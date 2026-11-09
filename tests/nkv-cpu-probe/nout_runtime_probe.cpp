// CPU-only probe for pillar 1: the n_outputs runtime parameter lane.
//
// One graph topology, one cycle of active n_outputs values, two context arms
// against one model. The ON arm sets GGML_TP5_UNION_BINDER=1, the OFF arm sets
// GGML_TP5_UNION_BINDER=0 (stock). Both arms receive the same batch at every
// step, so the only thing the two arms can disagree about is how the binder
// reaches the same result. Inside the cycle the token width is constant and
// only the set of tokens marked as outputs moves: that keeps the n_tokens lane
// out of the cycle measurement, so a build-count move there can only come from
// the output-count lane.
//
// WHAT THIS MEASURES, and nothing else:
//
//   (a) llama_graph_build_count(ON) does not move across the whole cycle after
//       the first decode. The graph topology did not change when the active
//       output count moved. This is the cross-n_outputs topology-hash
//       invariance the lane exists for, observed through the only public
//       counter that reports graph builds.
//
//   (b) Every requested logits row is byte-identical between the ON and OFF
//       arms at every active count in the cycle. A silent no-op (the gather
//       still reading the previous row set) or a wrong gather (rows taken from
//       the wrong token) shows up here as a difference.
//
//   (c) llama_decode(ON) returns 0 at every accepted step. This is not
//       decoration: process_ubatch() has a poka-yoke immediately before
//       compute,
//
//           const uint32_t out_bound = res->outputs_bound();
//           if (out_bound != 0 && out_bound != n_outputs) {
//               LLAMA_LOG_ERROR("%s: output gather bound at %u, active n_outputs %u\n", ...);
//               ret = GGML_STATUS_FAILED;
//               return nullptr;
//           }
//
//       (src/llama-context.cpp, end of process_ubatch). A non-zero decode means
//       the gather closure was not at the active width, or another guard
//       refused the step. So (a) + (b) + (c) together say: the gather executed
//       at the active count, on a graph that was never rebuilt. That chain is
//       the whole proof; there is no fourth observation hiding behind it.
//
// WHAT THIS DOES NOT MEASURE (no public API exposes it):
//
//   - There is no public bound-output count. llama_union_diagnostics has no
//     n_outputs field; its bound_tokens is st.record_bound_ntok, the bound
//     TOKEN width. This probe varies the token width between the cycle (4) and
//     the growth arm (5), so bound_tokens does move, but it moves with the
//     schedule, not with the output count. It is therefore reported only, and
//     never asserted: it is the n_tokens witness, not the n_outputs one.
//   - c_nout_grow_binds, c_nout_grow_rebuilds, c_refused_nout,
//     record_cap_nout, record_bound_nout and nout_high_water are internal
//     (src/llama-nout-state.h) and are not read. The growth arm below asserts
//     the build-count behaviour and row equality, which is what an external
//     caller can see; it does not claim the internal counter moved.
//   - The output envelope is max(active, nout_high_water) (graph_params), so a
//     width inside the cycle is a bind, never a rebuild. A request ABOVE the
//     mark is the one envelope-expanding rebuild; see the growth arm.
//
// THE SCHEDULE
//
//   cycle   every step submits 4 tokens at consecutive positions. Active output
//           counts run 1, 3, 2, 4, 1, 3, 2, 4, 2, 3. The first decode builds
//           the envelope at 1; the second (3) grows inside it; the rest bind.
//           The requested output rows are a moving subset of the 4 batch
//           positions, so a gather stuck on the previous row set cannot match
//           the OFF arm.
//   growth  three more steps request 5, 3, 5 outputs. n_outputs_max is 5 here,
//           so 5 is legal. batch.logits has one entry per submitted token, so a
//           5-output request needs a 5-token batch: the growth steps submit 5
//           tokens and the cycle submits 4. The cycle's high-water is 4, so the
//           first growth step is the envelope-expanding rebuild: exactly one
//           build there. The rebuild raises the mark to 5, so the following 3
//           and 5 bind in place and add no build. The token width change from
//           4 to 5 is a separate runtime key and LLAMA_EXPERIMENTAL_VARLEN_NTOK
//           is unset here, so it cannot be bound; it is part of the rebuild,
//           not of the bind. That is why the growth step is expected to build,
//           and why the two steps after it isolate the output count again.
//   refusal one final step asks for 6 outputs with n_outputs_max = 5. It must
//           fail on both arms, the ON arm must not build a graph, and the KV
//           position of the last accepted token must not move: the refusal
//           happens before compute, and decode removes the failed ubatch's
//           positions again.
//
// FAIL-CLOSED GATES (any failure exits 1; the exit code is 0 only on a full
// pass). Each one is printed as GATE_FAIL=<name> with its observed values:
//
//   G1  the OFF arm reports an enabled binder (a stock context allocates no
//       binder state at all, so "no state" is the correct answer, not a
//       failure)
//   G2  the ON arm reports no state, a disabled binder, or an
//       envelope_disabled binder (the ON gate is 1, so the state must exist)
//   G3  the ON arm's decode build total over the whole run is not exactly 2
//       (the envelope build plus the one rebuild at the growth step; that
//       rebuild is on a changed skeleton, not a pure n_outputs expansion)
//   G4  any cycle step after the first adds an ON decode build
//   G5  the growth step does not add exactly one ON decode build
//   G6  any step after the growth adds an ON decode build
//   G7  any step does not execute: a non-zero llama_decode on either arm
//   G8  any requested logits row differs between the arms, or a row cannot be
//       captured on either arm
//   G9  a request above n_outputs_max is accepted: the step must fail on both
//       arms, the ON arm must not have built a graph, and the KV position of
//       the last accepted token must not move
//   G10 the OFF arm's build count does not follow the stock reuse rule: with
//       no binder state the stock arm rebuilds exactly when the active output
//       count changes and reuses the graph when it does not
//       (llm_graph_input_out_ids::can_reuse compares bound_outputs against
//       params.n_outputs; allow_reuse sees an unchanged skeleton). The rule is
//       evaluated per step inside the cycle, where the skeleton is constant,
//       and as a whole-run relation over the growth arm, where the token width
//       also changes
//   G11 the run did not consume the whole token budget
//
// Exit codes: 0 pass, 1 gate failure, 2 usage or model-load failure.
#include "llama.h"

// The shared arm machinery: environment selection, model/context setup, the
// build-count delta and the gate-failure printer. Every probe in this
// directory uses it, so the arm contract is stated once.
#include "union_arm.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int cycle_width   = 4; // tokens per cycle step
constexpr int growth_width  = 5; // tokens per growth step: one more output than the cycle
constexpr int refusal_width = 5; // tokens per refusal step

// Active output counts of the cycle, and the batch positions requested as
// outputs for each. The positions are a moving subset of 0..cycle_width-1.
struct cycle_step {
    int  n_outputs;
    bool output[cycle_width];
};

constexpr cycle_step cycle[] = {
    {1, {true,  false, false, false}},
    {3, {false, true,  true,  true }},
    {2, {true,  true,  false, false}},
    {4, {true,  true,  true,  true }},
    {1, {false, false, false, true }},
    {3, {true,  false, true,  true }},
    {2, {false, true,  false, true }},
    {4, {true,  true,  true,  true }},
    {2, {false, false, true,  true }},
    {3, {true,  true,  true,  false}},
};
constexpr int cycle_steps = int(sizeof(cycle) / sizeof(cycle[0]));

// Growth arm: 5 outputs (above the cycle's high-water of 4), then 3, then 5
// again. n_outputs_max is growth_width = 5, so 5 is legal.
constexpr int growth_outputs[3] = {growth_width, 3, growth_width};
constexpr int growth_steps      = 3;

// One request above n_outputs_max.
constexpr int refusal_outputs = growth_width + 1;

constexpr int total_steps  = cycle_steps + growth_steps + 1;
constexpr int total_tokens = cycle_width * cycle_steps +
                             growth_width * growth_steps +
                             refusal_width;

struct step_plan {
    int              width     = 0;
    int              n_outputs = 0;
    bool             refuse    = false; // expects a refusal on both arms
    std::vector<int> indices;           // batch token indices requested as outputs
};

// Growth step `i` requests growth_outputs[i] rows: the first `n` batch
// positions, ascending. With growth_width tokens submitted, every requested
// index is a legal batch position.
step_plan plan_growth(int i) {
    step_plan p;
    p.width     = growth_width;
    p.n_outputs = growth_outputs[i];
    for (int j = 0; j < p.n_outputs; ++j) {
        p.indices.push_back(j);
    }
    return p;
}

step_plan plan_cycle(int i) {
    const cycle_step & c = cycle[i];
    step_plan p;
    p.width     = cycle_width;
    p.n_outputs = c.n_outputs;
    for (int j = 0; j < cycle_width; ++j) {
        if (c.output[j]) {
            p.indices.push_back(j);
        }
    }
    return p;
}

step_plan plan_refusal() {
    step_plan p;
    p.width     = refusal_width;
    p.n_outputs = refusal_outputs;
    p.refuse    = true;
    for (int j = 0; j < p.n_outputs; ++j) {
        p.indices.push_back(j);
    }
    return p;
}

step_plan plan_step(int step) {
    if (step < cycle_steps) {
        return plan_cycle(step);
    }
    if (step < cycle_steps + growth_steps) {
        return plan_growth(step - cycle_steps);
    }
    return plan_refusal();
}

bool capture(llama_context * ctx, const std::vector<int> & indices, int vocab,
             std::vector<float> & rows) {
    rows.clear();
    for (int index : indices) {
        const float * logits = llama_get_logits_ith(ctx, index);
        if (!logits) {
            return false;
        }
        rows.insert(rows.end(), logits, logits + vocab);
    }
    return true;
}

// The gate printer is the shared one. This wrapper keeps the probe's
// `ok = gate_fail(...)` idiom: the shared function prints and returns void,
// the probe needs a value to assign.
bool gate_fail(const char * name, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buf[512];
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    union_arm::gate_fail(name, "%s", buf);
    return false;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s model.gguf\n", argv[0]);
        return 2;
    }

    // Model and both arms come from the shared header. The two arms differ in
    // exactly the variables arm_env names, and each states all of them, so no
    // arm inherits a setting from the other.
    union_arm::model_handle m;
    if (!union_arm::open_model(argv[1], m)) {
        return 2;
    }
    const int vocab = m.vocab;

    llama_context_params cp = union_arm::cpu_params(total_tokens, growth_width, growth_width);
    cp.n_ctx_kv = total_tokens;
    cp.no_perf  = false;

    llama_context * on  = union_arm::make_context(m, union_arm::arm_env::binder_arm(), cp);
    llama_context * off = union_arm::make_context(m, union_arm::arm_env::stock_arm(),  cp);
    if (!on || !off) {
        std::fprintf(stderr, "FAIL: context creation on=%p off=%p\n", (void *) on, (void *) off);
        if (on)  llama_free(on);
        if (off) llama_free(off);
        union_arm::close_model(m);
        return 2;
    }

    bool ok = true;

    // ---- G1/G2: arm identity before any decode -----------------------------
    //
    // The two arms are NOT symmetric. A stock context allocates no binder
    // state: the slot is created only when the gate is on (llama-context.cpp,
    // constructor), and llama_context::union_diagnostics returns false when
    // the slot does not exist. So for the OFF arm "no state" is the normal,
    // correct answer, and only a state that exists AND reports enabled is a
    // failure. For the ON arm the gate is 1, so the state must exist, be
    // enabled and not have disabled its envelope.
    llama_union_diagnostics d = {}, off_d = {};
    const bool off_has_state = llama_get_union_diagnostics(off, &off_d);
    const bool off_stock     = !off_has_state || !off_d.enabled;
    if (!off_stock) {
        ok = gate_fail("G1_off_binder", "off_has_state=%d off_enabled=%d",
                       int(off_has_state), int(off_d.enabled));
    }
    const bool on_has_state = llama_get_union_diagnostics(on, &d);
    if (!on_has_state || !d.enabled || d.envelope_disabled) {
        ok = gate_fail("G2_on_binder", "on_has_state=%d on_enabled=%d envelope_disabled=%d",
                       int(on_has_state), int(d.enabled), int(d.envelope_disabled));
    }
    if (!ok) {
        // No decode on a misidentified arm: a later pass would then describe
        // the wrong configuration.
        std::fprintf(stderr, "PROBE_RC=1\n");
        llama_free(on);
        llama_free(off);
        union_arm::close_model(m);
        return 1;
    }

    // Initialization reserve builds are separate from decode builds; the
    // deltas below are measured from here.
    const uint64_t on_init  = llama_graph_build_count(on);
    const uint64_t off_init = llama_graph_build_count(off);
    std::fprintf(stderr, "BEGIN steps=%d cycle=%d growth=%d cycle_width=%d growth_width=%d "
                 "n_outputs_max=%d on_init_builds=%llu off_init_builds=%llu\n",
                 total_steps, cycle_steps, growth_steps, cycle_width, growth_width,
                 int(cp.n_outputs_max), (unsigned long long) on_init, (unsigned long long) off_init);

    int pos            = 0; // tokens accepted into the KV cache, by the schedule
    int tokens_refused = 0; // tokens submitted by a step that was refused
    int on_decode_builds  = 0;
    int off_decode_builds = 0;
    int rows_compared     = 0;
    int prev_outputs      = -1; // active count of the previous accepted step

    std::vector<float> a, b;
    for (int step = 0; ok && step < total_steps; ++step) {
        const step_plan p = plan_step(step);

        llama_batch batch = llama_batch_init(p.width, 0, 1);
        if (!llama_batch_is_complete(&batch, p.width, false, true)) {
            ok = gate_fail("G7_execute", "step=%d batch allocation failed", step);
            llama_batch_free(batch);
            batch = llama_batch{};
            break;
        }
        for (int j = 0; j < p.width; ++j) {
            batch.token[j]     = 3 + (llama_token) (((int64_t) (pos + j) * 7 + 1) % (vocab - 3));
            batch.pos[j]       = pos + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = 0;
        }
        for (int index : p.indices) {
            batch.logits[index] = 1;
        }
        batch.n_tokens = p.width;

        const union_arm::build_delta on_delta(on);
        const union_arm::build_delta off_delta(off);
        const int on_rc  = llama_decode(on, batch);
        const int off_rc = llama_decode(off, batch);
        llama_batch_free(batch);
        const int on_build  = int(on_delta.delta());
        const int off_build = int(off_delta.delta());
        on_decode_builds  += on_build;
        off_decode_builds += off_build;

        if (p.refuse) {
            // ---- G9: a request above n_outputs_max must be refused ---------
            if (on_rc == 0 || off_rc == 0) {
                ok = gate_fail("G9_above_max", "step=%d requested=%d max=%d on_rc=%d off_rc=%d",
                               step, p.n_outputs, int(cp.n_outputs_max), on_rc, off_rc);
            } else if (on_build != 0) {
                ok = gate_fail("G9_above_max", "step=%d refused but ON built %d graph(s)",
                               step, on_build);
            }
            // The refused ubatch must leave no trace in the KV cache: decode
            // removes the failed ubatch's positions again. `pos` is not
            // advanced here, so the budget gate below still counts the tokens
            // the schedule submitted.
            const llama_pos on_pos_max  = llama_memory_seq_pos_max(llama_get_memory(on),  0);
            const llama_pos off_pos_max = llama_memory_seq_pos_max(llama_get_memory(off), 0);
            if (ok && (on_pos_max != pos - 1 || off_pos_max != pos - 1)) {
                ok = gate_fail("G9_above_max", "step=%d refused but KV pos_max on=%d off=%d, expected %d",
                               step, int(on_pos_max), int(off_pos_max), pos - 1);
            }
            std::fprintf(stderr, "STEP step=%d kind=refusal requested=%d max=%d indices=%zu "
                         "on_rc=%d off_rc=%d on_build=%d off_build=%d kv_pos_max=%d/%d\n",
                         step, p.n_outputs, int(cp.n_outputs_max), p.indices.size(),
                         on_rc, off_rc, on_build, off_build, int(on_pos_max), int(off_pos_max));
            // The refused tokens were never computed, so the budget gate must
            // not count them: the refusal is the last step of the schedule.
            tokens_refused += p.width;
            continue;
        }

        // ---- G7: the step executed on both arms ------------------------------
        if (on_rc != 0 || off_rc != 0) {
            ok = gate_fail("G7_execute", "step=%d requested=%d on_rc=%d off_rc=%d",
                           step, p.n_outputs, on_rc, off_rc);
            break;
        }
        if (int(p.indices.size()) != p.n_outputs) {
            ok = gate_fail("G7_execute", "step=%d plan says %d outputs, %zu indices requested",
                           step, p.n_outputs, p.indices.size());
            break;
        }

        // ---- G4/G5/G6: build-count shape ------------------------------------
        if (step == 0) {
            if (on_build != 1) {
                ok = gate_fail("G3_on_total_builds", "step=0 first decode built %d graph(s), expected 1",
                               on_build);
            }
        } else if (step < cycle_steps) {
            if (on_build != 0) {
                ok = gate_fail("G4_cycle_rebuild", "step=%d requested=%d ON added %d build(s)",
                               step, p.n_outputs, on_build);
            }
        } else if (step == cycle_steps) {
            if (on_build != 1) {
                ok = gate_fail("G5_growth_rebuild", "step=%d requested=%d ON added %d build(s), expected 1",
                               step, p.n_outputs, on_build);
            }
        } else if (on_build != 0) {
            ok = gate_fail("G6_post_growth_rebuild", "step=%d requested=%d ON added %d build(s)",
                           step, p.n_outputs, on_build);
        }

        // ---- G10: the OFF arm follows the stock reuse rule ------------------
        // No binder state on this arm, so reuse is decided only by
        // res->can_reuse(): allow_reuse(ignore_runtime=true) plus each input's
        // can_reuse(). Inside the cycle every step has the same skeleton
        // (4 tokens, one sequence, one token per sequence), so the only key
        // that can differ is the active count, compared by
        // llm_graph_input_out_ids::can_reuse as bound_outputs == n_outputs.
        // Stock therefore rebuilds exactly when the count changes and reuses
        // otherwise. The growth arm also changes the token width, which this
        // rule cannot model, so it is checked as a whole-run relation below.
        if (step < cycle_steps) {
            const bool count_changed = step == 0 || p.n_outputs != prev_outputs;
            const int  expected      = count_changed ? 1 : 0;
            if (off_build != expected) {
                ok = gate_fail("G10_off_stock_reuse",
                               "step=%d requested=%d count_changed=%d OFF added %d build(s), expected %d",
                               step, p.n_outputs, int(count_changed), off_build, expected);
            }
        }

        // ---- G8: per-row byte equality --------------------------------------
        if (!capture(on, p.indices, vocab, a) || !capture(off, p.indices, vocab, b) ||
                a.size() != b.size() || a.size() != size_t(p.n_outputs) * size_t(vocab)) {
            ok = gate_fail("G8_rows", "step=%d requested=%d capture failed (on=%zu off=%zu floats)",
                           step, p.n_outputs, a.size(), b.size());
            break;
        }
        // Row r of both captures is the logits row of batch token
        // p.indices[r] (llama_get_logits_ith takes the batch index), so
        // position in the capture and batch index name the same row on both
        // arms.
        // The whole capture is compared byte for byte. `rows_equal` is the
        // shared helper: equal sizes, non-empty, memcmp over every float.
        const bool rows_identical = union_arm::rows_equal(a, b);
        const int  equal_rows     = rows_identical ? p.n_outputs : 0;
        rows_compared += p.n_outputs;
        if (!rows_identical) {
            ok = gate_fail("G8_rows", "step=%d requested=%d equal_rows=%d/%d",
                           step, p.n_outputs, equal_rows, p.n_outputs);
        }

        std::fprintf(stderr, "STEP step=%d kind=%s requested=%d indices=%zu on_rc=%d off_rc=%d "
                     "on_build=%d off_build=%d bitwise_rows=%d/%d\n",
                     step, step < cycle_steps ? "cycle" : "growth",
                     p.n_outputs, p.indices.size(), on_rc, off_rc, on_build, off_build,
                     equal_rows, p.n_outputs);
        prev_outputs = p.n_outputs;
        pos += p.width;
    }
    if (!ok) {
        // A step failed. The counters below still describe what ran, so print
        // them: the build total is the first thing a reader needs.
        std::fprintf(stderr, "ABORT on_decode_builds=%d off_decode_builds=%d accepted_tokens=%d\n",
                     on_decode_builds, off_decode_builds, pos);
    }

    // ---- G3: the whole run built exactly the two graphs --------------------
    // The first decode's envelope build plus one rebuild at the growth step,
    // and nothing else. That growth rebuild is caused by a changed skeleton:
    // the token width moves from 4 to 5 and LLAMA_EXPERIMENTAL_VARLEN_NTOK is
    // unset, so the n_tokens lane cannot bind it. The n_outputs part of that
    // rebuild is not separable from outside; what is proven here is that no
    // other step needed a rebuild, i.e. the whole cycle ran on one topology.
    // Evaluated even after a failed step, so a run that broke early does not
    // hide a build total behind the earlier gate.
    if (on_decode_builds != 2) {
        ok = gate_fail("G3_on_total_builds", "ON decode builds over %d steps = %d, expected 2",
                       total_steps, on_decode_builds);
    }

    // ---- G10 (whole run): the stock arm must rebuild more than the binder --
    // The cycle rule above is exact only where the skeleton is constant. The
    // growth arm changes the token width as well, so its per-step count is a
    // property of the schedule, not of stock-ness. What must hold over the
    // whole run is that the stock arm rebuilds strictly more often than the
    // binder arm: that is the point of the lane.
    if (off_decode_builds <= on_decode_builds) {
        ok = gate_fail("G10_off_stock_reuse",
                       "OFF built %d graph(s), ON built %d: the stock arm must rebuild more",
                       off_decode_builds, on_decode_builds);
    }

    // ---- G11: the whole token budget was submitted -------------------------
    if (pos + tokens_refused != total_tokens) {
        ok = gate_fail("G11_budget", "accepted %d + refused %d of %d tokens",
                       pos, tokens_refused, total_tokens);
    }

    llama_union_diagnostics on_end = {};
    const bool on_end_ok = llama_get_union_diagnostics(on, &on_end);
    std::fprintf(stderr, "END steps=%d tokens=%d rows_compared=%d on_decode_builds=%d "
                 "off_decode_builds=%d on_patch_binds=%llu on_rebuild_fallbacks=%llu "
                 "on_lane_declines=%llu on_bound_tokens=%u on_envelope_disabled=%d\n",
                 total_steps, pos, rows_compared, on_decode_builds, off_decode_builds,
                 (unsigned long long) on_end.patch_binds,
                 (unsigned long long) on_end.rebuild_fallbacks,
                 (unsigned long long) on_end.lane_declines,
                 on_end.bound_tokens,
                 on_end_ok ? int(on_end.envelope_disabled) : -1);
    // bound_tokens is reported, never asserted: it is record_bound_ntok (the
    // token width), not an output count. See the header comment.

    llama_free(on);
    llama_free(off);
    union_arm::close_model(m);

    std::fprintf(stderr, "PROBE_RC=%d\n", ok ? 0 : 1);
    return ok ? 0 : 1;
}
