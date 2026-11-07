#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

//
// test-rsz-runtime-param
//
// rs_z (the recurrent-state cell build_rs() zeroes) as a RUNTIME parameter.
//
// Gate: GGML_TP5_RSZ_RUNTIME_PARAM, exactly "1", and only under
// GGML_TP5_UNION_BINDER. Anything else is OFF and byte-identical to stock.
//
// The test drives one tiny recurrent fixture through >= 4 distinct rs_z
// values and asserts:
//   (a) graph builds == 1 across every rs_z variation (no rebuild),
//   (b) logits and recurrent state are bitwise equal to the OFF arm for
//       every variation,
//   (c) the outputs are non-degenerate (rows differ, nonzero spread), so a
//       bitwise match cannot be vacuously true,
//   (d) the named fallback counter is 0 for the supported fixture, and > 0
//       for an explicitly unsupported one (rs_z == -1, no free zero cell),
//       proving the fallback is explicit and not silent.
//
// It also covers the parameter SURFACE, not only the mechanism:
//   (e) boundary cases, each with its own fixture and its own named counter:
//       rs_z == -1 -> _cell; n_rs_seq > 0 -> _n_rs_seq; a context that
//       cannot arm -> a named reason; the live zero view covers exactly the
//       fixture's state cells and every bound rs_z is inside it. The
//       _envelope branch (rs_z >= zero_view_rows) cannot be produced by the
//       live allocator and is COMPILE_DEFERRED_13022 with a hand-driven
//       value; this test states that gap instead of faking it;
//   (f) the lane total is the sum of the named counters, and a refusal is
//       counted once (never once per decode);
//   (g) the interaction with the umbrella gate and the recurrent runtime
//       parameters (rs_z, rs_head) is in test-rsz-union-combo.cpp.
//
// The pure parser contract (unset / "" / "0" / "true" / "01" / " 1" / "2" /
// "1") needs no model and lives in test-union-binder-gate.cpp.
//
// usage: test-rsz-runtime-param model.gguf
//
// The fixture must be a recurrent (MAMBA / MAMBA2) GGUF. The test is skipped
// with exit code 77 (ctest SKIP_RETURN_CODE) when no model is supplied, and
// fails when the supplied model is not recurrent: this lane is only defined
// for a recurrent/hybrid graph.

namespace {

constexpr int k_skip = 77;

struct arm_stats {
    uint64_t builds = 0;
    llama_union_diagnostics d = {};
    bool have = false;
};

arm_stats read_stats(llama_context * ctx) {
    arm_stats s;
    s.builds = llama_graph_build_count(ctx);
    s.have   = llama_get_union_diagnostics(ctx, &s.d);
    return s;
}

// one decode of a single token on one sequence; returns the logits row
bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos, llama_seq_id seq,
                std::vector<float> & out, int n_vocab) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
        llama_batch_free(batch);
        return false;
    }
    batch.n_tokens    = 1;
    batch.token[0]    = tok;
    batch.pos[0]      = pos;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = seq;
    batch.logits[0]   = 1;
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        return false;
    }
    const float * row = llama_get_logits_ith(ctx, 0);
    if (row == nullptr) {
        return false;
    }
    out.assign(row, row + n_vocab);
    return true;
}

// copy the recurrent state of one sequence out of the cache
bool read_state(llama_context * ctx, llama_seq_id seq, std::vector<uint8_t> & out) {
    size_t size = llama_state_seq_get_size(ctx, seq);
    if (size == 0) {
        return false;
    }
    out.resize(size);
    return llama_state_seq_get_data(ctx, out.data(), size, seq) == size;
}

// non-degeneracy: at least two distinct values and a nonzero spread
bool non_degenerate(const std::vector<float> & v) {
    if (v.size() < 2) {
        return false;
    }
    float lo = v[0], hi = v[0];
    bool distinct = false;
    for (float x : v) {
        lo = x < lo ? x : lo;
        hi = x > hi ? x : hi;
        distinct = distinct || x != v[0];
    }
    return distinct && hi > lo;
}


// The helper OWNS the gate environment: it clears every knob the arms depend
// on before the context is created and clears them again afterwards, so a
// caller's setenv() immediately before the call is the only thing that
// decides the arm. Without this, `GGML_TP5_UNION_BINDER=1 ./test ...` would
// silently turn the OFF arm into a second binder arm and the bitwise ON/OFF
// assertions would compare two binder contexts and still pass.
//
// Both variables are read exactly once, in the llama_context constructor:
// GGML_TP5_UNION_BINDER at src/llama-context.cpp:536-537 and
// GGML_TP5_RSZ_RUNTIME_PARAM at :576. Clearing them after llama_init_from_model()
// returns cannot change the context that just captured them.
llama_context * make_ctx_slots(llama_model * model, uint32_t n_ctx, uint32_t n_seq_max, uint32_t n_rs_slots,
                               uint32_t n_rs_seq = 0) {
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    unsetenv("GGML_TP5_UNION_BINDER");
    unsetenv("GGML_TP5_RSZ_RUNTIME_PARAM");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_ctx_kv        = n_ctx;
    cp.n_batch         = 8;
    cp.n_ubatch        = 8;
    cp.n_outputs_max   = 8;
    cp.n_seq_max       = n_seq_max;
    cp.n_rs_slots      = n_rs_slots;
    cp.n_rs_seq        = n_rs_seq;
    cp.n_threads       = 1;
    cp.n_threads_batch = 1;
    cp.kv_unified      = true;
    cp.offload_kqv     = false;
    cp.op_offload      = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.embeddings      = false;
    llama_context * ctx = llama_init_from_model(model, cp);

    // leave the environment clean for the next call site
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    unsetenv("GGML_TP5_UNION_BINDER");
    unsetenv("GGML_TP5_RSZ_RUNTIME_PARAM");
    return ctx;
}

// the lane total must be the sum of the named counters: one refusal, one
// counter plus the total. Returns false and prints which counter broke it.
// One refusal, one named counter plus the lane total (llama-nout-state.h,
// rsz_rt_refuse()). All nine named causes are summed, including the two the
// relay ring added.
bool fallbacks_add_up(const llama_union_diagnostics & d, const char * label) {
    const uint64_t sum = d.rsz_rt_refused_gate + d.rsz_rt_refused_arch +
                         d.rsz_rt_refused_no_rs_z + d.rsz_rt_refused_cell +
                         d.rsz_rt_refused_envelope + d.rsz_rt_refused_shape +
                         d.rsz_rt_refused_n_rs_seq + d.rsz_rt_refused_ring +
                         d.rsz_rt_refused_mixed_depth;
    if (sum != d.rsz_rt_fallbacks) {
        std::fprintf(stderr,
                     "FAIL %s: named refusals sum to %llu, lane total is %llu "
                     "(gate=%llu arch=%llu no_rs_z=%llu cell=%llu envelope=%llu shape=%llu n_rs_seq=%llu)\n",
                     label, (unsigned long long) sum, (unsigned long long) d.rsz_rt_fallbacks,
                     (unsigned long long) d.rsz_rt_refused_gate,
                     (unsigned long long) d.rsz_rt_refused_arch,
                     (unsigned long long) d.rsz_rt_refused_no_rs_z,
                     (unsigned long long) d.rsz_rt_refused_cell,
                     (unsigned long long) d.rsz_rt_refused_envelope,
                     (unsigned long long) d.rsz_rt_refused_shape,
                     (unsigned long long) d.rsz_rt_refused_n_rs_seq);
        return false;
    }
    return true;
}

// Six physical recurrent-state cells (n_rs_slots <= n_seq_max, so both are 6;
// llama-context.cpp:264-266 -- size = mem_size = n_rs_slots).
//
// rs_z is the first zero-refcount cell in [min, max], where min/max are taken
// over the tails of the sequences IN THE BATCH (llama-memory-recurrent.cpp
// :598-599, :698-699) and head is set to min (:782). The search range is
// therefore as wide as the spread of the batch's tails, and it widens further
// because seq_rm() releases a cell and resets its src to -1 (:278-285), which
// makes that cell a zero-refcount candidate inside [min, max].
//
// The batch always holds exactly two sequences with one token each, so
// n_seq_tokens == 1 and equal_seqs() holds (llama-batch.cpp:775,
// :842-843). Positions per sequence stay consecutive, which is what
// llama_batch_allocr::init() checks (llama-batch.cpp:293-314); the two
// sequences may be at different positions.
llama_context * make_ctx(llama_model * model, uint32_t n_ctx) {
    return make_ctx_slots(model, n_ctx, 6, 6);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s model.gguf (recurrent fixture)\n", argv[0]);
        return k_skip;
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    ggml_backend_dev_t devices[] = { nullptr };
    mp.devices = devices;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        std::fprintf(stderr, "FAIL: model load\n");
        llama_backend_free();
        return 1;
    }
    if (!llama_model_is_recurrent(model)) {
        std::fprintf(stderr, "FAIL: fixture is not recurrent; this lane is defined for recurrent graphs only\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (n_vocab < 4) {
        std::fprintf(stderr, "FAIL: fixture vocabulary too small (%d)\n", n_vocab);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // OFF arm: stock. make_ctx_slots() clears every gate itself, so this is
    // stock whatever the caller's shell exported.
    llama_context * off = make_ctx(model, 64);
    if (off == nullptr) {
        std::fprintf(stderr, "FAIL: OFF context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // ON arm: the runtime lane under the umbrella binder gate.
    setenv("GGML_TP5_UNION_BINDER", "1", 1);
    setenv("GGML_TP5_RSZ_RUNTIME_PARAM", "1", 1);
    llama_context * on = make_ctx(model, 64);
    if (on == nullptr) {
        std::fprintf(stderr, "FAIL: ON context\n");
        llama_free(off);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    bool ok = true;

    // ---- supported fixture: >= 4 distinct rs_z values -------------------
    //
    // The batch always holds exactly two sequences with one token each, so
    // n_seq_tokens == 1 and equal_seqs() holds. pos[] tracks one next
    // position per sequence; positions stay consecutive per sequence, which
    // is what llama_batch_allocr::init() requires.
    //
    // rs_z is the first zero-refcount cell in [min, max], and min/max are the
    // tails of the sequences in the batch. A cell becomes a candidate by
    // being released: seq_rm() sets its src to -1 (llama-memory-recurrent.cpp
    // :278-285). So rs_z walks by (a) keeping the batch's two tails far
    // apart, which widens [min, max], and (b) releasing the cells the search
    // should land on.
    //
    // The table below is the output of a full simulation of find_slot()
    // (:652-790) plus seq_rm() (:198-293), run over the exact steps[] below.
    // The simulation models cells (pos, src, src0, seq_id), tails, head,
    // used, the capacity guard (:560-582), the placement loop (:664-700),
    // the re-order (:702-724), the pos update (:727-747) and the
    // refcount/rs_z search (:749-779). It reports no capacity refusal, no
    // :677 abort, and no :734 non-consecutive-position warning on any row.
    //
    //  k  pair   release  head_in nec min max refc[0..5]            rs_z  s_copy_extra
    //  0  (3,5)  {4}      0       0   0   1   [0,0,0,0,0,0]          0     -
    //  1  (1,3)  {5}      0       1   0   1   [1,0,0,0,0,0]          1     -
    //  2  (0,5)  {4,2}    0       2   2   3   [1,1,0,0,0,0]          2     -
    //  3  (1,4)  {}       2       4   0   4   [1,1,1,1,0,0]          4     (2<-2)(3<-3)(4<-1)
    //  4  (0,2)  {3,5,4}  0       1   1   2   [1,0,1,0,0,0]          1     -
    //  5  (0,5)  {}       1       3   1   3   [1,1,1,0,0,0]          3     (3<-2)
    //
    // `head_in` is `min` of the previous row (:782); `nec` is
    // next_empty_cell at the start of the walk (:654). `refc[i]` counts cells
    // whose src == i, computed BEFORE the src = i staging at :778, so a fresh
    // cache (src == -1 everywhere, llama-memory-recurrent.h:103) has all-zero
    // refcounts and rs_z = 0 on row 0.
    //
    // s_copy_extra lists (cell <- src0) for cells [head+n_seqs, head+n_rs):
    // those are the cells copied FROM the zeroed rs_z row, i.e. the observable
    // effect of the selection. Rows 3 and 5 exercise it.
    //
    // Distinct values: {0, 1, 2, 3, 4}. The test reads them back from the
    // diagnostics and requires >= 4; it never assumes this table's outcome.
    //
    // No row can double-place a sequence. find_slot() would abort at :677
    // only if a batch sequence needs a cell while next_empty_cell points at
    // a live cell, i.e. with zero empty cells. A sequence needs a cell when
    // its tail is -1 or its cell is shared; both states are produced by
    // seq_rm()/seq_cp(), and each of those frees a cell first (:278-285,
    // :311-321). So "no empty cell + a needy sequence" is unreachable here,
    // and every row above places two sequences into two distinct cells.
    struct fixture_step {
        llama_seq_id a, b;
        llama_seq_id release[3];
    };
    const fixture_step steps[] = {
        { 3, 5, { 4, -1, -1 } },
        { 1, 3, { 5, -1, -1 } },
        { 0, 5, { 4,  2, -1 } },
        { 1, 4, { -1, -1, -1 } },
        { 0, 2, { 3,  5,  4 } },
        { 0, 5, { -1, -1, -1 } },
    };
    const uint32_t n_steps = sizeof(steps) / sizeof(steps[0]);

    std::vector<int32_t> seen;
    uint64_t builds_after_first = 0;
    uint64_t binds_after_first  = 0;
    bool     first_done         = false;
    uint32_t active_steps       = 0;
    llama_pos pos[6] = { 0, 0, 0, 0, 0, 0 };

    for (uint32_t step = 0; ok && step < n_steps; ++step) {
        const fixture_step & fs = steps[step];
        for (int r = 0; r < 3; ++r) {
            const llama_seq_id drop = fs.release[r];
            if (drop < 0) {
                continue;
            }
            const bool ok_on  = llama_memory_seq_rm(llama_get_memory(on),  drop, -1, -1);
            const bool ok_off = llama_memory_seq_rm(llama_get_memory(off), drop, -1, -1);
            if (!ok_on || !ok_off) {
                std::fprintf(stderr, "FAIL: seq_rm step=%u seq=%d\n", step, (int) drop);
                ok = false;
                break;
            }
            pos[drop] = 0;
        }
        if (!ok) {
            break;
        }

        // one batch, two sequences, one token each: equal_seqs() holds
        const llama_seq_id pair[2] = { fs.a, fs.b };
        llama_batch batch_on  = llama_batch_init(2, 0, 1);
        llama_batch batch_off = llama_batch_init(2, 0, 1);
        const bool batch_ok = batch_on.token  && batch_on.pos  && batch_on.n_seq_id  &&
                              batch_on.seq_id && batch_on.logits &&
                              batch_off.token && batch_off.pos && batch_off.n_seq_id &&
                              batch_off.seq_id && batch_off.logits;
        if (batch_ok) {
            batch_on.n_tokens  = 2;
            batch_off.n_tokens = 2;
            for (int i = 0; i < 2; ++i) {
                const llama_token tok = (llama_token) (1 + (int) ((step * 7 + i * 3) % (n_vocab - 1)));
                batch_on.token[i]      = tok;
                batch_on.pos[i]        = pos[pair[i]];
                batch_on.n_seq_id[i]   = 1;
                batch_on.seq_id[i][0]  = pair[i];
                batch_on.logits[i]     = 1;
                batch_off.token[i]     = tok;
                batch_off.pos[i]       = pos[pair[i]];
                batch_off.n_seq_id[i]  = 1;
                batch_off.seq_id[i][0] = pair[i];
                batch_off.logits[i]    = 1;
            }
        }
        const int rc_on  = batch_ok ? llama_decode(on,  batch_on)  : -1;
        const int rc_off = batch_ok ? llama_decode(off, batch_off) : -1;
        llama_batch_free(batch_on);
        llama_batch_free(batch_off);
        if (rc_on != 0 || rc_off != 0) {
            std::fprintf(stderr, "FAIL: decode step=%u on=%d off=%d\n", step, rc_on, rc_off);
            ok = false;
            break;
        }
        for (int i = 0; i < 2; ++i) {
            ++pos[pair[i]];
        }

        arm_stats s_on = read_stats(on);
        if (!s_on.have || !s_on.d.enabled) {
            std::fprintf(stderr, "FAIL: ON diagnostics missing or binder disabled\n");
            ok = false;
            break;
        }
        // rs_z == -1 is a legitimate, counted refusal: the lane cannot
        // express it and says so. Every other step must have the lane active.
        if (s_on.d.rsz_rt_enabled) {
            if (s_on.d.rsz_rt_bound_rs_z < 0) {
                if (s_on.d.rsz_rt_active) {
                    std::fprintf(stderr, "FAIL: lane active with rs_z=%d step=%u\n",
                                 s_on.d.rsz_rt_bound_rs_z, step);
                    ok = false;
                    break;
                }
            } else {
                if (!s_on.d.rsz_rt_active) {
                    std::fprintf(stderr, "FAIL: lane inactive at rs_z=%d (fallbacks=%llu) step=%u\n",
                                 s_on.d.rsz_rt_bound_rs_z,
                                 (unsigned long long) s_on.d.rsz_rt_fallbacks, step);
                    ok = false;
                    break;
                }
                ++active_steps;
            }
        }
        for (int i = 0; i < 2; ++i) {
            const float * row_on  = llama_get_logits_ith(on,  i);
            const float * row_off = llama_get_logits_ith(off, i);
            if (row_on == nullptr || row_off == nullptr) {
                std::fprintf(stderr, "FAIL: missing logits step=%u row=%d\n", step, i);
                ok = false;
                break;
            }
            const std::vector<float> va(row_on,  row_on  + n_vocab);
            const std::vector<float> vb(row_off, row_off + n_vocab);
            if (!non_degenerate(va) || !non_degenerate(vb)) {
                std::fprintf(stderr, "FAIL: degenerate logits step=%u row=%d\n", step, i);
                ok = false;
                break;
            }
            if (va != vb) {
                std::fprintf(stderr, "FAIL: ON/OFF logits differ step=%u row=%d\n", step, i);
                ok = false;
                break;
            }
            std::vector<uint8_t> sa, sb;
            if (!read_state(on, pair[i], sa) || !read_state(off, pair[i], sb) || sa != sb) {
                std::fprintf(stderr, "FAIL: ON/OFF recurrent state differs step=%u row=%d\n", step, i);
                ok = false;
                break;
            }
        }
        if (!ok) {
            break;
        }
        seen.push_back(s_on.d.rsz_rt_bound_rs_z);
        if (!first_done) {
            // baseline for the no-rebuild / no-bind assertions: the first
            // decode pays the reserve builds and the build's own bind
            builds_after_first = s_on.builds;
            binds_after_first  = s_on.d.patch_binds;
            first_done         = true;
        }
    }

    // distinct rs_z values actually exercised
    std::vector<int32_t> distinct;
    for (int32_t v : seen) {
        bool dup = false;
        for (int32_t d : distinct) {
            dup = dup || d == v;
        }
        if (!dup) {
            distinct.push_back(v);
        }
    }

    arm_stats fin_on  = read_stats(on);
    arm_stats fin_off = read_stats(off);

    // (a) no rebuild across every rs_z variation.
    //
    // llama_graph_build_count() also counts the reserve builds that
    // sched_reserve() performs on the first decode (three graph_reserve()
    // calls: pp, tg, pp again -- see llama-context.cpp sched_reserve), so the
    // absolute count is not 1. What the lane guarantees is that the count does
    // not move once the first decode has built its graph.
    if (ok && fin_on.builds != builds_after_first) {
        std::fprintf(stderr, "FAIL: ON graph builds moved from %llu to %llu across rs_z variation\n",
                     (unsigned long long) builds_after_first, (unsigned long long) fin_on.builds);
        ok = false;
    }
    // the lane must have been active on the steps with a free zero cell
    if (ok && (!fin_on.d.rsz_rt_enabled || active_steps == 0)) {
        std::fprintf(stderr, "FAIL: rs_z runtime lane never active on a supported fixture\n");
        ok = false;
    }
    // (b) >= 4 distinct rs_z values across the run. The fixture widens
    // [min, max] through the batch's tail spread and releases the low cells,
    // so the search can return 0, 1, 2, 3, 4; the values are read back from
    // the diagnostics, never assumed.
    if (ok && distinct.size() < 4) {
        std::fprintf(stderr, "FAIL: only %zu distinct rs_z values observed, need >= 4\n", distinct.size());
        ok = false;
    }
    // (c) the fallback counter must hold exactly the steps with rs_z == -1
    // and nothing else: a fallback on a bindable step would mean the lane
    // silently gave up, and a missing one would mean it silently ran stock.
    {
        uint32_t neg_rounds = 0;
        for (int32_t v : seen) {
            neg_rounds += (v < 0) ? 1 : 0;
        }
        if (ok && fin_on.d.rsz_rt_fallbacks != neg_rounds) {
            std::fprintf(stderr, "FAIL: %llu rs_z fallbacks, expected %u (the rs_z == -1 steps)\n",
                         (unsigned long long) fin_on.d.rsz_rt_fallbacks, neg_rounds);
            ok = false;
        }
    }
    // rs_z itself must not be bound: the lane makes it an input write, so no
    // union patch may be applied for an rs_z move. The first decode may bind
    // once for an unrelated reason (n_kv exact width), so compare against the
    // count taken after the first decode instead of against zero.
    if (ok && fin_on.d.patch_binds != binds_after_first) {
        std::fprintf(stderr, "FAIL: rs_z variation applied %llu union patch(es) (was %llu after the first decode)\n",
                     (unsigned long long) fin_on.d.patch_binds,
                     (unsigned long long) binds_after_first);
        ok = false;
    }

    std::printf("RSZ_RT_STATS on_builds=%llu on_builds_after_first=%llu off_builds=%llu "
                "on_patch_binds=%llu on_patch_binds_after_first=%llu "
                "on_fallbacks=%llu rsz_rt=%d active=%d active_steps=%u bound_rs_z=%d "
                "zero_view_rows=%u distinct_rs_z=%zu\n",
                (unsigned long long) fin_on.builds, (unsigned long long) builds_after_first,
                (unsigned long long) fin_off.builds,
                (unsigned long long) fin_on.d.patch_binds, (unsigned long long) binds_after_first,
                (unsigned long long) fin_on.d.rsz_rt_fallbacks,
                fin_on.d.rsz_rt_enabled ? 1 : 0, fin_on.d.rsz_rt_active ? 1 : 0,
                active_steps,
                fin_on.d.rsz_rt_bound_rs_z, fin_on.d.rsz_rt_zero_view_rows, distinct.size());

    // ---- unsupported fixture: rs_z == -1 must refuse by name ------------
    //
    // The rs_z == -1 case is reached with a two-cell cache and ONE live
    // sequence. find_slot() computes min/max over the cells of the sequences
    // IN THE BATCH (llama-memory-recurrent.cpp:598-599, :698-699), so with one
    // sequence min == max == its tail; that cell's src is itself after the
    // staging at :775-781, so refcounts[min] == 1 and the rs_z search over
    // [min, max] finds nothing (:760-766). The capacity guard at :579 only
    // rejects required > size (1 > 2 is false), so the batch is admitted.
    // Stock then builds a ZERO-SIZED zero view, which a selection row cannot
    // express; llama_rsz_rt_armable() sees rs_z < 0 and does not arm, so the
    // record build counts c_rsz_rt_refused_cell. A zero counter here would
    // mean a silent fallback.
    {
        setenv("GGML_TP5_UNION_BINDER", "1", 1);
        setenv("GGML_TP5_RSZ_RUNTIME_PARAM", "1", 1);
        llama_context * dis = make_ctx_slots(model, 64, 2, 2);
        if (dis == nullptr) {
            std::fprintf(stderr, "FAIL: unsupported-fixture context\n");
            ok = false;
        } else {
            std::vector<float> a;
            const bool dec = decode_one(dis, 1, 0, 0, a, n_vocab);
            arm_stats ds = read_stats(dis);
            std::printf("RSZ_RT_UNSUPPORTED builds=%llu fallbacks=%llu refused_gate=%llu "
                        "refused_arch=%llu refused_no_rs_z=%llu refused_cell=%llu "
                        "refused_shape=%llu active=%d bound_rs_z=%d\n",
                        (unsigned long long) ds.builds,
                        (unsigned long long) ds.d.rsz_rt_fallbacks,
                        (unsigned long long) ds.d.rsz_rt_refused_gate,
                        (unsigned long long) ds.d.rsz_rt_refused_arch,
                        (unsigned long long) ds.d.rsz_rt_refused_no_rs_z,
                        (unsigned long long) ds.d.rsz_rt_refused_cell,
                        (unsigned long long) ds.d.rsz_rt_refused_shape,
                        ds.d.rsz_rt_active ? 1 : 0, ds.d.rsz_rt_bound_rs_z);
            if (!dec) {
                std::fprintf(stderr, "FAIL: unsupported-fixture decode\n");
                ok = false;
            } else if (!ds.have || ds.d.rsz_rt_fallbacks == 0) {
                // the fallback must be explicit: a zero counter here means
                // the lane silently ran stock
                std::fprintf(stderr, "FAIL: unsupported fixture did not count a fallback\n");
                ok = false;
            } else if (ds.d.rsz_rt_active) {
                std::fprintf(stderr, "FAIL: unsupported fixture claimed an active lane\n");
                ok = false;
            } else if (ds.d.rsz_rt_bound_rs_z != -1) {
                std::fprintf(stderr, "FAIL: unsupported fixture bound rs_z=%d, expected -1\n",
                             ds.d.rsz_rt_bound_rs_z);
                ok = false;
            } else if (ds.d.rsz_rt_refused_cell == 0) {
                std::fprintf(stderr, "FAIL: rs_z == -1 did not increment the named _cell counter\n");
                ok = false;
            } else if (!fallbacks_add_up(ds.d, "rs_z == -1 fixture")) {
                ok = false;
            }
            llama_free(dis);
        }
    }

    // ---- boundary: the zero-view extent the lane actually carries --------
    //
    // set_input() writes the selection into an I32 [1] tensor and asserts
    // rs_z >= 0; the context additionally refuses rs_z >= zero_view_rows as
    // _envelope before any reuse decision or input write.
    //
    // What the live run can prove is asserted here:
    //   - the recorded zero view has exactly the fixture's state-cell count
    //     (6 cells: make_ctx uses n_seq_max = n_rs_slots = 6), so the row
    //     range the lane clears is the cell range and not something wider;
    //   - every rs_z the run bound is inside [0, zero_view_rows), which is
    //     the precondition the _envelope check exists to enforce;
    //   - the run never hit _envelope, because rs_z is always a cell index.
    // The _envelope branch itself cannot be driven from a live allocator
    // (rs_z is a cell index and the cache has exactly that many rows), so it
    // is COMPILE_DEFERRED_13022 and needs a hand-driven value; this test does
    // not pretend to cover it.
    if (fin_on.have) {
        if (!fin_on.d.rsz_rt_active) {
            std::fprintf(stderr, "FAIL: lane not active at the end of the supported run\n");
            ok = false;
        } else if (fin_on.d.rsz_rt_zero_view_rows != 6) {
            std::fprintf(stderr,
                         "FAIL: live zero view has %u rows, expected the 6 state cells\n",
                         fin_on.d.rsz_rt_zero_view_rows);
            ok = false;
        } else if (fin_on.d.rsz_rt_refused_envelope != 0) {
            std::fprintf(stderr,
                         "FAIL: %llu _envelope refusals on a fixture whose rs_z is always a cell\n",
                         (unsigned long long) fin_on.d.rsz_rt_refused_envelope);
            ok = false;
        } else {
            for (int32_t v : seen) {
                if (v >= 0 && (uint32_t) v >= fin_on.d.rsz_rt_zero_view_rows) {
                    std::fprintf(stderr,
                                 "FAIL: bound rs_z=%d is outside the live zero view (%u rows)\n",
                                 v, fin_on.d.rsz_rt_zero_view_rows);
                    ok = false;
                }
            }
        }
    }

    // ---- boundary: a context whose state rows are not cells --------------
    //
    // n_rs_seq > 0 widens the state tensors to mem_size*(1+n_rs_seq) rows and
    // makes s_copy() return idx*size + src0, so a row index is no longer a
    // cell index (see build_rs()). The lane must refuse that by name
    // (_n_rs_seq) and never arm, instead of silently clearing the wrong row.
    // The model must support recurrent rollback for n_rs_seq to survive the
    // constructor clamp; when it does not, the request is clamped to 0 and
    // this case cannot be built -- the test then says so and skips only this
    // case, never the rest.
    {
        setenv("GGML_TP5_UNION_BINDER", "1", 1);
        setenv("GGML_TP5_RSZ_RUNTIME_PARAM", "1", 1);
        llama_context * nrs = make_ctx_slots(model, 64, 6, 6, /*n_rs_seq =*/ 2);
        if (nrs == nullptr) {
            std::fprintf(stderr, "FAIL: n_rs_seq fixture context\n");
            ok = false;
        } else if (llama_n_rs_seq(nrs) == 0) {
            std::printf("RSZ_RT_N_RS_SEQ skipped: the fixture arch does not support recurrent rollback\n");
            llama_free(nrs);
        } else {
            std::vector<float> a;
            const bool dec = decode_one(nrs, 1, 0, 0, a, n_vocab);
            arm_stats ns = read_stats(nrs);
            std::printf("RSZ_RT_N_RS_SEQ n_rs_seq=%u fallbacks=%llu refused_n_rs_seq=%llu active=%d\n",
                        llama_n_rs_seq(nrs),
                        (unsigned long long) ns.d.rsz_rt_fallbacks,
                        (unsigned long long) ns.d.rsz_rt_refused_n_rs_seq,
                        ns.d.rsz_rt_active ? 1 : 0);
            if (!dec) {
                std::fprintf(stderr, "FAIL: n_rs_seq fixture decode\n");
                ok = false;
            } else if (!ns.have || ns.d.rsz_rt_active) {
                std::fprintf(stderr,
                             "FAIL: n_rs_seq > 0 armed the lane; a row index is not a cell index there\n");
                ok = false;
            } else if (ns.d.rsz_rt_refused_n_rs_seq == 0) {
                std::fprintf(stderr,
                             "FAIL: n_rs_seq > 0 did not increment the named _n_rs_seq counter\n");
                ok = false;
            } else if (ns.d.rsz_rt_refused_n_rs_seq != 1) {
                // exactly one: graph_params() counts the refusal once per
                // build (llama-context.cpp, the rsz_rows_ok branch), and the
                // record build does not count it again. A count above one
                // means the same refusal is charged twice, which would make
                // the lane total meaningless.
                std::fprintf(stderr,
                             "FAIL: n_rs_seq > 0 counted %llu _n_rs_seq refusals, expected exactly 1\n",
                             (unsigned long long) ns.d.rsz_rt_refused_n_rs_seq);
                ok = false;
            } else if (!fallbacks_add_up(ns.d, "n_rs_seq fixture")) {
                ok = false;
            }
            llama_free(nrs);
        }
    }

    // ---- boundary: a context that cannot arm ----------------------------
    //
    // This fixture (two cells, one live sequence) yields rs_z == -1, so the
    // reachable refusal is _cell. _arch (no recurrent-state input) and
    // _no_rs_z (zero view not provable) need a non-recurrent or out-of-range
    // graph, which this lane is not defined for; the test asserts the
    // CONTRACT instead: when the gate is on and the lane is not active, at
    // least one named counter moved and the total is their sum, so the
    // reason is always visible.
    {
        setenv("GGML_TP5_UNION_BINDER", "1", 1);
        setenv("GGML_TP5_RSZ_RUNTIME_PARAM", "1", 1);
        llama_context * dis = make_ctx_slots(model, 64, 2, 2);
        if (dis == nullptr) {
            std::fprintf(stderr, "FAIL: cannot-arm fixture context\n");
            ok = false;
        } else {
            std::vector<float> a;
            const bool dec = decode_one(dis, 1, 0, 0, a, n_vocab);
            arm_stats cs = read_stats(dis);
            if (!dec) {
                std::fprintf(stderr, "FAIL: cannot-arm fixture decode\n");
                ok = false;
            } else if (!cs.have || !cs.d.rsz_rt_enabled) {
                std::fprintf(stderr, "FAIL: cannot-arm fixture did not read the gate as on\n");
                ok = false;
            } else if (cs.d.rsz_rt_active) {
                std::fprintf(stderr, "FAIL: cannot-arm fixture armed the lane\n");
                ok = false;
            } else if (!fallbacks_add_up(cs.d, "cannot-arm fixture")) {
                ok = false;
            } else {
                // exactly one of the reasons the built graph can show
                const uint64_t named = cs.d.rsz_rt_refused_arch + cs.d.rsz_rt_refused_no_rs_z +
                                       cs.d.rsz_rt_refused_shape + cs.d.rsz_rt_refused_cell +
                                       cs.d.rsz_rt_refused_n_rs_seq + cs.d.rsz_rt_refused_envelope;
                if (named == 0) {
                    std::fprintf(stderr,
                                 "FAIL: lane inactive with no named refusal (silent fallback)\n");
                    ok = false;
                }
                std::printf("RSZ_RT_CANNOT_ARM arch=%llu no_rs_z=%llu shape=%llu cell=%llu "
                            "n_rs_seq=%llu envelope=%llu\n",
                            (unsigned long long) cs.d.rsz_rt_refused_arch,
                            (unsigned long long) cs.d.rsz_rt_refused_no_rs_z,
                            (unsigned long long) cs.d.rsz_rt_refused_shape,
                            (unsigned long long) cs.d.rsz_rt_refused_cell,
                            (unsigned long long) cs.d.rsz_rt_refused_n_rs_seq,
                            (unsigned long long) cs.d.rsz_rt_refused_envelope);
            }
            llama_free(dis);
        }
    }

    std::printf("RSZ_RT_TEST %s\n", ok ? "PASS" : "FAIL");

    llama_free(on);
    llama_free(off);
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
