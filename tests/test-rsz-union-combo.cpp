#include "llama.h"

#include "ggml.h"

#include "llama-memory-recurrent.h"
#include "llama-nout-state.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <stdlib.h> // setenv/unsetenv

//
// test-rsz-union-combo
//
// The rs_z runtime lane (GGML_TP5_RSZ_RUNTIME_PARAM=1) crossed with the
// other recurrent runtime parameters of the same union binder, on the real
// CPU decode path.
//
// What is actually crossed, stated precisely: on a PURE recurrent fixture the
// n_kv lane does not move. union_nkv_now() only reads llama_kv_cache_context
// / hybrid attention sub-caches (llama-context.cpp), so a MAMBA graph has no
// n_kv width at all and the n_kv kind declines on every step. The runtime
// parameters that move here are the recurrent ones of the same union binder:
// rs_z (the selection the rs_z lane turns into an input write) and rs_head
// (the head-dependent copy destination, bound by the rs_head shape list).
//
// The question this test answers: when rs_z and rs_head both move between
// two decodes, does the graph still bind in place, and is the result still
// bitwise equal to the stock arm? The control arm runs with
// LLAMA_GRAPH_REUSE_DISABLE=1, so it rebuilds on every decode; that is what
// makes "the binder did not rebuild" a claim that could fail.
//
// Fixture: three live sequences with distinct prompt lengths (5 / 2 / 7, no
// padding), then one token per sequence per step. A sequence that is
// released and rejoins at position 0 becomes the zero row, so rs_z takes
// 0 / 1 / 2 and the `-1` no-op. The stock arm rebuilds on every rs_z move,
// the binder arm must not. (A hybrid fixture with a real KV cache would be
// needed to cross n_kv itself; that is not what this test claims.)
//
// Assertions, all behavioural (no incidental values):
//   1. after the warm decode, graph_builds and patch_binds do not move on a
//      step where rs_z AND rs_head both change;
//   2. the binder arm is bitwise equal to the stock arm: per-sequence logits
//      rows and every recurrent-state cell;
//   3. the rows are not degenerate: the three sequences' logits differ, and
//      the stock arm reports no graph reuse on the rs_z moves, so the
//      scenario is not a no-op;
//   4. the lane is active and has counted no refusal;
//   5. the reuse-disabled control rebuilds on every steady-state step, so the
//      binder arm's reuse is not vacuous.
//
// usage: test-rsz-union-combo <tiny-recurrent.gguf>
//
// Exits 77 (ctest SKIP_RETURN_CODE) when no model is given, 1 on any failed
// assertion, 0 otherwise.

namespace {

constexpr int k_skip = 77;
constexpr int N_SEQ      = 3;
constexpr int N_RS_SLOTS = 8;
// the fixture's state cells. rsz_rt_zero_view_rows reports the CELL CAPACITY
// (mem->size), not a view height: the lane has no zero view, the zero-row op
// clears one row of the state tensor, so with n_rs_seq == 0 the capacity and
// the state height coincide.

// per-arm failure count: run_arm() resets it on entry and reports its own
// number, so one arm's failure never masks the other's result
int g_fail = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n"); \
        ++g_fail; \
    } \
} while (0)

struct token_step {
    llama_token  tok;
    llama_pos    pos;
    llama_seq_id seq;
    bool         logits;
};

// Every gate the test depends on is set explicitly before the context is
// constructed, so the caller's shell cannot change the arm under test.
//
// The two arms are genuinely different code paths, and each variable is read
// exactly once, in the llama_context constructor:
//
//   binder == true   GGML_TP5_UNION_BINDER=1 (src/llama-context.cpp:536-537)
//                    and GGML_TP5_RSZ_RUNTIME_PARAM=1 (:576). The context
//                    builds the envelope graph once and reuses it across the
//                    rs_z / rs_head moves.
//   binder == false  LLAMA_GRAPH_REUSE_DISABLE=1 (src/llama-context.cpp:
//                    500-501). Every decode builds a fresh graph, so the
//                    stock arm rebuilds on each parameter move. That is the
//                    control: without a rebuild on the stock side, "the
//                    binder did not rebuild" would prove nothing.
//
// No variable is unset while a context that read it is alive: make_ctx()
// writes before llama_init_from_model() and clears in its tail, so the next
// call starts clean.
llama_context * make_ctx(llama_model * model, bool binder) {
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    unsetenv("GGML_TP5_RSZ_RUNTIME_PARAM");
    unsetenv("GGML_TP5_UNION_BINDER");
    if (binder) {
        setenv("GGML_TP5_UNION_BINDER", "1", 1);
        setenv("GGML_TP5_RSZ_RUNTIME_PARAM", "1", 1);
    } else {
        setenv("LLAMA_GRAPH_REUSE_DISABLE", "1", 1);
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 64;
    cp.n_batch         = 8;
    cp.n_ubatch        = 8;
    cp.n_seq_max       = N_RS_SLOTS;
    cp.n_rs_slots      = N_RS_SLOTS;
    cp.n_rs_seq        = 0;            // no rollback snapshots: replay_len stays 0
    cp.n_outputs_max   = cp.n_seq_max; // contract: n_outputs_max >= n_seq_max
    cp.n_threads       = 1;
    cp.n_threads_batch = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv     = false;
    cp.no_perf         = true;

    llama_context * ctx = llama_init_from_model(model, cp);

    // the constructor has captured every variable above
    unsetenv("GGML_TP5_UNION_BINDER");
    unsetenv("GGML_TP5_RSZ_RUNTIME_PARAM");
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    return ctx;
}

bool decode_batch(llama_context * ctx, const std::vector<token_step> & ts) {
    llama_batch b = llama_batch_init((int32_t) ts.size(), 0, 1);
    if (!llama_batch_is_complete(&b, (int32_t) ts.size(), false, true)) {
        llama_batch_free(b);
        GGML_ABORT("llama_batch_init failed");
    }
    b.n_tokens = (int32_t) ts.size();
    for (size_t i = 0; i < ts.size(); ++i) {
        b.token[i]     = ts[i].tok;
        b.pos[i]       = ts[i].pos;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = ts[i].seq;
        b.logits[i]    = ts[i].logits ? 1 : 0;
    }
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc == 0;
}

// deterministic token ids inside the tiny vocabulary (3 .. 261)
llama_token tok_at(llama_seq_id seq, llama_pos pos) {
    return (llama_token) (3 + (37 * (int) seq + 11 * (int) pos) % 259);
}

llama_memory_recurrent * recr(llama_context * ctx) {
    return dynamic_cast<llama_memory_recurrent *>(llama_get_memory(ctx));
}

// the whole recurrent state must match the stock context byte for byte
bool compare_all_states(llama_context * a, llama_context * b, int step) {
    auto * ma = recr(a);
    auto * mb = recr(b);
    if (ma == nullptr || mb == nullptr) {
        std::fprintf(stderr, "FAIL step %d: missing recurrent memory\n", step);
        return false;
    }
    if (ma->r_l.size() != mb->r_l.size() || ma->s_l.size() != mb->s_l.size()) {
        std::fprintf(stderr, "FAIL step %d: layer count differs\n", step);
        return false;
    }
    for (size_t il = 0; il < ma->r_l.size(); ++il) {
        for (int which = 0; which < 2; ++which) {
            const ggml_tensor * ta = which == 0 ? ma->r_l[il] : ma->s_l[il];
            const ggml_tensor * tb = which == 0 ? mb->r_l[il] : mb->s_l[il];
            if (ta == nullptr || tb == nullptr) {
                std::fprintf(stderr, "FAIL step %d: layer %zu %s tensor is null\n",
                             step, il, which == 0 ? "r_l" : "s_l");
                return false;
            }
            const size_t na = ggml_nbytes(ta);
            const size_t nb = ggml_nbytes(tb);
            if (na != nb || std::memcmp(ta->data, tb->data, na) != 0) {
                std::fprintf(stderr, "FAIL step %d: layer %zu %s state differs (%zu vs %zu bytes)\n",
                             step, il, which == 0 ? "r_l" : "s_l", na, nb);
                return false;
            }
        }
    }
    return true;
}

bool logits_match(llama_context * a, int32_t ia, llama_context * b, int32_t ib,
                  int n_vocab, int step, int seq) {
    const float * la = llama_get_logits_ith(a, ia);
    const float * lb = llama_get_logits_ith(b, ib);
    if (la == nullptr || lb == nullptr) {
        std::fprintf(stderr, "FAIL step %d seq %d: missing logits\n", step, seq);
        return false;
    }
    if (std::memcmp(la, lb, (size_t) n_vocab * sizeof(float)) != 0) {
        std::fprintf(stderr, "FAIL step %d seq %d: binder logits differ from stock\n", step, seq);
        return false;
    }
    return true;
}

bool logits_differ(llama_context * a, int32_t ia, int32_t ib, int n_vocab, int step) {
    const float * la = llama_get_logits_ith(a, ia);
    const float * lb = llama_get_logits_ith(a, ib);
    if (la == nullptr || lb == nullptr) {
        std::fprintf(stderr, "FAIL step %d: missing logits for the non-degeneracy check\n", step);
        return false;
    }
    if (std::memcmp(la, lb, (size_t) n_vocab * sizeof(float)) == 0) {
        std::fprintf(stderr, "FAIL step %d: rows %d and %d are identical (vacuous comparison)\n",
                     step, ia, ib);
        return false;
    }
    return true;
}

bool fallbacks_add_up(const llama_union_diagnostics & d, const char * label) {
    const uint64_t sum = d.rsz_rt_refused_gate + d.rsz_rt_refused_arch +
                         d.rsz_rt_refused_no_rs_z + d.rsz_rt_refused_cell +
                         d.rsz_rt_refused_envelope + d.rsz_rt_refused_shape +
                         d.rsz_rt_refused_n_rs_seq + d.rsz_rt_refused_ring +
                         d.rsz_rt_refused_mixed_depth;
    if (sum != d.rsz_rt_fallbacks) {
        std::fprintf(stderr, "FAIL %s: named refusals sum to %llu, lane total is %llu\n",
                     label, (unsigned long long) sum, (unsigned long long) d.rsz_rt_fallbacks);
        return false;
    }
    return true;
}

// one arm of the comparison: binder+rsz vs the reuse-disabled control.
// Returns the number of failed checks IN THIS ARM (0 = pass), so the caller
// can report which arm failed instead of a mixed total.
int run_arm(llama_model * model, const char * label, int n_vocab) {
    g_fail = 0;
    llama_context * ctx_on  = make_ctx(model, /*binder=*/true);
    llama_context * ctx_off = make_ctx(model, /*binder=*/false);
    if (ctx_on == nullptr || ctx_off == nullptr) {
        std::fprintf(stderr, "FAIL %s: context creation failed\n", label);
        if (ctx_on)  llama_free(ctx_on);
        if (ctx_off) llama_free(ctx_off);
        return g_fail + 1;
    }

    bool ok = true;

    // the internal binder state, read-only: the recorded rs_z / rs_head the
    // live graph is bound at. The lane's own view of those is in the public
    // diagnostics; this one also carries the head.
    const llama_nkv_ctx_state & st = llama_nkv_ctx_state_get(ctx_on);
    if (!st.nkv_enabled) {
        std::fprintf(stderr, "FAIL %s: binder state not enabled\n", label);
        ok = false;
    }

    llama_union_diagnostics d = {};
    if (!llama_get_union_diagnostics(ctx_on, &d)) {
        std::fprintf(stderr, "FAIL %s: diagnostics unavailable\n", label);
        ok = false;
    } else if (!d.enabled || !d.rsz_rt_enabled) {
        std::fprintf(stderr, "FAIL %s: binder or rs_z gate not on (enabled=%d rsz_rt=%d)\n",
                     label, d.enabled ? 1 : 0, d.rsz_rt_enabled ? 1 : 0);
        ok = false;
    }

    // ---- prefill: one prompt per sequence, distinct lengths, no padding ----
    const int prefill_len[N_SEQ] = { 5, 2, 7 };
    CHECK(prefill_len[0] != prefill_len[1] && prefill_len[1] != prefill_len[2] &&
          prefill_len[0] != prefill_len[2],
          "%s: prefill lengths must differ (no padding)", label);

    llama_pos next_pos[N_SEQ] = { 0, 0, 0 };
    for (int s = 0; ok && s < N_SEQ; ++s) {
        std::vector<token_step> ts;
        for (int i = 0; i < prefill_len[s]; ++i) {
            ts.push_back({ tok_at(s, i), (llama_pos) i, (llama_seq_id) s, i + 1 == prefill_len[s] });
        }
        if (!decode_batch(ctx_on, ts) || !decode_batch(ctx_off, ts)) {
            std::fprintf(stderr, "FAIL %s: prefill seq %d decode\n", label, s);
            ok = false;
            break;
        }
        ok = logits_match(ctx_on, prefill_len[s] - 1, ctx_off, prefill_len[s] - 1, n_vocab, -1, s) && ok;
        next_pos[s] = prefill_len[s];
    }

    // ---- steady state: rs_z AND rs_head move together ----
    struct step {
        const char * name;
        int          rm_seq;      // -1: every sequence continues
        int32_t      expect_rs_z;
    };
    const step steps[] = {
        { "S0 warm",                -1, -1 },
        { "S1 seq0 rejoins",         0,  0 },
        { "S2 seq1 rejoins",         1,  1 },
        { "S3 seq2 rejoins",         2,  2 },
        { "S4 seq0 rejoins again",   0,  0 },
        { "S5 all continue",        -1, -1 },
    };
    const size_t n_steps = sizeof(steps) / sizeof(steps[0]);

    size_t   warm_builds = 0;
    size_t   warm_binds  = 0;
    int      reuse_on    = 0;
    int      reuse_off   = 0;
    int32_t  prev_rs_z   = -2;
    // the recurrent runtime parameters that actually move on this graph
    uint64_t rs_z_moves  = 0;
    uint64_t head_moves  = 0;
    int32_t  prev_rs_z_live = -2;
    uint32_t prev_head      = 0;

    for (size_t i = 0; ok && i < n_steps; ++i) {
        const step & s = steps[i];
        const int step_i = (int) i;

        if (s.rm_seq >= 0) {
            const llama_seq_id q = (llama_seq_id) s.rm_seq;
            CHECK(llama_memory_seq_rm(llama_get_memory(ctx_on),  q, -1, -1),
                  "%s step %d (%s): ON seq_rm failed", label, step_i, s.name);
            CHECK(llama_memory_seq_rm(llama_get_memory(ctx_off), q, -1, -1),
                  "%s step %d (%s): OFF seq_rm failed", label, step_i, s.name);
            next_pos[q] = 0;
        }

        std::vector<token_step> ts;
        for (int q = 0; q < N_SEQ; ++q) {
            ts.push_back({ tok_at(q, next_pos[q]), next_pos[q], (llama_seq_id) q, true });
        }
        CHECK(ts[0].pos != ts[1].pos && ts[1].pos != ts[2].pos && ts[0].pos != ts[2].pos,
              "%s step %d (%s): positions are not distinct, the batch skeleton is not constant",
              label, step_i, s.name);

        llama_perf_context_reset(ctx_on);
        llama_perf_context_reset(ctx_off);
        if (!decode_batch(ctx_on, ts) || !decode_batch(ctx_off, ts)) {
            std::fprintf(stderr, "FAIL %s step %d (%s): decode\n", label, step_i, s.name);
            ok = false;
            break;
        }
        const llama_perf_context_data pon  = llama_perf_context(ctx_on);
        const llama_perf_context_data poff = llama_perf_context(ctx_off);

        llama_union_diagnostics di = {};
        if (!llama_get_union_diagnostics(ctx_on, &di)) {
            std::fprintf(stderr, "FAIL %s step %d (%s): diagnostics unavailable\n", label, step_i, s.name);
            ok = false;
            break;
        }
        ok = fallbacks_add_up(di, label) && ok;

        // the live recurrent selection, read from the allocator, never assumed
        auto * mem_on  = recr(ctx_on);
        auto * mem_off = recr(ctx_off);
        if (mem_on == nullptr || mem_off == nullptr) {
            std::fprintf(stderr, "FAIL %s step %d (%s): missing recurrent memory\n", label, step_i, s.name);
            ok = false;
            break;
        }
        CHECK(mem_on->rs_z  == s.expect_rs_z, "%s step %d (%s): ON rs_z %d != %d",
              label, step_i, s.name, mem_on->rs_z, s.expect_rs_z);
        CHECK(mem_off->rs_z == s.expect_rs_z, "%s step %d (%s): OFF rs_z %d != %d",
              label, step_i, s.name, mem_off->rs_z, s.expect_rs_z);

        // the recurrent runtime parameters really moved: rs_z (the value the
        // lane turns into an input write) and rs_head (the head-dependent
        // copy destination). Both are read from the live allocator.
        if (i > 0) {
            if (mem_on->rs_z != prev_rs_z_live) {
                ++rs_z_moves;
            }
            if (mem_on->head != prev_head) {
                ++head_moves;
            }
        }
        prev_rs_z_live = mem_on->rs_z;
        prev_head      = mem_on->head;

        // the cell capacity the lane reports must be exactly the fixture's
        // state cells (N_RS_SLOTS), and every rs_z the allocator names must be
        // a cell inside it. This is the invariant the decode-time _envelope
        // check enforces; an out-of-range value here would mean the check
        // let one through.
        CHECK(di.rsz_rt_zero_view_rows == (uint32_t) N_RS_SLOTS,
              "%s step %d (%s): zero view has %u rows, expected the %d state cells",
              label, step_i, s.name, di.rsz_rt_zero_view_rows, N_RS_SLOTS);
        if (mem_on->rs_z >= 0) {
            CHECK((uint32_t) mem_on->rs_z < di.rsz_rt_zero_view_rows,
                  "%s step %d (%s): live rs_z %d is outside the zero view (%u rows)",
                  label, step_i, s.name, mem_on->rs_z, di.rsz_rt_zero_view_rows);
        }
        // the head-dependent copy destination is a shape the binder still
        // binds (rs_head): the recorded value must follow the live head. This
        // one is real evidence, because rs_head is PATCHED by the binder
        // rather than copied from the live value at record time.
        CHECK(st.record_bound_rs_head == mem_on->head,
              "%s step %d (%s): recorded rs_head %u != live head %u",
              label, step_i, s.name, st.record_bound_rs_head, mem_on->head);

        // the lane must be live on every step with a nameable rs_z
        if (s.expect_rs_z >= 0) {
            CHECK(di.rsz_rt_active, "%s step %d (%s): lane inactive at rs_z=%d (fallbacks=%llu)",
                  label, step_i, s.name, s.expect_rs_z, (unsigned long long) di.rsz_rt_fallbacks);
            CHECK(di.rsz_rt_refused_envelope == 0,
                  "%s step %d (%s): _envelope refusal at a legal rs_z=%d",
                  label, step_i, s.name, s.expect_rs_z);
        }

        // bitwise equality with stock, row by row, plus non-degeneracy
        for (int q = 0; q < N_SEQ && ok; ++q) {
            ok = logits_match(ctx_on, q, ctx_off, q, n_vocab, step_i, q) && ok;
        }
        ok = logits_differ(ctx_on, 0, 1, n_vocab, step_i) && ok;
        ok = logits_differ(ctx_on, 0, 2, n_vocab, step_i) && ok;
        ok = logits_differ(ctx_on, 1, 2, n_vocab, step_i) && ok;
        ok = compare_all_states(ctx_on, ctx_off, step_i) && ok;

        // warm-graph gate: after the first steady-state decode nothing may
        // rebuild or bind, even though both rs_z and rs_head move
        if (i == 0) {
            warm_builds = (size_t) llama_graph_build_count(ctx_on);
            warm_binds  = (size_t) di.patch_binds;
        } else {
            CHECK((size_t) llama_graph_build_count(ctx_on) == warm_builds,
                  "%s step %d (%s): graph_builds %llu != warm %zu (rebuild on an rs_z+rs_head move)",
                  label, step_i, s.name, (unsigned long long) llama_graph_build_count(ctx_on), warm_builds);
            CHECK((size_t) di.patch_binds == warm_binds,
                  "%s step %d (%s): patch_binds %zu != warm %zu (an rs_z move called the binder)",
                  label, step_i, s.name, (size_t) di.patch_binds, warm_binds);
            if (pon.n_reused == 1) {
                ++reuse_on;
            }
        }
        // the stock arm must NOT reuse on an rs_z move: otherwise the
        // scenario would be a no-op and the equality above vacuous
        if (i > 0) {
            const int expect_off_reuse = prev_rs_z == s.expect_rs_z ? 1 : 0;
            CHECK(poff.n_reused == expect_off_reuse,
                  "%s step %d (%s): stock reused %d != %d (stock must rebuild on an rs_z move)",
                  label, step_i, s.name, poff.n_reused, expect_off_reuse);
            if (poff.n_reused == 0) {
                ++reuse_off;
            }
        }
        prev_rs_z = s.expect_rs_z;

        for (int q = 0; q < N_SEQ; ++q) {
            next_pos[q] += 1;
        }
    }

    if (ok) {
        CHECK(rs_z_moves > 0,
              "%s: rs_z never moved between steps, the scenario is a no-op", label);
        CHECK(head_moves > 0,
              "%s: rs_head never moved between steps, only rs_z was exercised", label);
        CHECK(reuse_on == (int) n_steps - 1,
              "%s: binder reused %d times, expected %zu", label, reuse_on, n_steps - 1);
        CHECK(reuse_off > 0, "%s: stock never rebuilt, the scenario is a no-op", label);
        CHECK(d.rsz_rt_active, "%s: lane never armed", label);
    }

    std::printf("RSZ_COMBO_STATS arm=%s warm_builds=%zu warm_binds=%zu reuse_on=%d "
                "reuse_off=%d rs_z_moves=%llu head_moves=%llu fallbacks=%llu active=%d "
                "zero_view_rows=%u\n",
                label, warm_builds, warm_binds,
                reuse_on, reuse_off, (unsigned long long) rs_z_moves,
                (unsigned long long) head_moves,
                (unsigned long long) d.rsz_rt_fallbacks, d.rsz_rt_active ? 1 : 0,
                d.rsz_rt_zero_view_rows);

    llama_free(ctx_on);
    llama_free(ctx_off);
    return g_fail + (ok ? 0 : 1);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <tiny-recurrent.gguf>\n", argv[0]);
        return k_skip;
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    ggml_backend_dev_t devices[] = { nullptr };
    mp.devices = devices;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        std::fprintf(stderr, "FAIL: cannot load model %s\n", argv[1]);
        llama_backend_free();
        return 1;
    }
    if (!llama_model_is_recurrent(model)) {
        std::fprintf(stderr, "FAIL: fixture is not recurrent; this lane is defined for recurrent graphs only\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    const int n_vocab = (int) llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (n_vocab < 262) {
        std::fprintf(stderr, "FAIL: fixture has %d tokens, expected >= 262\n", n_vocab);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // The binder arm and the reuse-disabled control. The control exists so
    // the "no rebuild on an rs_z+rs_head move" assertion is falsifiable: on
    // that arm llama_context builds a fresh graph per decode, so the same
    // scenario DOES rebuild. Each arm reports its own failure count, so the
    // summary names the arm that failed.
    const int fails_binder  = run_arm(model, "binder", n_vocab);
    const int fails_control = run_arm(model, "control", n_vocab);

    llama_model_free(model);
    llama_backend_free();

    if (fails_binder != 0 || fails_control != 0) {
        std::printf("RSZ_COMBO_TEST FAIL binder=%d control=%d\n", fails_binder, fails_control);
        return 1;
    }
    std::printf("RSZ_COMBO_TEST PASS\n");
    return 0;
}
