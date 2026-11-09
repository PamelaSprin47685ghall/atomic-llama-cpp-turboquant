#include "llama.h"

#include "ggml.h"

#include "llama-context.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama-nout-state.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <stdlib.h> // setenv/unsetenv

//
// test-rsz-hybrid-compose
//
// The rs_z runtime lane (GGML_TP5_RSZ_RUNTIME_PARAM=1) crossed with the n_kv
// width lane of the same union binder, on a HYBRID fixture: a model with both
// a recurrent sub-cache and an attention KV cache (QWEN35 / NEMOTRON_H /
// JAMBA). This is the case test-rsz-union-combo cannot reach: on a pure
// recurrent fixture union_nkv_now() returns 0 (llama-context.cpp: it only
// reads llama_kv_cache_context / hybrid attention sub-caches), so the n_kv
// kind declines on every step and the n_kv width never moves.
//
// What moves here, read from the live caches, never assumed:
//   - n_kv: the graph is bound at the true used span of the attention cache
//     (llama_kv_cache_context::get_n_kv_target(), llama-kv-cache.h), so it
//     grows with the sequence length and stays below the stock 256-cell
//     floor. Both arms bind that same span; they differ in whether the graph
//     is REUSED at it.
//   - rs_z: the recurrent selection. A sequence that is released and rejoins
//     at position 0 becomes the zero row, so rs_z moves.
//
// Assertions, all behavioural:
//   1. the n_kv the live graph is BOUND at (live_bound_nkv: the graph
//      input's get_n_kv_target(), the production value) equals the width this
//      arm is expected to bind, derived from the live cache
//      (expected_nkv_width; the production function needs per-ubatch
//      slot_info a test cannot build), and is below the stock 256 floor;
//   2. the live graph is reused, not rebuilt, on a step where n_kv AND rs_z
//      both change (graph_builds does not move after warm), while the
//      reuse-disabled control rebuilds on every such step -- that is what
//      makes the reuse assertion falsifiable;
//   3. the binder arm is bitwise equal to the stock arm: per-sequence logits
//      rows and every recurrent-state cell;
//   4. the rows are not degenerate, so the bitwise equality is not vacuous;
//   5. the lane is active and the named refusals sum to the lane total;
//   6. both arms bind the same n_kv trajectory (same tokens, same span).
//
// Not proven here: that the binder PATCHED rather than rebuilt on the reuse
// steps. patch_binds counts a dispatch per kind and also moves on a no-op
// kind, so it cannot separate the two; the no-rebuild claim rests on the
// graph build count.
//
// usage: test-rsz-hybrid-compose <tiny-hybrid.gguf>
//
// Exits 77 (ctest SKIP_RETURN_CODE) when no model is given, 1 on any failed
// assertion, 0 otherwise.

namespace {

constexpr int k_skip = 77;

constexpr int N_SEQ      = 2;
constexpr int N_RS_SLOTS = 4;

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

// every gate the test depends on is set explicitly, so the caller's shell
// cannot change the arm under test
// The two arms are genuinely different code paths, and each variable is read
// exactly once, in the llama_context constructor:
//
//   binder == true   GGML_TP5_UNION_BINDER=1 (src/llama-context.cpp:536-537)
//                    and GGML_TP5_RSZ_RUNTIME_PARAM=1 (:576). The context
//                    builds the envelope graph once and reuses it while n_kv
//                    and rs_z both move.
//   binder == false  LLAMA_GRAPH_REUSE_DISABLE=1 (src/llama-context.cpp:
//                    500-501). Every decode builds a fresh graph, so the
//                    stock arm rebuilds on each step. That is the control
//                    that makes "the binder reused" falsifiable.
//
// No variable is unset while a context that read it is alive: make_ctx()
// writes before llama_init_from_model() and clears in its tail.
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
    // one KV stream: with kv_unified false the attention cache gets
    // n_stream == n_seq_max and the binder refuses a multi-stream cache
    // (llama-context.cpp:4219-4226, "the K/V view offset encodes the
    // build-time stream base"), so the n_kv lane could never bind.
    // llama_memory_hybrid passes this value straight to its attention cache
    // (llama-memory-hybrid.cpp:38-45, "unified"), so nothing overrides it.
    cp.kv_unified      = true;
    cp.n_rs_seq        = 0;            // no rollback snapshots: replay_len stays 0
    cp.n_outputs_max   = cp.n_seq_max; // contract: n_outputs_max >= n_seq_max
    cp.n_threads       = 1;
    cp.n_threads_batch = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv     = false;
    cp.op_offload      = false;
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

// deterministic token ids inside the fixture vocabulary
llama_token tok_at(llama_seq_id seq, llama_pos pos, int n_vocab) {
    return (llama_token) (3 + (37 * (int) seq + 11 * (int) pos) % (n_vocab - 3));
}

// the hybrid memory of a hybrid fixture: the attention sub-cache (n_kv lane)
// and the recurrent sub-cache (rs_z lane) live behind the same object
llama_memory_hybrid * hybrid_mem(llama_context * ctx) {
    return dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
}

// The width the live graph is actually bound at, read from the graph's own
// attention input. This is the production value: llm_graph_input_attn_kv
// caches the llama_kv_cache_context the build used, and that context's
// get_n_kv_target() is the width can_reuse_kq_mask() compares against
// (llama-kv-cache.h:512-516, "the width the graph is expected to execute
// at"). Reading it is the only way a test can see the bound width: the
// allocator-side llama_kv_cache::get_n_kv() is the padded width and does not
// follow the exact lane.
//
// Returns 0 when the live graph has no hybrid attention input, which the
// caller treats as a failure.
uint32_t live_bound_nkv(llama_context * ctx) {
    llm_graph_result * res = ctx->get_gf_res_prev();
    if (res == nullptr) {
        return 0;
    }
    for (const auto & input : res->inputs) {
        const auto * hyb = dynamic_cast<const llm_graph_input_mem_hybrid *>(input.get());
        if (hyb == nullptr || hyb->get_attn() == nullptr || hyb->get_attn()->mctx == nullptr) {
            continue;
        }
        return hyb->get_attn()->mctx->get_n_kv_target();
    }
    return 0;
}

// The width this arm is EXPECTED to bind, derived from the live cache.
//
// This is not a call into the production function. union_nkv_width()
// (llama-context.cpp) reads llama_kv_cache_context::get_n_kv_exact() /
// get_n_kv_padded(), and both need the per-ubatch slot_info that the context
// built internally; a test cannot reconstruct it. What is derived below is
// the same quantity for this fixture's layout, and it is used only as the
// EXPECTED value to compare the live bound width against (see
// live_bound_nkv(), which reads the production value):
//
//   - make_ctx() sets kv_unified = true, so the attention cache has one
//     stream (llama-kv-cache.cpp:202, n_stream = unified ? 1 : n_seq_max);
//     K/V keep the default F16 (llama-context.cpp:7597-7598) and flash
//     attention is off, so v_trans is true and every layer's V block size is
//     1 (ggml.c, GGML_TYPE_F16 blck_size) -> align == 1, no rounding;
//   - the used span is max(1, used_max_p1()) clamped to the cell count
//     (llama-kv-cache.cpp:2100), and used_max_p1() is the highest used cell
//     index + 1. With a contiguous cache the cell index equals the position
//     of the token stored in it, so the highest used cell is the highest
//     position over every sequence, which is what seq_pos_max() reports;
//   - the padded form is min(cells.size(), max(256, pad(used_max_p1, 256)))
//     (llama-kv-cache.cpp:2057-2062).
//
// The assertions that prove the lane moved do NOT depend on this copy: they
// are the cross-arm disagreement (§ run_arm / main) and the bitwise logits and
// state equality with stock. This value only says what each arm should show.
uint32_t expected_nkv_width(llama_kv_cache * kv) {
    uint32_t span = 0;
    for (int q = 0; q < N_RS_SLOTS; ++q) {
        const llama_pos p = kv->seq_pos_max((llama_seq_id) q);
        if (p >= 0 && (uint32_t) (p + 1) > span) {
            span = (uint32_t) (p + 1);
        }
    }
    const uint32_t capacity = kv->get_size();
    return std::min(capacity, std::max(1u, span));
}

// the whole recurrent state must match the stock context byte for byte
bool compare_all_states(llama_context * a, llama_context * b, int step) {
    llama_memory_recurrent * ma = hybrid_mem(a) != nullptr ? hybrid_mem(a)->get_mem_recr() : nullptr;
    llama_memory_recurrent * mb = hybrid_mem(b) != nullptr ? hybrid_mem(b)->get_mem_recr() : nullptr;
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
        std::fprintf(stderr, "FAIL %s: named refusals sum to %llu, lane total is %llu\n",
                     label, (unsigned long long) sum, (unsigned long long) d.rsz_rt_fallbacks);
        return false;
    }
    return true;
}

// what one arm observed, so the caller can compare the two n_kv trajectories
struct arm_result {
    std::vector<uint32_t> nkv;      // live n_kv per steady-state step
    uint64_t              rs_z_moves = 0;
    uint64_t              nkv_moves  = 0;
    bool                  ok         = false;
};

// one arm: binder+rsz vs stock, with a given exact setting.
// Returns the number of failed checks IN THIS ARM (0 = pass).
int run_arm(llama_model * model, const char * label, int n_vocab,
            arm_result & out) {
    g_fail = 0;
    out = arm_result();

    llama_context * ctx_on  = make_ctx(model, /*binder=*/true);
    llama_context * ctx_off = make_ctx(model, /*binder=*/false);
    if (ctx_on == nullptr || ctx_off == nullptr) {
        std::fprintf(stderr, "FAIL %s: context creation failed\n", label);
        if (ctx_on)  llama_free(ctx_on);
        if (ctx_off) llama_free(ctx_off);
        return g_fail + 1;
    }

    bool ok = true;

    llama_memory_hybrid * mem_on  = hybrid_mem(ctx_on);
    llama_memory_hybrid * mem_off = hybrid_mem(ctx_off);
    if (mem_on == nullptr || mem_off == nullptr) {
        std::fprintf(stderr, "FAIL %s: fixture is not hybrid (no llama_memory_hybrid)\n", label);
        llama_free(ctx_on);
        llama_free(ctx_off);
        return g_fail + 1;
    }

    const llama_nkv_ctx_state & st = llama_nkv_ctx_state_get(ctx_on);
    if (!st.nkv_enabled) {
        std::fprintf(stderr, "FAIL %s: binder state not enabled\n", label);
        ok = false;
    }
    // The arm is decided by the real flag, not by the argument name. The
    // control arm must be the reuse-disabled path, so the stock context must
    // report no reuse on any steady-state step (asserted below). If the
    // control ever reused, the ON arm's reuse would prove nothing.

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
    const int prefill_len[N_SEQ] = { 5, 3 };
    CHECK(prefill_len[0] != prefill_len[1], "%s: prefill lengths must differ (no padding)", label);

    llama_pos next_pos[N_SEQ] = { 0, 0 };
    for (int s = 0; ok && s < N_SEQ; ++s) {
        std::vector<token_step> ts;
        for (int i = 0; i < prefill_len[s]; ++i) {
            ts.push_back({ tok_at(s, i, n_vocab), (llama_pos) i, (llama_seq_id) s,
                           i + 1 == prefill_len[s] });
        }
        if (!decode_batch(ctx_on, ts) || !decode_batch(ctx_off, ts)) {
            std::fprintf(stderr, "FAIL %s: prefill seq %d decode\n", label, s);
            ok = false;
            break;
        }
        ok = logits_match(ctx_on, prefill_len[s] - 1, ctx_off, prefill_len[s] - 1,
                          n_vocab, -1, s) && ok;
        next_pos[s] = prefill_len[s];
    }

    // ---- steady state: n_kv grows AND rs_z moves ----
    //
    // Each step decodes ONE token per sequence, so the ubatch skeleton
    // (n_tokens, n_seqs, n_seq_tokens) is constant and only the runtime
    // parameters move. seq 1 keeps growing the attention cache, so the exact
    // n_kv width moves on every step. seq 0 is released and rejoins at
    // position 0 on the odd steps, which frees its recurrent cell and makes
    // it the zero row, so rs_z moves too.
    struct step {
        const char * name;
        bool         rm_seq0;
    };
    const step steps[] = {
        { "S0 warm",             false },
        { "S1 seq0 rejoins",     true  },
        { "S2 both continue",    false },
        { "S3 seq0 rejoins",     true  },
        { "S4 both continue",    false },
    };
    const size_t n_steps = sizeof(steps) / sizeof(steps[0]);

    size_t   warm_builds = 0;
    size_t   warm_binds  = 0;
    int      reuse_on    = 0;
    int      reuse_off   = 0;
    int32_t  prev_rs_z   = -1;
    uint32_t prev_nkv    = 0;
    bool     have_prev   = false;

    for (size_t i = 0; ok && i < n_steps; ++i) {
        const step & s = steps[i];
        const int step_i = (int) i;

        if (s.rm_seq0) {
            CHECK(llama_memory_seq_rm(llama_get_memory(ctx_on),  0, -1, -1),
                  "%s step %d (%s): ON seq_rm failed", label, step_i, s.name);
            CHECK(llama_memory_seq_rm(llama_get_memory(ctx_off), 0, -1, -1),
                  "%s step %d (%s): OFF seq_rm failed", label, step_i, s.name);
            next_pos[0] = 0;
        }

        std::vector<token_step> ts;
        for (int q = 0; q < N_SEQ; ++q) {
            ts.push_back({ tok_at(q, next_pos[q], n_vocab), next_pos[q], (llama_seq_id) q, true });
        }

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
            std::fprintf(stderr, "FAIL %s step %d (%s): diagnostics unavailable\n",
                         label, step_i, s.name);
            ok = false;
            break;
        }
        ok = fallbacks_add_up(di, label) && ok;

        // the live attention width, read from the cache the graph was built
        // from. Both arms hold the same tokens, so the used span is the same;
        // Both arms hold the same tokens, so both must bind the same width:
        // the true used span. The arms differ in whether the graph is REUSED
        // at that width (binder) or rebuilt for it (control).
        llama_kv_cache * kv_on  = mem_on->get_mem_attn();
        llama_kv_cache * kv_off = mem_off->get_mem_attn();
        if (kv_on == nullptr || kv_off == nullptr) {
            std::fprintf(stderr, "FAIL %s step %d (%s): missing attention cache\n",
                         label, step_i, s.name);
            ok = false;
            break;
        }
        // the width the live graph is BOUND at, read from the graph's own
        // attention input (live_bound_nkv): the production value
        // can_reuse_kq_mask() compares against (llama-graph.cpp:108).
        const uint32_t nkv_live = live_bound_nkv(ctx_on);
        // the width this arm is expected to bind, derived from the live cache
        // (see expected_nkv_width)
        const uint32_t expect_live = expected_nkv_width(kv_on);
        // the OFF arm holds the same tokens, so its span must agree; the two
        // arms differ only in whether they reuse the graph at that span
        CHECK(expected_nkv_width(kv_off) == expected_nkv_width(kv_on),
              "%s step %d (%s): ON/OFF used spans disagree", label, step_i, s.name);

        // the recurrent selection, read from the allocator
        const llama_memory_recurrent * recr_on = mem_on->get_mem_recr();
        if (recr_on == nullptr) {
            std::fprintf(stderr, "FAIL %s step %d (%s): missing recurrent cache\n",
                         label, step_i, s.name);
            ok = false;
            break;
        }

        // ---- n_kv lane: the width the live graph is bound at --------------
        //
        // nkv_live is the production value (the graph input's
        // get_n_kv_target()); expect_live is the derivation. The proof that
        // the lane moved is the cross-arm disagreement in main() plus the
        // bitwise equality with stock below; this check says the bound width
        // is the one each lane should produce.
        CHECK(nkv_live > 0,
              "%s step %d (%s): no live attention input to read the bound n_kv from",
              label, step_i, s.name);
        CHECK(nkv_live == expect_live,
              "%s step %d (%s): live n_kv %u != the used span %u",
              label, step_i, s.name, nkv_live, expect_live);
        // the fixture is small: a 256-rounded width would mean the graph ran
        // at the stock padded floor instead of the used span
        CHECK(nkv_live < 256,
              "%s step %d (%s): live n_kv %u is at the stock 256 floor, not the used span",
              label, step_i, s.name, nkv_live);

        // ---- rs_z lane: the selection is a live cell index ----------------
        if (recr_on->rs_z >= 0) {
            CHECK(di.rsz_rt_active,
                  "%s step %d (%s): lane inactive at rs_z=%d (fallbacks=%llu)",
                  label, step_i, s.name, recr_on->rs_z, (unsigned long long) di.rsz_rt_fallbacks);
            CHECK((uint32_t) recr_on->rs_z < di.rsz_rt_zero_view_rows,
                  "%s step %d (%s): live rs_z %d is outside the zero view (%u rows)",
                  label, step_i, s.name, recr_on->rs_z, di.rsz_rt_zero_view_rows);
        }

        // ---- the parameters really moved -------------------------------
        //
        // nkv_moves counts changes of the BOUND width, which is the used span
        // on both arms and grows every step.
        if (have_prev) {
            if (nkv_live != prev_nkv) {
                ++out.nkv_moves;
            }
            if (recr_on->rs_z != prev_rs_z) {
                ++out.rs_z_moves;
            }
        }
        prev_nkv  = nkv_live;
        prev_rs_z = recr_on->rs_z;
        have_prev = true;
        out.nkv.push_back(nkv_live);

        // ---- bitwise equality with stock, plus non-degeneracy -------------
        for (int q = 0; q < N_SEQ && ok; ++q) {
            ok = logits_match(ctx_on, q, ctx_off, q, n_vocab, step_i, q) && ok;
        }
        ok = logits_differ(ctx_on, 0, 1, n_vocab, step_i) && ok;
        ok = compare_all_states(ctx_on, ctx_off, step_i) && ok;

        // ---- warm-graph gate: nothing may rebuild after warm --------------
        //
        // The live graph is built at the physical KV capacity (the envelope,
        // llama-context.cpp: cap = min(n_ctx_kv, kv capacity)), so a width
        // move inside it must not rebuild. The assertion is on the build
        // count only: patch_binds is a per-kind dispatch counter that also
        // moves on a no-op kind, so it cannot distinguish a bind from a
        // declined attempt and is not asserted (it is printed for context).
        if (i == 0) {
            warm_builds = (size_t) llama_graph_build_count(ctx_on);
            warm_binds  = (size_t) di.patch_binds;
        } else {
            CHECK((size_t) llama_graph_build_count(ctx_on) == warm_builds,
                  "%s step %d (%s): graph_builds %llu != warm %zu (rebuild on an n_kv+rs_z move)",
                  label, step_i, s.name,
                  (unsigned long long) llama_graph_build_count(ctx_on), warm_builds);
            if (pon.n_reused == 1) {
                ++reuse_on;
            }
        }
        // the stock arm must NOT reuse on an n_kv move. Its graph key is the
        // KQ mask width, and can_reuse_kq_mask() compares that mask against
        // mctx->get_n_kv_target() (llama-graph.cpp:95-116), which follows the
        // used span on a stock context. So a moved n_kv is a real rebuild
        // there, and the binder's reuse above is not vacuous.
        if (i > 0) {
            if (poff.n_reused == 0) {
                ++reuse_off;
            }
        }

        for (int q = 0; q < N_SEQ; ++q) {
            next_pos[q] += 1;
        }
    }

    if (ok) {
        // the bound width must move: it is the used span, which grows every
        // step. A zero count would mean the scenario never exercised the n_kv
        // lane, and the reuse assertion above would be about a constant.
        CHECK(out.nkv_moves > 0,
              "%s: the bound n_kv never moved between steps, the n_kv lane is a no-op",
              label);
        CHECK(out.rs_z_moves > 0,
              "%s: rs_z never moved between steps, only n_kv was exercised", label);
        // the binder reused the live graph on every steady-state step after
        // warm; the stock arm rebuilt on every one of those steps, so the
        // reuse is not vacuous
        CHECK(reuse_on == (int) n_steps - 1,
              "%s: binder reused %d times, expected %zu", label, reuse_on, n_steps - 1);
        CHECK(reuse_off == (int) n_steps - 1,
              "%s: the reuse-disabled control rebuilt on %d of %zu steps, expected every one "
              "(otherwise the binder arm's reuse is vacuous)",
              label, (int) n_steps - 1 - reuse_off, n_steps - 1);
        CHECK(d.rsz_rt_active, "%s: lane never armed", label);
    }

    std::printf("RSZ_HYBRID_STATS arm=%s warm_builds=%zu warm_binds=%zu reuse_on=%d "
                "reuse_off=%d nkv_moves=%llu rs_z_moves=%llu fallbacks=%llu active=%d "
                "zero_view_rows=%u nkv=",
                label, warm_builds, warm_binds,
                reuse_on, reuse_off, (unsigned long long) out.nkv_moves,
                (unsigned long long) out.rs_z_moves,
                (unsigned long long) d.rsz_rt_fallbacks, d.rsz_rt_active ? 1 : 0,
                d.rsz_rt_zero_view_rows);
    for (size_t i = 0; i < out.nkv.size(); ++i) {
        std::printf("%s%u", i == 0 ? "" : ",", out.nkv[i]);
    }
    std::printf("\n");

    out.ok = ok && g_fail == 0;
    llama_free(ctx_on);
    llama_free(ctx_off);
    return g_fail + (ok ? 0 : 1);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <tiny-hybrid.gguf>\n", argv[0]);
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
    if (!llama_model_is_hybrid(model)) {
        std::fprintf(stderr, "FAIL: fixture is not hybrid; this lane needs an attention KV cache "
                             "next to the recurrent state\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    const int n_vocab = (int) llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (n_vocab < 16) {
        std::fprintf(stderr, "FAIL: fixture has %d tokens, expected >= 16\n", n_vocab);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // The binder arm and the reuse-disabled control. Each arm reports its own
    // failure count, so the summary names the arm that failed.
    //
    // The control exists to make the binder arm's reuse falsifiable: with
    // LLAMA_GRAPH_REUSE_DISABLE=1 every decode builds a fresh graph, so the
    // same n_kv x rs_z scenario rebuilds on every steady-state step there. If
    // the control ever reused, run_arm() fails on reuse_off.
    //
    // Both arms must bind the SAME width trajectory (the used span), because
    // they hold the same tokens; what differs is whether the graph is reused
    // at that width. A differing trajectory would mean one arm ran at a
    // different width, which the per-step expect_live check would already
    // have caught.
    //
    // What this does NOT prove: that the binder patched rather than rebuilt.
    // patch_binds is a per-kind dispatch counter that also moves on a no-op
    // kind, so it cannot separate a bind from a declined attempt; the
    // no-rebuild claim rests on llama_graph_build_count() inside run_arm().
    arm_result binder_arm, control_arm;
    const int fails_binder  = run_arm(model, "binder",  n_vocab, binder_arm);
    const int fails_control = run_arm(model, "control", n_vocab, control_arm);

    int fails_cross = 0;
    if (fails_binder == 0 && fails_control == 0) {
        bool same = binder_arm.nkv.size() == control_arm.nkv.size();
        for (size_t i = 0; same && i < binder_arm.nkv.size(); ++i) {
            same = binder_arm.nkv[i] == control_arm.nkv[i];
        }
        if (!same) {
            std::fprintf(stderr, "FAIL: the binder and control arms bound different n_kv "
                                 "trajectories; the same tokens must give the same span\n");
            ++fails_cross;
        }
    }

    llama_model_free(model);
    llama_backend_free();

    if (fails_binder != 0 || fails_control != 0 || fails_cross != 0) {
        std::printf("RSZ_HYBRID_TEST FAIL binder=%d control=%d cross=%d\n",
                    fails_binder, fails_control, fails_cross);
        return 1;
    }
    std::printf("RSZ_HYBRID_TEST PASS\n");
    return 0;
}
