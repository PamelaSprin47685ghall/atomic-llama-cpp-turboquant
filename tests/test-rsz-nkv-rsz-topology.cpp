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
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <stdlib.h> // setenv/unsetenv

//
// test-rsz-nkv-rsz-topology
//
// The rs_z runtime lane (GGML_TP5_RSZ_RUNTIME_PARAM=1) crossed with the n_kv
// width lane of the same union binder, on a HYBRID fixture (recurrent state
// AND an attention KV cache: QWEN35 / NEMOTRON_H / JAMBA). The claim under
// test is TOPOLOGY INVARIANCE: on a step where both runtime parameters move,
// the live graph is not rebuilt, and the two parameters still reach the graph
// as input writes.
//
// The scenario moves both on the same steps, by construction:
//   - 2 sequences, distinct prefill lengths (5 and 3), no padding;
//   - then one token per sequence per step, so the ubatch skeleton
//     (n_tokens, n_seqs, n_seq_tokens) is constant;
//   - on odd steps seq 0 is released with llama_memory_seq_rm(mem, 0, -1, -1)
//     and rejoins at position 0, so the recurrent selection rs_z moves;
//   - seq 1 keeps growing the attention cache, so the bound n_kv moves on
//     every step.
//
// What is read, never assumed:
//   - the bound n_kv: walk ctx->get_gf_res_prev()->inputs, take the
//     llm_graph_input_mem_hybrid, and read get_attn()->mctx->get_n_kv_target()
//     (the production value can_reuse_kq_mask() compares against);
//   - rs_z and head: llama_memory_hybrid::get_mem_recr();
//   - the selection row: the live graph's llm_graph_input_rs::s_zero, read
//     through the hybrid input's get_recr(). The tensor is host-backed
//     (union_rsz_bind_index() asserts it), so its int32 is the ROW the graph
//     will clear. With n_rs_seq == 0 that row is the cell index rs_z.
//
// Assertions, all behavioural:
//   a. topology invariance: after the warm decode, llama_graph_build_count()
//      does not move on any step where BOTH n_kv and rs_z move; the stock arm
//      rebuilds on those steps (llama_perf_context().n_reused == 0), so the
//      invariance is not vacuous;
//   b. both parameters really moved: >= 2 distinct bound n_kv values and >= 2
//      distinct rs_z values, else the scenario degenerated and the test fails
//      loudly;
//   c. the lane is live: rsz_rt_active on every step with rs_z >= 0,
//      rsz_rt_zero_view_rows (the CELL CAPACITY, mem->size, not a view
//      height: there is no zero view on this lane) equals the recurrent cell
//      count, every live
//      rs_z < zero_view_rows, rsz_rt_refused_envelope == 0;
//   d. the selection reached the graph: on every step
//      di.rsz_rt_bound_rs_z == live rs_z AND the live graph's s_zero host
//      value == live rs_z. THIS is the assertion that would have caught the
//      hybrid defect: the hybrid set_input() sites copied s_copy but never
//      wrote the selection row, so can_reuse() accepted a moved rs_z while the
//      row kept the value of the last standalone write;
//   e. bitwise: per-sequence logits rows and every recurrent r_l/s_l cell equal
//      the stock arm, and the two sequences' logits differ (non-degeneracy);
//   f. the named refusals sum to rsz_rt_fallbacks.
//
// usage: test-rsz-nkv-rsz-topology <tiny-hybrid.gguf>
//
// Exits 77 (ctest SKIP_RETURN_CODE) when no model is given, 1 on any failed
// assertion, 0 otherwise.

namespace {

constexpr int k_skip = 77;

constexpr int N_SEQ      = 2;
constexpr int N_RS_SLOTS = 4;

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

// The helper owns the environment: every gate this test depends on is set
// explicitly before llama_init_from_model() and cleared after it, so the
// caller's shell cannot change the arm under test.
//
// The two arms are genuinely different code paths:
//
//   on  == true   GGML_TP5_UNION_BINDER=1 and GGML_TP5_RSZ_RUNTIME_PARAM=1.
//                 Both are read ONCE, in the llama_context constructor
//                 (src/llama-context.cpp:536-537 for the umbrella gate,
//                 :576 for the lane gate). The context then builds the
//                 envelope graph and reuses it across the parameter moves.
//
//   on  == false  LLAMA_GRAPH_REUSE_DISABLE=1. Read once, in the same
//                 constructor (src/llama-context.cpp:500-501, "graph reuse
//                 disabled"), it makes every decode build a fresh graph, so
//                 the stock arm rebuilds on every step. That is the control
//                 the invariance assertion needs: without it the stock arm
//                 could also reuse and the ON arm's reuse would prove nothing.
//                 GGML_TP5_UNION_BINDER is left UNSET so the binder does not
//                 arm on this arm either.
//
// Neither arm un-sets a variable while its context is alive: the variables are
// written before llama_init_from_model() and cleared only after the context
// has captured them, in make_ctx()'s tail. A later make_ctx() call therefore
// starts from a clean environment and cannot inherit the previous arm.
llama_context * make_ctx(llama_model * model, bool on) {
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    unsetenv("GGML_TP5_UNION_BINDER");
    unsetenv("GGML_TP5_RSZ_RUNTIME_PARAM");
    if (on) {
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
    // n_stream == n_seq_max and the binder refuses a multi-stream cache, so
    // the n_kv lane could never bind
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

    // the constructor has captured every variable above; clearing them now
    // cannot change this context and keeps the next make_ctx() honest
    unsetenv("GGML_TP5_UNION_BINDER");
    unsetenv("GGML_TP5_RSZ_RUNTIME_PARAM");
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    return ctx;
}

bool decode_batch(llama_context * ctx, const std::vector<token_step> & ts) {
    llama_batch b = llama_batch_init((int32_t) ts.size(), 0, 1);
    if (!b.token || !b.pos || !b.n_seq_id || !b.seq_id || !b.logits) {
        llama_batch_free(b);
        return false;
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

llama_memory_hybrid * hybrid_mem(llama_context * ctx) {
    return dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
}

// The live hybrid input of the previous graph, or null when the live graph has
// none (the caller treats null as a failure).
const llm_graph_input_mem_hybrid * live_hybrid_inp(llama_context * ctx) {
    llm_graph_result * res = ctx->get_gf_res_prev();
    if (res == nullptr) {
        return nullptr;
    }
    for (const auto & input : res->inputs) {
        const auto * hyb = dynamic_cast<const llm_graph_input_mem_hybrid *>(input.get());
        if (hyb != nullptr) {
            return hyb;
        }
    }
    return nullptr;
}

// The width the live graph is bound at, read from the graph's own attention
// input: llm_graph_input_attn_kv caches the llama_kv_cache_context the build
// used, and get_n_kv_target() is the width can_reuse_kq_mask() compares
// against. Returns 0 when the live graph has no hybrid attention input.
uint32_t live_bound_nkv(llama_context * ctx) {
    const llm_graph_input_mem_hybrid * hyb = live_hybrid_inp(ctx);
    if (hyb == nullptr || hyb->get_attn() == nullptr || hyb->get_attn()->mctx == nullptr) {
        return 0;
    }
    return hyb->get_attn()->mctx->get_n_kv_target();
}

// The value the live graph's zero-row index holds, or INT32_MIN when the live
// graph carries no index (the lane is not armed). The tensor is host-backed:
// union_rsz_bind_index() asserts ggml_backend_buffer_is_host(index->buffer)
// (src/llama-graph.cpp:542), and it is the I32 [1] tensor named
// "union_rs_zero_index" that build_rs_inp_impl() creates when
// llm_graph_params::rs_row_armed is set (src/llama-graph.cpp:4286-4293).
//
// The value is the ROW the gathers read (rs_idx*mem_size + rs_z), not the cell
// index. With n_rs_seq == 0 the two coincide, which is why this fixture can
// compare it against the live rs_z directly.
int32_t live_index_value(llama_context * ctx) {
    const llm_graph_input_mem_hybrid * hyb = live_hybrid_inp(ctx);
    if (hyb == nullptr || hyb->get_recr() == nullptr) {
        return std::numeric_limits<int32_t>::min();
    }
    const ggml_tensor * t = hyb->get_recr()->s_zero;
    if (t == nullptr || t->data == nullptr) {
        return std::numeric_limits<int32_t>::min();
    }
    return ((const int32_t *) t->data)[0];
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
        std::fprintf(stderr, "FAIL step %d seq %d: ON logits differ from stock\n", step, seq);
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
// relay ring added, so a counter that stops being counted shows up here.
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

// one step's observation, printed at the end as one stats line
struct step_obs {
    int32_t  rs_z   = -1;
    uint32_t nkv    = 0;
    uint64_t builds = 0;
    int      reused = 0;
    bool     both_moved = false;
};

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

    llama_context * ctx_on  = make_ctx(model, true);
    llama_context * ctx_off = make_ctx(model, false);
    if (ctx_on == nullptr || ctx_off == nullptr) {
        std::fprintf(stderr, "FAIL: context creation failed\n");
        if (ctx_on)  llama_free(ctx_on);
        if (ctx_off) llama_free(ctx_off);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_memory_hybrid * mem_on  = hybrid_mem(ctx_on);
    llama_memory_hybrid * mem_off = hybrid_mem(ctx_off);
    if (mem_on == nullptr || mem_off == nullptr) {
        std::fprintf(stderr, "FAIL: fixture is not hybrid (no llama_memory_hybrid)\n");
        llama_free(ctx_on);
        llama_free(ctx_off);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // the arm is decided by the real flag, not by the argument name
    const llama_nkv_ctx_state & st = llama_nkv_ctx_state_get(ctx_on);
    CHECK(st.nkv_enabled, "binder state not enabled on the ON arm");
    CHECK(st.rsz_rt_enabled, "rs_z gate not read as on");
    CHECK(!llama_nkv_ctx_state_get(ctx_off).nkv_enabled, "binder state enabled on the OFF arm");

    llama_union_diagnostics d0 = {};
    CHECK(llama_get_union_diagnostics(ctx_on, &d0), "diagnostics unavailable");
    CHECK(d0.enabled && d0.rsz_rt_enabled,
          "ON arm: binder or rs_z gate not on (enabled=%d rsz_rt=%d)",
          d0.enabled ? 1 : 0, d0.rsz_rt_enabled ? 1 : 0);

    // ---- prefill: one prompt per sequence, distinct lengths, no padding ----
    const int prefill_len[N_SEQ] = { 5, 3 };
    CHECK(prefill_len[0] != prefill_len[1], "prefill lengths must differ (no padding)");

    bool ok = true;
    llama_pos next_pos[N_SEQ] = { 0, 0 };
    for (int s = 0; ok && s < N_SEQ; ++s) {
        std::vector<token_step> ts;
        for (int i = 0; i < prefill_len[s]; ++i) {
            ts.push_back({ tok_at(s, i, n_vocab), (llama_pos) i, (llama_seq_id) s,
                           i + 1 == prefill_len[s] });
        }
        if (!decode_batch(ctx_on, ts) || !decode_batch(ctx_off, ts)) {
            std::fprintf(stderr, "FAIL: prefill seq %d decode\n", s);
            ok = false;
            break;
        }
        ok = logits_match(ctx_on, prefill_len[s] - 1, ctx_off, prefill_len[s] - 1,
                          n_vocab, -1, s) && ok;
        next_pos[s] = prefill_len[s];
    }

    // ---- steady state: n_kv grows AND rs_z moves on the same steps ---------
    struct step {
        const char * name;
        bool         rm_seq0;
    };
    const step steps[] = {
        { "S0 warm",          false },
        { "S1 seq0 rejoins",  true  },
        { "S2 both continue", false },
        { "S3 seq0 rejoins",  true  },
        { "S4 both continue", false },
        { "S5 seq0 rejoins",  true  },
    };
    const size_t n_steps = sizeof(steps) / sizeof(steps[0]);

    std::vector<step_obs> obs;
    obs.reserve(n_steps);

    size_t  warm_builds = 0;
    int     reuse_on    = 0;
    int     reuse_off   = 0;
    bool    have_prev   = false;
    int32_t prev_rs_z   = -1;
    uint32_t prev_nkv   = 0;
    llama_union_diagnostics d_last = {};

    for (size_t i = 0; ok && i < n_steps; ++i) {
        const step & s = steps[i];
        const int step_i = (int) i;

        if (s.rm_seq0) {
            CHECK(llama_memory_seq_rm(llama_get_memory(ctx_on),  0, -1, -1),
                  "step %d (%s): ON seq_rm failed", step_i, s.name);
            CHECK(llama_memory_seq_rm(llama_get_memory(ctx_off), 0, -1, -1),
                  "step %d (%s): OFF seq_rm failed", step_i, s.name);
            next_pos[0] = 0;
        }

        std::vector<token_step> ts;
        for (int q = 0; q < N_SEQ; ++q) {
            ts.push_back({ tok_at(q, next_pos[q], n_vocab), next_pos[q], (llama_seq_id) q, true });
        }

        llama_perf_context_reset(ctx_on);
        llama_perf_context_reset(ctx_off);
        if (!decode_batch(ctx_on, ts) || !decode_batch(ctx_off, ts)) {
            std::fprintf(stderr, "FAIL step %d (%s): decode\n", step_i, s.name);
            ok = false;
            break;
        }
        const llama_perf_context_data pon  = llama_perf_context(ctx_on);
        const llama_perf_context_data poff = llama_perf_context(ctx_off);

        llama_union_diagnostics di = {};
        if (!llama_get_union_diagnostics(ctx_on, &di)) {
            std::fprintf(stderr, "FAIL step %d (%s): diagnostics unavailable\n", step_i, s.name);
            ok = false;
            break;
        }
        d_last = di;
        ok = fallbacks_add_up(di, "ON arm") && ok;

        const llama_memory_recurrent * recr_on = mem_on->get_mem_recr();
        if (recr_on == nullptr) {
            std::fprintf(stderr, "FAIL step %d (%s): missing recurrent cache\n", step_i, s.name);
            ok = false;
            break;
        }
        const int32_t  rs_z   = recr_on->rs_z;
        const uint32_t nkv    = live_bound_nkv(ctx_on);
        const uint64_t builds = llama_graph_build_count(ctx_on);
        const int32_t  idx    = live_index_value(ctx_on);

        step_obs o;
        o.rs_z   = rs_z;
        o.nkv    = nkv;
        o.builds = builds;
        o.reused = pon.n_reused;

        // ---- (b) both parameters really moved -----------------------------
        if (have_prev) {
            o.both_moved = nkv != prev_nkv && rs_z != prev_rs_z;
        }
        prev_nkv = nkv;
        prev_rs_z = rs_z;
        have_prev = true;

        // ---- (c) the lane is live -----------------------------------------
        CHECK(nkv > 0, "step %d (%s): no live attention input to read the bound n_kv from",
              step_i, s.name);
        CHECK(di.rsz_rt_zero_view_rows == recr_on->get_size(),
              "step %d (%s): zero view has %u rows, recurrent cache has %u cells",
              step_i, s.name, di.rsz_rt_zero_view_rows, recr_on->get_size());
        if (rs_z >= 0) {
            CHECK(di.rsz_rt_active,
                  "step %d (%s): lane inactive at rs_z=%d (fallbacks=%llu)",
                  step_i, s.name, rs_z, (unsigned long long) di.rsz_rt_fallbacks);
            CHECK((uint32_t) rs_z < di.rsz_rt_zero_view_rows,
                  "step %d (%s): live rs_z %d is outside the zero view (%u rows)",
                  step_i, s.name, rs_z, di.rsz_rt_zero_view_rows);
            CHECK(di.rsz_rt_refused_envelope == 0,
                  "step %d (%s): %llu envelope refusals on a live lane",
                  step_i, s.name, (unsigned long long) di.rsz_rt_refused_envelope);

            // ---- (d) the selection reached the graph ----------------------
            //
            // The row is written by union_rsz_bind_index() from the HYBRID
            // set_input() (src/llama-graph.cpp:1386, :1460). Before that call
            // was added to the hybrid sites, can_reuse() accepted a moved rs_z
            // (s_zero != nullptr short-circuits the rs_z key,
            // src/llama-graph.cpp:614-616) while the row kept the value of the
            // last standalone write: this check is the one that catches it.
            CHECK(di.rsz_rt_bound_rs_z == rs_z,
                  "step %d (%s): diagnostics bound rs_z %d != live rs_z %d",
                  step_i, s.name, di.rsz_rt_bound_rs_z, rs_z);
            CHECK(idx == rs_z,
                  "step %d (%s): live graph selection row holds %d, live rs_z is %d "
                  "(the hybrid set_input() did not write the row)",
                  step_i, s.name, idx, rs_z);
        }

        // ---- (e) bitwise equality with stock, plus non-degeneracy ---------
        for (int q = 0; q < N_SEQ && ok; ++q) {
            ok = logits_match(ctx_on, q, ctx_off, q, n_vocab, step_i, q) && ok;
        }
        ok = logits_differ(ctx_on, 0, 1, n_vocab, step_i) && ok;
        ok = compare_all_states(ctx_on, ctx_off, step_i) && ok;

        // ---- (a) topology invariance --------------------------------------
        //
        // The assertion is on the build count only: patch_binds is a per-kind
        // dispatch counter that also moves on a no-op kind, so it cannot
        // separate a bind from a declined attempt.
        if (i == 0) {
            warm_builds = (size_t) builds;
        } else {
            CHECK((size_t) builds == warm_builds,
                  "step %d (%s): graph_builds %llu != warm %zu (rebuild on an n_kv+rs_z move)",
                  step_i, s.name, (unsigned long long) builds, warm_builds);
            if (pon.n_reused == 1) {
                ++reuse_on;
            }
            // the control arm must NOT reuse: it runs with
            // LLAMA_GRAPH_REUSE_DISABLE=1, which makes llama_context build a
            // fresh graph for every decode (src/llama-context.cpp:500-501 and
            // the reuse gate that reads graph_reuse_disable). So the ON arm's
            // reuse above is not vacuous -- the same scenario, the same
            // parameter moves, and the control rebuilds every time.
            if (poff.n_reused == 0) {
                ++reuse_off;
            }
        }

        obs.push_back(o);

        for (int q = 0; q < N_SEQ; ++q) {
            next_pos[q] += 1;
        }
    }

    if (ok) {
        // ---- (b) distinct values, not a degenerate scenario ---------------
        std::vector<uint32_t> nkv_seen;
        std::vector<int32_t>  rs_z_seen;
        size_t both_moved = 0;
        for (const step_obs & o : obs) {
            if (std::find(nkv_seen.begin(), nkv_seen.end(), o.nkv) == nkv_seen.end()) {
                nkv_seen.push_back(o.nkv);
            }
            if (std::find(rs_z_seen.begin(), rs_z_seen.end(), o.rs_z) == rs_z_seen.end()) {
                rs_z_seen.push_back(o.rs_z);
            }
            if (o.both_moved) {
                ++both_moved;
            }
        }
        CHECK(nkv_seen.size() >= 2,
              "the bound n_kv took %zu distinct value(s); the n_kv lane did not move",
              nkv_seen.size());
        CHECK(rs_z_seen.size() >= 2,
              "rs_z took %zu distinct value(s); the rs_z lane did not move",
              rs_z_seen.size());
        CHECK(both_moved >= 2,
              "only %zu step(s) moved BOTH n_kv and rs_z; the cross is vacuous", both_moved);

        // ---- (a) reuse on the ON arm, rebuild on the stock arm ------------
        CHECK(reuse_on == (int) n_steps - 1,
              "ON arm reused %d times, expected %zu", reuse_on, n_steps - 1);
        CHECK(reuse_off == (int) n_steps - 1,
              "the reuse-disabled control rebuilt on %d of %zu steps, expected every one "
              "(otherwise the ON arm's reuse is vacuous)",
              (int) n_steps - 1 - reuse_off, n_steps - 1);

        // ---- (c) the lane armed and never refused -------------------------
        CHECK(d_last.rsz_rt_active, "lane never armed");
        CHECK(d_last.rsz_rt_refused_envelope == 0, "envelope refusals on a live lane");
        CHECK(d_last.rsz_rt_fallbacks == d_last.rsz_rt_refused_gate + d_last.rsz_rt_refused_arch +
                                         d_last.rsz_rt_refused_no_rs_z + d_last.rsz_rt_refused_cell +
                                         d_last.rsz_rt_refused_envelope + d_last.rsz_rt_refused_shape +
                                         d_last.rsz_rt_refused_n_rs_seq + d_last.rsz_rt_refused_ring +
                                         d_last.rsz_rt_refused_mixed_depth,
              "named refusals do not sum to the lane total");
    }

    // ---- one stats line, then the verdict ---------------------------------
    std::printf("RSZ_NKV_RSZ_STATS warm_builds=%zu reuse_on=%d reuse_off=%d "
                "zero_view_rows=%u active=%d fallbacks=%llu nkv=",
                warm_builds, reuse_on, reuse_off, d_last.rsz_rt_zero_view_rows,
                d_last.rsz_rt_active ? 1 : 0, (unsigned long long) d_last.rsz_rt_fallbacks);
    for (size_t i = 0; i < obs.size(); ++i) {
        std::printf("%s%u", i == 0 ? "" : ",", obs[i].nkv);
    }
    std::printf(" rs_z=");
    for (size_t i = 0; i < obs.size(); ++i) {
        std::printf("%s%d", i == 0 ? "" : ",", obs[i].rs_z);
    }
    std::printf(" builds=");
    for (size_t i = 0; i < obs.size(); ++i) {
        std::printf("%s%llu", i == 0 ? "" : ",", (unsigned long long) obs[i].builds);
    }
    std::printf(" reused=");
    for (size_t i = 0; i < obs.size(); ++i) {
        std::printf("%s%d", i == 0 ? "" : ",", obs[i].reused);
    }
    std::printf("\n");

    llama_free(ctx_on);
    llama_free(ctx_off);
    llama_model_free(model);
    llama_backend_free();

    if (!ok || g_fail != 0) {
        std::printf("RSZ_NKV_RSZ_TEST FAIL %d\n", g_fail + (ok ? 0 : 1));
        return 1;
    }
    std::printf("RSZ_NKV_RSZ_TEST PASS\n");
    return 0;
}
