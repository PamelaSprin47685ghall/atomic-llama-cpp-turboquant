// CPU test: the FINAL gate of the union binder's exact n_kv lane.
//
// The claim this file tests is short and strict:
//
//   one binder context decodes a 300-step single-token history and produces,
//   at every step, the SAME logits row as a stock context that decodes the
//   same history -- and the binder context builds exactly ONE graph for the
//   whole decode while the stock context builds at least one per step.
//
// A second arm pair repeats the same equality for a multi-token batch: 65
// steps of 4 tokens (260 positions, all 4 rows marked and compared), so the
// claim covers every row of every step of a multi-token decode too.
//
// The 13022 gate split the old bitwise comparison into a numeric pair (same
// width) and a binder pair (exact vs stock graph-build counts). This file is
// the binder pair, and it is the only place where the two are claimed
// together. It is deliberately narrow: one model, one history, one arm pair
// per batch width.
//
// Arms (env read ONCE in the llama_context constructor, so set/unset around
// construction only, exactly as tests/test-union-four-param-matrix.cpp does):
//
//   binder/exact arm  GGML_TP5_UNION_BINDER=1
//                     LLAMA_EXPERIMENTAL_VARLEN_NTOK unset (the n_kv lane, not
//                       the token lane: the token lane replans the arena per
//                       width and is a different contract)
//                     LLAMA_GRAPH_REUSE_DISABLE unset
//   stock arm         LLAMA_GRAPH_REUSE_DISABLE=1 (llama-context.cpp:498-505)
//                     GGML_TP5_UNION_BINDER=1 is left set ON PURPOSE, and this
//                       is not a convenience: the same gate decides n_ctx
//                       rounding (llama-context.cpp:599-606), so leaving it
//                       set keeps both contexts' capacity identical. What
//                       makes the stock arm stock is that the record build
//                       requires !graph_reuse_disable
//                       (llama-context.cpp:5321): no binder record is ever
//                       taken, record_valid stays false, and the graph is
//                       rebuilt at every step.
//
// The two arms therefore differ ONLY in whether the binder's
// graph-reuse/exact-width record exists. The stock arm is NOT upstream
// llama.cpp: it is the same binder binary with graph reuse disabled and the
// record suppressed. The comparison proves "binder exact lane == this binary
// with the record absent", never "== upstream".
//
// Both contexts of an arm pair come from one llama_model, with the same n_ctx,
// n_batch, n_ubatch and n_outputs_max, flash attention disabled, CPU only
// (mp.devices = { nullptr }).
//
// What the test asserts, in order:
//
//   1. BSE_BITWISE  every logits row of BOTH positive arms is byte-identical
//                   (memcmp over n_vocab floats): 300 single-token rows plus
//                   260 multi-token rows.
//   2. BSE_STOCK_ARM  the distinctness witness, taken ONCE after the stock
//                   arm's first decode (before any decode neither context has
//                   a record, so a pre-decode check would be vacuous).
//                   llama_get_nkv_runtime_stats() on the stock context must
//                   return FALSE: the record build is gated on
//                   !graph_reuse_disable (llama-context.cpp:5321), so no
//                   binder record exists there and the stats path refuses
//                   (src/llama-context.cpp:1928-1932). If it returns true, the
//                   stock arm took the binder path and the comparison is
//                   binder-vs-binder. This replaces the old (unreachable)
//                   "equal build counts" check.
//   3. BSE_SINGLE   the gate's acceptance fields, read from the run:
//                   llama_graph_build_count() deltas named on_decode_builds
//                   (binder == 1 exactly) and off_decode_builds (stock >=
//                   steps); the stats path's graph_builds unchanged between
//                   the first and the last step -- the SAME monotonic counter
//                   read twice, not an independent measurement; and
//                   rebuild_fallbacks == 0 and pad_fallbacks == 0 across the
//                   decode.
//   4. the exact lane really was armed and really bound, on EVERY step:
//                   topology_hashed, has_nkv_dims, mask_filled > 0,
//                   mask_refused == 0, mask_n_kv == runtime_n_kv,
//                   bound_n_kv == runtime_n_kv; the mask read-back ran and
//                   agreed on every row it was owed (mask_row_checked > 0,
//                   mask_row_ok == mask_row_checked, mask_row_refused == 0,
//                   mask_rows_all == 1); the executed span crosses 256.
//                   The bind witness is patch_binds > 0 plus the per-step
//                   bound_n_kv == runtime_n_kv, NOT the width moving:
//                   runtime_n_kv is re-armed from the KV cache every ubatch
//                   (llama-context.cpp:5687) whether or not a bind applied.
//                   The stock arm's own state is witnessed by BSE_STOCK_ARM,
//                   not by a stock patch_binds reading.
//   5. BSE_KERNEL_SEL  the DECOUPLING gate. The exact lane is only legitimate
//                   when kernel selection is independent of the runtime
//                   length. That decoupling is produced by a sibling change
//                   (run union-kernel-sel-len-indep-r24m-20261104-1225) and is
//                   NOT in this tree. The probe below is fail-closed: it is
//                   false unless the decoupling patch defines the named
//                   symbol (see the "decoupling probe" block below), and a
//                   false probe prints ok=0, names the reason, and makes the
//                   test exit nonzero. It never skips and never prints ok=1
//                   without the symbol.
//   6. BSE_NEG      the falsifiability arm: the same comparison, run against a
//                   stock context whose numerics are deliberately changed, must
//                   report differ > 0. Without it BSE_BITWISE could be a
//                   comparison that cannot fail.
//   7. BSE_END      the conjunction.
//   8. BSE_MULTI    the 4-token arm's acceptance fields, same shape and same
//                   per-step assertions as BSE_SINGLE, with rows counted per
//                   step (4) and the executed span checked against
//                   4 * (step + 1).
//
// Every machine line goes to stderr -- the same stream the existing tests use,
// and the stream tests/nkv-cpu-probe/gate-nkv-runtime-13022.sh greps.
//
// What this test does NOT prove:
//   - that any ggml kernel reads the runtime width. No kernel does; the
//     consumer is the host-side KQ-mask fill (llama-graph.cpp:328-389), and
//     the K/V views take their width from the binder's in-place view patch.
//     The kernel-selection decoupling is a separate, unlanded change and is
//     gated here rather than assumed;
//   - anything about GPU/HIP, flash attention, relay, pipeline, the token lane
//     (LLAMA_EXPERIMENTAL_VARLEN_NTOK), batches whose rows are not all marked
//     as outputs, n_outputs collapsing (n_outputs < n_tokens), more than one
//     sequence, or token widths other than 1 and 4. The 4-token arm covers
//     n_tokens == 4 with n_outputs == 4 only;
//   - that upstream llama.cpp would have produced these rows. The stock arm is
//     the SAME binary with LLAMA_GRAPH_REUSE_DISABLE=1 and the binder record
//     suppressed; it is not an upstream build. What the test proves is that
//     the two arms produced the same rows while one held a live binder record
//     and built one graph for the whole decode and the other held no record
//     and rebuilt at every step.
//
// usage: test-union-binder-stock-equiv <model.gguf>
//        test-union-binder-stock-equiv --neg-control <model.gguf>
//   The plain form is the registration that carries the PASS_REGULAR_EXPRESSION
//   on BSE_END. --neg-control runs only the falsifiability arm and exits 0 only
//   when it reports BSE_NEG ok=1.

#include "llama.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// the decoupling probe (BSE_KERNEL_SEL)
// ---------------------------------------------------------------------------
//
// The exact lane binds the graph to the true used span. That is only
// legitimate when the ggml CPU kernels the attention nodes dispatch to do not
// depend on the runtime length. On a tree without the decoupling they do: the
// binder marks the softmax and the value product with
// GGML_TENSOR_FLAG_REDUCE_PAD_TAIL and pins their accumulation tree to a
// padded width in op_params[2], and the CPU kernels read it, so the exact lane
// is not allowed to claim bit-equality and this gate must say so. On the tree
// this test ships with that marking is gone -- the binder patches shapes in
// place and does nothing else -- and the macro below is defined.
//
// The probe is a preprocessor check on a symbol the decoupling patch must
// define. The symbol name is fixed here and nowhere else:
//
//     LLAMA_UNION_KERNEL_SEL_LEN_INDEP
//
// The decoupling patch turns this gate green by defining that macro in
// include/llama.h, the only llama header this TU includes. A definition
// anywhere else -- src/llama-union-binder.h included -- cannot turn the gate
// green, and this TU deliberately does not include the internal header: it
// would drag ggml internals into a public-API test for no gain. That is the
// fail-closed property we want: the macro must be visible to a consumer of the
// public header, not smuggled into an internal one. The patch must define it
// ONLY when the decoupling is real -- a kernel that selects on the runtime
// length must not leave it defined.
//
// It is a macro, not a function, on purpose: a function symbol would have to
// be declared somewhere the test can see, and a declaration the patch does not
// actually define would link-fail on 13022 instead of failing this gate with a
// reason. A macro cannot be declared without being defined.
//
// Fail-closed by construction: an undefined macro is 0 in #if, so the gate is
// red on every tree that does not carry the decoupling.
#if defined(LLAMA_UNION_KERNEL_SEL_LEN_INDEP) && LLAMA_UNION_KERNEL_SEL_LEN_INDEP
static constexpr bool bse_kernel_sel_len_indep = true;
#else
static constexpr bool bse_kernel_sel_len_indep = false;
#endif

// ---------------------------------------------------------------------------
// model + context helpers
// ---------------------------------------------------------------------------

struct bse_model {
    llama_model * model = nullptr;
    int           vocab = 0;
};

static bool bse_open(const char * model_path, bse_model & m) {
    llama_backend_init();

    auto mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = { nullptr };
    mp.devices      = devices;
    mp.n_gpu_layers = 0;

    m.model = llama_model_load_from_file(model_path, mp);
    if (m.model == nullptr) {
        std::fprintf(stderr, "FAIL: model load: %s\n", model_path);
        llama_backend_free();
        return false;
    }
    m.vocab = (int) llama_vocab_n_tokens(llama_model_get_vocab(m.model));
    if (m.vocab <= 3) {
        std::fprintf(stderr, "FAIL: vocab=%d\n", m.vocab);
        llama_model_free(m.model);
        m.model = nullptr;
        llama_backend_free();
        return false;
    }
    return true;
}

static void bse_close(bse_model & m) {
    if (m.model != nullptr) {
        llama_model_free(m.model);
        m.model = nullptr;
    }
    llama_backend_free();
}

// The context's token width is a BUILD parameter here: n_batch = n_ubatch =
// n_outputs_max = n_tok. The n_tokens lane that would patch the width in place
// is disabled in this test (LLAMA_EXPERIMENTAL_VARLEN_NTOK unset), so a
// 4-token arm needs its own context pair and cannot grow the single-token one.
// n_seq_max = 1 keeps the mask and the gather closure single-stream.
static llama_context * bse_context(const bse_model & m, int n_ctx, int n_tok) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx         = (uint32_t) n_ctx;
    cp.n_batch       = (uint32_t) n_tok;
    cp.n_ubatch      = (uint32_t) n_tok;
    cp.n_seq_max     = 1;
    cp.n_outputs_max = (uint32_t) n_tok;
    // this tree spells the field flash_attn_type (include/llama.h:390);
    // LLAMA_FLASH_ATTN_TYPE_DISABLED is 0 (include/llama.h:197)
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    return llama_init_from_model(m.model, cp);
}

// One single-token step at `pos`. `logits_out` receives the full row.
static int bse_step(llama_context * ctx, int vocab, int pos, std::vector<float> & logits_out) {
    logits_out.assign((size_t) vocab, 0.0f);

    llama_batch batch = llama_batch_init(1, 0, 1);
    if (!llama_batch_is_complete(&batch, 1, false, true)) {
        llama_batch_free(batch);
        GGML_ABORT("llama_batch_init failed");
    }
    batch.n_tokens     = 1;
    batch.token[0]     = (llama_token) (1 + (pos * 7) % (vocab - 1));
    batch.pos[0]       = (llama_pos) pos;
    batch.n_seq_id[0]  = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0]    = 1;

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        return rc;
    }
    const float * src = llama_get_logits_ith(ctx, 0);
    if (src == nullptr) {
        return -1;
    }
    std::memcpy(logits_out.data(), src, sizeof(float) * (size_t) vocab);
    return 0;
}

// Compare two full logits rows: 0 = bitwise equal, 1 = bitwise different,
// -1 = not comparable (size mismatch or an empty row). The three outcomes are
// distinct on purpose: a caller that summed 1 and -1 would let a size mismatch
// cancel a real difference and read as "no difference".
static int bse_diff_rows(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return -1;
    }
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// the multi-token batch helpers
// ---------------------------------------------------------------------------
//
// A 4-token batch, positions 4*step .. 4*step+3, all four rows marked as
// outputs. 65 steps put the executed span at 260 positions, so it crosses 256
// at the last step (step 63 executes at 256, step 64 at 260).
static constexpr int bse_multi_rows_per_step = 4;
static constexpr int bse_multi_steps         = 65;
static constexpr int bse_multi_rows          = bse_multi_steps * bse_multi_rows_per_step;
// 8 spare cells, the same margin the single-token arm uses: the build envelope
// is the raw n_ctx (the binder gate keeps it, llama-context.cpp:599-606), and
// the executed span must stay strictly inside it.
static constexpr int bse_multi_n_ctx         = bse_multi_rows + 8;

// One multi-token step. `logits_out` receives all four rows, row-major in batch
// order, each `vocab` floats. llama_get_logits_ith(ctx, i) is the row of the
// i-th token of the batch, and every token has logits[i] == 1, so all four are
// real output rows.
static int bse_step_multi(llama_context * ctx, int vocab, int step, std::vector<float> & logits_out) {
    const int n = bse_multi_rows_per_step;
    logits_out.assign((size_t) n * (size_t) vocab, 0.0f);

    llama_batch batch = llama_batch_init(n, 0, 1);
    if (!llama_batch_is_complete(&batch, n, false, true)) {
        llama_batch_free(batch);
        GGML_ABORT("llama_batch_init failed");
    }
    batch.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        const int pos = step * n + i;
        batch.token[i]     = (llama_token) (1 + (pos * 7) % (vocab - 1));
        batch.pos[i]       = (llama_pos) pos;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        return rc;
    }
    for (int i = 0; i < n; ++i) {
        const float * src = llama_get_logits_ith(ctx, i);
        if (src == nullptr) {
            return -1;
        }
        std::memcpy(logits_out.data() + (size_t) i * (size_t) vocab, src,
                sizeof(float) * (size_t) vocab);
    }
    return 0;
}

// Compare two multi-row logits buffers. The whole-buffer memcmp answers the
// binary question; the per-row walk then names the row that differs, so a
// FAIL line can say (step, row) instead of only "the batch differs". Returns
// the number of differing rows, or -1 when the buffers are not comparable.
static int bse_diff_rows_multi(const std::vector<float> & a, const std::vector<float> & b,
                               int rows, int vocab, int & first_bad) {
    first_bad = -1;
    if (rows <= 0 || vocab <= 0 || a.size() != b.size() ||
            a.size() != (size_t) rows * (size_t) vocab) {
        return -1;
    }
    if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0) {
        return 0;
    }
    int differ = 0;
    for (int i = 0; i < rows; ++i) {
        const size_t off = (size_t) i * (size_t) vocab;
        if (std::memcmp(a.data() + off, b.data() + off, sizeof(float) * (size_t) vocab) != 0) {
            if (first_bad < 0) {
                first_bad = i;
            }
            ++differ;
        }
    }
    return differ;
}

// ---------------------------------------------------------------------------
// the kernel-selection gate
// ---------------------------------------------------------------------------

// Prints the gate line and returns its verdict. Fail-closed: the only way to
// get ok=1 is a tree that defines LLAMA_UNION_KERNEL_SEL_LEN_INDEP.
static bool bse_kernel_sel() {
    if (bse_kernel_sel_len_indep) {
        std::fprintf(stderr, "BSE_KERNEL_SEL ok=1 reason=kernel_sel_len_indep\n");
        return true;
    }
    std::fprintf(stderr, "BSE_KERNEL_SEL ok=0 reason=kernel_sel_coupled\n");
    std::fprintf(stderr,
            "FAIL: binder-stock-equiv: kernel selection still depends on the runtime length;"
            " the exact lane cannot be claimed\n");
    return false;
}

// ---------------------------------------------------------------------------
// the positive arm pair
// ---------------------------------------------------------------------------

// 300 steps: the executed span runs 1..300, so it crosses the 256-cell KV
// boundary. A shorter run would leave the crossing claim untested.
static constexpr int bse_steps = 300;

// Why the exact lane did not deliver the true used span. Called only on the
// failing step, immediately before the FAIL line that reports it, so the
// refusal is diagnosable instead of merely visible as runtime_n_kv == 0.
// Reads llama_get_union_diagnostics (include/llama.h:1890-1903); a context
// that cannot report diagnostics says so rather than printing zeros that look
// like a real answer.
static void bse_print_lane_refused(llama_context * ctx, int step, const llama_nkv_runtime_stats & st) {
    llama_union_diagnostics d = {};
    if (!llama_get_union_diagnostics(ctx, &d)) {
        std::fprintf(stderr,
                "BSE_LANE_REFUSED step=%d runtime_n_kv=%u build_n_kv=%u diagnostics=unavailable\n",
                step, st.runtime_n_kv, st.build_n_kv);
        return;
    }
    std::fprintf(stderr,
            "BSE_LANE_REFUSED step=%d runtime_n_kv=%u build_n_kv=%u enabled=%d record_valid=%d"
            " envelope_disabled=%d lane_declines=%" PRIu64 " rebuild_fallbacks=%" PRIu64
            " pad_fallbacks=%" PRIu64 "\n",
            step, st.runtime_n_kv, st.build_n_kv,
            int(d.enabled), int(d.record_valid), int(d.envelope_disabled),
            d.lane_declines, d.rebuild_fallbacks, d.pad_fallbacks);
}

// One positive arm's verdict and the row counts the aggregate line needs.
// `stock_arm` is only meaningful on the single-token arm: that is where the
// stock-arm distinctness witness is taken, once, and it is reported once on
// the wire (BSE_STOCK_ARM).
struct bse_arm_result {
    bool ok        = false;
    bool stock_arm = false;
    int  rows      = 0;   // logits rows compared
    int  bitwise   = 0;   // logits rows bitwise equal
};

static bse_arm_result bse_positive(const bse_model & m) {
    bse_arm_result r;
    const int vocab = m.vocab;
    const int n_ctx = bse_steps + 8;

    // Environment audit. Every variable below is read by the llama_context
    // CONSTRUCTOR, once (llama-context.cpp:498-505 for
    // LLAMA_GRAPH_REUSE_DISABLE, :536-589 for GGML_TP5_UNION_BINDER and
    // LLAMA_EXPERIMENTAL_VARLEN_NTOK). None of them is read lazily, so
    // unsetting LLAMA_GRAPH_REUSE_DISABLE right after the stock context is
    // constructed cannot change that context's arm: the constructor already
    // consumed the value. What would be unsafe is changing a variable between
    // construction and a lazy read, and there is no lazy read here.
    //   LLAMA_NKV_MASK_OVERRIDE is unset for both arms: it is a per-set_input
    //   test hook (llama-graph.cpp:328-345) and an inherited value would
    //   change the mask width this test observes.
    setenv("GGML_TP5_UNION_BINDER", "1", 1);
    unsetenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK");
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    unsetenv("LLAMA_NKV_MASK_OVERRIDE");

    llama_context * ctx = bse_context(m, n_ctx, 1);
    if (ctx == nullptr) {
        std::fprintf(stderr, "FAIL: binder-stock-equiv: binder context init\n");
        return r;
    }

    // The stock arm: one context, same history, every step rebuilt. The flag
    // is set before this context is constructed and unset after; the
    // constructor already read it.
    setenv("LLAMA_GRAPH_REUSE_DISABLE", "1", 1);
    llama_context * stock = bse_context(m, n_ctx, 1);
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    if (stock == nullptr) {
        std::fprintf(stderr, "FAIL: binder-stock-equiv: stock context init\n");
        llama_free(ctx);
        return r;
    }

    // The stock-arm distinctness witness is taken after the stock arm's first
    // decode; see the loop below. `stock_arm` is its verdict and is folded
    // into this arm's result.
    bool stock_arm = true;
    bool ok = true;

    // Build-count baselines: taken after both contexts exist, so the
    // constructor/reserve builds are not counted as decode builds.
    const uint64_t binder_builds0 = llama_graph_build_count(ctx);
    const uint64_t stock_builds0  = llama_graph_build_count(stock);

    int      rows           = 0;
    int      bitwise        = 0;
    int64_t  min_nkv        = INT64_MAX;
    int64_t  max_nkv        = 0;
    uint32_t build_n_kv     = 0;
    bool     have_build     = false;
    bool     stock_checked  = false; // the stock-arm witness has been taken
    uint64_t patch_binds0   = 0;
    uint64_t patch_binds1   = 0;
    uint64_t rebuild_fb0    = 0;
    uint64_t rebuild_fb1    = 0;
    uint64_t pad_fb0        = 0;
    uint64_t pad_fb1        = 0;
    // The stats path's build counter at the first and the last step. It is the
    // SAME monotonic counter as llama_graph_build_count() (read out at
    // src/llama-context.cpp:1958), not an independent one: an unchanged value
    // here says the two readings agree with the delta, it does not measure a
    // second thing.
    uint64_t first_stats_builds = 0;
    uint64_t last_stats_builds  = 0;
    bool     have_first_stats   = false;

    {
        llama_union_diagnostics d0 = {};
        if (!llama_get_union_diagnostics(ctx, &d0)) {
            std::fprintf(stderr, "FAIL: binder-stock-equiv: no union diagnostics on the binder arm\n");
            ok = false;
        } else {
            patch_binds0 = d0.patch_binds;
            rebuild_fb0  = d0.rebuild_fallbacks;
            pad_fb0      = d0.pad_fallbacks;
        }
    }

    std::vector<float> got, want;

    for (int step = 0; step < bse_steps && ok; ++step) {
        if (bse_step(ctx, vocab, step, got) != 0) {
            std::fprintf(stderr, "FAIL: binder-stock-equiv: binder decode step=%d\n", step);
            ok = false;
            break;
        }

        // --- the exact lane must be armed and must have bound --------------
        llama_nkv_runtime_stats st = {};
        if (!llama_get_nkv_runtime_stats(ctx, &st)) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: no n_kv runtime stats at step=%d"
                    " (binder off or no record)\n", step);
            ok = false;
            break;
        }
        if (!st.topology_hashed) {
            std::fprintf(stderr, "FAIL: binder-stock-equiv: topology not hashed at step=%d\n", step);
            ok = false;
            break;
        }
        if (!st.has_nkv_dims) {
            std::fprintf(stderr, "FAIL: binder-stock-equiv: no n_kv runtime root at step=%d\n", step);
            ok = false;
            break;
        }
        if (st.mask_refused != 0) {
            std::fprintf(stderr, "FAIL: binder-stock-equiv: the mask fill refused at step=%d (refused=%u)\n",
                    step, st.mask_refused);
            ok = false;
            break;
        }
        if (st.mask_filled == 0) {
            std::fprintf(stderr, "FAIL: binder-stock-equiv: no KQ mask filled at step=%d\n", step);
            ok = false;
            break;
        }
        if (st.mask_n_kv != st.runtime_n_kv) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: mask filled at %u, graph executes at %u (step=%d)\n",
                    st.mask_n_kv, st.runtime_n_kv, step);
            ok = false;
            break;
        }
        // The read-back is the only independent observation of the mask. A
        // fill that was never compared proves nothing, so each of these is a
        // FAIL, not a skip.
        if (st.mask_row_checked == 0) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: the mask row was not read back at step=%d (filled=%u)\n",
                    step, st.mask_filled);
            ok = false;
            break;
        }
        if (st.mask_row_ok != st.mask_row_checked) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: the mask row disagrees at step=%d (ok=%u checked=%u)\n",
                    step, st.mask_row_ok, st.mask_row_checked);
            ok = false;
            break;
        }
        if (st.mask_row_refused != 0) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: the mask read-back refused %u row(s) at step=%d\n",
                    st.mask_row_refused, step);
            ok = false;
            break;
        }
        if (st.mask_rows_all != 1) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: only the last mask row was compared at step=%d\n", step);
            ok = false;
            break;
        }
        if (st.bound_n_kv != st.runtime_n_kv) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: bound n_kv %u != runtime n_kv %u (step=%d)\n",
                    st.bound_n_kv, st.runtime_n_kv, step);
            ok = false;
            break;
        }
        // the true used span after this step is step + 1
        if ((int64_t) st.runtime_n_kv != (int64_t) step + 1) {
            // A refused exact lane reads here as runtime_n_kv == 0, which is
            // true but says nothing about WHY. Print the binder's own
            // diagnostics first, so the reader can tell "gate off" from "no
            // record" from "envelope refused" from "multi-split". This is
            // diagnostic output on an already-failing path; it is not a pass
            // condition.
            bse_print_lane_refused(ctx, step, st);
            std::fprintf(stderr, "FAIL: binder-stock-equiv: runtime_n_kv=%u at step=%d (want %d)\n",
                    st.runtime_n_kv, step, step + 1);
            ok = false;
            break;
        }
        if (!have_build) {
            build_n_kv = st.build_n_kv;
            have_build = true;
        } else if (st.build_n_kv != build_n_kv) {
            std::fprintf(stderr, "FAIL: binder-stock-equiv: build width moved at step=%d (%u -> %u)\n",
                    step, build_n_kv, st.build_n_kv);
            ok = false;
            break;
        }
        if ((int64_t) st.runtime_n_kv < min_nkv) min_nkv = (int64_t) st.runtime_n_kv;
        if ((int64_t) st.runtime_n_kv > max_nkv) max_nkv = (int64_t) st.runtime_n_kv;

        // The build counter, read through the stats path. It must agree with
        // llama_graph_build_count() at the first step (the build has just
        // happened) and at the last (no rebuild happened in between). Both are
        // readings of one monotonic member (src/llama-context.cpp:1958), so
        // this is an agreement check, not a second measurement.
        if (!have_first_stats) {
            first_stats_builds = st.graph_builds;
            have_first_stats   = true;
            if (st.graph_builds != llama_graph_build_count(ctx)) {
                std::fprintf(stderr,
                        "FAIL: binder-stock-equiv: stats graph_builds %" PRIu64 " != graph_build_count %" PRIu64
                        " at the first step\n", st.graph_builds, llama_graph_build_count(ctx));
                ok = false;
                break;
            }
        }
        last_stats_builds = st.graph_builds;
        if (step == bse_steps - 1 && st.graph_builds != llama_graph_build_count(ctx)) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: stats graph_builds %" PRIu64 " != graph_build_count %" PRIu64
                    " at the last step\n", st.graph_builds, llama_graph_build_count(ctx));
            ok = false;
            break;
        }

        // --- the stock arm: the SAME step, once ---------------------------
        if (bse_step(stock, vocab, step, want) != 0) {
            std::fprintf(stderr, "FAIL: binder-stock-equiv: stock decode step=%d\n", step);
            ok = false;
            break;
        }

        // The distinctness witness, taken once, after the stock arm's FIRST
        // decode. Timing is the point: before any decode neither context holds
        // a binder record -- record_valid is set only inside the record build
        // (src/llama-context.cpp:5397) -- so a pre-decode check would pass on
        // both arms and prove nothing. After one stock decode the binder arm
        // has a live record (the stats call above asserts it on every step)
        // while the stock arm must still have none: the record build requires
        // !graph_reuse_disable (src/llama-context.cpp:5321), so
        // llama_get_nkv_runtime_stats() must refuse here. That asymmetry is
        // the witness.
        //
        // The diagnostics call is deliberately one-sided: when it is
        // available, only record_valid is required to be false. `enabled` is
        // NOT asserted (the call may be unavailable, and requiring it would
        // fail the arm for a reason unrelated to the record).
        //
        // One witness is enough: the multi-token arm runs the same env split
        // and the same per-step stats assertion, so a second copy would only
        // duplicate the same asymmetry.
        if (!stock_checked) {
            stock_checked = true;
            llama_nkv_runtime_stats sst = {};
            const bool stock_stats = llama_get_nkv_runtime_stats(stock, &sst);
            if (stock_stats) {
                std::fprintf(stderr,
                        "FAIL: binder-stock-equiv: the stock arm reports binder runtime stats"
                        " (it took the binder path)\n");
                stock_arm = false;
                ok = false;
            }
            llama_union_diagnostics sd = {};
            if (llama_get_union_diagnostics(stock, &sd) && sd.record_valid) {
                std::fprintf(stderr,
                        "FAIL: binder-stock-equiv: the stock arm holds a live binder record\n");
                stock_arm = false;
                ok = false;
            }
            std::fprintf(stderr, "BSE_STOCK_ARM ok=%d binder_stats=%d\n",
                    int(stock_arm), int(stock_stats));
        }
        ++rows;
        const int d = bse_diff_rows(got, want);
        if (d < 0) {
            std::fprintf(stderr, "FAIL: binder-stock-equiv: logits row not comparable at step=%d\n", step);
            ok = false;
        } else if (d == 0) {
            ++bitwise;
        } else {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: logits differ from stock at step=%d (runtime_n_kv=%u)\n",
                    step, st.runtime_n_kv);
            ok = false;
        }
    }

    const uint64_t binder_builds = llama_graph_build_count(ctx) - binder_builds0;
    const uint64_t stock_builds  = llama_graph_build_count(stock) - stock_builds0;

    {
        llama_union_diagnostics d1 = {};
        if (!llama_get_union_diagnostics(ctx, &d1)) {
            std::fprintf(stderr, "FAIL: binder-stock-equiv: no union diagnostics after the decode\n");
            ok = false;
        } else {
            patch_binds1 = d1.patch_binds;
            rebuild_fb1  = d1.rebuild_fallbacks;
            pad_fb1      = d1.pad_fallbacks;
        }
    }
    const uint64_t patch_binds  = patch_binds1 - patch_binds0;
    const uint64_t rebuild_fb   = rebuild_fb1 - rebuild_fb0;
    const uint64_t pad_fb       = pad_fb1 - pad_fb0;

    // The run must have produced every row it claims to compare. Without this
    // an early break with rows == 0 would let `bitwise == rows` hold on a
    // comparison that compared nothing.
    if (ok && rows != bse_steps) {
        std::fprintf(stderr,
                "FAIL: binder-stock-equiv: compared %d of %d rows\n", rows, bse_steps);
        ok = false;
    }

    // The exact lane must have been armed on a graph that executes at the true
    // span, and the span must cross 256. Both are read from the run.
    if (ok) {
        if (min_nkv >= 256 || max_nkv <= 256) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: the executed span did not cross 256 (min=%" PRId64 " max=%" PRId64 ")\n",
                    min_nkv, max_nkv);
            ok = false;
        }
        // The bind witness is patch_binds > 0 (below) together with the
        // per-step bound_n_kv == runtime_n_kv assertion: record_bound is
        // assigned only on a successful bind (src/llama-context.cpp:4890),
        // while runtime_n_kv is re-armed from the KV cache on every ubatch
        // (:5687), bind or no bind. The width moving on its own is therefore
        // NOT evidence that a bind happened.
        if (patch_binds == 0) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: patch_binds stayed 0 on the binder arm;"
                    " no bind was applied\n");
            ok = false;
        }
    }

    // The build counts. This is what makes the two arms genuinely different
    // paths: the binder arm builds exactly once for the whole decode and the
    // stock arm builds at least once per step. Equal counts can never hold
    // once binder_builds == 1 and stock_builds >= rows, so there is no
    // "equal counts" check here: it would be unreachable and therefore
    // evidence of nothing. The distinct-path evidence is BSE_STOCK_ARM (no
    // binder record on the stock arm) together with these two counts.
    if (ok) {
        if (binder_builds != 1) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: the binder arm built %" PRIu64 " graph(s) for %d steps (want 1)\n",
                    binder_builds, rows);
            ok = false;
        }
        if (stock_builds < (uint64_t) rows) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: the stock arm built %" PRIu64 " graph(s) for %d steps"
                    " (LLAMA_GRAPH_REUSE_DISABLE did not take effect)\n", stock_builds, rows);
            ok = false;
        }
    }

    // The two witnesses the gate reads besides the build counts.
    //
    // (i) The stats path's build counter must not move between the first and
    // the last step. It is the same monotonic counter as
    // llama_graph_build_count() (src/llama-context.cpp:1958), so this says the
    // stats path agrees with the delta; it is not an independent counter.
    // (ii) A rebuild fallback or a padded fallback means the exact lane did not
    // deliver the span: the graph was rebuilt at the stock width, or the KQ
    // mask was filled at the padded width. Either one makes the equality above
    // a comparison of something other than the exact lane.
    if (ok) {
        if (last_stats_builds != first_stats_builds) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: the stats path saw a build between the first and the last step"
                    " (%" PRIu64 " -> %" PRIu64 ")\n", first_stats_builds, last_stats_builds);
            ok = false;
        }
        if (rebuild_fb != 0) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: %" PRIu64 " rebuild fallback(s) during the decode;"
                    " the exact lane did not deliver the span\n", rebuild_fb);
            ok = false;
        }
        if (pad_fb != 0) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: %" PRIu64 " padded fallback(s) during the decode;"
                    " the exact lane did not deliver the span\n", pad_fb);
            ok = false;
        }
    }

    // The gate's acceptance line. rows= is compared / expected, so a short or
    // partially-different run cannot read as a pass; steps= is the number of
    // decode calls this arm made; on_decode_builds / off_decode_builds are the
    // two llama_graph_build_count() deltas under the names the gate consumes
    // (tests/nkv-cpu-probe/varlen_output_probe.cpp:343 and
    // gate-p1-merge-13022.sh:110,124 use the same spelling).
    std::fprintf(stderr,
            "BSE_SINGLE ok=%d rows=%d/%d steps=%d on_decode_builds=%" PRIu64
            " off_decode_builds=%" PRIu64 " rebuild_fallbacks=%" PRIu64 " pad_fallbacks=%" PRIu64 "\n",
            int(ok && bitwise == rows), bitwise, rows, bse_steps,
            binder_builds, stock_builds, rebuild_fb, pad_fb);

    r.ok        = ok;
    r.stock_arm = stock_arm;
    r.rows      = rows;
    r.bitwise   = bitwise;

    llama_free(stock);
    llama_free(ctx);
    return r;
}

// ---------------------------------------------------------------------------
// the second positive arm: a 4-token batch
// ---------------------------------------------------------------------------
//
// Same claim, different batch shape. Every step decodes 4 tokens at positions
// 4*step .. 4*step+3, marks all 4 as outputs, and compares all 4 rows. The
// span crosses 256 at the last step (step 63 executes at 256, step 64 at 260),
// so the same KV-boundary crossing is covered with 4 rows per step.
//
// The env, the model, the arm split and the per-step binder assertions are the
// same as the single-token arm. Only the context widths differ, and they must:
// n_batch/n_ubatch/n_outputs_max are build parameters on the n_kv lane.
static bse_arm_result bse_multi(const bse_model & m) {
    bse_arm_result r;
    const int vocab = m.vocab;

    setenv("GGML_TP5_UNION_BINDER", "1", 1);
    unsetenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK");
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    unsetenv("LLAMA_NKV_MASK_OVERRIDE");

    llama_context * ctx = bse_context(m, bse_multi_n_ctx, bse_multi_rows_per_step);
    if (ctx == nullptr) {
        std::fprintf(stderr, "FAIL: binder-stock-equiv: multi binder context init\n");
        return r;
    }

    setenv("LLAMA_GRAPH_REUSE_DISABLE", "1", 1);
    llama_context * stock = bse_context(m, bse_multi_n_ctx, bse_multi_rows_per_step);
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    if (stock == nullptr) {
        std::fprintf(stderr, "FAIL: binder-stock-equiv: multi stock context init\n");
        llama_free(ctx);
        return r;
    }

    bool ok = true;

    const uint64_t binder_builds0 = llama_graph_build_count(ctx);
    const uint64_t stock_builds0  = llama_graph_build_count(stock);

    int      rows           = 0;
    int      bitwise        = 0;
    int64_t  min_nkv        = INT64_MAX;
    int64_t  max_nkv        = 0;
    uint32_t build_n_kv     = 0;
    bool     have_build     = false;
    uint64_t patch_binds0   = 0;
    uint64_t patch_binds1   = 0;
    uint64_t rebuild_fb0    = 0;
    uint64_t rebuild_fb1    = 0;
    uint64_t pad_fb0        = 0;
    uint64_t pad_fb1        = 0;
    uint64_t first_stats_builds = 0;
    uint64_t last_stats_builds  = 0;
    bool     have_first_stats   = false;

    {
        llama_union_diagnostics d0 = {};
        if (!llama_get_union_diagnostics(ctx, &d0)) {
            std::fprintf(stderr, "FAIL: binder-stock-equiv: no union diagnostics on the multi binder arm\n");
            ok = false;
        } else {
            patch_binds0 = d0.patch_binds;
            rebuild_fb0  = d0.rebuild_fallbacks;
            pad_fb0      = d0.pad_fallbacks;
        }
    }

    std::vector<float> got, want;

    for (int step = 0; step < bse_multi_steps && ok; ++step) {
        if (bse_step_multi(ctx, vocab, step, got) != 0) {
            std::fprintf(stderr, "FAIL: binder-stock-equiv: multi binder decode step=%d\n", step);
            ok = false;
            break;
        }

        llama_nkv_runtime_stats st = {};
        if (!llama_get_nkv_runtime_stats(ctx, &st)) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: no n_kv runtime stats at multi step=%d"
                    " (binder off or no record)\n", step);
            ok = false;
            break;
        }
        if (!st.topology_hashed) {
            std::fprintf(stderr, "FAIL: binder-stock-equiv: multi topology not hashed at step=%d\n", step);
            ok = false;
            break;
        }
        if (!st.has_nkv_dims) {
            std::fprintf(stderr, "FAIL: binder-stock-equiv: no n_kv runtime root at multi step=%d\n", step);
            ok = false;
            break;
        }
        if (st.mask_refused != 0) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: the mask fill refused at multi step=%d (refused=%u)\n",
                    step, st.mask_refused);
            ok = false;
            break;
        }
        if (st.mask_filled == 0) {
            std::fprintf(stderr, "FAIL: binder-stock-equiv: no KQ mask filled at multi step=%d\n", step);
            ok = false;
            break;
        }
        if (st.mask_n_kv != st.runtime_n_kv) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: mask filled at %u, graph executes at %u (multi step=%d)\n",
                    st.mask_n_kv, st.runtime_n_kv, step);
            ok = false;
            break;
        }
        // Same read-back floor as the single-token arm. The 4-token batch is
        // below the fill's row cap, so every row must be compared.
        if (st.mask_row_checked == 0) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: the mask row was not read back at multi step=%d (filled=%u)\n",
                    step, st.mask_filled);
            ok = false;
            break;
        }
        if (st.mask_row_ok != st.mask_row_checked) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: the mask row disagrees at multi step=%d (ok=%u checked=%u)\n",
                    step, st.mask_row_ok, st.mask_row_checked);
            ok = false;
            break;
        }
        if (st.mask_row_refused != 0) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: the mask read-back refused %u row(s) at multi step=%d\n",
                    st.mask_row_refused, step);
            ok = false;
            break;
        }
        if (st.mask_rows_all != 1) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: only the last mask row was compared at multi step=%d\n", step);
            ok = false;
            break;
        }
        if (st.bound_n_kv != st.runtime_n_kv) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: bound n_kv %u != runtime n_kv %u (multi step=%d)\n",
                    st.bound_n_kv, st.runtime_n_kv, step);
            ok = false;
            break;
        }
        // the true used span after this step is 4 * (step + 1)
        const int64_t want_nkv = (int64_t) bse_multi_rows_per_step * ((int64_t) step + 1);
        if ((int64_t) st.runtime_n_kv != want_nkv) {
            bse_print_lane_refused(ctx, step, st);
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: runtime_n_kv=%u at multi step=%d (want %" PRId64 ")\n",
                    st.runtime_n_kv, step, want_nkv);
            ok = false;
            break;
        }
        if (!have_build) {
            build_n_kv = st.build_n_kv;
            have_build = true;
        } else if (st.build_n_kv != build_n_kv) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: build width moved at multi step=%d (%u -> %u)\n",
                    step, build_n_kv, st.build_n_kv);
            ok = false;
            break;
        }
        if ((int64_t) st.runtime_n_kv < min_nkv) min_nkv = (int64_t) st.runtime_n_kv;
        if ((int64_t) st.runtime_n_kv > max_nkv) max_nkv = (int64_t) st.runtime_n_kv;

        if (!have_first_stats) {
            first_stats_builds = st.graph_builds;
            have_first_stats   = true;
            if (st.graph_builds != llama_graph_build_count(ctx)) {
                std::fprintf(stderr,
                        "FAIL: binder-stock-equiv: stats graph_builds %" PRIu64 " != graph_build_count %" PRIu64
                        " at the first multi step\n", st.graph_builds, llama_graph_build_count(ctx));
                ok = false;
                break;
            }
        }
        last_stats_builds = st.graph_builds;
        if (step == bse_multi_steps - 1 && st.graph_builds != llama_graph_build_count(ctx)) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: stats graph_builds %" PRIu64 " != graph_build_count %" PRIu64
                    " at the last multi step\n", st.graph_builds, llama_graph_build_count(ctx));
            ok = false;
            break;
        }

        // --- the stock arm: the SAME 4-token step, once -------------------
        if (bse_step_multi(stock, vocab, step, want) != 0) {
            std::fprintf(stderr, "FAIL: binder-stock-equiv: multi stock decode step=%d\n", step);
            ok = false;
            break;
        }
        rows += bse_multi_rows_per_step;
        int first_bad = -1;
        const int d = bse_diff_rows_multi(got, want, bse_multi_rows_per_step, vocab, first_bad);
        if (d < 0) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: multi logits rows not comparable at step=%d\n", step);
            ok = false;
        } else if (d == 0) {
            bitwise += bse_multi_rows_per_step;
        } else {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: multi logits differ from stock at step=%d row=%d"
                    " (differing rows=%d, runtime_n_kv=%u)\n",
                    step, first_bad, d, st.runtime_n_kv);
            ok = false;
        }
    }

    const uint64_t binder_builds = llama_graph_build_count(ctx) - binder_builds0;
    const uint64_t stock_builds  = llama_graph_build_count(stock) - stock_builds0;

    {
        llama_union_diagnostics d1 = {};
        if (!llama_get_union_diagnostics(ctx, &d1)) {
            std::fprintf(stderr, "FAIL: binder-stock-equiv: no union diagnostics after the multi decode\n");
            ok = false;
        } else {
            patch_binds1 = d1.patch_binds;
            rebuild_fb1  = d1.rebuild_fallbacks;
            pad_fb1      = d1.pad_fallbacks;
        }
    }
    const uint64_t patch_binds = patch_binds1 - patch_binds0;
    const uint64_t rebuild_fb  = rebuild_fb1 - rebuild_fb0;
    const uint64_t pad_fb      = pad_fb1 - pad_fb0;

    if (ok && rows != bse_multi_rows) {
        std::fprintf(stderr,
                "FAIL: binder-stock-equiv: multi compared %d of %d rows\n", rows, bse_multi_rows);
        ok = false;
    }

    if (ok) {
        if (min_nkv >= 256 || max_nkv <= 256) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: the multi executed span did not cross 256"
                    " (min=%" PRId64 " max=%" PRId64 ")\n", min_nkv, max_nkv);
            ok = false;
        }
        // The bind witness is patch_binds > 0 (below) together with the
        // per-step bound_n_kv == runtime_n_kv assertion; runtime_n_kv is
        // re-armed from the KV cache every ubatch (src/llama-context.cpp:5687),
        // so its movement alone is not evidence that a bind happened.
        if (patch_binds == 0) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: multi patch_binds stayed 0; no bind was applied\n");
            ok = false;
        }
        if (last_stats_builds != first_stats_builds) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: the stats path saw a build between the first and the last"
                    " multi step (%" PRIu64 " -> %" PRIu64 ")\n",
                    first_stats_builds, last_stats_builds);
            ok = false;
        }
        if (rebuild_fb != 0) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: %" PRIu64 " rebuild fallback(s) during the multi decode;"
                    " the exact lane did not deliver the span\n", rebuild_fb);
            ok = false;
        }
        if (pad_fb != 0) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: %" PRIu64 " padded fallback(s) during the multi decode;"
                    " the exact lane did not deliver the span\n", pad_fb);
            ok = false;
        }
        if (binder_builds != 1) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: the multi binder arm built %" PRIu64
                    " graph(s) for %d steps (want 1)\n", binder_builds, bse_multi_steps);
            ok = false;
        }
        if (stock_builds < (uint64_t) bse_multi_steps) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: the multi stock arm built %" PRIu64 " graph(s) for %d steps"
                    " (LLAMA_GRAPH_REUSE_DISABLE did not take effect)\n", stock_builds, bse_multi_steps);
            ok = false;
        }
    }

    std::fprintf(stderr,
            "BSE_MULTI ok=%d rows=%d/%d steps=%d on_decode_builds=%" PRIu64
            " off_decode_builds=%" PRIu64 " rebuild_fallbacks=%" PRIu64 " pad_fallbacks=%" PRIu64 "\n",
            int(ok && bitwise == rows), bitwise, rows, bse_multi_steps,
            binder_builds, stock_builds, rebuild_fb, pad_fb);

    r.ok      = ok;
    r.rows    = rows;
    r.bitwise = bitwise;

    llama_free(stock);
    llama_free(ctx);
    return r;
}

// ---------------------------------------------------------------------------
// the falsifiability arm
// ---------------------------------------------------------------------------
//
// A comparison that cannot fail proves nothing, so this arm runs the same
// memcmp between the binder arm and a stock context that is deliberately fed a
// DIFFERENT token history, and requires the comparison to report differ > 0.
//
// The perturbation is one token id, at one step, on the perturbed arm only.
// It is the smallest perturbation that provably changes logits, and it needs
// no hook, no env var and no binder-only code path:
//
//   - the token id is the only input that differs between the two arms at that
//     step. Both arms have the same weights, the same positions, the same
//     n_ctx, the same KV type and the same batch shape (bse_step() is the one
//     decode helper for both arms);
//   - a different token id selects a different row of tok_embd, and every
//     later node of the step derives from it, so the logits row of that step
//     cannot be bitwise equal unless the embedding lookup is ignored;
//   - the history then stays different for every later step (the KV cache
//     holds the perturbed key/value), so the control does not depend on a
//     single step being sensitive.
//
// Why NOT LLAMA_NKV_MASK_OVERRIDE: that hook is read only by
// llm_graph_input_nkv_runtime::set_input (llama-graph.cpp:328-345), and that
// input exists only on a binder graph (build_inp_nkv_runtime returns nullptr
// without the binder, llama-graph.cpp:2985-2990). A stock arm never reads it,
// and a binder arm would abort on a width above its own mask ne[0]
// (llama-graph.cpp:380-385). It cannot perturb the stock arm, so it is not
// used here.
//
// Why NOT a different n_ctx bucket: the binder arm's capacity is the exact
// n_ctx (llama-context.cpp:691, runtime_len_capacity), and the stock arm's is
// the 256-cell bucket of the same value. Choosing a bucket that changes the
// visible set would also change the binder arm's capacity, so the two effects
// cannot be separated from the source alone. The token perturbation has no
// such coupling.
//
// The arm fails loudly when the perturbation is unreachable: a decode error, a
// row-size mismatch, or zero rows all fail it, and differ == 0 fails it too.
//
// The perturbed token id is `bse_step()`'s id + 1, wrapped into the same
// vocabulary range bse_step() uses ([1, vocab-1]). The two ids differ for
// every vocab > 2, which bse_open() already requires (vocab > 3).
static int bse_step_perturbed(llama_context * ctx, int vocab, int pos, std::vector<float> & logits_out) {
    logits_out.assign((size_t) vocab, 0.0f);

    llama_batch batch = llama_batch_init(1, 0, 1);
    if (!llama_batch_is_complete(&batch, 1, false, true)) {
        llama_batch_free(batch);
        GGML_ABORT("llama_batch_init failed");
    }
    batch.n_tokens     = 1;
    // bse_step()'s id is 1 + (pos*7) % (vocab-1), in [1, vocab-1]; +1 wrapped
    // into the same range is a different id for vocab > 2.
    const int base = 1 + (pos * 7) % (vocab - 1);
    batch.token[0]     = (llama_token) (1 + (base - 1 + 1) % (vocab - 1));
    batch.pos[0]       = (llama_pos) pos;
    batch.n_seq_id[0]  = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0]    = 1;

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        return rc;
    }
    const float * src = llama_get_logits_ith(ctx, 0);
    if (src == nullptr) {
        return -1;
    }
    std::memcpy(logits_out.data(), src, sizeof(float) * (size_t) vocab);
    return 0;
}

// The control decodes 32 steps: enough that the perturbed history has moved
// past the perturbed step for most of the run, so a comparison that only
// happens to differ at one step still counts as a difference.
static constexpr int bse_neg_steps = 32;

static bool bse_neg_control(const bse_model & m) {
    const int vocab = m.vocab;
    const int n_ctx = bse_steps + 8;

    // This function builds its own context pair. The constructors already read
    // the arm variables for bse_positive()'s contexts, and nothing here can
    // change those; these setenv calls select the arms for the two contexts
    // created below and for no others.
    setenv("GGML_TP5_UNION_BINDER", "1", 1);
    unsetenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK");
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    unsetenv("LLAMA_NKV_MASK_OVERRIDE");

    llama_context * ctx = bse_context(m, n_ctx, 1);
    if (ctx == nullptr) {
        std::fprintf(stderr, "FAIL: binder-stock-equiv: neg binder context init\n");
        return false;
    }

    setenv("LLAMA_GRAPH_REUSE_DISABLE", "1", 1);
    llama_context * stock = bse_context(m, n_ctx, 1);
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    if (stock == nullptr) {
        std::fprintf(stderr, "FAIL: binder-stock-equiv: neg stock context init\n");
        llama_free(ctx);
        return false;
    }

    bool ok = true;
    int  differ = 0;
    int  rows   = 0;
    std::vector<float> got, want;

    for (int step = 0; step < bse_neg_steps && ok; ++step) {
        if (bse_step(ctx, vocab, step, got) != 0) {
            std::fprintf(stderr, "FAIL: binder-stock-equiv: neg binder decode step=%d\n", step);
            ok = false;
            break;
        }
        // the perturbed arm: the same position, a different token id
        if (bse_step_perturbed(stock, vocab, step, want) != 0) {
            std::fprintf(stderr,
                    "FAIL: binder-stock-equiv: neg perturbed decode step=%d"
                    " (the perturbation is not reachable)\n", step);
            ok = false;
            break;
        }
        ++rows;
        const int d = bse_diff_rows(got, want);
        if (d < 0) {
            std::fprintf(stderr, "FAIL: binder-stock-equiv: neg row size mismatch at step=%d\n", step);
            ok = false;
            break;
        }
        differ += d;
    }

    // The arm is a control, so its verdict is inverted: it passes only when
    // the comparison DID differ. differ == 0 means the bitwise comparison
    // cannot distinguish a context fed a different token from the binder arm,
    // and the positive arm's equality would then be vacuous. rows == 0 means
    // the loop never compared anything, which is the same vacuity reached by a
    // different route.
    if (ok && rows != bse_neg_steps) {
        std::fprintf(stderr,
                "FAIL: binder-stock-equiv: the negative control compared %d of %d rows\n",
                rows, bse_neg_steps);
        ok = false;
    }
    if (ok && differ == 0) {
        std::fprintf(stderr,
                "FAIL: binder-stock-equiv: the negative control produced identical logits;"
                " the bitwise comparison cannot fail\n");
        ok = false;
    }

    std::fprintf(stderr, "BSE_NEG ok=%d differ=%d\n", int(ok), differ);

    llama_free(stock);
    llama_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    const bool neg_only = argc == 3 && std::strcmp(argv[1], "--neg-control") == 0;
    const char * model_path = neg_only ? argv[2] : (argc == 2 ? argv[1] : nullptr);

    if (model_path == nullptr) {
        std::fprintf(stderr, "usage: %s model.gguf\n"
                             "       %s --neg-control model.gguf\n", argv[0], argv[0]);
        return 2;
    }
    if (model_path[0] == '\0') {
        std::fprintf(stderr, "FAIL: empty model path\n");
        return 2;
    }

    bse_model m;
    if (!bse_open(model_path, m)) {
        return 1;
    }

    bool ok = true;
    if (neg_only) {
        ok = bse_neg_control(m);
    } else {
        // The kernel-selection gate runs first and is reported first: when it
        // is red, the equality below is not a claim the exact lane may make,
        // and the exit status must say so whatever the numbers are. Both
        // positive arms still run, so their numbers are visible as evidence
        // rather than hidden behind the gate.
        const bool kernel_sel = bse_kernel_sel();
        const bse_arm_result single = bse_positive(m);
        const bse_arm_result multi  = bse_multi(m);
        const int total_rows    = single.rows + multi.rows;
        const int total_bitwise = single.bitwise + multi.bitwise;
        // BSE_BITWISE is about the COMPARISON only: every row of both arms was
        // produced and every produced row compared bitwise equal. It
        // deliberately does not fold in the arm verdicts: a build counter or a
        // fallback moving is a different claim, and folding it in here would
        // hide which claim failed. Those verdicts live on BSE_SINGLE /
        // BSE_MULTI, and the conjunction lives on BSE_END.
        const bool bitwise_ok = total_rows > 0 &&
            single.rows == bse_steps &&
            multi.rows  == bse_multi_rows &&
            total_bitwise == total_rows;
        std::fprintf(stderr, "BSE_BITWISE ok=%d rows=%d/%d\n",
                int(bitwise_ok), total_bitwise, total_rows);
        const bool neg = bse_neg_control(m);
        ok = kernel_sel && single.stock_arm && single.ok && multi.ok && neg;
        std::fprintf(stderr, "BSE_END ok=%d\n", int(ok));
    }

    bse_close(m);
    return ok ? 0 : 1;
}
