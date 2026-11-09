#include "llama.h"

#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

// CPU regression: the token lane binds in place, rebuilds nothing, and
// reproduces the per-length rebuild bit for bit.
//
// Claim, in one sentence: with GGML_TP5_UNION_BINDER=1 and
// LLAMA_EXPERIMENTAL_VARLEN_NTOK=1, ONE context decodes a schedule of variable
// token widths (1, 3, 2, 4), applies an in-place bind on every step after the
// record build, builds no graph after its first decode, and produces logits
// byte-identical to a fresh context that rebuilt its graph at each captured
// length.
//
// Arms (both variables are read ONCE in the llama_context constructor, so they
// are set around construction only):
//
//   binder arm    GGML_TP5_UNION_BINDER=1, LLAMA_EXPERIMENTAL_VARLEN_NTOK=1,
//                 LLAMA_GRAPH_REUSE_DISABLE unset. ONE context, n_ctx = the
//                 total token count. The gate keeps n_ctx raw (see the
//                 runtime_len_capacity branch of the llama_context
//                 constructor), so the capacity is the live
//                 span, not a 256-cell bucket. The first decode builds the
//                 record at the live KV, token and output widths; every later
//                 step must move the token and KV axes by an in-place bind.
//
//   baseline arm  GGML_TP5_UNION_BINDER=1, LLAMA_GRAPH_REUSE_DISABLE=1,
//                 LLAMA_EXPERIMENTAL_VARLEN_NTOK unset. A FRESH context per
//                 captured step, replaying the same prefix. The binder gate is
//                 left ON on purpose: the same gate decides n_ctx rounding, so
//                 both arms have the same raw capacity and the only difference
//                 is the binder record (the construction of
//                 test-union-fixed-topology's native arm). Reuse is disabled,
//                 so every decode builds. The test ASSERTS that measured count
//                 after every decode (min_step_builds >= 1, builds >= decodes)
//                 instead of assuming it.
//
// What is asserted, each as its own machine line:
//
//   NTOK_BIND    per step: the graph build count, patch_binds,
//                rebuild_fallbacks, pad_fallbacks, bound_tokens, bound_n_kv,
//                runtime_n_kv, build_n_tokens, has_nkv_dims, the lane flags
//                and the topology hash. Required: the lane is enabled, the
//                envelope is not disabled, the record and the token map are
//                valid, the K/V width dimensions are recorded (has_nkv_dims:
//                the K and V write views, the V one reshaped to the 2-D
//                cache-write layout because flash attention is off),
//                bound_tokens == the batch width, bound_n_kv == pos + width,
//                runtime_n_kv == pos + width (the consumer-visible width the
//                mask fill and the attention kernels read, so the committed
//                runtime_n_kv=0 defect is covered directly), build_n_tokens ==
//                the first width, rebuild_fallbacks == 0, pad_fallbacks == 0,
//                the build count does not move after the first decode, and
//                patch_binds strictly increases on every step after the record
//                build.
//
//                patch_binds alone is NOT sufficient evidence:
//                union_rebind_apply increments c_patch_binds BEFORE its
//                can_reuse() post-check, and that post-check clears
//                record_valid on failure. The step assertion therefore requires
//                record_valid && !envelope_disabled together with the bind
//                count, and the next step's llama_get_nkv_runtime_stats()
//                returns false once the record is gone. A counted-but-rejected
//                bind cannot pass this test.
//
//                The record build itself applies no patch when nothing has
//                moved yet (union_bind_is_noop), so its bind count is not
//                required to move.
//   NTOK_BASE    per captured step: the baseline's decode count, total measured
//                graph build count, and the MINIMUM per-decode build delta.
//                min_step_builds >= 1 is asserted, so every decode of the
//                baseline really built a graph; the total alone could hide one
//                reused decode behind another decode's extra build.
//   NTOK_REPLAY  per captured step: the bitwise verdict against the baseline.
//   NTOK_END     the conjunction: steps, final binds (> 0), final rebuilds
//                (0), bitwise == compared == steps, and the FINAL step's
//                record_valid && !envelope_disabled.
//
// What this test does NOT prove:
//   - anything about GPU/HIP, flash attention, quantized V or multi-sequence
//     batches: the lane refuses them by construction (token_candidate);
//   - that the baseline is upstream llama.cpp: it is the same binary with graph
//     reuse disabled, not an upstream build;
//   - that every intermediate layer matches: only the requested logits row is
//     compared.
//
// usage: test-ntok-bind-replay <model.gguf> [rounds]
//   rounds defaults to 3 (12 steps, 30 tokens) and must be in [1, 16].
//   A missing or unreadable model is an error, never a skip.

namespace {

const int k_widths[]   = { 1, 3, 2, 4 };
const int k_cycle      = (int) (sizeof(k_widths) / sizeof(k_widths[0]));
const int k_per_round  = 1 + 3 + 2 + 4;
const int k_rounds_max = 16;

struct tok_model {
    llama_model * model = nullptr;
    int           vocab = 0;
};

bool tok_open(const char * path, tok_model & m) {
    llama_backend_init();

    auto mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = { nullptr };
    mp.devices      = devices;
    mp.n_gpu_layers = 0;

    m.model = llama_model_load_from_file(path, mp);
    if (m.model == nullptr) {
        std::fprintf(stderr, "FAIL: model load: %s\n", path);
        llama_backend_free();
        return false;
    }
    m.vocab = (int) llama_vocab_n_tokens(llama_model_get_vocab(m.model));
    if (m.vocab <= 4) {
        std::fprintf(stderr, "FAIL: vocab=%d, need > 4\n", m.vocab);
        llama_model_free(m.model);
        m.model = nullptr;
        llama_backend_free();
        return false;
    }
    return true;
}

void tok_close(tok_model & m) {
    if (m.model != nullptr) {
        llama_model_free(m.model);
        m.model = nullptr;
    }
    llama_backend_free();
}

// One context shape for BOTH arms. n_batch = n_ubatch = the widest batch in
// the schedule, so the token lane is never limited by the batch width.
llama_context * tok_context(const tok_model & m, uint32_t n_ctx) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_ctx_kv        = n_ctx;
    cp.n_batch         = (uint32_t) k_widths[3];
    cp.n_ubatch        = (uint32_t) k_widths[3];
    cp.n_outputs_max   = 1;
    cp.n_seq_max       = 1;
    cp.n_threads       = 1;
    cp.n_threads_batch = 1;
    cp.kv_unified      = true;
    cp.offload_kqv     = false;
    cp.op_offload      = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.embeddings      = false;
    cp.no_perf         = true;
    return llama_init_from_model(m.model, cp);
}

// One token id per position, in [1, vocab-1] so the embedding lookup is always
// a real row. The same function feeds both arms.
int tok_token(int pos, int vocab) {
    return 1 + (pos * 7 + 1) % (vocab - 1);
}

// One batch of `width` tokens at `pos`. Only the last row requests logits, so
// n_outputs stays 1 and the token and KV axes are the only moving runtime
// parameters. `row` receives the full logits row of that last token.
int tok_decode(llama_context * ctx, const tok_model & m, int pos, int width, std::vector<float> & row) {
    llama_batch batch = llama_batch_init(width, 0, 1);
    if (!llama_batch_is_complete(&batch, width, false, true)) {
        std::fprintf(stderr, "FAIL: batch allocation width=%d\n", width);
        llama_batch_free(batch);
        batch = llama_batch{};
        return -1;
    }
    batch.n_tokens = width;
    for (int i = 0; i < width; ++i) {
        batch.token[i]     = (llama_token) tok_token(pos + i, m.vocab);
        batch.pos[i]       = (llama_pos) (pos + i);
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == width - 1) ? 1 : 0;
    }

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: decode pos=%d width=%d rc=%d\n", pos, width, rc);
        return rc;
    }

    const float * src = llama_get_logits_ith(ctx, width - 1);
    if (src == nullptr) {
        std::fprintf(stderr, "FAIL: no logits row pos=%d width=%d\n", pos, width);
        return -1;
    }
    row.assign((size_t) m.vocab, 0.0f);
    std::memcpy(row.data(), src, sizeof(float) * (size_t) m.vocab);
    return 0;
}

struct bind_cell {
    int      step  = 0;
    int      width = 0;
    int      pos   = 0;
    uint64_t builds = 0;
    uint64_t binds  = 0;
    uint64_t rebuilds = 0;
    uint64_t pads   = 0;
    uint32_t bound_tokens   = 0;
    uint32_t bound_n_kv     = 0;
    uint32_t runtime_n_kv   = 0;
    uint32_t build_n_tokens = 0;
    uint64_t topology_hash  = 0;
    bool     ok = false;
    bool     record_valid = false;
    bool     envelope_disabled = false;
    std::vector<float> row;
};

// The binder arm: one context, the whole schedule, one bind per moved step.
bool tok_binder_arm(const tok_model & m, int rounds, std::vector<bind_cell> & cells) {
    setenv("GGML_TP5_UNION_BINDER", "1", 1);
    setenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK", "1", 1);
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    unsetenv("LLAMA_NKV_MASK_OVERRIDE");

    const int      n_steps = rounds * k_cycle;
    const uint32_t n_ctx   = (uint32_t) (rounds * k_per_round);

    llama_context * ctx = tok_context(m, n_ctx);
    if (ctx == nullptr) {
        std::fprintf(stderr, "FAIL: binder arm: context init\n");
        return false;
    }

    bool     ok                 = true;
    uint64_t builds_after_first = 0;
    uint64_t previous_binds     = 0;
    int      pos                = 0;

    for (int step = 0; step < n_steps; ++step) {
        const int width = k_widths[step % k_cycle];

        bind_cell c;
        c.step  = step;
        c.width = width;
        c.pos   = pos;

        if (tok_decode(ctx, m, pos, width, c.row) != 0) {
            ok = false;
            break;
        }

        const uint64_t builds_now = llama_graph_build_count(ctx);
        if (step == 0) {
            builds_after_first = builds_now;
        }

        llama_union_diagnostics d  = {};
        llama_nkv_runtime_stats st = {};
        const bool have_d  = llama_get_union_diagnostics(ctx, &d);
        const bool have_st = llama_get_nkv_runtime_stats(ctx, &st);

        c.builds          = builds_now;
        c.binds           = d.patch_binds;
        c.rebuilds        = d.rebuild_fallbacks;
        c.pads            = d.pad_fallbacks;
        c.bound_tokens    = d.bound_tokens;
        c.bound_n_kv      = d.bound_n_kv;
        c.runtime_n_kv    = st.runtime_n_kv;
        c.build_n_tokens  = st.build_n_tokens;
        c.topology_hash   = d.topology_hash;
        c.record_valid      = d.record_valid;
        c.envelope_disabled = d.envelope_disabled;

        std::fprintf(stderr,
                "NTOK_BIND step=%d width=%d pos=%d builds=%" PRIu64 " binds=%" PRIu64
                " rebuilds=%" PRIu64 " pads=%" PRIu64 " bound_tokens=%u bound_n_kv=%u"
                " runtime_n_kv=%u build_n_tokens=%u nkv_dims=%d token_map=%d enabled=%d"
                " envelope_disabled=%d record=%d hash=%016" PRIx64 "\n",
                step, width, pos, c.builds, c.binds, c.rebuilds, c.pads,
                c.bound_tokens, c.bound_n_kv, c.runtime_n_kv, c.build_n_tokens,
                int(st.has_nkv_dims), int(d.token_map_valid), int(d.enabled),
                int(d.envelope_disabled), int(d.record_valid), c.topology_hash);

        bool cell_ok = have_d && have_st;
        if (!have_d) {
            std::fprintf(stderr, "FAIL: step=%d: no union diagnostics\n", step);
        }
        if (!have_st) {
            std::fprintf(stderr, "FAIL: step=%d: no n_kv runtime stats (no live record)\n", step);
        }
        if (cell_ok && (!d.enabled || d.envelope_disabled || !d.record_valid || !d.token_map_valid)) {
            std::fprintf(stderr,
                    "FAIL: step=%d: lane not live (enabled=%d envelope_disabled=%d record=%d token_map=%d)\n",
                    step, int(d.enabled), int(d.envelope_disabled), int(d.record_valid),
                    int(d.token_map_valid));
            cell_ok = false;
        }
        if (cell_ok && (!st.has_ntok_dims || st.build_n_tokens != (uint32_t) k_widths[0])) {
            std::fprintf(stderr,
                    "FAIL: step=%d: token dimensions not recorded at the first width"
                    " (has_ntok=%d build_n_tokens=%u want=%d)\n",
                    step, int(st.has_ntok_dims), st.build_n_tokens, k_widths[0]);
            cell_ok = false;
        }
        if (cell_ok && !st.has_nkv_dims) {
            std::fprintf(stderr,
                    "FAIL: step=%d: the K/V width dimensions are not recorded"
                    " (has_nkv_dims=0); the V write reshape is not covered\n",
                    step);
            cell_ok = false;
        }
        if (cell_ok && (d.bound_tokens != (uint32_t) width || st.bound_n_tokens != (uint32_t) width)) {
            std::fprintf(stderr,
                    "FAIL: step=%d: bound token width disagrees with the batch"
                    " (diag=%u stats=%u want=%d)\n",
                    step, d.bound_tokens, st.bound_n_tokens, width);
            cell_ok = false;
        }
        if (cell_ok && d.bound_n_kv != (uint32_t) (pos + width)) {
            std::fprintf(stderr,
                    "FAIL: step=%d: bound_n_kv=%u, want the used span %d\n",
                    step, d.bound_n_kv, pos + width);
            cell_ok = false;
        }
        // The consumer-visible width, not only the binder's own bound value:
        // the mask fill and every attention kernel read runtime_n_kv. The
        // committed defect left it 0 for every width above 1 while
        // bound_n_kv looked right, so this assertion is the direct cover.
        if (cell_ok && st.runtime_n_kv != (uint32_t) (pos + width)) {
            std::fprintf(stderr,
                    "FAIL: step=%d: runtime_n_kv=%u, want the used span %d\n",
                    step, st.runtime_n_kv, pos + width);
            cell_ok = false;
        }
        if (cell_ok && d.rebuild_fallbacks != 0) {
            std::fprintf(stderr, "FAIL: step=%d: %" PRIu64 " rebuild fallback(s)\n",
                    step, d.rebuild_fallbacks);
            cell_ok = false;
        }
        if (cell_ok && d.pad_fallbacks != 0) {
            std::fprintf(stderr, "FAIL: step=%d: %" PRIu64 " padded fallback(s)\n",
                    step, d.pad_fallbacks);
            cell_ok = false;
        }
        if (cell_ok && builds_now != builds_after_first) {
            std::fprintf(stderr,
                    "FAIL: step=%d: the graph was rebuilt after the first decode"
                    " (%" PRIu64 " -> %" PRIu64 ")\n",
                    step, builds_after_first, builds_now);
            cell_ok = false;
        }
        // The record build applies no patch when nothing has moved yet
        // (union_bind_is_noop), so the first step is not required to bind. Every
        // step after it moves the token and KV axes, so every one must bind.
        if (cell_ok && step > 0 && d.patch_binds <= previous_binds) {
            std::fprintf(stderr,
                    "FAIL: step=%d: no in-place bind (patch_binds=%" PRIu64
                    ", previous=%" PRIu64 ")\n",
                    step, d.patch_binds, previous_binds);
            cell_ok = false;
        }

        c.ok = cell_ok;
        ok   = ok && cell_ok;
        previous_binds = d.patch_binds;
        cells.push_back(std::move(c));
        pos += width;

        if (!cell_ok) {
            break;
        }
    }

    if (ok && pos != (int) n_ctx) {
        std::fprintf(stderr, "FAIL: binder arm decoded %d of %d tokens\n", pos, (int) n_ctx);
        ok = false;
    }
    if (ok && !cells.empty() && cells.back().binds == 0) {
        std::fprintf(stderr, "FAIL: binder arm applied no in-place bind at all\n");
        ok = false;
    }

    llama_free(ctx);
    return ok;
}

// The baseline arm: a fresh context per captured step with graph reuse
// disabled, replaying the same prefix. Every decode builds, so the graph at
// the captured step was built at that step's live widths. The build count is
// sampled after EVERY decode: a total alone could hide one reused decode
// behind another decode's extra build.
bool tok_baseline_arm(const tok_model & m, int upto_step, uint32_t n_ctx,
                      std::vector<float> & row, uint64_t & builds,
                      uint64_t & min_step_builds, int & decodes) {
    setenv("GGML_TP5_UNION_BINDER", "1", 1);
    setenv("LLAMA_GRAPH_REUSE_DISABLE", "1", 1);
    unsetenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK");
    unsetenv("LLAMA_NKV_MASK_OVERRIDE");

    llama_context * ctx = tok_context(m, n_ctx);
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE"); // the constructor already read it
    if (ctx == nullptr) {
        std::fprintf(stderr, "FAIL: baseline arm: context init at step=%d\n", upto_step);
        return false;
    }

    const uint64_t b0    = llama_graph_build_count(ctx);
    uint64_t       prev  = b0;
    uint64_t       min_step = UINT64_MAX;
    bool ok = true;
    int  pos = 0;
    for (int step = 0; step <= upto_step; ++step) {
        const int width = k_widths[step % k_cycle];
        if (tok_decode(ctx, m, pos, width, row) != 0) {
            ok = false;
            break;
        }
        const uint64_t now   = llama_graph_build_count(ctx);
        const uint64_t delta = now - prev;
        if (delta < min_step) {
            min_step = delta;
        }
        if (delta == 0) {
            std::fprintf(stderr,
                    "FAIL: baseline step=%d reused its graph; the per-length rebuild did not happen\n",
                    step);
            ok = false;
        }
        prev = now;
        pos += width;
    }

    builds          = prev - b0;
    min_step_builds = min_step == UINT64_MAX ? 0 : min_step;
    decodes         = upto_step + 1;

    llama_free(ctx);
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s model.gguf [rounds]\n", argv[0]);
        return 2;
    }
    if (argv[1][0] == '\0') {
        std::fprintf(stderr, "FAIL: empty model path\n");
        return 2;
    }

    int rounds = 3;
    if (argc == 3) {
        char * end = nullptr;
        errno = 0;
        const long v = std::strtol(argv[2], &end, 10);
        if (errno != 0 || end == argv[2] || *end != '\0' || v < 1 || v > k_rounds_max) {
            std::fprintf(stderr, "FAIL: rounds must be an integer in [1, %d]\n", k_rounds_max);
            return 2;
        }
        rounds = (int) v;
    }

    tok_model m;
    if (!tok_open(argv[1], m)) {
        return 1;
    }

    const int total       = rounds * k_per_round;
    const int n_ctx_train = llama_model_n_ctx_train(m.model);
    if (total > n_ctx_train) {
        std::fprintf(stderr, "FAIL: %d tokens exceed the model context %d\n", total, n_ctx_train);
        tok_close(m);
        return 2;
    }

    std::vector<bind_cell> cells;
    bool ok = tok_binder_arm(m, rounds, cells);

    int      bitwise         = 0;
    int      compared        = 0;
    uint64_t baseline_builds = 0;

    for (const bind_cell & c : cells) {
        if (!c.ok) {
            continue;
        }

        std::vector<float> base_row;
        uint64_t builds = 0;
        uint64_t min_step_builds = 0;
        int      decodes = 0;
        if (!tok_baseline_arm(m, c.step, (uint32_t) total, base_row, builds,
                              min_step_builds, decodes)) {
            std::fprintf(stderr, "FAIL: baseline arm: step=%d\n", c.step);
            ok = false;
            continue;
        }
        baseline_builds += builds;

        // The baseline must really have rebuilt at EVERY decode. The minimum
        // per-decode delta is the direct measurement; the total is printed for
        // context but cannot prove per-step rebuilds on its own.
        const bool rebuilt = builds >= (uint64_t) decodes && min_step_builds >= 1;
        std::fprintf(stderr,
                "NTOK_BASE step=%d len=%d decodes=%d builds=%" PRIu64
                " min_step_builds=%" PRIu64 " rebuilt=%d\n",
                c.step, c.pos + c.width, decodes, builds, min_step_builds, int(rebuilt));
        if (!rebuilt) {
            std::fprintf(stderr,
                    "FAIL: baseline step=%d did not rebuild at every decode"
                    " (%d decode(s), %" PRIu64 " build(s), min_step=%" PRIu64 ")\n",
                    c.step, decodes, builds, min_step_builds);
            ok = false;
        }

        int diff = -1;
        if (base_row.size() == c.row.size() && !c.row.empty()) {
            diff = std::memcmp(c.row.data(), base_row.data(),
                               c.row.size() * sizeof(float)) == 0 ? 0 : 1;
        }
        std::fprintf(stderr, "NTOK_REPLAY step=%d width=%d bitwise=%d\n",
                c.step, c.width, diff == 0 ? 1 : 0);
        if (diff < 0) {
            std::fprintf(stderr, "FAIL: logits not comparable at step=%d\n", c.step);
            ok = false;
        } else if (diff != 0) {
            std::fprintf(stderr,
                    "FAIL: logits differ from the per-length rebuild at step=%d"
                    " width=%d bound_n_kv=%u\n",
                    c.step, c.width, c.bound_n_kv);
            ok = false;
        } else {
            ++bitwise;
        }
        ++compared;
    }

    const uint64_t final_binds    = cells.empty() ? 0 : cells.back().binds;
    const uint64_t final_rebuilds = cells.empty() ? 0 : cells.back().rebuilds;
    const bool     final_record   = !cells.empty() && cells.back().record_valid;
    const bool     final_envelope = !cells.empty() && cells.back().envelope_disabled;
    const bool end_ok = ok &&
                        compared == (int) cells.size() &&
                        bitwise  == compared &&
                        final_binds > 0 &&
                        final_rebuilds == 0 &&
                        final_record &&
                        !final_envelope;

    std::fprintf(stderr,
            "NTOK_END ok=%d steps=%zu compared=%d binds=%" PRIu64 " rebuilds=%" PRIu64
            " bitwise=%d/%d record=%d envelope_disabled=%d baseline_builds=%" PRIu64 "\n",
            int(end_ok), cells.size(), compared, final_binds, final_rebuilds,
            bitwise, compared, int(final_record), int(final_envelope), baseline_builds);

    tok_close(m);
    return end_ok ? 0 : 1;
}
