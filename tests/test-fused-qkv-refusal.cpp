// G4: the fused-QKV projection must FAIL CLOSED on the token lane.
//
// Claim, in one sentence: given a GGUF whose attention projection is one fused
// `blk.N.attn_qkv.weight` (Q rows || K rows || V rows), a context built with
// GGML_TP5_UNION_BINDER=1 and LLAMA_EXPERIMENTAL_VARLEN_NTOK=1 names the
// refusal, disables the binder envelope, applies no in-place bind, and
// produces logits byte-identical to a stock context at every decode step.
//
// Why the token lane cannot bind a fused projection
// -------------------------------------------------
// llm_graph_context::build_qkv (src/llama-graph.cpp) slices the single fused
// matmul result into Q/K/V with ggml_view_3d. Q starts at row 0, K at row
// n_embd_q and V at row n_embd_q + n_embd_k, each with the fused row stride --
// a non-contiguous row slice of one tensor, not three independent projections.
// When the token lane is armed (`strict_ntok`) the three views are tagged with
// LLM_FUSED_QKV_SLICE_MARKER (src/llama-graph.h). union_shape_analyze
// (src/llama-context.cpp, GGML_OP_VIEW) refuses such a view and sets
// map.fused_qkv_refused; process_ubatch counts it in c_fused_qkv_refused, logs
// "token lane refused: fused QKV projection", sets env_disabled, invalidates
// the record and rebuilds the stock graph at the stock width.
//
// Arms (every environment variable below is read ONCE, in the llama_context
// constructor, so it is set around construction only):
//
//   refusal arm   GGML_TP5_UNION_BINDER=1
//                 LLAMA_EXPERIMENTAL_VARLEN_NTOK=1
//                 everything else unset. One context decodes k_steps
//                 single-token steps; the first decode builds the token-lane
//                 record, the shape pass refuses it, and every step after that
//                 runs the stock graph.
//
//   stock arm     GGML_TP5_UNION_BINDER=0 (the documented stock value; the
//                 gate is also the only thing that keeps n_ctx raw, see
//                 below). One fresh context decodes the SAME token/position
//                 sequence. It takes no binder record and applies no bind.
//
//                 The presence of the diagnostics slot is NOT the witness: the
//                 product creates the per-context state lazily from
//                 process_ubatch's bind-trace block
//                 (src/llama-context.cpp, "auto & st_dbg =
//                 llama_nkv_ctx_state_get(this)"), which runs on every ubatch
//                 whether the gate is on or off. With the gate off that state
//                 is default-constructed and nothing reads it for a decision.
//                 What makes this arm stock is the CONTENT of the diagnostics:
//                 enabled=0, patch_binds=0, env_disabled=0. A run that shows
//                 any of those moved took the binder path and is a failure.
//
// n_ctx is 256 on purpose. With the binder gate on, the exact sub-lane keeps
// n_ctx raw; with the gate off the constructor rounds n_ctx up to the 256-cell
// KV block (src/llama-context.cpp, "stock rounds n_ctx up"). At 256 both
// roundings are the identity, so the two arms run with the same capacity and
// the same stock graph width, and a bitwise logits comparison is comparing the
// refusal decision, not a capacity difference. The refusal arm's fallback
// rebuild uses the same stock width for the same reason.
//
// What is asserted, in order (each failure prints a named "FAIL:" line):
//
//   1. FUSED_QKV_STEP   per step: the refusal arm's logits row is byte-
//                       identical (memcmp over n_vocab floats) to the stock
//                       arm's row at the same token and position.
//   2. FUSED_QKV_ARM    the refusal arm's public diagnostics:
//                       llama_get_union_diagnostics() returned true (the
//                       binder state exists), enabled, envelope_disabled == 1
//                       (the "env_disabled=1" of the gate), patch_binds == 0
//                       (no in-place bind was applied), and lane_declines >= 1
//                       (the refusal went through the counted fallback path).
//   3. FUSED_QKV_REFUSED  the product's own log line
//                       "token lane refused: fused QKV projection" appears at
//                       least once, and the NKV_BINDER_STATS line the context
//                       destructor prints carries fused_qkv_refused >= 1.
//                       `fused_qkv_refused` is NOT exposed by any public
//                       struct (include/llama.h: llama_nkv_runtime_stats has
//                       no such field and llama_union_diagnostics stops at
//                       patch_binds/envelope_disabled), so the log line is the
//                       only source; this harness captures the log through
//                       llama_log_set, forwards every line to stderr unchanged
//                       (the gate greps the product's lines there) and parses
//                       the one counter it cannot read through the API. Every
//                       other field on the summary line comes from the public
//                       structs.
//   4. FUSED_QKV_STOCK  the stock arm took no binder path: if the diagnostics
//                       slot exists, enabled=0, patch_binds=0 and
//                       env_disabled=0 (the slot itself is allocated by a
//                       trace block, not by the gate); both arms decoded the
//                       same number of steps.
//   5. FUSED_QKV_REFUSAL  the conjunction, printed immediately before return:
//                       ok=1 fused_qkv_refused=<n> refusals=<n> patch_binds=0
//                       env_disabled=1 ... On any failure the same prefix is
//                       printed with ok=0 and the reason, and the process exits
//                       1 (0 on success, 2 on a usage error).
//
// The harness also rejects a captured "GGML_ASSERT" or "GGML_ABORT" line: an
// abort in the product would otherwise look like a clean nonzero exit.
//
// What this test does NOT prove:
//   - that the fixture's weights are the source model's: that is the fixture
//     generator's verify pass (scripts/make-fused-qkv-fixture.py);
//   - anything about GPU/HIP, flash attention, multi-sequence batches or
//     quantized V: token_candidate refuses them by construction;
//   - that the stock arm is upstream llama.cpp: it is the same binary with
//     GGML_TP5_UNION_BINDER=0, not an upstream build;
//   - that a fused model would refuse on the n_kv / n_outputs / rs_z lanes:
//     the refusal is specific to the token lane's strict shape pass.
//
// usage: test-fused-qkv-refusal <fused-model.gguf>
//   A missing or unreadable model is an error, never a skip.

#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

const uint32_t k_n_ctx  = 256; // a multiple of the stock 256-cell KV block
const int      k_steps  = 4;   // >= 4 decode steps per the G4 contract

// ---------------------------------------------------------------------------
// log capture
// ---------------------------------------------------------------------------
//
// The product prints its refusal and its counters through LLAMA_LOG_INFO. The
// gate greps those lines, so the callback must forward them to stderr exactly
// as the default logger would; it additionally keeps a copy so the harness can
// assert on the two facts no public struct exposes (the named refusal line and
// fused_qkv_refused).

std::string g_log;

void capture_log(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    if (text == nullptr) {
        return;
    }
    g_log += text;
    std::fputs(text, stderr);
    std::fflush(stderr);
}

size_t count_occurrences(const std::string & hay, const char * needle) {
    const size_t n = std::strlen(needle);
    size_t count = 0;
    for (size_t pos = hay.find(needle); pos != std::string::npos;
            pos = hay.find(needle, pos + n)) {
        ++count;
    }
    return count;
}

// The last `key=<decimal>` in the captured text. Returns false when the key is
// absent or is not followed by a decimal number.
bool find_counter(const std::string & hay, const char * key, uint64_t & out) {
    const std::string needle = std::string(key) + "=";
    const size_t pos = hay.rfind(needle);
    if (pos == std::string::npos) {
        return false;
    }
    size_t i = pos + needle.size();
    if (i >= hay.size() || hay[i] < '0' || hay[i] > '9') {
        return false;
    }
    uint64_t value = 0;
    while (i < hay.size() && hay[i] >= '0' && hay[i] <= '9') {
        value = value * 10 + (uint64_t) (hay[i] - '0');
        ++i;
    }
    out = value;
    return true;
}

// ---------------------------------------------------------------------------
// arms
// ---------------------------------------------------------------------------

enum arm_kind {
    ARM_REFUSAL,
    ARM_STOCK,
};

// Every variable here is read once by the llama_context constructor. Unsetting
// the ones the harness does not select keeps the arm independent of whatever
// the caller's environment carries: the gate command itself exports the
// refusal arm's two variables, and the stock arm must not inherit them.
void arm_env(arm_kind kind) {
    if (kind == ARM_REFUSAL) {
        setenv("GGML_TP5_UNION_BINDER", "1", 1);
        setenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK", "1", 1);
    } else {
        setenv("GGML_TP5_UNION_BINDER", "0", 1);
        unsetenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK");
    }
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    unsetenv("GGML_TP5_UNION_BINDER_EXACT");
    unsetenv("GGML_TP5_RSZ_RUNTIME_PARAM");
    unsetenv("LLAMA_NKV_MASK_OVERRIDE");
    unsetenv("LLAMA_GRAPH_UNION_OUTPUTS_RUNTIME");
}

// One context shape for BOTH arms. n_batch = n_ubatch = 1 so every decode is a
// one-token, one-sequence batch: that is what token_candidate requires
// (ubatch.n_seqs == ubatch.n_tokens, n_seqs_unq == 1, n_seq_tokens == 1) and
// it keeps the token lane's contract narrow. kv_unified is likewise required
// by token_candidate and is not the stock default.
llama_context * make_ctx(llama_model * model) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = k_n_ctx;
    cp.n_ctx_kv        = k_n_ctx;
    cp.n_batch         = 1;
    cp.n_ubatch        = 1;
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
    return llama_init_from_model(model, cp);
}

// One token id per position, in [1, vocab-1] so the embedding lookup is always
// a real row. The same function feeds both arms.
llama_token token_at(int pos, int vocab) {
    return (llama_token) (1 + (pos * 7 + 1) % (vocab - 1));
}

int decode_one(llama_context * ctx, llama_token token, llama_pos pos,
               std::vector<float> & row, int vocab) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    if (!llama_batch_is_complete(&batch, 1, false, true)) {
        std::fprintf(stderr, "FAIL: batch allocation\n");
        llama_batch_free(batch);
        batch = llama_batch{};
        return -1;
    }
    batch.n_tokens     = 1;
    batch.token[0]     = token;
    batch.pos[0]       = pos;
    batch.n_seq_id[0]  = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0]    = 1;

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: decode pos=%d rc=%d\n", (int) pos, rc);
        return rc;
    }

    const float * src = llama_get_logits_ith(ctx, 0);
    if (src == nullptr) {
        std::fprintf(stderr, "FAIL: no logits row pos=%d\n", (int) pos);
        return -1;
    }
    row.assign((size_t) vocab, 0.0f);
    std::memcpy(row.data(), src, sizeof(float) * (size_t) vocab);
    return 0;
}

struct arm_result {
    std::vector<std::vector<float>> rows;
    bool     init_ok   = false;
    bool     decode_ok = true;
    // public diagnostics, read before the context is freed
    bool     diagnostics       = false;
    bool     enabled           = false;
    bool     env_disabled      = false;
    bool     record_valid      = false;
    uint64_t patch_binds       = 0;
    uint64_t lane_declines     = 0;
    uint64_t rebuild_fallbacks = 0;
    bool     nkv_stats         = false; // llama_get_nkv_runtime_stats() returned true
    uint64_t graph_builds      = 0;
    // parsed from the captured log (no public accessor exists)
    uint64_t fused_qkv_refused = 0;
    size_t   refusal_lines     = 0;
    bool     has_fused_counter = false;
    bool     abort_line        = false;
    std::string log;
};

arm_result run_arm(llama_model * model, arm_kind kind,
                   const std::vector<llama_token> & tokens, int vocab) {
    arm_result r;

    arm_env(kind);
    g_log.clear();

    llama_context * ctx = make_ctx(model);
    if (ctx == nullptr) {
        r.log = g_log;
        return r;
    }
    r.init_ok = true;

    for (size_t i = 0; i < tokens.size(); ++i) {
        std::vector<float> row;
        if (decode_one(ctx, tokens[i], (llama_pos) i, row, vocab) != 0) {
            r.decode_ok = false;
            break;
        }
        r.rows.push_back(std::move(row));
    }

    // Read the public state BEFORE llama_free(): the diagnostics live in the
    // context's binder slot and the logits buffers die with the context.
    llama_union_diagnostics diag = {};
    r.diagnostics = llama_get_union_diagnostics(ctx, &diag);
    if (r.diagnostics) {
        r.enabled           = diag.enabled;
        r.env_disabled      = diag.envelope_disabled;
        r.record_valid      = diag.record_valid;
        r.patch_binds       = diag.patch_binds;
        r.lane_declines     = diag.lane_declines;
        r.rebuild_fallbacks = diag.rebuild_fallbacks;
    }
    llama_nkv_runtime_stats nkv = {};
    r.nkv_stats    = llama_get_nkv_runtime_stats(ctx, &nkv);
    r.graph_builds = llama_graph_build_count(ctx);

    // The destructor prints NKV_BINDER_STATS (with fused_qkv_refused) when the
    // binder state exists; that print lands in g_log through capture_log.
    llama_free(ctx);
    r.log = g_log;

    if (kind == ARM_REFUSAL) {
        r.refusal_lines     = count_occurrences(r.log, "token lane refused: fused QKV projection");
        r.has_fused_counter = find_counter(r.log, "fused_qkv_refused", r.fused_qkv_refused);
        r.abort_line        = r.log.find("GGML_ASSERT") != std::string::npos ||
                              r.log.find("GGML_ABORT")  != std::string::npos;
    }
    return r;
}

void fail(const char * reason) {
    std::fprintf(stderr, "FAIL: %s\n", reason);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s fused-model.gguf\n", argv[0]);
        return 2;
    }
    if (argv[1][0] == '\0') {
        std::fprintf(stderr, "FAIL: empty model path\n");
        return 2;
    }

    llama_log_set(capture_log, nullptr);
    llama_backend_init();

    auto mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = { nullptr };
    mp.devices      = devices;
    mp.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        fail("fused model load: the fixture was not accepted by the model loader");
        std::fprintf(stderr, "FUSED_QKV_REFUSAL ok=0 reason=model_load model=%s\n", argv[1]);
        llama_backend_free();
        return 1;
    }

    const int vocab = (int) llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (vocab <= 4) {
        std::fprintf(stderr, "FUSED_QKV_REFUSAL ok=0 reason=vocab vocab=%d\n", vocab);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::vector<llama_token> tokens;
    tokens.reserve((size_t) k_steps);
    for (int i = 0; i < k_steps; ++i) {
        tokens.push_back(token_at(i, vocab));
    }

    bool ok = true;

    // ---- refusal arm ------------------------------------------------------
    const arm_result r = run_arm(model, ARM_REFUSAL, tokens, vocab);

    if (!r.init_ok) {
        fail("refusal arm: context init");
        ok = false;
    }
    if (!r.decode_ok) {
        fail("refusal arm: decode");
        ok = false;
    }
    if (!r.diagnostics) {
        fail("refusal arm: llama_get_union_diagnostics() returned false"
             " (the binder state does not exist)");
        ok = false;
    } else {
        if (!r.enabled) {
            fail("refusal arm: diagnostics.enabled=0 (the binder gate was not armed)");
            ok = false;
        }
        if (!r.env_disabled) {
            fail("refusal arm: envelope_disabled=0 (the refusal did not disable the lane)");
            ok = false;
        }
        if (r.patch_binds != 0) {
            std::fprintf(stderr, "FAIL: refusal arm applied %" PRIu64
                                 " in-place bind(s); the contract requires 0\n", r.patch_binds);
            ok = false;
        }
        if (r.lane_declines == 0) {
            fail("refusal arm: lane_declines=0 (the refusal did not take the counted path)");
            ok = false;
        }
    }
    if (!r.has_fused_counter) {
        fail("refusal arm: no fused_qkv_refused= counter in the captured log");
        ok = false;
    } else if (r.fused_qkv_refused == 0) {
        fail("refusal arm: fused_qkv_refused=0");
        ok = false;
    }
    if (r.refusal_lines == 0) {
        fail("refusal arm: no \"token lane refused: fused QKV projection\" line");
        ok = false;
    }
    if (r.abort_line) {
        fail("refusal arm: GGML_ASSERT/GGML_ABORT in the log");
        ok = false;
    }

    std::fprintf(stderr,
            "FUSED_QKV_ARM arm=refusal enabled=%d env_disabled=%d patch_binds=%" PRIu64
            " lane_declines=%" PRIu64 " rebuild_fallbacks=%" PRIu64
            " record_valid=%d nkv_stats=%d graph_builds=%" PRIu64
            " fused_qkv_refused=%" PRIu64 " refusals=%zu\n",
            (int) r.enabled, (int) r.env_disabled, r.patch_binds, r.lane_declines,
            r.rebuild_fallbacks, (int) r.record_valid, (int) r.nkv_stats, r.graph_builds,
            r.fused_qkv_refused, r.refusal_lines);

    // ---- stock arm --------------------------------------------------------
    const arm_result s = run_arm(model, ARM_STOCK, tokens, vocab);

    if (!s.init_ok) {
        fail("stock arm: context init");
        ok = false;
    }
    if (!s.decode_ok) {
        fail("stock arm: decode");
        ok = false;
    }
    // The slot's presence is not the witness (see the header comment): the
    // product allocates it lazily from process_ubatch's bind-trace block even
    // with the gate off. The content is: the gate never set nkv_enabled, no
    // bind was applied, and the envelope was never disabled.
    if (s.diagnostics && (s.enabled || s.patch_binds != 0 || s.env_disabled)) {
        fail("stock arm took a binder path: enabled=1 / patch_binds>0 / env_disabled=1");
        ok = false;
    }
    if (s.rows.size() != r.rows.size()) {
        fail("arms decoded a different number of steps");
        ok = false;
    }

    std::fprintf(stderr,
            "FUSED_QKV_ARM arm=stock enabled=%d env_disabled=%d patch_binds=%" PRIu64
            " diagnostics=%d graph_builds=%" PRIu64 "\n",
            (int) s.enabled, (int) s.env_disabled, s.patch_binds,
            (int) s.diagnostics, s.graph_builds);

    // ---- bitwise comparison ----------------------------------------------
    int compared = 0;
    int bitwise  = 0;
    const size_t steps = std::min(r.rows.size(), s.rows.size());
    for (size_t i = 0; i < steps; ++i) {
        const bool same = r.rows[i].size() == s.rows[i].size() &&
                          !r.rows[i].empty() &&
                          std::memcmp(r.rows[i].data(), s.rows[i].data(),
                                      r.rows[i].size() * sizeof(float)) == 0;
        std::fprintf(stderr, "FUSED_QKV_STEP step=%zu bitwise=%d\n", i, same ? 1 : 0);
        if (!same) {
            fail("logits differ from the stock arm");
            ok = false;
        } else {
            ++bitwise;
        }
        ++compared;
    }
    if (compared != k_steps) {
        std::fprintf(stderr, "FAIL: compared %d of %d steps\n", compared, k_steps);
        ok = false;
    }

    llama_model_free(model);
    llama_backend_free();

    const bool end_ok = ok && compared == k_steps && bitwise == compared &&
                        r.fused_qkv_refused >= 1 && r.refusal_lines >= 1 &&
                        r.patch_binds == 0 && r.env_disabled;

    std::fprintf(stderr,
            "FUSED_QKV_REFUSAL ok=%d fused_qkv_refused=%" PRIu64 " refusals=%zu"
            " patch_binds=%" PRIu64 " env_disabled=%d lane_declines=%" PRIu64
            " steps=%d bitwise=%d/%d graph_builds_refusal=%" PRIu64
            " graph_builds_stock=%" PRIu64 "\n",
            (int) end_ok, r.fused_qkv_refused, r.refusal_lines, r.patch_binds,
            (int) r.env_disabled, r.lane_declines, k_steps, bitwise, compared,
            r.graph_builds, s.graph_builds);

    return end_ok ? 0 : 1;
}
