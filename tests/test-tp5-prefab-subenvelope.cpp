// TP5 factor-prefab sub-envelope KQ-mask regression (CPU only).
//
// An envelope graph has n_env rows. A ubatch with n_tokens < n_env runs on it
// with padded rows: only rows [0, n_tokens) are real. The KQ mask therefore
// has n_env rows, but the fill must write exactly the real rows at the
// tensor's row stride, and the prefab wrapper must replicate the last real row
// into the pad rows. The stock fill's n_tps == n_tokens equality is a
// stock-layout contract, not a property of the envelope mask.
//
// This test compares user-visible logits of one token sequence under:
//   stock  : prefab off,
//   exact  : envelopes 1 and 3, so every ubatch fits its envelope,
//   padded : envelope 4 only, so a 3-token and two 1-token ubatches pad.
// Every step must return the same logits. The padded arm must also report
// n_sub_envelope/n_pad_rows with n_ubatch_stock == 0, so a run that silently
// fell back to the stock builder cannot pass.
//
// A GGUF whose raw F32 tensor bytes are not finite (the /tmp/min-llama.gguf
// generator writes arbitrary bytes) produces non-finite logits. The test then
// reports `numeric-skip` and falls back to raw-bit agreement with the stock
// arm; use a finite-weight GGUF (e.g. /tmp/min-llama-finite.gguf) for the
// numeric arm.
//
// Build and run against a real tiny CPU GGUF:
//   cmake --build build-ws4-sched-ctest -j --target test-tp5-prefab-subenvelope
//   LLAMA_TEST_MODEL=/tmp/min-llama.gguf ./build-ws4-sched-ctest/bin/test-tp5-prefab-subenvelope
//
// Without a model the test prints a skip and exits 0; the prefab arms are not
// proven then.

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            ++g_failures;                                                                \
        }                                                                                \
    } while (0)

void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value != nullptr) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

// The factor lane is selected by exact-string gates. Clear every lane-selection
// flag first so a developer's exported prefab environment cannot change what
// this test exercises. The union binder is turned off too: this test observes
// the factor-prefab mask fill, and the binder's own token/n_kv lanes must not
// decide the baseline or intercept the prefab arm.
void clear_prefab_env() {
    set_env("LLAMA_PREFAB_ENABLE", nullptr);
    set_env("LLAMA_PREFAB_EXPERIMENTAL", nullptr);
    set_env("LLAMA_PREFAB_ENVELOPES", nullptr);
    set_env("LLAMA_PREFAB_MTP", nullptr);
    set_env("GGML_TP5_PREFAB_IN_GRAPH", nullptr);
    set_env("GGML_TP5_PREFAB_AB", nullptr);
    set_env("GGML_TP5_PREFAB_DECOUPLE", nullptr);
    set_env("GGML_TP5_PREFAB_NOCLONE", nullptr);
    set_env("GGML_TP5_PREFAB_STATS", "1");
    set_env("GGML_TP5_UNION_BINDER", "0");
    set_env("LLAMA_EXPERIMENTAL_VARLEN_NTOK", nullptr);
}

struct arm {
    llama_context * ctx = nullptr;
    std::vector<std::vector<float>> rows; // rows[step][token * n_vocab + vocab]
    llama_prefab_stats stats = {};
};

struct step {
    std::vector<llama_token> tokens;
    llama_pos pos0;
};

std::vector<float> copy_step_logits(llama_context * ctx, int n_tokens, int n_vocab) {
    std::vector<float> out((size_t) n_tokens * (size_t) n_vocab, 0.0f);
    for (int i = 0; i < n_tokens; ++i) {
        const float * row = llama_get_logits_ith(ctx, i);
        if (row == nullptr) {
            return {};
        }
        std::memcpy(out.data() + (size_t) i * (size_t) n_vocab, row, (size_t) n_vocab * sizeof(float));
    }
    return out;
}

bool decode_step(llama_context * ctx, const step & s, int n_vocab, std::vector<float> & out) {
    llama_batch batch = llama_batch_init((int32_t) s.tokens.size(), 0, 1);
    if (!llama_batch_is_complete(&batch, (int32_t) s.tokens.size(), false, true)) {
        llama_batch_free(batch);
        GGML_ABORT("llama_batch_init failed");
    }
    batch.n_tokens = (int32_t) s.tokens.size();
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i]      = s.tokens[i];
        batch.pos[i]        = s.pos0 + i;
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = 0;
        batch.logits[i]     = 1;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        std::fprintf(stderr, "FAIL llama_decode step pos=%d n=%d rc=%d\n", s.pos0, (int) s.tokens.size(), rc);
        ++g_failures;
        return false;
    }
    out = copy_step_logits(ctx, (int) s.tokens.size(), n_vocab);
    if (out.empty()) {
        std::fprintf(stderr, "FAIL null logits step pos=%d\n", s.pos0);
        ++g_failures;
        return false;
    }
    return true;
}

bool run_arm(llama_model * model, const char * envelopes, bool kv_unified, const std::vector<step> & steps,
             int n_vocab, arm & a) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx            = 64;
    cparams.n_batch          = 4;
    cparams.n_ubatch         = 4;
    cparams.n_threads        = 1;
    cparams.n_threads_batch  = 1;
    cparams.kv_unified       = kv_unified;

    if (envelopes != nullptr) {
        set_env("LLAMA_PREFAB_ENABLE", "1");
        set_env("LLAMA_PREFAB_EXPERIMENTAL", "1");
        set_env("LLAMA_PREFAB_ENVELOPES", envelopes);
    }

    try {
        a.ctx = llama_init_from_model(model, cparams);
    } catch (...) {
        a.ctx = nullptr;
    }
    if (a.ctx == nullptr) {
        std::fprintf(stderr, "FAIL context init failed (envelopes=%s)\n", envelopes ? envelopes : "off");
        ++g_failures;
        return false;
    }
    llama_prefab_reset_stats(a.ctx);

    for (const step & s : steps) {
        std::vector<float> row;
        if (!decode_step(a.ctx, s, n_vocab, row)) {
            return false;
        }
        a.rows.push_back(std::move(row));
    }

    llama_prefab_get_stats(a.ctx, &a.stats);
    return true;
}

float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return INFINITY;
    }
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return INFINITY;
        }
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

bool raw_equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() &&
        (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

bool finite_rows(const std::vector<std::vector<float>> & rows) {
    for (const auto & row : rows) {
        for (float v : row) {
            if (!std::isfinite(v)) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : std::getenv("LLAMA_TEST_MODEL");

    clear_prefab_env();
    llama_backend_init();

    if (model_path == nullptr || model_path[0] == '\0') {
        std::printf("PREFAB_SUBENVELOPE_TEST skip (no LLAMA_TEST_MODEL)\n");
        llama_backend_free();
        std::printf("PREFAB_SUBENVELOPE_TEST PASS\n");
        return 0;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "PREFAB_SUBENVELOPE_TEST FAIL (model load failed: %s)\n", model_path);
        llama_backend_free();
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    CHECK(n_vocab > 0);

    // 3-token prompt (one pad row in envelope 4), then two 1-token steps
    // (three pad rows each).
    const std::vector<step> steps = {
        { { 1, 2, 3 }, 0 },
        { { 4 },       3 },
        { { 5 },       4 },
    };

    arm stock;
    arm exact;
    arm padded;

    set_env("LLAMA_PREFAB_ENABLE", nullptr);
    set_env("LLAMA_PREFAB_EXPERIMENTAL", nullptr);
    set_env("LLAMA_PREFAB_ENVELOPES", nullptr);
    CHECK(run_arm(model, nullptr, true, steps, n_vocab, stock));
    CHECK(run_arm(model, "1,3", true, steps, n_vocab, exact));
    CHECK(run_arm(model, "4", true, steps, n_vocab, padded));

    // The retention fixture runs the default non-unified cache (n_seq_max 1, so
    // one stream). Stock itself cannot fill a multi-token non-unified mask
    // (n_tps = n_seq_tokens = 1), so this arm pair stays inside the prefab lane
    // and compares padded against exact under the fixture's own setting.
    arm exact_nu;
    arm padded_nu;
    CHECK(run_arm(model, "1,3", false, steps, n_vocab, exact_nu));
    CHECK(run_arm(model, "4", false, steps, n_vocab, padded_nu));

    CHECK(stock.rows.size() == steps.size());
    CHECK(exact.rows.size() == steps.size());
    CHECK(padded.rows.size() == steps.size());
    CHECK(exact_nu.rows.size() == steps.size());
    CHECK(padded_nu.rows.size() == steps.size());

    // A GGUF with raw random F32 tensors can contain NaN/Inf weights (the
    // /tmp/gen-min-gguf.py fixture does). Its logits are then non-finite and
    // cannot be compared numerically. Use bitwise agreement with the stock
    // arm as the fallback state check, and say so. On a finite-weight model
    // the numeric comparison below is the real user-visible check.
    const bool stock_finite = finite_rows(stock.rows);
    if (!stock_finite) {
        std::printf("PREFAB_SUBENVELOPE_TEST numeric-skip (stock logits are non-finite: "
                    "fixture weights are not finite; comparing raw bits instead)\n");
    }

    // The exact arm must stay exact, and the padded arm must really pad.
    CHECK(exact.stats.n_ubatch_prefab == steps.size());
    CHECK(exact.stats.n_ubatch_stock == 0);
    CHECK(exact.stats.n_sub_envelope == 0);
    CHECK(exact.stats.n_pad_rows == 0);
    CHECK(exact.stats.last_n_env == 1);
    CHECK(exact.stats.last_n_tokens == 1);

    CHECK(padded.stats.n_ubatch_prefab == steps.size());
    CHECK(padded.stats.n_ubatch_stock == 0);
    CHECK(padded.stats.n_sub_envelope == steps.size());
    CHECK(padded.stats.n_pad_rows == 1 + 3 + 3);
    CHECK(padded.stats.last_n_env == 4);
    CHECK(padded.stats.last_n_tokens == 1);

    CHECK(exact_nu.stats.n_ubatch_prefab == steps.size());
    CHECK(exact_nu.stats.n_ubatch_stock == 0);
    CHECK(exact_nu.stats.n_sub_envelope == 0);
    CHECK(exact_nu.stats.n_pad_rows == 0);

    CHECK(padded_nu.stats.n_ubatch_prefab == steps.size());
    CHECK(padded_nu.stats.n_ubatch_stock == 0);
    CHECK(padded_nu.stats.n_sub_envelope == steps.size());
    CHECK(padded_nu.stats.n_pad_rows == 1 + 3 + 3);
    CHECK(padded_nu.stats.last_n_env == 4);
    CHECK(padded_nu.stats.last_n_tokens == 1);

    // The padded arm must match both the exact prefab arm and the stock
    // baseline. The tolerance is the numerical slack of a wider batch; it is
    // tight enough that a wrong real-row mask or an unmasked scratch cell
    // changes the observed logits by orders of magnitude.
    constexpr float kTolerance = 1e-5f;
    for (size_t i = 0; i < steps.size(); ++i) {
        const float d_pe = max_abs_diff(padded.rows[i], exact.rows[i]);
        const float d_ps = max_abs_diff(padded.rows[i], stock.rows[i]);
        const float d_es = max_abs_diff(exact.rows[i], stock.rows[i]);
        const float d_nu = max_abs_diff(padded_nu.rows[i], exact_nu.rows[i]);
        std::printf("PREFAB_SUBENVELOPE_TEST step=%zu n=%zu pad_exact=%g pad_stock=%g exact_stock=%g "
                    "nu_pad_exact=%g raw(pe=%d ps=%d es=%d nu=%d)\n",
                i, steps[i].tokens.size(), (double) d_pe, (double) d_ps, (double) d_es, (double) d_nu,
                (int) raw_equal(padded.rows[i], exact.rows[i]),
                (int) raw_equal(padded.rows[i], stock.rows[i]),
                (int) raw_equal(exact.rows[i], stock.rows[i]),
                (int) raw_equal(padded_nu.rows[i], exact_nu.rows[i]));
        if (stock_finite) {
            // max_abs_diff returns INFINITY for any non-finite value, so a
            // prefab arm that introduced NaN/Inf fails here even on a finite
            // fixture.
            CHECK(d_pe <= kTolerance);
            CHECK(d_ps <= kTolerance);
            CHECK(d_es <= kTolerance);
            CHECK(d_nu <= kTolerance);
        } else {
            CHECK(raw_equal(padded.rows[i], exact.rows[i]));
            CHECK(raw_equal(padded.rows[i], stock.rows[i]));
            CHECK(raw_equal(exact.rows[i], stock.rows[i]));
            CHECK(raw_equal(padded_nu.rows[i], exact_nu.rows[i]));
        }
    }

    const bool ok = g_failures == 0;
    std::printf("PREFAB_SUBENVELOPE_TEST stats exact(pad=%" PRIu64 ",sub=%" PRIu64 ") "
                "padded(pad=%" PRIu64 ",sub=%" PRIu64 ",stock=%" PRIu64 ",env=%" PRIu64 ",n=%" PRIu64 ") "
                "nu_padded(pad=%" PRIu64 ",sub=%" PRIu64 ",stock=%" PRIu64 ")\n",
            exact.stats.n_pad_rows, exact.stats.n_sub_envelope,
            padded.stats.n_pad_rows, padded.stats.n_sub_envelope, padded.stats.n_ubatch_stock,
            padded.stats.last_n_env, padded.stats.last_n_tokens,
            padded_nu.stats.n_pad_rows, padded_nu.stats.n_sub_envelope, padded_nu.stats.n_ubatch_stock);

    llama_free(stock.ctx);
    llama_free(exact.ctx);
    llama_free(padded.ctx);
    llama_free(exact_nu.ctx);
    llama_free(padded_nu.ctx);
    llama_model_free(model);
    llama_backend_free();

    if (!ok) {
        std::fprintf(stderr, "PREFAB_SUBENVELOPE_TEST FAIL failures=%d\n", g_failures);
        return 1;
    }
    std::printf("PREFAB_SUBENVELOPE_TEST PASS\n");
    return 0;
}
