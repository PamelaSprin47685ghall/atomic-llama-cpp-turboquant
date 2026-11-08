// TP5 WS4 pending-context and model ownership test (CPU only).
//
// e3c9ba1e8 retained a pending llama_context by releasing individual members;
// memory/kv_stream_phase_arena/cvec/loras/balloc/sampling were still freed by
// the destructor and the model was caller-owned without a retained reference.
// The fix retains the whole context (llama_free returns without deleting it)
// and increments a model retained-context count that defers llama_model_free.
//
// This test observes:
//   * llama_model_free defers while a retained-context reference exists,
//   * a clean model still frees normally,
//   * with LLAMA_TEST_MODEL (or argv[1]), a real CPU context marked unproven is
//     retained by llama_free and the model free is deferred; the context is
//     never destructed, so memory/arena/balloc/sampling cannot be freed behind
//     an in-flight copy,
//   * without a model the context part is skipped; the model part still runs.
//
// Build (ctest registers it; Release or ASAN):
//   cmake --build build-ws4-sched-ctest -j --target test-tp5-context-retain
//   LLAMA_TEST_MODEL=/path/to/model.gguf ctest -R test-tp5-context-retain

#include "llama.h"
#include "llama-ext.h"

#include "../src/llama-context.h"
#include "../src/llama-model.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            ++g_failures;                                                                \
        }                                                                                \
    } while (0)

llama_model * make_mock_model() {
    llama_quant_model_desc desc = {};
    desc.architecture  = "llama";
    desc.n_embd        = 64;
    desc.n_ff          = 128;
    desc.n_layer       = 2;
    desc.n_head        = 4;
    desc.n_head_kv     = 4;
    desc.n_expert      = 0;
    desc.n_embd_head_k = 16;
    desc.n_embd_head_v = 16;
    return llama_quant_model_from_metadata(&desc);
}

// A model retained by a pending context must not be deleted by
// llama_model_free. The retained model object stays readable.
void test_model_retention() {
    llama_model * retained = make_mock_model();
    CHECK(retained != nullptr);
    if (retained == nullptr) {
        return;
    }
    CHECK(retained->retained_contexts() == 0);
    retained->retain_context();
    CHECK(retained->retained_contexts() == 1);

    llama_model_free(retained);
    CHECK(retained->retained_contexts() == 1); // deferred, not deleted
    CHECK(!retained->name.empty());            // object still readable

    // A clean model frees normally. Do not read it after the free.
    llama_model * clean = make_mock_model();
    CHECK(clean != nullptr);
    if (clean != nullptr) {
        CHECK(clean->retained_contexts() == 0);
        llama_model_free(clean);
    }
}

// With a real model, create a CPU context, force the scheduler's unproven
// state, and free the context. llama_free must retain the whole context and
// the model free must defer. The context destructor is never reached (a direct
// delete aborts by design).
void test_pending_context_retention(const char * model_path) {
    if (model_path == nullptr || model_path[0] == '\0') {
        std::printf("TP5_CONTEXT_RETAIN_TEST context-skip (no LLAMA_TEST_MODEL)\n");
        return;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        std::printf("TP5_CONTEXT_RETAIN_TEST context-skip (model load failed)\n");
        return;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx            = 64;
    cparams.n_batch          = 4;
    cparams.n_threads        = 1;
    cparams.n_threads_batch  = 1;

    llama_context * ctx = nullptr;
    try {
        ctx = llama_init_from_model(model, cparams);
    } catch (...) {
        ctx = nullptr;
    }
    if (ctx == nullptr) {
        std::printf("TP5_CONTEXT_RETAIN_TEST context-skip (context creation failed)\n");
        llama_model_free(model);
        return;
    }

    CHECK(model->retained_contexts() == 0);
    ctx->debug_mark_unproven();
    CHECK(ctx->has_unproven());

    llama_free(ctx); // must retain the whole context, not delete it
    CHECK(model->retained_contexts() == 1);

    llama_model_free(model); // must defer while the context is retained
    CHECK(model->retained_contexts() == 1);
    CHECK(!model->name.empty()); // model object still alive
}

// A clean context takes the normal destruction path: the scheduler is freed
// before the backend members, and the model can then be freed.
void test_clean_context_retention(const char * model_path) {
    if (model_path == nullptr || model_path[0] == '\0') {
        return;
    }
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        return;
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 64;
    cparams.n_batch         = 4;
    cparams.n_threads       = 1;
    cparams.n_threads_batch = 1;
    llama_context * ctx = nullptr;
    try {
        ctx = llama_init_from_model(model, cparams);
    } catch (...) {
        ctx = nullptr;
    }
    if (ctx == nullptr) {
        llama_model_free(model);
        return;
    }

    CHECK(model->retained_contexts() == 0);
    llama_free(ctx); // clean: deletes the context
    llama_model_free(model); // clean: deletes the model
}

} // namespace

int main(int argc, char ** argv) {
    llama_backend_init();

    const char * model_path = argc > 1 ? argv[1] : std::getenv("LLAMA_TEST_MODEL");

    test_model_retention();
    test_pending_context_retention(model_path);
    test_clean_context_retention(model_path);

    llama_backend_free();

    if (g_failures != 0) {
        std::fprintf(stderr, "TP5_CONTEXT_RETAIN_TEST FAIL failures=%d\n", g_failures);
        return 1;
    }
    std::printf("TP5_CONTEXT_RETAIN_TEST PASS\n");
    return 0;
}
