// Explicit MoE cache policy at the real llama_init_from_model entry.
//
// An explicit ON is a request: when no cacheable expert provider/tensor is
// available, the resolved mode is OFF and the context still succeeds. The
// requested and resolved modes are logged, not hidden. AUTO and the unspecified
// default may also resolve to OFF. This test needs a model (LLAMA_TEST_MODEL or
// argv[1]) and skips otherwise; it registers no provider, so the cache is
// ineligible, and does not need a GPU.
//
// Context-level ON provider-dormant (an eligible model with a provider whose
// session_create returns null) is not covered: the CPU backend registers no
// real MoE provider, and injecting one would require a fake interface. The
// setter-level DORMANT path is covered by test-moe-cache-session.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int g_checks   = 0;
int g_failures = 0;
std::string g_log;

void log_cb(ggml_log_level, const char * text, void *) {
    if (text != nullptr) {
        g_log += text;
    }
}

void check(bool condition, const char * message) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
}

} // namespace

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : std::getenv("LLAMA_TEST_MODEL");
    if (model_path == nullptr || model_path[0] == '\0') {
        std::printf("MOE_CONTEXT_POLICY_TEST SKIP no LLAMA_TEST_MODEL\n");
        return 77;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "MOE_CONTEXT_POLICY_TEST SKIP model load failed: %s\n", model_path);
        return 77;
    }
    llama_log_set(log_cb, nullptr);

    // The legal default is unspecified; an ineligible model resolves it to OFF
    // and still creates a context.
    {
        g_log.clear();
        llama_context_params cparams = llama_context_default_params();
        check(cparams.moe_cache_mode == LLAMA_MOE_CACHE_MODE_UNSPECIFIED,
              "default context params keep the mode unspecified");
        llama_context * ctx = llama_init_from_model(model, cparams);
        check(ctx != nullptr, "unspecified default creates a context on an ineligible model");
        llama_free(ctx);
    }

    // Explicit AUTO may resolve to OFF.
    {
        g_log.clear();
        llama_context_params cparams = llama_context_default_params();
        cparams.moe_cache_mode = LLAMA_MOE_CACHE_MODE_AUTO;
        llama_context * ctx = llama_init_from_model(model, cparams);
        check(ctx != nullptr, "auto creates a context on an ineligible model");
        llama_free(ctx);
    }

    // Explicit ON is a request. On an ineligible model it resolves to OFF and
    // the context still succeeds; the log records both modes.
    {
        g_log.clear();
        llama_context_params cparams = llama_context_default_params();
        cparams.moe_cache_mode = LLAMA_MOE_CACHE_MODE_ON;
        llama_context * ctx = llama_init_from_model(model, cparams);
        check(ctx != nullptr, "explicit on resolves to off and creates a context on an ineligible model");
        check(g_log.find("MoE cache requested=on resolved=off") != std::string::npos,
              "explicit on logs the requested and resolved modes");
        llama_free(ctx);
    }

    llama_log_set(nullptr, nullptr);
    llama_model_free(model);

    if (g_failures != 0) {
        std::fprintf(stderr, "MOE_CONTEXT_POLICY_TEST FAIL checks=%d failures=%d\n", g_checks, g_failures);
        return 1;
    }
    std::printf("MOE_CONTEXT_POLICY_TEST PASS checks=%d\n", g_checks);
    return 0;
}
