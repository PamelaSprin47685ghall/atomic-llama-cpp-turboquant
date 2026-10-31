// m3-probe-hc-mix.cpp — test-only trampoline that reaches the REAL
// llama_model_qwen4exp::graph::build_hc_mix.
//
// The production builder is protected inside llama_model_qwen4exp::graph and is
// constructed from (llm_graph_result, llm_graph_params).  This TU derives from
// the real graph type to expose that one member and nothing else, so the device
// test exercises exactly the code the model runs: the Q8_0 door, the
// inject/mixer split, the production exact-"1" gate and the
// GGML_TENSOR_FLAG_NO_REUSE decision.  No part of the builder is re-implemented.
//
// The trampoline does not create its own ggml context or cgraph: the device test
// owns both.  Testing-only hooks declared in src/llama-graph.h under
// M3_PROBE_GRAPH_HOOK (micro/build_mixer_prod.py prints the exact blocks) let the
// result point at that context while leaving every production build untouched.
// After the build, the site tensors are found by the debug names the production
// builder sets ("hc_lo", "hc_up"), so nothing depends on node positions.
//
// Authored, NOT built or run.

#include "models.h"
#include "llama-graph.h"
#include "llama-model.h"

#include <cstring>

// ---- state installed by the device test -----------------------------------
static struct {
    llama_model * model = nullptr;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
} g_probe;

namespace {

// Non-owning result: its context is the test's, so the graph object writes into
// the caller's memory and the caller can run it on its own backend.
struct probe_result : llm_graph_result {
    explicit probe_result(int64_t max_nodes) : llm_graph_result(max_nodes) {}
    // reset_context_view() exists only when M3_PROBE_GRAPH_HOOK is defined, which
    // micro/build_mixer_prod.py puts on the command line; the view does not own
    // the test's context.
    void reset_nonowning(ggml_context * c, ggml_cgraph * g) { this->reset_context_view(c, g); }
};

struct probe_graph : llama_model_qwen4exp::graph {
    probe_graph(const llama_model & model, const llm_graph_params & params) :
        llama_model_qwen4exp::graph(model, params) {}

    ggml_tensor * hc_mix(ggml_tensor * x, ggml_tensor * w_norm, ggml_tensor * w_down,
                         ggml_tensor * w_up, ggml_tensor * w_inject,
                         ggml_tensor ** inject, int il) {
        return build_hc_mix(x, w_norm, w_down, w_up, w_inject, inject, il);
    }
};

} // namespace

extern "C" {

bool m3_probe_attach(llama_model * model, ggml_context * ctx, ggml_cgraph * gf) {
    if (model == nullptr || ctx == nullptr || gf == nullptr) {
        return false;
    }
    g_probe.model = model;
    g_probe.ctx   = ctx;
    g_probe.gf    = gf;
    return true;
}

// Build one HC site with the production builder and report the tensors it made.
// The Q8_0/F32 tensors are created and seeded by the device test, so the
// production Q8_0 door is the thing under test.  `inject_out` receives nullptr
// when the caller selected the mixer site.
ggml_tensor * m3_probe_build_hc_mix(ggml_tensor * xn, ggml_tensor * w_norm,
                                    ggml_tensor * w_down, ggml_tensor * w_up,
                                    ggml_tensor * w_inject, int64_t nt, int il,
                                    ggml_tensor ** lo_out, ggml_tensor ** up_out,
                                    ggml_tensor ** inject_out) {
    if (g_probe.model == nullptr || g_probe.ctx == nullptr || g_probe.gf == nullptr) {
        return nullptr;
    }
    llm_graph_params params;
    params.hparams     = g_probe.model->hparams;
    params.n_outputs   = uint32_t(nt);
    params.sched       = nullptr;
    params.backend_cpu = nullptr;
    params.mctx        = nullptr;

    probe_result res(4096);
    res.reset_nonowning(g_probe.ctx, g_probe.gf);
    params.res = &res;

    probe_graph g(*g_probe.model, params);
    ggml_tensor * inject = w_inject;
    ggml_tensor * mixed = g.hc_mix(xn, w_norm, w_down, w_up,
                                   w_inject != nullptr ? &inject : nullptr, int(il));
    if (mixed == nullptr) {
        return nullptr;
    }
    if (lo_out != nullptr) {
        *lo_out = ggml_graph_get_tensor(g_probe.gf, "hc_lo");
    }
    if (up_out != nullptr) {
        *up_out = ggml_graph_get_tensor(g_probe.gf, "hc_up");
    }
    if (inject_out != nullptr) {
        *inject_out = w_inject != nullptr ? inject : nullptr;
    }
    return mixed;
}

// True when the production gate armed retention for this site.  Read straight off
// the tensor the builder mutated; the harness itself never sets the flag.
bool m3_probe_lo_retained(ggml_tensor * lo) {
    return lo != nullptr && (lo->flags & 32) != 0;
}

} // extern "C"
