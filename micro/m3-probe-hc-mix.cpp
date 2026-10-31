// m3-probe-hc-mix.cpp — test-only trampoline that reaches the REAL
// llama_model_qwen4exp::graph::build_hc_mix.
//
// Design constraints this file obeys (2026-10-31 scheduler review):
//   * no macro may change the layout of a shipped class: src/llama-graph.h is
//     byte-identical to the production header, and this TU compiles with exactly
//     the same macros as the rest of the library, so there is no ODR/ABI split;
//   * the production builder reads its context and graph through
//     `res->get_ctx()` / `res->get_gf()` inside the library TU, so a test must
//     hand the library a real `llm_graph_result` whose `ctx_compute` owns the
//     test's ggml context -- not a shadow member this TU could read but the
//     library could not;
//   * therefore the probe gives ownership of the test context to
//     llm_graph_result::ctx_compute, whose deleter is ggml_context_deleter
//     (`ggml_free`), exactly like a normal build.  The test allocates the context
//     with ggml_init(), builds the graph inside it, and hands the raw pointer over
//     exactly once; from that moment the result is the only owner and the test
//     must not call ggml_free() itself -- one owner, one free, no double free.
//   * the result outlives the build for as long as the caller needs its tensors:
//     m3_probe_release() destroys it (and with it the context) when the caller has
//     finished reading back.  Weights, because they live in the same context, must
//     be re-created for the next site.
//
// There is no new production API: only llama_model_qwen4exp::graph (protected
// member) is subclassed, and llm_graph_context's constructor is public.
//
// Authored, NOT built or run.

#include "models.h"
#include "llama-graph.h"
#include "llama-model.h"
#include "ggml-cpp.h"
#include "ggml-backend-impl.h"   // struct ggml_backend_buffer: the `context` tag used below

#include <memory>

// ---- state installed by the device test -----------------------------------
static struct {
    llama_model * model = nullptr;
    ggml_context * ctx = nullptr;   // owned by g_probe.res after a build
    ggml_cgraph * gf = nullptr;
    llm_graph_result * res = nullptr; // sole owner of ctx once set
} g_probe;

namespace {

// Signature copied from src/models/qwen4exp.cpp: a drift is a compile error, and
// no argument is ever faked (w_inject is a real Q8_0 weight; only the inject
// output slot is null, which is how the production builder selects a mixer site).
struct probe_graph : llama_model_qwen4exp::graph {
    probe_graph(const llama_model & model, const llm_graph_params & params) :
        llama_model_qwen4exp::graph(model, params) {}

    ggml_tensor * hc_mix(
            ggml_tensor *  x,
            ggml_tensor *  w_norm,
            ggml_tensor *  w_down,
            ggml_tensor *  w_up,
            ggml_tensor *  w_inject,
            ggml_tensor ** inject,
            int            il) {
        return build_hc_mix(x, w_norm, w_down, w_up, w_inject, inject, il);
    }
};

} // namespace

extern "C" {

// Install the model, the test-created context and the (single) graph the builder
// should populate.  Refused when an earlier call still owns a context.
bool m3_probe_attach(llama_model * model, ggml_context * ctx, ggml_cgraph * gf) {
    if (model == nullptr || ctx == nullptr || gf == nullptr) {
        return false;
    }
    if (g_probe.ctx != nullptr || g_probe.res != nullptr) {
        return false;   // outstanding owner from a previous call: refuse instead of double-free
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
//
// Lifetime contract: this call CONSUMES the context and graph installed by
// m3_probe_attach; the returned tensors stay valid until the caller calls
// m3_probe_release().  A second build without a release is refused, and so is a
// re-attach while a result is outstanding (see m3_probe_attach).
ggml_tensor * m3_probe_build_hc_mix(ggml_tensor * xn, ggml_tensor * w_norm,
                                    ggml_tensor * w_down, ggml_tensor * w_up,
                                    ggml_tensor * w_inject, int64_t nt, int il,
                                    ggml_tensor ** lo_out, ggml_tensor ** up_out,
                                    ggml_tensor ** inject_out) {
    if (g_probe.model == nullptr || g_probe.ctx == nullptr || g_probe.gf == nullptr) {
        return nullptr;
    }
    // The weights, the input and the graph the builder will fill must belong to the
    // context handed over below; otherwise the library would write into a context the
    // test keeps and frees separately.  The context a tensor lives in is not exposed
    // by the public API, so the backend buffer that backs it is checked instead: the
    // harness allocates weights and input with ggml_backend_alloc_ctx_tensors(ctx),
    // which tags the buffer with its ggml context.  A mismatch is refused rather than
    // leaked or double-freed, and the harness uses one context per site.
    const auto in_ctx = [](const ggml_tensor * t, ggml_context * owner) {
        return t == nullptr || (t->buffer != nullptr && t->buffer->context == owner);
    };
    if (!in_ctx(xn, g_probe.ctx) || !in_ctx(w_norm, g_probe.ctx) || !in_ctx(w_down, g_probe.ctx) ||
            !in_ctx(w_up, g_probe.ctx) || !in_ctx(w_inject, g_probe.ctx)) {
        return nullptr;
    }
    // Ownership: the context is handed to this result exactly once, through the
    // same unique_ptr type the shipped result uses, so its deleter (ggml_free) is
    // the only free and the library reads res->get_ctx() / res->get_gf() from the
    // very context the test created.  There is no second owner and no setter.
    std::unique_ptr<llm_graph_result> res(new llm_graph_result(/*max_nodes=*/4096,
            std::unique_ptr<ggml_context, ggml_context_deleter>(g_probe.ctx)));
    g_probe.ctx = nullptr;
    g_probe.gf  = nullptr;

    llm_graph_params params;
    params.hparams     = g_probe.model->hparams;
    params.n_outputs   = uint32_t(nt);
    params.sched       = nullptr;
    params.backend_cpu = nullptr;
    params.mctx        = nullptr;
    params.res         = res.get();

    probe_graph g(*g_probe.model, params);
    ggml_tensor * inject_slot = w_inject;
    ggml_tensor ** inject_arg = w_inject != nullptr ? &inject_slot : nullptr;
    ggml_tensor * mixed = g.hc_mix(xn, w_norm, w_down, w_up, w_inject, inject_arg, int(il));
    if (mixed == nullptr) {
        return nullptr;
    }
    // The builder's tensors live in res->get_ctx(); read the graph the library used.
    if (lo_out != nullptr) {
        *lo_out = ggml_graph_get_tensor(res->get_gf(), "hc_lo");
    }
    if (up_out != nullptr) {
        *up_out = ggml_graph_get_tensor(res->get_gf(), "hc_up");
    }
    if (inject_out != nullptr) {
        *inject_out = inject_arg != nullptr ? inject_slot : nullptr;
    }
    // Keep the result alive until the caller has read the tensors back.
    g_probe.res = res.release();
    return mixed;
}

// Destroy the result handed out by the previous build; its deleter frees the
// context and everything allocated in it, exactly once.
void m3_probe_release() {
    delete g_probe.res;
    g_probe.res = nullptr;
    g_probe.gf  = nullptr;
}

// True when the production gate armed retention for this site.  Read straight off
// the tensor the builder mutated; the harness itself never sets the flag.
bool m3_probe_lo_retained(ggml_tensor * lo) {
    return lo != nullptr && (lo->flags & 32) != 0;
}

} // extern "C"
