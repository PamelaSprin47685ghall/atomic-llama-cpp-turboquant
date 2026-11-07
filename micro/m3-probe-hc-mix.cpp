// m3-probe-hc-mix.cpp — test-only trampoline that reaches the REAL
// llama_model_qwen4exp::graph::build_hc_mix.
//
// Design constraints this file obeys (2026-10-31 reviews, rounds 4-5):
//   * src/llama-graph.h stays byte-identical to production: no macro, no member,
//     no inline change, no new API;
//   * the shipped llm_graph_result has exactly two public entry points that matter
//     here: `llm_graph_result(int64_t max_nodes)` — which calls reset() and creates
//     its own context + graph — and `get_ctx()` / `get_gf()`.  There is no
//     constructor that adopts an external context, and there must not be one.
//   * so the probe uses the shipped ctor: the result OWNS its context, the probe
//     creates the weights, the input and the readback tensors inside
//     res->get_ctx(), and the library's build_hc_mix reads that very context
//     through res->get_ctx().  Nothing about the container is faked.
//   * `ggml_backend_buffer::context` is the backend's private context (a CPU
//     malloc base address, etc.), never a ggml_context *, so it is not used here.
//   * the graph the result created is an empty cgraph; the probe fills it, and the
//     caller reserves it with ggml_gallocr_* exactly like any other graph.
//
// There is no observable retention decision the probe can query (llama/ggml
// expose none, and no production API is added for the test).  What the probe
// reports instead: the tensor identity/type of lo and xn, and per site whether the
// fused kernel ran.  The latter is read from the tail diagnostics the model TU
// already emits under M3_TAIL_BUILD/M3_TAIL_WHY (Witness A), and the numeric
// A/B (Witness B) comes from diag/m3-n1-rowref.py over the row images.
//
// Authored, NOT built or run.

#include "models.h"
#include "llama-graph.h"
#include "llama-model.h"
#include "ggml-cpp.h"

#include <memory>

// A site description owned by the harness: the tensors the builder created (in the
// result's context) and the graph it filled.
struct m3_probe_site {
    bool mixer = false;
    m3_probe_site * tensors_out = nullptr;  // when set, receives the created tensors
    ggml_tensor * norm   = nullptr;
    ggml_tensor * down   = nullptr;
    ggml_tensor * up     = nullptr;
    ggml_tensor * inject = nullptr;
    ggml_tensor * x      = nullptr;
    ggml_cgraph * gf     = nullptr;
};

// ---- state installed by the device test -----------------------------------
static struct {
    llama_model * model = nullptr;    // real qwen4exp model, used for hparams only
    llm_graph_result * res = nullptr; // owns the context/graph the probe builds in
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

// Install the model.  Refused while a previous result is still outstanding.
bool m3_probe_attach(llama_model * model) {
    if (model == nullptr || g_probe.res != nullptr) {
        return false;
    }
    g_probe.model = model;
    return true;
}

// Build one HC site with the shipped builder.
//
// The result uses its own public constructor, so it creates and owns an empty
// ggml context and an empty cgraph.  Everything the builder needs is then created
// INSIDE that context via res->get_ctx(), which is also the pointer the library
// reads inside build_hc_mix (it dereferences res->get_ctx() in its own TU), so the
// probe and the library use the same context by construction.
//
// Lifetime: the created site lives in the result; the caller uses the returned
// pointers until it calls m3_probe_release(), which deletes the result (and with
// it the context, the graph and every tensor created here).
//
// A second build without a release is refused, and so is an attach while a result
// is outstanding: there is never more than one owner.
ggml_tensor * m3_probe_build_hc_mix(llama_model * model, int64_t n_embd, int64_t hc,
                                    const m3_probe_site & spec, int64_t nt, int il,
                                    ggml_tensor ** lo_out, ggml_tensor ** up_out,
                                    ggml_tensor ** inject_out) {
    if (model == nullptr || g_probe.res != nullptr) {
        return nullptr;   // one site at a time; a stale result must be released first
    }
    // Production shapes only: the Q8_0 door rejects anything else, and the probe
    // must not silently test a different shape than the model builds.
    if (n_embd != 2560 || hc != 4 || nt < 1 || nt > 4) {
        return nullptr;
    }

    std::unique_ptr<llm_graph_result> res(new llm_graph_result(/*max_nodes=*/4096));
    ggml_context * ctx = res->get_ctx();
    if (ctx == nullptr) {
        return nullptr;
    }

    // Weights and input, in the result's own context, exactly as the model stores
    // them: norm F32 [n_embd, hc], down Q8_0 [hc*n_embd, 320], up Q8_0
    // [320, hc*n_embd], inject Q8_0 [hc*n_embd, hc] (trunk) or absent (mixer).
    ggml_tensor * w_norm   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  n_embd, hc);
    ggml_tensor * w_down   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, hc * n_embd, 320);
    ggml_tensor * w_up     = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 320, hc * n_embd);
    ggml_tensor * w_inject = spec.mixer ? nullptr :
            ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, hc * n_embd, hc);
    ggml_tensor * xn = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, nt);
    if (w_norm == nullptr || w_down == nullptr || w_up == nullptr || xn == nullptr ||
            (!spec.mixer && w_inject == nullptr)) {
        return nullptr;
    }
    ggml_set_name(w_norm, "m3_probe_w_norm");
    ggml_set_name(w_down, "m3_probe_w_down");
    ggml_set_name(w_up, "m3_probe_w_up");
    ggml_set_name(xn, "m3_probe_x");
    if (w_inject != nullptr) {
        ggml_set_name(w_inject, "m3_probe_w_inject");
    }
    ggml_set_input(xn);
    // The builder reads the context through res->get_ctx() and the graph through
    // res->get_gf(); the graph is the empty cgraph the result created.
    ggml_cgraph * gf = res->get_gf();

    llm_graph_params params;
    params.hparams     = model->hparams;
    params.n_outputs   = uint32_t(nt);
    params.sched       = nullptr;
    params.backend_cpu = nullptr;
    params.mctx        = nullptr;
    params.res         = res.get();

    probe_graph g(*model, params);
    ggml_tensor * inject_slot = w_inject;
    ggml_tensor ** inject_arg = w_inject != nullptr ? &inject_slot : nullptr;
    // Seven-argument production call: the real weight is passed, and only the
    // inject OUTPUT slot is null, which is how the builder selects a mixer site.
    ggml_tensor * mixed = g.hc_mix(xn, w_norm, w_down, w_up, w_inject, inject_arg, int(il));
    // ggml_cgraph is opaque in the public headers, so the node count is read
    // through the documented accessor rather than by touching the struct.
    if (mixed == nullptr || ggml_graph_n_nodes(gf) == 0) {
        return nullptr;
    }
    if (lo_out != nullptr) {
        *lo_out = ggml_graph_get_tensor(gf, "hc_lo");
    }
    if (up_out != nullptr) {
        *up_out = ggml_graph_get_tensor(gf, "hc_up");
    }
    if (inject_out != nullptr) {
        *inject_out = inject_arg != nullptr ? inject_slot : nullptr;
    }
    // Hand the tensors back to the harness for seeding and for the graph allocator.
    // The offsets are for the weight/input tensors the harness fills.
    if (spec.tensors_out != nullptr) {
        m3_probe_site * out = spec.tensors_out;
        out->norm   = w_norm;
        out->down   = w_down;
        out->up     = w_up;
        out->inject = w_inject;
        out->x      = xn;
        out->gf     = gf;
    }
    g_probe.res = res.release();
    return mixed;
}

// Destroy the result (and with it the context, the graph and every tensor of the
// last site) exactly once; frees ggml memory through the shipped deleter.
void m3_probe_release() {
    delete g_probe.res;
    g_probe.res = nullptr;
}

// The retention flag is an internal decision with no public probe.  This helper
// exists only so the harness can LOG the raw flag bits next to the kernel witness;
// it must not be read as the pass criterion (see the header comment).
bool m3_probe_lo_retained(ggml_tensor * lo) {
    return lo != nullptr && (lo->flags & GGML_TENSOR_FLAG_NO_REUSE) != 0;
}

} // extern "C"
