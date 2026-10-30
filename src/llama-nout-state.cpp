//
// llama-nout-state.cpp
//
// ABI-stable side tables for the combined runtime-engine patches
// (env: LLAMA_GRAPH_UNION_NKV_RUNTIME, LLAMA_GRAPH_UNION_OUTPUTS_RUNTIME).
// See llama-nout-state.h for the layout-preservation rationale.
//
// All state here is keyed by llama_context * and session-scoped; entries are
// erased in llama_context::~llama_context() (llama-context.cpp), which is the
// single owner of the context lifetime. The kv-cache override and the
// out_ids bind-env maps live in their owning TUs instead
// (llama-kv-cache.cpp / llama-graph.cpp) with matching destructor cleanup.
//

#include "llama-nout-state.h"

#include "llama-context.h"
#include "llama-graph.h"

#include <unordered_map>

namespace {
std::unordered_map<llama_context *, llama_nkv_ctx_state> g_nkv_ctx_state;

// reverse index: which llama_context owns a given llm_graph_result; written
// by llama_nkv_ctx_state_get() (every context registers on first use, i.e.
// in its constructor env blocks) and erased together with the main entry
std::unordered_map<const llm_graph_result *, llama_context *> g_result_owner;
} // namespace

llama_nkv_ctx_state & llama_nkv_ctx_state_get(llama_context * ctx) {
    auto [it, inserted] = g_nkv_ctx_state.try_emplace(ctx);
    if (inserted) {
        // register the context's long-lived result object for the reverse
        // lookup used by llama_nout_runtime_enabled(); llama_context keeps
        // gf_res_prev alive for its whole lifetime (rebuilt only via
        // graph_reserve(), which calls this again to refresh the mapping)
        if (llm_graph_result * res = ctx->get_gf_res_prev()) {
            g_result_owner[res] = ctx;
        }
    }
    return it->second;
}

void llama_nkv_ctx_state_erase(llama_context * ctx) {
    auto & st = g_nkv_ctx_state[ctx];

    // drop the reverse mapping before the result object dies with the context
    if (llm_graph_result * res = ctx->get_gf_res_prev()) {
        g_result_owner.erase(res);
    }

    // destroy the held scratch llm_graph_result (stored opaque in the table
    // so this TU does not need llm_graph_result_ptr in the header)
    if (st.nkv_scratch != nullptr) {
        llm_graph_result_ptr res;
        res.reset(static_cast<llm_graph_result *>(st.nkv_scratch));
    }

    g_nkv_ctx_state.erase(ctx);
}

bool llama_nout_runtime_enabled(const llm_graph_params & params) {
    // the effective flag is a pure function of the owning context's env state
    // (LLAMA_GRAPH_UNION_OUTPUTS_RUNTIME == "1" and no backend samplers);
    // derived on the fly instead of stored in llm_graph_params.
    // The owning llama_context is located via the per-context result object
    // (params.res), which llama_context owns for its whole lifetime.
    const auto it = g_result_owner.find(params.res);
    if (it == g_result_owner.end()) {
        return false; // unregistered result: feature off (fail-closed)
    }
    return g_nkv_ctx_state[it->second].nout_enabled;
}

bool llama_nout1_runtime_enabled(const llm_graph_params & params) {
    // floor1 lane: same owner-map derivation as the envelope lane; the flag is
    // a pure function of the owning context's env state
    // (LLAMA_NOUT_FLOOR1_REUSE == "1", envelope lane off, no backend samplers)
    const auto it = g_result_owner.find(params.res);
    if (it == g_result_owner.end()) {
        return false; // unregistered result: feature off (fail-closed)
    }
    return g_nkv_ctx_state[it->second].nout1_enabled;
}

void llama_nkv_record_restore(llama_context * ctx) {
    // restore every non-fixed recorded tensor's src[]/view_src pointers from
    // the pre-allocation record (undoes the scheduler's split rewrites); the
    // record was taken pre-alloc, so it holds the pristine build-time
    // topology. -1 = null src (unchanged), -2 = back-edge (leave untouched).
    auto & st = llama_nkv_ctx_state_get(ctx);
    if (!st.record_valid) {
        // no valid pre-alloc record for the live graph: the table may hold
        // pointers into an already-discarded graph (any build path that
        // replaced the graph without re-snapshotting); restoring from it
        // would write dangling src pointers into the current graph (UAF)
        return;
    }
    const size_t n = st.record_t.size();
    for (size_t k = 0; k < n; ++k) {
        if (st.record_fixed[k]) {
            continue;
        }
        const int32_t vsrc = st.record_src[k * (GGML_MAX_SRC + 1) + GGML_MAX_SRC];
        if (vsrc >= 0) {
            st.record_t[k]->view_src = st.record_t[vsrc];
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const int32_t v = st.record_src[k * (GGML_MAX_SRC + 1) + j];
            if (v >= 0) {
                st.record_t[k]->src[j] = st.record_t[v];
            }
        }
    }
}

void llama_nkv_record_clear_alloc(llama_context * ctx) {
    // clear stale allocation state (see the nkv patch in llama-context.cpp
    // for the full rationale; same rule, shared implementation)
    auto & st = llama_nkv_ctx_state_get(ctx);
    if (!st.record_valid) {
        // no valid record for the live graph: never clear allocation state
        // of tensors that no longer belong to the current graph
        return;
    }
    const size_t n = st.record_t.size();
    for (size_t k = 0; k < n; ++k) {
        if (st.record_fixed[k]) {
            continue;
        }
        const int32_t vsrc = st.record_src[k * (GGML_MAX_SRC + 1) + GGML_MAX_SRC];
        if (vsrc >= 0 && st.record_fixed[vsrc]) {
            // view of a fixed tensor (KV view): keep its buffer/data
            continue;
        }
        ggml_tensor * d = st.record_t[k];
        d->buffer = nullptr;
        d->data   = nullptr;
        d->extra  = nullptr;
    }
}
