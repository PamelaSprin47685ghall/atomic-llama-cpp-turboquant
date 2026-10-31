//
// llama-nout-state.cpp
//
// Per-context state for the combined runtime-engine patches
// (env: GGML_TP5_UNION_BINDER, exact "1").
// See llama-nout-state.h for the state-placement rationale.
//
// Storage: the state lives in a heap slot owned by the llama_context itself
// (the `union_binder_state` unique_ptr member declared in llama-context.h).
// The slot is created lazily by the llama_context constructor - and only when
// the GGML_TP5_UNION_BINDER gate is exactly "1" - so the OFF path allocates
// no binder state; ~llama_context destroys the member (exception-safe: a
// constructor-body throw still runs member destruction), and the state's
// out-of-line destructor releases the held scratch graph result.
// There is no process-global map: nothing is shared between contexts, no
// lookup is keyed by a raw address, and a context can never observe another
// context's record/counters (previously possible if a llama_context address
// was reused after destruction and the erase had been skipped).
//
// The nkv and n_outputs lanes reconcile through the binder's shared record;
// there is no side table anywhere (the old kv-cache override map was removed
// with the envelope build).

#include "llama-nout-state.h"

#include "llama-context.h"
#include "llama-graph.h"

llama_nkv_ctx_state & llama_nkv_ctx_state_get(llama_context * ctx) {
    return ctx->union_binder_state_get();
}

// out-of-line destructor: the only place that needs llm_graph_result complete.
// Releases the scratch graph result held as a void* (llama-context.h holds
// this struct behind a unique_ptr to the incomplete type; that unique_ptr is
// destroyed by ~llama_context, which is also out-of-line in llama-context.cpp).
llama_nkv_ctx_state::~llama_nkv_ctx_state() {
    if (nkv_scratch != nullptr) {
        llm_graph_result_ptr res;
        res.reset(static_cast<llm_graph_result *>(nkv_scratch));
    }
}
