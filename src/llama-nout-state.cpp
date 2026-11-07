//
// llama-nout-state.cpp
//
// Per-context state for the combined runtime-engine patches
// (env: GGML_TP5_UNION_BINDER, LLAMA_GRAPH_UNION_OUTPUTS_RUNTIME).
// See llama-nout-state.h for the state-placement rationale.
//
// Storage: the state lives in a heap slot owned by the llama_context itself
// (the `union_binder_state` unique_ptr member declared in llama-context.h).
// The constructor creates the slot only when the binder gate is on. The
// context owns it, including during constructor exception cleanup.
// There is no process-global map: nothing is shared between contexts, no
// lookup is keyed by a raw address, and a context can never observe another
// context's record/counters (previously possible if a llama_context address
// was reused after destruction and the erase had been skipped).
//
// The kv-cache override and the out_ids bind-env maps live in their owning
// TUs instead (llama-kv-cache.cpp / llama-graph.cpp) with matching
// destructor cleanup.
//

#include "llama-nout-state.h"

#include "llama-context.h"

llama_nkv_ctx_state & llama_nkv_ctx_state_get(llama_context * ctx) {
    return ctx->union_binder_state_get();
}

// out-of-line destructor: keeps the state's destruction in the TU that owns
// the definition. llama-context.h holds this struct behind a unique_ptr to
// the incomplete type; that unique_ptr is destroyed by ~llama_context, which
// is also out-of-line in llama-context.cpp.


llama_nkv_ctx_state::~llama_nkv_ctx_state() = default;
