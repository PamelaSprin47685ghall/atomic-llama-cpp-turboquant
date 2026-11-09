// Permanent build-tree regression for the public TP5 API declarations.
// The installed-package find_package/symbol checks are a separate staged
// verification; this test keeps the public header and link surface covered.
#include <ggml-backend.h>
#include <llama.h>

#include <cstdio>
#include <type_traits>

static_assert(LLAMA_MEMORY_HYBRID_IDX_TRACE_VERSION == 1);
static_assert(std::is_standard_layout_v<llama_memory_hybrid_idx_trace>);

int main() {
    using set_nextn_t  = void  (*)(llama_context *, bool, bool);
    using get_nextn_t  = float * (*)(llama_context *, int32_t);
    using trace_t      = bool  (*)(llama_memory_t, llama_seq_id, llama_memory_hybrid_idx_trace *);
    using hook_t       = bool  (*)(bool (*)(void *), void *);

    const set_nextn_t set_nextn = &llama_set_embeddings_nextn;
    const get_nextn_t get_nextn = &llama_get_embeddings_nextn_ith;
    const trace_t     trace_fn  = &llama_memory_hybrid_idx_get_trace;
    const hook_t      ready_fn  = &ggml_backend_meta_set_submit_ready_hook;
    const hook_t      submit_fn = &ggml_backend_meta_set_submitted_hook;

    (void) set_nextn;
    (void) get_nextn;
    (void) submit_fn;

    llama_memory_hybrid_idx_trace trace{};
    trace.version = LLAMA_MEMORY_HYBRID_IDX_TRACE_VERSION;
    trace.struct_size = sizeof(trace);
    const bool trace_ok = trace_fn(nullptr, 0, &trace);
    const bool ready_ok = ready_fn(nullptr, nullptr);

    std::printf("public_tp5_api trace=%d hook=%d sizeof_trace=%zu version=%u\n",
                int(trace_ok), int(ready_ok), sizeof(trace),
                unsigned(LLAMA_MEMORY_HYBRID_IDX_TRACE_VERSION));
    return 0;
}
