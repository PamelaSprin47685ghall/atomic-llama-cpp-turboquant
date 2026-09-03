#pragma once

#include "llama.h"
#include "llama-graph.h"
#include "llama-ext.h"
#include "llama-rerot.h"

#include <map>
#include <memory>
#include <functional>

struct llama_ubatch;

class llama_batch_allocr;

class llama_io_write_i;
class llama_io_read_i;

struct llama_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;

    llama_context_type ctx_type;

    llama_memory_t mem_other;

    bool triattention_enabled = false;
    const char * triattention_stats = nullptr;
};

enum llama_memory_status {
    LLAMA_MEMORY_STATUS_SUCCESS = 0,
    LLAMA_MEMORY_STATUS_NO_UPDATE,
    LLAMA_MEMORY_STATUS_FAILED_PREPARE,
    LLAMA_MEMORY_STATUS_FAILED_COMPUTE,
};

// helper function for combining the status of two memory contexts
// useful for implementing hybrid memory types (e.g. iSWA)
llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1);

// helper function for checking if a memory status indicates a failure
bool llama_memory_status_is_fail(llama_memory_status status);

// the interface for managing the memory context during batch processing
// this interface is implemented per memory type. see:
//   - llama_kv_cache_context
//   - llama_kv_cache_iswa_context
//   ...
//
// the only method that should mutate the memory and the memory context is llama_memory_i::apply()
struct llama_memory_context_i {
    virtual ~llama_memory_context_i() = default;

    // consume the current ubatch from the context and proceed to the next one
    // return false if we are done
    virtual bool next() = 0;

    // apply the memory state for the current ubatch to the memory object
    // return false on failure
    virtual bool apply() = 0;

    // get the current ubatch
    virtual const llama_ubatch & get_ubatch() const = 0;

    // get the status of the memory context - used for error handling and checking if any updates would be applied
    virtual llama_memory_status get_status() const = 0;

    // TurboQuant: get rotation tensors for pre-rotate-queries optimization
    // Returns null for non-turbo memory types. Override in KV cache contexts.
    virtual ggml_tensor * get_turbo_rot_forward() const { return nullptr; }
    virtual ggml_tensor * get_turbo_rot_inverse() const { return nullptr; }

    // TurboQuant InnerQ: get per-channel scale_inv tensor for Q/V equalization
    // Returns nullptr when InnerQ is not active. Override in KV cache contexts.
    virtual ggml_tensor * get_turbo_innerq_scale_inv() const { return nullptr; }
};

using llama_memory_context_ptr = std::unique_ptr<llama_memory_context_i>;

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
struct llama_memory_i {
    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    // this callback is used to specify which layers should reuse memory from other layers
    // return negative value to indicate that the layer il should not reuse memory
    using layer_reuse_cb = std::function<int32_t(int32_t il)>;

    using layer_share_cb = std::function<int32_t(int32_t il)>;

    virtual ~llama_memory_i() = default;

    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a context object containing the ubatches and memory state required to process them
    // check the llama_memory_context_i::get_status() for the result
    virtual llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual llama_memory_context_ptr init_full() = 0;

    // prepare for any pending memory updates, such as shifts, copies, etc.
    // status == LLAMA_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) = 0;

    // getters
    virtual bool get_can_shift() const = 0;

    virtual uint32_t get_kv_capacity() const { return 0; }
    virtual uint32_t get_kv_used()     const { return 0; }
    virtual uint32_t get_kv_seq_used(llama_seq_id seq_id) const { GGML_UNUSED(seq_id); return 0; }

    virtual uint32_t get_recurrent_capacity() const { return 0; }
    virtual uint32_t get_recurrent_used()     const { return 0; }
    virtual uint32_t get_recurrent_seq_used(llama_seq_id seq_id) const { GGML_UNUSED(seq_id); return 0; }

    //
    // ops
    //

    // if data == true, the data buffers will also be cleared together with the metadata
    virtual void clear(bool data) = 0;

    virtual bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) = 0;
    virtual void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) = 0;
    virtual void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) = 0;

    // Component-selective operations used by RERoT. Attention-only memory is
    // the common case, so the default delegates to the ordinary sequence op;
    // recurrent-only defaults to a successful no-op. Recurrent and hybrid
    // memory implementations override these methods explicitly.
    virtual bool seq_rm_attention(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
        return seq_rm(seq_id, p0, p1);
    }
    virtual void seq_cp_attention(
            llama_seq_id seq_id_src,
            llama_seq_id seq_id_dst,
            llama_pos p0,
            llama_pos p1) {
        seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
    virtual bool seq_rm_recurrent(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
        GGML_UNUSED(seq_id);
        GGML_UNUSED(p0);
        GGML_UNUSED(p1);
        return true;
    }
    virtual void seq_cp_recurrent(
            llama_seq_id seq_id_src,
            llama_seq_id seq_id_dst,
            llama_pos p0,
            llama_pos p1) {
        GGML_UNUSED(seq_id_src);
        GGML_UNUSED(seq_id_dst);
        GGML_UNUSED(p0);
        GGML_UNUSED(p1);
    }

    // RERoT physical-cell classification. Implementations with no attention
    // cache report unsupported. Composite memories must validate publication
    // across all attention components before mutating any component.
    virtual bool rerot_set_write_tag(llama_seq_id seq_id, const llama_kv_rerot_meta & tag) {
        GGML_UNUSED(seq_id);
        GGML_UNUSED(tag);
        return false;
    }
    virtual void rerot_clear_write_tag(llama_seq_id seq_id) {
        GGML_UNUSED(seq_id);
    }
    virtual bool rerot_can_publish_run(
            uint64_t episode_id,
            llama_rerot_run_id run_id,
            size_t * count) const {
        GGML_UNUSED(episode_id);
        GGML_UNUSED(run_id);
        if (count) {
            *count = 0;
        }
        return false;
    }
    virtual size_t rerot_publish_run(
            uint64_t episode_id,
            llama_rerot_run_id run_id,
            uint64_t publish_epoch) {
        GGML_UNUSED(episode_id);
        GGML_UNUSED(run_id);
        GGML_UNUSED(publish_epoch);
        return 0;
    }

    virtual llama_pos seq_pos_min(llama_seq_id seq_id) const = 0;
    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const = 0;

    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;

    //
    // state write/read
    //

    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const = 0;
    virtual void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) = 0;

    // Lossy KV reclaim — allows the server to request freeing KV cells
    // Default implementation returns unsupported (no reclaim possible)
    virtual llama_memory_kv_reclaim_result reclaim_kv(const llama_memory_kv_reclaim_request & request) {
        GGML_UNUSED(request);
        llama_memory_kv_reclaim_result result;
        result.supported = false;
        return result;
    }

    // Returns true if positions in [seq_pos_min, seq_pos_max] may have gaps
    // (e.g. after TriAttention eviction). Callers must not assume all positions
    // in the range exist. Default is false (dense positions).
    virtual bool positions_are_sparse() const { return false; }
};

using llama_memory_ptr = std::unique_ptr<llama_memory_i>;
