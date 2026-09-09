#pragma once

#include "llama.h"
#include "llama-graph.h"
#include "llama-ext.h"
#include "llama-rerot.h"

#include <algorithm>
#include <cstdint>
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

    // XKV parameters
    enum llama_xkv_mode xkv_mode = LLAMA_XKV_MODE_OFF;
    uint32_t xkv_segment_tokens  = 0;
    uint32_t xkv_chunk_tokens    = 0;
    uint32_t n_ubatch            = 0;
    uint32_t n_batch             = 0;
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

    // Post-compute hooks called exactly around graph_compute
    // Only success transitions applied hot_writing rows to hot_committed.
    // Failure leaves them uncommitted and triggers rollback/cleanup.
    // Both report success/failure: success commits (and only then releases
    // deferred victims); failure runs one coordinated rollback transaction.
    // Callers must propagate a false return instead of continuing as if the
    // cache committed.
    virtual bool postcompute_success() { return true; }
    virtual bool postcompute_failure() { return true; }

    // get the current ubatch
    virtual const llama_ubatch & get_ubatch() const = 0;

    // Graph-facing bounded-hot physical views (exact spelling required by the
    // XKV attention path). Default nullptr is fail-closed for memories without
    // bounded hot; the kv/iswa/hybrid contexts override and forward.
    virtual ggml_tensor * get_xkv_hot_k(ggml_context * /*ctx*/, int32_t /*il*/) const { return nullptr; }
    virtual ggml_tensor * get_xkv_hot_v(ggml_context * /*ctx*/, int32_t /*il*/) const { return nullptr; }

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


// Finalize a partially filled snapshot: derive safe_next_ubatch (raw token
// bound, may be 0) and the tightest-domain limit_reason.
// - Token domains only: logical cells, hot slots, factor_safe_tokens,
//   workspace_safe_tokens. Recurrent capacity holds sequence slots, not
//   token slots, and never enters this minimum (see
//   recurrent_blocks_new_seq below).
// - hot_free is used exactly as provided (it already excludes hot_reserved);
//   it is only clamped into [0, hot_capacity]. It is never recomputed from
//   capacity minus used, so reserved slots are never over-admitted.
// - Byte domains limit solely through their planner token bounds:
//   UINT32_MAX means unconstrained. Raw free-byte counts are informational
//   and never bind (nonzero bytes may still be insufficient).
// - limit_reason names the minimum finite bound even when it is positive;
//   NONE means unknown snapshot only (no finite domain). Priority on ties:
//   logical cells, hot slots, factor store, workspace.
inline void llama_memory_admission_finalize(llama_memory_admission_snapshot & snap) {
    constexpr uint32_t INF = UINT32_MAX;
    uint32_t logical_free = INF;
    if (snap.logical_capacity != 0) {
        logical_free = snap.logical_capacity - std::min(snap.logical_used, snap.logical_capacity);
    }
    uint32_t hot_bound = INF;
    if (snap.hot_capacity != 0) {
        hot_bound = std::min(snap.hot_free, snap.hot_capacity);
    }
    const uint32_t factor_bound = snap.factor_safe_tokens;
    const uint32_t workspace_bound = snap.workspace_safe_tokens;

    uint32_t safe = INF;
    safe = std::min(safe, logical_free);
    safe = std::min(safe, hot_bound);
    safe = std::min(safe, factor_bound);
    safe = std::min(safe, workspace_bound);

    snap.limit_reason = LLAMA_MEMORY_LIMIT_NONE;
    snap.safe_next_ubatch = 0;
    if (safe != INF) {
        snap.safe_next_ubatch = safe;
        if (logical_free == safe) {
            snap.limit_reason = LLAMA_MEMORY_LIMIT_LOGICAL_CELLS;
        } else if (hot_bound == safe) {
            snap.limit_reason = LLAMA_MEMORY_LIMIT_HOT_SLOTS;
        } else if (factor_bound == safe) {
            snap.limit_reason = LLAMA_MEMORY_LIMIT_FACTOR_STORE;
        } else {
            snap.limit_reason = LLAMA_MEMORY_LIMIT_WORKSPACE;
        }
    }

    // Recurrent gating is separate from the token bound: full recurrent
    // slots block only NEW sequences; continuation batches proceed.
    snap.recurrent_blocks_new_seq =
        snap.recurrent_capacity != 0 &&
        snap.recurrent_used >= snap.recurrent_capacity;
}

// Combine an attention-domain snapshot with recurrent capacity/used
// (hybrid memories). Attention token bounds and reason are preserved (the
// recurrent domain never enters the token minimum); only the separate
// recurrent new-sequence gate is recomputed.
inline llama_memory_admission_snapshot llama_memory_admission_combine(
        const llama_memory_admission_snapshot & attn,
        uint32_t recurrent_capacity,
        uint32_t recurrent_used) {
    llama_memory_admission_snapshot out = attn;
    out.recurrent_capacity = recurrent_capacity;
    out.recurrent_used     = std::min(recurrent_used, recurrent_capacity);
    out.recurrent_blocks_new_seq =
        out.recurrent_capacity != 0 &&
        out.recurrent_used >= out.recurrent_capacity;
    return out;
}

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

    void set_backend_sched(ggml_backend_sched_t sched) {
        backend_sched = sched;
    }

protected:
    ggml_backend_sched_t backend_sched = nullptr;

public:
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

    // Physical hot backing capacity across streams
    virtual uint32_t get_kv_hot_capacity() const { return get_kv_capacity(); }
    virtual bool can_use_legacy_attention() const { return true; }
    virtual bool is_xkv_bounded_hot() const { return false; }

    // POD admission snapshot combining every resource domain this memory
    // manages. The default derives exact legacy values (logical cells from
    // get_kv_capacity()/get_kv_used(), hot slots from get_kv_hot_capacity(),
    // recurrent from get_recurrent_capacity()/get_recurrent_used()) and
    // reports zero factor store / workspace bytes (XKV OFF mapping).
    // Memories that own an XKV factor store override this to fill the byte
    // domains. Wrapper memories override it to combine sub-memory domains.
    // Returns false when out is null or no domain has capacity.
    virtual bool get_admission_snapshot(struct llama_memory_admission_snapshot * out) const {
        if (out == nullptr) {
            return false;
        }
        llama_memory_admission_snapshot snap = {};
        snap.logical_capacity = get_kv_capacity();
        snap.logical_used     = std::min(get_kv_used(), snap.logical_capacity);
        snap.hot_capacity     = get_kv_hot_capacity();
        snap.hot_used         = std::min(get_kv_used(), snap.hot_capacity);
        snap.hot_free         = snap.hot_capacity - snap.hot_used;
        snap.hot_reserved     = 0;
        // No bound store: planner token bounds stay unconstrained.
        snap.factor_safe_tokens    = UINT32_MAX;
        snap.workspace_safe_tokens = UINT32_MAX;
        snap.recurrent_capacity = get_recurrent_capacity();
        snap.recurrent_used     = std::min(get_recurrent_used(), snap.recurrent_capacity);
        llama_memory_admission_finalize(snap);
        *out = snap;
        return snap.logical_capacity != 0 || snap.recurrent_capacity != 0;
    }

    // Attempt cache-owned safe-boundary maintenance before the caller falls
    // back to legacy TriAttention reclaim or slot preemption. Default is a
    // no-op returning NO_ACTION; the runtime coordinator overrides this
    // where an XKV store is bound. The server proceeds to Tri/atomic
    // fallback only on FLOOR_EXHAUSTED or NO_ACTION, and takes the error
    // path on ERROR (never masked as success).
    virtual llama_memory_maintenance_status maintain_safe_boundary() {
        return LLAMA_MEMORY_MAINTENANCE_NO_ACTION;
    }

    // Observed XKV runtime/store snapshot. Default reports unbound (false)
    // so callers show empty/not_evaluated; the bound runtime overrides to
    // fill the authoritative public POD. Wrapper memories forward.
    virtual bool get_xkv_runtime_snapshot(struct llama_memory_xkv_runtime_snapshot * out) const {
        if (out != nullptr) {
            *out = {};
        }
        return false;
    }

    virtual uint32_t get_recurrent_capacity() const { return 0; }
    virtual uint32_t get_recurrent_used()     const { return 0; }
    virtual uint32_t get_recurrent_seq_used(llama_seq_id seq_id) const { GGML_UNUSED(seq_id); return 0; }

    // RERoT grouped layout (§§B.3.2, B.6, B.13 Phase 3).
    virtual void set_grouped_layout(uint32_t n_brains, uint32_t n_hands) {
        GGML_UNUSED(n_brains);
        GGML_UNUSED(n_hands);
    }
    virtual uint32_t get_brain_capacity() const { return get_recurrent_capacity(); }
    virtual uint32_t get_hand_capacity()  const { return get_recurrent_capacity(); }
    virtual uint32_t get_brain_used()     const { return get_recurrent_used(); }
    virtual uint32_t get_hand_used()      const { return get_recurrent_used(); }

    //
    // ops
    //

    // if data == true, the data buffers will also be cleared together with the metadata
    virtual void clear(bool data) = 0;

    // Failure-reporting clear. The default funnels legacy void clear() and
    // converts an exception into false so no C++ exception crosses the C
    // ABI. Memories with fallible commit steps (XKV store/pool) override this
    // with real bool logic and keep their void clear() as a fatal wrapper.
    virtual bool try_clear(bool data, std::string * err = nullptr) {
        try {
            clear(data);
        } catch (const std::exception & e) {
            if (err) *err = e.what();
            return false;
        }
        return true;
    }

    virtual bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) = 0;
    virtual void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) = 0;
    virtual void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) = 0;

    // Component-selective operations used by RERoT (§6.3). Attention-only
    // memory is the common case, so the default delegates to the ordinary
    // sequence op; recurrent-only defaults to a successful no-op. Recurrent
    // and hybrid memory implementations override these methods explicitly
    // (hybrid forwards each side to its sub-memory). The inapplicable side
    // must stay vacuously successful: release paths call both sides
    // unconditionally, so a failure default would break retire on
    // single-component memories.
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
    virtual void rerot_release_episode(uint64_t episode_id) {
        GGML_UNUSED(episode_id);
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
    virtual bool rerot_can_reclassify_run(
            uint64_t episode_id,
            llama_rerot_run_id run_id,
            llama_rerot_visibility expected,
            llama_rerot_visibility replacement,
            uint64_t publish_epoch,
            size_t * count) const {
        GGML_UNUSED(episode_id);
        GGML_UNUSED(run_id);
        GGML_UNUSED(expected);
        GGML_UNUSED(replacement);
        GGML_UNUSED(publish_epoch);
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
    virtual size_t rerot_reclassify_run(
            uint64_t episode_id,
            llama_rerot_run_id run_id,
            llama_rerot_visibility expected,
            llama_rerot_visibility replacement,
            uint64_t publish_epoch) {
        GGML_UNUSED(episode_id);
        GGML_UNUSED(run_id);
        GGML_UNUSED(expected);
        GGML_UNUSED(replacement);
        GGML_UNUSED(publish_epoch);
        return 0;
    }
    virtual bool rerot_can_add_run_ref(
            uint64_t episode_id,
            llama_rerot_run_id run_id,
            llama_seq_id seq_id,
            size_t * count) const {
        GGML_UNUSED(episode_id);
        GGML_UNUSED(run_id);
        GGML_UNUSED(seq_id);
        if (count) {
            *count = 0;
        }
        return false;
    }
    virtual size_t rerot_add_run_ref(
            uint64_t episode_id,
            llama_rerot_run_id run_id,
            llama_seq_id seq_id) {
        GGML_UNUSED(episode_id);
        GGML_UNUSED(run_id);
        GGML_UNUSED(seq_id);
        return 0;
    }
    virtual bool rerot_set_reader_view(llama_seq_id seq_id, const llama_rerot_reader_state & view) {
        GGML_UNUSED(seq_id);
        GGML_UNUSED(view);
        return false;
    }
    virtual void rerot_clear_reader_view(llama_seq_id seq_id) {
        GGML_UNUSED(seq_id);
    }

    // Shared fork hand seed (§16.1, §16.4, §B.6.4)
    virtual size_t rerot_hand_seed_size(llama_seq_id source_seq) const {
        GGML_UNUSED(source_seq);
        return 0;
    }
    virtual bool rerot_capture_hand_seed(llama_seq_id source_seq, std::vector<uint8_t> & seed_out) {
        GGML_UNUSED(source_seq);
        GGML_UNUSED(seed_out);
        return false;
    }
    virtual bool rerot_apply_hand_seed(llama_seq_id dest_seq, const std::vector<uint8_t> & seed_in) {
        GGML_UNUSED(dest_seq);
        GGML_UNUSED(seed_in);
        return false;
    }

    virtual bool rerot_commit_rbb_frontier(
            uint32_t person_id,
            const llama_seq_id * candidate_seqs,
            const uint8_t * is_public_write,
            size_t n_candidates) {
        GGML_UNUSED(person_id);
        GGML_UNUSED(candidate_seqs);
        GGML_UNUSED(is_public_write);
        GGML_UNUSED(n_candidates);
        return false;
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
