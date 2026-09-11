#pragma once

#include "ggml.h"
#include "llama-cparams.h"
#include "llama-rerot.h"
#include "llama-xkv-cache.h"
#include "llama-xkv-codec.h"
#include "llama-xkv-landmark.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llama_xkv {

// Online softmax state maintaining running (m, z, u) in FP32
// m: running max logit across all pieces
// z: running normalizer sum exp(logit - m)
// u: running weighted value accumulator sum(exp(logit - m) * v)
// Final output = u / z (or 0 if z == 0)
struct xkv_online_softmax_state {
    float max_score = -INFINITY;
    float sum_exp   = 0.0f;
    float * accum_ptr = nullptr;
    std::vector<float> accum; // fallback backing when accum_ptr is nullptr

    void init(float * ptr, uint32_t dv) {
        max_score = -INFINITY;
        sum_exp = 0.0f;
        accum_ptr = ptr;
        if (accum_ptr) {
            std::memset(accum_ptr, 0, dv * sizeof(float));
        } else {
            accum.assign(dv, 0.0f);
        }
    }

    float * get_accum() {
        return accum_ptr ? accum_ptr : accum.data();
    }
    const float * get_accum() const {
        return accum_ptr ? accum_ptr : accum.data();
    }

    void reset(uint32_t dv);
    void update_single(float score, const float * v, uint32_t dv);
    void update_sink(float sink_score, uint32_t dv);
    void merge_block(float block_max, float block_sum, const float * block_accum, uint32_t dv);
    void finalize(float * dst, uint32_t dv) const;
};

// Key identifying a decoded B feature/rank tile in the global cache.
// B tiles are immutable for a given segment version and descriptor: the key carries
// segment_id/version, role, owning layer, kv head, tile, exact feature slice
// (offset + dim), codec epoch, device, domain, and descriptor fingerprint.
// Content/binding epochs are deliberately EXCLUDED: hot-token commits and physical
// relocations must not invalidate immutable B tiles (that guaranteed steady misses).
// Invalidation happens only on retired segment versions, codec teardown, or device
// teardown. The codec fingerprint is kept exact.
struct b_tile_cache_key {
    uint64_t     segment_id = 0;
    uint64_t     segment_version = 0;
    factor_role  role = factor_role::b_k;
    uint32_t     owning_layer = 0;
    uint32_t     kv_head = 0;
    uint32_t     tile_index = 0;
    uint64_t     codec_epoch = 0;
    uint32_t     feature_offset = 0;
    uint32_t     feature_dim = 0;
    uint32_t     device_id = 0;
    value_domain domain = value_domain::canonical;
    uint64_t     desc_fingerprint = 0;

    bool operator==(const b_tile_cache_key & o) const {
        return segment_id == o.segment_id &&
               segment_version == o.segment_version &&
               role == o.role &&
               owning_layer == o.owning_layer &&
               kv_head == o.kv_head &&
               tile_index == o.tile_index &&
               codec_epoch == o.codec_epoch &&
               feature_offset == o.feature_offset &&
               feature_dim == o.feature_dim &&
               device_id == o.device_id &&
               domain == o.domain &&
               desc_fingerprint == o.desc_fingerprint;
    }
    bool operator!=(const b_tile_cache_key & o) const {
        return !(*this == o);
    }
};

struct b_tile_cache_key_hash {
    size_t operator()(const b_tile_cache_key & k) const noexcept {
        uint64_t h = 0xcbf29ce484222325ULL;
        auto mix = [&h](uint64_t v) {
            h ^= v;
            h *= 0x100000001b3ULL;
        };
        mix(k.segment_id);
        mix(k.segment_version);
        mix(static_cast<uint64_t>(k.role));
        mix(k.owning_layer);
        mix(k.kv_head);
        mix(k.tile_index);
        mix(k.codec_epoch);
        mix(k.feature_offset);
        mix(k.feature_dim);
        mix(k.device_id);
        mix(static_cast<uint64_t>(k.domain));
        mix(k.desc_fingerprint);
        return static_cast<size_t>(h);
    }
};

// Internal record tracking lifetime of a cached entry and its leases
struct b_tile_record {
    std::vector<float> data;
    size_t allocated_bytes = 0;
    size_t active_leases = 0;
    bool in_cache = true;
};

class xkv_b_tile_cache;

// RAII Lease for a cached B tile. Holding a lease retains access.
// As long as any lease is active, the underlying record cannot be freed.
// When clear() or evict occurs, entries with active leases remain accounted in allocated_bytes until release.
class xkv_b_tile_lease {
public:
    xkv_b_tile_lease() = default;
    ~xkv_b_tile_lease();

    xkv_b_tile_lease(const xkv_b_tile_lease &) = delete;
    xkv_b_tile_lease & operator=(const xkv_b_tile_lease &) = delete;

    xkv_b_tile_lease(xkv_b_tile_lease && o) noexcept;
    xkv_b_tile_lease & operator=(xkv_b_tile_lease && o) noexcept;

    void release();

    const float * data() const { return ptr_; }
    size_t size() const { return count_; }
    bool valid() const { return ptr_ != nullptr && count_ > 0; }
    explicit operator bool() const { return valid(); }

    // Public static factory for workspace-owned / uncached tile buffers
    static xkv_b_tile_lease uncached(std::shared_ptr<b_tile_record> rec);
    static xkv_b_tile_lease from_buffer(const float * ptr, size_t count);

private:
    friend class xkv_b_tile_cache;
    xkv_b_tile_lease(std::shared_ptr<b_tile_record> rec, xkv_b_tile_cache * owner);

    std::shared_ptr<b_tile_record> record_;
    const float * ptr_ = nullptr;
    size_t count_ = 0;
    xkv_b_tile_cache * owner_ = nullptr;
};

// Global optional B-tile cache with hard byte bounds, explicit lease accounting, and LRU eviction.
// Leased entries are NEVER evicted while active leases exist. If unleased entries are insufficient
// to accommodate a new entry within max_bytes, insertion is refused and put_and_lease returns an uncached lease.
class xkv_b_tile_cache {
public:
    explicit xkv_b_tile_cache(size_t max_bytes);
    ~xkv_b_tile_cache() = default;

    xkv_b_tile_cache(const xkv_b_tile_cache &) = delete;
    xkv_b_tile_cache & operator=(const xkv_b_tile_cache &) = delete;

    // Exact helper computing total entry bytes for a given number of float elements
    static size_t compute_entry_bytes(size_t n_elements);

    // Get an existing B tile lease. Returns invalid lease on miss.
    xkv_b_tile_lease get(const b_tile_cache_key & key);

    // Put decoded tile data into cache and immediately return a lease to avoid local+cached duplicate copies.
    // If budget cannot accommodate (e.g. unleased entries insufficient to free space), returns an uncached lease.
    xkv_b_tile_lease put_and_lease(const b_tile_cache_key & key, std::vector<float> tile_data);

    // Copy [data, data + n_elements) into a cache-owned bounded record when budget allows
    // (may evict unleased entries). When budget refuses, returns a non-owning lease borrowing
    // the caller's buffer, which the caller must keep alive through use. The refuse path never
    // heap-allocates; the insert path allocates only the counted cache-owned record. Lets a
    // bounded reader decode straight into caller workspace and still warm the cache on miss.
    xkv_b_tile_lease put_and_lease_copy(const b_tile_cache_key & key, const float * data, size_t n_elements);

    void invalidate_segment(uint64_t segment_id, uint64_t segment_version);
    // Codec teardown only. content_epoch is accepted for wrapper ABI stability and
    // ignored: B tiles are immutable across content/binding epochs by key design.
    void invalidate_epoch(uint64_t content_epoch, uint64_t codec_epoch);
    void clear();

    size_t total_allocated_bytes() const;
    size_t active_cache_bytes() const;
    size_t leased_bytes() const;
    size_t max_bytes() const;
    uint64_t hit_count() const;
    uint64_t miss_count() const;
    uint64_t eviction_count() const;

private:
    friend class xkv_b_tile_lease;
    void on_lease_released(const std::shared_ptr<b_tile_record> & rec);
    bool evict_unleased_entry_locked();

    mutable std::mutex mtx_;
    size_t max_bytes_;
    size_t total_bytes_ = 0; // Total actual memory across both cache entries and leased out-of-cache records
    size_t active_cache_bytes_ = 0;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t evictions_ = 0;

    struct entry {
        b_tile_cache_key key;
        std::shared_ptr<b_tile_record> record;
    };

    std::list<entry> lru_list_;
    std::unordered_map<b_tile_cache_key, std::list<entry>::iterator, b_tile_cache_key_hash> map_;
};

// Hot flat row description (resident hot KV cache)
struct xkv_hot_row {
    uint32_t     row_index = 0;
    int64_t      storage_pos = 0;
    const float * k_ptr = nullptr; // pointer to head_dim_k floats
    const float * v_ptr = nullptr; // pointer to head_dim_v floats
    uint32_t     group_index = 0;  // DDVR group index
    bool         is_valid = true;  // causal or mask validity

    // Explicit per-query visibility membership (if empty, visible to all queries)
    std::vector<bool> query_visibility;
    // Optional per-query DDVR group index mapping, parallel to the batch query order
    // (position q, NOT query.query_index). If non-empty, its length must equal the
    // query count exactly; batch position q uses query_group_indices[q] and single
    // reads use query_group_indices[0]. Values for invisible queries are ignored
    // (UINT32_MAX recommended). If empty, group_index is used for all visible queries.
    std::vector<uint32_t> query_group_indices;

    bool is_visible_to_query(uint32_t query_idx) const {
        if (!is_valid) return false;
        if (query_visibility.empty()) return true;
        if (query_idx < query_visibility.size()) return query_visibility[query_idx];
        return false;
    }

    uint32_t get_group_for_query(uint32_t query_idx) const {
        if (!query_group_indices.empty() && query_idx < query_group_indices.size()) {
            return query_group_indices[query_idx];
        }
        return group_index;
    }
};

// Factored segment view for reading.
// Invariant: MUST own an immutable snapshot via xkv_reader_pin. Raw pointer or unpinned access is rejected.
struct xkv_segment_read_view {
    xkv_reader_pin pin;

    uint64_t segment_version_id = 0; // Segment version lineage for multi-segment row identity
    uint64_t storage_generation = 0; // Segment storage generation for multi-segment row identity
    uint32_t factor_group_index = UINT32_MAX; // Factor group index (UINT32_MAX if unspecified, derived from owning_layer)
    uint32_t owning_layer = UINT32_MAX;       // Owning attention layer (UINT32_MAX if unspecified)
    uint32_t kv_head = 0;

    // Derive exact B feature offset for K from immutable group metadata plus KV head
    uint32_t get_b_feature_offset_k(uint32_t head_dim_k, std::string * err = nullptr) const;
    // Derive exact B feature offset for V from immutable group metadata plus KV head
    uint32_t get_b_feature_offset_v(uint32_t head_dim_v, std::string * err = nullptr) const;

    // Resolve the unique owning factor group payload in the segment bundle
    const xkv_factor_group_payload * resolve_group(std::string * err = nullptr) const;

    // Selected rows within this segment (dense mode: all live rows)
    std::vector<uint32_t> selected_rows;
    // Storage position for each selected row (REQUIRED: exact length == selected_rows.size())
    std::vector<int64_t>  storage_positions;
    // Storage generation for each selected row (exact length == selected_rows.size())
    std::vector<uint64_t> row_generations;
    // DDVR group index for each selected row (exact length == selected_rows.size())
    std::vector<uint32_t> group_indices;
    // Per-query membership mask for this segment's selected rows (optional, or length == selected_rows.size())
    std::vector<bool>     membership_mask;
    // Per-query row visibility: outer size n_queries, inner size selected_rows.size().
    // [q][r] is true iff selected row r is visible to query q.
    std::vector<std::vector<bool>> query_row_visibility;
    // Per-query visible selected row indices (subset of selected_rows visible to query q)
    std::vector<std::vector<uint32_t>> query_selected_row_indices;

    const xkv_segment * get_segment() const {
        return pin.get();
    }

    uint64_t get_row_generation(size_t idx) const {
        if (idx < row_generations.size() && row_generations[idx] != 0) {
            return row_generations[idx];
        }
        return storage_generation;
    }

    segment_row_ref get_row_ref(size_t idx) const {
        segment_row_ref ref;
        const auto * seg = get_segment();
        ref.segment_id = seg ? seg->segment_id : 0;
        ref.segment_version = segment_version_id != 0 ? segment_version_id : (seg ? seg->segment_version : 0);
        ref.storage_generation = get_row_generation(idx);
        ref.row = (idx < selected_rows.size()) ? selected_rows[idx] : 0;
        return ref;
    }

    // Comprehensive validation of view structural integrity and segment metadata
    bool validate(std::string * err = nullptr) const;
};

// Query configuration for reading attention
struct xkv_query_input {
    uint32_t query_index = 0;

    // Q vectors: either single Q per query head, or per DDVR group
    // If q_groups is non-empty, q_groups[group_idx] has [n_q_heads * head_dim_k] floats
    // Otherwise q_vec has [n_q_heads * head_dim_k] floats for group 0
    std::vector<float> q_vec;
    std::vector<std::vector<float>> q_groups;

    uint32_t n_q_heads = 1; // GQA: number of query heads sharing this KV head
    uint32_t head_dim_k = 0;
    uint32_t head_dim_v = 0;

    float scale = 0.0f;         // Scale before softcap (0.0f -> 1.0f / sqrt(head_dim_k))
    float logit_softcap = 0.0f; // 0.0f means disabled

    // Sinks: per query head logit contribution (-INFINITY if none)
    std::vector<float> sink_logits;

    // Optional causal cutoff storage position (-1 means no cutoff)
    int64_t causal_limit_pos = -1;

    bool validate(std::string * err = nullptr) const;
};

// Configuration parameters for pre-warming or sizing the reusable reader workspace
struct xkv_reader_workspace_config {
    size_t   capacity_bytes = 16 * 1024 * 1024; // Hard memory limit (default 16 MiB)
    uint32_t max_queries = 64;
    uint32_t max_q_heads = 32;
    uint32_t max_head_dim_k = 256;
    uint32_t max_head_dim_v = 256;
    uint32_t max_tile_size = 128;
    uint32_t max_rank_k = 512;
    uint32_t max_rank_v = 512;
    uint32_t max_csr_entries = 8192;
    // Cap for the decode_tmp byte region. 0 = derive one max-rank FP32 row:
    // align64(max(max_rank_k, max_rank_v) * sizeof(float)), which covers the F16
    // (2x) and Q8_0 (~1.06x) row sizes for any padded width within the rank maxima.
    // Explicit values make the slice independently one-byte-short testable.
    uint32_t max_decode_tmp_bytes = 0;
};

// Paired row ref and DDVR group index for per-query membership and group resolution
struct xkv_csr_entry {
    segment_row_ref ref;
    uint32_t group_index = 0;

    bool operator<(const xkv_csr_entry & o) const {
        return ref < o.ref;
    }
    bool operator<(const segment_row_ref & r) const {
        return ref < r;
    }
};
inline bool operator<(const segment_row_ref & r, const xkv_csr_entry & e) {
    return r < e.ref;
}

// Scratch spans leased from the workspace for a single/batch read execution
struct xkv_reader_scratch {
    // Online softmax accumulators buffer: [total_heads * head_dim_v] floats
    float * states_accum = nullptr;
    size_t  states_accum_capacity = 0;

    // A and B factor tile decode buffers
    float * tile_a_k = nullptr;
    size_t  tile_a_k_capacity = 0;
    float * tile_a_v = nullptr;
    size_t  tile_a_v_capacity = 0;
    float * tile_b_k = nullptr;
    size_t  tile_b_k_capacity = 0;
    float * tile_b_v = nullptr;
    size_t  tile_b_v_capacity = 0;

    // Reconstructed K and V tile buffers
    float * tile_k = nullptr;
    size_t  tile_k_capacity = 0;
    float * tile_v = nullptr;
    size_t  tile_v_capacity = 0;

    // Phase scratch buffer for RoPE / transform
    float * transformed_k = nullptr;
    size_t  transformed_k_capacity = 0;

    // Integer index scratch buffers
    uint64_t * tile_row_indices = nullptr;
    size_t     tile_row_indices_capacity = 0;
    uint64_t * b_row_indices = nullptr;
    size_t     b_row_indices_capacity = 0;

    // CSR lookup scratch
    xkv_csr_entry *   csr_entries = nullptr;
    size_t            csr_entries_capacity = 0;
    segment_row_ref * csr_refs = nullptr;
    size_t            csr_refs_capacity = 0;

    // Byte scratch for factor-row decode tmp (F16/Q8_0). The codec queries the exact
    // need per descriptor via decode_rows_scratch_bytes; strict reads fail closed when
    // it exceeds decode_tmp_capacity. Null with 0 capacity when the carve has none.
    uint8_t * decode_tmp = nullptr;
    size_t    decode_tmp_capacity = 0;

    // Exact scratch bytes reserved
    size_t leased_bytes = 0;
    bool   valid = false;
};

// Sorted span for CSR row ref lookup without heap allocations
struct xkv_csr_lookup_span {
    const segment_row_ref * data = nullptr;
    size_t size = 0;

    bool contains(const segment_row_ref & ref) const {
        if (!data || size == 0) return false;
        return std::binary_search(data, data + size, ref);
    }
};

class xkv_reader_workspace;

// RAII lease on the reader workspace scratch
class xkv_workspace_scratch_lease {
public:
    xkv_workspace_scratch_lease() = default;
    xkv_workspace_scratch_lease(xkv_reader_workspace * ws, xkv_reader_scratch scratch)
        : ws_(ws), scratch_(scratch) {}
    ~xkv_workspace_scratch_lease() { release(); }

    xkv_workspace_scratch_lease(const xkv_workspace_scratch_lease &) = delete;
    xkv_workspace_scratch_lease & operator=(const xkv_workspace_scratch_lease &) = delete;

    xkv_workspace_scratch_lease(xkv_workspace_scratch_lease && o) noexcept
        : ws_(o.ws_), scratch_(o.scratch_) {
        o.ws_ = nullptr;
        o.scratch_.valid = false;
    }
    xkv_workspace_scratch_lease & operator=(xkv_workspace_scratch_lease && o) noexcept {
        if (this != &o) {
            release();
            ws_ = o.ws_;
            scratch_ = o.scratch_;
            o.ws_ = nullptr;
            o.scratch_.valid = false;
        }
        return *this;
    }

    void release();

    const xkv_reader_scratch & get() const { return scratch_; }
    bool valid() const { return scratch_.valid; }
    explicit operator bool() const { return valid(); }

private:
    xkv_reader_workspace * ws_ = nullptr;
    xkv_reader_scratch scratch_;
};

// Reusable bounded CPU reader workspace owned/configured by caller/store.
// Eliminates per-tile/per-head allocations during read calls.
// Guarantees capacities and pointers do not grow after warmup under fixed configured maxima.
class xkv_reader_workspace {
public:
    explicit xkv_reader_workspace(size_t capacity_bytes = 0, bool preallocate = true);
    explicit xkv_reader_workspace(const llama_cparams & cparams);
    explicit xkv_reader_workspace(const llama_xkv_cache_store & store);
    ~xkv_reader_workspace() = default;

    // Non-copyable and non-movable to guarantee address stability for active leases
    xkv_reader_workspace(const xkv_reader_workspace &) = delete;
    xkv_reader_workspace & operator=(const xkv_reader_workspace &) = delete;
    xkv_reader_workspace(xkv_reader_workspace &&) = delete;
    xkv_reader_workspace & operator=(xkv_reader_workspace &&) = delete;

    // Warm up the workspace scratch to configured maxima
    bool warmup(const xkv_reader_workspace_config & cfg, std::string * err = nullptr);
    bool warmup(size_t capacity_bytes = 0, std::string * err = nullptr);

    // Carve the workspace from caller-owned contiguous backing (e.g. a store arena
    // region supplied by the workspace owner) instead of allocating: zero heap
    // allocation, zero copy. Requires 64-byte-aligned backing with
    // backing_bytes >= layout total for cfg (see xkv_estimate_workspace_layout);
    // capacity becomes backing_bytes. The caller must keep backing alive longer than
    // the workspace and all its leases; the workspace never frees it. Replaces any
    // owned buffer; fails closed with prior state preserved while a lease is active.
    // clear() on an external workspace drops the carve without freeing the backing.
    // Lifetime rule (owner-enforced): a carve is valid only while the store snapshot
    // stamp equals the carve-time stamp; any epoch mismatch means the owner drops the
    // carve and re-carves. The reader never reads stale data through a dropped carve:
    // every read re-verifies its expected_stamp before and after, and scratch holds
    // no persistent semantic state across reads.
    // Accounting: the backing owner (arena) counts the carved region; allocated_bytes()
    // below counts only workspace-owned heap (side tables), so the region is never
    // double-counted. Usage within the carve is still tracked once via live/peak lease
    // bytes (diagnostic high-water marks, not additive bytes).
    bool warmup_external(void * backing, size_t backing_bytes,
                           const xkv_reader_workspace_config & cfg,
                           std::string * err = nullptr);
    bool is_external() const;

    // Preflight check against capacity
    bool preflight(size_t needed_bytes) const;

    // Acquire scratch view for an execution requiring needed_bytes
    bool acquire(size_t needed_bytes, xkv_reader_scratch & out_scratch);

    // RAII acquisition helper
    xkv_workspace_scratch_lease acquire_lease(size_t needed_bytes);

    // Release scratch reservation
    void release(const xkv_reader_scratch & scratch);

    // Metrics
    size_t capacity_bytes() const;
    size_t allocated_bytes() const;
    size_t peak_bytes() const;
    size_t live_bytes() const;
    // Live/peak track only scratch-lease bytes (the per-read estimate) counted once at acquire
    // and released at lease end. B-tile cache bytes (when a b_cache is configured) are owned
    // and counted by xkv_b_tile_cache separately and are never added here: no double counting.
    // Reported read peak_workspace_bytes is the checked upper-bound estimate (assumes B miss),
    // so it covers both cache hits and misses.
    bool is_warmed_up() const;
    bool has_active_lease() const;
    const xkv_reader_workspace_config & config() const;

    void reset_peak();
    void clear();

    // Stability verification inspectors (for testing that capacities/pointers do not grow)
    const void * scratch_ptr() const;
    const float * tile_k_buffer() const;
    const float * tile_v_buffer() const;
    const float * tile_a_k_buffer() const;
    const float * tile_a_v_buffer() const;
    const float * tile_b_k_buffer() const;
    const float * tile_b_v_buffer() const;
    const xkv_csr_entry * csr_entries_buffer() const;
    const segment_row_ref * csr_refs_buffer() const;

    // Pre-allocated states and CSR spans storage
    xkv_online_softmax_state * get_states_storage();
    size_t get_states_storage_capacity() const;
    xkv_csr_lookup_span * get_csr_spans_storage();
    size_t get_csr_spans_storage_capacity() const;

private:
    mutable std::mutex mtx_;
    size_t capacity_bytes_ = 0;
    size_t live_bytes_ = 0;
    size_t peak_bytes_ = 0;
    bool warmed_up_ = false;
    bool active_lease_ = false; // Strict exclusive single lease enforcement
    xkv_reader_workspace_config cfg_;

    // Primary pre-allocated contiguous scratch buffer (64-byte aligned)
    std::vector<uint8_t> buffer_;

    // Externally carved backing (non-owning). When set, it replaces buffer_ as the
    // scratch source; buffer_ stays empty so allocated_bytes() never double-counts
    // the carved region. Only mutated under mtx_.
    uint8_t * ext_base_ = nullptr;
    size_t    ext_size_ = 0;

    // Pre-allocated online softmax states and CSR spans arrays
    std::vector<xkv_online_softmax_state> state_storage_;
    std::vector<xkv_csr_lookup_span> csr_spans_storage_;

    // Pre-calculated fixed offsets for warmup layout
    size_t offset_states_accum_ = 0;
    size_t offset_tile_a_k_ = 0;
    size_t offset_tile_a_v_ = 0;
    size_t offset_tile_b_k_ = 0;
    size_t offset_tile_b_v_ = 0;
    size_t offset_tile_k_ = 0;
    size_t offset_tile_v_ = 0;
    size_t offset_transformed_k_ = 0;
    size_t offset_tile_row_indices_ = 0;
    size_t offset_b_row_indices_ = 0;
    size_t offset_csr_entries_ = 0;
    size_t offset_csr_refs_ = 0;
    size_t offset_decode_tmp_ = 0;
    size_t fixed_layout_total_bytes_ = 0;

    // Laid-out decode_tmp bytes (exact, set at warmup from cfg); published by acquire.
    size_t decode_tmp_bytes_ = 0;

    // Backing base for locked sections: the external carve when set, else buffer_.
    const uint8_t * backing_base_locked() const {
        if (ext_base_ != nullptr) return ext_base_;
        return buffer_.empty() ? nullptr : buffer_.data();
    }
    uint8_t * backing_base_locked() {
        if (ext_base_ != nullptr) return ext_base_;
        return buffer_.empty() ? nullptr : buffer_.data();
    }

    bool setup_layout_locked(const xkv_reader_workspace_config & cfg, std::string * err = nullptr);
};

// Reader execution config & resource limits
struct xkv_reader_config {
    uint32_t tile_size = 32;                            // Row tile size for streaming decode
    size_t   workspace_budget_bytes = 16 * 1024 * 1024; // 16 MiB hard limit
    std::shared_ptr<xkv_b_tile_cache> b_cache = nullptr;// Optional global B cache
    uint32_t device_id = 0;
    xkv_reader_workspace * workspace = nullptr;         // Optional reusable workspace
    // Standalone CPU oracle/debug only (default false). Semantics:
    // - workspace == nullptr: legacy oracle path, heap scratch allowed (existing behavior).
    //   Production MUST configure a workspace; unwired production reads stay legacy-heap.
    // - workspace != nullptr && !reference_allow_heap: strict bounded mode. Any slice that
    //   does not fit returns workspace_exhausted before producing output or mutating cache;
    //   fallback heap vectors are never allocated.
    // - workspace != nullptr && reference_allow_heap: oracle/debug mode, heap fallbacks
    //   permitted alongside a configured workspace for small standalone tests only.
    bool reference_allow_heap = false;
};

// Status of read operation
enum class xkv_read_status : uint8_t {
    success = 0,
    retry_stale_stamp = 1,
    invalid_argument = 2,
    workspace_exceeded = 3,
    codec_error = 4,
};

// Result of reading attention for a query
struct xkv_read_result {
    xkv_read_status status = xkv_read_status::success;
    std::vector<float> output; // [n_q_heads * head_dim_v]
    size_t peak_workspace_bytes = 0;
    bool b_cache_hit = false;
    std::string error_message;
};

// Batch read result
struct xkv_batch_read_result {
    xkv_read_status status = xkv_read_status::success;
    std::vector<xkv_read_result> per_query;
    size_t peak_workspace_bytes = 0;
    std::string error_message;
};

// Checked arithmetic helper: safe addition with overflow check
inline bool safe_add(size_t a, size_t b, size_t & out) {
    if (SIZE_MAX - a < b) return false;
    out = a + b;
    return true;
}

// Checked arithmetic helper: safe multiplication with overflow check
inline bool safe_mul(size_t a, size_t b, size_t & out) {
    if (a != 0 && b > SIZE_MAX / a) return false;
    out = a * b;
    return true;
}

inline bool safe_align(size_t n, size_t alignment, size_t & out) {
    if (alignment == 0) {
        out = n;
        return true;
    }
    size_t rem = n % alignment;
    if (rem == 0) {
        out = n;
        return true;
    }
    size_t add = alignment - rem;
    return safe_add(n, add, out);
}

inline uint32_t xkv_compute_padded_rank(uint32_t rank, ggml_type type = GGML_TYPE_TURBO4_0) {
    if (rank == 0) return 0;
    uint32_t align = 128; // Default alignment for Turbo types (group size 128)
    if (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) {
        align = 1;
    } else if (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0) {
        align = 32;
    }
    uint32_t rem = rank % align;
    return rem == 0 ? rank : (rank + (align - rem));
}

// Sanity cap for a single query dimension (heads or K/V width). Larger values are rejected
// as invalid_argument before any allocation; legitimate head dimensions are orders of
// magnitude smaller and workspace maxima are uint32-bounded regardless.
inline constexpr uint32_t xkv_reader_max_dim = 1u << 20;

// Exact checked byte size of the fixed workspace layout for cfg: the contiguous buffer
// footprint a successful warmup commits to (excludes the separately-counted state/span side
// tables). Pure checked arithmetic, never allocates. A capacity of exactly out_total_bytes
// warms successfully while out_total_bytes - 1 fails, so tests can prove byte-exactness.
bool xkv_estimate_workspace_layout(const xkv_reader_workspace_config & cfg, size_t & out_total_bytes, std::string * err = nullptr);

// Checked workspace calculation for a single query BEFORE allocating any memory
bool xkv_preflight_workspace(
    const xkv_query_input & query,
    const std::vector<xkv_segment_read_view> & segment_views,
    const xkv_reader_config & config,
    size_t & out_peak_workspace,
    std::string * err = nullptr
);

// Checked workspace calculation for a batch of queries BEFORE allocating any memory
bool xkv_preflight_batch_workspace(
    const std::vector<xkv_query_input> & queries,
    const std::vector<xkv_segment_read_view> & segment_views,
    const sr_batch_selection_result & batch_selection,
    const xkv_reader_config & config,
    size_t & out_peak_workspace,
    std::string * err = nullptr
);

// Single query attention reading
xkv_read_result xkv_read_attention(
    const xkv_query_input & query,
    const std::vector<xkv_hot_row> & hot_rows,
    const std::vector<xkv_segment_read_view> & segment_views,
    const phase_transform_fn & phase_tx,
    const xkv_snapshot_stamp & expected_stamp,
    llama_xkv_cache_store * store,
    const xkv_reader_config & config = {}
);

xkv_read_result xkv_read_attention(
    xkv_reader_workspace & ws,
    const xkv_query_input & query,
    const std::vector<xkv_hot_row> & hot_rows,
    const std::vector<xkv_segment_read_view> & segment_views,
    const phase_transform_fn & phase_tx,
    const xkv_snapshot_stamp & expected_stamp,
    llama_xkv_cache_store * store,
    const xkv_reader_config & config = {}
);

// Batch query attention reading with single union decode & CSR isolation
xkv_batch_read_result xkv_read_attention_batch(
    const std::vector<xkv_query_input> & queries,
    const std::vector<xkv_hot_row> & hot_rows,
    const std::vector<xkv_segment_read_view> & segment_views,
    const sr_batch_selection_result & batch_selection,
    const phase_transform_fn & phase_tx,
    const xkv_snapshot_stamp & expected_stamp,
    llama_xkv_cache_store * store,
    const xkv_reader_config & config = {}
);

xkv_batch_read_result xkv_read_attention_batch(
    xkv_reader_workspace & ws,
    const std::vector<xkv_query_input> & queries,
    const std::vector<xkv_hot_row> & hot_rows,
    const std::vector<xkv_segment_read_view> & segment_views,
    const sr_batch_selection_result & batch_selection,
    const phase_transform_fn & phase_tx,
    const xkv_snapshot_stamp & expected_stamp,
    llama_xkv_cache_store * store,
    const xkv_reader_config & config = {}
);

// Dense reference attention oracle for testing and validation
std::vector<float> xkv_dense_attention_reference(
    const xkv_query_input & query,
    const std::vector<xkv_hot_row> & hot_rows,
    const std::vector<std::vector<float>> & cold_keys,   // [head_dim_k] per cold row
    const std::vector<std::vector<float>> & cold_values, // [head_dim_v] per cold row
    const std::vector<uint32_t> & cold_group_indices,
    const std::vector<bool> & cold_mask
);

} // namespace llama_xkv
