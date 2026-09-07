#pragma once

#include "ggml.h"
#include "llama.h"
#include "llama-cparams.h"
#include "llama-xkv-codec.h"
#include "llama-xkv-factor.h"
#include "ggml-xkv.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct kv_pack_plan;

namespace llama_xkv {

enum class xkv_location_kind : uint8_t {
    hot = 0,
    flat_quantized = 1,
    factored = 2,
};

// Lifecycle state discipline for payloads and candidates:
// HOT_WRITING -> HOT_COMMITTED -> SEAL_CANDIDATE -> FLAT_TQ / FACTORED
enum class xkv_state : uint8_t {
    hot_writing    = 0,
    hot_committed  = 1,
    seal_candidate = 2,
    flat_tq        = 3,
    factored       = 4,
};

const char * xkv_state_to_str(xkv_state state);

// Explicit skip reasons for failed / skipped sealing candidates
enum class xkv_skip_reason : uint8_t {
    none = 0,
    not_committed = 1,
    unsupported_config = 2,
    preflight_oom = 3,
    factorization_failed = 4,
    codec_error = 5,
    error_threshold_exceeded = 6,
    no_saving = 7,
    aborted = 8,
    landmark_required = 9,
    store_capacity_exceeded = 10,
};

const char * xkv_skip_reason_to_str(xkv_skip_reason reason);

// Globally unambiguous row reference within a specific segment snapshot version
struct segment_row_ref {
    uint64_t segment_id = 0;
    uint64_t segment_version = 0;
    uint64_t storage_generation = 0;
    uint32_t row = 0;

    bool operator==(const segment_row_ref & o) const {
        return segment_id == o.segment_id &&
               segment_version == o.segment_version &&
               storage_generation == o.storage_generation &&
               row == o.row;
    }
    bool operator!=(const segment_row_ref & o) const {
        return !(*this == o);
    }
    bool operator<(const segment_row_ref & o) const {
        if (segment_id != o.segment_id) return segment_id < o.segment_id;
        if (segment_version != o.segment_version) return segment_version < o.segment_version;
        if (storage_generation != o.storage_generation) return storage_generation < o.storage_generation;
        return row < o.row;
    }
};

struct segment_row_ref_hash {
    size_t operator()(const segment_row_ref & r) const noexcept {
        uint64_t h = 0xcbf29ce484222325ULL;
        h ^= r.segment_id; h *= 0x100000001b3ULL;
        h ^= r.segment_version; h *= 0x100000001b3ULL;
        h ^= r.storage_generation; h *= 0x100000001b3ULL;
        h ^= r.row; h *= 0x100000001b3ULL;
        return static_cast<size_t>(h);
    }
};

// Location of a payload in physical storage.
// Invariant: MUST NOT duplicate semantic position, seq refs, visibility, or RERoT metadata.
struct xkv_location {
    xkv_location_kind kind = xkv_location_kind::hot;
    uint64_t segment_id = 0;
    uint64_t segment_version = 0;
    uint32_t row = 0;
    uint64_t storage_generation = 0;
    xkv_state state = xkv_state::hot_writing;
    uint64_t seal_tx_nonce = 0; // Per-location seal transaction nonce/id

    bool operator==(const xkv_location & o) const {
        return kind == o.kind && segment_id == o.segment_id && segment_version == o.segment_version &&
               row == o.row && storage_generation == o.storage_generation &&
               state == o.state && seal_tx_nonce == o.seal_tx_nonce;
    }
    bool operator!=(const xkv_location & o) const {
        return !(*this == o);
    }
};

// Combined snapshot stamp
struct xkv_snapshot_stamp {
    llama_rerot_view_stamp view = {0, 0, 0}; // Existing RERoT view stamp
    uint64_t live_epoch = 0;                 // Permanent retention set changes
    uint64_t content_epoch = 0;              // Factor refit / re-quantization changes
    uint64_t codec_epoch = 0;                // Codec / scale / rotation changes
    uint64_t binding_epoch = 0;              // Physical-only moves / rebinding

    bool operator==(const xkv_snapshot_stamp & o) const {
        return view.topology_epoch == o.view.topology_epoch &&
               view.publish_epoch == o.view.publish_epoch &&
               view.layout_epoch == o.view.layout_epoch &&
               live_epoch == o.live_epoch &&
               content_epoch == o.content_epoch &&
               codec_epoch == o.codec_epoch &&
               binding_epoch == o.binding_epoch;
    }
    bool operator!=(const xkv_snapshot_stamp & o) const {
        return !(*this == o);
    }
};

// Removal callback result context passed back to the coordinator.
// If the removal callback used store remove_payloads (which advanced
// live/binding/content), it sets has_store_stamp = true and store_stamp to
// the resulting store stamp. If no removal ran, has_store_stamp stays false
// and the provider applies the reserved content advance instead (never both).
struct xkv_removal_feedback {
    bool has_store_stamp = false;
    xkv_snapshot_stamp store_stamp = {};
};

// Opaque reservation token for two-phase checked epoch publication.
// Moved here from llama-xkv-transaction.h: the store is the authoritative
// owner (transaction.h includes this header and must not redeclare these).
struct xkv_stamp_reservation {
    uint64_t token = 0;
    xkv_snapshot_stamp base_stamp = {};
    xkv_snapshot_stamp reserved_stamp = {};
    bool valid = false;
};

// Authoritative stamp provider interface, implemented directly by
// llama_xkv_cache_store below. The transaction coordinator binds the store
// (no raw adapter lifetime/fake) and delegates all snapshot queries and
// epoch publications here.
struct xkv_stamp_provider {
    virtual ~xkv_stamp_provider() = default;

    // Read current authoritative stamp.
    virtual xkv_snapshot_stamp current_stamp() const = 0;

    // Phase 1: Preflight reservation for content epoch advance before removal
    // callback runs. Reserves exclusively: unrelated store mutators refuse
    // while held. Fails (false + err, 'overflow' on epoch exhaustion) when
    // busy or unbumpable; zero mutation on failure.
    virtual bool preflight_content_advance(xkv_stamp_reservation * out_res, std::string * err) = 0;

    // Phase 2: Infallible commit of the reservation token after removal
    // callback succeeds. If feedback carries the store stamp from an actual
    // token-carried batch removal, apply validates it (exact match incl.
    // view/codec, so feedback can never mask a mismatch) and clears without
    // a second bump. Otherwise applies the reserved content advance once.
    // MUST NOT allocate, MUST NOT fail.
    virtual xkv_snapshot_stamp apply_content_advance(const xkv_stamp_reservation & res,
                                                    const xkv_removal_feedback & feedback) noexcept = 0;

    // Abort a preflight reservation without publishing. Legal only when the
    // staged mutation did not commit; otherwise a no-op.
    virtual void abort_content_advance(const xkv_stamp_reservation & res) noexcept = 0;

    // Checked advance called on final_fence to install stable RERoT view.
    virtual bool install_stable_view(const llama_rerot_view_stamp & stable_view,
                                     xkv_snapshot_stamp * out_new_stamp,
                                     std::string * err) = 0;

    // Checked advance called on context shift: updates view + advances binding.
    virtual bool advance_context_shift(const llama_rerot_view_stamp & shifted_view,
                                       xkv_snapshot_stamp * out_new_stamp,
                                       std::string * err) = 0;
};

// Immutable per-chunk landmark metadata, parallel to landmark chunks.
// Explicit per-chunk landmark metadata (row interval, error bound, and exact
// final-factor/phase source fingerprint).
// Stride is configurable (--xkv-chunk-tokens), arbitrary widths >= 1, strictly
// advancing with final end == n_rows. Error bounds must be finite and >= 0.0f.
struct xkv_landmark_chunk {
    uint32_t row_begin = 0;
    uint32_t row_count = 0;
    float error_bound = 0.0f;
    uint64_t source_fingerprint = 0;

    bool operator==(const xkv_landmark_chunk & o) const {
        return row_begin == o.row_begin &&
               row_count == o.row_count &&
               error_bound == o.error_bound &&
               source_fingerprint == o.source_fingerprint;
    }
    bool operator!=(const xkv_landmark_chunk & o) const {
        return !(*this == o);
    }
};

uint64_t compute_landmark_table_fingerprint(const std::vector<xkv_landmark_chunk> & chunks);

// Payload of a single factor group inside an xkv_segment bundle
struct xkv_factor_group_payload {
    uint32_t group_index = 0;
    std::vector<uint32_t> owning_layers;

    uint32_t rank_k = 16;
    uint32_t rank_v = 16;

    // Feature offsets and dimensions within B_K and B_V for each layer in owning_layers
    std::vector<uint32_t> layer_feature_offsets_k;
    std::vector<uint32_t> layer_feature_dims_k;
    std::vector<uint32_t> layer_feature_offsets_v;
    std::vector<uint32_t> layer_feature_dims_v;

    uint32_t total_dim_k = 0;
    uint32_t total_dim_v = 0;

    // Four independent factor code streams
    // A is token-major; B is feature_major_transposed and immutable shared handle
    encoded_matrix a_k;
    std::shared_ptr<const encoded_matrix> b_k;
    encoded_matrix a_v;
    std::shared_ptr<const encoded_matrix> b_v;

    // Quantized landmark code stream for this factor group (empty if not configured)
    encoded_matrix landmark;

    // Explicit per-chunk landmark metadata. Size equals landmark row count.
    std::vector<xkv_landmark_chunk> landmark_chunks;
    uint64_t landmark_table_fingerprint = 0;

    // Fingerprints
    uint64_t descriptor_fingerprint = 0;
    uint64_t config_fingerprint = 0;

    // Immutable original-flat baseline: exact runtime-measured bytes this
    // group covers (never derived from compressed streams or the last seal).
    // row = Σ per-layer hot row bytes; total = row × rows at seal time.
    // Survivor packs recompute live covered bytes as row × n_live_rows.
    uint64_t baseline_original_row_bytes = 0;
    uint64_t baseline_original_bytes = 0;

    // Byte counters
    size_t bytes_a_k = 0;
    size_t bytes_b_k = 0;
    size_t bytes_a_v = 0;
    size_t bytes_b_v = 0;
    size_t bytes_landmark = 0;
    size_t bytes_metadata = 0;
    size_t total_allocated_bytes = 0;

    void update_byte_counters();
    void refresh_descriptor_fingerprint();

    // Helper setters for B
    void set_b_k(encoded_matrix em) {
        b_k = std::make_shared<const encoded_matrix>(std::move(em));
    }
    void set_b_v(encoded_matrix em) {
        b_v = std::make_shared<const encoded_matrix>(std::move(em));
    }

    // Lookup helper: find index of layer inside owning_layers
    int32_t find_owning_layer_index(uint32_t layer) const {
        for (size_t i = 0; i < owning_layers.size(); ++i) {
            if (owning_layers[i] == layer) return (int32_t)i;
        }
        return -1;
    }
};

// Forward declarations
struct xkv_backend_store_reservation;
class llama_xkv_cache_store;

// Move-only RAII store capacity reservation handle
class xkv_capacity_reservation {
public:
    xkv_capacity_reservation() noexcept = default;
    ~xkv_capacity_reservation();

    xkv_capacity_reservation(const xkv_capacity_reservation &) = delete;
    xkv_capacity_reservation & operator=(const xkv_capacity_reservation &) = delete;

    xkv_capacity_reservation(xkv_capacity_reservation && o) noexcept;
    xkv_capacity_reservation & operator=(xkv_capacity_reservation && o) noexcept;

    bool valid() const noexcept { return store_ != nullptr && token_ != 0; }
    explicit operator bool() const noexcept { return valid(); }

    uint64_t token() const noexcept { return token_; }
    size_t reserved_bytes() const noexcept { return reserved_bytes_; }

    const std::vector<uint64_t> & expected_removal_payload_ids() const noexcept { return expected_removal_payload_ids_; }
    const std::vector<uint64_t> & expected_removal_generations() const noexcept { return expected_removal_generations_; }

    xkv_backend_store_reservation backend_reservation() const noexcept;

    void release() noexcept;

private:
    friend class llama_xkv_cache_store;
    xkv_capacity_reservation(llama_xkv_cache_store * store, uint64_t token, size_t reserved_bytes,
                             std::vector<uint64_t> expected_pids = {},
                             std::vector<uint64_t> expected_gens = {}) noexcept
        : store_(store), token_(token), reserved_bytes_(reserved_bytes),
          expected_removal_payload_ids_(std::move(expected_pids)),
          expected_removal_generations_(std::move(expected_gens)) {}

    llama_xkv_cache_store * store_ = nullptr;
    uint64_t token_ = 0;
    size_t reserved_bytes_ = 0;
    std::vector<uint64_t> expected_removal_payload_ids_;
    std::vector<uint64_t> expected_removal_generations_;
};

// Move-only RAII transient device staging reservation handle (workspace arena / device peak limit)
class xkv_device_staging_reservation {
public:
    xkv_device_staging_reservation() noexcept = default;
    ~xkv_device_staging_reservation();

    xkv_device_staging_reservation(const xkv_device_staging_reservation &) = delete;
    xkv_device_staging_reservation & operator=(const xkv_device_staging_reservation &) = delete;

    xkv_device_staging_reservation(xkv_device_staging_reservation && o) noexcept;
    xkv_device_staging_reservation & operator=(xkv_device_staging_reservation && o) noexcept;

    bool valid() const noexcept { return store_ != nullptr && reserved_bytes_ > 0; }
    explicit operator bool() const noexcept { return valid(); }
    size_t reserved_bytes() const noexcept { return reserved_bytes_; }
    void release() noexcept;

private:
    friend class llama_xkv_cache_store;
    xkv_device_staging_reservation(llama_xkv_cache_store * store, size_t reserved_bytes) noexcept
        : store_(store), reserved_bytes_(reserved_bytes) {}

    llama_xkv_cache_store * store_ = nullptr;
    size_t reserved_bytes_ = 0;
};

// Exact deduplicated accounting breakdown
// Residency seam (see ggml/include/ggml-xkv.h): counters below measure host
// encoded bytes owned by the store. Production Vulkan device-owned upload
// (one exact-bytes upload per immutable segment, then host release, tensor
// reuse across graphs) lives outside the store in runtime/graph; the store
// MUST NOT fake device-resident savings here. API preserved until the upload
// contract lands.
struct xkv_accounting {
    size_t live_payload_bytes = 0;      // Deduplicated bytes of active live payloads (A live rows + unique B + live landmarks + logical metadata)
    size_t allocated_bytes = 0;         // Total actual memory currently allocated across published and pinned-retired segments (deduplicating shared B)
    size_t baseline_factored_bytes = 0; // Live covered ORIGINAL bytes (Σ group row-baseline × live rows over published segments; never compressed bytes)
    size_t reserved_bytes = 0;          // Actual reserved memory (allocated segments + arena reserved scratch capacity)
    size_t workspace_budget_bytes = 0;  // Configured unallocated workspace limit/budget (cparams.xkv_workspace_mib)
    size_t hot_bytes = 0;               // Hot KV cache bytes
    size_t factored_bytes = 0;          // Factored segments bytes (A + unique B + landmark + logical metadata)
    size_t active_segments = 0;         // Number of active published segments
    size_t total_payloads = 0;          // Number of unique tracked payloads
    size_t pinned_segments = 0;         // Number of currently pinned segments (published or retired)

    // Deduplicated shared B matrix metrics
    size_t unique_b_matrices = 0;       // Number of distinct B matrices allocated across all segment versions and groups
    size_t shared_b_bytes = 0;          // Deduplicated byte sum of all distinct B matrices

    // Bounded host workspace arena actual live/reserved/peak counters distinct from budgets
    size_t arena_live_bytes = 0;        // Currently reserved live workspace bytes
    size_t arena_reserved_bytes = 0;    // Allocated arena scratch capacity
    size_t arena_peak_bytes = 0;        // High-water mark of live arena bytes
    size_t arena_capacity_bytes = 0;    // Configured hard arena capacity limit

    // §16 detailed metrics (measured host bytes, deduplicated):
    size_t factor_metadata_bytes = 0;   // Metadata across published + pinned-retired segments
    size_t factor_padding_bytes = 0;    // Row padding overhead (padded cols - logical cols)
    size_t factor_live_bytes = 0;       // Live-row factor stream bytes only (A + unique B)
    size_t factor_fp16_equivalent_bytes = 0; // FP16 byte size of identical factor elements
    size_t landmark_payload_bytes = 0;  // Encoded landmark stream bytes
    size_t landmark_metadata_bytes = 0; // Landmark chunk descriptor table bytes
    size_t landmark_exception_bytes = 0;// Unchunked outlier exception bytes (0 for native)
    size_t index_bytes = 0;             // Common row payload IDs + live bits metadata
    size_t codec_shared_bytes = 0;      // Global shared codec table bytes (WHT tables)
    size_t snapshot_pinned_bytes = 0;   // Retained bytes of pinned-retired segment versions
    size_t aliased_payloads = 0;        // Count of payloads mapped via layer aliases
    size_t host_peak_bytes = 0;         // High-water mark of host-resident allocated bytes
    size_t device_peak_bytes = 0;       // High-water mark of device-resident allocated bytes
    size_t dedup_scratch_bytes = 0;     // Store-owned scratch vector capacity bytes for zero-allocation accounting

    bool operator==(const xkv_accounting & o) const {
        return live_payload_bytes == o.live_payload_bytes &&
               allocated_bytes == o.allocated_bytes &&
               baseline_factored_bytes == o.baseline_factored_bytes &&
               reserved_bytes == o.reserved_bytes &&
               workspace_budget_bytes == o.workspace_budget_bytes &&
               hot_bytes == o.hot_bytes &&
               factored_bytes == o.factored_bytes &&
               active_segments == o.active_segments &&
               total_payloads == o.total_payloads &&
               pinned_segments == o.pinned_segments &&
               unique_b_matrices == o.unique_b_matrices &&
               shared_b_bytes == o.shared_b_bytes &&
               dedup_scratch_bytes == o.dedup_scratch_bytes &&
               arena_live_bytes == o.arena_live_bytes &&
               arena_reserved_bytes == o.arena_reserved_bytes &&
               arena_peak_bytes == o.arena_peak_bytes &&
               arena_capacity_bytes == o.arena_capacity_bytes &&
               factor_metadata_bytes == o.factor_metadata_bytes &&
               factor_padding_bytes == o.factor_padding_bytes &&
               factor_live_bytes == o.factor_live_bytes &&
               factor_fp16_equivalent_bytes == o.factor_fp16_equivalent_bytes &&
               landmark_payload_bytes == o.landmark_payload_bytes &&
               landmark_metadata_bytes == o.landmark_metadata_bytes &&
               landmark_exception_bytes == o.landmark_exception_bytes &&
               index_bytes == o.index_bytes &&
               codec_shared_bytes == o.codec_shared_bytes &&
               snapshot_pinned_bytes == o.snapshot_pinned_bytes &&
               aliased_payloads == o.aliased_payloads &&
               host_peak_bytes == o.host_peak_bytes &&
               device_peak_bytes == o.device_peak_bytes;
    }
    bool operator!=(const xkv_accounting & o) const {
        return !(*this == o);
    }
};

// Deeply immutable published segment bundle snapshot.
// Invariant: One xkv_segment is one time-row bundle with vector<xkv_factor_group_payload> groups.
// The payload locator remains exactly one (segment_id, segment_version, row, generation).
// Once published, an xkv_segment snapshot is NEVER mutated in place.
struct xkv_segment {
    uint64_t segment_id = 0;
    uint64_t segment_version = 1;

    llama_xkv_storage_profile profile = LLAMA_XKV_STORAGE_PROFILE_REFERENCE;
    llama_xkv_source source = LLAMA_XKV_SOURCE_DECODED_HOT;

    // Immutable persisted residency/capability metadata. DEVICE_OWNED requires
    // a complete verified backend bundle and permits cleared host bytes;
    // REFERENCE_HOST requires exact host bytes and no bundle. Set from the
    // effective backend at seal time; COW versions preserve it.
    ggml_xkv_residency residency = GGML_XKV_RES_REFERENCE_HOST;

    // Immutable shared backend bundle (device streams), BackendResidency-owned.
    // Forward-declared to avoid an include cycle; the store only shares
    // ownership, never builds or verifies. Null for REFERENCE/host execution.
    // COW versions reset to null (compacted bytes differ; re-adopt after
    // re-upload). Retired versions keep theirs until reclaimed.
    std::shared_ptr<const class xkv_backend_batch_result> backend_bundle;

    // Fingerprints
    uint64_t profile_fingerprint = 0;
    uint64_t source_fingerprint = 0;
    uint64_t layer_group_map_fingerprint = 0; // Deterministic fingerprint over configured factor group topology
    uint64_t descriptor_fingerprint = 0;      // Combined bundle descriptor fingerprint

    // Immutable original-flat baseline total (Σ group totals at seal time).
    uint64_t baseline_original_bytes = 0;

    // Factor groups in this time-row bundle
    std::vector<xkv_factor_group_payload> groups;

    // Common row mapping: row index in bundle -> stable payload_id
    std::vector<uint64_t> row_payload_ids;

    // Common live-row bitmask within bundle. Deeply immutable once published.
    std::vector<bool> live_rows;

    uint32_t n_rows = 0;
    uint32_t n_live_rows = 0;

    // Reader pin counter. Protects segment buffers while readers are active.
    mutable std::atomic<uint32_t> pin_count{0};

    // Bundle total allocated bytes (sum of group allocations + common row metadata)
    size_t total_allocated_bytes = 0;
    size_t bytes_metadata_logical = 0;

    void update_byte_counters();

    const xkv_factor_group_payload * find_group(uint32_t group_index) const {
        for (const auto & g : groups) {
            if (g.group_index == group_index) return &g;
        }
        return nullptr;
    }

    const xkv_factor_group_payload * find_group_for_layer(uint32_t layer) const {
        for (const auto & g : groups) {
            if (g.find_owning_layer_index(layer) >= 0) return &g;
        }
        return nullptr;
    }
};

// Bounded host workspace arena with RAII reservation and peak tracking.
// Shared backing state below: leases hold a shared_ptr to it, so a lease
// outliving its owning store/arena (async graph) never dangles.
struct xkv_arena_state {
    static constexpr size_t kMaxBlocks = 1024;

    ~xkv_arena_state() {
        // Final shared owner frees: a lease outliving its arena keeps the
        // backing alive until now. Plain delete of null is a no-op.
        ::operator delete(base, std::align_val_t(64));
    }

    std::mutex mtx;
    // Single preallocated 64B-aligned backing buffer (allocated once at init /
    // set_capacity, never per op). Null when capacity is 0.
    void * base = nullptr;
    size_t capacity = 0;   // Committed buffer bytes (multiple of 64).
    size_t live = 0;       // Currently leased bytes (rounded).
    size_t live_count = 0; // Currently outstanding leases.
    size_t peak = 0;       // High-water leased bytes.

    struct free_node {
        size_t offset = 0;
        size_t size = 0;
        int32_t next = -1;
        bool used = false;
    };
    // Preallocated free-list metadata: fixed pool, intrusive indices.
    free_node blocks[kMaxBlocks];
    int32_t free_head = -1;
    int32_t free_nodes = -1; // Stack of spare metadata nodes.
};

class xkv_workspace_arena;

class xkv_arena_lease {
public:
    xkv_arena_lease() = default;
    ~xkv_arena_lease();

    xkv_arena_lease(const xkv_arena_lease &) = delete;
    xkv_arena_lease & operator=(const xkv_arena_lease &) = delete;

    xkv_arena_lease(xkv_arena_lease && o) noexcept;
    xkv_arena_lease & operator=(xkv_arena_lease && o) noexcept;

    void release();

    // Backing block: 64B-aligned start, `size()` rounded bytes, non-overlapping
    // with any other live lease. Null unless the lease is valid. Safe to call
    // even after the owning arena is destroyed (shared state).
    void * data() const;
    const void * const_data() const { return data(); }
    template <typename T>
    T * as() const { return static_cast<T *>(data()); }
    template <typename T>
    const T * as_const() const { return static_cast<const T *>(data()); }

    size_t size() const { return leased_size; }
    bool valid() const { return state != nullptr && leased_size > 0; }
    explicit operator bool() const { return valid(); }

private:
    friend class xkv_workspace_arena;
    xkv_arena_lease(std::shared_ptr<xkv_arena_state> state, size_t offset, size_t size, int32_t spare);

    // Shared ownership: the backing buffer outlives any destroyed arena while
    // leases are outstanding. `spare` is a metadata node reserved for this
    // lease's eventual release, so release can never fail for lack of metadata.
    std::shared_ptr<xkv_arena_state> state;
    size_t leased_offset = 0;
    size_t leased_size = 0;
    int32_t spare_node = -1;
};

class xkv_workspace_arena {
public:
    static constexpr size_t kAlignment = 64;
    // Max concurrent leases: one metadata node is reserved per live lease, so
    // release can never fail. Acquire refuses past this bound explicitly.
    static constexpr size_t kMaxBlocks = xkv_arena_state::kMaxBlocks;

    explicit xkv_workspace_arena(size_t capacity_bytes = 0);
    ~xkv_workspace_arena();

    xkv_workspace_arena(const xkv_workspace_arena &) = delete;
    xkv_workspace_arena & operator=(const xkv_workspace_arena &) = delete;

    bool preflight(size_t needed_bytes) const;
    xkv_arena_lease acquire(size_t size_bytes);

    size_t get_live_bytes() const;
    size_t get_reserved_bytes() const;
    size_t get_peak_bytes() const;
    size_t get_capacity_bytes() const;
    // Init-time realloc. Fails (false, no mutation) while leases are outstanding.
    bool set_capacity_bytes(size_t bytes);
    void reset_peak();

    // Introspection for tests/peers (null when capacity is 0).
    const void * base_data() const;

private:
    friend class xkv_arena_lease;
    // Shared state outlives the arena while leases are outstanding.
    std::shared_ptr<xkv_arena_state> state = std::make_shared<xkv_arena_state>();
};

namespace xkv_arena_detail {
// Checked round-up to the arena alignment; false on overflow.
inline bool checked_round_up(size_t n, size_t alignment, size_t & out) {
    if (n > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        return false;
    }
    const size_t add = n + (alignment - 1);
    out = add & ~(alignment - 1);
    return true;
}
} // namespace xkv_arena_detail

namespace xkv_store_detail {
bool streams_equal(const encoded_matrix & a, const encoded_matrix & b);
bool shared_stream_equal(const std::shared_ptr<const encoded_matrix> & a,
                         const std::shared_ptr<const encoded_matrix> & b);
bool backend_handles_equal(const std::shared_ptr<const xkv_backend_batch_result> & a,
                           const std::shared_ptr<const xkv_backend_batch_result> & b);
bool segments_exact_equal(const xkv_segment & a, const xkv_segment & b);
} // namespace xkv_store_detail

// RAII handle for pinning a segment bundle during reading.
class xkv_reader_pin {
public:
    xkv_reader_pin() = default;
    explicit xkv_reader_pin(std::shared_ptr<const xkv_segment> seg);
    ~xkv_reader_pin();

    xkv_reader_pin(const xkv_reader_pin &) = delete;
    xkv_reader_pin & operator=(const xkv_reader_pin &) = delete;

    xkv_reader_pin(xkv_reader_pin && o) noexcept;
    xkv_reader_pin & operator=(xkv_reader_pin && o) noexcept;

    void release();

    const xkv_segment * get() const { return segment.get(); }
    const xkv_segment & operator*() const { return *segment; }
    const xkv_segment * operator->() const { return segment.get(); }
    std::shared_ptr<const xkv_segment> handle() const { return segment; }
    explicit operator bool() const { return segment != nullptr; }

private:
    std::shared_ptr<const xkv_segment> segment;
};

// Snapshot record of a hot payload binding for safe storage-only enumeration
struct xkv_hot_payload_binding {
    uint64_t payload_id = 0;
    uint32_t hot_slot_row = 0;
    uint64_t storage_generation = 0;
    xkv_state state = xkv_state::hot_writing;

    bool operator==(const xkv_hot_payload_binding & o) const {
        return payload_id == o.payload_id &&
               hot_slot_row == o.hot_slot_row &&
               storage_generation == o.storage_generation &&
               state == o.state;
    }
    bool operator<(const xkv_hot_payload_binding & o) const {
        return payload_id < o.payload_id;
    }
};

// Snapshot listing of hot payload bindings under one snapshot stamp
struct xkv_hot_bindings_snapshot {
    xkv_snapshot_stamp stamp = {};
    std::vector<xkv_hot_payload_binding> bindings; // Deterministically sorted by payload_id
};

// Snapshot item of a published segment location with pin for a queried payload
struct xkv_payload_segment_view {
    uint64_t payload_id = 0;
    xkv_location location = {};
    xkv_reader_pin pin = {};
};

struct xkv_payload_segments_snapshot {
    xkv_snapshot_stamp stamp = {};
    std::vector<xkv_payload_segment_view> views; // Deterministically sorted by payload_id
};

// Post-factor landmark generator callback type per factor group.
// Consumes the FINAL ENCODED streams (never decoded mirrors: the store must not
// materialize full A/B dense matrices) plus exact per-row storage positions
// for phase-aware chunking and a caller arena carve for tile buffers. The
// implementation streams tiles (bounded memory) and fails closed when the
// span is insufficient. No V inputs: landmarks summarize reconstructed K.
using xkv_group_landmark_factory_fn = std::function<bool(
    uint32_t group_index,
    const encoded_matrix & enc_a_k,
    const encoded_matrix & enc_b_k,
    const int64_t * row_positions,
    uint64_t n_rows,
    factor_workspace_span scratch,
    encoded_matrix & out_landmark,
    std::vector<xkv_landmark_chunk> & out_chunks,
    std::string * err
)>;

// Semantic landmark rebuild callback for LANDMARKS-profile COW.
// The store NEVER fabricates landmark bytes by copying the first N chunks after
// arbitrary survivor deletion: chunk summaries are position-sensitive and row
// deletion shifts chunk boundaries, and duplicating the whole layout just to
// change the live mask breaks the byte budget. With a callback, the store
// compacts survivor A rows and the callback atomically produces valid landmark
// streams for the compacted layout before publication; return false to refuse
// (no partial state). WITHOUT a callback, any LANDMARKS removal/pack/
// displacement refuses with zero mutation and the old version stays valid and
// published. (Shared-handle tombstones await the BackendResidency handle design.)
using xkv_landmark_rebuild_fn = std::function<bool(
    const xkv_segment & old_seg,
    const std::vector<uint32_t> & surviving_rows,
    xkv_segment & new_seg,
    std::string * err
)>;

// Input data for sealing a single factor group inside a bundle
struct xkv_factor_group_input {
    uint32_t group_index = 0;
    std::vector<uint32_t> owning_layers;

    // Explicit per-group ranks. REQUIRED nonzero: seal rejects rank_k == 0 or
    // rank_v == 0 atomically (no fallback to params.rank_k/rank_v). The bundle
    // params ranks remain full-group maxima/defaults, never forced tail values.
    uint32_t rank_k = 0;
    uint32_t rank_v = 0;

    std::vector<uint32_t> layer_feature_offsets_k;
    std::vector<uint32_t> layer_feature_dims_k;
    std::vector<uint32_t> layer_feature_offsets_v;
    std::vector<uint32_t> layer_feature_dims_v;

    uint32_t total_dim_k = 0;
    uint32_t total_dim_v = 0;

    const float * canonical_k_data = nullptr;
    uint64_t k_rows = 0;
    uint64_t k_cols = 0;

    const float * canonical_v_data = nullptr;
    uint64_t v_rows = 0;
    uint64_t v_cols = 0;
    // Exact per-row storage positions in signed 64-bit phase space (row order
    // == payload order; negative/large admitted), populated by the runtime
    // from live cells. REQUIRED and size-validated (== n_rows); distinct from
    // hot physical rows. Never nullptr-inferred or captured implicitly.
    std::vector<int64_t> row_positions;
    // Exact measured hot-row storage bytes per owning layer, derived from
    // the live cache tensors' type/ne/nb (ggml_row_size over ne[0]):
    // includes Turbo row padding and V-transpose layout width. Parallel to
    // owning_layers. The store sums these checked for the seal baseline
    // instead of assuming logical dim x type; zero/missing entries fail
    // validation (never silently fall back to theory).
    std::vector<uint64_t> hot_bytes_per_row_k;
    std::vector<uint64_t> hot_bytes_per_row_v;
};

// Parameters for sealing an entire multi-group segment bundle
uint64_t compute_layer_group_map_fingerprint(const std::vector<xkv_factor_group_input> & group_inputs);
uint64_t compute_layer_group_map_fingerprint(const std::vector<xkv_factor_group_payload> & groups);

struct xkv_bundle_sealing_params {
    llama_xkv_storage_profile profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS;
    llama_xkv_source          source  = LLAMA_XKV_SOURCE_DECODED_HOT;

    uint32_t rank_k = 16;
    uint32_t rank_v = 16;
    factor_balance balance = LLAMA_XKV_FACTOR_BALANCE_UPSTREAM;
    uint64_t seed = 6362273814452121649ULL;

    ggml_type factor_a_k = GGML_TYPE_TURBO4_0;
    ggml_type factor_b_k = GGML_TYPE_TURBO4_0;
    ggml_type factor_a_v = GGML_TYPE_TURBO4_0;
    ggml_type factor_b_v = GGML_TYPE_TURBO4_0;
    ggml_type landmark_type = GGML_TYPE_Q8_0;

    double max_relative_error = 0.25;
    double min_saving_ratio   = 0.10;
    size_t min_saving_bytes   = 1;

    // Expected group count and deterministic group map fingerprint for bundle completeness validation
    uint32_t expected_group_count = 0;
    uint64_t expected_group_map_fingerprint = 0;

    // Post-factor landmark factory callback per group
    xkv_group_landmark_factory_fn landmark_factory = nullptr;

    // Lazy canonical group source callback. If non-null, called per group index to retrieve
    // canonical K and V data pointers borrowed for that group's iteration only.
    // Allows RuntimeSealer to stream one group at a time without keeping all dense matrices live.
    using canonical_source_fn = std::function<bool(
        uint32_t group_index,
        const float ** out_k,
        const float ** out_v,
        std::string * err
    )>;
    canonical_source_fn canonical_source = nullptr;

    // Configurable landmark chunk token stride (default 8; e.g. 1, 7, 16).
    uint32_t chunk_tokens = 8;

    ggml_type flat_type_k = GGML_TYPE_TURBO4_0;
    ggml_type flat_type_v = GGML_TYPE_TURBO4_0;
    // SHADOW evaluate-only: run capture/factor/gates but publish nothing and
    // emit an empty release plan. Payloads are restored to hot_committed.
    bool evaluate_only = false;
    // Expected code-stream residency. REFERENCE_HOST (default): backend bundle
    // optional (host/CPU execution, tests). DEVICE_OWNED: the gate must attach
    // a verified backend_bundle to the candidate; publish refuses without one
    // (fail-closed, never silent host fallback on device paths).
    ggml_xkv_residency expected_residency = GGML_XKV_RES_REFERENCE_HOST;

    // Optional capacity reservation handle. If provided and valid:
    // 1. Used during preflight and publish capacity checks by excluding its
    //    reserved_bytes from pending_reserved_store_bytes_, eliminating double-counting.
    // 2. Reconciled and consumed upon successful publication commit.
    // 3. Ignored in evaluate_only mode (no reservation consumed/required).
    xkv_capacity_reservation * capacity_reservation = nullptr;
};

// Checked conservative persistent bytes estimator for an entire segment bundle
// across all profiles (TQ_FACTORS, TQ_FACTORS_LANDMARKS, REFERENCE).
// Uses exact descriptor, configurable chunk_tokens, and metadata size formulas.
// Returns false on arithmetic overflow or invalid parameters.
bool estimate_segment_bundle_persistent_bytes(
    const std::vector<xkv_factor_group_input> & group_inputs,
    const xkv_bundle_sealing_params & params,
    size_t * out_bytes,
    std::string * err = nullptr,
    size_t * out_encoded_stream_bytes = nullptr
);

// Seal transaction precommit gate. Fired after all factor/landmark work,
// backend-upload allocations, and failure hooks, but BEFORE the no-fail
// locator/segment commit. The owner of hot-slot lifecycle (runtime/hot pool)
// validates (and stages the release of) the exact hot slots here; return false
// to refuse with zero store mutation. After true returns, the store commit is
// allocation-free/noexcept. Invoked with the store lock held: the callback
// MUST NOT re-enter the store. Skipped entirely in evaluate-only mode.
struct xkv_seal_precommit_ctx {
    uint64_t segment_id = 0; // Candidate about to publish
    uint64_t segment_version = 0;
    std::vector<uint64_t> payload_ids;
    std::vector<uint32_t> physical_rows; // Refreshed hot rows at precommit
    std::vector<uint64_t> generations;
    // Mutable pre-publish candidate: the gate performs upload + verification
    // off-side and attaches candidate->backend_bundle here before commit.
    // Ignored by pool-only gates. Never null when a gate is installed.
    std::shared_ptr<xkv_segment> candidate;
};
using xkv_seal_precommit_fn = std::function<bool(const xkv_seal_precommit_ctx &, std::string * err)>;

// Plan for integration to unbind / release dense hot-source storage after successful sealing
struct xkv_hot_release_plan {
    std::vector<uint64_t> released_payload_ids;
    std::vector<uint32_t> released_physical_rows;
    std::vector<uint64_t> released_generations;
    uint64_t expected_segment_id = 0;
    uint64_t expected_segment_version = 0;
    size_t dense_bytes_freed = 0;
    xkv_snapshot_stamp stamp = {};
    // True when a seal precommit gate callback owned the hot-pool release at
    // commit time. The runtime MUST NOT re-apply pool release for a consumed
    // plan (record/observe only); applying twice would double-release slots.
    // False means the pool still holds the rows and the plan is actionable.
    bool pool_released_at_precommit = false;
};

// Result of sealing API
struct xkv_sealing_result {
    bool success = false;
    xkv_skip_reason skip_reason = xkv_skip_reason::none;
    std::string message;

    uint64_t segment_id = 0;
    uint64_t segment_version = 0;
    size_t flat_source_bytes = 0;
    size_t factored_bytes = 0;
    size_t saved_bytes = 0;
    double compression_ratio = 0.0;
    double relative_error_k = 0.0;
    double relative_error_v = 0.0;

    double factor_quant_seconds = 0.0;   // Wall time in factorize + shadow
    double landmark_quant_seconds = 0.0; // Wall time in landmark construction

    xkv_snapshot_stamp stamp = {};
    xkv_hot_release_plan release_plan = {};
};

// Batch-removal precommit gate. Fired after clones are prepared and after
// revalidation + epoch preflight + reserves, but BEFORE any map/segment
// mutation. Lets the hot-pool owner validate/release rows atomically with the
// store commit (the staged alternative to store-first + compensation, which
// cannot roll back factored bindings); false/throw refuses with zero mutation.
// After true, the commit is allocation-free/noexcept. Invoked with the store
// lock held: the callback MUST NOT re-enter the store. Covers remove_payloads
// and execute_mutation_transaction. Pure packs (no removals) still fire it so
// the owner observes every version bump.
struct xkv_removal_precommit_ctx {
    std::vector<uint64_t> payload_ids;          // Removals requested (maybe empty)
    std::vector<uint64_t> affected_segment_ids;
    std::vector<uint64_t> new_segment_versions; // 0 = segment fully retired
};
using xkv_removal_precommit_fn = std::function<bool(const xkv_removal_precommit_ctx &, std::string * err)>;

// Batch COW transaction structures
struct xkv_batch_mutation {
    std::vector<uint64_t> payload_removals;
    std::vector<uint64_t> segments_to_pack;
    std::function<void()> test_failure_hook = nullptr;
    // Optional semantic landmark rebuild for LANDMARKS-profile segments.
    // Null (default) refuses LANDMARKS removal/pack with zero mutation.
    xkv_landmark_rebuild_fn landmark_rebuild = nullptr;
    // Optional external precommit gate (hot-pool owner).
    xkv_removal_precommit_fn removal_precommit = nullptr;
};

struct xkv_mutation_result {
    bool success = false;
    std::string error;
    std::vector<uint64_t> affected_segments;
    std::vector<uint64_t> new_segment_versions;
    xkv_snapshot_stamp stamp = {};
};

// Result of atomic batch payload removal
struct xkv_payload_removal_result {
    bool success = false;
    std::vector<uint64_t> removed_payload_ids;
    std::vector<uint32_t> released_hot_rows;        // Physical rows of removed hot payloads for cache pool release
    std::vector<xkv_location_kind> removed_kinds;   // Storage kind of each removed payload (hot vs factored)
    std::vector<uint64_t> affected_segment_ids;
    std::vector<uint64_t> new_segment_versions;
    xkv_snapshot_stamp stamp = {};
};

// Shared store owner
// Shared store owner. Authoritative xkv_stamp_provider for the transaction
// coordinator (bound directly, no adapter).
// Forward-declared backend ID generator (src/llama-xkv-backend.h); held by
// unique_ptr so this header stays decoupled.
class xkv_allocation_id_generator;

class llama_xkv_cache_store : public xkv_stamp_provider {
public:
    explicit llama_xkv_cache_store(const llama_cparams & cparams);
    ~llama_xkv_cache_store();

    llama_xkv_cache_store(const llama_xkv_cache_store &) = delete;
    llama_xkv_cache_store & operator=(const llama_xkv_cache_store &) = delete;

    // Location queries & mutations
    bool find_location(uint64_t payload_id, xkv_location & out_loc) const;
    bool register_hot_payload(
        uint64_t payload_id,
        uint32_t physical_row,
        uint64_t generation,
        xkv_state state = xkv_state::hot_writing,
        std::string * err = nullptr
    );
    bool register_hot_payloads(
        const std::vector<xkv_hot_payload_binding> & bindings,
        std::string * err = nullptr,
        std::function<void()> test_failure_hook = nullptr
    );
    bool remove_payload(
        uint64_t payload_id,
        std::string * err = nullptr,
        xkv_landmark_rebuild_fn landmark_rebuild = nullptr,
        xkv_removal_precommit_fn removal_precommit = nullptr,
        uint64_t reservation_token = 0
    );
    bool remove_payloads(
        const std::vector<uint64_t> & payload_ids,
        const std::vector<uint64_t> & expected_generations,
        xkv_payload_removal_result * result = nullptr,
        std::string * err = nullptr,
        std::function<void()> test_failure_hook = nullptr,
        xkv_landmark_rebuild_fn landmark_rebuild = nullptr,
        xkv_removal_precommit_fn removal_precommit = nullptr,
        uint64_t reservation_token = 0
    );
    // Fallible cache-coordinated mutators. can_clear/can_apply_cells_packed are
    // read-only preflights the cache MUST call before touching cells/pool; a
    // false return (with err) aborts the whole operation with zero mutation
    // anywhere. clear rechecks internally and reports failure instead of
    // silently skipping, so every caller can distinguish success.
    // (Cell-compact pack notification was removed: logical cell indices and
    // store hot physical rows are different domains. Factored row packing
    // happens via the batch mutation transaction, not cell compact.)
    bool can_clear(std::string * err = nullptr) const;
    bool clear(std::string * err = nullptr);

    // State discipline transitions
    bool commit_hot_payload(uint64_t payload_id);
    bool commit_hot_payloads(const std::vector<uint64_t> & payload_ids);
    // Atomically validates (payload_id, generation) pairs are hot_committed,
    // then marks them seal_candidate under one fresh nonzero monotonic tx
    // nonce returned via out_nonce. Any mismatch (missing payload, wrong
    // state, stale generation, size mismatch) fails with zero mutation.
    // Returns false (no mutation) while an epoch reservation is held.
    bool mark_seal_candidates(const std::vector<uint64_t> & payload_ids,
                              const std::vector<uint64_t> & generations,
                              uint64_t * out_nonce,
                              std::string * err = nullptr);
    // Reverts ONLY payloads still carrying the given nonce to hot_committed;
    // payloads sealed under a different (newer/foreign) nonce are untouched.
    // This closes the ABA hole where a stale abort by PID could revert a
    // later overlapping seal. Returns false (no mutation) while an epoch
    // reservation is held.
    bool abort_seal_candidates(const std::vector<uint64_t> & payload_ids,
                               uint64_t nonce,
                               xkv_skip_reason reason);
    bool find_payload_state(uint64_t payload_id, xkv_state & out_state) const;

    // Metrics counters
    size_t get_sealed_count() const;
    size_t get_skipped_count(xkv_skip_reason reason) const;

    // Epochs
    xkv_snapshot_stamp current_stamp() const override;
    uint64_t live_epoch() const;
    uint64_t content_epoch() const;
    uint64_t codec_epoch() const;
    uint64_t binding_epoch() const;

    // Central epoch bump preflight mask
    enum epoch_bump_flags : uint8_t {
        bump_flag_live    = 1 << 0,
        bump_flag_content = 1 << 1,
        bump_flag_codec   = 1 << 2,
        bump_flag_binding = 1 << 3,
    };

    bool can_bump(uint8_t mask, std::string * err = nullptr) const;
    void bump_prechecked(uint8_t mask) noexcept;

    // Test-only method to set an epoch to a specific value (e.g. UINT64_MAX to test overflow policy)
    void set_epoch_for_testing(uint8_t mask, uint64_t val) {
        std::lock_guard<std::mutex> lock(mtx);
        if (mask & bump_flag_live) stamp.live_epoch = val;
        if (mask & bump_flag_content) stamp.content_epoch = val;
        if (mask & bump_flag_codec) stamp.codec_epoch = val;
        if (mask & bump_flag_binding) stamp.binding_epoch = val;
    }

    // Test-only seam to drive the reservation token counter near exhaustion.
    void set_next_res_token_for_testing(uint64_t val) {
        std::lock_guard<std::mutex> lock(mtx);
        next_res_token = val;
    }

    // Test-only seam to drive capacity reservation token counter near exhaustion.
    void set_next_cap_res_token_for_testing(uint64_t val) {
        std::lock_guard<std::mutex> lock(mtx);
        next_cap_res_token_ = val;
    }

    bool bump_live_epoch(std::string * err = nullptr);
    bool bump_content_epoch(std::string * err = nullptr);
    bool bump_codec_epoch(std::string * err = nullptr);
    bool bump_binding_epoch(std::string * err = nullptr);
    void update_view_stamp(const llama_rerot_view_stamp & view);

    // xkv_stamp_provider implementation (authoritative stamp + epochs).
    bool preflight_content_advance(xkv_stamp_reservation * out_res, std::string * err) override;
    xkv_snapshot_stamp apply_content_advance(const xkv_stamp_reservation & res,
                                              const xkv_removal_feedback & feedback) noexcept override;
    void abort_content_advance(const xkv_stamp_reservation & res) noexcept override;
    bool install_stable_view(const llama_rerot_view_stamp & stable_view,
                               xkv_snapshot_stamp * out_new_stamp,
                               std::string * err) override;
    bool advance_context_shift(const llama_rerot_view_stamp & shifted_view,
                                 xkv_snapshot_stamp * out_new_stamp,
                                 std::string * err) override;
    // Reservation token validation for coordinators (true iff live token).
    // Arbitrary callers cannot suppress epochs: every mutator refuses while a
    // reservation is held unless carrying the live token (single remove path).
    bool validate_reservation_token(uint64_t token) const;

    // COW Candidate Lifecycle (internal candidate builder before freezing to const snapshot)
    std::shared_ptr<xkv_segment> create_candidate_segment(
        llama_xkv_storage_profile profile,
        llama_xkv_source source,
        const std::vector<xkv_factor_group_payload> & groups
    );
    bool validate_candidate(const std::shared_ptr<const xkv_segment> & candidate, std::string * err = nullptr) const;
    bool publish_candidate(
        std::shared_ptr<xkv_segment> candidate,
        const std::vector<uint64_t> & payload_ids,
        const std::vector<uint64_t> & generations,
        std::string * err = nullptr,
        xkv_landmark_rebuild_fn landmark_rebuild = nullptr,
        xkv_seal_precommit_fn precommit = nullptr,
        xkv_capacity_reservation * capacity_reservation = nullptr
    );
    void abort_candidate(std::shared_ptr<xkv_segment> & candidate);

    // Multi-group Sealing API: consumes all factor groups in bundle + payload bindings,
    // factorizes independent K/V for each group, evaluates configured final four-stream codecs,
    // gates measured bytes/error/min saving across all groups, atomically publishes the bundle,
    // and emits a single hot_release_plan.
    xkv_sealing_result seal_segment_bundle(
        const std::vector<xkv_factor_group_input> & group_inputs,
        const std::vector<uint64_t> & payload_ids,
        const std::vector<uint64_t> & generations,
        const xkv_bundle_sealing_params & params,
        xkv_seal_precommit_fn precommit = nullptr,
        xkv_capacity_reservation * capacity_reservation = nullptr
    );

    // SHADOW evaluate-only API: same args as seal_segment_bundle, runs
    // factorization/encode/gates and discards the candidate without publication
    // (payloads stay hot_committed, epochs/locations untouched, empty release
    // plan). Equivalent to seal with params.evaluate_only = true. SHADOW paths
    // call this and record shadow metrics only; it never rebinds or releases.
    xkv_sealing_result evaluate_segment_bundle(
        const std::vector<xkv_factor_group_input> & group_inputs,
        const std::vector<uint64_t> & payload_ids,
        const std::vector<uint64_t> & generations,
        const xkv_bundle_sealing_params & params
    );

    // Atomic snapshot import for state restore. The bridge materializes and
    // validates everything off-side; the store only installs on success.
    // Serialized high-waters are CONSUMED (never derived): next_segment_id
    // must exceed the present max, next_seal_tx_nonce must be nonzero,
    // stamp/sealed_count adopted exactly. Refused unless the store holds zero
    // factored state (no published/retired segments, no factored locations)
    // and no epoch reservation is active; dedup allocated bytes must fit the
    // store cap. Single no-throw commit; old state untouched on refusal.
    // Backend handle high-waters stay BackendResidency-side (not owned here).
    struct xkv_snapshot_import_bundle {
        std::vector<std::shared_ptr<const xkv_segment>> segments;
        std::vector<std::pair<uint64_t, xkv_location>> locations;
        xkv_snapshot_stamp stamp = {};
        uint64_t sealed_count = 0;
        uint64_t next_segment_id = 1;
        uint64_t next_seal_tx_nonce = 1;
        // Serialized backend allocation-ID high-water for the persistent
        // generator (never reset below issued ids once seeded).
        uint64_t next_alloc_id = 1;
    };
    bool import_snapshot_segments(const xkv_snapshot_import_bundle & bundle, std::string * err = nullptr,
                                  xkv_capacity_reservation * capacity_reservation = nullptr);

    // Reader pins & segment access (external callers receive const immutable snapshots only)
    xkv_reader_pin pin_segment(uint64_t segment_id);
    xkv_reader_pin pin_segment_version(uint64_t segment_id, uint64_t segment_version);
    std::shared_ptr<const xkv_segment> get_segment(uint64_t segment_id) const;
    std::shared_ptr<const xkv_segment> get_segment_version(uint64_t segment_id, uint64_t segment_version) const;
    void reclaim_retired_segments();

    // Storage-only immutable snapshot enumeration APIs (feeding cache-owned semantic joins)
    xkv_hot_bindings_snapshot list_hot_payload_bindings(
        const std::vector<xkv_state> & state_filter = {}
    ) const;

    xkv_payload_segments_snapshot query_payload_segments(
        const std::vector<uint64_t> & payload_ids
    );

    bool validate_hot_release_plan(
        const xkv_hot_release_plan & plan,
        std::string * err = nullptr
    ) const;

    // Byte-preserving encoded-A pack with immutable shared B. Survivor A rows are
    // copied verbatim (no re-encode); per-row baselines travel verbatim with the
    // total recomputed for survivors. DEVICE_OWNED segments refuse before any host
    // decode/memcpy and require a backend-native byte-preserving transaction.
    // Any landmark stream presence (any profile) requires a semantic landmark
    // rebuild callback for atomic rechunking and refuses without one (zero
    // mutation, old version intact). Retired versions are preserved until the
    // last pin releases; metadata commits only after every data move succeeds.
    bool pack_segment(
        uint64_t segment_id,
        std::string * err = nullptr,
        xkv_landmark_rebuild_fn landmark_rebuild = nullptr
    );

    // Two-phase batch COW transaction API (prepare off-lock, commit under lock).
    // Same DEVICE_OWNED/landmark/host-byte preconditions as pack_segment; every
    // failure path preserves published bytes, epochs, and handles.
    bool execute_mutation_transaction(
        const xkv_batch_mutation & mutation,
        xkv_mutation_result * result = nullptr,
        std::string * err = nullptr
    );

    // Layer aliasing & deduplicated accounting
    void register_layer_alias(uint32_t model_layer, uint32_t owning_layer);
    uint32_t resolve_owning_layer(uint32_t model_layer) const;
    xkv_accounting get_accounting() const;

    // Workspace arena access
    xkv_workspace_arena & get_arena() { return workspace_arena; }
    const xkv_workspace_arena & get_arena() const { return workspace_arena; }
    xkv_arena_lease acquire_workspace_lease(size_t bytes) { return workspace_arena.acquire(bytes); }

    // Persistent backend allocation-ID generator (BackendResidency-owned type).
    // Seeded from the image high-water at restore; failed batches burn ids by
    // design, so it is never rewound. Import/pack paths take it by reference.
    xkv_allocation_id_generator & allocation_id_generator();

    // Hard persistent-store cap from xkv_store_mib (checked bytes; 0 = unlimited).
    // Exact deduplicated preflight for BackendResidency builders: fits the live
    // published + retired + given new segments (shared B / backend handles once
    // each) against the cap. False + err on overflow, misfit, or (for the
    // checked form) allocation failure while walking.
    size_t store_capacity_bytes() const;
    bool check_store_capacity_for(const std::vector<std::shared_ptr<const xkv_segment>> & new_segments,
                                   std::string * err = nullptr) const;

    // Computes the exact incremental allocated bytes a candidate bundle would add
    // to the persistent store relative to current live/retired state, taking into
    // account deduplication of shared B matrices and device backend handles.
    // Used by runtime native and CPU sealing paths for min-saving and metrics.
    // Returns false on arithmetic overflow or null candidate.
    bool candidate_incremental_bytes(const std::shared_ptr<const xkv_segment> & candidate,
                                     size_t * out_bytes,
                                     std::string * err = nullptr) const;
    // Prepares accounting scratch buffers for a candidate segment (including attached backend bundle).
    // Safe to call before candidate_incremental_bytes or publish_candidate.
    bool prepare_accounting_scratch(const std::shared_ptr<const xkv_segment> & candidate, std::string * err = nullptr);

    // Atomic preflight & RAII store capacity reservation (xkv_store_mib hard cap)
    bool preflight_store_capacity(size_t expected_bytes, size_t * out_deficit = nullptr,
                                   std::string * err = nullptr) const;
    xkv_capacity_reservation reserve_capacity(size_t expected_bytes, std::string * err = nullptr,
                                              size_t * out_deficit = nullptr,
                                              const std::vector<uint64_t> & expected_removal_pids = {},
                                              const std::vector<uint64_t> & expected_removal_gens = {});

    // Transient device staging reservation (workspace arena / device peak limit)
    xkv_device_staging_reservation reserve_device_staging(size_t bytes, std::string * err = nullptr,
                                                          size_t * out_deficit = nullptr);

    size_t get_pending_reserved_store_bytes() const;
    size_t get_device_staging_reserved_bytes() const;
    size_t get_device_staging_peak_bytes() const;

    const llama_cparams & get_cparams() const { return cparams; }

private:
    void reclaim_retired_segments_locked();
    bool ensure_accounting_scratch_locked(const std::vector<std::shared_ptr<const xkv_segment>> & candidate_extras, std::string * err = nullptr) const;
    friend class xkv_capacity_reservation;
    friend class xkv_device_staging_reservation;
    void release_capacity_reservation_locked(uint64_t token) noexcept;
    void release_device_staging_reservation_locked(size_t bytes) noexcept;

    // Epoch preflight with mtx already held (for reservation-atomic paths).
    bool can_bump_locked(uint8_t mask, std::string * err) const;

    // Exact deduplicated allocated bytes over published + retired (+ extras
    // for COW peak: old + new until old unpins/reclaims). Shared B counted
    // once. Caller holds mtx. False on arithmetic overflow.
    bool deduplicated_allocated_locked(const std::vector<std::shared_ptr<const xkv_segment>> & extras,
                                       size_t & out_bytes) const;
    // Cap enforcement with mtx held. True when unlimited or fitting.
    bool store_capacity_fits_locked(const std::vector<std::shared_ptr<const xkv_segment>> & extras,
                                    std::string * err,
                                    size_t exclude_reserved_bytes = 0) const;

    mutable std::mutex mtx;

    llama_cparams cparams;
    xkv_snapshot_stamp stamp;

    // Epoch reservation state backing xkv_stamp_provider. While active_token
    // is nonzero, every map/epoch mutator refuses unless carrying the live
    // token on the single allowed batch-remove path. RERoT view flow is
    // unaffected (update_view_stamp stays open; feedback validation treats
    // view drift as mismatch, fail-safe). Call check_reservation_locked with
    // mtx held; true means refuse.
    uint64_t active_res_token = 0;
    xkv_snapshot_stamp reserve_base = {};
    bool reserve_spent = false;
    uint64_t next_res_token = 1;

    bool check_reservation_locked(uint64_t token) const {
        if (active_res_token == 0) {
            return token != 0; // No reservation: forged/stale tokens refused.
        }
        if (token == 0 || token != active_res_token || reserve_spent) {
            return true;
        }
        return !(stamp.live_epoch == reserve_base.live_epoch &&
                 stamp.content_epoch == reserve_base.content_epoch &&
                 stamp.codec_epoch == reserve_base.codec_epoch &&
                 stamp.binding_epoch == reserve_base.binding_epoch);
    }

    uint64_t next_segment_id = 1;

    std::unordered_map<uint64_t, xkv_location> payload_locations;
    std::unordered_map<uint64_t, std::shared_ptr<const xkv_segment>> published_segments;
    std::vector<std::shared_ptr<const xkv_segment>> retired_segments;
    std::unordered_map<uint32_t, uint32_t> layer_aliases;
    xkv_workspace_arena workspace_arena;

    mutable size_t host_peak_bytes_ = 0;
    mutable size_t device_peak_bytes_ = 0;
    // Persistent backend allocation-ID generator (unique_ptr: incomplete here).
    std::unique_ptr<xkv_allocation_id_generator> alloc_id_gen;

    size_t sealed_count = 0;
    std::unordered_map<uint8_t, size_t> skipped_counts;
    uint64_t next_seal_tx_nonce = 1;

    // Store capacity reservation ledger
    uint64_t next_cap_res_token_ = 1;
    size_t pending_reserved_store_bytes_ = 0;
    std::unordered_map<uint64_t, size_t> pending_reservations_;

    // Transient device staging reservation tracking
    size_t device_staging_reserved_bytes_ = 0;
    size_t device_staging_peak_bytes_ = 0;

    // Bounded reusable scratch buffers for zero-allocation accounting walks under mtx
    mutable std::vector<const encoded_matrix *> scratch_unique_b_k_;
    mutable std::vector<const encoded_matrix *> scratch_unique_b_v_;
    mutable std::vector<uint64_t> scratch_backend_alloc_ids_;
    mutable std::vector<uint8_t> scratch_backend_alloc_accounted_;
};

} // namespace llama_xkv
