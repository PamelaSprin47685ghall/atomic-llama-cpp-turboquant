#pragma once

#include "ggml.h"
#include "llama.h"
#include "llama-cparams.h"
#include "llama-triattention.h"
#include "llama-xkv-cache.h"
#include "llama-xkv-codec.h"
#include "llama-xkv-factor.h"
#include "llama-xkv-landmark.h"
#include "llama-kv-cells.h"
#include "llama-ext.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace llama_xkv {

// Layer slicing info for an owning layer within a factor group.
// Invariant: No equal-layer-width assumptions; each layer has explicit feature offsets and dims.
struct xkv_layer_slice {
    uint32_t model_layer = 0;
    uint32_t owning_layer = 0;
    uint32_t feature_offset_k = 0; // element offset along feature axis of B_K
    uint32_t feature_dim_k = 0;    // total feature dimension for this layer in B_K
    uint32_t feature_offset_v = 0;
    uint32_t feature_dim_v = 0;
    uint32_t n_kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t rotary_dim = 0;       // dimensions with RoPE applied (<= head_dim)
    uint32_t rope_style = 0;       // MUST be 0 (half/NeoX pairing; IMRoPE pairing is also NeoX half)
};

// Candidate item for Tri scoring and policy evaluation
struct xkv_tri_candidate {
    uint32_t cell_index = 0;
    int32_t  storage_pos = 0;
    uint64_t payload_id = 0;
    uint64_t storage_generation = 0;
    uint64_t segment_version = 0;
    segment_row_ref row_ref = {};
    xkv_location location = {};

    std::vector<llama_seq_id> seq_ids; // sequences holding references to this cell
    bool is_protected = false;         // hard guard (e.g. pending record)
    bool device_owned = false;         // factor bytes live on device; host decode/readback forbidden
    uint32_t semantic_episode_id = 0;  // 0 if none
    llama_rerot_visibility visibility = llama_rerot_visibility::public_live;
};

// Per-sequence specification matching existing reclaim loop
struct xkv_tri_seq_info {
    llama_seq_id seq_id = 0;
    uint32_t logical_tokens = 0; // distinct logical length L
    int64_t  frontier_pos = 0;   // decode frontier / max pos for this sequence
    uint32_t tail_guard = 128;   // recent window
    bool     eligible = true;
    uint32_t semantic_episode_id = 0;
    std::vector<llama_seq_id> semantic_seq_ids; // reader-tail HARD-GUARD inputs
};

struct xkv_tri_config {
    uint32_t tile_size = 32;          // bounded tile size for factor streaming
    uint32_t recent_window = 128;     // default tail guard
    double   ratio = 3.0 / 32.0;      // default retention ratio (3/32)
    uint32_t pool_radius = 2;         // max-pool radius
    bool     normalize_scores = true; // fixed production: per-head z-score normalization across entire candidate set
    bool     is_ablation = false;     // explicit flag required to override fixed production parameters
    enum triattention_agg head_agg = TRIATTENTION_AGG_MAX; // fixed production head aggregation: normalized max/union
    enum triattention_agg agg = TRIATTENTION_AGG_MEAN;
    bool     disable_trig = false;    // ablation: norm-only scoring
    bool     disable_mlr = false;
    size_t   workspace_budget_bytes = 0; // optional hard budget limit (0 = unbounded/checked via arena)
};

// Pressure resource type targeting specific bottleneck
enum class xkv_tri_pressure_resource {
    legacy_payloads = 0, // cell / payload count deficit
    hot_slots = 1,       // physical hot cache slot shortage
    store_bytes = 2,     // factor / landmark device or host byte shortage
};

struct xkv_tri_pressure_state {
    bool kv_pressure = false;
    bool recurrent_pressure = false;
    bool maintenance_due = false;
    xkv_tri_pressure_resource resource = xkv_tri_pressure_resource::legacy_payloads;
    uint64_t required_units = 0; // required units of resource (slots, bytes, or payloads)
};

// Forward declaration / table for explicit sequence frontiers
struct xkv_tri_frontier_entry {
    llama_seq_id seq_id = 0;
    int64_t frontier_pos = -1;
};
using xkv_tri_frontier_table = std::vector<xkv_tri_frontier_entry>;


// Exact (payload_id, seq_id) reference removal
struct xkv_tri_ref_removal {
    uint64_t payload_id = 0;
    llama_seq_id seq_id = 0;
    uint32_t cell_index = 0;

    bool operator==(const xkv_tri_ref_removal & o) const {
        return payload_id == o.payload_id && seq_id == o.seq_id && cell_index == o.cell_index;
    }
};

// Result of mutation planning
struct xkv_tri_mutation_plan {
    // Exact per-sequence reference removals emitted by policy
    std::vector<xkv_tri_ref_removal> ref_removals;

    // Physical survivors (at least one keeper ref remaining)
    std::vector<uint64_t> survivor_payloads;
    std::vector<uint32_t> survivor_cells;

    // Physically evicted (all keeper refs removed)
    std::vector<uint64_t> evicted_payloads;
    std::vector<uint32_t> evicted_cells;

    // Affected segments containing evicted rows (require A-row repacking)
    std::vector<uint64_t> affected_segments;

    // Accounting metrics (matching tri_* specifications)
    uint32_t total_candidates = 0;
    uint32_t target_references = 0;
    uint32_t hard_keep = 0;
    uint32_t shared_keep = 0;
    uint32_t references_removed = 0;
    uint32_t physical_freed = 0;
    uint32_t physical_before = 0;
    uint32_t physical_after = 0;

    // Resource-specific outcomes (capacity_satisfied compares required units
    // of the pressured resource only; physical_freed stays the generic metric)
    uint32_t hot_slots_freed = 0;
    uint32_t factored_rows_freed = 0;
    uint64_t factored_bytes_reclaimed = 0;
    bool capacity_satisfied = false;

    bool changed = false; // TRUE for any reference removal, even when no physical row frees
    bool recurrent_only_bypass = false;
};

// Provider callback for reading canonical pre-RoPE K from hot cache
using hot_k_provider_fn = std::function<bool(
    uint32_t model_layer,
    uint32_t kv_head,
    const std::vector<uint32_t> & cell_indices,
    float * dst_pre_rope_k_half, // [cell_count * head_dim] in half layout
    size_t capacity_elements
)>;

// Zero-heap span variant of the hot K provider: cell indices as pointer/len
// so carved scoring never builds a vector per tile. Preferred for production;
// the vector form remains as a legacy adapter (one bounded allocation).
using hot_k_span_provider_fn = std::function<bool(
    uint32_t model_layer,
    uint32_t kv_head,
    const uint32_t * cell_indices,
    size_t n_cells,
    float * dst_pre_rope_k_half, // [n_cells * head_dim] in half layout
    size_t capacity_elements
)>;

// Preflighted atomic store mutation proposal wrapping batch COW transaction
struct xkv_store_mutation_proposal {
    xkv_batch_mutation batch;
    xkv_snapshot_stamp expected_stamp = {};

    // Validate that store can execute all deletions and packs atomically
    bool preflight(const llama_xkv_cache_store & store, std::string * err = nullptr) const;

    // Apply the mutations atomically using store's execute_mutation_transaction.
    // On any failure, old locations stay valid. Landmark invalidation occurs only after successful commit.
    bool apply(
        llama_xkv_cache_store & store,
        landmark_table * lm_table = nullptr,
        std::string * err = nullptr
    ) const;
};

// Proposal outcome status (strictly distinguishing floor-exhausted from failures)
enum class xkv_tri_proposal_status {
    success = 0,
    floor_exhausted = 1,
    stale_stamp = 2,
    workspace_exhausted = 3,
    codec_error = 4,
    score_error = 5,
    invalid_argument = 6,
    store_error = 7,
    bypass_recurrent_only = 8,
    device_scoring_unavailable = 9,
};

inline const char * xkv_tri_proposal_status_name(xkv_tri_proposal_status status) {
    switch (status) {
        case xkv_tri_proposal_status::success:               return "success";
        case xkv_tri_proposal_status::floor_exhausted:      return "floor-exhausted";
        case xkv_tri_proposal_status::stale_stamp:          return "stale-stamp";
        case xkv_tri_proposal_status::workspace_exhausted:  return "workspace-exhausted";
        case xkv_tri_proposal_status::codec_error:          return "codec-error";
        case xkv_tri_proposal_status::score_error:          return "score-error";
        case xkv_tri_proposal_status::invalid_argument:      return "invalid-argument";
        case xkv_tri_proposal_status::store_error:          return "store-error";
        case xkv_tri_proposal_status::bypass_recurrent_only:return "bypass-recurrent-only";
        case xkv_tri_proposal_status::device_scoring_unavailable: return "device-scoring-unavailable";
        default:                                             return "unknown";
    }
}

// Memory requirements planned for scratch/buffers during proposal generation
struct xkv_tri_workspace_requirements {
    size_t candidate_array_bytes = 0;
    size_t candidate_order_bytes = 0;
    size_t score_buffers_bytes = 0;
    size_t tile_buffers_bytes = 0;
    size_t tile_a_bytes = 0;
    size_t union_maps_bytes = 0;
    size_t total_bytes = 0;
};

// Callback for device-owned factor scoring (e.g. Vulkan carry-chain / GPU compute).
// Used when factor streams have GGML_XKV_RES_DEVICE_OWNED and cannot be host-decoded.
using device_factor_scoring_fn = std::function<bool(
    uint64_t segment_id,
    uint64_t segment_version,
    const uint32_t * seg_rows,
    uint32_t n_rows,
    const xkv_layer_slice & slice,
    uint32_t sampled_head_idx,
    int64_t frontier_pos,
    float * out_scores,
    std::string * err
)>;

// Distinct fail-closed error for device-owned factors without a device scorer.
// Caught separately from score_error so the integrator can dispatch natively.
struct xkv_device_scoring_error : public std::runtime_error {
    explicit xkv_device_scoring_error(const std::string & msg) : std::runtime_error(msg) {}
};

// Bulk selected-K fetch primitive (native device path contract).
// Fills dst[n_rows * head_dim] with pre-RoPE half-layout K for exactly the
// requested segment rows. The implementation MUST serve the whole batch in
// one device graph + one sync (selected rows only; never full factor or
// full-history D2H). Native implementations may reconstruct selected
// post-RoPE K in one graph/sync then invert RoPE on the selected rows only.
// Host-backed implementations must use bounded tile decodes. Returns false
// (with err) to fail closed; Tri never falls back to host full decode.
using xkv_tri_selected_k_fetch_fn = std::function<bool(
    uint64_t segment_id,
    uint64_t segment_version,
    const uint32_t * seg_rows,
    uint32_t n_rows,
    const xkv_layer_slice & slice,
    uint32_t kv_head,
    float * dst_pre_rope, // [n_rows * slice.head_dim] half layout
    size_t dst_capacity_elements,
    std::string * err
)>;

// Options for proposal generation
struct xkv_tri_reclaim_options {
    xkv_landmark_rebuild_fn landmark_rebuild = nullptr;
    hot_k_provider_fn hot_provider = nullptr;
    hot_k_span_provider_fn hot_span_provider = nullptr;
    device_factor_scoring_fn device_scoring = nullptr;
    // Bulk selected-K fetch for device-owned lanes: exactly one call per
    // (segment-group run) covering every selected row of the tile, then Tri
    // scores each sampled head locally (sampled-head reuse of one fetch).
    // The fetch MUST be one device graph + one sync internally (or a bounded
    // host decode for host-resident segments); never full-factor/history D2H.
    xkv_tri_selected_k_fetch_fn selected_k_fetch = nullptr;
    const xkv_tri_frontier_table * explicit_frontiers = nullptr;
    xkv_workspace_arena * arena = nullptr;
    size_t workspace_budget_bytes = 0;
};

// Comprehensive reclaim proposal returned to llama-kv-cache
struct xkv_tri_reclaim_proposal {
    xkv_tri_proposal_status status = xkv_tri_proposal_status::success;
    std::string message;

    xkv_tri_mutation_plan plan;
    xkv_store_mutation_proposal store_proposal;

    std::vector<uint64_t> released_hot_payloads;
    std::vector<uint32_t> released_hot_rows;

    std::vector<uint64_t> factored_pack_segments;
    std::vector<uint64_t> factored_tombstone_segments;
    std::vector<uint64_t> factored_survivor_payloads;

    uint32_t hot_slots_freed = 0;
    uint32_t factored_rows_freed = 0;
    uint64_t factored_bytes_reclaimed = 0;
    bool capacity_satisfied = false;

    xkv_snapshot_stamp expected_stamp = {};

    size_t required_workspace_bytes = 0;
    size_t peak_workspace_bytes = 0;

    bool is_success() const { return status == xkv_tri_proposal_status::success; }
    bool is_floor_exhausted() const { return status == xkv_tri_proposal_status::floor_exhausted; }
};

// ---- Production pressure integration (fixed order contract) ----
//
// Under XKV ON pressure the integrator preserves this exact order:
//   1. lossless seal/pack first (store/runtime side, atomic);
//   2. Tri proposal across HOT+FACTORED semantic payloads (this slice);
//   3. apply the store transaction, then refresh hot/store/KV usage;
//   4. ONLY on floor-exhausted with remaining deficit: atomic victim
//      handling (idle demotion / active preemption, server side).
// Recurrent-only pressure bypasses BOTH KV paths and never reaches Tri.
// The adapter never decides fallback itself; it only reports the decision.
enum class xkv_tri_pressure_decision {
    reclaimed = 0,          // proposal ready; apply + refresh usage
    floor_exhausted_victim, // floor reached, deficit may remain -> victim handling
    bypass_recurrent_only,  // recurrent-only: both KV paths bypassed
    idle_noop,              // fill-first/idle: explicit no-op success
    retry_stale,            // store moved under the proposal: re-enumerate + retry
    error,                  // score/codec/store/workspace failure: no mutation
};

inline const char * xkv_tri_pressure_decision_name(xkv_tri_pressure_decision d) {
    switch (d) {
        case xkv_tri_pressure_decision::reclaimed: return "reclaimed";
        case xkv_tri_pressure_decision::floor_exhausted_victim: return "floor-exhausted-victim";
        case xkv_tri_pressure_decision::bypass_recurrent_only: return "bypass-recurrent-only";
        case xkv_tri_pressure_decision::idle_noop: return "idle-noop";
        case xkv_tri_pressure_decision::retry_stale: return "retry-stale";
        case xkv_tri_pressure_decision::error: return "error";
        default: return "unknown";
    }
}

// Integrator-side hook bundle: same fields as reclaim options, named for the
// production call site (lossless seal/pack callbacks stay runtime-owned).
struct xkv_tri_pressure_hooks {
    hot_k_span_provider_fn hot_span_provider = nullptr;
    hot_k_provider_fn hot_provider = nullptr; // legacy vector form
    xkv_tri_selected_k_fetch_fn selected_k_fetch = nullptr;
    device_factor_scoring_fn device_scoring = nullptr;
    xkv_landmark_rebuild_fn landmark_rebuild = nullptr;
    const xkv_tri_frontier_table * explicit_frontiers = nullptr;
    xkv_workspace_arena * arena = nullptr;
    size_t workspace_budget_bytes = 0;
};

// Refreshed usage snapshot after applying a proposal (read again post-commit).
struct xkv_tri_usage_snapshot {
    uint32_t cells_used = 0;
    size_t live_payload_bytes = 0;
    size_t factored_bytes = 0;
    size_t hot_bytes = 0;
    size_t active_segments = 0;
};

// Read hot/store/KV usage from cells + store (call again after apply).
xkv_tri_usage_snapshot xkv_tri_read_usage(const llama_kv_cells & cells,
                                           const llama_xkv_cache_store & store);

// Enumerate candidate items from llama_kv_cells joined to store locations.
// Guarantees deterministic cell_index order.
std::vector<xkv_tri_candidate> enumerate_candidates_from_cells(
    const llama_kv_cells & cells,
    const llama_xkv_cache_store & store
);

class xkv_tri_adapter {
public:
    // Takes ONLY an existing validated calibration view plus exact runtime omega and freq_scale_sq.
    // Validates finite values and rejects rope_style != 0.
    xkv_tri_adapter(
        const triattention_calibration & calib,
        const float * runtime_omega,
        const float * runtime_freq_scale_sq,
        const xkv_tri_config & cfg = {}
    );

    ~xkv_tri_adapter() = default;

    // Non-copyable
    xkv_tri_adapter(const xkv_tri_adapter &) = delete;
    xkv_tri_adapter & operator=(const xkv_tri_adapter &) = delete;
    xkv_tri_adapter(xkv_tri_adapter &&) noexcept = default;
    xkv_tri_adapter & operator=(xkv_tri_adapter &&) noexcept = default;

    bool valid() const;

    // Stream final quantized factor K reconstruction for sampled owning layers & KV heads in bounded tiles.
    // Preserves pre-RoPE half-layout and partial rotary tail.
    // Validates every shape, slice, head, row, live, pointer, segment version pin, and overflow;
    // decodes ONLY exact B head feature rows, never whole B!
    void stream_factored_k_head(
        const xkv_segment & segment,
        const xkv_layer_slice & slice,
        uint32_t kv_head,
        const uint32_t * seg_row_indices,
        uint32_t n_rows,
        float * dst_half,
        size_t dst_capacity_elements,
        uint32_t tile_size = 32
    ) const;

    // Compute Tri importance scores for a specific subset of candidates at a given sequence frontier.
    // Streams bounded tiles into per-head O(n) score arrays, reuses one K decode
    // across multiple exact sampled Q-head calibration entries sharing the KV head,
    // and normalizes scores across the candidate subset.
    // Missing slice, segment, or hot provider is an error; never returns dummy zeros.
    void score_candidate_subset(
        const std::vector<xkv_tri_candidate> & candidates,
        const std::vector<uint32_t> & subset_indices,
        const std::vector<xkv_layer_slice> & slices,
        const llama_xkv_cache_store & store,
        int64_t seq_frontier_position,
        float * out_pooled_scores,
        float * out_raw_combined = nullptr,
        const hot_k_provider_fn & hot_provider = nullptr,
        const device_factor_scoring_fn & device_scoring = nullptr,
        const hot_k_span_provider_fn & hot_span = nullptr,
        xkv_arena_lease * scratch_lease = nullptr,
        const xkv_tri_selected_k_fetch_fn & fetch = nullptr
    ) const;

    // Build mutation plan matching existing reclaim loop exactly:
    // For each sequence:
    //   1. Build its own candidate subset
    //   2. Compute seq target = max(tail_guard, ceil(L * ratio))
    //   3. Protect recent (pos >= frontier - tail_guard + 1), pending, foreign, and semantic-reader-tail
    //   4. Score with that seq.frontier_pos and local-pool within that subset
    //   5. Choose references to evict and emit exact (payload_id, seq_id) reference removals
    // Then union physical survivors: cell survives if any keeper ref remains.
    // changed is true if references_removed > 0.
    xkv_tri_mutation_plan build_mutation_plan(
        const std::vector<xkv_tri_candidate> & candidates,
        const std::vector<xkv_tri_seq_info> & seqs,
        const std::vector<xkv_layer_slice> & slices,
        const llama_xkv_cache_store & store,
        const xkv_tri_pressure_state & pressure,
        const hot_k_provider_fn & hot_provider = nullptr,
        const xkv_tri_frontier_table * explicit_frontiers = nullptr,
        const device_factor_scoring_fn & device_scoring = nullptr,
        const hot_k_span_provider_fn & hot_span = nullptr,
        xkv_arena_lease * scratch_lease = nullptr,
        const xkv_tri_selected_k_fetch_fn & fetch = nullptr
    ) const;

    // Build preflighted atomic store mutation proposal from mutation plan
    xkv_store_mutation_proposal create_mutation_proposal(
        const xkv_tri_mutation_plan & plan,
        const llama_xkv_cache_store & store,
        xkv_landmark_rebuild_fn landmark_rebuild = nullptr
    ) const;

    // Calculate required workspace layout and memory requirements
    bool estimate_workspace_requirements(
        uint32_t n_candidates,
        const std::vector<xkv_tri_seq_info> & seqs,
        const std::vector<xkv_layer_slice> & slices,
        xkv_tri_workspace_requirements & out_reqs,
        std::string * err = nullptr,
        const llama_xkv_cache_store * store = nullptr,
        const std::vector<xkv_tri_candidate> * candidates = nullptr
    ) const;

    // Primary deterministic reclaim proposal API consumed by llama-kv-cache later
    xkv_tri_reclaim_proposal create_reclaim_proposal(
        const std::vector<xkv_tri_candidate> & candidates,
        const std::vector<xkv_tri_seq_info> & seqs,
        const std::vector<xkv_layer_slice> & slices,
        const llama_xkv_cache_store & store,
        const xkv_tri_pressure_state & pressure,
        const xkv_tri_reclaim_options & options = {}
    ) const;

    // Overload taking llama_kv_cells directly
    xkv_tri_reclaim_proposal create_reclaim_proposal(
        const llama_kv_cells & cells,
        const std::vector<xkv_tri_seq_info> & seqs,
        const std::vector<xkv_layer_slice> & slices,
        const llama_xkv_cache_store & store,
        const xkv_tri_pressure_state & pressure,
        const xkv_tri_reclaim_options & options = {}
    ) const;

    // Production bridge: generic reclaim request + cells -> proposal.
    // Frontiers come authoritatively from cells (seq_pos_max); logical
    // lengths, guards, eligibility and semantic refs come from the hints.
    // In XKV reclaim_kv, generic required_free represents physical hot/KV
    // slots, so resource is xkv_tri_pressure_resource::hot_slots and only
    // hot removals count toward capacity satisfaction; factored-only deletion
    // cannot satisfy a physical hot deficit. drain_to_floor (or any deficit)
    // raises KV pressure. Recurrent pressure arrives as a parameter so the
    // caller gate stays the single bypass authority.
    xkv_tri_reclaim_proposal plan_pressure(
        const llama_kv_cells & cells,
        const llama_memory_kv_reclaim_request & request,
        const std::vector<xkv_layer_slice> & slices,
        const llama_xkv_cache_store & store,
        bool recurrent_pressure,
        const xkv_tri_pressure_hooks & hooks = {},
        xkv_tri_pressure_resource resource = xkv_tri_pressure_resource::hot_slots
    ) const;

    // Explicit store-byte pressure entry point for cap admission / factor budget:
    // required_bytes specifies the store byte deficit; only factored bytes
    // reclaimed count toward capacity satisfaction.
    xkv_tri_reclaim_proposal plan_store_byte_pressure(
        const llama_kv_cells & cells,
        uint64_t required_bytes,
        const std::vector<xkv_layer_slice> & slices,
        const llama_xkv_cache_store & store,
        const xkv_tri_pressure_hooks & hooks = {}
    ) const;

    // Map a finished proposal to the fixed-order decision (no fallback taken).
    xkv_tri_pressure_decision decide(const xkv_tri_reclaim_proposal & proposal) const;

    // Map a finished proposal to the generic reclaim result for the
    // integrator to merge after applying the transaction + refreshing usage.
    llama_memory_kv_reclaim_result result_from_proposal(
        const xkv_tri_reclaim_proposal & proposal,
        const llama_memory_kv_reclaim_request & request,
        uint32_t physical_before) const;

    // Host-backed selected-K fetch over store segments (tests, CPU reference,
    // host-resident fallback). Selected rows only, bounded tiles, zero-heap
    // scratch beyond the caller's arena when provided. Refuses device_owned
    // candidates fail-closed: production device segments need a native fetch.
    xkv_tri_selected_k_fetch_fn make_host_selected_k_fetch(
        const llama_xkv_cache_store & store,
        xkv_arena_lease * scratch = nullptr,
        uint64_t * call_counter = nullptr) const;

    const xkv_tri_config & get_config() const { return config; }
    const triattention_calibration & get_calibration() const { return cal; }

private:
    const triattention_calibration & cal;
    xkv_tri_config config;
    std::vector<float> omega;
    std::vector<float> freq_scale_sq;
    std::vector<float> offsets;
};

} // namespace llama_xkv
