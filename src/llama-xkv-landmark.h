#pragma once

#include "ggml.h"
#include "llama-cparams.h"
#include "llama-rerot.h"
#include "llama-xkv-cache.h"
#include "llama-xkv-codec.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llama_xkv {

// Key identifying a legally valid, homogeneous landmark fragment
struct landmark_fragment_key {
    uint64_t segment_id = 0;
    uint64_t segment_version = 0;
    uint64_t storage_generation = 0;
    uint32_t owning_layer = 0;
    uint32_t kv_head = 0;
    uint64_t live_epoch = 0;
    uint64_t content_epoch = 0;
    uint64_t codec_epoch = 0;
    uint64_t view_topology_epoch = 0;
    uint64_t view_publish_epoch = 0;
    uint64_t view_layout_epoch = 0;
    uint64_t live_set_fingerprint = 0;        // Deterministic fingerprint of live payloads & storage generations
    uint64_t phase_tx_fingerprint = 0;       // Deterministic fingerprint of RoPE / phase transform
    uint64_t binding_epoch = 0;              // Physical-only moves / rebinding epoch at build time
    uint64_t source_fingerprint = 0;         // Immutable segment source/profile/config fingerprint at build time
    uint64_t landmark_codec_fp = 0;          // Fingerprint of the landmark codec descriptor (set at encode time)
    uint32_t run_id = 0;
    int64_t  storage_pos0 = 0;
    int64_t  phase_delta = 0;                // storage_base - virtual_base for DDVR
    uint32_t ddvr_group = 0;                 // explicit reader/DDVR group slot (§9.3)
    uint8_t  visibility = 0;                 // llama_rerot_visibility
    int64_t  causal_cutoff = -1;             // -1 means no causal cutoff, otherwise max visible storage_pos

    bool operator==(const landmark_fragment_key & o) const {
        return segment_id == o.segment_id &&
               segment_version == o.segment_version &&
               storage_generation == o.storage_generation &&
               owning_layer == o.owning_layer &&
               kv_head == o.kv_head &&
               live_epoch == o.live_epoch &&
               content_epoch == o.content_epoch &&
               codec_epoch == o.codec_epoch &&
               view_topology_epoch == o.view_topology_epoch &&
               view_publish_epoch == o.view_publish_epoch &&
               view_layout_epoch == o.view_layout_epoch &&
               live_set_fingerprint == o.live_set_fingerprint &&
               phase_tx_fingerprint == o.phase_tx_fingerprint &&
               binding_epoch == o.binding_epoch &&
               source_fingerprint == o.source_fingerprint &&
               landmark_codec_fp == o.landmark_codec_fp &&
               run_id == o.run_id &&
               storage_pos0 == o.storage_pos0 &&
               phase_delta == o.phase_delta &&
               ddvr_group == o.ddvr_group &&
               visibility == o.visibility &&
               causal_cutoff == o.causal_cutoff;
    }
    bool operator!=(const landmark_fragment_key & o) const {
        return !(*this == o);
    }
};

struct landmark_fragment_key_hash {
    size_t operator()(const landmark_fragment_key & k) const noexcept {
        uint64_t h = 0xcbf29ce484222325ULL;
        auto mix = [&h](uint64_t v) {
            h ^= v;
            h *= 0x100000001b3ULL;
        };
        mix(k.segment_id);
        mix(k.segment_version);
        mix(k.storage_generation);
        mix(k.owning_layer);
        mix(k.kv_head);
        mix(k.live_epoch);
        mix(k.content_epoch);
        mix(k.codec_epoch);
        mix(k.view_topology_epoch);
        mix(k.view_publish_epoch);
        mix(k.view_layout_epoch);
        mix(k.live_set_fingerprint);
        mix(k.phase_tx_fingerprint);
        mix(k.binding_epoch);
        mix(k.source_fingerprint);
        mix(k.landmark_codec_fp);
        mix(k.run_id);
        mix(static_cast<uint64_t>(k.storage_pos0));
        mix(static_cast<uint64_t>(k.phase_delta));
        mix(static_cast<uint64_t>(k.ddvr_group));
        mix(static_cast<uint64_t>(k.visibility));
        mix(static_cast<uint64_t>(k.causal_cutoff));
        return static_cast<size_t>(h);
    }
};

// Row metadata needed for fragment partitioning and reader legality check
struct row_meta {
    uint32_t segment_row = 0;
    uint64_t payload_id = 0;
    uint64_t storage_generation = 0;
    uint32_t run_id = 0;
    int64_t  storage_pos = 0;
    int64_t  virtual_pos = 0;
    uint32_t ddvr_group = 0;                 // explicit reader/DDVR group slot
    uint8_t  visibility = 0;
    bool     is_live = true;
    bool     reader_visible = true; // explicitly verified reader-visible
};

// Legal fragment descriptor and its encoded landmark code stream
struct legal_fragment {
    landmark_fragment_key key;

    uint32_t row_begin = 0;
    uint32_t row_count = 0;
    std::vector<uint32_t> row_indices;       // row indices in segment A
    std::vector<int64_t>  storage_positions; // storage positions per row
    std::vector<uint64_t> payload_ids;       // stable payload IDs per row
    std::vector<uint64_t> generations;       // storage generations per row

    // Independent quantized landmark code stream (Q8_0 or Turbo4_0)
    encoded_matrix landmark_matrix;

    // Conservative error bound epsilon on landmark vector: |q^T (L_hat - L)| <= ||q||_2 * error_bound
    float error_bound = 0.0f;

    // Source fingerprint derived from the final factor K and RoPE configuration
    uint64_t source_fingerprint = 0;

    size_t allocated_bytes() const {
        size_t b = sizeof(*this);
        b += row_indices.capacity() * sizeof(uint32_t);
        b += storage_positions.capacity() * sizeof(int64_t);
        b += payload_ids.capacity() * sizeof(uint64_t);
        b += generations.capacity() * sizeof(uint64_t);
        b += landmark_matrix.bytes.capacity();
        return b;
    }
    // Immutable stored-base reference (group.landmark row) for intact original chunks.
    // When attached, selectors decode base_landmark[base_landmark_row] instead of the
    // duplicate landmark_matrix bytes (cleared on attach). Shared ownership keeps the
    // base alive across fragment deletion tombstones; derived (partial) fragments never
    // attach. allocated_bytes() excludes base bytes: they are billed to the owning
    // segment/group, and epoch-keyed table invalidation drops derived entries only.
    std::shared_ptr<const encoded_matrix> base_landmark;
    uint32_t base_landmark_row = 0;
    bool uses_base_landmark() const { return base_landmark != nullptr; }

    // Per-query visibility mask: size n_queries, true iff fragment is visible to query q
    std::vector<bool> query_visibility;
    // Explicit parent query ID for §9.3 per-query fragment plans (UINT32_MAX if shared/unconstrained)
    uint32_t parent_query_id = UINT32_MAX;
    // Explicit parent DDVR group ID
    uint32_t parent_group_id = 0;

    // Derived partial flag and device native rebuild marker (§9.3)
    bool is_derived_partial = false;
    bool requires_native_rebuild = false;
};

// Query for Selective Reconstruction (SR)
struct sr_query {
    uint32_t query_index = 0;
    int64_t  query_virtual_pos = 0;
    int64_t  causal_limit_pos = -1; // -1 means no causal cutoff, otherwise max visible storage_pos

    // Query vector for the layer/head group: float array of head_dim * n_q_heads
    std::vector<float> q_vec;
    // Indices of query heads sharing this KV head (GQA group)
    std::vector<uint32_t> q_head_indices;
    // Dimension of one head
    uint32_t head_dim = 0;

    // Attention scale factor (e.g. 1.0f / sqrt(head_dim))
    float scale = 0.0f;
};

// Configuration for SR selection
// Internal fault stage for landmark rescore failure injection (§15.4).
enum class landmark_fault_stage : uint8_t {
    none = 0,          // no injection (production default)
    refine_rescore = 1 // fail one boundary-refine rebuild event
};
// Config-scoped injection control. Plain data, default none.
struct landmark_fault_inject {
    landmark_fault_stage stage = landmark_fault_stage::none;
    uint32_t occurrence = 0; // 0-based rebuild-event index that fails
};
struct sr_selection_config {
    uint32_t sr_budget = 0; // Number of chunks or fragments to select
    ggml_type landmark_type = GGML_TYPE_Q8_0;
    llama_xkv_landmark_refine refine_mode = LLAMA_XKV_LANDMARK_REFINE_NONE;
    uint32_t refine_max_rows = 64;
    // Internal failure injection for landmark rescore (§15.4). Config-scoped test-only
    // control: default none; never enters fingerprints, keys, descriptors, or persistent
    // state; no globals, callbacks, or allocations. Exact field for FaultMatrix:
    // `sr_selection_config::fault` of type `landmark_fault_inject`.
    // {stage=none} disables. {stage=refine_rescore, occurrence=k} fails the k-th
    // (0-based) boundary-refine rebuild within one selection, before any caller
    // output/count/cache mutation (bounded: counts stay zero, payloads untouched;
    // reference: throws before result emission). Occurrences beyond the executed
    // rebuilds never trigger.
    landmark_fault_inject fault;

    // Mandatory protection sets as unambiguous segment_row_refs
    std::vector<segment_row_ref> hot_rows;
    std::vector<segment_row_ref> recent_rows;
    std::vector<segment_row_ref> outlier_rows;
};

// Result of SR selection for a single query
struct sr_selection_result {
    std::vector<segment_row_ref> selected_rows; // Deduplicated, sorted row refs
    std::vector<float>           scores;        // Landmark scores for selected fragments / chunks
    uint32_t rows_refined = 0;
    bool refine_cap_hit = false;
};

// Result of SR selection for a batch of queries with CSR membership
struct sr_batch_selection_result {
    std::vector<segment_row_ref> gather_rows; // Unique union of all selected row refs across queries
    std::vector<uint32_t> csr_ptrs;           // Size n_queries + 1
    std::vector<uint32_t> csr_indices;        // Indices into gather_rows for each query
    // Effective group per CSR entry, parallel to csr_indices (UINT32_MAX = invalid /
    // untracked). The allocating reference batch fills invalid; the production bounded
    // batch fills each entry's effective group from the source fragment view, so an
    // expanded per-(query,group) selection merges back without losing the group.
    std::vector<uint32_t> csr_group_indices;

    std::vector<sr_selection_result> per_query;

    uint32_t total_refined_rows = 0;
    uint32_t total_refine_cap_hits = 0;
};

// Dynamic cache table for legal fragments with deterministic LRU eviction, external lease awareness, and hard byte bounds
class landmark_table {
public:
    explicit landmark_table(size_t max_bytes = 64 * 1024 * 1024);
    ~landmark_table() = default;

    // Published/table getters return const fragments
    std::shared_ptr<const legal_fragment> find(const landmark_fragment_key & key);
    bool insert(std::shared_ptr<const legal_fragment> frag);

    void invalidate_segment(uint64_t segment_id);
    void invalidate_epoch(uint64_t segment_id, uint64_t live_epoch);
    void invalidate_content_epoch(uint64_t segment_id, uint64_t content_epoch);
    void invalidate_codec_epoch(uint64_t segment_id, uint64_t codec_epoch);
    void invalidate_stamp(const xkv_snapshot_stamp & current_stamp);
    void clear();
    // Binding/view/phase/source invalidation: any mismatch against the entry key
    // (binding epoch, full RERoT view stamp, phase-tx fingerprint, segment source
    // fingerprint) evicts the entry. Phase/source invalidation is segment-scoped.
    void invalidate_binding_epoch(uint64_t segment_id, uint64_t binding_epoch);
    void invalidate_view(uint64_t segment_id, const xkv_snapshot_stamp & current_stamp);
    void invalidate_phase_source(uint64_t segment_id, uint64_t phase_tx_fingerprint,
        uint64_t source_fingerprint);
    // Reject-or-evict helper: returns false when frag must not be cached (stale stamp,
    // phase/source mismatch, or oversize); performs atomic preflight before mutation.
    bool can_cache(const legal_fragment & frag, const xkv_snapshot_stamp & current_stamp,
        uint64_t phase_tx_fingerprint, std::string * err = nullptr) const;

    size_t total_allocated_bytes() const;
    size_t fragment_count() const;

private:
    void evict_if_needed();

    mutable std::mutex mutex_;
    size_t max_bytes_;
    size_t current_bytes_ = 0;

    // Deterministic LRU list of keys (front = most recently used, back = least recently used)
    std::list<landmark_fragment_key> lru_order_;

    struct table_entry {
        std::shared_ptr<const legal_fragment> frag;
        std::list<landmark_fragment_key>::iterator lru_it;
    };
    std::unordered_map<landmark_fragment_key, table_entry, landmark_fragment_key_hash> table_;
};

// Phase transform callback: given a pre-RoPE key vector and storage position, produces post-RoPE vector
using phase_transform_fn = std::function<void(const float * src_key, int64_t storage_pos, uint32_t head_idx, float * dst_key)>;

// 1. Build legal fragments from a segment given row metadata, chunk size, snapshot stamp, and causal/layout constraints
std::vector<legal_fragment> build_legal_fragments(
    const xkv_segment & segment,
    const xkv_snapshot_stamp & stamp,
    uint64_t segment_version,
    uint64_t default_storage_generation,
    uint32_t owning_layer,
    uint32_t kv_head,
    const std::vector<row_meta> & rows,
    uint32_t chunk_size,
    uint64_t phase_tx_fingerprint = 0,
    int64_t causal_cutoff = -1
);

// 2. Encode fragment landmark: reconstructs final factor K for fragment rows, decodes ONLY exact B^T feature slice [feature_offset, feature_offset + feature_dim),
// applies phase transform (storage-position RoPE), aggregates landmark vector (mean pooling), encodes independent Q8 or Turbo4 stream,
// computes conservative error bounds and fingerprint. Turbo4 logical feature dimensions are allowed with standard block padding.
// Unsupported landmark type throws std::invalid_argument.
void encode_fragment_landmark(
    legal_fragment & frag,
    const xkv_segment & segment,
    uint32_t feature_offset,
    uint32_t feature_dim,
    const phase_transform_fn & phase_tx,
    uint64_t phase_tx_fingerprint,
    ggml_type landmark_type = GGML_TYPE_Q8_0
);

// Overload accepting factor group inside segment bundle
void encode_fragment_landmark(
    legal_fragment & frag,
    const xkv_segment & segment,
    const xkv_factor_group_payload & group,
    uint32_t feature_offset,
    uint32_t feature_dim,
    const phase_transform_fn & phase_tx,
    uint64_t phase_tx_fingerprint,
    ggml_type landmark_type = GGML_TYPE_Q8_0
);

// 3. Score legal fragments against a query with per-Q-head z-score normalization across all candidates before max pooling,
// global top-k across legal DDVR spans, hot/recent/outlier union + dedup, single-query causal enforcement/rebuild, and optional interval refinement.
sr_selection_result select_sr_query(
    const sr_query & query,
    const std::vector<legal_fragment> & legal_frags,
    const sr_selection_config & config,
    const xkv_segment * segment = nullptr, // optional segment for boundary refinement or causal rebuild
    uint32_t feature_offset = 0,
    uint32_t feature_dim = 0,
    const phase_transform_fn & phase_tx = nullptr,
    uint64_t phase_tx_fingerprint = 0
);

// 4. Batch SR selection across multiple queries: guarantees multi-query isolation (future draft queries never affect prior selections),
// re-encodes partial-causal fragment landmarks from only legal rows, performs gather union and builds per-query CSR membership.
sr_batch_selection_result select_sr_batch(
    const std::vector<sr_query> & queries,
    const std::vector<legal_fragment> & legal_frags,
    const sr_selection_config & config,
    const xkv_segment * segment = nullptr,
    uint32_t feature_offset = 0,
    uint32_t feature_dim = 0,
    const phase_transform_fn & phase_tx = nullptr,
    uint64_t phase_tx_fingerprint = 0
);
// Production-bounded landmark workspace: zero-heap query path over caller-owned memory.
// The allocating functions above remain as the small CPU-reference convenience wrapper.
struct landmark_workspace_config {
    uint32_t max_fragments = 0;
    uint32_t max_rows_per_fragment = 0;
    uint32_t max_rows_total = 0;
    uint32_t max_queries = 0;
    uint32_t max_q_heads_per_query = 0;
    uint32_t max_head_dim = 0;
    uint32_t max_protected_rows = 0;
    uint32_t max_padded_rank = 0;
};
struct landmark_fragment_view {
    const landmark_fragment_key * key = nullptr;
    uint32_t row_count = 0;
    const uint32_t * row_indices = nullptr;
    const int64_t * storage_positions = nullptr;
    const uint64_t * payload_ids = nullptr;
    const uint64_t * generations = nullptr;
    const encoded_matrix * landmark = nullptr;
    float error_bound = 0.0f;
    const encoded_matrix * base_landmark = nullptr; // non-null iff attached intact chunk
    uint32_t base_landmark_row = 0;
    // Effective DDVR/group identity for this fragment's rows (UINT32_MAX = unspecified,
    // mirroring xkv_segment_read_view). Carried verbatim per row into selection outputs so
    // an expanded per-(query,group) selection merges back without losing the group.
    uint32_t group_index = UINT32_MAX;
};
inline landmark_fragment_view view_of_fragment(const legal_fragment & f) {
    landmark_fragment_view v;
    v.key = &f.key;
    v.row_count = f.row_count;
    v.row_indices = f.row_indices.data();
    v.storage_positions = f.storage_positions.data();
    v.payload_ids = f.payload_ids.data();
    v.generations = f.generations.empty() ? nullptr : f.generations.data();
    v.landmark = &f.landmark_matrix;
    v.error_bound = f.error_bound;
    v.base_landmark = f.base_landmark ? f.base_landmark.get() : nullptr;
    v.base_landmark_row = f.base_landmark_row;
    // Explicit per-(parent, slot) assignment rides through verbatim (slot 0
    // included); shared/unconstrained fragments stay UINT32_MAX
    // (unspecified), matching the reference batch which fills invalid.
    v.group_index = (f.parent_query_id != UINT32_MAX) ? f.parent_group_id : UINT32_MAX;
    return v;
}
class landmark_workspace {
public:
    landmark_workspace() = default;
    ~landmark_workspace() = default;
    landmark_workspace(const landmark_workspace &) = delete;
    landmark_workspace & operator=(const landmark_workspace &) = delete;
    bool init(const landmark_workspace_config & cfg, std::string * err = nullptr);
    void clear();
    bool valid() const { return !buffer_.empty() && align_off_ < buffer_.size(); }
    size_t capacity_bytes() const { return buffer_.empty() ? 0 : buffer_.size() - align_off_; }
    uint8_t * data() { return buffer_.empty() ? nullptr : buffer_.data() + align_off_; }
    const uint8_t * data() const { return buffer_.empty() ? nullptr : buffer_.data() + align_off_; }
    const landmark_workspace_config & config() const { return cfg_; }
private:
    landmark_workspace_config cfg_;
    std::vector<uint8_t> buffer_;
    // Start offset of the 64-byte aligned usable region inside buffer_ (buffer is
    // over-allocated by 63 bytes once at init; all float/uint64 carves use data()).
    size_t align_off_ = 0;
};
bool landmark_select_scratch_bytes(uint32_t n_frags, uint32_t n_q_heads, uint32_t head_dim,
    uint32_t max_rows_per_fragment, uint32_t padded_rank, size_t & out_bytes,
    std::string * err = nullptr);
bool landmark_encode_scratch_bytes(uint32_t feature_dim, uint32_t padded_rank, uint32_t row_count,
    size_t & out_bytes, std::string * err = nullptr);
bool landmark_workspace_required_bytes(const landmark_workspace_config & cfg, size_t & out_bytes,
    std::string * err = nullptr);
struct landmark_fragment_plan {
    landmark_fragment_key key;
    uint32_t row_begin = 0;
    uint32_t row_count = 0;
    uint32_t rows_offset = 0;
};
// Cross-segment attribution for global SR selection over concatenated fragments.
// Per-row source identity (parallel to the final selected/gather row order):
// - row_frag_index[k]: src-index of the source fragment in the caller's concat order.
// - row_frag_key[k]: EFFECTIVE fragment key; partial-causal rows carry the base key
//   with causal_cutoff overridden to the selecting query's cutoff.
// Per-segment CSR over the final row order (distinct segment id/version, first
// appearance; indices address the final row array). Lets a global top-k/refine
// attribute exact rows across segments. All caller-owned; null struct = no attribution.
struct sr_bounded_attribution {
    uint32_t * row_frag_index = nullptr;
    landmark_fragment_key * row_frag_key = nullptr;
    size_t rows_cap = 0;
    uint64_t * seg_ids = nullptr;
    uint64_t * seg_versions = nullptr;
    size_t segs_cap = 0;
    size_t * out_n_segs = nullptr;
    uint32_t * seg_ptrs = nullptr;
    uint32_t * seg_indices = nullptr;
    size_t seg_idx_cap = 0;
    size_t * out_n_seg_idx = nullptr;
    // Caller contract: seg_ptrs holds segs_cap + 1 slots (final sentinel included),
    // seg_indices holds seg_idx_cap slots, row arrays hold rows_cap slots. Any shortfall
    // fails closed. Partial-causal rebuilds resolve the owning segment per fragment via
    // an exact (segment_id, segment_version) table (new trailing params); misses fail
    // closed instead of silently reusing a foreign segment's factors. Budget and refine
    // stay global across all concatenated fragments (no per-segment split).
};
bool build_legal_fragment_plans(const xkv_segment & segment, const xkv_snapshot_stamp & stamp,
    uint64_t segment_version, uint64_t default_storage_generation, uint32_t owning_layer,
    uint32_t kv_head, const row_meta * rows, size_t n_rows, uint32_t chunk_size,
    uint64_t phase_tx_fingerprint, int64_t causal_cutoff, landmark_fragment_plan * out_plans,
    size_t plans_cap, size_t * out_n_plans, uint32_t * out_row_indices, int64_t * out_positions,
    uint64_t * out_payload_ids, uint64_t * out_generations, size_t rows_cap, size_t * out_n_rows,
    std::string * err = nullptr);
// Attach an intact original chunk to its immutable stored-base summary row
// (group.landmark[base_row]). Self-verifying, init-time only (may allocate decode temps):
// proves exact epoch match (live/content/codec/binding/view), phase match, intact row
// identity (payload/generation/position equality against the chunk mapping supplied by
// the sealer/store), codec-type and feature-width match, and BITWISE decoded-value
// equality with the fragment's own landmark row. On success the duplicate
// landmark_matrix bytes are released. Partial (visibility/live/causal) fragments must
// never attach: any row mismatch fails closed. Deleting the fragment keeps the shared
// base alive (tombstone-safe); table invalidation drops derived entries by key epochs.
bool attach_base_landmark(legal_fragment & frag, std::shared_ptr<const encoded_matrix> base,
    uint32_t base_row, const xkv_snapshot_stamp & stamp, uint64_t phase_tx_fingerprint,
    uint64_t chunk_source_fingerprint,
    const uint64_t * chunk_payload_ids, const uint64_t * chunk_generations,
    const int64_t * chunk_positions, size_t chunk_n, std::string * err = nullptr);
// Immutable stored-base summary table (group.landmark) for runtime binding.
// The sealer persists, alongside group.landmark ([n_chunks x dim] over the sealed rows):
// per-chunk row identity (flat payload/generation/position arrays cut by row_offsets),
// the per-chunk conservative quant-error bounds produced at seal (builder outputs —
// they must persist here, not be dropped when chunk outputs are concatenated), and the
// stamp + phase fingerprint the base was sealed under. All pointers + counts, no ownership
// except the shared landmark handle. The store/sealer owns persistence; this module only
// validates and references.
struct landmark_base_table {
    std::shared_ptr<const encoded_matrix> landmark;
    const uint64_t * row_payload_ids = nullptr;  // flat, length n_rows_total
    const uint64_t * row_generations = nullptr;  // flat, length n_rows_total
    const int64_t * row_positions = nullptr;     // flat storage positions
    const float * chunk_error_bounds = nullptr;  // per chunk, length n_chunks
    const uint64_t * chunk_source_fingerprints = nullptr; // exact final-factor/phase source per chunk
    const uint32_t * chunk_row_offsets = nullptr; // length n_chunks + 1, [0] == 0
    size_t n_rows_total = 0;
    size_t n_chunks = 0;
    xkv_snapshot_stamp stamp;                    // seal-time epochs
    uint64_t phase_tx_fingerprint = 0;           // seal-time phase identity
    // Expected closure fingerprint over the whole table (offsets, bound bits, source fingerprints,
    // row identity, counts, stamp epochs, phase fp, landmark desc fingerprint).
    // Computed by compute_base_table_fingerprint; the binder requires nonzero and
    // an exact match, so no persisted field can be tampered independently.
    uint64_t bounds_fingerprint = 0;
};
// Deterministic FNV-1a closure fingerprint for a base table: chunk offsets,
// error-bound bit patterns, row identity arrays, counts, all 7 stamp epochs, phase
// fingerprint, and the landmark descriptor fingerprint, in that order. The state
// envelope must persist this exact value; the binder recomputes and requires a match.
// Null mappings or a null landmark yield a defined (non-matching, unless the table is
// empty in exactly the fingerprinted way) value — the binder still fails closed on
// structural defects before comparing.
uint64_t compute_base_table_fingerprint(const landmark_base_table & base);
// Bounded runtime binder: bind each fragment that exactly matches an intact base chunk
// (row count + elementwise payload/generation/position equality, exact key epochs vs the
// table stamp, phase match, sane base descriptor). Bound fragments reference the table
// row, adopt its persisted quant-error bound and codec fingerprint, and keep no duplicate
// bytes — so SR never sees empty landmark data for intact chunks. Fragments with own
// encoded bytes are left untouched; stale/partial fragments are left unbound (not an
// error) for scratch rebuild; structural table defects fail closed. Zero heap.
// Returns the bound count via out_n_bound (0 is success with nothing intact).
bool bind_base_landmarks(legal_fragment * frags, size_t n_frags,
    const landmark_base_table & base, size_t * out_n_bound, std::string * err = nullptr);
// ------------------------------------------------------------------------------------------------
// Production bounded landmark builder (§8.6/8.7): seal-time construction from FINAL
// encoded streams with no full decoded factor/KV/landmark mirror.
//
// Seal-time construction consumes final encoded A_K/B_K plus spans and streams only
// required A rows and current B rank tiles into a caller arena span — no full decoded
// factor/KV/landmark mirror is ever materialized on this path. K landmarks need no V
// inputs (no V params exist).
// Chunk payload encoding is caller-injected, but the module ships a production zero-heap
// chunk encoder below supporting F32, Q8_0, and Turbo4_0 with exact codec_desc, row
// padding, seed, and fingerprint behavior bit-identical to encode_matrix. A null encoder
// fails closed, as does any other landmark type. Encoders must support a size query
// (out == nullptr: set *out_n/*out_desc, write nothing) and must not heap-allocate on
// the encode path. The builder measures the conservative error bound itself by decoding
// the just-written bytes (zero-heap), so encoders report only a provisional bound.
// ------------------------------------------------------------------------------------------------
// Chunk payload encoder: mean row (dim floats) -> code stream bytes.
typedef bool (*landmark_chunk_encoder_fn)(const float * mean, uint32_t dim, ggml_type type,
    uint64_t seed, uint8_t * out, size_t out_cap, size_t * out_n, codec_desc * out_desc,
    float * out_error_bound, std::string * err);
// Zero-heap F32 row encoder (bit-exact memcpy; oracle/tests/F32 profiles).
bool landmark_encode_f32_row(const float * mean, uint32_t dim, ggml_type type, uint64_t seed,
    uint8_t * out, size_t out_cap, size_t * out_n, codec_desc * out_desc,
    float * out_error_bound, std::string * err = nullptr);
// Production zero-heap chunk encoder for F32, Q8_0, and Turbo4_0. Same signature and
// size-query contract as the injected type above, so RuntimeSealer can pass it straight
// into build_landmarks_bounded: full 32-/128-element groups quantize directly from the
// caller mean row, the zero-padded tail stages through fixed alignas(64) stack buffers,
// and Turbo output uses per-group memcpy (misaligned-dst safe). Output bytes and the
// descriptor (including seed-derived table fingerprint) are bit-identical to
// encode_matrix for the same inputs. No heap allocation on any path; all other types
// fail closed (including Turbo2/3: Turbo3 reads process-global WHT state).
// Each written block/group is canonical-decoded back from the output bytes into fixed
// stack scratch and accumulated into a deterministic conservative L2 error bound
// (|q^T(L_hat-L)| <= ||q||_2 * bound) with the 1.10 margin and 1e-4 floor convention;
// F32 reports exactly 0.0, size queries report 0.0 (the builder always re-measures
// authoritatively post-encode).
bool landmark_encode_quant_row(const float * mean, uint32_t dim, ggml_type type, uint64_t seed,
    uint8_t * out, size_t out_cap, size_t * out_n, codec_desc * out_desc,
    float * out_error_bound, std::string * err = nullptr);
// One owning-layer slice of the concatenated landmark row.
struct landmark_build_layer {
    uint32_t owning_layer = 0;
    uint32_t feature_offset = 0; // B_K slice offset for this layer
    uint32_t feature_dim = 0;    // B_K slice width (head_dim multiple)
    uint32_t kv_head = 0;        // phase identity for this slice
};
// Seal-time build spec over final encoded streams (all pointers + counts, no ownership).
struct landmark_build_spec {
    const encoded_matrix * a_k = nullptr;    // final encoded A_K, token-major
    const encoded_matrix * b_k = nullptr;    // final encoded B_K, feature-major-transposed
    const uint32_t * surviving_rows = nullptr; // A row indices, length n_rows
    size_t n_rows = 0;
    const int64_t * storage_positions = nullptr; // parallel to surviving_rows
    const landmark_build_layer * layers = nullptr; // contiguous slices from offset 0
    size_t n_layers = 0;
    uint32_t chunk_tokens = 0;               // rows per output landmark chunk
    ggml_type landmark_type = GGML_TYPE_Q8_0;
    uint64_t phase_tx_fingerprint = 0;       // 0 = identity phase sentinel
    uint64_t seed_extra = 0;                 // immutable source/profile/config mix-in
};
// Per-chunk output descriptor into the caller arena.
struct landmark_build_output {
    codec_desc desc;
    size_t byte_offset = 0;
    size_t byte_size = 0;
    float error_bound = 0.0f;
    uint64_t source_fingerprint = 0; // content mix over phased rows of this chunk
};
// Exact checked estimators: workspace scratch, arena bytes, chunk count. False on any
// invalid input or integer overflow. Arena sizing queries the encoder (size-query mode).
bool landmark_build_workspace_bytes(const landmark_build_spec & spec,
    landmark_chunk_encoder_fn encoder, size_t & out_workspace, size_t & out_arena,
    size_t & out_n_chunks, std::string * err = nullptr);
// Production bounded builder: stream-decodes required A rows + current B rank tiles,
// reconstructs + phases each row before chunk-local mean pooling, encodes each chunk
// into caller arena slices. No heap after init; fails closed on any shortfall/mismatch.
bool build_landmarks_bounded(const landmark_build_spec & spec, const phase_transform_fn & phase_tx,
    landmark_chunk_encoder_fn encoder, uint8_t * scratch, size_t scratch_bytes, uint8_t * arena,
    size_t arena_bytes, landmark_build_output * out_chunks, size_t chunks_cap,
    size_t * out_n_chunks, std::string * err = nullptr);
bool select_sr_query_bounded(const sr_query & query, const landmark_fragment_view * frags,
    size_t n_frags, const sr_selection_config & config, const xkv_segment * segment,
    uint32_t feature_offset, uint32_t feature_dim, const phase_transform_fn & phase_tx,
    uint64_t phase_tx_fingerprint, uint8_t * scratch, size_t scratch_bytes,
    segment_row_ref * out_rows, size_t rows_cap, size_t * out_n_rows, float * out_scores,
    size_t scores_cap, size_t * out_n_scores, uint32_t * out_rows_refined, bool * out_cap_hit,
    std::string * err, const xkv_segment * const * segments = nullptr,
    size_t n_segments = 0, sr_bounded_attribution * attrib = nullptr,
    uint32_t * out_row_groups = nullptr, size_t row_groups_cap = 0);
bool select_sr_batch_bounded(const sr_query * queries, size_t n_queries,
    const landmark_fragment_view * frags, size_t n_frags, const sr_selection_config & config,
    const xkv_segment * segment, uint32_t feature_offset, uint32_t feature_dim,
    const phase_transform_fn & phase_tx, uint64_t phase_tx_fingerprint, uint8_t * scratch,
    size_t scratch_bytes, segment_row_ref * out_gather, size_t gather_cap, size_t * out_n_gather,
    uint32_t * out_csr_ptrs, size_t csr_ptrs_cap, uint32_t * out_csr_indices, size_t csr_cap,
    size_t * out_n_csr, uint32_t * out_total_refined, uint32_t * out_total_cap_hits,
    std::string * err, const xkv_segment * const * segments = nullptr,
    size_t n_segments = 0, sr_bounded_attribution * attrib = nullptr,
    uint32_t * out_csr_group_indices = nullptr, size_t csr_group_cap = 0);

} // namespace llama_xkv
