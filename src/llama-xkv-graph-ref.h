#pragma once

#include <algorithm>
#include <cstring>

#include "ggml.h"
#include "llama-xkv-cache.h"
#include "llama-xkv-reader.h"
#include "llama-xkv-landmark.h"
#include "llama-xkv-transaction.h"

#include "ggml-xkv.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace llama_xkv {

// Physical hot storage layout descriptor (filled by the snapshot builder).
// Defined before xkv_graph_snapshot: the snapshot holds it by value.
struct xkv_hot_storage_layout {
    ggml_type k_type = GGML_TYPE_F32;
    ggml_type v_type = GGML_TYPE_F32;
    uint32_t head_dim_k = 0; // logical (unpadded) K width per head
    uint32_t head_dim_v = 0; // logical (unpadded) V width per head
    uint32_t padded_k = 0;   // storage width per head (0 = head_dim_k)
    uint32_t padded_v = 0;   // storage width per head (0 = head_dim_v)
    // Transposed V layouts are unsupported and rejected before allocation.
    bool v_transposed = false;

    uint32_t eff_padded_k() const { return padded_k != 0 ? padded_k : head_dim_k; }
    uint32_t eff_padded_v() const { return padded_v != 0 ? padded_v : head_dim_v; }
};

// Combined zero-heap callback layout (P0-WS). One snapshot-owned lease covers
// callback temps (sink, hot bulk, dequant, gather outputs, rot tmp) plus the
// reader subspan plus accounted struct estimates, as nonoverlapping 64-aligned
// slices of a single backing buffer. All math checked; overflow fails.
struct xkv_callback_layout {
    size_t total_bytes = 0;
    size_t sink_off = 0, sink_bytes = 0;
    size_t bulk_k_off = 0, bulk_k_bytes = 0;
    size_t bulk_v_off = 0, bulk_v_bytes = 0;
    size_t deq_off = 0, deq_bytes = 0;
    size_t gather_off = 0, gather_bytes = 0;
    size_t rot_off = 0, rot_bytes = 0;
    size_t reader_off = 0, reader_bytes = 0;
    size_t sr_off = 0, sr_bytes = 0; // bounded-selector scratch (XkvLandmarkBounded)
};

// Pure layout computation for the combined lease. Bulk regions reserve
// n_gather * cell stride (upper bound on any [min,max] span). Query/hot/gather
// descriptors are runtime-preallocated snapshot metadata (see
// preallocate_compute_state), sized from the same build-known dims — never
// heap-allocated inside compute.
bool xkv_compute_callback_layout(uint32_t n_gather, size_t cell_stride_k, size_t cell_stride_v,
    size_t nk_floats, size_t nv_floats, uint32_t head_dim_k, uint32_t head_dim_v, uint32_t pad_max,
    size_t sink_floats, size_t reader_bytes, size_t sr_scratch_bytes,
    xkv_callback_layout & out, std::string * err = nullptr);

// Managed CPU-reference graph snapshot for a single attention layer compute pass.
// Owns:
//  - Copied bounded hot K/V row storage so hot cache modifications cannot race during compute.
//  - Move-only segment pins and read views ensuring immutable segment lifetime for callback duration.
//  - Exact expected XKV stamp.
//  - DDVR query/group metadata and effective query configurations.
//  - SR CSR selection mappings and gather structures.
//  - Phase transform callback (e.g. RoPE at storage position).
//  - Hard reader and workspace execution limits.
//  - Store pointer (for stamp verification and workspace arena lease).
class xkv_graph_snapshot {
public:
    xkv_graph_snapshot() = default;
    ~xkv_graph_snapshot() = default;

    // Move-only snapshot to guarantee single ownership and deterministic lifetime
    xkv_graph_snapshot(const xkv_graph_snapshot &) = delete;
    xkv_graph_snapshot & operator=(const xkv_graph_snapshot &) = delete;

    xkv_graph_snapshot(xkv_graph_snapshot &&) noexcept = default;
    xkv_graph_snapshot & operator=(xkv_graph_snapshot &&) noexcept = default;

    // Hot row backing store (copied bounded rows)
    struct hot_row_data {
        uint64_t payload_id = 0;         // Payload ID (optional/if tracked)
        uint32_t row_index = 0;
        int64_t storage_pos = 0;
        uint64_t storage_generation = 0; // Captured storage generation for validation
        xkv_state expected_state = xkv_state::hot_committed; // Expected storage state
        uint32_t group_index = 0;
        bool is_valid = true;
        std::vector<bool> query_visibility;
        std::vector<float> k_data; // [head_dim_k]
        std::vector<float> v_data; // [head_dim_v]

        bool is_visible_to_query(uint32_t query_idx) const {
            if (!is_valid) return false;
            if (query_visibility.empty()) return true;
            if (query_idx < query_visibility.size()) return query_visibility[query_idx];
            return false;
        }
        // Per-query group mapping, parallel to query_visibility (filled by
        // the snapshot builder): visible queries carry this row's exact DDVR
        // group, invisible slots stay UINT32_MAX. Ordinary path uses group 0.
        // Empty = legacy (all visible queries use group_index). Non-empty
        // must cover every query exactly (validated downstream); UINT32_MAX
        // on a visible query fails closed.
        std::vector<uint32_t> query_group_indices;
        // Source-gather metadata (bounded-hot graph wiring). cell >= 0 means
        // gather+dequantize the exact physical slot from the explicit storage
        // dependency tensors inside compute, after writes. cell < 0 keeps the
        // legacy test-only fallback using owned k_data/v_data above.
        int64_t cell = -1;    // physical cell row in storage view tensors
        uint32_t kv_head = 0; // KV head slice within the storage row
        uint32_t stream = 0;  // cache stream (row-space selector)

        bool use_storage_gather() const { return cell >= 0; }
    };

    std::vector<hot_row_data> hot_data;

    // Hard bounds on hot row storage (0 = unlimited / reader budget limit)
    size_t max_hot_rows = 0;
    size_t max_hot_bytes = 0;

    // Factored segment views with move-only pins
    std::vector<xkv_segment_read_view> segment_views;

    // SR batch selection result (CSR indices and gather rows)
    sr_batch_selection_result sr_selection;

    // Exact expected stamp
    xkv_snapshot_stamp expected_stamp;

    // Store pointer (optional, used for stamp verification and arena leasing)
    llama_xkv_cache_store * store = nullptr;

    // Move-owned coordinator reader lease: acquired before any cell/store
    // read, held through compute/postcompute. Blocks maintenance, final
    // fence, and context shift for this snapshot's lifetime; segment pins
    // alone protect bytes, not epochs/layout. Empty when no coordinator
    // exists (standalone use).
    xkv_transaction_coordinator::reader_lease reader_guard;

    // Phase transform function
    phase_transform_fn phase_tx;

    // Reader configuration and budget limits
    xkv_reader_config reader_config;

    // Head and query dimensions for this layer
    uint32_t head_dim_k = 0;
    uint32_t head_dim_v = 0;
    uint32_t n_q_heads  = 1;
    uint32_t n_queries  = 0;

    // Per-query parameters
    float scale = 0.0f;
    float logit_softcap = 0.0f;
    std::vector<int64_t> query_causal_limits; // Per-query causal cutoff storage position (-1 = none)
    std::vector<std::vector<float>> query_sink_logits; // [n_queries][n_q_heads]

    // DDVR group count per query (0 = single vector in q tensor, >0 = q tensor contains groups)
    uint32_t n_ddvr_groups = 0;

    // Per-query variable DDVR group counts (if non-empty, size must equal n_queries)
    std::vector<uint32_t> query_ddvr_group_counts;

    // Optional per-query element offsets in q_tensor data (if non-empty, size must equal n_queries)
    std::vector<size_t> query_ddvr_group_offsets;

    // ---- Bounded-hot graph wiring (per owning layer + KV head) ----
    uint32_t kv_head_index = 0;   // which KV head this snapshot covers
    uint32_t q_group_begin = 0;   // first global Q head index of this GQA group
    uint32_t n_q_heads_total = 0; // total Q heads (0 = n_q_heads, single group)
    xkv_hot_storage_layout hot_layout; // physical hot storage descriptor
    int k_storage_dep = -1; // dst->src index of K storage view tensor (-1 = none)
    int v_storage_dep = -1; // dst->src index of V storage view tensor (-1 = none)
    int sink_dep = -1;      // dst->src index of F32 sink tensor [n_q_total] or [n_q_total, n_queries]
    std::vector<float> hot_k_inv_rot; // row-major custom inverse attention rotation (empty = identity)
    uint32_t hot_k_rot_dim = 0;
    std::vector<float> hot_v_inv_rot; // row-major custom inverse attention rotation (empty = identity)
    uint32_t hot_v_rot_dim = 0;
    std::vector<float> hot_k_channel_mul; // per-channel post-rotation multiplier, size padded_k (empty = identity)
    std::vector<float> hot_v_channel_mul; // per-channel post-rotation multiplier, size padded_v (empty = identity)
    uint64_t expected_turbo_fp_k = 0; // compiled layout fingerprint guard, 0 = skip
    uint64_t expected_turbo_fp_v = 0; // compiled layout fingerprint guard, 0 = skip
    // ---- Snapshot-owned combined workspace (P0-WS) ----
    // Single lease owns ALL callback memory: workspace_base() addresses
    // lease.data() (+64-alignment offset). No separate byte vector exists.
    // Lease size must cover alignment slack + every live region. Production
    // bounded snapshots set workspace_required with a backing lease;
    // budget>0 without backing fails closed (never legacy heap). Default
    // snapshots (tests/legacy) leave it false and use the preflight-bounded
    // heap path. Builder acquires ONE store lease, binds the reader workspace
    // to the reader subspan via warmup_external, fills layout, and sizes the
    // descriptor caches below. Compute then allocates zero heap (outputs
    // excluded: reader-owned result vectors, accounted separately).
    bool workspace_required = false;
    xkv_arena_lease workspace_lease; // single lease: accounting + backing memory
    size_t workspace_base_off = 0; // 64-alignment offset into lease.data()
    xkv_callback_layout workspace_layout;
    std::shared_ptr<xkv_reader_workspace> workspace_reader; // bound to reader subspan
    // Runtime-preallocated immutable metadata (sized once at build by
    // preallocate_compute_state; compute only fills, never allocates).
    std::vector<xkv_query_input> query_cache;
    std::vector<xkv_hot_row> hot_cache;
    std::vector<int64_t> gather_cells_cache; // size n_gather, build-known cells
    std::vector<int64_t> gather_pos_cache;  // size n_hot, bulk positions (-1 skip)
    uint8_t * workspace_base() {
        if (!workspace_lease) return nullptr;
        uint8_t * d = static_cast<uint8_t *>(workspace_lease.data());
        return d ? d + workspace_base_off : nullptr;
    }
    const uint8_t * workspace_base() const {
        if (!workspace_lease) return nullptr;
        const uint8_t * d = static_cast<const uint8_t *>(workspace_lease.const_data());
        return d ? d + workspace_base_off : nullptr;
    }
    bool workspace_ready() const {
        if (!workspace_lease || workspace_layout.total_bytes == 0) return false;
        size_t end = 0;
        if (!safe_add(workspace_base_off, workspace_layout.total_bytes, end)) return false;
        return end <= workspace_lease.size();
    }
    // Size descriptor caches from build-known dims (query counts/groups/heads,
    // hot rows + visibility, gather cells). BUILD-time allocation only.
    // Returns false (with err) on overflow or inconsistent dims.
    bool preallocate_compute_state(std::string * err = nullptr);
    // Validate caches against current dims (compute calls when required).
    bool compute_caches_valid(std::string * err = nullptr) const;
    // Runtime refresh hook (installed by the snapshot builder): resolve fresh
    // cells/stamps/positions/visibility/causal/groups/pins within the
    // preallocated capacities recorded below. Invoked by graph set_input
    // after guard acquire; failure sets force_retry_stale (callback + status
    // report retry so the outer loop rebuilds). Absent = unrefreshable.
    std::function<bool(std::string &)> snapshot_refresh_fn;
    // Set by failed refresh; callback pre-check and status fns report stale
    // retry while set. Cleared by successful refresh or snapshot rebuild.
    bool force_retry_stale = false;
    // Capacity reservations recorded at build (upper bounds for refresh
    // growth; refresh beyond them fails closed and refuses reuse).
    struct xkv_snapshot_capacities {
        uint32_t n_queries_max = 0; // 0 = exact current n_queries
        uint32_t hot_max = 0;       // 0 = exact current hot rows
        uint32_t group_max = 0;     // 0 = exact current max groups/query
        uint32_t seg_max = 0;       // 0 = exact current segment views
    } capacities;
    // Coordinator reader guard hooks (installed by the snapshot builder when
    // the runtime owns a guard; empty = no guard installed). The graph/input
    // lifecycle invokes acquire at set_input start (fresh guard before any
    // reuse/refresh) and release after every poll path plus destruction, so
    // a cached graph can never pin maintenance forever. Segment pins are
    // untouched (COW-immutable).
    std::function<void()> guard_release_hook;
    std::function<bool(std::string &)> guard_acquire_hook;
    void release_reader_guard() {
        if (guard_release_hook) guard_release_hook();
    }
    bool acquire_reader_guard(std::string & err) {
        if (!guard_acquire_hook) return true; // no guard installed: vacuous
        return guard_acquire_hook(err);
    }
    // True when storage/sink dependency buffers are host-dereferenceable.
    // False forces explicit bounded backend subrange copies: one bulk copy
    // per K/V layer, i.e. two syncs total (K bulk + V bulk). Single-sync
    // needs async K/V queued on one backend handle (builder-threaded
    // follow-up). Defaults to false: fail-safe on accelerators.
    // The graph sets it from actual caps/backend; builders must not rely on
    // the default for device-resident storage.
    bool hot_storage_host_resident = false;

    // ---- SR-after-Q selection (SR mode runs selection in compute) ----
    // sr_mode 0: use precomputed sr_selection (dense/off path; dense streams
    //   all legal rows via segment_views). sr_mode 1: run select_sr_batch
    //   after Q is available, independently per query/head, and use the
    //   computed CSR instead of sr_selection.
    int sr_mode = 0;
    sr_selection_config sr_config; // budget/refine/landmark type (builder)
    std::vector<legal_fragment> sr_legal_frags; // legal landmark fragments
    uint32_t sr_feature_offset = 0; // exact B feature slice for landmarks
    uint32_t sr_feature_dim = 0;    // 0 = head_dim_k
    uint64_t sr_phase_fingerprint = 0;

    // ---- Native reconstruct arenas (ggml-xkv.h contract) ----
    // Backend-resident code-stream / refs / positions / group-meta tensors
    // travel as explicit deps (indices into dst->src); no per-decode upload.
    // Absent until the builder wires segment arenas: native build then fails
    // closed instead of falling back silently.
    bool native_arenas_present = false;
    // Borrowed backend-resident code streams for ONE (segment, factor group),
    // filled by the snapshot builder. All tensors must be allocated
    // (scheduler-visible buffers), immutable for the snapshot lifetime
    // (segment pins in segment_views cover lifetime), and use exact
    // codec-padded layouts (validated per dispatch against group metadata).
    // The graph builds only small per-build metadata (refs/positions/
    // group_meta/layer_meta/entries/offsets/sinks/status) in ctx0 and never
    // uploads code streams or retains dense mirrors. Empty = builder has not
    // wired arenas: native build fails closed with an explicit reason.
    struct xkv_native_group_arenas {
        uint64_t segment_id = 0;
        uint64_t segment_version = 0;
        uint32_t group_index = 0; // factor group within the segment
        ggml_tensor *a_k = nullptr;
        ggml_tensor *b_k = nullptr;
        ggml_tensor *a_v = nullptr;
        ggml_tensor *b_v = nullptr;
        ggml_tensor *landmarks = nullptr; // backend-resident landmark code stream for this arena
    };
    std::vector<xkv_native_group_arenas> native_group_arenas;
    // Model-constant rope tables (F32[2*Fc] omega/mag), borrowed shared table
    // uploaded once per backend (BackendResidency batch path), NOT per bundle.
    // Null iff rotary_dim==0 (graph substitutes an empty tensor).
    ggml_tensor *native_rope_tables = nullptr;
    // Exact bounded fallback when the runtime cannot yet cache a borrowed
    // backend tensor: [omega(Fc), direct_mag(Fc)] for this owning layer.
    // Graph upload is allowed only from these model-derived values; it never
    // invents/defaults a RoPE table. Empty when native_rope_tables is set or
    // rotary_dim is zero.
    std::vector<float> native_rope_table_data;
    // Backend-resident quantized landmark code stream (F16/Q8_0/Turbo4_0)
    // covering legal fragments, borrowed from BackendResidency / factorizer.
    // When present in SR mode on device, wired to ggml_xkv_landmark selector v3.
    ggml_tensor *native_landmarks = nullptr;
    // ---- Native Landmark Rebuild Descriptors (DEVICE_OWNED §9.3) ----
    struct xkv_native_landmark_rebuild_desc {
        uint32_t fragment_index = 0;
        uint64_t segment_id = 0;
        uint64_t segment_version = 0;
        uint32_t group_index = 0;
        uint32_t owning_layer = 0;
        uint32_t kv_head = 0;
        std::vector<uint32_t> row_indices;
        std::vector<int64_t> storage_positions;
        uint32_t feature_offset = 0;
        uint32_t feature_dim = 0;
        uint64_t phase_tx_fingerprint = 0;
        ggml_type landmark_type = GGML_TYPE_Q8_0;
        float error_bound = 0.0f;
    };
    std::vector<xkv_native_landmark_rebuild_desc> native_landmark_rebuild_requests;
    // ---- Bounded SR selection resources (XkvLandmarkBounded; sized at build) ----
    // Fragment views parallel sr_legal_frags (view_of_fragment, no copy).
    // Expanded per-(query,group) selector queries with sized q_vec floats.
    // Segment table (pinned views, frag first-appearance order) for owning-
    // segment rebuild/refine attribution. Expanded + merged CSR buffers and
    // the merged result share one global budget (never per-segment split).
    // Scratch comes from the lease sr slice (span) on the required path.
    std::vector<landmark_fragment_view> sr_frag_views;
    std::vector<sr_query> sr_selector_queries;
    std::vector<uint32_t> sr_selector_parents; // parent query per expanded entry
    std::vector<const xkv_segment *> sr_seg_table;
    std::vector<segment_row_ref> sr_expanded_gather;   // bounded-call union out
    std::vector<uint32_t> sr_expanded_ptrs;            // n_expanded+1
    std::vector<uint32_t> sr_expanded_indices;         // gather cap
    sr_batch_selection_result sr_result_cache;         // merged per-parent out
    uint32_t sr_padded_rank = 0;   // max A padded rank over views (scratch sizing)
    uint32_t sr_max_frag_rows = 0; // max frag row_count (scratch sizing)
    size_t sr_scratch_bytes = 0;   // from landmark_select_scratch_bytes
    // Exact per-group/adaptive rank + dims + RoPE + fingerprints for the
    // native op (filled by the builder; version must be GGML_XKV_VERSION).
    ggml_xkv_reconstruct_params native_params = {};

    // Explicit source dependency tensors (Q remains source 0; dependencies are source 1..N)
    std::vector<const struct ggml_tensor *> explicit_dependencies;

    // Execution status and diagnostic reporting
    mutable xkv_read_status last_status = xkv_read_status::invalid_argument;
    mutable std::string last_error = "uncomputed";
    mutable bool is_computed = false;
    mutable size_t peak_workspace_bytes = 0;

    // Helper to query effective DDVR group count for query q
    uint32_t get_query_group_count(uint32_t q) const;

    // Helper to compute total required elements in Q tensor
    bool get_required_q_elements(size_t & out_elements, std::string * err = nullptr) const;

    // Helper to validate hot rows dimensions, visibility, groups, and explicit bounds
    bool validate_hot_rows(std::string * err = nullptr) const;

    // Helper to validate current hot payload locations against store before and after read
    bool validate_hot_store_bindings(std::string * err = nullptr) const;

    // Helper to assemble xkv_hot_row views pointing to the owned hot_data
    std::vector<xkv_hot_row> build_hot_rows() const;

    // Helper to assemble xkv_query_input structures consuming a given Q tensor buffer
    bool build_query_inputs(
        const float * q_tensor_data,
        size_t q_tensor_elements,
        std::vector<xkv_query_input> & out_queries,
        std::string * err = nullptr
    ) const;
};

// Handle owning the snapshot and op execution state throughout GGML graph compute.
// Builder returns std::shared_ptr<xkv_graph_op_handle>. Runtime owners (e.g. llm_graph_input_xkv)
// retain shared ownership through graph lifetime.
class xkv_graph_op_handle {
public:
    static constexpr uint64_t HANDLE_MAGIC = 0x584B565F4F504844ULL; // 'XKV_OPHD'

    explicit xkv_graph_op_handle(std::unique_ptr<xkv_graph_snapshot> snapshot);
    ~xkv_graph_op_handle();

    xkv_graph_op_handle(const xkv_graph_op_handle &) = delete;
    xkv_graph_op_handle & operator=(const xkv_graph_op_handle &) = delete;

    xkv_graph_op_handle(xkv_graph_op_handle &&) noexcept = default;
    xkv_graph_op_handle & operator=(xkv_graph_op_handle &&) noexcept = default;

    xkv_graph_snapshot * snapshot() { return snapshot_.get(); }
    const xkv_graph_snapshot * snapshot() const { return snapshot_.get(); }

    // GGML custom op entrypoint adhering to ggml_custom_op_t:
    // void (*ggml_custom_op_t)(struct ggml_tensor * dst, int ith, int nth, void * userdata);
    static void ggml_custom_op_callback(struct ggml_tensor * dst, int ith, int nth, void * userdata);

    // Fail-closed execution without throwing across C ABI
    void compute(struct ggml_tensor * dst, int ith, int nth) noexcept;

    // Check if compute succeeded
    bool succeeded() const {
        return snapshot_ && snapshot_->is_computed && snapshot_->last_status == xkv_read_status::success;
    }

    xkv_read_status status() const {
        return snapshot_ ? snapshot_->last_status : xkv_read_status::invalid_argument;
    }

    const std::string & error_message() const {
        static const std::string uncomputed_msg = "uncomputed";
        return snapshot_ ? snapshot_->last_error : uncomputed_msg;
    }

    uint64_t magic = HANDLE_MAGIC;

private:
    std::unique_ptr<xkv_graph_snapshot> snapshot_;
};

// Builds a managed GGML custom op for attention.
// Returns the output ggml_tensor with shape [head_dim_v * n_q_heads, n_queries, 1, 1] of type GGML_TYPE_F32,
// and assigns a shared_ptr of the handle to out_handle. Runtime input managers retain shared ownership
// through graph lifetime.
// Accepts optional explicit source dependency tensors (Q remains source 0; dependencies are source 1..N).
struct ggml_tensor * xkv_build_graph_attention_ref(
    struct ggml_context * ctx,
    struct ggml_tensor * q_tensor,
    std::unique_ptr<xkv_graph_snapshot> snapshot,
    std::shared_ptr<xkv_graph_op_handle> & out_handle,
    const std::vector<struct ggml_tensor *> & dependencies = {}
);

// ==================== Bounded-hot graph runtime wiring ====================
//
// Cache-facing contract (built by llama_kv_cache_context::build_xkv_graph_snapshot,
// owned by XkvRuntimeSealer; consumed here, never redefined): one snapshot per
// owning layer and KV head. The graph slices Q heads per GQA group and concats
// per-head outputs. Storage/write/sink tensors travel as explicit scheduler
// sources (Q is src 0); exact physical hot slots are gathered and dequantized
// inside compute, after writes, never from stale pre-compute copies.
//
// Hot K canonicalization: Turbo-decode to canonical domain (inverts the
// write-time WHT) then inverse custom attention rotation only (hot_k_inv_rot).
// Inverse RoPE is NEVER applied: hot K stays storage-position post-RoPE, cold K
// is canonical pre-RoPE phased exactly once at storage position, V stays
// canonical. Optional per-channel multipliers (hot_k/v_channel_mul, empty =
// identity) apply after the inverse rotation, before truncation to logical dims.

// Physical hot storage layout descriptor (filled by the snapshot builder).
// Capability snapshot for fail-closed graph validation. Plain data only, so
// graph code maps hparams/cparams/sched into it without new includes.
struct xkv_graph_build_caps {
    int xkv_mode = 0; // llama_xkv_mode value: 0=OFF 1=SHADOW 2=DENSE 3=SR
    bool storage_needs_native = false; // single-source residency: TQ profile on a device factorizer
    bool native_backend_registered = false; // sched backend in the native registry (graph fills)
    bool use_alibi = false;
    int rope_type = 0; // ggml rope mode int; GGML_ROPE_TYPE_VISION is rejected
    bool v_transposed = false;
    bool has_kq_bias = false;
    ggml_type k_type = GGML_TYPE_F32;
    ggml_type v_type = GGML_TYPE_F32;
    bool backend_is_cpu = true;
};

// True only for bounded-hot DENSE/SR. OFF/SHADOW never create the XKV op and
// preserve the stock graph and reuse.
inline bool xkv_graph_covers_bounded_hot(const xkv_graph_build_caps & caps) {
    return caps.xkv_mode == 2 || caps.xkv_mode == 3;
}

// Fail-closed validation before any allocation. Rejects ALiBi, VISION/
// multimodal RoPE coordinates, transposed V, unsupported codecs/layouts and
// kq bias. Returns false with err set on rejection.
bool xkv_validate_build_caps(const xkv_graph_build_caps & caps, std::string * err = nullptr);

// Execution branch selection. CPU reference is always available; native
// reconstruct requires a registered capability (owned by XkvVulkanFactorOps).
// Returns false with err set when the required branch is unavailable.
enum class xkv_exec_branch : uint8_t { cpu_reference = 0, native_reconstruct = 1 };
bool xkv_select_exec_branch(const xkv_graph_build_caps & caps, xkv_exec_branch & out_branch, std::string * err = nullptr);

// Native reconstruct capability seam, keyed per backend (absent by default).
// The owner registers each proven backend (e.g. "Vulkan"); unregistered
// backends fail closed. Codec support is additionally validated per dispatch
// with ggml_xkv_reconstruct_supports on the actual arena tensors.
void xkv_register_native_reconstruct_capability(const char * backend_name, bool available);
bool xkv_native_reconstruct_available_for(const char * backend_name);

// Layer gate: bounded cparams mode alone never routes a layer to XKV. The
// cache/runtime answers per owning layer (SWA/recurrent/MTP layers report
// false and stay stock against their own full cache). Pure helper so both
// the graph and unit tests share the exact decision.
inline bool xkv_use_bounded_path(int xkv_mode, bool layer_enabled) {
    return (xkv_mode == 2 || xkv_mode == 3) && layer_enabled;
}

// Validate native arena tensors with the real ggml API before building the
// native op. Returns false with err on any unsupported shape/codec/param.
bool xkv_validate_native_tensors(
    const struct ggml_tensor * a_k, const struct ggml_tensor * b_k,
    const struct ggml_tensor * a_v, const struct ggml_tensor * b_v,
    const struct ggml_tensor * refs, const struct ggml_tensor * positions,
    const struct ggml_tensor * group_meta, const struct ggml_tensor * rope_tables,
    const struct ggml_tensor * layer_meta,
    const struct ggml_tensor * dst,
    const ggml_xkv_reconstruct_params * params, std::string * err = nullptr);

// Graph reuse policy: bounded XKV disables reuse until safe snapshot
// replacement after previous compute is proven.
inline bool xkv_graph_reuse_allowed(int xkv_mode) {
    return xkv_mode != 2 && xkv_mode != 3;
}

// Post-compute action mapping. Stale stamps request retry; codec/workspace/
// argument failures are hard errors. A zeroed output tensor alone is never
// success: callers must consult this, never the tensor contents.
enum class xkv_post_action : uint8_t { ok = 0, retry = 1, hard_error = 2 };
xkv_post_action xkv_graph_postcompute_action(xkv_read_status status);
bool xkv_graph_postcompute_ok(const xkv_graph_snapshot & snap, std::string * err = nullptr);

// Caller-owned metadata footprint of one snapshot (capacities, bytes) for
// graph/decode workspace reserve accounting (fit.cpp hookup,
// XkvAdmissionAccounting). Lease-backed spans report layout.total_bytes
// (arena-counted); shared segment pins are not double-counted.

// Exact arena match for one (segment, version, group); fail closed on
// miss/ambiguity (never host fallback). Shared by the chain builder and the
// llm dispatch precheck so both resolve identically.
const xkv_graph_snapshot::xkv_native_group_arenas * xkv_find_native_arenas(
    const xkv_graph_snapshot & snap, uint64_t seg_id, uint64_t seg_version,
    uint32_t group_index, std::string * err = nullptr);

// Vulkan narrowing mirror for one K/V codec pair side: matched Turbo pairs
// (same type) or canonical pairs; exactly-one-turbo rejected. CPU allows
// mixed pairs (it decodes). Used by llm dispatch on non-CPU backends for
// explicit init-time failure; the backend hook enforces at compute regardless.
bool xkv_native_codec_pairs_ok(ggml_type a_k, ggml_type b_k, ggml_type a_v, ggml_type b_v,
    bool backend_is_cpu, std::string * err = nullptr);

// SR-after-Q selection and required-descriptor fills, exposed for focused
// graph-runtime tests (defined in llama-xkv-graph-ref.cpp, same semantics
// as the in-compute paths).
bool run_sr_after_q(const xkv_graph_snapshot & snap, const std::vector<xkv_query_input> & queries,
    struct sr_batch_selection_result & out_sel, std::string * err = nullptr);
bool fill_required_hot(const xkv_graph_snapshot & snap, std::string * err = nullptr);
size_t xkv_snapshot_metadata_bytes(const xkv_graph_snapshot & snap);

// One-shot host fill for a graph-created native metadata tensor
// (refs/positions/group_meta/layer_meta/entries/offsets/cold-dummy).
// Recorded at build; applied at set_input with exact-size validation
// (refuse truncation, never silent partial fill).
struct xkv_native_fill_item {
    ggml_tensor * tensor = nullptr; // owned by the graph ctx, set_input-marked
    std::vector<uint8_t> bytes;     // exact ggml_nbytes(tensor) content
};

// Builds the native tiled XKV attention chain for one KV head (ggml-xkv.h).
// Consumes the snapshot as borrowed config: hot storage views and sink head
// tensor are explicit graph tensors; cold code streams come from
// snap.native_group_arenas (borrowed backend-resident, builder-wired).
// Small per-build metadata (refs/positions/group_meta/layer_meta/entries/
// offsets/cold-dummy) is created in ctx as set_input tensors; their exact
// bytes are returned in out_fills for set_input application. One I32[1]
// status tensor per attention tile is appended to out_status_tensors for
// postcompute validation (nonzero status = hard failure, never silent).
// Returns per-head output [head_dim_v_logic * n_q_heads, n_queries] F32,
// identical in shape to the reference builder output. Fail-closed: returns
// nullptr with err on any unsupported/mismatched input (SR-after-Q mode,
// custom attention rotation, transposed V, missing arenas, mixed codecs on
// non-CPU, per-query-varying sinks, multi-stream hot, unmapped groups,
// empty queries). Never falls back, never partial topology.
// NOTE: codec pairing (matched Turbo vs canonical) is validated per dispatch
// by ggml_xkv_reconstruct_supports on real tensors; the Vulkan backend hook
// additionally narrows mixed pairs at compute. backend_is_cpu selects the
// pairing rule (CPU decodes mixed pairs; non-CPU fails closed explicitly).
struct ggml_tensor * xkv_build_attention_native(
    struct ggml_context * ctx,
    struct ggml_tensor * q_h,
    const xkv_graph_snapshot & snap,
    struct ggml_tensor * k_store,
    struct ggml_tensor * v_store,
    struct ggml_tensor * sinks_head,
    bool backend_is_cpu,
    std::vector<struct ggml_tensor *> & out_status_tensors,
    std::vector<xkv_native_fill_item> & out_fills,
    std::string * err = nullptr);

// ---- Device SR metadata emission / rows-output partition -------------
// Graph side of ggml_xkv_landmark_rows (ggml-vulkan-landmark.h): emit exact
// device metadata from plain fragment/arena tables. The production graph
// consumes the device ROWS tensors directly; the partition helper below is
// only a CPU oracle/debug validator and is never a device D2H bridge.
//
// Conventions (locked with the kernel owner):
// - One GLOBAL rows call over concatenated segments. Global frag id = index
//   into frags. Row ids are GLOBAL: per (segment,group) arena ranges assigned
//   cumulative in arenas order, disjoint across arenas, so rows-op dedup by
//   row collapses only true duplicates (shared prefix within one arena).
// - frag_kv[f] = (semantic group_index, GLOBAL tile ordinal = arena index,
//   kv head). refs[0] carries the global row id; the partitioner subtracts
//   the tile arena base into arena-relative per-tile refs and cross-checks
//   refs[1] against the range-decoded arena group (fail closed on mismatch).
// - Legality is global eligible-bit plus an explicit per-expanded-query
//   eligibility bitset, causal position, and DDVR group. Row generation is
//   diagnostic identity, never compared with the content epoch.
// - frag_meta[2] (segment_id) is kernel-ignored; emission writes a
//   graph-assigned segment ordinal for debuggability.
struct xkv_sr_device_frag {
    uint64_t segment_id = 0;
    uint64_t segment_version = 0;
    uint32_t group_index = 0; // semantic factor group (matches store/backend identity)
    uint32_t ddvr_group = 0;  // reader-visible DDVR slot used only for selector legality
    uint32_t owning_layer = 0;
    uint32_t kv_head = 0;
    // Phase-domain identity (verified against layer-expected values at emission;
    // a stale/wrong-transform landmark fails closed here, never scores wrong).
    uint64_t phase_tx_fingerprint = 0; // must equal expected (forward-RoPE transform)
    uint64_t landmark_codec_fp = 0; // must equal expected (landmark codec descriptor)
    uint64_t storage_generation = 0; // diagnostic row-generation identity (I32-range)
    uint32_t row_begin = 0; // contiguous base (local arena row; must be 0 if sparse)
    uint32_t row_count = 0; // >0, <= GGML_XKV_LANDMARK_MAX_FRAG_ROWS
    bool sparse = false;
    std::vector<int32_t> sparse_rows; // iff sparse: row_count local ids, -1 = hole
    std::vector<int64_t> storage_positions; // per row (row_count entries; holes ignored)
    int32_t eligible = 1; // 0/1 only: eligible bit (never a query bitmask)
    int32_t frag_pos = 0; // representative storage pos (frag_positions[f], causal checks)
    std::vector<bool> query_visibility; // per-query visibility from runtime (§9.3)
};
struct xkv_sr_device_query {
    uint32_t parent_query = UINT32_MAX; // UINT32_MAX => identity index in emitter
    int64_t causal_limit_pos = -1; // -1 = no cutoff, else max visible storage_pos
    int64_t query_pos = 0;
    uint32_t n_q_heads = 1;
    uint32_t kv_head = 0;
    int32_t vis_group = 0; // DDVR slot; -1 permits every visible fragment group
};
struct xkv_sr_device_arena {
    uint64_t segment_id = 0;
    uint64_t segment_version = 0;
    uint32_t group_index = 0;
    uint64_t landmark_codec_fp = 0; // exact descriptor fingerprint for this arena
    uint64_t row_count = 0; // arena rows; global base assigned cumulative in order
};
// Emitted I32 buffers in exact oracle/builder layouts: frag_meta [6,nf],
// frag_positions [nf], frag_gen [nf], query_meta
// [causal,query_pos,n_q_heads,kv_head,vis_group,reserved] [6,nq], frag_row_off [nf+1]
// CSR, frag_row_ids pool (sparse rows only), frag_kv [3,nf], row_pos
// [n_rows_total] (-1 for never-fragged globals). Fail-closed on any
// unsupported input (empty frags, OOB rows, pos/gen overflow, arena miss,
// conflicting positions for one global row, phase/codec fingerprint mismatch).
// Per-query legality is the emitted elig_bits mask plus causal/DDVR checks.
// storage_generation is diagnostic identity only: it is never compared with a
// global content epoch (row generations and content epochs are different domains).
// Never truncates.
struct xkv_sr_device_inputs {
    std::vector<int32_t> frag_meta;
    std::vector<int32_t> frag_positions;
    std::vector<int32_t> frag_gen;
    std::vector<int32_t> query_meta;
    std::vector<int32_t> frag_row_off;
    std::vector<int32_t> frag_row_ids;
    std::vector<int32_t> frag_kv;
    std::vector<int32_t> row_pos;
    std::vector<int32_t> elig_bits; // [(n_frags + 31)/32, n_queries]
    uint32_t n_frags = 0;
    uint32_t n_queries = 0;
    uint32_t n_rows_total = 0;
    uint32_t max_frag_rows = 0; // rows-params max_frag_rows value
    bool sparse = false;
};
bool xkv_sr_emit_device_inputs(const std::vector<xkv_sr_device_frag> & frags,
    const std::vector<xkv_sr_device_query> & queries,
    const std::vector<xkv_sr_device_arena> & arenas,
    uint64_t expected_phase_tx_fp,
    xkv_sr_device_inputs & out, std::string * err = nullptr);
// Partition rows-op outputs into per-arena tile ref lists (arena order,
// non-empty only). refs are arena-relative quads [local_row,group,slot,head]
// with slot = arena index; positions are storage positions cross-checked
// against the emitted row_pos. Policy: status[0] != OK fails closed;
// status[2] > 0 (clamped queries) sets stale=true (caller retries with a
// larger refine_cap, never silently continues); padding past per-query
// counts must be -1 quads/positions. Verifies per-query ascending order.
struct xkv_sr_device_tile_rows {
    uint32_t arena_index = 0;
    std::vector<int32_t> refs; // quads, arena-relative
    std::vector<int32_t> positions;
};
struct xkv_sr_partition_result {
    std::vector<xkv_sr_device_tile_rows> tiles;
    bool stale = false;
    uint32_t total_rows = 0;
    uint32_t clamped_queries = 0;
};
bool xkv_sr_partition_rows_outputs(const int32_t * row_ptrs, const int32_t * row_refs,
    const int32_t * out_row_pos, const int32_t * row_status, uint32_t n_queries,
    uint32_t refine_cap, const std::vector<xkv_sr_device_arena> & arenas,
    const std::vector<int32_t> & row_pos, xkv_sr_partition_result & out,
    std::string * err = nullptr);

// ---- Same-shape snapshot content refresh (reuse gate) ----
// set_input/can_reuse call the installed snapshot_refresh_fn; with no hook
// installed reuse is always refused. This mechanism implements the hook:
// fresh content-epoch data (new stamp, cells, positions, visibility,
// causal, sinks) swaps into a snapshot with FROZEN topology (same hot/
// query/view counts, same view epochs, same segment identities, same
// group structure). Anything beyond preallocated capacities or changing
// shape fails closed (caller sets force_retry_stale so the outer loop
// rebuilds); stale output is never committed. Segment pins are untouched
// (COW-immutable); the reader guard is reacquired by set_input, not here.
// After the swap, descriptor caches rebuild via preallocate_compute_state
// (same sizes, no growth) and must validate.
struct xkv_snapshot_refresh_data {
    xkv_snapshot_stamp stamp; // new expected stamp (view.* epochs must match)
    std::vector<xkv_graph_snapshot::hot_row_data> hot_rows; // same count as snapshot hot_data
    std::vector<int64_t> query_causal_limits; // empty = keep, else size n_queries
    std::vector<std::vector<float>> query_sink_logits; // empty = keep, else [n_queries][n_q_heads]
    std::vector<uint32_t> query_ddvr_group_counts; // empty = keep, else effective-topology equal
};
bool xkv_snapshot_refresh_content(xkv_graph_snapshot & snap,
    const xkv_snapshot_refresh_data & fresh, std::string * err = nullptr);
// Provider reads fresh store content (Sealer-owned); the installer binds it
// to the content swap above. Install only after the snapshot address is
// stable (post unique_ptr placement); the lambda captures the raw pointer.
using xkv_snapshot_refresh_provider =
    std::function<bool(xkv_snapshot_refresh_data &, std::string &)>;
inline void xkv_install_snapshot_refresh(xkv_graph_snapshot * snap,
    xkv_snapshot_refresh_provider provider) {
    if (!snap || !provider) return;
    snap->snapshot_refresh_fn = [snap, provider](std::string & err) {
        xkv_snapshot_refresh_data fresh;
        if (!provider(fresh, err)) return false;
        return xkv_snapshot_refresh_content(*snap, fresh, &err);
    };
}

} // namespace llama_xkv
