#pragma once

// llama-xkv-runtime: cache-owned runtime coordinator for target-trunk XKV.
//
// Owns, for one llama_kv_cache instance:
//   - deterministic alias-deduplicated layer groups built from actual hparams/model/cache state;
//   - enumeration of eligible homogeneous committed PUBLIC logical cells joined to exact
//     hot/factored locations (llama_kv_cells remains the sole semantic owner; the XKV
//     store owns payload locations only);
//   - bounded segment selection and atomic multi-group sealing at quiescent boundaries;
//   - POD admission accounting with a limiting-reason enum (XKV OFF maps legacy capacity/usage);
//   - explicit readiness/capability reporting so graph/build paths fail closed until the
//     XKV attention path exists (never legacy attention on truncated bounded-hot tensors);
//   - per-KV-head graph snapshot construction delegating cell semantics (implemented in
//     llama-xkv-runtime.cpp against llama-xkv-graph-ref.h; only forward-declared here);
//   - a default phase/position-aware landmark rebuild callback for survivor-pack paths.
//
// XKV OFF and MTP/draft contexts never create a runtime (create() returns nullptr) and
// never allocate/store/branch through XKV code.

#include "llama-cparams.h"
#include "llama-xkv-cache.h"
#include "llama-xkv-landmark.h"

#include "ggml-xkv.h"

#include <cstdint>
#include <functional>
#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>


class llama_kv_cache;
class llama_kv_cache_context;
struct llama_hparams;
class llama_kv_cells;

namespace llama_xkv {

// Forward declarations to avoid include cycles. Complete types are only
// needed in llama-xkv-runtime.cpp (and by graph consumers of xkv_graph_snapshot).
class xkv_canonical_layer_kv_snapshot;
class xkv_graph_snapshot;
class xkv_transaction_coordinator;
struct canonical_batch_stats;
class xkv_b_tile_cache;
struct xkv_decode_cache_metrics;

// Per-layer exact phase specification for landmark construction: the forward
// half-layout RoPE applied to each layer slice before pooling. These types
// precede removal-capture declarations because that state owns phase vectors.
struct xkv_landmark_phase_layer {
    uint32_t feature_offset = 0;
    uint32_t feature_dim = 0;
    uint32_t n_heads = 0;
    uint32_t head_dim = 0;
    uint32_t rotary_dim = 0;
    std::vector<float> omega;
    std::vector<float> freq_scale_sq;
};

struct xkv_group_phase_spec {
    uint32_t group_index = 0;
    std::vector<xkv_landmark_phase_layer> layers;
    uint64_t fingerprint = 0;
};

// Why admission/maintenance cannot proceed. Part of the POD admission snapshot.
// Readiness is computed from actual hardware/profile state at init, never a
// fixed compile bool. CPU host tensors + CPU factorizer + decoded-hot source
// supports the CPU/reference graph path; Vulkan TQ stays false until native
// full-attention dispatch lands (not merely the reconstruct op); CUDA/Metal
// and unsupported layouts are false with a precise reason.
struct xkv_path_readiness {
    bool ready = false;
    char reason[128] = {};
};

// Positions-aware pack/removal rebuild capture for seq_rm/seq_keep/
// overwrite/cancel paths. Built from live cells BEFORE any mutation:
// every old live payload row maps to its cells-owned storage_pos plus the
// exact cached group phase. The bound store closure fails on any missing
// position, so callers validate before mutating. Capture buffers are caller-
// owned and reusable across removals (clear+refill retains capacity).
struct xkv_pack_rebuild_capture {
    std::unordered_map<uint64_t, int64_t> pid_to_pos;
    std::vector<xkv_group_phase_spec> phase_specs;
    ggml_type landmark_type = GGML_TYPE_Q8_0;
    uint32_t chunk_tokens = 8;

    void clear() {
        pid_to_pos.clear();
        phase_specs.clear();
    }

    // True when every payload id has a captured cells-owned position.
    bool validate(const std::vector<uint64_t> & payload_ids, std::string * err = nullptr) const;

    // Store-compatible closure over the capture (positions guaranteed by a
    // prior validate). Refuses on any lookup gap or phase mismatch.
    xkv_landmark_rebuild_fn bind() const;
};

// Standalone store-level quiescence primitive. Captures the store stamp and the
// bounded hot pool reservation level; validate() fails when either moved, i.e.
// the cache was not quiescent across the guarded range. Initially consumed by
// runtime maintenance; available to future transactional consumers.
class xkv_quiescence_guard {
public:
    xkv_quiescence_guard() = default;
    // Capture from a live cache. Never mutates.
    explicit xkv_quiescence_guard(const llama_kv_cache & kv);

    // Re-capture current state from the cache.
    void capture(const llama_kv_cache & kv);
    // True when nothing observable moved since capture (or since construction).
    bool validate(const llama_kv_cache & kv, std::string * err = nullptr) const;
    bool armed() const { return armed_; }

private:
    bool armed_ = false;
    bool have_store_ = false;
    bool have_pool_ = false;
    xkv_snapshot_stamp stamp_ = {};
    uint32_t pool_reserved_ = 0;
    uint32_t pool_bound_ = 0;
};

// One eligible logical payload joined to its exact physical location.
// No semantic duplication: payload identity, positions, and visibility are read
// from llama_kv_cells; the physical row/generation comes from the XKV store
// (or the bounded hot pool when one exists).
struct xkv_layer_row_view {
    uint64_t payload_id = 0;
    uint64_t generation = 0;
    uint32_t physical_row = 0; // exact hot row for canonical reads / release plans
    int32_t  storage_pos = 0;  // cells-owned storage position (RoPE authority)
    uint32_t logical_cell = 0; // representative cells-owned logical cell index
    uint32_t stream = 0;
    uint8_t  visibility = 0;   // llama_rerot_visibility value at enumeration time
    // Exact semantic sharing domain (cells-owned, never duplicated):
    std::vector<llama_seq_id> seq_ids; // sorted keeper refs of the representative cell
    uint64_t episode_id = 0;           // RERoT episode (0 = ordinary)
    uint64_t run_id = 0;               // RERoT run (INVALID = ordinary)
    uint64_t publish_epoch = 0;        // RERoT publication epoch (0 = unpublished)
    uint64_t seq_set_fp = 0;           // fingerprint of seq_ids (ordinary rows only)
};

// Semantic sharing domain key for sealing. Rows seal together only under an
// identical key: same storage stream, same visibility class, same RERoT
// episode/run/publish identity, and (for ordinary rows) the exact keeper seq
// set. Joint low-rank B factors never couple unrelated/private histories;
// shared-prefix rows keep their run domain and seal with identically-shared
// siblings. Never mix across keys.
struct xkv_seal_domain_key {
    uint32_t stream = 0;
    uint8_t  visibility = 0; // llama_rerot_visibility: normal or public_live
    uint64_t episode_id = 0;
    uint64_t run_id = 0;
    uint64_t publish_epoch = 0;
    uint64_t seq_set_fp = 0; // ordinary rows: exact seq set; public rows: 0

    bool operator==(const xkv_seal_domain_key & o) const {
        return stream == o.stream && visibility == o.visibility &&
               episode_id == o.episode_id && run_id == o.run_id &&
               publish_epoch == o.publish_epoch && seq_set_fp == o.seq_set_fp;
    }
    bool operator<(const xkv_seal_domain_key & o) const {
        if (stream != o.stream) return stream < o.stream;
        if (visibility != o.visibility) return visibility < o.visibility;
        if (episode_id != o.episode_id) return episode_id < o.episode_id;
        if (run_id != o.run_id) return run_id < o.run_id;
        if (publish_epoch != o.publish_epoch) return publish_epoch < o.publish_epoch;
        return seq_set_fp < o.seq_set_fp;
    }
};

uint64_t fingerprint_seal_domain(const xkv_seal_domain_key & key);
uint64_t fingerprint_seq_set(const std::vector<llama_seq_id> & seq_ids);
// Splits one ordered factor group into maximal placement-homogeneous runs.
// Offsets are rebased run-local; group_index values are reassigned by the
// caller. Pure logic over caller-supplied placement keys (testable without
// devices); the runtime derives keys from live K/V buffer types.
std::vector<layer_group> split_group_by_placement(
    const layer_group & group,
    const std::vector<std::string> & placements);
// Buffer-type identity key for one owning layer (K+V buft names).
std::string layer_placement_key(ggml_tensor * k, ggml_tensor * v);

// Deterministic proportional tail rank: ceil(base * nlayers / group_size)
// then geometry clamp. Full groups keep base rank; tails scale down so
// small trailing groups are not over-ranked. Zero on degenerate input.
inline uint32_t xkv_scaled_group_rank(uint32_t base_rank, uint32_t nlayers,
                                       uint32_t group_size, uint32_t n_rows,
                                       uint64_t total_dim) {
    if (base_rank == 0 || nlayers == 0 || group_size == 0 || n_rows == 0 || total_dim == 0) {
        return 0;
    }
    uint64_t scaled = ((uint64_t) base_rank * nlayers + group_size - 1) / group_size;
    if (scaled < 1) {
        scaled = 1;
    }
    if (scaled > n_rows) {
        scaled = n_rows;
    }
    if (scaled > total_dim) {
        scaled = total_dim;
    }
    return (uint32_t) scaled;
}

// Safe read of a cell's full keeper seq set (sorted). Cells-owned; never
// duplicated. Shared with rollback consumers needing exact ref sets.
std::vector<llama_seq_id> xkv_cell_seq_ids(const ::llama_kv_cells & cells, uint32_t cell);
// Bounded runtime failure/counter reporting. last_error is a truncated,
// fixed-capacity diagnostic (never unbounded).
struct xkv_runtime_stats {
    uint64_t maintenance_runs = 0;
    uint64_t sealed_segments = 0;
    uint64_t sealed_rows = 0;
    uint64_t shadow_evals = 0;  // evaluate-only shadow transactions (no rebind)
    uint64_t deferred_runs = 0; // runs with no full safe segment available
    uint64_t quiescence_skips = 0;
    uint64_t last_sync_count = 0; // canonical backend syncs in the last seal path
    uint64_t last_domain_fp = 0; // sharing-domain fingerprint of the last sealed segment
    uint64_t last_factored_bytes = 0; // measured factored bytes of the last sealed segment
    uint32_t last_factored_rows = 0;  // sealed rows of the last sealed segment
    // Sealed-totals for the observed runtime snapshot (measured at each
    // seal from the published immutable segment; removals may drift live
    // state below these sealed watermarks).
    uint64_t sealed_streams = 0;
    uint64_t sealed_landmarks = 0;
    uint64_t cum_ak_bytes = 0;
    uint64_t cum_bk_bytes = 0;
    uint64_t cum_av_bytes = 0;
    uint64_t cum_bv_bytes = 0;
    uint64_t cum_lm_bytes = 0;
    uint64_t cum_flat_source_bytes = 0;
    uint64_t cum_factored_bytes = 0;
    double   last_compression_ratio = 0.0;
    bool     compression_goal_met = false;
    uint64_t sync_total = 0;
    double   cum_seal_seconds = 0.0;
    double   cum_factor_quant_seconds = 0.0;
    double   cum_landmark_quant_seconds = 0.0;
    // Postcompute & graph counters (§16, llama_xkv_runtime)
    double   cum_select_seconds = 0.0;
    double   cum_refine_seconds = 0.0;
    double   cum_reconstruct_seconds = 0.0;
    double   cum_read_seconds = 0.0;
    double   cum_pack_seconds = 0.0;
    uint64_t sr_selected_rows = 0;
    uint64_t sr_fragments = 0;
    uint64_t landmark_refine_rows = 0;
    uint64_t landmark_refine_cap_hits = 0;
    uint64_t spec_stale_total = 0;
    uint64_t transaction_abort_total = 0;
    uint64_t last_segment_id = 0;
    uint64_t last_segment_version = 0;
    uint64_t last_codec_fp = 0;
    uint64_t last_backend_fp = 0;
    uint64_t last_source_fp = 0;
    uint64_t last_profile_fp = 0;
    uint32_t last_rank_k_obs = 0;
    uint32_t last_rank_v_obs = 0;
    std::vector<uint64_t> skipped_by_reason; // indexed by xkv_skip_reason
    char last_error[256] = {};

    void record_skip(xkv_skip_reason reason);
    void record_error(const std::string & msg);
};

// Precise per-call maintenance outcome. Unscoped so existing boolean checks
// keep working (error == 0 is falsy). The server calls maintain() repeatedly
// until the hot deficit clears: sealed/evaluated mean progress, no_action
// means nothing due, retry_stale means state moved (re-query then retry).
// Exactly one segment seals per call; there is no segment-count cap.
enum xkv_maintenance_outcome : uint8_t {
    xkv_maintain_error = 0,   // hard failure; err set; must take the error path
    xkv_maintain_sealed = 1,  // sealed exactly one segment; re-query snapshot
    xkv_maintain_no_action = 2, // fill-first/deferred; nothing due
    xkv_maintain_retry_stale = 3, // contention/state moved; re-query then retry
    xkv_maintain_evaluated = 4, // SHADOW evaluate-only ran; no publication
};

// Preallocated, bounded control/metadata workspace owned by the runtime coordinator.
// Sized at init_workspace() from maximum hot capacity and group count, then reused
// across all maintain() calls with zero per-call heap allocation.
struct xkv_maintain_control_scratch {
    // Row enumeration & windowing
    std::vector<xkv_layer_row_view> eligible;
    std::vector<size_t> eligible_indices; // index permutation for in-place domain sorting
    std::vector<size_t> eligible_deduped; // pid-deduped indices (enumeration order)
    std::vector<size_t> window_idx; // chosen domain window (indices into eligible)
    std::vector<uint64_t> pids;
    std::vector<uint64_t> gens;
    std::vector<int64_t> positions;
    std::vector<uint32_t> physical_rows;
    // Reusable release plan (vectors retain capacity; refilled per run).
    xkv_hot_release_plan release_plan;


    // Sealing structures
    std::vector<xkv_factor_group_input> inputs;
    std::vector<xkv_group_phase_spec> phase_specs;

    // Slice descriptors
    struct slice_desc {
        float * k = nullptr;
        float * v = nullptr;
    };
    std::vector<slice_desc> slices;

    // Streaming read head scratch (single head at max dim, reused across layers)
    std::vector<float> head_scratch;
    // Canonical int32 positions parallel to window_idx (storage_pos is int32;
    // int64 window positions are preserved separately, never narrowed).
    std::vector<int32_t> canonical_positions;
    // Physical hot rows parallel to window_idx for canonical reads.
    std::vector<uint32_t> canonical_cells;



    // Landmark/factor scratch carved per run (no per-group heap)
    std::vector<float> lm_decode_a;
    std::vector<float> lm_decode_b;
    std::vector<float> lm_recon;
    std::vector<float> lm_phased;
    std::vector<float> lm_accum;
    std::vector<uint64_t> lm_row_idx;
    std::vector<uint32_t> lm_identity_rows; // 0..n-1 for full-segment factories
    std::vector<landmark_build_layer> lm_spec_layers; // per-layer slices for the bounded builder
    std::vector<landmark_build_output> lm_out_chunks; // per-chunk output descriptors
    std::vector<xkv_seal_domain_key> domain_keys;

    bool initialized = false;
    size_t capacity_rows = 0;
    size_t capacity_groups = 0;
    size_t capacity_head_dim = 0;

    void init(size_t max_rows, size_t max_groups, size_t max_head_dim);
    void clear_for_run();
};

// Reusable removal-capture state (capacity retained across removals).
struct xkv_removal_capture_state {
    uint64_t generation = 0;
    // Sorted (pid -> storage_pos) records, capped by exact eligible row
    // count and accounted in workspace; binary-searched, never hashed.
    // Capacity retained across removals: no per-removal hash allocations.
    std::vector<std::pair<uint64_t, int64_t>> pid_to_pos;
    void clear_keep_capacity() { pid_to_pos.clear(); }
    // Lower-bound lookup; end() miss.
    std::vector<std::pair<uint64_t, int64_t>>::const_iterator find(uint64_t pid) const {
        auto it = std::lower_bound(pid_to_pos.begin(), pid_to_pos.end(), pid,
            [](const std::pair<uint64_t, int64_t> & e, uint64_t v) { return e.first < v; });
        if (it != pid_to_pos.end() && it->first == pid) {
            return it;
        }
        return pid_to_pos.end();
    }
};

// Explicit immutable effective config: every sealing knob validated once at
// construction, fingerprinted, and used verbatim. There are no hidden
// fallbacks (a zero segment/chunk size is invalid, not silently replaced)
// and no per-call magic numbers.
// Knobs without a cparams/profile home live in the versioned sealing profile
// below. Immutable: any change requires a version bump (which changes every
// effective fingerprint). Shared by name/value with docs (XKV-SR.md section 16),
// state writers, and metrics; the construction log prints the effective set.
struct xkv_sealing_profile {
    static constexpr uint32_t version = 1;
    // Sealed-bundle error gate: maximum relative reconstruction error.
    static constexpr double max_relative_error = 0.25;
    // Absolute floor for counted byte savings.
    static constexpr size_t min_saving_bytes = 1;
};
struct xkv_effective_config {
    uint32_t profile_version = 0;
    uint32_t group_size = 0;
    uint32_t rank_k = 0;
    uint32_t rank_v = 0;
    uint32_t segment_tokens = 0;
    uint32_t chunk_tokens = 0;
    uint64_t factor_seed = 0;
    double max_relative_error = 0.0;
    double min_saving_ratio = 0.0;
    size_t min_saving_bytes = 0;

    // Full §16 coverage: every numeric/selection input participates in the
    // fingerprint. Copied immutably at construction from cparams.
    int32_t mode = 0;
    int32_t storage_profile = 0;
    int32_t source = 0;
    int32_t factor_a_k = 0;
    int32_t factor_b_k = 0;
    int32_t factor_a_v = 0;
    int32_t factor_b_v = 0;
    int32_t factor_balance = 0;
    int32_t landmark_type = 0;
    int32_t landmark_refine = 0;
    uint32_t landmark_refine_max_rows = 0;
    uint32_t sr_budget = 0;
    int32_t factorizer = 0;
    uint32_t workspace_mib = 0;
    uint32_t decode_cache_mib = 0;
    uint32_t store_mib = 0;
    double min_factor_coverage = 0.0;

    bool valid(std::string * err = nullptr) const;
    uint64_t fingerprint() const;
};

xkv_effective_config make_effective_config(const llama_cparams & cparams);

// One-time device upload hook (backend owner, e.g. Vulkan): uploads the
// exact code-stream bytes of one immutable published segment, verifies, and
// reports success. Host release of published bytes stays with the store's
// publish path. Null means no device path exists (fail closed, never
// silently host-factor a device profile).
using xkv_device_upload_fn = std::function<bool(
    uint64_t segment_id,
    uint64_t segment_version,
    std::string * err
)>;

// Semantic landmark rebuild callback for survivor-pack paths. Rebuilds the
// group landmark from the FINAL factor streams restricted to surviving_rows,
// phased per row at its storage position (see xkv_landmark_phase_layer),
// chunked deterministically. Must never be a first-N byte copy after
// arbitrary deletion, and never an unphased pooled mean.
using xkv_group_landmark_rebuild_fn = std::function<bool(
    const xkv_segment & segment,
    const xkv_factor_group_payload & group,
    const std::vector<uint32_t> & surviving_rows,
    const std::vector<int64_t> & storage_positions,
    const std::vector<xkv_landmark_phase_layer> & phase_layers,
    uint64_t phase_tx_fingerprint,
    encoded_matrix & out_landmark,
    std::string * err
)>;

// Fingerprint keying the exact phase transform (rope style, dims, tables,
// layer order). The engine recomputes it from the passed layers and refuses
// on mismatch: a fingerprint alone never substitutes for the transform.
uint64_t fingerprint_landmark_phase(const std::vector<xkv_landmark_phase_layer> & layers);

// Phased engine: reconstructs canonical K rows from the FINAL factor
// streams (decoded full-width matrices), applies the exact forward phase per
// layer slice at each row's storage position, mean-pools phased rows per
// chunk, and encodes the quantized landmark. storage_positions is REQUIRED
// (parallel to surviving_rows); phase_tx_fingerprint must equal
// fingerprint_landmark_phase(phase_layers). Any violation refuses.
bool rebuild_landmarks_mean_pool_phased(
    const float * a_dec, uint64_t a_rows, uint64_t a_pad,
    const float * b_dec, uint64_t b_dim, uint64_t b_pad, uint64_t rank,
    const std::vector<uint32_t> & surviving_rows,
    const std::vector<int64_t> & storage_positions,
    const std::vector<xkv_landmark_phase_layer> & phase_layers,
    uint64_t phase_tx_fingerprint,
    ggml_type landmark_type,
    uint32_t chunk_tokens,
    float * recon_scratch, size_t recon_cap,
    float * phased_scratch, size_t phased_cap,
    float * lm_scratch, size_t lm_cap,
    encoded_matrix & out_landmark,
    std::string * err = nullptr
);

// Builds exact per-layer phase specifications for one factor group, mirroring
// the canonical RoPE table derivation (same builder, model, and cparams, so
// the tables are bit-identical to canonical reads). Refuses unsupported rope
// types, degenerate dims, and table build failures. The returned fingerprint
// equals fingerprint_landmark_phase(layers).
bool build_landmark_phase_layers(
    const llama_kv_cache & kv,
    const llama_cparams & cparams,
    const llama_hparams & hparams,
    const layer_group & group,
    std::vector<xkv_landmark_phase_layer> & out_layers,
    uint64_t & out_fingerprint,
    std::string * err = nullptr);

// Positions-carrying candidate rebuild for maintenance-driven packs: every
// group landmark is rebuilt phased from the source FINAL factors. Refuses
// when any group lacks a phase spec, positions mismatch, or the fingerprint
// does not key the spec.
bool rebuild_candidate_landmarks_with_positions(
    const xkv_segment & source,
    const std::vector<uint32_t> & surviving_rows,
    const std::vector<int64_t> & storage_positions,
    const std::vector<xkv_group_phase_spec> & phase_specs,
    xkv_segment & candidate,
    ggml_type landmark_type,
    uint32_t chunk_tokens,
    std::string * err = nullptr
);

// Store-level adapter: rebuilds every group landmark of a mutable pack
// candidate. Positions and phase are cells-owned and unavailable inside the
// store, and an unphased pooled mean is mathematically wrong, so this ALWAYS
// refuses: the caller must keep tombstones until the runtime supplies
// positions+phase via rebuild_candidate_landmarks_with_positions. Never a
// first-N byte copy, never a phase-less mean masquerading as a landmark.
bool rebuild_candidate_landmarks(
    const xkv_segment & source,
    const std::vector<uint32_t> & surviving_rows,
    xkv_segment & candidate,
    ggml_type landmark_type,
    uint32_t chunk_tokens,
    std::string * err = nullptr
);

// Cache-owned runtime coordinator. Created via create() (nullptr for XKV OFF
// and MTP/draft contexts). All methods are deterministic given identical cache
// state; enumeration order is (storage_pos, payload_id).
class llama_xkv_runtime {
public:
    // nullptr when cparams.xkv_mode == OFF or cparams.ctx_type == MTP.
    static std::unique_ptr<llama_xkv_runtime> create(const llama_cparams & cparams);

    // Explicit end-to-end capability gate: false (with reason) for any
    // mode/profile/source/factorizer combination that is not fully wired
    // (PREROPE_CAPTURE staging, non-CPU factorization). Checked at create()
    // time so misconfiguration fails at init, never at first pressure.
    // OFF/MTP trivially pass (no runtime by design).
    static bool config_supported(const llama_cparams & cparams, std::string * err = nullptr);

    explicit llama_xkv_runtime(const llama_cparams & cparams);

    llama_xkv_runtime(const llama_xkv_runtime &) = delete;
    llama_xkv_runtime & operator=(const llama_xkv_runtime &) = delete;


    // Explicit effective config (validated, fingerprinted). maintain() fails
    // closed while invalid.
    const xkv_effective_config & effective_config() const { return eff_; }
    bool config_valid(std::string * err = nullptr) const { return eff_.valid(err); }
    // Full §16 fingerprint for segment/state/metrics stamping.
    uint64_t config_fingerprint() const { return eff_.fingerprint(); }


    // Single transaction coordinator gating seal/pack/MTP/fence readers.
    // Non-owning; set once by the cache owner. Null is allowed only for
    // standalone/test use (no concurrent subsystem exists there).
    void set_coordinator(xkv_transaction_coordinator * coord) { coord_ = coord; }
    xkv_transaction_coordinator * get_coordinator() const { return coord_; }

    // Device upload hook for DEVICE_OWNED profiles (backend owner sets it).
    void set_device_upload_fn(xkv_device_upload_fn fn) { upload_fn_ = std::move(fn); }

    // One shared global B-tile decode cache per shared-store runtime family
    // (checked MiB capacity from xkv_decode_cache_mib). Passed to every CPU
    // reader snapshot via reader_config; the bounded reader path uses
    // put_and_lease_copy borrowed-workspace fallback internally, so cache +
    // workspace never exceed budgets and no per-reader cache exists. Native
    // Vulkan bypass reports honest zeros through decode_cache_metrics().
    std::shared_ptr<xkv_b_tile_cache> get_b_tile_cache() const { return b_tile_cache_; }
    xkv_decode_cache_metrics decode_cache_metrics() const;
    // Invalidate on codec/view/state events (seal calls epoch invalidation
    // internally; removal/publish paths call these explicitly).
    void invalidate_decode_cache_segment(uint64_t segment_id, uint64_t segment_version);
    void invalidate_decode_cache_epochs(uint64_t content_epoch, uint64_t codec_epoch);
    void clear_decode_cache();
    const llama_cparams & get_cparams() const { return cparams_; }

    // -- readiness / capability -------------------------------------------
    // True only for DENSE/SR caches with a bounded hot pool.
    bool bounded_hot_configured() const { return bounded_hot_; }
    // Computed readiness (never a fixed bool): set by refresh_readiness()
    // from actual hot-tensor backends + profile/source. Callers fail closed
    // while false. Graph rechecks per op.
    bool attention_path_ready() const { return readiness_.ready; }
    const char * readiness_reason() const { return readiness_.reason; }
    const xkv_path_readiness & readiness() const { return readiness_; }
    // Recompute readiness from live cache state (target-layer hot tensor
    // backends + capability gate). Called automatically by ensure_groups.
    void refresh_readiness(const llama_kv_cache & kv) const;
    // Legacy attention is safe only when hot tensors are not truncated.
    bool legacy_attention_safe() const { return !bounded_hot_; }
    // Per-layer XKV gate for graph routing: true only for target-trunk
    // owning layers (il < n_layer() && has_kv && !recr && !swa as captured at
    // the last rebuild_groups). False for SWA/recurrent/MTP layers and when
    // no groups were built (fail closed). Global bounded mode alone must not
    // route SWA layers onto the XKV path.
    bool xkv_layer_enabled(uint32_t il) const;

    // -- layer groups -------------------------------------------------------
    // Rebuild deterministic alias-deduplicated target-trunk groups from actual
    // hparams/model/cache state. Target trunk layers satisfy
    // il < n_layer() && has_kv(il) && !is_recr(il) && !is_swa(il); MTP/draft
    // layers are excluded via n_layer(). Aliased model layers collapse to one
    // owning layer (cache resolution order), preserving first-seen order; the
    // tail group may be smaller than group_size.
    bool rebuild_groups(const llama_hparams & hparams, const llama_kv_cache & kv, std::string * err = nullptr) const;
    const layer_group_map & group_map() const { return groups_; }
    bool groups_valid() const { return groups_valid_; }
    // Single device-placement identity all groups were validated against
    // (empty when groups are invalid). Consumed by fingerprinting and the
    // native graph backend check.

    // Per-group placement identity (parallel group_map().groups).
    const std::vector<std::string> & group_placements() const { return group_placements_; }
    const std::string & placement_identity() const { return placement_id_; }

    // -- immutable read helpers ----------------------------------------------
    // Collect eligible homogeneous committed PUBLIC rows: ordinary (normal) or
    // public_live visibility; store state hot_committed with kind hot; exact
    // hot binding with matching generation; single stream (homogeneous).
    // Excludes MTP/draft, recurrent/SWA layers (via groups), PRIVATE_CONTROL,
    // PENDING_RECORD, hot_writing, shared semantic mismatches (conflicting
    // storage_pos/generation for one payload), in-flight reservations
    // (quiescence pre-check), and already-factored payloads. Deduplicates
    // shared-prefix payloads to one row. Sorted by (storage_pos, payload_id).
    bool collect_eligible_rows(
        const llama_kv_cache & kv,
        std::vector<xkv_layer_row_view> & out,
        std::string * err = nullptr) const;

    // Bulk-canonicalize one owning layer for exactly the given rows (physical
    // hot rows + cells-owned storage positions) into concatenated per-head
    // canonical K/V matrices: out_k/out_v are [n_rows, n_embd_k/v_gqa].
    bool read_layer_canonical(
        const llama_kv_cache & kv,
        uint32_t owning_layer,
        const std::vector<xkv_layer_row_view> & rows,
        matrix & out_k,
        matrix & out_v,
        std::string * err = nullptr) const;

    // -- maintenance ----------------------------------------------------------
    // Fixed Tri fill-first interaction: when a bounded hot pool exists and
    // forced=false, maintain() NEVER seals or rebinds while hot slots
    // remain free (hot_free >= upcoming_tokens). It returns true with
    // no action taken (fill-first preservation). When pressure hits
    // (hot_free < upcoming_tokens, or forced=true), it holds the
    // transaction maintenance exclusion, partitions by domain, and seals
    // the oldest full segment. SHADOW mode may evaluate without release
    // regardless of pressure (observability).
    xkv_maintenance_outcome maintain(
        llama_kv_cache & kv, uint32_t upcoming_tokens = 0, bool forced = false, std::string * err = nullptr);

    // -- admission --------------------------------------------------------------
    // Fills the authoritative public snapshot (domains: logical, hot,
    // factor store, workspace) and finalizes it. Recurrent fields stay zero
    // (hybrid memories combine). Factor/workspace token bounds come from
    // real segment/batch planners, never defaulted.
    bool fill_admission_snapshot(
        const llama_kv_cache & kv,
        llama_memory_admission_snapshot & out,
        std::string * err = nullptr) const;

    // Fills the authoritative observed runtime snapshot from immutable
    // effective config, exact store accounting/stream descriptors, backend
    // residency, tile cache, and runtime stats. Never synthesizes missing
    // fields (unknown stays zero). Armed only after path readiness plus at
    // least one live runtime observation (a maintain run).
    bool fill_runtime_snapshot(
        const llama_kv_cache & kv,
        llama_memory_xkv_runtime_snapshot & out,
        std::string * err = nullptr) const;

    // -- graph snapshot ----------------------------------------------------------
    const xkv_runtime_stats & stats() const { return stats_; }

    // Default per-group landmark rebuild bound to this runtime's profile.
    xkv_group_landmark_rebuild_fn group_landmark_rebuild_fn() const;
    // Positions-aware removal hook for seq_rm/seq_keep/overwrite/cancel:
    // validates cells-owned positions + refreshes cached phase BEFORE any
    // mutation (false fails closed). Shape matches the cache setter.
    std::function<bool(const std::vector<std::pair<uint32_t, uint32_t>> &, std::string *)>
        bind_removal_hook(const llama_kv_cache & kv);
    // Store-compatible closure over the last validated capture (generation
    // guarded; refuses on gaps). Wired by the cache into store removals.
    xkv_landmark_rebuild_fn bound_removal_callback() const;
    // Direct capture builder (also used by the hook above).
    bool capture_pack_rebuild_ctx(const llama_kv_cache & kv, std::string * err = nullptr) const;
    // Context-shift coupling: factored canonical K needs no byte change,
    // but stored SR landmarks are means of K phased at OLD storage
    // positions. Before cells commit shifted positions, the shift path calls
    // this to prebuild exact new phase-aware landmarks from final A_K/B_K +
    // shifted positions (cells NOT yet mutated: pid_to_new_pos overrides
    // cells-owned positions for shifted payloads). Returns per-group encoded
    // landmarks in segment group order for COW-publication; the coordinator
    // fence then commits cells positions/hot rotation/stamp atomically.
    // Refuses on any missing position or phase (fail closed, no mutation).
    bool rebuild_landmarks_for_shift(
        const llama_kv_cache & kv,
        uint64_t segment_id,
        // Sorted by pid ascending (binary-searched, never hashed).
        const std::vector<std::pair<uint64_t, int64_t>> & pid_to_new_pos,
        std::vector<encoded_matrix> & out_landmarks,
        std::string * err = nullptr) const;
    // Store-level pack callback (xkv_landmark_rebuild_fn, owned by
    // llama-xkv-cache.h) bound to this runtime's profile, for wiring into
    // remove_payload(s)/pack_segment/mutation pack paths.
    xkv_landmark_rebuild_fn store_pack_rebuild_fn() const;

private:
    bool ensure_groups(const llama_hparams & hparams, const llama_kv_cache & kv, std::string * err) const;
    xkv_bundle_sealing_params sealing_params() const;
    // Scratch enumeration: fills eligible + eligible_deduped (pid-deduped,
    // enumeration order) with zero per-call heap after warmup. fill_seq_ids
    // populates keeper sets (public/test path); maintain path computes only
    // the fingerprint inline.

    // Refresh cached per-group phase specs when groups changed (capacity
    // retained; cold-path allocation only on topology change).
    bool refresh_phase_cache(const llama_kv_cache & kv, std::string * err) const;
    bool enumerate_into_scratch(const llama_kv_cache & kv, bool fill_seq_ids, std::string * err) const;

    // Streams one owning layer into workspace slices (see .cpp): temp bound
    // is one head, never a full layer; syncs accumulate into stats.
    bool stream_layer_canonical(
        const llama_kv_cache & kv,
        uint32_t owning_layer,
        float * k_dst, uint64_t k_row_stride, uint32_t k_off,
        float * v_dst, uint64_t v_row_stride, uint32_t v_off,
        canonical_batch_stats * stats,
        std::string * err) const;

    llama_cparams cparams_;
    xkv_effective_config eff_;
    bool bounded_hot_ = false;
    mutable layer_group_map groups_;
    mutable bool groups_valid_ = false;
    mutable std::vector<char> layer_enabled_; // per model layer, from last rebuild

    mutable std::vector<std::string> group_placements_; // parallel groups_.groups
    mutable std::string placement_id_; // first group placement (informational)
    xkv_transaction_coordinator * coord_ = nullptr;
    xkv_device_upload_fn upload_fn_;
    std::shared_ptr<xkv_b_tile_cache> b_tile_cache_;
    xkv_runtime_stats stats_;
    uint64_t next_maint_id_ = 1;
    mutable xkv_maintain_control_scratch scratch_;
    // Serializes maintain() across every cache sharing this runtime: the
    // shared-store family drives one logical maintenance path.
    mutable std::mutex maint_mtx_;
    mutable xkv_path_readiness readiness_;
    mutable uint64_t groups_version_ = 0;
    mutable std::vector<xkv_group_phase_spec> cached_phase_specs_;
    mutable uint64_t cached_phase_version_ = 0;
    mutable xkv_removal_capture_state removal_capture_;
    // Scratch topology versions: inputs/phase rebuilt only when groups change.
    uint64_t scratch_inputs_version_ = 0;
    uint64_t scratch_phase_version_ = 0;
};

// OFF fallback: POD admission snapshot mapping legacy capacity/usage exactly.
// Canonical bytes per admitted token for the target trunk (exact geometry
// sum, no group/state dependency): sum over eligible layers of
// (n_embd_k_gqa + n_embd_v_gqa) * sizeof(float) plus one head-temp bound.
uint64_t canonical_bytes_per_token(const llama_hparams & hparams);

// Shared decode-cache metrics (honest zeros when the native path bypasses).
struct xkv_decode_cache_metrics {
    size_t max_bytes = 0;
    size_t active_bytes = 0;
    size_t total_allocated_bytes = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
};

// Code-stream residency conformance (ggml/include/ggml-xkv.h contract):
// reference/host-assisted profiles retain host bytes; Vulkan TQ profiles are
// device-owned via one exact-bytes upload per immutable segment bundle with
// host release by the backend owner. The runtime never uploads per decode and
// never retains dense mirrors; it only decides the expectation, preflights
// codec support, and reports measured bytes.
ggml_xkv_residency xkv_expected_residency(const llama_cparams & cparams);

struct xkv_code_stream_residency {
    uint64_t segment_id = 0;
    uint64_t segment_version = 0;
    ggml_xkv_residency residency = GGML_XKV_RES_REFERENCE_HOST;
    size_t host_bytes = 0;   // retained host code-stream bytes
    size_t device_bytes = 0; // exact device bytes (0 for reference-host)
    bool verified = false;   // device upload verification is the backend owner's job
    char backend[16] = {};   // factorizer name, e.g. "host", "vulkan"
};

// Measure a published segment against the residency contract. Fails closed on
// invalid descriptors or unsupported codec/residency combinations (never
// silently mirrored). Does not move any bytes.
bool report_code_stream_residency(
    const xkv_segment & segment,
    const llama_cparams & cparams,
    xkv_code_stream_residency & out,
    std::string * err = nullptr);

} // namespace llama_xkv
