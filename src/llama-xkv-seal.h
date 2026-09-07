#pragma once

// llama-xkv-seal.h — Production native sealing bridge (XKV).
//
// Builds ordered async per-group graphs reusing ONE max-group transient
// allocation (via ggml_gallocr) while persistent compact outputs (A_K, B_K,
// A_V, B_V, and optional landmark streams) accumulate in ctx_out. Single
// backend queue execution with ONE final host sync at the end, status-only
// D2H plus singular and exact all-row tiled final-codec residual telemetry.
// Returns descriptor-mapped immutable backend handles (via BackendResidency
// adoption) ready for atomic store publish.
//
// Design boundaries (see local://xkv-vulkan-runtime.md + bridge contracts):
//  - BackendResidency owns allocation/adoption/lifetime: the bridge calls
//    xkv_backend_adopt_device_tensors with its DAG tensors + store-owned
//    id_gen + reservation. No host full K/V/factor bytes exist at any point.
//  - RuntimeSealer owns group config, call site, and publish: it fills
//    xkv_native_seal_config from live cache state and calls
//    xkv_native_seal_build; the store adopts-or-refuses the returned handles.
//  - GraphRuntime owns decode-time graphs: untouched here.
//  - Landmark generation: supported natively via ggml_xkv_landmark_build
//    when want_landmarks=true, reconstructing from final encoded A_K/B_K
//    streams with per-row forward RoPE, chunk mean-pooling, and Q8_0 or
//    Turbo4_0 quant encoding in the same DAG.
//  - Error gating: exact all-row tiled final-codec residuals (A+B, A-only,
//    B-only) are evaluated device-side and reported as scalar metrics;
//    A+B relative error is gated against max_rel_error before publication.
//
// Owned by XkvVulkanFactorizer. No edits to cache/runtime/store/graph.

#include "ggml.h"
#include "ggml-backend.h"
#include "llama-xkv-codec.h"
#include "llama-xkv-backend.h"
#include "llama-xkv-factor.h" // factor_error_report fields (gate evidence)
#include "llama-xkv-cache.h"  // xkv_landmark_chunk definition

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llama_xkv {

class xkv_device_staging_reservation;

// One owning layer's hot inputs. Hot tensors are BORROWED cache storage
// (backend-resident; Sealer guarantees quiescence/pins through the call).
// Host tables are copied in by the bridge (tiny metadata uploads).
struct xkv_native_seal_hot_layer {
    ggml_tensor * hot_k = nullptr; // [n_heads*padded_k, n_phys] typed hot K
    ggml_tensor * hot_v = nullptr; // [n_heads*padded_v, n_phys] typed hot V
    uint32_t n_heads = 0;
    uint32_t head_dim_k = 0;       // logical widths (<= padded)
    uint32_t head_dim_v = 0;
    // K inverse-RoPE tables (host, [Fc] each; empty iff rotary_dim_k==0):
    uint32_t rotary_dim_k = 0;
    int32_t rope_mode_k = 0;       // GGML_XKV_ROPE_HALF / _INTERLEAVED
    std::vector<float> rope_omega_k;
    std::vector<float> rope_mag_k;
    // Optional attention rotations ([HxH] each; empty = skip):
    std::vector<float> hadamard_k;
    std::vector<float> hadamard_v;
    // Optional channel scales: null/empty/all-ones only; anything else
    // fails closed (no kernel support).
    std::vector<float> channel_scales_k;
    std::vector<float> channel_scales_v;
};

// One factor group's seal inputs.
struct xkv_native_seal_group {
    uint32_t group_index = 0;
    std::vector<xkv_native_seal_hot_layer> layers; // >= 1 owning layer
    uint32_t rank_k = 0;                           // clamped, > 0
    uint32_t rank_v = 0;
    ggml_type codec_a_k = GGML_TYPE_TURBO4_0;
    ggml_type codec_b_k = GGML_TYPE_TURBO4_0;
    ggml_type codec_a_v = GGML_TYPE_TURBO4_0;
    ggml_type codec_b_v = GGML_TYPE_TURBO4_0;
    uint64_t seed_k = 42;                          // K pair seed (V pair +1 convention is Sealer-side)
    uint64_t seed_v = 43;
    uint32_t oversampling = 16;
    uint32_t power_iterations = 2;                 // >= 1
    uint32_t balance_mode = 1;                     // GGML_XKV_BALANCE_SQRT production default
    bool want_landmarks = false;                   // true => build landmark stream from final A_K/B_K
    ggml_type landmark_type = GGML_TYPE_TURBO4_0;  // Q8_0 (8) or TURBO4_0 (44)
    uint64_t landmark_seed = 777;                  // canonical codec table seed (default 777)
    uint32_t chunk_tokens = 8;                     // rows per landmark chunk (>0)
    uint64_t phase_tx_fingerprint = 0;             // phase transform identity
};

// Standalone bounded native landmark rebuild (for context-shift COW and per-seq compaction).
// Operates directly on immutable backend-resident A_K/B_K handles without host factors.
struct xkv_native_landmark_rebuild_request {
    std::shared_ptr<const xkv_backend_allocation> a_k; // immutable token-major A_K handle
    std::shared_ptr<const xkv_backend_allocation> b_k; // immutable feature-major B_K handle
    std::vector<int32_t> surviving_rows;              // [n_rows] row indices into A_K
    std::vector<int64_t> storage_positions;           // [n_rows] signed 64-bit phase positions
    uint32_t chunk_tokens = 8;                        // rows per landmark chunk (>0)
    ggml_type landmark_type = GGML_TYPE_TURBO4_0;     // Q8_0 or TURBO4_0
    uint64_t seed = 777;                              // canonical landmark codec seed (default 777)
    uint64_t phase_tx_fingerprint = 0;
    std::vector<xkv_native_seal_hot_layer> layers;     // layer phase geometry (offsets/dims/rope)
    std::shared_ptr<struct ggml_backend> executor;    // REQUIRED non-null backend owner
    ggml_backend_buffer_type_t buft = nullptr;        // placement buffer type
    xkv_allocation_id_generator * id_gen = nullptr;   // store ID generator
    const xkv_backend_store_reservation * store_reservation = nullptr; // gates output handle
    const xkv_device_staging_reservation * staging_reservation = nullptr; // gates transient scratch
};

struct xkv_native_landmark_rebuild_result {
    std::shared_ptr<xkv_backend_allocation> landmark_handle; // adopted immutable device handle
    codec_desc desc;
    std::vector<xkv_landmark_chunk> chunks;
    uint64_t table_fingerprint = 0;
    uint64_t stream_fingerprint = 0;
    size_t exact_bytes = 0;
    size_t transient_bytes = 0;
    uint64_t sync_count = 0;
};

// Rebuilds landmark stream on device via single async submission and sync.
// Output untouched on any failure.
bool xkv_native_landmark_rebuild(
    const xkv_native_landmark_rebuild_request & req,
    xkv_native_landmark_rebuild_result & out,
    std::string * err = nullptr);

struct xkv_native_seal_estimate_result {
    size_t persistent_dest_bytes = 0; // exact adopted streams (destination)
    size_t max_transient_bytes   = 0; // max transient gallocr workspace (scratch + intermediate)
    size_t combined_peak_bytes   = 0; // persistent_dest_bytes + max_transient_bytes (telemetry / auto-fit)
};

struct xkv_native_seal_config;

// Pure checked geometry estimator: calculates exact persistent_dest_bytes and
// max_transient_bytes before any backend allocation or ID consumption.
// The runtime acquires reservations matching these exact numbers.
bool xkv_native_seal_estimate(
    const xkv_native_seal_config & config,
    xkv_native_seal_estimate_result & out,
    std::string * err = nullptr);

// Whole-bundle seal request.
struct xkv_native_seal_config {
    // Persistent executor owner: REQUIRED non-null with ggml_backend_free deleter.
    // Raw backend is derived from executor.get(); fake no-op shared_ptr is rejected.
    std::shared_ptr<struct ggml_backend> executor;
    ggml_backend_t backend = nullptr;              // raw alias; if non-null must == executor.get()
    ggml_backend_buffer_type_t buft = nullptr;     // placement (borrowed)
    ggml_xkv_residency residency = GGML_XKV_RES_DEVICE_OWNED;
    uint32_t n_rows = 0;
    std::vector<int32_t> physical_rows;            // [n] selected hot rows (shared layers)
    std::vector<int64_t> storage_positions;        // [n] full signed 64-bit phase space (I64 tensor, no truncation)
    std::vector<xkv_native_seal_group> groups;     // >= 1
    xkv_allocation_id_generator * id_gen = nullptr; // store-owned, never reset (required)
    // Split reservations:
    // store_reservation gates persistent destination bytes only (xkv_store_mib budget).
    // staging_reservation gates transient workspace arena bytes only (device peak budget).
    // Neither policy cap is compared against combined_peak!
    const xkv_backend_store_reservation * store_reservation = nullptr; // null = unenforced
    const xkv_device_staging_reservation * staging_reservation = nullptr; // null = unenforced
    std::string placement_identity;                // must equal backend/executor name if nonempty
    // Publish gate: device-measured A+B final-codec relative error must not
    // exceed this (Sealer sets from profile max_relative_error; required > 0).
    // A-only/B-only residuals are evaluated device-side for diagnostics;
    // only the A+B product is reported and gates publication.
    double max_rel_error = 0.25;
};

// One adopted sealed stream: immutable handle + store-facing descriptor.
struct xkv_native_sealed_stream {
    std::shared_ptr<xkv_backend_allocation> handle;
    codec_desc desc;
    uint64_t stream_fingerprint = 0;
    size_t exact_bytes = 0;
};

// One sealed group: four adopted streams + device telemetry.
struct xkv_native_sealed_group {
    uint32_t group_index = 0;
    uint32_t rank_k = 0;   // effective ranks (clamped to geometry)
    uint32_t rank_v = 0;
    xkv_native_sealed_stream a_k, b_k, a_v, b_v;
    // Optional landmark stream (populated iff want_landmarks == true)
    bool has_landmarks = false;
    xkv_native_sealed_stream landmark;
    std::vector<float> landmark_eb;      // per-chunk error bound (telemetry)
    std::vector<uint64_t> landmark_srcfp; // per-chunk FNV source fingerprint
    std::vector<float> singular_k; // device S (sorted desc, telemetry for gates)
    std::vector<float> singular_v;
    factor_error_report err_k;    // device-measured FINAL-codec A+B residual
    factor_error_report err_v;    // (decode on device, scalar D2H only)
    int32_t status_canon = -1;     // nonzero if any layer canonicalize failed
    int32_t status_fact_k = -1;
    int32_t status_fact_v = -1;
    int32_t status_lmbuild = -1;   // 0 if landmark built OK, -1 if skipped
};

// Whole-bundle result. Assigned ONLY on success (failure-atomic: untouched
// otherwise). storage keeps every adopted tensor valid; Sealer holds the
// bundle until store publish (adoption already attached handles).
struct xkv_native_seal_bundle {
    std::vector<xkv_native_sealed_group> groups;
    uint64_t bundle_fingerprint = 0;
    size_t factored_bytes = 0;   // exact adopted stream bytes (dedup-free sum)
    size_t scratch_bytes = 0;    // device scratch peak (compute-side, freed with storage)
    size_t preflight_bytes = 0;  // exact bridge-buffer bytes preflighted (T-1 gate observable)
    size_t peak_source_bytes = 0; // exact DAG buffer footprint (from_buft_size)
    size_t peak_dest_bytes = 0;   // exact adopted-stream footprint (alloc sizes)
    // preflight_bytes = peak_source + peak_dest (combined peak alive
    // simultaneously during adoption copy; the gated quantity).
    // Persistent executor handle (shared RAII owner, ggml_backend_free deleter):
    // keeps the device executor alive with the bundle/store lifetime so
    // subsequent operations (readers, Tri, pack) never leak or re-init backends.
    std::shared_ptr<struct ggml_backend> executor;
    // Committed adoption batch result holding all adopted stream allocations,
    // executor ownership, and success state. RuntimeSealer attaches this
    // directly to candidate->backend_bundle for atomic candidate publish.
    std::shared_ptr<const xkv_backend_batch_result> backend_bundle;
    uint64_t sync_count = 0;     // DAG syncs: exactly 1 on success
    uint64_t adopt_sync_count = 0;
};

// Failure-atomic single-DAG native seal. Returns true with `out` fully
// populated, or false with `out` bit-identical to entry and `err` set.
// No host full K/V/factor bytes on any path; D2H is statuses + S telemetry.
bool xkv_native_seal_build(
    const xkv_native_seal_config & config,
    xkv_native_seal_bundle & out,
    std::string * err = nullptr);

} // namespace llama_xkv
