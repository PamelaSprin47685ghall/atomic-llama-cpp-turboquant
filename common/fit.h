#pragma once

#include "ggml.h"
#include "llama.h"

#include <utility>
#include <vector>

enum common_params_fit_status {
    COMMON_PARAMS_FIT_STATUS_SUCCESS = 0, // found allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_FAILURE = 1, // could not find allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_ERROR   = 2, // a hard error occurred, e.g. because no model could be found at the specified path
};

// fits mparams and cparams to free device memory (assumes system memory is unlimited)
//   - returns true if the parameters could be successfully modified to fit device memory
//   - this function is NOT thread safe because it modifies the global llama logger state
//   - only parameters that have the same value as in llama_default_model_params are modified
//     with the exception of the context size which is modified if and only if equal to 0
common_params_fit_status common_fit_params(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams,
                              float * tensor_split,          // writable buffer for tensor split, needs at least llama_max_devices elements
   llama_model_tensor_buft_override * tensor_buft_overrides, // writable buffer for overrides, needs at least llama_max_tensor_buft_overrides elements
                             size_t * margins,               // margins of memory to leave per device in bytes
                           uint32_t   n_ctx_min,             // minimum context size to set when trying to reduce memory use
                     ggml_log_level   log_level);            // minimum log level to print during fitting, lower levels go to debug log

common_params_fit_status common_fit_kv_cache(
                         const char * path_model,
           const llama_model_params * mparams,
               llama_context_params * cparams,
        const std::vector<std::pair<ggml_backend_dev_t, size_t>> & reserve,
        const llama_context_params * extra_cparams,
                     ggml_log_level   log_level);

common_params_fit_status common_fit_recurrent_cache(
                         const char * path_model,
           const llama_model_params * mparams,
               llama_context_params * cparams,
        const std::vector<std::pair<ggml_backend_dev_t, size_t>> & reserve,
        const llama_context_params * extra_cparams,
                     ggml_log_level   log_level);

struct common_rerot_fit_result {
    common_params_fit_status status = COMMON_PARAMS_FIT_STATUS_FAILURE;
    uint32_t b_people = 0;
    uint32_t p_pens = 0;
    uint32_t k_tokens = 0;
    uint32_t k_min = 0;
    int64_t  min_device_margin = 0;
};

// XKV auto-fit reserve breakdown (bytes). The decode tile cache is a global
// sub-budget of the workspace (§8.6/16): workspace_mib is the TOTAL transient
// budget; decode_cache lives inside it, and factor/graph/source peaks must
// fit the remainder. total contributes workspace ONCE (never the sum).
struct common_xkv_fit_reserve {
    uint64_t workspace_bytes     = 0; // xkv_workspace_mib in bytes
    uint64_t decode_cache_bytes = 0; // xkv_decode_cache_mib in bytes
    uint64_t factor_scratch_bytes = 0; // exact seal-scratch requirement (inside)
    uint64_t dedup_scratch_bytes = 0; // store-owned host dedup vectors (outside arena)
    uint64_t total_bytes        = 0; // == workspace_bytes (once; device-partitioned)
};

// Compute the exact-once XKV reserve for auto-fit probes. scratch_bytes is
// the exact seal-scratch requirement (common_xkv_scratch_bytes): it must fit
// the workspace remainder after the decode sub-budget (scratch + decode <=
// workspace), else false (fail closed) with total UINT64_MAX. dedup_bytes
// is the store-owned host dedup-scratch upper bound
// (common_xkv_dedup_scratch_bytes): host-resident, outside the workspace
// arena, carried for host-charge accounting (never device-partitioned).
// XKV OFF (or null args) yields true with all zeros. total is workspace
// once.
bool common_xkv_fit_reserve_bytes(const llama_context_params * cparams, uint64_t scratch_bytes,
        common_xkv_fit_reserve * out, uint64_t dedup_scratch_bytes = 0);

// Persistent factor store budget in bytes, checked. This is separate from
// the transient reserve above: workspace/decode/scratch are seal-time peaks,
// the store budget caps persistent factor bytes. Returns 0 when XKV is OFF,
// the mode keeps no persistent store (SHADOW reference), or no explicit
// budget is configured (0 = auto-derive). UINT64_MAX on overflow.
uint64_t common_xkv_store_budget_bytes(const llama_context_params * cparams);

// Device-owned share of a persistent store budget, derived from the effective
// residency predicate (llama_xkv_profile_is_device_owned), never from the
// storage profile alone: tq-* + cpu-reference is host-resident (0 device
// bytes, full host charge); TQ profiles on device factorizers are fully
// device-owned. The host share is exactly store_bytes minus the return.
// Hermetic and fully unit-tested.
uint64_t common_xkv_store_device_bytes(const llama_context_params * cparams, uint64_t store_bytes);

// Derive the persistent store budget (MiB) from a fitted logical KV
// capacity with a single dense-context probe: E = C*(1-min_saving), minimum
// 1 MiB, rounded up. Returns 0 when the probe fails, overflows, or the
// capacity is zero. Never conflates transient workspace peaks.
uint32_t common_xkv_derive_store_mib(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        uint32_t n_ctx_kv_fit,
        ggml_log_level log_level);

// Pure store-budget math: MiB for dense-equivalent context bytes C.
// E = C*(1-min_saving), minimum 1 MiB, rounded up. Returns 0 on overflow,
// zero input, or non-finite saving. Hermetic and fully unit-tested.
uint32_t common_xkv_store_mib_for_bytes(uint64_t dense_ctx_bytes, double min_saving);

// Store budget with worst-case one-segment seal/COW overlap: the base E for
// C plus one segment's factored share (same ratio, capped at E), so an
// in-flight COW replacement (old + new versions live) never exceeds budget
// and runtime maintenance is not refused at pressure. Backend alignment is
// covered by the MiB round-up. Returns 0 on any failure. Hermetic.
uint32_t common_xkv_store_mib_with_overlap(
        uint64_t dense_ctx_bytes,
        double min_saving,
        uint32_t seg_tokens,
        uint32_t k_tokens);

// Partition a global XKV byte budget across devices by target-layer
// placement weights (per-device context bytes). Shares sum to exactly
// total (remainder goes to the largest weight); all-zero weights split
// equally. Checked; returns false (shares zeroed) on overflow or bad args.
// n==0 returns true (nothing to do). Hermetic and fully unit-tested.
bool common_xkv_partition_budget(uint64_t total, const uint64_t * weights, uint64_t * shares, size_t n);

// Exact seal-scratch bound for one segment group with feature widths
// Dk/Dv (largest alias-dedup target group): rSVD factorize peak via the
// exact factor estimators, plus the quantized-shadow peak (live FP factors +
// exact candidate encoded bytes via encoded_matrix_bytes, rank-128 Turbo
// padding included: encoded can EXCEED FP at tiny rank, never inferred from
// FP logical size), canonical pre-RoPE capture peak, and landmark stream
// bound. All checked; false on overflow or unsupported types. Hermetic and
// fully unit-tested. Replaces any rank*segment_tokens*sizeof(float) toy
// estimate.
bool common_xkv_scratch_for_group(
        uint32_t seg_tokens,
        uint32_t chunk_tokens,
        uint64_t feat_k,
        uint64_t feat_v,
        uint32_t rank_k,
        uint32_t rank_v,
        ggml_type landmark_type,
        uint64_t * out_bytes,
        ggml_type type_a_k = GGML_TYPE_TURBO4_0,
        ggml_type type_b_k = GGML_TYPE_TURBO4_0,
        ggml_type type_a_v = GGML_TYPE_TURBO4_0,
        ggml_type type_b_v = GGML_TYPE_TURBO4_0);

// Seal-scratch bound from model hparams: walks trunk attention layers
// (il < n_layer, has_kv, !recr, !swa), takes the largest group_size-wide
// alias-dedup group widths, and calls common_xkv_scratch_for_group.
// Loads only a no-alloc metadata model (never XKV runtime). False when the
// model cannot be inspected. Comparison against the instrumented runtime
// peak is integration work after FactorizerSafety lands.
// Optional out_dedup_scratch_bytes receives the store-owned host
// dedup-scratch upper bound from the fit topology (segments x groups x
// backend handles); host-resident, outside the workspace arena, charged to
// the host bucket, never device-partitioned.
bool common_xkv_scratch_bytes(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        uint64_t * out_bytes,
        uint64_t * out_dedup_scratch_bytes = nullptr);

// VRAM-only Three-Capacity Auto-Fit for RERoT (§§B.0, B.10, B.13 Phase 8):
// Jointly selects people B, pens P, and maximum feasible aligned KV capacity K without user capacity knobs.
common_rerot_fit_result common_fit_rerot_capacities(
                         const char * path_model,
           const llama_model_params * mparams,
               llama_context_params * cparams,
        const std::vector<std::pair<ggml_backend_dev_t, size_t>> & reserve,
        const llama_context_params * extra_cparams,
                     ggml_log_level   log_level);

// print estimated memory to stdout
void common_fit_print(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams);

void common_memory_breakdown_print(const llama_context * ctx);

struct common_device_memory_data {
    int64_t total;
    int64_t free;
    size_t  model;
    size_t  context;
    size_t  compute;
};

using common_device_memory_data_vec = std::vector<common_device_memory_data>;

// Load a model + context with no_alloc and return the per-device memory breakdown.
common_device_memory_data_vec common_get_device_memory_data(
                         const char * path_model,
           const llama_model_params * mparams,
         const llama_context_params * cparams,
    std::vector<ggml_backend_dev_t> & devs,
                           uint32_t & hp_ngl,
                           uint32_t & hp_n_ctx_train,
                           uint32_t & hp_n_expert,
                     ggml_log_level   log_level);
