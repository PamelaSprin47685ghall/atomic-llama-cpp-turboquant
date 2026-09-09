#pragma once

#include "llama.h"
#include "llama-xkv-codec.h"

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace llama_xkv {

// Row-major 2D float matrix
struct matrix {
    uint64_t rows = 0;
    uint64_t cols = 0;
    std::vector<float> data;
    float * external_data = nullptr;
    const float * const_external_data = nullptr;

    matrix() = default;
    matrix(uint64_t r, uint64_t c, float init_val = 0.0f);
    matrix(uint64_t r, uint64_t c, std::vector<float> d);
    matrix(uint64_t r, uint64_t c, float * ext_ptr, float init_val = 0.0f);
    matrix(uint64_t r, uint64_t c, const float * ext_ptr)
        : rows(r), cols(c), const_external_data(ext_ptr) {}

    float & at(uint64_t r, uint64_t c);
    const float & at(uint64_t r, uint64_t c) const;
    float * row_ptr(uint64_t r);
    const float * row_ptr(uint64_t r) const;
    uint64_t elements() const;
    bool empty() const;
    bool has_non_finite() const;
    bool operator==(const matrix & o) const;
    bool operator!=(const matrix & o) const { return !(*this == o); }
};

// Matrix operations
matrix matrix_transpose(const matrix & m);
matrix matrix_matmul(const matrix & a, const matrix & b);
// Reconstruct from A (n x r) and B^T (m x r): X_{ij} = sum_{k=0}^{r-1} A_{ik} * B^T_{jk}
matrix matrix_reconstruct(const matrix & a, const matrix & b_transposed);

// Layer grouping and owning layer mapping
struct layer_group {
    uint32_t group_index = 0;
    std::vector<uint32_t> owning_layers; // Non-contiguous owning layers (e.g. [3, 7, 11, 15])

    // Slicing within concatenated group matrix
    std::vector<uint32_t> layer_feature_offsets_k;
    std::vector<uint32_t> layer_feature_dims_k;
    std::vector<uint32_t> layer_feature_offsets_v;
    std::vector<uint32_t> layer_feature_dims_v;

    uint32_t total_dim_k = 0;
    uint32_t total_dim_v = 0;
};

struct layer_group_map {
    uint32_t group_size = 4;
    std::vector<layer_group> groups;
    std::vector<uint32_t> unique_owning_layers;
    // Map from model_layer -> group_index (UINT32_MAX if non-attention or invalid)
    std::vector<uint32_t> model_layer_to_group;
    // Map from model_layer -> offset within its group (UINT32_MAX if non-attention)
    std::vector<uint32_t> model_layer_to_group_offset;
};

// Build layer group mapping: deduplicates aliases, preserves non-contiguous layer ordering,
// handles group sizes 1, 2, 4, and tail groups.
layer_group_map build_layer_group_map(
    const std::vector<uint32_t> & owning_layers,
    uint32_t group_size = 4,
    uint32_t dim_k = 128,
    uint32_t dim_v = 128
);

layer_group_map build_layer_group_map_ex(
    const std::vector<uint32_t> & model_to_owning,
    const std::vector<bool>     & is_attention_layer,
    uint32_t group_size,
    const std::vector<uint32_t> & dim_k_per_layer,
    const std::vector<uint32_t> & dim_v_per_layer
);

// Factor balancing options (alias to llama_xkv_factor_balance)
using factor_balance = llama_xkv_factor_balance;
const char * factor_balance_to_str(factor_balance balance);

enum class factor_fault_injection : uint32_t {
    none = 0,
    fail_qr = 1,
    fail_svd = 2,
};

// Factorizer configuration
struct factor_config {
    uint32_t algorithm_version = 2; // 2 = deterministic rSVD with integer Rademacher Omega, power iterations and CGS2
    uint32_t random_distribution = 0; // 0 = Rademacher (+-1) for exact cross-libm determinism
    uint32_t rank_k = 384;
    uint32_t rank_v = 576;
    factor_balance balance = LLAMA_XKV_FACTOR_BALANCE_UPSTREAM;
    uint64_t seed = 42;
    uint32_t group_size = 4;

    uint32_t oversampling = 16;
    uint32_t power_iterations = 2; // at least 1 checked power iteration; 0 is rejected

    // Quantized shadow codec types
    ggml_type factor_a_k = GGML_TYPE_TURBO4_0;
    ggml_type factor_b_k = GGML_TYPE_TURBO4_0;
    ggml_type factor_a_v = GGML_TYPE_TURBO4_0;
    ggml_type factor_b_v = GGML_TYPE_TURBO4_0;

    double svd_tolerance = 1e-9;
    uint32_t max_svd_sweeps = 30;

    // Hard memory limits for workspace preflight
    uint64_t max_workspace_bytes = 1024ULL * 1024ULL * 1024ULL; // 1 GiB default ceiling

    // Fault injection hooks (XKV-SR §15.4: zero-cost default, excluded from fingerprint)
    factor_fault_injection fault = factor_fault_injection::none;
    uint32_t fault_occurrence = 0; // 0 = trigger on first match

    uint64_t fingerprint() const;
};

struct factor_pair;

class xkv_arena_lease;

struct factor_workspace_span {
    void * data = nullptr;
    size_t size_bytes = 0;

    factor_workspace_span() = default;
    factor_workspace_span(void * d, size_t sz) : data(d), size_bytes(sz) {}
    template <typename Lease, typename = decltype(std::declval<const Lease&>().data()), typename = decltype(std::declval<const Lease&>().size())>
    factor_workspace_span(const Lease & lease) : data(const_cast<void*>(lease.data())), size_bytes(lease.size()) {}

    bool valid() const { return data != nullptr && size_bytes > 0; }
    explicit operator bool() const { return valid(); }
};

struct factor_execution_context {
    factor_workspace_span workspace;

    factor_execution_context() = default;
    factor_execution_context(factor_workspace_span ws) : workspace(ws) {}
    factor_execution_context(void * ptr, size_t sz) : workspace(ptr, sz) {}
};

// Exact peak workspace estimator with checked arithmetic.
// Returns false on integer overflow or invalid dimension, setting *out_bytes = UINT64_MAX.
// Includes output A (n x r), B^T (m x r), S (r), plus all simultaneously live scratch temporaries.
bool estimate_factorize_matrix_workspace_bytes(
    uint64_t rows,
    uint64_t cols,
    uint32_t requested_rank,
    uint32_t oversampling,
    uint32_t power_iterations,
    uint64_t * out_bytes,
    std::string * err = nullptr
);

// Separate outputs and scratch estimators for exact simultaneous terms
bool estimate_factor_output_bytes(
    uint64_t rows,
    uint64_t cols,
    uint32_t rank,
    uint64_t * out_bytes,
    std::string * err = nullptr
);

bool estimate_factorize_kv_workspace_bytes(
    const matrix & x_k,
    const matrix & x_v,
    const factor_config & config,
    uint64_t * out_bytes,
    std::string * err = nullptr
);

// Checked estimator for quantized shadow peak workspace:
// Accounts for live immutable FP factors (K and V), 4 encoded streams, and decode/streaming tile scratch.
bool estimate_quantized_shadow_workspace_bytes(
    const matrix & x_k,
    const matrix & x_v,
    const factor_pair & factor_k,
    const factor_pair & factor_v,
    const factor_config & config,
    uint64_t * out_bytes,
    std::string * err = nullptr
);

// Error report for unquantized and quantized reconstructions
struct factor_error_report {
    double frobenius_norm_original = 0.0;
    double frobenius_norm_error    = 0.0;
    double relative_error          = 0.0;
    double max_absolute_error      = 0.0;
};

// Compute error report by streaming over pre-materialized matrices or tiled streaming
factor_error_report compute_error_report(const matrix & original, const matrix & approx);

// Streaming error computation between X (n x m) and factors A (n x r) and B^T (m x r) without full recon matrix
factor_error_report compute_factor_residual_report_tiled(
    const matrix & x,
    const matrix & a,
    const matrix & b_transposed,
    uint32_t tile_rows = 64
);

// Low-rank factor pair: X \approx A * (B^T)^T with B^T authoritative
struct factor_pair {
    matrix a;             // n x r (token-major)
    matrix b_transposed;  // m x r (feature-major transposed: feature rows across rank)
    uint32_t rank = 0;
    std::vector<float> singular_values;
    factor_error_report residual_report;
};

// Complete factorization result for both K and V
struct factor_result {
    bool success = false;
    std::string error_message;
    factor_pair k;
    factor_pair v;
    uint64_t config_fingerprint = 0;
};

// Error report for shadow stream reconstructions
struct shadow_stream_errors {
    factor_error_report a_only;     // (A_hat * B) vs X
    factor_error_report b_only;     // (A * B_hat) vs X
    factor_error_report ab_product; // (A_hat * B_hat) vs X
};

// Four-stream quantized shadow container (contains only descriptors, encoded streams, and errors - no dense recon mirrors!)
struct factor_quantized_shadow {
    bool success = false;
    std::string error_message;

    // The four independent encoded streams
    encoded_matrix stream_a_k;
    encoded_matrix stream_b_k;
    encoded_matrix stream_a_v;
    encoded_matrix stream_b_v;

    // Detailed errors computed via tiled streaming
    shadow_stream_errors errors_k;
    shadow_stream_errors errors_v;
};

// Factorize a single matrix into A (n x r) and authoritative B^T (m x r)
// Uses deterministic randomized SVD for low-rank and small Jacobi for full-rank / small matrices (bounded <= 64).
bool factorize_matrix(
    const matrix & x,
    uint32_t requested_rank,
    factor_balance balance,
    uint64_t seed,
    factor_pair & out_pair,
    std::string * err = nullptr,
    double tolerance = 1e-9,
    uint32_t max_sweeps = 30,
    uint32_t oversampling = 16,
    uint32_t power_iterations = 2,
    uint64_t max_workspace_bytes = UINT64_MAX,
    factor_workspace_span workspace = {},
    factor_fault_injection fault = factor_fault_injection::none
);

bool factorize_matrix_bounded(
    const matrix & x,
    uint32_t requested_rank,
    factor_balance balance,
    uint64_t seed,
    factor_pair & out_pair,
    factor_workspace_span workspace,
    std::string * err = nullptr,
    double tolerance = 1e-9,
    uint32_t max_sweeps = 30,
    uint32_t oversampling = 16,
    uint32_t power_iterations = 2,
    uint64_t max_workspace_bytes = UINT64_MAX,
    factor_fault_injection fault = factor_fault_injection::none
);

// Factorize both K and V concatenated matrices according to config.
// Performs atomic failure: if validation or preflight fails, output is unmodified.
factor_result factorize_kv(
    const matrix & x_k,
    const matrix & x_v,
    const factor_config & config,
    factor_workspace_span workspace = {}
);

factor_result factorize_kv_bounded(
    const matrix & x_k,
    const matrix & x_v,
    const factor_config & config,
    factor_workspace_span workspace
);

// Create a four-stream quantized shadow from immutable factors and original matrices.
// Streams residual comparisons in tiles, retaining no O(nm) reconstruction mirrors.
factor_quantized_shadow evaluate_quantized_shadow(
    const matrix & x_k,
    const matrix & x_v,
    const factor_pair & factor_k,
    const factor_pair & factor_v,
    const factor_config & config,
    factor_workspace_span workspace = {}
);

factor_quantized_shadow evaluate_quantized_shadow_bounded(
    const matrix & x_k,
    const matrix & x_v,
    const factor_pair & factor_k,
    const factor_pair & factor_v,
    const factor_config & config,
    factor_workspace_span workspace
);

} // namespace llama_xkv
