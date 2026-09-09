#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-cparams.h"
#include "llama-kv-cache.h"
#include "llama-triattention.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llama_xkv {

// Accounting and execution stats for canonical batch reads
struct canonical_batch_stats {
    uint32_t sync_count = 0;        // Total backend synchronization calls across layer batches
    uint32_t cells_read = 0;        // Total cells processed
    size_t   bytes_transferred = 0; // Total bytes transferred from backend storage
};

// Batch specification for canonical reads over physical payload cells
struct canonical_read_batch {
    const uint32_t * cell_indices = nullptr; // Physical cell indices
    const int32_t  * positions    = nullptr; // Physical storage positions for RoPE inversion (REQUIRED for decoded_hot)
    const uint64_t * payload_ids  = nullptr; // Optional stable payload IDs
    uint32_t         n_cells      = 0;       // Number of cells in batch
    llama_seq_id     seq_id       = 0;       // Owning sequence ID
};

// Explicit caller-provided backend execution context for D2H readback.
// If backend_k and backend_v are the same active backend handle, K and V are queued async
// and synchronized once (sync_count = 1). If different, each is synchronized (sync_count = 2).
// If null, honest synchronous `ggml_backend_tensor_get` is used and each synchronous get counts 1 sync.
// Host memory buffers require 0 synchronizations.
struct canonical_backend_ctx {
    ggml_backend_t backend_k = nullptr;
    ggml_backend_t backend_v = nullptr;
};

// Separate synchronization planner helper:
// Computes the exact distinct backend synchronizations needed for K and V transfers.
uint32_t plan_canonical_synchronizations(
    ggml_backend_t backend_k, bool k_needs_sync,
    ggml_backend_t backend_v, bool v_needs_sync
);

// Precomputed, validated RoPE omega and frequency scaling tables per owning layer.
// Storing this avoids ad-hoc dereferences of model state on every snapshot.
struct canonical_rope_context {
    uint32_t rotary_dim = 0;
    std::vector<float> omega;
    std::vector<float> freq_scale_sq;
    bool has_factors = false;
    bool valid = false;

    // Build precomputed tables from model parameters or explicit factors
    static canonical_rope_context create_text_imrope(
        uint32_t rotary_dim,
        float freq_base,
        float freq_scale = 1.0f,
        int32_t n_ctx_orig = 4096,
        float ext_factor = 0.0f,
        float attn_factor = 1.0f,
        float beta_fast = 32.0f,
        float beta_slow = 1.0f,
        const float * freq_factors = nullptr
    );
};

// ============================================================================
// Low-level dequantization, inverse attention rotation, and inverse-RoPE helpers
// ============================================================================

// Dequantize one KV head from contiguous row bytes using public ggml type traits or public ggml_dequantize_turbo_row.
// Fixed descriptor: TurboQuant KV cache is always group_size = 128.
bool dequant_k_head_from_rows(
    float          * out,
    const uint8_t  * rows,
    size_t           row_bytes,
    ggml_type        k_type,
    const uint32_t * cell_indices,
    uint32_t         kv_head_idx,
    uint32_t         n_cells,
    uint32_t         head_dim,
    uint32_t         padded_head_dim,
    std::string    * err = nullptr
);

// Dequantize one V head from contiguous row bytes (non-transposed V).
bool dequant_v_head_from_rows(
    float          * out,
    const uint8_t  * rows,
    size_t           row_bytes,
    ggml_type        v_type,
    const uint32_t * cell_indices,
    uint32_t         kv_head_idx,
    uint32_t         n_cells,
    uint32_t         head_dim,
    uint32_t         padded_head_dim,
    std::string    * err = nullptr
);

// Invert orthonormal Hadamard attention rotation in-place on vectors of length head_dim.
// Order invariant: must be applied BEFORE inverse RoPE (K store order is RoPE -> H -> Turbo,
// so canonical decode order is Turbo decode -> H^{-1} -> RoPE^{-1}).
bool invert_attention_rotation(
    float       * data,
    uint32_t      n_vectors,
    uint32_t      dim,
    const float * hadamard_matrix,
    uint32_t      hadamard_dim,
    std::string * err = nullptr
);

// Invert RoPE on post-RoPE key vectors to recover canonical pre-RoPE K in standardized half layout.
// Only supports pairing styles actually implemented: NeoX and text-only IMRoPE.
// Rejects missing positions or negative counts.
bool invert_rope_k(
    float         * out,
    const float   * post_rope_k,
    const int32_t * positions,
    const float   * omega,
    const float   * freq_scale_sq,
    uint32_t        n_cells,
    uint32_t        head_dim,
    uint32_t        rotary_dim,
    std::string   * err = nullptr
);

// ============================================================================
// Bounded Staging Capture API for Committed Seal Ranges (§5.3)
// ============================================================================

struct prerope_staging_config {
    uint32_t max_budget_tokens = 0; // Hard bounded budget in tokens
    uint32_t head_dim_k        = 0;
    uint32_t head_dim_v        = 0;
    uint32_t n_kv_heads        = 0;
    size_t   max_memory_bytes  = 0; // Derived byte budget for staging
};

// Staging buffer holding pre-RoPE K and feature-domain V captured at committed seal boundaries.
// Bounded by budget; checked arithmetic; rejects PRIVATE, PENDING, and tentative tokens.
class xkv_prerope_staging {
public:
    xkv_prerope_staging() = default;
    explicit xkv_prerope_staging(const prerope_staging_config & cfg);

    bool stage_range(
        const llama_kv_cache & kv,
        llama_seq_id           seq_id,
        uint32_t               cell_start,
        uint32_t               cell_count,
        const float          * k_pre_norm, // [cell_count, n_kv_heads * head_dim_k]
        const float          * v_feature,  // [cell_count, n_kv_heads * head_dim_v]
        std::string          * err = nullptr
    );

    void release();

    bool is_valid() const { return valid; }
    uint32_t get_staged_count() const { return staged_tokens; }
    size_t get_allocated_bytes() const { return k_data.size() * sizeof(float) + v_data.size() * sizeof(float); }

    const float * get_k_head(uint32_t kv_head_idx) const;
    const float * get_v_head(uint32_t kv_head_idx) const;

private:
    prerope_staging_config config;
    bool valid = false;
    uint32_t staged_tokens = 0;
    std::vector<float> k_data;
    std::vector<float> v_data;
};

// ============================================================================
// Combined Layer-Level Snapshot (Single-Sync Bulk Readback)
// ============================================================================

// Combined K+V layer snapshot that operates on caller-provided backend handles:
// - Same backend: queues async D2H reads for K and V, synchronizes once (sync_count = 1).
// - Different backends: synchronizes each backend (sync_count = 2).
// - Host memory buffers: requires no synchronization (sync_count = 0).
// - No backend handles: honest synchronous readback counting each synchronous transfer.
class xkv_canonical_layer_kv_snapshot {
public:
    xkv_canonical_layer_kv_snapshot() = default;
    ~xkv_canonical_layer_kv_snapshot() = default;

    bool init(
        const llama_kv_cache          & kv,
        const llama_cparams           & cparams,
        uint32_t                        il,
        const uint32_t                * cell_indices,
        const int32_t                 * positions,
        uint32_t                      n_cells,
        llama_seq_id                  seq_id,
        bool                          need_k,
        bool                            need_v,
        llama_xkv_source                source = LLAMA_XKV_SOURCE_DECODED_HOT,
        const xkv_prerope_staging     * staging = nullptr,
        const canonical_backend_ctx   * bctx = nullptr,
        const canonical_rope_context  * rope_ctx = nullptr,
        canonical_batch_stats         * stats = nullptr,
        std::string                   * err = nullptr
    );

    bool read_k_head(
        uint32_t    kv_head_idx,
        float     * dst,
        size_t      dst_capacity_elements,
        std::string * err = nullptr
    ) const;

    bool read_v_head(
        uint32_t    kv_head_idx,
        float     * dst,
        size_t      dst_capacity_elements,
        std::string * err = nullptr
    ) const;

    uint32_t get_head_dim_k() const { return head_dim_k; }
    uint32_t get_head_dim_v() const { return head_dim_v; }
    uint32_t get_n_cells() const { return n_cells; }
    uint32_t get_n_kv_heads() const { return n_kv_heads; }
    llama_xkv_source get_source() const { return source; }

private:
    uint32_t il = 0;
    uint32_t n_cells = 0;
    uint32_t n_kv_heads = 0;
    uint32_t head_dim_k = 0;
    uint32_t padded_head_dim_k = 0;
    uint32_t head_dim_v = 0;
    uint32_t padded_head_dim_v = 0;
    uint32_t rotary_dim = 0;
    uint32_t kv_size = 0;

    ggml_type k_type = GGML_TYPE_F32;
    ggml_type v_type = GGML_TYPE_F32;
    bool v_trans = false;

    // Persisted selected attention rotation parameters from KV cache
    bool has_attn_rot_k = false;
    uint32_t attn_rot_k_nrot = 0;
    std::vector<float> attn_rot_k_matrix;

    bool has_attn_rot_v = false;
    uint32_t attn_rot_v_nrot = 0;
    std::vector<float> attn_rot_v_matrix;

    std::vector<float> omega;
    std::vector<float> freq_scale_sq;

    std::vector<uint32_t> cell_indices_buf_k;
    std::vector<uint32_t> cell_indices_buf_v;
    std::vector<int32_t>  positions_buf;

    std::vector<uint8_t> rows_snapshot_k;
    std::vector<uint8_t> rows_snapshot_v;
    size_t row_bytes_k = 0;
    size_t row_bytes_v = 0;
    const uint8_t * rows_ptr_k = nullptr;
    const uint8_t * rows_ptr_v = nullptr;

    llama_xkv_source source = LLAMA_XKV_SOURCE_DECODED_HOT;
    const xkv_prerope_staging * staging_ptr = nullptr;
};

// Typed wrappers for individual K or V snapshot reading
class xkv_canonical_layer_k_snapshot {
public:
    xkv_canonical_layer_k_snapshot() = default;

    bool init(
        const llama_kv_cache          & kv,
        const llama_cparams           & cparams,
        uint32_t                        il,
        const uint32_t                * cell_indices,
        const int32_t                 * positions,
        uint32_t                      n_cells,
        llama_seq_id                  seq_id,
        llama_xkv_source              source = LLAMA_XKV_SOURCE_DECODED_HOT,
        const xkv_prerope_staging     * staging = nullptr,
        const canonical_backend_ctx   * bctx = nullptr,
        const canonical_rope_context  * rope_ctx = nullptr,
        canonical_batch_stats         * stats = nullptr,
        std::string                   * err = nullptr
    ) {
        return impl.init(kv, cparams, il, cell_indices, positions, n_cells, seq_id, true, false, source, staging, bctx, rope_ctx, stats, err);
    }

    bool read_kv_head(uint32_t kv_head_idx, float * dst, size_t dst_capacity_elements, std::string * err = nullptr) const {
        return impl.read_k_head(kv_head_idx, dst, dst_capacity_elements, err);
    }

    uint32_t get_head_dim() const { return impl.get_head_dim_k(); }
    uint32_t get_n_cells() const { return impl.get_n_cells(); }
    uint32_t get_n_kv_heads() const { return impl.get_n_kv_heads(); }
    llama_xkv_source get_source() const { return impl.get_source(); }

private:
    xkv_canonical_layer_kv_snapshot impl;
};

class xkv_canonical_layer_v_snapshot {
public:
    xkv_canonical_layer_v_snapshot() = default;

    bool init(
        const llama_kv_cache          & kv,
        const llama_cparams           & cparams,
        uint32_t                        il,
        const uint32_t                * cell_indices,
        uint32_t                      n_cells,
        llama_seq_id                  seq_id,
        llama_xkv_source              source = LLAMA_XKV_SOURCE_DECODED_HOT,
        const xkv_prerope_staging     * staging = nullptr,
        const canonical_backend_ctx   * bctx = nullptr,
        canonical_batch_stats         * stats = nullptr,
        std::string                   * err = nullptr
    ) {
        return impl.init(kv, cparams, il, cell_indices, nullptr, n_cells, seq_id, false, true, source, staging, bctx, nullptr, stats, err);
    }

    bool read_kv_head(uint32_t kv_head_idx, float * dst, size_t dst_capacity_elements, std::string * err = nullptr) const {
        return impl.read_v_head(kv_head_idx, dst, dst_capacity_elements, err);
    }

    uint32_t get_head_dim() const { return impl.get_head_dim_v(); }
    uint32_t get_n_cells() const { return impl.get_n_cells(); }
    uint32_t get_n_kv_heads() const { return impl.get_n_kv_heads(); }
    llama_xkv_source get_source() const { return impl.get_source(); }

private:
    xkv_canonical_layer_kv_snapshot impl;
};

// ============================================================================
// Batched Canonical Read Boundary API (§5.3 / §8.1)
// ============================================================================

bool read_k_canonical(
    const llama_kv_cache          & kv,
    const llama_cparams           & cparams,
    uint32_t                        il,
    const canonical_read_batch    & batch,
    uint32_t                        kv_head_idx,
    float                         * dst,
    size_t                          dst_capacity_elements,
    llama_xkv_source                source = LLAMA_XKV_SOURCE_DECODED_HOT,
    const xkv_prerope_staging     * staging = nullptr,
    const canonical_backend_ctx   * bctx = nullptr,
    const canonical_rope_context  * rope_ctx = nullptr,
    canonical_batch_stats         * stats = nullptr,
    std::string                   * err = nullptr
);

bool read_v_canonical(
    const llama_kv_cache          & kv,
    const llama_cparams           & cparams,
    uint32_t                        il,
    const canonical_read_batch    & batch,
    uint32_t                        kv_head_idx,
    float                         * dst,
    size_t                          dst_capacity_elements,
    llama_xkv_source                source = LLAMA_XKV_SOURCE_DECODED_HOT,
    const xkv_prerope_staging     * staging = nullptr,
    const canonical_backend_ctx   * bctx = nullptr,
    canonical_batch_stats         * stats = nullptr,
    std::string                   * err = nullptr
);

} // namespace llama_xkv
