#include "llama-xkv-canonical.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace llama_xkv {

// ============================================================================
// canonical_rope_context implementation
// ============================================================================

canonical_rope_context canonical_rope_context::create_text_imrope(
    uint32_t rotary_dim,
    float freq_base,
    float freq_scale,
    int32_t n_ctx_orig,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow,
    const float * freq_factors
) {
    canonical_rope_context ctx;
    ctx.rotary_dim = rotary_dim;
    if (rotary_dim == 0 || (rotary_dim & 1u) != 0) {
        return ctx;
    }
    const uint32_t freq_count = rotary_dim / 2;
    ctx.omega.resize(freq_count);
    ctx.freq_scale_sq.resize(freq_count);

    if (triattention_build_rope_tables(
            ctx.omega.data(), ctx.freq_scale_sq.data(), rotary_dim,
            freq_base, freq_scale, n_ctx_orig,
            ext_factor, attn_factor, beta_fast, beta_slow, freq_factors)) {
        ctx.has_factors = (freq_factors != nullptr);
        ctx.valid = true;
    }
    return ctx;
}

uint32_t plan_canonical_synchronizations(
    ggml_backend_t backend_k, bool k_needs_sync,
    ggml_backend_t backend_v, bool v_needs_sync
) {
    uint32_t count = 0;
    if (backend_k && k_needs_sync) {
        count += 1;
    }
    if (backend_v && v_needs_sync) {
        if (!backend_k || !k_needs_sync || backend_v != backend_k) {
            count += 1;
        }
    }
    return count;
}

// ============================================================================
// Low-level dequantization and transformation helpers
// ============================================================================

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
    std::string    * err
) {
    if (!out || !rows) {
        if (err) *err = "null out or rows pointer in dequant_k_head_from_rows";
        return false;
    }
    if (n_cells == 0) {
        return true;
    }
    if (head_dim == 0 || padded_head_dim < head_dim) {
        if (err) *err = "invalid head_dim or padded_head_dim";
        return false;
    }

    const size_t head_offset_bytes = ggml_row_size(k_type, (uint64_t) kv_head_idx * padded_head_dim);
    std::vector<float> dequant_tmp(padded_head_dim, 0.0f);

    const bool is_turbo = (k_type == GGML_TYPE_TURBO2_0 ||
                           k_type == GGML_TYPE_TURBO3_0 ||
                           k_type == GGML_TYPE_TURBO4_0);

    const auto * traits = ggml_get_type_traits(k_type);

    for (uint32_t ci = 0; ci < n_cells; ++ci) {
        const uint32_t cell = cell_indices ? cell_indices[ci] : ci;
        const uint8_t * src = rows + (size_t) cell * row_bytes + head_offset_bytes;
        float * dst = out + (size_t) ci * head_dim;

        float * deq_target = (padded_head_dim != head_dim) ? dequant_tmp.data() : dst;

        if (is_turbo) {
            const int group_size = 128;
            if (padded_head_dim % group_size != 0) {
                if (err) *err = "padded_head_dim " + std::to_string(padded_head_dim) + " must be multiple of group_size 128 for TurboQuant";
                return false;
            }
            if (!ggml_dequantize_turbo_row(k_type, src, deq_target, (int64_t) padded_head_dim, group_size, GGML_TURBO_DECODE_CANONICAL)) {
                if (err) *err = "ggml_dequantize_turbo_row failed for type " + std::string(ggml_type_name(k_type));
                return false;
            }
        } else if (k_type == GGML_TYPE_F32) {
            std::memcpy(deq_target, src, padded_head_dim * sizeof(float));
        } else if (traits && traits->to_float) {
            traits->to_float(src, deq_target, (int64_t) padded_head_dim);
        } else {
            if (err) *err = "unsupported or missing to_float for type " + std::string(ggml_type_name(k_type));
            return false;
        }

        if (deq_target != dst) {
            std::memcpy(dst, deq_target, head_dim * sizeof(float));
        }
    }

    return true;
}

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
    std::string    * err
) {
    return dequant_k_head_from_rows(
        out, rows, row_bytes, v_type, cell_indices, kv_head_idx,
        n_cells, head_dim, padded_head_dim, err
    );
}

bool invert_attention_rotation(
    float       * data,
    uint32_t      n_vectors,
    uint32_t      dim,
    const float * hadamard_matrix,
    uint32_t      hadamard_dim,
    std::string * err
) {
    if (!data) {
        if (err) *err = "null data pointer in invert_attention_rotation";
        return false;
    }
    if (!hadamard_matrix || hadamard_dim == 0) {
        if (err) *err = "null hadamard_matrix or zero hadamard_dim";
        return false;
    }
    if (dim < hadamard_dim || dim % hadamard_dim != 0) {
        if (err) *err = "vector dim " + std::to_string(dim) + " is not divisible by hadamard_dim " + std::to_string(hadamard_dim);
        return false;
    }

    std::vector<float> tmp(hadamard_dim);

    for (uint32_t v = 0; v < n_vectors; ++v) {
        float * vec = data + (size_t) v * dim;
        for (uint32_t b = 0; b < dim; b += hadamard_dim) {
            float * blk = vec + b;
            for (uint32_t i = 0; i < hadamard_dim; ++i) {
                float sum = 0.0f;
                const float * h_row = hadamard_matrix + (size_t) i * hadamard_dim;
                for (uint32_t j = 0; j < hadamard_dim; ++j) {
                    sum += h_row[j] * blk[j];
                }
                tmp[i] = sum;
            }
            std::memcpy(blk, tmp.data(), hadamard_dim * sizeof(float));
        }
    }

    return true;
}

bool invert_rope_k(
    float         * out,
    const float   * post_rope_k,
    const int32_t * positions,
    const float   * omega,
    const float   * freq_scale_sq,
    uint32_t        n_cells,
    uint32_t        head_dim,
    uint32_t        rotary_dim,
    std::string   * err
) {
    if (!out || !post_rope_k) {
        if (err) *err = "null out or post_rope_k pointer in invert_rope_k";
        return false;
    }
    if (n_cells == 0) {
        return true;
    }
    if (!positions) {
        if (err) *err = "positions pointer is null (required for text-only RoPE inversion)";
        return false;
    }
    if (rotary_dim == 0) {
        std::memcpy(out, post_rope_k, (size_t) n_cells * head_dim * sizeof(float));
        return true;
    }
    if (rotary_dim == 0 || (rotary_dim & 1u) != 0 || rotary_dim > head_dim) {
        if (err) *err = "invalid or odd rotary_dim: " + std::to_string(rotary_dim);
        return false;
    }
    if (!omega) {
        if (err) *err = "omega frequencies pointer is null";
        return false;
    }

    const uint32_t freq_count = rotary_dim / 2;

    for (uint32_t i = 0; i < n_cells; ++i) {
        const float * src = post_rope_k + (size_t) i * head_dim;
        float       * dst = out         + (size_t) i * head_dim;
        const float   pos = (float) positions[i];

        for (uint32_t f = 0; f < freq_count; ++f) {
            float angle = omega[f] * pos;
            float c = cosf(angle);
            float s = sinf(angle);
            const float scale = (freq_scale_sq && freq_scale_sq[f] > 0.0f)
                ? sqrtf(freq_scale_sq[f])
                : 1.0f;

            float re = src[f] / scale;
            float im = src[f + freq_count] / scale;

            dst[f]              = re * c + im * s;
            dst[f + freq_count] = im * c - re * s;
        }

        for (uint32_t d = rotary_dim; d < head_dim; ++d) {
            dst[d] = src[d];
        }
    }

    return true;
}

// ============================================================================
// xkv_prerope_staging implementation
// ============================================================================

xkv_prerope_staging::xkv_prerope_staging(const prerope_staging_config & cfg)
    : config(cfg) {
    if (config.max_budget_tokens > 0 && config.head_dim_k > 0 && config.head_dim_v > 0 && config.n_kv_heads > 0) {
        uint64_t total_head_dim = (uint64_t) config.head_dim_k + (uint64_t) config.head_dim_v;
        uint64_t total_elements_per_token = total_head_dim * (uint64_t) config.n_kv_heads;
        if (total_elements_per_token > 0 && (uint64_t) config.max_budget_tokens <= UINT64_MAX / (total_elements_per_token * sizeof(float))) {
            config.max_memory_bytes = (size_t) ((uint64_t) config.max_budget_tokens * total_elements_per_token * sizeof(float));
            valid = true;
        }
    }
}

bool xkv_prerope_staging::stage_range(
    const llama_kv_cache & kv,
    llama_seq_id           seq_id,
    uint32_t               cell_start,
    uint32_t               cell_count,
    const float          * k_pre_norm,
    const float          * v_feature,
    std::string          * err
) {
    if (!valid) {
        if (err) *err = "xkv_prerope_staging is not configured or invalid";
        return false;
    }
    if (cell_count == 0) {
        if (err) *err = "cell_count is zero";
        return false;
    }
    if (cell_start > UINT32_MAX - cell_count) {
        if (err) *err = "cell range arithmetic overflow";
        return false;
    }
    if (cell_count > config.max_budget_tokens) {
        if (err) *err = "requested staging count " + std::to_string(cell_count) + " exceeds budget " + std::to_string(config.max_budget_tokens);
        return false;
    }
    if (!k_pre_norm || !v_feature) {
        if (err) *err = "k_pre_norm or v_feature input is null";
        return false;
    }

    if (!kv.can_capture_prerope_range(seq_id, cell_start, cell_count, err)) {
        return false;
    }

    const uint64_t k_elements_64 = (uint64_t) cell_count * config.n_kv_heads * config.head_dim_k;
    const uint64_t v_elements_64 = (uint64_t) cell_count * config.n_kv_heads * config.head_dim_v;

    if (k_elements_64 > (uint64_t) std::numeric_limits<size_t>::max() ||
        v_elements_64 > (uint64_t) std::numeric_limits<size_t>::max()) {
        if (err) *err = "staging element count exceeds size_t maximum";
        return false;
    }

    const size_t k_elements = (size_t) k_elements_64;
    const size_t v_elements = (size_t) v_elements_64;

    try {
        k_data.resize(k_elements);
        v_data.resize(v_elements);
    } catch (const std::bad_alloc &) {
        k_data.clear();
        v_data.clear();
        staged_tokens = 0;
        if (err) *err = "out of memory allocating staging buffers";
        return false;
    }

    std::memcpy(k_data.data(), k_pre_norm, k_elements * sizeof(float));
    std::memcpy(v_data.data(), v_feature,  v_elements * sizeof(float));

    staged_tokens = cell_count;
    return true;
}

void xkv_prerope_staging::release() {
    k_data.clear();
    k_data.shrink_to_fit();
    v_data.clear();
    v_data.shrink_to_fit();
    staged_tokens = 0;
}

const float * xkv_prerope_staging::get_k_head(uint32_t kv_head_idx) const {
    if (staged_tokens == 0 || kv_head_idx >= config.n_kv_heads) return nullptr;
    return k_data.data() + (size_t) kv_head_idx * config.head_dim_k;
}

const float * xkv_prerope_staging::get_v_head(uint32_t kv_head_idx) const {
    if (staged_tokens == 0 || kv_head_idx >= config.n_kv_heads) return nullptr;
    return v_data.data() + (size_t) kv_head_idx * config.head_dim_v;
}

// ============================================================================
// xkv_canonical_layer_kv_snapshot implementation (Single-Sync Unified Engine)
// ============================================================================

bool xkv_canonical_layer_kv_snapshot::init(
    const llama_kv_cache          & kv,
    const llama_cparams           & cparams,
    uint32_t                        il_param,
    const uint32_t                * cell_indices,
    const int32_t                 * positions,
    uint32_t                        n_cells_param,
    llama_seq_id                    seq_id,
    bool                            need_k,
    bool                            need_v,
    llama_xkv_source                source_param,
    const xkv_prerope_staging     * staging,
    const canonical_backend_ctx   * bctx,
    const canonical_rope_context  * rope_ctx,
    canonical_batch_stats         * stats,
    std::string                   * err
) {
    il = il_param;
    n_cells = n_cells_param;
    source = source_param;
    staging_ptr = staging;

    if (n_cells == 0) {
        return true;
    }

    // Validate seq_id and stream bounds
    if (!kv.validate_seq_id(seq_id)) {
        if (err) *err = "invalid seq_id " + std::to_string(seq_id);
        return false;
    }
    const uint32_t stream = kv.get_stream_for_seq(seq_id);

    const auto & hparams = kv.get_hparams();
    const auto & model = kv.get_model();

    n_kv_heads = hparams.n_head_kv(il);
    head_dim_k = hparams.n_embd_head_k(il);
    head_dim_v = hparams.n_embd_head_v(il);
    v_trans = kv.get_v_trans();
    kv_size = kv.get_hot_size();

    if (n_kv_heads == 0 || head_dim_k == 0 || head_dim_v == 0) {
        if (err) *err = "invalid dimensions for layer " + std::to_string(il);
        return false;
    }

    if (source == LLAMA_XKV_SOURCE_PREROPE_CAPTURE) {
        if (!staging_ptr || !staging_ptr->is_valid() || staging_ptr->get_staged_count() < n_cells) {
            if (err) *err = "prerope_capture requested but staging buffer is invalid or insufficient";
            return false;
        }
        return true;
    }

    if (need_k) {
        rotary_dim = (hparams.rope_type == LLAMA_ROPE_TYPE_NONE) ? 0 : hparams.n_rot(il);
        if (hparams.rope_type == LLAMA_ROPE_TYPE_NONE || rotary_dim == 0) {
            rotary_dim = 0;
        } else {
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NEOX && hparams.rope_type != LLAMA_ROPE_TYPE_IMROPE) {
            if (err) *err = "unsupported RoPE type for canonical reading (only NeoX and text-only IMRoPE supported)";
            return false;
        }

        if (rotary_dim == 0 || (rotary_dim & 1u) != 0 || rotary_dim > head_dim_k) {
            if (err) *err = "invalid or odd rotary_dim " + std::to_string(rotary_dim) + " for layer " + std::to_string(il);
            return false;
        }

        const uint32_t freq_count = rotary_dim / 2;

        // Use precomputed canonical_rope_context if provided, or safely derive from model/cparams
        if (rope_ctx) {
            if (!rope_ctx->valid) {
                if (err) *err = "provided canonical_rope_context is invalid";
                return false;
            }
            if (rope_ctx->rotary_dim != rotary_dim || rope_ctx->omega.size() != freq_count) {
                if (err) *err = "provided canonical_rope_context dimension mismatch";
                return false;
            }
            omega = rope_ctx->omega;
            freq_scale_sq = rope_ctx->freq_scale_sq;
        } else {
            omega.resize(freq_count);
            freq_scale_sq.resize(freq_count);

            std::vector<float> freq_factors;
            const float * factor_ptr = nullptr;

            // Safely check if layers[il] has rope factors before calling get_rope_factors
            if ((size_t) il < model.layers.size()) {
                if (ggml_tensor * factor_tensor = model.get_rope_factors(cparams, (int) il)) {
                    if (factor_tensor->type != GGML_TYPE_F32 || factor_tensor->ne[0] < (int64_t) freq_count) {
                        if (err) *err = "RoPE factor tensor must be F32 with sufficient frequency entries";
                        return false;
                    }
                    freq_factors.resize(freq_count);
                    ggml_backend_tensor_get(factor_tensor, freq_factors.data(), 0, freq_count * sizeof(float));
                    factor_ptr = freq_factors.data();
                }
            }

            const float freq_base = (cparams.rope_freq_base > 0.0f) ? model.get_rope_freq_base(cparams, (int) il) : model.hparams.rope_freq_base_train;
            const float freq_scale = (cparams.rope_freq_scale > 0.0f) ? model.get_rope_freq_scale(cparams, (int) il) : 1.0f;
            const float attn_factor = (cparams.yarn_attn_factor > 0.0f) ? cparams.yarn_attn_factor : 1.0f;
            if (!triattention_build_rope_tables(
                    omega.data(), freq_scale_sq.data(), rotary_dim,
                    freq_base, freq_scale,
                    (int32_t) cparams.n_ctx_orig_yarn,
                    cparams.yarn_ext_factor,
                    attn_factor,
                    cparams.yarn_beta_fast,
                    cparams.yarn_beta_slow,
                    factor_ptr)) {
                if (err) *err = "triattention_build_rope_tables failed for layer " + std::to_string(il);
                return false;
            }
        }
        }
    }

    // Persist selected attention rotation parameters
    has_attn_rot_k = kv.get_attn_rot_k();
    if (has_attn_rot_k) {
        int32_t nrot_k = kv.get_attn_rot_k_nrot();
        if (nrot_k <= 0) {
            if (err) *err = "invalid selected attention rotation nrot for K";
            return false;
        }
        attn_rot_k_nrot = (uint32_t) nrot_k;
        const auto & hadamards = kv.get_attn_rot_hadamard();
        auto it = hadamards.find((int64_t) attn_rot_k_nrot);
        if (it == hadamards.end() || it->second.empty()) {
            if (err) *err = "missing pre-computed attention Hadamard matrix for K (nrot=" + std::to_string(attn_rot_k_nrot) + ")";
            return false;
        }
        attn_rot_k_matrix = it->second;
    }

    has_attn_rot_v = kv.get_attn_rot_v();
    if (has_attn_rot_v) {
        int32_t nrot_v = kv.get_attn_rot_v_nrot();
        if (nrot_v <= 0) {
            if (err) *err = "invalid selected attention rotation nrot for V";
            return false;
        }
        attn_rot_v_nrot = (uint32_t) nrot_v;
        const auto & hadamards = kv.get_attn_rot_hadamard();
        auto it = hadamards.find((int64_t) attn_rot_v_nrot);
        if (it == hadamards.end() || it->second.empty()) {
            if (err) *err = "missing pre-computed attention Hadamard matrix for V (nrot=" + std::to_string(attn_rot_v_nrot) + ")";
            return false;
        }
        attn_rot_v_matrix = it->second;
    }

    cell_indices_buf_k.resize(n_cells);
    cell_indices_buf_v.resize(n_cells);
    if (cell_indices) {
        std::copy(cell_indices, cell_indices + n_cells, cell_indices_buf_k.begin());
        std::copy(cell_indices, cell_indices + n_cells, cell_indices_buf_v.begin());
    } else {
        std::iota(cell_indices_buf_k.begin(), cell_indices_buf_k.end(), 0);
        std::iota(cell_indices_buf_v.begin(), cell_indices_buf_v.end(), 0);
    }

    if (need_k) {
        if (!positions) {
            if (err) *err = "positions are required for decoded_hot canonical K read";
            return false;
        }
        positions_buf.resize(n_cells);
        std::copy(positions, positions + n_cells, positions_buf.begin());
    }

    ggml_tensor * k_tensor = need_k ? kv.get_k_storage((int32_t) il) : nullptr;
    ggml_tensor * v_tensor = need_v ? kv.get_v_storage((int32_t) il) : nullptr;

    if (need_k && !k_tensor) {
        if (err) *err = "missing K tensor for layer " + std::to_string(il);
        return false;
    }
    if (need_v && !v_tensor) {
        if (err) *err = "missing V tensor for layer " + std::to_string(il);
        return false;
    }

    ggml_backend_t backend_k = bctx ? bctx->backend_k : nullptr;
    ggml_backend_t backend_v = bctx ? bctx->backend_v : nullptr;

    // Explicit unallocated-buffer rejection before backend get:
    // Dummy buffers created under no_alloc have get_size == 0 or get_base == nullptr (for host buft).
    if (need_k && (!k_tensor->buffer || ggml_backend_buffer_get_size(k_tensor->buffer) == 0 || (ggml_backend_buffer_is_host(k_tensor->buffer) && ggml_backend_buffer_get_base(k_tensor->buffer) == nullptr))) {
        if (err) *err = "K tensor buffer is not allocated";
        return false;
    }
    if (need_v && (!v_tensor->buffer || ggml_backend_buffer_get_size(v_tensor->buffer) == 0 || (ggml_backend_buffer_is_host(v_tensor->buffer) && ggml_backend_buffer_get_base(v_tensor->buffer) == nullptr))) {
        if (err) *err = "V tensor buffer is not allocated";
        return false;
    }

    bool k_is_host = k_tensor && k_tensor->buffer && ggml_backend_buffer_is_host(k_tensor->buffer);
    bool v_is_host = v_tensor && v_tensor->buffer && ggml_backend_buffer_is_host(v_tensor->buffer);

    // 1. Prepare K transfer
    if (need_k) {
        k_type = k_tensor->type;
        const bool k_is_turbo = (k_type == GGML_TYPE_TURBO2_0 || k_type == GGML_TYPE_TURBO3_0 || k_type == GGML_TYPE_TURBO4_0);
        padded_head_dim_k = (k_is_turbo && head_dim_k % 128 != 0)
            ? ((head_dim_k + 127) / 128) * 128
            : head_dim_k;

        row_bytes_k = ggml_row_size(k_type, k_tensor->ne[0]);
        uint32_t min_cell_k = *std::min_element(cell_indices_buf_k.begin(), cell_indices_buf_k.end());
        uint32_t max_cell_k = *std::max_element(cell_indices_buf_k.begin(), cell_indices_buf_k.end());
        if (max_cell_k >= kv_size) {
            if (err) *err = "K cell index " + std::to_string(max_cell_k) + " exceeds kv_size " + std::to_string(kv_size);
            return false;
        }
        size_t span_cells_k = (size_t) max_cell_k - min_cell_k + 1;
        size_t span_bytes_k = span_cells_k * row_bytes_k;
        // Stream offset for K: stream * kv_size * row_bytes_k + min_cell_k * row_bytes_k
        size_t stream_offset_k = (size_t) stream * kv_size * row_bytes_k;
        size_t offset_bytes_k = stream_offset_k + (size_t) min_cell_k * row_bytes_k;

        if (offset_bytes_k + span_bytes_k > ggml_nbytes(k_tensor)) {
            if (err) *err = "K span exceeds tensor bounds for layer " + std::to_string(il);
            return false;
        }

        if (k_is_host) {
            rows_ptr_k = ((const uint8_t *) k_tensor->data) + offset_bytes_k;
        } else {
            rows_snapshot_k.resize(span_bytes_k);
            if (backend_k) {
                ggml_backend_tensor_get_async(backend_k, k_tensor, rows_snapshot_k.data(), offset_bytes_k, span_bytes_k);
            } else {
                ggml_backend_tensor_get(k_tensor, rows_snapshot_k.data(), offset_bytes_k, span_bytes_k);
                if (stats) {
                    stats->sync_count += 1;
                }
            }
            rows_ptr_k = rows_snapshot_k.data();
            if (stats) {
                stats->bytes_transferred += span_bytes_k;
            }
        }

        for (uint32_t & c : cell_indices_buf_k) {
            c -= min_cell_k;
        }
    }

    // 2. Prepare V transfer
    if (need_v) {
        v_type = v_tensor->type;
        const bool v_is_turbo = (v_type == GGML_TYPE_TURBO2_0 || v_type == GGML_TYPE_TURBO3_0 || v_type == GGML_TYPE_TURBO4_0);
        padded_head_dim_v = (v_is_turbo && head_dim_v % 128 != 0)
            ? ((head_dim_v + 127) / 128) * 128
            : head_dim_v;

        if (!v_trans) {
            row_bytes_v = ggml_row_size(v_type, v_tensor->ne[0]);
            uint32_t min_cell_v = *std::min_element(cell_indices_buf_v.begin(), cell_indices_buf_v.end());
            uint32_t max_cell_v = *std::max_element(cell_indices_buf_v.begin(), cell_indices_buf_v.end());
            if (max_cell_v >= kv_size) {
            if (err) *err = "V cell index " + std::to_string(max_cell_v) + " exceeds kv_size " + std::to_string(kv_size);
            return false;
            }
            size_t span_cells_v = (size_t) max_cell_v - min_cell_v + 1;
            size_t span_bytes_v = span_cells_v * row_bytes_v;
            // Stream offset for non-transposed V: stream * kv_size * row_bytes_v + min_cell_v * row_bytes_v
            size_t stream_offset_v = (size_t) stream * kv_size * row_bytes_v;
            size_t offset_bytes_v = stream_offset_v + (size_t) min_cell_v * row_bytes_v;

            if (offset_bytes_v + span_bytes_v > ggml_nbytes(v_tensor)) {
                if (err) *err = "V span exceeds tensor bounds for layer " + std::to_string(il);
                return false;
            }

            if (v_is_host) {
                rows_ptr_v = ((const uint8_t *) v_tensor->data) + offset_bytes_v;
            } else {
                rows_snapshot_v.resize(span_bytes_v);
                if (backend_v) {
                    ggml_backend_tensor_get_async(backend_v, v_tensor, rows_snapshot_v.data(), offset_bytes_v, span_bytes_v);
                } else {
                    ggml_backend_tensor_get(v_tensor, rows_snapshot_v.data(), offset_bytes_v, span_bytes_v);
                    if (stats) {
                        stats->sync_count += 1;
                    }
                }
                rows_ptr_v = rows_snapshot_v.data();
                if (stats) {
                    stats->bytes_transferred += span_bytes_v;
                }
            }

            for (uint32_t & c : cell_indices_buf_v) {
                c -= min_cell_v;
            }
        } else {
            if (ggml_is_quantized(v_type)) {
                if (err) *err = "quantized transposed V is not supported by backend";
                return false;
            }
            size_t tensor_bytes = ggml_nbytes(v_tensor);
            // Stream offset for transposed V: stream * kv_size * n_embd_v_gqa * v_size_el
            const uint64_t n_embd_v_gqa = v_tensor->ne[0];
            const size_t v_size_el = ggml_type_size(v_type);
            const size_t stream_offset_vt = (size_t) stream * kv_size * n_embd_v_gqa * v_size_el;
            if (v_is_host) {
                rows_ptr_v = ((const uint8_t *) v_tensor->data) + stream_offset_vt;
            } else {
                rows_snapshot_v.resize(tensor_bytes);
                if (backend_v) {
                    ggml_backend_tensor_get_async(backend_v, v_tensor, rows_snapshot_v.data(), stream_offset_vt, (tensor_bytes - stream_offset_vt));
                } else {
                    ggml_backend_tensor_get(v_tensor, rows_snapshot_v.data(), stream_offset_vt, (tensor_bytes - stream_offset_vt));
                    if (stats) {
                        stats->sync_count += 1;
                    }
                }
                rows_ptr_v = rows_snapshot_v.data();
                if (stats) {
                    stats->bytes_transferred += (tensor_bytes - stream_offset_vt);
                }
            }
        }
    }

    // Backend synchronization contract:
    if (backend_k && !k_is_host) {
        ggml_backend_synchronize(backend_k);
        if (stats) {
            stats->sync_count += 1;
        }
    }
    if (backend_v && !v_is_host && backend_v != backend_k) {
        ggml_backend_synchronize(backend_v);
        if (stats) {
            stats->sync_count += 1;
        }
    }

    if (stats) {
        stats->cells_read += n_cells;
    }

    return true;
}

bool xkv_canonical_layer_kv_snapshot::read_k_head(
    uint32_t    kv_head_idx,
    float     * dst,
    size_t      dst_capacity_elements,
    std::string * err
) const {
    if (n_cells == 0) {
        return true;
    }
    if (kv_head_idx >= n_kv_heads) {
        if (err) *err = "kv_head_idx " + std::to_string(kv_head_idx) + " out of bounds";
        return false;
    }
    if (dst_capacity_elements < (size_t) n_cells * head_dim_k) {
        if (err) *err = "dst capacity smaller than required " + std::to_string(n_cells * head_dim_k);
        return false;
    }

    if (source == LLAMA_XKV_SOURCE_PREROPE_CAPTURE) {
        for (uint32_t ci = 0; ci < n_cells; ++ci) {
            const float * src_head = staging_ptr->get_k_head(kv_head_idx) + (size_t) ci * n_kv_heads * head_dim_k;
            std::memcpy(dst + (size_t) ci * head_dim_k, src_head, head_dim_k * sizeof(float));
        }
        return true;
    }

    std::vector<float> dequant_buf((size_t) n_cells * head_dim_k);

    // Step 1: Dequantize + inverse WHT if TurboQuant
    if (!dequant_k_head_from_rows(
            dequant_buf.data(), rows_ptr_k, row_bytes_k, k_type,
            cell_indices_buf_k.data(), kv_head_idx, n_cells,
            head_dim_k, padded_head_dim_k, err)) {
        return false;
    }

    // Step 2: Inverse attention Hadamard rotation BEFORE inverse RoPE
    if (has_attn_rot_k && !attn_rot_k_matrix.empty()) {
        if (!invert_attention_rotation(
                dequant_buf.data(), n_cells, head_dim_k,
                attn_rot_k_matrix.data(), attn_rot_k_nrot, err)) {
            return false;
        }
    }

    // Step 3: Inverse RoPE -> canonical pre-RoPE K in half layout
    if (!invert_rope_k(
            dst, dequant_buf.data(), positions_buf.data(),
            omega.data(), freq_scale_sq.data(), n_cells,
            head_dim_k, rotary_dim, err)) {
        return false;
    }

    return true;
}

bool xkv_canonical_layer_kv_snapshot::read_v_head(
    uint32_t    kv_head_idx,
    float     * dst,
    size_t      dst_capacity_elements,
    std::string * err
) const {
    if (n_cells == 0) {
        return true;
    }
    if (kv_head_idx >= n_kv_heads) {
        if (err) *err = "kv_head_idx " + std::to_string(kv_head_idx) + " out of bounds";
        return false;
    }
    if (dst_capacity_elements < (size_t) n_cells * head_dim_v) {
        if (err) *err = "dst capacity smaller than required " + std::to_string(n_cells * head_dim_v);
        return false;
    }

    if (source == LLAMA_XKV_SOURCE_PREROPE_CAPTURE) {
        for (uint32_t ci = 0; ci < n_cells; ++ci) {
            const float * src_head = staging_ptr->get_v_head(kv_head_idx) + (size_t) ci * n_kv_heads * head_dim_v;
            std::memcpy(dst + (size_t) ci * head_dim_v, src_head, head_dim_v * sizeof(float));
        }
        return true;
    }

    if (!v_trans) {
        if (!dequant_v_head_from_rows(
                dst, rows_ptr_v, row_bytes_v, v_type,
                cell_indices_buf_v.data(), kv_head_idx, n_cells,
                head_dim_v, padded_head_dim_v, err)) {
            return false;
        }
    } else {
        const size_t v_size_el = ggml_type_size(v_type);
        const uint32_t head_start_feat = kv_head_idx * head_dim_v;

        for (uint32_t ci = 0; ci < n_cells; ++ci) {
            const uint32_t cell = cell_indices_buf_v[ci];
            if (cell >= kv_size) {
                if (err) *err = "cell index " + std::to_string(cell) + " exceeds kv_size " + std::to_string(kv_size);
                return false;
            }
            float * dst_cell = dst + (size_t) ci * head_dim_v;

            for (uint32_t j = 0; j < head_dim_v; ++j) {
                const size_t el_offset = ((size_t) (head_start_feat + j) * kv_size + cell) * v_size_el;
                const uint8_t * el_ptr = rows_ptr_v + el_offset;
                switch (v_type) {
                    case GGML_TYPE_F32:
                        dst_cell[j] = *(const float *) el_ptr;
                        break;
                    case GGML_TYPE_F16:
                        dst_cell[j] = ggml_fp16_to_fp32(*(const ggml_fp16_t *) el_ptr);
                        break;
                    case GGML_TYPE_BF16:
                        dst_cell[j] = ggml_bf16_to_fp32(*(const ggml_bf16_t *) el_ptr);
                        break;
                    default:
                        if (err) *err = "unsupported transposed V type: " + std::string(ggml_type_name(v_type));
                        return false;
                }
            }
        }
    }

    // Optional attention rotation inverse for V
    if (has_attn_rot_v && !attn_rot_v_matrix.empty()) {
        if (!invert_attention_rotation(
                dst, n_cells, head_dim_v,
                attn_rot_v_matrix.data(), attn_rot_v_nrot, err)) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// Batched Canonical Read Boundary API implementations
// ============================================================================

bool read_k_canonical(
    const llama_kv_cache          & kv,
    const llama_cparams           & cparams,
    uint32_t                        il,
    const canonical_read_batch    & batch,
    uint32_t                        kv_head_idx,
    float                         * dst,
    size_t                          dst_capacity_elements,
    llama_xkv_source                source,
    const xkv_prerope_staging     * staging,
    const canonical_backend_ctx   * bctx,
    const canonical_rope_context  * rope_ctx,
    canonical_batch_stats         * stats,
    std::string                   * err
) {
    xkv_canonical_layer_kv_snapshot snapshot;
    if (!snapshot.init(
            kv, cparams, il, batch.cell_indices, batch.positions,
            batch.n_cells, batch.seq_id, true, false, source, staging, bctx, rope_ctx, stats, err)) {
        return false;
    }
    return snapshot.read_k_head(kv_head_idx, dst, dst_capacity_elements, err);
}

bool read_v_canonical(
    const llama_kv_cache          & kv,
    const llama_cparams           & cparams,
    uint32_t                        il,
    const canonical_read_batch    & batch,
    uint32_t                        kv_head_idx,
    float                         * dst,
    size_t                          dst_capacity_elements,
    llama_xkv_source                source,
    const xkv_prerope_staging     * staging,
    const canonical_backend_ctx   * bctx,
    canonical_batch_stats         * stats,
    std::string                   * err
) {
    xkv_canonical_layer_kv_snapshot snapshot;
    if (!snapshot.init(
            kv, cparams, il, batch.cell_indices, nullptr,
            batch.n_cells, batch.seq_id, false, true, source, staging, bctx, nullptr, stats, err)) {
        return false;
    }
    return snapshot.read_v_head(kv_head_idx, dst, dst_capacity_elements, err);
}

} // namespace llama_xkv
