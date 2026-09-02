// TriAttention: Trigonometric KV Cache Importance Scorer for llama.cpp
// Based on arXiv 2604.04921 (MIT/NVIDIA/ZJU)
//
// Phase 1: Pure "importance oracle" implementation.
//   1. Binary calibration file loader (.triattention format)
//   2. RoPE inversion (post-RoPE K -> pre-RoPE K)
//   3. Trigonometric key importance scoring (Eqs. 6-10 from paper)
//   4. Streaming combined scoring across all sampled heads (max/union)
//
// All math references cite equation numbers from: "TriAttention: Decoding-Time
// Trigonometric Key Cache Eviction for Long-Context LLM Inference" (2604.04921)

#include "llama-triattention.h"

#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include <cuda_runtime.h>
#include <map>
#endif

// Block types and dequant declarations are in ggml-common.h (ggml/src/)
// which is not on the include path for src/. We declare the dequant
// functions with void* parameters and cast at call sites.
// Block sizes (bytes per 128 elements): turbo2=10, turbo3=14, turbo4=68, q8_0=34

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <vector>

// For timing
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/time.h>
#endif

// Pre-computed WHT inverse rotation matrix R^T (128x128)
// Used to convert turbo2/turbo3 dequant output from WHT-rotated space
// back to the original post-RoPE embedding space.
// turbo4 dequant already applies R^T internally, so this is only needed
// for turbo2_0 and turbo3_0 types.
#include "turbo-rotation-data.h"

// TurboQuant dequant function declarations (from ggml-turbo-quant.c)
// Using void* since block type definitions live in ggml-common.h (not on include path)
extern "C" {
    void dequantize_row_turbo2_0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    void dequantize_row_turbo3_0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    void dequantize_row_turbo4_0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
}

// Standard ggml dequant for Q8_0, F16, etc.
extern "C" {
    void dequantize_row_q8_0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
}

// ============================================================================
// Internal helpers
// ============================================================================

static double triattention_time_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart * 1000.0;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
#endif
}

// Matrix-vector multiply: out[i] = sum_j mat[i*d + j] * vec[j]
// Used for inverse WHT rotation on turbo2/turbo3 dequant output
static void matvec_128(const float * mat, const float * vec, float * out) {
    for (int i = 0; i < 128; i++) {
        float sum = 0.0f;
        const float * row = mat + i * 128;
        for (int j = 0; j < 128; j++) {
            sum += row[j] * vec[j];
        }
        out[i] = sum;
    }
}

// Helper: z-score normalize an array in-place
// After normalization: mean=0, std=1
static void zscore_normalize(float * scores, uint32_t n) {
    if (n <= 1) return;

    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) sum += scores[i];
    double mean = sum / n;

    double var_sum = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        double d = scores[i] - mean;
        var_sum += d * d;
    }
    double std = sqrt(var_sum / n);
    if (std < 1e-10) std = 1e-10;

    for (uint32_t i = 0; i < n; i++) {
        scores[i] = (float)((scores[i] - mean) / std);
    }
}

// ============================================================================
// Binary calibration file I/O
// ============================================================================

// Load .triattention calibration file
// Returns nullptr on any error, with diagnostic printed to stderr
static triattention_calibration * triattention_load_calibration(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[TriAttention] ERROR: cannot open calibration file: %s\n", path);
        return nullptr;
    }

    // Read and validate magic
    uint32_t magic;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || magic != TRIATTENTION_MAGIC) {
        fprintf(stderr, "[TriAttention] ERROR: invalid magic in %s (got 0x%08x, expected 0x%08x)\n",
                path, magic, TRIATTENTION_MAGIC);
        fclose(f);
        return nullptr;
    }

    // Read and validate version
    uint32_t version;
    if (fread(&version, sizeof(uint32_t), 1, f) != 1 || version != TRIATTENTION_VERSION) {
        fprintf(stderr, "[TriAttention] ERROR: unsupported version %u in %s (expected %u)\n",
                version, path, TRIATTENTION_VERSION);
        fclose(f);
        return nullptr;
    }

    auto * cal = new triattention_calibration();
    memset(cal, 0, sizeof(triattention_calibration));

    // Read header fields
    bool ok = true;
    ok = ok && fread(&cal->head_dim,        sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->num_layers,      sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->num_attn_heads,  sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->num_kv_heads,    sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->rope_theta,      sizeof(double),   1, f) == 1;
    ok = ok && fread(&cal->rope_style,      sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->n_sampled,       sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->freq_count,      sizeof(uint32_t), 1, f) == 1;

    if (!ok) {
        fprintf(stderr, "[TriAttention] ERROR: truncated header in %s\n", path);
        delete cal;
        fclose(f);
        return nullptr;
    }

    // Read model name
    uint32_t name_len;
    if (fread(&name_len, sizeof(uint32_t), 1, f) != 1 || name_len == 0 || name_len > 255) {
        fprintf(stderr, "[TriAttention] ERROR: invalid model name length %u in %s\n", name_len, path);
        delete cal;
        fclose(f);
        return nullptr;
    }
    if (fread(cal->model_name, 1, name_len, f) != name_len) {
        fprintf(stderr, "[TriAttention] ERROR: truncated model name in %s\n", path);
        delete cal;
        fclose(f);
        return nullptr;
    }
    cal->model_name[name_len] = '\0';

    // Validate basic field consistency
    if (cal->freq_count != cal->head_dim / 2) {
        fprintf(stderr, "[TriAttention] ERROR: freq_count (%u) != head_dim/2 (%u) in %s\n",
                cal->freq_count, cal->head_dim / 2, path);
        delete cal;
        fclose(f);
        return nullptr;
    }

    if (cal->num_attn_heads == 0 || cal->num_kv_heads == 0 ||
        cal->num_attn_heads % cal->num_kv_heads != 0) {
        fprintf(stderr, "[TriAttention] ERROR: invalid head counts (attn=%u, kv=%u) in %s\n",
                cal->num_attn_heads, cal->num_kv_heads, path);
        delete cal;
        fclose(f);
        return nullptr;
    }

    cal->num_kv_groups = cal->num_attn_heads / cal->num_kv_heads;

    // Allocate per-head arrays
    cal->sampled_layer = new uint32_t[cal->n_sampled];
    cal->sampled_head  = new uint32_t[cal->n_sampled];
    cal->head_stats    = new triattention_head_stats[cal->n_sampled];

    const uint32_t fc = cal->freq_count;

    for (uint32_t h = 0; h < cal->n_sampled; h++) {
        // Read layer and head indices
        ok = true;
        ok = ok && fread(&cal->sampled_layer[h], sizeof(uint32_t), 1, f) == 1;
        ok = ok && fread(&cal->sampled_head[h],  sizeof(uint32_t), 1, f) == 1;

        if (!ok) {
            fprintf(stderr, "[TriAttention] ERROR: truncated head entry %u in %s\n", h, path);
            // Cleanup partially allocated heads
            for (uint32_t j = 0; j < h; j++) {
                delete[] cal->head_stats[j].q_mean_real;
                delete[] cal->head_stats[j].q_mean_imag;
                delete[] cal->head_stats[j].q_abs_mean;
            }
            delete[] cal->sampled_layer;
            delete[] cal->sampled_head;
            delete[] cal->head_stats;
            delete cal;
            fclose(f);
            return nullptr;
        }

        // Validate indices
        if (cal->sampled_layer[h] >= cal->num_layers ||
            cal->sampled_head[h] >= cal->num_attn_heads) {
            fprintf(stderr, "[TriAttention] ERROR: head entry %u has invalid indices (layer=%u, head=%u) in %s\n",
                    h, cal->sampled_layer[h], cal->sampled_head[h], path);
            for (uint32_t j = 0; j < h; j++) {
                delete[] cal->head_stats[j].q_mean_real;
                delete[] cal->head_stats[j].q_mean_imag;
                delete[] cal->head_stats[j].q_abs_mean;
            }
            delete[] cal->sampled_layer;
            delete[] cal->sampled_head;
            delete[] cal->head_stats;
            delete cal;
            fclose(f);
            return nullptr;
        }

        // Allocate and read per-frequency arrays
        auto & hs = cal->head_stats[h];
        hs.q_mean_real  = new float[fc];
        hs.q_mean_imag  = new float[fc];
        hs.q_abs_mean   = new float[fc];
        hs.q_mean_abs   = nullptr;  // computed at init time
        hs.extra_weight = nullptr;  // computed at init time

        ok = true;
        ok = ok && fread(hs.q_mean_real, sizeof(float), fc, f) == fc;
        ok = ok && fread(hs.q_mean_imag, sizeof(float), fc, f) == fc;
        ok = ok && fread(hs.q_abs_mean,  sizeof(float), fc, f) == fc;

        // Read R_f (validation data — not stored at runtime, just skip)
        float * r_f_tmp = new float[fc];
        ok = ok && fread(r_f_tmp, sizeof(float), fc, f) == fc;
        delete[] r_f_tmp;

        if (!ok) {
            fprintf(stderr, "[TriAttention] ERROR: truncated stats for head %u in %s\n", h, path);
            // Free this head's arrays
            delete[] hs.q_mean_real;
            delete[] hs.q_mean_imag;
            delete[] hs.q_abs_mean;
            // Free previous heads
            for (uint32_t j = 0; j < h; j++) {
                delete[] cal->head_stats[j].q_mean_real;
                delete[] cal->head_stats[j].q_mean_imag;
                delete[] cal->head_stats[j].q_abs_mean;
                delete[] cal->head_stats[j].q_mean_abs;
                delete[] cal->head_stats[j].extra_weight;
            }
            delete[] cal->sampled_layer;
            delete[] cal->sampled_head;
            delete[] cal->head_stats;
            delete cal;
            fclose(f);
            return nullptr;
        }
    }

    fclose(f);

    fprintf(stderr, "[TriAttention] Loaded calibration: model=%s, layers=%u, attn_heads=%u, kv_heads=%u, "
            "head_dim=%u, sampled=%u, rope_theta=%.1f\n",
            cal->model_name, cal->num_layers, cal->num_attn_heads,
            cal->num_kv_heads, cal->head_dim, cal->n_sampled, cal->rope_theta);

    return cal;
}

static void triattention_free_calibration(triattention_calibration * cal) {
    if (!cal) return;

    for (uint32_t h = 0; h < cal->n_sampled; h++) {
        delete[] cal->head_stats[h].q_mean_real;
        delete[] cal->head_stats[h].q_mean_imag;
        delete[] cal->head_stats[h].q_abs_mean;
        delete[] cal->head_stats[h].q_mean_abs;
        delete[] cal->head_stats[h].extra_weight;
    }
    delete[] cal->sampled_layer;
    delete[] cal->sampled_head;
    delete[] cal->head_stats;
    delete cal;
}

// ============================================================================
// Precomputation at init time
// ============================================================================

// Build RoPE frequency array: omega[f] = rope_theta^(-2f/head_dim)
// Paper Eq. 1: theta_f = base^{-2f/d}
static void triattention_build_omega(float * omega, uint32_t freq_count, uint32_t head_dim, double rope_theta) {
    for (uint32_t f = 0; f < freq_count; f++) {
        double exponent = -2.0 * (double)f / (double)head_dim;
        omega[f] = (float)pow(rope_theta, exponent);
    }
}

// Build frequency scaling squared: freq_scale_sq[f] = cos^2(omega[f]*0) + sin^2(omega[f]*0)
// For standard RoPE this is always 1.0, but for scaled RoPE (YaRN etc.)
// the scaling factors at position 0 capture any frequency-dependent scaling.
// Paper Section 3.2: "frequency scaling factor"
static void triattention_build_freq_scale_sq(float * freq_scale_sq, const float * omega, uint32_t freq_count) {
    for (uint32_t f = 0; f < freq_count; f++) {
        // At position 0: cos(omega*0)=1, sin(omega*0)=0
        // So freq_scale_sq = 1.0 for all standard RoPE variants.
        // If we later support YaRN scaling, this would use the actual scaling factors.
        float c = cosf(omega[f] * 0.0f);
        float s = sinf(omega[f] * 0.0f);
        freq_scale_sq[f] = c * c + s * s;
    }
}

// Build geometric offset array: {1, 2, 4, 8, ..., offset_max}
// Paper Eq. 9: D = {2^0, 2^1, ..., 2^{log2(max_length)}}
static uint32_t triattention_build_offsets(float * offsets, uint32_t offset_max) {
    uint32_t n = 0;
    for (uint32_t d = 1; d <= offset_max; d *= 2) {
        offsets[n++] = (float)d;
    }
    return n;
}

// Precompute derived quantities per head from calibration stats:
//   q_mean_abs[f] = sqrt(q_mean_real[f]^2 + q_mean_imag[f]^2)  = ||E[q_f]||
//   extra_weight[f] = q_abs_mean[f] - q_mean_abs[f]              = E[||q_f||] - ||E[q_f]||
// Paper Eq. 8: the "norm excess" term weighted by (1 - R_f)
static void triattention_precompute_head_derived(triattention_head_stats * hs, uint32_t freq_count, bool disable_mlr) {
    hs->q_mean_abs   = new float[freq_count];
    hs->extra_weight = new float[freq_count];

    for (uint32_t f = 0; f < freq_count; f++) {
        float re = hs->q_mean_real[f];
        float im = hs->q_mean_imag[f];
        hs->q_mean_abs[f] = sqrtf(re * re + im * im);

        if (disable_mlr) {
            // Ablation: use q_abs_mean directly as the norm contribution
            hs->extra_weight[f] = hs->q_abs_mean[f];
        } else {
            // Standard: MLR-weighted norm excess = E[||q_f||] - ||E[q_f]||
            // This is >= 0 because ||E[x]|| <= E[||x||] (Jensen's inequality)
            hs->extra_weight[f] = hs->q_abs_mean[f] - hs->q_mean_abs[f];
            if (hs->extra_weight[f] < 0.0f) {
                hs->extra_weight[f] = 0.0f;  // Numerical safety
            }
        }
    }
}

// ============================================================================
// Core scoring functions: CPU implementations
// ============================================================================

// Invert RoPE rotation on post-RoPE key vectors.
// Paper Eq. 4: recover pre-RoPE K from post-RoPE K using known positions.
//
// For "half" style (Llama/Qwen): dimensions split as [real | imag]
//   k_pre[f]    = k_post[f]*cos(omega[f]*pos) + k_post[f+fc]*sin(omega[f]*pos)
//   k_pre[f+fc] = k_post[f+fc]*cos(omega[f]*pos) - k_post[f]*sin(omega[f]*pos)
//
// For "interleaved" style: pairs are (2f, 2f+1)
//   k_pre[2f]   = k_post[2f]*cos(omega[f]*pos) + k_post[2f+1]*sin(omega[f]*pos)
//   k_pre[2f+1] = k_post[2f+1]*cos(omega[f]*pos) - k_post[2f]*sin(omega[f]*pos)
void triattention_invert_rope(
    float       * out,
    const float * post_rope_k,
    const int32_t * positions,
    const float * omega,
    uint32_t n_keys,
    uint32_t head_dim,
    uint32_t freq_count,
    uint32_t rope_style)
{
    for (uint32_t i = 0; i < n_keys; i++) {
        const float * src = post_rope_k + (size_t)i * head_dim;
        float       * dst = out         + (size_t)i * head_dim;
        const float   pos = (float)positions[i];

        if (rope_style == 0) {
            // Half style: [real_0..real_{fc-1} | imag_0..imag_{fc-1}]
            for (uint32_t f = 0; f < freq_count; f++) {
                float angle = omega[f] * pos;
                float c = cosf(angle);
                float s = sinf(angle);
                float re = src[f];
                float im = src[f + freq_count];
                // Invert rotation: multiply by conjugate rotation matrix
                dst[f]              = re * c + im * s;
                dst[f + freq_count] = im * c - re * s;
            }
        } else {
            // Interleaved style: [re_0, im_0, re_1, im_1, ...]
            for (uint32_t f = 0; f < freq_count; f++) {
                float angle = omega[f] * pos;
                float c = cosf(angle);
                float s = sinf(angle);
                float re = src[2 * f];
                float im = src[2 * f + 1];
                dst[2 * f]     = re * c + im * s;
                dst[2 * f + 1] = im * c - re * s;
            }
        }
    }
}

// Score cached keys for a single (layer, attention_head) pair.
// Paper Eqs. 6-10: trigonometric importance scoring with MLR norm term.
//
// For each key at position p_k with base distance Delta = round_start - p_k:
//   1. Convert pre-RoPE K to complex representation
//   2. Compute amplitude: amp_f = ||E[q_f]|| * |k_f|
//   3. Compute phase: phi_f = angle(E[q_f] * conj(k_f))
//   4. Compute trig score: S_trig(Delta+delta) = sum_f amp_f * fscale_sq_f * cos(omega_f*(Delta+delta) + phi_f)
//   5. Compute norm score: S_norm = sum_f extra_f * fscale_sq_f * |k_f|
//   6. Aggregate over geometric offsets
void triattention_score_keys(
    float       * out_scores,
    const float * pre_rope_k,
    const triattention_head_stats * stats,
    const float * omega,
    const float * freq_scale_sq,
    const float * offsets,
    const int32_t * key_positions,
    int64_t  round_start,
    uint32_t n_keys,
    uint32_t head_dim,
    uint32_t freq_count,
    uint32_t n_offsets,
    enum triattention_agg agg,
    bool disable_trig)
{
    const float inv_n_offsets = 1.0f / (float)n_offsets;

    for (uint32_t i = 0; i < n_keys; i++) {
        const float * k = pre_rope_k + (size_t)i * head_dim;
        const float   base_delta = (float)(round_start - key_positions[i]);

        // Precompute per-frequency quantities for this key
        // Using "half" layout: k_re = k[f], k_im = k[f + freq_count]
        // (interleaved would be k[2f], k[2f+1] — handled at invert_rope stage,
        //  output from invert_rope is always in half layout for scoring)

        float total_score = 0.0f;

        if (!disable_trig) {
            // Full scoring: trigonometric + norm terms
            for (uint32_t d = 0; d < n_offsets; d++) {
                float delta = base_delta + offsets[d];
                float offset_score = 0.0f;

                for (uint32_t f = 0; f < freq_count; f++) {
                    float k_re = k[f];
                    float k_im = k[f + freq_count];
                    float k_mag = sqrtf(k_re * k_re + k_im * k_im);

                    // Amplitude: ||E[q_f]|| * |k_f|  (Paper Eq. 7)
                    float amp = stats->q_mean_abs[f] * k_mag;

                    // Phase from conj multiply: E[q_f] * conj(k_f)
                    // = (q_re + i*q_im) * (k_re - i*k_im)
                    // = (q_re*k_re + q_im*k_im) + i*(q_im*k_re - q_re*k_im)
                    float conj_re = stats->q_mean_real[f] * k_re + stats->q_mean_imag[f] * k_im;
                    float conj_im = stats->q_mean_imag[f] * k_re - stats->q_mean_real[f] * k_im;
                    float phi = atan2f(conj_im, conj_re);

                    // Trigonometric score (Paper Eq. 6):
                    // S_trig += amp * fscale^2 * cos(omega * delta + phi)
                    float phase = omega[f] * delta + phi;
                    offset_score += amp * freq_scale_sq[f] * cosf(phase);

                    // Norm excess term (Paper Eq. 8):
                    // S_norm += extra_weight * fscale^2 * |k_f|
                    offset_score += stats->extra_weight[f] * freq_scale_sq[f] * k_mag;
                }

                if (agg == TRIATTENTION_AGG_MAX) {
                    total_score = (d == 0) ? offset_score : fmaxf(total_score, offset_score);
                } else {
                    total_score += offset_score;
                }
            }

            if (agg == TRIATTENTION_AGG_MEAN) {
                total_score *= inv_n_offsets;
            }
        } else {
            // Ablation: norm-only scoring (disable_trig=true)
            // Only the position-independent norm term
            for (uint32_t f = 0; f < freq_count; f++) {
                float k_re = k[f];
                float k_im = k[f + freq_count];
                float k_mag = sqrtf(k_re * k_re + k_im * k_im);
                total_score += stats->extra_weight[f] * freq_scale_sq[f] * k_mag;
            }
        }

        out_scores[i] = total_score;
    }
}

// ============================================================================
// KV cache dequantization helper
// ============================================================================

// Dequantize K values for a specific KV head from the cache tensor.
// Handles all supported quantization types and applies inverse WHT
// rotation for turbo2/turbo3 types.
//
// Parameters:
//   out         — [n_cells, padded_head_dim] dequantized float output
//   k_tensor    — the raw K cache tensor for this layer
//   cell_indices— [n_cells] which cell slots to extract
//   kv_head_idx — which KV head (0..n_kv_heads-1)
//   n_cells     — number of cells to dequantize
//   padded_hd   — padded head dimension (128-aligned for turbo types)
//   n_kv_heads  — total number of KV heads
//   need_wht_inv— whether to apply inverse WHT rotation (turbo2/turbo3)
//
// Note: This function copies data from potentially GPU-resident tensors
// to CPU memory, which involves a synchronous transfer. This is acceptable
// because pruning happens infrequently (every divide_length tokens).
static void triattention_dequant_kv_head(
    float              * out,
    const ggml_tensor  * k_tensor,
    const uint32_t     * cell_indices,
    uint32_t             kv_head_idx,
    uint32_t             n_cells,
    uint32_t             padded_hd,
    uint32_t             n_kv_heads,
    bool                 need_wht_inv)
{
    (void)n_kv_heads;  // kept for API symmetry, offset computed via kv_head_idx * padded_hd

    const ggml_type k_type = k_tensor->type;
    const uint64_t  n_embd_k_gqa = k_tensor->ne[0];  // total K embedding (all KV heads)
    const size_t    row_bytes = ggml_row_size(k_type, n_embd_k_gqa);

    // Byte offset to this KV head within a row
    const size_t head_offset_bytes = ggml_row_size(k_type, (uint64_t)kv_head_idx * padded_hd);
    const size_t head_bytes = ggml_row_size(k_type, padded_hd);

    // Temporary buffer for one quantized head block
    std::vector<uint8_t> quant_buf(head_bytes);

    // Temporary buffer for dequantized values (before WHT inverse)
    std::vector<float> dequant_tmp(padded_hd);

    for (uint32_t ci = 0; ci < n_cells; ci++) {
        const uint32_t cell_idx = cell_indices[ci];

        // Byte offset in the full tensor: row_bytes * cell_idx + head_offset_bytes
        // This addresses stream 0 (the common case for unified KV caches)
        const size_t tensor_offset = (size_t)cell_idx * row_bytes + head_offset_bytes;

        // Copy quantized data from backend (may be GPU) to CPU
        ggml_backend_tensor_get(k_tensor, quant_buf.data(), tensor_offset, head_bytes);

        // Dequantize based on type
        float * dst = need_wht_inv ? dequant_tmp.data() : (out + (size_t)ci * padded_hd);

        switch (k_type) {
            case GGML_TYPE_TURBO3_0:
                dequantize_row_turbo3_0(quant_buf.data(), dst, padded_hd);
                break;
            case GGML_TYPE_TURBO4_0:
                dequantize_row_turbo4_0(quant_buf.data(), dst, padded_hd);
                break;
            case GGML_TYPE_TURBO2_0:
                dequantize_row_turbo2_0(quant_buf.data(), dst, padded_hd);
                break;
            case GGML_TYPE_Q8_0:
                dequantize_row_q8_0(quant_buf.data(), dst, padded_hd);
                break;
            case GGML_TYPE_F16: {
                const ggml_fp16_t * src16 = (const ggml_fp16_t *)quant_buf.data();
                for (uint32_t j = 0; j < padded_hd; j++) {
                    dst[j] = ggml_fp16_to_fp32(src16[j]);
                }
                break;
            }
            case GGML_TYPE_BF16: {
                const ggml_bf16_t * src16 = (const ggml_bf16_t *)quant_buf.data();
                for (uint32_t j = 0; j < padded_hd; j++) {
                    dst[j] = ggml_bf16_to_fp32(src16[j]);
                }
                break;
            }
            case GGML_TYPE_F32: {
                memcpy(dst, quant_buf.data(), padded_hd * sizeof(float));
                break;
            }
            default:
                fprintf(stderr, "[TriAttention] ERROR: unsupported K cache type %d\n", k_type);
                memset(out + (size_t)ci * padded_hd, 0, padded_hd * sizeof(float));
                continue;
        }

        // Apply inverse WHT rotation for turbo2/turbo3
        // turbo4 dequant already applies R^T internally
        if (need_wht_inv) {
            float * final_dst = out + (size_t)ci * padded_hd;
            // Process in 128-element blocks (WHT block size)
            for (uint32_t b = 0; b < padded_hd; b += 128) {
                matvec_128(TURBO_ROTATION_RT, dequant_tmp.data() + b, final_dst + b);
            }
        }
    }
}

// ============================================================================
// triattention_scorer: pimpl implementation
// ============================================================================

struct triattention_scorer::impl {
    triattention_calibration * cal = nullptr;
    triattention_scorer_config cfg;

    // Precomputed arrays (allocated once at init)
    float *   omega = nullptr;          // [freq_count]  RoPE frequencies: theta^(-2f/d)
    float *   freq_scale_sq = nullptr;  // [freq_count]  frequency scaling^2
    float *   offsets = nullptr;        // [n_offsets]   geometric {1,2,4,...,offset_max}
    uint32_t  n_offsets = 0;

    bool      is_valid = false;

#ifdef GGML_USE_CUDA
    // GPU state cache: one triattention_gpu_state per (device_id, k_type) pair.
    // Key: device_id * 1000 + (int)k_type
    // This handles layer-adaptive K types and multi-GPU placement.
    std::map<uint64_t, triattention_gpu_state *> gpu_states;

    ~impl() {
        // Free GPU states
        for (auto & [key, state] : gpu_states) {
            if (state) {
                triattention_gpu_free(state);
            }
        }
        gpu_states.clear();
        if (cal) {
            triattention_free_calibration(cal);
            cal = nullptr;
        }
        delete[] omega;
        delete[] freq_scale_sq;
        delete[] offsets;
    }

    // Get or create GPU state for a specific (device, k_type) combination.
    // Returns nullptr if initialization fails.
    triattention_gpu_state * get_gpu_state(int device_id, ggml_type k_type) {
        const uint64_t key = (uint64_t)device_id * 1000 + (uint64_t)k_type;
        auto it = gpu_states.find(key);
        if (it != gpu_states.end()) {
            return it->second;
        }

        // Initialize GPU state for this (device, k_type)
        triattention_gpu_config gcfg = {};
        gcfg.head_dim      = cal->head_dim;
        gcfg.freq_count    = cal->freq_count;
        gcfg.n_kv_heads    = cal->num_kv_heads;
        gcfg.n_sampled     = cal->n_sampled;
        gcfg.n_offsets     = n_offsets;
        gcfg.k_type        = k_type;
        gcfg.need_wht_inv  = (k_type == GGML_TYPE_TURBO2_0 || k_type == GGML_TYPE_TURBO3_0);
        gcfg.disable_trig  = cfg.disable_trig;

        // Build per-head calibration arrays
        std::vector<triattention_gpu_head_calib> head_calibs(cal->n_sampled);
        for (uint32_t h = 0; h < cal->n_sampled; h++) {
            head_calibs[h].q_mean_real  = cal->head_stats[h].q_mean_real;
            head_calibs[h].q_mean_imag  = cal->head_stats[h].q_mean_imag;
            head_calibs[h].q_mean_abs   = cal->head_stats[h].q_mean_abs;
            head_calibs[h].extra_weight = cal->head_stats[h].extra_weight;
        }

        // Set CUDA device before init
        cudaSetDevice(device_id);

        triattention_gpu_state * state = triattention_gpu_init(
            &gcfg, head_calibs.data(), omega, freq_scale_sq, offsets, nullptr);

        if (state) {
            gpu_states[key] = state;
        }
        return state;
    }
#else
    ~impl() {
        if (cal) {
            triattention_free_calibration(cal);
            cal = nullptr;
        }
        delete[] omega;
        delete[] freq_scale_sq;
        delete[] offsets;
    }
#endif

    // Find the sampled head index for a given (model_layer, kv_head).
    // Returns -1 if no sampled head matches.
    int32_t find_sampled_head(uint32_t model_layer, uint32_t kv_head) const {
        for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
            if (cal->sampled_layer[sh] == model_layer) {
                uint32_t attn_head = cal->sampled_head[sh];
                uint32_t kv_h = attn_head / cal->num_kv_groups;
                if (kv_h == kv_head) {
                    return (int32_t)sh;
                }
            }
        }
        return -1;
    }
};

// ============================================================================
// triattention_scorer: public API
// ============================================================================

triattention_scorer::triattention_scorer(
    const char * stats_path,
    const triattention_scorer_config & cfg,
    double rope_theta,
    uint32_t head_dim,
    uint32_t n_kv_heads)
{
    pimpl = std::make_unique<impl>();
    pimpl->cfg = cfg;

    // Load calibration file
    triattention_calibration * cal = triattention_load_calibration(stats_path);
    if (!cal) {
        return;
    }

    // Validate model compatibility
    if (cal->head_dim != head_dim) {
        fprintf(stderr, "[TriAttention] ERROR: head_dim mismatch (calibration=%u, model=%u)\n",
                cal->head_dim, head_dim);
        triattention_free_calibration(cal);
        return;
    }
    if (cal->num_kv_heads != n_kv_heads) {
        fprintf(stderr, "[TriAttention] ERROR: n_kv_heads mismatch (calibration=%u, model=%u)\n",
                cal->num_kv_heads, n_kv_heads);
        triattention_free_calibration(cal);
        return;
    }
    // Warn if rope_theta differs significantly (>1% relative)
    if (fabs(cal->rope_theta - rope_theta) / fmax(cal->rope_theta, 1.0) > 0.01) {
        fprintf(stderr, "[TriAttention] WARNING: rope_theta mismatch (calibration=%.1f, model=%.1f)\n",
                cal->rope_theta, rope_theta);
    }

    pimpl->cal = cal;

    const uint32_t fc = cal->freq_count;

    // Build precomputed arrays
    pimpl->omega = new float[fc];
    triattention_build_omega(pimpl->omega, fc, head_dim, rope_theta);

    pimpl->freq_scale_sq = new float[fc];
    triattention_build_freq_scale_sq(pimpl->freq_scale_sq, pimpl->omega, fc);

    // Geometric offsets — max 17 elements for offset_max=65536
    pimpl->offsets = new float[32];  // generous allocation
    pimpl->n_offsets = triattention_build_offsets(pimpl->offsets, cfg.offset_max);

    // Precompute derived head stats
    for (uint32_t h = 0; h < cal->n_sampled; h++) {
        triattention_precompute_head_derived(&cal->head_stats[h], fc, cfg.disable_mlr);
    }

    pimpl->is_valid = true;

    fprintf(stderr, "[TriAttention] Scorer initialized: agg=%d, offsets=%u, "
            "normalize=%d, sampled_heads=%u\n",
            (int)cfg.agg, pimpl->n_offsets, (int)cfg.normalize_scores, cal->n_sampled);
}

triattention_scorer::~triattention_scorer() = default;

bool triattention_scorer::valid() const {
    return pimpl && pimpl->is_valid;
}

void triattention_scorer::score_head(
    float * out_scores,
    const ggml_tensor * k_tensor,
    const uint32_t * cell_indices,
    const int32_t * positions,
    uint32_t kv_head_idx,
    uint32_t n_candidates,
    int64_t frontier_position) const
{
    if (!valid() || !k_tensor || n_candidates == 0) {
        if (out_scores && n_candidates > 0) {
            memset(out_scores, 0, n_candidates * sizeof(float));
        }
        return;
    }

    const auto * cal = pimpl->cal;
    const uint32_t fc = cal->freq_count;
    const uint32_t hd = cal->head_dim;
    const uint32_t padded_hd = ((hd + 127) / 128) * 128;

    // We need to find the sampled head for this (layer, kv_head) pair.
    // The caller passes a single k_tensor for a specific layer.
    // We search all sampled heads for one whose kv_head matches.
    // Since we don't know the model layer index here (score_head is per-tensor),
    // we search for any sampled head with this kv_head_idx.
    // If multiple sampled heads share the same kv_head (different layers),
    // we use the first match. For precise (layer, head) scoring, use score_combined.
    int32_t sh = -1;
    for (uint32_t s = 0; s < cal->n_sampled; s++) {
        uint32_t attn_head = cal->sampled_head[s];
        uint32_t kv_h = attn_head / cal->num_kv_groups;
        if (kv_h == kv_head_idx) {
            sh = (int32_t)s;
            break;
        }
    }

    if (sh < 0) {
        // No calibration for this kv_head — zero scores
        memset(out_scores, 0, n_candidates * sizeof(float));
        return;
    }

    const ggml_type k_type = k_tensor->type;
    const bool need_wht_inv = (k_type == GGML_TYPE_TURBO2_0 || k_type == GGML_TYPE_TURBO3_0);

#ifdef GGML_USE_CUDA
    // GPU fast path: if K tensor is on a CUDA device, score directly on GPU
    // without copying K data to host. Only the score array is transferred back.
    if (k_tensor->buffer && k_tensor->buffer->buft) {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(k_tensor->buffer->buft);
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            // Get the CUDA device index
            const char * dev_name = ggml_backend_dev_name(dev);
            // Try to find device index by matching against CUDA devices
            int cuda_dev_id = -1;
            int n_cuda_devs = 0;
            cudaGetDeviceCount(&n_cuda_devs);
            for (int d = 0; d < n_cuda_devs; d++) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, d);
                if (strstr(dev_name, prop.name) != nullptr) {
                    cuda_dev_id = d;
                    break;
                }
            }
            // Fallback: use device 0 if we can't match
            if (cuda_dev_id < 0 && n_cuda_devs > 0) {
                cuda_dev_id = 0;
            }

            if (cuda_dev_id >= 0) {
                auto * gpu_state = pimpl->get_gpu_state(cuda_dev_id, k_type);
                if (gpu_state) {
                    // Get device pointer to K data
                    void * k_data_dev = k_tensor->data;
                    const size_t row_bytes = ggml_row_size(k_type, k_tensor->ne[0]);

                    // Upload cell indices and positions to device
                    uint32_t * cell_indices_dev = nullptr;
                    int32_t  * positions_dev = nullptr;
                    triattention_gpu_upload_cells(
                        &cell_indices_dev, &positions_dev,
                        cell_indices, positions, n_candidates, nullptr);

                    // Allocate score buffer on device
                    float * scores_dev = triattention_gpu_alloc_scores(n_candidates, nullptr);

                    // Launch kernel
                    triattention_gpu_score_head(
                        gpu_state, k_data_dev,
                        (uint64_t)k_tensor->ne[0], row_bytes,
                        kv_head_idx, (uint32_t)sh,
                        cell_indices_dev, positions_dev,
                        n_candidates, frontier_position,
                        (int)pimpl->cfg.agg, scores_dev, nullptr);

                    // Copy scores back to host
                    triattention_gpu_scores_to_host(
                        out_scores, scores_dev, n_candidates, nullptr);

                    // Cleanup device buffers
                    triattention_gpu_free_dev(cell_indices_dev);
                    triattention_gpu_free_dev(positions_dev);
                    triattention_gpu_free_dev(scores_dev);

                    return;
                }
            }
        }
    }
#endif

    // CPU path: dequantize, invert RoPE, score
    // 1. Dequantize K for this KV head for all candidate cells
    std::vector<float> dequant_buf((size_t)n_candidates * padded_hd);
    triattention_dequant_kv_head(
        dequant_buf.data(),
        k_tensor,
        cell_indices,
        kv_head_idx,
        n_candidates,
        padded_hd,
        cal->num_kv_heads,
        need_wht_inv);

    // 2. Invert RoPE -> pre-RoPE K
    std::vector<float> unrot_buf((size_t)n_candidates * padded_hd);
    triattention_invert_rope(
        unrot_buf.data(),
        dequant_buf.data(),
        positions,
        pimpl->omega,
        n_candidates,
        padded_hd,
        fc,
        cal->rope_style);

    // 3. Score keys
    triattention_score_keys(
        out_scores,
        unrot_buf.data(),
        &cal->head_stats[sh],
        pimpl->omega,
        pimpl->freq_scale_sq,
        pimpl->offsets,
        positions,
        frontier_position,
        n_candidates,
        padded_hd,
        fc,
        pimpl->n_offsets,
        pimpl->cfg.agg,
        pimpl->cfg.disable_trig);
}

void triattention_scorer::score_combined(
    float * combined,
    ggml_tensor * const * k_tensors,
    uint32_t n_kv_layers,
    const int32_t * layer_map,
    const uint32_t * cell_indices,
    const int32_t * positions,
    uint32_t n_candidates,
    int64_t frontier_position) const
{
    if (!valid() || n_candidates == 0) {
        if (combined && n_candidates > 0) {
            memset(combined, 0, n_candidates * sizeof(float));
        }
        return;
    }

    const auto * cal = pimpl->cal;

    // Initialize combined to -1e30f (will take max)
    for (uint32_t i = 0; i < n_candidates; i++) {
        combined[i] = -1e30f;
    }

    // Streaming: one temp score vector, reused per sampled head
    std::vector<float> temp_scores(n_candidates);

    for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
        const uint32_t model_layer = cal->sampled_layer[sh];
        const uint32_t attn_head   = cal->sampled_head[sh];
        const uint32_t kv_head     = attn_head / cal->num_kv_groups;

        // Find internal layer index from layer_map
        int32_t ikv = -1;
        for (uint32_t l = 0; l < n_kv_layers; l++) {
            if (layer_map[l] == (int32_t)model_layer) {
                ikv = (int32_t)l;
                break;
            }
        }
        if (ikv < 0) {
            // Layer not in cache — skip (zero contribution to max)
            continue;
        }

        const ggml_tensor * k_tensor = k_tensors[ikv];

        // Score this head into temp_scores
        score_head(
            temp_scores.data(),
            k_tensor,
            cell_indices,
            positions,
            kv_head,
            n_candidates,
            frontier_position);

        // Z-score normalize per head if configured
        if (pimpl->cfg.normalize_scores) {
            zscore_normalize(temp_scores.data(), n_candidates);
        }

        // combined[i] = max(combined[i], temp[i])
        for (uint32_t i = 0; i < n_candidates; i++) {
            if (temp_scores[i] > combined[i]) {
                combined[i] = temp_scores[i];
            }
        }
    }

    // If no sampled head matched any layer, all combined entries remain -1e30f.
    // Replace with 0 so the caller doesn't see sentinel values.
    bool any_valid = false;
    for (uint32_t i = 0; i < n_candidates; i++) {
        if (combined[i] > -1e29f) {
            any_valid = true;
            break;
        }
    }
    if (!any_valid) {
        memset(combined, 0, n_candidates * sizeof(float));
    }
}

// ============================================================================
// Accessors
// ============================================================================

uint32_t triattention_scorer::get_head_dim() const {
    return pimpl && pimpl->cal ? pimpl->cal->head_dim : 0;
}

uint32_t triattention_scorer::get_n_kv_heads() const {
    return pimpl && pimpl->cal ? pimpl->cal->num_kv_heads : 0;
}

uint32_t triattention_scorer::get_n_sampled() const {
    return pimpl && pimpl->cal ? pimpl->cal->n_sampled : 0;
}

uint32_t triattention_scorer::get_freq_count() const {
    return pimpl && pimpl->cal ? pimpl->cal->freq_count : 0;
}

const char * triattention_scorer::get_model_name() const {
    return pimpl && pimpl->cal ? pimpl->cal->model_name : "";
}

// ============================================================================
// Print info
// ============================================================================

void triattention_scorer::print_info(FILE * stream) const {
    if (!valid() || !pimpl || !pimpl->cal) {
        fprintf(stream, "[TriAttention] Scorer not initialized\n");
        return;
    }

    const auto * cal = pimpl->cal;
    const auto & cfg = pimpl->cfg;

    fprintf(stream, "\n=== TriAttention Scorer ===\n");
    fprintf(stream, "  Model:            %s\n", cal->model_name);
    fprintf(stream, "  Head dim:         %u\n", cal->head_dim);
    fprintf(stream, "  Layers:           %u\n", cal->num_layers);
    fprintf(stream, "  Attention heads:  %u\n", cal->num_attn_heads);
    fprintf(stream, "  KV heads:         %u\n", cal->num_kv_heads);
    fprintf(stream, "  KV groups:        %u\n", cal->num_kv_groups);
    fprintf(stream, "  RoPE theta:       %.1f\n", cal->rope_theta);
    fprintf(stream, "  RoPE style:       %s\n", cal->rope_style == 0 ? "half" : "interleaved");
    fprintf(stream, "  Sampled heads:    %u of %u\n", cal->n_sampled, cal->num_attn_heads);
    fprintf(stream, "  Freq count:       %u\n", cal->freq_count);
    fprintf(stream, "  ---\n");
    fprintf(stream, "  Score aggregation: %s\n",
            cfg.agg == TRIATTENTION_AGG_MEAN ? "mean" : "max");
    fprintf(stream, "  Geometric offsets: %u (max %u)\n", pimpl->n_offsets, cfg.offset_max);
    fprintf(stream, "  Normalize scores:  %s\n", cfg.normalize_scores ? "on" : "off");
    fprintf(stream, "  Disable MLR:       %s\n", cfg.disable_mlr ? "on" : "off");
    fprintf(stream, "  Disable trig:      %s\n", cfg.disable_trig ? "on" : "off");
    fprintf(stream, "===========================\n\n");
}
