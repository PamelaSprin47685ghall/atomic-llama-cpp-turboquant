#pragma once

// TriAttention: Trigonometric KV Cache Importance Scorer for llama.cpp
// Based on arXiv 2604.04921 (MIT/NVIDIA/ZJU)
//
// Phase 1: Pure "importance oracle" — takes K tensors + candidate cells +
// positions + frontier and returns importance scores. No cache ownership,
// no position tracking, no trigger logic, no GPU.
//
// Scores cached keys using pre-RoPE Q/K concentration statistics and
// trigonometric importance estimation. The caller is responsible for
// deciding which cells to evict based on the returned scores.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <memory>

// ============================================================================
// Binary calibration file format (.triattention)
// ============================================================================
//
// Header:
//   magic          uint32  0x54524941 ("TRIA")
//   version        uint32  1
//   head_dim       uint32  e.g. 128
//   num_layers     uint32  e.g. 36
//   num_attn_heads uint32  e.g. 36  (total attention heads, not KV heads)
//   num_kv_heads   uint32  e.g. 4   (grouped query attention KV heads)
//   rope_theta     float64 e.g. 10000.0
//   rope_style     uint32  0=half, 1=interleaved
//   n_sampled      uint32  number of (layer, head) pairs with stats
//   freq_count     uint32  head_dim / 2
//   name_len       uint32  length of model name string (including null)
//   name           char[name_len]  UTF-8 null-terminated model name
//
// Per sampled head (repeated n_sampled times):
//   layer_idx      uint32
//   head_idx       uint32  (attention head index, 0..num_attn_heads-1)
//   q_mean_real    float32[freq_count]  Re(E[q_f])
//   q_mean_imag    float32[freq_count]  Im(E[q_f])
//   q_abs_mean     float32[freq_count]  E[||q_f||]
//   r_f            float32[freq_count]  ||E[q_f]|| / E[||q_f||] (validation)

#define TRIATTENTION_MAGIC   0x54524941u  // "TRIA" in little-endian
#define TRIATTENTION_VERSION 2u  // v2 adds rotary_dim for partial RoPE support

// ============================================================================
// Enums
// ============================================================================

// Score aggregation over geometric future offsets D = {1,2,4,...,offset_max}
enum triattention_agg {
    TRIATTENTION_AGG_MEAN = 0,  // Paper default: mean over offsets
    TRIATTENTION_AGG_MAX  = 1,  // Alternative: max over offsets
};

// ============================================================================
// Data structures
// ============================================================================

// Per-(layer, head) calibration statistics
// Stores the complex mean of pre-RoPE Q vectors and the mean magnitude
// per frequency band f in [0, freq_count), where freq_count = head_dim/2
struct triattention_head_stats {
    float * q_mean_real;    // [freq_count]  Re(E[q_f])
    float * q_mean_imag;    // [freq_count]  Im(E[q_f])
    float * q_abs_mean;     // [freq_count]  E[||q_f||]

    // Precomputed at init time from the above:
    float * q_mean_abs;     // [freq_count]  ||E[q_f]|| = sqrt(re^2 + im^2)
    float * extra_weight;   // [freq_count]  E[||q_f||] - ||E[q_f]|| (norm excess, MLR-weighted)
};

// Model calibration data loaded from .triattention file
struct triattention_calibration {
    uint32_t head_dim;
    uint32_t num_layers;
    uint32_t num_attn_heads;      // total attention heads
    uint32_t num_kv_heads;        // GQA KV heads
    uint32_t num_kv_groups;       // = num_attn_heads / num_kv_heads
    double   rope_theta;
    uint32_t rope_style;          // 0 = half, 1 = interleaved
    uint32_t freq_count;          // = rotary_dim / 2
    uint32_t rotary_dim;          // dimensions with RoPE applied (<= head_dim). v2 field; v1 defaults to head_dim
    uint32_t n_sampled;           // number of (layer, head) pairs

    // Per sampled head arrays — length n_sampled
    uint32_t * sampled_layer;     // [n_sampled]  layer index
    uint32_t * sampled_head;      // [n_sampled]  attention head index
    triattention_head_stats * head_stats;  // [n_sampled]

    char model_name[256];
};

// ============================================================================
// Scorer config — no budget, no trigger, no prefill protection
// ============================================================================

struct triattention_scorer_config {
    uint32_t offset_max = 65536;    // max geometric offset for scoring
    enum triattention_agg agg = TRIATTENTION_AGG_MEAN; // paper default: mean over future offsets
    bool normalize_scores = true;   // z-score per head (AGENTS.md: normalize=on)
    bool disable_mlr = false;       // ablation
    bool disable_trig = false;      // ablation
    bool enable_logging = false;
};

// ============================================================================
// Pure importance oracle — holds calibration + precomputed data, produces scores
// ============================================================================

class triattention_scorer {
public:
    // Load calibration from file, validate against model params
    // Returns invalid() on failure (file not found, format error, model mismatch)
    triattention_scorer(const char * stats_path,
                        const triattention_scorer_config & cfg,
                        double rope_theta,
                        uint32_t head_dim,
                        uint32_t n_kv_heads);
    ~triattention_scorer();

    bool valid() const;

    // Score candidates for a single (layer, head) pair
    // out_scores: [n_candidates] importance scores
    // k_tensor: K cache tensor for this layer
    // cell_indices: [n_candidates] physical cell indices into k_tensor
    // positions: [n_candidates] logical positions of each cell
    // kv_head_idx: which KV head (0..n_kv_heads-1)
    // frontier_position: current decode position (for delta computation)
    void score_head(
        float * out_scores,
        const ggml_tensor * k_tensor,
        const uint32_t * cell_indices,
        const int32_t * positions,
        uint32_t kv_head_idx,
        uint32_t n_candidates,
        int64_t frontier_position
    ) const;

    // Combined score across ALL sampled heads using max/union (global mode)
    // combined: [n_candidates] output — max over all sampled heads
    // k_tensors: [n_kv_layers] K cache tensors indexed by internal layer id
    // layer_map: [n_kv_layers] maps internal layer index -> model layer index
    // Uses streaming scratch: O(n_candidates) host memory, not O(n_sampled * n_candidates)
    void score_combined(
        float * combined,
        ggml_tensor * const * k_tensors,
        uint32_t n_kv_layers,
        const int32_t * layer_map,
        const uint32_t * cell_indices,
        const int32_t * positions,
        uint32_t n_candidates,
        int64_t frontier_position
    ) const;

    // Accessors
    uint32_t get_head_dim() const;
    uint32_t get_n_kv_heads() const;
    uint32_t get_n_sampled() const;
    uint32_t get_freq_count() const;
    const char * get_model_name() const;

    // Print calibration info to stream
    void print_info(FILE * stream) const;

private:
    // Score one exact calibration sample. Unlike score_head(), the sampled
    // attention head is explicit; score_combined() uses this so heads that
    // share a KV head still contribute their distinct calibration statistics.
    void score_sampled_head(
        float * out_scores,
        const ggml_tensor * k_tensor,
        const uint32_t * cell_indices,
        const int32_t * positions,
        uint32_t sampled_head_idx,
        uint32_t kv_head_idx,
        uint32_t n_candidates,
        int64_t frontier_position
    ) const;

    struct impl;
    std::unique_ptr<impl> pimpl;
};

// Spread a strong token score to contiguous neighboring positions. This avoids
// selecting only fragments of multi-token facts such as numbers and identifiers.
void triattention_max_pool_scores(
    float * pooled,
    const float * scores,
    const int32_t * positions,
    uint32_t n_candidates,
    uint32_t radius);

// ============================================================================
// Core scoring functions (CPU implementations)
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// Invert RoPE rotation on post-RoPE key vectors to recover pre-RoPE keys.
// Paper Eq. 4: k_base = invert_rope(k_rotated, position, omega)
//
// For "half" style (Llama/Qwen): first d/2 = real, second d/2 = imaginary
//   out[f]           = in[f]*cos(w_f*pos) + in[f+fc]*sin(w_f*pos)
//   out[f+fc]        = in[f+fc]*cos(w_f*pos) - in[f]*sin(w_f*pos)
//
// For "interleaved" style: pairs are (2f, 2f+1) in input, converted to half layout in output
//   out[f]           = in[2f]*cos(w_f*pos) + in[2f+1]*sin(w_f*pos)
//   out[f+fc]        = in[2f+1]*cos(w_f*pos) - in[2f]*sin(w_f*pos)
//
// For partial RoPE (rotary_dim < head_dim): only the first rotary_dim dimensions
// are processed; dimensions [rotary_dim, head_dim) are copied unchanged.
//
// Output is always in half layout regardless of rope_style.
//
// Parameters:
//   out           — [n_keys, head_dim] pre-RoPE K output (half layout)
//   post_rope_k   — [n_keys, head_dim] post-RoPE K input (dequantized)
//   positions     — [n_keys] absolute positions of each key
//   omega         — [freq_count] RoPE frequencies
//   n_keys        — number of keys to process
//   head_dim      — full head dimension (for buffer sizing)
//   rotary_dim    — dimensions with RoPE applied (<= head_dim)
//   freq_count    — rotary_dim / 2
//   rope_style    — 0=half, 1=interleaved
void triattention_invert_rope(
    float       * out,
    const float * post_rope_k,
    const int32_t * positions,
    const float * omega,
    uint32_t n_keys,
    uint32_t head_dim,
    uint32_t rotary_dim,
    uint32_t freq_count,
    uint32_t rope_style);

// Score cached keys for a single (layer, head) pair.
// Paper Eqs. 6-10: trigonometric scoring + MLR norm term
//
// For each key i at position p_k:
//   For each freq band f:
//     k_f = complex(pre_rope_k[i*hd+f], pre_rope_k[i*hd+f+fc])
//     amp_f = q_mean_abs[f] * |k_f|
//     phi_f = atan2(Im(E[q_f]*conj(k_f)), Re(E[q_f]*conj(k_f)))
//     extra_f = extra_weight[f] * |k_f|
//   For each offset d in offsets:
//     S_trig += sum_f amp_f * fscale_sq[f] * cos(omega[f]*(Delta+d) + phi_f)
//   S_norm = sum_f extra_f * fscale_sq[f]
//   score = aggregate(S_trig + S_norm) over offsets
//
// Parameters:
//   out_scores    — [n_keys] output importance scores
//   pre_rope_k    — [n_keys, head_dim] pre-RoPE keys (from invert_rope)
//   stats         — calibration statistics for this (layer, head)
//   omega         — [freq_count] RoPE frequencies
//   freq_scale_sq — [freq_count] frequency scaling^2
//   offsets       — [n_offsets] geometric future offsets
//   key_positions — [n_keys] absolute positions
//   round_start   — current decode position (absolute)
//   n_keys        — number of keys
//   head_dim      — full head dimension
//   freq_count    — head_dim / 2
//   n_offsets     — number of geometric offsets
//   agg           — score aggregation method (mean or max)
//   disable_trig  — if true, drop trigonometric term (norm-only ablation)
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
    bool disable_trig);

#ifdef __cplusplus
}
#endif
