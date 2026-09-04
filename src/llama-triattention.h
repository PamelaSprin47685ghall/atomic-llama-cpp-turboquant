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
//   version        uint32  1 or 2
//   head_dim       uint32  e.g. 128
//   num_layers     uint32  e.g. 36
//   num_attn_heads uint32  e.g. 36  (total attention heads, not KV heads)
//   num_kv_heads   uint32  e.g. 4   (grouped query attention KV heads)
//   rope_theta     float64 e.g. 10000.0
//   rope_style     uint32  0=half/NeoX pairing, 1=even-odd pairing
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
                        uint32_t n_kv_heads,
                        uint32_t runtime_rotary_dim = 0,
                        const float * runtime_omega = nullptr,
                        const float * runtime_freq_scale_sq = nullptr,
                        int32_t expected_rope_style = -1);
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

// Build the effective phase frequency and magnitude scale used by ggml RoPE.
// `freq_factors` follows ggml_rope_ext semantics: theta is divided by the
// corresponding factor when present. The YaRN branch mirrors ggml's phase
// interpolation and mscale behavior.
bool triattention_build_rope_tables(
    float * omega,
    float * freq_scale_sq,
    uint32_t rotary_dim,
    float freq_base,
    float freq_scale,
    int32_t n_ctx_orig,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow,
    const float * freq_factors);

// ============================================================================
// RERoT shared-memory reclaim policy (Stage 10 part 1 — §§23, A.4)
// ============================================================================
//
// TriAttention is the legal lossy reclaimer for RERoT PUBLIC shared memory.
// These helpers are pure policy: they own no cache, track no sequence, emit
// no tri_* metrics, and reinterpret no visibility. The scorer above stays the
// importance oracle; the KV cache stays the sole physical metadata owner;
// the RERoT document stays the sole PAC-DFS owner. Callers wire the three
// together as below. With RERoT OFF (no rerot_enabled / episode-active gate)
// none of this runs and existing scorer behavior is bit-for-bit unchanged.
//
// Ownership trichotomy (§A.4.1) — three facts the caller must never conflate:
//   * physical cell: an ephemeral llama_kv_cells index. Compaction, eviction,
//     and restore renumber it. It is valid only between those events.
//   * semantic owner: the stable (episode_id, node_id, run_id) triple in
//     llama_kv_rerot_meta plus the document run table. It survives repacking.
//   * retention ref: one seq-bitset membership (exec / archive / parked /
//     ancestor / shared-prefix). One PUBLIC cell routinely carries several at
//     once — e.g. exec_seq + archive_seq after rerot_freeze_to_archive.
// The scorer sees only (cell, storage_pos) pairs. Retention fan-out must
// NEVER inflate importance: the retention target below is computed over real
// logical history (distinct storage positions / run token_counts), never over
// reader or bookkeeping-ref counts. Use tri_rerot_dedup_candidates() to fold
// per-seq enumerations to distinct physical cells before scoring, and
// tri_rerot_count_distinct_positions() to size history without keeper
// multiplication.
//
// READ-ONLY KV use: fill the (cell, pos) inputs below from resident-only
// enumeration (llama_kv_cells::rerot_collect_run /
// llama_kv_cache::rerot_find_run_cells), guarded by rerot_has_active[_seq].
// Never cache a returned index across compaction, eviction, or restore —
// re-enumerate. Never write cell metadata from Tri code.
//
// Reclaim-then-view path (§A.4.2):
//   1. Enumerate resident PUBLIC cells only (ordinary prefix + public_live
//      cells visible to the reader under the layout builder's predicate;
//      private_control / pending_record stay owner-local).
//   2. Score + evict to tri_rerot_target_retention(); compact.
//   3. Re-enumerate residents (indices changed) in PAC-DFS rank order, sort
//      within each rank with tri_rerot_order_by_storage(), and densely pack
//      virtual positions [0, L). Virtual density is reader-virtual density,
//      NOT physical/history density: storage gaps after reclaim are expected
//      and must never be treated as a [p0,p1) span hole. Never assume run
//      storage continuity.
//
// Pressure order (unchanged product contract, queryable below):
//   fill-first -> drain-to-floor at 3/32 (tri_rerot_default_ratio()) ->
//   sticky maintenance (compressed seqs only) -> floor-exhausted ->
//   atomic fallback (idle demotion / active preemption; whole-episode
//   HARD_ABORT for RERoT). Recurrent-only pressure NEVER enters Tri reclaim:
//   tri_rerot_should_reclaim() is false for it by construction.
//
// Metrics: existing tri_* semantics are unchanged. These helpers emit no
// metrics themselves; the reclaim_kv caller keeps reporting
// references_removed per seq-ref (may exceed physical_freed for shared
// cells), target_references summed from deduped logical history, hard_keep
// for tail/in-flight protection, and shared_keep for multi-ref physical
// cells, plus score/pack timings and floor/fallback counters.
//
// The orchestrator's later matrix (RERoT FullKV vs 3/32, pressure + fork /
// queue / final fence, floor-exhausted + preemption) drives these query
// helpers directly; tri_rerot_policy_selfcheck() deterministically exercises
// the reclaim-then-view path (dedup, target, pressure gate, sparse order,
// dense virtual pack after a simulated eviction).

// One resident candidate as the scorer sees it: physical cell + semantic
// storage position. Owner/visibility/retention live in KV metadata, not here.
struct tri_rerot_candidate {
    uint32_t cell = 0;
    int32_t  pos  = 0;
};

// Pressure classification for the reclaim gate. MAINTENANCE is KV-side sticky
// upkeep (compressed seqs above floor with no new deficit), not new pressure.
enum tri_rerot_pressure {
    TRI_REROT_PRESSURE_NONE      = 0,
    TRI_REROT_PRESSURE_KV        = 1,
    TRI_REROT_PRESSURE_RECURRENT = 2,
    TRI_REROT_PRESSURE_BOTH      = 3,
    TRI_REROT_PRESSURE_MAINTENANCE = 4,
};

// Legal reclaim phases in order. FILL_FIRST performs no reclaim; DRAIN and
// STICKY both run score+evict (DRAIN covers every eligible seq, STICKY only
// compressed ones); FLOOR_EXHAUSTED means the floor was reached without
// satisfying the deficit; ATOMIC_FALLBACK is the non-Tri owner path.
enum tri_rerot_phase {
    TRI_REROT_PHASE_FILL_FIRST     = 0,
    TRI_REROT_PHASE_DRAIN_TO_FLOOR = 1,
    TRI_REROT_PHASE_STICKY         = 2,
    TRI_REROT_PHASE_FLOOR_EXHAUSTED = 3,
    TRI_REROT_PHASE_ATOMIC_FALLBACK = 4,
};

// Configured drain floor: first KV pressure drains eligible history to 3/32.
double tri_rerot_default_ratio();

// Retention target over REAL logical history: ceil(logical_tokens * ratio)
// lower-bounded by tail_guard. logical_tokens must count distinct logical
// tokens (e.g. tri_rerot_count_distinct_positions() over residents, or the
// run token_count sum) — never the sum over keeper refs. An invalid ratio
// falls back to tri_rerot_default_ratio().
uint32_t tri_rerot_target_retention(uint32_t logical_tokens, double ratio, uint32_t tail_guard);

// Classify pressure for the gate. maintenance_due upgrades an otherwise idle
// system to MAINTENANCE; any KV deficit dominates it.
tri_rerot_pressure tri_rerot_classify_pressure(bool kv_pressure, bool recurrent_pressure, bool maintenance_due);

// True only for KV pressure or sticky maintenance. Recurrent-only pressure
// (and idle) return false: TriAttention reclaims KV only and must never run
// for recurrent shortfall.
bool tri_rerot_should_reclaim(bool kv_pressure, bool recurrent_pressure, bool maintenance_due);

// Map a reclaim outcome to the next legal phase: !changed, or floor reached
// without satisfying the deficit, advances to FLOOR_EXHAUSTED (caller then
// takes ATOMIC_FALLBACK); any satisfying reclaim returns STICKY.
tri_rerot_phase tri_rerot_phase_after_reclaim(bool changed, bool floor_reached, bool capacity_satisfied);

// Stable name for logs / matrix assertions. Never null.
const char * tri_rerot_phase_name(tri_rerot_phase phase);

// Fold a per-seq enumeration (same PUBLIC cell may appear once per keeper
// ref: exec + archive + parked) to distinct physical cells in first-seen
// order. Score the folded list once. Returns the folded count; with null
// outputs it only counts. Null inputs with n > 0 yield 0.
uint32_t tri_rerot_dedup_candidates(
    const uint32_t * cells,
    const int32_t  * positions,
    uint32_t n,
    uint32_t * out_cells,
    int32_t  * out_positions);

// Count distinct storage positions in [positions, positions + n): the keeper-
// multiplication-free history length for tri_rerot_target_retention(). Null
// input yields 0.
uint32_t tri_rerot_count_distinct_positions(const int32_t * positions, uint32_t n);

// Stable sparse-safe order for one PAC-DFS rank group: ascending storage_pos,
// then ascending frontier (when frontier_or_null != null), then ascending
// cell (when cells_or_null != null), then input order. Writes the permutation
// of [0, n) into order. Dense virtual positions are simply the rank in this
// order, so storage gaps pack away. No-op for n == 0; null storage_pos or
// order with n > 0 is a no-op guard (order left untouched). The caller
// concatenates per-rank groups in document rank order (see doc above).
void tri_rerot_order_by_storage(
    const int32_t  * storage_pos,
    const uint64_t * frontier_or_null,
    const uint32_t * cells_or_null,
    uint32_t n,
    uint32_t * order);

// Deterministic exercise of the reclaim-then-view path: archive-fan-out
// dedup, keeper-free target sizing, recurrent-only gate, sparse ordering,
// and dense virtual repack after a simulated eviction. True when every
// invariant holds; false (with a stderr diagnostic) otherwise.
bool tri_rerot_policy_selfcheck();
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
// For adjacent/even-odd style (ggml NORMAL): pairs are (2f, 2f+1) in input,
// converted to half layout in output. ggml IMROPE is not this style; its
// vector pairing remains NeoX/half.
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
//   freq_scale_sq — [freq_count] squared RoPE magnitude scaling (nullptr => 1)
//   n_keys        — number of keys to process
//   head_dim      — full head dimension (for buffer sizing)
//   rotary_dim    — dimensions with RoPE applied (<= head_dim)
//   freq_count    — rotary_dim / 2
//   rope_style    — 0=half/NeoX, 1=adjacent even-odd
void triattention_invert_rope(
    float       * out,
    const float * post_rope_k,
    const int32_t * positions,
    const float * omega,
    const float * freq_scale_sq,
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
