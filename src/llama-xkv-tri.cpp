#include "llama-xkv-tri.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace llama_xkv {

// Bounded scratch carving: all planning/scoring temporaries come from one
// contiguous arena span (or a single heap block on the legacy path). No
// unordered_map/set or growing STL containers on the carved path; sorted
// contiguous spans + CSR runs instead. Every region is 64B-aligned.
static size_t tri_round64(size_t n) { return (n + size_t(63)) & ~size_t(63); }

struct tri_bump {
    uint8_t * base = nullptr;
    size_t cap = 0;
    size_t off = 0;
    bool overflow = false;
    void * carve(size_t n) {
        const size_t s = tri_round64(off);
        if (n > cap || s > cap - n) { overflow = true; return nullptr; }
        void * p = base + s;
        off = s + n;
        return p;
    }
    size_t mark() const { return off; }
    void reset_to(size_t m) { off = m; if (off > cap) overflow = true; }
};

template <typename T>
static T * tri_carve(tri_bump & b, size_t n) {
    if (n == 0) return nullptr;
    if (n > std::numeric_limits<size_t>::max() / sizeof(T)) { b.overflow = true; return nullptr; }
    return static_cast<T *>(b.carve(n * sizeof(T)));
}

// Keeper-ref bitset words per candidate (LLAMA_MAX_SEQ bits, branch-free ops)
static constexpr uint32_t TRI_REF_WORDS = (LLAMA_MAX_SEQ + 63u) / 64u;
static bool tri_ref_test(const uint64_t * bits, llama_seq_id s) {
    if (s < 0 || (uint64_t)s >= (uint64_t)LLAMA_MAX_SEQ) return false;
    return (bits[(uint32_t)s >> 6] >> ((uint32_t)s & 63u)) & 1ull;
}
static void tri_ref_clear(uint64_t * bits, llama_seq_id s) {
    if (s < 0 || (uint64_t)s >= (uint64_t)LLAMA_MAX_SEQ) return;
    bits[(uint32_t)s >> 6] &= ~(1ull << ((uint32_t)s & 63u));
}
static bool tri_ref_empty(const uint64_t * bits) {
    for (uint32_t w = 0; w < TRI_REF_WORDS; ++w) if (bits[w]) return false;
    return true;
}
static uint32_t tri_ref_count(const uint64_t * bits) {
    uint32_t n = 0;
    for (uint32_t w = 0; w < TRI_REF_WORDS; ++w) n += (uint32_t)__builtin_popcountll(bits[w]);
    return n;
}

// Explicit frontier lookup: seqs first, then the caller-supplied table.
// Unknown ref => false (caller must preserve/fail closed, never drop guard).
static bool tri_lookup_frontier(const std::vector<xkv_tri_seq_info> & seqs,
                                 const xkv_tri_frontier_table * tab,
                                 const xkv_tri_config & cfg,
                                 llama_seq_id ref,
                                 int64_t & frontier_out,
                                 uint32_t & guard_out) {
    for (const auto & s : seqs) {
        if (s.seq_id == ref) {
            frontier_out = s.frontier_pos;
            guard_out = s.tail_guard > 0 ? s.tail_guard : cfg.recent_window;
            return true;
        }
    }
    if (tab) {
        for (const auto & e : *tab) {
            if (e.seq_id == ref) {
                frontier_out = e.frontier_pos;
                guard_out = cfg.recent_window;
                return true;
            }
        }
    }
    return false;
}

// Max A/B padded widths over factored candidates' actual descriptors.
// Falls back to slice-derived widths when the store is unavailable.
static void tri_max_pads(const llama_xkv_cache_store * store,
                          const xkv_tri_candidate * cands,
                          size_t n_cands,
                          const uint32_t * subset,
                          size_t n_sub,
                          bool has_subset,
                          const std::vector<xkv_layer_slice> & slices,
                          uint32_t head_dim,
                          uint32_t & a_pad_out,
                          uint32_t & b_pad_out,
                          size_t & tmp_out) {
    uint32_t a_pad = 0, b_pad = 0;
    size_t tmp = 0;
    if (store && cands) {
        const size_t n = has_subset ? n_sub : n_cands;
        for (size_t i = 0; i < n; ++i) {
            const size_t ci = has_subset ? subset[i] : i;
            if (ci >= n_cands) continue;
            const auto & c = cands[ci];
            if (c.device_owned) continue; // never host-decoded; no pad needed
            if (c.location.kind != xkv_location_kind::factored || c.location.segment_id == 0) continue;
            auto seg = store->get_segment(c.location.segment_id);
            if (!seg) continue;
            for (const auto & g : seg->groups) {
                const uint32_t ap = (uint32_t)g.a_k.desc.padded_shape.cols;
                if (ap > a_pad) a_pad = ap;
                if (g.b_k) {
                    const uint32_t bp = (uint32_t)g.b_k->desc.padded_shape.cols;
                    if (bp > b_pad) b_pad = bp;
                }
                size_t q = 0;
                if (decode_rows_scratch_bytes(g.a_k.desc, q, nullptr) && q > tmp) tmp = q;
                if (g.b_k && decode_rows_scratch_bytes(g.b_k->desc, q, nullptr) && q > tmp) tmp = q;
            }
        }
    }
    if (a_pad == 0 || b_pad == 0) {
        uint32_t mr = 0;
        for (const auto & s : slices) {
            uint32_t r = s.head_dim;
            if (s.feature_dim_k > r) r = s.feature_dim_k;
            if (r > mr) mr = r;
        }
        if (mr == 0) mr = head_dim;
        const uint32_t pr = (mr + 127u) / 128u * 128u;
        if (a_pad == 0) a_pad = pr;
        if (b_pad == 0) b_pad = pr;
    }
    a_pad_out = a_pad;
    b_pad_out = b_pad;
    tmp_out = tmp;
}

// Per-row factored byte cost of one evicted row in a segment (A strides only;
// shared B/metadata amortize and are never double-counted per row).
static size_t tri_factored_row_bytes(const xkv_segment & seg) {
    size_t b = 0;
    for (const auto & g : seg.groups) {
        b += g.a_k.desc.row_stride_bytes;
        b += g.a_v.desc.row_stride_bytes;
    }
    return b;
}

// Two-pass z-score normalization over a candidate subset
static void zscore_normalize_subset(float * scores, uint32_t n) {
    if (n <= 1) {
        return;
    }
    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        sum += scores[i];
    }
    double mean = sum / (double)n;

    double var_sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        double d = scores[i] - mean;
        var_sum += d * d;
    }
    double std = std::sqrt(var_sum / (double)n);
    if (std < 1e-10) {
        std = 1e-10;
    }

    for (uint32_t i = 0; i < n; ++i) {
        scores[i] = (float)((scores[i] - mean) / std);
    }
}

// ============================================================================
// Constructor
// ============================================================================

xkv_tri_adapter::xkv_tri_adapter(
    const triattention_calibration & calib,
    const float * runtime_omega,
    const float * runtime_freq_scale_sq,
    const xkv_tri_config & cfg
) : cal(calib), config(cfg) {
    if (cal.head_dim == 0 || cal.num_kv_heads == 0 || cal.rotary_dim == 0 || cal.n_sampled == 0) {
        throw std::invalid_argument("xkv_tri_adapter: invalid calibration dimensions");
    }
    // Reject legacy interleaved rope_style; IMRoPE pairing is also NeoX half (rope_style == 0)
    if (cal.rope_style != 0) {
        throw std::invalid_argument("xkv_tri_adapter: unsupported rope_style != 0 (must be half/NeoX pairing)");
    }
    if (!runtime_omega) {
        throw std::invalid_argument("xkv_tri_adapter: runtime_omega cannot be null");
    }
    // Production head aggregation contract: normalized max/union is fixed.
    // Non-default head aggregation or disabling normalization requires explicit is_ablation flag.
    if (!config.is_ablation) {
        if (config.head_agg != TRIATTENTION_AGG_MAX) {
            throw std::invalid_argument("xkv_tri_adapter: non-max head aggregation requires is_ablation=true in production");
        }
        if (!config.normalize_scores) {
            throw std::invalid_argument("xkv_tri_adapter: disabling normalization requires is_ablation=true in production");
        }
    }

    const uint32_t fc = cal.freq_count;
    if (fc != cal.rotary_dim / 2) {
        throw std::invalid_argument("xkv_tri_adapter: freq_count != rotary_dim / 2");
    }

    // Validate finite values in runtime tables and config
    if (config.ratio <= 0.0 || config.ratio > 1.0 || std::isnan(config.ratio)) {
        throw std::invalid_argument("xkv_tri_adapter: invalid non-finite or out-of-range ratio");
    }

    omega.resize(fc);
    for (uint32_t f = 0; f < fc; ++f) {
        if (!std::isfinite(runtime_omega[f])) {
            throw std::invalid_argument("xkv_tri_adapter: non-finite value in runtime_omega");
        }
        omega[f] = runtime_omega[f];
    }

    freq_scale_sq.resize(fc);
    for (uint32_t f = 0; f < fc; ++f) {
        float s = runtime_freq_scale_sq ? runtime_freq_scale_sq[f] : 1.0f;
        if (!std::isfinite(s) || s < 0.0f) {
            throw std::invalid_argument("xkv_tri_adapter: non-finite or negative value in runtime_freq_scale_sq");
        }
        freq_scale_sq[f] = s;
    }

    // Precompute geometric offsets D = {1, 2, 4, 8, ... <= 65536}
    offsets.clear();
    for (uint32_t d = 1; d <= 65536; d *= 2) {
        offsets.push_back((float)d);
    }
}

bool xkv_tri_adapter::valid() const {
    return cal.head_dim > 0 && cal.n_sampled > 0 && !omega.empty();
}

// ============================================================================
// stream_factored_k_head
// ============================================================================

void xkv_tri_adapter::stream_factored_k_head(
    const xkv_segment & segment,
    const xkv_layer_slice & slice,
    uint32_t kv_head,
    const uint32_t * seg_row_indices,
    uint32_t n_rows,
    float * dst_half,
    size_t dst_capacity_elements,
    uint32_t tile_size
) const {
    if (n_rows == 0) {
        return;
    }

    const auto * g = segment.find_group_for_layer(slice.owning_layer);
    if (!g) {
        throw std::invalid_argument("stream_factored_k_head: no factor group found for owning_layer " + std::to_string(slice.owning_layer));
    }

    if (!dst_half) {
        throw std::invalid_argument("stream_factored_k_head: dst_half is null");
    }
    if (!seg_row_indices) {
        throw std::invalid_argument("stream_factored_k_head: seg_row_indices is null");
    }
    if (tile_size == 0) {
        tile_size = 32;
    }

    const uint32_t head_dim = slice.head_dim;
    const uint32_t rotary_dim = slice.rotary_dim;
    if (head_dim == 0 || rotary_dim == 0 || rotary_dim > head_dim) {
        throw std::invalid_argument("stream_factored_k_head: invalid slice dimensions");
    }
    if (slice.rope_style != 0) {
        throw std::invalid_argument("stream_factored_k_head: slice rope_style must be 0 (half layout)");
    }
    if (kv_head >= slice.n_kv_heads && slice.n_kv_heads > 0) {
        throw std::out_of_range("stream_factored_k_head: kv_head out of range for slice");
    }

    if (slice.feature_offset_k > UINT32_MAX - (kv_head + 1) * head_dim) {
        throw std::overflow_error("stream_factored_k_head: feature offset overflow");
    }
    const uint32_t head_feat_start = slice.feature_offset_k + kv_head * head_dim;
    if (head_feat_start + head_dim > slice.feature_offset_k + slice.feature_dim_k) {
        throw std::out_of_range("stream_factored_k_head: head feature span exceeds layer feature_dim_k");
    }

    if ((uint64_t)n_rows > UINT64_MAX / head_dim) {
        throw std::overflow_error("stream_factored_k_head: n_rows * head_dim overflow");
    }
    const size_t req_elements = (size_t)n_rows * head_dim;
    if (dst_capacity_elements < req_elements) {
        throw std::invalid_argument("stream_factored_k_head: dst capacity smaller than required elements");
    }

    // Validate immutable B_K
    if (!g->b_k) {
        throw std::invalid_argument("stream_factored_k_head: group b_k is null");
    }
    const encoded_matrix & b_k_ref = *g->b_k;
    if (b_k_ref.desc.orient != orientation::feature_major_transposed) {
        throw std::invalid_argument("stream_factored_k_head: b_k must be feature_major_transposed");
    }
    if (head_feat_start + head_dim > b_k_ref.desc.logical_shape.rows) {
        throw std::out_of_range("stream_factored_k_head: head feature span exceeds b_k logical rows");
    }

    // Decode ONLY exact B head feature rows, never whole B!
    std::vector<uint64_t> b_row_indices(head_dim);
    for (uint32_t d = 0; d < head_dim; ++d) {
        b_row_indices[d] = head_feat_start + d;
    }

    const uint64_t b_pad_cols = b_k_ref.desc.padded_shape.cols;
    const uint64_t rank_k = b_k_ref.desc.logical_shape.cols;
    const uint64_t a_pad_cols = g->a_k.desc.padded_shape.cols;

    std::vector<float> decoded_b_head(head_dim * b_pad_cols);
    decode_rows(
        b_k_ref,
        b_row_indices.data(),
        head_dim,
        decoded_b_head.data(),
        decoded_b_head.size(),
        value_domain::canonical
    );

    // Validate A rows and live mask
    for (uint32_t r = 0; r < n_rows; ++r) {
        uint32_t row_idx = seg_row_indices[r];
        if (row_idx >= segment.n_rows) {
            throw std::out_of_range("stream_factored_k_head: row index out of bounds in segment");
        }
        if (row_idx < segment.live_rows.size() && !segment.live_rows[row_idx]) {
            throw std::runtime_error("stream_factored_k_head: requested row is dead in segment");
        }
    }

    // Stream bounded tiles of A and reconstruct K_pre = A_row @ B_feat^T
    std::vector<uint64_t> tile_row_indices;
    std::vector<float> tile_a_buf;

    for (uint32_t tile_start = 0; tile_start < n_rows; tile_start += tile_size) {
        const uint32_t cur_tile_size = std::min(tile_size, n_rows - tile_start);

        tile_row_indices.resize(cur_tile_size);
        for (uint32_t r = 0; r < cur_tile_size; ++r) {
            tile_row_indices[r] = seg_row_indices[tile_start + r];
        }

        tile_a_buf.resize((size_t)cur_tile_size * a_pad_cols);
        decode_rows(
            g->a_k,
            tile_row_indices.data(),
            cur_tile_size,
            tile_a_buf.data(),
            tile_a_buf.size(),
            value_domain::canonical
        );

        for (uint32_t r = 0; r < cur_tile_size; ++r) {
            const float * a_row = tile_a_buf.data() + (size_t)r * a_pad_cols;
            float * out_row = dst_half + (size_t)(tile_start + r) * head_dim;

            for (uint32_t d = 0; d < head_dim; ++d) {
                const float * b_row = decoded_b_head.data() + (size_t)d * b_pad_cols;
                float sum = 0.0f;
                for (uint64_t k = 0; k < rank_k; ++k) {
                    sum += a_row[k] * b_row[k];
                }
                out_row[d] = sum;
            }
        }
    }
}

// ============================================================================
// score_candidate_subset
// ============================================================================

// Forward declarations: carved core + geometry live below the wrappers.
static size_t tri_score_total(uint32_t n_sub, uint32_t tile_size, uint32_t hd, uint32_t a_pad,
                               uint32_t b_pad, size_t tmp_bytes, uint32_t n_sampled, uint32_t n_slices);
static void tri_score_carved(const triattention_calibration & cal, const float * omega,
                              const float * fsq, const float * offsets, uint32_t n_offsets,
                              const xkv_tri_config & config,
                              const std::vector<xkv_tri_candidate> & candidates, const uint32_t * subset,
                              uint32_t n_sub, const std::vector<xkv_layer_slice> & slices,
                              const llama_xkv_cache_store & store, int64_t seq_frontier_position,
                              float * out_pooled_scores, float * out_raw_combined,
                              const hot_k_provider_fn & hot_provider,
                              const device_factor_scoring_fn & device_scoring,
                              const hot_k_span_provider_fn & hot_span, uint32_t a_pad, uint32_t b_pad,
                              size_t tmp_bytes, const xkv_tri_selected_k_fetch_fn & fetch,
                              tri_bump & bump);
static void tri_build_carved(const triattention_calibration & cal, const float * omega,
                              const float * fsq, const float * offsets, uint32_t n_offsets,
                              const xkv_tri_config & config,
                              const std::vector<xkv_tri_candidate> & candidates,
                              const std::vector<xkv_tri_seq_info> & seqs,
                              const std::vector<xkv_layer_slice> & slices,
                              const llama_xkv_cache_store & store, const xkv_tri_pressure_state & pressure,
                              const hot_k_provider_fn & hot_provider,
                              const xkv_tri_frontier_table * explicit_frontiers,
                              const device_factor_scoring_fn & device_scoring,
                              const hot_k_span_provider_fn & hot_span, uint32_t a_pad, uint32_t b_pad,
                              size_t tmp_bytes, const xkv_tri_selected_k_fetch_fn & fetch,
                              xkv_tri_mutation_plan & plan, tri_bump & bump);

void xkv_tri_adapter::score_candidate_subset(
    const std::vector<xkv_tri_candidate> & candidates,
    const std::vector<uint32_t> & subset_indices,
    const std::vector<xkv_layer_slice> & slices,
    const llama_xkv_cache_store & store,
    int64_t seq_frontier_position,
    float * out_pooled_scores,
    float * out_raw_combined,
    const hot_k_provider_fn & hot_provider,
    const device_factor_scoring_fn & device_scoring,
    const hot_k_span_provider_fn & hot_span,
    xkv_arena_lease * scratch_lease,
    const xkv_tri_selected_k_fetch_fn & fetch
) const {
    const uint32_t n_sub = (uint32_t)subset_indices.size();
    if (n_sub == 0 || !valid()) {
        return;
    }

    const uint32_t hd = cal.head_dim;

    // Scratch: arena-carved when a lease is supplied (zero heap), else one
    // bounded heap block. Layout matches tri_score_geometry exactly.
    const uint32_t tile_size = config.tile_size > 0 ? config.tile_size : 32;
    uint32_t a_pad = 0, b_pad = 0;
    size_t tmp_bytes = 0;
    tri_max_pads(&store, candidates.data(), candidates.size(), subset_indices.data(), n_sub, true,
                 slices, hd, a_pad, b_pad, tmp_bytes);
    const size_t need =
        tri_score_total(n_sub, tile_size, hd, a_pad, b_pad, tmp_bytes, cal.n_sampled,
                        (uint32_t)slices.size());
    std::vector<uint8_t> heap_scratch;
    tri_bump bump;
    if (scratch_lease && scratch_lease->valid() && scratch_lease->size() >= need) {
        bump.base = static_cast<uint8_t *>(scratch_lease->data());
        bump.cap = scratch_lease->size();
    } else if (scratch_lease && scratch_lease->valid()) {
        throw std::runtime_error("workspace: score scratch lease one-byte-short");
    } else {
        heap_scratch.resize(need);
        bump.base = heap_scratch.data();
        bump.cap = heap_scratch.size();
    }
    tri_score_carved(cal, omega.data(), freq_scale_sq.data(), offsets.data(),
                       (uint32_t)offsets.size(), config, candidates,
                       subset_indices.data(), n_sub, slices, store, seq_frontier_position,
                       out_pooled_scores, out_raw_combined, hot_provider, device_scoring, hot_span,
                       a_pad, b_pad, tmp_bytes, fetch, bump);
    if (bump.overflow) {
        throw std::runtime_error("workspace: score scratch carve overflow");
    }
}

// ---- carved scoring geometry: exact region accounting shared by the
// estimator, the heap fallback, and the arena path (one-byte-short exact).
struct tri_score_geom {
    size_t raw = 0, pos = 0, grp = 0;
    size_t ktile = 0, bhead = 0, segk = 0, hotk = 0, tilea = 0;
    size_t brow = 0, tup = 0, hotidx = 0, gtab = 0, dtmp = 0, trow = 0, pord = 0;
    // Per-carve sort/order/row temps: hord, dord, srows (u32 each) + devtmp (f32).
    size_t srt = 0;
    size_t total = 0;
};

static tri_score_geom tri_score_geometry(uint32_t n_sub, uint32_t tile_size, uint32_t hd,
                                          uint32_t a_pad, uint32_t b_pad, size_t tmp_bytes,
                                          uint32_t n_sampled) {
    tri_score_geom g;
    // Worst case: every sampled head shares one (layer, KV-head) group.
    const size_t max_grp = n_sampled > 0 ? n_sampled : 1;
    g.raw    = tri_round64((size_t)n_sub * sizeof(float));
    g.pos    = tri_round64((size_t)n_sub * sizeof(int32_t));
    g.grp    = tri_round64(max_grp * (size_t)n_sub * sizeof(float));
    g.ktile  = tri_round64((size_t)tile_size * hd * sizeof(float));
    g.bhead  = tri_round64((size_t)hd * b_pad * sizeof(float));
    g.segk   = tri_round64((size_t)tile_size * hd * sizeof(float));
    g.hotk   = tri_round64((size_t)tile_size * hd * sizeof(float));
    g.tilea  = tri_round64((size_t)tile_size * a_pad * sizeof(float));
    g.brow   = tri_round64((size_t)hd * sizeof(uint64_t));
    // Host + device tuple lanes: (seg_id, seg_ver, tile_pos, row) each.
    g.tup    = tri_round64((size_t)tile_size * 4 * sizeof(uint64_t)) * 2;
    // hpos + hcell carved as two separate spans.
    g.hotidx = tri_round64((size_t)tile_size * sizeof(uint32_t)) * 2;
    // hord + dord + srows + devtmp + dpos carved as separate spans.
    g.srt = tri_round64((size_t)tile_size * sizeof(uint32_t)) * 3 +
            tri_round64((size_t)tile_size * sizeof(float)) +
            tri_round64((size_t)tile_size * sizeof(int32_t)); // dpos: compact run positions
    // Sorted group table: keys + member order + CSR bounds.
    g.gtab   = tri_round64((size_t)n_sampled * sizeof(uint64_t))
             + tri_round64((size_t)n_sampled * sizeof(uint32_t))
             + tri_round64((size_t)(n_sampled + 1) * sizeof(uint32_t));
    g.dtmp   = tri_round64(tmp_bytes);
    g.trow   = tri_round64((size_t)tile_size * sizeof(uint64_t));
    g.pord   = tri_round64((size_t)n_sub * sizeof(uint32_t));
    g.total  = g.raw + g.pos + g.grp + g.ktile + g.bhead + g.segk + g.hotk + g.tilea
             + g.brow + g.tup + g.hotidx + g.gtab + g.dtmp + g.trow + g.pord + g.srt;
    return g;
}

static size_t tri_score_total(uint32_t n_sub, uint32_t tile_size, uint32_t hd,
                               uint32_t a_pad, uint32_t b_pad, size_t tmp_bytes, uint32_t n_sampled,
                               uint32_t /*n_slices*/) {
    return tri_score_geometry(n_sub, tile_size, hd, a_pad, b_pad, tmp_bytes, n_sampled).total;
}

static const xkv_layer_slice * tri_find_slice(const std::vector<xkv_layer_slice> & slices, uint32_t ml) {
    for (const auto & s : slices) {
        if (s.model_layer == ml) return &s;
    }
    return nullptr;
}

// Bounded factored K reconstruction for the carved path: same validation and
// exact-B-head-row semantics as xkv_tri_adapter::stream_factored_k_head, but
// every temporary (decoded B head rows, decoded A tiles, codec scratch) comes
// from caller-carved buffers via the zero-heap decode_rows overload.
static void tri_stream_k_bounded(
    const xkv_segment & segment,
    const xkv_layer_slice & slice,
    uint32_t kv_head,
    const uint32_t * seg_row_indices,
    uint32_t n_rows,
    float * dst_half,
    size_t dst_capacity_elements,
    float * tilea_buf,
    size_t tilea_cap,
    float * bhead_buf,
    size_t bhead_cap,
    uint64_t * brow_buf,
    size_t brow_cap,
    uint64_t * trow_buf,
    size_t trow_cap,
    uint8_t * dtmp,
    size_t dtmp_bytes,
    uint32_t tile_size) {
    if (n_rows == 0) return;
    const auto * g = segment.find_group_for_layer(slice.owning_layer);
    if (!g) {
        throw std::invalid_argument("tri_stream_k_bounded: no factor group for owning_layer " +
                                      std::to_string(slice.owning_layer));
    }
    if (!dst_half || !seg_row_indices) {
        throw std::invalid_argument("tri_stream_k_bounded: null dst or row indices");
    }
    if (tile_size == 0) tile_size = 32;
    const uint32_t head_dim = slice.head_dim;
    const uint32_t rotary_dim = slice.rotary_dim;
    if (head_dim == 0 || rotary_dim == 0 || rotary_dim > head_dim) {
        throw std::invalid_argument("tri_stream_k_bounded: invalid slice dimensions");
    }
    if (slice.rope_style != 0) {
        throw std::invalid_argument("tri_stream_k_bounded: slice rope_style must be 0 (half layout)");
    }
    if (kv_head >= slice.n_kv_heads && slice.n_kv_heads > 0) {
        throw std::out_of_range("tri_stream_k_bounded: kv_head out of range for slice");
    }
    if (slice.feature_offset_k > UINT32_MAX - (kv_head + 1) * head_dim) {
        throw std::overflow_error("tri_stream_k_bounded: feature offset overflow");
    }
    const uint32_t head_feat_start = slice.feature_offset_k + kv_head * head_dim;
    if (head_feat_start + head_dim > slice.feature_offset_k + slice.feature_dim_k) {
        throw std::out_of_range("tri_stream_k_bounded: head span exceeds layer feature_dim_k");
    }
    if ((uint64_t)n_rows > UINT64_MAX / head_dim) {
        throw std::overflow_error("tri_stream_k_bounded: n_rows * head_dim overflow");
    }
    if (dst_capacity_elements < (size_t)n_rows * head_dim) {
        throw std::invalid_argument("tri_stream_k_bounded: dst capacity too small");
    }
    if (!g->b_k) throw std::invalid_argument("tri_stream_k_bounded: group b_k is null");
    const encoded_matrix & b_k_ref = *g->b_k;
    if (b_k_ref.desc.orient != orientation::feature_major_transposed) {
        throw std::invalid_argument("tri_stream_k_bounded: b_k must be feature_major_transposed");
    }
    if (head_feat_start + head_dim > b_k_ref.desc.logical_shape.rows) {
        throw std::out_of_range("tri_stream_k_bounded: head span exceeds b_k logical rows");
    }
    const uint64_t b_pad_cols = b_k_ref.desc.padded_shape.cols;
    const uint64_t rank_k = b_k_ref.desc.logical_shape.cols;
    const uint64_t a_pad_cols = g->a_k.desc.padded_shape.cols;
    if (brow_cap < head_dim || bhead_cap < (size_t)head_dim * b_pad_cols ||
        tilea_cap < (size_t)tile_size * a_pad_cols || trow_cap < tile_size) {
        throw std::runtime_error("workspace: bounded K stream buffers too small for descriptors");
    }
    for (uint32_t d = 0; d < head_dim; ++d) brow_buf[d] = head_feat_start + d;
    decode_rows(b_k_ref, brow_buf, head_dim, bhead_buf, (size_t)head_dim * b_pad_cols, dtmp,
                 dtmp_bytes, value_domain::canonical);
    for (uint32_t r = 0; r < n_rows; ++r) {
        const uint32_t row_idx = seg_row_indices[r];
        if (row_idx >= segment.n_rows) {
            throw std::out_of_range("tri_stream_k_bounded: row index out of bounds");
        }
        if (row_idx < segment.live_rows.size() && !segment.live_rows[row_idx]) {
            throw std::runtime_error("tri_stream_k_bounded: requested row is dead");
        }
    }
    for (uint32_t ts = 0; ts < n_rows; ts += tile_size) {
        const uint32_t cur = std::min(tile_size, n_rows - ts);
        for (uint32_t r = 0; r < cur; ++r) trow_buf[r] = seg_row_indices[ts + r];
        decode_rows(g->a_k, trow_buf, cur, tilea_buf, (size_t)cur * a_pad_cols, dtmp,
                     dtmp_bytes, value_domain::canonical);
        for (uint32_t r = 0; r < cur; ++r) {
            const float * a_row = tilea_buf + (size_t)r * a_pad_cols;
            float * out_row = dst_half + (size_t)(ts + r) * head_dim;
            for (uint32_t d = 0; d < head_dim; ++d) {
                const float * b_row = bhead_buf + (size_t)d * b_pad_cols;
                float sum = 0.0f;
                for (uint64_t k = 0; k < rank_k; ++k) sum += a_row[k] * b_row[k];
                out_row[d] = sum;
            }
        }
    }
}

// Carved scoring core: sorted contiguous spans + CSR head-group runs.
// All temporaries come from `bump`; no map/set/vector growth here.
// `a_pad`/`b_pad` must match the geometry the caller sized (see tri_max_pads).
static void tri_score_carved(
    const triattention_calibration & cal,
    const float * omega,
    const float * fsq,
    const float * offsets,
    uint32_t n_offsets,
    const xkv_tri_config & config,
    const std::vector<xkv_tri_candidate> & candidates,
    const uint32_t * subset,
    uint32_t n_sub,
    const std::vector<xkv_layer_slice> & slices,
    const llama_xkv_cache_store & store,
    int64_t seq_frontier_position,
    float * out_pooled_scores,
    float * out_raw_combined,
    const hot_k_provider_fn & hot_provider,
    const device_factor_scoring_fn & device_scoring,
    const hot_k_span_provider_fn & hot_span,
    uint32_t a_pad,
    uint32_t b_pad,
    size_t tmp_bytes,
    const xkv_tri_selected_k_fetch_fn & fetch,
    tri_bump & bump) {
    const uint32_t fc = cal.freq_count;
    const uint32_t hd = cal.head_dim;
    const uint32_t tile_size = config.tile_size > 0 ? config.tile_size : 32;
    const uint32_t n_sampled = cal.n_sampled;

    float * raw_combined = tri_carve<float>(bump, n_sub);
    int32_t * sub_positions = tri_carve<int32_t>(bump, n_sub);
    if (!raw_combined || !sub_positions) throw std::runtime_error("workspace: score scratch carve overflow");
    for (uint32_t i = 0; i < n_sub; ++i) {
        const uint32_t cand_idx = subset[i];
        if (cand_idx >= candidates.size()) {
            throw std::out_of_range("score_candidate_subset: subset index out of bounds");
        }
        sub_positions[i] = candidates[cand_idx].storage_pos;
        raw_combined[i] = (config.head_agg == TRIATTENTION_AGG_MAX) ? -1e30f : 0.0f;
    }

    // Sorted (layer, KV-head) group table: keys + member order + CSR bounds.
    uint64_t * gkeys = tri_carve<uint64_t>(bump, n_sampled);
    uint32_t * gorder = tri_carve<uint32_t>(bump, n_sampled);
    uint32_t * gbounds = tri_carve<uint32_t>(bump, n_sampled + 1);
    float * gscores = tri_carve<float>(bump, (size_t)n_sampled * n_sub);
    if (!gkeys || !gorder || !gbounds || !gscores) {
        throw std::runtime_error("workspace: score scratch carve overflow");
    }
    for (uint32_t sh = 0; sh < n_sampled; ++sh) {
        const uint32_t ml = cal.sampled_layer[sh];
        const uint32_t qh = cal.sampled_head[sh];
        const uint32_t kv = cal.num_kv_groups > 0 ? qh / cal.num_kv_groups : 0;
        gkeys[sh] = ((uint64_t)ml << 32) | (uint64_t)kv;
        gorder[sh] = sh;
    }
    std::sort(gorder, gorder + n_sampled, [&](uint32_t a, uint32_t b) {
        if (gkeys[a] != gkeys[b]) return gkeys[a] < gkeys[b];
        return a < b;
    });
    uint32_t ngroups = 0;
    for (uint32_t i = 0; i < n_sampled; ) {
        uint32_t j = i + 1;
        while (j < n_sampled && gkeys[gorder[j]] == gkeys[gorder[i]]) ++j;
        gbounds[ngroups++] = i;
        i = j;
    }
    gbounds[ngroups] = n_sampled;

    // Tile buffers (fixed maxima, reused across groups).
    float * ktile = tri_carve<float>(bump, (size_t)tile_size * hd);
    float * bhead = tri_carve<float>(bump, (size_t)hd * b_pad);
    float * segk  = tri_carve<float>(bump, (size_t)tile_size * hd);
    float * hotk  = tri_carve<float>(bump, (size_t)tile_size * hd);
    float * tilea = tri_carve<float>(bump, (size_t)tile_size * a_pad);
    uint64_t * brow = tri_carve<uint64_t>(bump, hd);
    uint64_t * htup = tri_carve<uint64_t>(bump, (size_t)tile_size * 4);
    uint64_t * dtup = tri_carve<uint64_t>(bump, (size_t)tile_size * 4);
    uint32_t * hord = tri_carve<uint32_t>(bump, tile_size);
    uint32_t * dord = tri_carve<uint32_t>(bump, tile_size);
    uint32_t * hpos = tri_carve<uint32_t>(bump, tile_size);
    uint32_t * hcell = tri_carve<uint32_t>(bump, tile_size);
    uint32_t * srows = tri_carve<uint32_t>(bump, tile_size);
    float * devtmp = tri_carve<float>(bump, tile_size);
    uint64_t * trow = tri_carve<uint64_t>(bump, tile_size);
    uint8_t * dtmp = tri_carve<uint8_t>(bump, tmp_bytes);
    uint32_t * pord = tri_carve<uint32_t>(bump, n_sub);
    int32_t * dpos = tri_carve<int32_t>(bump, tile_size);
    if (!ktile || !bhead || !segk || !hotk || !tilea || !brow || !htup || !dtup ||
        !hord || !dord || !hpos || !hcell || !srows || !devtmp || !trow || !pord || !dpos ||
        (tmp_bytes > 0 && !dtmp)) {
        throw std::runtime_error("workspace: score scratch carve overflow");
    }

    for (uint32_t gi = 0; gi < ngroups; ++gi) {
        const uint32_t run0 = gbounds[gi];
        const uint32_t run1 = gbounds[gi + 1];
        const uint32_t run_len = run1 - run0;
        const uint32_t sh0 = gorder[run0];
        const uint32_t ml = cal.sampled_layer[sh0];
        const uint32_t qh0 = cal.sampled_head[sh0];
        const uint32_t kv_h = cal.num_kv_groups > 0 ? qh0 / cal.num_kv_groups : 0;

        const xkv_layer_slice * slice_p = tri_find_slice(slices, ml);
        if (!slice_p) {
            throw std::runtime_error("score_candidate_subset: missing required layer slice for model_layer " +
                                     std::to_string(ml));
        }
        const xkv_layer_slice & slice = *slice_p;
        for (uint32_t k = 0; k < run_len * n_sub; ++k) gscores[k] = 0.0f;

        for (uint32_t tile_start = 0; tile_start < n_sub; tile_start += tile_size) {
            const uint32_t cur_tile = std::min(tile_size, n_sub - tile_start);
            uint32_t nh = 0, nht = 0, ndt = 0;
            for (uint32_t t = 0; t < cur_tile; ++t) {
                const uint32_t cand_idx = subset[tile_start + t];
                const auto & c = candidates[cand_idx];
                if (c.device_owned) {
                    dtup[ndt * 4 + 0] = c.location.segment_id;
                    dtup[ndt * 4 + 1] = c.segment_version;
                    dtup[ndt * 4 + 2] = t;
                    dtup[ndt * 4 + 3] = c.location.row;
                    ++ndt;
                } else if (c.location.kind == xkv_location_kind::factored && c.location.segment_id > 0) {
                    htup[nht * 4 + 0] = c.location.segment_id;
                    htup[nht * 4 + 1] = c.segment_version;
                    htup[nht * 4 + 2] = t;
                    htup[nht * 4 + 3] = c.location.row;
                    ++nht;
                } else if (c.location.kind == xkv_location_kind::flat_quantized) {
                    throw std::runtime_error(
                        "score_candidate_subset: flat_quantized candidates unsupported by hot provider");
                } else {
                    hpos[nh] = t;
                    hcell[nh] = c.cell_index;
                    ++nh;
                }
            }

            // Host hot rows: one batched provider call per (layer, KV-head).
            if (nh > 0) {
                if (hot_span) {
                    // Zero-heap span path (production).
                    if (!hot_span(ml, kv_h, hcell, nh, hotk, (size_t)nh * hd)) {
                        throw std::runtime_error("score_candidate_subset: hot provider failed to read canonical K");
                    }
                } else {
                    if (!hot_provider) {
                        throw std::runtime_error(
                            "score_candidate_subset: missing required hot_provider for hot candidates");
                    }
                    // Legacy vector adaptation (one bounded allocation, documented).
                    const std::vector<uint32_t> cells_tmp(hcell, hcell + nh);
                    if (!hot_provider(ml, kv_h, cells_tmp, hotk, (size_t)nh * hd)) {
                        throw std::runtime_error("score_candidate_subset: hot_provider failed to read canonical K");
                    }
                }
                for (uint32_t k = 0; k < nh; ++k) {
                    std::memcpy(ktile + (size_t)hpos[k] * hd, hotk + (size_t)k * hd, hd * sizeof(float));
                }
            }

            // Host factored rows: sort tuple order by (seg_id, seg_ver), CSR runs.
            for (uint32_t k = 0; k < nht; ++k) hord[k] = k;
            for (uint32_t k = 1; k < nht; ++k) {
                const uint32_t v = hord[k];
                uint32_t j = k;
                while (j > 0) {
                    const uint32_t u = hord[j - 1];
                    const bool less =
                        (htup[(size_t)v * 4] < htup[(size_t)u * 4]) ||
                        (htup[(size_t)v * 4] == htup[(size_t)u * 4] &&
                         (htup[(size_t)v * 4 + 1] < htup[(size_t)u * 4 + 1] ||
                          (htup[(size_t)v * 4 + 1] == htup[(size_t)u * 4 + 1] && v < u)));
                    if (!less) break;
                    hord[j] = u;
                    --j;
                }
                hord[j] = v;
            }
            for (uint32_t r0 = 0; r0 < nht; ) {
                uint32_t r1 = r0 + 1;
                while (r1 < nht && htup[(size_t)hord[r1] * 4] == htup[(size_t)hord[r0] * 4] &&
                       htup[(size_t)hord[r1] * 4 + 1] == htup[(size_t)hord[r0] * 4 + 1]) {
                    ++r1;
                }
                const uint64_t seg_id = htup[(size_t)hord[r0] * 4];
                const uint64_t seg_ver = htup[(size_t)hord[r0] * 4 + 1];
                std::shared_ptr<const xkv_segment> seg_handle =
                    seg_ver > 0 ? store.get_segment_version(seg_id, seg_ver) : store.get_segment(seg_id);
                if (!seg_handle) {
                    throw std::runtime_error("score_candidate_subset: failed to pin segment " +
                                             std::to_string(seg_id));
                }
                xkv_reader_pin pin(seg_handle);
                const uint32_t rn = r1 - r0;
                for (uint32_t k = 0; k < rn; ++k) {
                    srows[k] = (uint32_t)htup[(size_t)hord[r0 + k] * 4 + 3];
                }
                tri_stream_k_bounded(*pin, slice, kv_h, srows, rn, segk, (size_t)rn * hd, tilea,
                                     (size_t)tile_size * a_pad, bhead, (size_t)hd * b_pad, brow, hd,
                                     trow, tile_size, dtmp, tri_round64(tmp_bytes), tile_size);
                for (uint32_t k = 0; k < rn; ++k) {
                    const uint32_t t = (uint32_t)htup[(size_t)hord[r0 + k] * 4 + 2];
                    std::memcpy(ktile + (size_t)t * hd, segk + (size_t)k * hd, hd * sizeof(float));
                }
                r0 = r1;
            }

            // Zero device-owned tile slots: host scoring must not read stale K.
            for (uint32_t k = 0; k < ndt; ++k) {
                std::memset(ktile + (size_t)dtup[(size_t)k * 4 + 2] * hd, 0, hd * sizeof(float));
            }

            // One K decode reused across every sampled Q head in the group.
            for (uint32_t hh = 0; hh < run_len; ++hh) {
                const uint32_t sh = gorder[run0 + hh];
                float * out_h_scores = gscores + (size_t)hh * n_sub + tile_start;
                triattention_score_keys(out_h_scores, ktile, &cal.head_stats[sh], omega, fsq, offsets,
                                        sub_positions + tile_start, seq_frontier_position, cur_tile, hd, fc,
                                        n_offsets, config.agg, config.disable_trig);
            }

            // Device-owned rows: native dispatch per segment run and head; never host-decoded.
            for (uint32_t k = 0; k < ndt; ++k) dord[k] = k;
            for (uint32_t k = 1; k < ndt; ++k) {
                const uint32_t v = dord[k];
                uint32_t j = k;
                while (j > 0) {
                    const uint32_t u = dord[j - 1];
                    const bool less =
                        (dtup[(size_t)v * 4] < dtup[(size_t)u * 4]) ||
                        (dtup[(size_t)v * 4] == dtup[(size_t)u * 4] &&
                         (dtup[(size_t)v * 4 + 1] < dtup[(size_t)u * 4 + 1] ||
                          (dtup[(size_t)v * 4 + 1] == dtup[(size_t)u * 4 + 1] && v < u)));
                    if (!less) break;
                    dord[j] = u;
                    --j;
                }
                dord[j] = v;
            }
            for (uint32_t r0 = 0; r0 < ndt; ) {
                uint32_t r1 = r0 + 1;
                while (r1 < ndt && dtup[(size_t)dord[r1] * 4] == dtup[(size_t)dord[r0] * 4] &&
                       dtup[(size_t)dord[r1] * 4 + 1] == dtup[(size_t)dord[r0] * 4 + 1]) {
                    ++r1;
                }
                if (!device_scoring && !fetch) {
                    throw xkv_device_scoring_error(
                        "score_candidate_subset: device-owned factors require a native device scoring callback");
                }
                const uint64_t seg_id = dtup[(size_t)dord[r0] * 4];
                const uint64_t seg_ver = dtup[(size_t)dord[r0] * 4 + 1];
                const uint32_t rn = r1 - r0;
                for (uint32_t k = 0; k < rn; ++k) {
                    srows[k] = (uint32_t)dtup[(size_t)dord[r0 + k] * 4 + 3];
                }
                if (fetch) {
                    // Batched native path: ONE fetch covers every selected row
                    // of the run; each sampled head then scores the fetched K
                    // locally (sampled-head reuse, no per-head round trips).
                    std::string ferr;
                    if (!fetch(seg_id, seg_ver, srows, rn, slice, kv_h, segk, (size_t)rn * hd,
                               &ferr)) {
                        throw xkv_device_scoring_error(
                            "score_candidate_subset: device fetch failed: " + ferr);
                    }
                    for (uint32_t k = 0; k < rn; ++k) {
                        const uint32_t t = (uint32_t)dtup[(size_t)dord[r0 + k] * 4 + 2];
                        dpos[k] = sub_positions[tile_start + t];
                    }
                    for (uint32_t hh = 0; hh < run_len; ++hh) {
                        const uint32_t sh = gorder[run0 + hh];
                        triattention_score_keys(devtmp, segk, &cal.head_stats[sh], omega, fsq,
                                                offsets, dpos, seq_frontier_position, rn, hd, fc,
                                                n_offsets, config.agg, config.disable_trig);
                        for (uint32_t k = 0; k < rn; ++k) {
                            const uint32_t t = (uint32_t)dtup[(size_t)dord[r0 + k] * 4 + 2];
                            gscores[(size_t)hh * n_sub + tile_start + t] = devtmp[k];
                        }
                    }
                    r0 = r1;
                    continue;
                }
                for (uint32_t hh = 0; hh < run_len; ++hh) {
                    const uint32_t sh = gorder[run0 + hh];
                    std::string derr;
                    if (!device_scoring(seg_id, seg_ver, srows, rn, slice, sh, seq_frontier_position,
                                        devtmp, &derr)) {
                        throw xkv_device_scoring_error(
                            "score_candidate_subset: device scoring callback failed: " + derr);
                    }
                    for (uint32_t k = 0; k < rn; ++k) {
                        const uint32_t t = (uint32_t)dtup[(size_t)dord[r0 + k] * 4 + 2];
                        gscores[(size_t)hh * n_sub + tile_start + t] = devtmp[k];
                    }
                }
                r0 = r1;
            }
        }

        for (uint32_t hh = 0; hh < run_len; ++hh) {
            float * scores = gscores + (size_t)hh * n_sub;
            if (config.normalize_scores) {
                zscore_normalize_subset(scores, n_sub);
            }
            for (uint32_t i = 0; i < n_sub; ++i) {
                if (config.head_agg == TRIATTENTION_AGG_MAX) {
                    raw_combined[i] = std::max(raw_combined[i], scores[i]);
                } else {
                    raw_combined[i] += scores[i] / (float)cal.n_sampled;
                }
            }
        }
    }

    if (config.head_agg == TRIATTENTION_AGG_MAX) {
        for (uint32_t i = 0; i < n_sub; ++i) {
            if (raw_combined[i] <= -1e29f) raw_combined[i] = 0.0f;
        }
    }

    if (out_raw_combined) {
        std::memcpy(out_raw_combined, raw_combined, (size_t)n_sub * sizeof(float));
    }

    if (out_pooled_scores) {
        // Local clustering, carved: identical algorithm to
        // triattention_max_pool_scores (which heap-allocates its order
        // vector), so the arena path performs zero heap allocations.
        for (uint32_t i = 0; i < n_sub; ++i) pord[i] = i;
        std::sort(pord, pord + n_sub,
                    [&](uint32_t a, uint32_t b) { return sub_positions[a] < sub_positions[b]; });
        for (uint32_t rank = 0; rank < n_sub; ++rank) {
            const uint32_t candidate = pord[rank];
            const int64_t center = sub_positions[candidate];
            float best = raw_combined[candidate];
            for (uint32_t left = rank; left > 0; --left) {
                const uint32_t neighbor = pord[left - 1];
                if (center - sub_positions[neighbor] > config.pool_radius) break;
                best = std::max(best, raw_combined[neighbor]);
            }
            for (uint32_t right = rank + 1; right < n_sub; ++right) {
                const uint32_t neighbor = pord[right];
                if ((int64_t)sub_positions[neighbor] - center > config.pool_radius) break;
                best = std::max(best, raw_combined[neighbor]);
            }
            out_pooled_scores[candidate] = best;
        }
    }
}

// ============================================================================
// build_mutation_plan
// ============================================================================

xkv_tri_mutation_plan xkv_tri_adapter::build_mutation_plan(
    const std::vector<xkv_tri_candidate> & candidates,
    const std::vector<xkv_tri_seq_info> & seqs,
    const std::vector<xkv_layer_slice> & slices,
    const llama_xkv_cache_store & store,
    const xkv_tri_pressure_state & pressure,
    const hot_k_provider_fn & hot_provider,
    const xkv_tri_frontier_table * explicit_frontiers,
    const device_factor_scoring_fn & device_scoring,
    const hot_k_span_provider_fn & hot_span,
    xkv_arena_lease * scratch_lease,
    const xkv_tri_selected_k_fetch_fn & fetch
) const {
    xkv_tri_mutation_plan plan;
    const uint32_t n_cand = (uint32_t)candidates.size();
    plan.total_candidates = n_cand;
    plan.physical_before = n_cand;

    if (n_cand == 0) {
        return plan;
    }

    // Check pressure gate: Tri reclaims KV only
    if (!tri_rerot_should_reclaim(pressure.kv_pressure, pressure.recurrent_pressure, pressure.maintenance_due)) {
        if (pressure.recurrent_pressure && !pressure.kv_pressure) {
            plan.recurrent_only_bypass = true;
        }
        for (uint32_t i = 0; i < n_cand; ++i) {
            plan.survivor_payloads.push_back(candidates[i].payload_id);
            plan.survivor_cells.push_back(candidates[i].cell_index);
        }
        plan.physical_after = n_cand;
        return plan;
    }

    // Scratch sizing: build regions + worst-case nested score geometry.
    const uint32_t hd = cal.head_dim;
    const uint32_t tile_size = config.tile_size > 0 ? config.tile_size : 32;
    uint32_t a_pad = 0, b_pad = 0;
    size_t tmp_bytes = 0;
    tri_max_pads(&store, candidates.data(), n_cand, nullptr, 0, false, slices, hd, a_pad, b_pad,
                 tmp_bytes);
    const tri_score_geom sg =
        tri_score_geometry(n_cand, tile_size, hd, a_pad, b_pad, tmp_bytes, cal.n_sampled);
    const size_t refbits_bytes = tri_round64((size_t)n_cand * TRI_REF_WORDS * sizeof(uint64_t));
    const size_t seqidx_bytes = tri_round64((size_t)n_cand * sizeof(uint32_t));
    const size_t pooled_bytes = tri_round64((size_t)n_cand * sizeof(float));
    const size_t order_bytes = tri_round64((size_t)n_cand * sizeof(uint32_t));
    const size_t pload_bytes = tri_round64((size_t)n_cand * sizeof(uint64_t));
    const size_t rowref_bytes = tri_round64((size_t)n_cand * sizeof(segment_row_ref));
    const size_t aff_bytes = tri_round64((size_t)n_cand * sizeof(uint64_t));
    const size_t pidk_bytes = tri_round64((size_t)n_cand * sizeof(uint64_t));
    const size_t pidv_bytes = tri_round64((size_t)n_cand * sizeof(uint32_t));
    const size_t need = refbits_bytes + seqidx_bytes + pooled_bytes + order_bytes + pload_bytes +
                        rowref_bytes + aff_bytes + pidk_bytes + pidv_bytes + sg.total;
    std::vector<uint8_t> heap_scratch;
    tri_bump bump;
    if (scratch_lease && scratch_lease->valid() && scratch_lease->size() >= need) {
        bump.base = static_cast<uint8_t *>(scratch_lease->data());
        bump.cap = scratch_lease->size();
    } else if (scratch_lease && scratch_lease->valid()) {
        throw std::runtime_error("workspace: build scratch lease one-byte-short");
    } else {
        heap_scratch.resize(need);
        bump.base = heap_scratch.data();
        bump.cap = heap_scratch.size();
    }

    tri_build_carved(cal, omega.data(), freq_scale_sq.data(), offsets.data(),
                       (uint32_t)offsets.size(), config, candidates, seqs, slices, store, pressure,
                       hot_provider, explicit_frontiers, device_scoring, hot_span, a_pad, b_pad,
                       tmp_bytes, fetch, plan, bump);
    if (bump.overflow) {
        throw std::runtime_error("workspace: build scratch carve overflow");
    }
    return plan;
}

// Carved planning core: keeper refs as bitsets, dedup/union via sorted spans.
static void tri_build_carved(
    const triattention_calibration & cal,
    const float * omega,
    const float * fsq,
    const float * offsets,
    uint32_t n_offsets,
    const xkv_tri_config & config,
    const std::vector<xkv_tri_candidate> & candidates,
    const std::vector<xkv_tri_seq_info> & seqs,
    const std::vector<xkv_layer_slice> & slices,
    const llama_xkv_cache_store & store,
    const xkv_tri_pressure_state & pressure,
    const hot_k_provider_fn & hot_provider,
    const xkv_tri_frontier_table * explicit_frontiers,
    const device_factor_scoring_fn & device_scoring,
    const hot_k_span_provider_fn & hot_span,
    uint32_t a_pad,
    uint32_t b_pad,
    size_t tmp_bytes,
    const xkv_tri_selected_k_fetch_fn & fetch,
    xkv_tri_mutation_plan & plan,
    tri_bump & bump) {
    const uint32_t n_cand = (uint32_t)candidates.size();

    plan.total_candidates = n_cand;
    plan.physical_before = n_cand;
    plan.physical_after = n_cand;
    plan.capacity_satisfied = (pressure.required_units == 0);
    if (n_cand == 0) return;

    // Validate + dedup via sorted contiguous spans (no hash sets).
    uint64_t * pload = tri_carve<uint64_t>(bump, n_cand);
    segment_row_ref * rowrefs = tri_carve<segment_row_ref>(bump, n_cand);
    if (!pload || !rowrefs) throw std::runtime_error("workspace: build scratch carve overflow");
    uint32_t nfact = 0;
    for (uint32_t i = 0; i < n_cand; ++i) {
        const auto & c = candidates[i];
        if (c.payload_id == 0) {
            throw std::invalid_argument("build_mutation_plan: payload_id cannot be 0");
        }
        pload[i] = c.payload_id;
        if (c.location.kind == xkv_location_kind::factored) {
            if (c.row_ref.segment_id != c.location.segment_id || c.row_ref.row != c.location.row) {
                throw std::invalid_argument("build_mutation_plan: candidate row_ref and location mismatch");
            }
            rowrefs[nfact++] = c.row_ref;
        }
    }
    std::sort(pload, pload + n_cand);
    for (uint32_t i = 1; i < n_cand; ++i) {
        if (pload[i] == pload[i - 1]) {
            throw std::invalid_argument("build_mutation_plan: duplicate payload_id in candidate set");
        }
    }
    std::sort(rowrefs, rowrefs + nfact);
    for (uint32_t i = 1; i < nfact; ++i) {
        if (!(rowrefs[i - 1] < rowrefs[i])) {
            throw std::invalid_argument("build_mutation_plan: duplicate segment_row_ref in candidate set");
        }
    }

    // Keeper refs: one LLAMA_MAX_SEQ bitset per candidate, carved contiguously.
    uint64_t * refbits = tri_carve<uint64_t>(bump, (size_t)n_cand * TRI_REF_WORDS);
    if (!refbits) throw std::runtime_error("workspace: build scratch carve overflow");
    for (uint32_t i = 0; i < n_cand; ++i) {
        uint64_t * bits = refbits + (size_t)i * TRI_REF_WORDS;
        for (uint32_t w = 0; w < TRI_REF_WORDS; ++w) bits[w] = 0;
        for (llama_seq_id s : candidates[i].seq_ids) {
            if (s >= 0 && (uint64_t)s < (uint64_t)LLAMA_MAX_SEQ) {
                bits[(uint32_t)s >> 6] |= (1ull << ((uint32_t)s & 63u));
            }
        }
    }

    uint32_t * seqidx = tri_carve<uint32_t>(bump, n_cand);
    float * pooled = tri_carve<float>(bump, n_cand);
    uint32_t * order = tri_carve<uint32_t>(bump, n_cand);
    if (!seqidx || !pooled || !order) throw std::runtime_error("workspace: build scratch carve overflow");

    // Match existing reclaim loop exactly:
    // For each sequence:
    //   1. Build its own candidate subset
    //   2. Compute seq target = max(tail_guard, ceil(L * ratio))
    //   3. Protect recent (pos >= frontier - tail_guard + 1), pending, foreign, and semantic-reader-tail
    //   4. Score with that seq.frontier_pos and local-pool within that subset
    //   5. Choose references to evict and emit exact (payload_id, seq_id) reference removals
    for (const auto & seq : seqs) {
        if (!seq.eligible) {
            continue;
        }

        const llama_seq_id sid = seq.seq_id;
        const uint32_t tail_guard = seq.tail_guard > 0 ? seq.tail_guard : config.recent_window;
        const uint32_t logical_L = seq.logical_tokens > 0 ? seq.logical_tokens : (uint32_t)std::max<int64_t>(0, seq.frontier_pos + 1);

        const uint32_t seq_target = tri_rerot_target_retention(logical_L, config.ratio, tail_guard);
        plan.target_references += seq_target;

        const int64_t seq_recent_threshold = (seq.frontier_pos >= (int64_t)tail_guard)
            ? (seq.frontier_pos - (int64_t)tail_guard + 1)
            : 0;

        uint32_t seq_protected_count = 0;
        uint32_t n_scorable = 0; // seqidx[0..n_scorable) after the loop

        for (uint32_t i = 0; i < n_cand; ++i) {
            uint64_t * bits = refbits + (size_t)i * TRI_REF_WORDS;
            if (!tri_ref_test(bits, sid)) {
                continue;
            }

            const auto & c = candidates[i];
            const bool is_recent = (c.storage_pos >= seq_recent_threshold);

            // Existing parity guards from llama-kv-cache.cpp:
            // 1. Pending structural record guard
            const bool rerot_pending = c.is_protected || (c.visibility == llama_rerot_visibility::pending_record);

            // 2. Semantic foreign tag guard
            const bool semantic_foreign_tag = (seq.semantic_episode_id != 0 &&
                                               c.semantic_episode_id != 0 &&
                                               (c.semantic_episode_id != seq.semantic_episode_id ||
                                                c.visibility != llama_rerot_visibility::public_live));

            // 3. Semantic reader tail HARD GUARD (semantic_seq_ids are reader-tail protection inputs)
            // Unknown reader frontier => preserve/fail closed: the tail guard is
            // never dropped for a sequence we cannot place. An explicit frontier
            // table may supply frontiers for sequences outside `seqs`.
            bool semantic_reader_tail = false;
            if (seq.semantic_episode_id != 0) {
                for (const llama_seq_id ref : seq.semantic_seq_ids) {
                    if (ref == sid || !tri_ref_test(bits, ref)) {
                        continue;
                    }
                    // Each referenced seq contributes its OWN frontier/threshold.
                    int64_t other_frontier = 0;
                    uint32_t other_guard = config.recent_window;
                    if (!tri_lookup_frontier(seqs, explicit_frontiers, config, ref, other_frontier,
                                             other_guard)) {
                        semantic_reader_tail = true; // unknown frontier: preserve
                        break;
                    }
                    const int64_t other_recent_threshold =
                        (other_frontier >= (int64_t)other_guard) ? (other_frontier - (int64_t)other_guard + 1)
                                : 0;
                            if (c.storage_pos >= other_recent_threshold) {
                                semantic_reader_tail = true;
                        break;
                    }
                }
            }

            if (is_recent || rerot_pending || semantic_foreign_tag || semantic_reader_tail) {
                seq_protected_count++;
            } else {
                seqidx[n_scorable++] = i;
            }
        }

        plan.hard_keep += seq_protected_count;

        const uint32_t total_seq_cells = seq_protected_count + n_scorable;

        if (total_seq_cells <= seq_target || n_scorable == 0) {
            continue;
        }

        const uint32_t to_keep = (seq_target > seq_protected_count)
            ? std::min(seq_target - seq_protected_count, n_scorable)
            : 0;

        // Score this sequence's candidate subset with THAT sequence's frontier_pos!
        // Nested carved scoring reuses the same bump (save/restore around it).
        float * seq_pooled_scores = pooled;
        const size_t score_mark = bump.mark();
        tri_score_carved(cal, omega, fsq, offsets, n_offsets, config, candidates, seqidx, n_scorable,
                           slices, store, seq.frontier_pos, seq_pooled_scores, nullptr, hot_provider,
                           device_scoring, hot_span, a_pad, b_pad, tmp_bytes, fetch, bump);
        bump.reset_to(score_mark);
        if (bump.overflow) throw std::runtime_error("workspace: build scratch carve overflow");

        for (uint32_t k = 0; k < n_scorable; ++k) order[k] = k;

        if (to_keep < n_scorable) {
            if (to_keep > 0) {
                std::partial_sort(
                    order,
                    order + to_keep,
                    order + n_scorable,
                    [&](uint32_t a, uint32_t b) {
                        // Deterministic sorting with secondary pos tie-breaking and tertiary cell_index tie-breaking
                        if (pooled[a] != pooled[b]) {
                            return pooled[a] > pooled[b];
                        }
                        if (candidates[seqidx[a]].storage_pos != candidates[seqidx[b]].storage_pos) {
                            return candidates[seqidx[a]].storage_pos > candidates[seqidx[b]].storage_pos;
                        }
                        return candidates[seqidx[a]].cell_index < candidates[seqidx[b]].cell_index;
                    }
                );
            }
            // Evict reference from candidates in order[to_keep .. n_scorable - 1]
            for (uint32_t k = to_keep; k < n_scorable; ++k) {
                const uint32_t cand_idx = seqidx[order[k]];
                const auto & c = candidates[cand_idx];
                uint64_t * bits = refbits + (size_t)cand_idx * TRI_REF_WORDS;

                // Mirror legacy llama_kv_cache::reclaim_kv semantic-cell behavior exactly:
                // A semantic PUBLIC/base cell (episode_id matches and visibility is public_live, or active is false)
                // represents one semantic base/PUBLIC cell regardless of archive/exec bookkeeping refs.
                // Remove ALL present refs in seq.semantic_seq_ids for that episode, not only current sid;
                // do not remove foreign episode refs.
                const bool semantic_cell = (seq.semantic_episode_id != 0 &&
                                            (c.semantic_episode_id == 0 ||
                                             (c.semantic_episode_id == seq.semantic_episode_id &&
                                              c.visibility == llama_rerot_visibility::public_live)));

                if (semantic_cell) {
                    for (const llama_seq_id ref : seq.semantic_seq_ids) {
                        if (tri_ref_test(bits, ref)) {
                xkv_tri_ref_removal rem;
                rem.payload_id = c.payload_id;
                            rem.seq_id = ref;
                rem.cell_index = c.cell_index;
                plan.ref_removals.push_back(rem);

                            tri_ref_clear(bits, ref);
                plan.references_removed++;
            }
            }
                } else {
                xkv_tri_ref_removal rem;
                rem.payload_id = c.payload_id;
                rem.seq_id = sid;
                rem.cell_index = c.cell_index;
                plan.ref_removals.push_back(rem);

                tri_ref_clear(bits, sid);
                plan.references_removed++;
            }
            }
        }
    }

    // Union physical survivors: a cell survives if ANY keeper reference remains.
    // Affected segments via sorted span dedup; resource outcomes per pressure.
    uint64_t * aff = tri_carve<uint64_t>(bump, n_cand);
    if (!aff) throw std::runtime_error("workspace: build scratch carve overflow");
    uint32_t naff = 0;

    for (uint32_t i = 0; i < n_cand; ++i) {
        const auto & c = candidates[i];
        const uint64_t * bits = refbits + (size_t)i * TRI_REF_WORDS;
        if (!tri_ref_empty(bits)) {
            plan.survivor_payloads.push_back(c.payload_id);
            plan.survivor_cells.push_back(c.cell_index);
            if (tri_ref_count(bits) > 1) {
                plan.shared_keep++;
            }
        } else {
            plan.evicted_payloads.push_back(c.payload_id);
            plan.evicted_cells.push_back(c.cell_index);
            plan.physical_freed++;

            if (c.location.kind == xkv_location_kind::hot) {
                plan.hot_slots_freed++;
            } else if (c.location.kind == xkv_location_kind::factored) {
                plan.factored_rows_freed++;
                if (c.location.segment_id > 0) {
                    aff[naff++] = c.location.segment_id;
                    auto seg = store.get_segment(c.location.segment_id);
                    if (seg) plan.factored_bytes_reclaimed += tri_factored_row_bytes(*seg);
                }
            }
        }
    }

    std::sort(aff, aff + naff);
    naff = (uint32_t)(std::unique(aff, aff + naff) - aff);
    plan.affected_segments.assign(aff, aff + naff);

    // capacity_satisfied compares required units of the pressured resource only.
    switch (pressure.resource) {
        case xkv_tri_pressure_resource::hot_slots:
            plan.capacity_satisfied = plan.hot_slots_freed >= pressure.required_units;
            break;
        case xkv_tri_pressure_resource::store_bytes:
            plan.capacity_satisfied = plan.factored_bytes_reclaimed >= pressure.required_units;
            break;
        case xkv_tri_pressure_resource::legacy_payloads:
        default:
            plan.capacity_satisfied = plan.physical_freed >= pressure.required_units;
            break;
    }

    plan.physical_after = plan.physical_before - plan.physical_freed;
    // changed is true if ANY reference was removed, even when physical_freed == 0!
    plan.changed = (plan.references_removed > 0);
}

// ============================================================================
// create_mutation_proposal & xkv_store_mutation_proposal
// ============================================================================

xkv_store_mutation_proposal xkv_tri_adapter::create_mutation_proposal(
    const xkv_tri_mutation_plan & plan,
    const llama_xkv_cache_store & store,
    xkv_landmark_rebuild_fn landmark_rebuild
) const {
    xkv_store_mutation_proposal prop;
    prop.batch.payload_removals = plan.evicted_payloads;
    prop.batch.segments_to_pack = plan.affected_segments;
    prop.batch.landmark_rebuild = landmark_rebuild;
    prop.expected_stamp = store.current_stamp();
    return prop;
}

// ============================================================================
// enumerate_candidates_from_cells
// ============================================================================

std::vector<xkv_tri_candidate> enumerate_candidates_from_cells(
    const llama_kv_cells & cells,
    const llama_xkv_cache_store & store
) {
    std::vector<xkv_tri_candidate> result;
    const uint32_t n_cells = cells.size();
    result.reserve(cells.get_used());

    for (uint32_t i = 0; i < n_cells; ++i) {
        if (cells.is_empty(i)) {
            continue;
        }

        xkv_tri_candidate c;
        c.cell_index = i;
        c.storage_pos = (int32_t)cells.pos_get(i);
        c.payload_id = cells.payload_id_get(i);
        c.storage_generation = cells.storage_generation_get(i);

        // Collect sequences referencing this cell
        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (cells.seq_has(i, s)) {
                c.seq_ids.push_back(s);
            }
        }

        // Join with store location
        xkv_location loc;
        if (c.payload_id != 0 && store.find_location(c.payload_id, loc)) {
            c.location = loc;
            if (loc.kind == xkv_location_kind::factored) {
                c.row_ref.segment_id = loc.segment_id;
                c.row_ref.row = loc.row;
                auto seg = store.get_segment(loc.segment_id);
                if (seg) {
                    c.segment_version = seg->segment_version;
                }
            }
        } else {
            // Nonexistent in store or hot uncommitted: default to hot cell mapping
            c.location.kind = xkv_location_kind::hot;
            c.location.row = i;
        }

        const auto & meta = cells.rerot_get(i);
        if (meta.active()) {
            c.semantic_episode_id = (uint32_t)meta.episode_id;
            c.visibility = meta.visibility;
            if (meta.visibility == llama_rerot_visibility::pending_record) {
                c.is_protected = true;
            }
        }

        result.push_back(std::move(c));
    }

    return result;
}

bool xkv_tri_adapter::estimate_workspace_requirements(
    uint32_t n_candidates,
    const std::vector<xkv_tri_seq_info> & seqs,
    const std::vector<xkv_layer_slice> & slices,
    xkv_tri_workspace_requirements & out_reqs,
    std::string * err,
    const llama_xkv_cache_store * store,
    const std::vector<xkv_tri_candidate> * candidates
) const {
    (void)seqs;
    (void)err;
    out_reqs = {};
    if (n_candidates == 0) {
        return true;
    }

    const uint32_t tile_size = config.tile_size > 0 ? config.tile_size : 32;
    const uint32_t hd = cal.head_dim;

    // Actual descriptor widths when the store is available, else slice fallback.
    uint32_t a_pad = 0, b_pad = 0;
    size_t tmp_bytes = 0;
    tri_max_pads(store, candidates ? candidates->data() : nullptr,
                 candidates ? candidates->size() : n_candidates, nullptr, 0, false, slices, hd,
                 a_pad, b_pad, tmp_bytes);

    // Build regions (must match tri_build_carved carve order/sizes).
    const size_t refbits_bytes = tri_round64((size_t)n_candidates * TRI_REF_WORDS * sizeof(uint64_t));
    const size_t seqidx_bytes = tri_round64((size_t)n_candidates * sizeof(uint32_t));
    const size_t pooled_bytes = tri_round64((size_t)n_candidates * sizeof(float));
    const size_t order_bytes = tri_round64((size_t)n_candidates * sizeof(uint32_t));
    const size_t pload_bytes = tri_round64((size_t)n_candidates * sizeof(uint64_t));
    const size_t rowref_bytes = tri_round64((size_t)n_candidates * sizeof(segment_row_ref));
    const size_t aff_bytes = tri_round64((size_t)n_candidates * sizeof(uint64_t));
    // Proposal payload-id index entries (pid + candidate index, carved as one span).
    const size_t pidk_bytes = tri_round64((size_t)n_candidates * (sizeof(uint64_t) + sizeof(uint32_t)));
    const size_t pidv_bytes = 0;

    // Nested worst-case score geometry (n_sub = n_candidates).
    const tri_score_geom sg =
        tri_score_geometry(n_candidates, tile_size, hd, a_pad, b_pad, tmp_bytes, cal.n_sampled);

    out_reqs.candidate_array_bytes = refbits_bytes + pload_bytes + rowref_bytes;
    out_reqs.candidate_order_bytes = seqidx_bytes + pooled_bytes + order_bytes;
    out_reqs.score_buffers_bytes = sg.raw + sg.pos + sg.grp;
    out_reqs.tile_buffers_bytes = sg.ktile + sg.bhead + sg.segk + sg.hotk + sg.dtmp;
    out_reqs.tile_a_bytes = sg.tilea;
    out_reqs.union_maps_bytes =
        aff_bytes + pidk_bytes + pidv_bytes + sg.brow + sg.tup + sg.hotidx + sg.gtab + sg.trow +
        sg.pord + sg.srt;
    out_reqs.total_bytes = refbits_bytes + seqidx_bytes + pooled_bytes + order_bytes + pload_bytes +
                             rowref_bytes + aff_bytes + pidk_bytes + pidv_bytes + sg.total;

    return true;
}

xkv_tri_reclaim_proposal xkv_tri_adapter::create_reclaim_proposal(
    const std::vector<xkv_tri_candidate> & candidates,
    const std::vector<xkv_tri_seq_info> & seqs,
    const std::vector<xkv_layer_slice> & slices,
    const llama_xkv_cache_store & store,
    const xkv_tri_pressure_state & pressure,
    const xkv_tri_reclaim_options & options
) const {
    xkv_tri_reclaim_proposal prop;
    prop.expected_stamp = store.current_stamp();

    // 1. Recurrent pressure check: recurrent-only bypasses Tri reclaim entirely
    if (!tri_rerot_should_reclaim(pressure.kv_pressure, pressure.recurrent_pressure, pressure.maintenance_due)) {
        if (pressure.recurrent_pressure && !pressure.kv_pressure) {
            prop.status = xkv_tri_proposal_status::bypass_recurrent_only;
            prop.message = "recurrent-only pressure bypassed TriAttention reclaim";
            prop.plan.recurrent_only_bypass = true;
            for (const auto & c : candidates) {
                prop.plan.survivor_payloads.push_back(c.payload_id);
                prop.plan.survivor_cells.push_back(c.cell_index);
            }
            prop.plan.physical_before = (uint32_t)candidates.size();
            prop.plan.physical_after = prop.plan.physical_before;
            prop.plan.capacity_satisfied = true;
            prop.capacity_satisfied = true;
            return prop;
        } else {
            // Under idle or fill-first with no pressure: return explicit no-op success, NEVER floor_exhausted!
            prop.status = xkv_tri_proposal_status::success;
            prop.message = "fill-first or idle: no KV reclaim pressure";
            for (const auto & c : candidates) {
                prop.plan.survivor_payloads.push_back(c.payload_id);
                prop.plan.survivor_cells.push_back(c.cell_index);
            }
            prop.plan.physical_before = (uint32_t)candidates.size();
            prop.plan.physical_after = prop.plan.physical_before;
            prop.plan.capacity_satisfied = true;
            prop.capacity_satisfied = true;
            return prop;
        }
    }

    // 2. Workspace preflight checks
    xkv_tri_workspace_requirements reqs;
    estimate_workspace_requirements((uint32_t)candidates.size(), seqs, slices, reqs, nullptr, &store,
                                     &candidates);
    prop.required_workspace_bytes = reqs.total_bytes;
    prop.peak_workspace_bytes = reqs.total_bytes;

    size_t hard_budget = options.workspace_budget_bytes > 0 ? options.workspace_budget_bytes : config.workspace_budget_bytes;
    if (hard_budget > 0 && reqs.total_bytes > hard_budget) {
        prop.status = xkv_tri_proposal_status::workspace_exhausted;
        prop.message = "workspace budget exceeded: needed " + std::to_string(reqs.total_bytes) + " bytes, budget is " + std::to_string(hard_budget);
        return prop;
    }

    xkv_arena_lease lease;
    if (options.arena) {
        if (!options.arena->preflight(reqs.total_bytes)) {
            prop.status = xkv_tri_proposal_status::workspace_exhausted;
            prop.message = "workspace arena preflight failed for " + std::to_string(reqs.total_bytes) + " bytes";
            return prop;
        }
        lease = options.arena->acquire(reqs.total_bytes);
        if (!lease) {
            prop.status = xkv_tri_proposal_status::workspace_exhausted;
            prop.message = "failed to acquire workspace arena lease of " + std::to_string(reqs.total_bytes) + " bytes";
            return prop;
        }
    }

    // 3. Build mutation plan
    // One lease backs the entire pipeline: carved build (with nested carved
    // scoring) followed by a carved payload-id index. No hash maps anywhere.
    const uint32_t n_cand_all = (uint32_t)candidates.size();
    uint32_t a_pad_build = 0, b_pad_build = 0;
    size_t tmp_build = 0;
    tri_max_pads(&store, candidates.data(), n_cand_all, nullptr, 0, false, slices, cal.head_dim,
                 a_pad_build, b_pad_build, tmp_build);
    std::vector<uint8_t> heap_scratch;
    tri_bump bump;
    if (options.arena) {
        bump.base = static_cast<uint8_t *>(lease.data());
        bump.cap = lease.size();
    } else {
        heap_scratch.resize(reqs.total_bytes);
        bump.base = heap_scratch.data();
        bump.cap = heap_scratch.size();
    }
    try {
        tri_build_carved(cal, omega.data(), freq_scale_sq.data(), offsets.data(),
                           (uint32_t)offsets.size(), config, candidates, seqs, slices, store, pressure,
                           options.hot_provider, options.explicit_frontiers, options.device_scoring,
                           options.hot_span_provider, a_pad_build, b_pad_build, tmp_build,
                           options.selected_k_fetch, prop.plan, bump);
        if (bump.overflow) {
            throw std::runtime_error("workspace: proposal scratch carve overflow");
        }
    } catch (const xkv_device_scoring_error & e) {
        prop.status = xkv_tri_proposal_status::device_scoring_unavailable;
        prop.message = e.what();
        return prop;
    } catch (const std::invalid_argument & e) {
        prop.status = xkv_tri_proposal_status::invalid_argument;
        prop.message = e.what();
        return prop;
    } catch (const std::out_of_range & e) {
        prop.status = xkv_tri_proposal_status::codec_error;
        prop.message = e.what();
        return prop;
    } catch (const std::runtime_error & e) {
        if (std::string(e.what()).rfind("workspace:", 0) == 0) {
            prop.status = xkv_tri_proposal_status::workspace_exhausted;
        } else {
            prop.status = xkv_tri_proposal_status::score_error;
        }
        prop.message = e.what();
        return prop;
    } catch (const std::exception & e) {
        prop.status = xkv_tri_proposal_status::score_error;
        prop.message = e.what();
        return prop;
    }

    // 4. Floor exhausted check: if no references could be removed, floor is exhausted
    if (prop.plan.references_removed == 0) {
        prop.status = xkv_tri_proposal_status::floor_exhausted;
        prop.message = "TriAttention floor exhausted: no further references can be evicted without violating floor/recent guards";
        return prop;
    }

    // 5. Partition released hot items and factored survivor/tombstone/pack requests
    // Sorted payload-id index carved from the same bump (estimator-sized).
    struct tri_pid_ent {
        uint64_t pid = 0;
        uint32_t idx = 0;
    };
    tri_pid_ent * pidx = tri_carve<tri_pid_ent>(bump, n_cand_all);
    if (!pidx && n_cand_all > 0) {
        prop.status = xkv_tri_proposal_status::workspace_exhausted;
        prop.message = "workspace: proposal pid index carve overflow";
        return prop;
    }
    for (uint32_t i = 0; i < n_cand_all; ++i) {
        pidx[i].pid = candidates[i].payload_id;
        pidx[i].idx = i;
    }
    if (n_cand_all > 0) {
        std::sort(pidx, pidx + n_cand_all, [](const tri_pid_ent & a, const tri_pid_ent & b) {
            return a.pid < b.pid;
        });
    }
    auto tri_pid_find = [&](uint64_t pid) -> const xkv_tri_candidate * {
        uint32_t lo = 0, hi = n_cand_all;
        while (lo < hi) {
            const uint32_t mid = lo + (hi - lo) / 2;
            if (pidx[mid].pid < pid) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo < n_cand_all && pidx[lo].pid == pid) return &candidates[pidx[lo].idx];
        return nullptr;
    };

    for (uint64_t pid : prop.plan.evicted_payloads) {
        const auto * c = tri_pid_find(pid);
        if (c && c->location.kind == xkv_location_kind::hot) {
            prop.released_hot_payloads.push_back(pid);
            prop.released_hot_rows.push_back(c->location.row);
        }
    }

    for (uint64_t pid : prop.plan.survivor_payloads) {
        const auto * c = tri_pid_find(pid);
        if (c && c->location.kind == xkv_location_kind::factored) {
            prop.factored_survivor_payloads.push_back(pid);
        }
    }

    // Mirror resource-specific outcomes onto the proposal.
    prop.hot_slots_freed = prop.plan.hot_slots_freed;
    prop.factored_rows_freed = prop.plan.factored_rows_freed;
    prop.factored_bytes_reclaimed = prop.plan.factored_bytes_reclaimed;
    prop.capacity_satisfied = prop.plan.capacity_satisfied;

    // Classify affected segments: pack vs tombstone
    for (uint64_t seg_id : prop.plan.affected_segments) {
        auto seg = store.get_segment(seg_id);
        if (seg && seg->profile == LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS) {
            if (options.landmark_rebuild) {
                prop.factored_pack_segments.push_back(seg_id);
            } else {
                prop.factored_tombstone_segments.push_back(seg_id);
            }
        } else {
            prop.factored_pack_segments.push_back(seg_id);
        }
    }

    // 6. Build store mutation proposal
    prop.store_proposal = create_mutation_proposal(prop.plan, store, options.landmark_rebuild);

    // Verify stamp hasn't drifted
    if (store.current_stamp() != prop.expected_stamp) {
        prop.status = xkv_tri_proposal_status::stale_stamp;
        prop.message = "store stamp advanced during proposal generation";
        return prop;
    }

    // Preflight store proposal
    std::string preflight_err;
    if (!prop.store_proposal.preflight(store, &preflight_err)) {
        if (store.current_stamp() != prop.expected_stamp) {
            prop.status = xkv_tri_proposal_status::stale_stamp;
        } else {
            prop.status = xkv_tri_proposal_status::store_error;
        }
        prop.message = preflight_err;
        return prop;
    }

    prop.status = xkv_tri_proposal_status::success;
    return prop;
}

xkv_tri_reclaim_proposal xkv_tri_adapter::create_reclaim_proposal(
    const llama_kv_cells & cells,
    const std::vector<xkv_tri_seq_info> & seqs,
    const std::vector<xkv_layer_slice> & slices,
    const llama_xkv_cache_store & store,
    const xkv_tri_pressure_state & pressure,
    const xkv_tri_reclaim_options & options
) const {
    auto candidates = enumerate_candidates_from_cells(cells, store);
    return create_reclaim_proposal(candidates, seqs, slices, store, pressure, options);
}

// ============================================================================
// Production pressure integration bridge
// ============================================================================

xkv_tri_reclaim_proposal xkv_tri_adapter::plan_pressure(
    const llama_kv_cells & cells,
    const llama_memory_kv_reclaim_request & request,
    const std::vector<xkv_layer_slice> & slices,
    const llama_xkv_cache_store & store,
    bool recurrent_pressure,
    const xkv_tri_pressure_hooks & hooks,
    xkv_tri_pressure_resource resource) const {
    std::vector<xkv_tri_seq_info> seqs;
    seqs.reserve(request.seq_hints.size());
    for (const auto & h : request.seq_hints) {
        xkv_tri_seq_info s;
        s.seq_id = h.seq_id;
        s.logical_tokens = h.logical_tokens;
        s.frontier_pos = cells.seq_pos_max(h.seq_id); // authoritative; -1 when absent
        s.tail_guard = h.tail_guard;
        s.eligible = h.eligible;
        s.semantic_episode_id = (uint32_t)h.semantic_episode_id;
        s.semantic_seq_ids = h.semantic_seq_ids;
        seqs.push_back(std::move(s));
    }

    xkv_tri_pressure_state pressure;
    pressure.kv_pressure = request.drain_to_floor || request.required_free > 0;
    pressure.recurrent_pressure = recurrent_pressure;
    pressure.maintenance_due = false;
    pressure.resource = resource;
    pressure.required_units = request.required_free;

    xkv_tri_reclaim_options opts;
    opts.hot_provider = hooks.hot_provider;
    opts.hot_span_provider = hooks.hot_span_provider;
    opts.selected_k_fetch = hooks.selected_k_fetch;
    opts.device_scoring = hooks.device_scoring;
    opts.landmark_rebuild = hooks.landmark_rebuild;
    opts.explicit_frontiers = hooks.explicit_frontiers;
    opts.arena = hooks.arena;
    opts.workspace_budget_bytes = hooks.workspace_budget_bytes;
    return create_reclaim_proposal(cells, seqs, slices, store, pressure, opts);
}

xkv_tri_reclaim_proposal xkv_tri_adapter::plan_store_byte_pressure(
    const llama_kv_cells & cells,
    uint64_t required_bytes,
    const std::vector<xkv_layer_slice> & slices,
    const llama_xkv_cache_store & store,
    const xkv_tri_pressure_hooks & hooks) const {
    // Generate seqs across all resident sequences with authoritatively discovered frontiers
    std::vector<xkv_tri_seq_info> seqs;
    for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
        int64_t pmax = cells.seq_pos_max(s);
        if (pmax >= 0) {
            xkv_tri_seq_info sq;
            sq.seq_id = s;
            sq.logical_tokens = (uint32_t)(pmax + 1);
            sq.frontier_pos = pmax;
            sq.tail_guard = config.recent_window;
            sq.eligible = true;
            seqs.push_back(std::move(sq));
        }
    }

    xkv_tri_pressure_state pressure;
    pressure.kv_pressure = required_bytes > 0;
    pressure.recurrent_pressure = false;
    pressure.maintenance_due = false;
    pressure.resource = xkv_tri_pressure_resource::store_bytes;
    pressure.required_units = required_bytes;

    xkv_tri_reclaim_options opts;
    opts.hot_provider = hooks.hot_provider;
    opts.hot_span_provider = hooks.hot_span_provider;
    opts.selected_k_fetch = hooks.selected_k_fetch;
    opts.device_scoring = hooks.device_scoring;
    opts.landmark_rebuild = hooks.landmark_rebuild;
    opts.explicit_frontiers = hooks.explicit_frontiers;
    opts.arena = hooks.arena;
    opts.workspace_budget_bytes = hooks.workspace_budget_bytes;
    return create_reclaim_proposal(cells, seqs, slices, store, pressure, opts);
}

xkv_tri_pressure_decision xkv_tri_adapter::decide(const xkv_tri_reclaim_proposal & proposal) const {
    switch (proposal.status) {
        case xkv_tri_proposal_status::success:
            return proposal.plan.references_removed > 0 ? xkv_tri_pressure_decision::reclaimed
                                                         : xkv_tri_pressure_decision::idle_noop;
        case xkv_tri_proposal_status::floor_exhausted:
            return xkv_tri_pressure_decision::floor_exhausted_victim;
        case xkv_tri_proposal_status::bypass_recurrent_only:
            return xkv_tri_pressure_decision::bypass_recurrent_only;
        case xkv_tri_proposal_status::stale_stamp:
            return xkv_tri_pressure_decision::retry_stale;
        default:
            return xkv_tri_pressure_decision::error;
    }
}

llama_memory_kv_reclaim_result xkv_tri_adapter::result_from_proposal(
    const xkv_tri_reclaim_proposal & proposal,
    const llama_memory_kv_reclaim_request & request,
    uint32_t physical_before) const {
    (void)request;
    llama_memory_kv_reclaim_result r;
    r.supported = true;
    r.changed = proposal.plan.changed;
    // Map actual resource outcome to capacity_satisfied:
    // If hot_slots was requested, satisfaction reflects freed hot slots;
    // if store_bytes was requested, satisfaction reflects reclaimed factor bytes.
    r.capacity_satisfied = proposal.capacity_satisfied;
    r.floor_reached = (proposal.status == xkv_tri_proposal_status::floor_exhausted);
    r.physical_before = physical_before;
    // physical_freed reflects actual hot slots freed when hot_slots is tracked,
    // or generic physical freed if hot_slots is zero and no hot_slots_freed counted
    r.physical_freed = proposal.hot_slots_freed > 0 ? proposal.hot_slots_freed : proposal.plan.physical_freed;
    r.physical_after = physical_before >= r.physical_freed ? physical_before - r.physical_freed : 0;
    r.references_removed = proposal.plan.references_removed;
    r.target_references = proposal.plan.target_references;
    r.hard_keep = proposal.plan.hard_keep;
    r.shared_keep = proposal.plan.shared_keep;
    r.score_us = 0; // timed by the integrator around apply
    r.pack_us = 0;
    return r;
}

xkv_tri_selected_k_fetch_fn xkv_tri_adapter::make_host_selected_k_fetch(
    const llama_xkv_cache_store & store,
    xkv_arena_lease * scratch,
    uint64_t * call_counter) const {
    (void)scratch; // host reference path uses bounded internal tiles
    // TEST / CPU-reference double ONLY: selected rows, bounded tiles, one
    // call per invocation. Production device-owned segments MUST be served
    // by a native fetch over device tensors (BackendResidency handles +
    // GraphRuntime kernel); this helper must never back device traffic.
    return [this, &store, call_counter](uint64_t seg_id, uint64_t seg_ver, const uint32_t * rows,
                                         uint32_t n, const xkv_layer_slice & slice, uint32_t kv_h,
                                         float * dst, size_t cap, std::string * err) -> bool {
        if (call_counter) ++(*call_counter);
        auto seg = seg_ver > 0 ? store.get_segment_version(seg_id, seg_ver) : store.get_segment(seg_id);
        if (!seg) {
            if (err) *err = "host fetch: segment not found";
            return false;
        }
        try {
            this->stream_factored_k_head(*seg, slice, kv_h, rows, n, dst, cap,
                                          this->config.tile_size > 0 ? this->config.tile_size : 32);
        } catch (const std::exception & e) {
            if (err) *err = std::string("host fetch: ") + e.what();
            return false;
        }
        return true;
    };
}

xkv_tri_usage_snapshot xkv_tri_read_usage(const llama_kv_cells & cells,
                                           const llama_xkv_cache_store & store) {
    xkv_tri_usage_snapshot u;
    u.cells_used = cells.get_used();
    const xkv_accounting acc = store.get_accounting();
    u.live_payload_bytes = acc.live_payload_bytes;
    u.factored_bytes = acc.factored_bytes;
    u.hot_bytes = acc.hot_bytes;
    u.active_segments = acc.active_segments;
    return u;
}

bool xkv_store_mutation_proposal::preflight(const llama_xkv_cache_store & store, std::string * err) const {
    xkv_snapshot_stamp cur = store.current_stamp();
    if (cur != expected_stamp) {
        if (err) *err = "preflight failed: store stamp changed since proposal generation";
        return false;
    }

    for (uint64_t pid : batch.payload_removals) {
        xkv_location loc;
        if (!store.find_location(pid, loc)) {
            if (err) *err = "preflight failed: payload " + std::to_string(pid) + " not found in store";
            return false;
        }
    }

    for (uint64_t seg_id : batch.segments_to_pack) {
        auto seg = store.get_segment(seg_id);
        if (!seg) {
            if (err) *err = "preflight failed: segment " + std::to_string(seg_id) + " not found in store";
            return false;
        }
    }

    return true;
}

bool xkv_store_mutation_proposal::apply(
    llama_xkv_cache_store & store,
    landmark_table * lm_table,
    std::string * err
) const {
    if (batch.payload_removals.empty() && batch.segments_to_pack.empty()) {
        return true;
    }

    if (!preflight(store, err)) {
        return false;
    }

    // Execute atomic batch COW transaction on store:
    // clones affected segments into new segment_versions (sharing immutable B),
    // atomically rebinds locations to new versions/rows, and retires old versions untouched.
    xkv_mutation_result res;
    bool ok = store.execute_mutation_transaction(batch, &res, err);
    if (!ok) {
        return false;
    }

    // Invalidate affected landmark table entries ONLY AFTER successful commit!
    if (lm_table) {
        for (uint64_t seg_id : res.affected_segments) {
            lm_table->invalidate_segment(seg_id);
        }
    }

    return true;
}

} // namespace llama_xkv
