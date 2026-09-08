// llama-xkv-runtime: cache-owned runtime coordinator for target-trunk XKV.
//
// Implements deterministic group construction, eligible-row enumeration joining
// cells semantics to exact hot locations, bulk canonicalization, atomic sealing,
// admission accounting, readiness gating, and graph-snapshot construction.

#include "llama-xkv-runtime.h"

#include "llama-hparams.h"
#include "llama-kv-cache.h"
#include "llama-kv-cells.h"
#include "llama-xkv-canonical.h"
#include "llama-xkv-graph-ref.h"
#include "llama-xkv-transaction.h"
#include "llama-xkv-tri.h"
#include "llama-xkv-backend.h"
#include "llama-xkv-seal.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-triattention.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <unordered_set>
#include <unordered_map>

namespace llama_xkv {

uint64_t fingerprint_seq_set(const std::vector<llama_seq_id> & seq_ids) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (llama_seq_id q : seq_ids) {
        h ^= (uint64_t)(uint32_t) q;
        h *= 0x100000001b3ULL;
    }
    h ^= (uint64_t) seq_ids.size() * 0x9e3779b97f4a7c15ULL;
    h *= 0x100000001b3ULL;
    return h;
}

// Checked arithmetic for exact workspace accounting (false on overflow).
static bool ck_mul_u64(uint64_t a, uint64_t b, uint64_t & out);
static bool ck_add_u64(uint64_t a, uint64_t b, uint64_t & out);
static bool next_segment_bytes_per_token(
    const llama_hparams & hparams,
    const xkv_effective_config & eff,
    const llama_cparams & cparams,
    uint64_t & out_per_token);

uint64_t fingerprint_seal_domain(const xkv_seal_domain_key & key) {
    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 0x100000001b3ULL;
    };
    mix(key.stream); mix(key.visibility);
    mix(key.episode_id); mix(key.run_id); mix(key.publish_epoch);
    mix(key.seq_set_fp);
    return h;
}

std::string layer_placement_key(ggml_tensor * k, ggml_tensor * v) {
    const char * kb = (k && k->buffer) ? ggml_backend_buft_name(ggml_backend_buffer_get_type(k->buffer)) : "?";
    const char * vb = (v && v->buffer) ? ggml_backend_buft_name(ggml_backend_buffer_get_type(v->buffer)) : "?";
    return std::string(kb ? kb : "?") + "+" + std::string(vb ? vb : "?");
}

std::vector<layer_group> split_group_by_placement(
    const layer_group & group,
    const std::vector<std::string> & placements) {
    std::vector<layer_group> out;
    const size_t nl = group.owning_layers.size();
    if (placements.size() != nl || nl == 0) {
        return out;
    }
    size_t run_start = 0;
    for (size_t i = 1; i <= nl; ++i) {
        if (i == nl || placements[i] != placements[run_start]) {
            layer_group r;
            r.group_index = group.group_index;
            uint32_t off_k = 0, off_v = 0;
            for (size_t li = run_start; li < i; ++li) {
                r.owning_layers.push_back(group.owning_layers[li]);
                r.layer_feature_dims_k.push_back(group.layer_feature_dims_k[li]);
                r.layer_feature_dims_v.push_back(group.layer_feature_dims_v[li]);
                r.layer_feature_offsets_k.push_back(off_k);
                r.layer_feature_offsets_v.push_back(off_v);
                off_k += group.layer_feature_dims_k[li];
                off_v += group.layer_feature_dims_v[li];
            }
            r.total_dim_k = off_k;
            r.total_dim_v = off_v;
            out.push_back(std::move(r));
            run_start = i;
        }
    }
    return out;
}

std::vector<llama_seq_id> xkv_cell_seq_ids(const ::llama_kv_cells & cells, uint32_t cell) {
    std::vector<llama_seq_id> out;
    if (cell >= cells.size() || cells.is_empty(cell)) {
        return out;
    }
    for (llama_seq_id q = 0; q < LLAMA_MAX_SEQ; ++q) {
        if (cells.seq_has(cell, q)) {
            out.push_back(q);
        }
    }
    return out;
}

void xkv_runtime_stats::record_skip(xkv_skip_reason reason) {
    const size_t idx = static_cast<size_t>(reason);
    if (skipped_by_reason.size() <= idx) {
        skipped_by_reason.resize(idx + 1, 0);
    }
    skipped_by_reason[idx]++;
}

void xkv_runtime_stats::record_error(const std::string & msg) {
    if (msg.empty()) {
        return;
    }
    const size_t cap = sizeof(last_error) - 1;
    const size_t n = std::min(msg.size(), cap);
    std::memcpy(last_error, msg.data(), n);
    last_error[n] = '\0';
}

void xkv_maintain_control_scratch::init(size_t max_rows, size_t max_groups, size_t max_head_dim) {
    capacity_rows = std::max<size_t>(max_rows, 512);
    capacity_groups = std::max<size_t>(max_groups, 16);
    size_t head_cap = std::max<size_t>(capacity_rows * max_head_dim, 4096);

    eligible.reserve(capacity_rows);
    eligible_indices.reserve(capacity_rows);
    eligible_deduped.reserve(capacity_rows);
    window_idx.reserve(capacity_rows);
    pids.reserve(capacity_rows);
    gens.reserve(capacity_rows);
    positions.reserve(capacity_rows);
    physical_rows.reserve(capacity_rows);

    inputs.reserve(capacity_groups);
    phase_specs.reserve(capacity_groups);
    slices.reserve(capacity_groups);

    // Head scratch is consumed via data()/size() as a fixed buffer, so it
    // must be sized (not merely reserved): the stream reader fail-closes
    // on size(). Retains storage across runs; re-sized only on growth.
    head_scratch.assign(head_cap, 0.0f);
    capacity_head_dim = max_head_dim;
    initialized = true;
}

void xkv_maintain_control_scratch::clear_for_run() {
    eligible.clear();
    eligible_indices.clear();
    eligible_deduped.clear();
    window_idx.clear();
    pids.clear();
    gens.clear();
    positions.clear();
    physical_rows.clear();
    slices.clear();
}

// ---------------------------------------------------------------------------
// Default landmark rebuild: mean-pool per chunk of reconstructed canonical K.
// ---------------------------------------------------------------------------

uint64_t fingerprint_landmark_phase(const std::vector<xkv_landmark_phase_layer> & layers) {
    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 0x100000001b3ULL;
    };
    mix(layers.size());
    for (const auto & l : layers) {
        mix(l.feature_offset); mix(l.feature_dim);
        mix(l.n_heads); mix(l.head_dim); mix(l.rotary_dim);
        for (float f : l.omega) {
            uint32_t u = 0;
            std::memcpy(&u, &f, sizeof(u));
            mix(u);
        }
        for (float f : l.freq_scale_sq) {
            uint32_t u = 0;
            std::memcpy(&u, &f, sizeof(u));
            mix(u);
        }
    }
    return h;
}

// Exact forward half-layout RoPE: the algebraic inverse of invert_rope_k.
// post[f]      = (pre[f] * c - pre[f+Fc] * s) * sc
// post[f + Fc] = (pre[f] * s + pre[f+Fc] * c) * sc, tail passthrough.
// Zero-heap F16 landmark chunk encoder (matches landmark_chunk_encoder_fn
// contract incl. size-query mode): bit-exact fp32->fp16 row convert with a
// measured max-abs error bound. No heap on any path.
static bool landmark_encode_f16_row(const float * mean, uint32_t dim,
    ggml_type type, uint64_t seed, uint8_t * out, size_t out_cap,
    size_t * out_n, codec_desc * out_desc, float * out_error_bound,
    std::string * err) {
    auto fail = [&](const char * m) -> bool { if (err) *err = m; return false; };
    if (!out_n || !out_desc || !out_error_bound) return fail("f16 row: null output");
    if (type != GGML_TYPE_F16) return fail("f16 row: type must be F16");
    if (dim == 0) return fail("f16 row: zero dim");
    codec_desc d;
    try {
        d = make_codec_desc(factor_role::landmark, GGML_TYPE_F16, orientation::token_major,
            {1, dim}, 0, seed);
    } catch (const std::exception &) {
        return fail("f16 row: descriptor rejected");
    }
    uint64_t need = 0;
    try {
        need = encoded_matrix_bytes(d);
    } catch (const std::exception &) {
        return fail("f16 row: byte size overflow");
    }
    *out_n = static_cast<size_t>(need);
    *out_desc = d;
    *out_error_bound = 0.0f;
    if (!out) return true; // size query: mean untouched
    if (!mean) return fail("f16 row: null mean");
    if (out_cap < static_cast<size_t>(need)) return fail("f16 row: output short");
    ggml_fp16_t * dst16 = reinterpret_cast<ggml_fp16_t *>(out);
    ggml_fp32_to_fp16_row(mean, dst16, dim);
    float worst = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        const float back = ggml_fp16_to_fp32(dst16[i]);
        float diff = mean[i] - back;
        if (diff < 0.0f) diff = -diff;
        if (diff > worst) worst = diff;
    }
    *out_error_bound = worst;
    return true;
}
static void apply_forward_rope_head(
    const float * src, float * dst,
    uint32_t head_dim, uint32_t rotary_dim,
    const float * omega, const float * freq_scale_sq,
    int64_t storage_pos) {
    const uint32_t fc = rotary_dim / 2;
    const float pos = (float) storage_pos;
    for (uint32_t f = 0; f < fc; ++f) {
        const float angle = omega[f] * pos;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float sc = (freq_scale_sq && freq_scale_sq[f] > 0.0f) ? sqrtf(freq_scale_sq[f]) : 1.0f;
        const float x0 = src[f];
        const float x1 = src[f + fc];
        dst[f]        = (x0 * c - x1 * s) * sc;
        dst[f + fc]   = (x0 * s + x1 * c) * sc;
    }
    for (uint32_t d = rotary_dim; d < head_dim; ++d) {
        dst[d] = src[d];
    }
}

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
    std::string * err) {
    const uint64_t n = surviving_rows.size();
    if (n == 0) {
        if (err) *err = "phased landmarks: no surviving rows";
        return false;
    }
    if (!a_dec || !b_dec || !recon_scratch || !phased_scratch || !lm_scratch) {
        if (err) *err = "phased landmarks: null decode/scratch buffers";
        return false;
    }
    // Positions are REQUIRED and parallel: phasing needs every row's own
    // storage position. A missing position refuses; fragment-time selection
    // can never repair a pre-pooled mean.
    if (storage_positions.size() != n) {
        if (err) *err = "phased landmarks: storage_positions must parallel surviving_rows";
        return false;
    }
    if (chunk_tokens == 0) {
        if (err) *err = "phased landmarks: chunk_tokens must be non-zero";
        return false;
    }
    // The fingerprint must key the exact transform just applied.
    if (phase_tx_fingerprint == 0 || phase_tx_fingerprint != fingerprint_landmark_phase(phase_layers)) {
        if (err) *err = "phased landmarks: phase fingerprint does not key the transform";
        return false;
    }
    if (rank == 0 || a_pad < rank || b_pad < rank || b_dim == 0) {
        if (err) *err = "phased landmarks: degenerate decoded factor rank";
        return false;
    }
    // Phase layers must contiguously cover the concatenated feature width.
    uint64_t dim = 0;
    for (const auto & l : phase_layers) {
        if (l.feature_offset != dim) {
            if (err) *err = "phased landmarks: phase layers do not contiguously cover the row";
            return false;
        }
        if (l.feature_dim != (uint64_t) l.n_heads * l.head_dim) {
            if (err) *err = "phased landmarks: phase layer dim mismatch";
            return false;
        }
        if (l.rotary_dim == 0 || (l.rotary_dim & 1u) != 0 || l.rotary_dim > l.head_dim) {
            if (err) *err = "phased landmarks: invalid rotary_dim";
            return false;
        }
        if (l.omega.size() != l.rotary_dim / 2 || l.freq_scale_sq.size() != l.rotary_dim / 2) {
            if (err) *err = "phased landmarks: phase table size mismatch";
            return false;
        }
        dim += l.feature_dim;
    }
    if (dim == 0 || dim != b_dim) {
        if (err) *err = "phased landmarks: phase coverage does not match B width";
        return false;
    }
    // Checked ceil division: chunk_tokens is nonzero (checked above), and the
    // numerator must not overflow (no silent wrap to a short count).
    uint64_t chunk_ceil_num = 0;
    if (!ck_add_u64(n, (uint64_t) chunk_tokens - 1u, chunk_ceil_num)) {
        if (err) *err = "phased landmarks: chunk count overflow";
        return false;
    }
    const uint64_t chunks = chunk_ceil_num / chunk_tokens;
    uint64_t lm_need = 0;
    if (!ck_mul_u64(chunks, dim, lm_need) || recon_cap < dim || phased_cap < dim ||
        lm_cap < lm_need) {
        if (err) *err = "phased landmarks: scratch capacity insufficient";
        return false;
    }
    float * recon = recon_scratch;
    float * phased = phased_scratch;
    float * lm = lm_scratch;
    for (uint64_t i = 0; i < lm_need; ++i) {
        lm[i] = 0.0f;
    }
    for (uint64_t r = 0; r < n; ++r) {
        const uint32_t srow = surviving_rows[(size_t) r];
        if ((uint64_t) srow >= a_rows) {
            if (err) *err = "phased landmarks: surviving row out of range";
            return false;
        }
        // Reconstruct the canonical (pre-RoPE) row from FINAL streams.
        const float * a_row = a_dec + (size_t) srow * a_pad;
        for (uint64_t c = 0; c < dim; ++c) {
            const float * b_row = b_dec + (size_t) c * b_pad;
            float sum = 0.0f;
            for (uint64_t k = 0; k < rank; ++k) {
                sum += a_row[k] * b_row[k];
            }
            recon[c] = sum;
        }
        // Phase every layer slice at this row's own storage position.
        const int64_t pos = storage_positions[(size_t) r];
        for (const auto & l : phase_layers) {
            for (uint32_t h = 0; h < l.n_heads; ++h) {
                apply_forward_rope_head(
                    recon + l.feature_offset + (uint64_t) h * l.head_dim,
                    phased + l.feature_offset + (uint64_t) h * l.head_dim,
                    l.head_dim, l.rotary_dim, l.omega.data(), l.freq_scale_sq.data(), pos);
            }
        }
        float * acc = lm + (r / chunk_tokens) * dim;
        for (uint64_t c = 0; c < dim; ++c) {
            acc[c] += phased[c];
        }
    }
    for (uint64_t ch = 0; ch < chunks; ++ch) {
        const uint64_t r0 = ch * chunk_tokens;
        const uint64_t r1 = std::min<uint64_t>(n, r0 + chunk_tokens);
        const float inv = 1.0f / (float)(r1 - r0);
        float * acc = lm + ch * dim;
        for (uint64_t c = 0; c < dim; ++c) {
            acc[c] *= inv;
        }
    }
    try {
        // Codec table seed is the phase fingerprint: the codebook is keyed
        // to the exact transform, never an unrelated magic number.
        const codec_desc desc = make_codec_desc(
            factor_role::landmark, landmark_type, orientation::token_major,
            {chunks, dim}, 0, phase_tx_fingerprint);
        out_landmark = encode_matrix(desc, lm, (size_t)(chunks * dim));
    } catch (const std::exception & e) {
        if (err) *err = std::string("phased landmarks: encode failed: ") + e.what();
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Quiescence guard.
// ---------------------------------------------------------------------------

xkv_quiescence_guard::xkv_quiescence_guard(const llama_kv_cache & kv) {
    capture(kv);
}

void xkv_quiescence_guard::capture(const llama_kv_cache & kv) {
    armed_ = true;
    auto store = kv.get_xkv_store();
    have_store_ = (store != nullptr);
    if (have_store_) {
        stamp_ = store->current_stamp();
    }
    auto pool = kv.get_hot_slot_pool();
    have_pool_ = (pool != nullptr);
    if (have_pool_) {
        pool_reserved_ = pool->get_reserved();
        pool_bound_ = pool->get_bound();
    } else {
        pool_reserved_ = 0;
        pool_bound_ = 0;
    }
}

bool xkv_quiescence_guard::validate(const llama_kv_cache & kv, std::string * err) const {
    if (!armed_) {
        if (err) *err = "quiescence guard is not armed";
        return false;
    }
    auto store = kv.get_xkv_store();
    const bool has_store = (store != nullptr);
    if (has_store != have_store_) {
        if (err) *err = "quiescence violation: store presence changed";
        return false;
    }
    if (has_store && !(store->current_stamp() == stamp_)) {
        if (err) *err = "quiescence violation: store stamp moved during seal preparation";
        return false;
    }
    auto pool = kv.get_hot_slot_pool();
    const bool has_pool = (pool != nullptr);
    if (has_pool != have_pool_) {
        if (err) *err = "quiescence violation: hot pool presence changed";
        return false;
    }
    if (has_pool && (pool->get_reserved() != pool_reserved_ || pool->get_bound() != pool_bound_)) {
        if (err) *err = "quiescence violation: hot pool live set changed during seal preparation";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Runtime.
// ---------------------------------------------------------------------------

std::unique_ptr<llama_xkv_runtime> llama_xkv_runtime::create(const llama_cparams & cparams) {
    if (cparams.xkv_mode == LLAMA_XKV_MODE_OFF) {
        return nullptr;
    }
    if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
        return nullptr;
    }
    std::string cerr;
    if (!config_supported(cparams, &cerr)) {
        LLAMA_LOG_ERROR("%s: XKV configuration rejected: %s\n", __func__, cerr.c_str());
        return nullptr;
    }
    return std::unique_ptr<llama_xkv_runtime>(new llama_xkv_runtime(cparams));
}

bool llama_xkv_runtime::config_supported(const llama_cparams & cparams, std::string * err) {
    if (cparams.xkv_mode == LLAMA_XKV_MODE_OFF || cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
        return true; // no runtime by design
    }
    if (cparams.xkv_source == LLAMA_XKV_SOURCE_PREROPE_CAPTURE) {
        if (err) *err = "PREROPE_CAPTURE source requires a staging ring/graph capture (not wired)";
        return false;
    }
    // Factorizer/profile matrix (explicit, never reinterpreted):
    // - CPU_REFERENCE: CPU path for every profile.
    // - VULKAN + TQ_FACTORS: native bridge. VULKAN + LANDMARKS stays on the
    //   CPU path (no native landmark kernel exists; bridge refuses it).
    // - VULKAN + REFERENCE: contradictory (reference expects host, vulkan
    //   demands device) — no valid path, reject.
    // - HYBRID: split unimplemented — reject until real.
    // - CUDA: unsupported on this build — reject.
    if (cparams.xkv_factorizer == LLAMA_XKV_FACTORIZER_CUDA) {
        if (err) *err = "CUDA factorizer is not supported on this build";
        return false;
    }
    if (cparams.xkv_factorizer == LLAMA_XKV_FACTORIZER_VULKAN_HYBRID) {
        if (err) *err = "VULKAN_HYBRID split is not implemented; use vulkan or cpu-reference";
        return false;
    }
    if (cparams.xkv_factorizer == LLAMA_XKV_FACTORIZER_VULKAN &&
        cparams.xkv_storage_profile == LLAMA_XKV_STORAGE_PROFILE_REFERENCE) {
        if (err) *err = "VULKAN factorizer contradicts REFERENCE host profile; no valid path";
        return false;
    }
    return true;
}

void llama_xkv_runtime::refresh_readiness(const llama_kv_cache & kv) const {
    readiness_.ready = false;
    readiness_.reason[0] = '\0';
    auto fail = [&](const std::string & r) {
        std::strncpy(readiness_.reason, r.c_str(), sizeof(readiness_.reason) - 1);
    };
    std::string cerr;
    if (!config_supported(cparams_, &cerr)) {
        fail(cerr);
        return;
    }
    if (!groups_valid_) {
        fail("layer groups not built");
        return;
    }
    // Inspect actual target-layer hot tensor backends: CPU host buffers
    // support the CPU/reference graph path; anything else stays false
    // (Vulkan TQ until native full-attention dispatch; CUDA/Metal always).
    for (const auto & g : groups_.groups) {
        for (uint32_t ol : g.owning_layers) {
            ggml_tensor * k = kv.get_k_storage((int32_t) ol);
            ggml_tensor * v = kv.get_v_storage((int32_t) ol);
            if (!k || !v || !k->buffer || !v->buffer ||
                !ggml_backend_buffer_is_host(k->buffer) ||
                !ggml_backend_buffer_is_host(v->buffer)) {
                fail("hot tensor of layer " + std::to_string(ol) + " is not host-resident");
                return;
            }
        }
    }
    const ggml_type stream_types[5] = {
        cparams_.xkv_factor_a_k, cparams_.xkv_factor_b_k, cparams_.xkv_factor_a_v,
        cparams_.xkv_factor_b_v, cparams_.xkv_landmark_type,
    };
    for (int i = 0; i < 5; ++i) {
        if (!ggml_xkv_codec_supported(stream_types[i])) {
            fail("factor/landmark codec is not supported on this build");
            return;
        }
    }
    readiness_.ready = true;
}

llama_xkv_runtime::llama_xkv_runtime(const llama_cparams & cparams)
    : cparams_(cparams),
      eff_(make_effective_config(cparams)),
      bounded_hot_(cparams.xkv_mode == LLAMA_XKV_MODE_DENSE || cparams.xkv_mode == LLAMA_XKV_MODE_SR) {
    // Checked MiB capacity: saturating multiply, shared by the whole family.
    uint64_t dcap = (uint64_t) cparams.xkv_decode_cache_mib * 1024 * 1024;
    if (dcap > (uint64_t) SIZE_MAX) {
        dcap = (uint64_t) SIZE_MAX;
    }
    b_tile_cache_ = std::make_shared<xkv_b_tile_cache>((size_t) dcap);
    LLAMA_LOG_INFO("%s: xkv effective config: profile v%u groups=%u rank=(%u,%u) seg=%u chunk=%u seed=%llu max_err=%.3f min_save=%.3f/%zu fp=%016llx\n",
        __func__, eff_.profile_version, eff_.group_size, eff_.rank_k, eff_.rank_v,
        eff_.segment_tokens, eff_.chunk_tokens, (unsigned long long) eff_.factor_seed,
        eff_.max_relative_error, eff_.min_saving_ratio, eff_.min_saving_bytes,
        (unsigned long long) eff_.fingerprint());
}

xkv_decode_cache_metrics llama_xkv_runtime::decode_cache_metrics() const {
    xkv_decode_cache_metrics m;
    if (!b_tile_cache_) {
        return m; // honest zeros: no cache, no use
    }
    m.max_bytes = b_tile_cache_->max_bytes();
    m.active_bytes = b_tile_cache_->active_cache_bytes();
    m.total_allocated_bytes = b_tile_cache_->total_allocated_bytes();
    m.hits = b_tile_cache_->hit_count();
    m.misses = b_tile_cache_->miss_count();
    m.evictions = b_tile_cache_->eviction_count();
    return m;
}

void llama_xkv_runtime::invalidate_decode_cache_segment(uint64_t segment_id, uint64_t segment_version) {
    if (b_tile_cache_) {
        b_tile_cache_->invalidate_segment(segment_id, segment_version);
    }
}

void llama_xkv_runtime::invalidate_decode_cache_epochs(uint64_t content_epoch, uint64_t codec_epoch) {
    if (b_tile_cache_) {
        b_tile_cache_->invalidate_epoch(content_epoch, codec_epoch);
    }
}

void llama_xkv_runtime::clear_decode_cache() {
    if (b_tile_cache_) {
        b_tile_cache_->clear();
    }
}

xkv_effective_config make_effective_config(const llama_cparams & cparams) {
    xkv_effective_config e;
    e.group_size = cparams.xkv_group_size;
    e.rank_k = cparams.xkv_rank_k;
    e.rank_v = cparams.xkv_rank_v;
    e.segment_tokens = cparams.xkv_segment_tokens;
    e.chunk_tokens = cparams.xkv_chunk_tokens;
    e.profile_version = xkv_sealing_profile::version;
    // Configured factor seed from cparams; fallback to factor_config default
    // only when 0 (unspecified). Never hardcoded.
    e.factor_seed = (cparams.xkv_seed != 0) ? cparams.xkv_seed : factor_config{}.seed;
    e.max_relative_error = xkv_sealing_profile::max_relative_error;
    e.min_saving_ratio = cparams.xkv_min_saving;
    e.min_saving_bytes = xkv_sealing_profile::min_saving_bytes;
    e.mode = (int32_t) cparams.xkv_mode;
    e.storage_profile = (int32_t) cparams.xkv_storage_profile;
    e.source = (int32_t) cparams.xkv_source;
    e.factor_a_k = (int32_t) cparams.xkv_factor_a_k;
    e.factor_b_k = (int32_t) cparams.xkv_factor_b_k;
    e.factor_a_v = (int32_t) cparams.xkv_factor_a_v;
    e.factor_b_v = (int32_t) cparams.xkv_factor_b_v;
    e.factor_balance = (int32_t) cparams.xkv_factor_balance;
    e.landmark_type = (int32_t) cparams.xkv_landmark_type;
    e.landmark_refine = (int32_t) cparams.xkv_landmark_refine;
    e.landmark_refine_max_rows = cparams.xkv_landmark_refine_max_rows;
    e.sr_budget = cparams.xkv_sr_budget;
    e.factorizer = (int32_t) cparams.xkv_factorizer;
    e.workspace_mib = cparams.xkv_workspace_mib;
    e.decode_cache_mib = cparams.xkv_decode_cache_mib;
    e.store_mib = cparams.xkv_store_mib;
    e.min_factor_coverage = cparams.xkv_min_factor_coverage;
    return e;
}

bool xkv_effective_config::valid(std::string * err) const {
    if (group_size == 0) {
        if (err) *err = "effective config: group_size must be >= 1";
        return false;
    }
    if (rank_k == 0 || rank_v == 0) {
        if (err) *err = "effective config: ranks must be >= 1";
        return false;
    }
    if (segment_tokens == 0) {
        if (err) *err = "effective config: segment_tokens must be >= 1 (no hidden fallback)";
        return false;
    }
    if (chunk_tokens == 0) {
        if (err) *err = "effective config: chunk_tokens must be >= 1 (no hidden fallback)";
        return false;
    }
    if (!(max_relative_error > 0.0) || !(max_relative_error < 1.0)) {
        if (err) *err = "effective config: max_relative_error must be in (0, 1)";
        return false;
    }
    if (!(min_saving_ratio >= 0.0) || !(min_saving_ratio < 1.0)) {
        if (err) *err = "effective config: min_saving_ratio must be in [0, 1)";
        return false;
    }
    return true;
}

uint64_t xkv_effective_config::fingerprint() const {
    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 0x100000001b3ULL;
    };
    mix(group_size); mix(rank_k); mix(rank_v);
    mix(profile_version);
    mix(segment_tokens); mix(chunk_tokens); mix(factor_seed);
    mix((uint64_t)(uint32_t) mode); mix((uint64_t)(uint32_t) storage_profile);
    mix((uint64_t)(uint32_t) source);
    mix((uint64_t)(uint32_t) factor_a_k); mix((uint64_t)(uint32_t) factor_b_k);
    mix((uint64_t)(uint32_t) factor_a_v); mix((uint64_t)(uint32_t) factor_b_v);
    mix((uint64_t)(uint32_t) factor_balance);
    mix((uint64_t)(uint32_t) landmark_type);
    mix((uint64_t)(uint32_t) landmark_refine);
    mix(landmark_refine_max_rows); mix(sr_budget);
    mix((uint64_t)(uint32_t) factorizer);
    mix(workspace_mib); mix(decode_cache_mib); mix(store_mib);
    mix(static_cast<uint64_t>(min_factor_coverage * 1e9));
    mix(static_cast<uint64_t>(max_relative_error * 1e9));
    mix(static_cast<uint64_t>(min_saving_ratio * 1e9));
    mix(min_saving_bytes);
    return h;
}

bool llama_xkv_runtime::rebuild_groups(
    const llama_hparams & hparams, const llama_kv_cache & kv, std::string * err) const {
    const uint32_t n_all = hparams.n_layer_all;
    if (n_all == 0) {
        if (err) *err = "rebuild_groups: model has no layers";
        return false;
    }
    const uint32_t n_trunk = hparams.n_layer();
    std::vector<uint32_t> model_to_owning(n_all, UINT32_MAX);
    std::vector<bool> is_attention(n_all, false);
    std::vector<uint32_t> dim_k(n_all, 0);
    std::vector<uint32_t> dim_v(n_all, 0);
    for (uint32_t il = 0; il < n_all; ++il) {
        const bool eligible = (il < n_trunk) && hparams.has_kv(il) && !hparams.is_recr(il) && !hparams.is_swa(il);
        is_attention[il] = eligible;
        // Alias resolution through the live cache: model layer -> owning layer.
        model_to_owning[il] = kv.get_owning_layer(il);
        if (eligible) {
            dim_k[il] = hparams.n_embd_k_gqa(il);
            dim_v[il] = hparams.n_embd_v_gqa(il);
            if (dim_k[il] == 0 || dim_v[il] == 0) {
                if (err) *err = "rebuild_groups: zero KV dim on eligible layer " + std::to_string(il);
                return false;
            }
        } else {
            dim_k[il] = 1;
            dim_v[il] = 1;
        }
    }
    uint32_t group_size = cparams_.xkv_group_size;
    if (group_size == 0) {
        group_size = 4;
    }
    try {
        groups_ = build_layer_group_map_ex(model_to_owning, is_attention, group_size, dim_k, dim_v);
    } catch (const std::exception & e) {
        if (err) *err = std::string("rebuild_groups: ") + e.what();
        return false;
    }
    if (groups_.groups.empty()) {
        if (err) *err = "rebuild_groups: no eligible target-trunk layers";
        return false;
    }
    // Backend placement split: a shared B factor spans exactly one device
    // placement. Groups crossing device boundaries split into maximal
    // homogeneous runs (offsets rebased, group indices reassigned, tail
    // ranks recomputed per final group downstream). Placement identity per
    // final group is retained for native gating and fingerprinting.
    std::vector<layer_group> split_groups;
    std::vector<std::string> split_placements;
    for (const auto & grp : groups_.groups) {
        std::vector<std::string> keys;
        keys.reserve(grp.owning_layers.size());
        for (uint32_t ol : grp.owning_layers) {
            ggml_tensor * k = kv.get_k_storage((int32_t) ol);
            ggml_tensor * v = kv.get_v_storage((int32_t) ol);
            if (!k || !v || !k->buffer || !v->buffer) {
                if (err) *err = "rebuild_groups: layer " + std::to_string(ol) +
                                " hot tensors are not allocated";
                return false;
            }
            keys.push_back(layer_placement_key(k, v));
        }
        std::vector<layer_group> runs = split_group_by_placement(grp, keys);
        if (runs.empty()) {
            if (err) *err = "rebuild_groups: placement split failed";
            return false;
        }
        size_t ri = 0;
        for (auto & r : runs) {
            split_placements.push_back(keys[ri]);
            ri += r.owning_layers.size();
            split_groups.push_back(std::move(r));
        }
    }
    groups_.groups.swap(split_groups);
    for (size_t gi = 0; gi < groups_.groups.size(); ++gi) {
        groups_.groups[gi].group_index = (uint32_t) gi;
    }
    group_placements_.swap(split_placements);
    placement_id_ = group_placements_.empty() ? std::string() : group_placements_[0];
    layer_enabled_.assign(n_all, 0);
    for (uint32_t il = 0; il < n_all; ++il) {
        layer_enabled_[il] = is_attention[il] ? 1 : 0;
    }
    groups_valid_ = true;
    groups_version_++;
    return true;
}

bool llama_xkv_runtime::xkv_layer_enabled(uint32_t il) const {
    if (!groups_valid_) {
        return false;
    }
    if (il >= layer_enabled_.size()) {
        return false;
    }
    return layer_enabled_[il] != 0;
}

bool llama_xkv_runtime::ensure_groups(
    const llama_hparams & hparams, const llama_kv_cache & kv, std::string * err) const {
    if (groups_valid_) {
        refresh_readiness(kv);
        return true;
    }
    if (!rebuild_groups(hparams, kv, err)) {
        return false;
    }
    refresh_readiness(kv);
    return true;
}


bool llama_xkv_runtime::enumerate_into_scratch(
    const llama_kv_cache & kv, bool fill_seq_ids, std::string * err) const {
    auto & eligible = scratch_.eligible;
    auto & order = scratch_.eligible_indices;
    auto & deduped = scratch_.eligible_deduped;
    eligible.clear();
    order.clear();
    deduped.clear();
    auto store = kv.get_xkv_store();
    if (!store) {
        if (err) *err = "enumerate: no XKV store (OFF/MTP context?)";
        return false;
    }
    auto pool = kv.get_hot_slot_pool();
    if (pool && pool->get_reserved() != 0) {
        if (err) *err = "enumerate: hot pool has in-flight reservations";
        return false;
    }
    // Pool-less (SHADOW) bindings: linear scan over the snapshot vector
    // (no hash map heap). Snapshot dies with this call.
    xkv_hot_bindings_snapshot shadow_bindings;
    if (!pool) {
        shadow_bindings = store->list_hot_payload_bindings();
    }
    const uint32_t n_stream = kv.get_n_stream();
    for (uint32_t s = 0; s < n_stream; ++s) {
        if (!kv.validate_seq_id((llama_seq_id) s) || kv.get_stream_for_seq((llama_seq_id) s) != s) {
            continue;
        }
        const llama_kv_cells & cells = kv.get_cells((llama_seq_id) s);
        const uint32_t n = cells.size();
        for (uint32_t i = 0; i < n; ++i) {
            if (cells.is_empty(i)) {
                continue;
            }
            const uint64_t pid = cells.payload_id_get(i);
            if (pid == 0) {
                continue;
            }
            const uint64_t gen = cells.storage_generation_get(i);
            const llama_pos spos = cells.pos_get(i);
            const llama_kv_rerot_meta & meta = cells.rerot_get(i);
            const auto vis = meta.visibility;
            if (vis != llama_rerot_visibility::normal &&
                vis != llama_rerot_visibility::public_live) {
                continue;
            }
            xkv_location loc;
            if (!store->find_location(pid, loc)) {
                continue;
            }
            if (loc.state != xkv_state::hot_committed || loc.kind != xkv_location_kind::hot) {
                continue;
            }
            uint32_t physical_row = 0;
            if (pool) {
                xkv_hot_slot_info info;
                if (!pool->find_payload(pid, info) || info.storage_generation != gen) {
                    continue;
                }
                physical_row = info.slot;
            } else {
                bool found = false;
                for (const auto & b : shadow_bindings.bindings) {
                    if (b.payload_id == pid && b.storage_generation == gen) {
                        physical_row = b.hot_slot_row;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    continue;
                }
            }
            if (physical_row >= kv.get_hot_size()) {
                continue;
            }
            xkv_layer_row_view row;
            row.payload_id = pid;
            row.generation = gen;
            row.physical_row = physical_row;
            row.storage_pos = (int32_t) spos;
            row.logical_cell = i;
            row.stream = s;
            row.visibility = static_cast<uint8_t>(vis);
            row.episode_id = meta.episode_id;
            row.run_id = (uint64_t) meta.run_id;
            row.publish_epoch = meta.publish_epoch;
            if (fill_seq_ids) {
                row.seq_ids = xkv_cell_seq_ids(cells, i);
                row.seq_set_fp = (vis == llama_rerot_visibility::normal)
                    ? fingerprint_seq_set(row.seq_ids) : 0;
            } else {
                // Fingerprint inline without retaining the keeper set.
                uint64_t fp = 0xcbf29ce484222325ULL;
                uint64_t cnt = 0;
                if (vis == llama_rerot_visibility::normal) {
                    for (llama_seq_id q = 0; q < LLAMA_MAX_SEQ; ++q) {
                        if (cells.seq_has(i, q)) {
                            fp ^= (uint64_t)(uint32_t) q;
                            fp *= 0x100000001b3ULL;
                            cnt++;
                        }
                    }
                    fp ^= cnt * 0x9e3779b97f4a7c15ULL;
                    fp *= 0x100000001b3ULL;
                }
                row.seq_set_fp = fp;
            }
            eligible.push_back(std::move(row));
        }
    }
    // Sort indices by payload for linear duplicate runs (no hash map).
    order.resize(eligible.size());
    for (size_t i = 0; i < eligible.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return eligible[a].payload_id < eligible[b].payload_id;
    });
    // Keep first row of each consistent run; drop mismatched payloads wholly.
    for (size_t lo = 0; lo < order.size();) {
        size_t hi = lo + 1;
        bool consistent = true;
        while (hi < order.size() && eligible[order[hi]].payload_id == eligible[order[lo]].payload_id) {
            const auto & A = eligible[order[lo]];
            const auto & B = eligible[order[hi]];
            const bool same_seq = fill_seq_ids ? (A.seq_ids == B.seq_ids)
                                               : (A.seq_set_fp == B.seq_set_fp);
            if (A.generation != B.generation || A.storage_pos != B.storage_pos ||
                A.physical_row != B.physical_row || A.stream != B.stream ||
                A.episode_id != B.episode_id || A.run_id != B.run_id ||
                A.publish_epoch != B.publish_epoch || !same_seq) {
                consistent = false;
            }
            ++hi;
        }
        if (consistent) {
            deduped.push_back(order[lo]);
        }
        lo = hi;
    }
    return true;
}

bool llama_xkv_runtime::collect_eligible_rows(
    const llama_kv_cache & kv,
    std::vector<xkv_layer_row_view> & out,
    std::string * err) const {
    out.clear();
    if (!enumerate_into_scratch(kv, true, err)) {
        return false;
    }
    auto & order = scratch_.eligible_indices;
    auto & deduped = scratch_.eligible_deduped;
    auto & eligible = scratch_.eligible;
    order = deduped;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (eligible[a].stream != eligible[b].stream) return eligible[a].stream < eligible[b].stream;
        if (eligible[a].storage_pos != eligible[b].storage_pos) {
            return eligible[a].storage_pos < eligible[b].storage_pos;
        }
        return eligible[a].payload_id < eligible[b].payload_id;
    });
    out.reserve(order.size());
    for (size_t i : order) {
        out.push_back(eligible[i]);
    }
    return true;
}

bool llama_xkv_runtime::read_layer_canonical(
    const llama_kv_cache & kv,
    uint32_t owning_layer,
    const std::vector<xkv_layer_row_view> & rows,
    matrix & out_k,
    matrix & out_v,
    std::string * err) const {
    const uint32_t n = (uint32_t) rows.size();
    if (n == 0) {
        if (err) *err = "read_layer_canonical: no rows";
        return false;
    }
    const uint32_t stream = rows[0].stream;
    for (const auto & r : rows) {
        if (r.stream != stream) {
            if (err) *err = "read_layer_canonical: heterogeneous streams in one read";
            return false;
        }
    }
    if (!kv.validate_seq_id((llama_seq_id) stream)) {
        if (err) *err = "read_layer_canonical: no valid seq id for stream";
        return false;
    }
    std::vector<uint32_t> cell_indices(n);
    std::vector<int32_t> positions(n);
    for (uint32_t i = 0; i < n; ++i) {
        cell_indices[i] = rows[i].physical_row;
        positions[i] = rows[i].storage_pos;
    }
    xkv_canonical_layer_kv_snapshot snap;
    if (!snap.init(kv, cparams_, owning_layer,
                   cell_indices.data(), positions.data(), n, (llama_seq_id) stream,
                   true, true, cparams_.xkv_source,
                   nullptr, nullptr, nullptr, nullptr, err)) {
        return false;
    }
    const uint32_t n_heads = snap.get_n_kv_heads();
    const uint32_t hd_k = snap.get_head_dim_k();
    const uint32_t hd_v = snap.get_head_dim_v();
    if (n_heads == 0 || hd_k == 0 || hd_v == 0) {
        if (err) *err = "read_layer_canonical: degenerate head geometry";
        return false;
    }
    const uint64_t total_k = (uint64_t) n_heads * hd_k;
    const uint64_t total_v = (uint64_t) n_heads * hd_v;
    out_k = matrix(n, total_k);
    out_v = matrix(n, total_v);
    std::vector<float> head_buf;
    head_buf.reserve(std::max(hd_k, hd_v) * 1);
    for (uint32_t h = 0; h < n_heads; ++h) {
        head_buf.assign(n * hd_k, 0.0f);
        if (!snap.read_k_head(h, head_buf.data(), head_buf.size(), err)) {
            return false;
        }
        for (uint32_t r = 0; r < n; ++r) {
            float * dst = out_k.row_ptr(r) + (uint64_t) h * hd_k;
            const float * src = head_buf.data() + (uint64_t) r * hd_k;
            std::memcpy(dst, src, hd_k * sizeof(float));
        }
        head_buf.assign(n * hd_v, 0.0f);
        if (!snap.read_v_head(h, head_buf.data(), head_buf.size(), err)) {
            return false;
        }
        for (uint32_t r = 0; r < n; ++r) {
            float * dst = out_v.row_ptr(r) + (uint64_t) h * hd_v;
            const float * src = head_buf.data() + (uint64_t) r * hd_v;
            std::memcpy(dst, src, hd_v * sizeof(float));
        }
    }
    return true;
}

xkv_bundle_sealing_params llama_xkv_runtime::sealing_params() const {
    xkv_bundle_sealing_params p;
    p.profile = cparams_.xkv_storage_profile;
    p.source = cparams_.xkv_source;
    p.rank_k = eff_.rank_k;
    p.rank_v = eff_.rank_v;
    p.balance = cparams_.xkv_factor_balance;
    p.seed = eff_.factor_seed;
    p.factor_a_k = cparams_.xkv_factor_a_k;
    p.factor_b_k = cparams_.xkv_factor_b_k;
    p.factor_a_v = cparams_.xkv_factor_a_v;
    p.factor_b_v = cparams_.xkv_factor_b_v;
    p.landmark_type = cparams_.xkv_landmark_type;
    p.max_relative_error = eff_.max_relative_error;
    p.min_saving_ratio = eff_.min_saving_ratio;
    p.min_saving_bytes = eff_.min_saving_bytes;
    return p;
}

// Checked arithmetic for exact workspace accounting (false on overflow).
static bool ck_mul_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (a == 0 || b == 0) {
        out = 0;
        return true;
    }
    if (a > std::numeric_limits<uint64_t>::max() / b) {
        return false;
    }
    out = a * b;
    return true;
}

static bool ck_add_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (a > std::numeric_limits<uint64_t>::max() - b) {
        return false;
    }
    out = a + b;
    return true;
}

// Streams one owning layer into its group workspace slice, one head at a
// time: temp storage is bounded by a single head (n * max head dim), never
// a full layer, and there is no duplicate group backing. Canonical backend
// synchronizations are accumulated into stats when provided.
bool llama_xkv_runtime::stream_layer_canonical(
    const llama_kv_cache & kv,
    uint32_t owning_layer,
    float * k_dst, uint64_t k_row_stride, uint32_t k_off,
    float * v_dst, uint64_t v_row_stride, uint32_t v_off,
    canonical_batch_stats * stats,
    std::string * err) const {
    auto & eligible = scratch_.eligible;
    auto & window = scratch_.window_idx;
    auto & cell_idx = scratch_.canonical_cells;
    auto & pos32 = scratch_.canonical_positions;
    auto & head = scratch_.head_scratch;
    const uint32_t n = (uint32_t) window.size();
    if (n == 0 || !k_dst || !v_dst) {
        if (err) *err = "stream_layer_canonical: no rows or null workspace";
        return false;
    }
    const uint32_t stream = eligible[window[0]].stream;
    for (size_t wi : window) {
        if (eligible[wi].stream != stream) {
            if (err) *err = "stream_layer_canonical: heterogeneous streams";
            return false;
        }
    }
    if (!kv.validate_seq_id((llama_seq_id) stream)) {
        if (err) *err = "stream_layer_canonical: no valid seq id for stream";
        return false;
    }
    cell_idx.resize(n);
    pos32.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        cell_idx[i] = eligible[window[i]].physical_row;
        pos32[i] = eligible[window[i]].storage_pos;
    }
    xkv_canonical_layer_kv_snapshot snap;
    if (!snap.init(kv, cparams_, owning_layer,
                   cell_idx.data(), pos32.data(), n, (llama_seq_id) stream,
                   true, true, cparams_.xkv_source,
                   nullptr, nullptr, nullptr, stats, err)) {
        return false;
    }
    const uint32_t n_heads = snap.get_n_kv_heads();
    const uint32_t hd_k = snap.get_head_dim_k();
    const uint32_t hd_v = snap.get_head_dim_v();
    if (n_heads == 0 || hd_k == 0 || hd_v == 0) {
        if (err) *err = "stream_layer_canonical: degenerate head geometry";
        return false;
    }
    uint64_t need_k = 0, need_v = 0;
    if (!ck_mul_u64(n_heads, hd_k, need_k) || !ck_mul_u64(n_heads, hd_v, need_v) ||
        (uint64_t) k_off + need_k > k_row_stride || (uint64_t) v_off + need_v > v_row_stride) {
        if (err) *err = "stream_layer_canonical: layer slice exceeds workspace stride";
        return false;
    }
    const size_t head_need = (size_t) n * std::max(hd_k, hd_v);
    if (head.size() < head_need) {
        if (err) *err = "stream_layer_canonical: head scratch short (geometry changed)";
        return false;
    }
    for (uint32_t h = 0; h < n_heads; ++h) {
        if (!snap.read_k_head(h, head.data(), (size_t) n * hd_k, err)) {
            return false;
        }
        for (uint32_t r = 0; r < n; ++r) {
            std::memcpy(k_dst + (size_t) r * k_row_stride + k_off + (uint64_t) h * hd_k,
                          head.data() + (size_t) r * hd_k, hd_k * sizeof(float));
        }
        if (!snap.read_v_head(h, head.data(), (size_t) n * hd_v, err)) {
            return false;
        }
        for (uint32_t r = 0; r < n; ++r) {
            std::memcpy(v_dst + (size_t) r * v_row_stride + v_off + (uint64_t) h * hd_v,
                          head.data() + (size_t) r * hd_v, hd_v * sizeof(float));
        }
    }
    return true;
}

xkv_maintenance_outcome llama_xkv_runtime::maintain(
    llama_kv_cache & kv, uint32_t upcoming_tokens, bool forced, std::string * err) {
    std::lock_guard<std::mutex> family_lock(maint_mtx_);
    stats_.maintenance_runs++;
    std::string cfg_err;
    if (!eff_.valid(&cfg_err)) {
        stats_.record_error(cfg_err);
        if (err) *err = cfg_err;
        return xkv_maintain_error;
    }
    const bool is_shadow = (cparams_.xkv_mode == LLAMA_XKV_MODE_SHADOW);
    if (!is_shadow && cparams_.xkv_mode != LLAMA_XKV_MODE_DENSE &&
        cparams_.xkv_mode != LLAMA_XKV_MODE_SR) {
        if (err) *err = "maintain: runtime created for unsupported mode";
        return xkv_maintain_error;
    }
    // Factorization itself is CPU-reference here (host canonicalization +
    // store-side CPU factorization) or native Vulkan via the bridge below.
    // Supported factorizers: CPU_REFERENCE (host canonicalize + CPU SVD),
    // and VULKAN / VULKAN_HYBRID via the native bridge above. CUDA remains
    // unsupported and fails closed explicitly.
    if (cparams_.xkv_factorizer == LLAMA_XKV_FACTORIZER_CUDA) {
        stats_.record_skip(xkv_skip_reason::unsupported_config);
        if (err) *err = "maintain: CUDA factorizer is not supported on this build";
        return xkv_maintain_error;
    }
    auto store = kv.get_xkv_store();
    if (!store) {
        if (err) *err = "maintain: no XKV store";
        return xkv_maintain_error;
    }
    // Fixed Tri fill-first interaction: when a bounded hot pool exists and
    // forced=false, maintain() NEVER seals or rebinds while hot slots remain
    // free for the upcoming requirement. Fill-first requires keeping data
    // dense as long as physical space allows. Only when pressure hits
    // (hot_free < upcoming_tokens, or forced=true) does sealing proceed.
    // SHADOW mode evaluates without release and bypasses this check.
    auto pool = kv.get_hot_slot_pool();
    if (!is_shadow && pool && !forced) {
        const uint32_t hot_free = pool->get_free();
        const uint32_t req = upcoming_tokens > 0 ? upcoming_tokens : 1;
        if (hot_free >= req) {
            stats_.deferred_runs++;
            return xkv_maintain_no_action; // fill-first: preserve dense hot rows while space exists
        }
    }
    // Single-coordinator maintenance exclusion (seal/pack/MTP/fence readers
    // gate on the same coordinator). Contention defers, never half-seals.
    std::optional<xkv_maintenance_guard> mguard;
    if (coord_ != nullptr) {
        // Re-sync the exclusion gate to the authoritative store stamp
        // before taking it (never gate on a stale zero stamp).
        if (auto store0 = kv.get_xkv_store()) {
            coord_->sync_from_store(store0->current_stamp());
        }
        std::string merr;
        mguard.emplace(coord_, xkv_maintenance_op::seal, next_maint_id_++, &merr);
        if (!mguard->held()) {
            stats_.quiescence_skips++;
            stats_.record_error(merr);
            if (err) *err = merr;
            return xkv_maintain_retry_stale;
        }
    }
    // Thin stamp/pool validator used under the exclusion above.
    xkv_quiescence_guard guard(kv);
    const llama_hparams & hparams = kv.get_hparams();
    if (!ensure_groups(hparams, kv, err)) {
        return xkv_maintain_error;
    }
    // Size control scratch once per geometry (reserves on first call or
    // geometry growth only); every run after is zero-heap on this path.
    {
        size_t max_hd = 256;
        for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
            if (hparams.has_kv(il)) {
                max_hd = std::max<size_t>(max_hd, hparams.n_embd_head_k(il));
                max_hd = std::max<size_t>(max_hd, hparams.n_embd_v_gqa(il));
            }
        }
        const size_t need_groups = std::max<size_t>(groups_.groups.size(), 1);
        if (!scratch_.initialized || scratch_.capacity_rows < kv.get_hot_size() ||
            scratch_.capacity_groups < need_groups || scratch_.capacity_head_dim < max_hd) {
            scratch_.init(kv.get_hot_size(), need_groups, max_hd);
        }
    }
    scratch_.clear_for_run();
    std::string collect_err;
    if (!enumerate_into_scratch(kv, false, &collect_err)) {
        stats_.quiescence_skips++;
        stats_.record_error(collect_err);
        if (err) *err = collect_err;
        return xkv_maintain_retry_stale;
    }
    auto & eligible = scratch_.eligible;
    auto & deduped = scratch_.eligible_deduped;
    auto & order = scratch_.eligible_indices;
    auto & window_idx = scratch_.window_idx;
    auto & pids = scratch_.pids;
    auto & gens = scratch_.gens;
    auto & positions = scratch_.positions;
    const uint32_t seg = eff_.segment_tokens;
    // Partition by semantic sharing domain; oldest domain with a full
    // segment wins. Joint B factors never couple unrelated histories.
    // Sort deduped indices by (domain key, storage_pos, payload) with zero
    // heap (reused index array), then scan runs for the oldest full domain.
    order = deduped;
    auto domain_of = [&](size_t i, xkv_seal_domain_key & k) {
        const auto & r = eligible[i];
        k.stream = r.stream;
        k.visibility = r.visibility;
        k.episode_id = r.episode_id;
        k.run_id = r.run_id;
        k.publish_epoch = r.publish_epoch;
        k.seq_set_fp = (r.visibility == (uint8_t) llama_rerot_visibility::normal) ? r.seq_set_fp : 0;
    };
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        xkv_seal_domain_key ka, kb;
        domain_of(a, ka);
        domain_of(b, kb);
        if (ka < kb) return true;
        if (kb < ka) return false;
        if (eligible[a].storage_pos != eligible[b].storage_pos) {
            return eligible[a].storage_pos < eligible[b].storage_pos;
        }
        return eligible[a].payload_id < eligible[b].payload_id;
    });
    bool have_window = false;
    xkv_seal_domain_key chosen_key;
    int64_t best_oldest = INT64_MAX;
    size_t best_lo = 0;
    size_t best_len = 0;
    // Pass 1: look for a full segment domain (hi - lo >= seg).
    // Pass 2 (under forced/pressure deficit): pick the oldest/largest partial
    // domain with >= 1 rows; adaptive ranks and exact error/saving gates
    // decide if beneficial, preventing lossy preemption. Fill-first still
    // strictly forbids partial seals without pressure.
    // SHADOW observes the production seal geometry without publishing. It
    // never turns every safe-boundary call into a partial-segment factor run;
    // wait for a complete domain just as fill-first production would absent
    // physical pressure.
    bool under_pressure = forced && cparams_.xkv_mode != LLAMA_XKV_MODE_SHADOW;
    if (!under_pressure && pool) {
        const uint32_t req = upcoming_tokens > 0 ? upcoming_tokens : 1;
        under_pressure = (pool->get_free() < req);
    }
    for (size_t lo = 0; lo < order.size();) {
        size_t hi = lo + 1;
        xkv_seal_domain_key k0;
        domain_of(order[lo], k0);
        while (hi < order.size()) {
            xkv_seal_domain_key kh;
            domain_of(order[hi], kh);
            if (!(kh == k0)) break;
            ++hi;
        }
        if (hi - lo >= seg) {
            const int64_t oldest = eligible[order[lo]].storage_pos;
            if (!have_window || oldest < best_oldest || (oldest == best_oldest && k0 < chosen_key)) {
                have_window = true;
                best_oldest = oldest;
                best_lo = lo;
                best_len = seg;
                chosen_key = k0;
            }
        }
        lo = hi;
    }
    if (!have_window && under_pressure) {
        // Fallback under pressure: pick the largest (tie-break oldest) partial run.
        size_t max_run = 0;
        for (size_t lo = 0; lo < order.size();) {
            size_t hi = lo + 1;
            xkv_seal_domain_key k0;
            domain_of(order[lo], k0);
            while (hi < order.size()) {
                xkv_seal_domain_key kh;
                domain_of(order[hi], kh);
                if (!(kh == k0)) break;
                ++hi;
            }
            const size_t run_len = hi - lo;
            const int64_t oldest = eligible[order[lo]].storage_pos;
            if (run_len > max_run || (run_len == max_run && oldest < best_oldest)) {
                have_window = true;
                max_run = run_len;
                best_oldest = oldest;
                best_lo = lo;
                best_len = run_len;
                chosen_key = k0;
            }
            lo = hi;
        }
    }
    if (!have_window) {
        stats_.deferred_runs++;
        return xkv_maintain_no_action;
    }
    const uint32_t n = (uint32_t) best_len;
    window_idx.clear();
    for (uint32_t i = 0; i < n; ++i) {
        window_idx.push_back(order[best_lo + i]);
    }
    const uint32_t stream = eligible[window_idx[0]].stream;
    for (size_t wi : window_idx) {
        if (eligible[wi].stream != stream) {
            stats_.record_error("maintain: domain leaked across streams");
            if (err) *err = "maintain: domain leaked across streams";
            return xkv_maintain_error;
        }
    }
    stats_.last_domain_fp = fingerprint_seal_domain(chosen_key);
    pids.resize(n);
    gens.resize(n);
    positions.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        const auto & r = eligible[window_idx[i]];
        pids[i] = r.payload_id;
        gens[i] = r.generation;
        positions[i] = r.storage_pos;
    }
    // Device-residency preflight before any fallible work.
    xkv_bundle_sealing_params params = sealing_params();
    if (xkv_expected_residency(cparams_) == GGML_XKV_RES_DEVICE_OWNED) {
        const ggml_type stream_types[5] = {
            params.factor_a_k, params.factor_b_k, params.factor_a_v,
            params.factor_b_v, params.landmark_type,
        };
        for (int i = 0; i < 5; ++i) {
            if (!ggml_xkv_residency_supported(stream_types[i], GGML_XKV_RES_DEVICE_OWNED)) {
                stats_.record_skip(xkv_skip_reason::unsupported_config);
                if (err) *err = "maintain: factor/landmark codec is not device-residency capable";
                return xkv_maintain_error;
            }
        }
    }
    // Native bridge if and only if device residency is expected on a
    // Production native sealing path: build ONE device-local DAG via
    // xkv_native_seal_build, attach the returned immutable backend handles
    // to the pre-publish candidate, and publish atomically via
    // publish_candidate with the pool release precommit gate.
    // Production native sealing path: covers both TQ_FACTORS and
    // TQ_FACTORS_LANDMARKS device-owned profiles using xkv_native_seal_build.
    const bool use_native_bridge =
        (xkv_expected_residency(cparams_) == GGML_XKV_RES_DEVICE_OWNED);
    if (use_native_bridge) {
        // Single-backend native bridge: all groups must share one placement.
        // Split groups are CPU-sealable per group; native fails closed here
        // with exact layers/devices rather than scattering one bundle.
        const auto & gpl = group_placements_;
        if (!gpl.empty()) {
            for (size_t gi = 0; gi < groups_.groups.size() && gi < gpl.size(); ++gi) {
                if (gpl[gi] != gpl[0]) {
                    stats_.record_skip(xkv_skip_reason::unsupported_config);
                    if (err) {
                        *err = "maintain: native seal needs single-device placement, group " +
                            std::to_string(groups_.groups[gi].group_index) + " is [" +
                            gpl[gi] + "] vs [" + gpl[0] + "]";
                    }
                    return xkv_maintain_error;
                }
            }
        }
    }
    if (use_native_bridge) {
        xkv_native_seal_config ncfg;
        ncfg.n_rows = n;
        ncfg.physical_rows.reserve(n);
        for (size_t wi : scratch_.window_idx) {
            ncfg.physical_rows.push_back(scratch_.eligible[wi].physical_row);
        }
        ncfg.storage_positions.assign(positions.begin(), positions.end());
        ncfg.residency = GGML_XKV_RES_DEVICE_OWNED;
        ncfg.placement_identity = placement_id_;

        // 1. Borrow placement buft from the first target layer's hot tensor (no backend initialized yet!).
        ggml_tensor * k0 = kv.get_k_storage((int32_t) groups_.groups[0].owning_layers[0]);
        ggml_backend_buffer_type_t buft0 = (k0 && k0->buffer) ? ggml_backend_buffer_get_type(k0->buffer) : nullptr;
        ggml_backend_dev_t dev0 = buft0 ? ggml_backend_buft_get_device(buft0) : nullptr;
        if (!buft0 || !dev0) {
            stats_.record_skip(xkv_skip_reason::unsupported_config);
            if (err) *err = "maintain: native seal requires valid device buft/device on hot tensors";
            return xkv_maintain_error;
        }
        ncfg.buft = buft0;
        ncfg.id_gen = &store->allocation_id_generator();

        // 2. Refresh phase cache to supply exact phase specs to native groups.
        if (!refresh_phase_cache(kv, err)) {
            return xkv_maintain_error;
        }

        // 3. Populate all native group descriptors and geometry BEFORE calling estimator or reserving!
        for (size_t gi = 0; gi < groups_.groups.size(); ++gi) {
            const auto & g = groups_.groups[gi];
            xkv_native_seal_group ng;
            ng.group_index = g.group_index;
            ng.rank_k = xkv_scaled_group_rank(eff_.rank_k, (uint32_t) g.owning_layers.size(),
                eff_.group_size, n, g.total_dim_k);
            ng.rank_v = xkv_scaled_group_rank(eff_.rank_v, (uint32_t) g.owning_layers.size(),
                eff_.group_size, n, g.total_dim_v);
            ng.codec_a_k = cparams_.xkv_factor_a_k;
            ng.codec_b_k = cparams_.xkv_factor_b_k;
            ng.codec_a_v = cparams_.xkv_factor_a_v;
            ng.codec_b_v = cparams_.xkv_factor_b_v;
            // Canonical per-group seed: seed + group_index * 1000 (V pair is +1)
            // Aligned with CPU/store estimator and factorizer contracts.
            const uint64_t fseed_g = eff_.factor_seed + (uint64_t) g.group_index * 1000ULL;
            ng.seed_k = fseed_g;
            ng.seed_v = fseed_g + 1;
            // Landmarks on native bridge: enabled when storage profile requires them
            const bool build_native_lm = (cparams_.xkv_storage_profile == LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS);
            ng.want_landmarks = build_native_lm;
            ng.landmark_type = cparams_.xkv_landmark_type;
            ng.chunk_tokens = eff_.chunk_tokens;
            for (const auto & sp : cached_phase_specs_) {
                if (sp.group_index == g.group_index) {
                    ng.phase_tx_fingerprint = sp.fingerprint;
                    break;
                }
            }
            for (uint32_t ol : g.owning_layers) {
                xkv_native_seal_hot_layer lay;
                lay.hot_k = kv.get_k_storage((int32_t) ol);
                lay.hot_v = kv.get_v_storage((int32_t) ol);
                lay.n_heads = hparams.n_head_kv(ol);
                lay.head_dim_k = hparams.n_embd_head_k(ol);
                lay.head_dim_v = hparams.n_embd_head_v(ol);
                lay.rotary_dim_k = hparams.n_rot(ol);
                if (hparams.rope_type == LLAMA_ROPE_TYPE_NORM) {
                    lay.rope_mode_k = GGML_XKV_ROPE_INTERLEAVED;
                } else if (hparams.rope_type == LLAMA_ROPE_TYPE_NEOX ||
                           hparams.rope_type == LLAMA_ROPE_TYPE_MROPE ||
                           hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE) {
                    lay.rope_mode_k = GGML_XKV_ROPE_HALF;
                } else if (hparams.rope_type == LLAMA_ROPE_TYPE_NONE || lay.rotary_dim_k == 0) {
                    lay.rope_mode_k = GGML_XKV_ROPE_HALF; // rotary_dim_k == 0 governs passthrough
                } else {
                    if (err) *err = "maintain: unsupported rope type for native seal";
                    return xkv_maintain_error;
                }
                const uint32_t fc = lay.rotary_dim_k / 2;
                lay.rope_omega_k.assign(fc, 0.0f);
                lay.rope_mag_k.assign(fc, 1.0f);
                if (fc > 0 && hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
                    const float freq_base = (cparams_.rope_freq_base > 0.0f) ? kv.get_model().get_rope_freq_base(cparams_, (int) ol) : hparams.rope_freq_base_train;
                    const float freq_scale = (cparams_.rope_freq_scale > 0.0f) ? kv.get_model().get_rope_freq_scale(cparams_, (int) ol) : 1.0f;
                    const float attn_factor = (cparams_.yarn_attn_factor > 0.0f) ? cparams_.yarn_attn_factor : 1.0f;
                    if (!triattention_build_rope_tables(
                        lay.rope_omega_k.data(), lay.rope_mag_k.data(), lay.rotary_dim_k,
                        freq_base, freq_scale,
                        (int32_t) cparams_.n_ctx_orig_yarn,
                        cparams_.yarn_ext_factor, attn_factor,
                        cparams_.yarn_beta_fast, cparams_.yarn_beta_slow, nullptr)) {
                        if (err) *err = "maintain: failed to build native RoPE tables";
                        return xkv_maintain_error;
                    }
                    for (float & scale_sq : lay.rope_mag_k) {
                        if (!std::isfinite(scale_sq) || scale_sq < 0.0f) {
                            if (err) *err = "maintain: invalid native RoPE scale";
                            return xkv_maintain_error;
                        }
                        scale_sq = std::sqrt(scale_sq);
                    }
                }
                if (kv.get_attn_rot_k()) {
                    int32_t nr = kv.get_attn_rot_k_nrot();
                    const auto & hads = kv.get_attn_rot_hadamard();
                    auto hit = hads.find((int64_t) nr);
                    if (hit != hads.end()) lay.hadamard_k = hit->second;
                }
                if (kv.get_attn_rot_v()) {
                    int32_t nr = kv.get_attn_rot_v_nrot();
                    const auto & hads = kv.get_attn_rot_hadamard();
                    auto hit = hads.find((int64_t) nr);
                    if (hit != hads.end()) lay.hadamard_v = hit->second;
                }
                ng.layers.push_back(std::move(lay));
            }

            xkv_factor_group_input estimate_in;
            estimate_in.group_index = g.group_index;
            estimate_in.rank_k = ng.rank_k;
            estimate_in.rank_v = ng.rank_v;
            estimate_in.k_rows = n;
            estimate_in.v_rows = n;
            estimate_in.owning_layers = g.owning_layers;
            uint64_t off_k = 0;
            uint64_t off_v = 0;
            for (const auto & lay : ng.layers) {
                uint64_t dim_k = 0;
                uint64_t dim_v = 0;
                if (!ck_mul_u64(lay.n_heads, lay.head_dim_k, dim_k) ||
                    !ck_mul_u64(lay.n_heads, lay.head_dim_v, dim_v) ||
                    off_k > UINT32_MAX || off_v > UINT32_MAX ||
                    dim_k > UINT32_MAX || dim_v > UINT32_MAX ||
                    !ck_add_u64(off_k, dim_k, off_k) ||
                    !ck_add_u64(off_v, dim_v, off_v) ||
                    off_k > UINT32_MAX || off_v > UINT32_MAX) {
                    if (err) *err = "maintain: native group feature geometry overflow";
                    return xkv_maintain_error;
                }
                estimate_in.layer_feature_offsets_k.push_back((uint32_t) (off_k - dim_k));
                estimate_in.layer_feature_dims_k.push_back((uint32_t) dim_k);
                estimate_in.layer_feature_offsets_v.push_back((uint32_t) (off_v - dim_v));
                estimate_in.layer_feature_dims_v.push_back((uint32_t) dim_v);
            }
            estimate_in.total_dim_k = (uint32_t) off_k;
            estimate_in.total_dim_v = (uint32_t) off_v;
            estimate_in.k_cols = off_k;
            estimate_in.v_cols = off_v;
            scratch_.inputs.push_back(std::move(estimate_in));
            ncfg.groups.push_back(std::move(ng));
        }

        // 4. Pure estimation via xkv_native_seal_estimate (no backend needed!).
        xkv_native_seal_estimate_result nest;
        std::string nest_err;
        if (!xkv_native_seal_estimate(ncfg, nest, &nest_err)) {
            stats_.record_skip(xkv_skip_reason::unsupported_config);
            if (err) *err = "maintain: xkv_native_seal_estimate failed: " + nest_err;
            return xkv_maintain_error;
        }

        // 5. Acquire two independent RAII reservations based on exact estimates:
        // SHADOW special case: no persistent store reservation; its would-be dest
        // allocations are transient, so reserve device staging for max_transient + persistent_dest.
        // Production (DENSE/SR): reserve staging for max_transient_bytes and store for persistent + metadata.
        xkv_capacity_reservation native_cap_res;
        xkv_device_staging_reservation native_staging_res;
        xkv_backend_store_reservation b_res = {};

        if (is_shadow) {
            uint64_t shadow_staging_bytes = 0;
            if (!ck_add_u64(nest.max_transient_bytes, nest.persistent_dest_bytes, shadow_staging_bytes)) {
                stats_.record_skip(xkv_skip_reason::unsupported_config);
                if (err) *err = "maintain: shadow staging bytes overflow";
                return xkv_maintain_error;
            }
            size_t s_deficit = 0;
            std::string s_err;
            native_staging_res = store->reserve_device_staging(
                (size_t) shadow_staging_bytes, &s_err, &s_deficit);
            if (store->get_arena().get_capacity_bytes() > 0 && !native_staging_res.valid()) {
                stats_.record_skip(xkv_skip_reason::preflight_oom);
                if (err) *err = "maintain: native shadow staging reservation failed: " + s_err;
                return xkv_maintain_error;
            }
            ncfg.staging_reservation = &native_staging_res;
            ncfg.store_reservation = nullptr;
        } else {
            // The native estimator covers device stream payloads only.  The
            // store estimator covers the same encoded streams plus host-side
            // segment/group metadata.  Reserve native destination bytes plus
            // only that metadata delta; max()/fallback would under-reserve
            // whenever native stream alignment exceeds the host encoding.
            size_t host_total_bytes = 0;
            size_t host_stream_bytes = 0;
            std::string est_m_err;
            if (!estimate_segment_bundle_persistent_bytes(
                    scratch_.inputs, params, &host_total_bytes, &est_m_err, &host_stream_bytes)) {
                stats_.record_skip(xkv_skip_reason::unsupported_config);
                if (err) *err = "maintain: native store metadata estimate failed: " + est_m_err;
                return xkv_maintain_error;
            }
            if (host_total_bytes < host_stream_bytes) {
                stats_.record_skip(xkv_skip_reason::unsupported_config);
                if (err) *err = "maintain: native store metadata estimate is inconsistent";
                return xkv_maintain_error;
            }
            const size_t metadata_bytes = host_total_bytes - host_stream_bytes;
            if (nest.persistent_dest_bytes > std::numeric_limits<size_t>::max() - metadata_bytes) {
                stats_.record_skip(xkv_skip_reason::unsupported_config);
                if (err) *err = "maintain: native persistent reservation overflow";
                return xkv_maintain_error;
            }
            const size_t persistent_total = nest.persistent_dest_bytes + metadata_bytes;
            size_t native_deficit = 0;
            std::string native_res_err;
            native_cap_res = store->reserve_capacity(persistent_total, &native_res_err, &native_deficit);
            if (store->store_capacity_bytes() > 0 && !native_cap_res.valid()) {
                stats_.record_skip(xkv_skip_reason::preflight_oom);
                if (err) *err = "maintain: native store capacity reservation failed (deficit " +
                                std::to_string(native_deficit) + " bytes): " + native_res_err;
                return xkv_maintain_error;
            }
            b_res = native_cap_res.backend_reservation();
            ncfg.store_reservation = &b_res;

            size_t staging_deficit = 0;
            std::string staging_err;
            native_staging_res = store->reserve_device_staging(
                nest.max_transient_bytes, &staging_err, &staging_deficit);
            if (store->get_arena().get_capacity_bytes() > 0 && !native_staging_res.valid()) {
                stats_.record_skip(xkv_skip_reason::preflight_oom);
                if (err) *err = "maintain: native staging reservation failed (deficit " +
                                std::to_string(staging_deficit) + " bytes): " + staging_err;
                return xkv_maintain_error;
            }
            ncfg.staging_reservation = &native_staging_res;
        }

        // 6. Obtain the KV-cache-owned persistent executor per device/buft:
        // avoids executor churn/leaks every segment and shares backend ownership.
        ncfg.executor = kv.get_xkv_executor(buft0);
        if (!ncfg.executor) {
            stats_.record_skip(xkv_skip_reason::unsupported_config);
            if (err) *err = "maintain: persistent XKV backend executor unavailable";
            return xkv_maintain_error;
        }
        ncfg.backend = ncfg.executor.get();

        xkv_native_seal_bundle nbundle;
        std::string nerr;
        if (is_shadow) {
            // Native SHADOW evaluate-only: executes the full Vulkan DAG
            // (canonicalize-hot, device factorize, singular telemetry, gates),
            // records measured telemetry and timing, and discards all
            // candidate device allocations without candidate marking,
            // publication, or pool release. Store/hot epochs stay untouched.
            if (!xkv_native_seal_build(ncfg, nbundle, &nerr)) {
                stats_.record_skip(xkv_skip_reason::factorization_failed);
                stats_.record_error(nerr);
                if (err) *err = "maintain: native shadow build failed: " + nerr;
                return xkv_maintain_error;
            }
            // Run the same numerical / telemetry gates as publish:
            for (const auto & sg : nbundle.groups) {
                if (sg.status_canon != 0 || sg.status_fact_k != 0 || sg.status_fact_v != 0) {
                    stats_.record_skip(xkv_skip_reason::factorization_failed);
                    if (err) *err = "maintain: native shadow group status nonzero";
                    return xkv_maintain_error;
                }
            }
            stats_.shadow_evals++;
            stats_.sync_total += nbundle.sync_count;
            stats_.last_factored_bytes = nbundle.factored_bytes;
            stats_.last_factored_rows = n;
            // nbundle goes out of scope here: all device stream allocations
            // are freed cleanly; zero mutation of store/cells/pool state.
            return xkv_maintain_evaluated;
        }
        // Store-owned monotonic allocation-id generator (never null, never
        // reset): null would collide backend allocation ids across seals.
        ncfg.id_gen = &store->allocation_id_generator();

        // Preflight hot release plan BEFORE marking seal candidates!
        // validate_hot_release checks that payloads are hot_committed (nonce==0);
        // once marked as seal_candidate with a nonzero nonce, validate_hot_release
        // strictly rejects them as locked in a seal transaction.
        const xkv_snapshot_stamp pre_stamp = store->current_stamp();
        auto & pre_plan = scratch_.release_plan;
        pre_plan.released_payload_ids.assign(pids.begin(), pids.end());
        pre_plan.released_generations.assign(gens.begin(), gens.end());
        pre_plan.released_physical_rows.clear();
        for (size_t wi : scratch_.window_idx) {
            pre_plan.released_physical_rows.push_back(scratch_.eligible[wi].physical_row);
        }
        pre_plan.stamp = pre_stamp;
        if (!kv.validate_hot_release(pre_plan, err)) {
            return xkv_maintain_error;
        }

        // Mark payloads as seal_candidates under generation checks before
        // publishing; any failure rolls them back to hot_committed atomically.
        // Mark payloads as seal_candidates under atomic (pid, generation)
        // validation; the tx nonce scopes the guard's abort below so a stale
        // abort can never revert a later overlapping seal.
        uint64_t seal_nonce = 0;
        if (!store->mark_seal_candidates(pids, gens, &seal_nonce, err)) {
            stats_.record_skip(xkv_skip_reason::not_committed);
            if (err && err->empty()) *err = "maintain: mark_seal_candidates failed";
            return xkv_maintain_error;
        }
        struct candidate_scope_guard {
            llama_xkv_cache_store * store;
            const std::vector<uint64_t> & pids;
            uint64_t nonce = 0;
            bool active = true;
            ~candidate_scope_guard() {
                if (active && store) {
                    store->abort_seal_candidates(pids, nonce, xkv_skip_reason::aborted);
                }
            }
        } cand_guard{store.get(), pids, seal_nonce, true};
        if (!xkv_native_seal_build(ncfg, nbundle, &nerr)) {
            stats_.record_skip(xkv_skip_reason::factorization_failed);
            stats_.record_error(nerr);
            if (err) *err = "maintain: native seal build failed: " + nerr;
            return xkv_maintain_error;
        }
        // Build candidate groups matching the native bundle streams.
        uint64_t native_baseline = 0;
        for (const auto & g : groups_.groups) {
            for (uint32_t ol : g.owning_layers) {
                ggml_tensor * kt = kv.get_k_storage((int32_t) ol);
                ggml_tensor * vt = kv.get_v_storage((int32_t) ol);
                if (!kt || !vt) {
                    stats_.record_skip(xkv_skip_reason::unsupported_config);
                    if (err) *err = "maintain: missing hot tensors for native baseline";
                    return xkv_maintain_error;
                }
                uint64_t rb = 0;
                if (!ck_add_u64(ggml_row_size(kt->type, kt->ne[0]),
                                ggml_row_size(vt->type, vt->ne[0]), rb) ||
                    !ck_mul_u64(rb, n, rb) || !ck_add_u64(native_baseline, rb, native_baseline)) {
                    stats_.record_skip(xkv_skip_reason::unsupported_config);
                    if (err) *err = "maintain: native baseline overflow";
                    return xkv_maintain_error;
                }
            }
        }
        std::vector<xkv_factor_group_payload> cand_groups;
        for (size_t gi = 0; gi < groups_.groups.size(); ++gi) {
            const auto & g = groups_.groups[gi];
            const auto & sg = nbundle.groups[gi];
            xkv_factor_group_payload cg;
            cg.group_index = g.group_index;
            cg.owning_layers = g.owning_layers;
            cg.rank_k = sg.rank_k;
            cg.rank_v = sg.rank_v;
            cg.layer_feature_offsets_k = g.layer_feature_offsets_k;
            cg.layer_feature_dims_k = g.layer_feature_dims_k;
            cg.layer_feature_offsets_v = g.layer_feature_offsets_v;
            cg.layer_feature_dims_v = g.layer_feature_dims_v;
            cg.total_dim_k = g.total_dim_k;
            cg.total_dim_v = g.total_dim_v;
            cg.a_k.desc = sg.a_k.desc;
            cg.set_b_k(encoded_matrix{sg.b_k.desc, {}});
            cg.a_v.desc = sg.a_v.desc;
            cg.set_b_v(encoded_matrix{sg.b_v.desc, {}});
            // Measured per-group hot row bytes and baseline
            uint64_t g_row_bytes = 0;
            for (uint32_t ol : g.owning_layers) {
                ggml_tensor * kt = kv.get_k_storage((int32_t) ol);
                ggml_tensor * vt = kv.get_v_storage((int32_t) ol);
                if (kt && vt) {
                    g_row_bytes += ggml_row_size(kt->type, kt->ne[0]) + ggml_row_size(vt->type, vt->ne[0]);
                }
            }
            cg.baseline_original_row_bytes = g_row_bytes;
            cg.baseline_original_bytes = g_row_bytes * n;
            // Set landmark metadata on candidate group if native landmarks built
            if (sg.has_landmarks) {
                cg.landmark.desc = sg.landmark.desc;
                cg.landmark_chunks.clear();
                const uint32_t chunk_t = eff_.chunk_tokens;
                if (chunk_t == 0 || sg.landmark_eb.size() != sg.landmark_srcfp.size()) {
                    stats_.record_skip(xkv_skip_reason::codec_error);
                    if (err) *err = "maintain: native landmark telemetry size mismatch";
                    return xkv_maintain_error;
                }
                for (size_t ci = 0; ci < sg.landmark_eb.size(); ++ci) {
                    xkv_landmark_chunk lmc;
                    lmc.row_begin = (uint32_t)(ci * chunk_t);
                    lmc.row_count = (uint32_t)(std::min<uint64_t>(n, (ci + 1) * chunk_t) - lmc.row_begin);
                    lmc.error_bound = sg.landmark_eb[ci];
                    lmc.source_fingerprint = sg.landmark_srcfp[ci];
                    cg.landmark_chunks.push_back(lmc);
                }
                cg.landmark_table_fingerprint = compute_landmark_table_fingerprint(cg.landmark_chunks);
            }
            cand_groups.push_back(std::move(cg));
        }
        auto cand = store->create_candidate_segment(
            cparams_.xkv_storage_profile, cparams_.xkv_source, cand_groups);
        if (!cand) {
            stats_.record_skip(xkv_skip_reason::preflight_oom);
            if (err) *err = "maintain: candidate segment creation failed";
            return xkv_maintain_error;
        }
        cand->residency = GGML_XKV_RES_DEVICE_OWNED;
        cand->descriptor_fingerprint = nbundle.bundle_fingerprint;
        cand->profile_fingerprint = config_fingerprint();
        cand->layer_group_map_fingerprint = compute_layer_group_map_fingerprint(cand_groups);
        cand->n_rows = n;
        cand->n_live_rows = n;
        cand->baseline_original_bytes = native_baseline;
        cand->row_payload_ids = pids;
        cand->live_rows.assign(n, true);
        // Native gates (same bar as CPU): min-saving on exact bytes,
        // finite/numerical checks on singular telemetry and build status.
        // Reject/abort rolls candidates back; rows stay hot and bound.
        for (const auto & sg : nbundle.groups) {
            if (sg.status_canon != 0 || sg.status_fact_k != 0 || sg.status_fact_v != 0) {
                stats_.record_skip(xkv_skip_reason::factorization_failed);
                if (err) *err = "maintain: native group build status nonzero";
                return xkv_maintain_error;
            }
            for (float s : sg.singular_k) {
                if (!std::isfinite(s) || s < 0.0f) {
                    stats_.record_skip(xkv_skip_reason::factorization_failed);
                    if (err) *err = "maintain: native singular telemetry non-finite";
                    return xkv_maintain_error;
                }
            }
            for (float s : sg.singular_v) {
                if (!std::isfinite(s) || s < 0.0f) {
                    stats_.record_skip(xkv_skip_reason::factorization_failed);
                    if (err) *err = "maintain: native singular telemetry non-finite";
                    return xkv_maintain_error;
                }
            }
        }
        // Directly attach the verified, committed adoption bundle from the bridge:
        if (!nbundle.backend_bundle || !nbundle.backend_bundle->is_success() || !nbundle.backend_bundle->is_committed()) {
            stats_.record_skip(xkv_skip_reason::aborted);
            if (err) *err = "maintain: native seal returned invalid or uncommitted backend_bundle";
            return xkv_maintain_error;
        }
        cand->backend_bundle = nbundle.backend_bundle;
        cand->update_byte_counters();

        // Exact candidate incremental live bytes under store accounting:
        // dedupes shared B and backend allocation IDs + includes metadata and alignment.
        size_t cand_inc_bytes = 0;
        std::string inc_err;
        if (!store->candidate_incremental_bytes(cand, &cand_inc_bytes, &inc_err)) {
            stats_.record_skip(xkv_skip_reason::aborted);
            if (err) *err = "maintain: candidate_incremental_bytes failed: " + inc_err;
            return xkv_maintain_error;
        }
        const uint64_t native_saved = (native_baseline > cand_inc_bytes)
            ? (native_baseline - cand_inc_bytes) : 0;
        const double native_ratio = native_baseline > 0
            ? (double) native_saved / (double) native_baseline : 0.0;
        if (native_saved < eff_.min_saving_bytes || native_ratio < eff_.min_saving_ratio) {
            stats_.record_skip(xkv_skip_reason::no_saving);
            if (err) *err = "maintain: native bundle below min-saving gate (baseline=" +
                            std::to_string(native_baseline) + " inc=" + std::to_string(cand_inc_bytes) + ")";
            return xkv_maintain_error;
        }

        // Validate candidate before hot precommit
        std::string cand_val_err;
        if (!store->validate_candidate(cand, &cand_val_err)) {
            stats_.record_skip(xkv_skip_reason::aborted);
            stats_.record_error(cand_val_err);
            if (err) *err = "maintain: candidate validation failed: " + cand_val_err;
            return xkv_maintain_error;
        }

        // Precommit gate: runs pool release atomically with the store publish commit.
        // Reference capture: pre_plan aliases scratch_.release_plan (a runtime
        // member outliving this synchronous publish); the closure stays within
        // std::function SSO. Expected ids are stamped on the scratch plan
        // itself, which is rebuilt every maintain before validation.
        xkv_seal_precommit_fn ngate = [&kv, &pre_plan](
            const xkv_seal_precommit_ctx & pctx, std::string * gerr) -> bool {
            pre_plan.expected_segment_id = pctx.segment_id;
            pre_plan.expected_segment_version = pctx.segment_version;
            return kv.commit_hot_release(pre_plan, gerr);
        };
        std::string pub_err;
        if (!store->publish_candidate(cand, pids, gens, &pub_err, nullptr, std::move(ngate), &native_cap_res)) {
            stats_.record_skip(xkv_skip_reason::aborted);
            stats_.record_error(pub_err);
            if (err) *err = "maintain: native candidate publish failed: " + pub_err;
            return xkv_maintain_error;
        }
        cand_guard.active = false; // committed: do not abort candidates
        stats_.sealed_segments++;
        stats_.sealed_rows += n;
        stats_.sync_total += nbundle.sync_count + nbundle.adopt_sync_count;
        stats_.last_factored_bytes = nbundle.factored_bytes;
        stats_.last_factored_rows = n;
        stats_.compression_goal_met = true;
        return xkv_maintain_sealed;
    }
    // Max-group streaming workspace accounting: size the arena lease to the
    // SINGLE LARGEST group's n * (Dk + Dv) floats. One reusable arena slice
    // is populated on demand per group via params.canonical_source, eliminating
    // cumulative multi-group dense memory bloat completely.
    uint64_t max_group_elems = 0;
    uint64_t max_group_dim_k = 0;
    uint64_t max_group_dim_v = 0;
    for (const auto & g : groups_.groups) {
        uint64_t gk = 0, gv = 0, g_tot = 0;
        if (!ck_mul_u64(n, g.total_dim_k, gk) || !ck_mul_u64(n, g.total_dim_v, gv) ||
            !ck_add_u64(gk, gv, g_tot)) {
            stats_.record_skip(xkv_skip_reason::unsupported_config);
            if (err) *err = "maintain: workspace size overflow";
            return xkv_maintain_error;
        }
        max_group_elems = std::max(max_group_elems, g_tot);
        max_group_dim_k = std::max(max_group_dim_k, (uint64_t) g.total_dim_k);
        max_group_dim_v = std::max(max_group_dim_v, (uint64_t) g.total_dim_v);
    }
    uint64_t ws_bytes = 0;
    if (!ck_mul_u64(max_group_elems, sizeof(float), ws_bytes) || ws_bytes > (uint64_t) SIZE_MAX) {
        stats_.record_skip(xkv_skip_reason::unsupported_config);
        if (err) *err = "maintain: workspace size overflow";
        return xkv_maintain_error;
    }
    if (!store->get_arena().preflight((size_t) ws_bytes)) {
        stats_.record_skip(xkv_skip_reason::preflight_oom);
        if (err) *err = "maintain: workspace arena cannot cover the segment";
        return xkv_maintain_error;
    }
    xkv_arena_lease ws_lease = store->acquire_workspace_lease((size_t) ws_bytes);
    if (!ws_lease.valid()) {
        stats_.record_skip(xkv_skip_reason::preflight_oom);
        if (err) *err = "maintain: workspace lease refused";
        return xkv_maintain_error;
    }
    float * ws_ptr = ws_lease.as<float>();
    if (!ws_ptr) {
        stats_.record_skip(xkv_skip_reason::preflight_oom);
        if (err) *err = "maintain: workspace arena lease has null buffer";
        return xkv_maintain_error;
    }

    // Single reusable 64B-aligned max-group slice: zero cumulative offset!
    // K starts at 0, V starts immediately after max_group_dim_k * n floats.
    float * const single_k_slice = ws_ptr;
    float * const single_v_slice = ws_ptr + (size_t) n * max_group_dim_k;

    // Quiescence verification before sealing
    if (!guard.validate(kv, err)) {
        stats_.quiescence_skips++;
        return xkv_maintain_retry_stale;
    }

    canonical_batch_stats bstats;

    // Lazy canonical group source: streams exactly ONE requested group into
    // single_k_slice / single_v_slice at the moment seal_segment_bundle is ready
    // to factorize it. Pointers are borrowed for that group iteration only;
    // store factorizes and encodes into the off-side candidate before the next call.
    params.canonical_source = [&](
        uint32_t group_idx,
        const float ** out_k,
        const float ** out_v,
        std::string * src_err) -> bool {
        const layer_group * target_grp = nullptr;
        for (const auto & g : groups_.groups) {
            if (g.group_index == group_idx) {
                target_grp = &g;
                break;
            }
        }
        if (!target_grp) {
            if (src_err) *src_err = "canonical_source: unknown group index " + std::to_string(group_idx);
            return false;
        }
        const layer_group & g = *target_grp;
        if (g.owning_layers.size() != g.layer_feature_dims_k.size() ||
            g.owning_layers.size() != g.layer_feature_dims_v.size()) {
            if (src_err) *src_err = "canonical_source: group feature map size mismatch";
            return false;
        }
        // Zero the reusable single-group slice
        std::memset(single_k_slice, 0, (size_t) n * g.total_dim_k * sizeof(float));
        std::memset(single_v_slice, 0, (size_t) n * g.total_dim_v * sizeof(float));
        for (size_t li = 0; li < g.owning_layers.size(); ++li) {
            if (!stream_layer_canonical(kv, g.owning_layers[li],
                    single_k_slice, g.total_dim_k, g.layer_feature_offsets_k[li],
                    single_v_slice, g.total_dim_v, g.layer_feature_offsets_v[li],
                    &bstats, src_err)) {
                return false;
            }
        }
        *out_k = single_k_slice;
        *out_v = single_v_slice;
        return true;
    };

    // Seal inputs: topology (layer maps, hot-row bytes) rebuilt only when
    // groups change; per-run refresh touches ranks + data pointers only.
    // Dense data pointers (canonical_k_data/canonical_v_data) are left NULL;
    // store factorizer obtains them via params.canonical_source sequentially!
    if (scratch_inputs_version_ != groups_version_) {
        scratch_.inputs.clear();
        for (size_t gi = 0; gi < groups_.groups.size(); ++gi) {
            const layer_group & g = groups_.groups[gi];
            xkv_factor_group_input in;
            in.group_index = g.group_index;
            in.owning_layers = g.owning_layers;
            in.total_dim_k = g.total_dim_k;
            in.total_dim_v = g.total_dim_v;
            in.layer_feature_offsets_k = g.layer_feature_offsets_k;
            in.layer_feature_dims_k = g.layer_feature_dims_k;
            in.layer_feature_offsets_v = g.layer_feature_offsets_v;
            in.layer_feature_dims_v = g.layer_feature_dims_v;
            in.hot_bytes_per_row_k.assign(g.owning_layers.size(), 0);
            in.hot_bytes_per_row_v.assign(g.owning_layers.size(), 0);
            scratch_.inputs.push_back(std::move(in));
        }
        scratch_inputs_version_ = groups_version_;
    }
    std::vector<xkv_factor_group_input> & inputs = scratch_.inputs;
    for (size_t gi = 0; gi < groups_.groups.size(); ++gi) {
        const layer_group & g = groups_.groups[gi];
        xkv_factor_group_input & in = inputs[gi];
        for (size_t li = 0; li < g.owning_layers.size(); ++li) {
            ggml_tensor * kt = kv.get_k_storage((int32_t) g.owning_layers[li]);
            ggml_tensor * vt = kv.get_v_storage((int32_t) g.owning_layers[li]);
            if (!kt || !vt) {
                if (err) *err = "maintain: missing hot tensors for baseline measure";
                return xkv_maintain_error;
            }
            const uint64_t bk = ggml_row_size(kt->type, kt->ne[0]);
            const uint64_t bv = ggml_row_size(vt->type, vt->ne[0]);
            if (bk == 0 || bv == 0) {
                if (err) *err = "maintain: degenerate hot row bytes";
                return xkv_maintain_error;
            }
            in.hot_bytes_per_row_k[li] = bk;
            in.hot_bytes_per_row_v[li] = bv;
        }
        in.rank_k = xkv_scaled_group_rank(eff_.rank_k, (uint32_t) g.owning_layers.size(),
            eff_.group_size, n, g.total_dim_k);
        in.rank_v = xkv_scaled_group_rank(eff_.rank_v, (uint32_t) g.owning_layers.size(),
            eff_.group_size, n, g.total_dim_v);
        if (in.rank_k == 0 || in.rank_v == 0) {
            if (err) *err = "maintain: degenerate group rank";
            return xkv_maintain_error;
        }
        // Null pointers: store consumes via params.canonical_source per group!
        in.canonical_k_data = nullptr;
        in.k_rows = n;
        in.k_cols = g.total_dim_k;
        in.canonical_v_data = nullptr;
        in.v_rows = n;
        in.v_cols = g.total_dim_v;
        // Exact int64 window positions (no narrowing; store validates size).
        in.row_positions.assign(positions.begin(), positions.end());
    }
    params.expected_group_count = (uint32_t) inputs.size();
    params.expected_group_map_fingerprint = compute_layer_group_map_fingerprint(inputs);

    // Exact per-group phase specs for landmark construction.
    // R3 + R0/R1: reference profile always builds configured landmarks
    // (test/shadow/ablation-only, incl. dense; FP16 without selection);
    // production tq-factors stays landmark-free; tq-factors-landmarks builds.
    const bool need_landmarks =
        (params.profile != LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS);
    if (need_landmarks) {
        // Topology (inputs + phase specs) is rebuilt only when groups change;
        // per-run work only refreshes data pointers/ranks (zero heap after).
        // (Inputs topology is built alongside below; phase specs cached here.)
        if (scratch_phase_version_ != groups_version_) {
            scratch_.phase_specs.clear();
            for (const auto & g : groups_.groups) {
                xkv_group_phase_spec spec;
                spec.group_index = g.group_index;
                if (!build_landmark_phase_layers(kv, cparams_, hparams, g, spec.layers, spec.fingerprint, err)) {
                    return xkv_maintain_error;
                }
                scratch_.phase_specs.push_back(std::move(spec));
            }
            scratch_phase_version_ = groups_version_;
        }
        const ggml_type lm_type = cparams_.xkv_landmark_type;
        const uint32_t chunk = eff_.chunk_tokens;
        // Size landmark scratch once per geometry (retain capacity after).
        uint64_t lm_max_dim = 0;
        for (const auto & g : groups_.groups) {
            lm_max_dim = std::max(lm_max_dim, (uint64_t) g.total_dim_k);
        }
        const uint64_t lm_chunks = (n + chunk - 1) / chunk;
        try {
            scratch_.lm_recon.resize((size_t) lm_max_dim);
            scratch_.lm_phased.resize((size_t) lm_max_dim);
            scratch_.lm_accum.assign((size_t)(lm_chunks * lm_max_dim), 0.0f);
            scratch_.lm_identity_rows.resize(n);
        } catch (const std::bad_alloc &) {
            if (err) *err = "maintain: landmark scratch allocation failed";
            return xkv_maintain_error;
        }
        for (uint32_t r = 0; r < n; ++r) {
            scratch_.lm_identity_rows[r] = r;
        }
        params.landmark_factory = [this, lm_type, chunk](
            uint32_t group_index,
            const encoded_matrix & enc_a_k,
            const encoded_matrix & enc_b_k,
            const int64_t * row_positions,
            uint64_t span_n_rows,
            factor_workspace_span scratch,
            encoded_matrix & out_lm,
            std::vector<xkv_landmark_chunk> & out_chunks,
            std::string * factory_err) -> bool {
            out_chunks.clear();
            const xkv_group_phase_spec * spec = nullptr;
            for (const auto & s : scratch_.phase_specs) {
                if (s.group_index == group_index) {
                    spec = &s;
                    break;
                }
            }
            if (!spec) {
                if (factory_err) *factory_err = "landmark factory: group lacks a phase spec";
                return false;
            }
            // Exact passed int64 positions (no narrowing, no substitution).
            if (!row_positions || span_n_rows == 0) {
                if (factory_err) *factory_err = "landmark factory: row count mismatch";
                return false;
            }
            // Phase uniformity: one shared table set must cover every layer
            // slice for the single phase closure; heterogeneous groups refuse.
            if (spec->layers.empty()) {
                if (factory_err) *factory_err = "landmark factory: empty phase spec";
                return false;
            }
            const auto & ref0 = spec->layers[0];
            for (const auto & l : spec->layers) {
                if (l.head_dim != ref0.head_dim || l.rotary_dim != ref0.rotary_dim ||
                    l.omega != ref0.omega || l.freq_scale_sq != ref0.freq_scale_sq) {
                    if (factory_err) *factory_err = "landmark factory: heterogeneous rotary across group";
                    return false;
                }
            }
            // Zero-heap chunk encoder: only F32 is available in-module.
            // Per-layer slices for the bounded builder (reused scratch).
            auto & blayers = scratch_.lm_spec_layers;
            blayers.clear();
            for (const auto & l : spec->layers) {
                // Resolve owning layer + head split from the group map.
                landmark_build_layer bl;
                bl.feature_offset = l.feature_offset;
                bl.feature_dim = l.feature_dim;
                bl.kv_head = 0;
                bl.owning_layer = 0;
                for (const auto & g : groups_.groups) {
                    if (g.group_index != group_index) {
                        continue;
                    }
                    for (size_t li = 0; li < g.owning_layers.size(); ++li) {
                        if (g.layer_feature_offsets_k[li] == l.feature_offset) {
                            bl.owning_layer = g.owning_layers[li];
                            break;
                        }
                    }
                }
                blayers.push_back(bl);
            }
            // Identity surviving rows (full segment, row order).
            auto & ident = scratch_.lm_identity_rows;
            ident.clear();
            for (uint64_t r = 0; r < span_n_rows; ++r) {
                ident.push_back((uint32_t) r);
            }
            // Phase closure over the validated-uniform shared tables.
            // arrive row-major/layer-minor; a cursor maps each call to its
            // layer slice, and the position periodicity (pos repeats every
            // n_layers) verifies the order — any divergence fails closed.
            struct phase_cursor { size_t calls = 0; bool ok = true; };
            phase_cursor cursor;
            const size_t nL = blayers.size();
            const uint32_t uhd = ref0.head_dim;
            const uint32_t urd = ref0.rotary_dim;
            const float * uom = ref0.omega.data();
            const float * ufs = ref0.freq_scale_sq.data();
            phase_transform_fn phase_tx = [&](
                const float * src, int64_t pos, uint32_t /*head_idx*/, float * dst) {
                const size_t k = cursor.calls++;
                const size_t li = (nL > 0) ? (k % nL) : 0;
                const uint64_t row = (nL > 0) ? (k / nL) : 0;
                if (nL == 0 || li >= blayers.size() || row >= span_n_rows ||
                    pos != row_positions[row]) {
                    cursor.ok = false;
                    return;
                }
                // Width from our own slice table (exact, never inferred).
                const uint32_t w = blayers[li].feature_dim;
                if (w == 0 || w % uhd != 0) {
                    cursor.ok = false;
                    return;
                }
                for (uint32_t h = 0; h < w / uhd; ++h) {
                    apply_forward_rope_head(
                        src + (uint64_t) h * uhd, dst + (uint64_t) h * uhd,
                        uhd, urd, uom, ufs, pos);
                }
            };
            landmark_build_spec bspec;
            bspec.a_k = &enc_a_k;
            bspec.b_k = &enc_b_k;
            bspec.surviving_rows = scratch_.lm_identity_rows.data();
            bspec.n_rows = span_n_rows;
            bspec.storage_positions = row_positions;
            bspec.layers = blayers.data();
            bspec.n_layers = blayers.size();
            bspec.chunk_tokens = chunk;
            bspec.landmark_type = lm_type;
            bspec.phase_tx_fingerprint = spec->fingerprint;
            bspec.seed_extra = eff_.fingerprint();
            size_t ws_need = 0, arena_need = 0, n_chunks = 0;
            // Encoder dispatch: production zero-heap F32/Q8/Turbo4 encoder,
            // local zero-heap F16 encoder; all other types fail closed.
            landmark_chunk_encoder_fn chunk_enc = nullptr;
            if (lm_type == GGML_TYPE_F16) {
                chunk_enc = landmark_encode_f16_row;
            } else {
                chunk_enc = landmark_encode_quant_row;
            }
            if (!landmark_build_workspace_bytes(bspec, chunk_enc,
                    ws_need, arena_need, n_chunks, factory_err)) {
            return false;
            }
            if (n_chunks == 0 || !scratch.data || scratch.size_bytes < ws_need + arena_need) {
                if (factory_err) *factory_err = "landmark factory: scratch span short";
            return false;
            }
            try {
                scratch_.lm_out_chunks.resize(n_chunks);
            } catch (const std::bad_alloc &) {
                if (factory_err) *factory_err = "landmark factory: output table alloc failed";
            return false;
            }
            uint8_t * base = (uint8_t *) scratch.data;
            if (!build_landmarks_bounded(bspec, phase_tx, chunk_enc,
                    base, ws_need, base + ws_need, arena_need,
                    scratch_.lm_out_chunks.data(), scratch_.lm_out_chunks.size(),
                    &n_chunks, factory_err)) {
            return false;
            }
            if (!cursor.ok) {
                if (factory_err) *factory_err = "landmark factory: phase call order diverged";
            return false;
            }
            // Assemble the final multirow code stream once (the published
            // artifact; single allocation, not scratch).
            uint64_t total_dim = 0;
            for (const auto & l : spec->layers) {
                total_dim += l.feature_dim;
            }
            size_t total_bytes = 0;
            for (size_t c = 0; c < n_chunks; ++c) {
                const auto & oc = scratch_.lm_out_chunks[c];
                if (oc.desc.type != lm_type) {
                    if (factory_err) *factory_err = "landmark factory: chunk codec drift";
            return false;
                }
                total_bytes += oc.byte_size;
            }
            try {
                out_lm.desc = make_codec_desc(factor_role::landmark, lm_type,
                    orientation::token_major, {n_chunks, total_dim}, 0, spec->fingerprint);
                out_lm.bytes.assign(total_bytes, 0);
            } catch (const std::exception & e) {
                if (factory_err) {
                    *factory_err = std::string("landmark factory: assemble failed: ") + e.what();
                }
            return false;
            }
            size_t off = 0;
            for (size_t c = 0; c < n_chunks; ++c) {
                const auto & oc = scratch_.lm_out_chunks[c];
                std::memcpy(out_lm.bytes.data() + off, base + ws_need + oc.byte_offset, oc.byte_size);
                off += oc.byte_size;
                xkv_landmark_chunk lmc;
                lmc.row_begin = (uint32_t)(c * chunk);
                lmc.row_count = (uint32_t)(std::min<uint64_t>(span_n_rows, (c + 1) * chunk) - c * chunk);
                lmc.error_bound = oc.error_bound;
                lmc.source_fingerprint = oc.source_fingerprint;
                out_chunks.push_back(lmc);
            }
            return true;
        };
    }
    // Store capacity reservation: do NOT reserve for evaluate-only SHADOW.
    // For DENSE/SR, compute the exact checked candidate bytes estimate from
    // inputs and reserve capacity using estimate_segment_bundle_persistent_bytes;
    // pass &store_cap_res into params so store publish consumes it without double-counting.
    // Ordering: runs AFTER landmark_factory installation so the estimate
    // includes landmark bytes; estimating earlier under-reserves exactly the
    // sealed landmark footprint and trips the publish capacity check.
    size_t store_deficit = 0;
    std::string cap_res_err;
    xkv_capacity_reservation store_cap_res;
    if (!is_shadow && store->store_capacity_bytes() > 0) {
        size_t exact_dest_bytes = 0;
        std::string est_err;
        if (!estimate_segment_bundle_persistent_bytes(inputs, params, &exact_dest_bytes, &est_err)) {
            stats_.record_skip(xkv_skip_reason::unsupported_config);
            if (err) *err = "maintain: estimate_segment_bundle_persistent_bytes failed: " + est_err;
            return xkv_maintain_error;
            }
        store_cap_res = store->reserve_capacity(
            exact_dest_bytes, &cap_res_err, &store_deficit);
        if (!store_cap_res.valid()) {
            stats_.record_skip(xkv_skip_reason::preflight_oom);
            if (err) *err = "maintain: store capacity reservation failed (deficit " +
                            std::to_string(store_deficit) + " bytes): " + cap_res_err;
            return xkv_maintain_error;
    }
        params.capacity_reservation = &store_cap_res;
    }
    if (is_shadow) {
        // Evaluate-only: factorization/encode/gates run, the candidate is
        // discarded, payloads stay hot_committed, epochs/locations untouched.
        const xkv_sealing_result res = store->evaluate_segment_bundle(inputs, pids, gens, params);
        if (!res.success) {
            stats_.record_skip(res.skip_reason);
            stats_.record_error(res.message);
            if (err) *err = res.message;
            return xkv_maintain_error;
        }
        stats_.shadow_evals++;
        return xkv_maintain_evaluated;
    }
    // One atomic multi-group seal. The pool release (+ device upload) runs
    // Release protocol is split halves: read-only validate_hot_release runs
    // pre-seal (zero mutation); the
    // pool release (+ device upload) runs as the store precommit gate after
    // all allocations/hooks and before the allocation-free publish. Any
    // gate failure rolls back both subsystems.
    const xkv_snapshot_stamp pre_stamp = store->current_stamp();
    const int64_t t_seal_start = ggml_time_ms();
    auto & pre_plan = scratch_.release_plan;
    pre_plan.released_payload_ids.assign(pids.begin(), pids.end());
    pre_plan.released_generations.assign(gens.begin(), gens.end());
    pre_plan.released_physical_rows.clear();
    for (size_t wi : scratch_.window_idx) {
        pre_plan.released_physical_rows.push_back(scratch_.eligible[wi].physical_row);
    }
    pre_plan.expected_segment_id = 0;
    pre_plan.expected_segment_version = 0;
    pre_plan.stamp = pre_stamp;
    if (!kv.validate_hot_release(pre_plan, err)) {
        return xkv_maintain_error; // pre-seal refusal: nothing mutated, rows stay hot
    }
    const bool need_upload = (xkv_expected_residency(cparams_) == GGML_XKV_RES_DEVICE_OWNED);
    // No post-publish adopt: native backend batches build off-side and attach
    // to the candidate before publish; the gate below covers the pool half.
    // Device profiles cannot reach here (factorizer gate fails closed above).
    // Reference capture (see native gate above): pre_plan aliases the
    // scratch member; dropping `this` keeps the closure within SSO.
    xkv_seal_precommit_fn gate = [&kv, &pre_plan](
        const xkv_seal_precommit_ctx & pctx, std::string * gerr) -> bool {
        pre_plan.expected_segment_id = pctx.segment_id;
        pre_plan.expected_segment_version = pctx.segment_version;
        // Gate half: pool release only, zero store calls, single pool lock,
        // at most once per seal after a successful validate with unbroken
        // quiescence, never retried after true. SHADOW never reaches here.
        return kv.commit_hot_release(pre_plan, gerr);
    };
    const xkv_sealing_result res = store->seal_segment_bundle(inputs, pids, gens, params, std::move(gate));
    if (!res.success) {
        stats_.record_skip(res.skip_reason);
        stats_.record_error(res.message);
        if (err) *err = res.message;
        return xkv_maintain_error;
    }
    stats_.sealed_segments++;
    stats_.sealed_rows += n;
    stats_.sync_total += bstats.sync_count;
    stats_.cum_seal_seconds += (double) (ggml_time_ms() - t_seal_start) * 0.001;
    stats_.cum_factor_quant_seconds += res.factor_quant_seconds;
    stats_.cum_landmark_quant_seconds += res.landmark_quant_seconds;
    // Observe the published immutable segment for the runtime snapshot:
    // exact per-stream bytes, fingerprints, ranks. Measured, never assumed.
    if (auto published = store->get_segment(res.segment_id)) {
        uint32_t max_rk = 0, max_rv = 0;
        for (const auto & in : inputs) {
            max_rk = std::max(max_rk, in.rank_k);
            max_rv = std::max(max_rv, in.rank_v);
        }
        stats_.last_rank_k_obs = max_rk;
        stats_.last_rank_v_obs = max_rv;
        stats_.last_segment_id = published->segment_id;
        stats_.last_segment_version = published->segment_version;
        stats_.last_source_fp = published->source_fingerprint;
        stats_.last_profile_fp = published->profile_fingerprint;
        uint64_t codec_fp = 0xcbf29ce484222325ULL;
        auto fmix = [&codec_fp](uint64_t v) {
            codec_fp ^= v;
            codec_fp *= 0x100000001b3ULL;
        };
        for (const auto & g : published->groups) {
            // Retained validated counters (never host .bytes size: those go
            // to zero after device-owned host release; absence must never
            // be inferred from cleared bytes). Descriptors likewise remain.
            stats_.cum_ak_bytes += g.bytes_a_k;
            stats_.cum_av_bytes += g.bytes_a_v;
            stats_.cum_bk_bytes += g.bytes_b_k;
            stats_.cum_bv_bytes += g.bytes_b_v;
            stats_.sealed_streams += 4;
            fmix(g.a_k.desc.fingerprint());
            fmix(g.a_v.desc.fingerprint());
            if (g.b_k) fmix(g.b_k->desc.fingerprint());
            if (g.b_v) fmix(g.b_v->desc.fingerprint());
            if (g.bytes_landmark > 0) {
                stats_.cum_lm_bytes += g.bytes_landmark;
                stats_.sealed_landmarks++;
                fmix(g.landmark.desc.fingerprint());
            }
        }
        stats_.last_codec_fp = codec_fp;
        uint64_t backend_fp = 0xcbf29ce484222325ULL;
        for (char ch : placement_id_) {
            backend_fp ^= (uint64_t)(unsigned char) ch;
            backend_fp *= 0x100000001b3ULL;
        }
        stats_.last_backend_fp = backend_fp;
        stats_.cum_flat_source_bytes += res.flat_source_bytes;
        stats_.cum_factored_bytes += res.factored_bytes;
        stats_.last_factored_bytes = res.factored_bytes;
        stats_.last_factored_rows = n;
        stats_.last_compression_ratio = res.compression_ratio;
        stats_.compression_goal_met = true; // gates passed or seal would have failed
    }
    // No post-publish apply: the release already ran inside the gate.
    // Logical cells stay resident; only physical hot backing was released.
    return xkv_maintain_sealed;
}

uint64_t canonical_bytes_per_token(const llama_hparams & hparams) {
    uint64_t total = 0;
    const uint32_t n_trunk = hparams.n_layer();
    for (uint32_t il = 0; il < n_trunk; ++il) {
        if (!hparams.has_kv(il) || hparams.is_recr(il) || hparams.is_swa(il)) {
            continue;
        }
        total += (uint64_t) hparams.n_embd_k_gqa(il) + (uint64_t) hparams.n_embd_v_gqa(il);
    }
    total *= sizeof(float);
    // Exact one-head streaming temp bound (no hardcoded width).
    uint64_t max_hd = 0;
    for (uint32_t il = 0; il < n_trunk; ++il) {
        if (!hparams.has_kv(il) || hparams.is_recr(il) || hparams.is_swa(il)) {
            continue;
        }
        max_hd = std::max<uint64_t>(max_hd, hparams.n_embd_head_k(il));
        max_hd = std::max<uint64_t>(max_hd, hparams.n_embd_head_v(il));
    }
    total += max_hd * sizeof(float);
    return total;
}

// Exact per-token workspace cost of the next segment from shared estimators
// over actual group dims/codecs: canonical total + factor estimates +
// landmark share, amortized over the segment. Checked math; false on
// overflow or estimator refusal (fail closed, never defaulted).
static bool next_segment_bytes_per_token(
    const llama_hparams & hparams,
    const xkv_effective_config & eff,
    const llama_cparams & cparams,
    uint64_t & out_per_token) {
    const uint32_t seg = eff.segment_tokens > 0 ? eff.segment_tokens : 1;
    const uint32_t rk = eff.rank_k > 0 ? eff.rank_k : 1;
    const uint32_t rv = eff.rank_v > 0 ? eff.rank_v : 1;
    const factor_config fcfg;
    uint64_t total = 0;
    const uint32_t n_trunk = hparams.n_layer();
    for (uint32_t il = 0; il < n_trunk; ++il) {
        if (!hparams.has_kv(il) || hparams.is_recr(il) || hparams.is_swa(il)) {
            continue;
        }
        const uint64_t dk = hparams.n_embd_k_gqa(il);
        const uint64_t dv = hparams.n_embd_v_gqa(il);
        if (dk == 0 || dv == 0) {
            return false;
        }
        // Canonical slice for a full segment (exact).
        uint64_t ck = 0, cv = 0;
        if (!ck_mul_u64(seg, dk, ck) || !ck_mul_u64(seg, dv, cv)) {
            return false;
        }
        uint64_t cb = 0;
        if (!ck_mul_u64(ck + cv, sizeof(float), cb) || !ck_add_u64(total, cb, total)) {
            return false;
        }
        // Factor estimates per layer (conservative vs grouped amortization)
        // with configured ranks clamped exactly like sealing.
        const uint32_t lr_k = std::min<uint32_t>({rk, seg, (uint32_t) std::min<uint64_t>(dk, UINT32_MAX)});
        const uint32_t lr_v = std::min<uint32_t>({rv, seg, (uint32_t) std::min<uint64_t>(dv, UINT32_MAX)});
        uint64_t ek = 0, ev = 0;
        std::string eerr;
        if (!estimate_factorize_matrix_workspace_bytes(seg, dk, lr_k,
                fcfg.oversampling, fcfg.power_iterations, &ek, &eerr) ||
            !estimate_factorize_matrix_workspace_bytes(seg, dv, lr_v,
                fcfg.oversampling, fcfg.power_iterations, &ev, &eerr)) {
            return false;
        }
        if (!ck_add_u64(total, ek, total) || !ck_add_u64(total, ev, total)) {
            return false;
        }
        // Landmark share: one phased row per token across K widths.
        uint64_t lm = 0;
        if (!ck_mul_u64(dk, sizeof(float), lm) || !ck_add_u64(total, lm, total)) {
            return false;
        }
    }
    if (total == 0) {
        return false;
    }
    // Amortize + one head temp, rounding up (conservative).
    uint64_t max_hd = 0;
    for (uint32_t il = 0; il < n_trunk; ++il) {
        if (!hparams.has_kv(il) || hparams.is_recr(il) || hparams.is_swa(il)) {
            continue;
        }
        max_hd = std::max<uint64_t>(max_hd, hparams.n_embd_head_k(il));
        max_hd = std::max<uint64_t>(max_hd, hparams.n_embd_head_v(il));
    }
    (void) cparams;
    out_per_token = total / seg + max_hd * sizeof(float) + 1;
    return out_per_token > 0;
}

bool llama_xkv_runtime::fill_admission_snapshot(
    const llama_kv_cache & kv,
    llama_memory_admission_snapshot & out,
    std::string * err) const {
    (void) err;
    out = llama_memory_admission_snapshot{};
    const uint32_t kv_cap = kv.get_kv_capacity();
    const uint32_t kv_used = std::min(kv.get_kv_used(), kv_cap);
    out.logical_capacity = kv_cap;
    out.logical_used = kv_used;
    auto pool = kv.get_hot_slot_pool();
    if (pool) {
        // hot_free is authoritative and already excludes reserved slots.
        out.hot_capacity = kv.get_kv_hot_capacity();
        out.hot_used = std::min(pool->get_live(), out.hot_capacity);
        out.hot_free = std::min(pool->get_free(), out.hot_capacity);
        out.hot_reserved = pool->get_reserved();
    } else {
        // Unbounded (OFF/SHADOW): hot maps legacy exactly.
        out.hot_capacity = kv_cap;
        out.hot_used = kv_used;
        out.hot_free = kv_cap - kv_used;
        out.hot_reserved = 0;
    }
    auto store = kv.get_xkv_store();
    const uint64_t per_token = canonical_bytes_per_token(kv.get_hparams());
    if (store) {
        const xkv_accounting acct = store->get_accounting();
        const uint64_t store_budget = (uint64_t) cparams_.xkv_store_mib * 1024 * 1024;
        out.factor_live_bytes = acct.live_payload_bytes;
        out.factor_reserved_bytes = acct.allocated_bytes;
        out.factor_budget_bytes = store_budget;
        // Factor free: hard-cap against total allocated/reserved bytes
        // (includes live payloads + pinned-retired + COW candidates), never
        // live payloads alone.
        out.factor_free_bytes = (store_budget > acct.allocated_bytes)
            ? (store_budget - acct.allocated_bytes) : 0;
        // Real planner bound: remaining budget over measured-or-estimated
        // factored bytes per token (measured after the first seal, canonical
        // geometry estimate before). Zero budget means unconstrained.
        uint64_t fpt = 0;
        if (!next_segment_bytes_per_token(kv.get_hparams(), eff_, cparams_, fpt)) {
            fpt = per_token;
        }
        if (stats_.last_factored_rows > 0 && stats_.last_factored_bytes > 0) {
            fpt = std::max<uint64_t>(1, stats_.last_factored_bytes / stats_.last_factored_rows);
        }
        // DENSE/SR requires a store budget; zero store budget is allowed
        // only for SHADOW (where publication never occurs). Unconstrained
        // DENSE/SR with zero budget is forbidden and reports 0 safe tokens.
        const bool is_shadow = (cparams_.xkv_mode == LLAMA_XKV_MODE_SHADOW);
        if (store_budget == 0 && !is_shadow) {
            out.factor_safe_tokens = 0;
        } else if (store_budget == 0) {
            out.factor_safe_tokens = UINT32_MAX;
        } else {
            uint64_t bound = out.factor_free_bytes / std::max<uint64_t>(fpt, 1);
            out.factor_safe_tokens = bound > UINT32_MAX ? UINT32_MAX : (uint32_t) bound;
        }
        // Workspace accounting: total live/peak/reserved sums arena plus the
        // shared B-tile decode cache (which consumes its subbudget
        // separately). Free bytes subtract total live from the unified
        // workspace budget (xkv_workspace_mib).
        const uint64_t ws_budget = (uint64_t) cparams_.xkv_workspace_mib * 1024 * 1024;
        const uint64_t arena_live = store->get_arena().get_live_bytes();
        const uint64_t arena_peak = store->get_arena().get_peak_bytes();
        uint64_t cache_res = 0;
        if (b_tile_cache_) {
            cache_res  = b_tile_cache_->total_allocated_bytes();
        }
        // Cache slabs remain allocated even when no entry is active and can
        // coexist with the arena peak; max(arena,cache) undercounts VRAM.
        if (!ck_add_u64(arena_live, cache_res, out.workspace_live_bytes) ||
            !ck_add_u64(arena_peak, cache_res, out.workspace_peak_bytes)) {
            if (err) *err = "XKV workspace accounting overflow";
            return false;
        }
        out.workspace_budget_bytes = ws_budget;
        const uint64_t ws_total_live = out.workspace_live_bytes;
        out.workspace_free_bytes = (ws_budget > ws_total_live) ? (ws_budget - ws_total_live) : 0;
        // Real planner bound: remaining workspace over exact per-token
        // cost from shared estimators. Zero budget = unconstrained.
        uint64_t ws_per_token = 0;
        if (!next_segment_bytes_per_token(kv.get_hparams(), eff_, cparams_, ws_per_token)) {
            ws_per_token = per_token;
        }
        if (ws_budget == 0 || ws_per_token == 0) {
            out.workspace_safe_tokens = UINT32_MAX;
        } else {
            uint64_t bound = out.workspace_free_bytes / ws_per_token;
            out.workspace_safe_tokens = bound > UINT32_MAX ? UINT32_MAX : (uint32_t) bound;
        }
    } else {
        // No bound store (OFF): zero store bytes, unconstrained planners.
        out.factor_live_bytes = 0;
        out.factor_reserved_bytes = 0;
        out.factor_budget_bytes = 0;
        out.factor_free_bytes = 0;
        out.factor_safe_tokens = UINT32_MAX;
        out.workspace_live_bytes = 0;
        out.workspace_peak_bytes = 0;
        out.workspace_budget_bytes = 0;
        out.workspace_free_bytes = 0;
        out.workspace_safe_tokens = UINT32_MAX;
    }
    out.recurrent_capacity = 0;
    out.recurrent_used = 0;
    out.recurrent_blocks_new_seq = false;
    llama_memory_admission_finalize(out);
    return true;
}

bool llama_xkv_runtime::fill_runtime_snapshot(
    const llama_kv_cache & kv,
    llama_memory_xkv_runtime_snapshot & out,
    std::string * err) const {
    out = llama_memory_xkv_runtime_snapshot{};
    out.struct_size = (uint32_t) sizeof(out);
    out.version = 1;
    auto store = kv.get_xkv_store();
    if (!store) {
        return false; // no store: caller reports not_evaluated
    }
    // Armed only after path readiness plus at least one live observation.
    out.armed = readiness_.ready && stats_.maintenance_runs > 0;
    out.mode = cparams_.xkv_mode;
    out.effective_profile = cparams_.xkv_storage_profile;
    out.source = cparams_.xkv_source;
    out.rank_k = stats_.last_rank_k_obs;
    out.rank_v = stats_.last_rank_v_obs;
    out.effective_chunk_size = eff_.chunk_tokens;
    out.factor_streams = stats_.sealed_streams;
    out.codec_fingerprint = stats_.last_codec_fp;
    out.backend_fingerprint = stats_.last_backend_fp;
    out.source_fingerprint = stats_.last_source_fp;
    out.profile_fingerprint = stats_.last_profile_fp;
    const xkv_accounting acct = store->get_accounting();
    // Compute live hot baseline bytes from actual bounded hot pool rows
    // (get_bound() / used after compact) × exact per-layer K/V row bytes.
    // Dedup unique owning layers across groups so aliased layers are counted once.
    uint64_t hot_bytes_live = 0;
    auto pool = kv.get_hot_slot_pool();
    const uint32_t hot_rows_live = pool ? pool->get_live() : 0;
    if (hot_rows_live > 0) {
        std::unordered_set<uint32_t> seen_layers;
        uint64_t single_row_hot_bytes = 0;
        for (const auto & g : groups_.groups) {
            for (uint32_t ol : g.owning_layers) {
                if (seen_layers.insert(ol).second) {
                    ggml_tensor * kt = kv.get_k_storage((int32_t) ol);
                    ggml_tensor * vt = kv.get_v_storage((int32_t) ol);
                    if (kt && vt) {
                        const uint64_t bk = ggml_row_size(kt->type, kt->ne[0]);
                        const uint64_t bv = ggml_row_size(vt->type, vt->ne[0]);
                        uint64_t row_bytes = 0;
                        if (!ck_add_u64(bk, bv, row_bytes) ||
                            !ck_add_u64(single_row_hot_bytes, row_bytes, single_row_hot_bytes)) {
                            if (err) *err = "XKV hot byte accounting overflow";
                            return false;
                        }
                    }
                }
            }
        }
        if (!ck_mul_u64((uint64_t) hot_rows_live, single_row_hot_bytes, hot_bytes_live)) {
            if (err) *err = "XKV hot byte accounting overflow";
            return false;
        }
    }
    out.hot_bytes = hot_bytes_live;
    out.flat_bytes = 0; // this runtime never publishes FLAT_TQ bundles
    out.factor_ak_bytes = stats_.cum_ak_bytes;
    out.factor_bk_bytes = stats_.cum_bk_bytes;
    out.factor_av_bytes = stats_.cum_av_bytes;
    out.factor_bv_bytes = stats_.cum_bv_bytes;
    out.factor_payload_bytes = acct.factor_live_bytes;
    out.factor_metadata_bytes = acct.factor_metadata_bytes;
    out.factor_padding_bytes = acct.factor_padding_bytes;
    out.landmark_payload_bytes = acct.landmark_payload_bytes;
    out.landmark_metadata_bytes = acct.landmark_metadata_bytes;
    out.landmark_exception_bytes = acct.landmark_exception_bytes;
    out.index_bytes = acct.index_bytes;
    out.codec_shared_bytes = acct.codec_shared_bytes;
    out.decode_tile_cache_bytes = b_tile_cache_ ? b_tile_cache_->active_cache_bytes() : 0;
    out.capture_bytes = 0;   // no staging ring; decoded-hot only
    out.candidate_bytes = scratch_.slices.capacity() * sizeof(xkv_maintain_control_scratch::slice_desc);
    out.snapshot_pinned_bytes = acct.snapshot_pinned_bytes;
    out.allocator_live_bytes = acct.arena_live_bytes;
    out.allocator_reserved_bytes = acct.arena_reserved_bytes;
    out.device_peak_bytes = acct.device_peak_bytes;
    out.host_peak_bytes = acct.host_peak_bytes > 0 ? acct.host_peak_bytes : acct.arena_peak_bytes;
    out.unique_payloads = acct.total_payloads;
    out.aliased_payloads = acct.aliased_payloads;
    // §16 math:
    // 1. Coverage = baseline-equivalent bytes of rows ACTUALLY factored / total eligible baseline bytes.
    //    Numerator is acct.baseline_factored_bytes (live covered original bytes across published segments).
    //    Denominator is eligible baseline bytes: active live factored baseline plus resident hot KV bytes.
    const size_t total_eligible_baseline = acct.baseline_factored_bytes + (size_t) hot_bytes_live;
    out.baseline_covered_bytes = acct.baseline_factored_bytes;
    out.covered_compressed_bytes = acct.live_payload_bytes; // actual live representation bytes covering the baseline
    out.factored_baseline_byte_coverage = (total_eligible_baseline > 0 && acct.baseline_factored_bytes > 0)
        ? (double) acct.baseline_factored_bytes / (double) total_eligible_baseline : 0.0;

    // 2. Net extra compression ratio (live representation basis):
    //    same-row live baseline / actual live representation bytes (factor payload + landmark + index/metadata).
    //    This is exact acct.baseline_factored_bytes / acct.live_payload_bytes, NOT cumulative historical publish bytes.
    out.net_extra_compression_ratio = (acct.live_payload_bytes > 0 && acct.baseline_factored_bytes > 0)
        ? (double) acct.baseline_factored_bytes / (double) acct.live_payload_bytes : 0.0;

    // 3. Net extra compression ratio (reserved representation basis):
    //    same-row live baseline / actual reserved storage bytes (allocated segments + arena reserved capacity).
    out.net_extra_compression_ratio_reserved = (acct.reserved_bytes > 0 && acct.baseline_factored_bytes > 0)
        ? (double) acct.baseline_factored_bytes / (double) acct.reserved_bytes : 0.0;

    out.factor_quant_ratio = (acct.factor_live_bytes > 0 && acct.factor_fp16_equivalent_bytes > 0)
        ? (double) acct.factor_fp16_equivalent_bytes / (double) acct.factor_live_bytes
        : stats_.last_compression_ratio;
    out.ratios_evaluated = (acct.live_payload_bytes > 0 && acct.baseline_factored_bytes > 0);
    out.seal_seconds = stats_.cum_seal_seconds;
    out.factor_quant_seconds = stats_.cum_factor_quant_seconds;
    out.landmark_quant_seconds = stats_.cum_landmark_quant_seconds;
    out.seal_timers_evaluated = stats_.cum_seal_seconds > 0.0;
    out.quant_timers_evaluated = (stats_.cum_factor_quant_seconds > 0.0 || stats_.cum_landmark_quant_seconds > 0.0);
    out.codec_a_k = (int32_t) cparams_.xkv_factor_a_k;
    out.codec_b_k = (int32_t) cparams_.xkv_factor_b_k;
    out.codec_a_v = (int32_t) cparams_.xkv_factor_a_v;
    out.codec_b_v = (int32_t) cparams_.xkv_factor_b_v;
    out.codec_landmark = (int32_t) cparams_.xkv_landmark_type;
    out.codec_factorizer = (int32_t) cparams_.xkv_factorizer;
    out.codec_balance = (int32_t) cparams_.xkv_factor_balance;
    out.tri_ratio = cparams_.triattention_ratio;
    out.tri_recent_window = cparams_.triattention_recent_window;
    out.tri_scorer_valid = false;
    if (kv.tri_scorer && kv.tri_scorer->valid()) {
        out.tri_scorer_valid = true;
        kv.tri_scorer->calibration_content_sha256(out.tri_calibration_sha256);
        out.tri_calibration_fingerprint = kv.tri_scorer->calibration_content_fingerprint();
    }
    out.factor_seed = eff_.factor_seed;
    // §16: Compression goal must combine global coverage >= configured min and
    // actual live net saving >= configured min; per-segment historical
    // stats_.compression_goal_met alone is insufficient.
    // Evaluated only when we have live factored baseline and positive eligible baseline.
    out.compression_goal_evaluated = (acct.live_payload_bytes > 0 &&
                                      acct.baseline_factored_bytes > 0 &&
                                      total_eligible_baseline > 0 &&
                                      acct.reserved_bytes > 0);
    if (out.compression_goal_evaluated) {
        const double live_saving_fraction = out.net_extra_compression_ratio > 0.0
            ? (1.0 - (1.0 / out.net_extra_compression_ratio)) : 0.0;
        const bool coverage_pass = (out.factored_baseline_byte_coverage >= cparams_.xkv_min_factor_coverage);
        const bool saving_pass = (live_saving_fraction >= cparams_.xkv_min_saving);
        out.compression_goal_met = stats_.compression_goal_met && coverage_pass && saving_pass;
    } else {
        out.compression_goal_met = false;
    }
    out.select_seconds = stats_.cum_select_seconds;
    out.refine_seconds = stats_.cum_refine_seconds;
    out.reconstruct_seconds = stats_.cum_reconstruct_seconds;
    out.read_seconds = stats_.cum_read_seconds;
    out.pack_seconds = stats_.cum_pack_seconds;
    out.graph_timings_evaluated = (out.select_seconds > 0.0 || out.refine_seconds > 0.0 ||
                                   out.reconstruct_seconds > 0.0 || out.read_seconds > 0.0);
    out.pack_timer_evaluated = (out.pack_seconds > 0.0);
    out.segments_sealed = stats_.sealed_segments;
    for (size_t i = 0; i < LLAMA_MEMORY_XKV_SKIP_REASON_COUNT; ++i) {
        out.skip_counts[i] = i < stats_.skipped_by_reason.size() ? stats_.skipped_by_reason[i] : 0;
    }
    out.sr_selected_rows = stats_.sr_selected_rows;
    out.sr_fragments = stats_.sr_fragments;
    out.landmark_refine_rows = stats_.landmark_refine_rows;
    out.landmark_refine_cap_hits = stats_.landmark_refine_cap_hits;
    out.sr_counters_evaluated = (out.sr_selected_rows > 0 || out.sr_fragments > 0 ||
                                 out.landmark_refine_rows > 0 || out.landmark_refine_cap_hits > 0);
    out.spec_stale_total = stats_.spec_stale_total;
    out.transaction_abort_total = stats_.transaction_abort_total;
    out.spec_counters_evaluated = (out.spec_stale_total > 0 || out.transaction_abort_total > 0);
    out.synchronize_total = stats_.sync_total;
    return true;
}

} // namespace llama_xkv

namespace llama_xkv {

// ---- Per-(parent query, DDVR slot) SR fragment plans (§9.3) -----------------
// Replaces ANY-row union visibility: each expanded selector query (parent q,
// DDVR slot g; E expanded queries in parent-major order) derives fragment
// metadata ONLY from its own legal cold rows (membership in q, slot match,
// causal cutoff, phase, visibility). Shared physical rows duplicate across
// slots as separate per-slot fragments; parent query + slot travel explicitly
// on every fragment (parent_query_id/parent_group_id, single-query
// query_visibility). Metadata is bounded by construction: plan rows ⊆ segment
// cold rows, global fragments ≤ E × cold rows (checked), every size conversion
// checked, no chunk-size fallbacks (zero chunk_tokens fails closed).
//
// Fragment -> arena landmark mapping (device snapshot contract): the arena
// `landmarks` tensor is the sealed base chunk table (row c covers segment rows
// [chunks[c].row_begin, +row_count), rows == landmark_chunks.size()). Intact
// fragments bind the exact persisted per-chunk source_fingerprint and map to
// local row base_landmark_row (== chunk index c). Partial fragments carry no
// shared row: host rebuilds them from FINAL factors restricted to legal rows;
// DEVICE_OWNED partials emit xkv_native_landmark_rebuild_desc entries (no host
// decode of device factors) for device rebuild-then-score. The graph SCOREs
// per (arena, fragment) restricted by parent ids + visibility, MERGEs global,
// and maps ROWS E->N with [parent, slot] selector columns, dedup (row, slot).
struct sr_plan_cold_row {
    uint32_t seg_row = 0;
    uint64_t pid = 0;
    uint64_t gen = 0;
    int64_t  spos = 0;
    uint32_t run_id = 0;
    uint8_t  vis = 0;
};

// Sealed landmark chunk table must cover the segment exactly: contiguous from
// row 0, every chunk nonempty and in bounds, final end == n_rows, and every
// chunk carries a nonzero persisted source fingerprint. Anything else fails
// closed (never silently re-chunked or defaulted).
static bool sr_chunk_table_covers_segment(
    const std::vector<xkv_landmark_chunk> & chunks,
    uint32_t n_rows,
    std::string * err) {
    auto fail = [&](const char * m) -> bool { if (err) *err = m; return false; };
    uint64_t expect = 0;
    for (size_t c = 0; c < chunks.size(); ++c) {
        const auto & chk = chunks[c];
        if (chk.row_count == 0) return fail("build_xkv_graph_snapshot: empty sealed landmark chunk");
        if ((uint64_t) chk.row_begin != expect) {
            return fail("build_xkv_graph_snapshot: sealed landmark chunks not contiguous");
        }
        uint64_t end = 0;
        if (!ck_add_u64(expect, (uint64_t) chk.row_count, end) || end > (uint64_t) n_rows) {
            return fail("build_xkv_graph_snapshot: sealed landmark chunk out of segment bounds");
        }
        if (chk.source_fingerprint == 0) {
            return fail("build_xkv_graph_snapshot: sealed chunk missing source fingerprint");
        }
        expect = end;
    }
    if (expect != (uint64_t) n_rows) {
        return fail("build_xkv_graph_snapshot: sealed landmark chunks do not cover segment");
    }
    return true;
}

// Exact device-snapshot stream match: full codec descriptor fingerprint +
// role + type + orientation + logical/padded shape (never handle order).
// Exactly one bundle handle must match and its tensor (+buffer) must be live.
static ggml_tensor * sr_match_device_stream(
    const xkv_backend_batch_result & bundle,
    const codec_desc & want,
    const char * stream_name,
    std::string * err) {
    auto fail = [&](const std::string & m) -> ggml_tensor * {
        if (err) *err = std::string("build_xkv_graph_snapshot: ") + m;
        return nullptr;
    };
    std::string werr;
    if (!want.validate(&werr)) return fail(std::string("invalid ") + stream_name + " descriptor: " + werr);
    const uint64_t wfp = want.fingerprint();
    ggml_tensor * found = nullptr;
    ggml_backend_buffer_t found_buf = nullptr;
    for (const auto & h : bundle.handles) {
        if (!h) continue;
        if (h->get_descriptor_fingerprint() != wfp) continue;
        const codec_desc & d = h->get_desc();
        if (d.role != want.role || d.type != want.type || d.orient != want.orient ||
            !(d.logical_shape == want.logical_shape) || !(d.padded_shape == want.padded_shape)) {
            continue;
        }
        if (found) return fail(std::string("ambiguous duplicate handle for ") + stream_name);
        found = h->get_tensor();
        found_buf = h->get_buffer();
    }
    if (!found) return fail(std::string("no backend handle matches ") + stream_name);
    if (!found_buf) return fail(std::string("matched handle has no live buffer for ") + stream_name);
    return found;
}

// One plan: fragments for exactly (parent_query, parent_slot) on one pinned
// segment from that plan's legal cold rows (pre-filtered, sorted, spos-unique).
// Intact chunk sets bind the exact persisted source fingerprints (no duplicate
// bytes); partials rebuild (host) or emit native rebuild descriptors
// (DEVICE_OWNED, never decoding device factors on host).
static bool build_sr_fragment_plan(
    const xkv_segment & seg,
    const xkv_factor_group_payload & gp,
    const std::shared_ptr<const xkv_segment> & seg_hold,
    uint32_t owning_layer,
    uint32_t kv_head,
    uint32_t parent_query,
    uint32_t parent_slot,
    int64_t causal_cutoff,
    uint32_t chunk_tokens,
    uint64_t phase_fp,
    const phase_transform_fn & frag_phase,
    ggml_type landmark_type,
    const xkv_snapshot_stamp & stamp,
    size_t n_queries,
    const std::vector<sr_plan_cold_row> & plan_rows,
    bool is_device_owned,
    uint32_t sr_feature_offset,
    uint32_t sr_feature_dim,
    std::vector<legal_fragment> & out_frags,
    std::vector<xkv_graph_snapshot::xkv_native_landmark_rebuild_desc> & out_rebuilds,
    std::string * err) {
    auto fail = [&](const std::string & m) -> bool {
        if (err) *err = std::string("build_xkv_graph_snapshot: ") + m;
        return false;
    };
    if (plan_rows.empty()) return true;
    if (plan_rows.size() > (size_t) UINT32_MAX) return fail("plan row count overflow");
    if (n_queries == 0) return fail("no parent queries");
    if ((size_t) parent_query >= n_queries) return fail("parent query out of range");
    if (chunk_tokens == 0) return fail("chunk_tokens must be non-zero (no fallback)");
    std::vector<row_meta> metas;
    metas.reserve(plan_rows.size());
    for (const auto & r : plan_rows) {
        if (r.seg_row >= seg.n_rows) return fail("plan row outside segment bounds");
        row_meta m;
        m.segment_row = r.seg_row;
        m.payload_id = r.pid;
        m.storage_generation = r.gen;
        m.run_id = r.run_id;
        m.storage_pos = r.spos;
        m.virtual_pos = r.spos; // ordinary layout; rerot grouping is graph-owned
        m.ddvr_group = parent_slot;
        m.visibility = r.vis;
        m.is_live = true;
        m.reader_visible = true;
        metas.push_back(m);
    }
    std::vector<legal_fragment> frags;
    try {
        frags = build_legal_fragments(seg, stamp, seg.segment_version, 0, owning_layer,
            kv_head, metas, chunk_tokens, phase_fp, causal_cutoff);
    } catch (const std::exception & e) {
        return fail(std::string("fragment build refused: ") + e.what());
    }
    // Intact binding against the sealed base chunk table (exact persisted
    // source fingerprints; stale/partial fragments stay unbound, not an error).
    // DEVICE_OWNED deliberately releases host landmark bytes after native
    // publication. Intactness depends on the persisted descriptor/chunk
    // provenance plus row identity, not on a redundant host mirror.
    if (!gp.landmark_chunks.empty()) {
        if (!sr_chunk_table_covers_segment(gp.landmark_chunks, seg.n_rows, err)) return false;
        if (seg.row_payload_ids.size() != (size_t) seg.n_rows) {
            return fail("segment row identity incomplete");
        }
        landmark_base_table btab;
        btab.landmark = std::shared_ptr<const encoded_matrix>(seg_hold, &gp.landmark);
        std::vector<uint64_t> base_gens((size_t) seg.n_rows, 0);
        std::vector<int64_t> base_poss((size_t) seg.n_rows, 0);
        for (const auto & r : plan_rows) {
            base_gens[(size_t) r.seg_row] = r.gen;
            base_poss[(size_t) r.seg_row] = r.spos;
        }
        std::vector<float> cbounds;
        std::vector<uint64_t> csource;
        std::vector<uint32_t> coffs;
        cbounds.reserve(gp.landmark_chunks.size());
        csource.reserve(gp.landmark_chunks.size());
        coffs.reserve(gp.landmark_chunks.size() + 1);
        for (const auto & chk : gp.landmark_chunks) {
            coffs.push_back(chk.row_begin);
            cbounds.push_back(chk.error_bound);
            csource.push_back(chk.source_fingerprint);
        }
        coffs.push_back(seg.n_rows);
        btab.row_payload_ids = seg.row_payload_ids.data();
        btab.row_generations = base_gens.data();
        btab.row_positions = base_poss.data();
        btab.chunk_error_bounds = cbounds.data();
        btab.chunk_source_fingerprints = csource.data();
        btab.chunk_row_offsets = coffs.data();
        btab.n_rows_total = (size_t) seg.n_rows;
        btab.n_chunks = gp.landmark_chunks.size();
        btab.stamp = stamp;
        btab.phase_tx_fingerprint = phase_fp;
        btab.bounds_fingerprint = compute_base_table_fingerprint(btab);
        size_t n_bound = 0;
        if (!bind_base_landmarks(frags.data(), frags.size(), btab, &n_bound, err)) return false;
    }
    for (size_t fi = 0; fi < frags.size(); ++fi) {
        legal_fragment & f = frags[fi];
        f.parent_query_id = parent_query;
        f.parent_group_id = parent_slot;
        f.query_visibility.assign(n_queries, false);
        f.query_visibility[(size_t) parent_query] = true;
        if (f.uses_base_landmark()) {
            if (f.source_fingerprint == 0) return fail("intact fragment missing source fingerprint");
            if ((size_t) f.base_landmark_row >= gp.landmark_chunks.size()) {
                return fail("intact fragment base row out of range");
            }
            f.is_derived_partial = false;
            f.requires_native_rebuild = false;
            continue;
        }
        // Partial/causal fragment: rebuild from FINAL factors restricted to
        // this plan's legal rows (never a first-N copy, never unphased).
        f.is_derived_partial = true;
        if (is_device_owned) {
            // DEVICE_OWNED: explicit native rebuild descriptor; never decode
            // device factors on host. The fragment carries no landmark bytes.
            if (!f.landmark_matrix.bytes.empty()) {
                return fail("device partial must not carry host landmark bytes");
            }
            const uint64_t frag_idx = (uint64_t) out_frags.size() + (uint64_t) fi;
            if (frag_idx > (uint64_t) UINT32_MAX) return fail("fragment index overflow");
            f.requires_native_rebuild = true;
            f.source_fingerprint = seg.source_fingerprint;
            f.key.source_fingerprint = seg.source_fingerprint;
            f.key.landmark_codec_fp = gp.landmark.desc.fingerprint();
            xkv_graph_snapshot::xkv_native_landmark_rebuild_desc rd;
            rd.fragment_index = (uint32_t) frag_idx;
            rd.segment_id = seg.segment_id;
            rd.segment_version = seg.segment_version;
            rd.group_index = gp.group_index;
            rd.owning_layer = owning_layer;
            rd.kv_head = kv_head;
            rd.row_indices = f.row_indices;
            rd.storage_positions = f.storage_positions;
            rd.feature_offset = sr_feature_offset;
            rd.feature_dim = sr_feature_dim;
            rd.phase_tx_fingerprint = phase_fp;
            rd.landmark_type = landmark_type;
            rd.error_bound = 0.0f; // measured at device rebuild time
            out_rebuilds.push_back(std::move(rd));
        } else {
            if (!frag_phase) return fail("host rebuild needs phase transform");
            f.requires_native_rebuild = false;
            try {
                encode_fragment_landmark(f, seg, gp, sr_feature_offset, sr_feature_dim,
                    frag_phase, phase_fp, landmark_type);
            } catch (const std::exception & e) {
                return fail(std::string("fragment rebuild failed: ") + e.what());
            }
        }
    }
    for (size_t fi = 0; fi < frags.size(); ++fi) {
        out_frags.push_back(std::move(frags[fi]));
    }
    return true;
}

} // namespace llama_xkv

bool llama_kv_cache_context::build_xkv_graph_snapshot(
    uint32_t il,
    uint32_t kv_head,
    float kq_scale,
    float logit_softcap,
    std::unique_ptr<llama_xkv::xkv_graph_snapshot> & out,
    std::string * err) const {
    using namespace llama_xkv;
    out.reset();
    if (!kv) {
        if (err) *err = "build_xkv_graph_snapshot: context has no cache";
        return false;
    }
    llama_xkv_runtime * rt = kv->get_xkv_runtime();
    if (!rt) {
        if (err) *err = "build_xkv_graph_snapshot: no XKV runtime (OFF/MTP)";
        return false;
    }
    if (kv->is_xkv_bounded_hot() && !rt->attention_path_ready()) {
        if (err) *err = "build_xkv_graph_snapshot: XKV attention path not ready";
        return false;
    }
    auto store = kv->get_xkv_store();
    if (!store) {
        if (err) *err = "build_xkv_graph_snapshot: no XKV store";
        return false;
    }
    // Coordinator reader lease first: blocks maintenance/final fence/
    // context shift for this snapshot's lifetime (pins protect bytes, not
    // epochs/layout). Parks while a quiesce gate is held; failure (or no
    // coordinator in standalone use is fine) fails closed when present.
    xkv_transaction_coordinator::reader_lease reader_guard;
    if (auto * coord = rt->get_coordinator()) {
        reader_guard = coord->acquire_reader();
        if (!reader_guard) {
            if (err) *err = "build_xkv_graph_snapshot: reader lease refused";
            return false;
        }
    }
    if (i_cur >= ubatches.size() || i_cur >= sinfos.size()) {
        if (err) *err = "build_xkv_graph_snapshot: context has no current ubatch";
        return false;
    }
    const llama_ubatch & ubatch = ubatches[i_cur];
    if (ubatch.n_tokens <= 0) {
        if (err) *err = "build_xkv_graph_snapshot: current ubatch is empty";
        return false;
    }
    const llama_hparams & hparams = kv->get_hparams();
    if (!hparams.has_kv(il)) {
        if (err) *err = "build_xkv_graph_snapshot: layer has no KV";
        return false;
    }
    const uint32_t n_kv_heads = hparams.n_head_kv(il);
    if (n_kv_heads == 0 || kv_head >= n_kv_heads) {
        if (err) *err = "build_xkv_graph_snapshot: kv_head out of range";
        return false;
    }
    if (!rt->groups_valid() && !rt->rebuild_groups(hparams, *kv, err)) {
        return false;
    }
    const uint32_t owning = kv->get_owning_layer(il);
    if (!rt->xkv_layer_enabled(owning)) {
        if (err) *err = "build_xkv_graph_snapshot: layer is not on the XKV path (SWA/recurrent/MTP)";
        return false;
    }
    const layer_group * grp = nullptr;
    for (const auto & g : rt->group_map().groups) {
        for (uint32_t ol : g.owning_layers) {
            if (ol == owning) {
                grp = &g;
                break;
            }
        }
        if (grp) {
            break;
        }
    }
    if (!grp) {
        if (err) *err = "build_xkv_graph_snapshot: owning layer is not in any runtime group";
        return false;
    }
    const uint32_t hd_k = hparams.n_embd_head_k(il);
    const uint32_t hd_v = hparams.n_embd_head_v(il);
    if (hd_k == 0 || hd_v == 0) {
        if (err) *err = "build_xkv_graph_snapshot: degenerate head dims";
        return false;
    }
    const uint32_t n_q = hparams.n_head(il);
    const uint32_t q_per_kv = n_q / std::max<uint32_t>(n_kv_heads, 1);
    // Collect current-ubatch rows: payload/positions remain cells-owned.
    // Enumerate ALL visible semantic cells per query (history + current).
    // Hot rows become storage-gather descriptors (no pre-copy, no
    // canonicalization; gathered in compute after writes). Current
    // hot_writing payloads are accepted only when this context applied them.
    // Factored rows are grouped into pinned segment views below.
    struct snap_hot {
        uint64_t pid; uint64_t gen; uint32_t prow; int64_t spos;
        uint32_t stream; xkv_state state; uint32_t group_index = 0;
    };
    struct snap_cold {
        uint64_t pid; uint64_t gen; int64_t spos;
        uint64_t seg_id; uint64_t seg_ver; uint32_t seg_row; uint64_t seg_gen;
        uint32_t run_id; uint8_t vis; uint32_t ddvr_grp = 0;
    };
    const int32_t nq = ubatch.n_tokens;
    std::vector<snap_hot> hot_all;
    std::vector<snap_cold> cold_all;
    // Descriptors are keyed by (payload, DDVR group): the same payload in
    // different effective-position groups MUST NOT share one descriptor.
    struct desc_key {
        uint64_t pid = 0;
        uint32_t grp = 0;
        bool operator==(const desc_key & o) const { return pid == o.pid && grp == o.grp; }
    };
    struct desc_hash {
        size_t operator()(const desc_key & k) const noexcept {
            uint64_t h = k.pid ^ ((uint64_t) k.grp * 0x9e3779b97f4a7c15ULL);
            h ^= h >> 33;
            h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33;
            return (size_t) h;
        }
    };
    std::unordered_map<desc_key, uint32_t, desc_hash> hot_idx, cold_idx;
    std::vector<std::vector<uint32_t>> q_hot(nq), q_cold(nq);
    std::vector<std::unordered_set<uint32_t>> q_hot_set(nq), q_cold_set(nq);
    auto pool = kv->get_hot_slot_pool();
    std::unordered_map<uint64_t, xkv_hot_payload_binding> bindings;
    if (!pool) {
        for (const auto & b : store->list_hot_payload_bindings().bindings) {
            bindings.emplace(b.payload_id, b);
        }
    }
    const uint32_t n_stream = kv->get_n_stream();
    const bool use_rerot = rerot_active();
    const llama_rerot_attn_layout * rerot_layout_ptr = nullptr;
    // Validated per-query layout runs: exactly one contiguous group run per
    // query; every entry must fall inside its query's run. Malformed layouts
    // fail closed here, never silently mapping to group 0.
    struct layout_run { uint32_t g0 = UINT32_MAX; uint32_t count = 0; };
    std::vector<layout_run> layout_runs;
    if (use_rerot) {
        if (kv->get_n_stream() != 1) {
            if (err) *err = "build_xkv_graph_snapshot: RERoT layout requires a unified stream";
            return false;
        }
        try {
            rerot_layout_ptr = &get_rerot_attn_layout();
        } catch (const std::exception & e) {
            if (err) *err = std::string("build_xkv_graph_snapshot: rerot layout threw: ") + e.what();
            return false;
        }
        if (!rerot_layout_ptr || rerot_layout_ptr->empty()) {
            if (err) *err = "build_xkv_graph_snapshot: empty RERoT layout for an active batch";
            return false;
        }
        const auto & L = *rerot_layout_ptr;
        const uint32_t cells_size = kv->get_cells(0).size();
        std::string layout_err;
        if (!L.validate(cells_size, &layout_err)) {
            if (err) *err = std::string("build_xkv_graph_snapshot: invalid RERoT layout: ") + layout_err;
            return false;
        }
        if ((int32_t) L.n_queries != nq) {
            if (err) *err = "build_xkv_graph_snapshot: layout query count mismatches ubatch";
            return false;
        }
        if (L.query_offsets.size() != (size_t) nq + 1) {
            if (err) *err = "build_xkv_graph_snapshot: layout offsets size mismatch";
            return false;
        }
        for (int32_t q = 0; q < nq; ++q) {
            if (L.query_offsets[(size_t) q] > L.query_offsets[(size_t) q + 1] ||
                L.query_offsets[(size_t) q + 1] > L.entries.size()) {
                if (err) *err = "build_xkv_graph_snapshot: layout offsets not contiguous";
                return false;
            }
        }
        layout_runs.assign((size_t) nq, layout_run{});
        for (size_t gi = 0; gi < L.groups.size(); ++gi) {
            const uint32_t q = L.groups[gi].query_index;
            if (q >= (uint32_t) nq) {
                if (err) *err = "build_xkv_graph_snapshot: layout group query out of range";
                return false;
            }
            layout_run & r = layout_runs[(size_t) q];
            if (r.g0 == UINT32_MAX) {
                r.g0 = (uint32_t) gi;
                r.count = 1;
            } else {
                if (gi != r.g0 + r.count) {
                    if (err) *err = "build_xkv_graph_snapshot: query groups not contiguous";
                    return false;
                }
                r.count++;
            }
        }
    }

    for (int32_t q = 0; q < nq; ++q) {
        if (use_rerot && rerot_layout_ptr && !rerot_layout_ptr->empty()) {
            // RERoT path: membership, DDVR groups, and effective RoPE
            // positions come from the layout entries (never seq-intersection).
            // PRIVATE_CONTROL is visible to its owner; PENDING follows
            // publish rules.
            const uint32_t q_begin = rerot_layout_ptr->query_offsets[(size_t) q];
            const uint32_t q_end = rerot_layout_ptr->query_offsets[(size_t) q + 1];
            // Validated run for this query (contiguity proven above). NOTE:
            // no storage-pos secondary cutoff here: layout entries are the
            // authoritative PAC-DFS visibility and may legally exceed the
            // lane's physical query position.
            const layout_run & run = layout_runs[(size_t) q];
            const auto & cells = kv->get_cells(0); // unified stream
            for (uint32_t e = q_begin; e < q_end; ++e) {
                const auto & entry = rerot_layout_ptr->entries[e];
                const uint32_t cell = entry.key_index;
                if (cells.is_empty(cell)) {
                    continue;
                }
                const uint64_t pid = cells.payload_id_get(cell);
                if (pid == 0) {
                    continue;
                }
                const uint64_t gen = cells.storage_generation_get(cell);
                const llama_pos cp = cells.pos_get(cell);
                const auto & meta = cells.rerot_get(cell);
                xkv_location loc;
                if (!store->find_location(pid, loc)) {
                    if (err) *err = "build_xkv_graph_snapshot: payload without a store location";
                    return false;
                }
                if (run.g0 == UINT32_MAX) {
                    if (err) *err = "build_xkv_graph_snapshot: layout entry without query groups";
                    return false;
                }
                if (entry.group_index < run.g0 || entry.group_index >= run.g0 + run.count) {
                    if (err) *err = "build_xkv_graph_snapshot: layout entry outside its query run";
                    return false;
                }
                const uint32_t ddvr_grp = entry.group_index - run.g0;
                const desc_key key{pid, ddvr_grp};
                if (loc.kind == xkv_location_kind::hot) {
                    if (loc.state == xkv_state::hot_writing) {
                        // A layout entry can expose another in-flight writer:
                        // accept only this context's applied payloads.
                        bool ours = false;
                        for (const auto & ae : applied_entries) {
                            if (ae.pid == pid && ae.gen == gen) {
                                ours = true;
                                break;
                            }
                        }
                        if (!ours) {
                            continue;
                        }
                    } else if (loc.state != xkv_state::hot_committed) {
                        continue;
                    }
                    uint32_t prow = 0;
                    if (pool) {
                        xkv_hot_slot_info info;
                        if (!pool->find_payload(pid, info) || info.storage_generation != gen) {
                            if (err) *err = "build_xkv_graph_snapshot: hot binding missing or stale";
                            return false;
                        }
                        if (info.state != llama_xkv::xkv_slot_state::bound) {
                            if (err) *err = "build_xkv_graph_snapshot: hot slot is not committed/bound";
                            return false;
                        }
                        prow = info.slot;
                    } else {
                        auto bit = bindings.find(pid);
                        if (bit == bindings.end() || bit->second.storage_generation != gen) {
                            if (err) *err = "build_xkv_graph_snapshot: hot binding missing or stale";
                            return false;
                        }
                        prow = bit->second.hot_slot_row;
                    }
                    if (prow >= kv->get_hot_size()) {
                        if (err) *err = "build_xkv_graph_snapshot: hot row outside storage views";
                        return false;
                    }
                    uint32_t hi = 0;
                    auto hit = hot_idx.find(key);
                    if (hit == hot_idx.end()) {
                        hi = (uint32_t) hot_all.size();
                        hot_idx.emplace(key, hi);
                        hot_all.push_back({pid, gen, prow, (int64_t) cp, 0, loc.state, ddvr_grp});
                    } else {
                        hi = hit->second;
                    }
                    if (q_hot_set[(size_t) q].insert(hi).second) {
                        q_hot[(size_t) q].push_back(hi);
                    }
                } else if (loc.kind == xkv_location_kind::factored) {
                    uint32_t ci = 0;
                    auto cit = cold_idx.find(key);
                    if (cit == cold_idx.end()) {
                        ci = (uint32_t) cold_all.size();
                        cold_idx.emplace(key, ci);
                        cold_all.push_back({pid, gen, (int64_t) cp, loc.segment_id,
                            loc.segment_version, loc.row, loc.storage_generation,
                            meta.run_id, (uint8_t) meta.visibility, ddvr_grp});
                    } else {
                        ci = cit->second;
                    }
                    if (q_cold_set[(size_t) q].insert(ci).second) {
                        q_cold[(size_t) q].push_back(ci);
                    }
                } else {
                    if (err) *err = "build_xkv_graph_snapshot: flat quantized rows have no snapshot path";
                    return false;
                }
            }
            continue;
        }
        // Ordinary path: query-specific sequence intersection + causal check.
        std::vector<uint32_t> qstreams;
        for (int32_t k = 0; k < ubatch.n_seq_id[q]; ++k) {
            const llama_seq_id sq = ubatch.seq_id[q][k];
            if (!kv->validate_seq_id(sq)) {
                if (err) *err = "build_xkv_graph_snapshot: query seq id invalid";
                return false;
            }
            const uint32_t st = kv->get_stream_for_seq(sq);
            if (st >= n_stream) {
                if (err) *err = "build_xkv_graph_snapshot: query stream out of range";
                return false;
            }
            if (std::find(qstreams.begin(), qstreams.end(), st) == qstreams.end()) {
                qstreams.push_back(st);
            }
        }
        for (uint32_t st : qstreams) {
            llama_seq_id rep = -1;
            for (int32_t k = 0; k < ubatch.n_seq_id[q]; ++k) {
                if (kv->get_stream_for_seq(ubatch.seq_id[q][k]) == st) {
                    rep = ubatch.seq_id[q][k];
                    break;
                }
            }
            if (rep < 0) {
                if (err) *err = "build_xkv_graph_snapshot: no representative seq for stream";
                return false;
            }
            const llama_kv_cells & cells = kv->get_cells(rep);
            for (uint32_t c = 0; c < cells.size(); ++c) {
                if (cells.is_empty(c)) {
                    continue;
                }
                // Query-specific visibility: the cell seq bitset must
                // intersect this query token's seq-id set (never seq_id[0]
                // alone); shared-prefix refs match each keeping query.
                bool seq_hit = false;
                for (int32_t k = 0; k < ubatch.n_seq_id[q]; ++k) {
                    if (cells.seq_has(c, ubatch.seq_id[q][k])) {
                        seq_hit = true;
                        break;
                    }
                }
                if (!seq_hit) {
                    continue;
                }
                const bool causal = rt->get_cparams().causal_attn;
                const llama_pos cp = cells.pos_get(c);
                if (causal && cp > ubatch.pos[q]) {
                    continue; // causal: future rows invisible to this query
                }
                const auto & meta = cells.rerot_get(c);
                const auto vis = meta.visibility;
                if (vis != llama_rerot_visibility::normal &&
                    vis != llama_rerot_visibility::public_live) {
                    continue; // PRIVATE_CONTROL / PENDING_RECORD stay isolated
                }
                const uint64_t pid = cells.payload_id_get(c);
                if (pid == 0) {
                    continue;
                }
                const uint64_t gen = cells.storage_generation_get(c);
                xkv_location loc;
                if (!store->find_location(pid, loc)) {
                    if (err) *err = "build_xkv_graph_snapshot: payload without a store location";
                    return false;
                }
                if (loc.kind == xkv_location_kind::hot) {
                    if (loc.state != xkv_state::hot_committed && loc.state != xkv_state::hot_writing) {
                        continue;
                    }
                    if (loc.state == xkv_state::hot_writing) {
                        // Current tentative writes are accepted only when this
                        // context applied them (HotSafety applied_entries).
                        bool ours = false;
                        for (const auto & e : applied_entries) {
                            if (e.pid == pid) {
                                ours = true;
                                break;
                            }
                        }
                        if (!ours) {
                            continue; // other tentative writes are not visible
                        }
                    }
                    uint32_t prow = 0;
                    if (pool) {
                        xkv_hot_slot_info info;
                        if (!pool->find_payload(pid, info) || info.storage_generation != gen) {
                            if (err) *err = "build_xkv_graph_snapshot: hot binding missing or stale";
                            return false;
                        }
                        if (info.state != llama_xkv::xkv_slot_state::bound) {
                            if (err) *err = "build_xkv_graph_snapshot: hot slot is not committed/bound";
                            return false;
                        }
                        prow = info.slot;
                    } else {
                        auto bit = bindings.find(pid);
                        if (bit == bindings.end() || bit->second.storage_generation != gen) {
                            if (err) *err = "build_xkv_graph_snapshot: hot binding missing or stale";
                            return false;
                        }
                        prow = bit->second.hot_slot_row;
                    }
                    if (prow >= kv->get_hot_size()) {
                        if (err) *err = "build_xkv_graph_snapshot: hot row outside storage views";
                        return false;
                    }
                    uint32_t hi = 0;
                    auto hit = hot_idx.find(desc_key{pid, 0});
                    if (hit == hot_idx.end()) {
                        hi = (uint32_t) hot_all.size();
                        hot_idx.emplace(desc_key{pid, 0}, hi);
                        hot_all.push_back({pid, gen, prow, (int64_t) cp, st, loc.state});
                    } else {
                        hi = hit->second;
                    }
                    if (q_hot_set[(size_t) q].insert(hi).second) {
                        q_hot[(size_t) q].push_back(hi);
                    }
                } else if (loc.kind == xkv_location_kind::factored) {
                    uint32_t ci = 0;
                    auto cit = cold_idx.find(desc_key{pid, 0});
                    if (cit == cold_idx.end()) {
                        ci = (uint32_t) cold_all.size();
                        cold_idx.emplace(desc_key{pid, 0}, ci);
                        cold_all.push_back({pid, gen, (int64_t) cp, loc.segment_id,
                            loc.segment_version, loc.row, loc.storage_generation,
                            meta.run_id, (uint8_t) vis, 0});
                    } else {
                        ci = cit->second;
                    }
                    if (q_cold_set[(size_t) q].insert(ci).second) {
                        q_cold[(size_t) q].push_back(ci);
                    }
                } else {
                    if (err) *err = "build_xkv_graph_snapshot: flat quantized rows have no snapshot path";
                    return false;
                }
            }
        }
    }
    if (hot_all.empty() && cold_all.empty()) {
        if (err) *err = "build_xkv_graph_snapshot: no visible rows for any query";
        return false;
    }
    // Group factored rows per segment for pinned read views.
    struct seg_group {
        uint64_t seg_id = 0; uint64_t seg_ver = 0;
        std::vector<uint32_t> cold_indices; // into cold_all
    };
    std::vector<seg_group> seg_groups;
    for (uint32_t ci = 0; ci < (uint32_t) cold_all.size(); ++ci) {
        const auto & c = cold_all[ci];
        seg_group * sg = nullptr;
        for (auto & g : seg_groups) {
            if (g.seg_id == c.seg_id && g.seg_ver == c.seg_ver) {
                sg = &g;
                break;
            }
        }
        if (!sg) {
            seg_groups.push_back({c.seg_id, c.seg_ver, {}});
            sg = &seg_groups.back();
        }
        sg->cold_indices.push_back(ci);
    }
    auto snap = std::unique_ptr<xkv_graph_snapshot>(new xkv_graph_snapshot());
    snap->head_dim_k = hd_k;
    snap->head_dim_v = hd_v;
    snap->n_q_heads = q_per_kv;
    snap->n_q_heads_total = n_q;
    snap->n_queries = (uint32_t) nq;
    snap->scale = kq_scale;
    snap->logit_softcap = logit_softcap;
    snap->expected_stamp = store->current_stamp();
    snap->store = store.get();
    snap->kv_head_index = kv_head;
    snap->q_group_begin = kv_head * q_per_kv;
    if (use_rerot && rerot_layout_ptr && !rerot_layout_ptr->empty()) {
        // Populate exact per-query DDVR group counts and offsets.
        snap->query_ddvr_group_counts.resize((size_t) nq, 0);
        for (const auto & rg : rerot_layout_ptr->groups) {
            if (rg.query_index < (uint32_t) nq) {
                snap->query_ddvr_group_counts[rg.query_index]++;
            }
        }
        snap->n_ddvr_groups = 0;
        for (uint32_t c : snap->query_ddvr_group_counts) {
            snap->n_ddvr_groups = std::max(snap->n_ddvr_groups, c);
        }
    } else {
        snap->n_ddvr_groups = 0;
        snap->query_ddvr_group_counts.clear();
    }
    snap->query_causal_limits.assign((size_t) nq, -1);
    const bool causal = rt->get_cparams().causal_attn;
    if (!use_rerot && causal) {
        for (int32_t q = 0; q < nq; ++q) {
            snap->query_causal_limits[(size_t) q] = (int64_t) ubatch.pos[q];
        }
    }
    // RERoT path keeps -1 (no secondary storage cutoff): layout entries
    // are the authoritative PAC-DFS visibility and may legally exceed a
    // lane's physical query position.
    // Physical hot storage descriptor (graph validates views against it).
    snap->hot_layout.k_type = kv->layer_type_k(owning);
    snap->hot_layout.v_type = kv->layer_type_v(owning);
    snap->hot_layout.head_dim_k = hd_k;
    snap->hot_layout.head_dim_v = hd_v;
    const bool k_turbo = (snap->hot_layout.k_type == GGML_TYPE_TURBO2_0 || snap->hot_layout.k_type == GGML_TYPE_TURBO3_0 ||
                            snap->hot_layout.k_type == GGML_TYPE_TURBO4_0);
    const bool v_turbo = (snap->hot_layout.v_type == GGML_TYPE_TURBO2_0 || snap->hot_layout.v_type == GGML_TYPE_TURBO3_0 ||
                            snap->hot_layout.v_type == GGML_TYPE_TURBO4_0);
    snap->hot_layout.padded_k = (k_turbo && hd_k % 128 != 0) ? ((hd_k + 127) / 128 * 128) : 0;
    snap->hot_layout.padded_v = (v_turbo && hd_v % 128 != 0) ? ((hd_v + 127) / 128 * 128) : 0;
    snap->hot_layout.v_transposed = kv->get_v_trans();
    snap->hot_storage_host_resident =
        (xkv_expected_residency(rt->get_cparams()) == GGML_XKV_RES_REFERENCE_HOST);
    // Attention custom rotation setup:
    // When K or V attention rotation is enabled on the cache, populate
    // inverse rotation matrices and dimensions into snapshot so CPU hot gather
    // can invert them. If rotation is configured but matrix is missing, fail closed.
    if (kv->get_attn_rot_k()) {
        int32_t nr_k = kv->get_attn_rot_k_nrot();
        if (nr_k <= 0) {
            if (err) *err = "build_xkv_graph_snapshot: invalid attn_rot_k_nrot";
            return false;
        }
        const auto & hads = kv->get_attn_rot_hadamard();
        auto hit = hads.find((int64_t) nr_k);
        if (hit == hads.end() || hit->second.empty()) {
            if (err) *err = "build_xkv_graph_snapshot: missing attention rotation Hadamard matrix for K";
            return false;
        }
        snap->hot_k_inv_rot = hit->second;
        snap->hot_k_rot_dim = (uint32_t) nr_k;
    }
    if (kv->get_attn_rot_v()) {
        int32_t nr_v = kv->get_attn_rot_v_nrot();
        if (nr_v <= 0) {
            if (err) *err = "build_xkv_graph_snapshot: invalid attn_rot_v_nrot";
            return false;
        }
        const auto & hads = kv->get_attn_rot_hadamard();
        auto hit = hads.find((int64_t) nr_v);
        if (hit == hads.end() || hit->second.empty()) {
            if (err) *err = "build_xkv_graph_snapshot: missing attention rotation Hadamard matrix for V";
            return false;
        }
        snap->hot_v_inv_rot = hit->second;
        snap->hot_v_rot_dim = (uint32_t) nr_v;
    }
    // Hot rows: storage-gather descriptors, no owned copies. Every emitted
    // cell is below the hot storage bound (checked at enumeration).
    snap->hot_data.reserve(hot_all.size());
    for (size_t r = 0; r < hot_all.size(); ++r) {
        const auto & h = hot_all[r];
        xkv_graph_snapshot::hot_row_data hd;
        hd.payload_id = h.pid;
        hd.row_index = h.prow;
        hd.storage_pos = h.spos;
        hd.storage_generation = h.gen;
        hd.expected_state = h.state; // hot_writing accepted for current writes
        hd.group_index = h.group_index; // exact DDVR group index (RERoT layout)
        hd.is_valid = true;
        hd.query_visibility.assign((size_t) nq, false);
        hd.cell = (int64_t) h.prow;
        hd.kv_head = kv_head;
        hd.stream = h.stream;
        snap->hot_data.push_back(std::move(hd));
    }
    for (int32_t q = 0; q < nq; ++q) {
        for (uint32_t hi : q_hot[(size_t) q]) {
            snap->hot_data[(size_t) hi].query_visibility[(size_t) q] = true;
        }
    }
    // Factored rows: pinned per-segment read views with exact selections.
    for (const auto & sg : seg_groups) {
        xkv_reader_pin pin = store->pin_segment_version(sg.seg_id, sg.seg_ver);
        if (!pin) {
            if (err) *err = "build_xkv_graph_snapshot: cannot pin segment version";
            return false;
        }
        xkv_segment_read_view view;
        view.pin = std::move(pin);
        view.segment_version_id = sg.seg_ver;
        view.owning_layer = owning;
        view.kv_head = kv_head;
        for (uint32_t ci : sg.cold_indices) {
            const auto & c = cold_all[ci];
            view.selected_rows.push_back(c.seg_row);
            view.storage_positions.push_back(c.spos);
            view.row_generations.push_back(c.gen);
            view.group_indices.push_back(c.ddvr_grp); // exact DDVR group (RERoT)
        }
        // Populate per-query row visibility: [q][r] is true iff row r is in q_cold_set[q]
        view.query_row_visibility.resize((size_t) nq);
        view.query_selected_row_indices.resize((size_t) nq);
        for (int32_t q = 0; q < nq; ++q) {
            view.query_row_visibility[(size_t) q].resize(sg.cold_indices.size(), false);
            for (size_t r = 0; r < sg.cold_indices.size(); ++r) {
                const uint32_t ci = sg.cold_indices[r];
                if (q_cold_set[(size_t) q].count(ci)) {
                    view.query_row_visibility[(size_t) q][r] = true;
                    view.query_selected_row_indices[(size_t) q].push_back((uint32_t) r);
                }
            }
        }
        snap->segment_views.push_back(std::move(view));
    }
    // Exact device snapshot contract: every DEVICE_OWNED pinned segment wires
    // one arena per (segment, factor group) with a_k/b_k/a_v/b_v/landmarks
    // matched by full codec descriptor fingerprint + role + shape (never handle
    // order). native_arenas_present is set only after every device segment
    // validates completely; anything else fails closed (no host fallback).
    // Lifetime: tensors are borrowed from backend-bundle allocations pinned by
    // the segment views above. native_rope_tables stays backend-owner-wired
    // (the builder owns no backend context for backend-resident tables).
    bool have_device_segment = false;
    for (const auto & sg : seg_groups) {
        const xkv_segment * seg = nullptr;
        for (const auto & v : snap->segment_views) {
            const xkv_segment * s = v.get_segment();
            if (s && s->segment_id == sg.seg_id) {
                seg = s;
                break;
            }
        }
        if (!seg) {
            if (err) *err = "build_xkv_graph_snapshot: pinned segment lost";
            return false;
        }
        if (seg->residency != GGML_XKV_RES_DEVICE_OWNED) {
            continue;
        }
        have_device_segment = true;
        const xkv_factor_group_payload * agp = seg->find_group(grp->group_index);
        if (!agp) {
            if (err) *err = "build_xkv_graph_snapshot: group missing from pinned device segment";
            return false;
        }
        if (!seg->backend_bundle) {
            if (err) *err = "build_xkv_graph_snapshot: DEVICE_OWNED segment missing backend bundle";
            return false;
        }
        if (!agp->b_k || !agp->b_v) {
            if (err) *err = "build_xkv_graph_snapshot: device segment missing shared factor descriptors";
            return false;
        }
        const xkv_backend_batch_result & bundle = *seg->backend_bundle;
        xkv_graph_snapshot::xkv_native_group_arenas arenas;
        arenas.segment_id = seg->segment_id;
        arenas.segment_version = seg->segment_version;
        arenas.group_index = grp->group_index;
        arenas.a_k = sr_match_device_stream(bundle, agp->a_k.desc, "a_k", err);
        if (!arenas.a_k) return false;
        arenas.b_k = sr_match_device_stream(bundle, agp->b_k->desc, "b_k", err);
        if (!arenas.b_k) return false;
        arenas.a_v = sr_match_device_stream(bundle, agp->a_v.desc, "a_v", err);
        if (!arenas.a_v) return false;
        arenas.b_v = sr_match_device_stream(bundle, agp->b_v->desc, "b_v", err);
        if (!arenas.b_v) return false;
        arenas.landmarks = sr_match_device_stream(bundle, agp->landmark.desc, "landmarks", err);
        if (!arenas.landmarks) return false;
        snap->native_group_arenas.push_back(arenas);
    }
    if (have_device_segment) {
        snap->native_arenas_present = true;
    }
    snap->max_hot_rows = snap->hot_data.size();
    snap->max_hot_bytes = snap->hot_data.size() * sizeof(xkv_graph_snapshot::hot_row_data);
    // Exact B feature slice for this (owning layer, KV head).
    int32_t li = -1;
    for (size_t k = 0; k < grp->owning_layers.size(); ++k) {
        if (grp->owning_layers[k] == owning) {
            li = (int32_t) k;
            break;
        }
    }
    if (li < 0 || (size_t) li >= grp->layer_feature_offsets_k.size()) {
        if (err) *err = "build_xkv_graph_snapshot: layer missing from group map";
        return false;
    }
    snap->sr_feature_offset = grp->layer_feature_offsets_k[(size_t) li] + kv_head * hd_k;
    snap->sr_feature_dim = hd_k;
    snap->sr_config.sr_budget = rt->get_cparams().xkv_sr_budget;
    snap->sr_config.landmark_type = rt->get_cparams().xkv_landmark_type;
    snap->sr_config.refine_mode = rt->get_cparams().xkv_landmark_refine;
    snap->sr_config.refine_max_rows = rt->get_cparams().xkv_landmark_refine_max_rows;
    std::vector<xkv_landmark_phase_layer> pl;
    uint64_t pfp = 0;
    if (!build_landmark_phase_layers(*kv, rt->get_cparams(), hparams, *grp, pl, pfp, err)) {
        return false;
    }
    const xkv_landmark_phase_layer * play = nullptr;
    for (size_t pli = 0; pli < pl.size(); ++pli) {
        // pl parallels grp->owning_layers order.
        if (pli < grp->owning_layers.size() && grp->owning_layers[pli] == owning) {
            play = &pl[pli];
            break;
        }
    }
    if (!play) {
        if (err) *err = "build_xkv_graph_snapshot: phase tables missing for owning layer";
        return false;
    }
    const uint32_t phd = play->head_dim;
    const uint32_t prd = play->rotary_dim;
    const std::vector<float> pom = play->omega;
    const std::vector<float> pfs = play->freq_scale_sq;
    phase_transform_fn frag_phase = [phd, prd, pom, pfs](
        const float * src, int64_t pos, uint32_t /*head_idx*/, float * dst) {
        apply_forward_rope_head(src, dst, phd, prd, pom.data(), pfs.data(), pos);
    };
    snap->phase_tx = frag_phase;

    const llama_xkv_mode snap_mode = rt->get_cparams().xkv_mode;
    if (snap_mode == LLAMA_XKV_MODE_SR) {
        // SR-after-Q: selection runs in compute; builder pins legal
        // landmark fragments per segment for this head.
        snap->sr_mode = 1;
        snap->sr_phase_fingerprint = pfp;
        const uint32_t chunk = rt->effective_config().chunk_tokens;
        if (chunk == 0) {
            if (err) *err = "build_xkv_graph_snapshot: chunk_tokens must be non-zero (no fallback)";
            return false;
        }
        if (nq <= 0) {
            if (err) *err = "build_xkv_graph_snapshot: no parent queries";
            return false;
        }
        if ((size_t) nq > snap->query_causal_limits.size()) {
            if (err) *err = "build_xkv_graph_snapshot: causal limits missing for parent queries";
            return false;
        }
        // E expanded selector queries in parent-major order: ng slots per parent
        // query (RERoT layout counts, else exactly one slot). [parent, slot] are
        // the first two selector columns; rows dedup by (row, slot) downstream.
        const bool have_group_counts = use_rerot && rerot_layout_ptr && !rerot_layout_ptr->empty() &&
            !snap->query_ddvr_group_counts.empty();
        std::vector<uint32_t> query_ng((size_t) nq, 1);
        uint64_t n_expanded = 0;
        for (int32_t q = 0; q < nq; ++q) {
            uint32_t ng = 1;
            if (have_group_counts) {
                if ((size_t) q >= snap->query_ddvr_group_counts.size()) {
                    if (err) *err = "build_xkv_graph_snapshot: DDVR group counts missing for parent query";
                    return false;
                }
                ng = snap->query_ddvr_group_counts[(size_t) q];
                if (ng == 0) ng = 1;
            }
            query_ng[(size_t) q] = ng;
            if (!ck_add_u64(n_expanded, (uint64_t) ng, n_expanded)) {
                if (err) *err = "build_xkv_graph_snapshot: expanded query count overflow";
                return false;
            }
            // Consistency: every cold row's slot must lie inside its query's
            // expansion, or E is wrong and rows would be silently dropped.
            for (uint32_t ci : q_cold[(size_t) q]) {
                if ((size_t) ci >= cold_all.size()) {
                    if (err) *err = "build_xkv_graph_snapshot: cold index out of range";
                    return false;
                }
                if (cold_all[(size_t) ci].ddvr_grp >= ng) {
                    if (err) *err = "build_xkv_graph_snapshot: cold row slot outside expanded query count";
                    return false;
                }
            }
        }
        // Bounded metadata: plan rows are subsets of cold rows, so the global
        // fragment count never exceeds E x cold rows (checked, enforced below).
        uint64_t frag_cap = 0;
        if (!ck_mul_u64(n_expanded, (uint64_t) cold_all.size(), frag_cap)) {
            if (err) *err = "build_xkv_graph_snapshot: fragment cap overflow";
            return false;
        }
        // Borrowed-backend rope bridge: [omega, direct_mag] for this layer/head
        // from the exact phase tables above (never derived in graph). Empty iff
        // rotary_dim == 0, when the graph substitutes an empty tensor.
        snap->native_rope_table_data.clear();
        if (play->rotary_dim != 0) {
            if ((play->rotary_dim & 1u) != 0 || play->rotary_dim > play->head_dim) {
                if (err) *err = "build_xkv_graph_snapshot: invalid rotary geometry for rope bridge";
                return false;
            }
            const uint32_t fc = play->rotary_dim / 2u;
            if (play->omega.size() != (size_t) fc || play->freq_scale_sq.size() != (size_t) fc) {
                if (err) *err = "build_xkv_graph_snapshot: rope table size mismatch";
                return false;
            }
            std::vector<float> rope_data;
            rope_data.reserve((size_t) play->rotary_dim);
            for (uint32_t f = 0; f < fc; ++f) {
                const float w = play->omega[(size_t) f];
                if (!std::isfinite(w)) {
                    if (err) *err = "build_xkv_graph_snapshot: non-finite rope omega";
                    return false;
                }
                rope_data.push_back(w);
            }
            for (uint32_t f = 0; f < fc; ++f) {
                const float s = play->freq_scale_sq[(size_t) f];
                if (!std::isfinite(s) || !(s >= 0.0f)) {
                    if (err) *err = "build_xkv_graph_snapshot: invalid rope scale";
                    return false;
                }
                const float m = sqrtf(s);
                if (!std::isfinite(m) || !(m > 0.0f)) {
                    if (err) *err = "build_xkv_graph_snapshot: non-positive rope magnitude";
                    return false;
                }
                rope_data.push_back(m);
            }
            if (rope_data.size() != (size_t) play->rotary_dim) {
                if (err) *err = "build_xkv_graph_snapshot: rope table assembly short";
                return false;
            }
            snap->native_rope_table_data = std::move(rope_data);
        }
        for (const auto & sg : seg_groups) {
            const xkv_segment * seg = nullptr;
            for (const auto & v : snap->segment_views) {
                const xkv_segment * s = v.get_segment();
                if (s && s->segment_id == sg.seg_id) {
                    seg = s;
                    break;
                }
            }
            if (!seg) {
                if (err) *err = "build_xkv_graph_snapshot: pinned segment lost";
                return false;
            }
            // Factor group + pin for this segment (resolved once; every plan
            // below only reads through them).
            const xkv_factor_group_payload * gp = seg->find_group(grp->group_index);
            if (!gp) {
                if (err) *err = "build_xkv_graph_snapshot: group missing from pinned segment";
                return false;
            }
            std::shared_ptr<const xkv_segment> seg_hold =
                store->get_segment_version(sg.seg_id, sg.seg_ver);
            if (!seg_hold) {
                if (err) *err = "build_xkv_graph_snapshot: pinned segment version lost";
                return false;
            }
            const bool is_device_owned = (seg->residency == GGML_XKV_RES_DEVICE_OWNED);
            // Per-(parent query, DDVR slot) plans over this segment: metadata
            // derives only from the plan's legal cold rows (membership in q,
            // slot match, causal cutoff). Parent-major order; shared rows
            // duplicate across slots as separate fragments.
            for (int32_t q = 0; q < nq; ++q) {
                const int64_t cutoff = snap->query_causal_limits[(size_t) q];
                for (uint32_t g = 0; g < query_ng[(size_t) q]; ++g) {
                    std::vector<sr_plan_cold_row> plan;
                    for (uint32_t ci : sg.cold_indices) {
                        if ((size_t) ci >= cold_all.size()) {
                            if (err) *err = "build_xkv_graph_snapshot: cold index out of range";
                            return false;
                        }
                        if (q_cold_set[(size_t) q].count(ci) == 0) continue;
                        const auto & c = cold_all[(size_t) ci];
                        if (c.ddvr_grp != g) continue;
                        sr_plan_cold_row r;
                        r.seg_row = c.seg_row;
                        r.pid = c.pid;
                        r.gen = c.gen;
                        r.spos = c.spos;
                        r.run_id = c.run_id;
                        r.vis = c.vis;
                        plan.push_back(r);
                    }
                    if (plan.empty()) continue;
                    // build_legal_fragments needs strictly sorted, spos-unique
                    // rows; conflicting identities fail closed (never merged).
                    std::sort(plan.begin(), plan.end(),
                        [](const sr_plan_cold_row & a, const sr_plan_cold_row & b) {
                            if (a.spos != b.spos) return a.spos < b.spos;
                            return a.seg_row < b.seg_row;
                        });
                    for (size_t k = 1; k < plan.size(); ++k) {
                        if (plan[k].spos == plan[k - 1].spos) {
                            if (err) *err = "build_xkv_graph_snapshot: conflicting cold rows share a storage position";
                            return false;
                        }
                    }
                    if (!build_sr_fragment_plan(*seg, *gp, seg_hold, owning, kv_head,
                            (uint32_t) q, g, cutoff, chunk, pfp, frag_phase,
                            rt->get_cparams().xkv_landmark_type, snap->expected_stamp,
                            (size_t) nq, plan, is_device_owned,
                            snap->sr_feature_offset, snap->sr_feature_dim,
                            snap->sr_legal_frags, snap->native_landmark_rebuild_requests, err)) {
                        return false;
                    }
                    if ((uint64_t) snap->sr_legal_frags.size() > frag_cap) {
                        if (err) *err = "build_xkv_graph_snapshot: fragment metadata exceeds checked bound";
                        return false;
                    }
                }
            }
        }
    } else {
        // precomputed selection is the causal per-query factored set.
        // Hot rows flow through hot_data, not the CSR.
        snap->sr_mode = 0;
        sr_batch_selection_result sel;
        sel.csr_ptrs.assign((size_t) nq + 1, 0);
        sel.per_query.resize((size_t) nq);
        for (int32_t q = 0; q < nq; ++q) {
            sr_selection_result & pq = sel.per_query[(size_t) q];
            for (uint32_t ci : q_cold[(size_t) q]) {
                const auto & c = cold_all[ci];
                segment_row_ref ref;
                ref.segment_id = c.seg_id;
                ref.segment_version = c.seg_ver;
                ref.storage_generation = c.seg_gen;
                ref.row = c.seg_row;
                // Same row may be visible in several DDVR groups for one
                // query: gather it once (group fan-out lives in the views).
                bool dup = false;
                for (const auto & have : pq.selected_rows) {
                    if (have.segment_id == ref.segment_id &&
                        have.segment_version == ref.segment_version &&
                        have.storage_generation == ref.storage_generation &&
                        have.row == ref.row) {
                        dup = true;
                        break;
                    }
                }
                if (dup) {
                    continue;
                }
                pq.selected_rows.push_back(ref);
                sel.gather_rows.push_back(ref);
                sel.csr_indices.push_back((uint32_t) sel.gather_rows.size() - 1);
                sel.csr_group_indices.push_back(cold_all[ci].ddvr_grp);
            }
            sel.csr_ptrs[(size_t) q + 1] = (uint32_t) sel.csr_indices.size();
        }
        snap->sr_selection = std::move(sel);
    }
    snap->reader_config.workspace_budget_bytes =
        (size_t) rt->get_cparams().xkv_workspace_mib * 1024 * 1024;
    // Shared family B-tile cache (never a per-reader cache). Native paths
    // that bypass it report honest zeros via decode_cache_metrics().
    snap->reader_config.b_cache = rt->get_b_tile_cache();
    // Native arenas stay absent until the builder wires segment arenas;
    // native build then fails closed instead of falling back silently.
    if (!snap->validate_hot_rows(err)) {
        return false;
    }
    snap->reader_guard = std::move(reader_guard);
    // Guard acquire/release hooks for graph reuse (idempotent: acquire
    // replaces the guard, releasing the old one; release on empty is a
    // no-op; graph flags guard doubles). Installed only when the builder
    // acquired a coordinator lease above; otherwise vacuous.
    if (auto * coord = rt->get_coordinator()) {
        xkv_graph_snapshot * raw = snap.get();
        snap->guard_release_hook = [raw]() {
            raw->reader_guard.release();
        };
        snap->guard_acquire_hook = [raw, coord](std::string & hook_err) -> bool {
            raw->reader_guard = coord->acquire_reader();
            if (!raw->reader_guard) {
                hook_err = "build_xkv_graph_snapshot: reader lease re-acquire refused";
                return false;
            }
            return true;
        };
    }
    out = std::move(snap);
    return true;
}

namespace llama_xkv {

bool build_landmark_phase_layers(
    const llama_kv_cache & kv,
    const llama_cparams & cparams,
    const llama_hparams & hparams,
    const layer_group & group,
    std::vector<xkv_landmark_phase_layer> & out_layers,
    uint64_t & out_fingerprint,
    std::string * err) {
    out_layers.clear();
    out_fingerprint = 0;
    if (hparams.rope_type == LLAMA_ROPE_TYPE_NONE) {
        for (size_t li = 0; li < group.owning_layers.size(); ++li) {
            const uint32_t ol = group.owning_layers[li];
            xkv_landmark_phase_layer l;
            l.feature_offset = group.layer_feature_offsets_k[li];
            l.feature_dim = group.layer_feature_dims_k[li];
            l.n_heads = hparams.n_head_kv(ol);
            l.head_dim = hparams.n_embd_head_k(ol);
            l.rotary_dim = 0;
            out_layers.push_back(std::move(l));
        }
        out_fingerprint = fingerprint_landmark_phase(out_layers);
        return true;
    }
    if (hparams.rope_type != LLAMA_ROPE_TYPE_NEOX && hparams.rope_type != LLAMA_ROPE_TYPE_IMROPE) {
        if (err) *err = "phase layers: unsupported rope type (NeoX/text-IMRoPE only)";
        return false;
    }
    if (group.owning_layers.size() != group.layer_feature_offsets_k.size() ||
        group.owning_layers.size() != group.layer_feature_dims_k.size()) {
        if (err) *err = "phase layers: group feature map size mismatch";
        return false;
    }
    const llama_model & model = kv.get_model();
    for (size_t li = 0; li < group.owning_layers.size(); ++li) {
        const uint32_t ol = group.owning_layers[li];
        xkv_landmark_phase_layer l;
        l.feature_offset = group.layer_feature_offsets_k[li];
        l.feature_dim = group.layer_feature_dims_k[li];
        l.n_heads = hparams.n_head_kv(ol);
        l.head_dim = hparams.n_embd_head_k(ol);
        l.rotary_dim = hparams.n_rot(ol);
        if (l.n_heads == 0 || l.head_dim == 0) {
            if (err) *err = "phase layers: degenerate head geometry";
            return false;
        }
        if (l.feature_dim != (uint64_t) l.n_heads * l.head_dim) {
            if (err) *err = "phase layers: feature dim does not match head geometry";
            return false;
        }
        if ((l.rotary_dim & 1u) != 0 || l.rotary_dim > l.head_dim) {
            if (err) *err = "phase layers: odd or oversized rotary_dim";
            return false;
        }
        if (l.rotary_dim == 0) {
            out_layers.push_back(std::move(l));
            continue;
        }
        const uint32_t fc = l.rotary_dim / 2;
        l.omega.assign(fc, 0.0f);
        l.freq_scale_sq.assign(fc, 1.0f);
        // Mirror the canonical derivation exactly: explicit rope factors when
        // the model provides them, otherwise the shared table builder, so the
        // phase tables are bit-identical to canonical reads.
        std::vector<float> freq_factors;
        const float * factor_ptr = nullptr;
        if ((size_t) ol < model.layers.size()) {
            if (ggml_tensor * ft = model.get_rope_factors(cparams, (int) ol)) {
                if (ft->type != GGML_TYPE_F32 || ft->ne[0] < (int64_t) fc) {
                    if (err) *err = "phase layers: rope factor tensor must be F32 with sufficient entries";
                    return false;
                }
                freq_factors.assign(fc, 0.0f);
                ggml_backend_tensor_get(ft, freq_factors.data(), 0, fc * sizeof(float));
                factor_ptr = freq_factors.data();
            }
        }
        const float freq_base = (cparams.rope_freq_base > 0.0f) ? model.get_rope_freq_base(cparams, (int) ol) : model.hparams.rope_freq_base_train;
        const float freq_scale = (cparams.rope_freq_scale > 0.0f) ? model.get_rope_freq_scale(cparams, (int) ol) : 1.0f;
        const float attn_factor = (cparams.yarn_attn_factor > 0.0f) ? cparams.yarn_attn_factor : 1.0f;
        if (!triattention_build_rope_tables(
                l.omega.data(), l.freq_scale_sq.data(), l.rotary_dim,
                freq_base, freq_scale,
                (int32_t) cparams.n_ctx_orig_yarn,
                cparams.yarn_ext_factor, attn_factor,
                cparams.yarn_beta_fast, cparams.yarn_beta_slow, factor_ptr)) {
            if (err) *err = "phase layers: rope table build failed";
            return false;
        }
        out_layers.push_back(std::move(l));
    }
    out_fingerprint = fingerprint_landmark_phase(out_layers);
    if (out_fingerprint == 0) {
        if (err) *err = "phase layers: degenerate fingerprint";
        return false;
    }
    return true;
}

xkv_group_landmark_rebuild_fn llama_xkv_runtime::group_landmark_rebuild_fn() const {
    const ggml_type lm_type = cparams_.xkv_landmark_type;
    const uint32_t chunk = eff_.chunk_tokens;
    return [lm_type, chunk](
        const xkv_segment & /*segment*/,
        const xkv_factor_group_payload & group,
        const std::vector<uint32_t> & surviving_rows,
        const std::vector<int64_t> & storage_positions,
        const std::vector<xkv_landmark_phase_layer> & phase_layers,
        uint64_t phase_tx_fingerprint,
        encoded_matrix & out_landmark,
        std::string * err) -> bool {
        if (group.b_k == nullptr) {
            if (err) *err = "group rebuild: B_K is null";
            return false;
        }
        // Decode FINAL streams (padded) and unpad to logical matrices.
        std::vector<float> a_dec, b_dec;
        try {
            a_dec = decode_matrix(group.a_k);
            b_dec = decode_matrix(*group.b_k);
        } catch (const std::exception & e) {
            if (err) *err = std::string("group rebuild: decode failed: ") + e.what();
            return false;
        }
        const uint64_t a_rows = group.a_k.desc.logical_shape.rows;
        const uint64_t rank = group.a_k.desc.logical_shape.cols;
        const uint64_t a_pad = group.a_k.desc.padded_shape.cols;
        const uint64_t dim = group.total_dim_k;
        const uint64_t b_pad = group.b_k->desc.padded_shape.cols;
        if (rank == 0 || dim == 0 || a_pad < rank || b_pad < rank) {
            if (err) *err = "group rebuild: degenerate descriptors";
            return false;
        }
        if (a_dec.size() != a_rows * a_pad || b_dec.size() != dim * b_pad) {
            if (err) *err = "group rebuild: decoded size mismatch";
            return false;
        }
        matrix dec_a(a_rows, rank), dec_bt(dim, rank);
        // Cold pack path: local scratch (documented; the hot seal path uses
        // maintain-owned scratch instead). Unpad into logical-width views is
        // unnecessary: the engine strides padded rows directly.
        const uint64_t chunks = (surviving_rows.size() + chunk - 1) / (chunk > 0 ? chunk : 1);
        std::vector<float> recon(dim), phased(dim), lm(chunks * dim, 0.0f);
        return rebuild_landmarks_mean_pool_phased(
            a_dec.data(), a_rows, a_pad, b_dec.data(), dim, b_pad, rank,
            surviving_rows, storage_positions, phase_layers,
            phase_tx_fingerprint, lm_type, chunk,
            recon.data(), recon.size(), phased.data(), phased.size(),
            lm.data(), lm.size(), out_landmark, err);
    };
}

bool rebuild_candidate_landmarks(
    const xkv_segment & source,
    const std::vector<uint32_t> & surviving_rows,
    xkv_segment & candidate,
    ggml_type landmark_type,
    uint32_t chunk_tokens,
    std::string * err) {
    (void) source; (void) surviving_rows; (void) candidate;
    (void) landmark_type; (void) chunk_tokens;
    // Positions and phase are cells-owned and unavailable inside the store,
    // and an unphased pooled mean is mathematically wrong. ALWAYS refuse so
    // the caller keeps tombstones until the runtime supplies positions+phase
    // via rebuild_candidate_landmarks_with_positions.
    if (err) *err = "rebuild_candidate_landmarks: positions/phase unavailable; tombstones preserved";
    return false;
}

bool rebuild_candidate_landmarks_with_positions(
    const xkv_segment & source,
    const std::vector<uint32_t> & surviving_rows,
    const std::vector<int64_t> & storage_positions,
    const std::vector<xkv_group_phase_spec> & phase_specs,
    xkv_segment & candidate,
    ggml_type landmark_type,
    uint32_t chunk_tokens,
    std::string * err) {
    if (surviving_rows.empty()) {
        if (err) *err = "rebuild with positions: no surviving rows";
        return false;
    }
    if (chunk_tokens == 0) {
        if (err) *err = "rebuild with positions: chunk_tokens must be non-zero";
        return false;
    }
    for (auto & g : candidate.groups) {
        const xkv_factor_group_payload * src_g = source.find_group(g.group_index);
        if (!src_g) {
            if (err) *err = "rebuild with positions: candidate group missing in source";
            return false;
        }
        const xkv_group_phase_spec * spec = nullptr;
        for (const auto & s : phase_specs) {
            if (s.group_index == g.group_index) {
                spec = &s;
                break;
            }
        }
        if (!spec) {
            if (err) *err = "rebuild with positions: group lacks a phase spec";
            return false;
        }
        if (src_g->b_k == nullptr) {
            if (err) *err = "rebuild with positions: source B_K is null";
            return false;
        }
        std::vector<float> a_dec, b_dec;
        try {
            a_dec = decode_matrix(src_g->a_k);
            b_dec = decode_matrix(*src_g->b_k);
        } catch (const std::exception & e) {
            if (err) *err = std::string("rebuild with positions: decode failed: ") + e.what();
            return false;
        }
        const uint64_t a_rows = src_g->a_k.desc.logical_shape.rows;
        const uint64_t rank = src_g->a_k.desc.logical_shape.cols;
        const uint64_t a_pad = src_g->a_k.desc.padded_shape.cols;
        const uint64_t dim = src_g->total_dim_k;
        const uint64_t b_pad = src_g->b_k->desc.padded_shape.cols;
        if (rank == 0 || dim == 0 || a_pad < rank || b_pad < rank ||
            a_dec.size() != a_rows * a_pad || b_dec.size() != dim * b_pad) {
            if (err) *err = "rebuild with positions: decoded size mismatch";
            return false;
        }
        matrix dec_a(a_rows, rank), dec_bt(dim, rank);
        encoded_matrix rebuilt;
        // Checked ceil division with no fallback divisor: chunk_tokens == 0
        // was already refused above, and every size conversion is checked.
        uint64_t chunks_num = 0;
        if (!ck_add_u64((uint64_t) surviving_rows.size(), (uint64_t) chunk_tokens - 1u, chunks_num)) {
            if (err) *err = "rebuild with positions: chunk count overflow";
            return false;
        }
        const uint64_t chunks = chunks_num / chunk_tokens;
        uint64_t lm_elems = 0;
        if (!ck_mul_u64(chunks, dim, lm_elems) ||
            lm_elems > (uint64_t) std::numeric_limits<size_t>::max()) {
            if (err) *err = "rebuild with positions: landmark scratch overflow";
            return false;
        }
        if (dim > (uint64_t) std::numeric_limits<size_t>::max()) {
            if (err) *err = "rebuild with positions: feature dim overflow";
            return false;
        }
        std::vector<float> recon((size_t) dim), phased((size_t) dim), lm((size_t) lm_elems, 0.0f);
        if (!rebuild_landmarks_mean_pool_phased(
                a_dec.data(), a_rows, a_pad, b_dec.data(), dim, b_pad, rank,
                surviving_rows, storage_positions, spec->layers,
                spec->fingerprint, landmark_type, chunk_tokens,
                recon.data(), recon.size(), phased.data(), phased.size(),
                lm.data(), lm.size(), rebuilt, err)) {
            return false;
        }
        g.landmark = std::move(rebuilt);
        g.update_byte_counters();
    }
    candidate.update_byte_counters();
    return true;
}

xkv_landmark_rebuild_fn llama_xkv_runtime::store_pack_rebuild_fn() const {
    const ggml_type lm_type = cparams_.xkv_landmark_type;
    const uint32_t chunk = eff_.chunk_tokens;
    return [lm_type, chunk](
        const xkv_segment & source,
        const std::vector<uint32_t> & surviving_rows,
        xkv_segment & candidate,
        std::string * err) -> bool {
        return rebuild_candidate_landmarks(
            source, surviving_rows, candidate, lm_type, chunk, err);
    };
}

bool llama_xkv_runtime::rebuild_landmarks_for_shift(
    const llama_kv_cache & kv,
    uint64_t segment_id,
    const std::vector<std::pair<uint64_t, int64_t>> & pid_to_new_pos,
    std::vector<encoded_matrix> & out_landmarks,
    std::string * err) const {
    out_landmarks.clear();
    if (cparams_.xkv_factorizer != LLAMA_XKV_FACTORIZER_CPU_REFERENCE) {
        if (err) *err = "shift rebuild: non-CPU factorizer has no host rebuild path";
        return false;
    }
    auto store = kv.get_xkv_store();
    if (!store) {
        if (err) *err = "shift rebuild: no XKV store";
        return false;
    }
    auto seg = store->pin_segment(segment_id);
    if (!seg) {
        if (err) *err = "shift rebuild: segment not published";
        return false;
    }
    if (!refresh_phase_cache(kv, err)) {
        return false;
    }
    auto & cur_pos = removal_capture_.pid_to_pos;
    cur_pos.clear();
    const uint32_t n_stream = kv.get_n_stream();
    for (uint32_t s = 0; s < n_stream; ++s) {
        if (!kv.validate_seq_id((llama_seq_id) s) || kv.get_stream_for_seq((llama_seq_id) s) != s) {
            continue;
        }
        const llama_kv_cells & cells = kv.get_cells((llama_seq_id) s);
        for (uint32_t i = 0, n = cells.size(); i < n; ++i) {
            if (!cells.is_empty(i)) {
                const uint64_t pid = cells.payload_id_get(i);
                if (pid != 0) {
                    cur_pos.emplace_back(pid, (int64_t) cells.pos_get(i));
                }
            }
        }
    }
    auto pos_less = [](const std::pair<uint64_t, int64_t> & a,
                       const std::pair<uint64_t, int64_t> & b) { return a.first < b.first; };
    std::sort(cur_pos.begin(), cur_pos.end(), pos_less);
    auto find_pos = [&](const std::vector<std::pair<uint64_t, int64_t>> & v, uint64_t pid,
                        const std::pair<uint64_t, int64_t> *& out) -> bool {
        auto it = std::lower_bound(v.begin(), v.end(), pid,
            [](const std::pair<uint64_t, int64_t> & e, uint64_t p) { return e.first < p; });
        if (it == v.end() || it->first != pid) {
            return false;
        }
        out = &(*it);
        return true;
    };
    for (const auto & g : seg->groups) {
        const uint32_t nrows = seg->n_rows;
        std::vector<uint32_t> surviving(nrows);
        std::vector<int64_t> positions(nrows);
        for (uint32_t r = 0; r < nrows; ++r) {
            surviving[r] = r;
            const uint64_t pid = seg->row_payload_ids[r];
            const std::pair<uint64_t, int64_t> * hit = nullptr;
            if (find_pos(pid_to_new_pos, pid, hit) || find_pos(cur_pos, pid, hit)) {
                positions[r] = hit->second;
            } else {
                if (err) *err = "shift rebuild: row without old or new position";
                return false;
            }
        }
        const xkv_group_phase_spec * spec = nullptr;
        for (const auto & s : cached_phase_specs_) {
            if (s.group_index == g.group_index) {
                spec = &s;
                break;
            }
        }
        if (!spec) {
            if (err) *err = "shift rebuild: group lacks a cached phase spec";
            return false;
        }
        encoded_matrix rebuilt;
        std::vector<float> a_dec, b_dec;
        try {
            a_dec = decode_matrix(g.a_k);
            if (!g.b_k) {
                if (err) *err = "shift rebuild: source B_K is null";
                return false;
            }
            b_dec = decode_matrix(*g.b_k);
        } catch (const std::exception & e) {
            if (err) *err = std::string("shift rebuild: decode failed: ") + e.what();
            return false;
        }
        const uint64_t a_rows = g.a_k.desc.logical_shape.rows;
        const uint64_t rank = g.a_k.desc.logical_shape.cols;
        const uint64_t a_pad = g.a_k.desc.padded_shape.cols;
        const uint64_t dim = g.total_dim_k;
        const uint64_t b_pad = g.b_k->desc.padded_shape.cols;
        if (rank == 0 || dim == 0 || a_dec.size() != a_rows * a_pad ||
            b_dec.size() != dim * b_pad) {
            if (err) *err = "shift rebuild: decoded size mismatch";
            return false;
        }
        // Checked ceil division with no fallback divisor: zero chunk_tokens
        // fails closed here (never divided by a defaulted 1).
        if (eff_.chunk_tokens == 0) {
            if (err) *err = "shift rebuild: chunk_tokens must be non-zero";
            return false;
        }
        uint64_t chunks_num = 0;
        if (!ck_add_u64((uint64_t) nrows, (uint64_t) eff_.chunk_tokens - 1u, chunks_num)) {
            if (err) *err = "shift rebuild: chunk count overflow";
            return false;
        }
        const uint64_t chunks = chunks_num / eff_.chunk_tokens;
        uint64_t lm_elems = 0;
        if (!ck_mul_u64(chunks, dim, lm_elems) ||
            lm_elems > (uint64_t) std::numeric_limits<size_t>::max() ||
            dim > (uint64_t) std::numeric_limits<size_t>::max()) {
            if (err) *err = "shift rebuild: landmark scratch overflow";
            return false;
        }
        std::vector<float> recon((size_t) dim), phased((size_t) dim), lm((size_t) lm_elems, 0.0f);
        if (!rebuild_landmarks_mean_pool_phased(
                a_dec.data(), a_rows, a_pad, b_dec.data(), dim, b_pad, rank,
                surviving, positions, spec->layers, spec->fingerprint,
                cparams_.xkv_landmark_type, eff_.chunk_tokens,
                recon.data(), recon.size(), phased.data(), phased.size(),
                lm.data(), lm.size(), rebuilt, err)) {
            return false;
        }
        out_landmarks.push_back(std::move(rebuilt));
    }
    return true;
}

bool llama_xkv_runtime::refresh_phase_cache(const llama_kv_cache & kv, std::string * err) const {
    if (cached_phase_version_ == groups_version_ && !cached_phase_specs_.empty()) {
        return true;
    }
    const llama_hparams & hparams = kv.get_hparams();
    if (!ensure_groups(hparams, kv, err)) {
        return false;
    }
    cached_phase_specs_.clear();
    for (const auto & g : groups_.groups) {
        xkv_group_phase_spec spec;
        spec.group_index = g.group_index;
        if (!build_landmark_phase_layers(kv, cparams_, hparams, g, spec.layers, spec.fingerprint, err)) {
            cached_phase_specs_.clear();
            return false;
        }
        cached_phase_specs_.push_back(std::move(spec));
    }
    cached_phase_version_ = groups_version_;
    return true;
}

bool llama_xkv_runtime::capture_pack_rebuild_ctx(const llama_kv_cache & kv, std::string * err) const {
    if (!refresh_phase_cache(kv, err)) {
        return false;
    }
    removal_capture_.clear_keep_capacity();
    const uint32_t n_stream = kv.get_n_stream();
    for (uint32_t s = 0; s < n_stream; ++s) {
        if (!kv.validate_seq_id((llama_seq_id) s) || kv.get_stream_for_seq((llama_seq_id) s) != s) {
            continue;
        }
        const llama_kv_cells & cells = kv.get_cells((llama_seq_id) s);
        for (uint32_t i = 0, n = cells.size(); i < n; ++i) {
            if (cells.is_empty(i)) {
                continue;
            }
            const uint64_t pid = cells.payload_id_get(i);
            if (pid != 0) {
                removal_capture_.pid_to_pos.emplace_back(pid, (int64_t) cells.pos_get(i));
            }
        }
    }
    std::sort(removal_capture_.pid_to_pos.begin(), removal_capture_.pid_to_pos.end(),
        [](const std::pair<uint64_t, int64_t> & a, const std::pair<uint64_t, int64_t> & b) {
            return a.first < b.first;
        });
    removal_capture_.generation++;
    return true;
}

std::function<bool(const std::vector<std::pair<uint32_t, uint32_t>> &, std::string *)>
llama_xkv_runtime::bind_removal_hook(const llama_kv_cache & kv) {
    return [this, &kv](const std::vector<std::pair<uint32_t, uint32_t>> & doomed,
                       std::string * err) -> bool {
        (void) doomed;
        return capture_pack_rebuild_ctx(kv, err);
    };
}

xkv_landmark_rebuild_fn llama_xkv_runtime::bound_removal_callback() const {
    const ggml_type lm_type = cparams_.xkv_landmark_type;
    const uint32_t chunk = eff_.chunk_tokens;
    return [this, lm_type, chunk](
        const xkv_segment & source,
        const std::vector<uint32_t> & surviving_rows,
        xkv_segment & candidate,
        std::string * err) -> bool {
        const uint64_t gen = removal_capture_.generation;
        if (gen == 0) {
            if (err) *err = "bound removal: no validated capture (hook must run first)";
            return false;
        }
        std::vector<int64_t> positions;
        positions.reserve(surviving_rows.size());
        for (uint32_t r : surviving_rows) {
            if (r >= source.row_payload_ids.size()) {
                if (err) *err = "bound removal: surviving row out of range";
                return false;
            }
            const uint64_t pid = source.row_payload_ids[r];
            auto it = removal_capture_.find(pid);
            if (it == removal_capture_.pid_to_pos.end()) {
                if (err) *err = "bound removal: surviving payload lacks a captured position";
                return false;
            }
            positions.push_back(it->second);
        }
        if (gen != removal_capture_.generation) {
            if (err) *err = "bound removal: capture superseded mid-flight";
            return false;
        }
        return rebuild_candidate_landmarks_with_positions(source, surviving_rows, positions,
            cached_phase_specs_, candidate, lm_type, chunk, err);
    };
}

ggml_xkv_residency xkv_expected_residency(const llama_cparams & cparams) {
    if (llama_xkv_profile_is_device_owned(cparams.xkv_storage_profile, cparams.xkv_factorizer)) {
        return GGML_XKV_RES_DEVICE_OWNED;
    }
    return GGML_XKV_RES_REFERENCE_HOST;
}

bool report_code_stream_residency(
    const xkv_segment & segment,
    const llama_cparams & cparams,
    xkv_code_stream_residency & out,
    std::string * err) {
    out = xkv_code_stream_residency{};
    out.segment_id = segment.segment_id;
    out.segment_version = segment.segment_version;
    out.residency = segment.residency;
    const char * backend = "host";
    switch (cparams.xkv_factorizer) {
        case LLAMA_XKV_FACTORIZER_CPU_REFERENCE: backend = "host"; break;
        case LLAMA_XKV_FACTORIZER_VULKAN:        backend = "vulkan"; break;
        case LLAMA_XKV_FACTORIZER_VULKAN_HYBRID: backend = "vulkan-hybrid"; break;
        case LLAMA_XKV_FACTORIZER_CUDA:          backend = "cuda"; break;
        default: break;
    }
    std::strncpy(out.backend, backend, sizeof(out.backend) - 1);
    if (segment.residency == GGML_XKV_RES_DEVICE_OWNED) {
        // Device-owned: verify the immutable backend bundle handle.
        // Deduplicate allocation IDs across handles; sum exact device bytes.
        // Host bytes remain 0 truthfully (released post-verify).
        if (!segment.backend_bundle) {
            if (err) *err = "report_code_stream_residency: DEVICE_OWNED segment missing backend bundle";
            return false;
        }
        const auto & bundle = *segment.backend_bundle;
        std::unordered_set<uint64_t> seen_allocs;
        for (const auto & h : bundle.handles) {
            if (!h) continue;
            if (h->get_residency() != GGML_XKV_RES_DEVICE_OWNED) {
                if (err) *err = "report_code_stream_residency: handle residency mismatch in bundle";
                return false;
            }
            if (seen_allocs.insert(h->get_allocation_id()).second) {
                out.device_bytes += h->get_actual_bytes();
            }
        }
        out.host_bytes = 0;
        out.verified = bundle.is_success() && bundle.is_committed();
        return true;
    }
    // REFERENCE_HOST: exact host bytes from descriptors/retained vectors.
    out.device_bytes = 0;
    out.verified = true; // host memory needs no device verification
    for (const auto & g : segment.groups) {
        out.host_bytes += g.bytes_a_k + g.bytes_b_k + g.bytes_a_v + g.bytes_b_v + g.bytes_landmark;
    }
    return true;
}

} // namespace llama_xkv
