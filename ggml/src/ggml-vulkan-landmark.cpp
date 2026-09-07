// ggml-vulkan-landmark.cpp — CPU reference + graph builder + validation for native
// Vulkan phase-aware quantized landmark scoring, deterministic bounded global top-k,
// fragment→row CSR expansion, and deterministic multi-segment merge (v2).
//
// v2 history independence: NO nq*n_frags score matrix. Fragments stream in fixed
// TILE_FRAGS tiles; per-query carry holds at most top_k (idx+score) plus at most
// refine_cap rows. Scratch is O(nq*(2*top_k + refine_cap + TILE + const)).
// The CPU oracle mirrors the device tile order and carry merge exactly (same total
// order: score-desc, global-frag-id-asc), so streaming == one-shot global select.
//
// Capability maxima (header): budgets above max fail BEFORE graph build
// (supports/oracle false, status ERR_BUDGET). No silent truncation anywhere.
//
// Owned by XkvVulkanLandmark. Fully registered in GGML core, RPC wire protocol,
// and CPU/Vulkan backends. High-level graph files (src/llama-xkv-graph-ref.*)
// intentionally untouched.

#include "ggml-vulkan-landmark.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

static void xvk_err(char * err, size_t n, const char * msg) {
    if (err && n) snprintf(err, n, "%s", msg);
}

static bool xvk_codec_supported(uint32_t t) {
    return t == (uint32_t)GGML_TYPE_Q8_0 || t == (uint32_t)GGML_TYPE_TURBO4_0;
}

static bool xvk_finite_float(float v) { return v == v && v != std::numeric_limits<float>::infinity() && v != -std::numeric_limits<float>::infinity(); }

// ---- shared param/cap validation (v2) ----
static bool xvk_check_caps(const ggml_xkv_landmark_params * params, char * err, size_t err_size) {
    if (!params) { xvk_err(err, err_size, "xkv landmark: null params"); return false; }
    if (params->version != GGML_XKV_LANDMARK_VERSION &&
        params->version != GGML_XKV_LANDMARK_VERSION_MIN) {
        xvk_err(err, err_size, "xkv landmark: bad version"); return false;
    }
    if (params->n_queries == 0 || params->n_queries > GGML_XKV_LANDMARK_MAX_QUERIES) {
        xvk_err(err, err_size, "xkv landmark: n_queries out of capability range"); return false;
    }
    if (params->n_frags == 0) { xvk_err(err, err_size, "xkv landmark: empty frags"); return false; }
    // n_frags itself is unbounded (streaming); only residency per tile is bounded.
    if (params->head_dim == 0 || params->head_dim > GGML_XKV_LANDMARK_MAX_HEAD_DIM) {
        xvk_err(err, err_size, "xkv landmark: head_dim above maximum (no truncation)"); return false;
    }
    if (params->padded_dim < params->head_dim || params->padded_dim > GGML_XKV_LANDMARK_MAX_PADDED_DIM) {
        xvk_err(err, err_size, "xkv landmark: padded_dim out of range"); return false;
    }
    if (params->rotary_dim > params->head_dim || params->rotary_dim > GGML_XKV_LANDMARK_MAX_ROTARY_DIM ||
        (params->rotary_dim & 1u)) {
        xvk_err(err, err_size, "xkv landmark: bad rotary_dim"); return false;
    }
    if (params->rope_mode != GGML_XKV_LANDMARK_ROPE_HALF &&
        params->rope_mode != GGML_XKV_LANDMARK_ROPE_INTERLEAVED) {
        xvk_err(err, err_size, "xkv landmark: bad rope_mode"); return false;
    }
    if (!xvk_codec_supported(params->landmark_type)) {
        xvk_err(err, err_size, "xkv landmark: unsupported codec"); return false;
    }
    if (params->top_k == 0 || params->top_k > GGML_XKV_LANDMARK_MAX_TOP_K) {
        xvk_err(err, err_size, "xkv landmark: top_k above maximum (never select-all)"); return false;
    }
    if (params->max_top_k == 0 || params->max_top_k > GGML_XKV_LANDMARK_MAX_TOP_K) {
        xvk_err(err, err_size, "xkv landmark: max_top_k above maximum"); return false;
    }
    if (params->top_k > params->max_top_k) {
        xvk_err(err, err_size, "xkv landmark: top_k exceeds max_top_k"); return false;
    }
    if (params->top_k > params->n_frags) {
        xvk_err(err, err_size, "xkv landmark: top_k exceeds n_frags"); return false;
    }
    if (params->refine_cap > GGML_XKV_LANDMARK_MAX_REFINE_CAP) {
        xvk_err(err, err_size, "xkv landmark: refine_cap above maximum"); return false;
    }
    if (params->n_q_heads == 0 || params->n_q_heads > GGML_XKV_LANDMARK_MAX_Q_HEADS) {
        xvk_err(err, err_size, "xkv landmark: n_q_heads out of range"); return false;
    }
    if (!xvk_finite_float(params->scale) || params->scale == 0.0f) {
        xvk_err(err, err_size, "xkv landmark: bad scale"); return false;
    }
    const uint32_t known = GGML_XKV_LANDMARK_FLAG_SPARSE_ROWS | GGML_XKV_LANDMARK_FLAG_PERQ_LEGAL
        | GGML_XKV_LANDMARK_FLAG_CANONICAL_XPHASE;
    if (params->_reserved & ~known) { xvk_err(err, err_size, "xkv landmark: unknown flags"); return false; }
    return true;
}

// Phase fingerprint: FNV-1a over rope table bits + domain scalars. Both the
// landmark producer (build time) and the canonical-mode consumer compare these;
// mismatch fails closed. NULL tables allowed iff fc==0.
uint64_t ggml_xkv_landmark_phase_fingerprint(const float * omega_mag_2fc, uint32_t fc,
        uint32_t rope_mode, uint32_t rotary_dim, uint32_t head_dim,
        uint32_t landmark_type) {
    uint64_t h = 14695981039346656037ULL;
    auto mix_byte = [&](uint8_t b) { h ^= b; h *= 1099511628211ULL; };
    auto mix_u32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) mix_byte((uint8_t)(v >> (i * 8))); };
    if (omega_mag_2fc) {
        for (uint32_t i = 0; i < fc * 2; ++i) {
            uint32_t u; memcpy(&u, &omega_mag_2fc[i], 4);
            mix_u32(u);
        }
    }
    mix_u32(fc); mix_u32(rope_mode); mix_u32(rotary_dim); mix_u32(head_dim); mix_u32(landmark_type);
    return h == 0 ? 1 : h;
}

// f16 -> f32 helper (bit-exact portable)
static float xvk_f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign << 31; }
        else {
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; --exp; }
            mant &= 0x3ffu;
            f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xffu << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

static const float XVK_C4[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f,
};

static const float XVK_S1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};
static const float XVK_S2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

static bool xvk_decode_row(enum ggml_type t, const void * src_row, float * dst, int64_t n) {
    if (!src_row || !dst || n <= 0) return false;
    if (t == GGML_TYPE_Q8_0) {
        const uint8_t * b = (const uint8_t *)src_row;
        int64_t nblocks = (n + 31) / 32;
        for (int64_t blk = 0; blk < nblocks; ++blk) {
            uint16_t h = (uint16_t)b[0] | ((uint16_t)b[1] << 8);
            float d = xvk_f16_to_f32(h);
            for (int j = 0; j < 32 && blk * 32 + j < n; ++j) {
                int8_t q; memcpy(&q, b + 2 + j, 1);
                dst[blk * 32 + j] = d * (float)q;
            }
            b += 34;
        }
        return true;
    }
    if (t == GGML_TYPE_TURBO4_0) {
        if (n % 128 != 0) return false;
        const uint8_t * b = (const uint8_t *)src_row;
        int64_t ngroups = n / 128;
        std::vector<float> tmp(128);
        for (int64_t g = 0; g < ngroups; ++g) {
            uint16_t hn = (uint16_t)b[0] | ((uint16_t)b[1] << 8);
            float norm = xvk_f16_to_f32(hn);
            for (int j = 0; j < 128; ++j) {
                uint8_t qb = b[4 + j / 2];
                uint8_t idx = (j & 1) ? (qb >> 4) & 0xfu : qb & 0xfu;
                tmp[j] = norm * XVK_C4[idx];
            }
            for (int i = 0; i < 128; ++i) tmp[i] *= XVK_S2[i];
            for (int h = 1; h < 128; h *= 2) {
                for (int i = 0; i < 128; i += h * 2) {
                    for (int j = i; j < i + h; ++j) {
                        float a = tmp[j], c = tmp[j + h];
                        tmp[j] = a + c; tmp[j + h] = a - c;
                    }
                }
            }
            const float inv = 0.08838834764831845f;
            for (int i = 0; i < 128; ++i) dst[g * 128 + i] = tmp[i] * inv * XVK_S1[i];
            b += 68;
        }
        return true;
    }
    return false;
}

// NOTE (phase domain): intentionally no landmark re-phase helper. Persisted
// base/partial landmarks are row-wise forward-RoPE-then-mean per the runtime
// contract (encode_fragment_landmark phases each row at its OWN storage
// position, then means), and the host selector scores the decoded landmark
// as-is. Re-phasing at the representative frag position would double-phase
// and diverge host vs device whenever rotary_dim > 0, so the selector must
// not apply RoPE here. frag_positions stays for causal gating only.

// Canonical experimental single-position phase (HOST ORACLE ONLY, exact HALF /
// INTERLEAVED math). Applied solely under FLAG_CANONICAL_XPHASE with fingerprint
// match and all-single-row fragments; the production phased path never calls this.
static void xvk_rope_apply_single(float * vec, uint32_t rotary_dim, uint32_t mode,
                           int32_t pos, const float * omega, const float * mag) {
    if (rotary_dim == 0) return;
    uint32_t fc = rotary_dim / 2;
    if (mode == GGML_XKV_LANDMARK_ROPE_HALF) {
        for (uint32_t f = 0; f < fc; ++f) {
            float ang = (float)pos * omega[f];
            float mg = mag ? mag[f] : 1.0f;
            float c = cosf(ang), s = sinf(ang);
            float re = vec[f], im = vec[f + fc];
            vec[f] = (re * c - im * s) * mg;
            vec[f + fc] = (re * s + im * c) * mg;
        }
    } else {
        for (uint32_t f = 0; f < fc; ++f) {
            float ang = (float)pos * omega[f];
            float mg = mag ? mag[f] : 1.0f;
            float c = cosf(ang), s = sinf(ang);
            float re = vec[2 * f], im = vec[2 * f + 1];
            vec[2 * f] = (re * c - im * s) * mg;
            vec[2 * f + 1] = (re * s + im * c) * mg;
        }
    }
}

// Stride-8 (v3) resolvers: words [6]=fpos, [7]=gen ride in the meta; the array
// pointers must be NULL (ambiguity rejected by the wrappers).
static int32_t xvk_fpos_of(uint32_t f, const int32_t * meta, int32_t fstride,
        const int32_t * fpos_array) {
    if (fstride == GGML_XKV_LANDMARK_FRAG_META_STRIDE_V3) return meta[(size_t)f * 8 + 6];
    return fpos_array[f];
}
static int32_t xvk_gen_of(uint32_t f, const int32_t * meta, int32_t fstride,
        const int32_t * gen_array) {
    if (fstride == GGML_XKV_LANDMARK_FRAG_META_STRIDE_V3) return meta[(size_t)f * 8 + 7];
    return gen_array ? gen_array[f] : 0;
}

// ---- per-(query,frag) legality: global flag + causal + optional generation gate ----
static bool xvk_legal(uint32_t f, int32_t qq,
        const int32_t * frag_meta, int32_t fstride,
        const int32_t * frag_positions,
        const int32_t * query_meta, int32_t qstride,
        const int32_t * frag_gen, uint32_t flags) {
    int32_t fl = frag_meta[(size_t)f * fstride + 3];
    if (!(fl & 1)) return false;
    // v3 (stride-8): fpos/gen ride in meta words [6]/[7]; arrays must be NULL
    // (wrappers reject otherwise). v2 and below use the arrays.
    int32_t spos = xvk_fpos_of(f, frag_meta, fstride, frag_positions);
    int32_t causal = query_meta[(size_t)qq * qstride + 0];
    if (causal >= 0 && (int64_t)spos > (int64_t)causal) return false;
    if (flags & GGML_XKV_LANDMARK_FLAG_PERQ_LEGAL) {
        if (fstride != GGML_XKV_LANDMARK_FRAG_META_STRIDE_V3 && !frag_gen) return false;
        int32_t qepoch = query_meta[(size_t)qq * qstride + 4];
        if (xvk_gen_of(f, frag_meta, fstride, frag_gen) != qepoch) return false;
    }
    return true;
}

// Score one legal fragment (decode + GQA-max dot, no re-phase: landmarks are
// pre-phased persisted means). q_heads points at this
// query's q block [n_q_heads * head_dim]. Returns false on non-finite score.
static bool xvk_score_one(const float * q_heads,
        const void * row_src, enum ggml_type lt, uint32_t padded_dim,
        float * lvec, float * lph, uint32_t head_dim,
        uint32_t rotary_dim, uint32_t rope_mode, int32_t spos,
        const float * omega, const float * mag, uint32_t flags,
        uint32_t n_q_heads, float scale, float * out) {
    if (!xvk_decode_row(lt, row_src, lvec, padded_dim)) return false;
    memcpy(lph, lvec, (size_t)head_dim * sizeof(float));
    // Phase domain: production phased path scores the persisted landmark directly
    // (it is ALREADY row-wise forward-RoPE-then-mean). Canonical experimental
    // path (FLAG_CANONICAL_XPHASE) phases the decoded canonical landmark at this
    // single position (caller must guarantee single-row fragment + fp match).
    if (flags & GGML_XKV_LANDMARK_FLAG_CANONICAL_XPHASE) {
        xvk_rope_apply_single(lph, rotary_dim, rope_mode, spos, omega, mag);
    } else {
        (void)rotary_dim; (void)rope_mode; (void)spos; (void)omega; (void)mag;
    }
    float best = -std::numeric_limits<float>::infinity();
    for (uint32_t h = 0; h < n_q_heads; ++h) {
        const float * qp = q_heads + (size_t)h * head_dim;
        double acc = 0.0;
        for (uint32_t d = 0; d < head_dim; ++d) acc += (double)qp[d] * (double)lph[d];
        float s = (float)(acc * (double)scale);
        if (!(s == s)) return false;
        if (s > best) best = s;
    }
    *out = best;
    return true;
}

// Insert (score,frag) into a sorted carry of capacity top_k (desc/asc total order).
// carry_idx/carry_sc have top_k entries, ncarry = current count. Returns new count.
static uint32_t xvk_carry_insert(uint32_t * carry_idx, float * carry_sc, uint32_t ncarry,
        uint32_t top_k, float s, uint32_t f) {
    uint32_t pos = ncarry;
    for (uint32_t i = 0; i < ncarry; ++i) {
        float es = carry_sc[i];
        uint32_t ef = carry_idx[i];
        if (s > es || (s == es && f < ef)) { pos = i; break; }
    }
    if (pos >= top_k) return ncarry;
    uint32_t up = ncarry < top_k ? ncarry : top_k - 1;
    for (uint32_t i = up; i > pos; --i) {
        carry_idx[i] = carry_idx[i - 1];
        carry_sc[i] = carry_sc[i - 1];
    }
    carry_idx[pos] = f;
    carry_sc[pos] = s;
    return ncarry < top_k ? ncarry + 1 : top_k;
}

// ---- workspace: history-independent (no n_frags term) ----
bool ggml_xkv_landmark_workspace_bytes(const ggml_xkv_landmark_params * params,
        size_t * out_bytes, char * err, size_t err_size) {
    if (!params || !out_bytes) { xvk_err(err, err_size, "xkv landmark: null pointer"); return false; }
    if (!xvk_check_caps(params, err, err_size)) return false;
    uint64_t nq = params->n_queries, tk = params->top_k, rc = params->refine_cap;
    const uint64_t TILE = GGML_XKV_LANDMARK_TILE_FRAGS;
    __uint128_t words = (__uint128_t)nq * tk      // CARRY_IDX
        + (__uint128_t)nq * tk                     // CARRY_SC
        + (__uint128_t)nq                          // TAKE
        + (__uint128_t)nq                          // REF
        + (__uint128_t)nq                          // LEGAL
        + (__uint128_t)nq * rc                     // ROWBUF (0 when cap==0)
        + (__uint128_t)nq * TILE                   // TILE staging
        + 64;                                      // margin
    if (words > (size_t)-1 / sizeof(float)) { xvk_err(err, err_size, "xkv landmark: scratch overflow"); return false; }
    *out_bytes = (size_t)words * sizeof(float);
    return true;
}

bool ggml_xkv_landmark_scratch_map(const ggml_xkv_landmark_params * params,
        size_t * carry_idx_words, size_t * carry_sc_words,
        size_t * take_words, size_t * ref_words, size_t * legal_words,
        size_t * rowbuf_words, size_t * tile_words,
        char * err, size_t err_size) {
    if (!params) { xvk_err(err, err_size, "xkv landmark: null params"); return false; }
    if (!xvk_check_caps(params, err, err_size)) return false;
    uint64_t nq = params->n_queries, tk = params->top_k, rc = params->refine_cap;
    const uint64_t TILE = GGML_XKV_LANDMARK_TILE_FRAGS;
    size_t ci = 0, cs = (size_t)(nq * tk), tkoff = cs + (size_t)(nq * tk);
    size_t refoff = tkoff + (size_t)nq, legaloff = refoff + (size_t)nq;
    size_t rowoff = legaloff + (size_t)nq, tileoff = rowoff + (size_t)(nq * rc);
    if (carry_idx_words) *carry_idx_words = ci;
    if (carry_sc_words) *carry_sc_words = cs;
    if (take_words) *take_words = tkoff;
    if (ref_words) *ref_words = refoff;
    if (legal_words) *legal_words = legaloff;
    if (rowbuf_words) *rowbuf_words = rowoff;
    if (tile_words) *tile_words = tileoff;
    (void)TILE;
    return true;
}

bool ggml_xkv_landmark_supports(
        const struct ggml_tensor * q,
        const struct ggml_tensor * landmarks,
        const struct ggml_tensor * frag_positions,
        const struct ggml_tensor * frag_meta,
        const struct ggml_tensor * query_meta,
        const struct ggml_tensor * rope_tables,
        const struct ggml_tensor * scratch,
        const struct ggml_tensor * csr_ptrs,
        const struct ggml_tensor * topk_scores,
        const struct ggml_tensor * status,
        const struct ggml_tensor * dst_csr_indices,
        const ggml_xkv_landmark_params * params,
        char * err, size_t err_size) {
    if (!q || !landmarks || !frag_positions || !frag_meta || !query_meta || !rope_tables ||
        !scratch || !csr_ptrs || !topk_scores || !status || !dst_csr_indices || !params) {
        xvk_err(err, err_size, "xkv landmark: null tensor/params"); return false;
    }
    // Caps first: above-maximum budgets rejected BEFORE graph build (never truncate).
    if (!xvk_check_caps(params, err, err_size)) return false;
    if ((enum ggml_type)landmarks->type != (enum ggml_type)params->landmark_type) {
        xvk_err(err, err_size, "xkv landmark: tensor/type mismatch"); return false;
    }
    if (q->type != GGML_TYPE_F32) { xvk_err(err, err_size, "xkv landmark: q must be F32"); return false; }
    if (frag_positions->type != GGML_TYPE_I32 || frag_meta->type != GGML_TYPE_I32 ||
        query_meta->type != GGML_TYPE_I32 || csr_ptrs->type != GGML_TYPE_I32 ||
        dst_csr_indices->type != GGML_TYPE_I32 || status->type != GGML_TYPE_I32) {
        xvk_err(err, err_size, "xkv landmark: meta/csr/status must be I32"); return false;
    }
    if (topk_scores->type != GGML_TYPE_F32) { xvk_err(err, err_size, "xkv landmark: scores must be F32"); return false; }
    if (q->ne[0] != (int64_t)params->head_dim) { xvk_err(err, err_size, "xkv landmark: q dim mismatch"); return false; }
    if (q->ne[1] != (int64_t)params->n_q_heads * params->n_queries) {
        xvk_err(err, err_size, "xkv landmark: q heads mismatch"); return false;
    }
    if (landmarks->ne[0] != (int64_t)params->padded_dim || landmarks->ne[1] != (int64_t)params->n_frags) {
        xvk_err(err, err_size, "xkv landmark: landmarks shape mismatch"); return false;
    }
    if (frag_positions->ne[0] != (int64_t)params->n_frags) {
        if (frag_meta->ne[0] != GGML_XKV_LANDMARK_FRAG_META_STRIDE_V3) {
            xvk_err(err, err_size, "xkv landmark: frag_positions len"); return false;
        }
    }
    if ((frag_meta->ne[0] != GGML_XKV_LANDMARK_FRAG_META_STRIDE &&
         frag_meta->ne[0] != GGML_XKV_LANDMARK_FRAG_META_STRIDE_X &&
         frag_meta->ne[0] != GGML_XKV_LANDMARK_FRAG_META_STRIDE_V3) ||
        frag_meta->ne[1] != (int64_t)params->n_frags) {
        xvk_err(err, err_size, "xkv landmark: frag_meta stride must be 4, 6, or 8"); return false;
    }
    if (frag_meta->ne[0] == GGML_XKV_LANDMARK_FRAG_META_STRIDE_V3) {
        // v3: fpos/gen ride in meta words [6]/[7]; src2 carries the per-query
        // eligibility bitset I32[ceil(n_frags/32), n_queries], not positions.
        const uint64_t words = ((uint64_t)params->n_frags + 31u) / 32u;
        if (frag_positions->type != GGML_TYPE_I32 ||
            (uint64_t)ggml_nelements(frag_positions) != words * params->n_queries) {
            xvk_err(err, err_size, "xkv landmark v3: src2 must be elig bitset [ceil(nf/32),nq]");
            return false;
        }
    }
    if ((query_meta->ne[0] != GGML_XKV_LANDMARK_QUERY_META_STRIDE &&
         query_meta->ne[0] != GGML_XKV_LANDMARK_QUERY_META_STRIDE_X) ||
        query_meta->ne[1] != (int64_t)params->n_queries) {
        xvk_err(err, err_size, "xkv landmark: query_meta stride must be 4 or 6"); return false;
    }
    if ((params->_reserved & GGML_XKV_LANDMARK_FLAG_PERQ_LEGAL) && query_meta->ne[0] != 6) {
        xvk_err(err, err_size, "xkv landmark: PERQ_LEGAL needs stride-6 query_meta"); return false;
    }
    {
        uint32_t fc = params->rotary_dim / 2;
        int64_t want = fc == 0 ? 0 : (int64_t)fc * 2;
        if (rope_tables->type != GGML_TYPE_F32 || ggml_nelements(rope_tables) != want) {
            xvk_err(err, err_size, "xkv landmark: rope_tables must be F32[2*Fc]"); return false;
        }
    }
    if (csr_ptrs->ne[0] != (int64_t)params->n_queries + 1) {
        xvk_err(err, err_size, "xkv landmark: csr_ptrs len"); return false;
    }
    if (dst_csr_indices->ne[0] != (int64_t)params->top_k || dst_csr_indices->ne[1] != (int64_t)params->n_queries) {
        xvk_err(err, err_size, "xkv landmark: csr_indices shape"); return false;
    }
    if (topk_scores->ne[0] != (int64_t)params->top_k || topk_scores->ne[1] != (int64_t)params->n_queries) {
        xvk_err(err, err_size, "xkv landmark: topk_scores shape"); return false;
    }
    if (status->ne[0] != 4) { xvk_err(err, err_size, "xkv landmark: status must be I32[4]"); return false; }
    {
        size_t need = 0;
        if (!ggml_xkv_landmark_workspace_bytes(params, &need, err, err_size)) return false;
        if ((size_t)ggml_nbytes(scratch) < need) { xvk_err(err, err_size, "xkv landmark: scratch too small"); return false; }
    }
    if (landmarks->nb[0] != ggml_type_size(landmarks->type) ||
        landmarks->nb[1] < ggml_row_size(landmarks->type, landmarks->ne[0])) {
        xvk_err(err, err_size, "xkv landmark: landmarks nb[1] must be >= row_size(head_dim)"); return false;
    }
    // Feature slice validation: verify row stride nb1 is a multiple of quant block size
    if (landmarks->nb[1] % ggml_type_size(landmarks->type) != 0) {
        xvk_err(err, err_size, "xkv landmark: landmarks nb[1] must be element aligned"); return false;
    }
    // Strided full-group views: the byte stride rides the Vulkan push as u32.
    if ((uint64_t)landmarks->nb[1] > (uint64_t)UINT32_MAX) {
        xvk_err(err, err_size, "xkv landmark: landmarks stride overflow"); return false;
    }
    if (params->refine_cap > 0 && params->n_rows_total == 0) {
        xvk_err(err, err_size, "xkv landmark: refine_cap requires n_rows_total"); return false;
    }
    // Sparse fragments + select-side refinement: select assumes contiguous rows
    // [row_begin, row_begin+row_count), so select-side refinement with sparse
    // fragments is invalid and must fail closed. Pass refine_cap=0 on select;
    // the rows op enforces the row budget once via explicit lists.
    if ((params->_reserved & GGML_XKV_LANDMARK_FLAG_SPARSE_ROWS) && params->refine_cap > 0) {
        xvk_err(err, err_size, "xkv landmark: select-side refinement unsupported with SPARSE_ROWS (use rows op)");
        return false;
    }
    // Canonical experimental mode (FLAG_CANONICAL_XPHASE) is host-oracle-only;
    // device graph build must reject it explicitly.
    if (params->_reserved & GGML_XKV_LANDMARK_FLAG_CANONICAL_XPHASE) {
        xvk_err(err, err_size, "xkv landmark: CANONICAL_XPHASE unsupported on device"); return false;
    }
    return true;
}

struct ggml_tensor * ggml_xkv_landmark(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * landmarks,
        struct ggml_tensor  * frag_positions,
        struct ggml_tensor  * frag_meta,
        struct ggml_tensor  * query_meta,
        struct ggml_tensor  * rope_tables,
        struct ggml_tensor  * scratch,
        struct ggml_tensor  * csr_ptrs,
        struct ggml_tensor  * topk_scores,
        struct ggml_tensor  * status,
        const ggml_xkv_landmark_params * params) {
    if (!ctx || !q || !landmarks || !frag_positions || !frag_meta || !query_meta ||
        !rope_tables || !scratch || !csr_ptrs || !topk_scores || !status || !params) {
        return nullptr;
    }
    struct ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, params->top_k, params->n_queries);
    if (!dst) return nullptr;
    memcpy(dst->op_params, params, sizeof(*params));
    dst->op = (enum ggml_op)(GGML_OP_XKV_LANDMARK);
    dst->src[0] = q; dst->src[1] = landmarks; dst->src[2] = frag_positions;
    dst->src[3] = frag_meta; dst->src[4] = query_meta; dst->src[5] = rope_tables;
    dst->src[6] = scratch; dst->src[7] = csr_ptrs; dst->src[8] = topk_scores;
    dst->src[9] = status;
    return dst;
}

// ---- streaming core shared by the legacy-signature oracle and the extended oracle ----
static bool xvk_select_stream(
        const float * q_data, const void * land_data, enum ggml_type lt,
        const int32_t * frag_positions, const int32_t * frag_meta, int32_t fstride,
        const int32_t * query_meta, int32_t qstride, const int32_t * frag_gen,
        const float * rope_tables, const ggml_xkv_landmark_params * params,
        uint64_t expected_phase_fp,
        int32_t * out_csr_ptrs, int32_t * out_csr_indices, float * out_topk_scores,
        int32_t * out_status, char * err, size_t err_size) {
    const bool canonical_phase = (params->_reserved & GGML_XKV_LANDMARK_FLAG_CANONICAL_XPHASE) != 0;
    if (canonical_phase) {
        // Canonical mode constraints: every fragment MUST be single-row (mathematically
        // undefined to single-position phase a multi-row chunk), and the rope table
        // fingerprint MUST match the expected phase fingerprint.
        for (uint32_t f = 0; f < params->n_frags; ++f) {
            int32_t rc = frag_meta[(size_t)f * fstride + 1];
            if (rc != 1) {
                out_status[0] = GGML_XKV_LANDMARK_STATUS_ERR_INPUT;
                xvk_err(err, err_size, "xkv landmark: canonical mode requires row_count==1");
                return false;
            }
        }
        uint32_t fc = params->rotary_dim / 2;
        uint64_t actual_fp = ggml_xkv_landmark_phase_fingerprint(
            rope_tables, fc, params->rope_mode, params->rotary_dim,
            params->head_dim, params->landmark_type);
        if (actual_fp != expected_phase_fp) {
            out_status[0] = GGML_XKV_LANDMARK_STATUS_ERR_INPUT;
            xvk_err(err, err_size, "xkv landmark: phase fingerprint mismatch");
            return false;
        }
    }
    uint32_t fc = params->rotary_dim / 2;
    const float * omega = fc > 0 ? rope_tables : nullptr;
    const float * mag = fc > 0 ? rope_tables + fc : nullptr;
    if (fc > 0) {
        if (!rope_tables) { xvk_err(err, err_size, "xkv landmark oracle: rope missing"); return false; }
        for (uint32_t f = 0; f < fc; ++f) {
            if (!(omega[f] == omega[f]) || !(mag[f] == mag[f]) || !(mag[f] > 0.0f)) {
                xvk_err(err, err_size, "xkv landmark oracle: rope non-finite/bad mag"); return false;
            }
        }
    }
    size_t row_bytes = ggml_row_size(lt, params->padded_dim);
    std::vector<float> lvec(params->padded_dim), lph(params->head_dim);
    std::vector<uint32_t> carry_idx(params->top_k);
    std::vector<float> carry_sc(params->top_k);
    const uint32_t TILE = GGML_XKV_LANDMARK_TILE_FRAGS;

    out_csr_ptrs[0] = 0;
    int32_t total_refined = 0, cap_hits = 0, total_legal = 0;
    for (uint32_t qq = 0; qq < params->n_queries; ++qq) {
        const float * qh = q_data + (size_t)qq * params->n_q_heads * params->head_dim;
        uint32_t ncarry = 0, legal = 0;
        // Tiled streaming: fixed TILE_FRAGS residency, carry merge per tile.
        // Comparison is a total order (score desc, global id asc), so tile-chained
        // insertion == one-shot global top-k over the union (segment order free).
        for (uint32_t base = 0; base < params->n_frags; base += TILE) {
            uint32_t lim = base + TILE < params->n_frags ? base + TILE : params->n_frags;
            for (uint32_t f = base; f < lim; ++f) {
                if (!xvk_legal(f, (int32_t)qq, frag_meta, fstride, frag_positions,
                               query_meta, qstride, frag_gen, params->_reserved)) {
                    continue;
                }
                legal++;
                const void * row_src = (const char *)land_data + (size_t)f * row_bytes;
                float s = 0.0f;
                if (!xvk_score_one(qh, row_src, lt, params->padded_dim,
                                   lvec.data(), lph.data(), params->head_dim,
                                   params->rotary_dim, params->rope_mode,
                                   xvk_fpos_of(f, frag_meta, fstride, frag_positions),
                                   omega, mag, params->_reserved,
                                   params->n_q_heads, params->scale, &s)) {
                    xvk_err(err, err_size, "xkv landmark oracle: dequant/non-finite score"); return false;
                }
                ncarry = xvk_carry_insert(carry_idx.data(), carry_sc.data(), ncarry,
                                          params->top_k, s, f);
            }
        }
        total_legal += (int32_t)legal;
        uint32_t take = ncarry; // ncarry <= top_k by construction
        for (uint32_t i = 0; i < params->top_k; ++i) {
            if (i < take) {
                out_csr_indices[(size_t)qq * params->top_k + i] = (int32_t)carry_idx[i];
                out_topk_scores[(size_t)qq * params->top_k + i] = carry_sc[i];
            } else {
                out_csr_indices[(size_t)qq * params->top_k + i] = -1;
                out_topk_scores[(size_t)qq * params->top_k + i] = -1e30f;
            }
        }
        out_csr_ptrs[qq + 1] = out_csr_ptrs[qq] + (int32_t)take;
        uint32_t refined = 0;
        bool hit = false;
        if (params->refine_cap > 0 && take > 0) {
            std::vector<uint32_t> sel(carry_idx.begin(), carry_idx.begin() + take);
            std::sort(sel.begin(), sel.end());
            std::vector<uint32_t> rows;
            rows.reserve((size_t)take * params->frag_size);
            for (uint32_t s = 0; s < take; ++s) {
                uint32_t fidx = sel[s];
                int32_t rb = frag_meta[(size_t)fidx * fstride + 0];
                int32_t rc = frag_meta[(size_t)fidx * fstride + 1];
                if (rc <= 0) rc = (int32_t)params->frag_size;
                for (int32_t r = 0; r < rc; ++r) {
                    int64_t row = (int64_t)rb + r;
                    if (row < 0 || row >= (int64_t)params->n_rows_total) continue;
                    if (rows.size() >= params->refine_cap) break;
                    rows.push_back((uint32_t)row);
                }
                if (rows.size() >= params->refine_cap) break;
            }
            if (rows.size() >= params->refine_cap) hit = true;
            std::sort(rows.begin(), rows.end());
            rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
            if (rows.size() > params->refine_cap) { rows.resize(params->refine_cap); hit = true; }
            refined = (uint32_t)rows.size();
        }
        total_refined += (int32_t)refined;
        if (hit) cap_hits++;
    }
    out_status[0] = GGML_XKV_LANDMARK_STATUS_OK;
    out_status[1] = total_refined;
    out_status[2] = cap_hits;
    out_status[3] = total_legal;
    return true;
}

// Legacy-signature oracle (stride-4 meta, no generation gate). Behavior is v2
// streaming; status[3] now carries total legal fragments (was 0).
bool ggml_xkv_landmark_cpu_oracle(
        const float   * q_data,
        const void    * landmarks_data, enum ggml_type landmark_type,
        const int32_t * frag_positions,
        const int32_t * frag_meta,
        const int32_t * query_meta,
        const float   * rope_tables,
        const ggml_xkv_landmark_params * params,
        int32_t       * out_csr_ptrs,
        int32_t       * out_csr_indices,
        float         * out_topk_scores,
        int32_t       * out_status,
        char * err, size_t err_size) {
    if (!q_data || !landmarks_data || !frag_positions || !frag_meta || !query_meta ||
        !params || !out_csr_ptrs || !out_csr_indices || !out_topk_scores || !out_status) {
        xvk_err(err, err_size, "xkv landmark oracle: null pointer"); return false;
    }
    if (!xvk_check_caps(params, err, err_size)) {
        out_status[0] = GGML_XKV_LANDMARK_STATUS_ERR_BUDGET;
        return false;
    }
    if (!xvk_codec_supported(params->landmark_type) || landmark_type != (enum ggml_type)params->landmark_type) {
        out_status[0] = GGML_XKV_LANDMARK_STATUS_ERR_UNSUPPORTED;
        xvk_err(err, err_size, "xkv landmark oracle: unsupported codec"); return false;
    }
    if (params->_reserved & GGML_XKV_LANDMARK_FLAG_PERQ_LEGAL) {
        out_status[0] = GGML_XKV_LANDMARK_STATUS_ERR_INPUT;
        xvk_err(err, err_size, "xkv landmark oracle: PERQ_LEGAL needs _x entry"); return false;
    }
    if (params->_reserved & GGML_XKV_LANDMARK_FLAG_CANONICAL_XPHASE) {
        out_status[0] = GGML_XKV_LANDMARK_STATUS_ERR_INPUT;
        xvk_err(err, err_size, "xkv landmark oracle: CANONICAL_XPHASE needs _x entry with fingerprint"); return false;
    }
    return xvk_select_stream(q_data, landmarks_data, landmark_type, frag_positions,
        frag_meta, GGML_XKV_LANDMARK_FRAG_META_STRIDE, query_meta,
        GGML_XKV_LANDMARK_QUERY_META_STRIDE, nullptr, rope_tables, params, 0,
        out_csr_ptrs, out_csr_indices, out_topk_scores, out_status, err, err_size);
}

// Extended oracle: explicit strides + optional generation gate.
bool ggml_xkv_landmark_cpu_oracle_x(
        const float   * q_data,
        const void    * landmarks_data, enum ggml_type landmark_type,
        const int32_t * frag_positions,
        const int32_t * frag_meta, int32_t frag_meta_stride,
        const int32_t * query_meta, int32_t query_meta_stride,
        const int32_t * frag_gen,
        const float   * rope_tables,
        const ggml_xkv_landmark_params * params,
        uint64_t expected_phase_fp,
        int32_t       * out_csr_ptrs,
        int32_t       * out_csr_indices,
        float         * out_topk_scores,
        int32_t       * out_status,
        char * err, size_t err_size) {
    if (!q_data || !landmarks_data || !frag_positions || !frag_meta || !query_meta ||
        !params || !out_csr_ptrs || !out_csr_indices || !out_topk_scores || !out_status) {
        xvk_err(err, err_size, "xkv landmark oracle_x: null pointer"); return false;
    }
    if (frag_meta_stride != GGML_XKV_LANDMARK_FRAG_META_STRIDE &&
        frag_meta_stride != GGML_XKV_LANDMARK_FRAG_META_STRIDE_X) {
        xvk_err(err, err_size, "xkv landmark oracle_x: bad frag stride"); return false;
    }
    if (query_meta_stride != GGML_XKV_LANDMARK_QUERY_META_STRIDE &&
        query_meta_stride != GGML_XKV_LANDMARK_QUERY_META_STRIDE_X) {
        xvk_err(err, err_size, "xkv landmark oracle_x: bad query stride"); return false;
    }
    if (!xvk_check_caps(params, err, err_size)) {
        out_status[0] = GGML_XKV_LANDMARK_STATUS_ERR_BUDGET;
        return false;
    }
    if (!xvk_codec_supported(params->landmark_type) || landmark_type != (enum ggml_type)params->landmark_type) {
        out_status[0] = GGML_XKV_LANDMARK_STATUS_ERR_UNSUPPORTED;
        xvk_err(err, err_size, "xkv landmark oracle_x: unsupported codec"); return false;
    }
    if ((params->_reserved & GGML_XKV_LANDMARK_FLAG_PERQ_LEGAL) &&
        (query_meta_stride != GGML_XKV_LANDMARK_QUERY_META_STRIDE_X || !frag_gen)) {
        xvk_err(err, err_size, "xkv landmark oracle_x: PERQ_LEGAL needs stride-6 meta+gen"); return false;
    }
    return xvk_select_stream(q_data, landmarks_data, landmark_type, frag_positions,
        frag_meta, frag_meta_stride, query_meta, query_meta_stride, frag_gen,
        rope_tables, params, expected_phase_fp, out_csr_ptrs, out_csr_indices, out_topk_scores,
        out_status, err, err_size);
}

// v3 oracle: stride-8 frag_meta carrying fpos+gen. frag_positions and frag_gen
// MUST be NULL (ambiguity rejected).
bool ggml_xkv_landmark_v3_cpu_oracle(
        const float   * q_data,
        const void    * landmarks_data, enum ggml_type landmark_type,
        const int32_t * frag_meta,
        const int32_t * query_meta, int32_t query_meta_stride,
        const float   * rope_tables,
        const ggml_xkv_landmark_params * params,
        uint64_t expected_phase_fp,
        int32_t       * out_csr_ptrs,
        int32_t       * out_csr_indices,
        float         * out_topk_scores,
        int32_t       * out_status,
        char * err, size_t err_size) {
    if (!q_data || !landmarks_data || !frag_meta || !query_meta ||
        !params || !out_csr_ptrs || !out_csr_indices || !out_topk_scores || !out_status) {
        xvk_err(err, err_size, "xkv landmark v3 oracle: null pointer"); return false;
    }
    if (!xvk_check_caps(params, err, err_size)) {
        out_status[0] = GGML_XKV_LANDMARK_STATUS_ERR_BUDGET;
        return false;
    }
    if (!xvk_codec_supported(params->landmark_type) || landmark_type != (enum ggml_type)params->landmark_type) {
        out_status[0] = GGML_XKV_LANDMARK_STATUS_ERR_UNSUPPORTED;
        xvk_err(err, err_size, "xkv landmark v3 oracle: unsupported codec"); return false;
    }
    if (query_meta_stride != GGML_XKV_LANDMARK_QUERY_META_STRIDE &&
        query_meta_stride != GGML_XKV_LANDMARK_QUERY_META_STRIDE_X) {
        xvk_err(err, err_size, "xkv landmark v3 oracle: bad query stride"); return false;
    }
    return xvk_select_stream(q_data, landmarks_data, landmark_type, nullptr,
        frag_meta, GGML_XKV_LANDMARK_FRAG_META_STRIDE_V3, query_meta,
        query_meta_stride, nullptr, rope_tables, params, expected_phase_fp,
        out_csr_ptrs, out_csr_indices, out_topk_scores, out_status, err, err_size);
}

// ---- fragment→row CSR expansion ----
bool ggml_xkv_landmark_rows_workspace_bytes(uint32_t n_queries, uint32_t top_k, uint32_t refine_cap,
        size_t * out_bytes, char * err, size_t err_size) {
    if (!out_bytes) { xvk_err(err, err_size, "xkv rows: null out"); return false; }
    if (n_queries == 0 || n_queries > GGML_XKV_LANDMARK_MAX_QUERIES) {
        xvk_err(err, err_size, "xkv rows: n_queries out of range"); return false;
    }
    if (top_k == 0 || top_k > GGML_XKV_LANDMARK_MAX_TOP_K) {
        xvk_err(err, err_size, "xkv rows: top_k out of range"); return false;
    }
    if (refine_cap == 0 || refine_cap > GGML_XKV_LANDMARK_MAX_REFINE_CAP) {
        xvk_err(err, err_size, "xkv rows: refine_cap must be within (0,MAX]"); return false;
    }
    __uint128_t words = (__uint128_t)n_queries * refine_cap * 5  // refs(4)+pos per row
        + (__uint128_t)n_queries * 2 + 64;
    if (words > (size_t)-1 / sizeof(int32_t)) { xvk_err(err, err_size, "xkv rows: overflow"); return false; }
    *out_bytes = (size_t)words * sizeof(int32_t);
    return true;
}

bool ggml_xkv_landmark_rows_cpu_oracle(
        const int32_t * sel_indices,
        const int32_t * frag_meta,
        const int32_t * frag_row_off, const int32_t * frag_row_ids,
        const int32_t * frag_kv, const int32_t * row_pos,
        const int32_t * frag_positions,
        const ggml_xkv_landmark_rows_params * params,
        int32_t * out_row_ptrs, int32_t * out_row_refs, int32_t * out_row_pos,
        int32_t * out_row_entries,
        int32_t * out_row_status, char * err, size_t err_size) {
    if (!sel_indices || !frag_meta || !frag_kv || !params ||
        !frag_positions || !out_row_ptrs || !out_row_refs || !out_row_pos || !out_row_status) {
        xvk_err(err, err_size, "xkv rows: null pointer"); return false;
    }
    const uint32_t n_queries = params->n_queries;
    const uint32_t top_k = params->top_k;
    const uint32_t n_frags = params->n_frags;
    const uint32_t refine_cap = params->refine_cap;
    const int32_t frag_meta_stride = (int32_t)params->fstride;
    const uint32_t flags = params->flags;
    const uint32_t n_rows_total = params->n_rows_total;
    const uint32_t n_parent_queries = params->n_parent_queries;
    const uint32_t has_query_map = params->has_query_map;
    const uint32_t arena_filter = params->arena_filter;
    const uint32_t global_row_base = params->global_row_base;
    const uint32_t arena_row_count = params->arena_row_count;
    const uint32_t output_row_begin = params->output_row_begin;
    if (params->version != GGML_XKV_LANDMARK_VERSION) {
        xvk_err(err, err_size, "xkv rows: bad version"); return false;
    }
    if (params->_reserved != 0) {
        xvk_err(err, err_size, "xkv rows: _reserved must be zero"); return false;
    }
    if (n_queries == 0 || n_queries > GGML_XKV_LANDMARK_MAX_QUERIES) {
        xvk_err(err, err_size, "xkv rows: n_queries out of range"); return false;
    }
    if (top_k == 0 || top_k > GGML_XKV_LANDMARK_MAX_TOP_K) {
        out_row_status[0] = GGML_XKV_LANDMARK_STATUS_ERR_BUDGET;
        xvk_err(err, err_size, "xkv rows: top_k out of range"); return false;
    }
    if (refine_cap == 0 || refine_cap > GGML_XKV_LANDMARK_MAX_REFINE_CAP) {
        out_row_status[0] = GGML_XKV_LANDMARK_STATUS_ERR_BUDGET;
        xvk_err(err, err_size, "xkv rows: refine_cap out of range"); return false;
    }
    if (n_rows_total == 0) {
        xvk_err(err, err_size, "xkv rows: empty row domain"); return false;
    }
    if (frag_meta_stride != 4 && frag_meta_stride != 6) {
        xvk_err(err, err_size, "xkv rows: frag stride must be 4 or 6"); return false;
    }
    bool sparse = (flags & GGML_XKV_LANDMARK_FLAG_SPARSE_ROWS) != 0;
    if (sparse && (!frag_row_off || !frag_row_ids)) {
        xvk_err(err, err_size, "xkv rows: sparse needs row_off+row_ids"); return false;
    }
    const uint32_t known = GGML_XKV_LANDMARK_FLAG_SPARSE_ROWS | GGML_XKV_LANDMARK_FLAG_PERQ_LEGAL;
    if (flags & ~known) { xvk_err(err, err_size, "xkv rows: unknown flags"); return false; }
    if (has_query_map > 1 || (has_query_map == 0 && n_parent_queries != 0) ||
        (has_query_map != 0 && (n_parent_queries == 0 || n_parent_queries > n_queries ||
                                n_parent_queries > GGML_XKV_LANDMARK_MAX_QUERIES))) {
        xvk_err(err, err_size, "xkv rows: inconsistent parent query map"); return false;
    }
    // Per-arena contract: filter before cap accounting. ALL mode requires base 0
    // with full-domain count; filtered mode requires a non-empty window inside
    // the global row domain. All arithmetic fails closed.
    const bool filter_all = (arena_filter == UINT32_MAX);
    if (filter_all) {
        if (global_row_base != 0 || arena_row_count != n_rows_total) {
            xvk_err(err, err_size, "xkv rows: ALL arena mode needs base 0 and full count"); return false;
        }
    } else {
        if (arena_row_count == 0 || arena_row_count > n_rows_total ||
            global_row_base >= n_rows_total ||
            (uint64_t)global_row_base + arena_row_count > n_rows_total) {
            xvk_err(err, err_size, "xkv rows: arena window outside row domain"); return false;
        }
    }

    const uint32_t n_parents = has_query_map ? n_parent_queries : n_queries;
    const uint32_t sel_stride = top_k + (has_query_map ? 2u : 0u);
    for (uint32_t n = 0; n <= n_parents; ++n) out_row_ptrs[n] = 0;

    // Validate the complete map before publishing any row. Parent-major order
    // makes compact expanded-query output a valid parent CSR without a scatter.
    if (has_query_map) {
        int32_t prev_parent = -1;
        int32_t prev_slot = -1;
        for (uint32_t e = 0; e < n_queries; ++e) {
            const int32_t parent = sel_indices[(size_t)e * sel_stride + 0];
            const int32_t slot = sel_indices[(size_t)e * sel_stride + 1];
            if (parent < 0 || (uint32_t)parent >= n_parent_queries || slot < 0 ||
                parent < prev_parent || (parent == prev_parent && slot <= prev_slot)) {
                xvk_err(err, err_size, "xkv rows: query map must be parent-major with increasing slots");
                return false;
            }
            prev_parent = parent;
            prev_slot = slot;
        }
    }

    struct row_item {
        int32_t row;
        int32_t pos;
        int32_t slot;
        uint32_t owner;
        uint32_t order;
        int32_t parent;
    };

    int32_t hits = 0;
    std::vector<row_item> all;
    try {
        std::vector<uint32_t> sel;
        sel.reserve(top_k);
        std::vector<row_item> items;
        items.reserve((size_t)top_k * GGML_XKV_LANDMARK_MAX_FRAG_ROWS);
        all.reserve((size_t)n_queries * refine_cap);

        for (uint32_t e = 0; e < n_queries; ++e) {
            const int32_t parent = has_query_map ? sel_indices[(size_t)e * sel_stride + 0] : (int32_t)e;
            const int32_t mapped_slot = has_query_map ? sel_indices[(size_t)e * sel_stride + 1] : 0;
            sel.clear();
            for (uint32_t i = 0; i < top_k; ++i) {
                const int32_t f = sel_indices[(size_t)e * sel_stride + (has_query_map ? 2u : 0u) + i];
                if (f < 0) break;
                if ((uint32_t)f >= n_frags) {
                    xvk_err(err, err_size, "xkv rows: frag id OOB"); return false;
                }
                sel.push_back((uint32_t)f);
            }
            std::sort(sel.begin(), sel.end());
            items.clear();
            uint32_t item_order = 0;
            for (uint32_t f : sel) {
                const int32_t slot = has_query_map ? mapped_slot :
                    (frag_meta_stride >= 6 ? frag_meta[(size_t)f * frag_meta_stride + 5] : 0);
                if (slot < 0) {
                    xvk_err(err, err_size, "xkv rows: negative DDVR slot"); return false;
                }
                // Arena filter before cap accounting: frag_kv[1] is the arena ordinal.
                const uint32_t frag_arena = (uint32_t)frag_kv[(size_t)f * 3 + 1];
                if (!filter_all && frag_arena != arena_filter) continue;
                if (!sparse || frag_row_off[f] < 0) {
                    const int32_t rb = frag_meta[(size_t)f * frag_meta_stride + 0];
                    const int32_t rc = frag_meta[(size_t)f * frag_meta_stride + 1];
                    if (rb < 0 || rc <= 0 || rc > (int32_t)GGML_XKV_LANDMARK_MAX_FRAG_ROWS ||
                        (uint64_t)(uint32_t)rb + (uint32_t)rc > n_rows_total) {
                        xvk_err(err, err_size, "xkv rows: contiguous fragment outside row domain"); return false;
                    }
                    for (int32_t r = 0; r < rc; ++r) {
                        const int32_t grow = rb + r;
                        if ((uint32_t)grow < global_row_base ||
                            (uint32_t)grow >= global_row_base + arena_row_count) {
                            xvk_err(err, err_size, "xkv rows: row outside arena window"); return false;
                        }
                        const int32_t lrow = (int32_t)((uint32_t)grow - global_row_base);
                        items.push_back({lrow, row_pos ? row_pos[grow] : frag_positions[f], slot, f, item_order++, parent});
                    }
                } else {
                    const int32_t off = frag_row_off[f];
                    const int32_t nxt = frag_row_off[f + 1];
                    if (off < 0 || nxt < off || nxt - off > (int32_t)GGML_XKV_LANDMARK_MAX_FRAG_ROWS) {
                        xvk_err(err, err_size, "xkv rows: sparse list above maximum"); return false;
                    }
                    for (int32_t k = off; k < nxt; ++k) {
                        const int32_t grow = frag_row_ids[k];
                        if (grow < 0) continue;
                        if ((uint32_t)grow >= n_rows_total) {
                            xvk_err(err, err_size, "xkv rows: sparse row outside row domain"); return false;
                        }
                        if ((uint32_t)grow < global_row_base ||
                            (uint32_t)grow >= global_row_base + arena_row_count) {
                            xvk_err(err, err_size, "xkv rows: row outside arena window"); return false;
                        }
                        const int32_t lrow = (int32_t)((uint32_t)grow - global_row_base);
                        items.push_back({lrow, row_pos ? row_pos[grow] : frag_positions[f], slot, f, item_order++, parent});
                    }
                }
            }

            std::stable_sort(items.begin(), items.end(), [](const row_item & a, const row_item & b) {
                if (a.row != b.row) return a.row < b.row;
                if (a.slot != b.slot) return a.slot < b.slot;
                return a.order < b.order;
            });
            items.erase(std::unique(items.begin(), items.end(), [](const row_item & a, const row_item & b) {
                return a.row == b.row && a.slot == b.slot;
            }), items.end());

            const bool hit = items.size() > refine_cap;
            if (hit) {
                items.resize(refine_cap);
                ++hits;
            }
            for (const row_item & it : items) all.push_back(it);
        }
    } catch (const std::exception &) {
        xvk_err(err, err_size, "xkv rows: host workspace allocation failed");
        return false;
    }

    // Compact ordinal tile window over the flattened parent-major rows.
    const int32_t total_filtered = (int32_t)all.size();
    uint64_t tile_cap64 = (uint64_t)n_queries * refine_cap;
    if (tile_cap64 > 1024u) tile_cap64 = 1024u;
    const int32_t tile_cap = (int32_t)tile_cap64;
    const int32_t win_start = (uint32_t)output_row_begin >= (uint32_t)total_filtered
        ? total_filtered : (int32_t)output_row_begin;
    int32_t win_end = win_start + tile_cap;
    if (win_end > total_filtered) win_end = total_filtered;
    const int32_t emitted = win_end - win_start;
    for (int32_t i = 0; i < emitted; ++i) {
        const row_item & item = all[(size_t)win_start + (size_t)i];
        const size_t dst = (size_t)i * 4;
        // Reconstruct contract: feeds ggml_xkv_reconstruct with rp.n_groups=1,
        // so row_refs quad is [arena_local_a_row, 0, 0, kv_head].
        out_row_refs[dst + 0] = item.row;
        out_row_refs[dst + 1] = 0;
        out_row_refs[dst + 2] = 0;
        out_row_refs[dst + 3] = frag_kv[(size_t)item.owner * 3 + 2];
        out_row_pos[i] = item.pos;
        if (out_row_entries) {
            out_row_entries[dst + 0] = 2; // GGML_XKV_ATTN_SOURCE_COLD
            out_row_entries[dst + 1] = i;
            out_row_entries[dst + 2] = item.slot;
            out_row_entries[dst + 3] = 1;
        }
        out_row_ptrs[(uint32_t)item.parent + 1] += 1;
    }
    for (uint32_t n = 0; n < n_parents; ++n) {
        out_row_ptrs[n + 1] += out_row_ptrs[n];
    }
    // Pad the remainder of the tile with safe dummy row 0 refs and valid=0 entries.
    for (int32_t j = emitted; j < tile_cap; ++j) {
        const size_t dst = (size_t)j * 4;
        out_row_refs[dst + 0] = 0;
        out_row_refs[dst + 1] = 0;
        out_row_refs[dst + 2] = 0;
        out_row_refs[dst + 3] = 0;
        out_row_pos[(size_t)j] = 0;
        if (out_row_entries) {
            out_row_entries[dst + 0] = 2;
            out_row_entries[dst + 1] = j;
            out_row_entries[dst + 2] = 0;
            out_row_entries[dst + 3] = 0;
        }
    }
    out_row_status[0] = GGML_XKV_LANDMARK_STATUS_OK;
    out_row_status[1] = emitted;
    out_row_status[2] = hits;
    out_row_status[3] = total_filtered;
    return true;
}

// ---- deterministic merge ----
bool ggml_xkv_landmark_merge_cpu_oracle(
        const int32_t * set_indices,
        const float   * set_scores,
        uint32_t n_queries, uint32_t n_sets, uint32_t set_cap, uint32_t top_k,
        int32_t * out_indices, float * out_scores, char * err, size_t err_size) {
    if (!set_indices || !set_scores || !out_indices || !out_scores) {
        xvk_err(err, err_size, "xkv merge: null pointer"); return false;
    }
    if (n_queries == 0 || n_queries > GGML_XKV_LANDMARK_MAX_QUERIES) {
        xvk_err(err, err_size, "xkv merge: n_queries out of range"); return false;
    }
    if (n_sets == 0 || n_sets > GGML_XKV_LANDMARK_MAX_SEGMENTS) {
        xvk_err(err, err_size, "xkv merge: n_sets out of range"); return false;
    }
    if (set_cap == 0 || set_cap > GGML_XKV_LANDMARK_MAX_TOP_K) {
        xvk_err(err, err_size, "xkv merge: set_cap out of range"); return false;
    }
    if (top_k == 0 || top_k > GGML_XKV_LANDMARK_MAX_TOP_K) {
        xvk_err(err, err_size, "xkv merge: top_k out of range"); return false;
    }
    if ((uint64_t)top_k > (uint64_t)n_sets * set_cap) {
        xvk_err(err, err_size, "xkv merge: top_k exceeds candidates"); return false;
    }
    std::vector<uint32_t> ids;
    std::vector<float> sc;
    ids.reserve((size_t)n_sets * set_cap);
    sc.reserve((size_t)n_sets * set_cap);
    std::vector<uint32_t> order;
    for (uint32_t qq = 0; qq < n_queries; ++qq) {
        ids.clear(); sc.clear();
        for (uint32_t s = 0; s < n_sets; ++s) {
            for (uint32_t i = 0; i < set_cap; ++i) {
                size_t at = ((size_t)s * n_queries + qq) * set_cap + i;
                int32_t id = set_indices[at];
                float v = set_scores[at];
                if (id < 0) continue; // sentinel short list
                if (!(v == v)) { xvk_err(err, err_size, "xkv merge: NaN candidate"); return false; }
                if (v <= -1e29f) continue; // illegal sentinel
                ids.push_back((uint32_t)id);
                sc.push_back(v);
            }
        }
        order.resize(ids.size());
        for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            if (sc[a] != sc[b]) return sc[a] > sc[b];
            return ids[a] < ids[b];
        });
        // Duplicate global ids across sets (overlapping ranges) collapse to first.
        std::vector<uint32_t> took;
        took.reserve(top_k);
        std::vector<float> took_sc;
        took_sc.reserve(top_k);
        for (uint32_t oi = 0; oi < order.size() && took.size() < top_k; ++oi) {
            uint32_t id = ids[order[oi]];
            bool dup = false;
            for (uint32_t t = 0; t < took.size(); ++t) {
                if (took[t] == id) { dup = true; break; }
            }
            if (dup) continue;
            took.push_back(id);
            took_sc.push_back(sc[order[oi]]);
        }
        for (uint32_t i = 0; i < top_k; ++i) {
            if (i < took.size()) {
                out_indices[(size_t)qq * top_k + i] = (int32_t)took[i];
                out_scores[(size_t)qq * top_k + i] = took_sc[i];
            } else {
                out_indices[(size_t)qq * top_k + i] = -1;
                out_scores[(size_t)qq * top_k + i] = -1e30f;
            }
        }
    }
    return true;
}

// Merge CPU oracle with explicit per-set global ID offsets (base mapping).
bool ggml_xkv_landmark_merge_with_base_cpu_oracle(
        const int32_t  * set_indices,
        const float    * set_scores,
        const uint32_t * set_base,
        uint32_t n_queries, uint32_t n_sets, uint32_t set_cap, uint32_t top_k,
        int32_t * out_indices, float * out_scores, int32_t * out_status,
        char * err, size_t err_size) {
    if (!set_indices || !set_scores || !out_indices || !out_scores || !out_status) {
        xvk_err(err, err_size, "xkv merge: null pointer"); return false;
    }
    if (n_queries == 0 || n_queries > GGML_XKV_LANDMARK_MAX_QUERIES) {
        out_status[0] = GGML_XKV_LANDMARK_STATUS_ERR_INPUT;
        xvk_err(err, err_size, "xkv merge: n_queries out of range"); return false;
    }
    if (n_sets == 0 || n_sets > GGML_XKV_LANDMARK_MAX_SEGMENTS) {
        out_status[0] = GGML_XKV_LANDMARK_STATUS_ERR_INPUT;
        xvk_err(err, err_size, "xkv merge: n_sets out of range"); return false;
    }
    if (set_cap == 0 || set_cap > GGML_XKV_LANDMARK_MAX_TOP_K ||
        top_k == 0 || top_k > GGML_XKV_LANDMARK_MAX_TOP_K) {
        out_status[0] = GGML_XKV_LANDMARK_STATUS_ERR_BUDGET;
        xvk_err(err, err_size, "xkv merge: cap/top_k out of range"); return false;
    }
    if ((uint64_t)top_k > (uint64_t)n_sets * set_cap) {
        out_status[0] = GGML_XKV_LANDMARK_STATUS_ERR_BUDGET;
        xvk_err(err, err_size, "xkv merge: top_k exceeds candidates"); return false;
    }

    std::vector<uint32_t> ids;
    std::vector<float> sc;
    ids.reserve((size_t)n_sets * set_cap);
    sc.reserve((size_t)n_sets * set_cap);
    std::vector<uint32_t> order;

    uint32_t total_took = 0;
    for (uint32_t qq = 0; qq < n_queries; ++qq) {
        ids.clear(); sc.clear();
        for (uint32_t s = 0; s < n_sets; ++s) {
            uint32_t base = set_base ? set_base[s] : 0u;
            for (uint32_t i = 0; i < set_cap; ++i) {
                size_t at = ((size_t)s * n_queries + qq) * set_cap + i;
                int32_t local_id = set_indices[at];
                float v = set_scores[at];
                if (local_id < 0) continue;
                if (!(v == v)) {
                    out_status[0] = GGML_XKV_LANDMARK_STATUS_ERR_INPUT;
                    xvk_err(err, err_size, "xkv merge: NaN candidate"); return false;
                }
                if (v <= -1e29f) continue;
                ids.push_back(base + (uint32_t)local_id);
                sc.push_back(v);
            }
        }
        order.resize(ids.size());
        for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            if (sc[a] != sc[b]) return sc[a] > sc[b];
            return ids[a] < ids[b];
        });
        std::vector<uint32_t> took;
        took.reserve(top_k);
        std::vector<float> took_sc;
        took_sc.reserve(top_k);
        for (uint32_t oi = 0; oi < order.size() && took.size() < top_k; ++oi) {
            uint32_t gid = ids[order[oi]];
            bool dup = false;
            for (uint32_t t = 0; t < took.size(); ++t) {
                if (took[t] == gid) { dup = true; break; }
            }
            if (dup) continue;
            took.push_back(gid);
            took_sc.push_back(sc[order[oi]]);
        }
        total_took += (uint32_t)took.size();
        for (uint32_t i = 0; i < top_k; ++i) {
            if (i < took.size()) {
                out_indices[(size_t)qq * top_k + i] = (int32_t)took[i];
                out_scores[(size_t)qq * top_k + i] = took_sc[i];
            } else {
                out_indices[(size_t)qq * top_k + i] = -1;
                out_scores[(size_t)qq * top_k + i] = -1e30f;
            }
        }
    }
    out_status[0] = GGML_XKV_LANDMARK_STATUS_OK;
    out_status[1] = (int32_t)total_took;
    out_status[2] = 0;
    out_status[3] = 0;
    return true;
}

// ---- rows / merge device-op builders + validators (enum wiring by Attention) ----
static enum ggml_op xvk_extra_op(const char * want_rows) {
    if (strcmp(want_rows, "rows") == 0) return GGML_OP_XKV_LANDMARK_ROWS;
    return GGML_OP_XKV_LANDMARK_MERGE;
}

struct ggml_tensor * ggml_xkv_landmark_rows(
        struct ggml_context * ctx,
        struct ggml_tensor  * sel_idx,
        struct ggml_tensor  * frag_meta,
        struct ggml_tensor  * frag_row_off,
        struct ggml_tensor  * frag_row_ids,
        struct ggml_tensor  * frag_kv,
        struct ggml_tensor  * row_pos,
        struct ggml_tensor  * row_ptrs,
        struct ggml_tensor  * row_out_pos,
        struct ggml_tensor  * row_entries,
        struct ggml_tensor  * row_status,
        const ggml_xkv_landmark_rows_params * params) {
    if (!ctx || !sel_idx || !frag_meta || !frag_row_off || !frag_row_ids || !frag_kv ||
        !row_pos || !row_ptrs || !row_out_pos || !row_entries || !row_status || !params) {
        return nullptr;
    }
    uint64_t tile_cap = (uint64_t)params->n_queries * params->refine_cap;
    if (tile_cap > 1024u) tile_cap = 1024u;
    if (tile_cap == 0) return nullptr;
    struct ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, (int64_t)tile_cap);
    if (!dst) return nullptr;
    memcpy(dst->op_params, params, sizeof(*params));
    dst->op = xvk_extra_op("rows");
    dst->src[0] = sel_idx; dst->src[1] = frag_meta; dst->src[2] = frag_row_off;
    dst->src[3] = frag_row_ids; dst->src[4] = frag_kv; dst->src[5] = row_pos;
    dst->src[6] = row_ptrs; dst->src[7] = row_out_pos; dst->src[8] = row_entries;
    dst->src[9] = row_status;
    return dst;
}

bool ggml_xkv_landmark_rows_supports(
        const struct ggml_tensor * sel_idx,
        const struct ggml_tensor * frag_meta,
        const struct ggml_tensor * frag_row_off,
        const struct ggml_tensor * frag_row_ids,
        const struct ggml_tensor * frag_kv,
        const struct ggml_tensor * row_pos,
        const struct ggml_tensor * row_ptrs,
        const struct ggml_tensor * row_refs,
        const struct ggml_tensor * row_out_pos,
        const struct ggml_tensor * row_entries,
        const struct ggml_tensor * row_status,
        const ggml_xkv_landmark_rows_params * params,
        char * err, size_t err_size) {
    if (!sel_idx || !frag_meta || !frag_row_off || !frag_row_ids || !frag_kv || !row_pos ||
        !row_ptrs || !row_refs || !row_out_pos || !row_entries || !row_status || !params) {
        xvk_err(err, err_size, "xkv rows: null tensor/params"); return false;
    }
    if (params->version != GGML_XKV_LANDMARK_VERSION) { xvk_err(err, err_size, "xkv rows: bad version"); return false; }
    if (params->n_queries == 0 || params->n_queries > GGML_XKV_LANDMARK_MAX_QUERIES) {
        xvk_err(err, err_size, "xkv rows: n_queries out of range"); return false;
    }
    if (params->top_k == 0 || params->top_k > GGML_XKV_LANDMARK_MAX_TOP_K) {
        xvk_err(err, err_size, "xkv rows: top_k out of range"); return false;
    }
    if (params->n_frags == 0) { xvk_err(err, err_size, "xkv rows: empty frags"); return false; }
    if (params->refine_cap == 0 || params->refine_cap > GGML_XKV_LANDMARK_MAX_REFINE_CAP) {
        xvk_err(err, err_size, "xkv rows: refine_cap out of range"); return false;
    }
    if (params->fstride != 4 && params->fstride != 6) { xvk_err(err, err_size, "xkv rows: bad fstride"); return false; }
    const uint32_t known = GGML_XKV_LANDMARK_FLAG_SPARSE_ROWS | GGML_XKV_LANDMARK_FLAG_PERQ_LEGAL;
    if (params->flags & ~known) { xvk_err(err, err_size, "xkv rows: unknown flags"); return false; }
    if (params->max_frag_rows == 0 || params->max_frag_rows > GGML_XKV_LANDMARK_MAX_FRAG_ROWS) {
        xvk_err(err, err_size, "xkv rows: max_frag_rows out of range"); return false;
    }
    if (params->n_rows_total == 0) { xvk_err(err, err_size, "xkv rows: empty row domain"); return false; }
    if (params->arena_filter == UINT32_MAX) {
        if (params->global_row_base != 0 || params->arena_row_count != params->n_rows_total) {
            xvk_err(err, err_size, "xkv rows: ALL arena mode needs base 0 and full count"); return false;
        }
    } else {
        if (params->arena_row_count == 0 || params->arena_row_count > params->n_rows_total ||
            params->global_row_base >= params->n_rows_total ||
            (uint64_t)params->global_row_base + params->arena_row_count > params->n_rows_total) {
            xvk_err(err, err_size, "xkv rows: arena window outside row domain"); return false;
        }
    }
    if (params->has_query_map > 1 || params->_reserved != 0 ||
        (params->has_query_map == 0 && params->n_parent_queries != 0) ||
        (params->has_query_map != 0 &&
            (params->n_parent_queries == 0 || params->n_parent_queries > params->n_queries))) {
        xvk_err(err, err_size, "xkv rows: inconsistent query map params"); return false;
    }
    const int64_t sel_width = (int64_t)params->top_k + (params->has_query_map ? 2 : 0);
    if (sel_idx->type != GGML_TYPE_I32 || sel_idx->ne[0] != sel_width ||
        sel_idx->ne[1] != (int64_t)params->n_queries) {
        xvk_err(err, err_size, "xkv rows: sel_idx shape does not match query-map mode"); return false;
    }
    if (frag_meta->type != GGML_TYPE_I32 || frag_meta->ne[0] != (int64_t)params->fstride ||
        frag_meta->ne[1] < (int64_t)params->n_frags) {
        xvk_err(err, err_size, "xkv rows: frag_meta shape"); return false;
    }
    if (frag_kv->type != GGML_TYPE_I32 || frag_kv->ne[0] != 3 ||
        frag_kv->ne[1] < (int64_t)params->n_frags) {
        xvk_err(err, err_size, "xkv rows: frag_kv must be I32[3,nf]"); return false;
    }
    if (row_pos->type != GGML_TYPE_I32 || ggml_nelements(row_pos) != params->n_rows_total) {
        xvk_err(err, err_size, "xkv rows: row_pos must match n_rows_total"); return false;
    }
    if ((params->flags & GGML_XKV_LANDMARK_FLAG_SPARSE_ROWS) != 0 &&
        (frag_row_off->type != GGML_TYPE_I32 || ggml_nelements(frag_row_off) < params->n_frags + 1 ||
         frag_row_ids->type != GGML_TYPE_I32)) {
        xvk_err(err, err_size, "xkv rows: sparse row tensors malformed"); return false;
    }
    const uint32_t n_ptr_queries = params->has_query_map ? params->n_parent_queries : params->n_queries;
    if (row_ptrs->type != GGML_TYPE_I32 || ggml_nelements(row_ptrs) != (int64_t)n_ptr_queries + 1) {
        xvk_err(err, err_size, "xkv rows: row_ptrs length mismatch"); return false;
    }
    uint64_t tile_cap = (uint64_t)params->n_queries * params->refine_cap;
    if (tile_cap > 1024u) tile_cap = 1024u;
    if (row_refs->type != GGML_TYPE_I32 || row_refs->ne[0] != 4 ||
        ggml_nelements(row_refs) != (int64_t)4 * (int64_t)tile_cap) {
        xvk_err(err, err_size, "xkv rows: row_refs must hold 4*tile_capacity"); return false;
    }
    if (row_out_pos->type != GGML_TYPE_I32 ||
        ggml_nelements(row_out_pos) != (int64_t)tile_cap) {
        xvk_err(err, err_size, "xkv rows: row_out_pos must hold tile_capacity"); return false;
    }
    if (row_entries->type != GGML_TYPE_I32 || row_entries->ne[0] != 4 ||
        ggml_nelements(row_entries) != (int64_t)4 * (int64_t)tile_cap) {
        xvk_err(err, err_size, "xkv rows: row_entries must hold 4*tile_capacity"); return false;
    }
    if (row_status->type != GGML_TYPE_I32 || row_status->ne[0] != 4) {
        xvk_err(err, err_size, "xkv rows: row_status must be I32[4]"); return false;
    }
    return true;
}

struct ggml_tensor * ggml_xkv_landmark_merge(
        struct ggml_context * ctx,
        struct ggml_tensor  * set_idx,
        struct ggml_tensor  * set_sc,
        struct ggml_tensor  * set_base,
        struct ggml_tensor  * out_sc,
        struct ggml_tensor  * status,
        const ggml_xkv_landmark_merge_params * params) {
    if (!ctx || !set_idx || !set_sc || !out_sc || !status || !params) return nullptr;
    struct ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, params->top_k, params->n_queries);
    if (!dst) return nullptr;
    memcpy(dst->op_params, params, sizeof(*params));
    dst->op = xvk_extra_op("merge");
    dst->src[0] = set_idx; dst->src[1] = set_sc; dst->src[2] = set_base; dst->src[3] = out_sc; dst->src[4] = status;
    return dst;
}

bool ggml_xkv_landmark_merge_supports(
        const struct ggml_tensor * set_idx,
        const struct ggml_tensor * set_sc,
        const struct ggml_tensor * set_base,
        const struct ggml_tensor * out_sc,
        const struct ggml_tensor * status,
        const struct ggml_tensor * dst_out_idx,
        const ggml_xkv_landmark_merge_params * params,
        char * err, size_t err_size) {
    if (!set_idx || !set_sc || !out_sc || !status || !dst_out_idx || !params) {
        xvk_err(err, err_size, "xkv merge: null tensor/params"); return false;
    }
    if (params->version != GGML_XKV_LANDMARK_VERSION) { xvk_err(err, err_size, "xkv merge: bad version"); return false; }
    if (params->n_queries == 0 || params->n_queries > GGML_XKV_LANDMARK_MAX_QUERIES) {
        xvk_err(err, err_size, "xkv merge: n_queries out of range"); return false;
    }
    if (params->n_sets == 0 || params->n_sets > GGML_XKV_LANDMARK_MAX_SEGMENTS) {
        xvk_err(err, err_size, "xkv merge: n_sets out of range"); return false;
    }
    if (params->set_cap == 0 || params->set_cap > GGML_XKV_LANDMARK_MAX_TOP_K) {
        xvk_err(err, err_size, "xkv merge: set_cap out of range"); return false;
    }
    if (params->top_k == 0 || params->top_k > GGML_XKV_LANDMARK_MAX_TOP_K) {
        xvk_err(err, err_size, "xkv merge: top_k out of range"); return false;
    }
    if ((uint64_t)params->top_k > (uint64_t)params->n_sets * params->set_cap) {
        xvk_err(err, err_size, "xkv merge: top_k exceeds candidates"); return false;
    }
    if (params->_pad0 != 0 || params->_pad1 != 0) {
        xvk_err(err, err_size, "xkv merge: pads must be 0"); return false;
    }
    if (params->has_set_base != 0) {
        if (!set_base || set_base->type != GGML_TYPE_I32 || set_base->ne[0] != (int64_t)params->n_sets) {
            xvk_err(err, err_size, "xkv merge: set_base must be I32[n_sets]"); return false;
        }
    }
    if (set_idx->type != GGML_TYPE_I32 || set_idx->ne[0] != (int64_t)params->set_cap ||
        set_idx->ne[1] != (int64_t)params->n_queries || set_idx->ne[2] != (int64_t)params->n_sets) {
        xvk_err(err, err_size, "xkv merge: set_idx shape"); return false;
    }
    if (set_sc->type != GGML_TYPE_F32 || set_sc->ne[0] != (int64_t)params->set_cap ||
        set_sc->ne[1] != (int64_t)params->n_queries || set_sc->ne[2] != (int64_t)params->n_sets) {
        xvk_err(err, err_size, "xkv merge: set_sc shape"); return false;
    }
    if (out_sc->type != GGML_TYPE_F32 || out_sc->ne[0] != (int64_t)params->top_k ||
        out_sc->ne[1] != (int64_t)params->n_queries) {
        xvk_err(err, err_size, "xkv merge: out_sc shape"); return false;
    }
    if (dst_out_idx->type != GGML_TYPE_I32 || dst_out_idx->ne[0] != (int64_t)params->top_k ||
        dst_out_idx->ne[1] != (int64_t)params->n_queries) {
        xvk_err(err, err_size, "xkv merge: dst shape"); return false;
    }
    if (status->type != GGML_TYPE_I32 || status->ne[0] != 4) {
        xvk_err(err, err_size, "xkv merge: status must be I32[4]"); return false;
    }
    return true;
}
