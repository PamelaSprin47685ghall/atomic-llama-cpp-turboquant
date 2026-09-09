// llama-xkv-seal.cpp — Production native sealing bridge (XKV).
//
// See llama-xkv-seal.h for ownership/contracts.
//
// Architecture (Main-directed bounded workspace + liveness reuse):
//  1. Persistent output context (ctx_out): holds only the final packed outputs
//     (A_K, B_K, A_V, B_V, and optional landmark dst/eb/srcfp) plus statuses
//     across all groups. Allocated once from config.buft.
//  2. Transient per-group context (ctx_tr) + gallocr: builds canonicalize
//     and factorize DAG for each group sequentially. gallocr reserves on the
//     WORST-CASE (largest) group graph and reuses the SAME device buffer for
//     every group's dense staging and factor scratch.
//     Peak memory is strictly: final compact payload + MAX-group transient!
//  3. Single queue execution: all group graphs are submitted to the backend
//     asynchronously in group order; ONE final synchronize at the end.
//  4. Statuses + residual words + S telemetry are downloaded via scalar D2H.
//  5. Adoption wraps final outputs into immutable backend handles with
//     unbound-checksum and bundle provenance.

#include "llama-xkv-seal.h"

#include "ggml-xkv.h"
#include "ggml-xkv-factor.h"
#include "ggml-xkv-landmark-build.h"
#include "llama-xkv-cache.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <memory>
#include <vector>

namespace llama_xkv {
namespace {

bool ck_add_u64(uint64_t a, uint64_t b, uint64_t & out) {
    __uint128_t s = (__uint128_t)a + b;
    if (s > UINT64_MAX) return false;
    out = (uint64_t)s;
    return true;
}
bool ck_mul_u64(uint64_t a, uint64_t b, uint64_t & out) {
    __uint128_t p = (__uint128_t)a * b;
    if (p > UINT64_MAX) return false;
    out = (uint64_t)p;
    return true;
}

void seal_err(std::string * err, const std::string & msg) {
    if (err) *err = msg;
}

bool is_turbo(ggml_type t) {
    return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
}

bool scales_trivial(const std::vector<float> & s) {
    for (float v : s) {
        if (!(v == 1.0f)) return false;
    }
    return true;
}

bool seal_scratch_offsets(uint64_t n, uint64_t m, uint64_t r, uint64_t l,
                          uint64_t & off_s, uint64_t & out_total) {
    __uint128_t y0 = 0;
    __uint128_t z0 = y0 + (__uint128_t)n * l;
    __uint128_t c0 = z0 + (__uint128_t)m * l;
    __uint128_t g0 = c0 + (__uint128_t)l * m;
    __uint128_t v0 = g0 + (__uint128_t)l * l;
    __uint128_t s0 = v0 + (__uint128_t)l * l;
    __uint128_t total = s0 + l + l + (__uint128_t)n * r + (__uint128_t)m * r + 640;
    if (s0 > UINT64_MAX || total > UINT64_MAX) return false;
    off_s = (uint64_t)s0;
    out_total = (uint64_t)total;
    return true;
}

uint64_t seal_eff_l(uint64_t n, uint64_t m, uint32_t rank, uint32_t over) {
    uint64_t mn = std::min(n, m);
    uint64_t l = (uint64_t)rank + over;
    return std::min(mn, l);
}

struct native_group_geometry {
    uint64_t mk = 0;
    uint64_t mv = 0;
    uint32_t eff_rk = 0;
    uint32_t eff_rv = 0;
    uint32_t pra_k = 0;
    uint32_t prb_k = 0;
    uint32_t pra_v = 0;
    uint32_t prb_v = 0;
    uint64_t lk = 0;
    uint64_t lv = 0;
    std::vector<uint32_t> padded_k;
    std::vector<uint32_t> padded_v;
    std::vector<uint32_t> feat_off_k;
    std::vector<uint32_t> feat_off_v;
    std::vector<uint32_t> had_dim_k;
    std::vector<uint32_t> had_dim_v;
    uint32_t pad_D = 0;
    uint32_t n_chunks = 0;
    uint32_t max_feature_dim = 0;
};

bool native_group_geometry_checked(
    const xkv_native_seal_config & config,
    const xkv_native_seal_group & g,
    native_group_geometry & gg,
    const char * scope,
    std::string * err) {
    auto fail = [&](const std::string & message) {
        seal_err(err, std::string(scope) + ": " + message);
        return false;
    };
    if (g.layers.empty() || g.layers.size() > UINT32_MAX) return fail("invalid layer count");
    if (g.rank_k == 0 || g.rank_v == 0) return fail("zero rank");
    if (g.power_iterations == 0) return fail("power_iterations==0");
    if (g.balance_mode > 2) return fail("bad balance mode");
    if (g.want_landmarks && g.chunk_tokens == 0) return fail("chunk_tokens==0");
    if (g.want_landmarks && g.landmark_type != GGML_TYPE_Q8_0 &&
        g.landmark_type != GGML_TYPE_TURBO4_0) {
        return fail("unsupported landmark type");
    }
    const ggml_type codecs[4] = {g.codec_a_k, g.codec_b_k, g.codec_a_v, g.codec_b_v};
    for (ggml_type codec : codecs) {
        if (!ggml_xkv_codec_supported(codec)) return fail("unsupported factor codec");
    }

    try {
        gg.padded_k.reserve(g.layers.size());
        gg.padded_v.reserve(g.layers.size());
        gg.feat_off_k.reserve(g.layers.size());
        gg.feat_off_v.reserve(g.layers.size());
        gg.had_dim_k.reserve(g.layers.size());
        gg.had_dim_v.reserve(g.layers.size());
    } catch (const std::exception &) {
        return fail("geometry metadata allocation failed");
    }

    uint64_t off_k = 0;
    uint64_t off_v = 0;
    for (const auto & layer : g.layers) {
        if (!layer.hot_k || !layer.hot_v) return fail("null hot tensor");
        if (layer.n_heads == 0 || layer.head_dim_k == 0 || layer.head_dim_v == 0) {
            return fail("degenerate head geometry");
        }
        for (const ggml_tensor * hot : {layer.hot_k, layer.hot_v}) {
            if (!hot->buffer || ggml_backend_buffer_get_type(hot->buffer) != config.buft) {
                return fail("hot tensor off-placement");
            }
        }
        if (layer.hot_k->ne[0] <= 0 || layer.hot_v->ne[0] <= 0 ||
            (uint64_t)layer.hot_k->ne[0] % layer.n_heads != 0 ||
            (uint64_t)layer.hot_v->ne[0] % layer.n_heads != 0) {
            return fail("hot width not divisible by n_heads");
        }
        const uint64_t phk64 = (uint64_t)layer.hot_k->ne[0] / layer.n_heads;
        const uint64_t phv64 = (uint64_t)layer.hot_v->ne[0] / layer.n_heads;
        if (phk64 > UINT32_MAX || phv64 > UINT32_MAX) return fail("hot padded width overflow");
        const uint32_t phk = (uint32_t)phk64;
        const uint32_t phv = (uint32_t)phv64;
        if (phk < layer.head_dim_k || phv < layer.head_dim_v) {
            return fail("padded width narrower than head dim");
        }
        const uint64_t want_k = is_turbo(layer.hot_k->type)
            ? ((uint64_t)layer.head_dim_k + 127u) / 128u * 128u : layer.head_dim_k;
        const uint64_t want_v = is_turbo(layer.hot_v->type)
            ? ((uint64_t)layer.head_dim_v + 127u) / 128u * 128u : layer.head_dim_v;
        if (phk64 != want_k || phv64 != want_v) return fail("hot padded width mismatch");
        // Selected rows must address allocated hot storage on every owning
        // layer (fail closed before any backend work, IDs, or publication).
        if (layer.hot_k->ne[1] <= 0 || layer.hot_v->ne[1] <= 0) {
            return fail("hot height degenerate");
        }
        if (config.physical_rows.size() != config.n_rows) {
            return fail("rows length mismatch");
        }
        for (uint32_t i = 0; i < config.n_rows; ++i) {
            if ((int64_t)config.physical_rows[i] < 0 ||
                (int64_t)config.physical_rows[i] >= layer.hot_k->ne[1] ||
                (int64_t)config.physical_rows[i] >= layer.hot_v->ne[1]) {
                return fail("physical row out of range");
            }
        }

        const uint32_t freq_count = layer.rotary_dim_k / 2;
        if ((layer.rotary_dim_k & 1u) || layer.rotary_dim_k > layer.head_dim_k) {
            return fail("bad rotary_dim_k");
        }
        if (layer.rotary_dim_k > 0) {
            if (layer.rope_omega_k.size() != freq_count || layer.rope_mag_k.size() != freq_count) {
                return fail("rope table size mismatch");
            }
            for (uint32_t f = 0; f < freq_count; ++f) {
                if (!std::isfinite(layer.rope_omega_k[f]) || !std::isfinite(layer.rope_mag_k[f]) ||
                    !(layer.rope_mag_k[f] > 0.0f)) {
                    return fail("rope table non-finite/bad mag");
                }
            }
            if (layer.rope_mode_k != GGML_XKV_ROPE_HALF &&
                layer.rope_mode_k != GGML_XKV_ROPE_INTERLEAVED) {
                return fail("bad rope_mode_k");
            }
        }

        auto hadamard_dim = [&](const std::vector<float> & values, uint32_t head_dim,
                                uint32_t & result) {
            if (values.empty()) {
                result = 0;
                return true;
            }
            const uint64_t h = (uint64_t)llround(std::sqrt((double)values.size()));
            if (h == 0 || h > UINT32_MAX || h * h != values.size() || head_dim % (uint32_t)h != 0) {
                return false;
            }
            for (float value : values) if (!std::isfinite(value)) return false;
            result = (uint32_t)h;
            return true;
        };
        uint32_t had_k = 0;
        uint32_t had_v = 0;
        if (!hadamard_dim(layer.hadamard_k, layer.head_dim_k, had_k) ||
            !hadamard_dim(layer.hadamard_v, layer.head_dim_v, had_v)) {
            return fail("bad hadamard transform");
        }
        if (!scales_trivial(layer.channel_scales_k) || !scales_trivial(layer.channel_scales_v)) {
            return fail("nontrivial channel scales unsupported");
        }

        uint64_t dim_k = 0;
        uint64_t dim_v = 0;
        if (!ck_mul_u64(layer.n_heads, layer.head_dim_k, dim_k) ||
            !ck_mul_u64(layer.n_heads, layer.head_dim_v, dim_v) ||
            dim_k > UINT32_MAX || dim_v > UINT32_MAX ||
            off_k > UINT32_MAX || off_v > UINT32_MAX) {
            return fail("layer feature width overflow");
        }
        gg.feat_off_k.push_back((uint32_t)off_k);
        gg.feat_off_v.push_back((uint32_t)off_v);
        if (!ck_add_u64(off_k, dim_k, off_k) || !ck_add_u64(off_v, dim_v, off_v) ||
            off_k > UINT32_MAX || off_v > UINT32_MAX) {
            return fail("group width overflow");
        }
        gg.padded_k.push_back(phk);
        gg.padded_v.push_back(phv);
        gg.had_dim_k.push_back(had_k);
        gg.had_dim_v.push_back(had_v);
        gg.max_feature_dim = std::max(gg.max_feature_dim, (uint32_t)dim_k);
    }

    gg.mk = off_k;
    gg.mv = off_v;
    if (gg.mk == 0 || gg.mv == 0) return fail("empty group width");
    const uint64_t n = config.n_rows;
    gg.eff_rk = (uint32_t)std::min<uint64_t>(g.rank_k, std::min(n, gg.mk));
    gg.eff_rv = (uint32_t)std::min<uint64_t>(g.rank_v, std::min(n, gg.mv));
    if (gg.eff_rk == 0 || gg.eff_rv == 0) return fail("degenerate effective rank");

    const int32_t pra_k = ggml_xkv_padded_rank(g.codec_a_k, gg.eff_rk);
    const int32_t prb_k = ggml_xkv_padded_rank(g.codec_b_k, gg.eff_rk);
    const int32_t pra_v = ggml_xkv_padded_rank(g.codec_a_v, gg.eff_rv);
    const int32_t prb_v = ggml_xkv_padded_rank(g.codec_b_v, gg.eff_rv);
    if (pra_k <= 0 || prb_k <= 0 || pra_v <= 0 || prb_v <= 0) {
        return fail("padded rank unsupported or overflowed");
    }
    gg.pra_k = (uint32_t)pra_k;
    gg.prb_k = (uint32_t)prb_k;
    gg.pra_v = (uint32_t)pra_v;
    gg.prb_v = (uint32_t)prb_v;
    if (gg.pra_k != gg.prb_k || gg.pra_v != gg.prb_v) {
        return fail("per-pair padded width mismatch");
    }

    gg.lk = seal_eff_l(n, gg.mk, gg.eff_rk, g.oversampling);
    gg.lv = seal_eff_l(n, gg.mv, gg.eff_rv, g.oversampling);
    if (gg.lk == 0 || gg.lv == 0 || gg.lk > 512 || gg.lv > 512) {
        return fail("core width out of device bounds");
    }
    if (g.want_landmarks) {
        const int32_t pad_d = ggml_xkv_padded_rank(g.landmark_type, (uint32_t)gg.mk);
        if (pad_d <= 0) return fail("landmark padded width unsupported or overflowed");
        gg.pad_D = (uint32_t)pad_d;
        const uint64_t chunks = (n + g.chunk_tokens - 1u) / g.chunk_tokens;
        if (chunks == 0 || chunks > UINT32_MAX) return fail("landmark chunk count overflow");
        gg.n_chunks = (uint32_t)chunks;
    }
    return true;
}

bool checked_matrix_bytes(ggml_type type, uint64_t cols, uint64_t rows, uint64_t & bytes) {
    if (cols == 0 || rows == 0 || cols > INT64_MAX) return false;
    const size_t row_bytes = ggml_row_size(type, (int64_t)cols);
    return row_bytes != 0 && ck_mul_u64(row_bytes, rows, bytes) && bytes <= SIZE_MAX;
}

struct seal_ctx_deleter {
    void operator()(ggml_context * ctx) const { if (ctx) ggml_free(ctx); }
};

bool native_persistent_alloc_bytes(
    const xkv_native_seal_config & config,
    const std::vector<native_group_geometry> & geoms,
    size_t & bytes,
    std::string * err) {
    uint64_t tensor_slots = 0;
    uint64_t ctx_bytes = 0;
    if (geoms.size() != config.groups.size() ||
        !ck_mul_u64(config.groups.size(), 5, tensor_slots) ||
        !ck_add_u64(tensor_slots, 32, tensor_slots) ||
        !ck_mul_u64(tensor_slots, ggml_tensor_overhead(), ctx_bytes) ||
        !ck_add_u64(ctx_bytes, 65536, ctx_bytes) || ctx_bytes > SIZE_MAX) {
        seal_err(err, "seal: persistent context estimate overflow");
        return false;
    }
    std::vector<uint8_t> context_memory;
    try {
        context_memory.resize((size_t)ctx_bytes);
    } catch (const std::exception &) {
        seal_err(err, "seal: persistent context estimate allocation failed");
        return false;
    }
    ggml_init_params params = {(size_t)ctx_bytes, context_memory.data(), true};
    std::unique_ptr<ggml_context, seal_ctx_deleter> ctx(ggml_init(params));
    if (!ctx) {
        seal_err(err, "seal: persistent context estimate init failed");
        return false;
    }
    for (size_t gi = 0; gi < config.groups.size(); ++gi) {
        const auto & group = config.groups[gi];
        const auto & gg = geoms[gi];
        if (!ggml_new_tensor_2d(ctx.get(), group.codec_a_k, gg.pra_k, config.n_rows) ||
            !ggml_new_tensor_2d(ctx.get(), group.codec_b_k, gg.prb_k, (int64_t)gg.mk) ||
            !ggml_new_tensor_2d(ctx.get(), group.codec_a_v, gg.pra_v, config.n_rows) ||
            !ggml_new_tensor_2d(ctx.get(), group.codec_b_v, gg.prb_v, (int64_t)gg.mv)) {
            seal_err(err, "seal: persistent output tensor allocation failed");
            return false;
        }
        if (group.want_landmarks) {
            if (!ggml_new_tensor_2d(ctx.get(), group.landmark_type, gg.pad_D, gg.n_chunks)) {
                seal_err(err, "seal: persistent landmark tensor allocation failed");
                return false;
            }
        }
    }
    bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), config.buft);
    if (bytes == 0) {
        seal_err(err, "seal: persistent allocator size probe failed");
        return false;
    }
    return true;
}

bool native_aux_alloc_bytes(
    const xkv_native_seal_config & config,
    const std::vector<native_group_geometry> & geoms,
    size_t & bytes,
    std::string * err) {
    uint64_t tensor_count = 0;
    for (const auto & group : config.groups) {
        uint64_t count = 2 + (group.want_landmarks ? 3 : 0);
        uint64_t layer_count = 0;
        if (!ck_mul_u64(group.layers.size(), 2, layer_count) ||
            !ck_add_u64(count, layer_count, count) ||
            !ck_add_u64(tensor_count, count, tensor_count)) {
            seal_err(err, "estimate: telemetry tensor count overflow");
            return false;
        }
    }
    uint64_t context_bytes = 0;
    if (geoms.size() != config.groups.size() ||
        !ck_add_u64(tensor_count, 32, tensor_count) ||
        !ck_mul_u64(tensor_count, ggml_tensor_overhead(), context_bytes) ||
        !ck_add_u64(context_bytes, 65536, context_bytes) || context_bytes > SIZE_MAX) {
        seal_err(err, "estimate: telemetry context overflow");
        return false;
    }
    std::vector<uint8_t> memory;
    try {
        memory.resize((size_t)context_bytes);
    } catch (const std::exception &) {
        seal_err(err, "estimate: telemetry context allocation failed");
        return false;
    }
    ggml_init_params params = {(size_t)context_bytes, memory.data(), true};
    std::unique_ptr<ggml_context, seal_ctx_deleter> ctx(ggml_init(params));
    if (!ctx) {
        seal_err(err, "estimate: telemetry context init failed");
        return false;
    }
    for (size_t gi = 0; gi < config.groups.size(); ++gi) {
        const auto & group = config.groups[gi];
        const auto & gg = geoms[gi];
        if (!ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 10) ||
            !ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 10)) {
            seal_err(err, "estimate: telemetry status tensor allocation failed");
            return false;
        }
        for (size_t li = 0; li < group.layers.size(); ++li) {
            if (!ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1) ||
                !ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1)) {
                seal_err(err, "estimate: telemetry layer status allocation failed");
                return false;
            }
        }
        if (group.want_landmarks) {
            if (!ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, gg.n_chunks) ||
                !ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, gg.n_chunks) ||
                !ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 4)) {
                seal_err(err, "estimate: telemetry landmark tensor allocation failed");
                return false;
            }
        }
    }
    bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), config.buft);
    if (bytes == 0) {
        seal_err(err, "estimate: telemetry allocator size probe failed");
        return false;
    }
    return true;
}

} // namespace

bool xkv_native_seal_estimate(
    const xkv_native_seal_config & config,
    xkv_native_seal_estimate_result & out,
    std::string * err) {
    if (!config.buft) {
        seal_err(err, "estimate: null buft");
        return false;
    }
    if (config.n_rows == 0 || config.groups.empty()) {
        seal_err(err, "estimate: empty rows or groups");
        return false;
    }
    if (config.physical_rows.size() != config.n_rows ||
        config.storage_positions.size() != config.n_rows) {
        seal_err(err, "estimate: rows/positions length mismatch");
        return false;
    }
    uint32_t n = config.n_rows;
    uint64_t raw_stream_bytes = 0;
    uint64_t max_transient = 0;
    std::vector<native_group_geometry> geoms;
    try {
        geoms.resize(config.groups.size());
    } catch (const std::exception &) {
        seal_err(err, "estimate: geometry allocation failed");
        return false;
    }

    for (size_t gi = 0; gi < config.groups.size(); ++gi) {
        const auto & g = config.groups[gi];
        auto & gg = geoms[gi];
        if (!native_group_geometry_checked(config, g, gg, "estimate", err)) return false;

        uint64_t streams[4] = {};
        if (!checked_matrix_bytes(g.codec_a_k, gg.pra_k, n, streams[0]) ||
            !checked_matrix_bytes(g.codec_b_k, gg.prb_k, gg.mk, streams[1]) ||
            !checked_matrix_bytes(g.codec_a_v, gg.pra_v, n, streams[2]) ||
            !checked_matrix_bytes(g.codec_b_v, gg.prb_v, gg.mv, streams[3])) {
            seal_err(err, "estimate: persistent stream size overflow");
            return false;
        }
        for (uint64_t bytes : streams) {
            if (!ck_add_u64(raw_stream_bytes, bytes, raw_stream_bytes)) {
                seal_err(err, "estimate: persistent destination overflow");
                return false;
            }
        }
        if (g.want_landmarks) {
            uint64_t landmark_bytes = 0;
            if (!checked_matrix_bytes(g.landmark_type, gg.pad_D, gg.n_chunks, landmark_bytes) ||
                !ck_add_u64(raw_stream_bytes, landmark_bytes, raw_stream_bytes)) {
                seal_err(err, "estimate: landmark destination overflow");
                return false;
            }
        }

        // Transient scratch estimation for this group:
        ggml_xkv_factorize_params fpk = {}, fpv = {};
        fpk.version = GGML_XKV_FACTOR_VERSION;
        fpk.rows_n = n; fpk.cols_m = (uint32_t)gg.mk; fpk.requested_rank = gg.eff_rk;
        fpk.oversampling = g.oversampling; fpk.power_iterations = g.power_iterations;
        fpk.balance_mode = g.balance_mode;
        fpk.type_a = (uint32_t)g.codec_a_k; fpk.type_b = (uint32_t)g.codec_b_k;
        fpk.pad_r_a = gg.pra_k; fpk.pad_r_b = gg.prb_k;

        fpv = fpk;
        fpv.cols_m = (uint32_t)gg.mv; fpv.requested_rank = gg.eff_rv;
        fpv.type_a = (uint32_t)g.codec_a_v; fpv.type_b = (uint32_t)g.codec_b_v;
        fpv.pad_r_a = gg.pra_v; fpv.pad_r_b = gg.prb_v;

        size_t need_k = 0, need_v = 0;
        char e0[256] = {};
        if (!ggml_xkv_factorize_scratch_bytes(&fpk, &need_k, e0, sizeof(e0)) ||
            !ggml_xkv_factorize_scratch_bytes(&fpv, &need_v, e0, sizeof(e0))) {
            seal_err(err, std::string("estimate: factorize scratch error: ") + e0);
            return false;
        }
        uint64_t dense_k = 0;
        uint64_t dense_v = 0;
        uint64_t grp_transient = 0;
        if (!ck_mul_u64(gg.mk, n, dense_k) || !ck_mul_u64(dense_k, sizeof(float), dense_k) ||
            !ck_mul_u64(gg.mv, n, dense_v) || !ck_mul_u64(dense_v, sizeof(float), dense_v) ||
            !ck_add_u64(dense_k, dense_v, grp_transient) ||
            !ck_add_u64(grp_transient, need_k, grp_transient) ||
            !ck_add_u64(grp_transient, need_v, grp_transient)) {
            seal_err(err, "estimate: transient workspace overflow");
            return false;
        }
        if (g.want_landmarks) {
            ggml_xkv_landmark_build_params lbp = {};
            lbp.version = GGML_XKV_LANDMARK_BUILD_VERSION;
            lbp.n_rows = n;
            lbp.n_chunks = gg.n_chunks;
            lbp.chunk_tokens = g.chunk_tokens;
            lbp.total_dim = (uint32_t)gg.mk;
            lbp.padded_dim = gg.pad_D;
            lbp.rank = gg.eff_rk;
            lbp.pad_rank = gg.pra_k;
            lbp.n_layers = (uint32_t)g.layers.size();
            lbp.max_feature_dim = gg.max_feature_dim;
            lbp.landmark_type = (uint32_t)g.landmark_type;
            lbp.a_type = (uint32_t)g.codec_a_k;
            lbp.b_type = (uint32_t)g.codec_b_k;
            lbp.seed = (uint32_t)g.landmark_seed;
            lbp.phase_lo = (uint32_t)g.phase_tx_fingerprint;
            lbp.phase_hi = (uint32_t)(g.phase_tx_fingerprint >> 32);
            size_t lm_need = 0;
            if (!ggml_xkv_landmark_build_scratch_bytes(&lbp, &lm_need, e0, sizeof(e0))) {
                seal_err(err, std::string("estimate: landmark scratch error: ") + e0);
                return false;
            }
            if (!ck_add_u64(grp_transient, lm_need, grp_transient)) {
                seal_err(err, "estimate: landmark workspace overflow");
                return false;
            }
        }
        if (grp_transient > max_transient) {
            max_transient = grp_transient;
        }
    }

    size_t persistent_dest_bytes = 0;
    if (!native_persistent_alloc_bytes(config, geoms, persistent_dest_bytes, err) ||
        (uint64_t)persistent_dest_bytes < raw_stream_bytes) {
        if (err && err->empty()) *err = "estimate: allocator size below encoded stream bytes";
        return false;
    }
    size_t aux_bytes = 0;
    if (!native_aux_alloc_bytes(config, geoms, aux_bytes, err) ||
        !ck_add_u64(max_transient, aux_bytes, max_transient)) {
        if (err && err->empty()) *err = "estimate: telemetry workspace overflow";
        return false;
    }
    uint64_t combined = 0;
    if (max_transient > SIZE_MAX ||
        !ck_add_u64(persistent_dest_bytes, max_transient, combined) || combined > SIZE_MAX) {
        seal_err(err, "estimate: result exceeds size_t");
        return false;
    }
    out.persistent_dest_bytes = persistent_dest_bytes;
    out.max_transient_bytes = (size_t)max_transient;
    out.combined_peak_bytes = (size_t)combined;
    return true;
}

bool xkv_native_seal_build(
    const xkv_native_seal_config & config,
    xkv_native_seal_bundle & out,
    std::string * err) {
    // ---- 0. config validation ----
    if (!config.executor || !config.buft || !config.id_gen) {
        seal_err(err, "seal: null executor/buft/id_gen");
        return false;
    }
    ggml_backend_t backend = config.executor.get();
    if (!backend) {
        seal_err(err, "seal: executor holds null backend");
        return false;
    }
    if (config.backend && config.backend != backend) {
        seal_err(err, "seal: config.backend does not match config.executor");
        return false;
    }
    if (config.n_rows == 0 || config.groups.empty()) {
        seal_err(err, "seal: empty rows or groups");
        return false;
    }
    if (!(config.max_rel_error > 0.0) || !std::isfinite(config.max_rel_error)) {
        seal_err(err, "seal: max_rel_error must be positive and finite");
        return false;
    }
    if (config.physical_rows.size() != config.n_rows ||
        config.storage_positions.size() != config.n_rows) {
        seal_err(err, "seal: rows/positions length mismatch");
        return false;
    }
    for (uint32_t i = 0; i < config.n_rows; ++i) {
        if (config.physical_rows[i] < 0) {
            seal_err(err, "seal: negative physical row");
            return false;
        }
    }
    if (!config.placement_identity.empty()) {
        const char * name = ggml_backend_name(backend);
        if (!name || config.placement_identity != name) {
            seal_err(err, "seal: placement identity mismatch (split/cross-device refused)");
            return false;
        }
    }

    // Estimation and execution share exactly one checked geometry derivation.
    size_t worst_group_idx = 0;
    uint64_t worst_group_metric = 0;
    std::vector<native_group_geometry> geoms;
    try {
        geoms.resize(config.groups.size());
    } catch (const std::exception &) {
        seal_err(err, "seal: geometry allocation failed");
        return false;
    }
    for (size_t gi = 0; gi < config.groups.size(); ++gi) {
        const auto & g = config.groups[gi];
        auto & gg = geoms[gi];
        if (!native_group_geometry_checked(config, g, gg, "seal", err)) return false;
        uint64_t width = 0;
        uint64_t metric = 0;
        if (!ck_add_u64(gg.mk, gg.mv, width) || !ck_mul_u64(width, config.n_rows, metric)) {
            seal_err(err, "seal: group workspace metric overflow");
            return false;
        }
        if (metric > worst_group_metric) {
            worst_group_metric = metric;
            worst_group_idx = gi;
        }
    }

    uint32_t n = config.n_rows;

    // ---- 1. Persistent stream-only context ----
    // Telemetry and source metadata live in the reusable transient arena;
    // retaining them beside code streams would inflate steady-state VRAM.
    uint64_t n_out_tensors = 0;
    uint64_t out_ctx_mem_u64 = 0;
    if (!ck_mul_u64(config.groups.size(), 5, n_out_tensors) ||
        !ck_add_u64(n_out_tensors, 32, n_out_tensors) ||
        !ck_mul_u64(n_out_tensors, ggml_tensor_overhead(), out_ctx_mem_u64) ||
        !ck_add_u64(out_ctx_mem_u64, 65536, out_ctx_mem_u64) || out_ctx_mem_u64 > SIZE_MAX) {
        seal_err(err, "seal: persistent context metadata overflow");
        return false;
    }
    const size_t out_ctx_mem = (size_t)out_ctx_mem_u64;
    std::vector<uint8_t> out_ctx_buf;
    try {
        out_ctx_buf.resize(out_ctx_mem);
    } catch (const std::exception &) {
        seal_err(err, "seal: persistent context metadata allocation failed");
        return false;
    }
    ggml_init_params oip = {out_ctx_mem, out_ctx_buf.data(), true};
    ggml_context * ctx_out = ggml_init(oip);
    if (!ctx_out) { seal_err(err, "seal: ctx_out init failed"); return false; }

    struct ctx_deleter { void operator()(ggml_context * c) const { if (c) ggml_free(c); } };
    std::unique_ptr<ggml_context, ctx_deleter> ctx_out_hold(ctx_out);

    struct persistent_group {
        ggml_tensor *a_k = nullptr, *b_k = nullptr;
        ggml_tensor *a_v = nullptr, *b_v = nullptr;
        ggml_tensor *l_dst = nullptr;
    };
    std::vector<persistent_group> pg(config.groups.size());

    for (size_t gi = 0; gi < config.groups.size(); ++gi) {
        const auto & g = config.groups[gi];
        const auto & gg = geoms[gi];
        auto & P = pg[gi];
        P.a_k = ggml_new_tensor_2d(ctx_out, g.codec_a_k, gg.pra_k, n);
        P.b_k = ggml_new_tensor_2d(ctx_out, g.codec_b_k, gg.prb_k, (int64_t)gg.mk);
        P.a_v = ggml_new_tensor_2d(ctx_out, g.codec_a_v, gg.pra_v, n);
        P.b_v = ggml_new_tensor_2d(ctx_out, g.codec_b_v, gg.prb_v, (int64_t)gg.mv);
        if (!P.a_k || !P.b_k || !P.a_v || !P.b_v) {
            seal_err(err, "seal: persistent stream tensor allocation failed");
            return false;
        }
        if (g.want_landmarks) {
            P.l_dst = ggml_new_tensor_2d(ctx_out, g.landmark_type, gg.pad_D, gg.n_chunks);
            if (!P.l_dst) {
                seal_err(err, "seal: persistent landmark tensor allocation failed");
                return false;
            }
        }
    }

    size_t dest_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_out, config.buft);
    if (dest_bytes == 0) {
        seal_err(err, "seal: dest buft size probe failed");
        return false;
    }

    // ---- 2. Build Transient Graphs (per group) & Measure Max Scratch ----
    // Build transient graph helper
    struct transient_group {
        std::vector<uint8_t> ctx_mem;
        std::unique_ptr<ggml_context, ctx_deleter> ctx;
        ggml_cgraph * graph = nullptr;
        ggml_tensor *rows_t = nullptr, *pos_t = nullptr, *empty_f32 = nullptr;
        ggml_tensor *scratch_k = nullptr, *scratch_v = nullptr;
        ggml_tensor *st_fk = nullptr, *st_fv = nullptr;
        ggml_tensor *l_meta = nullptr, *l_rope = nullptr, *l_scratch = nullptr;
        ggml_tensor *l_eb = nullptr, *l_srcfp = nullptr, *l_status = nullptr;
        std::vector<ggml_tensor *> had_k, had_v, rope_k, st_ck, st_cv;
        std::vector<int32_t> lmeta_data;
        std::vector<float> lrope_data;
        std::vector<int32_t> host_st_fk, host_st_fv, host_st_ck, host_st_cv;
        std::vector<float> host_singular_k, host_singular_v, host_l_eb;
        std::vector<int64_t> host_l_srcfp;
        std::vector<int32_t> host_l_status;
    };
    std::vector<transient_group> tg(config.groups.size());

    uint64_t aux_tensor_count = 0;
    for (const auto & group : config.groups) {
        uint64_t group_aux = 2 + (group.want_landmarks ? 3 : 0);
        uint64_t layer_aux = 0;
        if (!ck_mul_u64(group.layers.size(), 2, layer_aux) ||
            !ck_add_u64(group_aux, layer_aux, group_aux) ||
            !ck_add_u64(aux_tensor_count, group_aux, aux_tensor_count)) {
            seal_err(err, "seal: telemetry tensor count overflow");
            return false;
        }
    }
    uint64_t aux_ctx_mem_u64 = 0;
    if (!ck_add_u64(aux_tensor_count, 32, aux_tensor_count) ||
        !ck_mul_u64(aux_tensor_count, ggml_tensor_overhead(), aux_ctx_mem_u64) ||
        !ck_add_u64(aux_ctx_mem_u64, 65536, aux_ctx_mem_u64) || aux_ctx_mem_u64 > SIZE_MAX) {
        seal_err(err, "seal: telemetry context metadata overflow");
        return false;
    }
    std::vector<uint8_t> aux_ctx_buf;
    try {
        aux_ctx_buf.resize((size_t)aux_ctx_mem_u64);
    } catch (const std::exception &) {
        seal_err(err, "seal: telemetry context metadata allocation failed");
        return false;
    }
    ggml_init_params aip = {(size_t)aux_ctx_mem_u64, aux_ctx_buf.data(), true};
    std::unique_ptr<ggml_context, seal_ctx_deleter> ctx_aux(ggml_init(aip));
    if (!ctx_aux) {
        seal_err(err, "seal: telemetry context init failed");
        return false;
    }
    for (size_t gi = 0; gi < config.groups.size(); ++gi) {
        auto & T = tg[gi];
        const auto & group = config.groups[gi];
        const auto & gg = geoms[gi];
        T.st_fk = ggml_new_tensor_1d(ctx_aux.get(), GGML_TYPE_I32, 10);
        T.st_fv = ggml_new_tensor_1d(ctx_aux.get(), GGML_TYPE_I32, 10);
        if (!T.st_fk || !T.st_fv) {
            seal_err(err, "seal: telemetry status tensor allocation failed");
            return false;
        }
        T.st_ck.resize(group.layers.size());
        T.st_cv.resize(group.layers.size());
        for (size_t li = 0; li < group.layers.size(); ++li) {
            T.st_ck[li] = ggml_new_tensor_1d(ctx_aux.get(), GGML_TYPE_I32, 1);
            T.st_cv[li] = ggml_new_tensor_1d(ctx_aux.get(), GGML_TYPE_I32, 1);
            if (!T.st_ck[li] || !T.st_cv[li]) {
                seal_err(err, "seal: telemetry layer status allocation failed");
                return false;
            }
        }
        if (group.want_landmarks) {
            T.l_eb = ggml_new_tensor_1d(ctx_aux.get(), GGML_TYPE_F32, gg.n_chunks);
            T.l_srcfp = ggml_new_tensor_1d(ctx_aux.get(), GGML_TYPE_I64, gg.n_chunks);
            T.l_status = ggml_new_tensor_1d(ctx_aux.get(), GGML_TYPE_I32, 4);
            if (!T.l_eb || !T.l_srcfp || !T.l_status) {
                seal_err(err, "seal: telemetry landmark tensor allocation failed");
                return false;
            }
        }
    }
    const size_t aux_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_aux.get(), config.buft);
    if (aux_bytes == 0) {
        seal_err(err, "seal: telemetry allocator size probe failed");
        return false;
    }

    auto build_group_graph = [&](size_t gi) -> bool {
        const auto & g = config.groups[gi];
        const auto & gg = geoms[gi];
        auto & T = tg[gi];
        auto & P = pg[gi];
        uint64_t graph_cap_u64 = 0;
        uint64_t mem_u64 = 0;
        if (!ck_mul_u64(g.layers.size(), 32, graph_cap_u64) ||
            !ck_add_u64(graph_cap_u64, 256, graph_cap_u64) || graph_cap_u64 > SIZE_MAX ||
            !ck_mul_u64(graph_cap_u64, ggml_tensor_overhead(), mem_u64) ||
            !ck_add_u64(mem_u64, ggml_graph_overhead_custom((size_t)graph_cap_u64, false), mem_u64) ||
            !ck_add_u64(mem_u64, 65536, mem_u64) || mem_u64 > SIZE_MAX) {
            seal_err(err, "seal: transient graph metadata overflow");
            return false;
        }
        const size_t graph_cap = (size_t)graph_cap_u64;
        const size_t mem = (size_t)mem_u64;
        try {
            T.ctx_mem.resize(mem);
            T.host_st_fk.resize(10);
            T.host_st_fv.resize(10);
            T.host_st_ck.resize(g.layers.size());
            T.host_st_cv.resize(g.layers.size());
            T.host_singular_k.resize(gg.eff_rk);
            T.host_singular_v.resize(gg.eff_rv);
            if (g.want_landmarks) {
                T.host_l_eb.resize(gg.n_chunks);
                T.host_l_srcfp.resize(gg.n_chunks);
                T.host_l_status.resize(4);
            }
        } catch (const std::exception &) {
            seal_err(err, "seal: transient host metadata allocation failed");
            return false;
        }
        ggml_init_params tip = {mem, T.ctx_mem.data(), true};
        T.ctx.reset(ggml_init(tip));
        if (!T.ctx) return false;

        ggml_context * c = T.ctx.get();
        T.rows_t = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
        T.pos_t = ggml_new_tensor_1d(c, GGML_TYPE_I64, n);
        T.empty_f32 = ggml_new_tensor_1d(c, GGML_TYPE_F32, 0);
        ggml_tensor * Xk = ggml_new_tensor_2d(c, GGML_TYPE_F32, (int64_t)gg.mk, n);
        ggml_tensor * Xv = ggml_new_tensor_2d(c, GGML_TYPE_F32, (int64_t)gg.mv, n);
        if (!T.rows_t || !T.pos_t || !T.empty_f32 || !Xk || !Xv) {
            seal_err(err, "seal: transient staging tensor allocation failed");
            return false;
        }

        ggml_xkv_factorize_params fpk = {}, fpv = {};
        fpk.version = GGML_XKV_FACTOR_VERSION;
        fpk.rows_n = n; fpk.cols_m = (uint32_t)gg.mk; fpk.requested_rank = gg.eff_rk;
        fpk.oversampling = g.oversampling; fpk.power_iterations = g.power_iterations;
        fpk.balance_mode = g.balance_mode;
        fpk.type_a = (uint32_t)g.codec_a_k; fpk.type_b = (uint32_t)g.codec_b_k;
        fpk.seed_low = (uint32_t)g.seed_k; fpk.seed_high = (uint32_t)(g.seed_k >> 32);
        fpk.pad_r_a = gg.pra_k; fpk.pad_r_b = gg.prb_k;
        fpv = fpk;
        fpv.cols_m = (uint32_t)gg.mv; fpv.requested_rank = gg.eff_rv;
        fpv.type_a = (uint32_t)g.codec_a_v; fpv.type_b = (uint32_t)g.codec_b_v;
        fpv.seed_low = (uint32_t)g.seed_v; fpv.seed_high = (uint32_t)(g.seed_v >> 32);
        fpv.pad_r_a = gg.pra_v; fpv.pad_r_b = gg.prb_v;

        size_t need_k = 0, need_v = 0;
        char e0[256] = {};
        if (!ggml_xkv_factorize_scratch_bytes(&fpk, &need_k, e0, sizeof(e0)) ||
            !ggml_xkv_factorize_scratch_bytes(&fpv, &need_v, e0, sizeof(e0))) {
            seal_err(err, std::string("seal: factorize scratch query failed: ") + e0);
            return false;
        }
        if ((need_k % 4) != 0 || (need_v % 4) != 0 || need_k > (uint64_t)INT64_MAX * 4u ||
            need_v > (uint64_t)INT64_MAX * 4u) {
            seal_err(err, "seal: factorize scratch size misaligned or overflowed");
            return false;
        }
        T.scratch_k = ggml_new_tensor_1d(c, GGML_TYPE_F32, (int64_t)(need_k / 4));
        T.scratch_v = ggml_new_tensor_1d(c, GGML_TYPE_F32, (int64_t)(need_v / 4));
        if (!T.scratch_k || !T.scratch_v) {
            seal_err(err, "seal: factorize scratch tensor allocation failed");
            return false;
        }

        T.graph = ggml_new_graph_custom(c, graph_cap, false);
        if (!T.graph) return false;

        T.had_k.resize(g.layers.size()); T.had_v.resize(g.layers.size());
        T.rope_k.resize(g.layers.size());

        for (size_t li = 0; li < g.layers.size(); ++li) {
            const auto & L = g.layers[li];
            uint32_t wk = L.n_heads * L.head_dim_k;
            uint32_t wv = L.n_heads * L.head_dim_v;
            size_t off_k = (size_t)gg.feat_off_k[li] * sizeof(float);
            size_t off_v = (size_t)gg.feat_off_v[li] * sizeof(float);
            ggml_tensor * view_k = ggml_view_2d(c, Xk, wk, n, Xk->nb[1], off_k);
            ggml_tensor * view_v = ggml_view_2d(c, Xv, wv, n, Xv->nb[1], off_v);
            if (!view_k || !view_v) {
                seal_err(err, "seal: group column view allocation failed");
                return false;
            }

            uint32_t fck = L.rotary_dim_k / 2;
            T.rope_k[li] = ggml_new_tensor_1d(c, GGML_TYPE_F32, fck ? (int64_t)fck * 2 : 0);
            uint32_t Hk = gg.had_dim_k[li], Hv = gg.had_dim_v[li];
            T.had_k[li] = Hk ? ggml_new_tensor_2d(c, GGML_TYPE_F32, Hk, Hk) : T.empty_f32;
            T.had_v[li] = Hv ? ggml_new_tensor_2d(c, GGML_TYPE_F32, Hv, Hv) : T.empty_f32;
            if (!T.rope_k[li] || !T.had_k[li] || !T.had_v[li]) {
                seal_err(err, "seal: rope/hadamard tensor allocation failed");
                return false;
            }

            ggml_xkv_canonicalize_params cpk = {}, cpv = {};
            cpk.version = GGML_XKV_FACTOR_VERSION;
            cpk.n_rows = n; cpk.n_layers = 1; cpk.n_heads = L.n_heads;
            cpk.head_dim = L.head_dim_k; cpk.padded_head_dim = gg.padded_k[li];
            cpk.total_feat = wk;
            cpk.rotary_dim = L.rotary_dim_k; cpk.rope_mode = (uint32_t)L.rope_mode_k;
            cpk.input_type = (uint32_t)L.hot_k->type; cpk.is_k = 1;
            cpk.hadamard_dim = Hk;
            cpv = cpk;
            cpv.head_dim = L.head_dim_v; cpv.padded_head_dim = gg.padded_v[li];
            cpv.total_feat = wv;
            cpv.rotary_dim = 0; cpv.rope_mode = (uint32_t)GGML_XKV_ROPE_HALF;
            cpv.input_type = (uint32_t)L.hot_v->type; cpv.is_k = 0;
            cpv.hadamard_dim = Hv;

            ggml_tensor * ck = ggml_xkv_canonicalize(c, L.hot_k, T.rows_t, T.pos_t,
                T.rope_k[li], T.had_k[li], T.st_ck[li], &cpk);
            ggml_tensor * cv = ggml_xkv_canonicalize(c, L.hot_v, T.rows_t, T.pos_t,
                T.empty_f32, T.had_v[li], T.st_cv[li], &cpv);
            ggml_tensor * mk = ggml_cpy(c, ck, view_k);
            ggml_tensor * mv = ggml_cpy(c, cv, view_v);
            if (!ck || !cv || !mk || !mv) {
                seal_err(err, "seal: canonicalize/copy node creation failed");
                return false;
            }
            ggml_build_forward_expand(T.graph, mk);
            ggml_build_forward_expand(T.graph, mv);
        }

        ggml_tensor * ak = ggml_xkv_factorize(c, Xk, T.scratch_k, P.b_k, T.st_fk, &fpk);
        ggml_tensor * av = ggml_xkv_factorize(c, Xv, T.scratch_v, P.b_v, T.st_fv, &fpv);
        ggml_tensor * cpy_ak = ggml_cpy(c, ak, P.a_k);
        ggml_tensor * cpy_av = ggml_cpy(c, av, P.a_v);
        if (!ak || !av || !cpy_ak || !cpy_av) {
            seal_err(err, "seal: factorize/copy node creation failed");
            return false;
        }
        ggml_build_forward_expand(T.graph, cpy_ak);
        ggml_build_forward_expand(T.graph, cpy_av);

        if (g.want_landmarks) {
            ggml_xkv_landmark_build_params lbp = {};
            lbp.version = GGML_XKV_LANDMARK_BUILD_VERSION;
            lbp.n_rows = n; lbp.n_chunks = gg.n_chunks; lbp.chunk_tokens = g.chunk_tokens;
            lbp.total_dim = (uint32_t)gg.mk; lbp.padded_dim = gg.pad_D;
            lbp.rank = gg.eff_rk; lbp.pad_rank = gg.pra_k;
            lbp.n_layers = (uint32_t)g.layers.size(); lbp.max_feature_dim = gg.max_feature_dim;
            lbp.landmark_type = (uint32_t)g.landmark_type;
            lbp.a_type = (uint32_t)g.codec_a_k; lbp.b_type = (uint32_t)g.codec_b_k;
            lbp.seed = (uint32_t)g.landmark_seed;
            lbp.phase_lo = (uint32_t)g.phase_tx_fingerprint;
            lbp.phase_hi = (uint32_t)(g.phase_tx_fingerprint >> 32);

            size_t lm_need = 0;
            char lmerr[256] = {};
            if (!ggml_xkv_landmark_build_scratch_bytes(&lbp, &lm_need, lmerr, sizeof(lmerr))) {
                seal_err(err, std::string("seal: landmark scratch query failed: ") + lmerr);
                return false;
            }
            if ((lm_need % 4) != 0 || lm_need > (uint64_t)INT64_MAX * 4u) {
                seal_err(err, "seal: landmark scratch size misaligned or overflowed");
                return false;
            }
            T.l_meta = ggml_new_tensor_2d(c, GGML_TYPE_I32, GGML_XKV_LANDMARK_BUILD_LAYER_META_STRIDE, g.layers.size());
            size_t rope_total_floats = 0;
            for (const auto & L : g.layers) {
                uint32_t fc = L.rotary_dim_k / 2;
                if (fc > 0) rope_total_floats += size_t(fc) * 2;
            }
            T.l_rope = ggml_new_tensor_1d(c, GGML_TYPE_F32, (int64_t)rope_total_floats);
            T.l_scratch = ggml_new_tensor_1d(c, GGML_TYPE_F32, (int64_t)(lm_need / 4));
            if (!T.l_meta || !T.l_rope || !T.l_scratch) {
                seal_err(err, "seal: landmark transient tensor allocation failed");
                return false;
            }

            ggml_tensor * ldst = ggml_xkv_landmark_build(c, ak, P.b_k, T.rows_t, T.pos_t,
                T.l_meta, T.l_rope, T.l_scratch, T.l_eb, T.l_srcfp, T.l_status, &lbp);
            ggml_tensor * cpy_lm = ggml_cpy(c, ldst, P.l_dst);
            if (!ldst || !cpy_lm) {
                seal_err(err, "seal: landmark build/copy node creation failed");
                return false;
            }
            ggml_build_forward_expand(T.graph, cpy_lm);

            T.lmeta_data.resize(g.layers.size() * GGML_XKV_LANDMARK_BUILD_LAYER_META_STRIDE);
            uint32_t cur_roff = 0;
            for (size_t li = 0; li < g.layers.size(); ++li) {
                const auto & L = g.layers[li];
                uint32_t fd = L.n_heads * L.head_dim_k;
                uint32_t fc = L.rotary_dim_k / 2;
                T.lmeta_data[li * 6 + 0] = (int32_t)gg.feat_off_k[li];
                T.lmeta_data[li * 6 + 1] = (int32_t)fd;
                T.lmeta_data[li * 6 + 2] = (int32_t)L.head_dim_k;
                T.lmeta_data[li * 6 + 3] = (int32_t)L.rotary_dim_k;
                T.lmeta_data[li * 6 + 4] = (int32_t)L.rope_mode_k;
                T.lmeta_data[li * 6 + 5] = fc > 0 ? (int32_t)cur_roff : 0;
                if (fc > 0) {
                    for (uint32_t f = 0; f < fc; ++f) T.lrope_data.push_back(L.rope_omega_k[f]);
                    for (uint32_t f = 0; f < fc; ++f) T.lrope_data.push_back(L.rope_mag_k[f]);
                    cur_roff += fc * 2;
                }
            }
        }
        return true;
    };

    for (size_t gi = 0; gi < config.groups.size(); ++gi) {
        if (!build_group_graph(gi)) {
            seal_err(err, "seal: group graph build failed");
            return false;
        }
    }

    // `dest_bytes` is the backend allocator's exact size for the stream-only
    // context, including alignment. Gate it before allocating anything.
    const xkv_backend_store_reservation * store_res = config.store_reservation;
    if (store_res &&
        ((uint64_t)dest_bytes > store_res->reserved_bytes ||
         (store_res->cap_bytes > 0 && (uint64_t)dest_bytes > store_res->cap_bytes))) {
        seal_err(err, "seal: persistent destination exceeds store reservation");
        return false;
    }
    struct buf_deleter { void operator()(ggml_backend_buffer_t b) const { if (b) ggml_backend_buffer_free(b); } };
    std::unique_ptr<std::remove_pointer<ggml_backend_buffer_t>::type, buf_deleter> buf_out(
        ggml_backend_alloc_ctx_tensors_from_buft(ctx_out, config.buft));
    if (!buf_out || ggml_backend_buffer_get_size(buf_out.get()) != dest_bytes) {
        seal_err(err, "seal: persistent output buffer allocation/size mismatch");
        return false;
    }
    std::unique_ptr<std::remove_pointer<ggml_backend_buffer_t>::type, buf_deleter> buf_aux(
        ggml_backend_alloc_ctx_tensors_from_buft(ctx_aux.get(), config.buft));
    if (!buf_aux || ggml_backend_buffer_get_size(buf_aux.get()) != aux_bytes) {
        seal_err(err, "seal: telemetry buffer allocation/size mismatch");
        return false;
    }

    // Measure max transient workspace using worst-case group graph
    struct galloc_deleter { void operator()(ggml_gallocr_t g) const { if (g) ggml_gallocr_free(g); } };
    std::unique_ptr<std::remove_pointer<ggml_gallocr_t>::type, galloc_deleter> galloc(ggml_gallocr_new(config.buft));
    if (!galloc) { seal_err(err, "seal: gallocr new failed"); return false; }

    // Measure every group graph and reserve the true maximum: the (mk+mv)*n
    // heuristic cannot rank landmark/scratch-heavy groups, and reserving a
    // non-maximum graph would force a mid-flight gallocr reallocation.
    size_t measured_max = 0;
    worst_group_idx = 0;
    for (size_t gi = 0; gi < config.groups.size(); ++gi) {
        size_t measured_one[1] = {0};
        ggml_gallocr_reserve_n_size(galloc.get(), tg[gi].graph, nullptr, nullptr, measured_one);
        if (measured_one[0] == 0) {
            seal_err(err, "seal: gallocr measured zero transient bytes for group " + std::to_string(gi));
            return false;
        }
        if (measured_one[0] > measured_max) {
            measured_max = measured_one[0];
            worst_group_idx = gi;
        }
    }
    if (measured_max == 0 || aux_bytes > SIZE_MAX - measured_max) {
        seal_err(err, "seal: gallocr measured zero transient bytes");
        return false;
    }
    const size_t max_transient_bytes = measured_max + aux_bytes;

    // Combined peak = persistent outputs + max transient
    uint64_t combined_peak = 0;
    if (!ck_add_u64(dest_bytes, max_transient_bytes, combined_peak)) {
        seal_err(err, "seal: combined peak overflow");
        return false;
    }

    // Staging reservation gates the one reusable transient workspace. The
    // persistent reservation was checked against `dest_bytes` above.
    if (config.staging_reservation) {
        if (max_transient_bytes > config.staging_reservation->reserved_bytes()) {
            seal_err(err, "seal: staging reservation short for max transient workspace (T-1 refused)");
            return false;
        }
    }

    // Allocate transient workspace buffer once (sized for worst-case group)
    if (!ggml_gallocr_reserve(galloc.get(), tg[worst_group_idx].graph)) {
        seal_err(err, "seal: gallocr reserve failed");
        return false;
    }

    // ---- 3. Ordered Async Per-Group Execution (Single Sync) ----
    bool backend_work_pending = false;
    struct sync_on_failure {
        ggml_backend_t backend;
        bool * pending;
        ~sync_on_failure() { if (backend && pending && *pending) ggml_backend_synchronize(backend); }
    } sync_guard{backend, &backend_work_pending};
    for (size_t gi = 0; gi < config.groups.size(); ++gi) {
        const auto & g = config.groups[gi];
        auto & T = tg[gi];

        // Allocate transient nodes for this group using the reused gallocr buffer!
        if (!ggml_gallocr_alloc_graph(galloc.get(), T.graph)) {
            seal_err(err, "seal: gallocr alloc failed for group " + std::to_string(gi));
            return false;
        }

        // Upload source row metadata and group-specific transforms into the
        // reused arena after this graph has been bound to it.
        backend_work_pending = true;
        ggml_backend_tensor_set_async(backend, T.rows_t, config.physical_rows.data(), 0, size_t(n) * sizeof(int32_t));
        ggml_backend_tensor_set_async(backend, T.pos_t, config.storage_positions.data(), 0, size_t(n) * sizeof(int64_t));
        for (size_t li = 0; li < g.layers.size(); ++li) {
            const auto & L = g.layers[li];
            uint32_t fck = L.rotary_dim_k / 2;
            if (fck > 0) {
                std::vector<float> rm(size_t(fck) * 2);
                for (uint32_t f = 0; f < fck; ++f) { rm[f] = L.rope_omega_k[f]; rm[fck + f] = L.rope_mag_k[f]; }
                ggml_backend_tensor_set_async(backend, T.rope_k[li], rm.data(), 0, rm.size() * 4);
            }
            if (!L.hadamard_k.empty()) {
                ggml_backend_tensor_set_async(backend, T.had_k[li], L.hadamard_k.data(), 0, L.hadamard_k.size() * 4);
            }
            if (!L.hadamard_v.empty()) {
                ggml_backend_tensor_set_async(backend, T.had_v[li], L.hadamard_v.data(), 0, L.hadamard_v.size() * 4);
            }
        }
        if (g.want_landmarks) {
            ggml_backend_tensor_set_async(backend, T.l_meta, T.lmeta_data.data(), 0, T.lmeta_data.size() * 4);
            if (!T.lrope_data.empty()) {
                ggml_backend_tensor_set_async(backend, T.l_rope, T.lrope_data.data(), 0, T.lrope_data.size() * 4);
            }
        }

        // Compute graph for this group asynchronously (queues into backend command stream)
        if (ggml_backend_graph_compute(backend, T.graph) != GGML_STATUS_SUCCESS) {
            seal_err(err, "seal: graph compute failed for group " + std::to_string(gi));
            return false;
        }

        // Queue telemetry copies before the same arena is rebound for the
        // next group. Queue order preserves each group's values; one final
        // synchronize completes all graph work and all D2H copies.
        ggml_backend_tensor_get_async(backend, T.st_fk, T.host_st_fk.data(), 0, 10 * sizeof(int32_t));
        ggml_backend_tensor_get_async(backend, T.st_fv, T.host_st_fv.data(), 0, 10 * sizeof(int32_t));
        for (size_t li = 0; li < g.layers.size(); ++li) {
            ggml_backend_tensor_get_async(backend, T.st_ck[li], &T.host_st_ck[li], 0, sizeof(int32_t));
            ggml_backend_tensor_get_async(backend, T.st_cv[li], &T.host_st_cv[li], 0, sizeof(int32_t));
        }
        uint64_t singular_off_k = 0;
        uint64_t singular_total_k = 0;
        uint64_t singular_off_v = 0;
        uint64_t singular_total_v = 0;
        if (geoms[gi].lk == 0 || geoms[gi].lv == 0 ||
            !seal_scratch_offsets(n, geoms[gi].mk, geoms[gi].eff_rk,
                geoms[gi].lk, singular_off_k, singular_total_k) ||
            !seal_scratch_offsets(n, geoms[gi].mv, geoms[gi].eff_rv,
                geoms[gi].lv, singular_off_v, singular_total_v) ||
            singular_off_k > SIZE_MAX / sizeof(float) || singular_off_v > SIZE_MAX / sizeof(float) ||
            singular_total_k > (uint64_t)ggml_nelements(T.scratch_k) ||
            singular_total_v > (uint64_t)ggml_nelements(T.scratch_v)) {
            seal_err(err, "seal: singular-value readback offset overflow");
            return false;
        }
        ggml_backend_tensor_get_async(backend, T.scratch_k, T.host_singular_k.data(),
            (size_t)singular_off_k * sizeof(float), T.host_singular_k.size() * sizeof(float));
        ggml_backend_tensor_get_async(backend, T.scratch_v, T.host_singular_v.data(),
            (size_t)singular_off_v * sizeof(float), T.host_singular_v.size() * sizeof(float));
        if (g.want_landmarks) {
            ggml_backend_tensor_get_async(backend, T.l_eb, T.host_l_eb.data(), 0,
                T.host_l_eb.size() * sizeof(float));
            ggml_backend_tensor_get_async(backend, T.l_srcfp, T.host_l_srcfp.data(), 0,
                T.host_l_srcfp.size() * sizeof(int64_t));
            ggml_backend_tensor_get_async(backend, T.l_status, T.host_l_status.data(), 0,
                T.host_l_status.size() * sizeof(int32_t));
        }
    }

    // THE one single synchronization for the entire multi-group bundle!
    ggml_backend_synchronize(backend);
    backend_work_pending = false;
    uint64_t sync_count = 1;

    // ---- 4. Check Statuses and Extract Telemetry ----
    xkv_native_seal_bundle bundle;
    bundle.groups.resize(config.groups.size());
    std::vector<xkv_backend_device_stream> astreams;

    struct pending_stream {
        size_t gi = 0;
        int which = 0; // 0=a_k, 1=b_k, 2=a_v, 3=b_v, 4=lm
        codec_desc desc;
        ggml_tensor * tensor = nullptr;
        size_t exact = 0;
    };
    std::vector<pending_stream> pending;

    struct resid_words { float fo = 0, fe = 0, mx = 0; };
    auto read_resid = [&](const std::vector<int32_t> & words, int base, resid_words & r) {
        memcpy(&r.fo, &words[(size_t)base + 0], sizeof(float));
        memcpy(&r.fe, &words[(size_t)base + 1], sizeof(float));
        memcpy(&r.mx, &words[(size_t)base + 2], sizeof(float));
    };

    for (size_t gi = 0; gi < config.groups.size(); ++gi) {
        const auto & g = config.groups[gi];
        const auto & gg = geoms[gi];
        auto & P = pg[gi];
        auto & T = tg[gi];
        auto & bg = bundle.groups[gi];
        bg.group_index = g.group_index;
        bg.rank_k = gg.eff_rk;
        bg.rank_v = gg.eff_rv;

        const int32_t sk = T.host_st_fk[0];
        const int32_t sv = T.host_st_fv[0];
        if (sk != 0 || sv != 0) {
            seal_err(err, "seal: factorize status nonzero (sk=" + std::to_string(sk) + " sv=" + std::to_string(sv) + ")");
            return false;
        }
        bg.status_fact_k = 0;
        bg.status_fact_v = 0;

        for (size_t li = 0; li < g.layers.size(); ++li) {
            const int32_t ck = T.host_st_ck[li];
            const int32_t cv = T.host_st_cv[li];
            if (ck != 0 || cv != 0) {
                seal_err(err, "seal: canonicalize status nonzero (ck=" + std::to_string(ck) + " cv=" + std::to_string(cv) + " gi=" + std::to_string(gi) + " li=" + std::to_string(li) + ")");
                return false;
            }
        }
        bg.status_canon = 0;

        resid_words ab_k, a_k, b_k, ab_v, a_v, b_v;
        read_resid(T.host_st_fk, 1, ab_k);
        read_resid(T.host_st_fk, 4, a_k);
        read_resid(T.host_st_fk, 7, b_k);
        read_resid(T.host_st_fv, 1, ab_v);
        read_resid(T.host_st_fv, 4, a_v);
        read_resid(T.host_st_fv, 7, b_v);

        auto fill = [&](const resid_words & rw, factor_error_report & rep, const char * what) -> bool {
            if (!std::isfinite(rw.fo) || !std::isfinite(rw.fe) || !std::isfinite(rw.mx) ||
                rw.fo < 0.0f || rw.fe < 0.0f || rw.mx < 0.0f) {
                seal_err(err, std::string("seal: non-finite residual ") + what);
                return false;
            }
            rep.frobenius_norm_original = rw.fo;
            rep.frobenius_norm_error = rw.fe;
            rep.relative_error = (rw.fo > 1e-12f) ? (double)rw.fe / (double)rw.fo : (double)rw.fe;
            rep.max_absolute_error = rw.mx;
            if (!(rep.relative_error <= config.max_rel_error)) {
                seal_err(err, "seal: relative error gate refused (" + std::string(what) + "_rel=" +
                    std::to_string(rep.relative_error) + " max=" + std::to_string(config.max_rel_error) + ")");
                return false;
            }
            return true;
        };
        if (!fill(ab_k, bg.err_k, "K") || !fill(ab_v, bg.err_v, "V")) return false;

        uint32_t grpk = is_turbo(g.codec_a_k) ? 128 : 0;
        uint32_t grpv = is_turbo(g.codec_a_v) ? 128 : 0;
        try {
            codec_desc d_ak = make_codec_desc(factor_role::a_k, g.codec_a_k,
                orientation::token_major, matrix_shape{n, gg.eff_rk}, grpk, g.seed_k);
            codec_desc d_bk = make_codec_desc(factor_role::b_k, g.codec_b_k,
                orientation::feature_major_transposed, matrix_shape{(uint64_t)gg.mk, gg.eff_rk}, grpk, g.seed_k);
            codec_desc d_av = make_codec_desc(factor_role::a_v, g.codec_a_v,
                orientation::token_major, matrix_shape{n, gg.eff_rv}, grpv, g.seed_v);
            codec_desc d_bv = make_codec_desc(factor_role::b_v, g.codec_b_v,
                orientation::feature_major_transposed, matrix_shape{(uint64_t)gg.mv, gg.eff_rv}, grpv, g.seed_v);
            if (!d_ak.validate() || !d_bk.validate() || !d_av.validate() || !d_bv.validate()) {
                seal_err(err, "seal: descriptor validation failed");
                return false;
            }
            bg.a_k.desc = d_ak; bg.b_k.desc = d_bk; bg.a_v.desc = d_av; bg.b_v.desc = d_bv;
        } catch (const std::exception & e) {
            seal_err(err, std::string("seal: descriptor build failed: ") + e.what());
            return false;
        }

        bg.singular_k = T.host_singular_k;
        bg.singular_v = T.host_singular_v;

        uint64_t exact_ak = 0;
        uint64_t exact_bk = 0;
        uint64_t exact_av = 0;
        uint64_t exact_bv = 0;
        if (!checked_matrix_bytes(g.codec_a_k, gg.pra_k, n, exact_ak) ||
            !checked_matrix_bytes(g.codec_b_k, gg.prb_k, gg.mk, exact_bk) ||
            !checked_matrix_bytes(g.codec_a_v, gg.pra_v, n, exact_av) ||
            !checked_matrix_bytes(g.codec_b_v, gg.prb_v, gg.mv, exact_bv)) {
            seal_err(err, "seal: sealed stream byte size overflow");
            return false;
        }
        bg.a_k.exact_bytes = (size_t)exact_ak;
        bg.b_k.exact_bytes = (size_t)exact_bk;
        bg.a_v.exact_bytes = (size_t)exact_av;
        bg.b_v.exact_bytes = (size_t)exact_bv;
        bg.a_k.stream_fingerprint = bg.a_k.desc.fingerprint();
        bg.b_k.stream_fingerprint = bg.b_k.desc.fingerprint();
        bg.a_v.stream_fingerprint = bg.a_v.desc.fingerprint();
        bg.b_v.stream_fingerprint = bg.b_v.desc.fingerprint();

        pending.push_back({gi, 0, bg.a_k.desc, P.a_k, bg.a_k.exact_bytes});
        pending.push_back({gi, 1, bg.b_k.desc, P.b_k, bg.b_k.exact_bytes});
        pending.push_back({gi, 2, bg.a_v.desc, P.a_v, bg.a_v.exact_bytes});
        pending.push_back({gi, 3, bg.b_v.desc, P.b_v, bg.b_v.exact_bytes});

        if (g.want_landmarks) {
            if (T.host_l_status[0] != 0) {
                seal_err(err, "seal: landmark build status nonzero");
                return false;
            }
            if (T.host_l_eb.size() != gg.n_chunks || T.host_l_srcfp.size() != (size_t)gg.n_chunks) {
                seal_err(err, "seal: landmark telemetry size mismatch");
                return false;
            }
            for (uint32_t ci = 0; ci < gg.n_chunks; ++ci) {
                if (!std::isfinite(T.host_l_eb[(size_t)ci]) || !(T.host_l_eb[(size_t)ci] > 0.0f)) {
                    seal_err(err, "seal: landmark error bound non-finite/non-positive");
                    return false;
                }
                if (T.host_l_srcfp[(size_t)ci] == 0) {
                    seal_err(err, "seal: landmark source fingerprint is zero");
                    return false;
                }
            }
            bg.has_landmarks = true;
            bg.status_lmbuild = 0;
            bg.landmark_eb = T.host_l_eb;
            bg.landmark_srcfp.assign(T.host_l_srcfp.begin(), T.host_l_srcfp.end());

            try {
                codec_desc d_lm = make_codec_desc(factor_role::landmark, g.landmark_type,
                    orientation::token_major, matrix_shape{gg.n_chunks, (uint64_t)gg.mk},
                    is_turbo(g.landmark_type) ? 128 : 0, g.landmark_seed);
                if (!d_lm.validate()) {
                    seal_err(err, "seal: landmark descriptor validation failed");
                    return false;
                }
                bg.landmark.desc = d_lm;
                bg.landmark.stream_fingerprint = d_lm.fingerprint();
                uint64_t exact_landmark = 0;
                if (!checked_matrix_bytes(g.landmark_type, gg.pad_D, gg.n_chunks, exact_landmark)) {
                    seal_err(err, "seal: landmark stream byte size overflow");
                    return false;
                }
                bg.landmark.exact_bytes = (size_t)exact_landmark;
                pending.push_back({gi, 4, d_lm, P.l_dst, bg.landmark.exact_bytes});
            } catch (const std::exception & e) {
                seal_err(err, std::string("seal: landmark desc failed: ") + e.what());
                return false;
            }
        }
    }

    // Bundle fingerprint
    {
        uint64_t h = 14695981039346656037ULL;
        auto mix = [&](uint64_t v) {
            for (int i = 0; i < 8; ++i) {
                h ^= (uint8_t)(v >> (i * 8));
                h *= 1099511628211ULL;
            }
        };
        mix(n);
        for (const auto & bg : bundle.groups) {
            mix(bg.group_index);
            mix(bg.a_k.stream_fingerprint); mix(bg.b_k.stream_fingerprint);
            mix(bg.a_v.stream_fingerprint); mix(bg.b_v.stream_fingerprint);
            if (bg.has_landmarks) mix(bg.landmark.stream_fingerprint);
        }
        bundle.bundle_fingerprint = h == 0 ? 1 : h;
    }

    // Transfer the already-populated stream-only allocation into immutable
    // handles. No second device allocation, copy, or synchronization.
    {
        astreams.reserve(pending.size());
        for (const auto & ps : pending) {
            xkv_backend_device_stream ds;
            ds.desc = ps.desc;
            ds.tensor = ps.tensor;
            ds.provenance = bundle.bundle_fingerprint | 1ULL;
            astreams.push_back(ds);
        }
        xkv_backend_batch_config aconfig = {};
        aconfig.residency = config.residency;
        aconfig.enforce_residency = true;
        aconfig.backend_owner = config.executor;
        aconfig.inject_alloc_fail_at = -1;
        aconfig.inject_upload_fail_at = -1;
        aconfig.inject_readback_fail_at = -1;
        aconfig.inject_checksum_fail_at = -1;
        aconfig.inject_post_submit_fail_at = -1;
        aconfig.inject_copy_fail_at = -1;

        std::shared_ptr<xkv_backend_batch_storage> storage;
        try {
            storage = std::make_shared<xkv_backend_batch_storage>();
            storage->ctx = ctx_out_hold.release();
            storage->buffer = buf_out.release();
            storage->buffer_bytes = dest_bytes;
            storage->context_memory = std::move(out_ctx_buf);
            storage->backend_executor = config.executor;
        } catch (const std::exception &) {
            seal_err(err, "seal: transfer storage allocation failed");
            return false;
        }
        xkv_backend_adopt_result ares;
        if (!xkv_backend_take_device_tensors(backend, config.buft, std::move(storage), astreams,
                aconfig, *config.id_gen, ares, err, store_res)) {
            return false;
        }
        if (ares.handles.size() != pending.size()) {
            seal_err(err, "seal: adoption handle count mismatch");
            return false;
        }
        bundle.adopt_sync_count = 0;
        bundle.executor = config.executor;
        // Convert adoption result into a genuine committed/success batch result
        // with the real backend executor owner preserved BEFORE distributing handles:
        bundle.backend_bundle = ares.make_batch_result(backend, config.executor);
        if (!bundle.backend_bundle || !bundle.backend_bundle->is_success() || !bundle.backend_bundle->is_committed()) {
            seal_err(err, "seal: adoption batch result construction failed");
            return false;
        }
        for (size_t i = 0; i < pending.size(); ++i) {
            auto & bg = bundle.groups[pending[i].gi];
            std::shared_ptr<xkv_backend_allocation> h = std::move(ares.handles[i]);
            if (!h) { seal_err(err, "seal: null adopted handle"); return false; }
            switch (pending[i].which) {
                case 0: bg.a_k.handle = std::move(h); break;
                case 1: bg.b_k.handle = std::move(h); break;
                case 2: bg.a_v.handle = std::move(h); break;
                case 3: bg.b_v.handle = std::move(h); break;
                default: bg.landmark.handle = std::move(h); break;
            }
        }
    }

    bundle.factored_bytes = 0;
    for (const auto & bg : bundle.groups) {
        uint64_t b = bundle.factored_bytes;
        if (!ck_add_u64(b, bg.a_k.exact_bytes, b) ||
            !ck_add_u64(b, bg.b_k.exact_bytes, b) ||
            !ck_add_u64(b, bg.a_v.exact_bytes, b) ||
            !ck_add_u64(b, bg.b_v.exact_bytes, b)) {
            seal_err(err, "seal: byte accounting overflow");
            return false;
        }
        if (bg.has_landmarks) {
            if (!ck_add_u64(b, bg.landmark.exact_bytes, b)) {
                seal_err(err, "seal: landmark byte accounting overflow");
                return false;
            }
        }
        bundle.factored_bytes = b;
    }

    bundle.scratch_bytes = max_transient_bytes;
    bundle.preflight_bytes = (size_t)combined_peak;
    bundle.peak_source_bytes = max_transient_bytes;
    bundle.peak_dest_bytes = dest_bytes;
    bundle.sync_count = sync_count;

    out = std::move(bundle);
    return true;
}

bool xkv_native_landmark_rebuild(
    const xkv_native_landmark_rebuild_request & req,
    xkv_native_landmark_rebuild_result & out,
    std::string * err) {
    if (!req.executor || !req.buft || !req.id_gen) {
        seal_err(err, "lm_rebuild: null executor/buft/id_gen");
        return false;
    }
    if (!req.a_k || !req.b_k || !req.a_k->get_tensor() || !req.b_k->get_tensor()) {
        seal_err(err, "lm_rebuild: null a_k or b_k allocation handle");
        return false;
    }
    if (req.surviving_rows.empty()) {
        seal_err(err, "lm_rebuild: surviving_rows empty");
        return false;
    }
    if (req.surviving_rows.size() != req.storage_positions.size()) {
        seal_err(err, "lm_rebuild: rows and positions count mismatch");
        return false;
    }
    if (req.chunk_tokens == 0) {
        seal_err(err, "lm_rebuild: chunk_tokens == 0");
        return false;
    }
    if (req.landmark_type != GGML_TYPE_Q8_0 && req.landmark_type != GGML_TYPE_TURBO4_0) {
        seal_err(err, "lm_rebuild: unsupported landmark type");
        return false;
    }
    if (req.layers.empty()) {
        seal_err(err, "lm_rebuild: layers empty");
        return false;
    }

    ggml_backend_t backend = req.executor.get();
    if (!backend) { seal_err(err, "lm_rebuild: executor holds null backend"); return false; }
    if (req.surviving_rows.size() > UINT32_MAX || req.layers.size() > (size_t)INT64_MAX ||
        req.surviving_rows.size() > (size_t)INT64_MAX) {
        seal_err(err, "lm_rebuild: row/layer count overflow");
        return false;
    }
    const uint32_t n_rows = (uint32_t)req.surviving_rows.size();
    uint64_t chunks_u64 = ((uint64_t)n_rows + req.chunk_tokens - 1u) / req.chunk_tokens;
    if (chunks_u64 == 0 || chunks_u64 > UINT32_MAX) {
        seal_err(err, "lm_rebuild: chunk count overflow");
        return false;
    }
    const uint32_t n_chunks = (uint32_t)chunks_u64;

    uint64_t total_dim_u64 = 0;
    uint64_t max_fd_u64 = 0;
    std::vector<uint32_t> feat_offsets;
    try { feat_offsets.reserve(req.layers.size()); } catch (const std::exception &) {
        seal_err(err, "lm_rebuild: offset allocation failed"); return false;
    }
    for (size_t li = 0; li < req.layers.size(); ++li) {
        const auto & L = req.layers[li];
        if (L.n_heads == 0 || L.head_dim_k == 0) {
            seal_err(err, "lm_rebuild: degenerate layer head geometry");
            return false;
        }
        uint64_t fd = 0;
        if (!ck_mul_u64(L.n_heads, L.head_dim_k, fd) || fd == 0 || fd > UINT32_MAX ||
            total_dim_u64 > UINT32_MAX) {
            seal_err(err, "lm_rebuild: feature width overflow");
            return false;
        }
        feat_offsets.push_back((uint32_t)total_dim_u64);
        if (!ck_add_u64(total_dim_u64, fd, total_dim_u64) || total_dim_u64 == 0 ||
            total_dim_u64 > UINT32_MAX) {
            seal_err(err, "lm_rebuild: total dim overflow");
            return false;
        }
        if (fd > max_fd_u64) max_fd_u64 = fd;
        if ((L.rotary_dim_k & 1u) || L.rotary_dim_k > L.head_dim_k) {
            seal_err(err, "lm_rebuild: bad rotary_dim_k"); return false;
        }
        const uint32_t fc = L.rotary_dim_k / 2;
        if (fc > 0) {
            if (L.rope_omega_k.size() != fc || L.rope_mag_k.size() != fc) {
                seal_err(err, "lm_rebuild: rope table size mismatch"); return false;
            }
            for (uint32_t f = 0; f < fc; ++f) {
                if (!std::isfinite(L.rope_omega_k[f]) || !std::isfinite(L.rope_mag_k[f]) ||
                    !(L.rope_mag_k[f] > 0.0f)) {
                    seal_err(err, "lm_rebuild: rope table non-finite/bad mag"); return false;
                }
            }
            if (L.rope_mode_k != GGML_XKV_ROPE_HALF && L.rope_mode_k != GGML_XKV_ROPE_INTERLEAVED) {
                seal_err(err, "lm_rebuild: bad rope_mode_k"); return false;
            }
        }
    }
    const uint32_t total_dim = (uint32_t)total_dim_u64;
    const uint32_t max_fd = (uint32_t)max_fd_u64;

    ggml_tensor * ta = req.a_k->get_tensor();
    ggml_tensor * tb = req.b_k->get_tensor();
    if (!ta || !tb || !ta->buffer || !tb->buffer) {
        seal_err(err, "lm_rebuild: a_k/b_k tensor has no device buffer"); return false;
    }
    if (ggml_backend_buffer_get_type(ta->buffer) != req.buft ||
        ggml_backend_buffer_get_type(tb->buffer) != req.buft) {
        seal_err(err, "lm_rebuild: a_k/b_k off-placement"); return false;
    }
    if (ta->ne[0] <= 0 || tb->ne[0] <= 0 || ta->ne[0] != tb->ne[0] ||
        ta->ne[0] > UINT32_MAX || tb->ne[1] != (int64_t)total_dim) {
        seal_err(err, "lm_rebuild: a_k/b_k shape mismatch with layer geometry"); return false;
    }
    for (int32_t r : req.surviving_rows) {
        if (r < 0 || (int64_t)r >= ta->ne[1]) {
            seal_err(err, "lm_rebuild: surviving row out of range"); return false;
        }
    }
    if (ta->type != tb->type) { seal_err(err, "lm_rebuild: a_k/b_k type mismatch"); return false; }
    const uint32_t pad_rk = (uint32_t)ta->ne[0];
    uint64_t rk_u64 = req.a_k->get_desc().logical_shape.cols;
    if (rk_u64 == 0 || rk_u64 > pad_rk) rk_u64 = pad_rk;
    if (rk_u64 == 0 || rk_u64 > UINT32_MAX) { seal_err(err, "lm_rebuild: bad rank"); return false; }
    const uint32_t rk = (uint32_t)rk_u64;

    const int32_t pad_d_i32 = ggml_xkv_padded_rank(req.landmark_type, total_dim);
    if (pad_d_i32 <= 0) { seal_err(err, "lm_rebuild: landmark padded width unsupported"); return false; }
    const uint32_t pad_D = (uint32_t)pad_d_i32;
    uint64_t exact_dest = 0;
    if (!checked_matrix_bytes(req.landmark_type, pad_D, n_chunks, exact_dest)) {
        seal_err(err, "lm_rebuild: landmark byte size overflow"); return false;
    }

    ggml_xkv_landmark_build_params lbp = {};
    lbp.version = GGML_XKV_LANDMARK_BUILD_VERSION;
    lbp.n_rows = n_rows;
    lbp.n_chunks = n_chunks;
    lbp.chunk_tokens = req.chunk_tokens;
    lbp.total_dim = total_dim;
    lbp.padded_dim = pad_D;
    lbp.rank = rk;
    lbp.pad_rank = pad_rk;
    if (req.layers.size() > UINT32_MAX) { seal_err(err, "lm_rebuild: layer count overflow"); return false; }
    lbp.n_layers = (uint32_t)req.layers.size();
    lbp.max_feature_dim = max_fd;
    lbp.landmark_type = (uint32_t)req.landmark_type;
    lbp.a_type = (uint32_t)ta->type;
    lbp.b_type = (uint32_t)tb->type;
    lbp.seed = (uint32_t)req.seed;
    lbp.phase_lo = (uint32_t)req.phase_tx_fingerprint;
    lbp.phase_hi = (uint32_t)(req.phase_tx_fingerprint >> 32);

    size_t lm_need = 0;
    char lmerr[256] = {};
    if (!ggml_xkv_landmark_build_scratch_bytes(&lbp, &lm_need, lmerr, sizeof(lmerr))) {
        seal_err(err, std::string("lm_rebuild: scratch estimate failed: ") + lmerr);
        return false;
    }
    if ((lm_need % 4) != 0 || lm_need > (uint64_t)INT64_MAX * 4u) {
        seal_err(err, "lm_rebuild: scratch size misaligned or overflowed"); return false;
    }
    // Full preflight before any allocation: the transient gate precedes the
    // persistent context/buffer below, so short reservations refuse with
    // zero mutation, zero IDs, and `out` untouched.
    if (req.staging_reservation && lm_need > req.staging_reservation->reserved_bytes()) {
        seal_err(err, "lm_rebuild: staging reservation short for transient scratch (T-1 refused)");
        return false;
    }

    // ---- stream-only persistent context: holds ONLY the final landmark stream.
    uint64_t out_ctx_u64 = 0;
    if (!ck_mul_u64(1, ggml_tensor_overhead(), out_ctx_u64) ||
        !ck_add_u64(out_ctx_u64, 65536, out_ctx_u64) || out_ctx_u64 > SIZE_MAX) {
        seal_err(err, "lm_rebuild: persistent context overflow"); return false;
    }
    std::vector<uint8_t> out_ctx_buf;
    try { out_ctx_buf.resize((size_t)out_ctx_u64); } catch (const std::exception &) {
        seal_err(err, "lm_rebuild: persistent context allocation failed"); return false;
    }
    ggml_init_params oip = {(size_t)out_ctx_u64, out_ctx_buf.data(), true};
    ggml_context * ctx_out_raw = ggml_init(oip);
    if (!ctx_out_raw) { seal_err(err, "lm_rebuild: ctx_out init failed"); return false; }
    struct ctx_deleter { void operator()(ggml_context * c) const { if (c) ggml_free(c); } };
    std::unique_ptr<ggml_context, ctx_deleter> ctx_out_hold(ctx_out_raw);
    ggml_tensor * l_dst = ggml_new_tensor_2d(ctx_out_raw, req.landmark_type, pad_D, n_chunks);
    if (!l_dst) { seal_err(err, "lm_rebuild: persistent landmark allocation failed"); return false; }
    const size_t dest_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_out_raw, req.buft);
    if (dest_bytes == 0 || (uint64_t)dest_bytes < exact_dest) {
        seal_err(err, "lm_rebuild: persistent allocator size probe failed"); return false;
    }
    if (req.store_reservation) {
        if ((uint64_t)dest_bytes > req.store_reservation->reserved_bytes) {
            seal_err(err, "lm_rebuild: store reservation short for landmark destination (T-1 refused)");
            return false;
        }
        if (req.store_reservation->cap_bytes > 0 && (uint64_t)dest_bytes > req.store_reservation->cap_bytes) {
            seal_err(err, "lm_rebuild: store cap exceeded by landmark destination");
            return false;
        }
    }
    struct buf_deleter { void operator()(ggml_backend_buffer_t b) const { if (b) ggml_backend_buffer_free(b); } };
    std::unique_ptr<std::remove_pointer<ggml_backend_buffer_t>::type, buf_deleter> buf_out(
        ggml_backend_alloc_ctx_tensors_from_buft(ctx_out_raw, req.buft));
    if (!buf_out || ggml_backend_buffer_get_size(buf_out.get()) != dest_bytes) {
        seal_err(err, "lm_rebuild: persistent buffer allocation/size mismatch"); return false;
    }


    // ---- transient context: inputs, scratch, telemetry, and the transient build node.
    uint64_t tr_ctx_u64 = 0;
    if (!ck_mul_u64(32, ggml_tensor_overhead(), tr_ctx_u64) ||
        !ck_add_u64(tr_ctx_u64, ggml_graph_overhead_custom(32, false), tr_ctx_u64) ||
        !ck_add_u64(tr_ctx_u64, 65536, tr_ctx_u64) || tr_ctx_u64 > SIZE_MAX) {
        seal_err(err, "lm_rebuild: transient context overflow"); return false;
    }
    std::vector<uint8_t> tr_ctx_buf;
    try { tr_ctx_buf.resize((size_t)tr_ctx_u64); } catch (const std::exception &) {
        seal_err(err, "lm_rebuild: transient context allocation failed"); return false;
    }
    ggml_init_params tip = {(size_t)tr_ctx_u64, tr_ctx_buf.data(), true};
    std::unique_ptr<ggml_context, ctx_deleter> ctx_tr(ggml_init(tip));
    if (!ctx_tr) { seal_err(err, "lm_rebuild: ctx_tr init failed"); return false; }
    ggml_context * ctx = ctx_tr.get();
    ggml_tensor * rows_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_rows);
    ggml_tensor * pos_t  = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_rows);
    ggml_tensor * l_meta = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, GGML_XKV_LANDMARK_BUILD_LAYER_META_STRIDE, (int64_t)req.layers.size());
    if (!rows_t || !pos_t || !l_meta) { seal_err(err, "lm_rebuild: transient input allocation failed"); return false; }
    uint64_t rope_floats_u64 = 0;
    for (const auto & L : req.layers) {
        const uint32_t fc = L.rotary_dim_k / 2;
        if (fc > 0) {
            uint64_t add = 0;
            if (!ck_mul_u64(fc, 2, add) || !ck_add_u64(rope_floats_u64, add, rope_floats_u64) ||
                rope_floats_u64 > (uint64_t)INT64_MAX) {
                seal_err(err, "lm_rebuild: rope size overflow"); return false;
            }
        }
    }
    if (rope_floats_u64 > (uint64_t)INT64_MAX) { seal_err(err, "lm_rebuild: rope size overflow"); return false; }
    ggml_tensor * l_rope = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)rope_floats_u64);
    ggml_tensor * l_scratch = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)(lm_need / 4));
    ggml_tensor * l_eb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_chunks);
    ggml_tensor * l_srcfp = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_chunks);
    ggml_tensor * l_status = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    if (!l_rope || !l_scratch || !l_eb || !l_srcfp || !l_status) {
        seal_err(err, "lm_rebuild: transient scratch/telemetry allocation failed"); return false;
    }
    ggml_tensor * tnode = ggml_xkv_landmark_build(ctx, ta, tb, rows_t, pos_t, l_meta, l_rope,
        l_scratch, l_eb, l_srcfp, l_status, &lbp);
    if (!tnode) { seal_err(err, "lm_rebuild: landmark build node creation failed"); return false; }
    ggml_tensor * cpy = ggml_cpy(ctx, tnode, l_dst);
    if (!cpy) { seal_err(err, "lm_rebuild: landmark copy node creation failed"); return false; }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    if (!graph) { seal_err(err, "lm_rebuild: graph allocation failed"); return false; }
    ggml_build_forward_expand(graph, cpy);
    std::unique_ptr<std::remove_pointer<ggml_backend_buffer_t>::type, buf_deleter> buf_tr(
        ggml_backend_alloc_ctx_tensors_from_buft(ctx, req.buft));
    if (!buf_tr) { seal_err(err, "lm_rebuild: transient buffer allocation failed"); return false; }

    std::vector<int32_t> lmeta_data;
    std::vector<float> lrope_data;
    try {
        lmeta_data.resize(req.layers.size() * GGML_XKV_LANDMARK_BUILD_LAYER_META_STRIDE);
    } catch (const std::exception &) { seal_err(err, "lm_rebuild: meta allocation failed"); return false; }
    uint64_t cur_roff_u64 = 0;
    for (size_t li = 0; li < req.layers.size(); ++li) {
        const auto & L = req.layers[li];
        const uint32_t fd = L.n_heads * L.head_dim_k;
        const uint32_t fc = L.rotary_dim_k / 2;
        if (feat_offsets[li] > (uint32_t)INT32_MAX || fd > (uint32_t)INT32_MAX ||
            L.head_dim_k > (uint32_t)INT32_MAX || L.rotary_dim_k > (uint32_t)INT32_MAX ||
            L.rope_mode_k > (uint32_t)INT32_MAX || cur_roff_u64 > (uint64_t)INT32_MAX) {
            seal_err(err, "lm_rebuild: layer meta overflow"); return false;
        }
        lmeta_data[li * 6 + 0] = (int32_t)feat_offsets[li];
        lmeta_data[li * 6 + 1] = (int32_t)fd;
        lmeta_data[li * 6 + 2] = (int32_t)L.head_dim_k;
        lmeta_data[li * 6 + 3] = (int32_t)L.rotary_dim_k;
        lmeta_data[li * 6 + 4] = (int32_t)L.rope_mode_k;
        lmeta_data[li * 6 + 5] = fc > 0 ? (int32_t)cur_roff_u64 : 0;
        if (fc > 0) {
            if (L.rope_omega_k.size() < fc || L.rope_mag_k.size() < fc) {
                seal_err(err, "lm_rebuild: rope table short"); return false;
            }
            for (uint32_t f = 0; f < fc; ++f) lrope_data.push_back(L.rope_omega_k[f]);
            for (uint32_t f = 0; f < fc; ++f) lrope_data.push_back(L.rope_mag_k[f]);
            cur_roff_u64 += (uint64_t)fc * 2u;
        }
    }

    bool work_pending = false;
    struct rebuild_sync_guard {
        ggml_backend_t be; bool * p;
        ~rebuild_sync_guard() { if (be && p && *p) ggml_backend_synchronize(be); }
    } guard{backend, &work_pending};
    work_pending = true;
    ggml_backend_tensor_set_async(backend, rows_t, req.surviving_rows.data(), 0, (size_t)n_rows * 4);
    ggml_backend_tensor_set_async(backend, pos_t, req.storage_positions.data(), 0, (size_t)n_rows * sizeof(int64_t));
    ggml_backend_tensor_set_async(backend, l_meta, lmeta_data.data(), 0, lmeta_data.size() * 4);
    if (!lrope_data.empty()) {
        ggml_backend_tensor_set_async(backend, l_rope, lrope_data.data(), 0, lrope_data.size() * 4);
    }
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        seal_err(err, "lm_rebuild: graph compute failed");
        return false;
    }
    std::vector<float> eb_vals;
    std::vector<int64_t> fp_raw;
    int32_t lmst[4] = {-1, -1, -1, -1};
    try { eb_vals.resize(n_chunks); fp_raw.resize(n_chunks); } catch (const std::exception &) {
        seal_err(err, "lm_rebuild: host telemetry allocation failed"); return false;
    }
    ggml_backend_tensor_get_async(backend, l_eb, eb_vals.data(), 0, (size_t)n_chunks * sizeof(float));
    ggml_backend_tensor_get_async(backend, l_srcfp, fp_raw.data(), 0, (size_t)n_chunks * sizeof(int64_t));
    ggml_backend_tensor_get_async(backend, l_status, lmst, 0, sizeof(lmst));
    ggml_backend_synchronize(backend);
    work_pending = false;
    if (lmst[0] != 0) {
        seal_err(err, "lm_rebuild: device status nonzero: " + std::to_string(lmst[0]));
        return false;
    }

    codec_desc d_lm;
    try {
        d_lm = make_codec_desc(factor_role::landmark, req.landmark_type,
            orientation::token_major, matrix_shape{n_chunks, total_dim},
            is_turbo(req.landmark_type) ? 128 : 0, req.seed);
        if (!d_lm.validate()) { seal_err(err, "lm_rebuild: descriptor invalid"); return false; }
    } catch (const std::exception & e) {
        seal_err(err, std::string("lm_rebuild: descriptor build failed: ") + e.what()); return false;
    }
    for (uint32_t ci = 0; ci < n_chunks; ++ci) {
        if (!std::isfinite(eb_vals[ci]) || !(eb_vals[ci] > 0.0f) || fp_raw[ci] == 0) {
            seal_err(err, "lm_rebuild: landmark telemetry non-finite/non-positive/zero-fp");
            return false;
        }
    }

    // Ownership-transfer the stream-only persistent buffer: zero copies, zero adoption syncs.
    std::vector<xkv_backend_device_stream> astreams(1);
    astreams[0].desc = d_lm;
    astreams[0].tensor = l_dst;
    astreams[0].provenance = d_lm.fingerprint() | 1ULL;
    xkv_backend_batch_config aconfig = {};
    aconfig.residency = ggml_backend_buft_is_host(req.buft) ? GGML_XKV_RES_REFERENCE_HOST : GGML_XKV_RES_DEVICE_OWNED;
    aconfig.enforce_residency = false;
    aconfig.backend_owner = req.executor;
    std::shared_ptr<xkv_backend_batch_storage> storage;
    try {
        storage = std::make_shared<xkv_backend_batch_storage>();
        storage->ctx = ctx_out_hold.release();
        storage->buffer = buf_out.release();
        storage->buffer_bytes = dest_bytes;
        storage->context_memory = std::move(out_ctx_buf);
        storage->backend_executor = req.executor;
    } catch (const std::exception &) { seal_err(err, "lm_rebuild: transfer storage allocation failed"); return false; }
    xkv_backend_adopt_result ares;
    if (!xkv_backend_take_device_tensors(backend, req.buft, std::move(storage), astreams,
            aconfig, *req.id_gen, ares, err, req.store_reservation)) {
        return false;
    }
    if (ares.handles.empty() || !ares.handles[0]) {
        seal_err(err, "lm_rebuild: transfer produced null handle");
        return false;
    }

    xkv_native_landmark_rebuild_result res;
    res.landmark_handle = ares.handles[0];
    res.desc = d_lm;
    res.stream_fingerprint = d_lm.fingerprint();
    res.exact_bytes = (size_t)exact_dest;
    res.transient_bytes = lm_need;
    res.sync_count = 1;
    res.chunks.resize(n_chunks);
    for (uint32_t ci = 0; ci < n_chunks; ++ci) {
        res.chunks[ci].row_begin = ci * req.chunk_tokens;
        res.chunks[ci].row_count = (ci + 1u == n_chunks) ? (n_rows - res.chunks[ci].row_begin) : req.chunk_tokens;
        res.chunks[ci].error_bound = eb_vals[ci];
        res.chunks[ci].source_fingerprint = (uint64_t)fp_raw[ci];
    }
    res.table_fingerprint = compute_landmark_table_fingerprint(res.chunks);

    out = std::move(res);
    return true;
}

} // namespace llama_xkv
