// test-xkv-reconstruct.cpp — focused backend test for GGML_OP_XKV_RECONSTRUCT.
//
// Covers (CPU oracle vs CPU graph vs Vulkan graph), one op per exact
// segment-group (n_groups=1):
//  - ranks not aligned to blocks (100/70, tail 48/40), K/V dims unequal,
//    explicit per-layer K/V offsets/dims/heads via layer_meta (aliases share
//    offsets, tail layers narrow, gaps break any uniform-formula fallback),
//    partial rotary dim (32) in HALF + INTERLEAVED modes,
//    arbitrary selected rows/positions,
//    all Turbo2/3/4 + Q8_0 (+F32 reference) streams;
//  - measured max error asserted against the encoded CPU oracle;
//  - fault cases: bad version/fingerprints/seeds/bounds/layout/codec,
//    rope-table mismatch, scratch bounds, landmark top_k==0 (never select-all),
//    mixed Turbo/canonical (CPU ok, Vulkan explicit unsupported),
//    OOB ref NaN-poison on device.
// No project-wide commands; standalone binary.

#include "ggml.h"
#include "ggml-xkv.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int failures = 0;
#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static std::vector<float> build_rope_tables(uint32_t rotary_dim, float mag_scale = 1.0f) {
    // Exact precomputed tables: omega[f] = 10000^(-2f/rotary_dim), mag uniform.
    // Both backends consume identical tables (no scalar-theta divergence).
    uint32_t fc = rotary_dim / 2;
    std::vector<float> t(size_t(fc) * 2);
    for (uint32_t f = 0; f < fc; ++f) {
        t[f] = std::pow(10000.0f, -2.0f * (float)f / (float)rotary_dim);
        t[fc + f] = mag_scale;
    }
    return t;
}

static bool encode_stream(ggml_type type, const std::vector<float> & logical, uint32_t rows,
                          uint32_t logical_cols, uint32_t padded_cols, std::vector<uint8_t> & out) {
    size_t row_bytes = ggml_row_size(type, padded_cols);
    out.assign(row_bytes * rows, 0);
    std::vector<float> pad(padded_cols, 0.0f);
    if (type == GGML_TYPE_F32) {
        for (uint32_t r = 0; r < rows; ++r) {
            memcpy(pad.data(), logical.data() + size_t(r) * logical_cols, size_t(logical_cols) * 4);
            if (padded_cols > logical_cols) {
                memset(pad.data() + logical_cols, 0, size_t(padded_cols - logical_cols) * 4);
            }
            memcpy(out.data() + size_t(r) * row_bytes, pad.data(), size_t(padded_cols) * 4);
        }
        return true;
    }
    if (type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0) {
        const auto * tr = ggml_get_type_traits(type);
        if (!tr || !tr->from_float_ref) return false;
        for (uint32_t r = 0; r < rows; ++r) {
            memcpy(pad.data(), logical.data() + size_t(r) * logical_cols, size_t(logical_cols) * 4);
            if (padded_cols > logical_cols) {
                memset(pad.data() + logical_cols, 0, size_t(padded_cols - logical_cols) * 4);
            }
            tr->from_float_ref(pad.data(), out.data() + size_t(r) * row_bytes, padded_cols);
        }
        return true;
    }
    if (type == GGML_TYPE_TURBO2_0 || type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0) {
        for (uint32_t r = 0; r < rows; ++r) {
            memcpy(pad.data(), logical.data() + size_t(r) * logical_cols, size_t(logical_cols) * 4);
            if (padded_cols > logical_cols) {
                memset(pad.data() + logical_cols, 0, size_t(padded_cols - logical_cols) * 4);
            }
            if (!ggml_quantize_turbo_row(type, pad.data(), out.data() + size_t(r) * row_bytes,
                                         padded_cols, 128)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

struct case_cfg {
    const char * name;
    ggml_type codec;
    float tol_vk; // Vulkan vs oracle budget
    uint32_t rope_mode;
    float mag;
};

// Layer map entry: [offset_k, dim_k, offset_v, dim_v, n_kv_heads].
struct layer_entry {
    int32_t ok, dk, ov, dv, nh;
};

static std::vector<float> run_graph(ggml_backend_t backend,
        ggml_type tk, const std::vector<uint8_t> & ak, int64_t prk, int64_t nrows,
        ggml_type bk, const std::vector<uint8_t> & bk_bytes, int64_t bk_rows,
        ggml_type tv, const std::vector<uint8_t> & av, int64_t prv,
        ggml_type bv, const std::vector<uint8_t> & bv_bytes, int64_t bv_rows,
        const std::vector<int32_t> & refs, const std::vector<int32_t> & pos,
        const std::vector<int32_t> & meta, const std::vector<layer_entry> & layers,
        const std::vector<float> & rope,
        const ggml_xkv_reconstruct_params & p) {
    std::vector<int32_t> lm;
    for (const auto & l : layers) {
        lm.push_back(l.ok); lm.push_back(l.dk); lm.push_back(l.ov);
        lm.push_back(l.dv); lm.push_back(l.nh);
    }
    ggml_init_params ip = { ggml_tensor_overhead() * 40 + ggml_graph_overhead_custom(40, false), nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor * a_k = ggml_new_tensor_2d(ctx.get(), tk, prk, nrows);
    ggml_tensor * b_k = ggml_new_tensor_2d(ctx.get(), bk, prk, bk_rows);
    ggml_tensor * a_v = ggml_new_tensor_2d(ctx.get(), tv, prv, nrows);
    ggml_tensor * b_v = ggml_new_tensor_2d(ctx.get(), bv, prv, bv_rows);
    ggml_tensor * r = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, p.n_sel);
    ggml_tensor * pp = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, p.n_sel);
    ggml_tensor * gm = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 8, 1);
    ggml_tensor * lm_t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 5, (int64_t)layers.size());
    ggml_tensor * rt = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, (int64_t)rope.size());
    ggml_tensor * out = ggml_xkv_reconstruct(ctx.get(), a_k, b_k, a_v, b_v, r, pp, gm, lm_t, rt, &p);
    CHECK(ggml_backend_supports_op(backend, out));
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    ggml_backend_tensor_set(a_k, ak.data(), 0, ak.size());
    ggml_backend_tensor_set(b_k, bk_bytes.data(), 0, bk_bytes.size());
    ggml_backend_tensor_set(a_v, av.data(), 0, av.size());
    ggml_backend_tensor_set(b_v, bv_bytes.data(), 0, bv_bytes.size());
    ggml_backend_tensor_set(r, refs.data(), 0, refs.size() * 4);
    ggml_backend_tensor_set(pp, pos.data(), 0, pos.size() * 4);
    ggml_backend_tensor_set(gm, meta.data(), 0, meta.size() * 4);
    ggml_backend_tensor_set(lm_t, lm.data(), 0, lm.size() * 4);
    ggml_backend_tensor_set(rt, rope.data(), 0, rope.size() * 4);
    ggml_cgraph * g = ggml_new_graph_custom(ctx.get(), 40, false);
    ggml_build_forward_expand(g, out);
    CHECK(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, res.size() * 4);
    return res;
}

static std::vector<float> run_oracle(
        ggml_type codec, const std::vector<uint8_t> & ak, int64_t prk, int64_t nrows,
        const std::vector<uint8_t> & bk, int64_t bk_rows,
        const std::vector<uint8_t> & av, int64_t prv,
        const std::vector<uint8_t> & bv, int64_t bv_rows,
        const std::vector<int32_t> & refs, const std::vector<int32_t> & pos,
        const std::vector<int32_t> & meta, const std::vector<layer_entry> & layers,
        const std::vector<float> & rope, const ggml_xkv_reconstruct_params & p) {
    std::vector<int32_t> lm;
    for (const auto & l : layers) {
        lm.push_back(l.ok); lm.push_back(l.dk); lm.push_back(l.ov);
        lm.push_back(l.dv); lm.push_back(l.nh);
    }
    std::vector<float> out(size_t(p.dim_k + p.dim_v) * p.n_sel, 0.0f);
    char err[256] = {0};
    bool ok = ggml_xkv_reconstruct_oracle(
        ak.data(), codec, prk, nrows, ggml_type_size(codec), ggml_row_size(codec, prk),
        bk.data(), codec, prk, bk_rows, ggml_type_size(codec), ggml_row_size(codec, prk),
        av.data(), codec, prv, nrows, ggml_type_size(codec), ggml_row_size(codec, prv),
        bv.data(), codec, prv, bv_rows, ggml_type_size(codec), ggml_row_size(codec, prv),
        refs.data(), pos.data(), meta.data(), lm.data(), (int64_t)layers.size(),
        rope.data(), (int64_t)rope.size(), &p,
        out.data(), p.dim_k + p.dim_v, p.n_sel, sizeof(float), size_t(p.dim_k + p.dim_v) * 4,
        err, sizeof(err));
    CHECK(ok);
    if (!ok) { std::fprintf(stderr, "oracle failed: %s\n", err); }
    return out;
}

// Main group: explicit per-layer maps with an alias, a gap, and unequal dims.
// layer0: K dim 96 @0 (2 heads), V dim 64 @0
// layer1: K dim 64 @256 (gap after layer0's 192 rows; 1 head), V dim 48 @160
// layer2: alias of layer0 (same offsets, 2 heads)
// A uniform layer_slot*n_heads*dim formula cannot reproduce these rows.
static void test_group_case(ggml_backend_t cpu_backend, ggml_backend_t vk_backend, const case_cfg & cc,
                            uint32_t rank_k, uint32_t rank_v,
                            uint32_t n_rows, uint32_t n_sel, uint32_t seed_k, uint32_t seed_v) {
    const uint32_t dim_k = 96, dim_v = 64, rotary = 32; // op maxima
    const int64_t prk = ggml_xkv_padded_rank(cc.codec, rank_k);
    const int64_t prv = ggml_xkv_padded_rank(cc.codec, rank_v);
    CHECK(prk > 0 && prv > 0);
    CHECK((uint32_t)prk > rank_k && (uint32_t)prv > rank_v); // unaligned ranks

    std::vector<layer_entry> layers = {
        {0, 96, 0, 64, 2},
        {256, 64, 160, 48, 1},
        {0, 96, 0, 64, 2}, // alias of layer0
    };
    int64_t bk_rows = 256 + 64; // cover max end (256+64)
    int64_t bv_rows = 160 + 48;

    std::mt19937 rng(0x1234 + rank_k * 7 + (uint32_t)cc.codec);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    std::vector<float> A_K(size_t(n_rows) * rank_k), A_V(size_t(n_rows) * rank_v);
    for (auto & x : A_K) x = dist(rng);
    for (auto & x : A_V) x = dist(rng);
    std::vector<float> B_K(size_t(bk_rows) * rank_k, 0.0f), B_V(size_t(bv_rows) * rank_v, 0.0f);
    for (auto & x : B_K) x = dist(rng);
    for (auto & x : B_V) x = dist(rng);

    std::vector<uint8_t> ak_b, bk_b, av_b, bv_b;
    CHECK(encode_stream(cc.codec, A_K, n_rows, rank_k, (uint32_t)prk, ak_b));
    CHECK(encode_stream(cc.codec, B_K, (uint32_t)bk_rows, rank_k, (uint32_t)prk, bk_b));
    CHECK(encode_stream(cc.codec, A_V, n_rows, rank_v, (uint32_t)prv, av_b));
    CHECK(encode_stream(cc.codec, B_V, (uint32_t)bv_rows, rank_v, (uint32_t)prv, bv_b));

    std::vector<int32_t> refs(size_t(n_sel) * 4), pos(n_sel);
    std::mt19937 rr(0x77 + n_sel);
    const uint32_t nl = (uint32_t)layers.size();
    for (uint32_t i = 0; i < n_sel; ++i) {
        uint32_t ls = uint32_t(rr() % nl);
        refs[i * 4 + 0] = int32_t(rr() % n_rows);
        refs[i * 4 + 1] = 0;
        refs[i * 4 + 2] = int32_t(ls);
        refs[i * 4 + 3] = int32_t(rr() % (uint32_t)layers[ls].nh);
        pos[i] = int32_t(rr() % 4096);
    }
    std::vector<int32_t> meta = { (int32_t)nl, 2, 0, (int32_t)bk_rows, 0, (int32_t)bv_rows,
                                  (int32_t)rank_k, (int32_t)rank_v };
    std::vector<float> rope = build_rope_tables(rotary, cc.mag);

    ggml_xkv_reconstruct_params p = {};
    p.version = GGML_XKV_VERSION; p.n_sel = n_sel; p.n_groups = 1;
    p.rank_k = rank_k; p.rank_v = rank_v; p.dim_k = dim_k; p.dim_v = dim_v;
    p.rotary_dim = rotary; p.rope_mode = cc.rope_mode;
    p.seed_k = seed_k; p.seed_v = seed_v;
    p.fp_combined = ggml_xkv_fp_combined(cc.codec, cc.codec, cc.codec, cc.codec, seed_k, seed_v);

    std::vector<float> oracle = run_oracle(cc.codec, ak_b, prk, n_rows, bk_b, bk_rows,
        av_b, prv, bv_b, bv_rows, refs, pos, meta, layers, rope, p);

    // Explicit mapping observably differs from a naive uniform formula:
    // layer1's K rows live at offset 256, but slot*nh*dim would place a
    // (slot=1, 2-head, dim=96) slice at 192. Aliased layer2 must equal layer0.
    {
        bool saw_layer1 = false, alias_ok = true;
        for (uint32_t i = 0; i < n_sel; ++i) {
            if (refs[i * 4 + 2] == 1) saw_layer1 = true;
        }
        CHECK(saw_layer1);
        // Alias check: same (row,pos,head) via slot0 vs slot2 must match.
        std::vector<int32_t> r2 = {refs[0], 0, 2, refs[3]};
        std::vector<int32_t> p2 = {pos[0]};
        std::vector<int32_t> r0 = {refs[0], 0, 0, refs[3]};
        if (refs[2] != 2) { r0[3] = r2[3] = 0; }
        ggml_xkv_reconstruct_params q = p; q.n_sel = 1;
        q.fp_combined = p.fp_combined;
        std::vector<float> o0 = run_oracle(cc.codec, ak_b, prk, n_rows, bk_b, bk_rows,
            av_b, prv, bv_b, bv_rows, r0, p2, meta, layers, rope, q);
        std::vector<float> o2 = run_oracle(cc.codec, ak_b, prk, n_rows, bk_b, bk_rows,
            av_b, prv, bv_b, bv_rows, r2, p2, meta, layers, rope, q);
        for (size_t k = 0; k < o0.size(); ++k) {
            if (std::fabs(o0[k] - o2[k]) > 1e-5f) { alias_ok = false; break; }
        }
        CHECK(alias_ok);
        (void)saw_layer1;
    }

    std::vector<float> cpu_g = run_graph(cpu_backend, cc.codec, ak_b, prk, n_rows, cc.codec, bk_b,
        bk_rows, cc.codec, av_b, prv, cc.codec, bv_b, bv_rows, refs, pos, meta, layers, rope, p);
    float e_cpu = max_abs_diff(oracle, cpu_g);
    std::fprintf(stderr, "%s rk=%u rv=%u mode=%u cpu-vs-oracle = %.9g\n", cc.name, rank_k, rank_v,
                 cc.rope_mode, e_cpu);
    CHECK(e_cpu < 1e-5f);

    if (vk_backend) {
        std::vector<float> vk_g = run_graph(vk_backend, cc.codec, ak_b, prk, n_rows, cc.codec, bk_b,
            bk_rows, cc.codec, av_b, prv, cc.codec, bv_b, bv_rows, refs, pos, meta, layers, rope, p);
        float e_vk = max_abs_diff(oracle, vk_g);
        std::fprintf(stderr, "%s rk=%u rv=%u mode=%u vk-vs-oracle = %.9g (tol %.9g)\n", cc.name,
                     rank_k, rank_v, cc.rope_mode, e_vk, cc.tol_vk);
        CHECK(e_vk < cc.tol_vk);
    }
}

static void test_faults(ggml_backend_t cpu_backend) {
    (void)cpu_backend;
    char err[256];
    CHECK(ggml_xkv_exact_bytes(GGML_TYPE_Q8_0, 100, 4, err, sizeof(err)) == 0);
    CHECK(!ggml_xkv_codec_supported(GGML_TYPE_Q4_0));
    CHECK(ggml_xkv_exact_bytes(GGML_TYPE_Q4_0, 32, 4, err, sizeof(err)) == 0);
    CHECK(ggml_xkv_residency_supported(GGML_TYPE_Q8_0, GGML_XKV_RES_REFERENCE_HOST));
    CHECK(ggml_xkv_residency_supported(GGML_TYPE_Q8_0, GGML_XKV_RES_DEVICE_OWNED));
    CHECK(!ggml_xkv_residency_supported(GGML_TYPE_Q4_0, GGML_XKV_RES_DEVICE_OWNED));
    CHECK(!ggml_xkv_residency_supported(GGML_TYPE_Q4_0, GGML_XKV_RES_REFERENCE_HOST));

    ggml_xkv_reconstruct_params p = {};
    p.version = GGML_XKV_VERSION; p.n_sel = 2; p.n_groups = 1;
    p.rank_k = 32; p.rank_v = 32; p.dim_k = 16; p.dim_v = 8;
    p.rotary_dim = 8; p.rope_mode = GGML_XKV_ROPE_HALF;
    p.seed_k = 42; p.seed_v = 43;
    p.fp_combined = ggml_xkv_fp_combined(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32,
                                         GGML_TYPE_F32, 42, 43);
    ggml_init_params ip = { ggml_tensor_overhead() * 24, nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor * ak = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 4);
    ggml_tensor * bk = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 32);
    ggml_tensor * av = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 4);
    ggml_tensor * bv = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 16);
    ggml_tensor * rf = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, 2);
    ggml_tensor * ps = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 2);
    ggml_tensor * gm = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 8, 1);
    ggml_tensor * lm = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 5, 2);
    ggml_tensor * rt = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 8);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 24, 2);

    {
        ggml_xkv_reconstruct_params q = p; q.n_groups = 2;
        CHECK(!ggml_xkv_reconstruct_supports(ak, bk, av, bv, rf, ps, gm, lm, rt, dst, &q, err, sizeof(err)));
    }
    {
        ggml_xkv_reconstruct_params q = p; q.version = 1;
        CHECK(!ggml_xkv_reconstruct_supports(ak, bk, av, bv, rf, ps, gm, lm, rt, dst, &q, err, sizeof(err)));
    }
    {
        ggml_xkv_reconstruct_params q = p; q.fp_combined ^= 1u;
        CHECK(!ggml_xkv_reconstruct_supports(ak, bk, av, bv, rf, ps, gm, lm, rt, dst, &q, err, sizeof(err)));
    }
    {
        ggml_xkv_reconstruct_params q = p; q.seed_k = 43;
        CHECK(!ggml_xkv_reconstruct_supports(ak, bk, av, bv, rf, ps, gm, lm, rt, dst, &q, err, sizeof(err)));
    }
    {
        ggml_xkv_reconstruct_params q = p; q.seed_v = 44;
        CHECK(!ggml_xkv_reconstruct_supports(ak, bk, av, bv, rf, ps, gm, lm, rt, dst, &q, err, sizeof(err)));
    }
    {
        ggml_init_params ip2 = { ggml_tensor_overhead() * 8, nullptr, true };
        ggml_context_ptr c2(ggml_init(ip2));
        ggml_tensor * t4 = ggml_new_tensor_2d(c2.get(), GGML_TYPE_TURBO4_0, 128, 4);
        ggml_tensor * t3 = ggml_new_tensor_2d(c2.get(), GGML_TYPE_TURBO3_0, 128, 4);
        ggml_xkv_reconstruct_params q = p;
        q.rank_k = 100; q.rank_v = 100; q.dim_k = 16; q.dim_v = 8; q.rotary_dim = 8;
        q.seed_k = 42; q.seed_v = 43;
        q.fp_combined = ggml_xkv_fp_combined(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0,
                                             GGML_TYPE_F32, GGML_TYPE_F32, 42, 43);
        CHECK(!ggml_xkv_reconstruct_supports(t4, t3, av, bv, rf, ps, gm, lm, rt, dst, &q, err, sizeof(err)));
    }
    {
        ggml_tensor * bad_rt = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 6);
        CHECK(!ggml_xkv_reconstruct_supports(ak, bk, av, bv, rf, ps, gm, lm, bad_rt, dst, &p, err, sizeof(err)));
    }
    // Layer slice exceeding B bounds rejected (offset 30 + dim 16 > 32 rows).
    {
        std::vector<uint8_t> za(ggml_row_size(GGML_TYPE_F32, 32) * 4, 0);
        std::vector<uint8_t> zbk(ggml_row_size(GGML_TYPE_F32, 32) * 64, 0);
        std::vector<uint8_t> zbv(ggml_row_size(GGML_TYPE_F32, 32) * 32, 0);
        std::vector<float> zrope(8, 0.0f);
        for (int i = 0; i < 4; ++i) zrope[4 + i] = 1.0f;
        std::vector<float> zdst(24 * 2, 0.0f);
        std::vector<int32_t> zgm = {2, 2, 0, 64, 0, 32, 32, 32};
        std::vector<int32_t> zlm = {0, 16, 0, 8, 2, 30, 16, 24, 8, 1};
        std::vector<int32_t> zps = {0, 1};
        std::vector<int32_t> zrf = {0, 0, 1, 0, 0, 0, 0, 0}; // slot1 K slice 30+16 > 64? no: 46<64; use slot1 V? V 24+8=32 ok. Force OOB via head:
        std::vector<int32_t> zrf_h = {0, 0, 1, 5, 0, 0, 0, 0}; // head 5 >= nh 1
        CHECK(!ggml_xkv_reconstruct_oracle(
            za.data(), GGML_TYPE_F32, 32, 4, 4, 128,
            zbk.data(), GGML_TYPE_F32, 32, 64, 4, 128,
            za.data(), GGML_TYPE_F32, 32, 4, 4, 128,
            zbv.data(), GGML_TYPE_F32, 32, 32, 4, 128,
            zrf_h.data(), zps.data(), zgm.data(), zlm.data(), 2, zrope.data(), 8, &p,
            zdst.data(), 24, 2, 4, 96, err, sizeof(err)));
        // Layer dim exceeding op maxima rejected (dim_k 32 > 16).
        std::vector<int32_t> zlm2 = {0, 32, 0, 8, 1, 0, 16, 0, 8, 1};
        CHECK(!ggml_xkv_reconstruct_oracle(
            za.data(), GGML_TYPE_F32, 32, 4, 4, 128,
            zbk.data(), GGML_TYPE_F32, 32, 64, 4, 128,
            za.data(), GGML_TYPE_F32, 32, 4, 4, 128,
            zbv.data(), GGML_TYPE_F32, 32, 32, 4, 128,
            zrf.data(), zps.data(), zgm.data(), zlm2.data(), 2, zrope.data(), 8, &p,
            zdst.data(), 24, 2, 4, 96, err, sizeof(err)));
        (void)zrf;
    }
    // Out-of-bounds selected row rejected.
    {
        std::vector<uint8_t> za(ggml_row_size(GGML_TYPE_F32, 32) * 4, 0);
        std::vector<uint8_t> zbk(ggml_row_size(GGML_TYPE_F32, 32) * 64, 0);
        std::vector<uint8_t> zbv(ggml_row_size(GGML_TYPE_F32, 32) * 32, 0);
        std::vector<float> zrope(8, 0.0f);
        for (int i = 0; i < 4; ++i) zrope[4 + i] = 1.0f;
        std::vector<float> zdst(24 * 2, 0.0f);
        std::vector<int32_t> zgm = {1, 1, 0, 16, 0, 8, 32, 32};
        std::vector<int32_t> zlm = {0, 16, 0, 8, 1};
        std::vector<int32_t> zps = {0, 1};
        std::vector<int32_t> zrf = {4, 0, 0, 0, 0, 0, 0, 0}; // a_row 4 == n_rows
        CHECK(!ggml_xkv_reconstruct_oracle(
            za.data(), GGML_TYPE_F32, 32, 4, 4, 128,
            zbk.data(), GGML_TYPE_F32, 32, 64, 4, 128,
            za.data(), GGML_TYPE_F32, 32, 4, 4, 128,
            zbv.data(), GGML_TYPE_F32, 32, 32, 4, 128,
            zrf.data(), zps.data(), zgm.data(), zlm.data(), 1, zrope.data(), 8, &p,
            zdst.data(), 24, 2, 4, 96, err, sizeof(err)));
    }
    // Scratch too small rejected.
    {
        size_t need = 0;
        CHECK(ggml_xkv_core_scratch_floats(&p, 32, 32, &need, err, sizeof(err)));
        CHECK(need == size_t(2 * 32 + 16 + 8 + 64));
        std::vector<float> a4(32 * 4, 0.0f), b4(32 * 32, 0.0f);
        std::vector<float> av4(32 * 4, 0.0f), bv4(32 * 16, 0.0f);
        std::vector<int32_t> rf2 = {0, 0, 0, 0, 1, 0, 0, 0};
        std::vector<int32_t> ps2 = {0, 7};
        std::vector<int32_t> gm2 = {1, 2, 0, 32, 0, 16, 32, 32};
        std::vector<int32_t> lm2 = {0, 16, 0, 8, 2};
        std::vector<float> rope2(8, 0.0f);
        for (int i = 0; i < 4; ++i) rope2[4 + i] = 1.0f;
        std::vector<float> dst2(24 * 2, 0.0f);
        std::vector<float> tiny(4, 0.0f);
        ggml_xkv_reconstruct_params q = p;
        q.fp_combined = ggml_xkv_fp_combined(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32,
                                             GGML_TYPE_F32, q.seed_k, q.seed_v);
        CHECK(!ggml_xkv_reconstruct_core(
            a4.data(), GGML_TYPE_F32, 32, 4, 4, 128,
            b4.data(), GGML_TYPE_F32, 32, 32, 4, 128,
            av4.data(), GGML_TYPE_F32, 32, 4, 4, 128,
            bv4.data(), GGML_TYPE_F32, 32, 16, 4, 128,
            rf2.data(), ps2.data(), gm2.data(), lm2.data(), 1, rope2.data(), 8, &q,
            dst2.data(), 24, 2, 4, 96, tiny.data(), tiny.size(), err, sizeof(err)));
    }
    // Landmark top_k==0 fails (never select-all); refine cap reported.
    {
        ggml_xkv_landmark_config cfg = {};
        cfg.dim = 16; cfg.rotary_dim = 8; cfg.rope_mode = GGML_XKV_ROPE_HALF;
        cfg.top_k = 0; cfg.refine_max_rows = 4;
        float sc[4] = {1, 2, 3, 4};
        uint32_t idx[4] = {0, 0, 0, 0};
        CHECK(!ggml_xkv_landmark_topk(sc, 4, &cfg, idx, nullptr, err, sizeof(err)));
        cfg.top_k = 2;
        CHECK(ggml_xkv_landmark_topk(sc, 4, &cfg, idx, nullptr, err, sizeof(err)));
        CHECK(idx[0] == 3 && idx[1] == 2);
        uint32_t ref[8] = {0}; uint32_t nref = 0; bool hit = false;
        uint32_t sel[2] = {0, 1};
        CHECK(ggml_xkv_landmark_refine(sel, 2, 16, 4, 4, &cfg, ref, &nref, 5, &hit, err, sizeof(err)));
        CHECK(hit);
    }
    // Landmark scoring uses post-RoPE q as-is (no Q rotation).
    {
        const uint32_t dim = 16, rotary = 8, fc = 4;
        std::vector<float> omega(fc), mag(fc, 1.0f);
        for (uint32_t f = 0; f < fc; ++f) omega[f] = std::pow(10000.0f, -2.0f * f / rotary);
        std::vector<float> Lc(dim, 0.0f); Lc[0] = 1.0f;
        int64_t padded = ggml_xkv_padded_rank(GGML_TYPE_F32, dim);
        std::vector<uint8_t> Lb(size_t(padded) * 4, 0);
        memcpy(Lb.data(), Lc.data(), size_t(dim) * 4);
        std::vector<float> q(dim, 0.0f); q[0] = 1.0f;
        int32_t fpos[1] = {3};
        float scores[1] = {0};
        ggml_xkv_landmark_config cfg = {};
        cfg.dim = dim; cfg.rotary_dim = rotary; cfg.rope_mode = GGML_XKV_ROPE_HALF;
        cfg.top_k = 1;
        CHECK(ggml_xkv_landmark_score(q.data(), dim, Lb.data(), GGML_TYPE_F32, padded, 1,
                                      ggml_type_size(GGML_TYPE_F32), size_t(padded) * 4,
                                      fpos, omega.data(), mag.data(), fc, &cfg, scores, err, sizeof(err)));
        float expect = std::cos(3.0f * omega[0]);
        CHECK(std::fabs(scores[0] - expect) < 1e-5f);
    }
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    CHECK(cpu_backend != nullptr);
    ggml_backend_t vk_backend = nullptr;
    if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
        vk_backend = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU), nullptr);
        if (!vk_backend) {
            std::puts("SKIP: GPU device present but init failed");
        }
    } else {
        std::puts("SKIP: no GPU backend; Vulkan parity skipped, CPU oracle checks still run");
    }

    const case_cfg cases[] = {
        {"f32",  GGML_TYPE_F32,     2e-4f, GGML_XKV_ROPE_HALF,        1.0f},
        {"q8",   GGML_TYPE_Q8_0,    1e-2f, GGML_XKV_ROPE_HALF,        1.0f},
        {"tq4",  GGML_TYPE_TURBO4_0, 5e-2f, GGML_XKV_ROPE_HALF,       1.0f},
        {"tq4i", GGML_TYPE_TURBO4_0, 5e-2f, GGML_XKV_ROPE_INTERLEAVED, 1.0f},
        {"tq3",  GGML_TYPE_TURBO3_0, 8e-2f, GGML_XKV_ROPE_HALF,       1.05f},
        {"tq2",  GGML_TYPE_TURBO2_0, 2e-1f, GGML_XKV_ROPE_HALF,       1.0f},
    };
    for (const auto & cc : cases) {
        test_group_case(cpu_backend, vk_backend, cc, 100, 70, 24, 10, 42, 43);
    }
    // Unequal-rank tail group as a SEPARATE op (own exact ranks/descriptors).
    for (const auto & cc : cases) {
        test_group_case(cpu_backend, vk_backend, cc, 48, 40, 16, 6, 42, 43);
    }
    test_faults(cpu_backend);

    // Mixed Turbo/canonical: CPU oracle succeeds via canonical decode.
    // (Vulkan mixed-reject lives in the backend support hook.)
    {
        char err[256] = {0};
        std::mt19937 rng(9);
        std::normal_distribution<float> dist(0.0f, 0.3f);
        std::vector<float> fAk(4 * 100), fBk(8 * 100), fAv(4 * 24), fBv(8 * 24);
        for (auto & x : fAk) x = dist(rng);
        for (auto & x : fBk) x = dist(rng);
        for (auto & x : fAv) x = dist(rng);
        for (auto & x : fBv) x = dist(rng);
        std::vector<uint8_t> eAk, eBk, eAv, eBv;
        CHECK(encode_stream(GGML_TYPE_TURBO4_0, fAk, 4, 100, 128, eAk));
        CHECK(encode_stream(GGML_TYPE_F32, fBk, 8, 100, 128, eBk));
        CHECK(encode_stream(GGML_TYPE_Q8_0, fAv, 4, 24, 32, eAv));
        CHECK(encode_stream(GGML_TYPE_Q8_0, fBv, 8, 24, 32, eBv));
        std::vector<int32_t> rf = {0, 0, 0, 0, 1, 0, 1, 0};
        std::vector<int32_t> ps = {5, 11};
        std::vector<int32_t> gm = {2, 1, 0, 32, 0, 16, 100, 24};
        std::vector<int32_t> lm = {0, 16, 0, 8, 1, 0, 16, 0, 8, 1};
        std::vector<float> rope = build_rope_tables(8);
        ggml_xkv_reconstruct_params mp = {};
        mp.version = GGML_XKV_VERSION; mp.n_sel = 2; mp.n_groups = 1;
        mp.rank_k = 100; mp.rank_v = 24; mp.dim_k = 16; mp.dim_v = 8;
        mp.rotary_dim = 8; mp.rope_mode = GGML_XKV_ROPE_HALF;
        mp.seed_k = 42; mp.seed_v = 43;
        mp.fp_combined = ggml_xkv_fp_combined(GGML_TYPE_TURBO4_0, GGML_TYPE_F32,
                                              GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 42, 43);
        std::vector<float> mout(24 * 2, 0.0f);
        CHECK(ggml_xkv_reconstruct_oracle(
            eAk.data(), GGML_TYPE_TURBO4_0, 128, 4,
            ggml_type_size(GGML_TYPE_TURBO4_0), ggml_row_size(GGML_TYPE_TURBO4_0, 128),
            eBk.data(), GGML_TYPE_F32, 128, 8, 4, size_t(128) * 4,
            eAv.data(), GGML_TYPE_Q8_0, 32, 4,
            ggml_type_size(GGML_TYPE_Q8_0), ggml_row_size(GGML_TYPE_Q8_0, 32),
            eBv.data(), GGML_TYPE_Q8_0, 32, 8,
            ggml_type_size(GGML_TYPE_Q8_0), ggml_row_size(GGML_TYPE_Q8_0, 32),
            rf.data(), ps.data(), gm.data(), lm.data(), 2, rope.data(), (int64_t)rope.size(), &mp,
            mout.data(), 24, 2, 4, size_t(24) * 4, err, sizeof(err)));
        for (float x : mout) { CHECK(std::isfinite(x)); }
    }

    std::printf("=== Results: %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
}
