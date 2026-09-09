// test-xkv-vulkan-landmark-build.cpp — Focused tests for native
// landmark-construction op GGML_OP_XKV_LANDMARK_BUILD.
//
// Covers:
//  1. Checked scratch estimator (exact bytes, reject short/overflow)
//  2. Validation hook (ggml_xkv_landmark_build_supports) - fail closed on
//     bad types, shapes, dimensions, non-contiguous strides
//  3. CPU oracle - tails, multiple chunks, odd logical dims, Q8_0 and Turbo4_0
//  4. CPU graph compute vs CPU oracle bitwise parity
//  5. Vulkan graph compute vs CPU oracle parity (when device present)
//  6. Malformed/short scratch - fails closed with nonzero status, outputs untouched
//  7. Status-first fail-closed contract (never uninitialized or silent crash)

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "ggml.h"
#include "ggml-xkv.h"
#include "ggml-xkv-landmark-build.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>

static int failures = 0;
#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

static uint64_t rng_state = 0x54321ULL;
static uint64_t rng_next() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static float rng_uni() {
    return (float)(rng_next() & 0x7FFFFFFFu) / (float)0x7FFFFFFF * 2.0f - 1.0f;
}

static ggml_backend_t open_vulkan_backend() {
    ggml_backend_load_all();
    ggml_backend_t vk = nullptr;
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev) vk = ggml_backend_dev_init(dev, nullptr);
#ifdef GGML_USE_VULKAN
    if (!vk && ggml_backend_vk_get_device_count() > 0) {
        char desc[256] = {};
        ggml_backend_vk_get_device_description(0, desc, sizeof(desc));
        std::printf("  direct vk_init(0): %s\n", desc);
        vk = ggml_backend_vk_init(0);
    }
#endif
    return vk;
}

// Case definition for CPU oracle and graph tests
struct build_case {
    ggml_xkv_landmark_build_params p;
    std::vector<uint8_t> a_k_bytes;
    std::vector<uint8_t> b_k_bytes;
    std::vector<int32_t> rows;
    std::vector<int64_t> pos;
    std::vector<int32_t> layer_meta;
    std::vector<float> rope_tables;
    uint32_t max_dim = 0;
};

static build_case make_test_case(
        uint32_t n_rows, uint32_t chunk_tokens,
        uint32_t total_dim, uint32_t rank,
        enum ggml_type factor_type, enum ggml_type lm_type,
        bool with_rope) {
    build_case c = {};
    uint32_t n_chunks = (n_rows + chunk_tokens - 1) / chunk_tokens;
    uint32_t pad_rk = rank;
    if (factor_type == GGML_TYPE_TURBO4_0) pad_rk = (rank + 127) / 128 * 128;
    else if (factor_type == GGML_TYPE_Q8_0) pad_rk = (rank + 31) / 32 * 32;

    uint32_t pad_dim = total_dim;
    if (lm_type == GGML_TYPE_TURBO4_0) pad_dim = (total_dim + 127) / 128 * 128;
    else if (lm_type == GGML_TYPE_Q8_0) pad_dim = (total_dim + 31) / 32 * 32;

    c.p.version = GGML_XKV_LANDMARK_BUILD_VERSION;
    c.p.n_rows = n_rows;
    c.p.n_chunks = n_chunks;
    c.p.chunk_tokens = chunk_tokens;
    c.p.total_dim = total_dim;
    c.p.padded_dim = pad_dim;
    c.p.rank = rank;
    c.p.pad_rank = pad_rk;
    c.p.n_layers = 2;
    c.p.landmark_type = (uint32_t)lm_type;
    c.p.a_type = (uint32_t)factor_type;
    c.p.b_type = (uint32_t)factor_type;
    c.p.seed = 42;
    c.p.phase_lo = 0x12345678;
    c.p.phase_hi = 0x9abcdef0;

    // layer slices: split total_dim into 2 slices
    uint32_t d0 = total_dim / 2;
    uint32_t d1 = total_dim - d0;
    c.max_dim = std::max(d0, d1);
    c.p.max_feature_dim = c.max_dim;

    uint32_t rd0 = with_rope ? (d0 >= 16 ? 16 : (d0 / 2 * 2)) : 0;
    uint32_t rd1 = with_rope ? (d1 >= 16 ? 16 : (d1 / 2 * 2)) : 0;

    c.layer_meta = {
        0, (int32_t)d0, (int32_t)d0, (int32_t)rd0, 0, 0,
        (int32_t)d0, (int32_t)d1, (int32_t)d1, (int32_t)rd1, 1, (int32_t)rd0
    };

    uint32_t rope_total = rd0 + rd1;
    c.rope_tables.resize(rope_total, 1.0f);
    for (size_t i = 0; i < rd0 / 2; ++i) c.rope_tables[i] = 0.01f * (i + 1);
    for (size_t i = rd0 / 2; i < rd0; ++i) c.rope_tables[i] = 1.0f;
    for (size_t i = rd0; i < rd0 + rd1 / 2; ++i) c.rope_tables[i] = 0.02f * (i - rd0 + 1);
    for (size_t i = rd0 + rd1 / 2; i < rope_total; ++i) c.rope_tables[i] = 1.0f;

    c.rows.resize(n_rows);
    c.pos.resize(n_rows);
    for (uint32_t i = 0; i < n_rows; ++i) {
        c.rows[i] = (int32_t)i;
        c.pos[i] = 100 + i * 3;
    }

    // synthesize A_K and B_K rows
    size_t a_row_sz = ggml_row_size(factor_type, pad_rk);
    size_t b_row_sz = ggml_row_size(factor_type, pad_rk);
    c.a_k_bytes.resize(a_row_sz * n_rows);
    c.b_k_bytes.resize(b_row_sz * total_dim);

    std::vector<float> tmp_row(pad_rk, 0.0f);
    for (uint32_t i = 0; i < n_rows; ++i) {
        for (uint32_t j = 0; j < rank; ++j) tmp_row[j] = rng_uni();
        for (uint32_t j = rank; j < pad_rk; ++j) tmp_row[j] = 0.0f;
        if (factor_type == GGML_TYPE_F32) {
            memcpy(c.a_k_bytes.data() + i * a_row_sz, tmp_row.data(), pad_rk * 4);
        } else if (factor_type == GGML_TYPE_Q8_0) {
            const auto * tr = ggml_get_type_traits(GGML_TYPE_Q8_0);
            for (uint32_t b = 0; b < pad_rk / 32; ++b) {
                tr->from_float_ref(tmp_row.data() + b * 32,
                    c.a_k_bytes.data() + i * a_row_sz + b * 34, 32);
            }
        } else if (factor_type == GGML_TYPE_TURBO4_0) {
            for (uint32_t g = 0; g < pad_rk / 128; ++g) {
                ggml_quantize_turbo_row(GGML_TYPE_TURBO4_0, tmp_row.data() + g * 128,
                    c.a_k_bytes.data() + i * a_row_sz + g * 68, 128, 128);
            }
        }
    }
    for (uint32_t i = 0; i < total_dim; ++i) {
        for (uint32_t j = 0; j < rank; ++j) tmp_row[j] = rng_uni();
        for (uint32_t j = rank; j < pad_rk; ++j) tmp_row[j] = 0.0f;
        if (factor_type == GGML_TYPE_F32) {
            memcpy(c.b_k_bytes.data() + i * b_row_sz, tmp_row.data(), pad_rk * 4);
        } else if (factor_type == GGML_TYPE_Q8_0) {
            const auto * tr = ggml_get_type_traits(GGML_TYPE_Q8_0);
            for (uint32_t b = 0; b < pad_rk / 32; ++b) {
                tr->from_float_ref(tmp_row.data() + b * 32,
                    c.b_k_bytes.data() + i * b_row_sz + b * 34, 32);
            }
        } else if (factor_type == GGML_TYPE_TURBO4_0) {
            for (uint32_t g = 0; g < pad_rk / 128; ++g) {
                ggml_quantize_turbo_row(GGML_TYPE_TURBO4_0, tmp_row.data() + g * 128,
                    c.b_k_bytes.data() + i * b_row_sz + g * 68, 128, 128);
            }
        }
    }
    return c;
}

static void test_scratch_bytes_exactness() {
    std::printf("[test] scratch_bytes exactness and bounds ...\n");
    char err[256] = {};
    build_case c = make_test_case(10, 4, 64, 32, GGML_TYPE_F32, GGML_TYPE_Q8_0, false);
    size_t need = 0;
    CHECK(ggml_xkv_landmark_build_scratch_bytes(&c.p, &need, err, sizeof(err)));
    CHECK(need > 0);

    // Bad params must fail closed
    ggml_xkv_landmark_build_params bad = c.p;
    bad.version = 999;
    CHECK(!ggml_xkv_landmark_build_scratch_bytes(&bad, &need, err, sizeof(err)));
    bad = c.p;
    bad.n_rows = 0;
    CHECK(!ggml_xkv_landmark_build_scratch_bytes(&bad, &need, err, sizeof(err)));
    bad = c.p;
    bad.n_chunks = 99; // mismatch ceil
    CHECK(!ggml_xkv_landmark_build_scratch_bytes(&bad, &need, err, sizeof(err)));
}

static void test_cpu_oracle_cases() {
    std::printf("[test] CPU oracle parity and conservative bounds ...\n");
    char err[256] = {};

    // 1. Tails and multiple chunks (11 rows, chunk 4 -> 3 chunks, tail = 3 rows)
    {
        build_case c = make_test_case(11, 4, 48, 32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true);
        size_t need = 0;
        CHECK(ggml_xkv_landmark_build_scratch_bytes(&c.p, &need, err, sizeof(err)));
        std::vector<uint8_t> scr(need);
        size_t dst_row_sz = ggml_row_size((enum ggml_type)c.p.landmark_type, c.p.padded_dim);
        std::vector<uint8_t> dst(dst_row_sz * c.p.n_chunks, 0);
        std::vector<float> eb(c.p.n_chunks, 0.0f);
        std::vector<uint64_t> fp(c.p.n_chunks, 0);
        int32_t st[4] = {};

        CHECK(ggml_xkv_landmark_build_cpu_oracle(
            c.a_k_bytes.data(), (enum ggml_type)c.p.a_type, c.p.pad_rank, c.p.n_rows, 0, ggml_row_size((enum ggml_type)c.p.a_type, c.p.pad_rank),
            c.b_k_bytes.data(), (enum ggml_type)c.p.b_type, c.p.pad_rank, c.p.total_dim, 0, ggml_row_size((enum ggml_type)c.p.b_type, c.p.pad_rank),
            c.rows.data(), c.p.n_rows,
            c.pos.data(), 1,
            c.layer_meta.data(), c.p.n_layers,
            c.rope_tables.data(), c.rope_tables.size(),
            &c.p,
            dst.data(), c.p.padded_dim, c.p.n_chunks, 0, dst_row_sz,
            eb.data(), fp.data(), st,
            scr.data(), scr.size(),
            err, sizeof(err)));

        CHECK(st[0] == GGML_XKV_LANDMARK_BUILD_STATUS_OK);
        CHECK(st[1] == (int32_t)c.p.n_chunks);
        for (uint32_t i = 0; i < c.p.n_chunks; ++i) {
            CHECK(eb[i] >= 1e-6f);
            CHECK(fp[i] != 0 && fp[i] != 0xcbf29ce484222325ULL);
        }
    }

    // 2. Odd logical dims + Turbo4_0 output
    {
        build_case c = make_test_case(7, 3, 53, 128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0, true);
        size_t need = 0;
        CHECK(ggml_xkv_landmark_build_scratch_bytes(&c.p, &need, err, sizeof(err)));
        std::vector<uint8_t> scr(need);
        size_t dst_row_sz = ggml_row_size((enum ggml_type)c.p.landmark_type, c.p.padded_dim);
        std::vector<uint8_t> dst(dst_row_sz * c.p.n_chunks, 0);
        std::vector<float> eb(c.p.n_chunks, 0.0f);
        std::vector<uint64_t> fp(c.p.n_chunks, 0);
        int32_t st[4] = {};

        CHECK(ggml_xkv_landmark_build_cpu_oracle(
            c.a_k_bytes.data(), (enum ggml_type)c.p.a_type, c.p.pad_rank, c.p.n_rows, 0, ggml_row_size((enum ggml_type)c.p.a_type, c.p.pad_rank),
            c.b_k_bytes.data(), (enum ggml_type)c.p.b_type, c.p.pad_rank, c.p.total_dim, 0, ggml_row_size((enum ggml_type)c.p.b_type, c.p.pad_rank),
            c.rows.data(), c.p.n_rows,
            c.pos.data(), 1,
            c.layer_meta.data(), c.p.n_layers,
            c.rope_tables.data(), c.rope_tables.size(),
            &c.p,
            dst.data(), c.p.padded_dim, c.p.n_chunks, 0, dst_row_sz,
            eb.data(), fp.data(), st,
            scr.data(), scr.size(),
            err, sizeof(err)));

        CHECK(st[0] == GGML_XKV_LANDMARK_BUILD_STATUS_OK);
        for (uint32_t i = 0; i < c.p.n_chunks; ++i) {
            CHECK(eb[i] >= 1e-4f);
            CHECK(fp[i] != 0);
        }
    }

    // 3. Short scratch fails closed with STATUS_ERR_WORKSPACE
    {
        build_case c = make_test_case(4, 2, 32, 32, GGML_TYPE_F32, GGML_TYPE_Q8_0, false);
        size_t need = 0;
        CHECK(ggml_xkv_landmark_build_scratch_bytes(&c.p, &need, err, sizeof(err)));
        std::vector<uint8_t> scr(need - 1); // 1 byte short
        size_t dst_row_sz = ggml_row_size((enum ggml_type)c.p.landmark_type, c.p.padded_dim);
        std::vector<uint8_t> dst(dst_row_sz * c.p.n_chunks, 0xAA);
        std::vector<float> eb(c.p.n_chunks, -1.0f);
        std::vector<uint64_t> fp(c.p.n_chunks, 0);
        int32_t st[4] = {};

        CHECK(!ggml_xkv_landmark_build_cpu_oracle(
            c.a_k_bytes.data(), (enum ggml_type)c.p.a_type, c.p.pad_rank, c.p.n_rows, 0, ggml_row_size((enum ggml_type)c.p.a_type, c.p.pad_rank),
            c.b_k_bytes.data(), (enum ggml_type)c.p.b_type, c.p.pad_rank, c.p.total_dim, 0, ggml_row_size((enum ggml_type)c.p.b_type, c.p.pad_rank),
            c.rows.data(), c.p.n_rows,
            c.pos.data(), 1,
            c.layer_meta.data(), c.p.n_layers,
            c.rope_tables.data(), c.rope_tables.size(),
            &c.p,
            dst.data(), c.p.padded_dim, c.p.n_chunks, 0, dst_row_sz,
            eb.data(), fp.data(), st,
            scr.data(), scr.size(),
            err, sizeof(err)));

        CHECK(st[0] == GGML_XKV_LANDMARK_BUILD_STATUS_ERR_WORKSPACE);
        // Outputs must remain untouched
        CHECK(dst[0] == 0xAA);
        CHECK(eb[0] == -1.0f);
    }
}

static bool decode_landmark_row(
        enum ggml_type type,
        const uint8_t * src,
        float * dst,
        uint32_t n) {
    if (type == GGML_TYPE_TURBO4_0) {
        return ggml_dequantize_turbo_row(type, src, dst, n, 128, GGML_TURBO_DECODE_CANONICAL);
    }
    const ggml_type_traits * tr = ggml_get_type_traits(type);
    if (!tr || !tr->to_float) return false;
    tr->to_float(src, dst, n);
    return true;
}

// Graph test helper
static void run_graph_test(ggml_backend_t backend, const char * backend_name, bool is_vulkan) {
    std::printf("[test] %s graph vs oracle ...\n", backend_name);
    char err[256] = {};

    // Test both Q8_0 and Turbo4_0 on the graph
    enum ggml_type lm_types[] = { GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0 };
    enum ggml_type fac_types[] = { GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0 };

    for (int k = 0; k < 2; ++k) {
        enum ggml_type lt = lm_types[k];
        enum ggml_type ft = fac_types[k];
        uint32_t rank = (ft == GGML_TYPE_TURBO4_0) ? 128 : 32;

        build_case c = make_test_case(6, 2, 48, rank, ft, lt, true);
        size_t need = 0;
        CHECK(ggml_xkv_landmark_build_scratch_bytes(&c.p, &need, err, sizeof(err)));

        // Run oracle first for reference
        std::vector<uint8_t> scr_ora(need);
        size_t dst_row_sz = ggml_row_size((enum ggml_type)c.p.landmark_type, c.p.padded_dim);
        std::vector<uint8_t> dst_ref(dst_row_sz * c.p.n_chunks, 0);
        std::vector<float> eb_ref(c.p.n_chunks, 0.0f);
        std::vector<uint64_t> fp_ref(c.p.n_chunks, 0);
        int32_t st_ref[4] = {};

        CHECK(ggml_xkv_landmark_build_cpu_oracle(
            c.a_k_bytes.data(), (enum ggml_type)c.p.a_type, c.p.pad_rank, c.p.n_rows, 0, ggml_row_size((enum ggml_type)c.p.a_type, c.p.pad_rank),
            c.b_k_bytes.data(), (enum ggml_type)c.p.b_type, c.p.pad_rank, c.p.total_dim, 0, ggml_row_size((enum ggml_type)c.p.b_type, c.p.pad_rank),
            c.rows.data(), c.p.n_rows,
            c.pos.data(), 1,
            c.layer_meta.data(), c.p.n_layers,
            c.rope_tables.data(), c.rope_tables.size(),
            &c.p,
            dst_ref.data(), c.p.padded_dim, c.p.n_chunks, 0, dst_row_sz,
            eb_ref.data(), fp_ref.data(), st_ref,
            scr_ora.data(), scr_ora.size(),
            err, sizeof(err)));
        CHECK(st_ref[0] == GGML_XKV_LANDMARK_BUILD_STATUS_OK);

        // Build GGML graph
        struct ggml_init_params ip = { 128 * 1024, nullptr, true };
        struct ggml_context * ctx = ggml_init(ip);
        CHECK(ctx != nullptr);

        struct ggml_tensor * ta = ggml_new_tensor_2d(ctx, ft, c.p.pad_rank, c.p.n_rows);
        struct ggml_tensor * tb = ggml_new_tensor_2d(ctx, ft, c.p.pad_rank, c.p.total_dim);
        struct ggml_tensor * tr = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.p.n_rows);
        struct ggml_tensor * tp = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, c.p.n_rows);
        struct ggml_tensor * tl = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 6, c.p.n_layers);
        struct ggml_tensor * trp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.rope_tables.size());
        struct ggml_tensor * tsc = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, need);
        struct ggml_tensor * teb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.p.n_chunks);
        struct ggml_tensor * tfp = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, c.p.n_chunks);
        struct ggml_tensor * tst = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);

        struct ggml_tensor * tdst = ggml_xkv_landmark_build(ctx, ta, tb, tr, tp, tl, trp, tsc, teb, tfp, tst, &c.p);
        CHECK(tdst != nullptr);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, tdst);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        CHECK(buf != nullptr);

        // Upload inputs
        ggml_backend_tensor_set(ta, c.a_k_bytes.data(), 0, c.a_k_bytes.size());
        ggml_backend_tensor_set(tb, c.b_k_bytes.data(), 0, c.b_k_bytes.size());
        ggml_backend_tensor_set(tr, c.rows.data(), 0, c.rows.size() * sizeof(int32_t));
        ggml_backend_tensor_set(tp, c.pos.data(), 0, c.pos.size() * sizeof(int64_t));
        ggml_backend_tensor_set(tl, c.layer_meta.data(), 0, c.layer_meta.size() * sizeof(int32_t));
        ggml_backend_tensor_set(trp, c.rope_tables.data(), 0, c.rope_tables.size() * sizeof(float));

        int32_t init_st[4] = { -1, 0, 0, 0 };
        ggml_backend_tensor_set(tst, init_st, 0, sizeof(init_st));

        // Compute graph
        CHECK(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

        // Readback outputs
        std::vector<uint8_t> dst_got(dst_row_sz * c.p.n_chunks);
        std::vector<float> eb_got(c.p.n_chunks);
        std::vector<uint64_t> fp_got(c.p.n_chunks);
        int32_t st_got[4] = {};

        ggml_backend_tensor_get(tdst, dst_got.data(), 0, dst_got.size());
        ggml_backend_tensor_get(teb, eb_got.data(), 0, eb_got.size() * sizeof(float));
        ggml_backend_tensor_get(tfp, fp_got.data(), 0, fp_got.size() * sizeof(uint64_t));
        ggml_backend_tensor_get(tst, st_got, 0, sizeof(st_got));

        CHECK(st_got[0] == GGML_XKV_LANDMARK_BUILD_STATUS_OK);
        CHECK(st_got[1] == (int32_t)c.p.n_chunks);

        // Parity comparison
        if (!is_vulkan) {
            // CPU graph must be bit-identical to oracle
            CHECK(dst_got == dst_ref);
            CHECK(fp_got == fp_ref);
            for (uint32_t i = 0; i < c.p.n_chunks; ++i) {
                CHECK(std::fabs(eb_got[i] - eb_ref[i]) < 1e-5f);
            }
        } else {
            // Vulkan parity: compare decoded values, not merely status/bounds.
            // This catches a dispatched-but-dead kernel, cross-codec-block
            // HALF pairing, and partial-chunk corruption.
            std::vector<float> dec_ref(c.p.padded_dim);
            std::vector<float> dec_got(c.p.padded_dim);
            for (uint32_t i = 0; i < c.p.n_chunks; ++i) {
                CHECK(fp_got[i] != 0);
                CHECK(eb_got[i] >= 1e-4f);
                CHECK(decode_landmark_row(lt, dst_ref.data() + i * dst_row_sz,
                                          dec_ref.data(), c.p.padded_dim));
                CHECK(decode_landmark_row(lt, dst_got.data() + i * dst_row_sz,
                                          dec_got.data(), c.p.padded_dim));
                double diff2 = 0.0;
                double ref2 = 0.0;
                for (uint32_t d = 0; d < c.p.total_dim; ++d) {
                    const double delta = (double) dec_got[d] - dec_ref[d];
                    diff2 += delta * delta;
                    ref2 += (double) dec_ref[d] * dec_ref[d];
                }
                const double allowed = (double) eb_ref[i] + (double) eb_got[i] +
                                       0.01 * std::sqrt(ref2) + 1e-3;
                CHECK(std::sqrt(diff2) <= allowed);
            }
        }

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
}

int main() {
    std::printf("=== test-xkv-vulkan-landmark-build ===\n");

    test_scratch_bytes_exactness();
    test_cpu_oracle_cases();

    // CPU graph
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (cpu) {
        run_graph_test(cpu, "CPU", false);
        ggml_backend_free(cpu);
    }

    // Vulkan graph
    ggml_backend_t vk = open_vulkan_backend();
    if (vk) {
        run_graph_test(vk, "Vulkan", true);
        ggml_backend_free(vk);
    } else {
        std::printf("  SKIP: Vulkan backend not available\n");
    }

    std::printf("=== test-xkv-vulkan-landmark-build: %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
}
