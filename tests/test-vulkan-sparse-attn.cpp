// tests/test-vulkan-sparse-attn.cpp
//
// Hermetic backend regression test for Vulkan sparse Flash Attention (PR27970).
// Tests FLASH_ATTN_EXT op with n_kv_max > 0 (compaction + sparse execution)
// against CPU reference on identical quantized bytes.
//
// Checks:
//   1. Observability: verifies ggml_backend_vk_get_sparse_dispatch_count increments
//      on sparse dispatches (n_kv_max > 0) and does not increment when n_kv_max == 0 (dense).
//   2. Parity: compares dense (hint 0) and sparse (hint > 0) Vulkan results against
//      the numerical CPU oracle on identical quantized K/V bytes.
//   3. KV types: Q8_0 and TURBO4_0 target KV, plus F16 reference.
//   4. Mask structures & topologies:
//      - Multiple queries (prefill Nq > 1 and decode Nq == 1)
//      - Multiple streams (Nstream >= 1)
//      - GQA configurations (Hq > Hkv)
//      - All-masked rows (all -INFINITY -> exact zero output)
//      - Finite boundary values: -65504.0f (must NOT be dropped or treated as -inf)
//      - Finite count overflow: finite entries > n_kv_max (overflow fallback row)
//      - Odd tail active counts
//      - Non-zero view offsets on K/V tensors
//      - Mask mutation across repeated graph executions (verifies no stale indices in cache)
//   5. Hermetic bounds: bounded shapes (few MiB worst-case per device), single GPU by default.
//
// Usage: test-vulkan-sparse-attn [--device <int>] [--all-devices]

#include "ggml.h"
#include "../ggml/src/ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

typedef uint64_t (*get_sparse_dispatch_count_t)(ggml_backend_t);

static int g_failures = 0;

#define CHECK_TRUE(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL [%s:%d]: %s (%s)\n", __FILE__, __LINE__, #cond, msg); \
        g_failures++; \
        return false; \
    } \
} while (0)

#define CHECK_CLOSE(a, b, tol, msg) do { \
    float diff = std::fabs((a) - (b)); \
    if (std::isnan(a) || std::isnan(b) || diff > (tol)) { \
        std::fprintf(stderr, "FAIL [%s:%d]: |%f - %f| = %f > tol=%f (%s)\n", \
                     __FILE__, __LINE__, (double)(a), (double)(b), (double)diff, (double)(tol), msg); \
        g_failures++; \
        return false; \
    } \
} while (0)

struct test_config {
    int64_t hsk = 128;
    int64_t hsv = 128;
    int64_t nq = 1;
    int64_t nkv = 64;
    int64_t hq = 4;
    int64_t hkv = 2;
    int64_t nstream = 1;
    int64_t mask_ne1 = 0; // 0 means default to cfg.nq; > 0 tests padded mask rows (e.g. 32 > 1 or 32 > 4)
    int32_t n_kv_max = 32; // op_params[4]
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;
    bool all_masked = false;
    bool test_minus_65504 = false;
    bool force_overflow = false;
    bool odd_tail = false;
    bool use_view_offset = false;
    // Attach per-Q-head sinks to the FA node (exercises SINK_ENABLE_BIT bit 24
    // coexisting with the TP5 headmap flag bit 25).
    bool use_sinks = false;
    // TP5 QSA explicit head map (opt-in): per-local-Q-head KV head indices,
    // e.g. {0,0,1,1,1} for local Q5 over KV2 where the integer ratio (5/2)
    // is NOT the true assignment. Empty = ordinary node.
    std::vector<uint32_t> head_map;
    float scale = 0.0f; // 0 -> 1/sqrt(hsk)
};

// Encode float row to target quantization format (deterministic CPU quantizers)
static bool quantize_row_data(ggml_type type, const float * src, void * dst, int64_t n) {
    if (type == GGML_TYPE_F32) {
        std::memcpy(dst, src, n * sizeof(float));
        return true;
    }
    if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src, (ggml_fp16_t *) dst, n);
        return true;
    }
    if (type == GGML_TYPE_Q8_0) {
        const auto * traits = ggml_get_type_traits(type);
        if (!traits || !traits->from_float_ref) return false;
        traits->from_float_ref(src, dst, n);
        return true;
    }
    if (type == GGML_TYPE_TURBO4_0) {
        return ggml_quantize_turbo_row(type, src, dst, n, 128);
    }
    return false;
}

// Build a graph containing FLASH_ATTN_EXT
struct fa_graph_fixture {
    ggml_init_params ip;
    ggml_context * ctx = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * m = nullptr;
    // Pre-matmul projection input for replay eligibility (batch == 1 decode)
    ggml_tensor * x_proj = nullptr;
    ggml_tensor * w_proj = nullptr;
    bool with_replay_matmul = false;
    ggml_tensor * out = nullptr;
    ggml_tensor * sinks = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    // View parents if offset is tested
    ggml_tensor * k_parent = nullptr;
    ggml_tensor * v_parent = nullptr;

    fa_graph_fixture(const test_config & cfg, ggml_backend_t backend, bool add_replay_matmul = false) : with_replay_matmul(add_replay_matmul) {
        size_t mem_overhead = ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false);
        ip = { mem_overhead, nullptr, true };
        ctx = ggml_init(ip);
        if (!ctx) return;

        const float scale = cfg.scale > 0.0f ? cfg.scale : (1.0f / std::sqrt((float)cfg.hsk));

        q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cfg.hsk, cfg.nq, cfg.hq, cfg.nstream);
        ggml_set_name(q, "q");

        if (cfg.use_view_offset) {
            // Create properly typed 1D quantized parents for K and V with sufficient physical rows.
            // Alignment must satisfy both the 256-byte Vulkan device storage buffer alignment
            // and whole quantized row boundaries: offset = LCM(256, row_size).
            const size_t k_row_size = ggml_row_size(cfg.type_k, cfg.hsk);
            const size_t v_row_size = ggml_row_size(cfg.type_v, cfg.hsv);

            auto compute_lcm = [](size_t a, size_t b) -> size_t {
                size_t x = a, y = b;
                while (y != 0) { size_t t = y; y = x % y; x = t; }
                return (a / x) * b;
            };

            const size_t k_off = compute_lcm(256, k_row_size);
            const size_t v_off = compute_lcm(256, v_row_size);

            const size_t k_extra_rows = k_off / k_row_size;
            const size_t v_extra_rows = v_off / v_row_size;

            const size_t k_active_rows = (size_t)cfg.nkv * cfg.hkv * cfg.nstream;
            const size_t v_active_rows = (size_t)cfg.nkv * cfg.hkv * cfg.nstream;

            k_parent = ggml_new_tensor_1d(ctx, cfg.type_k, (int64_t)((k_active_rows + k_extra_rows) * cfg.hsk));
            v_parent = ggml_new_tensor_1d(ctx, cfg.type_v, (int64_t)((v_active_rows + v_extra_rows) * cfg.hsv));

            const size_t k_nb1 = k_row_size;
            const size_t k_nb2 = k_nb1 * cfg.nkv;
            const size_t k_nb3 = k_nb2 * cfg.hkv;
            k = ggml_view_4d(ctx, k_parent, cfg.hsk, cfg.nkv, cfg.hkv, cfg.nstream, k_nb1, k_nb2, k_nb3, k_off);

            const size_t v_nb1 = v_row_size;
            const size_t v_nb2 = v_nb1 * cfg.nkv;
            const size_t v_nb3 = v_nb2 * cfg.hkv;
            v = ggml_view_4d(ctx, v_parent, cfg.hsv, cfg.nkv, cfg.hkv, cfg.nstream, v_nb1, v_nb2, v_nb3, v_off);
        } else {
            k = ggml_new_tensor_4d(ctx, cfg.type_k, cfg.hsk, cfg.nkv, cfg.hkv, cfg.nstream);
            v = ggml_new_tensor_4d(ctx, cfg.type_v, cfg.hsv, cfg.nkv, cfg.hkv, cfg.nstream);
        }
        ggml_set_name(k, "k");
        ggml_set_name(v, "v");

        const int64_t m_ne1 = cfg.mask_ne1 > 0 ? cfg.mask_ne1 : cfg.nq;
        m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cfg.nkv, m_ne1, 1, cfg.nstream);
        ggml_set_name(m, "m");

        ggml_tensor * q_in = q;
        if (add_replay_matmul) {
            // Prepend a 1x1 matmul: q_proj = mul_mat(w_proj, x_proj) where x_proj has batch == 1
            // This makes the single-token decode graph eligible for Vulkan command replay
            w_proj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.hsk, cfg.hsk);
            x_proj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.hsk, 1);
            ggml_tensor * proj = ggml_mul_mat(ctx, w_proj, x_proj); // [hsk, 1]
            // Add proj to q to form a valid computational dependency
            q_in = ggml_add(ctx, q, proj);
        }

        if (cfg.head_map.empty()) {
            out = ggml_flash_attn_ext(ctx, q_in, k, v, m, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
            ggml_flash_attn_ext_set_n_kv_max(out, cfg.n_kv_max);
        } else {
            // Mapped fixture: build the FLASH_ATTN_EXT node directly.
            // ggml_flash_attn_ext asserts q->ne[2] % k->ne[2] == 0, which is
            // exactly the uniform-ratio assumption the TP5 head map exists
            // to break (5 Q heads over 2 KV heads, map [0,0,1,1,1]).
            const int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
            out = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
            float params[3] = { scale, 0.0f, 0.0f };
            ggml_set_op_params(out, params, sizeof(params));
            out->op = GGML_OP_FLASH_ATTN_EXT;
            out->src[0] = q_in;
            out->src[1] = k;
            out->src[2] = v;
            out->src[3] = m;
            ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
            ggml_flash_attn_ext_set_n_kv_max(out, cfg.n_kv_max);
        }
        if (!cfg.head_map.empty()) {
            // TP5 private local head map via the canonical setter
            // (ggml-backend.h ABI: slots 8/9/10). ggml_flash_attn_ext only
            // initializes slots 0..2 (set_prec/set_n_kv_max use 3 and 4).
            GGML_ASSERT(cfg.head_map.size() == (size_t) cfg.hq && cfg.head_map.size() <= 10);
            for (size_t h = 0; h < cfg.head_map.size(); ++h) {
                GGML_ASSERT(cfg.head_map[h] < 8 && (int64_t) cfg.head_map[h] < cfg.hkv);
            }
            uint8_t local_heads[GGML_TP5_HEADMAP_MAX_ENTRIES];
            for (size_t h = 0; h < cfg.head_map.size(); ++h) {
                local_heads[h] = (uint8_t) cfg.head_map[h];
            }
            ggml_tp5_headmap_set(out, local_heads, (int32_t) cfg.head_map.size());
        }
        if (cfg.use_sinks) {
            // Sinks are indexed by Q head (never by KV head), so they coexist
            // with the head map: SINK_ENABLE_BIT (24) and TP5_HEADMAP_FLAG
            // (25) live in different bits of mask_n_head_log2.
            sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.hq);
            ggml_flash_attn_ext_add_sinks(out, sinks);
        }
        ggml_set_name(out, "out");

        gf = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(gf, out);

        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }

    ~fa_graph_fixture() {
        if (buf) {
            ggml_backend_buffer_free(buf);
            buf = nullptr;
        }
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
    }
};

// Generate test data for Q, K, V, M
struct test_data {
    std::vector<float> q_f32;
    std::vector<uint8_t> k_bytes;
    std::vector<uint8_t> v_bytes;
    std::vector<ggml_fp16_t> mask_f16;
};

static test_data generate_test_inputs(const test_config & cfg, int seed = 1234) {
    test_data td;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist_q(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_kv(-1.5f, 1.5f);

    // Q: [hsk, nq, hq, nstream]
    const size_t q_elems = (size_t)cfg.hsk * cfg.nq * cfg.hq * cfg.nstream;
    td.q_f32.resize(q_elems);
    for (size_t i = 0; i < q_elems; ++i) {
        td.q_f32[i] = dist_q(rng);
    }

    // K: [hsk, nkv, hkv, nstream]
    const size_t k_rows = (size_t)cfg.nkv * cfg.hkv * cfg.nstream;
    const size_t k_row_size = ggml_row_size(cfg.type_k, cfg.hsk);
    td.k_bytes.resize(k_rows * k_row_size);
    std::vector<float> row_tmp_k(cfg.hsk);
    for (size_t r = 0; r < k_rows; ++r) {
        for (int64_t d = 0; d < cfg.hsk; ++d) {
            row_tmp_k[d] = dist_kv(rng);
        }
        bool ok = quantize_row_data(cfg.type_k, row_tmp_k.data(), td.k_bytes.data() + r * k_row_size, cfg.hsk);
        GGML_ASSERT(ok && "quantize K row failed");
    }

    // V: [hsv, nkv, hkv, nstream]
    const size_t v_rows = (size_t)cfg.nkv * cfg.hkv * cfg.nstream;
    const size_t v_row_size = ggml_row_size(cfg.type_v, cfg.hsv);
    td.v_bytes.resize(v_rows * v_row_size);
    std::vector<float> row_tmp_v(cfg.hsv);
    for (size_t r = 0; r < v_rows; ++r) {
        for (int64_t d = 0; d < cfg.hsv; ++d) {
            row_tmp_v[d] = dist_kv(rng);
        }
        bool ok = quantize_row_data(cfg.type_v, row_tmp_v.data(), td.v_bytes.data() + r * v_row_size, cfg.hsv);
        GGML_ASSERT(ok && "quantize V row failed");
    }

    // Mask: [nkv, m_ne1, 1, nstream]
    const int64_t m_ne1 = cfg.mask_ne1 > 0 ? cfg.mask_ne1 : cfg.nq;
    const size_t mask_elems = (size_t)cfg.nkv * m_ne1 * 1 * cfg.nstream;
    // Fill entire mask with -INFINITY by default so padding rows are explicitly masked out
    td.mask_f16.assign(mask_elems, ggml_fp32_to_fp16(-INFINITY));

    for (int64_t s = 0; s < cfg.nstream; ++s) {
        for (int64_t q_idx = 0; q_idx < cfg.nq; ++q_idx) {
            const size_t row_base = (size_t)(s * m_ne1 + q_idx) * cfg.nkv;

            if (cfg.all_masked) {
                for (int64_t kv_idx = 0; kv_idx < cfg.nkv; ++kv_idx) {
                    td.mask_f16[row_base + kv_idx] = ggml_fp32_to_fp16(-INFINITY);
                }
            } else if (cfg.force_overflow) {
                // Ensure number of finite entries exceeds n_kv_max
                const int count = std::min<int>((int)cfg.nkv, cfg.n_kv_max + 8);
                for (int64_t kv_idx = 0; kv_idx < cfg.nkv; ++kv_idx) {
                    if (kv_idx < count) {
                        td.mask_f16[row_base + kv_idx] = ggml_fp32_to_fp16(0.0f);
                    } else {
                        td.mask_f16[row_base + kv_idx] = ggml_fp32_to_fp16(-INFINITY);
                    }
                }
            } else if (cfg.odd_tail) {
                // Keep an odd number of finite elements (e.g. 7, 13)
                const int odd_count = std::min<int>((int)cfg.nkv, (cfg.n_kv_max > 7 ? 7 : 3));
                for (int64_t kv_idx = 0; kv_idx < cfg.nkv; ++kv_idx) {
                    if (kv_idx < odd_count) {
                        td.mask_f16[row_base + kv_idx] = ggml_fp32_to_fp16(0.0f);
                    } else {
                        td.mask_f16[row_base + kv_idx] = ggml_fp32_to_fp16(-INFINITY);
                    }
                }
            } else if (cfg.test_minus_65504) {
                // Make ALL selected mask entries -65504.0f, rest -INFINITY.
                // This guarantees that if -65504 is mistakenly treated as -inf or dropped,
                // the row behaves as all-masked or missing keys, causing significant parity failure.
                for (int64_t kv_idx = 0; kv_idx < cfg.nkv; ++kv_idx) {
                    if (kv_idx < 4) {
                        td.mask_f16[row_base + kv_idx] = ggml_fp32_to_fp16(-65504.0f);
                    } else {
                        td.mask_f16[row_base + kv_idx] = ggml_fp32_to_fp16(-INFINITY);
                    }
                }
            } else {
                // Standard causal or bounded window: distance <= n_kv_max
                for (int64_t kv_idx = 0; kv_idx < cfg.nkv; ++kv_idx) {
                    int64_t dist = q_idx + (cfg.nkv - cfg.nq) - kv_idx;
                    if (dist >= 0 && dist < cfg.n_kv_max) {
                        td.mask_f16[row_base + kv_idx] = ggml_fp32_to_fp16(0.0f);
                    } else {
                        td.mask_f16[row_base + kv_idx] = ggml_fp32_to_fp16(-INFINITY);
                    }
                }
            }
        }
    }

    return td;
}

// Load test inputs into fixture tensors
static void upload_fixture_inputs(fa_graph_fixture & fix, const test_config & cfg, const test_data & td) {
    if (fix.with_replay_matmul && fix.w_proj && fix.x_proj) {
        std::vector<float> w_zeros(cfg.hsk * cfg.hsk, 0.0f);
        std::vector<float> x_ones(cfg.hsk, 1.0f);
        // w is identity-like or zeros so proj is 0 or controlled
        for (int64_t i = 0; i < cfg.hsk; ++i) {
            w_zeros[i * cfg.hsk + i] = 0.01f;
        }
        ggml_backend_tensor_set(fix.w_proj, w_zeros.data(), 0, w_zeros.size() * sizeof(float));
        ggml_backend_tensor_set(fix.x_proj, x_ones.data(), 0, x_ones.size() * sizeof(float));
    }

    if (fix.sinks) {
        std::vector<float> sink_vals((size_t) cfg.hq);
        for (int64_t h = 0; h < cfg.hq; ++h) {
            sink_vals[h] = -2.0f + 0.5f * (float) h;
        }
        ggml_backend_tensor_set(fix.sinks, sink_vals.data(), 0, sink_vals.size() * sizeof(float));
    }

    ggml_backend_tensor_set(fix.q, td.q_f32.data(), 0, td.q_f32.size() * sizeof(float));

    if (cfg.use_view_offset) {
        // Initialize full backing parent first so no uninitialized bytes exist before or after the view window
        std::vector<uint8_t> k_parent_zeros(ggml_nbytes(fix.k_parent), 0);
        std::vector<uint8_t> v_parent_zeros(ggml_nbytes(fix.v_parent), 0);
        ggml_backend_tensor_set(fix.k_parent, k_parent_zeros.data(), 0, k_parent_zeros.size());
        ggml_backend_tensor_set(fix.v_parent, v_parent_zeros.data(), 0, v_parent_zeros.size());

        // Upload active rows into view parent at the view's exact byte view_offs
        ggml_backend_tensor_set(fix.k_parent, td.k_bytes.data(), fix.k->view_offs, td.k_bytes.size());
        ggml_backend_tensor_set(fix.v_parent, td.v_bytes.data(), fix.v->view_offs, td.v_bytes.size());
    } else {
        ggml_backend_tensor_set(fix.k, td.k_bytes.data(), 0, td.k_bytes.size());
        ggml_backend_tensor_set(fix.v, td.v_bytes.data(), 0, td.v_bytes.size());
    }

    ggml_backend_tensor_set(fix.m, td.mask_f16.data(), 0, td.mask_f16.size() * sizeof(ggml_fp16_t));
}

// Run graph and read back output floats
static bool compute_and_read_output(fa_graph_fixture & fix, ggml_backend_t backend, std::vector<float> & out) {
    const ggml_status status = ggml_backend_graph_compute(backend, fix.gf);
    if (status != GGML_STATUS_SUCCESS) return false;
    ggml_backend_synchronize(backend);

    const size_t n_out = ggml_nelements(fix.out);
    out.resize(n_out);
    ggml_backend_tensor_get(fix.out, out.data(), 0, n_out * sizeof(float));
    return true;
}

// Compare two float vectors with NMSE and check for NaNs / infinities
static bool check_results_parity(const std::vector<float> & actual, const std::vector<float> & expected,
                                 double max_nmse_tol, float max_abs_tol, const char * name) {
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "FAIL [%s]: size mismatch: actual=%zu expected=%zu\n", name, actual.size(), expected.size());
        g_failures++;
        return false;
    }

    double mse_diff = 0.0;
    double mse_ref = 0.0;
    float max_diff = 0.0f;

    for (size_t i = 0; i < actual.size(); ++i) {
        float a = actual[i];
        float e = expected[i];

        if (std::isnan(a) || std::isnan(e) || std::isinf(a) || std::isinf(e)) {
            std::fprintf(stderr, "FAIL [%s]: Non-finite value detected at index %zu: actual=%f, expected=%f\n", name, i, a, e);
            g_failures++;
            return false;
        }

        float diff = std::fabs(a - e);
        if (diff > max_diff) max_diff = diff;
        mse_diff += (double)diff * (double)diff;
        mse_ref += (double)e * (double)e;
    }

    double nmse = (mse_ref > 1e-12) ? (mse_diff / mse_ref) : mse_diff;

    if (nmse > max_nmse_tol && max_diff > max_abs_tol) {
        std::fprintf(stderr, "FAIL [%s]: NMSE=%e > tol=%e, max_diff=%e > tol=%e\n",
                     name, nmse, max_nmse_tol, (double)max_diff, (double)max_abs_tol);
        g_failures++;
        return false;
    }

    return true;
}

// Test case executor
static bool run_test_case(ggml_backend_t backend_gpu, ggml_backend_t backend_cpu,
                         get_sparse_dispatch_count_t get_sparse_count,
                         const test_config & cfg, const char * case_name) {
    std::printf("Running case: %s (Nq=%ld, Nkv=%ld, Hq=%ld, Hkv=%ld, Nstream=%ld, n_kv_max=%d, K=%s, V=%s%s)...\n",
                case_name, (long)cfg.nq, (long)cfg.nkv, (long)cfg.hq, (long)cfg.hkv, (long)cfg.nstream,
                cfg.n_kv_max, ggml_type_name(cfg.type_k), ggml_type_name(cfg.type_v),
                cfg.head_map.empty() ? "" : " TP5HEADMAP");

    test_data td = generate_test_inputs(cfg, 42);

    // 1. Run CPU reference on identical quantized bytes (hint 0 dense reference)
    test_config cfg_cpu = cfg;
    cfg_cpu.n_kv_max = 0; // CPU executes standard FA
    fa_graph_fixture fix_cpu(cfg_cpu, backend_cpu);
    CHECK_TRUE(fix_cpu.ctx && fix_cpu.buf, "CPU fixture allocation failed");
    upload_fixture_inputs(fix_cpu, cfg_cpu, td);
    std::vector<float> out_cpu;
    CHECK_TRUE(compute_and_read_output(fix_cpu, backend_cpu, out_cpu), "CPU graph compute failed");

    // 2. Query GPU sparse dispatch counter before GPU run
    uint64_t count_before = get_sparse_count ? get_sparse_count(backend_gpu) : 0;

    // 3. Run GPU with n_kv_max (sparse path)
    fa_graph_fixture fix_gpu(cfg, backend_gpu);
    CHECK_TRUE(fix_gpu.ctx && fix_gpu.buf, "GPU fixture allocation failed");
    upload_fixture_inputs(fix_gpu, cfg, td);
    std::vector<float> out_gpu_sparse;
    CHECK_TRUE(compute_and_read_output(fix_gpu, backend_gpu, out_gpu_sparse), "GPU sparse graph compute failed");

    // 4. Query GPU sparse dispatch counter after GPU run
    uint64_t count_after = get_sparse_count ? get_sparse_count(backend_gpu) : 0;

    if (cfg.n_kv_max > 0) {
        // Must have taken actual sparse branch
        CHECK_TRUE(count_after > count_before, "Observable sparse dispatch counter did not increment on n_kv_max > 0");
    } else {
        // Dense run must not increment sparse counter
        CHECK_TRUE(count_after == count_before, "Dense run erroneously incremented sparse dispatch counter");
    }

    // 5. If all_masked, verify output is exact zeros
    if (cfg.all_masked) {
        for (size_t i = 0; i < out_gpu_sparse.size(); ++i) {
            CHECK_CLOSE(out_gpu_sparse[i], 0.0f, 1e-6f, "All-masked row did not produce exact zero");
        }
    }

    // 6. Compare GPU sparse against CPU reference
    // Existing FA tolerances: 5e-4 NMSE for quantized/fp16 FA, 1e-3 max abs diff
    const double tol_nmse = (cfg.type_k == GGML_TYPE_F32) ? 1e-6 : 5e-4;
    const float tol_abs = 2e-3f;
    CHECK_TRUE(check_results_parity(out_gpu_sparse, out_cpu, tol_nmse, tol_abs, case_name), "GPU sparse vs CPU reference parity failed");

    // 7. Also compare GPU dense (n_kv_max = 0) vs GPU sparse to verify identical semantics
    uint64_t count_dense_before = get_sparse_count ? get_sparse_count(backend_gpu) : 0;
    test_config cfg_gpu_dense = cfg;
    cfg_gpu_dense.n_kv_max = 0;
    fa_graph_fixture fix_gpu_dense(cfg_gpu_dense, backend_gpu);
    CHECK_TRUE(fix_gpu_dense.ctx && fix_gpu_dense.buf, "GPU dense fixture allocation failed");
    upload_fixture_inputs(fix_gpu_dense, cfg_gpu_dense, td);
    std::vector<float> out_gpu_dense;
    CHECK_TRUE(compute_and_read_output(fix_gpu_dense, backend_gpu, out_gpu_dense), "GPU dense graph compute failed");
    uint64_t count_dense_after = get_sparse_count ? get_sparse_count(backend_gpu) : 0;
    CHECK_TRUE(count_dense_after == count_dense_before, "Dense GPU execution unexpectedly incremented sparse dispatch counter");

    CHECK_TRUE(check_results_parity(out_gpu_sparse, out_gpu_dense, 1e-5, 1e-3f, "GPU sparse vs GPU dense parity"),
               "GPU sparse vs GPU dense parity failed");

    std::printf("  Case %s passed (sparse dispatches: +%llu).\n", case_name, (unsigned long long)(count_after - count_before));
    return true;
}

// Test mask changes on repeated graph executions (verifying cache/replay integrity)
static bool test_repeated_graph_mask_mutation(ggml_backend_t backend_gpu, ggml_backend_t backend_cpu,
                                              get_sparse_dispatch_count_t get_sparse_count) {
    std::printf("Running test_repeated_graph_mask_mutation with decode replay...\n");
    test_config cfg;
    cfg.hsk = 128;
    cfg.hsv = 128;
    cfg.nq = 1; // single query decode batch == 1
    cfg.nkv      = 512;
    cfg.hq = 4;
    cfg.hkv = 2;
    cfg.nstream = 1;
    cfg.n_kv_max = 300;
    cfg.type_k = GGML_TYPE_Q8_0;
    cfg.type_v = GGML_TYPE_TURBO4_0;

    fa_graph_fixture fix_gpu(cfg, backend_gpu, true); // with replay matmul
    test_config      cfg_offset = cfg;
    cfg_offset.use_view_offset  = true;
    fa_graph_fixture fix_gpu_offset(cfg_offset, backend_gpu, true);
    fa_graph_fixture fix_cpu(cfg, backend_cpu, true);
    fix_cpu.out->op_params[4] = 0; // CPU dense reference

    typedef void (*get_replay_stats_t)(ggml_backend_t, uint64_t *, uint64_t *);
    ggml_backend_reg_t reg_gpu = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_gpu));
    get_replay_stats_t get_replay_stats = reg_gpu ?
        (get_replay_stats_t) ggml_backend_reg_get_proc_address(reg_gpu, "ggml_backend_vk_get_replay_stats") : nullptr;

    uint64_t replay_hits_start = 0, replay_misses_start = 0;
    if (get_replay_stats) get_replay_stats(backend_gpu, &replay_hits_start, &replay_misses_start);

    // Generate fixed Q, K, V data once so that only the mask mutates across iterations
    test_data td = generate_test_inputs(cfg, 555);

    for (int step = 0; step < 4; ++step) {
        // Mutate the mask specifically in each step:
        // Cross both subgroup and 256-key compaction tile boundaries.
        // The same logical data in different allocations/views must retain
        // exactly the same attention reduction order.
        for (int64_t kv = 0; kv < cfg.nkv; ++kv) {
            bool active = false;
            if (step == 0)
                active = (kv < 65);
            else if (step == 1)
                active = (kv >= 383);
            else if (step == 2)
                active = (kv % 2 == 0);
            else
                active = (kv >= 10 && kv < 287);
            td.mask_f16[kv] = ggml_fp32_to_fp16(active ? 0.0f : -INFINITY);
        }

        upload_fixture_inputs(fix_gpu, cfg, td);
        upload_fixture_inputs(fix_gpu_offset, cfg_offset, td);
        upload_fixture_inputs(fix_cpu, cfg, td);

        std::vector<float> out_gpu;
        std::vector<float> out_gpu_offset;
        std::vector<float> out_cpu;
        CHECK_TRUE(compute_and_read_output(fix_gpu, backend_gpu, out_gpu), "GPU repeated compute failed");
        CHECK_TRUE(compute_and_read_output(fix_gpu_offset, backend_gpu, out_gpu_offset), "GPU offset replay failed");
        CHECK_TRUE(out_gpu.size() == out_gpu_offset.size() &&
                       std::memcmp(out_gpu.data(), out_gpu_offset.data(), out_gpu.size() * sizeof(float)) == 0,
                   "Sparse attention changed with allocation/view placement");
        CHECK_TRUE(compute_and_read_output(fix_cpu, backend_cpu, out_cpu), "CPU repeated compute failed");

        char step_name[64];
        std::snprintf(step_name, sizeof(step_name), "mutation_step_%d", step);
        CHECK_TRUE(check_results_parity(out_gpu, out_cpu, 5e-4, 2e-3f, step_name), "Mutation parity failed");
    }

    if (get_replay_stats) {
        uint64_t replay_hits_end = 0, replay_misses_end = 0;
        get_replay_stats(backend_gpu, &replay_hits_end, &replay_misses_end);
        std::printf("  Replay stats during mask mutation: hits=%llu (+%llu), misses=%llu (+%llu)\n",
                    (unsigned long long)replay_hits_end, (unsigned long long)(replay_hits_end - replay_hits_start),
                    (unsigned long long)replay_misses_end, (unsigned long long)(replay_misses_end - replay_misses_start));
    }

    std::printf("  test_repeated_graph_mask_mutation passed.\n");
    return true;
}

// TP5 head-mapped decode with mask mutation across repeated graph executions:
// verifies no stale head-map or compaction indices survive in the dispatch
// cache, and GPU/CPU parity per step under the explicit map.
static bool test_tp5_headmap_mask_mutation(ggml_backend_t backend_gpu, ggml_backend_t backend_cpu) {
    std::printf("Running test_tp5_headmap_mask_mutation...\n");
    test_config cfg;
    cfg.nkv = 64;
    cfg.hq = 5;
    cfg.hkv = 2;
    cfg.n_kv_max = 32;
    cfg.type_k = GGML_TYPE_Q8_0;
    cfg.type_v = GGML_TYPE_TURBO4_0;
    cfg.head_map = {0, 0, 1, 1, 1};

    test_data td = generate_test_inputs(cfg, 777);
    fa_graph_fixture fix_gpu(cfg, backend_gpu);
    CHECK_TRUE(fix_gpu.ctx && fix_gpu.buf, "mapped fixture allocation failed");
    // CPU oracle convention: dense reference (n_kv_max = 0), matching
    // run_test_case's cfg_cpu. The CPU backend ignores slot 4 anyway.
    test_config cfg_cpu = cfg;
    cfg_cpu.n_kv_max = 0;
    fa_graph_fixture fix_cpu(cfg_cpu, backend_cpu);
    CHECK_TRUE(fix_cpu.ctx && fix_cpu.buf, "mapped CPU fixture allocation failed");

    for (int step = 0; step < 3; ++step) {
        for (int64_t kv = 0; kv < cfg.nkv; ++kv) {
            bool active = (step == 0) ? (kv < 9) : (step == 1) ? (kv >= 40) : (kv % 3 == 0);
            td.mask_f16[kv] = ggml_fp32_to_fp16(active ? 0.0f : -INFINITY);
        }
        upload_fixture_inputs(fix_gpu, cfg, td);
        upload_fixture_inputs(fix_cpu, cfg_cpu, td);
        std::vector<float> out_gpu;
        std::vector<float> out_cpu;
        CHECK_TRUE(compute_and_read_output(fix_gpu, backend_gpu, out_gpu), "mapped GPU compute failed");
        CHECK_TRUE(compute_and_read_output(fix_cpu, backend_cpu, out_cpu), "mapped CPU compute failed");
        char step_name[64];
        std::snprintf(step_name, sizeof(step_name), "tp5_headmap_mutation_step_%d", step);
        CHECK_TRUE(check_results_parity(out_gpu, out_cpu, 5e-4, 2e-3f, step_name), "mapped mutation parity failed");
    }
    std::printf("  test_tp5_headmap_mask_mutation passed.\n");
    return true;
}

// TP5 fail-closed capability oracles: proves the scheduler-facing contract
// (ggml_backend_dev_supports_op returns false, no abort) for every mapped
// combination the Vulkan backend declares unsupported (mapped+ALiBi,
// mapped with a multi-head mask, invalid map), and that valid mapped nodes
// (shared mask or no mask) are accepted.
static bool test_tp5_headmap_fail_closed(ggml_backend_t backend_gpu) {
    std::printf("Running test_tp5_headmap_fail_closed...\n");
    ggml_backend_dev_t dev_gpu = ggml_backend_get_device(backend_gpu);
    CHECK_TRUE(dev_gpu != nullptr, "no Vulkan device for supports_op query");

    const int64_t hsk = 128, nq = 1, nkv = 64, hq = 5, hkv = 2;

    // Build one bare mapped FA node per scenario in a scratch context. The
    // graph is never scheduled; only the capability verdict matters. The
    // tensors are CPU-buffer-backed (data nullptr is fine for a query).
    auto supports_mapped = [&](float max_bias, int64_t mask_ne2, int32_t bad_count) -> bool {
        size_t mem = ggml_tensor_overhead() * 16;
        ggml_init_params ip = { mem, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) return false;

        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hsk, nq, hq, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hsk, nkv, hkv, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hsk, nkv, hkv, 1);
        ggml_tensor * m = (mask_ne2 > 0) ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nkv, nq, mask_ne2, 1) : nullptr;

        const int64_t ne[4] = { v->ne[0], hq, nq, 1 };
        ggml_tensor * out = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
        float params[3] = { 0.1f, max_bias, 0.0f };
        ggml_set_op_params(out, params, sizeof(params));
        out->op = GGML_OP_FLASH_ATTN_EXT;
        out->src[0] = q;
        out->src[1] = k;
        out->src[2] = v;
        out->src[3] = m;

        if (bad_count > 0) {
            // Present-but-invalid: count != q->ne[2]. Must return false from
            // the capability query, never abort the scheduler.
            out->op_params[8]  = (int32_t) GGML_TP5_HEADMAP_MAGIC;
            out->op_params[9]  = 0;
            out->op_params[10] = bad_count;
        } else {
            const uint8_t map5[5] = { 0, 0, 1, 1, 1 };
            ggml_tp5_headmap_set(out, map5, 5);
        }

        const bool supported = ggml_backend_dev_supports_op(dev_gpu, out);
        ggml_free(ctx);
        return supported;
    };

    // Valid mapped node (shared mask) must be SUPPORTED so the scheduler can
    // actually dispatch it on Vulkan.
    CHECK_TRUE(supports_mapped(0.0f, 1, 0), "valid mapped FA node not supported by Vulkan");
    // Valid mapped node, no mask: supported.
    CHECK_TRUE(supports_mapped(0.0f, 0, 0), "mapped FA node without mask not supported");
    // mapped + ALiBi (max_bias > 0): the map overlays m0/m1, so unsupported.
    CHECK_TRUE(!supports_mapped(0.5f, 1, 0), "mapped FA node with ALiBi must be rejected");
    // mapped + per-KV-head mask (ne[2] == hkv): mask-head selection is not
    // map-aware, so unsupported (shared mask ne[2] == 1 is the supported case).
    CHECK_TRUE(!supports_mapped(0.0f, 2, 0), "mapped FA node with multi-head mask must be rejected");
    // Present-but-invalid map (count 7 != q->ne[2] 5): rejected, not aborted.
    CHECK_TRUE(!supports_mapped(0.0f, 1, 7), "invalid mapped FA node must be rejected, not supported");

    std::printf("  test_tp5_headmap_fail_closed passed.\n");
    return true;
}

// Mapped FA with sinks: GPU sparse vs graph-based CPU oracle, cross-checked
// against an INDEPENDENT hand-rolled reference (direct per-head softmax over
// the F16 K/V bytes, computed here, distinct from the CPU backend kernels).
// Proves SINK_ENABLE_BIT (24) coexists with TP5_HEADMAP_FLAG (25) and that the
// sink is indexed by Q head, not by the mapped KV head.
static bool test_tp5_headmap_sinks(ggml_backend_t backend_gpu, ggml_backend_t backend_cpu) {
    std::printf("Running test_tp5_headmap_sinks...\n");
    test_config cfg;
    cfg.hsk = 128;
    cfg.hsv = 128;
    cfg.nq = 1;
    cfg.nkv = 64;
    cfg.hq = 5;
    cfg.hkv = 2;
    cfg.n_kv_max = 32;
    cfg.type_k = GGML_TYPE_F16;
    cfg.type_v = GGML_TYPE_F16;
    cfg.use_sinks = true;
    cfg.head_map = {0, 0, 1, 1, 1};

    test_data td = generate_test_inputs(cfg, 888);

    fa_graph_fixture fix_gpu(cfg, backend_gpu);
    CHECK_TRUE(fix_gpu.ctx && fix_gpu.buf, "sinks GPU fixture allocation failed");
    test_config cfg_cpu = cfg;
    cfg_cpu.n_kv_max = 0;
    fa_graph_fixture fix_cpu(cfg_cpu, backend_cpu);
    CHECK_TRUE(fix_cpu.ctx && fix_cpu.buf, "sinks CPU fixture allocation failed");

    upload_fixture_inputs(fix_gpu, cfg, td);
    upload_fixture_inputs(fix_cpu, cfg_cpu, td);

    std::vector<float> out_gpu;
    std::vector<float> out_cpu;
    CHECK_TRUE(compute_and_read_output(fix_gpu, backend_gpu, out_gpu), "sinks GPU compute failed");
    CHECK_TRUE(compute_and_read_output(fix_cpu, backend_cpu, out_cpu), "sinks CPU compute failed");

    // GPU vs graph-based CPU oracle.
    CHECK_TRUE(check_results_parity(out_gpu, out_cpu, 1e-5, 1e-3f, "tp5_headmap_sinks_gpu_vs_cpu"),
               "mapped+sinks GPU vs CPU parity failed");

    // Independent hand-rolled reference: for each Q head h, softmax over
    //   scores[j] = scale * dot(Q[h], K[map[h]][j]) + mask[j]
    // plus the sink logit sinks[h]; output = sum_j P[j] * V[map[h]][j]
    // (+ sink contributes exp(sinks[h] - M) to the denominator, zero vector
    // to the numerator, exactly the CPU reference's sink convention).
    const float scale = cfg.scale > 0.0f ? cfg.scale : (1.0f / std::sqrt((float) cfg.hsk));
    std::vector<float> sink_vals(cfg.hq);
    for (int64_t h = 0; h < cfg.hq; ++h) {
        sink_vals[h] = -2.0f + 0.5f * (float) h;
    }
    std::vector<float> out_ref(out_cpu.size(), 0.0f);
    for (int64_t h = 0; h < cfg.hq; ++h) {
        const uint32_t kv_head = cfg.head_map[h];
        // scores and running max/sum
        float m = -INFINITY;
        std::vector<float> scores(cfg.nkv);
        std::vector<char> active(cfg.nkv);
        for (int64_t j = 0; j < cfg.nkv; ++j) {
            const ggml_fp16_t mv = td.mask_f16[j];
            const float mvf = ggml_fp16_to_fp32(mv);
            if (mvf == -INFINITY) {
                active[j] = 0;
                scores[j] = -INFINITY;
                continue;
            }
            active[j] = 1;
            float dot = 0.0f;
            for (int64_t d = 0; d < cfg.hsk; ++d) {
                const float qv = td.q_f32[((size_t) h * cfg.nq + 0) * cfg.hsk + d];
                const ggml_fp16_t kf = ((const ggml_fp16_t *) td.k_bytes.data())
                    [((size_t) kv_head * cfg.nkv + j) * cfg.hsk + d];
                dot += qv * ggml_fp16_to_fp32(kf);
            }
            scores[j] = scale * dot + mvf;
            m = std::max(m, scores[j]);
        }
        m = std::max(m, sink_vals[h]);
        float denom = 0.0f;
        std::vector<float> probs(cfg.nkv);
        for (int64_t j = 0; j < cfg.nkv; ++j) {
            probs[j] = active[j] ? std::exp(scores[j] - m) : 0.0f;
            denom += probs[j];
        }
        denom += std::exp(sink_vals[h] - m);
        for (int64_t j = 0; j < cfg.nkv; ++j) {
            if (!active[j]) continue;
            for (int64_t d = 0; d < cfg.hsv; ++d) {
                const ggml_fp16_t vf = ((const ggml_fp16_t *) td.v_bytes.data())
                    [((size_t) kv_head * cfg.nkv + j) * cfg.hsv + d];
                out_ref[((size_t) h * cfg.nq + 0) * cfg.hsv + d] += probs[j] * ggml_fp16_to_fp32(vf);
            }
        }
        for (int64_t d = 0; d < cfg.hsv; ++d) {
            out_ref[((size_t) h * cfg.nq + 0) * cfg.hsv + d] /= denom;
        }
    }

    CHECK_TRUE(check_results_parity(out_gpu, out_ref, 1e-5, 1e-3f, "tp5_headmap_sinks_independent_ref"),
               "mapped+sinks GPU vs independent reference failed");
    CHECK_TRUE(check_results_parity(out_cpu, out_ref, 1e-6, 1e-4f, "tp5_headmap_sinks_cpu_independent_ref"),
               "mapped+sinks CPU vs independent reference failed");

    std::printf("  test_tp5_headmap_sinks passed.\n");
    return true;
}

int main(int argc, char ** argv) {
    int target_device = 0;
    bool all_devices = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--device" && i + 1 < argc) {
            target_device = std::atoi(argv[++i]);
        } else if (a == "--all-devices") {
            all_devices = true;
        }
    }

    ggml_backend_load_all();

    ggml_backend_reg_t reg_vk = ggml_backend_reg_by_name("Vulkan");
    if (!reg_vk) {
        std::printf("SKIP: Vulkan backend registry not found\n");
        return 0;
    }

    size_t dev_count = ggml_backend_reg_dev_count(reg_vk);
    if (dev_count == 0) {
        std::printf("SKIP: No Vulkan devices available\n");
        return 0;
    }

    ggml_backend_reg_t reg_cpu = ggml_backend_reg_by_name("CPU");
    if (!reg_cpu) {
        std::fprintf(stderr, "CPU backend registry not found\n");
        return 1;
    }
    ggml_backend_dev_t dev_cpu = ggml_backend_reg_dev_get(reg_cpu, 0);
    ggml_backend_t backend_cpu = ggml_backend_dev_init(dev_cpu, nullptr);
    if (!backend_cpu) {
        std::fprintf(stderr, "Failed to initialize CPU backend\n");
        return 1;
    }

    get_sparse_dispatch_count_t get_sparse_count =
        (get_sparse_dispatch_count_t) ggml_backend_reg_get_proc_address(reg_vk, "ggml_backend_vk_get_sparse_dispatch_count");

    const size_t num_devices_to_test = all_devices ? dev_count : 1;
    const size_t start_dev = all_devices ? 0 : (size_t)target_device;

    for (size_t d = start_dev; d < start_dev + num_devices_to_test && d < dev_count; ++d) {
        ggml_backend_dev_t dev_gpu = ggml_backend_reg_dev_get(reg_vk, d);
        if (!dev_gpu) continue;

        const char * desc = ggml_backend_dev_description(dev_gpu);
        std::printf("=== Testing Vulkan Device %zu: %s ===\n", d, desc ? desc : "Unknown");

        ggml_backend_t backend_gpu = ggml_backend_dev_init(dev_gpu, nullptr);
        if (!backend_gpu) {
            std::fprintf(stderr, "Failed to initialize Vulkan backend for device %zu\n", d);
            g_failures++;
            continue;
        }

        // Test 1: Single query decode (Nq=1), GQA (Hq=4, Hkv=2), F16 KV
        {
            test_config cfg;
            cfg.hsk = 128;
            cfg.hsv = 128;
            cfg.nq = 1;
            cfg.nkv = 64;
            cfg.hq = 4;
            cfg.hkv = 2;
            cfg.nstream = 1;
            cfg.n_kv_max = 32;
            cfg.type_k = GGML_TYPE_F16;
            cfg.type_v = GGML_TYPE_F16;
            run_test_case(backend_gpu, backend_cpu, get_sparse_count, cfg, "f16_decode_gqa");
        }

        // Test 2: Production Q8_0 K + TURBO4_0 V decode
        {
            test_config cfg;
            cfg.hsk = 128;
            cfg.hsv = 128;
            cfg.nq = 1;
            cfg.nkv = 128;
            cfg.hq = 4;
            cfg.hkv = 2;
            cfg.nstream = 1;
            cfg.n_kv_max = 48;
            cfg.type_k = GGML_TYPE_Q8_0;
            cfg.type_v = GGML_TYPE_TURBO4_0;
            run_test_case(backend_gpu, backend_cpu, get_sparse_count, cfg, "q8_0_k_turbo4_v_decode");
        }

        // Test 3: Multiquery prefill (Nq=4) with multi-stream (Nstream=2), GQA
        {
            test_config cfg;
            cfg.hsk = 128;
            cfg.hsv = 128;
            cfg.nq = 4;
            cfg.nkv = 64;
            cfg.hq = 8;
            cfg.hkv = 2;
            cfg.nstream = 2;
            cfg.n_kv_max = 32;
            cfg.type_k = GGML_TYPE_Q8_0;
            cfg.type_v = GGML_TYPE_TURBO4_0;
            run_test_case(backend_gpu, backend_cpu, get_sparse_count, cfg, "multiquery_multistream_prefill");
        }

        // Test 4: All-masked row (all -INF -> exact 0 output)
        {
            test_config cfg;
            cfg.hsk = 128;
            cfg.hsv = 128;
            cfg.nq = 1;
            cfg.nkv = 64;
            cfg.hq = 2;
            cfg.hkv = 2;
            cfg.nstream = 1;
            cfg.n_kv_max = 32;
            cfg.type_k = GGML_TYPE_F16;
            cfg.type_v = GGML_TYPE_F16;
            cfg.all_masked = true;
            run_test_case(backend_gpu, backend_cpu, get_sparse_count, cfg, "all_masked_zero_output");
        }

        // Test 5: Finite boundary values: -65504.0f
        {
            test_config cfg;
            cfg.hsk = 128;
            cfg.hsv = 128;
            cfg.nq = 2;
            cfg.nkv = 64;
            cfg.hq = 4;
            cfg.hkv = 2;
            cfg.nstream = 1;
            cfg.n_kv_max = 32;
            cfg.type_k = GGML_TYPE_Q8_0;
            cfg.type_v = GGML_TYPE_TURBO4_0;
            cfg.test_minus_65504 = true;
            run_test_case(backend_gpu, backend_cpu, get_sparse_count, cfg, "finite_minus_65504_boundary");
        }

        // Test 6: Finite count overflow (active entries > n_kv_max, fallback to full row)
        {
            test_config cfg;
            cfg.hsk = 128;
            cfg.hsv = 128;
            cfg.nq = 2;
            cfg.nkv = 64;
            cfg.hq = 4;
            cfg.hkv = 2;
            cfg.nstream = 1;
            cfg.n_kv_max = 16;
            cfg.type_k = GGML_TYPE_Q8_0;
            cfg.type_v = GGML_TYPE_TURBO4_0;
            cfg.force_overflow = true;
            run_test_case(backend_gpu, backend_cpu, get_sparse_count, cfg, "count_overflow_fallback");
        }

        // Test 7: Odd tail active counts (e.g. 7 or 13 active elements)
        {
            test_config cfg;
            cfg.hsk = 128;
            cfg.hsv = 128;
            cfg.nq = 2;
            cfg.nkv = 64;
            cfg.hq = 4;
            cfg.hkv = 2;
            cfg.nstream = 1;
            cfg.n_kv_max = 32;
            cfg.type_k = GGML_TYPE_Q8_0;
            cfg.type_v = GGML_TYPE_TURBO4_0;
            cfg.odd_tail = true;
            run_test_case(backend_gpu, backend_cpu, get_sparse_count, cfg, "odd_tail_active_counts");
        }

        // Test 8: Nonzero view offset on K and V
        {
            test_config cfg;
            cfg.hsk = 128;
            cfg.hsv = 128;
            cfg.nq = 1;
            cfg.nkv = 64;
            cfg.hq = 4;
            cfg.hkv = 2;
            cfg.nstream = 1;
            cfg.n_kv_max = 32;
            cfg.type_k = GGML_TYPE_Q8_0;
            cfg.type_v = GGML_TYPE_TURBO4_0;
            cfg.use_view_offset = true;
            run_test_case(backend_gpu, backend_cpu, get_sparse_count, cfg, "nonzero_view_offsets");
        }

        // Test 9: Padded mask rows: mask.ne1 > Nq (decode: 32 vs 1)
        {
            test_config cfg;
            cfg.hsk = 128;
            cfg.hsv = 128;
            cfg.nq = 1;
            cfg.mask_ne1 = 32; // mask rows padded to 32
            cfg.nkv = 64;
            cfg.hq = 4;
            cfg.hkv = 2;
            cfg.nstream = 1;
            cfg.n_kv_max = 32;
            cfg.type_k = GGML_TYPE_Q8_0;
            cfg.type_v = GGML_TYPE_TURBO4_0;
            run_test_case(backend_gpu, backend_cpu, get_sparse_count, cfg, "padded_mask_decode_32_vs_1");
        }

        // Test 10: Padded mask rows: mask.ne1 > Nq (prefill: 32 vs 4)
        {
            test_config cfg;
            cfg.hsk = 128;
            cfg.hsv = 128;
            cfg.nq = 4;
            cfg.mask_ne1 = 32; // mask rows padded to 32
            cfg.nkv = 64;
            cfg.hq = 8;
            cfg.hkv = 2;
            cfg.nstream = 2;
            cfg.n_kv_max = 32;
            cfg.type_k = GGML_TYPE_Q8_0;
            cfg.type_v = GGML_TYPE_TURBO4_0;
            run_test_case(backend_gpu, backend_cpu, get_sparse_count, cfg, "padded_mask_prefill_32_vs_4");
        }

        // Test 11: Repeated executions with mask mutations
        test_repeated_graph_mask_mutation(backend_gpu, backend_cpu, get_sparse_count);

        // Test 12: TP5 QSA explicit head map — local Q5 over KV2 with map
        // [0,0,1,1,1]. The integer ratio (5/2) is NOT the true assignment,
        // so any consumer assuming uniform GQA reads the wrong KV head for
        // heads 2..4. Every head must be distinguishable: K, V, Q, mask rows
        // and sparse counts all differ per head, so a swapped head fails
        // parity, including the LAST head (4 -> KV head 1).
        {
            test_config cfg;
            cfg.hsk = 128;
            cfg.hsv = 128;
            cfg.nq = 1;
            cfg.nkv = 64;
            cfg.hq = 5;
            cfg.hkv = 2;
            cfg.nstream = 1;
            cfg.n_kv_max = 32;
            cfg.type_k = GGML_TYPE_Q8_0;
            cfg.type_v = GGML_TYPE_TURBO4_0;
            cfg.head_map = {0, 0, 1, 1, 1};
            run_test_case(backend_gpu, backend_cpu, get_sparse_count, cfg, "tp5_headmap_q5_kv2_sparse");
        }

        // Test 13: same map, dense (n_kv_max = 0) — exercises the non-sparse
        // scalar path (and its split_k reduce) with the explicit map.
        {
            test_config cfg;
            cfg.hsk = 128;
            cfg.hsv = 128;
            cfg.nq = 1;
            cfg.nkv = 64;
            cfg.hq = 5;
            cfg.hkv = 2;
            cfg.nstream = 1;
            cfg.n_kv_max = 0;
            cfg.type_k = GGML_TYPE_Q8_0;
            cfg.type_v = GGML_TYPE_TURBO4_0;
            cfg.head_map = {0, 0, 1, 1, 1};
            run_test_case(backend_gpu, backend_cpu, get_sparse_count, cfg, "tp5_headmap_q5_kv2_dense");
        }

        // Test 14: mapped prefill (Nq = 4, multiple streams) with a padded
        // mask and view offsets on K/V — cache/view guards under the map.
        {
            test_config cfg;
            cfg.hsk = 128;
            cfg.hsv = 128;
            cfg.nq = 4;
            cfg.mask_ne1 = 32;
            cfg.nkv = 64;
            cfg.hq = 5;
            cfg.hkv = 2;
            cfg.nstream = 2;
            cfg.n_kv_max = 32;
            cfg.type_k = GGML_TYPE_Q8_0;
            cfg.type_v = GGML_TYPE_TURBO4_0;
            cfg.use_view_offset = true;
            cfg.head_map = {0, 0, 1, 1, 1};
            run_test_case(backend_gpu, backend_cpu, get_sparse_count, cfg, "tp5_headmap_q5_kv2_view_padded_mask");
        }

        // Test 15: mapped decode with mask mutation across repeated graph executions
        test_tp5_headmap_mask_mutation(backend_gpu, backend_cpu);

        // Test 16: mapped fail-closed capability oracles (supports_op verdicts)
        test_tp5_headmap_fail_closed(backend_gpu);

        // Test 17: mapped + sinks (SINK bit 24 coexists with headmap bit 25),
        // with an independent hand-rolled reference oracle
        test_tp5_headmap_sinks(backend_gpu, backend_cpu);



        ggml_backend_free(backend_gpu);
    }

    ggml_backend_free(backend_cpu);

    if (g_failures != 0) {
        std::fprintf(stderr, "test-vulkan-sparse-attn: %d FAILURES\n", g_failures);
        return 1;
    }

    std::printf("test-vulkan-sparse-attn: ALL TESTS PASSED\n");
    return 0;
}
