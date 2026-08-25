#include "q2-opt-signed.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace qwen35_prune;

namespace {

struct ctx_deleter { void operator()(ggml_context * p) const { if (p) ggml_free(p); } };
using ctx_ptr = std::unique_ptr<ggml_context, ctx_deleter>;

static bool run_backend(ggml_backend_t backend) {
    constexpr int K = 256;
    constexpr int M = 64;
    constexpr int N = 5;

    ggml_init_params params {
        /* .mem_size   = */ 16u * 1024u * 1024u,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ctx_ptr ctx(ggml_init(params));
    if (!ctx) return false;

    ggml_tensor * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q2_0, K, M);
    ggml_tensor * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, N);
    ggml_tensor * y = ggml_mul_mat(ctx.get(), a, b);
    ggml_set_name(a, "signed_q2_a");
    ggml_set_name(b, "input_b");
    ggml_set_name(y, "output_y");

    if (!ggml_backend_supports_op(backend, y)) {
        std::fprintf(stderr, "%s: Q2_0 MUL_MAT not supported\n", ggml_backend_name(backend));
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx.get(), backend);
    if (!buffer) {
        std::fprintf(stderr, "%s: allocation failed\n", ggml_backend_name(backend));
        return false;
    }

    std::vector<q2_opt_block> packed((size_t) M * K / Q2_OPT_BLOCK_SIZE);
    std::vector<float> a_ref((size_t) M * K);
    uint64_t negative = 0;
    uint64_t codes[4] = {};
    for (int r = 0; r < M; ++r) {
        for (int ib = 0; ib < K / Q2_OPT_BLOCK_SIZE; ++ib) {
            q2_opt_block & block = packed[(size_t) r * (K / Q2_OPT_BLOCK_SIZE) + ib];
            const float mag = 0.125f * (1 + ((r + ib) % 7));
            const float d = ((r + ib) & 1) ? -mag : mag;
            block.d = ggml_fp32_to_fp16(d);
            if (std::signbit(d)) ++negative;
            std::memset(block.qs, 0, sizeof(block.qs));
            for (int i = 0; i < Q2_OPT_BLOCK_SIZE; ++i) {
                const uint8_t code = (uint8_t) ((i + 3 * r + ib) & 3);
                block.qs[i / 4] |= code << (2 * (i % 4));
                ++codes[code];
            }
            decode_q2_opt(block, a_ref.data() + (size_t) r * K + ib * Q2_OPT_BLOCK_SIZE);
        }
    }

    if (negative == 0 || *std::min_element(std::begin(codes), std::end(codes)) == 0) {
        ggml_backend_buffer_free(buffer);
        return false;
    }

    std::vector<float> b_data((size_t) N * K);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            b_data[(size_t) n * K + k] =
                0.35f * std::sin(0.031f * (float) (1 + k + 17 * n)) +
                0.15f * std::cos(0.047f * (float) (3 + 5 * k + n));
        }
    }

    ggml_backend_tensor_set(a, packed.data(), 0, packed.size() * sizeof(q2_opt_block));
    ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 8, false);
    ggml_build_forward_expand(graph, y);
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: graph compute failed: %s\n",
                ggml_backend_name(backend), ggml_status_to_string(status));
        ggml_backend_buffer_free(buffer);
        return false;
    }

    std::vector<float> got((size_t) M * N);
    ggml_backend_tensor_get(y, got.data(), 0, got.size() * sizeof(float));

    double max_abs_err = 0.0;
    double ref_norm2 = 0.0;
    double err_norm2 = 0.0;
    for (int n = 0; n < N; ++n) {
        for (int r = 0; r < M; ++r) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k) {
                ref += (double) a_ref[(size_t) r * K + k] * b_data[(size_t) n * K + k];
            }
            const double actual = got[(size_t) n * M + r];
            const double diff = actual - ref;
            max_abs_err = std::max(max_abs_err, std::fabs(diff));
            ref_norm2 += ref * ref;
            err_norm2 += diff * diff;
        }
    }
    const double nmse = ref_norm2 > 0.0 ? err_norm2 / ref_norm2 : err_norm2;
    const bool ok = std::isfinite(nmse) && nmse <= 5e-5 && max_abs_err <= 0.05;
    std::printf("%s: negative-d=%llu code_hist=[%llu,%llu,%llu,%llu] nmse=%.9g max_abs_err=%.9g %s\n",
            ggml_backend_name(backend),
            (unsigned long long) negative,
            (unsigned long long) codes[0], (unsigned long long) codes[1],
            (unsigned long long) codes[2], (unsigned long long) codes[3],
            nmse, max_abs_err, ok ? "OK" : "FAIL");
    ggml_backend_buffer_free(buffer);
    return ok;
}

static bool run_backend_mul_mat_id(ggml_backend_t backend) {
    constexpr int K = 256;
    constexpr int M = 32;
    constexpr int N_MATS = 4;
    constexpr int N_USED = 2;
    constexpr int N_TOKENS = 3;

    ggml_init_params params {
        /* .mem_size   = */ 16u * 1024u * 1024u,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ctx_ptr ctx(ggml_init(params));
    if (!ctx) return false;

    ggml_tensor * experts = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q2_0, K, M, N_MATS);
    ggml_tensor * ids     = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, N_USED, N_TOKENS);
    ggml_tensor * input   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, K, N_USED, N_TOKENS);
    ggml_tensor * output  = ggml_mul_mat_id(ctx.get(), experts, input, ids);
    ggml_set_name(experts, "signed_q2_experts");
    ggml_set_name(ids, "expert_ids");
    ggml_set_name(input, "expert_input");
    ggml_set_name(output, "expert_output");

    if (!ggml_backend_supports_op(backend, output)) {
        std::fprintf(stderr, "%s: Q2_0 MUL_MAT_ID not supported\n", ggml_backend_name(backend));
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx.get(), backend);
    if (!buffer) {
        std::fprintf(stderr, "%s: MUL_MAT_ID allocation failed\n", ggml_backend_name(backend));
        return false;
    }

    const size_t blocks_per_row = K / Q2_OPT_BLOCK_SIZE;
    std::vector<q2_opt_block> packed((size_t) N_MATS * M * blocks_per_row);
    std::vector<float> expert_ref((size_t) N_MATS * M * K);
    uint64_t negative = 0;
    uint64_t codes[4] = {};
    for (int e = 0; e < N_MATS; ++e) {
        for (int r = 0; r < M; ++r) {
            for (int ib = 0; ib < (int) blocks_per_row; ++ib) {
                q2_opt_block & block = packed[((size_t) e * M + r) * blocks_per_row + ib];
                const float mag = 0.0625f * (1 + ((11 * e + 3 * r + ib) % 11));
                const float d = ((e + r + ib) & 1) ? -mag : mag;
                block.d = ggml_fp32_to_fp16(d);
                if (std::signbit(d)) ++negative;
                std::memset(block.qs, 0, sizeof(block.qs));
                for (int i = 0; i < Q2_OPT_BLOCK_SIZE; ++i) {
                    const uint8_t code = (uint8_t) ((i + e + 3 * r + ib) & 3);
                    block.qs[i / 4] |= code << (2 * (i % 4));
                    ++codes[code];
                }
                decode_q2_opt(
                    block,
                    expert_ref.data() + ((size_t) e * M + r) * K + ib * Q2_OPT_BLOCK_SIZE);
            }
        }
    }

    const std::vector<int32_t> id_data = {
        0, 3,
        2, 1,
        3, 0,
    };
    std::vector<float> input_data((size_t) N_TOKENS * N_USED * K);
    for (int t = 0; t < N_TOKENS; ++t) {
        for (int u = 0; u < N_USED; ++u) {
            float * x = input_data.data() + ((size_t) t * N_USED + u) * K;
            for (int k = 0; k < K; ++k) {
                x[k] = 0.31f * std::sin(0.019f * (float) (1 + k + 13 * u + 29 * t)) +
                       0.17f * std::cos(0.041f * (float) (5 + 3 * k + 7 * u + t));
            }
        }
    }

    ggml_backend_tensor_set(experts, packed.data(), 0, packed.size() * sizeof(q2_opt_block));
    ggml_backend_tensor_set(ids, id_data.data(), 0, id_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 8, false);
    ggml_build_forward_expand(graph, output);
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: MUL_MAT_ID graph compute failed: %s\n",
                ggml_backend_name(backend), ggml_status_to_string(status));
        ggml_backend_buffer_free(buffer);
        return false;
    }

    if (output->ne[0] != M || output->ne[1] != N_USED || output->ne[2] != N_TOKENS) {
        std::fprintf(stderr, "%s: unexpected MUL_MAT_ID output shape [%lld,%lld,%lld,%lld]\n",
                ggml_backend_name(backend),
                (long long) output->ne[0], (long long) output->ne[1],
                (long long) output->ne[2], (long long) output->ne[3]);
        ggml_backend_buffer_free(buffer);
        return false;
    }

    std::vector<float> got((size_t) M * N_USED * N_TOKENS);
    ggml_backend_tensor_get(output, got.data(), 0, got.size() * sizeof(float));

    double max_abs_err = 0.0;
    double ref_norm2 = 0.0;
    double err_norm2 = 0.0;
    for (int t = 0; t < N_TOKENS; ++t) {
        for (int u = 0; u < N_USED; ++u) {
            const int expert = id_data[(size_t) t * N_USED + u];
            const float * x = input_data.data() + ((size_t) t * N_USED + u) * K;
            for (int r = 0; r < M; ++r) {
                const float * a = expert_ref.data() + ((size_t) expert * M + r) * K;
                double ref = 0.0;
                for (int k = 0; k < K; ++k) ref += (double) a[k] * x[k];
                const double actual = got[((size_t) t * N_USED + u) * M + r];
                const double diff = actual - ref;
                max_abs_err = std::max(max_abs_err, std::fabs(diff));
                ref_norm2 += ref * ref;
                err_norm2 += diff * diff;
            }
        }
    }

    const double nmse = ref_norm2 > 0.0 ? err_norm2 / ref_norm2 : err_norm2;
    const bool ok = negative > 0 &&
                    *std::min_element(std::begin(codes), std::end(codes)) > 0 &&
                    std::isfinite(nmse) && nmse <= 5e-5 && max_abs_err <= 0.05;
    std::printf("%s MUL_MAT_ID: negative-d=%llu code_hist=[%llu,%llu,%llu,%llu] nmse=%.9g max_abs_err=%.9g %s\n",
            ggml_backend_name(backend),
            (unsigned long long) negative,
            (unsigned long long) codes[0], (unsigned long long) codes[1],
            (unsigned long long) codes[2], (unsigned long long) codes[3],
            nmse, max_abs_err, ok ? "OK" : "FAIL");
    ggml_backend_buffer_free(buffer);
    return ok;
}

} // namespace

int main() {
    ggml_backend_load_all();

    bool ok = true;
    bool saw_cpu = false;
    bool saw_accelerator = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_CPU) saw_cpu = true;
        else saw_accelerator = true;

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            std::fprintf(stderr, "%s: backend init failed\n", ggml_backend_dev_name(dev));
            ok = false;
            continue;
        }
        ok = run_backend(backend) && ok;
        ok = run_backend_mul_mat_id(backend) && ok;
        ggml_backend_free(backend);
    }

    if (!saw_cpu) {
        std::fprintf(stderr, "CPU backend unavailable\n");
        ok = false;
    }
    if (!saw_accelerator) std::puts("accelerator backend unavailable: skipped");

    return ok ? 0 : 1;
}
