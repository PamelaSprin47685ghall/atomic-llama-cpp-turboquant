#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int K = 2560, M = 128, S = 10, E = 13;
using counter_fn = int64_t (*)(ggml_backend_t, const char *);

void check(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15;
    x *= 0x846ca68bU; return x ^ (x >> 16);
}

struct Weight {
    ggml_tensor * storage;
    ggml_tensor * view;
    std::vector<uint8_t> bytes;
};

Weight make_weight(ggml_context * ctx, bool padded, uint32_t seed) {
    check(ggml_blck_size(GGML_TYPE_Q2_0) == 64 && ggml_type_size(GGML_TYPE_Q2_0) == 18,
          "unexpected Q2_0 encoding");
    auto * storage = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, K + (padded ? 64 : 0), M + (padded ? 1 : 0), E);
    auto * view = padded ? ggml_view_3d(ctx, storage, K, M, E, storage->nb[1], storage->nb[2], 0) : storage;
    Weight result{storage, view, std::vector<uint8_t>(ggml_nbytes(storage), 0xff)};
    for (int e = 0; e < E; ++e) {
        for (int r = 0; r < M; ++r) {
            for (int b = 0; b < K / 64; ++b) {
                auto * dst = result.bytes.data() + size_t(e) * view->nb[2] + size_t(r) * view->nb[1] + b * 18;
                const uint32_t h = mix(seed + uint32_t((e * M + r) * (K / 64) + b));
                const float scale = (0.00317f + float(h % 97) * 0.000173f) * (h & 0x100 ? -1.0f : 1.0f);
                const auto half = ggml_fp32_to_fp16(scale);
                std::memcpy(dst, &half, 2);
                for (int j = 0; j < 16; ++j) dst[2+j] = uint8_t(mix(h + j * 13));
            }
        }
    }
    return result;
}

float decoded(const Weight & w, int expert, int row, int k) {
    const auto * b = w.bytes.data() + size_t(expert) * w.view->nb[2] + size_t(row) * w.view->nb[1] + size_t(k / 64) * 18;
    ggml_fp16_t d;
    std::memcpy(&d, b, 2);
    const int q = ((b[2 + (k % 64) / 4] >> (2 * (k % 4))) & 3) - 1;
    return ggml_fp16_to_fp32(d) * float(q);
}

struct Result {
    std::vector<float> values;
    std::vector<double> reference;
    int64_t fusions;
};

Result compute(ggml_backend_t backend, counter_fn counter, int input_slots, bool padded, bool invalid, bool fuse) {
    const ggml_init_params params{32 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false), nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    check(bool(ctx), "metadata allocation failed");
    auto wg = make_weight(ctx.get(), padded, 0x1234567);
    auto wu = make_weight(ctx.get(), padded, 0x9a87b65);
    auto * xs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, K + (padded ? 1 : 0), input_slots + (padded ? 1 : 0), 1);
    auto * x = padded ? ggml_view_3d(ctx.get(), xs, K, input_slots, 1, xs->nb[1], xs->nb[2], 0) : xs;
    auto * is = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, padded ? 2 : S, padded ? S : 1);
    auto * ids = padded ? ggml_view_2d(ctx.get(), ggml_transpose(ctx.get(), is), S, 1, sizeof(int32_t), 0) : is;
    std::vector<uint8_t> xb(ggml_nbytes(xs), 0xff), ib(ggml_nbytes(is), 0xff);
    std::vector<float> xv(size_t(input_slots) * K);
    std::vector<int> selected(S);
    for (int s = 0; s < input_slots; ++s) {
        for (int k = 0; k < K; ++k) {
            const float v = float(int(mix(uint32_t(s * K + k) + 713) % 65521) - 32760) / 65521.0f;
            xv[size_t(s) * K + k] = v;
            std::memcpy(xb.data() + size_t(s) * x->nb[1] + size_t(k) * x->nb[0], &v, sizeof(v));
        }
    }
    for (int s = 0; s < S; ++s) {
        selected[s] = (3 * s + 1) % E;
        if (invalid && s == 2) selected[s] = -1;
        if (invalid && s == 7) selected[s] = E;
        const int32_t id = selected[s];
        std::memcpy(ib.data() + size_t(s) * ids->nb[0], &id, sizeof(id));
    }
    auto * gate = ggml_mul_mat_id(ctx.get(), wg.view, x, ids);
    auto * up = ggml_mul_mat_id(ctx.get(), wu.view, x, ids);
    auto * output = ggml_swiglu_split(ctx.get(), gate, up);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    // Put leaf views before the three-node fusion region, as real shards are.
    for (auto * t : {wg.view, wu.view, x, ids}) ggml_build_forward_expand(graph, t);
    ggml_build_forward_expand(graph, output);
    if (!fuse) {
        // An actual external reader exercises the upstream use-count guard.
        ggml_set_output(gate);
        ggml_set_output(up);
        ggml_build_forward_expand(graph, ggml_sqr(ctx.get(), up));
    }
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    check(bool(buffer), "device buffer allocation failed");
    ggml_backend_tensor_set(wg.storage, wg.bytes.data(), 0, wg.bytes.size());
    ggml_backend_tensor_set(wu.storage, wu.bytes.data(), 0, wu.bytes.size());
    ggml_backend_tensor_set(xs, xb.data(), 0, xb.size());
    ggml_backend_tensor_set(is, ib.data(), 0, ib.size());
    const auto previous = counter(backend, "mul_mat_glu");
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph execution failed");
    ggml_backend_synchronize(backend);
    Result result{std::vector<float>(M * S), std::vector<double>(M * S), counter(backend, "mul_mat_glu") - previous};
    ggml_backend_tensor_get(output, result.values.data(), 0, result.values.size() * sizeof(float));
    check(fuse ? result.fusions > 0 : result.fusions == 0,
          fuse ? "candidate silently ran unfused" : "external projection reader was elided");
    for (int s = 0; s < S; ++s) {
        for (int r = 0; r < M; ++r) {
            const size_t pos = size_t(s) * M + r;
            if (selected[s] < 0 || selected[s] >= E) {
                check(std::isnan(result.values[pos]), "invalid expert did not poison its output");
                result.reference[pos] = NAN;
                continue;
            }
            double g = 0, u = 0;
            for (int k = 0; k < K; ++k) {
                const double a = xv[size_t(input_slots == 1 ? 0 : s) * K + k];
                g += double(decoded(wg, selected[s], r, k)) * a;
                u += double(decoded(wu, selected[s], r, k)) * a;
            }
            const double reference = (g / (1.0 + std::exp(-g))) * u;
            result.reference[pos] = reference;
            const double error = std::abs(double(result.values[pos]) - reference);
            check(std::isfinite(result.values[pos]) && error <= 2e-5 * (1.0 + std::abs(reference)), "independent F64 oracle failed");
        }
    }
    // Prove the kernel did not use padding or overwrite its original inputs.
    for (const auto & w : {wg, wu}) {
        std::vector<uint8_t> readback(w.bytes.size());
        ggml_backend_tensor_get(w.storage, readback.data(), 0, readback.size());
        check(readback == w.bytes, "original encoded weight or padding modified");
    }
    std::vector<uint8_t> read_x(xb.size()), read_ids(ib.size());
    ggml_backend_tensor_get(xs, read_x.data(), 0, read_x.size());
    ggml_backend_tensor_get(is, read_ids.data(), 0, read_ids.size());
    check(read_x == xb && read_ids == ib, "input, ID or padding modified");
    return result;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        check(argc == 2, "usage: q2-f32-glu <installed-backend-dir>");
        ggml_backend_load_all_from_path(argv[1]);
        size_t devices = 0, cases = 0, checked = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * dev = ggml_backend_dev_get(i);
            auto * reg = ggml_backend_dev_backend_reg(dev);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU || std::string(ggml_backend_reg_name(reg)) != "ROCm") continue;
            auto counter = reinterpret_cast<counter_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_fusion_count"));
            check(counter != nullptr, "fusion counter unavailable");
            std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(dev, "quantized-f32"), ggml_backend_free);
            check(bool(backend), "cannot initialize original-byte/F32 backend");
            for (int slots : {1, S}) {
                for (bool padded : {false, true}) {
                    for (bool invalid : {false, true}) {
                        // Invalid-ID striding is covered once per input mapping.
                        if (invalid && !padded) continue;
                        auto expected = compute(backend.get(), counter, slots, padded, invalid, false);
                        auto actual = compute(backend.get(), counter, slots, padded, invalid, true);
                        for (size_t j = 0; j < actual.values.size(); ++j) {
                            if (std::isnan(expected.values[j])) {
                                check(std::isnan(actual.values[j]), "poison mismatch");
                            } else {
                                uint32_t a, b;
                                std::memcpy(&a, &actual.values[j], 4); std::memcpy(&b, &expected.values[j], 4);
                                if (a != b) {
                                    std::fprintf(stderr, "mismatch device=%s slots=%d padded=%d index=%zu actual=%a expected=%a bits=%08x/%08x\n",
                                        ggml_backend_dev_name(dev), slots, int(padded), j, double(actual.values[j]), double(expected.values[j]), a, b);
                                    throw std::runtime_error("fused output differs from retained projection + SwiGLU arithmetic");
                                }
                                ++checked;
                            }
                        }
                        ++cases;
                        std::printf("Q2_F32_GLU_CASE device=%s slots=%d padded=%d invalid_ids=%d fusions=%lld bitwise=true f64=true sources_preserved=true\n",
                            ggml_backend_dev_name(dev), slots, int(padded), int(invalid), (long long)actual.fusions);
                    }
                }
            }
            ++devices;
        }
        check(devices == 5, "fixture requires the actual five ROCm devices");
        std::printf("Q2_F32_GLU_PASS devices=%zu cases=%zu finite_bit_checks=%zu\n", devices, cases, checked);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "Q2_F32_GLU_FAIL %s\n", e.what());
        return 1;
    }
}
