#include <ggml-backend.h>
#include <llama.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}
struct backend_delete { void operator()(ggml_backend * p) const { ggml_backend_free(p); } };
struct buffer_delete { void operator()(ggml_backend_buffer * p) const { ggml_backend_buffer_free(p); } };
struct context_delete { void operator()(ggml_context * p) const { ggml_free(p); } };
using buffer_owner = std::unique_ptr<ggml_backend_buffer, buffer_delete>;

buffer_owner allocate(ggml_backend_t backend, ggml_tensor * tensor) {
    const auto buft = ggml_backend_get_default_buffer_type(backend);
    const auto bytes = ggml_backend_buft_get_alloc_size(buft, tensor);
    buffer_owner buffer(ggml_backend_alloc_buffer(backend, bytes));
    check(buffer != nullptr, "device buffer allocation failed");
    check(ggml_backend_tensor_alloc(buffer.get(), tensor, ggml_backend_buffer_get_base(buffer.get())) ==
              GGML_STATUS_SUCCESS, "tensor allocation failed");
    return buffer;
}

// Build original block bytes, not quantized approximations of a float fixture.
// All codes, packed scale high bits and signs vary. Half scales are fixed dyadic
// values, so the CPU decoder + F64 dot is also an exact arithmetic oracle for
// these deliberately bounded integer numerators.
void fill_weights(const ggml_tensor * w, std::vector<unsigned char> & bytes, std::vector<float> & decoded) {
    const auto type = w->type;
    const auto block = ggml_blck_size(type);
    const auto size = ggml_type_size(type);
    const auto * traits = ggml_get_type_traits(type);
    check(traits && traits->to_float, "missing canonical CPU decoder");
    if (type == GGML_TYPE_Q2_0) {
        check(block == 64 && size == 18, "Q2_0 original-byte contract changed");
    }
    for (int64_t b3 = 0; b3 < w->ne[3]; ++b3) {
        for (int64_t b2 = 0; b2 < w->ne[2]; ++b2) {
            for (int64_t row = 0; row < w->ne[1]; ++row) {
                auto * dst = bytes.data() + b3 * w->nb[3] + b2 * w->nb[2] + row * w->nb[1];
                for (int64_t b = 0; b < w->ne[0] / block; ++b) {
                    auto * p = dst + b * size;
                    for (size_t j = 0; j < size; ++j) {
                        p[j] = static_cast<unsigned char>((j * 37 + b * 53 + row * 71 + b2 * 97 + b3 * 113) ^
                                                          (j * 11 + row * 17 + b2 * 29));
                    }
                    const auto scale = ggml_fp32_to_fp16((b + row + b2 + b3) % 2 ? 0.0009765625f : -0.0009765625f);
                    const auto minimum = ggml_fp32_to_fp16(0.0009765625f);
                    const size_t offset = type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q6_K ? size - 2 : 0;
                    std::memcpy(p + offset, &scale, sizeof(scale));
                    if (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                        std::memcpy(p + 2, &minimum, sizeof(minimum));
                    }
                }
                auto * out = decoded.data() + ((b3 * w->ne[2] + b2) * w->ne[1] + row) * w->ne[0];
                traits->to_float(dst, out, w->ne[0]);
                // Independently pin the nonstandard Q2_0 symbol mapping, without
                // trusting the canonical decoder for this actual model format.
                if (type == GGML_TYPE_Q2_0) {
                    for (int64_t i = 0; i < w->ne[0]; ++i) {
                        const auto * p = dst + (i / 64) * 18;
                        ggml_fp16_t d;
                        std::memcpy(&d, p, sizeof(d));
                        const int code = (p[2 + (i % 64) / 4] >> ((i % 4) * 2)) & 3;
                        check(out[i] == ggml_fp16_to_fp32(d) * (code - 1), "Q2_0 CPU symbol mapping mismatch");
                    }
                }
            }
        }
    }
}

void pad(ggml_tensor * t, bool strided_elements) {
    if (strided_elements) {
        t->nb[0] *= 2;
    }
    t->nb[1] = t->nb[0] * (t->ne[0] / ggml_blck_size(t->type)) + 32;
    t->nb[2] = t->nb[1] * t->ne[1] + 36;
    t->nb[3] = t->nb[2] * t->ne[2] + 40;
}

enum class column_layout { native, padded_flat, input_batch_gaps, output_batch_gaps };

void exercise(ggml_backend_t backend, ggml_type type, int k, int n, int tokens, bool padded,
              bool batches, bool indexed, bool broadcast, bool accepted = true, bool bad_shape = false,
              int bad_alignment = 0, column_layout columns = column_layout::native) {
    ggml_init_params params{ 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, context_delete> context(ggml_init(params));
    check(context != nullptr, "GGML metadata allocation failed");
    const int experts = indexed ? 5 : batches && !broadcast ? 2 : 1;
    const int w3 = batches && !indexed && !broadcast ? 2 : 1;
    const int slots = indexed ? 3 : tokens;
    auto * weights = ggml_new_tensor_4d(context.get(), type, k, n, experts, w3);
    auto * input = indexed ? ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, k, broadcast ? 1 : slots, tokens) :
                            ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, k, tokens, batches ? 4 : 1, batches ? 4 : 1);
    auto * ids = indexed ? ggml_new_tensor_2d(context.get(), GGML_TYPE_I32, slots, tokens) : nullptr;
    auto * output = indexed ? ggml_mul_mat_id(context.get(), weights, input, ids) :
                             ggml_mul_mat(context.get(), weights, input);
    if (padded) {
        pad(weights, false);
        pad(input, true);
        pad(output, true);
        if (ids) {
            pad(ids, true);
        }
    }
    if (columns != column_layout::native) {
        // Contiguous elements with padded columns can still flatten; gaps
        // between batches in either operand must retain the general mapping.
        pad(input, false);
        pad(output, false);
        if (columns != column_layout::input_batch_gaps) {
            input->nb[2] = input->nb[1] * input->ne[1];
            input->nb[3] = input->nb[2] * input->ne[2];
        }
        if (columns != column_layout::output_batch_gaps) {
            output->nb[2] = output->nb[1] * output->ne[1];
            output->nb[3] = output->nb[2] * output->ne[2];
        }
    }
    if (tokens == 1 && batches && broadcast && !indexed) {
        // Singleton strides need not equal the actual higher-axis column step.
        input->nb[1] += 4;
        output->nb[1] += 8;
    }
    // Invalid graphs are representable GGML tensors, but outside this numerical
    // policy. Graph preflight must reject them before touching any output byte.
    if (bad_shape) {
        output->ne[0] += 1;
    } else if (!accepted && type != GGML_TYPE_Q4_1 && bad_alignment == 0) {
        input->nb[0] += 1;
    }
    auto weight_buffer = allocate(backend, weights);
    auto input_buffer = allocate(backend, input);
    auto output_buffer = allocate(backend, output);
    buffer_owner ids_buffer;
    if (ids) {
        ids_buffer = allocate(backend, ids);
    }
    std::vector<unsigned char> w(ggml_nbytes(weights), 0xfa), x(ggml_nbytes(input), 0xfa);
    std::vector<unsigned char> y(ggml_nbytes(output), 0xfa), id_bytes(ids ? ggml_nbytes(ids) : 0, 0xfa);
    std::vector<float> decoded(static_cast<size_t>(k) * n * experts * w3);
    fill_weights(weights, w, decoded);
    for (int64_t b3 = 0; b3 < input->ne[3]; ++b3) {
        for (int64_t b2 = 0; b2 < input->ne[2]; ++b2) {
            for (int64_t row = 0; row < input->ne[1]; ++row) {
                for (int i = 0; i < k; ++i) {
                    const float value = float((i * 3 + row * 5 + b2 * 2 + b3) % 7 - 3) * 0.0078125f;
                    std::memcpy(x.data() + b3 * input->nb[3] + b2 * input->nb[2] + row * input->nb[1] + i * input->nb[0],
                                &value, sizeof(value));
                }
            }
        }
    }
    if (ids) {
        for (int token = 0; token < tokens; ++token) {
            for (int slot = 0; slot < slots; ++slot) {
                const int32_t expert = (token * 3 + slot * 2 + 1) % experts;
                std::memcpy(id_bytes.data() + token * ids->nb[1] + slot * ids->nb[0], &expert, sizeof(expert));
            }
        }
        ggml_backend_tensor_set(ids, id_bytes.data(), 0, id_bytes.size());
    }
    ggml_backend_tensor_set(weights, w.data(), 0, w.size());
    ggml_backend_tensor_set(input, x.data(), 0, x.size());
    ggml_backend_tensor_set(output, y.data(), 0, y.size());
    // Corrupt only device metadata after constructing a correctly aligned CPU
    // fixture. The canonical decoder must not itself read an invalid block.
    if (bad_alignment == 1) {
        weights->nb[1] += 2;
    } else if (bad_alignment == 2) {
        weights->data = static_cast<char *>(weights->data) + 2;
    } else if (bad_alignment == 3) {
        weights->nb[2] += 2;
    }
    auto * graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph, output);
    const auto status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    std::vector<unsigned char> observed(y.size());
    ggml_backend_tensor_get(output, observed.data(), 0, observed.size());
    if (!accepted) {
        check(status == GGML_STATUS_FAILED, "unsupported graph was not rejected");
        check(observed == y, "rejected graph modified its output");
        return;
    }
    check(status == GGML_STATUS_SUCCESS, "original-byte F32 projection graph failed");
    auto expected = y;
    for (int64_t b3 = 0; b3 < output->ne[3]; ++b3) {
        for (int64_t b2 = 0; b2 < output->ne[2]; ++b2) {
            for (int64_t row = 0; row < output->ne[1]; ++row) {
                int64_t wb2 = indexed ? 0 : b2 / (input->ne[2] / weights->ne[2]);
                const int64_t wb3 = indexed ? 0 : b3 / (input->ne[3] / weights->ne[3]);
                int64_t xr = row;
                if (indexed) {
                    int32_t expert;
                    std::memcpy(&expert, id_bytes.data() + b2 * ids->nb[1] + row * ids->nb[0], sizeof(expert));
                    wb2 = expert;
                    xr = row % input->ne[1];
                }
                for (int feature = 0; feature < n; ++feature) {
                    double sum = 0;
                    const auto weight_offset = ((wb3 * experts + wb2) * n + feature) * k;
                    for (int i = 0; i < k; ++i) {
                        float value;
                        std::memcpy(&value, x.data() + b3 * input->nb[3] + b2 * input->nb[2] + xr * input->nb[1] + i * input->nb[0],
                                    sizeof(value));
                        sum += double(decoded[weight_offset + i]) * double(value);
                    }
                    float actual;
                    const auto offset = b3 * output->nb[3] + b2 * output->nb[2] + row * output->nb[1] + feature * output->nb[0];
                    std::memcpy(&actual, observed.data() + offset, sizeof(actual));
                    if (!std::isfinite(actual) || actual != static_cast<float>(sum)) {
                        std::fprintf(stderr, "type=%s K=%d N=%d tokens=%d indexed=%d b3=%lld b2=%lld row=%lld feature=%d actual=%.9g ref=%.9g\n",
                                     ggml_type_name(type), k, n, tokens, indexed, (long long) b3, (long long) b2,
                                     (long long) row, feature, actual, static_cast<float>(sum));
                        check(false, "projection differs from canonical CPU decode / exact F64 dot oracle");
                    }
                    std::memcpy(expected.data() + offset, &actual, sizeof(actual));
                }
            }
        }
    }
    check(observed == expected, "kernel modified output padding");
    std::vector<unsigned char> after_weights(w.size()), after_input(x.size());
    ggml_backend_tensor_get(weights, after_weights.data(), 0, w.size());
    ggml_backend_tensor_get(input, after_input.data(), 0, x.size());
    check(after_weights == w && after_input == x, "kernel modified its input or padding");
    if (ids) {
        std::vector<unsigned char> after_ids(id_bytes.size());
        ggml_backend_tensor_get(ids, after_ids.data(), 0, after_ids.size());
        check(after_ids == id_bytes, "kernel modified device-resident expert IDs");
    }
}
}  // namespace

int main() {
    try {
        ggml_backend_load_all();
        size_t devices = 0, cases = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            const auto dev = ggml_backend_dev_get(i);
            const auto reg = ggml_backend_dev_backend_reg(dev);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU ||
                std::string(ggml_backend_reg_name(reg)) != "ROCm") {
                continue;
            }
            std::unique_ptr<ggml_backend, backend_delete> backend(ggml_backend_dev_init(dev, "quantized-f32"));
            check(backend != nullptr, "cannot initialize the declared HIP numerical policy");
            for (auto type : { GGML_TYPE_Q2_0, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
                               GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0 }) {
                const int block = int(ggml_blck_size(type));
                for (int tokens : { 1, 2, 3, 4, 5 }) {
                    for (int blocks : { 1, 3, 5 }) {
                        for (int n : { 1, 3, 4, 5, 19 }) {
                            for (bool padded : { false, true }) {
                                exercise(backend.get(), type, block * blocks, n, tokens, padded, false, false, false);
                                ++cases;
                            }
                        }
                    }
                }
                exercise(backend.get(), type, block * 3, 7, 3, true, true, false, false);
                exercise(backend.get(), type, block, 3, 2, true, false, false, false, false);
                exercise(backend.get(), type, block, 3, 2, false, false, false, false, false, true);
                cases += 3;
                // A single 2D weight matrix shared across both higher axes.
                // tokens=1 also makes nb[1] irrelevant to the flattened stride.
                for (int tokens : { 1, 3, 5 }) {
                    for (auto columns : { column_layout::native, column_layout::padded_flat,
                                          column_layout::input_batch_gaps, column_layout::output_batch_gaps }) {
                        exercise(backend.get(), type, block * 3, 7, tokens, false, true, false, true,
                                 true, false, 0, columns);
                        ++cases;
                    }
                }
                if (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                    for (int alignment_case : {1, 2, 3}) {
                        exercise(backend.get(), type, block, 3, 2, false, false, false, false,
                                 false, false, alignment_case);
                        ++cases;
                    }
                }
            }
            for (int tokens : { 1, 2, 5 }) {
                for (int k : { 64, 192, 320 }) {
                    for (bool broadcast : { false, true }) {
                        for (bool padded : { false, true }) {
                            exercise(backend.get(), GGML_TYPE_Q2_0, k, 7, tokens, padded, false, true, broadcast);
                            ++cases;
                        }
                    }
                }
            }
            exercise(backend.get(), GGML_TYPE_Q2_0, 192, 7, 2, true, false, true, true, false);
            exercise(backend.get(), GGML_TYPE_Q2_0, 192, 7, 2, false, false, true, true, false, true);
            exercise(backend.get(), GGML_TYPE_Q4_1, 64, 7, 2, false, false, false, false, false);
            cases += 3;
            ++devices;
        }
        check(devices > 0, "HIP hardware oracle requires an actual ROCm GPU");
        std::printf("Q8_F32_ORACLE_PASS devices=%zu cases=%zu formats=10 indexed_q2=true exact_dyadic=true "
                    "padding_preserved=true invalid_graphs_unchanged=true\n", devices, cases);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Q8_F32_ORACLE_FAIL: %s\n", error.what());
        return 1;
    }
}
