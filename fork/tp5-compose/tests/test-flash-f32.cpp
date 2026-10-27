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
constexpr int default_kv_length = 256;
constexpr double absolute_tolerance = 2e-5;
constexpr unsigned char guard_value = 0xa5;

enum class fixture { f16_exact, f32_non_f16_exact, f32_masked_nan };

void check(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

struct backend_delete {
    void operator()(ggml_backend * p) const { ggml_backend_free(p); }
};

struct buffer_delete {
    void operator()(ggml_backend_buffer * p) const { ggml_backend_buffer_free(p); }
};

struct context_delete {
    void operator()(ggml_context * p) const { ggml_free(p); }
};

using buffer_owner = std::unique_ptr<ggml_backend_buffer, buffer_delete>;

struct guarded_tensor {
    buffer_owner buffer;
    ggml_tensor * value;
    ggml_tensor * before;
    ggml_tensor * after;
};

guarded_tensor allocate(ggml_context * context, ggml_backend_t backend, ggml_tensor * tensor) {
    const auto buft = ggml_backend_get_default_buffer_type(backend);
    const auto alignment = ggml_backend_buft_get_alignment(buft);
    check(alignment > 0, "invalid backend buffer alignment");
    // The backend's allocation includes its permitted FA conversion tail. Guards
    // belong outside that region, not immediately after the logical tensor bytes.
    const auto bytes = ggml_backend_buft_get_alloc_size(buft, tensor);
    const auto padded_bytes = ((bytes + alignment - 1) / alignment) * alignment;
    guarded_tensor result{ buffer_owner(ggml_backend_alloc_buffer(backend, padded_bytes + 2 * alignment)), tensor,
                           ggml_new_tensor_1d(context, GGML_TYPE_I8, alignment),
                           ggml_new_tensor_1d(context, GGML_TYPE_I8, alignment) };
    check(result.buffer != nullptr, "device buffer allocation failed");
    auto * base = static_cast<unsigned char *>(ggml_backend_buffer_get_base(result.buffer.get()));
    check(ggml_backend_tensor_alloc(result.buffer.get(), result.before, base) == GGML_STATUS_SUCCESS &&
              ggml_backend_tensor_alloc(result.buffer.get(), tensor, base + alignment) == GGML_STATUS_SUCCESS &&
              ggml_backend_tensor_alloc(result.buffer.get(), result.after, base + alignment + padded_bytes) ==
                  GGML_STATUS_SUCCESS,
          "guarded tensor allocation failed");
    ggml_backend_buffer_clear(result.buffer.get(), guard_value);
    return result;
}

void check_guards(const guarded_tensor & tensor) {
    std::vector<unsigned char> bytes(ggml_nbytes(tensor.before));
    for (const auto * guard : { tensor.before, tensor.after }) {
        ggml_backend_tensor_get(guard, bytes.data(), 0, bytes.size());
        check(std::all_of(bytes.begin(), bytes.end(), [](unsigned char value) { return value == guard_value; }),
              "attention kernel modified an allocation guard");
    }
}

template <typename T>
void check_source(const guarded_tensor & tensor, const std::vector<T> & original) {
    std::vector<T> observed(original.size());
    ggml_backend_tensor_get(tensor.value, observed.data(), 0, observed.size() * sizeof(T));
    check(std::memcmp(observed.data(), original.data(), original.size() * sizeof(T)) == 0,
          "attention kernel modified a source tensor");
    check_guards(tensor);
}

bool exercise(ggml_backend_t backend, const char * device, int rows, int query_heads, int head_offset,
              fixture variant, int head_size = 128, int group_size = 4,
              ggml_type type_k = GGML_TYPE_F32, ggml_type type_v = GGML_TYPE_F32,
              int kv_length = default_kv_length, int visible_keys = -1) {
    const auto index = [head_size](int feature, int token, int head, int tokens) {
        return (static_cast<std::size_t>(head) * tokens + token) * head_size + feature;
    };
    const auto visible_count = [=](int row) {
        return visible_keys >= 0 ? visible_keys : kv_length - 19 - (rows - 1) + row;
    };
    const float scale = head_size == 256 ? 0.0625f : 0.125f;
    const bool f16_exact = variant == fixture::f16_exact;
    const char * fixture_name = f16_exact ? "f16-exact" :
                               variant == fixture::f32_masked_nan ? "f32-masked-nan" : "f32-non-f16-exact";
    ggml_init_params params{ 1024 * 1024, nullptr, true };
    std::unique_ptr<ggml_context, context_delete> context(ggml_init(params));
    check(context != nullptr, "GGML metadata allocation failed");
    const int kv_heads = query_heads / group_size;
    auto * q = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, head_size, rows, query_heads);
    auto * k = ggml_new_tensor_3d(context.get(), type_k, head_size, kv_length, kv_heads);
    auto * v = ggml_new_tensor_3d(context.get(), type_v, head_size, kv_length, kv_heads);
    auto * mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F16, kv_length, rows);
    auto * output = ggml_flash_attn_ext(context.get(), q, k, v, mask, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
    check(ggml_backend_supports_op(backend, output), "HIP backend does not support the requested F32 attention op");
    auto q_buffer = allocate(context.get(), backend, q);
    auto k_buffer = allocate(context.get(), backend, k);
    auto v_buffer = allocate(context.get(), backend, v);
    auto mask_buffer = allocate(context.get(), backend, mask);
    auto output_buffer = allocate(context.get(), backend, output);
    std::vector<float> queries(ggml_nelements(q)), keys(ggml_nelements(k)), values(ggml_nelements(v));
    std::vector<ggml_fp16_t> masks(ggml_nelements(mask));
    for (int head = 0; head < query_heads; ++head) {
        const int global_head = head_offset + head;
        for (int row = 0; row < rows; ++row) {
            for (int feature = 0; feature < head_size; ++feature) {
                queries[index(feature, row, head, rows)] =
                    float((global_head * 13 + row * 17 + feature * 7 + feature * feature % 19) % 65 - 32) / 128.0f;
                if (!f16_exact) {
                    queries[index(feature, row, head, rows)] +=
                        float((global_head + row * 3 + feature) % 7 + 1) / 65536.0f;
                }
            }
        }
    }
    for (int head = 0; head < kv_heads; ++head) {
        const int global_head = head_offset / group_size + head;
        for (int token = 0; token < kv_length; ++token) {
            for (int feature = 0; feature < head_size; ++feature) {
                const auto offset = index(feature, token, head, kv_length);
                keys[offset] = float((global_head * 19 + token * 11 + feature * 13 + token * feature % 23) % 65 - 32) /
                               128.0f;
                // Visible positive [1,2) dyadics expose VKQ accumulation error.
                // Non-NaN fixtures keep a finite, conspicuously larger tail.
                values[offset] = 1.0f + float((global_head * 137 + token * 29 + feature * 47 + token * feature % 31) %
                                              1024) / 1024.0f;
                if (token >= kv_length - 19) {
                    values[offset] = 8.0f + float((global_head * 3 + token * 7 + feature * 11) % 128) / 128.0f;
                }
                if (!f16_exact) {
                    keys[offset] += float((global_head * 3 + token + feature) % 7 + 1) / 65536.0f;
                    // With F32 storage, a sub-half-ULP positive offset makes
                    // rounding V to F16 wrong independently of accumulation
                    // precision. F16-storage cases round the host oracle below.
                    values[offset] += float(2 + (global_head + token + feature) % 2) / 8192.0f;
                }
                if (variant == fixture::f32_masked_nan &&
                        token >= (visible_keys >= 0 ? visible_keys : kv_length - 19)) {
                    // These positions are -INF masked for every query. They must
                    // be excluded before reading K/V, not multiplied by zero.
                    keys[offset] = values[offset] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
    }
    for (int row = 0; row < rows; ++row) {
        const int visible = visible_count(row);
        for (int token = 0; token < kv_length; ++token) {
            masks[static_cast<std::size_t>(row) * kv_length + token] =
                ggml_fp32_to_fp16(token < visible ? 0.0f : -std::numeric_limits<float>::infinity());
        }
    }
    // The F64 oracle must consume the actual stored K/V, not the pre-rounding
    // fixture. Keep Q's non-F16-exact F32 values unchanged in these cases.
    const auto store_f16 = [](std::vector<float> & source) {
        std::vector<ggml_fp16_t> stored(source.size());
        for (std::size_t i = 0; i < source.size(); ++i) {
            stored[i] = ggml_fp32_to_fp16(source[i]);
            source[i] = ggml_fp16_to_fp32(stored[i]);
        }
        return stored;
    };
    const auto stored_keys = type_k == GGML_TYPE_F16 ? store_f16(keys) : std::vector<ggml_fp16_t>{};
    const auto stored_values = type_v == GGML_TYPE_F16 ? store_f16(values) : std::vector<ggml_fp16_t>{};
    if (type_k == GGML_TYPE_F16 || type_v == GGML_TYPE_F16) {
        check(std::any_of(queries.begin(), queries.end(), [](float value) {
                  return value != ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
              }), "F16-cache oracle requires non-F16-exact F32 queries");
    }
    ggml_backend_tensor_set(q, queries.data(), 0, queries.size() * sizeof(float));
    ggml_backend_tensor_set(k, type_k == GGML_TYPE_F16 ? static_cast<const void *>(stored_keys.data()) : keys.data(),
                            0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, type_v == GGML_TYPE_F16 ? static_cast<const void *>(stored_values.data()) : values.data(),
                            0, ggml_nbytes(v));
    ggml_backend_tensor_set(mask, masks.data(), 0, masks.size() * sizeof(ggml_fp16_t));
    std::vector<float> observed(ggml_nelements(output), std::numeric_limits<float>::quiet_NaN());
    ggml_backend_tensor_set(output, observed.data(), 0, observed.size() * sizeof(float));
    auto * graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph, output);
    const auto status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    check(status == GGML_STATUS_SUCCESS, "F32 attention graph failed");
    ggml_backend_tensor_get(output, observed.data(), 0, observed.size() * sizeof(float));
    double max_error = 0.0;
    double worst_expected = 0.0;
    float worst_actual = 0.0f;
    std::size_t nonfinite = 0, errors = 0;
    std::vector<double> probabilities(kv_length);
    for (int head = 0; head < query_heads; ++head) {
        const int kv_head = head / group_size;
        for (int row = 0; row < rows; ++row) {
            const int visible = visible_count(row);
            double max_score = -std::numeric_limits<double>::infinity();
            for (int token = 0; token < visible; ++token) {
                double score = 0.0;
                for (int feature = 0; feature < head_size; ++feature) {
                    score += double(queries[index(feature, row, head, rows)]) *
                             double(keys[index(feature, token, kv_head, kv_length)]);
                }
                probabilities[token] = score * double(scale);
                max_score = std::max(max_score, probabilities[token]);
            }
            double denominator = 0.0;
            for (int token = 0; token < visible; ++token) {
                probabilities[token] = std::exp(probabilities[token] - max_score);
                denominator += probabilities[token];
            }
            for (int feature = 0; feature < head_size; ++feature) {
                double expected = 0.0;
                for (int token = 0; token < visible; ++token) {
                    expected += probabilities[token] * double(values[index(feature, token, kv_head, kv_length)]);
                }
                expected = denominator == 0.0 ? 0.0 : expected / denominator;
                // FA output is [feature, query_head, query_row], not Q's layout.
                const auto output_index = (static_cast<std::size_t>(row) * query_heads + head) * head_size + feature;
                const float actual = observed[output_index];
                if (!std::isfinite(actual)) {
                    ++nonfinite;
                    continue;
                }
                const double error = std::abs(double(actual) - expected);
                errors += error > absolute_tolerance;
                if (error > max_error) {
                    max_error = error;
                    worst_expected = expected;
                    worst_actual = actual;
                }
            }
        }
    }
    check_guards(output_buffer);
    check_source(q_buffer, queries);
    if (type_k == GGML_TYPE_F16) {
        check_source(k_buffer, stored_keys);
    } else {
        check_source(k_buffer, keys);
    }
    if (type_v == GGML_TYPE_F16) {
        check_source(v_buffer, stored_values);
    } else {
        check_source(v_buffer, values);
    }
    check_source(mask_buffer, masks);
    std::printf("FLASH_F32_CASE device=%s qdtype=f32 kvdtype=%s kdtype=%s vdtype=%s fixture=%s rows=%d qheads=%d kvheads=%d head_offset=%d kvlen=%d head_size=%d visible_keys=%d "
                "atol=%.9g maxerror=%.9g errors=%zu nonfinite=%zu worst_expected=%.9g worst_actual=%.9g "
                "guards_preserved=true sources_preserved=true\n",
                device, type_k == type_v ? ggml_type_name(type_k) : "mixed", ggml_type_name(type_k), ggml_type_name(type_v),
                fixture_name, rows, query_heads, kv_heads, head_offset, kv_length, head_size, visible_keys,
                absolute_tolerance, max_error, errors,
                nonfinite, worst_expected, double(worst_actual));
    std::fflush(stdout);
    return nonfinite == 0 && errors == 0;
}

}  // namespace

int main() {
    try {
        ggml_backend_load_all();
        std::size_t devices = 0, cases = 0, failed = 0;
        for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            const auto dev = ggml_backend_dev_get(i);
            const auto reg = ggml_backend_dev_backend_reg(dev);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU ||
                std::string(ggml_backend_reg_name(reg)) != "ROCm") {
                continue;
            }
            std::unique_ptr<ggml_backend, backend_delete> backend(ggml_backend_dev_init(dev, "quantized-f32"));
            check(backend != nullptr, "cannot initialize the declared HIP numerical policy");
            for (const auto variant : { fixture::f16_exact, fixture::f32_non_f16_exact, fixture::f32_masked_nan }) {
                for (int rows = 1; rows <= 4; ++rows) {
                    failed += !exercise(backend.get(), ggml_backend_dev_name(dev), rows, 32, 0, variant);
                    failed += !exercise(backend.get(), ggml_backend_dev_name(dev), rows, 4, 12, variant);
                    failed += !exercise(backend.get(), ggml_backend_dev_name(dev), rows, 8, 20, variant);
                    failed += !exercise(backend.get(), ggml_backend_dev_name(dev), rows, 24, 0, variant, 256, 12);
                    cases += 4;
                    if (variant != fixture::f16_exact) {
                        failed += !exercise(backend.get(), ggml_backend_dev_name(dev), rows, 24, 0,
                                            variant, 256, 12, GGML_TYPE_F16, GGML_TYPE_F16);
                        ++cases;
                    }
                }
            }
            // Short and non-aligned key ranges exercise masked tails and an
            // excluded row, with NaNs hidden in every masked K/V.
            for (const auto type : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
                failed += !exercise(backend.get(), ggml_backend_dev_name(dev), 1, 24, 0,
                                    fixture::f32_non_f16_exact, 256, 12, type, type, 3, 3);
                for (const int visible : { 0, 64, 65 }) {
                    failed += !exercise(backend.get(), ggml_backend_dev_name(dev), 1, 24, 0,
                                        fixture::f32_masked_nan, 256, 12, type, type, 256, visible);
                }
                cases += 4;
            }
            ++devices;
        }
        check(devices > 0, "F32 attention hardware oracle requires an actual ROCm GPU");
        check(failed == 0, "attention cases contain nonfinite outputs or exceed the fixed F64 softmax tolerance");
        std::printf("FLASH_F32_ORACLE_PASS devices=%zu cases=%zu kv_types=f32,f16 atol=%.9g f64_stable_softmax=true "
                    "fixtures=f16-exact,f32-non-f16-exact,f32-masked-nan masked_tail_finite=true masked_tail_nan=true "
                    "guards_preserved=true sources_preserved=true\n",
                    devices, cases, absolute_tolerance);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FLASH_F32_ORACLE_FAIL: %s\n", error.what());
        return 1;
    }
}
