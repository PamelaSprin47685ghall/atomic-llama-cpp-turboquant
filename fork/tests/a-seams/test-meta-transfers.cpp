#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr size_t guard_bytes = 32;
constexpr uint8_t guard_value = 0xa5;

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct split_config {
    ggml_backend_meta_split_axis axis;
    const char * name;
    int64_t ne[4];
};

ggml_backend_meta_split_state split_tensor(const ggml_tensor *, void * user_data) {
    const auto & config = *static_cast<const split_config *>(user_data);
    ggml_backend_meta_split_state result = {config.axis, {0}, {1}, 1};
    if (config.axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
        result.ne[0] = 3;
        result.ne[1] = 7;
    }
    return result;
}

struct tensor_fixture {
    ggml_context * context = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * tensor = nullptr;

    tensor_fixture() = default;
    tensor_fixture(const tensor_fixture &) = delete;
    tensor_fixture & operator=(const tensor_fixture &) = delete;

    ~tensor_fixture() {
        if (backend != nullptr) {
            ggml_backend_synchronize(backend);
        }
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (context != nullptr) {
            ggml_free(context);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }

    void init(ggml_backend_dev_t cpu, split_config & config, ggml_type type) {
        ggml_backend_dev_t children[] = {cpu, cpu};
        const auto device = ggml_backend_meta_device(children, 2, split_tensor, &config);
        require(device != nullptr, "cannot create the meta device");
        backend = ggml_backend_dev_init(device, nullptr);
        require(backend != nullptr, "cannot create the meta backend");
        context = ggml_init({16 * 1024, nullptr, true});
        require(context != nullptr, "cannot create the tensor context");
        tensor = ggml_new_tensor_4d(context, type, config.ne[0], config.ne[1], config.ne[2], config.ne[3]);
        require(tensor != nullptr, "cannot create the tensor");
        buffer = ggml_backend_alloc_ctx_tensors_from_buft(context, ggml_backend_dev_buffer_type(device));
        require(buffer != nullptr, "cannot allocate the split tensor");
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
};

std::vector<uint8_t> make_values(const ggml_tensor * tensor, int seed) {
    std::vector<uint8_t> result(ggml_nbytes(tensor));
    const size_t count = static_cast<size_t>(ggml_nelements(tensor));
    for (size_t i = 0; i < count; ++i) {
        const float value = float(int((i * 17) % 127) - 63 + seed) * 0.125f;
        if (tensor->type == GGML_TYPE_F32) {
            std::memcpy(result.data() + i * sizeof(value), &value, sizeof(value));
        } else {
            const ggml_fp16_t half = ggml_fp32_to_fp16(value);
            std::memcpy(result.data() + i * sizeof(half), &half, sizeof(half));
        }
    }
    return result;
}

std::vector<uint8_t> guarded_buffer(size_t bytes) {
    if (bytes > size_t(std::numeric_limits<std::ptrdiff_t>::max()) - 2 * guard_bytes) {
        throw std::length_error("the transfer is too large for a guarded buffer");
    }
    return std::vector<uint8_t>(bytes + 2 * guard_bytes, guard_value);
}

void check_range(const std::vector<uint8_t> & data, size_t offset, size_t bytes) {
    require(offset <= data.size() && bytes <= data.size() - offset, "the test transfer is outside the tensor");
}

void check_guards(const std::vector<uint8_t> & data) {
    for (size_t i = 0; i < guard_bytes; ++i) {
        require(data[i] == guard_value, "the transfer changed the leading guard");
        require(data[data.size() - 1 - i] == guard_value, "the transfer changed the trailing guard");
    }
}

void check_full_tensor(tensor_fixture & fixture, const std::vector<uint8_t> & expected) {
    auto actual = guarded_buffer(expected.size());
    ggml_backend_tensor_get(fixture.tensor, actual.data() + guard_bytes, 0, expected.size());
    check_guards(actual);
    require(std::memcmp(actual.data() + guard_bytes, expected.data(), expected.size()) == 0,
            "the tensor contents differ from the expected flat array");
}

void run_get(tensor_fixture & fixture, size_t offset, size_t bytes) {
    const auto expected = make_values(fixture.tensor, 0);
    check_range(expected, offset, bytes);
    ggml_backend_tensor_set(fixture.tensor, expected.data(), 0, expected.size());
    auto actual = guarded_buffer(bytes);
    ggml_backend_tensor_get_async(fixture.backend, fixture.tensor, actual.data() + guard_bytes, offset, bytes);
    ggml_backend_synchronize(fixture.backend);
    check_guards(actual);
    require(std::memcmp(actual.data() + guard_bytes, expected.data() + offset, bytes) == 0,
            "async get returned the wrong logical range");
    check_full_tensor(fixture, expected);
}

void run_set(tensor_fixture & fixture, size_t offset, size_t bytes) {
    auto expected = make_values(fixture.tensor, 0);
    check_range(expected, offset, bytes);
    ggml_backend_tensor_set(fixture.tensor, expected.data(), 0, expected.size());
    const auto replacement = make_values(fixture.tensor, 100);
    check_range(replacement, offset, bytes);
    auto source = guarded_buffer(bytes);
    std::memcpy(source.data() + guard_bytes, replacement.data() + offset, bytes);
    const auto source_before = source;
    std::memcpy(expected.data() + offset, replacement.data() + offset, bytes);
    ggml_backend_tensor_set_async(fixture.backend, fixture.tensor, source.data() + guard_bytes, offset, bytes);
    ggml_backend_synchronize(fixture.backend);
    require(source == source_before, "async set changed the caller's source bytes");
    check_full_tensor(fixture, expected);
}

} // namespace

int main(int argc, char ** argv) {
    bool do_get = true;
    bool do_set = true;
    bool zero_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--get-only") == 0 && do_get && do_set) {
            do_set = false;
        } else if (std::strcmp(argv[i], "--set-only") == 0 && do_get && do_set) {
            do_get = false;
        } else if (std::strcmp(argv[i], "--zero-only") == 0 && !zero_only) {
            zero_only = true;
        } else {
            std::fprintf(stderr, "usage: %s [--get-only | --set-only] [--zero-only]\n", argv[0]);
            return 1;
        }
    }

    try {
        const auto registry = ggml_backend_cpu_reg();
        require(registry != nullptr, "CPU registry is unavailable");
        const auto cpu = ggml_backend_reg_dev_get(registry, 0);
        require(cpu != nullptr, "CPU device is unavailable");
        static split_config configs[] = {
            {GGML_BACKEND_SPLIT_AXIS_0, "axis0", {10, 6, 1, 1}},
            {GGML_BACKEND_SPLIT_AXIS_1, "axis1", {5, 10, 6, 1}},
            {GGML_BACKEND_SPLIT_AXIS_2, "axis2", {5, 2, 10, 6}},
            {GGML_BACKEND_SPLIT_AXIS_MIRRORED, "mirrored", {10, 6, 1, 1}},
        };
        size_t cases = 0;
        for (auto & config : configs) {
            for (const auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
                tensor_fixture fixture;
                fixture.init(cpu, config, type);
                const int chunk_axis = config.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED ? 0 : int(config.axis);
                const size_t chunk_bytes = fixture.tensor->nb[chunk_axis + 1];
                const size_t ranges[][2] = {{0, 6}, {1, 2}, {5, 1}};
                for (const auto & range : ranges) {
                    if (zero_only && range[0] != 0) {
                        continue;
                    }
                    for (const bool is_set : {false, true}) {
                        if ((is_set && !do_set) || (!is_set && !do_get)) {
                            continue;
                        }
                        std::fprintf(stderr, "RUN %s %s %s start=%zu count=%zu\n", config.name, ggml_type_name(type),
                                     is_set ? "set" : "get", range[0], range[1]);
                        if (is_set) {
                            run_set(fixture, range[0] * chunk_bytes, range[1] * chunk_bytes);
                        } else {
                            run_get(fixture, range[0] * chunk_bytes, range[1] * chunk_bytes);
                        }
                        ++cases;
                    }
                }
            }
        }
        require(cases != 0, "no transfer case ran");
        std::printf("PASS %zu real meta transfer cases on two CPU children\n", cases);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
