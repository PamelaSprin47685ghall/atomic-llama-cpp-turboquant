// Bitwise and local-speed micro for the opt-in exact BF16 load4 N2/N3 path.
//
// Reference (exact oracle): one mul_mat_vec_bf16_load4<threads_per_row> launch
// per activation column, i.e. the admitted T=1 kernel run C times.
// Candidate: one mul_mat_vec_bf16_load4_multi<threads_per_row, C> launch.
// Current model path (information only, not bitwise comparable): the generic
// mul_mat_vec_f<bf16, float, C, block> launch recorded in the captured N3
// graph (grid=(rows,1,1), block=256, shared=128) for the shapes where that
// block selection applies.
//
// The translation unit includes the production source. Point TP5_MMVF_SOURCE
// at a lane copy with -DTP5_MMVF_SOURCE='"/abs/path/src/mmvf.multi.cu"'.
//
// Build (gfx1030, mirrors the lane compile flags):
//   /opt/rocm/core-10.0/lib/llvm/bin/clang++ -O3 -DNDEBUG -std=gnu++17 \
//     -funsafe-math-optimizations --offload-arch=gfx1030 -fPIC \
//     -DGGML_USE_HIP -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SHARED \
//     -DGGML_CUDA_NO_PEER_COPY -DGGML_HIP_NO_MMQ_MFMA -DGGML_HIP_NO_VMM \
//     -DGGML_SCHED_MAX_COPIES=4 -DUSE_PROF_API=1 -D_GNU_SOURCE \
//     -D_XOPEN_SOURCE=600 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
//     -I<tree>/ggml/include -iquote <tree>/ggml/src/ggml-cuda \
//     -x hip micro.cpp -o micro \
//     -L<install>/lib -lggml-hip -lggml-base \
//     -Wl,-rpath,<install>/lib:/opt/rocm/core-10.0/lib --hip-link -lamdhip64
//
// Usage: ./micro [device] [repeats]   (default device 0, repeats 200)
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef TP5_MMVF_SOURCE
#define TP5_MMVF_SOURCE "../../ggml/src/ggml-cuda/mmvf.cu"
#endif
#include TP5_MMVF_SOURCE

static void ck(hipError_t error, const char * what) {
    if (error != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(error));
        std::exit(2);
    }
}
#define CK(x) ck((x), #x)

static uint32_t next_value(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// Finite BF16 bit patterns. Exponent fields stay inside the normal range:
// default 2^-3..2^3, pattern 1 [1,2), pattern 2 about 2^16, pattern 3 about
// 2^-79, pattern 4 signed zero.
static uint16_t bf16_pattern(uint32_t & state, int pattern) {
    const uint32_t r = next_value(state);
    const uint16_t sign = uint16_t((r >> 15) & 0x8000u);
    switch (pattern) {
        case 1:  return uint16_t(sign | (uint16_t(127) << 7) | uint16_t(r & 0x7f));
        case 2:  return uint16_t(sign | (uint16_t(143) << 7) | uint16_t(r & 0x7f));
        case 3:  return uint16_t(sign | (uint16_t(48)  << 7) | uint16_t(r & 0x7f));
        case 4:  return uint16_t(r & 1 ? 0x0000u : 0x8000u);
        default: return uint16_t(sign | (uint16_t(124 + r % 7) << 7) | uint16_t(r & 0x7f));
    }
}

static float y_value(uint32_t & state, int pattern, int column) {
    const uint32_t r = next_value(state);
    switch (pattern) {
        case 1:  return column == 0 ? 0.0f : column == 1 ? 1.0f : ((r & 1) ? -1.0f : 1.0f);
        case 2:  return (int(r % 1024) - 512) * 1.5e15f;
        case 3:  return (int(r % 1024) - 512) * 3.0e-30f;
        case 4:  return 0.0f;
        default: return (int(r % 2048) - 1024) * (1.0f / 1024.0f);
    }
}

static bool best_block_is_256(int ncols) {
    // Same selection as launch_mul_mat_vec_f_cuda for RDNA2 wave32.
    const int warp_size = 32;
    int64_t best = warp_size;
    int64_t best_iter = (ncols + 2 * warp_size - 1) / (2 * warp_size);
    for (int64_t block = 2 * warp_size; block <= 256; block += warp_size) {
        const int64_t iter = (ncols + 2 * block - 1) / (2 * block);
        if (iter < best_iter) {
            best_iter = iter;
            best = block;
        }
    }
    return best == 256;
}

static void launch_reference(const nv_bfloat16 * x, const float * y, float * dst,
                             int ncols, int nrows, int ncols_dst, hipStream_t stream) {
    for (int c = 0; c < ncols_dst; ++c) {
        const float * yc = y + size_t(c) * size_t(ncols);
        float * dc = dst + size_t(c) * size_t(nrows);
        if (ncols <= 512) {
            mul_mat_vec_bf16_load4<32><<<dim3((nrows + 3) / 4), dim3(128), 0, stream>>>(x, yc, dc, ncols, nrows);
        } else {
            mul_mat_vec_bf16_load4<256><<<dim3(nrows), dim3(256), 0, stream>>>(x, yc, dc, ncols, nrows);
        }
    }
    CK(hipGetLastError());
}

static void launch_candidate(const nv_bfloat16 * x, const float * y, float * dst,
                             int ncols, int nrows, int ncols_dst, hipStream_t stream) {
    if (ncols <= 512) {
        if (ncols_dst == 2) {
            mul_mat_vec_bf16_load4_multi<32, 2><<<dim3((nrows + 3) / 4), dim3(128), 0, stream>>>(x, y, dst, ncols, nrows);
        } else {
            mul_mat_vec_bf16_load4_multi<32, 3><<<dim3((nrows + 3) / 4), dim3(128), 0, stream>>>(x, y, dst, ncols, nrows);
        }
    } else {
        if (ncols_dst == 2) {
            mul_mat_vec_bf16_load4_multi<256, 2><<<dim3(nrows), dim3(256), 0, stream>>>(x, y, dst, ncols, nrows);
        } else {
            mul_mat_vec_bf16_load4_multi<256, 3><<<dim3(nrows), dim3(256), 0, stream>>>(x, y, dst, ncols, nrows);
        }
    }
    CK(hipGetLastError());
}

template<int ncols_dst>
static void launch_generic_t(const nv_bfloat16 * x, const float * y, float * dst,
                             int ncols, int nrows, hipStream_t stream) {
    const ggml_cuda_mm_fusion_args_device fusion{};
    const uint3 zero = make_uint3(0, 0, 0);
    const uint3 one = init_fastdiv_values(1);
    const int ncols2 = ncols / 2;
    mul_mat_vec_f<nv_bfloat16, float, ncols_dst, 256, false, false>
        <<<dim3(nrows, 1, 1), dim3(256, 1, 1), 128, stream>>>(
            x, y, nullptr, fusion, dst,
            ncols2, zero,
            ncols, ncols2, nrows,
            one, ncols * nrows, ncols * ncols_dst, nrows * ncols_dst,
            one, ncols * nrows, ncols * ncols_dst, nrows * ncols_dst,
            0);
    CK(hipGetLastError());
}

static void launch_generic(const nv_bfloat16 * x, const float * y, float * dst,
                           int ncols, int nrows, int ncols_dst, hipStream_t stream) {
    if (ncols_dst == 2) {
        launch_generic_t<2>(x, y, dst, ncols, nrows, stream);
    } else {
        launch_generic_t<3>(x, y, dst, ncols, nrows, stream);
    }
}

static float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

typedef void (*launch_fn)(const nv_bfloat16 *, const float *, float *, int, int, int, hipStream_t);

static double time_us(launch_fn launch, const nv_bfloat16 * x, const float * y, float * dst,
                      int ncols, int nrows, int ncols_dst, hipStream_t stream, int repeats) {
    hipEvent_t begin, end;
    CK(hipEventCreate(&begin));
    CK(hipEventCreate(&end));
    for (int i = 0; i < 20; ++i) {
        launch(x, y, dst, ncols, nrows, ncols_dst, stream);
    }
    CK(hipStreamSynchronize(stream));
    std::vector<float> samples;
    for (int round = 0; round < 5; ++round) {
        CK(hipEventRecord(begin, stream));
        for (int i = 0; i < repeats; ++i) {
            launch(x, y, dst, ncols, nrows, ncols_dst, stream);
        }
        CK(hipEventRecord(end, stream));
        CK(hipEventSynchronize(end));
        float ms = 0.0f;
        CK(hipEventElapsedTime(&ms, begin, end));
        samples.push_back(ms * 1000.0f / float(repeats));
    }
    CK(hipEventDestroy(begin));
    CK(hipEventDestroy(end));
    return median(samples);
}

struct shape {
    int ncols;
    int nrows;
    const char * note;
};

static int run_shape(int device, int warp_size, int repeats, const shape & s) {
    constexpr int guard = 64;
    const int ncols_dst_max = 3;
    if (warp_size != 32) {
        std::fprintf(stderr, "device %d warpSize=%d is not the admitted wave32 target\n", device, warp_size);
        return 1;
    }

    const size_t weight_count = size_t(s.ncols) * size_t(s.nrows);
    const size_t act_count = size_t(s.ncols) * size_t(ncols_dst_max);
    const size_t out_count = size_t(s.nrows) * size_t(ncols_dst_max);

    std::vector<uint16_t> host_weights(weight_count);
    std::vector<float> host_y(act_count);
    std::vector<float> host_ref(out_count + 2 * guard, 123456.0f);
    std::vector<float> host_cand(out_count + 2 * guard, 123456.0f);
    std::vector<float> host_gen(out_count + 2 * guard, 123456.0f);

    uint16_t * dev_weights_bits = nullptr;
    float * dev_y = nullptr;
    float * dev_ref = nullptr;
    float * dev_cand = nullptr;
    float * dev_gen = nullptr;
    CK(hipMalloc(reinterpret_cast<void **>(&dev_weights_bits), weight_count * sizeof(uint16_t)));
    CK(hipMalloc(reinterpret_cast<void **>(&dev_y), act_count * sizeof(float)));
    CK(hipMalloc(reinterpret_cast<void **>(&dev_ref), host_ref.size() * sizeof(float)));
    CK(hipMalloc(reinterpret_cast<void **>(&dev_cand), host_cand.size() * sizeof(float)));
    CK(hipMalloc(reinterpret_cast<void **>(&dev_gen), host_gen.size() * sizeof(float)));

    const nv_bfloat16 * dev_weights = reinterpret_cast<const nv_bfloat16 *>(dev_weights_bits);
    float * dev_ref_used = dev_ref + guard;
    float * dev_cand_used = dev_cand + guard;
    float * dev_gen_used = dev_gen + guard;

    hipStream_t stream;
    CK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    int failures = 0;
    for (int pattern = 0; pattern < 5; ++pattern) {
        uint32_t state = 0x381a52f1u + uint32_t(pattern) * 0x9e3779b9u;
        for (size_t i = 0; i < weight_count; ++i) {
            host_weights[i] = bf16_pattern(state, pattern);
        }
        for (int c = 0; c < ncols_dst_max; ++c) {
            for (int col = 0; col < s.ncols; ++col) {
                host_y[size_t(c) * size_t(s.ncols) + size_t(col)] = y_value(state, pattern, c);
            }
        }
        std::fill(host_ref.begin(), host_ref.end(), 123456.0f);
        std::fill(host_cand.begin(), host_cand.end(), 123456.0f);
        std::fill(host_gen.begin(), host_gen.end(), 123456.0f);
        CK(hipMemcpy(dev_weights_bits, host_weights.data(), weight_count * sizeof(uint16_t), hipMemcpyHostToDevice));
        CK(hipMemcpy(dev_y, host_y.data(), act_count * sizeof(float), hipMemcpyHostToDevice));
        CK(hipMemcpy(dev_ref, host_ref.data(), host_ref.size() * sizeof(float), hipMemcpyHostToDevice));
        CK(hipMemcpy(dev_cand, host_cand.data(), host_cand.size() * sizeof(float), hipMemcpyHostToDevice));
        CK(hipMemcpy(dev_gen, host_gen.data(), host_gen.size() * sizeof(float), hipMemcpyHostToDevice));

        for (int ncols_dst = 2; ncols_dst <= 3; ++ncols_dst) {
            const size_t used = size_t(s.nrows) * size_t(ncols_dst);
            CK(hipMemsetAsync(dev_ref_used, 0x00, used * sizeof(float), stream));
            CK(hipMemsetAsync(dev_cand_used, 0x00, used * sizeof(float), stream));
            launch_reference(dev_weights, dev_y, dev_ref_used, s.ncols, s.nrows, ncols_dst, stream);
            launch_candidate(dev_weights, dev_y, dev_cand_used, s.ncols, s.nrows, ncols_dst, stream);
            CK(hipStreamSynchronize(stream));
            CK(hipMemcpy(host_ref.data(), dev_ref, host_ref.size() * sizeof(float), hipMemcpyDeviceToHost));
            CK(hipMemcpy(host_cand.data(), dev_cand, host_cand.size() * sizeof(float), hipMemcpyDeviceToHost));

            size_t different = 0, nonfinite_ref = 0, nonfinite_cand = 0;
            double max_abs = 0.0;
            for (size_t i = 0; i < used; ++i) {
                const float a = host_ref[guard + i], b = host_cand[guard + i];
                if (!std::isfinite(a)) ++nonfinite_ref;
                if (!std::isfinite(b)) ++nonfinite_cand;
                if (std::memcmp(&a, &b, sizeof(float)) != 0) {
                    ++different;
                    if (std::isfinite(a) && std::isfinite(b)) {
                        max_abs = std::max(max_abs, std::fabs(double(a) - double(b)));
                    }
                }
            }
            int guard_hits = 0;
            for (int i = 0; i < guard; ++i) {
                if (host_ref[i] != 123456.0f || host_cand[i] != 123456.0f ||
                    host_ref[guard + used + i] != 123456.0f || host_cand[guard + used + i] != 123456.0f) {
                    ++guard_hits;
                }
            }
            const bool pass = different == 0 && guard_hits == 0;
            if (!pass) {
                ++failures;
            }
            std::printf("{\"kind\":\"bitwise\",\"device\":%d,\"note\":\"%s\",\"ncols\":%d,\"nrows\":%d,\"ncols_dst\":%d,"
                        "\"pattern\":%d,\"values\":%zu,\"different_bits\":%zu,\"max_abs\":%.17g,"
                        "\"nonfinite_ref\":%zu,\"nonfinite_cand\":%zu,\"guard_writes\":%d,\"status\":\"%s\"}\n",
                        device, s.note, s.ncols, s.nrows, ncols_dst, pattern, used, different, max_abs,
                        nonfinite_ref, nonfinite_cand, guard_hits, pass ? "PASS" : "FAIL");
            std::fflush(stdout);
        }
    }

    // Local speed on the default pattern only. The reference arm is the exact
    // C-launch oracle; the generic arm is the current model path and is only
    // timed where its production block selection is 256.
    const bool generic_supported = best_block_is_256(s.ncols);
    for (int ncols_dst = 2; ncols_dst <= 3; ++ncols_dst) {
        const double reference_us = time_us(launch_reference, dev_weights, dev_y, dev_ref_used,
                                            s.ncols, s.nrows, ncols_dst, stream, repeats);
        const double candidate_us = time_us(launch_candidate, dev_weights, dev_y, dev_cand_used,
                                            s.ncols, s.nrows, ncols_dst, stream, repeats);
        const double generic_us = generic_supported
            ? time_us(launch_generic, dev_weights, dev_y, dev_gen_used, s.ncols, s.nrows, ncols_dst, stream, repeats)
            : -1.0;
        std::printf("{\"kind\":\"timing\",\"device\":%d,\"note\":\"%s\",\"ncols\":%d,\"nrows\":%d,\"ncols_dst\":%d,"
                    "\"generic_block256\":%s,\"reference_us\":%.9g,\"candidate_us\":%.9g,\"generic_us\":%.9g,"
                    "\"candidate_over_reference\":%.9g,\"candidate_over_generic\":%.9g}\n",
                    device, s.note, s.ncols, s.nrows, ncols_dst, generic_supported ? "true" : "false",
                    reference_us, candidate_us, generic_us,
                    candidate_us / reference_us, generic_supported ? candidate_us / generic_us : 0.0);
        std::fflush(stdout);
    }

    CK(hipStreamSynchronize(stream));
    CK(hipStreamDestroy(stream));
    CK(hipFree(dev_weights_bits));
    CK(hipFree(dev_y));
    CK(hipFree(dev_ref));
    CK(hipFree(dev_cand));
    CK(hipFree(dev_gen));
    return failures;
}

int main(int argc, char ** argv) {
    int device = 0;
    int repeats = 200;
    if (argc > 1) {
        device = std::atoi(argv[1]);
    }
    if (argc > 2) {
        repeats = std::atoi(argv[2]);
    }
    const int count = ggml_cuda_info().device_count;
    if (device < 0 || device >= count) {
        std::fprintf(stderr, "device %d outside 0..%d\n", device, count - 1);
        return 2;
    }
    CK(hipSetDevice(device));
    hipDeviceProp_t props{};
    CK(hipGetDeviceProperties(&props, device));
    std::printf("{\"kind\":\"start\",\"device\":%d,\"name\":\"%s\",\"gcn\":\"%s\",\"warp_size\":%d,\"repeats\":%d}\n",
                device, props.name, props.gcnArchName, props.warpSize, repeats);

    // Real N3 graph rows (captured grid.x) with model weight column counts.
    const std::array<shape, 10> shapes = {{
        {10240,    4, "hc_inject"},
        { 2560,    6, "ssm_ab_shard6"},
        { 2560,   12, "ssm_ab_shard12"},
        { 2560,   48, "ssm_ab_full"},
        { 2560,  128, "indexer_k"},
        { 2560,  512, "indexer_q"},
        { 2560, 2560, "ple_value"},
        {  320, 10240, "hc_up_short"},
        {  512,   48, "short_boundary"},
        {  516,   12, "long_boundary"},
    }};

    int failures = 0;
    for (const shape & s : shapes) {
        failures += run_shape(device, props.warpSize, repeats, s);
    }
    std::printf("{\"kind\":\"summary\",\"device\":%d,\"failures\":%d,\"status\":\"%s\"}\n",
                device, failures, failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
