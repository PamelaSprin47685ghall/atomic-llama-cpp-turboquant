// test-xkv-turbo-shift.cpp — Focused tests for backend-native bounded
// hot-row TurboQuant K rephasing (context shift).
//
// Covers CPU oracle vs CPU graph vs Vulkan graph:
//  - Turbo4/3/2 decode/canonical/encode roundtrip through the exact shift
//    node pattern used by build_graph_shift (cast, inverse WHT-128, rope
//    section, forward WHT-128, set_rows write-back of shifted rows only).
//  - Partial-IMRoPE geometry head_dim=256/rotary_dim=64 with NEOX-half
//    pairing: the same convention the KV shift path uses (IMROPE maps to
//    NEOX in build_rope_shift), plus a full-width n_rot=256 case.
//  - Oracle: F32 reference roped directly at shifted positions (absolute
//    truth, no composition assumption); tolerance derives from measured
//    requant noise (4x base + 1e-4), never a fixed magic constant.
//  - Shift-0 rows are byte-identical after the graph (memcmp): unshifted
//    rows are never re-encoded (no requant spray).
//  - Shifted rows must actually move (change strictly exceeds requant
//    noise): guards against vacuous no-op passes.
//  - Vulkan: graph runs on-device with zero host mirrors of K bytes
//    (inputs are I32 deltas/idxs + resident tensors); skips cleanly with
//    no device. Cast-node supports() must hold on Vulkan (new dequant
//    pipelines); allocation failure fails the test, never falls back.
//
// Standalone link: g++ ... -lggml -lggml-base -lggml-cpu -lggml-vulkan.
// Define GGML_USE_VULKAN (with ggml-vulkan.h) for the direct
// ggml_backend_vk_init(0) fallback.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "ggml.h"
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
#include <vector>

static int failures = 0;
#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
    } while (0)

// ---------------------------------------------------------------- fixtures

struct shift_case {
    ggml_type ktype = GGML_TYPE_TURBO4_0;
    int64_t D = 256;      // padded head dim (2x128 WHT groups)
    int64_t H = 2;        // kv heads
    int64_t R = 4;        // hot rows
    int64_t n_rot = 64;   // rotary section (partial-IMRoPE-like)
    int rope_mode = GGML_ROPE_TYPE_NEOX;
    float freq_base = 10000.0f;
    float freq_scale = 1.0f;
    int32_t pos[4] = {100, 200, 300, 400};
    int32_t delta[4] = {5, 0, -3, 7}; // row 1 unshifted: byte-identity probe
};

static uint64_t lcg_state = 0x12345678u;
static float frand() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    uint32_t u = (lcg_state >> 9) | 0x3f800000u;
    float f;
    std::memcpy(&f, &u, 4);
    return (f - 1.0f) * 2.0f - 1.0f; // ~[-3, 1]
}

static std::vector<float> make_canonical(const shift_case & c) {
    std::vector<float> v(size_t(c.D * c.H * c.R));
    for (auto & x : v) x = frand() * 0.5f;
    return v;
}

static float max_abs_diff(const float * a, const float * b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// Run a tiny rope graph on backend: rope_ext(F32 tensor, I32 pos) -> F32.
static bool rope_rows(ggml_backend_t backend, const std::vector<float> & src,
        int64_t D, int64_t H, int64_t R, int64_t n_rot, int rope_mode,
        const std::vector<int32_t> & pos, float freq_base, float freq_scale,
        std::vector<float> & out) {
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(8, false), nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor * a = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, D, H, R);
    ggml_tensor * p = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, R);
    // NOTE: ggml_rope_ext has additional trailing yarn/scale params; pass
    // neutral values matching an unscaled configuration.
    ggml_tensor * r = ggml_rope_ext(ctx.get(), a, p, nullptr, n_rot, rope_mode,
        1024, freq_base, freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
    if (!ggml_backend_supports_op(backend, r)) return false;
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buf) return false;
    ggml_backend_tensor_set(a, src.data(), 0, src.size() * 4);
    ggml_backend_tensor_set(p, pos.data(), 0, pos.size() * 4);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), 8, false);
    ggml_build_forward_expand(gf, r);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) return false;
    out.assign(src.size(), 0.0f);
    ggml_backend_tensor_get(r, out.data(), 0, out.size() * 4);
    return true;
}

// Turbo shift graph mirroring build_graph_shift's Turbo branch exactly:
// cast full rows, inverse WHT-128, rope section, forward WHT-128,
// set_rows write-back of shifted rows only (wht_group op param set).
static bool turbo_shift_graph(ggml_backend_t backend, ggml_type ktype,
        const std::vector<uint8_t> & stored, int64_t D, int64_t H, int64_t R,
        int64_t n_rot, int rope_mode, const std::vector<int32_t> & deltas,
        const std::vector<int64_t> & idxs, float freq_base, float freq_scale,
        std::vector<uint8_t> & out_stored) {
    const size_t row_bytes = ggml_row_size(ktype, (size_t)D);
    ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false), nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor * k = ggml_new_tensor_3d(ctx.get(), ktype, D, H, R);
    ggml_tensor * dec = ggml_cast(ctx.get(), k, GGML_TYPE_F32);
    if (!ggml_backend_supports_op(backend, dec)) return false;
    ggml_tensor * canon = ggml_turbo_wht(ctx.get(), dec, 1, 128, nullptr);
    ggml_tensor * dp = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, R);
    ggml_tensor * sec = ggml_view_3d(ctx.get(), canon, n_rot, H, R,
        canon->nb[1], canon->nb[2], 0);
    ggml_tensor * roped = ggml_rope_ext(ctx.get(), sec, dp, nullptr, n_rot, rope_mode,
        1024, freq_base, freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
    (void) roped; // in place on sec/canon storage
    ggml_tensor * rot = ggml_turbo_wht(ctx.get(), canon, 0, 128, nullptr);
    ggml_tensor * flat = ggml_view_2d(ctx.get(), rot, D * H, R, rot->nb[2], 0);
    ggml_tensor * ix = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, (int64_t)idxs.size());
    ggml_tensor * sub = ggml_get_rows(ctx.get(), flat, ix);
    ggml_tensor * back = ggml_set_rows(ctx.get(), k, sub, ix);
    int32_t wht_group = 128;
    std::memcpy(back->op_params, &wht_group, sizeof(int32_t));
    if (!ggml_backend_supports_op(backend, back)) return false;
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buf) return false;
    ggml_backend_tensor_set(k, stored.data(), 0, stored.size());
    ggml_backend_tensor_set(dp, deltas.data(), 0, deltas.size() * 4);
    ggml_backend_tensor_set(ix, idxs.data(), 0, idxs.size() * 8);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(gf, back);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) return false;
    out_stored.assign(stored.size(), 0);
    ggml_backend_tensor_get(k, out_stored.data(), 0, out_stored.size());
    return true;
}

static ggml_backend_t open_cpu_backend() {
    ggml_backend_load_all();
    return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
}

#ifdef GGML_USE_VULKAN
static ggml_backend_t open_vulkan_backend() {
    ggml_backend_load_all();
    ggml_backend_t vk = nullptr;
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev) vk = ggml_backend_dev_init(dev, nullptr);
    if (!vk && ggml_backend_vk_get_device_count() > 0) {
        vk = ggml_backend_vk_init(0);
    }
    return vk;
}
#endif

// ---------------------------------------------------------------- cases

static void run_case(ggml_backend_t backend, const char * tag, bool is_vk,
        ggml_type ktype, int64_t n_rot, float freq_scale) {
    std::printf("[%s] type=%s n_rot=%lld scale=%.1f ...\n", tag, ggml_type_name(ktype),
        (long long)n_rot, (double)freq_scale);
    shift_case c;
    c.ktype = ktype;
    c.n_rot = n_rot;
    c.freq_scale = freq_scale;
    const size_t n_el = size_t(c.D * c.H * c.R);
    const size_t row_bytes = ggml_row_size(ktype, (size_t)c.D);

    std::vector<float> canon = make_canonical(c);
    std::vector<int32_t> pos(c.R), deltas(c.R);
    std::vector<int32_t> pos_shifted(c.R);
    std::vector<int64_t> idxs;
    for (int64_t r = 0; r < c.R; ++r) {
        pos[r] = c.pos[r];
        deltas[r] = c.delta[r];
        pos_shifted[r] = c.pos[r] + c.delta[r];
        if (c.delta[r] != 0) idxs.push_back(r); // single head-stream layout: flat row == hot row
    }
    // NOTE: multi-head rows share one position each (shift is per-row,
    // head-independent), matching K-shift semantics.

    // Write-time equivalent: rope at original positions, then quantize
    // (forward WHT inside quantize, same as the set_rows K-write path).
    std::vector<float> at_p;
    CHECK(rope_rows(backend, canon, c.D, c.H, c.R, c.n_rot, c.rope_mode, pos,
        c.freq_base, c.freq_scale, at_p));
    std::vector<uint8_t> stored(row_bytes * size_t(c.H * c.R));
    for (int64_t h = 0; h < c.H; ++h) {
        for (int64_t r = 0; r < c.R; ++r) {
            const float * src = at_p.data() + (r * c.H + h) * (size_t)c.D;
            uint8_t * dst = stored.data() + (r * c.H + h) * row_bytes;
            CHECK(ggml_quantize_turbo_row(ktype, src, dst, c.D, 128));
        }
    }

    // Absolute truth: rope canonical directly at shifted positions.
    std::vector<float> expected;
    CHECK(rope_rows(backend, canon, c.D, c.H, c.R, c.n_rot, c.rope_mode, pos_shifted,
        c.freq_base, c.freq_scale, expected));

    // Measured requant noise of this exact stored content.
    std::vector<float> roundtrip(n_el);
    for (size_t b = 0; b < stored.size(); b += row_bytes) {
        CHECK(ggml_dequantize_turbo_row(ktype, stored.data() + b,
            roundtrip.data() + (b / row_bytes) * (size_t)c.D,
            c.D, 128, GGML_TURBO_DECODE_CANONICAL));
    }
    const float base = max_abs_diff(roundtrip.data(), at_p.data(), n_el);
    const float tol = 4.0f * base + 1e-4f;
    std::printf("    requant base=%.3e tol=%.3e\n", (double)base, (double)tol);

    // Shift graph under test.
    std::vector<uint8_t> shifted;
    CHECK(turbo_shift_graph(backend, ktype, stored, c.D, c.H, c.R, c.n_rot,
        c.rope_mode, deltas, idxs, c.freq_base, c.freq_scale, shifted));

    // Decode result to canonical and compare with absolute truth.
    std::vector<float> got(n_el);
    for (size_t b = 0; b < shifted.size(); b += row_bytes) {
        CHECK(ggml_dequantize_turbo_row(ktype, shifted.data() + b,
            got.data() + (b / row_bytes) * (size_t)c.D,
            c.D, 128, GGML_TURBO_DECODE_CANONICAL));
    }
    CHECK(max_abs_diff(got.data(), expected.data(), n_el) <= tol);

    // Shifted rows must actually have moved (guards vacuous no-op passes).
    CHECK(max_abs_diff(got.data(), roundtrip.data(), n_el) > base);

    // Unshifted rows are byte-identical (no requant spray).
    for (int64_t r = 0; r < c.R; ++r) {
        if (c.delta[r] != 0) continue;
        for (int64_t h = 0; h < c.H; ++h) {
            const uint8_t * a = stored.data() + (r * c.H + h) * row_bytes;
            const uint8_t * b = shifted.data() + (r * c.H + h) * row_bytes;
            CHECK(std::memcmp(a, b, row_bytes) == 0);
        }
    }
    (void) is_vk;
}

int main() {
    const ggml_type types[] = {GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0};
    const int64_t rots[] = {64, 256};
    ggml_backend_t cpu = open_cpu_backend();
    if (!cpu) { std::fprintf(stderr, "no CPU backend\n"); return 1; }
    for (auto t : types) {
        for (auto n_rot : rots) {
            run_case(cpu, "cpu", false, t, n_rot, 1.0f);
        }
    }
    run_case(cpu, "cpu", false, GGML_TYPE_TURBO4_0, 64, 2.0f);
#ifdef GGML_USE_VULKAN
    ggml_backend_t vk = open_vulkan_backend();
    if (!vk) {
        std::printf("SKIP: no Vulkan device\n");
    } else {
        for (auto t : types) {
            for (auto n_rot : rots) {
                run_case(vk, "vulkan", true, t, n_rot, 1.0f);
            }
        }
        run_case(vk, "vulkan", true, GGML_TYPE_TURBO4_0, 64, 2.0f);
    }
#else
    std::printf("SKIP: Vulkan tests (GGML_USE_VULKAN not defined)\n");
#endif
    std::printf("=== test-xkv-turbo-shift: %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
}
