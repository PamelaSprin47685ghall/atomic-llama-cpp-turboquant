#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-device-copy.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_failed = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        g_failed++;
    }
}

static void test_sanitize_rule() {
    const float nan = std::nanf("");
    check(ggml_sanitize_nonfinite_f32(nan) == 0.0f, "NaN -> 0");
    check(ggml_sanitize_nonfinite_f32(INFINITY) == GGML_SANITIZE_NONFINITE_F16_MAX, "+Inf -> f16 max");
    check(ggml_sanitize_nonfinite_f32(-INFINITY) == -GGML_SANITIZE_NONFINITE_F16_MAX, "-Inf -> -f16 max");
    check(ggml_sanitize_nonfinite_f32(1.0e6f) == 1.0e6f, "finite massive activation stays");
    check(ggml_sanitize_nonfinite_f32(-42.5f) == -42.5f, "finite negative stays");
    check(ggml_sanitize_nonfinite_f32(0.0f) == 0.0f, "zero stays");
}

// Host features_buf is token-major: token i holds [layer0 | layer1 | ...].
// The device path copies one [n_embd, n_tokens] plane per layer into
// [n_embd, n_tokens, n_aux], then permutes axes (0, 2, 1) and reshapes.
// Those two layouts must name the same element.
static void test_plane_permute_matches_host_interleave() {
    const int n_embd = 4;
    const int n_aux  = 3;
    const int n_tok  = 5;
    const int n_enc  = n_embd * n_aux;

    std::vector<float> layers(size_t(n_aux) * n_tok * n_embd);
    for (int k = 0; k < n_aux; ++k) {
        for (int i = 0; i < n_tok; ++i) {
            for (int j = 0; j < n_embd; ++j) {
                layers[size_t(k * n_tok + i) * n_embd + j] = float(1000 * k + 10 * i + j);
            }
        }
    }

    std::vector<float> host(size_t(n_tok) * n_enc);
    for (int k = 0; k < n_aux; ++k) {
        for (int i = 0; i < n_tok; ++i) {
            for (int j = 0; j < n_embd; ++j) {
                host[size_t(i) * n_enc + k * n_embd + j] =
                    layers[size_t(k * n_tok + i) * n_embd + j];
            }
        }
    }

    std::vector<float> planes(size_t(n_embd) * n_tok * n_aux);
    for (int k = 0; k < n_aux; ++k) {
        std::memcpy(planes.data() + size_t(k) * n_tok * n_embd,
                    layers.data() + size_t(k) * n_tok * n_embd,
                    size_t(n_tok) * n_embd * sizeof(float));
    }

    // permute(0, 2, 1) of [n_embd, n_tokens, n_aux] then contiguous reshape.
    std::vector<float> device(size_t(n_tok) * n_enc);
    for (int i = 0; i < n_tok; ++i) {
        for (int k = 0; k < n_aux; ++k) {
            for (int j = 0; j < n_embd; ++j) {
                device[size_t(i) * n_enc + size_t(k) * n_embd + j] =
                    planes[size_t(k) * n_tok * n_embd + size_t(i) * n_embd + j];
            }
        }
    }
    check(host == device, "device plane permute matches host interleave");
}

static bool device_roundtrip(ggml_backend_dev_t dev) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const bool has_copy = reg && ggml_backend_reg_get_proc_address(reg, "ggml_backend_device_copy_ranges");
    const bool has_san  = reg && ggml_backend_reg_get_proc_address(reg, "ggml_backend_device_sanitize_nonfinite");
    if (!has_copy || !has_san) {
        ggml_backend_free(backend);
        return false;
    }

    ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(dev));
    check(buf != nullptr, "device buffer allocated");
    if (!buf) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return true;
    }

    float raw[8] = { std::nanf(""), INFINITY, -INFINITY, 1.0e6f, -2.5f, 0.0f, 3.0f, 4.0f };
    ggml_backend_tensor_set(src, raw, 0, sizeof(raw));
    float zeros[8] = {};
    ggml_backend_tensor_set(dst, zeros, 0, sizeof(zeros));

    const ggml_device_copy_range range{src, dst, sizeof(float), 2 * sizeof(float), 4 * sizeof(float)};
    check(ggml_backend_device_copy_ranges(backend, &range, 1, true), "copy dry-run");
    check(ggml_backend_device_copy_ranges(backend, &range, 1, false), "copy record");
    check(ggml_backend_device_sanitize_nonfinite(backend, dst, 2 * sizeof(float), 4 * sizeof(float), true), "sanitize dry-run");
    check(ggml_backend_device_sanitize_nonfinite(backend, dst, 2 * sizeof(float), 4 * sizeof(float), false), "sanitize record");
    ggml_backend_synchronize(backend);

    float out[8] = {};
    ggml_backend_tensor_get(dst, out, 0, sizeof(out));
    check(out[0] == 0.0f && out[1] == 0.0f, "bytes outside the copy stay zero");
    check(out[2] == ggml_sanitize_nonfinite_f32(raw[1]), "copied +Inf sanitized");
    check(out[3] == ggml_sanitize_nonfinite_f32(raw[2]), "copied -Inf sanitized");
    check(out[4] == 1.0e6f, "copied finite massive value unchanged");
    check(out[5] == -2.5f, "copied finite value unchanged");
    check(out[6] == 0.0f && out[7] == 0.0f, "tail outside the copy stays zero");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

static void test_device_if_present() {
    ggml_backend_load_all();
    bool ran = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        std::fprintf(stderr, "device %s\n", ggml_backend_dev_name(dev));
        ran = device_roundtrip(dev) || ran;
    }
    if (!ran) {
        std::fprintf(stderr, "no device copy backend; host layout checks only\n");
    }
}

int main() {
    test_sanitize_rule();
    test_plane_permute_matches_host_interleave();
    test_device_if_present();
    if (g_failed) {
        std::fprintf(stderr, "%d checks failed\n", g_failed);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
