// Deterministic CPU Meta host<->shard copy test.
//
// The Meta buffer can report is_host=true when every child is a host buffer,
// while the top-level Meta tensor data is a virtual sentinel. The generic
// tensor copy must never dereference that sentinel, and the host<->Meta view
// slow path must use the real logical coordinates and child shard mapping.
//
// This test uses two real CPU child devices behind a Meta device. It covers
// non-view and non-zero view copies in both directions, a view that crosses
// the child boundary, different source/destination splits, empty tensors and
// alias no-ops. No GPU and no real model are involved.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

ggml_backend_meta_split_state split_half(const ggml_tensor * t, void *) {
    ggml_backend_meta_split_state s = {};
    s.axis = GGML_BACKEND_SPLIT_AXIS_0;
    s.n_segments = 1;
    s.nr[0] = 1;
    const int64_t n = t->ne[0];
    s.ne[0] = n / 2;
    s.ne[1] = n - n / 2;
    return s;
}

ggml_backend_meta_split_state split_uneven(const ggml_tensor * t, void *) {
    ggml_backend_meta_split_state s = {};
    s.axis = GGML_BACKEND_SPLIT_AXIS_0;
    s.n_segments = 1;
    s.nr[0] = 1;
    const int64_t n = t->ne[0];
    s.ne[0] = n / 4;
    s.ne[1] = n - n / 4;
    return s;
}

struct meta_fixture {
    ggml_backend_dev_t          dev  = nullptr;
    ggml_backend_buffer_type_t  buft = nullptr;
    ggml_context *              ctx  = nullptr;
    ggml_tensor *               root  = nullptr;
    ggml_tensor *               empty = nullptr;
    ggml_backend_buffer_t       buf   = nullptr;

    bool init(ggml_backend_dev_t cpu, ggml_backend_meta_get_split_state_t split) {
        ggml_backend_dev_t children[2] = { cpu, cpu };
        dev = ggml_backend_meta_device(children, 2, split, nullptr);
        if (dev == nullptr) {
            return false;
        }
        buft = ggml_backend_dev_buffer_type(dev);
        if (buft == nullptr) {
            return false;
        }
        const ggml_init_params params = { 1u << 20, nullptr, true };
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            return false;
        }
        root = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        empty = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 0);
        if (root == nullptr || empty == nullptr) {
            return false;
        }
        ggml_set_name(root, "meta_root");
        ggml_set_name(empty, "meta_empty");
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        return buf != nullptr;
    }

    ~meta_fixture() {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct host_fixture {
    ggml_backend_t        backend = nullptr;
    ggml_context *        ctx     = nullptr;
    ggml_tensor *         root    = nullptr;
    ggml_backend_buffer_t buf     = nullptr;

    bool init() {
        backend = ggml_backend_cpu_init();
        if (backend == nullptr) {
            return false;
        }
        const ggml_init_params params = { 1u << 20, nullptr, true };
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            return false;
        }
        root = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        if (root == nullptr) {
            return false;
        }
        ggml_set_name(root, "host_root");
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        return buf != nullptr;
    }

    ~host_fixture() {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

bool read_meta(const meta_fixture & m, std::vector<float> & out) {
    out.assign(64, 0.0f);
    ggml_backend_tensor_get(m.root, out.data(), 0, out.size() * sizeof(float));
    return true;
}

bool read_host(const host_fixture & h, std::vector<float> & out) {
    out.assign(64, 0.0f);
    ggml_backend_tensor_get(h.root, out.data(), 0, out.size() * sizeof(float));
    return true;
}

bool all_equal(const std::vector<float> & a, const std::vector<float> & b, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

bool range_equal(const std::vector<float> & a, size_t a_begin,
                 const std::vector<float> & b, size_t b_begin, size_t count) {
    if (a_begin + count > a.size() || b_begin + count > b.size()) {
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        if (a[a_begin + i] != b[b_begin + i]) {
            return false;
        }
    }
    return true;
}

bool all_value(const std::vector<float> & a, float value, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
        if (a[i] != value) {
            return false;
        }
    }
    return true;
}

void fill_sequence(std::vector<float> & v, float base) {
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = base + float(i);
    }
}

void test_non_view(ggml_backend_dev_t cpu) {
    meta_fixture m;
    host_fixture h;
    CHECK(m.init(cpu, split_half));
    CHECK(h.init());
    if (m.buf == nullptr || h.buf == nullptr) {
        return;
    }

    std::vector<float> src(64), dst(64), got;
    fill_sequence(src, 1000.0f);
    std::fill(dst.begin(), dst.end(), -1.0f);
    ggml_backend_tensor_set(h.root, src.data(), 0, src.size() * sizeof(float));
    ggml_backend_tensor_set(m.root, dst.data(), 0, dst.size() * sizeof(float));

    ggml_backend_tensor_copy(h.root, m.root);
    read_meta(m, got);
    CHECK(all_equal(got, src, 0, 64));

    std::fill(dst.begin(), dst.end(), 0.0f);
    ggml_backend_tensor_set(h.root, dst.data(), 0, dst.size() * sizeof(float));
    ggml_backend_tensor_copy(m.root, h.root);
    read_host(h, got);
    CHECK(all_equal(got, src, 0, 64));
}

void test_cross_rank_view(ggml_backend_dev_t cpu) {
    meta_fixture m;
    host_fixture h;
    CHECK(m.init(cpu, split_half));
    CHECK(h.init());
    if (m.buf == nullptr || h.buf == nullptr) {
        return;
    }

    // Parent [0,64) is split [0,32) | [32,64). The view is [8,40), so child 0
    // contributes [8,32) (24 values) and child 1 contributes [32,40) (8).
    ggml_tensor * host_view = ggml_view_1d(h.ctx, h.root, 32, 8 * sizeof(float));
    ggml_tensor * meta_view = ggml_view_1d(m.ctx, m.root, 32, 8 * sizeof(float));
    CHECK(host_view != nullptr && meta_view != nullptr);
    CHECK(ggml_backend_view_init(host_view) == GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_view_init(meta_view) == GGML_STATUS_SUCCESS);

    std::vector<float> sentinel(64, -1.0f);
    std::vector<float> view_src(32), got;
    fill_sequence(view_src, 2000.0f);
    ggml_backend_tensor_set(m.root, sentinel.data(), 0, sentinel.size() * sizeof(float));
    ggml_backend_tensor_set(host_view, view_src.data(), 0, view_src.size() * sizeof(float));

    // host view -> Meta view
    ggml_backend_tensor_copy(host_view, meta_view);
    read_meta(m, got);
    CHECK(range_equal(got, 8, view_src, 0, 32));
    CHECK(all_value(got, -1.0f, 0, 8));
    CHECK(all_value(got, -1.0f, 40, 64));

    // Meta view -> host view
    std::vector<float> meta_src(64), host_sentinel(64, 7.0f);
    fill_sequence(meta_src, 3000.0f);
    ggml_backend_tensor_set(m.root, meta_src.data(), 0, meta_src.size() * sizeof(float));
    ggml_backend_tensor_set(h.root, host_sentinel.data(), 0, host_sentinel.size() * sizeof(float));
    ggml_backend_tensor_copy(meta_view, host_view);
    read_host(h, got);
    CHECK(all_equal(got, meta_src, 8, 40));
    CHECK(all_value(got, 7.0f, 0, 8));
    CHECK(all_value(got, 7.0f, 40, 64));
}

void test_different_split(ggml_backend_dev_t cpu) {
    meta_fixture src;
    meta_fixture dst;
    CHECK(src.init(cpu, split_half));
    CHECK(dst.init(cpu, split_uneven));
    if (src.buf == nullptr || dst.buf == nullptr) {
        return;
    }

    std::vector<float> values(64), got;
    fill_sequence(values, 4000.0f);
    ggml_backend_tensor_set(src.root, values.data(), 0, values.size() * sizeof(float));
    std::vector<float> zero(64, 0.0f);
    ggml_backend_tensor_set(dst.root, zero.data(), 0, zero.size() * sizeof(float));

    ggml_backend_tensor_copy(src.root, dst.root);
    read_meta(dst, got);
    CHECK(all_equal(got, values, 0, 64));
}

void test_empty_and_alias(ggml_backend_dev_t cpu) {
    meta_fixture m;
    host_fixture h;
    CHECK(m.init(cpu, split_half));
    CHECK(h.init());
    if (m.buf == nullptr || h.buf == nullptr) {
        return;
    }

    // Empty tensors are a no-op and must not reach malloc(0) or a sentinel.
    ggml_tensor * empty_host = ggml_new_tensor_1d(h.ctx, GGML_TYPE_F32, 0);
    CHECK(empty_host != nullptr && m.empty != nullptr);
    ggml_backend_tensor_copy(empty_host, m.empty);
    ggml_backend_tensor_get(m.empty, nullptr, 0, 0);
    CHECK(true);

    // Alias copies are no-ops and preserve bytes.
    std::vector<float> values(64), got;
    fill_sequence(values, 5000.0f);
    ggml_backend_tensor_set(m.root, values.data(), 0, values.size() * sizeof(float));
    ggml_backend_tensor_copy(m.root, m.root);
    read_meta(m, got);
    CHECK(all_equal(got, values, 0, 64));

    ggml_tensor * view = ggml_view_1d(m.ctx, m.root, 16, 4 * sizeof(float));
    CHECK(view != nullptr);
    CHECK(ggml_backend_view_init(view) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_copy(view, view);
    read_meta(m, got);
    CHECK(all_equal(got, values, 0, 64));
}

} // namespace

int main() {
    ggml_backend_reg_t cpu_reg = ggml_backend_cpu_reg();
    if (cpu_reg == nullptr) {
        std::fprintf(stderr, "FAIL: no CPU backend registry\n");
        return 1;
    }
    ggml_backend_dev_t cpu = ggml_backend_reg_dev_get(cpu_reg, 0);
    if (cpu == nullptr) {
        std::fprintf(stderr, "FAIL: no CPU device\n");
        return 1;
    }

    test_non_view(cpu);
    test_cross_rank_view(cpu);
    test_different_split(cpu);
    test_empty_and_alias(cpu);

    if (g_failures != 0) {
        std::fprintf(stderr, "META_HOST_VIEW_COPY_TEST FAIL (%d)\n", g_failures);
        return 1;
    }
    std::printf("META_HOST_VIEW_COPY_TEST PASS\n");
    return 0;
}
