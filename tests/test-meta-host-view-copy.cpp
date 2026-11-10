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
#include <cstdint>
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

ggml_backend_meta_split_state split_segmented(const ggml_tensor * t, void *) {
    ggml_backend_meta_split_state s = {};
    s.axis = GGML_BACKEND_SPLIT_AXIS_0;
    s.n_segments = 2;
    s.nr[0] = 1;
    s.nr[1] = 1;
    const int64_t n = t->ne[0];
    s.ne[0] = n / 4;
    s.ne[1] = n / 4;
    s.ne[2] = n / 4;
    s.ne[3] = n - 3 * (n / 4);
    return s;
}

// Same expanded owner sequence as split_segmented, expressed as one segment
// repeated twice: child0/child1/child0/child1.
ggml_backend_meta_split_state split_segmented_repeat(const ggml_tensor * t, void *) {
    ggml_backend_meta_split_state s = {};
    s.axis = GGML_BACKEND_SPLIT_AXIS_0;
    s.n_segments = 1;
    s.nr[0] = 2;
    const int64_t n = t->ne[0];
    s.ne[0] = n / 4;
    s.ne[1] = n / 4;
    return s;
}

ggml_backend_meta_split_state split_repeated(const ggml_tensor * t, void *) {
    ggml_backend_meta_split_state s = {};
    s.axis = GGML_BACKEND_SPLIT_AXIS_0;
    s.n_segments = 1;
    s.nr[0] = 4;
    const int64_t n = t->ne[0];
    s.ne[0] = n / 8;
    s.ne[1] = n / 8;
    return s;
}

ggml_backend_meta_split_state split_q8(const ggml_tensor * t, void *) {
    ggml_backend_meta_split_state s = {};
    s.axis = GGML_BACKEND_SPLIT_AXIS_0;
    s.n_segments = 1;
    s.nr[0] = 1;
    const int64_t n = t->ne[0];
    s.ne[0] = n / 2;
    s.ne[1] = n - n / 2;
    return s;
}

ggml_backend_meta_split_state split_axis1_half(const ggml_tensor * t, void *) {
    ggml_backend_meta_split_state s = {};
    s.axis = GGML_BACKEND_SPLIT_AXIS_1;
    s.n_segments = 1;
    s.nr[0] = 1;
    const int64_t n = t->ne[1];
    s.ne[0] = n / 2;
    s.ne[1] = n - n / 2;
    return s;
}

struct meta_fixture {
    ggml_backend_dev_t          dev  = nullptr;
    ggml_backend_buffer_type_t  buft = nullptr;
    ggml_context *              ctx  = nullptr;
    ggml_tensor *               root  = nullptr;
    ggml_tensor *               empty = nullptr;
    ggml_backend_buffer_t       buf   = nullptr;

    bool init(ggml_backend_dev_t cpu, ggml_backend_meta_get_split_state_t split,
              ggml_type type = GGML_TYPE_F32, int64_t n = 64) {
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
        root = ggml_new_tensor_1d(ctx, type, n);
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

    bool init(ggml_type type = GGML_TYPE_F32, int64_t n = 64) {
        backend = ggml_backend_cpu_init();
        if (backend == nullptr) {
            return false;
        }
        const ggml_init_params params = { 1u << 20, nullptr, true };
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            return false;
        }
        root = ggml_new_tensor_1d(ctx, type, n);
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

void test_segmented_view(ggml_backend_dev_t cpu) {
    meta_fixture m;
    host_fixture h;
    CHECK(m.init(cpu, split_segmented));
    CHECK(h.init());
    if (m.buf == nullptr || h.buf == nullptr) return;
    ggml_tensor * hv = ggml_view_1d(h.ctx, h.root, 48, 8 * sizeof(float));
    ggml_tensor * mv = ggml_view_1d(m.ctx, m.root, 48, 8 * sizeof(float));
    CHECK(hv != nullptr && mv != nullptr);
    CHECK(ggml_backend_view_init(hv) == GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_view_init(mv) == GGML_STATUS_SUCCESS);
    std::vector<float> src(48), sentinel(64, -1.0f), got;
    fill_sequence(src, 6000.0f);
    ggml_backend_tensor_set(m.root, sentinel.data(), 0, sentinel.size() * sizeof(float));
    ggml_backend_tensor_set(hv, src.data(), 0, src.size() * sizeof(float));
    ggml_backend_tensor_copy(hv, mv);
    read_meta(m, got);
    CHECK(range_equal(got, 8, src, 0, 48));
    CHECK(all_value(got, -1.0f, 0, 8));
    CHECK(all_value(got, -1.0f, 56, 64));
}

void test_repeated_view(ggml_backend_dev_t cpu) {
    meta_fixture m;
    host_fixture h;
    CHECK(m.init(cpu, split_repeated));
    CHECK(h.init());
    if (m.buf == nullptr || h.buf == nullptr) return;
    ggml_tensor * hv = ggml_view_1d(h.ctx, h.root, 32, 4 * sizeof(float));
    ggml_tensor * mv = ggml_view_1d(m.ctx, m.root, 32, 4 * sizeof(float));
    CHECK(hv != nullptr && mv != nullptr);
    CHECK(ggml_backend_view_init(hv) == GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_view_init(mv) == GGML_STATUS_SUCCESS);
    std::vector<float> src(32), sentinel(64, -1.0f), got;
    fill_sequence(src, 7000.0f);
    ggml_backend_tensor_set(m.root, sentinel.data(), 0, sentinel.size() * sizeof(float));
    ggml_backend_tensor_set(hv, src.data(), 0, src.size() * sizeof(float));
    ggml_backend_tensor_copy(hv, mv);
    read_meta(m, got);
    CHECK(range_equal(got, 4, src, 0, 32));
    CHECK(all_value(got, -1.0f, 0, 4));
    CHECK(all_value(got, -1.0f, 36, 64));
}

// The parent split state describes one base period while the tensor holds
// two: a whole-period view keeps the base pattern with an internal group
// count, and a prefix partial view stays inside the first period.
static void test_parent_periods_group(ggml_backend_dev_t cpu) {
    meta_fixture m;
    CHECK(m.init(cpu, split_half));
    if (m.buf == nullptr) {
        return;
    }
    ggml_tensor * whole = ggml_view_1d(m.ctx, m.root, 64, 0);
    CHECK(whole != nullptr);
    CHECK(ggml_backend_view_init(whole) == GGML_STATUS_SUCCESS);
    std::vector<float> src(64), got(64, 0.0f);
    fill_sequence(src, 7000.0f);
    ggml_backend_tensor_set(whole, src.data(), 0, src.size() * sizeof(float));
    ggml_backend_tensor_get(m.root, got.data(), 0, got.size() * sizeof(float));
    CHECK(all_equal(got, src, 0, src.size()));

    ggml_tensor * part = ggml_view_1d(m.ctx, m.root, 16, 8 * sizeof(float));
    CHECK(part != nullptr);
    CHECK(ggml_backend_view_init(part) == GGML_STATUS_SUCCESS);
    std::vector<float> psrc(16), pgot(16, 0.0f);
    fill_sequence(psrc, 9000.0f);
    ggml_backend_tensor_set(part, psrc.data(), 0, psrc.size() * sizeof(float));
    ggml_backend_tensor_get(part, pgot.data(), 0, pgot.size() * sizeof(float));
    CHECK(all_equal(pgot, psrc, 0, psrc.size()));
    ggml_backend_tensor_get(m.root, got.data(), 0, got.size() * sizeof(float));
    CHECK(range_equal(got, 8, psrc, 0, psrc.size()));
}

// REPEAT tiles the whole source owner sequence. With split_segmented the
// source sequence is child0/child1/child0/child1; repeating to twice the
// extent must keep that order twice, not stretch each segment.
static void test_repeat_tiling(ggml_backend_dev_t cpu) {
    meta_fixture m;
    CHECK(m.init(cpu, split_segmented));
    if (m.buf == nullptr) {
        return;
    }
    const ggml_init_params rep_params = { 1u << 20, nullptr, true };
    ggml_context * rep_ctx = ggml_init(rep_params);
    CHECK(rep_ctx != nullptr);
    ggml_tensor * shape = ggml_new_tensor_1d(rep_ctx, GGML_TYPE_F32, 128);
    CHECK(shape != nullptr);
    ggml_tensor * rep = ggml_repeat(rep_ctx, m.root, shape);
    CHECK(rep != nullptr);
    rep->flags |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_backend_buffer_t rep_buf = ggml_backend_alloc_ctx_tensors_from_buft(rep_ctx, m.buft);
    CHECK(rep_buf != nullptr);

    ggml_backend_t backend = ggml_backend_dev_init(m.dev, nullptr);
    CHECK(backend != nullptr);
    ggml_cgraph * graph = ggml_new_graph(m.ctx);
    CHECK(graph != nullptr);
    ggml_build_forward_expand(graph, rep);
    std::vector<float> src(64);
    fill_sequence(src, 100.0f);
    ggml_backend_tensor_set(m.root, src.data(), 0, src.size() * sizeof(float));
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> got(128, 0.0f);
    ggml_backend_tensor_get(rep, got.data(), 0, got.size() * sizeof(float));
    for (size_t i = 0; i < got.size(); ++i) {
        CHECK(got[i] == src[i % src.size()]);
    }
    ggml_backend_free(backend);
    ggml_backend_buffer_free(rep_buf);
    ggml_free(rep_ctx);
}

// ADD of two sources: equivalent expanded owner sequences with a different
// segment/nr spelling must compute; truly different owner mappings must fail
// closed without aborting.
static void test_different_owner_add(ggml_backend_dev_t cpu) {
    {
        meta_fixture a;
        meta_fixture b;
        CHECK(a.init(cpu, split_segmented_repeat));
        CHECK(b.init(cpu, split_segmented));
        if (a.buf == nullptr || b.buf == nullptr) {
            return;
        }
        const ggml_init_params add_params = { 1u << 20, nullptr, true };
        ggml_context * add_ctx = ggml_init(add_params);
        CHECK(add_ctx != nullptr);
        ggml_tensor * add = ggml_add(add_ctx, a.root, b.root);
        CHECK(add != nullptr);
        add->flags |= GGML_TENSOR_FLAG_COMPUTE;
        ggml_backend_buffer_t add_buf = ggml_backend_alloc_ctx_tensors_from_buft(add_ctx, a.buft);
        CHECK(add_buf != nullptr);
        std::vector<float> av(64), bv(64);
        fill_sequence(av, 100.0f);
        fill_sequence(bv, 1000.0f);
        ggml_backend_tensor_set(a.root, av.data(), 0, av.size() * sizeof(float));
        ggml_backend_tensor_set(b.root, bv.data(), 0, bv.size() * sizeof(float));
        ggml_backend_t backend = ggml_backend_dev_init(a.dev, nullptr);
        CHECK(backend != nullptr);
        ggml_cgraph * graph = ggml_new_graph(add_ctx);
        CHECK(graph != nullptr);
        ggml_build_forward_expand(graph, add);
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        std::vector<float> got(64, 0.0f);
        ggml_backend_tensor_get(add, got.data(), 0, got.size() * sizeof(float));
        for (size_t i = 0; i < got.size(); ++i) {
            CHECK(got[i] == av[i] + bv[i]);
        }
        ggml_backend_free(backend);
        ggml_backend_buffer_free(add_buf);
        ggml_free(add_ctx);
    }
    {
        meta_fixture a;
        meta_fixture b;
        CHECK(a.init(cpu, split_half));
        CHECK(b.init(cpu, split_segmented));
        if (a.buf == nullptr || b.buf == nullptr) {
            return;
        }
        const ggml_init_params add_params = { 1u << 20, nullptr, true };
        ggml_context * add_ctx = ggml_init(add_params);
        CHECK(add_ctx != nullptr);
        ggml_tensor * add = ggml_add(add_ctx, a.root, b.root);
        CHECK(add != nullptr);
        CHECK(ggml_backend_tensor_alloc(a.buf, add,
                (char *) ggml_backend_buffer_get_base(a.buf)) == GGML_STATUS_FAILED);
        ggml_free(add_ctx);
    }
}

void test_q8_view(ggml_backend_dev_t cpu) {
    meta_fixture m;
    host_fixture h;
    CHECK(m.init(cpu, split_q8, GGML_TYPE_Q8_0, 128));
    CHECK(h.init(GGML_TYPE_Q8_0, 128));
    if (m.buf == nullptr || h.buf == nullptr) return;
    ggml_tensor * hv = ggml_view_1d(h.ctx, h.root, 64, 34);
    ggml_tensor * mv = ggml_view_1d(m.ctx, m.root, 64, 34);
    CHECK(hv != nullptr && mv != nullptr);
    CHECK(ggml_backend_view_init(hv) == GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_view_init(mv) == GGML_STATUS_SUCCESS);

    std::vector<uint8_t> src(68), sentinel(136, 0xff), got(136, 0);
    for (size_t i = 0; i < src.size(); ++i) src[i] = uint8_t(i * 3 + 1);
    ggml_backend_tensor_set(m.root, sentinel.data(), 0, sentinel.size());
    ggml_backend_tensor_set(hv, src.data(), 0, src.size());
    ggml_backend_tensor_copy(hv, mv);
    ggml_backend_tensor_get(m.root, got.data(), 0, got.size());
    CHECK(std::memcmp(got.data() + 34, src.data(), src.size()) == 0);
    for (size_t i = 0; i < 34; ++i) CHECK(got[i] == 0xff);
    for (size_t i = 102; i < 136; ++i) CHECK(got[i] == 0xff);

    std::vector<uint8_t> meta_src(136, 0x11), host_sentinel(136, 0xaa), host_got(136, 0);
    for (size_t i = 0; i < meta_src.size(); ++i) meta_src[i] = uint8_t(100 + i);
    ggml_backend_tensor_set(m.root, meta_src.data(), 0, meta_src.size());
    ggml_backend_tensor_set(h.root, host_sentinel.data(), 0, host_sentinel.size());
    ggml_backend_tensor_copy(mv, hv);
    ggml_backend_tensor_get(h.root, host_got.data(), 0, host_got.size());
    CHECK(std::memcmp(host_got.data() + 34, meta_src.data() + 34, 68) == 0);
    for (size_t i = 0; i < 34; ++i) CHECK(host_got[i] == 0xaa);
    for (size_t i = 102; i < 136; ++i) CHECK(host_got[i] == 0xaa);

    std::vector<uint8_t> part(34, 0x5a), part_got(34, 0);
    ggml_backend_tensor_set(mv, part.data(), 34, 34);
    ggml_backend_tensor_get(mv, part_got.data(), 34, 34);
    CHECK(std::memcmp(part_got.data(), part.data(), 34) == 0);
}

void test_noncontiguous_unaligned(ggml_backend_dev_t cpu) {
    ggml_backend_dev_t children[2] = { cpu, cpu };
    ggml_backend_dev_t dev = ggml_backend_meta_device(children, 2, split_axis1_half, nullptr);
    CHECK(dev != nullptr);
    if (dev == nullptr) return;
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    CHECK(buft != nullptr);
    ggml_init_params params = { 1u << 20, nullptr, true };
    ggml_context * mctx = ggml_init(params);
    ggml_context * hctx = ggml_init(params);
    ggml_tensor * mroot = ggml_new_tensor_2d(mctx, GGML_TYPE_F32, 4, 4);
    ggml_tensor * hroot = ggml_new_tensor_2d(hctx, GGML_TYPE_F32, 4, 4);
    CHECK(mroot != nullptr && hroot != nullptr);
    ggml_backend_buffer_t mbuf = ggml_backend_alloc_ctx_tensors_from_buft(mctx, buft);
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t hbuf = ggml_backend_alloc_ctx_tensors(hctx, cpu_backend);
    CHECK(mbuf != nullptr && hbuf != nullptr);
    ggml_tensor * mv = ggml_view_2d(mctx, mroot, 2, 2, mroot->nb[1], 1 * sizeof(float) + 1 * mroot->nb[1]);
    ggml_tensor * hv = ggml_view_2d(hctx, hroot, 2, 2, hroot->nb[1], 1 * sizeof(float) + 1 * hroot->nb[1]);
    CHECK(mv != nullptr && hv != nullptr);
    CHECK(ggml_backend_view_init(mv) == GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_view_init(hv) == GGML_STATUS_SUCCESS);

    std::vector<float> src = { 10, 11, 12, 13 };
    std::vector<float> sentinel(16, -1.0f), host_values(16, -1.0f), got(16, 0);
    host_values[1 * 4 + 1] = src[0];
    host_values[1 * 4 + 2] = src[1];
    host_values[2 * 4 + 1] = src[2];
    host_values[2 * 4 + 2] = src[3];
    ggml_backend_tensor_set(mroot, sentinel.data(), 0, sentinel.size() * sizeof(float));
    ggml_backend_tensor_set(hroot, host_values.data(), 0, host_values.size() * sizeof(float));
    ggml_backend_tensor_copy(hv, mv);
    ggml_backend_tensor_get(mroot, got.data(), 0, got.size() * sizeof(float));
    CHECK(got[1 * 4 + 1] == 10 && got[1 * 4 + 2] == 11);
    CHECK(got[2 * 4 + 1] == 12 && got[2 * 4 + 2] == 13);
    for (size_t i = 0; i < 16; ++i) {
        if (i != 1 * 4 + 1 && i != 1 * 4 + 2 && i != 2 * 4 + 1 && i != 2 * 4 + 2) {
            CHECK(got[i] == -1.0f);
        }
    }

    ggml_backend_buffer_free(mbuf);
    ggml_backend_buffer_free(hbuf);
    ggml_backend_free(cpu_backend);
    ggml_free(mctx);
    ggml_free(hctx);
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

    std::fprintf(stderr, "test_non_view\n");
    test_non_view(cpu);
    std::fprintf(stderr, "test_cross_rank_view\n");
    test_cross_rank_view(cpu);
    std::fprintf(stderr, "test_different_split\n");
    test_different_split(cpu);
    std::fprintf(stderr, "test_empty_and_alias\n");
    test_empty_and_alias(cpu);
    std::fprintf(stderr, "test_segmented_view\n");
    test_segmented_view(cpu);
    std::fprintf(stderr, "test_repeated_view\n");
    test_repeated_view(cpu);
    std::fprintf(stderr, "test_parent_periods_group\n");
    test_parent_periods_group(cpu);
    std::fprintf(stderr, "test_repeat_tiling\n");
    test_repeat_tiling(cpu);
    std::fprintf(stderr, "test_different_owner_add\n");
    test_different_owner_add(cpu);
    std::fprintf(stderr, "test_q8_view\n");
    test_q8_view(cpu);
    std::fprintf(stderr, "test_noncontiguous_unaligned\n");
    test_noncontiguous_unaligned(cpu);

    if (g_failures != 0) {
        std::fprintf(stderr, "META_HOST_VIEW_COPY_TEST FAIL (%d)\n", g_failures);
        return 1;
    }
    std::printf("META_HOST_VIEW_COPY_TEST PASS\n");
    return 0;
}
