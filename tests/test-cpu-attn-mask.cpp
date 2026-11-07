// CPU regressions for explicit attention masks and exact-capacity V storage.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__unix__)
#include <sys/mman.h>
#include <unistd.h>
#endif

struct guarded_buffer {
#if defined(__unix__)
    char * base;
    size_t bytes;
    size_t page;
    explicit guarded_buffer(size_t n) : page((size_t) sysconf(_SC_PAGESIZE)) {
        bytes = ((n + page - 1)/page + 1)*page;
        base = (char *) mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED || mprotect(base + bytes - page, page, PROT_NONE) != 0) std::abort();
    }
    ~guarded_buffer() { munmap(base, bytes); }
    void * end_aligned(size_t n) { return base + bytes - page - n; }
#else
    std::vector<unsigned char> storage;
    explicit guarded_buffer(size_t n) : storage(n) {}
    void * end_aligned(size_t) { return storage.data(); }
#endif
};

static void set_value(ggml_tensor * t, int row, int col, float value) {
    char * p = (char *) t->data + row*t->nb[1] + col*t->nb[0];
    if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(value);
        std::memcpy(p, &h, sizeof(h));
    } else {
        std::memcpy(p, &value, sizeof(value));
    }
}

static void set4(ggml_tensor * t, int i, int j, int h, int b, float value) {
    char * p = (char *) t->data + i*t->nb[0] + j*t->nb[1] + h*t->nb[2] + b*t->nb[3];
    if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t v = ggml_fp32_to_fp16(value);
        std::memcpy(p, &v, sizeof(v));
    } else {
        std::memcpy(p, &value, sizeof(value));
    }
}

static bool visible(int col, int query, int live) {
    return col < live && col != 2 && !(col > 2 && (col + query*3) % 11 == 0);
}

// Each graph uses the real CPU softmax and mask-aware ggml MUL_MAT.
// The live V tensor ends at a protected page, not in a wider allocation.
static bool check_case(ggml_type type, ggml_type mask_type, int live, int reference, int rows, int queries, bool compare_widths, bool pedantic) {
    ggml_init_params params = { 8*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_tensor * v = ggml_new_tensor_2d(ctx, type, live, rows);
    ggml_tensor * vref = ggml_new_tensor_2d(ctx, type, reference, rows);
    guarded_buffer guard(ggml_nbytes(v));
    v->data = guard.end_aligned(ggml_nbytes(v));
    ggml_tensor * s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, live, queries);
    ggml_tensor * sr = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, reference, queries);
    ggml_tensor * m = ggml_new_tensor_2d(ctx, mask_type, live, queries);
    ggml_tensor * mw = ggml_new_tensor_2d(ctx, mask_type, live + 3, queries);
    ggml_tensor * mr = ggml_new_tensor_2d(ctx, mask_type, reference, queries);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < reference; ++i) {
            // Column 2 is masked for every query, but the other interior
            // masks vary by query. Include both signs of zero in visible V.
            float x = ((r*29 + i*17) % 37 - 18) / 19.0f;
            if ((r + i) % 13 == 0) x = (r & 1) ? -0.0f : 0.0f;
            const float dirty = (i == 2 || i >= live) ? (i % 3 == 0 ? -INFINITY : nan) : x;
            set_value(vref, r, i, compare_widths ? dirty : (i == 2 ? 0.0f : x));
            if (i < live) set_value(v, r, i, dirty);
        }
    }
    for (int q = 0; q < queries; ++q) {
        for (int i = 0; i < reference; ++i) {
            const bool yes = visible(i, q, live);
            const float score = ((i*7 + q*5) % 23 - 11)/8.0f;
            set_value(sr, q, i, yes ? score : (compare_widths ? nan : 0.0f));
            set_value(mr, q, i, yes ? 0.0f : -INFINITY);
            if (i < live) {
                set_value(s, q, i, yes ? score : nan);
                set_value(m, q, i, yes ? 0.0f : -INFINITY);
                set_value(mw, q, i, yes ? 0.0f : -INFINITY);
            }
        }
        for (int i = live; i < live + 3; ++i) set_value(mw, q, i, -INFINITY);
    }
    ggml_tensor * p = ggml_soft_max_ext(ctx, s, m, 1.0f, 0.0f);
    ggml_tensor * pr = ggml_soft_max_ext(ctx, sr, mr, 1.0f, 0.0f);
    ggml_tensor * a = ggml_mul_mat(ctx, v, p);
    ggml_tensor * b = ggml_mul_mat(ctx, vref, pr);
    ggml_mul_mat_set_attn_mask(a, mw);
    ggml_mul_mat_set_attn_mask(b, mr);
    if (type == GGML_TYPE_F32 || pedantic) {
        ggml_mul_mat_set_prec(a, GGML_PREC_F32_PEDANTIC);
        ggml_mul_mat_set_prec(b, GGML_PREC_F32_PEDANTIC);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, a);
    ggml_build_forward_expand(graph, b);
    bool ok = ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS;
    if (ok) {
        for (int q = 0; q < queries; ++q) {
            if (std::memcmp((float *) p->data + q*live, (float *) pr->data + q*reference, live*sizeof(float)) ||
                std::memcmp((float *) a->data + q*rows, (float *) b->data + q*rows, rows*sizeof(float))) {
                std::fprintf(stderr, "mask mismatch type=%d mask=%d live=%d ref=%d rows=%d queries=%d q=%d widths=%d\n",
                             int(type), int(mask_type), live, reference, rows, queries, q, int(compare_widths));
                ok = false;
            }
        }
    }
    ggml_free(ctx);
    return ok;
}

// An unmasked score can underflow to zero; its NaN V must still propagate.
static bool check_unmasked_underflow(ggml_type type) {
    ggml_init_params params = { 1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_tensor * v = ggml_new_tensor_2d(ctx, type, 2, 1);
    ggml_tensor * s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    set_value(v, 0, 0, 1.0f); set_value(v, 0, 1, std::numeric_limits<float>::quiet_NaN());
    set_value(s, 0, 0, 0.0f); set_value(s, 0, 1, -1000.0f);
    set_value(s, 1, 0, 0.0f); set_value(s, 1, 1, -1000.0f);
    set_value(m, 0, 0, 0.0f); set_value(m, 0, 1, -INFINITY);
    set_value(m, 1, 0, 0.0f); set_value(m, 1, 1, 0.0f);
    ggml_tensor * p = ggml_soft_max_ext(ctx, s, m, 1.0f, 0.0f);
    ggml_tensor * out = ggml_mul_mat(ctx, v, p);
    ggml_mul_mat_set_attn_mask(out, m);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    const bool ok = ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS &&
                    ((float *) p->data)[3] == 0.0f && std::isfinite(((float *) out->data)[0]) &&
                    std::isnan(((float *) out->data)[1]);
    if (!ok) std::fprintf(stderr, "unmasked underflow suppressed nonfinite V\n");
    ggml_free(ctx);
    return ok;
}

static bool check_half_conversion() {
    ggml_init_params params = { 1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_tensor * v = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 3, 1);
    ggml_tensor * s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 1);
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 1);
    for (int i = 0; i < 3; ++i) {
        set_value(v, 0, i, i == 0 ? 1.0f : 0.0f);
        set_value(s, 0, i, 0.0f);
        set_value(m, 0, i, 0.0f);
    }
    ggml_tensor * p = ggml_soft_max_ext(ctx, s, m, 1.0f, 0.0f);
    ggml_tensor * out = ggml_mul_mat(ctx, v, p);
    ggml_mul_mat_set_attn_mask(out, m);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_t backend = ggml_backend_cpu_init();
    bool ok = backend && ggml_backend_supports_op(backend, p) &&
              ggml_backend_supports_op(backend, out) &&
              ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (backend) ggml_backend_free(backend);
    if (ok) {
        const float probability = ((const float *) p->data)[0];
        const float rounded = ggml_fp16_to_fp32(ggml_fp32_to_fp16(probability));
        ok = probability != rounded && std::memcmp(out->data, &rounded, sizeof(float)) == 0;
    }
    if (!ok) std::fprintf(stderr, "F16 probability conversion mismatch\n");
    ggml_free(ctx);
    return ok;
}

static bool check_all_masked(ggml_type type, bool use_sink) {
    ggml_init_params params = { 1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_tensor * v = ggml_new_tensor_2d(ctx, type, 3, 1);
    ggml_tensor * s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 1);
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 1);
    for (int i = 0; i < 3; ++i) {
        set_value(v, 0, i, std::numeric_limits<float>::quiet_NaN());
        set_value(s, 0, i, std::numeric_limits<float>::quiet_NaN());
        set_value(m, 0, i, -INFINITY);
    }
    ggml_tensor * p = ggml_soft_max_ext(ctx, s, m, 1.0f, 0.0f);
    if (use_sink) {
        ggml_tensor * sink = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ((float *) sink->data)[0] = 0.5f;
        ggml_soft_max_add_sinks(p, sink);
    }
    ggml_tensor * out = ggml_mul_mat(ctx, v, p);
    ggml_mul_mat_set_attn_mask(out, m);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    bool ok = ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS;
    const float zero = 0.0f;
    if (ok) {
        ok = std::memcmp(out->data, &zero, sizeof(zero)) == 0;
        for (int i = 0; i < 3; ++i) {
            ok = ok && std::memcmp((float *) p->data + i, &zero, sizeof(zero)) == 0;
        }
    }
    if (!ok) std::fprintf(stderr, "all-masked result mismatch type=%d sink=%d\n", int(type), int(use_sink));
    ggml_free(ctx);
    return ok;
}

static bool check_signed_zero(ggml_type type) {
    ggml_init_params params = { 1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_tensor * v = ggml_new_tensor_2d(ctx, type, 3, 1);
    ggml_tensor * s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 1);
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 1);
    for (int i = 0; i < 3; ++i) {
        set_value(v, 0, i, i == 1 ? std::numeric_limits<float>::quiet_NaN() : -0.0f);
        set_value(s, 0, i, 0.0f);
        set_value(m, 0, i, i == 1 ? -INFINITY : 0.0f);
    }
    ggml_tensor * p = ggml_soft_max_ext(ctx, s, m, 1.0f, 0.0f);
    ggml_tensor * out = ggml_mul_mat(ctx, v, p);
    ggml_mul_mat_set_attn_mask(out, m);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    bool ok = ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS;
    const float zero = 0.0f;
    if (ok) ok = std::memcmp(out->data, &zero, sizeof(zero)) == 0;
    if (!ok) std::fprintf(stderr, "signed zero mismatch type=%d\n", int(type));
    ggml_free(ctx);
    return ok;
}

static bool check_broadcast(ggml_type type, ggml_type mask_type) {
    constexpr int live = 5, reference = 9, rows = 3, queries = 2, heads = 4, batches = 2;
    ggml_init_params params = { 8*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_tensor * v = ggml_new_tensor_4d(ctx, type, live, rows, 2, 1);
    ggml_tensor * vr = ggml_new_tensor_4d(ctx, type, reference, rows, 2, 1);
    guarded_buffer guard(ggml_nbytes(v));
    v->data = guard.end_aligned(ggml_nbytes(v));
    ggml_tensor * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, live, queries, heads, batches);
    ggml_tensor * sr = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, reference, queries, heads, batches);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, mask_type, live, queries, 2, 1);
    ggml_tensor * mr = ggml_new_tensor_4d(ctx, mask_type, reference, queries, 2, 1);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int h = 0; h < 2; ++h) {
        for (int r = 0; r < rows; ++r) {
            for (int i = 0; i < reference; ++i) {
                const float x = (i == 2 || i >= live) ? nan : (float) (i + 3*r - 2*h)/8.0f;
                set4(vr, i, r, h, 0, x);
                if (i < live) set4(v, i, r, h, 0, x);
            }
        }
        for (int q = 0; q < queries; ++q) {
            for (int i = 0; i < reference; ++i) {
                const bool masked = i == 2 || i >= live || (i == 1 && q == 1 && h == 0);
                set4(mr, i, q, h, 0, masked ? -INFINITY : 0.0f);
                if (i < live) set4(m, i, q, h, 0, masked ? -INFINITY : 0.0f);
            }
        }
    }
    for (int b = 0; b < batches; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int q = 0; q < queries; ++q) {
                for (int i = 0; i < reference; ++i) {
                    const bool masked = i == 2 || i >= live || (i == 1 && q == 1 && h % 2 == 0);
                    const float x = masked ? nan : (float) (i - q + h + b)/7.0f;
                    set4(sr, i, q, h, b, x);
                    if (i < live) set4(s, i, q, h, b, x);
                }
            }
        }
    }
    ggml_tensor * p = ggml_soft_max_ext(ctx, s, m, 1.0f, 0.0f);
    ggml_tensor * pr = ggml_soft_max_ext(ctx, sr, mr, 1.0f, 0.0f);
    ggml_tensor * a = ggml_mul_mat(ctx, v, p);
    ggml_tensor * ar = ggml_mul_mat(ctx, vr, pr);
    ggml_mul_mat_set_attn_mask(a, m);
    ggml_mul_mat_set_attn_mask(ar, mr);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, a);
    ggml_build_forward_expand(graph, ar);
    bool ok = ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS;
    if (ok) ok = std::memcmp(a->data, ar->data, ggml_nbytes(a)) == 0;
    if (!ok) std::fprintf(stderr, "broadcast attention mismatch type=%d mask=%d\n", int(type), int(mask_type));
    ggml_free(ctx);
    return ok;
}

// Compare both the upstream dispatch and the marked CPU score path.
static bool check_kq_dispatch(int live, int reference, int queries,
                              bool ordered, ggml_type key_type, ggml_type mask_type) {
    constexpr int d = 32, rows = 16;
    ggml_init_params params = { 8*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_tensor * k = ggml_new_tensor_2d(ctx, key_type, d, live);
    ggml_tensor * kr = ggml_new_tensor_2d(ctx, key_type, d, reference);
    ggml_tensor * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, queries);
    ggml_tensor * v = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, live, rows);
    ggml_tensor * vr = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, reference, rows);
    ggml_tensor * m = ggml_new_tensor_2d(ctx, mask_type, live, queries);
    ggml_tensor * mr = ggml_new_tensor_2d(ctx, mask_type, reference, queries);
    guarded_buffer guard(ggml_nbytes(v));
    v->data = guard.end_aligned(ggml_nbytes(v));
    for (int row = 0; row < reference; ++row) {
        for (int i = 0; i < d; ++i) {
            const float x = ((i*31 + row*23 + 7) % 113 - 56)/32.0f;
            set_value(kr, row, i, ordered && row >= live ? std::numeric_limits<float>::quiet_NaN() : x);
            if (row < live) set_value(k, row, i, x);
        }
        for (int col = 0; col < queries; ++col) {
            set_value(mr, col, row, row < live ? 0.0f : -INFINITY);
            if (row < live) set_value(m, col, row, 0.0f);
        }
        for (int r = 0; r < rows; ++r) {
            const float x = row < live ? ((row*17 + r*11) % 41 - 20)/13.0f
                                       : std::numeric_limits<float>::quiet_NaN();
            set_value(vr, r, row, x);
            if (row < live) set_value(v, r, row, x);
        }
    }
    for (int col = 0; col < queries; ++col) {
        for (int i = 0; i < d; ++i) set_value(q, col, i, ((i*37 + col*13 + 3) % 97 - 48)/31.0f);
    }
    ggml_tensor * score = ggml_mul_mat(ctx, k, q);
    ggml_tensor * score_ref = ggml_mul_mat(ctx, kr, q);
    if (ordered) {
        ggml_mul_mat_set_attn_scores(score, m);
        ggml_mul_mat_set_attn_scores(score_ref, mr);
    }
    ggml_mul_mat_set_prec(score, GGML_PREC_F32);
    ggml_mul_mat_set_prec(score_ref, GGML_PREC_F32);
    ggml_tensor * p = ggml_soft_max_ext(ctx, score, m, 1.0f, 0.0f);
    ggml_tensor * pr = ggml_soft_max_ext(ctx, score_ref, mr, 1.0f, 0.0f);
    ggml_tensor * out = ggml_mul_mat(ctx, v, p);
    ggml_tensor * outr = ggml_mul_mat(ctx, vr, pr);
    ggml_mul_mat_set_attn_mask(out, m);
    ggml_mul_mat_set_attn_mask(outr, mr);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_build_forward_expand(graph, outr);
    bool ok = ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS;
    if (ok) {
        for (int col = 0; col < queries; ++col) {
            if (std::memcmp((float *) score->data + col*live, (float *) score_ref->data + col*reference, live*sizeof(float)) ||
                std::memcmp((float *) out->data + col*rows, (float *) outr->data + col*rows, rows*sizeof(float))) {
                std::fprintf(stderr, "KQ or output mismatch ordered=%d key=%d mask=%d live=%d ref=%d queries=%d\n",
                             int(ordered), int(key_type), int(mask_type), live, reference, queries);
                ok = false;
            }
            if (ordered) {
                const float * actual = (const float *) score->data + col*live;
                for (int row = 0; row < live; ++row) {
                    double expected = 0.0;
                    for (int i = 0; i < d; ++i) {
                        const char * kp = (const char *) k->data + row*k->nb[1] + i*k->nb[0];
                        const float kval = key_type == GGML_TYPE_F16
                            ? ggml_fp16_to_fp32(*(const ggml_fp16_t *) kp) : *(const float *) kp;
                        const float qval = *(const float *) ((const char *) q->data + col*q->nb[1] + i*q->nb[0]);
                        expected += (double) kval * (double) qval;
                    }
                    const float result = (float) expected;
                    if (!std::isfinite(actual[row]) || std::memcmp(&actual[row], &result, sizeof(result))) ok = false;
                }
                const float * ref = (const float *) score_ref->data + col*reference;
                for (int i = live; i < reference; ++i) {
                    if (ref[i] != 0.0f) ok = false;
                }
            }
        }
    }
    ggml_free(ctx);
    return ok;
}

static bool check_flash(ggml_type type, int width, int queries) {
    const int d = 32;
    ggml_init_params params = { 8*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, queries, 1, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, type, d, width, 1, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, type, d, width, 1, 1);
    ggml_tensor * kc = ggml_new_tensor_4d(ctx, type, d, width, 1, 1);
    ggml_tensor * vc = ggml_new_tensor_4d(ctx, type, d, width, 1, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, width, queries, 1, 1);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int j = 0; j < queries; ++j) {
        for (int i = 0; i < d; ++i) set_value(q, j, i, ((i + 3*j) % 9 - 4)/8.0f);
        for (int i = 0; i < width; ++i) set_value(mask, j, i, visible(i, j, width) ? 0.0f : -INFINITY);
    }
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < d; ++j) {
            const float x = ((i*5 + j*3) % 17 - 8)/16.0f;
            set_value(k, i, j, i == 2 ? nan : x);
            set_value(v, i, j, i == 2 ? nan : -x);
            set_value(kc, i, j, i == 2 ? 0.0f : x);
            set_value(vc, i, j, i == 2 ? 0.0f : -x);
        }
    }
    ggml_tensor * a = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 2.0f);
    ggml_tensor * b = ggml_flash_attn_ext(ctx, q, kc, vc, mask, 1.0f, 0.0f, 2.0f);
    ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);
    ggml_flash_attn_ext_set_prec(b, GGML_PREC_F32);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, a);
    ggml_build_forward_expand(graph, b);
    bool ok = ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS;
    if (ok) ok = std::memcmp(a->data, b->data, ggml_nbytes(a)) == 0;
    if (!ok) std::fprintf(stderr, "flash mask mismatch type=%d width=%d queries=%d\n", int(type), width, queries);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    const bool legacy_kq = argc == 2 && std::strcmp(argv[1], "--kq-dispatch") == 0;
    const bool ordered_kq = argc == 2 && std::strcmp(argv[1], "--ordered-kq") == 0;
    if (legacy_kq || ordered_kq) {
        bool ok = true;
        for (int live : { 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256, 257 }) {
            for (int queries : { 1, 4 }) {
                for (ggml_type key : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
                    for (ggml_type mask : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
                        if (legacy_kq && (key != GGML_TYPE_F16 || mask != GGML_TYPE_F32)) continue;
                        if (!check_kq_dispatch(live, std::max(256, live + 1), queries, ordered_kq, key, mask)) ok = false;
                    }
                }
            }
        }
        return ok ? 0 : 1;
    }
    const bool widths = argc == 2 && std::strcmp(argv[1], "--widths") == 0;
    if (argc != 1 && !widths) return 2;
    bool ok = true;
    if (!widths) {
        if (!check_unmasked_underflow(GGML_TYPE_F16)) ok = false;
        if (!check_unmasked_underflow(GGML_TYPE_F32)) ok = false;
        if (!check_half_conversion()) ok = false;
        for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
            if (!check_all_masked(type, false)) ok = false;
            if (!check_all_masked(type, true)) ok = false;
            if (!check_signed_zero(type)) ok = false;
            for (ggml_type mask : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
                if (!check_broadcast(type, mask)) ok = false;
            }
        }
    }
    if (!widths) {
        for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
            for (int width : { 7, 33, 257 }) {
                for (int queries : { 1, 32 }) {
                    if (!check_flash(type, width, queries)) ok = false;
                }
            }
        }
    }
    const int sizes[] = { 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512, 513 };
    for (int live : sizes) {
        const int reference = widths ? std::max(256, live + 1) : live;
        for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
            for (ggml_type mask : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
                for (int rows : { 15, 16 }) {
                    for (int queries : { 1, 4 }) {
                        for (bool pedantic : { false, true }) {
                            if (!check_case(type, mask, live, reference, rows, queries, widths, pedantic)) ok = false;
                        }
                    }
                }
            }
        }
    }
    return ok ? 0 : 1;
}
