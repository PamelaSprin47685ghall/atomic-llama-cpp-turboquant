// Compare active-width CPU attention reductions with the padded CPU contract.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

static_assert((GGML_TENSOR_FLAG_MASKED_TAIL_GEMM &
              (GGML_TENSOR_FLAG_ATTN_MASK | GGML_TENSOR_FLAG_ATTN_SOFT_MAX |
               GGML_TENSOR_FLAG_ATTN_SCORES)) == 0,
              "attention tensor flags must not overlap");

#if defined(__unix__)
#include <sys/mman.h>
#include <unistd.h>
#endif

extern "C" {
    double ggml_vec_soft_max_f32(int n, float * y, const float * x, float max);
    double ggml_vec_soft_max_f32_padded(int n, int padded, float * y, const float * x, float max);
    void ggml_vec_dot_f32(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by, int nrc);
    void ggml_vec_dot_f32_padded(int n, int padded, float * s, const float * x, const float * y);
    void ggml_vec_dot_f16(int n, float * s, size_t bs, ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);
    void ggml_vec_dot_f16_padded(int n, int padded, float * s, const ggml_fp16_t * x, const ggml_fp16_t * y);
}

struct guarded_row {
#if defined(__unix__)
    char * memory;
    size_t page;
    guarded_row() : page((size_t) sysconf(_SC_PAGESIZE)) {
        memory = (char *) mmap(nullptr, page * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (memory == MAP_FAILED || mprotect(memory + page, page, PROT_NONE) != 0) std::abort();
    }
    ~guarded_row() { munmap(memory, page * 2); }
    float * row(int n) {
        if ((size_t) n * sizeof(float) > page) std::abort();
        return (float *) (memory + page - n * sizeof(float));
    }
    ggml_fp16_t * row16(int n) {
        if ((size_t) n * sizeof(ggml_fp16_t) > page) std::abort();
        return (ggml_fp16_t *) (memory + page - n * sizeof(ggml_fp16_t));
    }
#else
    std::vector<float> memory = std::vector<float>(768);
    float * row(int n) { return memory.data() + memory.size() - n; }
    ggml_fp16_t * row16(int n) { return (ggml_fp16_t *) memory.data() + 2*memory.size() - n; }
#endif
};

static float sample(uint32_t & state) {
    state = state * 1664525u + 1013904223u;
    return (float) (int32_t(state >> 8) - 8388608) / 8388608.0f;
}

// Compare ordered CPU attention at the exact and reference widths. Both
// products carry the explicit mask; the stock V tail is deliberately nonzero.
static bool check_f16_kqv(int live, int reference, int rows, int columns, bool pedantic) {
    ggml_init_params params = { 8 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * v = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, live, rows);
    ggml_tensor * vp = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, reference, rows);
    ggml_tensor * scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, live, columns);
    ggml_tensor * scores_p = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, reference, columns);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, live, columns);
    ggml_tensor * mask_p = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, reference, columns);
    uint32_t state = 918273u + live * 17u + rows * 31u + columns;
    for (int row = 0; row < rows; ++row) {
        for (int i = 0; i < reference; ++i) {
            const ggml_fp16_t value = ggml_fp32_to_fp16(sample(state));
            ((ggml_fp16_t *) vp->data)[row * reference + i] = value;
            if (i < live) ((ggml_fp16_t *) v->data)[row * live + i] = value;
        }
    }
    for (int col = 0; col < columns; ++col) {
        for (int i = 0; i < reference; ++i) {
            const float value = sample(state);
            ((float *) scores_p->data)[col * reference + i] = value;
            ((float *) mask_p->data)[col * reference + i] = i < live ? 0.0f : -INFINITY;
            if (i < live) {
                ((float *) scores->data)[col * live + i] = value;
                ((float *) mask->data)[col * live + i] = 0.0f;
            }
        }
    }
    ggml_tensor * soft = ggml_soft_max_ext(ctx, scores, mask, 1.0f, 0.0f);
    ggml_tensor * soft_p = ggml_soft_max_ext(ctx, scores_p, mask_p, 1.0f, 0.0f);
    ggml_tensor * actual = ggml_mul_mat(ctx, v, soft);
    ggml_tensor * expected = ggml_mul_mat(ctx, vp, soft_p);
    ggml_mul_mat_set_attn_mask(actual, mask);
    ggml_mul_mat_set_attn_mask(expected, mask_p);
    if (pedantic) {
        ggml_mul_mat_set_prec(actual, GGML_PREC_F32_PEDANTIC);
        ggml_mul_mat_set_prec(expected, GGML_PREC_F32_PEDANTIC);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, actual);
    ggml_build_forward_expand(graph, expected);
    bool ok = ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS;
    if (ok) {
        for (int col = 0; col < columns; ++col) {
            const float * a = (const float *) soft->data + col * live;
            const float * b = (const float *) soft_p->data + col * reference;
            if (std::memcmp(a, b, live * sizeof(float)) != 0) {
                std::fprintf(stderr, "softmax mismatch live=%d ref=%d col=%d\n", live, reference, col);
                ok = false;
            }
        }
        for (int col = 0; col < columns; ++col) {
            const float * a = (const float *) actual->data + col * rows;
            const float * b = (const float *) expected->data + col * rows;
            if (std::memcmp(a, b, rows * sizeof(float)) != 0) {
                std::fprintf(stderr, "kqv-0 row mismatch live=%d ref=%d m=%d n=%d col=%d pedantic=%d\n",
                             live, reference, rows, columns, col, int(pedantic));
                ok = false;
            }
        }
    }
    ggml_free(ctx);
    return ok;
}

static int test_f16_kqv() {
    const int widths[] = { 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129, 249, 250, 255, 256, 257 };
    bool ok = true;
    for (int live : widths) {
        const int refs[] = { live, std::max(live, 250), (live + 255) / 256 * 256 };
        for (int reference : refs) {
            for (int rows : { 15, 16 }) {
                for (int columns : { 1, 4 }) {
                    for (bool pedantic : { false, true }) {
                        if (!check_f16_kqv(live, reference, rows, columns, pedantic)) ok = false;
                    }
                }
            }
        }
    }
    if (ok) std::puts("PASS: F16 ordered kqv rows match reference width bitwise");
    return ok ? 0 : 1;
}

// The TRUE_LEN_REDUCE contract has three arms over the same padded shapes:
//
//   p_pad           both flags, op_params[3] = live. The node the bind builds.
//                   It reduces over `live` and must leave a zeroed tail.
//   p_untrimmed     no flags. The padded reference: it reduces over `padded`
//                   and its tail is exp(score - max), strictly positive.
//   p_flagged_no_tail  TRUE_LEN_REDUCE only, no REDUCE_PAD_TAIL. Same live
//                   reduction, no padded SIMD grouping. Its first `live`
//                   outputs must equal p_pad's, which proves the trim comes
//                   from the flag and not from the padded helper.
//
// The falsifiable property is the tail: p_pad's tail is all zeros, and
// p_untrimmed's is not. Before the graph runs, p_pad's storage is filled with
// a sentinel, so a missing tail write leaves the sentinel in place and the
// zero assertion fails. The equality of the first `live` outputs against a
// graph shaped at `live` is the second half.
static bool check_true_len(int live, int padded, int rows, int columns) {
    ggml_init_params params = { 8 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    // The reference: everything shaped at `live`.
    ggml_tensor * v_live = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, live, rows);
    ggml_tensor * s_live = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, live, columns);
    ggml_tensor * m_live = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, live, columns);

    // The trimmed node: shaped at `padded`, flag + op_params[3] = live.
    ggml_tensor * v_pad = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, padded, rows);
    ggml_tensor * s_pad = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, padded, columns);
    ggml_tensor * m_pad = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, padded, columns);

    uint32_t state = 771234u + (uint32_t) live * 31u + (uint32_t) padded;
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < padded; ++i) {
            const float x = sample(state);
            ((float *) v_pad->data)[r * padded + i] = x;
            if (i < live) ((float *) v_live->data)[r * live + i] = x;
        }
    }
    for (int c = 0; c < columns; ++c) {
        for (int i = 0; i < padded; ++i) {
            const float x = sample(state);
            const float m = i < live ? 0.0f : -INFINITY;
            ((float *) s_pad->data)[c * padded + i] = x;
            ((float *) m_pad->data)[c * padded + i] = m;
            if (i < live) {
                ((float *) s_live->data)[c * live + i] = x;
                ((float *) m_live->data)[c * live + i] = m;
            }
        }
    }

    // The trimmed nodes carry exactly what the bind writes: both flags, and
    // op_params[2]/[4] = the padded extent, op_params[3] = live. A node with
    // only TRUE_LEN_REDUCE would not exercise the padded SIMD tail the binder
    // actually selects.
    ggml_tensor * p_live = ggml_soft_max_ext(ctx, s_live, m_live, 1.0f, 0.0f);
    ggml_tensor * p_pad  = ggml_soft_max_ext(ctx, s_pad,  m_pad,  1.0f, 0.0f);
    p_pad->flags |= GGML_TENSOR_FLAG_REDUCE_PAD_TAIL | GGML_TENSOR_FLAG_TRUE_LEN_REDUCE;
    p_pad->op_params[2] = padded;
    p_pad->op_params[3] = live;

    ggml_tensor * o_live = ggml_mul_mat(ctx, v_live, p_live);
    ggml_tensor * o_pad  = ggml_mul_mat(ctx, v_pad,  p_pad);
    o_pad->flags |= GGML_TENSOR_FLAG_REDUCE_PAD_TAIL | GGML_TENSOR_FLAG_TRUE_LEN_REDUCE;
    o_pad->op_params[2] = padded;
    o_pad->op_params[3] = live;
    o_pad->op_params[4] = padded;

    // The padded reference: the SAME padded shapes with no flags at all. It
    // reduces over `padded`, so its tail is exp(score - max), strictly
    // positive. That is what makes the trimmed node's zero tail meaningful:
    // the two arms differ exactly in the tail, and only the flag can explain
    // the difference.
    ggml_tensor * p_untrimmed = ggml_soft_max_ext(ctx, s_pad, m_pad, 1.0f, 0.0f);
    // The control that isolates the flag: the SAME padded shapes, the same
    // live reduction, but WITHOUT REDUCE_PAD_TAIL. If the kernel ignored
    // TRUE_LEN_REDUCE, this node would reduce over `padded` and its first
    // `live` outputs would differ from p_pad's.
    ggml_tensor * p_flagged_no_tail = ggml_soft_max_ext(ctx, s_pad, m_pad, 1.0f, 0.0f);
    p_flagged_no_tail->flags |= GGML_TENSOR_FLAG_TRUE_LEN_REDUCE;
    p_flagged_no_tail->op_params[3] = live;

    ggml_tensor * o_untrimmed = ggml_mul_mat(ctx, v_pad, p_untrimmed);

    // The sentinel. p_pad's storage is uninitialised (the context allocates
    // tensor data inline and does not zero it). 0x3f3f3f3f is a large positive
    // float and is never 0.0f, so if the kernel does not write the tail, the
    // sentinel survives and the all-zero assertion below fails.
    std::memset(p_pad->data, 0x3f, ggml_nbytes(p_pad));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, o_live);
    ggml_build_forward_expand(graph, o_pad);
    ggml_build_forward_expand(graph, o_untrimmed);
    ggml_build_forward_expand(graph, p_flagged_no_tail);
    bool ok = ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS;
    if (ok) {
        // The padded reference's tail must be non-zero, or the zero-tail
        // assertion below would hold for a degenerate reason.
        bool reference_tail_nonzero = false;
        for (int c = 0; c < columns && !reference_tail_nonzero; ++c) {
            const float * u = (const float *) p_untrimmed->data + c * padded;
            for (int i = live; i < padded; ++i) {
                if (u[i] != 0.0f) {
                    reference_tail_nonzero = true;
                    break;
                }
            }
        }
        if (!reference_tail_nonzero) {
            std::fprintf(stderr,
                    "true-len control: the padded reference has an all-zero tail, "
                    "so the tail assertion is vacuous (live=%d padded=%d)\n",
                    live, padded);
            ok = false;
        }
        // The trimmed node's tail must be exactly +0.0f, bit for bit. memcmp
        // against a zero vector, not `== 0.0f`, so -0.0f would fail.
        std::vector<float> zero_tail((size_t) (padded - live), 0.0f);
        for (int c = 0; c < columns; ++c) {
            const float * t = (const float *) p_pad->data + c * padded;
            if (std::memcmp(t + live, zero_tail.data(), zero_tail.size() * sizeof(float)) != 0) {
                std::fprintf(stderr,
                        "true-len tail not zeroed live=%d padded=%d col=%d\n",
                        live, padded, c);
                ok = false;
            }
        }
        // REDUCE_PAD_TAIL only changes the SIMD grouping of the reduction. The
        // two arms agree bit for bit only when the padded extent is a whole
        // number of lanes, so the comparison is made only there. 8 is the
        // smallest lane count on any target this test runs on.
        if (padded % 8 == 0) {
            for (int c = 0; c < columns; ++c) {
                const float * a = (const float *) p_flagged_no_tail->data + c * padded;
                const float * b = (const float *) p_pad->data            + c * padded;
                if (std::memcmp(a, b, live * sizeof(float)) != 0) {
                    std::fprintf(stderr,
                            "true-len flag-only arm mismatch live=%d padded=%d col=%d\n",
                            live, padded, c);
                    ok = false;
                }
            }
        }
        for (int c = 0; c < columns; ++c) {
            const float * a = (const float *) p_live->data + c * live;
            const float * b = (const float *) p_pad->data  + c * padded;
            if (std::memcmp(a, b, live * sizeof(float)) != 0) {
                std::fprintf(stderr, "true-len softmax mismatch live=%d padded=%d col=%d\n",
                             live, padded, c);
                ok = false;
            }
        }
        for (int c = 0; c < columns; ++c) {
            const float * a = (const float *) o_live->data + c * rows;
            const float * b = (const float *) o_pad->data  + c * rows;
            if (std::memcmp(a, b, rows * sizeof(float)) != 0) {
                std::fprintf(stderr, "true-len kqv mismatch live=%d padded=%d col=%d\n",
                             live, padded, c);
                ok = false;
            }
        }
    }
    ggml_free(ctx);
    return ok;
}

static int test_true_len() {
    const int widths[] = { 7, 8, 9, 127, 128, 129, 255, 256, 257 };
    bool ok = true;
    for (int live : widths) {
        for (int padded : { std::max(256, live), (live + 255) / 256 * 256, live + 3 }) {
            if (padded < live) continue;
            for (int rows : { 15, 16 }) {
                for (int columns : { 1, 4 }) {
                    if (!check_true_len(live, padded, rows, columns)) ok = false;
                }
            }
        }
    }
    if (ok) std::puts("PASS: TRUE_LEN_REDUCE matches a live-width graph bitwise");
    return ok ? 0 : 1;
}

// The measured-work counters must follow the runtime length, not the tensor
// extent. This arm runs through a real CPU backend, because that is the only
// path that carries a non-NULL cpu_ctx into the kernels
// (ggml_graph_compute_with_ctx leaves it NULL, so nothing is counted).
//
// Two graphs over the same tensors, one flagged and one not. The deltas are
// exact, not approximate: one softmax row per query column, `live` elements
// per row for the trimmed node and `ne00` for the unflagged one.
static bool check_counters(int live, int padded, int rows, int columns) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::fprintf(stderr, "counters: no CPU backend\n");
        return false;
    }
    // One thread makes the per-thread chunking deterministic, so the deltas
    // below are exact rather than "at most".
    ggml_backend_cpu_set_n_threads(backend, 1);

    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead()*64 + ggml_graph_overhead_custom(64, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        return false;
    }

    // Only softmax nodes. A mul_mat node would add its own rows to the
    // counter, and this arm is about the softmax identity alone.
    ggml_tensor * s_pad = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, padded, columns);
    ggml_tensor * m_pad = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, padded, columns);

    ggml_tensor * p_trimmed   = ggml_soft_max_ext(ctx, s_pad, m_pad, 1.0f, 0.0f);
    ggml_tensor * p_untrimmed = ggml_soft_max_ext(ctx, s_pad, m_pad, 1.0f, 0.0f);
    p_trimmed->flags |= GGML_TENSOR_FLAG_REDUCE_PAD_TAIL | GGML_TENSOR_FLAG_TRUE_LEN_REDUCE;
    p_trimmed->op_params[2] = padded;
    p_trimmed->op_params[3] = live;

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        std::fprintf(stderr, "counters: buffer allocation failed\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    std::vector<float> s_host((size_t) padded * (size_t) columns);
    std::vector<float> m_host((size_t) padded * (size_t) columns);
    uint32_t state = 551122u + (uint32_t) live * 17u + (uint32_t) padded;
    for (int c = 0; c < columns; ++c) {
        for (int i = 0; i < padded; ++i) {
            const size_t at = (size_t) c * (size_t) padded + (size_t) i;
            s_host[at] = sample(state);
            m_host[at] = i < live ? 0.0f : -INFINITY;
        }
    }
    ggml_backend_tensor_set(s_pad, s_host.data(), 0, s_host.size() * sizeof(float));
    ggml_backend_tensor_set(m_pad, m_host.data(), 0, m_host.size() * sizeof(float));

    uint64_t live0 = 0, pad0 = 0, rows0 = 0;
    ggml_backend_cpu_attn_reduce_stats(backend, &live0, &pad0, &rows0);

    bool ok = true;
    {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(graph, p_trimmed);
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "counters: trimmed compute failed\n");
            ok = false;
        }
    }

    uint64_t live1 = 0, pad1 = 0, rows1 = 0;
    ggml_backend_cpu_attn_reduce_stats(backend, &live1, &pad1, &rows1);

    if (ok) {
        const uint64_t d_live = live1 - live0;
        const uint64_t d_pad  = pad1  - pad0;
        const uint64_t d_rows = rows1 - rows0;
        if (d_rows != (uint64_t) columns) {
            std::fprintf(stderr,
                    "counters: trimmed rows %llu != columns %d (live=%d padded=%d)\n",
                    (unsigned long long) d_rows, columns, live, padded);
            ok = false;
        }
        if (d_live != (uint64_t) columns * (uint64_t) live) {
            std::fprintf(stderr,
                    "counters: trimmed live %llu != columns*live %llu (live=%d padded=%d)\n",
                    (unsigned long long) d_live,
                    (unsigned long long) ((uint64_t) columns * (uint64_t) live),
                    live, padded);
            ok = false;
        }
        if (d_pad != (uint64_t) columns * (uint64_t) padded) {
            std::fprintf(stderr,
                    "counters: trimmed pad %llu != columns*ne00 %llu (live=%d padded=%d)\n",
                    (unsigned long long) d_pad,
                    (unsigned long long) ((uint64_t) columns * (uint64_t) padded),
                    live, padded);
            ok = false;
        }
        if (live < padded && !(d_live < d_pad)) {
            std::fprintf(stderr,
                    "counters: trimmed live %llu is not below pad %llu (live=%d padded=%d)\n",
                    (unsigned long long) d_live, (unsigned long long) d_pad, live, padded);
            ok = false;
        }
    }

    {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(graph, p_untrimmed);
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "counters: unflagged compute failed\n");
            ok = false;
        }
    }

    uint64_t live2 = 0, pad2 = 0, rows2 = 0;
    ggml_backend_cpu_attn_reduce_stats(backend, &live2, &pad2, &rows2);

    if (ok) {
        const uint64_t d_live = live2 - live1;
        const uint64_t d_pad  = pad2  - pad1;
        const uint64_t d_rows = rows2 - rows1;
        // An unflagged softmax reduces over its own extent, so it is not a
        // trimmed row and must not be counted as one.
        if (d_rows != 0) {
            std::fprintf(stderr,
                    "counters: unflagged node moved rows by %llu (live=%d padded=%d)\n",
                    (unsigned long long) d_rows, live, padded);
            ok = false;
        }
        if (d_live != (uint64_t) columns * (uint64_t) padded) {
            std::fprintf(stderr,
                    "counters: unflagged live %llu != columns*ne00 %llu (live=%d padded=%d)\n",
                    (unsigned long long) d_live,
                    (unsigned long long) ((uint64_t) columns * (uint64_t) padded),
                    live, padded);
            ok = false;
        }
        if (d_pad != (uint64_t) columns * (uint64_t) padded) {
            std::fprintf(stderr,
                    "counters: unflagged pad %llu != columns*ne00 %llu (live=%d padded=%d)\n",
                    (unsigned long long) d_pad,
                    (unsigned long long) ((uint64_t) columns * (uint64_t) padded),
                    live, padded);
            ok = false;
        }
    }

    // With no_alloc the tensors live in the backend buffer, so the buffer
    // must go before the context.
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return ok;
}

static int test_counters() {
    const int widths[] = { 7, 8, 9, 127, 128, 129, 255, 256, 257 };
    bool ok = true;
    for (int live : widths) {
        for (int padded : { std::max(256, live), (live + 255) / 256 * 256, live + 3 }) {
            if (padded < live) continue;
            for (int rows : { 15, 16 }) {
                for (int columns : { 1, 4 }) {
                    if (!check_counters(live, padded, rows, columns)) ok = false;
                }
            }
        }
    }
    if (ok) std::puts("PASS: TRUE_LEN_REDUCE counters follow the runtime length");
    return ok ? 0 : 1;
}

// The masked-tail SGEMM is the prompt-processing path of the same reduction:
// ggml-cpu.c hands it (pad_len, live_len) in that order, and tinyBLAS loads
// every K block past `live_k` as zeros. Two properties follow, and both are
// checkable without knowing which kernel the dispatcher picked:
//
//   1. the result equals a dot over the live prefix alone, so the padded K
//      elements provably did not contribute;
//   2. the counters report live < pad, so the trim is visible in the measured
//      work and not only in the value.
//
// The K tail is filled with a large finite value, not with NaN: a NaN would
// make a wrong result obvious but would also make the comparison meaningless
// (any mismatch would be indistinguishable from the poisoning). A large finite
// tail makes a wrong result wrong by a measurable amount.
//
// The SGEMM only runs for n >= 2 queries, so every case here uses at least
// two columns.
static bool check_masked_tail_gemm(int live, int padded, int rows, int columns) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::fprintf(stderr, "gemm: no CPU backend\n");
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, 1);

    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead()*64 + ggml_graph_overhead_custom(64, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        return false;
    }

    // v [padded, rows] and p [padded, columns]: the mul_mat reduces over the
    // padded K span and must only read the live prefix of it.
    ggml_tensor * v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, padded, rows);
    ggml_tensor * p = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, padded, columns);
    ggml_tensor * o = ggml_mul_mat(ctx, v, p);
    o->flags |= GGML_TENSOR_FLAG_REDUCE_PAD_TAIL | GGML_TENSOR_FLAG_TRUE_LEN_REDUCE |
                GGML_TENSOR_FLAG_MASKED_TAIL_GEMM;
    o->op_params[2] = padded;
    o->op_params[3] = live;
    o->op_params[4] = padded;

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        std::fprintf(stderr, "gemm: buffer allocation failed\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    std::vector<float> v_host((size_t) padded * (size_t) rows);
    std::vector<float> p_host((size_t) padded * (size_t) columns);
    uint32_t state = 90210u + (uint32_t) live * 7u + (uint32_t) padded;
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < padded; ++i) {
            v_host[(size_t) r * (size_t) padded + (size_t) i] = i < live ? sample(state) : 1.0e6f;
        }
    }
    for (int c = 0; c < columns; ++c) {
        for (int i = 0; i < padded; ++i) {
            p_host[(size_t) c * (size_t) padded + (size_t) i] = i < live ? sample(state) : 1.0e6f;
        }
    }
    ggml_backend_tensor_set(v, v_host.data(), 0, v_host.size() * sizeof(float));
    ggml_backend_tensor_set(p, p_host.data(), 0, p_host.size() * sizeof(float));
    // The sentinel: a tail the kernel does not write keeps this value, so a
    // missing write cannot pass as a correct result.
    std::vector<float> sentinel((size_t) rows * (size_t) columns, 1.0e30f);
    ggml_backend_tensor_set(o, sentinel.data(), 0, sentinel.size() * sizeof(float));

    uint64_t live0 = 0, pad0 = 0, rows0 = 0;
    ggml_backend_cpu_attn_reduce_stats(backend, &live0, &pad0, &rows0);

    bool ok = true;
    {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(graph, o);
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "gemm: compute failed live=%d padded=%d\n", live, padded);
            ok = false;
        }
    }

    uint64_t live1 = 0, pad1 = 0, rows1 = 0;
    ggml_backend_cpu_attn_reduce_stats(backend, &live1, &pad1, &rows1);

    if (ok) {
        std::vector<float> got((size_t) rows * (size_t) columns);
        ggml_backend_tensor_get(o, got.data(), 0, got.size() * sizeof(float));

        // the reference: a dot over the live prefix, in double, so the
        // comparison is about the trim and not about the summation order
        for (int r = 0; r < rows && ok; ++r) {
            for (int c = 0; c < columns && ok; ++c) {
                double ref = 0.0;
                for (int i = 0; i < live; ++i) {
                    ref += (double) v_host[(size_t) r * (size_t) padded + (size_t) i] *
                           (double) p_host[(size_t) c * (size_t) padded + (size_t) i];
                }
                const double diff = std::fabs((double) got[(size_t) r * (size_t) columns + (size_t) c] - ref);
                const double scale = std::fabs(ref) > 1.0 ? std::fabs(ref) : 1.0;
                if (!(diff <= 1.0e-4 * scale)) {
                    std::fprintf(stderr,
                            "gemm: masked tail contributed live=%d padded=%d row=%d col=%d "
                            "(got %g ref %g)\n",
                            live, padded, r, c,
                            (double) got[(size_t) r * (size_t) columns + (size_t) c], ref);
                    ok = false;
                }
            }
        }

        // The sentinel must be gone from EVERY output element. A kernel that
        // wrote only part of the result would otherwise pass the value
        // comparison on the rows it did write.
        for (size_t i = 0; i < got.size(); ++i) {
            if (got[i] == 1.0e30f) {
                std::fprintf(stderr,
                        "gemm: output element %zu still holds the sentinel (live=%d padded=%d)\n",
                        i, live, padded);
                ok = false;
                break;
            }
        }

        // The counters must show the trim. This is unconditional: every cell
        // this arm walks has padded > live, and check_masked_tail_gemm
        // refuses a cell where it does not, so a cell that could not trim
        // cannot silently assert less than the arm claims.
        const uint64_t d_live = live1 - live0;
        const uint64_t d_pad  = pad1  - pad0;
        if (padded <= live) {
            std::fprintf(stderr,
                    "gemm: cell does not trim (live=%d padded=%d); the counter assertion "
                    "would be vacuous\n", live, padded);
            ok = false;
        } else if (!(d_live < d_pad)) {
            std::fprintf(stderr,
                    "gemm: counters did not trim live %llu pad %llu (live=%d padded=%d)\n",
                    (unsigned long long) d_live, (unsigned long long) d_pad, live, padded);
            ok = false;
        }
        if (d_live != (uint64_t) rows * (uint64_t) columns * (uint64_t) live) {
            std::fprintf(stderr,
                    "gemm: live %llu != rows*columns*live %llu (live=%d padded=%d)\n",
                    (unsigned long long) d_live,
                    (unsigned long long) ((uint64_t) rows * (uint64_t) columns * (uint64_t) live),
                    live, padded);
            ok = false;
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return ok;
}

static int test_masked_tail_gemm() {
    // Widths on both sides of a 256-cell boundary. Every padded span is
    // STRICTLY greater than its live length, so every cell exercises the
    // trim; a cell with padded == live is refused below rather than skipped,
    // because skipping it would leave the counter assertion unexercised while
    // the arm still claimed to have checked the trim.
    //
    // The three spans cover: one past the live length (the smallest trim), a
    // non-lane multiple, and the next 256-cell boundary above it.
    const int widths[] = { 7, 8, 9, 127, 128, 129, 255, 256, 257 };
    bool ok = true;
    int cells = 0;
    for (int live : widths) {
        for (int padded : { live + 1, live + 3, (live + 255) / 256 * 256 + 256 }) {
            if (padded <= live) {
                std::fprintf(stderr,
                        "gemm: cell does not trim (live=%d padded=%d); the arm would "
                        "assert less than it claims\n", live, padded);
                ok = false;
                continue;
            }
            for (int rows : { 15, 16 }) {
                for (int columns : { 2, 4 }) {
                    ++cells;
                    if (!check_masked_tail_gemm(live, padded, rows, columns)) ok = false;
                }
            }
        }
    }
    if (cells == 0) {
        std::fprintf(stderr, "gemm: no cell ran\n");
        ok = false;
    }
    if (ok) std::puts("PASS: masked-tail SGEMM reduces over the live length only");
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--f16-kqv") == 0) return test_f16_kqv();
    if (argc == 2 && std::strcmp(argv[1], "--counters") == 0) return test_counters();
    if (argc == 2 && std::strcmp(argv[1], "--true-len") == 0) return test_true_len();
    if (argc == 2 && std::strcmp(argv[1], "--masked-gemm") == 0) return test_masked_tail_gemm();
    if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--f16-kqv|--true-len|--counters|--masked-gemm]\n", argv[0]);
        return 2;
    }
    guarded_row gx, gy, gout, ghx, ghy;
    uint32_t state = 918273;
    for (int n = 1; n <= 608; ++n) {
        float * x = gx.row(n);
        float * y = gy.row(n);
        float * out = gout.row(n);
        for (int i = 0; i < n; ++i) { x[i] = sample(state); y[i] = sample(state); }
        ggml_fp16_t * hx = ghx.row16(n);
        ggml_fp16_t * hy = ghy.row16(n);
        for (int i = 0; i < n; ++i) {
            state = state * 1664525u + 1013904223u;
            hx[i] = (ggml_fp16_t) (0x3800u | (state & 0x83ffu));
            state = state * 1664525u + 1013904223u;
            hy[i] = (ggml_fp16_t) (0x3800u | (state & 0x83ffu));
        }
        const float max = *std::max_element(x, x + n);
        const int refs[] = {n, (n + 255) / 256 * 256, std::max(n, 250)};
        for (int padded : refs) {
            std::vector<float> xp(padded, -INFINITY), ref(padded), xd(padded, 0.0f), yd(padded, 0.0f);
            std::copy(x, x + n, xp.begin());
            const double expected_sum = ggml_vec_soft_max_f32(padded, ref.data(), xp.data(), max);
            const double actual_sum = ggml_vec_soft_max_f32_padded(n, padded, out, x, max);
            if (std::memcmp(&expected_sum, &actual_sum, sizeof(double)) || std::memcmp(ref.data(), out, n * sizeof(float))) {
                std::fprintf(stderr, "softmax mismatch n=%d padded=%d\n", n, padded);
                return 1;
            }
            std::copy(x, x + n, xd.begin());
            std::copy(y, y + n, yd.begin());
            float expected, actual;
            ggml_vec_dot_f32(padded, &expected, 0, xd.data(), 0, yd.data(), 0, 1);
            ggml_vec_dot_f32_padded(n, padded, &actual, x, y);
            if (std::memcmp(&expected, &actual, sizeof(float))) {
                std::fprintf(stderr, "dot mismatch n=%d padded=%d\n", n, padded);
                return 1;
            }
            std::vector<ggml_fp16_t> hxp(padded, 0), hyp(padded, 0);
            std::copy(hx, hx + n, hxp.begin());
            std::copy(hy, hy + n, hyp.begin());
            ggml_vec_dot_f16(padded, &expected, 0, hxp.data(), 0, hyp.data(), 0, 1);
            ggml_vec_dot_f16_padded(n, padded, &actual, hx, hy);
            if (std::memcmp(&expected, &actual, sizeof(float))) {
                std::fprintf(stderr, "f16 dot mismatch n=%d padded=%d\n", n, padded);
                return 1;
            }
        }
    }

    // The padded softmax helper reproduces the PADDED block grouping: it
    // rounds the tail up to `padded` lanes, which is a different addition
    // tree from the unpadded helper's. Below one lane of padding the two
    // agree exactly, because the helper falls back to the scalar tail
    // (vec.cpp: ref_np > n && np < n is false). Pin that boundary: it is
    // the width at which the padded grouping starts to matter, and nothing
    // else in the tree asserts it.
    //
    // 4 is the smallest lane count on any supported target (SSE2, NEON).
    // Above it, only the invariants are asserted: the live lanes are finite
    // and the returned sum is positive. Equality against the unpadded
    // helper is NOT asserted there, because the padded grouping legitimately
    // sums in a different order.
    for (int n = 1; n <= 64; ++n) {
        float * x = gx.row(n);
        float * out = gout.row(n);
        for (int i = 0; i < n; ++i) x[i] = sample(state);
        const float max = *std::max_element(x, x + n);
        std::vector<float> out_ref(n);
        const double ref_sum = ggml_vec_soft_max_f32(n, out_ref.data(), x, max);
        for (int padded : { n, n + 1, n + 7, n + 8, n + 15, n + 16, n + 63, (n + 255) / 256 * 256 }) {
            if (padded < n) continue;
            if (padded > 768) continue; // the guarded row is one page
            const double sum = ggml_vec_soft_max_f32_padded(n, padded, out, x, max);
            if (padded < n + 4) {
                if (std::memcmp(&ref_sum, &sum, sizeof(double)) ||
                    std::memcmp(out_ref.data(), out, n * sizeof(float))) {
                    std::fprintf(stderr,
                            "padded softmax drift n=%d padded=%d (below one lane)\n", n, padded);
                    return 1;
                }
            } else {
                if (!(sum > 0.0)) {
                    std::fprintf(stderr, "padded softmax sum not positive n=%d padded=%d\n", n, padded);
                    return 1;
                }
                for (int i = 0; i < n; ++i) {
                    if (!std::isfinite(out[i])) {
                        std::fprintf(stderr,
                                "padded softmax live lane not finite n=%d padded=%d i=%d\n",
                                n, padded, i);
                        return 1;
                    }
                }
            }
        }
    }
    std::puts("PASS: padded softmax helper keeps its live lanes for every padded width");
    std::puts("PASS: 608 widths, three reference widths each, guarded active rows");
    return 0;
}
