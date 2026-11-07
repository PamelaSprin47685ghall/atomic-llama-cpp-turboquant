#pragma once

//
// llama-union-shape-guard.h
//
// The symbolic dimension algebra of the union-graph analysis and the SET_ROWS
// shape contract as one pure predicate.
//
// The three SET_ROWS checks used to be inline code in union_shape_analyze()
// (llama-context.cpp). A test cannot call a static function, so the checks
// live here: the product path and tests/test-union-shape-guard.cpp call the
// same function, and the test does not restate the rules.
//
// Contract, from ggml_set_rows (ggml/src/ggml.c:4051-4077) and the CPU kernel
// (ggml/src/ggml-cpu/ops.cpp:5128-5203):
//
//   - The destination is the physical cache storage: its ne/nb are the
//     allocation envelope, and a bind must not move them. A dependent row
//     count would make the kernel's i1 < ne1 check reject a legal cell; a
//     dependent plane extent would break ne2 == ne02 / ne3 == ne03.
//   - The kernel's i02/i03 bounds are src0->ne[2]/ne[3], while the
//     destination's nb[2]/nb[3] are fixed cache spans. A runtime coefficient
//     on the source's plane axes would therefore write one full cache span per
//     token plane. The token axis belongs on the row axis (ne[1]); refuse
//     anything else.
//   - ggml_set_rows asserts b->ne[1] == c->ne[0], c->ne[3] == 1,
//     b->ne[2] % c->ne[1] == 0 and b->ne[3] % c->ne[2] == 0. The kernel reads
//     nr = src0->ne[1] indices per plane and wraps the index planes modulo
//     src1->ne[1]/ne[2], so the index shape must follow the value shape at
//     every bound width. The divisors are the index tensor's plane extents: a
//     dependent divisor has no exact rule, so it fails closed.
//   - After the destination shape is copied, re-check the kernel's shape
//     invariants against the value tensor (ne0 == nc, ne2 == ne02,
//     ne3 == ne03), the innermost stride against the element size, and, for
//     block-size-1 types, the contiguous-row stride. A quantized row spans
//     nb[0] * ne[0] / blck_size, which the coefficient form cannot express.
//
// This header is deliberately ggml-free: the caller passes the element and
// block sizes as plain integers, so a test TU can include it without linking
// ggml, llama or any backend. Keep it that way.

#include <cstdint>

struct union_dim_expr {
    int64_t base = 0;
    int64_t d_nkv = 0;
    int64_t d_nout = 0;
    int64_t d_head = 0;
    int64_t d_nrs = 0;
    int64_t d_ntok = 0;
    int64_t d_nkv_ntok = 0;
    bool invalid = false;

    bool dep() const { return invalid || d_nkv || d_nout || d_head || d_nrs || d_ntok || d_nkv_ntok; }
};

struct union_shape_expr {
    union_dim_expr ne[4];
    union_dim_expr nb[4];
    union_dim_expr view_offs;
    union_dim_expr root_offs;

    bool dep() const {
        for (int i = 0; i < 4; ++i) {
            if (ne[i].dep() || nb[i].dep()) {
                return true;
            }
        }
        return view_offs.dep() || root_offs.dep();
    }
};

inline union_dim_expr union_dim_const(int64_t v) {
    union_dim_expr e;
    e.base = v;
    return e;
}

inline union_dim_expr union_dim_add(const union_dim_expr & a, const union_dim_expr & b) {
    union_dim_expr out;
    out.invalid = a.invalid || b.invalid;
    out.invalid |= __builtin_add_overflow(a.base, b.base, &out.base);
    out.invalid |= __builtin_add_overflow(a.d_nkv, b.d_nkv, &out.d_nkv);
    out.invalid |= __builtin_add_overflow(a.d_nout, b.d_nout, &out.d_nout);
    out.invalid |= __builtin_add_overflow(a.d_nrs, b.d_nrs, &out.d_nrs);
    out.invalid |= __builtin_add_overflow(a.d_head, b.d_head, &out.d_head);
    out.invalid |= __builtin_add_overflow(a.d_ntok, b.d_ntok, &out.d_ntok);
    out.invalid |= __builtin_add_overflow(a.d_nkv_ntok, b.d_nkv_ntok, &out.d_nkv_ntok);
    return out;
}

inline bool union_dim_equal(const union_dim_expr & a, const union_dim_expr & b) {
    return !a.invalid && !b.invalid && a.base == b.base && a.d_nkv == b.d_nkv && a.d_nout == b.d_nout &&
           a.d_nrs == b.d_nrs && a.d_head == b.d_head && a.d_ntok == b.d_ntok &&
           a.d_nkv_ntok == b.d_nkv_ntok;
}

// Only n_kv * n_tokens may appear as a nonlinear term.
inline bool union_dim_mul(const union_dim_expr & a, const union_dim_expr & b, union_dim_expr & out) {
    if (a.invalid || b.invalid) return false;
    if (a.dep() && b.dep() &&
        (a.d_nout || a.d_nrs || a.d_head || a.d_nkv_ntok ||
         b.d_nout || b.d_nrs || b.d_head || b.d_nkv_ntok ||
         (a.d_nkv && b.d_nkv) || (a.d_ntok && b.d_ntok))) return false;
    out = {};
    auto accumulate = [](int64_t & target, int64_t x, int64_t y) {
        int64_t product;
        return !__builtin_mul_overflow(x, y, &product) &&
               !__builtin_add_overflow(target, product, &target);
    };
    return accumulate(out.base, a.base, b.base) &&
        accumulate(out.d_nkv, a.d_nkv, b.base) && accumulate(out.d_nkv, a.base, b.d_nkv) &&
        accumulate(out.d_nout, a.d_nout, b.base) && accumulate(out.d_nout, a.base, b.d_nout) &&
        accumulate(out.d_nrs, a.d_nrs, b.base) && accumulate(out.d_nrs, a.base, b.d_nrs) &&
        accumulate(out.d_head, a.d_head, b.base) && accumulate(out.d_head, a.base, b.d_head) &&
        accumulate(out.d_ntok, a.d_ntok, b.base) && accumulate(out.d_ntok, a.base, b.d_ntok) &&
        accumulate(out.d_nkv_ntok, a.d_nkv_ntok, b.base) &&
        accumulate(out.d_nkv_ntok, a.base, b.d_nkv_ntok) &&
        accumulate(out.d_nkv_ntok, a.d_nkv, b.d_ntok) &&
        accumulate(out.d_nkv_ntok, a.d_ntok, b.d_nkv);
}

// The SET_ROWS shape contract. `dst` is ggml_set_rows' src[2] (the cache),
// `values` its src[0], `indices` its src[1]: the ggml source order is values,
// indices, dst. `elem_size`/`blck_size` are ggml_type_size() /
// ggml_blck_size() of the destination type. On success out.ne/out.nb receive
// the destination layout; on failure they may be partially written and the
// caller must not read them.
inline bool union_set_rows_shape_ok(
        const union_shape_expr & dst,
        const union_shape_expr & values,
        const union_shape_expr & indices,
        int64_t elem_size,
        int64_t blck_size,
        union_shape_expr & out) {
    // The destination is the physical cache storage: its ne/nb are the
    // allocation envelope, and a bind must not move them. A dependent row
    // count would make the kernel's i1 < ne1 check reject a legal cell; a
    // dependent plane extent would break ne2 == ne02 / ne3 == ne03.
    for (int i = 0; i < 4; ++i) {
        if (dst.ne[i].dep() || dst.nb[i].dep()) {
            return false;
        }
    }
    // The kernel's i02/i03 bounds are src0->ne[2]/ne[3], while the
    // destination's nb[2]/nb[3] are fixed cache spans. A runtime coefficient
    // on the source's plane axes would therefore write one full cache span
    // per token plane. The token axis belongs on the row axis (ne[1]); refuse
    // anything else.
    if (values.ne[2].dep() || values.ne[3].dep()) {
        return false;
    }
    // ggml_set_rows asserts b->ne[1] == c->ne[0], c->ne[3] == 1,
    // b->ne[2] % c->ne[1] == 0 and b->ne[3] % c->ne[2] == 0. The kernel reads
    // nr = src0->ne[1] indices per plane and wraps the index planes modulo
    // src1->ne[1]/ne[2], so the index shape must follow the value shape at
    // every bound width. The divisors are the index tensor's plane extents: a
    // dependent divisor has no exact rule, so it fails closed.
    if (!union_dim_equal(values.ne[1], indices.ne[0]) ||
            indices.ne[3].dep() || indices.ne[3].base != 1) {
        return false;
    }
    if (indices.ne[1].dep() || indices.ne[2].dep() ||
            indices.ne[1].base <= 0 || indices.ne[2].base <= 0 ||
            values.ne[2].base % indices.ne[1].base != 0 ||
            values.ne[3].base % indices.ne[2].base != 0) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        out.ne[i] = dst.ne[i];
        out.nb[i] = dst.nb[i];
    }
    // Re-check the kernel's shape invariants against the value tensor after
    // the copy: ne0 == nc, ne2 == ne02, ne3 == ne03.
    if (!union_dim_equal(out.ne[0], values.ne[0]) ||
            !union_dim_equal(out.ne[2], values.ne[2]) ||
            !union_dim_equal(out.ne[3], values.ne[3])) {
        return false;
    }
    // ggml_is_contiguous_rows(dst) and the kernel's row write: one row is
    // ne[0] elements at nb[1] stride. The nb[1] relation is checked for
    // block-size-1 types only: a quantized row spans nb[0] * ne[0] /
    // blck_size, which the coefficient form cannot express.
    if (!union_dim_equal(out.nb[0], union_dim_const(elem_size))) {
        return false;
    }
    if (blck_size == 1) {
        union_dim_expr row_nb;
        if (!union_dim_mul(out.nb[0], out.ne[0], row_nb) ||
                !union_dim_equal(out.nb[1], row_nb)) {
            return false;
        }
    }
    return true;
}
