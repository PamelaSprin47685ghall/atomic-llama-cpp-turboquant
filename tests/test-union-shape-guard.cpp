#include "llama-union-shape-guard.h"

#include <cstdio>

//
// test-union-shape-guard
//
// The SET_ROWS symbolic shape contract (src/llama-union-shape-guard.h) as one
// pure predicate. No model, no context, no backend, no library.
//
// The footer is UNION_SHAPE_GUARD ok=1 cases=3. `cases` counts the three
// REFUSAL cases; the accepted control is folded into `ok`, because a guard
// that refuses everything would otherwise pass. Each refusal case carries
// exactly one fault, so a red case names the rule that stopped working:
//
//   g1_dst_nkv      dst.ne[1] carries d_nkv      -> G1: destination frozen
//   g2_index_width  values.ne[1]=5, indices.ne[0]=4 -> G2: nr vs index count
//   g3_value_plane  values.ne[2]=2, dst.ne[2]=1  -> G3: kernel ne2 re-check
//
// The shapes are a plain dense F16 KV-cache write (llama-kv-cache.cpp:2314,
// :2332): n_embd_gqa=512, n_kv=2048, n_tokens=3, indices 1-D I64.
//

namespace {

int  g_fail = 0;
int  g_refusals = 0;
bool g_control_ran = false;

union_shape_expr shape(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                       int64_t nb0, int64_t nb1, int64_t nb2, int64_t nb3) {
    union_shape_expr s;
    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    const int64_t nb[4] = { nb0, nb1, nb2, nb3 };
    for (int i = 0; i < 4; ++i) {
        s.ne[i] = union_dim_const(ne[i]);
        s.nb[i] = union_dim_const(nb[i]);
    }
    return s;
}

union_shape_expr control_dst() {
    return shape(512, 2048, 1, 1, 2, 1024, 2097152, 2097152);
}

union_shape_expr control_values() {
    return shape(512, 3, 1, 1, 2, 1024, 3072, 3072);
}

union_shape_expr control_indices() {
    return shape(3, 1, 1, 1, 8, 8, 8, 8);
}

bool run_case(const char * name, const union_shape_expr & dst,
              const union_shape_expr & values, const union_shape_expr & indices,
              int64_t elem_size, int64_t blck_size, bool expected) {
    ++g_refusals;
    union_shape_expr out;
    const bool got = union_set_rows_shape_ok(dst, values, indices, elem_size, blck_size, out);
    if (got != expected) {
        std::fprintf(stderr, "FAIL: %s expected %s got %s\n", name,
                     expected ? "true" : "false", got ? "true" : "false");
        ++g_fail;
    }
    return got;
}

} // namespace

int main() {
    // The accepted control: a dense KV-cache write. It must pass G1, G2 and
    // G3, and it must copy the destination layout into `out`.
    {
        g_control_ran = true;
        const union_shape_expr dst = control_dst();
        union_shape_expr out;
        if (!union_set_rows_shape_ok(dst, control_values(), control_indices(), 2, 1, out)) {
            std::fprintf(stderr, "FAIL: control expected true got false\n");
            ++g_fail;
        } else {
            for (int i = 0; i < 4; ++i) {
                if (!union_dim_equal(out.ne[i], dst.ne[i]) ||
                        !union_dim_equal(out.nb[i], dst.nb[i])) {
                    std::fprintf(stderr, "FAIL: control out.ne[%d]/nb[%d] differs from dst\n", i, i);
                    ++g_fail;
                    break;
                }
            }
        }
    }

    // g1_dst_nkv: the destination row count carries d_nkv. Every value is a
    // legal constant at base 8, so G2 and G3 would pass; only G1 refuses.
    {
        union_shape_expr dst = shape(512, 8, 1, 1, 2, 1024, 8192, 8192);
        dst.ne[1].d_nkv = 1;
        run_case("g1_dst_nkv", dst,
                 shape(512, 8, 1, 1, 2, 1024, 8192, 8192),
                 shape(8, 1, 1, 1, 8, 8, 8, 8), 2, 1, false);
    }

    // g2_index_width: the value tensor claims 5 rows per plane while the
    // index tensor has 4 entries. G1 passes (destination constant); G2's
    // values.ne[1] == indices.ne[0] fails.
    {
        run_case("g2_index_width", control_dst(),
                 shape(512, 5, 1, 1, 2, 1024, 5120, 5120),
                 shape(4, 1, 1, 1, 8, 8, 8, 8), 2, 1, false);
    }

    // g3_value_plane: the destination has one plane, the value tensor two.
    // G2 passes (2 % 1 == 0); G3's ne[2] re-check refuses.
    {
        run_case("g3_value_plane", control_dst(),
                 shape(512, 3, 2, 1, 2, 1024, 3072, 6144),
                 shape(3, 1, 1, 1, 8, 8, 8, 8), 2, 1, false);
    }

    if (!g_control_ran || g_refusals != 3) {
        std::fprintf(stderr, "UNION_SHAPE_GUARD ok=2 reason=unproven:case_count control=%d cases=%d\n",
                     g_control_ran ? 1 : 0, g_refusals);
        return 2;
    }
    std::fprintf(stderr, "UNION_SHAPE_GUARD ok=%d cases=%d\n", g_fail == 0 ? 1 : 0, g_refusals);
    return g_fail == 0 ? 0 : 1;
}
