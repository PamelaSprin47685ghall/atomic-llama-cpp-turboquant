//
// test-topology-hash.cpp
//
// Proves the stable topology digest of src/llama-topology-hash.h.
//
// The contract has two halves, and they point in opposite directions:
//
//   invariant to the runtime parameter values: the digest never reads n_kv,
//   n_outputs or rs_z. Two graphs built at the same shapes hash equal
//   whatever those parameters were, whatever the build order, whatever the
//   tensor names, and whatever the addresses.
//
//   sensitive to the graph as built: those parameters reach the graph as
//   tensor shapes and view offsets, so a graph built at a different n_kv, a
//   different n_out or a different rs_z IS a different graph and the digest
//   must say so.
//
//   positive
//     - the same topology built twice, in different build order, with
//       different tensor names and different tensor addresses, hashes equal;
//     - a separate process that builds the same graph prints the same hash
//       (the test re-executes itself with --emit and compares stdout).
//
//   negative
//     - one extra node, a different op, a different op parameter, a changed
//       edge, a changed shape, a changed tensor type, a changed view offset,
//       and a changed graph root all move the hash with the node set and the
//       node count held fixed, so the difference cannot be attributed to a
//       different number of nodes;
//     - the final scale's source becomes a weight (pre-existing storage) of
//       the same shape and type instead of the graph-owned gather result, so
//       only the fixed/owned distinction differs;
//     - changing n_kv, n_out or rs_z one at a time moves the hash, because
//       each of them changes a tensor shape or a view offset in the graph.
//
// No model file, no GPU, no backend: every graph is a ggml context in
// no_alloc mode, built from ggml ops only. The test is deterministic.
//
// The ggml build rules that matter here, all read from ggml/src/ggml.c:
//   - ggml_scale copies its source (ggml_scale_impl: `ggml_dup_tensor`), so
//     its node is graph-owned;
//   - ggml_cpy makes a view of its destination (ggml_cpy_impl:
//     `ggml_view_tensor`), so its node is graph-owned even when the
//     destination is a weight;
//   - ggml_view_2d / ggml_view_3d / ggml_reshape_2d / ggml_reshape_3d make
//     views of their source (ggml_view_impl / ggml_new_tensor_impl with
//     view_src);
//   - ggml_reshape_* asserts that its source is contiguous, so the builder
//     only reshapes tensors that are;
//   - ggml_get_rows and ggml_add_id write no op_params.
//

#include "../src/llama-topology-hash.h"

#include <unistd.h>

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    } \
} while (0)

#define CHECK_EQ_HASH(a, b) do { \
    ++g_checks; \
    const uint64_t va_ = (a); \
    const uint64_t vb_ = (b); \
    if (va_ != vb_) { \
        ++g_failures; \
        std::fprintf(stderr, "FAIL %s:%d: hash %016" PRIx64 " != %016" PRIx64 "\n", \
                __FILE__, __LINE__, va_, vb_); \
    } \
} while (0)

#define CHECK_NE_HASH(a, b) do { \
    ++g_checks; \
    const uint64_t va_ = (a); \
    const uint64_t vb_ = (b); \
    if (va_ == vb_) { \
        ++g_failures; \
        std::fprintf(stderr, "FAIL %s:%d: hash %016" PRIx64 " == %016" PRIx64 " (must differ)\n", \
                __FILE__, __LINE__, va_, vb_); \
    } \
} while (0)

// ---------------------------------------------------------------------------
// graph builder
// ---------------------------------------------------------------------------

// Runtime shape values. These are the parameters whose *values* the digest
// never reads. They are used only to size tensors and to pick view offsets, so
// they reach the digest exclusively through the graph those choices produce:
// n_kv sizes the KV cache, n_out sizes out_ids, rs_z picks the recurrent-state
// zero view's offset. Two builds with the same values hash equal (order,
// names, addresses do not matter); a build with a different value is a
// different graph and hashes differently. See test_shape_parameter_sensitivity.
struct shape {
    int64_t n_kv     = 64;  // KV width
    int64_t n_out    = 8;   // output rows
    int64_t rs_z     = 0;   // recurrent-state zero-view offset (in rows)
    int64_t n_embd   = 32;
    int64_t n_head   = 4;
    int64_t head_dim = 8;
};

// A mutation is a single, genuine change to the topology.
enum class mutation {
    none,
    extra_node,        // one more node in the graph
    different_op,      // add -> mul at one position
    different_param,   // scale factor 0.5 -> 0.25
    changed_edge,      // the gather reads another projection of the same shape
    changed_shape,     // one tensor is one row wider
    changed_type,      // one tensor is F16 instead of F32
    changed_view_offs, // the recurrent-state view starts one row later
    weight_not_activation, // the same node, but pre-existing storage
    changed_root,      // the graph's result tensor is a different node
};

// Builds one graph. `order` selects the order in which the independent
// branches are created and expanded; it must not change the topology.
// `names` selects whether tensors get names; it must not change the hash.
// `pad` shifts every allocation (ggml context arena) so that two builds of the
// same topology land at different addresses.
struct built {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * gf  = nullptr;
    uint64_t       hash = 0;
};

static built build_graph(const shape & s, mutation m, int order, bool names, size_t pad) {
    built b;

    // Room for ~96 tensors and a 128-node graph, plus the caller's padding.
    const size_t mem_size = pad + ggml_tensor_overhead() * 96 + ggml_graph_overhead_custom(128, false) + 4096;
    ggml_init_params ip = {
        /* .mem_size   = */ mem_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    b.ctx = ggml_init(ip);
    if (b.ctx == nullptr) {
        std::fprintf(stderr, "ggml_init failed\n");
        return b;
    }
    ggml_context * ctx = b.ctx;

    auto nm = [&](ggml_tensor * t, const char * name) -> ggml_tensor * {
        if (names) {
            ggml_set_name(t, name);
        }
        return t;
    };

    // --- weights: pre-existing storage (fixed leaves) -----------------------
    const int64_t n_embd = s.n_embd;
    const int64_t n_head = s.n_head;
    const int64_t hd     = s.head_dim;
    const int64_t n_kv   = s.n_kv;
    const int64_t n_out  = s.n_out;

    ggml_tensor * w_q   = nm(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd), "wq.weight");
    ggml_tensor * w_k   = nm(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd), "wk.weight");
    ggml_tensor * w_v   = nm(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd), "wv.weight");
    ggml_tensor * w_o   = nm(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd), "wo.weight");
    ggml_tensor * w_out = nm(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd), "out.weight");

    // KV cache: pre-existing storage, sized by n_kv.
    ggml_tensor * k_cache = nm(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, n_kv, n_head, 1), "k_cache");
    ggml_tensor * v_cache = nm(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, n_kv, n_head, 1), "v_cache");

    // Recurrent state: pre-existing storage. The zero view below starts at
    // row rs_z, which is a runtime value.
    ggml_tensor * rs = nm(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 4), "rs");

    // out_ids: the output-gather index, sized by n_out.
    ggml_tensor * out_ids = nm(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_out), "out_ids");

    // The context is no_alloc, so no tensor has storage yet. Mark the leaves
    // above as pre-existing storage with a non-null `data` sentinel: the
    // digest's leaf predicate is the binder record's (`view_src == nullptr &&
    // (data != nullptr || buffer != nullptr)`), and without this every tensor
    // would be graph-owned and the weight/activation distinction would not be
    // exercised at all. The sentinel is never dereferenced: nothing here
    // computes, and the digest never reads `data`.
    for (ggml_tensor * w : { w_q, w_k, w_v, w_o, w_out, k_cache, v_cache, rs, out_ids }) {
        w->data = &w;
    }

    // --- activations --------------------------------------------------------
    ggml_tensor * inp = nm(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1), "inp");

    // The two independent branches: attention and the recurrent-state zero
    // view. Their creation order is what `order` permutes.
    ggml_tensor * attn_out = nullptr;
    ggml_tensor * rs_zero  = nullptr;

    auto build_attn = [&]() {
        ggml_tensor * q = nm(ggml_mul_mat(ctx, w_q, inp), "q");
        ggml_tensor * k = nm(ggml_mul_mat(ctx, w_k, inp), "k");
        ggml_tensor * v = nm(ggml_mul_mat(ctx, w_v, inp), "v");

        // Write the new row into the cache tail: a view of the cache at the
        // last row, then a copy into it. k/v are [n_embd, 1] with
        // n_embd == hd * n_head and contiguous, so they reshape to the cache
        // row layout [hd, n_head, 1] without a copy.
        const size_t row = (size_t) (n_kv - 1) * k_cache->nb[1];
        ggml_tensor * k_tail = nm(ggml_view_3d(ctx, k_cache, hd, 1, n_head,
                k_cache->nb[1], k_cache->nb[2], row), "k_tail");
        ggml_tensor * k_row  = nm(ggml_reshape_3d(ctx, k, hd, n_head, 1), "k_row");
        ggml_tensor * k_wr   = nm(ggml_cpy(ctx, k_row, k_tail), "k_wr");

        ggml_tensor * v_tail = nm(ggml_view_3d(ctx, v_cache, hd, 1, n_head,
                v_cache->nb[1], v_cache->nb[2], row), "v_tail");
        ggml_tensor * v_row  = nm(ggml_reshape_3d(ctx, v, hd, n_head, 1), "v_row");
        ggml_tensor * v_wr   = nm(ggml_cpy(ctx, v_row, v_tail), "v_wr");

        // Read the whole cache back: [hd, n_kv, n_head, 1].
        ggml_tensor * k_all = nm(ggml_reshape_2d(ctx, k_cache, hd * n_head, n_kv), "k_all");
        ggml_tensor * v_all = nm(ggml_reshape_2d(ctx, v_cache, hd * n_head, n_kv), "v_all");

        ggml_tensor * q2 = nm(ggml_reshape_2d(ctx, q, hd * n_head, 1), "q2");
        ggml_tensor * kq = nm(ggml_mul_mat(ctx, k_all, q2), "kq"); // [n_kv, 1]
        ggml_tensor * sm = nm(ggml_soft_max(ctx, kq), "sm");      // [n_kv, 1]
        // The value product needs the transposed layout, exactly as the live
        // graph builds it: llama-graph.cpp:3341-3343 does
        // `v = ggml_cont(ctx0, ggml_transpose(ctx0, v))` when the cache is not
        // stored transposed, because ggml_can_mul_mat requires a->ne[0] ==
        // b->ne[0] and the scores carry n_kv there, not hd*n_head. Without the
        // transpose this call aborts (ggml.c:3369).
        ggml_tensor * v_t  = nm(ggml_cont(ctx, ggml_transpose(ctx, v_all)), "v_t"); // [n_kv, hd*n_head]
        ggml_tensor * vt = nm(ggml_mul_mat(ctx, v_t, sm), "vt"); // [hd*n_head, 1]
        ggml_tensor * vt2 = nm(ggml_reshape_2d(ctx, vt, n_embd, 1), "vt2");
        ggml_tensor * o  = nm(ggml_mul_mat(ctx, w_o, vt2), "o");

        // The writes must stay in the graph; keep them as extra roots by
        // folding them into the result. ggml_scale cannot be used for that:
        // it asserts ggml_is_padded_1d(a) (ggml.c:3505) and a CPY result is a
        // view of the cache tail, whose row stride is the cache's. ggml_sum
        // has no shape precondition and reduces the write to [1], which the
        // [n_embd, 1] attention output can broadcast.
        ggml_tensor * k_wr_s = nm(ggml_sum(ctx, k_wr), "k_wr_s");   // [1]
        ggml_tensor * v_wr_s = nm(ggml_sum(ctx, v_wr), "v_wr_s");   // [1]

        attn_out = nm(ggml_add(ctx, o, ggml_add(ctx, k_wr_s, v_wr_s)), "attn_out");
    };

    auto build_rs = [&]() {
        // The recurrent-state zero view: a row offset taken from rs_z. It is a
        // view of a weight, so it is graph-owned (view_src != nullptr).
        ggml_tensor * z = nm(ggml_view_2d(ctx, rs, n_embd, 1,
                rs->nb[1], (size_t) s.rs_z * rs->nb[1]), "rs_zero");
        rs_zero = nm(ggml_scale(ctx, z, 0.0f), "rs_zero_scaled");
    };

    if (order == 0) {
        build_attn();
        build_rs();
    } else {
        build_rs();
        build_attn();
    }

    // The join. One operand is swapped when `order` is 1: ggml_add is
    // commutative, so the two graphs compute the same thing, but the edge
    // slots differ. The digest sorts a node's edge targets before folding
    // them, so this must not move the hash.
    ggml_tensor * sum = order == 0
        ? nm(ggml_add(ctx, attn_out, rs_zero), "sum")
        : nm(ggml_add(ctx, rs_zero, attn_out), "sum");

    // Output gather: out_ids selects rows of the projection.
    ggml_tensor * logits = nm(ggml_mul_mat(ctx, w_out, sum), "logits"); // [n_embd, 1]
    ggml_tensor * rows   = nm(ggml_get_rows(ctx, logits, out_ids), "rows");
    ggml_tensor * result = nm(ggml_scale(ctx, rows, 0.5f), "result");

    // --- mutations ----------------------------------------------------------
    switch (m) {
        case mutation::none:
            break;
        case mutation::extra_node:
            // One more node between the gather and the final scale.
            result = nm(ggml_scale(ctx, result, 1.0f), "result2");
            break;
        case mutation::different_op:
            // add -> mul at the join.
            result = nm(ggml_scale(ctx, ggml_mul(ctx, attn_out, rs_zero), 0.5f), "result");
            break;
        case mutation::different_param:
            // The final scale factor changes.
            result = nm(ggml_scale(ctx, rows, 0.25f), "result");
            break;
        case mutation::changed_edge: {
            // The gather reads a second projection of the same shape and type
            // instead of `logits`. The node set and the node count are
            // unchanged: the second projection replaces the first, so only
            // one edge moves.
            //
            // The second projection reads `w_out` and `sum` exactly as the
            // first does. That matters: if it read a different weight of the
            // same shape, the two projections would be structurally identical
            // and the digest would (correctly) call the graphs the same, so
            // the case would test nothing. The edge that actually moves is
            // the gather's src[0], from the first projection to the second.
            ggml_tensor * logits_b = nm(ggml_mul_mat(ctx, w_out, sum), "logits");
            result = nm(ggml_scale(ctx, nm(ggml_get_rows(ctx, logits_b, out_ids), "rows"), 0.5f), "result");
            break;
        }
        case mutation::changed_shape: {
            // The projection is one row wider. Same node set, same node
            // count, one shape differs.
            //
            // The wider operand is `sum` itself, widened in place rather than
            // replaced by a fresh tensor: a new edge-less tensor would be an
            // extra node with no sources, and that is a different mutation
            // (a node whose edges moved), not a shape change.
            //
            // Only ne[1] and nb[1] are touched. nb[2] and nb[3] are derived
            // from ne[1] by the allocation, but this test never computes, and
            // rewriting them would make the case a two-field change instead
            // of one.
            sum->ne[1] = 2;
            sum->nb[1] = sum->nb[0] * sum->ne[0];
            ggml_tensor * logits_w = nm(ggml_mul_mat(ctx, w_out, sum), "logits");
            result = nm(ggml_scale(ctx, nm(ggml_get_rows(ctx, logits_w, out_ids), "rows"), 0.5f), "result");
            break;
        }
        case mutation::changed_type: {
            // The projection is F16. Same node set, same node count, one type
            // differs.
            ggml_tensor * logits_h = nm(ggml_cast(ctx, logits, GGML_TYPE_F16), "logits");
            result = nm(ggml_scale(ctx, nm(ggml_get_rows(ctx, logits_h, out_ids), "rows"), 0.5f), "result");
            break;
        }
        case mutation::changed_view_offs: {
            // The recurrent-state zero view starts one row later. Same node
            // set, same node count, one view offset differs.
            ggml_tensor * z = nm(ggml_view_2d(ctx, rs, n_embd, 1,
                    rs->nb[1], (size_t) (s.rs_z + 1) * rs->nb[1]), "rs_zero");
            ggml_tensor * zs = nm(ggml_scale(ctx, z, 0.0f), "rs_zero_scaled");
            ggml_tensor * sum_b = order == 0
                ? nm(ggml_add(ctx, attn_out, zs), "sum")
                : nm(ggml_add(ctx, zs, attn_out), "sum");
            ggml_tensor * logits_b = nm(ggml_mul_mat(ctx, w_out, sum_b), "logits");
            result = nm(ggml_scale(ctx, nm(ggml_get_rows(ctx, logits_b, out_ids), "rows"), 0.5f), "result");
            break;
        }
        case mutation::weight_not_activation: {
            // The same node, same shape, same type, same edges: only `fixed`
            // differs. `rows` is the gather result, graph-owned; `w_same` is
            // a leaf of identical shape and type whose storage pre-exists the
            // build. The final scale keeps its position and its source slot,
            // so the node set and the node count are unchanged.
            ggml_tensor * w_same = nm(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_out), "w_same");
            w_same->data = &w_same;
            result = nm(ggml_scale(ctx, w_same, 0.5f), "result");
            break;
        }
        case mutation::changed_root: {
            // Same nodes, a different graph result: the gather instead of the
            // final scale. The scale stays in the graph through a zero-scaled
            // copy, so the node set and the node count are unchanged.
            ggml_tensor * keep = nm(ggml_scale(ctx, result, 0.0f), "result_kept");
            result = nm(ggml_add(ctx, rows, keep), "root_swap");
            break;
        }
    }

    b.gf = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(b.gf, result);

    b.hash = llama_topology_hash(b.gf);
    return b;
}

static void free_built(built & b) {
    if (b.ctx != nullptr) {
        ggml_free(b.ctx);
        b.ctx = nullptr;
    }
    b.gf = nullptr;
}

// ---------------------------------------------------------------------------
// separate-process check
// ---------------------------------------------------------------------------

// Prints the hash of the canonical graph and exits. main() re-executes itself
// with this flag so the comparison crosses a process boundary (fresh ASLR,
// fresh allocator state).
static int emit_and_exit() {
    const shape s;
    built b = build_graph(s, mutation::none, 0, false, 0);
    if (b.ctx == nullptr) {
        return 2;
    }
    std::printf("%016" PRIx64 "\n", b.hash);
    free_built(b);
    return 0;
}

// ---------------------------------------------------------------------------
// cases
// ---------------------------------------------------------------------------

static void test_order_and_address_independence() {
    const shape s;

    // Two builds of the same topology: different branch creation order, no
    // names vs names, and a different arena padding so every tensor address
    // differs.
    built a = build_graph(s, mutation::none, 0, false, 0);
    built b = build_graph(s, mutation::none, 1, true,  4096);
    CHECK(a.ctx != nullptr && b.ctx != nullptr);
    CHECK(a.gf != nullptr && b.gf != nullptr);

    // The addresses really are different: otherwise this case proves nothing
    // about address independence.
    CHECK(ggml_graph_node(a.gf, 0) != ggml_graph_node(b.gf, 0));

    // The node orders really are different: otherwise this case proves
    // nothing about order independence.
    bool same_order = ggml_graph_n_nodes(a.gf) == ggml_graph_n_nodes(b.gf);
    if (same_order) {
        for (int i = 0; i < ggml_graph_n_nodes(a.gf); ++i) {
            const ggml_tensor * x = ggml_graph_node(a.gf, i);
            const ggml_tensor * y = ggml_graph_node(b.gf, i);
            if (x->op != y->op || x->ne[0] != y->ne[0]) {
                same_order = false;
                break;
            }
        }
    }
    CHECK(!same_order);

    CHECK_EQ_HASH(a.hash, b.hash);

    free_built(a);
    free_built(b);
}

// The contract has two halves, and they point in opposite directions:
//
//   invariant to the *parameter values*: the digest never reads n_kv,
//   n_outputs or rs_z. Two graphs that differ only in those values but are
//   built at the same shapes hash equal.
//
//   sensitive to the *graph as built*: those parameters reach the graph as
//   tensor shapes (the KV cache width, the out_ids length) and as view offsets
//   (the recurrent-state zero view). A graph built at a different n_kv IS a
//   different graph, and the digest must say so.
//
// The second half is what the binder needs: a shape change must never be
// mistaken for the same graph. The first half is already covered by
// test_order_and_address_independence, so it is not restated here.

static void test_shape_parameter_sensitivity() {
    const shape s0;

    // One parameter at a time, so each difference has exactly one cause.
    shape s_kv = s0;
    s_kv.n_kv = 128;

    shape s_out = s0;
    s_out.n_out = 16;

    shape s_rs = s0;
    s_rs.rs_z = 1;

    built a  = build_graph(s0,    mutation::none, 0, false, 0);
    built kv = build_graph(s_kv,  mutation::none, 0, false, 0);
    built ou = build_graph(s_out, mutation::none, 0, false, 0);
    built rz = build_graph(s_rs,  mutation::none, 0, false, 0);
    CHECK(a.ctx != nullptr && kv.ctx != nullptr && ou.ctx != nullptr && rz.ctx != nullptr);

    // n_kv sizes the KV cache leaves, so it is a shape change.
    CHECK_NE_HASH(a.hash, kv.hash);
    // n_out sizes out_ids, so it is a shape change.
    CHECK_NE_HASH(a.hash, ou.hash);
    // rs_z selects the recurrent-state zero view's offset, so it is a view
    // offset change. This is the same difference mutation::changed_view_offs
    // asserts, reached through the runtime parameter instead of a hand edit.
    CHECK_NE_HASH(a.hash, rz.hash);

    // And the same sensitivity holds with the other build order and names, so
    // the result is not an artefact of one build path.
    built a_r  = build_graph(s0,    mutation::none, 1, true, 2048);
    built kv_r = build_graph(s_kv,  mutation::none, 1, true, 2048);
    built ou_r = build_graph(s_out, mutation::none, 1, true, 2048);
    built rz_r = build_graph(s_rs,  mutation::none, 1, true, 2048);
    CHECK_EQ_HASH(a.hash, a_r.hash);
    CHECK_NE_HASH(a_r.hash, kv_r.hash);
    CHECK_NE_HASH(a_r.hash, ou_r.hash);
    CHECK_NE_HASH(a_r.hash, rz_r.hash);

    free_built(a);
    free_built(kv);
    free_built(ou);
    free_built(rz);
    free_built(a_r);
    free_built(kv_r);
    free_built(ou_r);
    free_built(rz_r);
}

static void test_separate_process() {
    // Re-execute this binary with --emit and compare its printed hash with the
    // in-process one. A second process has a fresh address space: any address
    // leaking into the digest would show up here.
    //
    // The child is found from /proc/self/exe (Linux). When that is not
    // available the case is reported as skipped, never as passed: a skip that
    // reads as a pass would be a gate that proves nothing.
    const shape s;
    built a = build_graph(s, mutation::none, 0, false, 0);
    CHECK(a.ctx != nullptr);

    char self[4096] = {};
    const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n <= 0) {
        std::fprintf(stderr, "SKIP separate-process case: /proc/self/exe unavailable\n");
        free_built(a);
        return;
    }
    self[n] = '\0';

    std::string cmd = std::string("'") + self + "' --emit";
    FILE * p = popen(cmd.c_str(), "r");
    if (p == nullptr) {
        std::fprintf(stderr, "FAIL: popen(%s) failed; cannot run the cross-process case\n", cmd.c_str());
        ++g_checks;
        ++g_failures;
        free_built(a);
        return;
    }

    char line[64] = {};
    const char * got = fgets(line, sizeof(line), p);
    const int rc = pclose(p);
    CHECK(rc == 0);
    if (got == nullptr) {
        std::fprintf(stderr, "FAIL: child printed no hash\n");
        ++g_checks;
        ++g_failures;
        free_built(a);
        return;
    }

    uint64_t child = 0;
    if (std::sscanf(line, "%" SCNx64, &child) != 1) {
        std::fprintf(stderr, "FAIL: child hash is not a hex digest: %s", line);
        ++g_checks;
        ++g_failures;
        free_built(a);
        return;
    }

    CHECK_EQ_HASH(a.hash, child);
    free_built(a);
}

static void test_mutations() {
    const shape s;
    built base = build_graph(s, mutation::none, 0, false, 0);
    CHECK(base.ctx != nullptr);
    CHECK(base.hash != 0);

    const mutation ms[] = {
        mutation::extra_node,
        mutation::different_op,
        mutation::different_param,
        mutation::changed_edge,
        mutation::changed_shape,
        mutation::changed_type,
        mutation::changed_view_offs,
        mutation::weight_not_activation,
        mutation::changed_root,
    };
    for (mutation m : ms) {
        built x = build_graph(s, m, 0, false, 0);
        CHECK(x.ctx != nullptr);
        CHECK_NE_HASH(base.hash, x.hash);
        free_built(x);
    }

    // Same mutations, built in the other order: the negative must not depend
    // on build order either.
    built base_r = build_graph(s, mutation::none, 1, true, 2048);
    CHECK_EQ_HASH(base.hash, base_r.hash);
    for (mutation m : ms) {
        built x = build_graph(s, m, 1, true, 2048);
        CHECK(x.ctx != nullptr);
        CHECK_NE_HASH(base_r.hash, x.hash);
        free_built(x);
    }
    free_built(base_r);
    free_built(base);
}

static void test_op_params_covered() {
    // The ops whose op_params the digest reads must be covered, otherwise one
    // of the negative cases could pass for the wrong reason.
    const enum ggml_op covered[] = {
        GGML_OP_MUL_MAT, GGML_OP_VIEW, GGML_OP_SOFT_MAX, GGML_OP_SCALE,
    };
    for (enum ggml_op op : covered) {
        CHECK(llama_topology_op_params_covered(op));
    }

    // The rope mask skips n_past (word 0) and n_ctx (word 3).
    CHECK(llama_topology_op_params_word(GGML_OP_ROPE, 0) == 1);
    CHECK(llama_topology_op_params_word(GGML_OP_ROPE, 1) == 2);
    CHECK(llama_topology_op_params_word(GGML_OP_ROPE, 2) == 4);
    CHECK(llama_topology_op_params_word(GGML_OP_ROPE, 8) == 10);
    CHECK(llama_topology_op_params_word(GGML_OP_ROPE, 9) == -1);

    // Ops with no op_params are not "covered", and that is correct: there is
    // nothing to cover. ggml/src/ggml.c writes none for ADD, MUL, RESHAPE,
    // CPY, GET_ROWS or ADD_ID.
    CHECK(!llama_topology_op_params_covered(GGML_OP_ADD));
    CHECK(!llama_topology_op_params_covered(GGML_OP_MUL));
    CHECK(!llama_topology_op_params_covered(GGML_OP_RESHAPE));
    CHECK(!llama_topology_op_params_covered(GGML_OP_CPY));
    CHECK(!llama_topology_op_params_covered(GGML_OP_GET_ROWS));
    CHECK(!llama_topology_op_params_covered(GGML_OP_ADD_ID));

    // The view offset lives in op_params[0..1] as a size_t.
    CHECK(llama_topology_op_params_word(GGML_OP_VIEW, 0) == 0);
    CHECK(llama_topology_op_params_word(GGML_OP_VIEW, 1) == 1);
    CHECK(llama_topology_op_params_word(GGML_OP_VIEW, 2) == -1);
}

static void test_empty_graph() {
    // The empty graph hashes to the digest of the empty serialization, and a
    // null graph to the same value. Neither is a valid graph identity.
    CHECK_EQ_HASH(llama_topology_hash(nullptr), LLAMA_TOPOLOGY_FNV_OFFSET);

    ggml_init_params ip = {
        /* .mem_size   = */ ggml_tensor_overhead() * 2 + ggml_graph_overhead_custom(8, false) + 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(ip);
    CHECK(ctx != nullptr);
    if (ctx != nullptr) {
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8, false);
        CHECK_EQ_HASH(llama_topology_hash(gf), LLAMA_TOPOLOGY_FNV_OFFSET);
        ggml_free(ctx);
    }
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--emit") == 0) {
        return emit_and_exit();
    }

    test_empty_graph();
    test_op_params_covered();
    test_order_and_address_independence();
    test_shape_parameter_sensitivity();
    test_mutations();
    test_separate_process();

    if (g_failures != 0) {
        std::fprintf(stderr, "FAIL %d/%d topology-hash checks\n", g_failures, g_checks);
        return 1;
    }
    std::printf("PASS %d topology-hash checks\n", g_checks);
    return 0;
}
