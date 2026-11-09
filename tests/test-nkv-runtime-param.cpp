// CPU unit test: the live n_kv lane of the union binder.
//
// Pillar 1 (n_kv, n_outputs, rs_z and n_tokens as runtime parameters of ONE
// fixed-topology graph) has ONE cross matrix:
// tests/test-union-four-param-matrix.cpp. This file keeps the two claims that
// matrix does not make, because they are about the live lane rather than about
// a parameter sweep.
//
// Part 1 (always runs, no model): the topology hash and the width table over
// the n_kv, n_outputs and rs_z lanes.
//   - llama_union_topology_hash::hash_nodes() must return the same value for
//     the same synthetic topology built at different combinations of the
//     three parameters, including an n_kv value that crosses the 256-cell
//     boundary and a non-aligned value;
//   - it must return a different value when something that is NOT a runtime
//     dimension changes: node count, op type, an op parameter, an edge, a
//     non-runtime shape dimension, a node name, or a view's offset when that
//     view is not a runtime root;
//   - llama_union_width_table::build() must find the n_kv dimension of the
//     three attention roots ("union_nkv_k", "union_nkv_v" in both layouts,
//     "attn_inp_kq_mask"), the n_outputs row of "union_out_ids", and the
//     length/head axes of the recurrent roots, and refuse a root whose shape
//     disagrees with the recorded value;
//   - set_runtime_n_kv() must fail closed for 0 and for a width above the
//     live arena width, and effective_n_kv() must report the armed value.
//     The bound is the arena, not the build width: the token lane builds at
//     the first live width and a successful replan widens the arena, so a
//     build-width bound would refuse a width the bind just applied.
//
// Part 2 (needs a model, and a missing model is an ERROR, not a skip): the
// live llama_decode path.
//   - decode a growing sequence with GGML_TP5_UNION_BINDER=1;
//   - read llama_get_nkv_runtime_stats() after every step: the topology hash
//     must be identical at every width, the build width must stay at the
//     allocation envelope, and runtime_n_kv must equal the 1-based position
//     (the true used span) after the first step;
//   - the widths must include a value below 256, one that crosses 256 and one
//     above 256, so the "same topology across the 256 boundary" claim is
//     exercised on a real graph, not only on synthetic tensors;
//   - the hash must have been RE-TAKEN over real binds: at least one step
//     must execute below the build envelope (bound_n_kv < build_n_kv), the
//     bound width must move between steps, and more than one distinct bound
//     width must be seen. Without this a hash captured once at build time
//     would look stable for the wrong reason. The cross matrix cannot make
//     this claim: it asserts zero rebuilds and a stable hash, not that a bind
//     happened at all.
//   - the mask read-back must have run on every step and agreed on every row
//     it was owed (mask_row_checked > 0, mask_row_ok == mask_row_checked,
//     mask_row_refused == 0, mask_rows_all == 1). The fill width and the
//     executed width both come from the binder; the read-back is the only
//     independent observation of the mask.
//
// Part 4 (needs a model): the mask path must be able to fail.
//   - positive: one decode fills the KQ mask at exactly the width the graph
//     reports it executes at, with no refusal, and the read-back compares
//     every row the batch owns and agrees on all of them;
//   - negative: the same decode with the runtime width forced above the
//     mask's ne[0] must abort in a child process with the named message
//     "runtime n_kv exceeds the KQ mask width". A step0 that cannot fail is
//     a defect, so this case exists to fail if the guard is ever removed.
//
// What this test does NOT prove:
//   - that a ggml kernel reads the runtime width for the MASK. The mask
//     consumer is the host: llm_graph_input_nkv_runtime::set_input fills the
//     KQ mask at the value it just wrote (llama_kv_cache::set_input_kq_mask
//     with an explicit width). The K/V views take their width from the
//     binder's in-place view patch, a separate mechanism;
//   - anything about GPU/HIP, flash attention, relay or pipeline graphs.
//
// What it DOES prove about the CPU attention work: the reduction length is a
// runtime value the kernels follow. The ordered softmax kernel adds
// attn_live_elems from the loop bound it executes (ne00, the width the bind
// patched in), one row per executed reduction, and the same value as
// attn_pad_elems. On this lane there is no padded work to report: the bind
// narrowed the mask and the K/V views to the runtime width, so the executed
// length IS the extent. The assertions require rows * runtime_n_kv == live
// and live == pad on every step, while the build envelope is separately
// asserted not to move. The counters measure only the runtime-length KV
// reduction: the generic mul_mat counts a node only when it carries
// GGML_TENSOR_FLAG_TRUE_LEN_REDUCE, so an unflagged product (the LM head, the
// projections) cannot move them.
//
// usage: test-nkv-runtime-param <model.gguf> [n_steps]
//        test-nkv-runtime-param --hash-only
//        test-nkv-runtime-param --mask-negative <model.gguf>   (part 4 child)
//   n_steps defaults to 300 and must be > 256 so the sequence crosses the
//   256-cell boundary. --hash-only runs part 1 alone (no model, no backend).
//
// Output lines (parsed by tests/nkv-cpu-probe/gate-nkv-runtime-13022.sh):
//   NKV_RT_HASH_ARM ok=1 cases=<n> widths=<n>
//   NKV_RT_TRIPARAM ok=1 combos=<n> roots=<n>
//   NKV_RT_LIVE ok=1 steps=<n> hash=<hex> min_width=<n> max_width=<n>
//                bind_steps=<n> distinct_bound=<n>
//   TRIPARAM_MASK_NEG ok=<0|1>
//   NKV_RT_END ok=<0|1>
//   NKV_RT_TRUELEN ok=<0|1> steps=<n> no_pad=<n> pad_steps=<n>

#include "llama.h"

#include "llama-union-topology.h"
#include "llama-union-width-table.h"

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// synthetic graph builder: a tiny but realistic skeleton of the three runtime
// lanes
// ---------------------------------------------------------------------------
//
//   k_view  = view_4d(head_k, n_head_kv, n_kv, ns)      "union_nkv_k"
//   v_view  = view_4d(head_v, n_head_kv, n_kv, ns)      "union_nkv_v"
//   kq      = mul_mat(k_view, q)                        "kq"
//   mask    = new_tensor_4d(n_kv, n_tokens, 1, 1)       "attn_inp_kq_mask"
//   kq_sm   = soft_max_ext(kq, mask, scale, bias)       "kq_soft_max"
//   kqv     = mul_mat(v_view, kq_sm)                    "kqv"
//   oids    = new_tensor_1d(I32, n_outputs)             "union_out_ids"
//   rows    = get_rows(kqv, oids)                       "rows"
//   scopy   = new_tensor_1d(I32, n_rs)                  "union_rs_copy"
//   sextra  = view_1d(scopy, n_rs - n_seqs, ...)        "union_rs_copy_extra"
//   sdst    = view_2d(state, state_size, n_rs - n_seqs, nb1, (head + n_seqs)*nb1)
//                                                       "union_rs_copy_dst"
//   states  = get_rows(state, scopy)                    "states"
//
// The tensors are allocated in a ggml context with no data buffer: only the
// metadata (ne/nb/op/op_params/src/name/view_offs) matters to the hash and the
// width table, so the test needs no backend and no allocation.
struct synth_graph {
    ggml_context * ctx = nullptr;
    std::vector<ggml_tensor *> nodes;
};

static void synth_free(synth_graph & g) {
    if (g.ctx != nullptr) {
        ggml_free(g.ctx);
        g.ctx = nullptr;
    }
    g.nodes.clear();
}

// `perturb` selects one deliberate non-runtime difference for the negative
// cases
enum synth_perturb {
    SYNTH_PLAIN          = 0,
    SYNTH_EXTRA_NODE     = 1, // node count changes
    SYNTH_OP_PARAM       = 2, // a non-runtime op parameter changes
    SYNTH_EDGE           = 3, // one edge is rewired
    SYNTH_NON_KV_SHAPE   = 4, // a non-runtime dimension changes
    SYNTH_NAME           = 5, // a node name changes
    SYNTH_VIEW_OFFSET    = 6, // a non-runtime view offset changes
};

struct synth_params {
    int64_t n_kv      = 64;
    int64_t n_tokens  = 1;
    int64_t n_outputs = 1;
    int64_t n_rs      = 8;
    int64_t n_seqs    = 1;
    int64_t rs_head   = 0;
};

static bool synth_build(synth_graph & g, const synth_params & p,
                        const char * v_name, int synth_perturb) {
    synth_free(g);

    ggml_init_params ip = {};
    ip.mem_size   = 8u * 1024u * 1024u;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    g.ctx = ggml_init(ip);
    if (g.ctx == nullptr) {
        return false;
    }

    const int64_t head_k  = 64;
    const int64_t head_v  = 64;
    const int64_t n_head  = 4;
    const int64_t state_size = 128;

    ggml_tensor * q = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, head_k, n_head, p.n_tokens);
    ggml_set_name(q, "q");

    ggml_tensor * k_cache = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, head_k, n_head, p.n_kv, 1);
    ggml_set_name(k_cache, "k_cache");
    // KV STORAGE: a fixed leaf, as in the live cache. The canonical walk stops
    // at a fixed node, so the live cache tensor is not a member of the hashed
    // set; only the views built over it are. Hashing it here would fold
    // ne[2] = n_kv, a dimension the binder patches on the views.
    k_cache->data = &k_cache;

    // the n_kv-carrying view: dimension 2
    ggml_tensor * k_view = ggml_view_4d(g.ctx, k_cache, head_k, n_head, p.n_kv, 1,
            ggml_row_size(k_cache->type, head_k),
            ggml_row_size(k_cache->type, head_k*n_head),
            ggml_row_size(k_cache->type, head_k*n_head*p.n_kv),
            0);
    ggml_set_name(k_view, "union_nkv_k");

    // the n_kv-carrying view: dimension 2 as well (non-transposed layout)
    ggml_tensor * v_view = ggml_view_4d(g.ctx, k_cache, head_v, n_head, p.n_kv, 1,
            ggml_row_size(k_cache->type, head_v),
            ggml_row_size(k_cache->type, head_v*n_head),
            ggml_row_size(k_cache->type, head_v*n_head*p.n_kv),
            0);
    ggml_set_name(v_view, v_name);

    // The attention products are built the way the live graph builds them
    // (llama-graph.cpp:3226-3228): the KV views keep n_kv in dimension 2 (the
    // width table's root rule), and the products read PERMUTED copies, so
    // ggml_can_mul_mat's a->ne[0] == b->ne[0] holds on the axis the real graph
    // uses. Without the permutes the fixture asked ggml for
    // mul_mat([head_v, n_head, n_kv], [n_head, n_tokens]), which cannot exist.
    ggml_tensor * k_perm = ggml_permute(g.ctx, k_view, 0, 2, 1, 3);   // [head_k, n_kv, n_head]
    ggml_set_name(k_perm, "k_perm");
    ggml_tensor * q_perm = ggml_permute(g.ctx, q, 0, 2, 1, 3);        // [head_k, n_tokens, n_head]
    ggml_set_name(q_perm, "q_perm");

    ggml_tensor * kq = ggml_mul_mat(g.ctx, k_perm, q_perm);           // [n_kv, n_tokens, n_head]
    ggml_set_name(kq, "kq");

    ggml_tensor * mask = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, p.n_kv, p.n_tokens, 1, 1);
    ggml_set_name(mask, "attn_inp_kq_mask");

    ggml_tensor * kq_sm = ggml_soft_max_ext(g.ctx, kq, mask, 1.0f, 0.0f);
    ggml_set_name(kq_sm, "kq_soft_max");

    // The value product reads the transposed layout, as the live graph does
    // when the cache is not stored transposed (llama-graph.cpp:3341-3343):
    // cont(transpose(permute(v))) is [n_kv, head_v, n_head], so a->ne[0] is
    // n_kv and the scores' ne[0] is n_kv too.
    ggml_tensor * v_perm = ggml_permute(g.ctx, v_view, 0, 2, 1, 3);   // [head_v, n_kv, n_head]
    ggml_set_name(v_perm, "v_perm");
    ggml_tensor * v_t = ggml_cont(g.ctx, ggml_transpose(g.ctx, v_perm)); // [n_kv, head_v, n_head]
    ggml_set_name(v_t, "v_t");
    ggml_tensor * kqv = ggml_mul_mat(g.ctx, v_t, kq_sm);              // [head_v, n_tokens, n_head]
    ggml_set_name(kqv, "kqv");

    // ggml_get_rows needs a 2-D source; the live graph reshapes the attention
    // output the same way (llama-graph.cpp:3300).
    ggml_tensor * kqv2 = ggml_reshape_2d(g.ctx, kqv, head_v, p.n_tokens * n_head);
    ggml_set_name(kqv2, "kqv_2d");

    // n_outputs lane: the gather width is the row count of out_ids
    ggml_tensor * oids = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, p.n_outputs);
    ggml_set_name(oids, "union_out_ids");
    ggml_tensor * rows = ggml_get_rows(g.ctx, kqv2, oids);
    ggml_set_name(rows, "rows");

    // rs_z lane: the recurrent state length and the head-selected copy views
    // The state's row count is the storage envelope: the head test moves the
    // view offset by rs_head rows, so the allocation must cover n_rs + rs_head
    // rows or ggml_view_2d refuses the view (measured: GGML_ASSERT in
    // ggml_new_tensor_impl from synth_build at rs_head=3). The state is a
    // fixed leaf, so its size does not enter the hash.
    ggml_tensor * state = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, state_size, p.n_rs + p.rs_head);
    ggml_set_name(state, "state");
    // The recurrent state is STORAGE: a fixed leaf whose address the graph never
    // owns. The digest's leaf predicate is the binder record's
    // (`view_src == nullptr && (data != nullptr || buffer != nullptr)`), and the
    // canonical walk stops at a fixed node, so the live cache tensor is not a
    // member of the hashed set -- only the views and gathers built over it are.
    // Without this the fixture hashes ne[1] = n_rs, a dimension the binder
    // patches on the views, and the hash moves with n_rs for a node the bind
    // never touches.
    state->data = &state;

    ggml_tensor * scopy = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, p.n_rs);
    ggml_set_name(scopy, "union_rs_copy");

    ggml_tensor * sextra = ggml_view_1d(g.ctx, scopy, p.n_rs - p.n_seqs, p.n_seqs * scopy->nb[0]);
    ggml_set_name(sextra, "union_rs_copy_extra");

    ggml_tensor * sdst = ggml_view_2d(g.ctx, state, state_size, p.n_rs - p.n_seqs,
            state->nb[1], (size_t) (p.rs_head + p.n_seqs) * state->nb[1]);
    ggml_set_name(sdst, "union_rs_copy_dst");

    ggml_tensor * states = ggml_get_rows(g.ctx, state, scopy);
    ggml_set_name(states, "states");

    // The model's state-store view: a constant shape whose base slot is
    // selected by the head through the view offset (llama-graph.cpp
    // build_rs_inp_impl names it union_rs_head_view). The width table needs it
    // to record the head axis; the hash skips a view's offset, so a head
    // change must not move it.
    ggml_tensor * shead = ggml_view_1d(g.ctx, state, state_size,
            (size_t) p.rs_head * state->nb[1]);
    ggml_set_name(shead, "union_rs_head_view");

    switch (synth_perturb) {
        case SYNTH_EXTRA_NODE:
            {
                ggml_tensor * extra = ggml_scale(g.ctx, kqv, 2.0f);
                ggml_set_name(extra, "extra");
            }
            break;
        case SYNTH_OP_PARAM:
            // the softmax scale is a non-runtime op parameter
            ggml_soft_max_ext(g.ctx, kq, mask, 0.5f, 0.0f);
            break;
        case SYNTH_EDGE:
            // the softmax now reads the un-masked scores. The mask must be
            // shaped for the tensor it is applied to: kqv is
            // [head_v, n_tokens, n_head], so its mask is [head_v, n_tokens].
            // (The edge is the mutation; the mask's own width is not.)
            {
                ggml_tensor * mask_v = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32,
                        head_v, p.n_tokens, 1, 1);
                ggml_set_name(mask_v, "kq_mask_v");
                ggml_soft_max_ext(g.ctx, kqv, mask_v, 1.0f, 0.0f);
            }
            break;
        case SYNTH_NON_KV_SHAPE:
            {
                // A non-runtime dimension of the attention output changes:
                // n_head becomes n_head + 1. ggml_add cannot express that
                // (broadcasting requires divisibility, and 4 % 5 != 0), so the
                // wider tensor is CONCATENATED on the head axis, which is the
                // same statement -- the graph now contains a tensor whose head
                // count is one larger -- with a legal op.
                ggml_tensor * wider = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32,
                        head_v, p.n_tokens, n_head + 1);
                ggml_set_name(wider, "wider");
                ggml_concat(g.ctx, kqv, wider, 2);
            }
            break;
        case SYNTH_NAME:
            ggml_set_name(kqv, "kqv_renamed");
            break;
        case SYNTH_VIEW_OFFSET:
            {
                // a view that is NOT a runtime root moves: the hash must see it
                ggml_tensor * shifted = ggml_view_2d(g.ctx, state, state_size, 1,
                        state->nb[1], (size_t) 2 * state->nb[1]);
                ggml_set_name(shifted, "shifted");
            }
            break;
        case SYNTH_PLAIN:
        default:
            break;
    }

    // graph order, source before consumer
    g.nodes = { q, k_cache, k_view, v_view, k_perm, q_perm, kq, mask, kq_sm,
                v_perm, v_t, kqv, kqv2, oids, rows, scopy, sextra, sdst, states,
                shead };

    // drop the storage leaf the fixed-leaf filter above cannot see: `k_cache` is
    // in the explicit list, so the append loop never re-checks it.
    for (size_t i = 0; i < g.nodes.size(); ) {
        ggml_tensor * t = g.nodes[i];
        const bool fixed_leaf = t->view_src == nullptr &&
                (t->data != nullptr || t->buffer != nullptr);
        if (fixed_leaf) {
            g.nodes.erase(g.nodes.begin() + (long) i);
        } else {
            ++i;
        }
    }

    // keep the perturbed nodes reachable through the context
    for (ggml_tensor * t = ggml_get_first_tensor(g.ctx); t != nullptr; t = ggml_get_next_tensor(g.ctx, t)) {
        bool seen = false;
        for (ggml_tensor * n : g.nodes) {
            if (n == t) { seen = true; break; }
        }
        // A fixed leaf is storage, not a member of the canonical set the digest
        // hashes (llama_topology_canonical_nodes stops at a fixed node). Adding
        // one here would hash a shape the live graph never hashes -- `state`
        // and `k_cache` carry n_rs and n_kv in ne[1]/ne[2] and are exactly
        // that case.
        const bool fixed_leaf = t->view_src == nullptr &&
                (t->data != nullptr || t->buffer != nullptr);
        if (!seen && !fixed_leaf) {
            g.nodes.push_back(t);
        }
    }
    return true;
}

// The runtime dimensions the BINDER patches on this skeleton: the same set
// union_shape_analyze() derives from the shape map's coefficient table. The
// NAME-based table alone cannot exclude the derived tensors -- `rows`'s ne[1]
// inherits out_ids's length and `states`'s ne[1] inherits scopy's -- and this
// test asserts the table's hash is stable across all three parameters, so the
// table must be given them.
static std::vector<llama_union_topology_excl> synth_runtime_dims(const synth_graph & g) {
    std::vector<llama_union_topology_excl> out(g.nodes.size());
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        out[i].node = (int32_t) i;
    }
    auto mark = [&g, &out](const char * name, int dim, int which) {
        for (size_t i = 0; i < g.nodes.size(); ++i) {
            if (std::strcmp(g.nodes[i]->name, name) != 0) {
                continue;
            }
            llama_union_topology_excl & e = out[i];
            // A runtime width on axis r moves ne[r] AND every stride above it:
            // nb[d] (d >= 1) is the product of ne[0..d-1] times the type size,
            // so nb[d] depends on r whenever r < d. union_contiguous_nb()
            // derives exactly those nonzero nb coefficients, and the binder
            // patches them, so the exclusion must name them.
            for (int d = dim; d < GGML_MAX_DIMS; ++d) {
                e.dim_mask |= (uint8_t) (1u << d);
            }
            switch (which) {
                case 0: e.dim      = dim; break;
                case 1: e.nout_dim = dim; break;
                case 2: e.head_dim = dim; break;
                case 3: e.nrs_dim  = dim; break;
                default: e.ntok_dim = dim; break;
            }
            return;
        }
    };
    mark("union_nkv_k", 2, 0);
    // union_nkv_v: the layout decides the axis (nb[1] > nb[2] -> dim 0, else
    // dim 2), exactly as the width table's own name rule reads it
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        if (std::strcmp(g.nodes[i]->name, "union_nkv_v") != 0) {
            continue;
        }
        const int dim = g.nodes[i]->nb[1] > g.nodes[i]->nb[2] ? 0 : 2;
        llama_union_topology_excl & e = out[i];
        // The view's strides above the runtime axis depend on n_kv too:
        // nb[3] is row_size * n_head * n_kv. Mark every axis from the runtime
        // one upward, exactly as mark() does; a single-axis mask leaves nb[3]
        // in the hash and the hash moves at n_kv=256 (measured).
        for (int d = dim; d < GGML_MAX_DIMS; ++d) {
            e.dim_mask |= (uint8_t) (1u << d);
        }
        e.dim = dim;
        break;
    }
    mark("attn_inp_kq_mask", 0, 0);
    // the mul_mat rule propagates n_kv to out.ne[0]; soft_max keeps it
    mark("kq", 0, 0);
    mark("kq_soft_max", 0, 0);
    // PERMUTE (0,2,1,3) moves n_kv from the view's dim 2 to the permuted dim 1,
    // and swaps the strides with it. Both axes are runtime.
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        const char * nm = g.nodes[i]->name;
        if (std::strcmp(nm, "k_perm") != 0 && std::strcmp(nm, "v_perm") != 0) {
            continue;
        }
        llama_union_topology_excl & e = out[i];
        // The permute swaps ne[1] and ne[2] but leaves nb[3] as the source
        // view's nb[3], which is row_size * n_head * n_kv. Mark every axis
        // from 1 upward, or nb[3] keeps n_kv in the hash.
        for (int d = 1; d < GGML_MAX_DIMS; ++d) {
            e.dim_mask |= (uint8_t) (1u << d);
        }
        e.dim = 1;
    }
    // The transpose ggml_transpose() inserts cannot be named in a whitelist:
    // ggml_view_tensor() + ggml_format_name("%s (transposed)") (ggml.c:3989)
    // builds "v_perm (transposed)" from the source name, so the intermediate
    // of cont(transpose(permute(v))) is invisible to the strcmp marks above.
    // Propagate the runtime axes by OP instead. The product's TRANSPOSE rule
    // (src/llama-context.cpp:4115-4145) maps source axis i to output axis
    // axes[i], out.ne[axes[i]] = s.ne[i], with axes = {1,0,2,3}; the same
    // mapping applied to the exclusion mask keeps this node's hash stable at
    // every n_kv. A source outside the hashed set is not hashed at all, so
    // its shape cannot move the hash and the node is skipped.
    static const int32_t transpose_axes[GGML_MAX_DIMS] = { 1, 0, 2, 3 };
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        const ggml_tensor * t = g.nodes[i];
        if (t->op != GGML_OP_TRANSPOSE || t->src[0] == nullptr) {
            continue;
        }
        size_t src = g.nodes.size();
        for (size_t j = 0; j < g.nodes.size(); ++j) {
            if (g.nodes[j] == t->src[0]) {
                src = j;
                break;
            }
        }
        if (src == g.nodes.size()) {
            continue;
        }
        uint8_t mask = 0;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (((out[src].dim_mask >> d) & 1u) != 0) {
                mask |= (uint8_t) (1u << transpose_axes[d]);
            }
        }
        out[i].dim_mask |= mask;
        // n_kv lands on the transposed output's axis 0. `dim` is only the
        // per-parameter example slot, so an existing value stays.
        if (out[i].dim == -1 && (mask & 1u) != 0) {
            out[i].dim = 0;
        }
    }
    // v_t = cont(transpose(permute(v))): ne[0] and every stride are n_kv, so
    // the whole tensor is runtime. Without this the hash moves at n_kv=256
    // (measured: v_t mask=0x0).
    mark("v_t", 0, 0);
    mark("union_out_ids", 0, 1);
    mark("rows", 1, 1);              // GET_ROWS: out.ne[1] = out_ids.ne[0]
    mark("union_rs_copy", 0, 3);
    mark("union_rs_copy_extra", 0, 3);
    mark("union_rs_copy_dst", 1, 3);
    mark("states", 1, 3);            // GET_ROWS: out.ne[1] = scopy.ne[0]
    return out;
}

static bool synth_table(const synth_graph & g, const synth_params & p, llama_union_width_table & table) {
    // 0: this synthetic graph carries no token-width runtime root, so the token
    // dimensions stay hashed as fixed shapes. The two factor arguments are
    // unused at width 0. The n_tokens lane has its own test
    // (tests/test-ntok-cross-matrix.cpp).
    return table.build(g.nodes, synth_runtime_dims(g), (uint32_t) p.n_kv,
                       (uint32_t) p.n_outputs, (uint32_t) p.n_rs, (uint32_t) p.rs_head,
                       0, 1, 1);
}

// ---------------------------------------------------------------------------
// part 1: synthetic topology hash + width table over all three parameters
// ---------------------------------------------------------------------------

static bool part1() {
    bool ok = true;

    // The n_kv values deliberately include:
    //   64   below the 256-cell bucket
    //   256  exactly the bucket
    //   257  the first value above it (crosses the boundary)
    //   300  an unaligned value well past it
    //   1024 a large aligned value
    const int64_t widths[] = { 64, 256, 257, 300, 1024 };
    const size_t  n_widths = sizeof(widths) / sizeof(widths[0]);

    // The n_outputs and rs_z values include 1, a value exactly at the 256-cell
    // bucket, and non-aligned values past it.
    const int64_t nouts[]  = { 1, 2, 257, 300 };
    const int64_t n_rs_v[] = { 8, 9, 260, 301 };
    const size_t  n_nouts  = sizeof(nouts)  / sizeof(nouts[0]);
    const size_t  n_nrs    = sizeof(n_rs_v) / sizeof(n_rs_v[0]);

    synth_params ref_p;
    ref_p.n_kv      = widths[0];
    ref_p.n_outputs = nouts[0];
    ref_p.n_rs      = n_rs_v[0];

    // reference: the plain topology at the first combination
    synth_graph ref;
    if (!synth_build(ref, ref_p, "union_nkv_v", SYNTH_PLAIN)) {
        std::fprintf(stderr, "FAIL: synth build\n");
        return false;
    }

    llama_union_width_table ref_table;
    if (!synth_table(ref, ref_p, ref_table)) {
        std::fprintf(stderr, "FAIL: width table build at n_kv=%" PRId64 "\n", ref_p.n_kv);
        synth_free(ref);
        return false;
    }
    if (!ref_table.has_nkv || !ref_table.has_nout || !ref_table.has_nrs) {
        std::fprintf(stderr, "FAIL: width table missed a runtime root (nkv=%d nout=%d nrs=%d)\n",
                int(ref_table.has_nkv), int(ref_table.has_nout), int(ref_table.has_nrs));
        synth_free(ref);
        return false;
    }

    // DIAGNOSTIC: which node's ne/nb moves with n_rs? Built once, printed to
    // stderr, so the exclusion set can be read off the graph instead of guessed.
    {
        synth_params q = ref_p;
        q.n_kv = ref_p.n_kv + 1;
        synth_graph g2;
        if (synth_build(g2, q, "union_nkv_v", SYNTH_PLAIN)) {
            for (size_t i = 0; i < ref.nodes.size() && i < g2.nodes.size(); ++i) {
                const ggml_tensor * a = ref.nodes[i];
                const ggml_tensor * b = g2.nodes[i];
                if (a == nullptr || b == nullptr || std::strcmp(a->name, b->name) != 0) {
                    continue;
                }
                bool ne_diff = false, nb_diff = false;
                for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                    if (a->ne[d] != b->ne[d]) ne_diff = true;
                    if (a->nb[d] != b->nb[d]) nb_diff = true;
                }
                if (ne_diff || nb_diff) {
                    // which axes are NOT excluded by the table built for this
                    // graph? Those are the ones that still reach the digest.
                    llama_union_width_table t2;
                    const std::vector<llama_union_topology_excl> dims2 = synth_runtime_dims(g2);
                    const bool built2 = synth_table(g2, q, t2);
                    uint8_t mask = 0;
                    for (const auto & e : dims2) {
                        if (e.node == (int32_t) i) { mask = e.dim_mask; }
                    }
                    std::fprintf(stderr, "NKV_DIAG %s ne=%d nb=%d mask=0x%x built=%d"
                            " a.ne=[%lld,%lld,%lld,%lld] a.nb=[%lld,%lld,%lld,%lld]\n",
                            a->name, int(ne_diff), int(nb_diff), (unsigned) mask, int(built2),
                            (long long) a->ne[0], (long long) a->ne[1], (long long) a->ne[2], (long long) a->ne[3],
                            (long long) a->nb[0], (long long) a->nb[1], (long long) a->nb[2], (long long) a->nb[3]);
                }
            }
            synth_free(g2);
        }
    }

    uint64_t ref_hash = 0;
    if (!ref_table.topology_hash(ref.nodes, ref_hash)) {
        std::fprintf(stderr, "FAIL: topology hash at n_kv=%" PRId64 "\n", ref_p.n_kv);
        synth_free(ref);
        return false;
    }

    int combos = 0;

    // positive: every combination hashes to the reference value, and the
    // table names every runtime root
    for (size_t w = 0; w < n_widths; ++w) {
        for (size_t o = 0; o < n_nouts; ++o) {
            for (size_t r = 0; r < n_nrs; ++r) {
                synth_params p;
                p.n_kv      = widths[w];
                p.n_outputs = nouts[o];
                p.n_rs      = n_rs_v[r];
                p.rs_head   = 0;

                synth_graph g;
                if (!synth_build(g, p, "union_nkv_v", SYNTH_PLAIN)) {
                    std::fprintf(stderr, "FAIL: synth build at n_kv=%" PRId64 "\n", p.n_kv);
                    ok = false;
                    synth_free(g);
                    continue;
                }
                llama_union_width_table table;
                if (!synth_table(g, p, table)) {
                    std::fprintf(stderr, "FAIL: width table build at n_kv=%" PRId64 " n_outputs=%" PRId64
                            " n_rs=%" PRId64 "\n", p.n_kv, p.n_outputs, p.n_rs);
                    ok = false;
                    synth_free(g);
                    continue;
                }
                uint64_t hash = 0;
                if (!table.topology_hash(g.nodes, hash)) {
                    std::fprintf(stderr, "FAIL: topology hash at n_kv=%" PRId64 "\n", p.n_kv);
                    ok = false;
                    synth_free(g);
                    continue;
                }
                if (hash != ref_hash) {
                    std::fprintf(stderr,
                            "FAIL: topology hash moved with a runtime parameter at"
                            " n_kv=%" PRId64 " n_outputs=%" PRId64 " n_rs=%" PRId64
                            " (ref=%016" PRIx64 " got=%016" PRIx64 ")\n",
                            p.n_kv, p.n_outputs, p.n_rs, ref_hash, hash);
                    ok = false;
                }
                // The table must name every runtime axis, and the SET is the
                // assertion -- not a count. A count cannot tell a correct set
                // from a wrong one of the same size, and it drifts whenever a
                // node is added. The set below is what union_shape_analyze()'s
                // rules produce for this skeleton: the three named roots plus
                // the tensors the rules PROPAGATE to (kq's ne[0] inherits the
                // KV width through mul_mat, kq_soft_max keeps it, rows' ne[1]
                // inherits out_ids' length through get_rows, states' ne[1]
                // inherits scopy's length the same way).
                //
                // Falsifiable: dropping any one entry -- kq is the one the
                // name-based table could not see -- makes the set differ and
                // this check fail.
                struct root_expect { const char * name; int which; int dim; };
                static const root_expect want[] = {
                    { "union_nkv_k",        0, 2 },
                    { "union_nkv_v",        0, 2 },
                    { "attn_inp_kq_mask",   0, 0 },
                    { "kq",                 0, 0 },
                    { "kq_soft_max",        0, 0 },
                    { "union_out_ids",      1, 0 },
                    { "rows",               1, 1 },
                    { "union_rs_copy",      3, 0 },
                    { "union_rs_copy_extra",3, 0 },
                    { "union_rs_copy_dst",  3, 1 },
                    { "states",             3, 1 },
                };
                for (const root_expect & w : want) {
                    int found = -1;
                    for (size_t i = 0; i < g.nodes.size(); ++i) {
                        if (std::strcmp(g.nodes[i]->name, w.name) == 0) {
                            found = (int) i;
                            break;
                        }
                    }
                    const int got = found < 0 ? -1
                            : (w.which == 0 ? table.excl[found].dim
                             : w.which == 1 ? table.excl[found].nout_dim
                             : w.which == 3 ? table.excl[found].nrs_dim
                             : table.excl[found].ntok_dim);
                    if (found < 0 || got != w.dim) {
                        std::fprintf(stderr,
                                "FAIL: width table root %s axis %d is %d at n_kv=%" PRId64
                                " (the analysis' rule set says %d)\n",
                                w.name, w.dim, got, p.n_kv, w.dim);
                        ok = false;
                    }
                }
                // every view node must skip the op-param copy of its offset
                for (size_t i = 0; i < g.nodes.size(); ++i) {
                    if (g.nodes[i]->view_src != nullptr && !table.excl[i].skip_view_data) {
                        std::fprintf(stderr, "FAIL: view %s does not skip its offset op param\n",
                                g.nodes[i]->name);
                        ok = false;
                    }
                }
                ++combos;
                synth_free(g);
            }
        }
    }
    std::fprintf(stderr, "NKV_RT_HASH_ARM ok=%d cases=%d widths=%zu\n", int(ok), combos, n_widths);
    std::fprintf(stderr, "NKV_RT_TRIPARAM ok=%d combos=%d roots=%d\n", int(ok), combos,
            int(ref_table.has_nkv) + int(ref_table.has_nout) + int(ref_table.has_nrs));

    // transposed V layout: n_kv moves to dimension 0, the hash must not care
    {
        synth_graph g;
        if (!synth_build(g, ref_p, "union_nkv_v", SYNTH_PLAIN)) {
            std::fprintf(stderr, "FAIL: synth build (transposed v)\n");
            ok = false;
        } else {
            // rewrite the V view as the transposed layout
            ggml_tensor * v = nullptr;
            for (ggml_tensor * t : g.nodes) {
                if (std::strcmp(t->name, "union_nkv_v") == 0) { v = t; break; }
            }
            if (v == nullptr) {
                std::fprintf(stderr, "FAIL: transposed v: no union_nkv_v node\n");
                ok = false;
            } else {
                // transposed: ne = (n_kv, n_head, head_v, ns); nb[1] > nb[2]
                v->ne[0] = ref_p.n_kv;
                v->ne[1] = 4;
                v->ne[2] = 64;
                v->ne[3] = 1;
                v->nb[0] = sizeof(float);
                v->nb[1] = v->nb[0] * (size_t) ref_p.n_kv;
                v->nb[2] = v->nb[1] * 4;
                v->nb[3] = v->nb[2] * 64;

                llama_union_width_table table;
                if (!synth_table(g, ref_p, table)) {
                    std::fprintf(stderr, "FAIL: transposed v: width table build\n");
                    ok = false;
                } else {
                    bool dim_ok = false;
                    for (const auto & e : table.excl) {
                        if (e.dim == 0) { dim_ok = true; }
                    }
                    if (!dim_ok) {
                        std::fprintf(stderr, "FAIL: transposed v: n_kv dimension not found in dimension 0\n");
                        ok = false;
                    }
                    uint64_t hash = 0;
                    if (!table.topology_hash(g.nodes, hash) || hash != ref_hash) {
                        std::fprintf(stderr, "FAIL: transposed v: topology hash moved\n");
                        ok = false;
                    }
                }
            }
            synth_free(g);
        }
    }

    // the recurrent head selects a view offset: a head change must not move
    // the hash, and the table must name the head axis
    {
        synth_params p = ref_p;
        p.rs_head = 3;
        synth_graph g;
        if (!synth_build(g, p, "union_nkv_v", SYNTH_PLAIN)) {
            std::fprintf(stderr, "FAIL: synth build (head)\n");
            ok = false;
        } else {
            llama_union_width_table table;
            uint64_t hash = 0;
            if (!synth_table(g, p, table)) {
                std::fprintf(stderr, "FAIL: head: width table build\n");
                ok = false;
            } else if (!table.has_rs_head) {
                std::fprintf(stderr, "FAIL: head: width table found no head root\n");
                ok = false;
            } else if (!table.topology_hash(g.nodes, hash) || hash != ref_hash) {
                std::fprintf(stderr, "FAIL: head: topology hash moved with the head\n");
                ok = false;
            }
            synth_free(g);
        }
    }

    // negative: every non-runtime difference must move the hash
    struct neg_case {
        const char *  name;
        synth_perturb perturb;
    };
    const neg_case negs[] = {
        { "extra_node",   SYNTH_EXTRA_NODE   },
        { "op_param",     SYNTH_OP_PARAM     },
        { "edge",         SYNTH_EDGE         },
        { "non_kv_shape", SYNTH_NON_KV_SHAPE },
        { "name",         SYNTH_NAME         },
        { "view_offset",  SYNTH_VIEW_OFFSET  },
    };
    for (const auto & n : negs) {
        synth_graph g;
        if (!synth_build(g, ref_p, "union_nkv_v", n.perturb)) {
            std::fprintf(stderr, "FAIL: synth build (%s)\n", n.name);
            ok = false;
            synth_free(g);
            continue;
        }
        llama_union_width_table table;
        uint64_t hash = 0;
        const bool built  = synth_table(g, ref_p, table);
        const bool hashed = built && table.topology_hash(g.nodes, hash);
        // a perturbation that adds a node may legitimately make the width
        // table refuse (the extra node is not a runtime root); in that case
        // the hash must still be computable through the reference table's
        // exclusions, extended to the longer node list
        if (!hashed) {
            llama_union_width_table any;
            any.excl = ref_table.excl;
            any.excl.resize(g.nodes.size());
            for (size_t i = 0; i < g.nodes.size(); ++i) {
                any.excl[i].node = (int32_t) i;
            }
            if (!any.topology_hash(g.nodes, hash)) {
                std::fprintf(stderr, "FAIL: %s: topology hash not computable\n", n.name);
                ok = false;
            } else if (hash == ref_hash) {
                std::fprintf(stderr, "FAIL: %s did not move the topology hash\n", n.name);
                ok = false;
            }
        } else if (hash == ref_hash) {
            std::fprintf(stderr, "FAIL: %s did not move the topology hash\n", n.name);
            ok = false;
        }
        synth_free(g);
    }

    // width table fail-closed rules
    {
        llama_union_width_table table;
        if (!synth_table(ref, ref_p, table)) {
            std::fprintf(stderr, "FAIL: fail-closed: table build\n");
            ok = false;
        } else {
            const uint32_t arena = (uint32_t) ref_p.n_kv;
            if (table.set_runtime_n_kv(0, arena)) {
                std::fprintf(stderr, "FAIL: set_runtime_n_kv(0) must fail closed\n");
                ok = false;
            }
            if (table.set_runtime_n_kv(arena + 1, arena)) {
                std::fprintf(stderr, "FAIL: set_runtime_n_kv above the arena must fail closed\n");
                ok = false;
            }
            if (table.runtime_n_kv != 0) {
                std::fprintf(stderr, "FAIL: a refused width must not be armed\n");
                ok = false;
            }
            if (table.effective_n_kv() != (uint32_t) ref_p.n_kv) {
                std::fprintf(stderr, "FAIL: unarmed table must report the build width\n");
                ok = false;
            }
            if (!table.set_runtime_n_kv(1, arena)) {
                std::fprintf(stderr, "FAIL: set_runtime_n_kv(1) must succeed\n");
                ok = false;
            }
            if (table.effective_n_kv() != 1) {
                std::fprintf(stderr, "FAIL: armed table must report the runtime width\n");
                ok = false;
            }
        }
    }

    // The arm's bound is the live arena, not the build width. The token lane
    // builds at the first live width (1) and a successful replan widens the
    // arena; the table must arm the widened width and refuse one above it.
    // This is the committed defect in miniature: build_n_kv == 1 while the
    // arena holds 4, and a build-width bound left runtime_n_kv at 0.
    {
        synth_params narrow_p = ref_p;
        narrow_p.n_kv = 1;
        synth_graph narrow;
        llama_union_width_table table;
        if (!synth_build(narrow, narrow_p, "union_nkv_v", SYNTH_PLAIN) ||
                !synth_table(narrow, narrow_p, table)) {
            std::fprintf(stderr, "FAIL: arena-bound: table build at n_kv=1\n");
            ok = false;
        } else {
            if (!table.set_runtime_n_kv(4, 4)) {
                std::fprintf(stderr, "FAIL: arena-bound: width 4 inside a 4-cell arena was refused\n");
                ok = false;
            } else if (table.effective_n_kv() != 4) {
                std::fprintf(stderr, "FAIL: arena-bound: armed width is %u, want 4\n",
                        table.effective_n_kv());
                ok = false;
            }
            if (table.set_runtime_n_kv(5, 4)) {
                std::fprintf(stderr, "FAIL: arena-bound: width 5 above a 4-cell arena was armed\n");
                ok = false;
            }
            if (table.runtime_n_kv != 0) {
                std::fprintf(stderr, "FAIL: arena-bound: a refused width must not be armed\n");
                ok = false;
            }
        }
        synth_free(narrow);
    }

    // a root whose shape disagrees with the recorded value must be refused
    {
        synth_params bad = ref_p;
        bad.n_kv += 7;
        llama_union_width_table table;
        if (synth_table(ref, bad, table)) {
            std::fprintf(stderr, "FAIL: width table accepted an n_kv that disagrees with the roots\n");
            ok = false;
        }
        bad = ref_p;
        bad.n_outputs += 3;
        if (synth_table(ref, bad, table)) {
            std::fprintf(stderr, "FAIL: width table accepted an n_outputs that disagrees with out_ids\n");
            ok = false;
        }
        bad = ref_p;
        bad.n_rs += 5;
        if (synth_table(ref, bad, table)) {
            std::fprintf(stderr, "FAIL: width table accepted an n_rs that disagrees with the copy roots\n");
            ok = false;
        }
    }

    synth_free(ref);
    return ok;
}

// ---------------------------------------------------------------------------
// part 2: the live llama_decode path
// ---------------------------------------------------------------------------

struct live_model {
    llama_model * model = nullptr;
    int           vocab = 0;
};

static bool live_open(const char * model_path, live_model & m) {
    llama_backend_init();

    auto mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = { nullptr };
    mp.devices      = devices;
    mp.n_gpu_layers = 0;

    m.model = llama_model_load_from_file(model_path, mp);
    if (m.model == nullptr) {
        std::fprintf(stderr, "FAIL: model load: %s\n", model_path);
        llama_backend_free();
        return false;
    }
    m.vocab = llama_vocab_n_tokens(llama_model_get_vocab(m.model));
    if (m.vocab <= 3) {
        std::fprintf(stderr, "FAIL: vocab=%d\n", m.vocab);
        llama_model_free(m.model);
        m.model = nullptr;
        llama_backend_free();
        return false;
    }
    return true;
}

static void live_close(live_model & m) {
    if (m.model != nullptr) {
        llama_model_free(m.model);
        m.model = nullptr;
    }
    llama_backend_free();
}

static bool part2(const live_model & m, int n_steps) {
    const int vocab = m.vocab;

    // Environment audit for this part. Every variable below is read by the
    // llama_context CONSTRUCTOR (llama-context.cpp:471 for
    // LLAMA_GRAPH_REUSE_DISABLE, :508 for GGML_TP5_UNION_BINDER, :517 for
    // LLAMA_EXPERIMENTAL_VARLEN_NTOK), never by the runner. Nothing here is
    // unset after a context was built, so no arm can be made identical to
    // another by an unset.
    //   GGML_TP5_UNION_BINDER=1        the binder arm: the runtime width, the
    //                                  topology hash and the mask observation
    //                                  only exist on a recorded graph.
    //   LLAMA_EXPERIMENTAL_VARLEN_NTOK unset: the token lane would replan the
    //                                  arena per width and is a different
    //                                  contract; this part tests the n_kv lane.
    //   LLAMA_GRAPH_REUSE_DISABLE      unset: the binder arm must REUSE the
    //                                  graph; with it set the arm would be the
    //                                  stock arm.
    setenv("GGML_TP5_UNION_BINDER", "1", 1);
    unsetenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK");
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = (uint32_t) (n_steps + 8);
    cp.n_batch   = 1;
    cp.n_ubatch  = 1;
    cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_context * ctx = llama_init_from_model(m.model, cp);
    if (ctx == nullptr) {
        std::fprintf(stderr, "FAIL: context init\n");
        return false;
    }

    bool ok = true;

    uint64_t ref_hash    = 0;
    bool     have_ref    = false;
    uint32_t build_width = 0;
    int64_t  min_width   = INT64_MAX;
    int64_t  max_width   = 0;
    int      steps_done  = 0;

    // A bind is a step whose executed width is below the build envelope. The
    // test must see real binds, not only the first build: a hash captured once
    // at build time and never re-taken would look stable for the wrong reason.
    //
    // The record is built at the context's KV capacity (record_cap = n_ctx,
    // 308 here), not at the first ubatch width, so every step executes below
    // the envelope and binds. n_ctx = n_steps + 8 leaves eight tokens of
    // headroom; the last used span is n_steps (300), so no step reaches the
    // envelope and the run cannot degenerate into "build once at the final
    // width".
    int      bind_steps  = 0;
    uint32_t last_bound  = 0;
    int      bound_moves = 0;
    std::vector<uint32_t> distinct_bound;

    // Measured CPU attention work. The ordered softmax adds the loop bound it
    // executed (ne00, the runtime width the bind patched in), so these deltas
    // are a measurement of what ran, not a restatement of the width the test
    // asked for. This lane has no padded work to report: the bind narrowed
    // the mask and the K/V views to the runtime width, so the executed length
    // and the extent it covers are the same value. `no_pad_steps` counts the
    // steps where that identity held; `pad_steps` counts steps that reported
    // padded work (a failure on this lane, kept as a counter for the line).
    uint64_t prev_live = 0;
    uint64_t prev_pad  = 0;
    uint64_t prev_rows = 0;
    int      no_pad_steps = 0;
    int      pad_steps    = 0;

    std::vector<llama_token> tokens(1);

    for (int step = 0; step < n_steps; ++step) {
        tokens[0] = (llama_token) (1 + (step * 7) % (vocab - 1));

        llama_batch batch = llama_batch_init(1, 0, 1);
        if (!llama_batch_is_complete(&batch, 1, false, true)) {
            llama_batch_free(batch);
            batch = llama_batch{};
            ok = false;
            break;
        }
        batch.n_tokens = 1;
        batch.token[0]    = tokens[0];
        batch.pos[0]      = (llama_pos) step;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]   = 1;

        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            std::fprintf(stderr, "FAIL: llama_decode rc=%d at step=%d\n", rc, step);
            ok = false;
            break;
        }
        ++steps_done;

        llama_nkv_runtime_stats st = {};
        if (!llama_get_nkv_runtime_stats(ctx, &st)) {
            std::fprintf(stderr, "FAIL: no n_kv runtime stats at step=%d (binder off or no record)\n", step);
            ok = false;
            break;
        }

        // The work this step actually did, as deltas of the measured counters.
        {
            const uint64_t live_d = st.attn_live_elems - prev_live;
            const uint64_t pad_d  = st.attn_pad_elems  - prev_pad;
            const uint64_t rows_d = st.attn_rows       - prev_rows;
            prev_live = st.attn_live_elems;
            prev_pad  = st.attn_pad_elems;
            prev_rows = st.attn_rows;

            if (live_d == 0 || pad_d == 0 || rows_d == 0) {
                std::fprintf(stderr,
                        "FAIL: no CPU attention work measured at step=%d (live=%" PRIu64
                        " pad=%" PRIu64 " rows=%" PRIu64 ")\n", step, live_d, pad_d, rows_d);
                ok = false;
                break;
            }
            // Self-consistency: every executed row reduced exactly the live
            // length. This fails if the two counters drift, which is the only
            // way a counter can lie about its own work.
            if (rows_d * (uint64_t) st.runtime_n_kv != live_d) {
                std::fprintf(stderr,
                        "FAIL: attention rows x live != live elems at step=%d "
                        "(rows=%" PRIu64 " live_n_kv=%u elems=%" PRIu64 ")\n",
                        step, rows_d, st.runtime_n_kv, live_d);
                ok = false;
                break;
            }
            // This lane has no padded work: the bind narrowed the mask and
            // the K/V views to the runtime width, so the ordered softmax's
            // executed length IS the extent it covers. live > pad is
            // impossible; live < pad would mean a padded tensor survived the
            // bind, which this lane must not do. The build envelope is still
            // asserted not to move below, so this is not a weaker claim: the
            // kernel really ran at the runtime width, and the graph really
            // stayed at the envelope.
            if (live_d != pad_d) {
                std::fprintf(stderr,
                        "FAIL: the runtime lane reported padded work at step=%d "
                        "(live=%" PRIu64 " pad=%" PRIu64 " rows=%" PRIu64 ")\n",
                        step, live_d, pad_d, rows_d);
                ok = false;
                break;
            }
            ++no_pad_steps;
        }
        if (!llama_get_nkv_runtime_stats(ctx, &st)) {
            std::fprintf(stderr, "FAIL: no n_kv runtime stats at step=%d (binder off or no record)\n", step);
            ok = false;
            break;
        }
        if (!st.topology_hashed) {
            std::fprintf(stderr, "FAIL: topology not hashed at step=%d\n", step);
            ok = false;
            break;
        }
        if (!st.has_nkv_dims) {
            std::fprintf(stderr, "FAIL: width table found no n_kv dimension at step=%d\n", step);
            ok = false;
            break;
        }
        if (!have_ref) {
            ref_hash    = st.topology_hash;
            build_width = st.build_n_kv;
            have_ref    = true;
        } else {
            if (st.topology_hash != ref_hash) {
                std::fprintf(stderr,
                        "FAIL: topology hash moved at step=%d (ref=%016" PRIx64 " got=%016" PRIx64 ")\n",
                        step, ref_hash, st.topology_hash);
                ok = false;
                break;
            }
            if (st.build_n_kv != build_width) {
                std::fprintf(stderr, "FAIL: build width moved at step=%d (%u -> %u): the graph was rebuilt\n",
                        step, build_width, st.build_n_kv);
                ok = false;
                break;
            }
        }
        // Step 0 executes at the first used span (1); after it the true used
        // span is step + 1. Assert both explicitly: including step 0 in the
        // range below is not a substitute for pinning its width.
        if ((int64_t) st.runtime_n_kv != (int64_t) step + 1) {
            std::fprintf(stderr, "FAIL: runtime_n_kv=%u at step=%d (want %d)\n",
                    st.runtime_n_kv, step, step + 1);
            ok = false;
            break;
        }
        if ((int64_t) st.runtime_n_kv < min_width) min_width = (int64_t) st.runtime_n_kv;
        if ((int64_t) st.runtime_n_kv > max_width) max_width = (int64_t) st.runtime_n_kv;

        // The mask observation. This runs on EVERY step, including step 0:
        // a step0 that cannot fail is the defect this assertion exists to
        // prevent. The fill must have consumed exactly the width the graph
        // reports it executes at, and it must not have refused.
        if (st.mask_refused != 0) {
            std::fprintf(stderr, "FAIL: the mask fill refused at step=%d (refused=%u)\n",
                    step, st.mask_refused);
            ok = false;
            break;
        }
        if (st.mask_filled == 0) {
            std::fprintf(stderr, "FAIL: no KQ mask was filled at step=%d\n", step);
            ok = false;
            break;
        }
        if (st.mask_n_kv != st.runtime_n_kv) {
            std::fprintf(stderr,
                    "FAIL: the mask was filled at width %u, the graph executes at %u (step=%d)\n",
                    st.mask_n_kv, st.runtime_n_kv, step);
            ok = false;
            break;
        }
        // The read-back is the only independent observation of the mask: the
        // fill width and the executed width both come from the binder. A fill
        // that was never compared proves nothing, so each of these is a FAIL,
        // not a skip.
        if (st.mask_row_checked == 0) {
            std::fprintf(stderr,
                    "FAIL: the mask row was not read back at step=%d (filled=%u checked=0)\n",
                    step, st.mask_filled);
            ok = false;
            break;
        }
        if (st.mask_row_ok != st.mask_row_checked) {
            std::fprintf(stderr,
                    "FAIL: the mask row disagrees with the KV cells at step=%d (ok=%u checked=%u)\n",
                    step, st.mask_row_ok, st.mask_row_checked);
            ok = false;
            break;
        }
        if (st.mask_row_refused != 0) {
            std::fprintf(stderr,
                    "FAIL: the mask read-back refused to model %u row(s) at step=%d\n",
                    st.mask_row_refused, step);
            ok = false;
            break;
        }
        if (st.mask_rows_all != 1) {
            std::fprintf(stderr,
                    "FAIL: only the last mask row was compared at step=%d (n_tokens=1, mask_rows_all=0)\n",
                    step);
            ok = false;
            break;
        }
        // a bind: the graph executes below the width it was built at
        if (st.bound_n_kv != 0 && st.bound_n_kv < st.build_n_kv) {
            ++bind_steps;
        }
        if (st.bound_n_kv != last_bound) {
            if (last_bound != 0) {
                ++bound_moves;
            }
            last_bound = st.bound_n_kv;
            bool seen = false;
            for (uint32_t v : distinct_bound) {
                if (v == st.bound_n_kv) { seen = true; break; }
            }
            if (!seen) {
                distinct_bound.push_back(st.bound_n_kv);
            }
        }
    }

    // the widths must actually cross the 256-cell boundary
    if (ok) {
        if (min_width >= 256 || max_width <= 256) {
            std::fprintf(stderr,
                    "FAIL: the run did not cross the 256 boundary (min=%" PRId64 " max=%" PRId64 ")\n",
                    min_width, max_width);
            ok = false;
        }
    }
    // the hash must have been re-taken over real binds, not only over the
    // first build: at least one step executed below the envelope, the bound
    // width moved between steps, and more than one bound width was seen
    if (ok) {
        if (bind_steps == 0) {
            std::fprintf(stderr, "FAIL: no step executed below the build envelope: no bind happened\n");
            ok = false;
        }
        if (bound_moves == 0) {
            std::fprintf(stderr, "FAIL: bound_n_kv never moved between steps\n");
            ok = false;
        }
        if (distinct_bound.size() < 2) {
            std::fprintf(stderr, "FAIL: only %zu distinct bound_n_kv value(s) seen\n",
                    distinct_bound.size());
            ok = false;
        }
    }

    // Every measured step must have reported the no-pad identity: the
    // ordered reduction executed exactly the runtime width per row and the
    // extent it covered was that same width. A step that reported padded
    // work was already failed inside the loop; this is the count that makes
    // the finding visible on the summary line.
    if (ok) {
        if (no_pad_steps != steps_done) {
            std::fprintf(stderr,
                    "FAIL: only %d of %d steps reported the no-pad identity\n",
                    no_pad_steps, steps_done);
            ok = false;
        }
        pad_steps = steps_done - no_pad_steps;
    }

    std::fprintf(stderr,
            "NKV_RT_LIVE ok=%d steps=%d hash=%016" PRIx64 " min_width=%" PRId64
            " max_width=%" PRId64 " bind_steps=%d distinct_bound=%zu\n",
            int(ok), steps_done, ref_hash, min_width, max_width,
            bind_steps, distinct_bound.size());
    // The line name is historical (it named the TRUE_LEN_REDUCE lane). On
    // this ordered lane it reports the no-pad identity: every measured step
    // executed its runtime width and covered exactly that extent.
    std::fprintf(stderr, "NKV_RT_TRUELEN ok=%d steps=%d no_pad=%d pad_steps=%d\n",
            int(ok), steps_done, no_pad_steps, pad_steps);

    llama_free(ctx);
    return ok;
}

// ---------------------------------------------------------------------------
// part 4: the mask path MUST be able to fail
// ---------------------------------------------------------------------------
//
// The one-token decode part 4 drives. It is deliberately the smallest form of
// the binder path: one token, one output row, position 0. The cross matrix
// that used to share these helpers moved to tests/test-union-four-param-matrix.cpp,
// which owns the four-parameter comparison; this part owns only the refusal.
struct tri_cell {
    int      n_tokens  = 1;
    uint32_t n_outputs = 1;
    int64_t  pos0      = 0;
};

static llama_context * tri_context(const live_model & m, int n_ctx) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx         = (uint32_t) n_ctx;
    cp.n_batch       = 4;
    cp.n_ubatch      = 4;
    cp.n_seq_max     = 1;
    cp.n_outputs_max = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    return llama_init_from_model(m.model, cp);
}

static int tri_decode(llama_context * ctx, const tri_cell & c, int vocab, std::vector<float> & logits_out) {
    logits_out.assign((size_t) c.n_outputs * (size_t) vocab, 0.0f);

    llama_batch batch = llama_batch_init(c.n_tokens, 0, 1);
    if (!llama_batch_is_complete(&batch, c.n_tokens, false, true)) {
        llama_batch_free(batch);
        batch = llama_batch{};
        return -1;
    }
    batch.n_tokens = c.n_tokens;
    for (int i = 0; i < c.n_tokens; ++i) {
        batch.token[i]     = (llama_token) (1 + (c.pos0 + i) * 7 % (vocab - 1));
        batch.pos[i]       = (llama_pos) (c.pos0 + i);
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i >= c.n_tokens - (int) c.n_outputs) ? 1 : 0;
    }

    const int rc = llama_decode(ctx, batch);
    if (rc != 0) {
        llama_batch_free(batch);
        return rc;
    }
    for (uint32_t r = 0; r < c.n_outputs; ++r) {
        const int32_t row = c.n_tokens - (int) c.n_outputs + (int) r;
        const float * src = llama_get_logits_ith(ctx, row);
        if (src == nullptr) {
            llama_batch_free(batch);
            return -1;
        }
        std::memcpy(logits_out.data() + (size_t) r * (size_t) vocab, src,
                sizeof(float) * (size_t) vocab);
    }
    llama_batch_free(batch);
    return 0;
}

// ---------------------------------------------------------------------------
// part 4: the mask path MUST be able to fail
// ---------------------------------------------------------------------------
//
// A step0 that cannot fail is a defect. Part 4 proves the refusal path is
// live: it runs one decode with LLAMA_NKV_MASK_OVERRIDE set to a width above
// the mask's own ne[0] and requires the process to abort with the named
// message. GGML_ABORT kills the process, so the case runs in a child.
//
// The child is this same binary with --mask-negative, so the injection goes
// through the same code path the binder uses (the runtime n_kv input), not a
// side door.

// The child of part 4: one decode with the override armed. It must abort.
static int mask_negative_child(const char * model_path) {
    setenv("GGML_TP5_UNION_BINDER", "1", 1);
    unsetenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK");
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");

    live_model m;
    if (!live_open(model_path, m)) {
        return 3;
    }
    llama_context * ctx = tri_context(m, 16);
    if (ctx == nullptr) {
        live_close(m);
        return 4;
    }
    tri_cell c;
    c.n_tokens  = 1;
    c.n_outputs = 1;
    c.pos0      = 0;
    std::vector<float> logits;
    (void) tri_decode(ctx, c, m.vocab, logits);
    // Reaching this line means the guard did not fire: the negative case
    // failed, and the exit code says so.
    llama_free(ctx);
    live_close(m);
    std::fprintf(stderr, "FAIL: the mask guard did not fire on an over-wide runtime n_kv\n");
    return 5;
}

static bool part4(const char * self_path, const char * model_path) {
    bool ok = true;

    // (a) positive: one decode with the binder on; the mask is filled at the
    // width the graph executes at, and nothing refused.
    {
        setenv("GGML_TP5_UNION_BINDER", "1", 1);
        unsetenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK");
        unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
        unsetenv("LLAMA_NKV_MASK_OVERRIDE");

        live_model m;
        if (!live_open(model_path, m)) {
            return false;
        }
        llama_context * ctx = tri_context(m, 16);
        bool pos_ok = ctx != nullptr;
        if (pos_ok) {
            tri_cell c;
            c.n_tokens  = 1;
            c.n_outputs = 1;
            c.pos0      = 0;
            std::vector<float> logits;
            pos_ok = tri_decode(ctx, c, m.vocab, logits) == 0;
            llama_nkv_runtime_stats st = {};
            pos_ok = pos_ok && llama_get_nkv_runtime_stats(ctx, &st);
            if (pos_ok) {
                if (st.mask_filled == 0 || st.mask_refused != 0 || st.mask_n_kv != st.runtime_n_kv) {
                    std::fprintf(stderr,
                            "FAIL: mask positive: filled=%u refused=%u mask_n_kv=%u runtime_n_kv=%u\n",
                            st.mask_filled, st.mask_refused, st.mask_n_kv, st.runtime_n_kv);
                    pos_ok = false;
                }
                // The read-back is the only independent observation of the
                // mask. A fill that was never compared proves nothing.
                if (st.mask_row_checked == 0 || st.mask_row_ok != st.mask_row_checked ||
                        st.mask_row_refused != 0 || st.mask_rows_all != 1) {
                    std::fprintf(stderr,
                            "FAIL: mask positive: read-back ok=%u checked=%u refused=%u rows_all=%u\n",
                            st.mask_row_ok, st.mask_row_checked, st.mask_row_refused, st.mask_rows_all);
                    pos_ok = false;
                }
            } else {
                std::fprintf(stderr, "FAIL: mask positive: decode or stats failed\n");
            }
            llama_free(ctx);
        } else {
            std::fprintf(stderr, "FAIL: mask positive: context init\n");
        }
        live_close(m);
        ok = ok && pos_ok;
    }

    // (b) negative: the same decode with a width above the mask must abort.
    // The child owns its environment; the parent never arms the override for
    // its own contexts.
    {
        std::string cmd = std::string("LLAMA_NKV_MASK_OVERRIDE=1000000 '") + self_path +
                          "' --mask-negative '" + model_path + "' 2>&1";
        FILE * pipe = popen(cmd.c_str(), "r");
        if (pipe == nullptr) {
            std::fprintf(stderr, "FAIL: mask negative: cannot spawn the child\n");
            return false;
        }
        std::string out;
        char buf[512];
        size_t nread;
        while ((nread = fread(buf, 1, sizeof(buf), pipe)) > 0) {
            out.append(buf, nread);
        }
        const int rc = pclose(pipe);
        const bool exited_nonzero = rc != 0;
        const bool named = out.find("runtime n_kv exceeds the KQ mask width") != std::string::npos;
        if (!exited_nonzero || !named) {
            std::fprintf(stderr,
                    "FAIL: mask negative did not abort as required (rc=%d, message=%s)\n",
                    rc, named ? "present" : "MISSING");
            ok = false;
        }
    }

    std::fprintf(stderr, "TRIPARAM_MASK_NEG ok=%d\n", int(ok));
    return ok;
}

int main(int argc, char ** argv) {
    // --hash-only runs part 1 alone: no model, no backend, no allocation.
    // It is the registration that always exists (test-nkv-runtime-param-hash).
    if (argc == 2 && std::strcmp(argv[1], "--hash-only") == 0) {
        const bool ok = part1();
        std::fprintf(stderr, "NKV_RT_END ok=%d\n", int(ok));
        return ok ? 0 : 1;
    }

    // --mask-negative is the child of part 4: one decode with the override
    // armed. It must abort; reaching its return means the guard did not fire.
    if (argc == 3 && std::strcmp(argv[1], "--mask-negative") == 0) {
        return mask_negative_child(argv[2]);
    }

    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s model.gguf [n_steps]\n"
                             "       %s --mask-negative model.gguf\n"
                             "       %s --hash-only\n", argv[0], argv[0], argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const int    n_steps    = argc == 3 ? std::atoi(argv[2]) : 300;
    if (n_steps <= 256) {
        std::fprintf(stderr, "FAIL: n_steps=%d must be > 256 to cross the 256-cell boundary\n", n_steps);
        return 2;
    }
    if (model_path == nullptr || model_path[0] == '\0') {
        std::fprintf(stderr, "FAIL: empty model path\n");
        return 2;
    }

    live_model m;
    if (!live_open(model_path, m)) {
        return 1;
    }

    // Every part runs even after one fails, so one invocation reports every
    // failure instead of only the first. The exit code is the conjunction.
    bool ok = true;
    ok &= part1();
    ok &= part2(m, n_steps);
    // Part 4 re-opens the model in its own contexts (it must not inherit this
    // process's live context state), so it takes the path, not `m`.
    ok &= part4(argv[0], model_path);

    live_close(m);

    std::fprintf(stderr, "NKV_RT_END ok=%d\n", int(ok));
    return ok ? 0 : 1;
}
