#pragma once

//
// llama-union-width-table.h
//
// Per-graph record of WHERE the runtime parameters live.
//
// The union binder patches runtime-coupled tensor shapes in place, so a single
// built graph serves every combination of n_kv, n_outputs and rs_z up to its
// allocation envelope. Two facts make that claim checkable rather than
// asserted:
//
//   1. which dimension of which node carries a runtime parameter (the topology
//      hash skips exactly those dimensions, see llama-union-topology.h);
//   2. what each parameter will be at execution time. `runtime_n_kv` is the
//      width the consumer must read: it is set from the KV cache's true used
//      span before every compute and is never a build-time constant.
//
// This table is not a second shape-patch mechanism. The patch itself stays in
// llama_context::union_rebind_apply (llama-context.cpp); the table is the
// audit surface: build it once per graph from the tensor names the
// construction sites set, then hash the topology through it.
//
// The runtime parameters and their roots (the same names the symbolic shape
// analysis keys on, llama-context.cpp union_shape_analyze):
//
//   n_kv      union_nkv_k       llama_kv_cache::get_k -> view_4d(head_k, n_head_kv, n_kv, ns)
//             union_nkv_v       llama_kv_cache::get_v -> view_4d(head_v, n_head_kv, n_kv, ns)
//                                                       or view_4d(n_kv, n_head_kv, head_v, ns)
//             attn_inp_kq_mask  build_attn_inp_kq_mask -> new_tensor_4d(n_kv, n_tokens/n_stream, 1, n_stream)
//   n_outputs union_out_ids     build_inp_out_ids -> new_tensor_1d(I32, n_outputs)
//                               (a graph-owned input leaf: a canonical member
//                               through the input-leaf rule of
//                               llama_topology_canonical_nodes)
//   rs_z      union_rs_copy     build_rs_inp_impl -> new_tensor_1d(I32, n_rs)
//             union_rs_copy_extra   view_1d(s_copy, n_rs - n_seqs, ...)
//             union_rs_copy_dst     view_2d(s, state_size, n_rs - n_seqs, nb1, (head + n_seqs)*nb1)
//             union_rs_head_view    view_1d(r_l, ..., head * row)
//   n_tokens  attn_inp_kq_mask  the same mask, dimension 1 (n_tokens/n_stream)
//             union_inp_tokens  build_inp_embd -> new_tensor_1d(I32, n_tokens)
//             union_inp_pos     build_inp_pos  -> new_tensor_1d(I32, n_tokens*n_pos_per_embd)
//             union_inp_k_idxs  llama_kv_cache::build_input_k_idxs -> new_tensor_1d(I64, n_tokens)
//             union_inp_v_idxs  llama_kv_cache::build_input_v_idxs -> new_tensor_1d(I64, n_tokens)
//                               or [n_tokens * n_embd_v_gqa_max] when V is transposed
//
// The n_tokens lane is recorded only when the caller passes a nonzero token
// width (the token lane is armed). On every other lane the token dimensions
// stay hashed exactly as before, so a token-width change still moves the hash
// and the bind is refused.
//
// The four token inputs are checked at their EXACT length, `n_tokens * factor`,
// where the factor is 1 for the tokens and K-index inputs, n_pos_per_embd for
// the position input, and n_embd_v_gqa_max for a transposed V index. The shape
// analysis derives the same factor from the tensor
// (llama-context.cpp union_shape_analyze: d_ntok = ne / base_ntok), so the
// patch moves the dimension by `factor * delta`. A multiple-of check would let
// a wrong layout pass, because [2 * n_tokens] and [n_embd_v_gqa_max * n_tokens]
// are then indistinguishable.
//
// The rs_z zero row itself is a runtime INDEX input (union_rs_zero_index), not
// a shape: it is a constant [1] tensor at every row, so it needs no table
// entry and cannot move the hash.
//
// Fail-closed rules:
//   - a root whose shape disagrees with the recorded parameter value makes
//     build() return false;
//   - an armed token lane (nonzero token width) with no token root in the
//     graph makes build() return false: the table cannot exclude a dimension
//     it did not find, and the bind would refuse itself;
//   - runtime_n_kv == 0 means "no runtime width is armed"; consumers must then
//     fall back to the width baked into the tensor, never to a guessed value.
//   - set_runtime_n_kv() refuses 0 and any width above the live arena width
//     the caller supplies; the arena, not the build width, is the allocation
//     that backs the patched shapes.

#include "ggml.h"
#include "llama-topology-hash.h"   // LLAMA_TOKEN_LANE_ROOTS: the roots this table marks ntok_dim
#include "llama-union-topology.h"

#include <cstdint>
#include <cstring>
#include <vector>

struct llama_union_width_table {
    // node index -> the dimensions that carry a runtime parameter (-1: none),
    // parallel to the recorded tensor list
    std::vector<llama_union_topology_excl> excl;

    // n_kv: the width the graph was built at (allocation envelope or exact
    // build) and the width it must execute at (0 = not armed)
    uint32_t build_n_kv   = 0;
    uint32_t runtime_n_kv = 0;

    // n_outputs: the out_ids row count the graph was built at
    uint32_t build_n_outputs = 0;

    // rs_z lane: the recurrent state length and head the graph was built at
    uint32_t build_n_rs   = 0;
    uint32_t build_rs_head = 0;

    // n_tokens: the token width the graph was built at (0 = the token lane is
    // not recorded for this graph)
    uint32_t build_n_tokens = 0;

    // The token-lane roots this table actually recorded, in the order it found
    // them. LLAMA_TOKEN_LANE_ROOTS is the superset the key's exclusion names;
    // a graph need not carry every root (a model without a transposed V index
    // carries one of the two layouts), so this is what the key's exclusion
    // must cover, not what it must equal. The matrix asserts
    // `llama_topology_key_excl::token_lane().token_lane_roots_match(recorded)`
    // on a graph that carries all four.
    std::vector<const char *> ntok_roots;

    // true when at least one node carries each runtime parameter
    bool has_nkv     = false;
    bool has_nout    = false;
    bool has_nrs     = false;
    bool has_rs_head = false;
    bool has_ntok    = false;

    void clear() {
        excl.clear();
        ntok_roots.clear();
        build_n_kv      = 0;
        runtime_n_kv    = 0;
        build_n_outputs = 0;
        build_n_rs      = 0;
        build_rs_head   = 0;
        build_n_tokens  = 0;
        has_nkv         = false;
        has_nout        = false;
        has_nrs         = false;
        has_rs_head     = false;
        has_ntok        = false;
    }

    // Record where the runtime parameters live in `nodes` (graph order, source
    // before consumer). build() takes eight parameters: the node list, the
    // five shape values the graph was constructed at (n_kv, n_outputs, n_rs,
    // rs_head, n_tokens) and the two token-lane factors (n_pos_per_embd,
    // n_embd_v_gqa_max). A zero shape value means "this graph has no such
    // root", not "check for zero": a root that is present must agree with its
    // value.
    //
    // build_n_tokens_ is nonzero only on the token lane. When it is zero the
    // token dimensions stay hashed as fixed shapes, which is the behaviour of
    // every other lane.
    //
    // Two token inputs are longer than the token width by a constant factor:
    // the position input is [n_tokens * n_pos_per_embd] and the transposed V
    // index input is [n_tokens * n_embd_v_gqa_max]. The caller passes both
    // factors so this table can check the exact length. The shape analysis
    // derives the same factor from the tensor itself
    // (llama-context.cpp union_shape_analyze, d_ntok = ne / base_ntok), so a
    // length that is not exactly `n_tokens * factor` is not the tensor this
    // table believes it is.

    // True when `name` is one of the token lane's roots. The list is
    // LLAMA_TOKEN_LANE_ROOTS (llama-topology-hash.h), which the single graph
    // cache key's exclusion also reads: the table and the key must name the
    // same set, or the key folds a dimension the bind patches and the bind
    // refuses itself. See llama_topology_key_excl::token_lane().
    static bool is_token_lane_root(const char * name) {
        for (int i = 0; i < LLAMA_TOKEN_LANE_ROOTS_N; ++i) {
            if (std::strcmp(name, LLAMA_TOKEN_LANE_ROOTS[i]) == 0) {
                return true;
            }
        }
        return false;
    }

    // The name-only form: the table excludes exactly the dimensions its
    // name-based records name. This is the whole exclusion set for a caller
    // that has no shape map (the model-free fixtures that pin the name rules),
    // and it is the fail-closed form for a graph whose derived tensors are
    // hashed: the hash then moves with a runtime parameter and the bind
    // refuses itself instead of patching a graph it cannot recognise. The
    // record build passes the shape map's coefficients through the other
    // overload; see union_width_runtime_dims().
    bool build(const std::vector<ggml_tensor *> & nodes,
               uint32_t build_n_kv_,
               uint32_t build_n_outputs_,
               uint32_t build_n_rs_,
               uint32_t build_rs_head_,
               uint32_t build_n_tokens_,
               uint32_t n_pos_per_embd,
               uint32_t n_embd_v_gqa_max) {
        return build(nodes, {}, build_n_kv_, build_n_outputs_, build_n_rs_,
                     build_rs_head_, build_n_tokens_, n_pos_per_embd, n_embd_v_gqa_max);
    }

    // `runtime_dims` is the set of dimensions the BINDER patches, resolved to
    // positions in `nodes` (see union_width_runtime_dims() in
    // llama-nout-state.h). It is the exclusion set; the name-based records
    // below are validation only. An empty list is legal and means "no runtime
    // dimension is known": the hash then sees every shape and the bind refuses
    // itself, which is the fail-closed direction.
    bool build(const std::vector<ggml_tensor *> & nodes,
               const std::vector<llama_union_topology_excl> & runtime_dims,
               uint32_t build_n_kv_,
               uint32_t build_n_outputs_,
               uint32_t build_n_rs_,
               uint32_t build_rs_head_,
               uint32_t build_n_tokens_,
               uint32_t n_pos_per_embd,
               uint32_t n_embd_v_gqa_max) {
        // Every failure path leaves an EMPTY table. A half-built table would
        // keep excl sized and the build values set while has_nkv stays false,
        // and topology_hash() would then succeed over partial data; the
        // invariant belongs to this type, not to each caller.
        auto fail = [this]() {
            clear();
            return false;
        };

        clear();
        build_n_kv      = build_n_kv_;
        build_n_outputs = build_n_outputs_;
        build_n_rs      = build_n_rs_;
        build_rs_head   = build_rs_head_;
        build_n_tokens  = build_n_tokens_;
        if (build_n_kv == 0) {
            return fail();
        }

        excl.resize(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) {
            excl[i].node = (int32_t) i;

            const ggml_tensor * t = nodes[i];
            if (t == nullptr) {
                return fail();
            }
            const char * name = t->name;
            if (std::strcmp(name, "union_nkv_k") == 0) {
                // get_k: n_kv is dimension 2
                if (t->ne[2] != (int64_t) build_n_kv) {
                    return fail();
                }
                excl[i].dim = 2;
                has_nkv = true;
            } else if (std::strcmp(name, "union_nkv_v") == 0) {
                // get_v: non-transposed view keeps n_kv in dimension 2,
                // the transposed view puts it in dimension 0. The strides
                // decide which layout this tensor is (same rule as the
                // shape analysis).
                const int dim = t->nb[1] > t->nb[2] ? 0 : 2;
                if (t->ne[dim] != (int64_t) build_n_kv) {
                    return fail();
                }
                excl[i].dim = dim;
                has_nkv = true;
            } else if (std::strcmp(name, "attn_inp_kq_mask") == 0) {
                // build_attn_inp_kq_mask: n_kv is dimension 0, the token width
                // is dimension 1 (n_tokens/n_stream; the token lane is
                // single-stream, so n_stream is 1)
                if (t->ne[0] != (int64_t) build_n_kv) {
                    return fail();
                }
                excl[i].dim = 0;
                has_nkv = true;
                if (build_n_tokens != 0) {
                    if (t->ne[1] != (int64_t) build_n_tokens) {
                        return fail();
                    }
                    excl[i].ntok_dim = 1;
                    has_ntok = true;
                }
            } else if (std::strcmp(name, "union_out_ids") == 0) {
                // build_inp_out_ids: the gather width is the row count. The
                // tensor is a graph-owned INPUT leaf, so it is a member of the
                // canonical set only because llama_topology_canonical_nodes()
                // admits an input-leaf source of a fixed node; without that
                // rule this branch never fired and has_nout stayed false on a
                // graph whose shape map had the root all along.
                if (t->ne[0] != (int64_t) build_n_outputs) {
                    return fail();
                }
                excl[i].nout_dim = 0;
                has_nout = true;
            } else if (std::strcmp(name, "union_rs_copy") == 0) {
                // build_rs_inp_impl: I32 [n_rs]
                if (t->ne[0] != (int64_t) build_n_rs) {
                    return fail();
                }
                excl[i].nrs_dim = 0;
                has_nrs = true;
            } else if (std::strcmp(name, "union_rs_copy_extra") == 0) {
                // view_1d(s_copy, n_rs - n_seqs, n_seqs * nb[0]): the length
                // lives in ne[0] and the n_seqs prefix in the view offset
                if (t->ne[0] > (int64_t) build_n_rs) {
                    return fail();
                }
                excl[i].nrs_dim = 0;
                has_nrs = true;
            } else if (std::strcmp(name, "union_rs_copy_dst") == 0) {
                // view_2d(s, state_size, n_rs - n_seqs, nb1, (head + n_seqs)*nb1):
                // the length is ne[1], the head is the view offset
                if (t->ne[1] > (int64_t) build_n_rs) {
                    return fail();
                }
                excl[i].nrs_dim = 1;
                has_nrs = true;
            } else if (std::strcmp(name, "union_rs_head_view") == 0) {
                // model state-store view: constant shape, the head selects the
                // base slot through the view offset
                excl[i].head_dim = 0;
                has_rs_head = true;
            } else if (build_n_tokens != 0 && is_token_lane_root(name)) {
                // The token-width input tensors. Each is 1-D and its length is
                // exactly the token width times a constant factor:
                //   union_inp_tokens  [n_tokens]                    factor 1
                //   union_inp_pos     [n_tokens * n_pos_per_embd]   factor n_pos_per_embd
                //   union_inp_k_idxs  [n_tokens]                    factor 1
                //   union_inp_v_idxs  [n_tokens]                    factor 1
                //                     [n_tokens * n_embd_v_gqa_max] when V is transposed
                // The shape analysis derives the same factor from the tensor
                // (d_ntok = ne / base_ntok), so the patch moves this dimension
                // by `factor * delta`. A length that is only a multiple of the
                // token width would let a wrong layout pass: [2 * n_tokens] and
                // [n_embd_v_gqa_max * n_tokens] are indistinguishable then.
                // Check the exact length instead.
                const int64_t tokens = (int64_t) build_n_tokens;
                int64_t factor = 1;
                if (std::strcmp(name, "union_inp_pos") == 0) {
                    factor = (int64_t) n_pos_per_embd;
                } else if (std::strcmp(name, "union_inp_v_idxs") == 0) {
                    // the transposed V index is longer; the non-transposed one
                    // is exactly n_tokens. Accept either, nothing else.
                    factor = t->ne[0] == tokens ? 1 : (int64_t) n_embd_v_gqa_max;
                }
                if (factor <= 0 || t->ne[1] != 1 || t->ne[2] != 1 || t->ne[3] != 1 ||
                        t->ne[0] != tokens * factor) {
                    return fail();
                }
                excl[i].ntok_dim = 0;
                has_ntok = true;
                ntok_roots.push_back(name);
            }

            // ggml stores a view's offset in op_params as well as in
            // view_offs. The binder patches view_offs, so the op-param copy is
            // a shape, not an op parameter, and the hash must skip it.
            if (t->view_src != nullptr) {
                excl[i].skip_view_data = true;
            }

            // The CPU attention reduction length lives in op_params[3] and the
            // bind rewrites it every step, so it is a runtime value and not a
            // topology fact. Recorded here, at build time, so the exclusion is
            // positional and does not depend on the value the slot holds.
            if ((t->flags & GGML_TENSOR_FLAG_TRUE_LEN_REDUCE) != 0) {
                excl[i].skip_op_param = 3;
            }
        }

        // The exclusion set the BINDER uses: every axis whose coefficient in
        // llama_union_shape_map is nonzero, resolved to this list by the
        // caller. The name-based records above cannot see a derived tensor --
        // the GET_ROWS result whose ne[1] inherits the copy index's length,
        // the matmul whose ne[0] inherits the KV width -- and those are
        // exactly the tensors the bind patches, so hashing them moved the
        // table's hash with a runtime parameter.
        //
        // Every entry must name a position in this list. A caller that
        // resolved a coefficient to a tensor outside the hashed set drops it
        // before calling (its shape does not enter the hash); an index that
        // survives to here is a caller bug, and a table that silently ignored
        // it would hash a dimension the bind moves.
        for (const llama_union_topology_excl & r : runtime_dims) {
            if (r.node < 0 || (size_t) r.node >= nodes.size()) {
                return fail();
            }
            if ((r.dim_mask & (uint8_t) ~0x0fu) != 0) {
                return fail();
            }
            llama_union_topology_excl & e = excl[r.node];
            e.node           = r.node;
            e.dim_mask      |= r.dim_mask;
            e.skip_view_data  = e.skip_view_data || r.skip_view_data;
            // The named slots are a per-parameter EXAMPLE axis, not the
            // exclusion: one slot cannot hold two axes of the same parameter
            // (a reshape that splits a runtime width across two axes), so a
            // later coefficient of the same parameter overwrites an earlier
            // one here. dim_mask above is the exact set and is what
            // hash_nodes() reads; nothing in this function's fail-closed
            // checks reads these slots, and has_* / ntok_roots are set by the
            // name-based records above, before this loop.
            if (r.dim      >= 0) e.dim      = r.dim;
            if (r.nout_dim >= 0) e.nout_dim = r.nout_dim;
            if (r.nrs_dim  >= 0) e.nrs_dim  = r.nrs_dim;
            if (r.head_dim >= 0) e.head_dim = r.head_dim;
            if (r.ntok_dim >= 0) e.ntok_dim = r.ntok_dim;
        }
        // A nonzero token width means the token lane is armed. If no node
        // carried it, the table cannot exclude it from the hash, and the bind
        // would refuse itself exactly as it did before the exclusion existed.
        // Refuse the table instead of pretending the lane is recorded.
        if (build_n_tokens != 0 && !has_ntok) {
            return fail();
        }
        return true;
    }

    // Arm the width the graph must execute at. `arena_n_kv` is the width the
    // live graph's allocation currently holds, not the width the graph was
    // built at: at record time the two are equal, and a successful replan is
    // the one operation that widens the arena without rebuilding the graph.
    // On the token lane the graph is built at the first live width and the
    // replan then widens the arena, so a build-width bound would refuse every
    // later width (measured on the committed tree: runtime_n_kv=0 at cell 1
    // while the arena held 4). Returns false for a width outside
    // (0, arena_n_kv]: zero is not a width, and a width above the arena has
    // no allocated backing. The bound is supplied by the caller because the
    // live arena width is context state, not a fact of this table.
    bool set_runtime_n_kv(uint32_t n_kv, uint32_t arena_n_kv) {
        if (n_kv == 0 || n_kv > arena_n_kv) {
            runtime_n_kv = 0;
            return false;
        }
        runtime_n_kv = n_kv;
        return true;
    }

    // Width the consumer must read: the armed runtime width, or the build
    // width when nothing is armed. Never returns 0 for a built table.
    uint32_t effective_n_kv() const {
        return runtime_n_kv != 0 ? runtime_n_kv : build_n_kv;
    }

    // Topology hash of `nodes` under this table's runtime dimensions. Equal
    // values across two combinations mean the same node set, edges, op types
    // and non-runtime parameters; only the recorded runtime dimensions may
    // differ.
    bool topology_hash(const std::vector<ggml_tensor *> & nodes, uint64_t & out) const {
        if (excl.size() != nodes.size()) {
            return false;
        }
        return llama_union_topology_hash::hash_nodes(nodes, excl, out);
    }
};
