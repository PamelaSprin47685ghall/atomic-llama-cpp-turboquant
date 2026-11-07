// CPU unit test: n_kv, n_outputs, rs_z and n_tokens are runtime parameters of
// ONE fixed-topology graph, not build-time constants.
//
// Pillar 1 asks for one graph topology reused across all four parameters, with
// their values supplied at execution time. This file is the single test for
// that claim. It replaces the two overlapping matrices it grew out of
// (tests/test-nkv-runtime-param.cpp part 3, three axes, and
// tests/test-ntok-cross-matrix.cpp part 2, four axes): one matrix, one
// registration, one place to read.
//
// Parts, and what each one needs:
//
// Part 1 (always runs, no model): the topology hash and the width table over
// all four parameters.
//   (a) the same synthetic skeleton built at every combination of
//       n_kv x n_outputs x n_rs hashes EQUAL through the width table
//       (llama_union_width_table::topology_hash), and a non-runtime change
//       still moves it;
//   (b) the same skeleton at seven token widths hashes EQUAL through the
//       token-lane exclusion and DIFFERENT without it;
//   (c) the SINGLE GRAPH CACHE KEY (llama_topology_key_of) hashes equal at
//       those token widths when the token lane's exclusion is passed, and
//       different without it. This is the pair the binder's own reuse gate
//       depends on: the gate compares a re-taken key, so a key that saw the
//       token width would refuse every token-width bind;
//   (d) the width table records the token dimension of the mask and of the
//       four token inputs, refuses a mask whose row count disagrees with the
//       recorded token width, and leaves the lane unrecorded at width 0;
//   (e) the token inputs are checked at their EXACT length, not at a length
//       that merely divides by the token width;
//   (f) an armed token lane whose graph carries no token root is refused;
//   (g) the plumbing: union_topology_key_excl() names all four token roots on
//       the token lane and nothing on every other lane;
//   (h) the key's exclusion set covers every tensor the width table marks as
//       a token root;
//   (i) the no-op rule's lane answer: the non-token lane answers the recorded
//       width, the token lane the live width;
//   (j) FALSIFIABILITY: a token width that leaks into the hash is refused.
//       A mask whose row count disagrees with the armed token width must be
//       refused, and the token width must reach the hash when the exclusion
//       is NOT passed. Without (j), (a) and (b) are satisfiable by an
//       exclusion that excludes everything.
//
// Part 2 (needs a model, and a missing model is an ERROR, not a skip): the
// four-parameter cross matrix on the live llama_decode path.
//   - one binder context decodes a fixed token history as batches of 1, 3, 2,
//     4, 7 and 5 tokens, with the output count moving independently and (on a
//     recurrent model) the rs_z lane moving, so n_kv, n_outputs, rs_z and
//     n_tokens all move between cells;
//   - ONE stock context walks the same history in the same order with
//     LLAMA_GRAPH_REUSE_DISABLE=1 set for its whole lifetime, so it rebuilds
//     for every cell;
//   - every cell must report the FIRST cell's topology hash, an unchanged
//     build width on all four axes (no rebuild), and the cell's own values as
//     the bound values;
//   - every compared logits row must be byte-identical (memcmp) to the stock
//     row. The denominator counts only cells that REQUESTED an output row: a
//     zero-output cell produces two empty buffers, and memcmp on two empty
//     ranges returns 0, so counting it would be a comparison that compared
//     nothing. That cell is proven by its own FOUR_NOUT_ZERO line instead;
//   - the two arms must be measurably different paths: the binder arm's
//     model.build_graph() delta is 0 and the stock arm's is >= cells. The
//     counts must also differ from each other; equal counts mean both arms
//     took the same path and the comparison proved nothing.
//
// Part 3 (needs a model): the mask refusal path must be able to fire. The
// child process decodes one token with LLAMA_NKV_MASK_OVERRIDE set above the
// mask's own ne[0] and must abort with the named message "runtime n_kv exceeds
// the KQ mask width". A guard that cannot be driven is a guard nobody has
// proven.
//
// The rs_z axis needs a recurrent or hybrid model (MAMBA / RWKV for the pure
// recurrent lane, QWEN35 / QWEN3NEXT for attention plus recurrent). A pure
// dense Llama has no recurrent state, so bound_rs_z stays -2; the test then
// prints FOUR_AXES with rs_z=0 and the reason, and crosses the other three
// axes. It never reports a four-axis pass it did not run.
//
// The FOUR_AXES line carries the CENSUS, not a flag: how many distinct values
// each axis actually took, counted from the values the binder arm reported.
// `n_kv` is the span the executed width covered; `n_outputs`, `rs_z` and
// `n_tokens` are distinct-value counts. A reader can therefore see that a
// dense-model run crossed three axes and that rs_z stayed inert, instead of
// having to trust a claim.
//
// The matrix's live graph carries all four token-lane roots, because the live
// graph does: llama-graph.cpp build_attn_inp_kv_impl builds
// union_inp_k_idxs and union_inp_v_idxs (src/llama-graph.cpp:3280-3281), so a
// key that folded their ne[0] would rebuild on every token-width cell. That
// is the case the exclusion must cover, and part 2 asserts rebuilds == 0 on
// it.
//
// What this test does NOT prove:
//   - that any ggml kernel reads a runtime width. The consumers are the host:
//     llm_graph_input_nkv_runtime::set_input fills the KQ mask at the width it
//     just wrote, and the binder patches the token tensors' shapes;
//   - that the stock arm is a different graph. It is: graph reuse is disabled
//     there, so every cell builds. What that does not prove is that the binder
//     arm would have failed without the binder; the rebuild delta is what
//     separates the two paths;
//   - anything about GPU/HIP, flash attention, relay or pipeline graphs.
//
// usage: test-union-four-param-matrix <model.gguf>
//        test-union-four-param-matrix --hash-only
//        test-union-four-param-matrix --matrix-only <model.gguf>
//        test-union-four-param-matrix --mask-negative <model.gguf>   (part 3 child)
//        test-union-four-param-matrix --mask-unproven <model.gguf>   (part 3 child)
//   --hash-only runs part 1 alone: no model, no backend, no allocation. That
//   registration always exists, so the hash contract is checked on every host.
//   --matrix-only runs part 1, part 2 and part 3 and nothing else. Its
//   PASS_REGULAR_EXPRESSION names the FOUR_MATRIX line, which part 2 prints,
//   so the regex covers part 2 only. A part-3 failure leaves FOUR_MATRIX ok=1
//   on the log and is carried by the exit code and by the FOUR_MASK_* lines;
//   a reader that trusts only the ctest result misses it.
//
// Output lines (parsed by tests/nkv-cpu-probe/gate-nkv-runtime-13022.sh,
// the only entry point for this matrix):
//   FOUR_HASH_ARM ok=<0|1> cases=<n> widths=<n> key_cases=<n>
//   FOUR_AXES n_kv=<span> n_outputs=<distinct> rs_z=<0|1|distinct> n_tokens=<distinct>
//             reason=<word>
//   FOUR_MATRIX ok=<0|1> cells=<n> hash=<hex> rebuilds=<binder>
//               stock_rebuilds=<stock> bitwise=<n>/<n> nout_zero=<k>
//               mask_cells=<n> mask_rows=<ok>/<checked>
//               mask_expect=<n> mask_cells_all=<n>
//     ok              every assertion part 2 made held (parts 1 and 3 gate
//                     the same flag, so a failure anywhere reads 0)
//     cells           cells the schedule executed
//     hash            topology hash of the binder arm's last cell; every cell
//                     must reproduce it (one hash across 124 binds)
//     rebuilds        model.build_graph() delta on the binder arm AFTER its
//                     initial build (the first cell's build is the graph the
//                     later cells reuse); 0 is the claim (bound in place,
//                     never rebuilt)
//     stock_rebuilds  the full-run delta on the stock arm, at least one build
//                     per cell including the first; asserted >= cells, so the
//                     cells really did take the rebuild path
//     bitwise         cells whose logits matched the stock arm bit for bit,
//                     over the cells that requested an output row
//                     (denominator = cells - nout_zero)
//     nout_zero       cells with n_outputs == 0; excluded from bitwise on
//                     both sides because two empty buffers memcmp equal
//     mask_cells      cells whose mask read-back reported at least one row
//                     compared; asserted == cells
//     mask_rows       rows that agreed at every column / rows compared, both
//                     summed over the cells; asserted ok == checked
//     mask_expect     rows the batches own: n_tokens * mask_filled summed
//                     over the cells; asserted checked == expect, so a
//                     read-back that compares one row per cell fails here
//                     even though checked >= cells would still hold
//     mask_cells_all  cells that compared EVERY row of their batch, taken
//                     from the measured mask_rows_all (not from the row-cap
//                     decision); asserted == cells
//   FOUR_NOUT_ZERO cell=<i> rc=<n> bound_n_outputs=<n> n_tokens=<n>
//   FOUR_LEAK_NEG ok=<0|1>            (part 1(j) as a whole: ok=1 only when
//                                      BOTH cases below actually ran and both
//                                      refused. A case that never ran leaves
//                                      its flag false, so deleting the arm
//                                      turns this line red.)
//   FOUR_LEAK_NEG_CASE name=<word> ...   (one per falsifiability case: the
//                                        observed value that had to be refused)
//   FOUR_MASK_NEG ok=<0|1>            (part 3 as a whole: both arms)
//   FOUR_MASK_ROW_NEG ok=<0|1>        (part 3(c) alone, in process on one
//                                      context: the live row must read as
//                                      keep-then-drop, the clean row must
//                                      agree, a corrupted column must
//                                      disagree, and restoring it must agree
//                                      again. Its own line, so deleting that
//                                      arm cannot leave the gate green.)
//   FOUR_MASK_ROW_NEG_STATS runtime_n_kv=<n> mask_n_kv=<n> mask_filled=<n>
//                          mask_refused=<n> mask_row_refused=<n> ne0=<n>
//                          tail_cols=<n>
//   FOUR_MASK_UNPROVEN ok=<0|1>       (part 3(d) alone: UNPROVEN, PASS and
//                                      FAIL must be three distinct values and
//                                      all three reachable. Its own line, so
//                                      deleting that arm cannot leave the gate
//                                      green.)

#include "llama.h"

#include "llama-nout-state.h"
#include "llama-topology-hash.h"
#include "llama-union-topology.h"
#include "llama-union-width-table.h"

#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// part 1: the synthetic skeleton
// ---------------------------------------------------------------------------
//
// Only metadata matters to the hash and the table, so the tensors live in a
// ggml context with no data buffer:
//
//   tokens  = new_tensor_1d(I32, n_tokens)              "union_inp_tokens"
//   pos     = new_tensor_1d(I32, n_tokens*n_pos)        "union_inp_pos"
//   k_idxs  = new_tensor_1d(I64, n_tokens)              "union_inp_k_idxs"
//   v_idxs  = new_tensor_1d(I64, n_tokens[*n_v_gqa])    "union_inp_v_idxs"
//   k_view  = view_4d(head_k, n_head_kv, n_kv, 1)       "union_nkv_k"
//   v_view  = view_4d(head_v, n_head_kv, n_kv, 1)       "union_nkv_v"
//   mask    = new_tensor_4d(F32, n_kv, n_tokens, 1, 1)  "attn_inp_kq_mask"
//   oids    = new_tensor_1d(I32, n_outputs)             "union_out_ids"
//   scopy   = new_tensor_1d(I32, n_rs)                  "union_rs_copy"
//   sextra  = view_1d(scopy, n_rs - n_seqs, ...)        "union_rs_copy_extra"
//   sdst    = view_2d(state, state_size, n_rs - n_seqs, ...)
//                                                       "union_rs_copy_dst"
//   states  = get_rows(state, scopy)                    "states"
//
// The list is closed under src[]/view_src, because hash_nodes() refuses a node
// whose source is outside it: every consumer above has its sources in the
// list. The mask has no consumer here on purpose - the width table keys on its
// name, and the hash folds its shape, so a consumer would add nothing.
//
// This skeleton is what part 1 walks. The single graph cache key is a
// different walk (it starts from a ggml_cgraph and stops at fixed leaves), so
// it has its own builder below.

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

struct synth_params {
    int64_t n_kv      = 8;
    int64_t n_tokens  = 4;
    int64_t n_outputs = 1;
    int64_t n_rs      = 8;
    int64_t n_seqs    = 1;
    int64_t rs_head   = 0;
    // the two constant factors the token inputs can carry: the position input
    // is [n_tokens * n_pos_per_embd] and a transposed V index is
    // [n_tokens * n_embd_v_gqa_max]. 1 means "the plain layout".
    int64_t n_pos_per_embd   = 1;
    int64_t n_embd_v_gqa_max = 1;
    // 0: v_idxs is [n_tokens] (non-transposed); 1: [n_tokens * n_embd_v_gqa_max]
    int     v_trans = 0;
    // the mask's row count (ne[1]). 0 means "same as n_tokens", which is the
    // only layout the real builder produces. A nonzero value that differs
    // from n_tokens is a LEAK fixture: the table must refuse it when the
    // token lane is armed (llama-union-width-table.h:200-206).
    int64_t mask_ntok = 0;
};

// `perturb` changes one thing the hash must see:
//   0: nothing
//   2: a non-runtime shape (the head dimension of the K view)
static bool synth_build(synth_graph & g, const synth_params & p, int perturb) {
    synth_free(g);

    ggml_init_params ip = {};
    ip.mem_size   = 16u * 1024u * 1024u;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    g.ctx = ggml_init(ip);
    if (g.ctx == nullptr) {
        return false;
    }

    const int64_t head_k = perturb == 2 ? 9 : 8;
    const int64_t head_v = 8;
    const int64_t n_head = 2;
    const int64_t state_size = 32;

    ggml_tensor * tokens = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, p.n_tokens);
    ggml_set_name(tokens, "union_inp_tokens");
    ggml_tensor * pos = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, p.n_tokens * p.n_pos_per_embd);
    ggml_set_name(pos, "union_inp_pos");
    ggml_tensor * k_idxs = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I64, p.n_tokens);
    ggml_set_name(k_idxs, "union_inp_k_idxs");
    ggml_tensor * v_idxs = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I64,
            p.v_trans ? p.n_tokens * p.n_embd_v_gqa_max : p.n_tokens);
    ggml_set_name(v_idxs, "union_inp_v_idxs");

    // The KV cache and the recurrent state are STORAGE: pre-existing leaves
    // whose address the graph never owns. The digest's leaf predicate is the
    // binder record's (`view_src == nullptr && (data != nullptr || buffer !=
    // nullptr)`), and the canonical walk stops at a fixed node, so these
    // tensors are not members of the hashed node set -- only the views built
    // over them are. The sentinel is never dereferenced: nothing here
    // computes, and the digest never reads `data`.
    //
    // Keeping them in `g.nodes` would hash their ne[2] = n_kv, which the live
    // graph never hashes, and the sweep below would then move the hash on a
    // parameter whose dimension the binder does not patch on a leaf.
    ggml_tensor * k_backing = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, head_k, n_head, p.n_kv, 1);
    ggml_tensor * v_backing = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, head_v, n_head, p.n_kv, 1);
    for (ggml_tensor * t : { k_backing, v_backing }) {
        t->data = &t;
    }

    ggml_tensor * k_view = ggml_view_4d(g.ctx, k_backing, head_k, n_head, p.n_kv, 1,
            k_backing->nb[1], k_backing->nb[2], k_backing->nb[3], 0);
    ggml_set_name(k_view, "union_nkv_k");
    ggml_tensor * v_view = ggml_view_4d(g.ctx, v_backing, head_v, n_head, p.n_kv, 1,
            v_backing->nb[1], v_backing->nb[2], v_backing->nb[3], 0);
    ggml_set_name(v_view, "union_nkv_v");

    // mask_ntok = 0 is the real layout (rows == n_tokens); a nonzero value
    // builds the leak fixture the negative arm needs.
    const int64_t mask_rows = p.mask_ntok != 0 ? p.mask_ntok : p.n_tokens;
    ggml_tensor * mask = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, p.n_kv, mask_rows, 1, 1);
    ggml_set_name(mask, "attn_inp_kq_mask");

    ggml_tensor * oids = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, p.n_outputs);
    ggml_set_name(oids, "union_out_ids");

    // rs_z lane: the recurrent state length and the copy views. `state` is the
    // recurrent state STORAGE (a fixed leaf, as in the live graph), so it is
    // not a member of the hashed set either; the copy views and the gather
    // built over it are.
    ggml_tensor * state = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, state_size, p.n_rs);
    ggml_set_name(state, "state");
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

    // graph order, sources before consumers. The three storage leaves
    // (k_backing, v_backing, state) are deliberately absent: the canonical
    // node set the digest hashes stops at a fixed node, so a hashed list that
    // contained them would be a graph the live path never hashes. Their views
    // fold the edge to them as EXTERNAL, exactly as the canonical hash does.
    g.nodes = { tokens, pos, k_idxs, v_idxs, k_view, v_view,
                mask, oids, scopy, sextra, sdst, states };
    return true;
}

// The shape map union_shape_analyze() derives for this skeleton, in the same
// resolved form the record build hands the width table: one entry per tensor
// the binder patches, naming the axes whose coefficient is nonzero.
//
// The values are the analysis' own rules for this graph, not a copy of the
// table's name list. `states` is the GET_ROWS result: its ne[1] inherits
// scopy->ne[0] = n_rs (the GET_ROWS rule out.ne[1] = s1.ne[0]), it carries no
// runtime name at all, and it is exactly the tensor the table could not see
// before the coefficient table was its source. `mask` appears in two lists
// because it carries two runtime axes (n_kv in ne[0], n_tokens in ne[1]).
//
// The storage leaves (k_backing, v_backing, state) have no entry: they are
// fixed, the analysis skips a fixed tensor, and the bind never patches one.
static std::vector<llama_union_topology_excl> synth_runtime_dims(
        const synth_graph & g, const synth_params & p, bool token_lane) {
    std::vector<llama_union_topology_excl> out(g.nodes.size());
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        out[i].node = (int32_t) i;
    }
    auto find = [&g](const char * name) -> int32_t {
        for (size_t i = 0; i < g.nodes.size(); ++i) {
            if (std::strcmp(g.nodes[i]->name, name) == 0) {
                return (int32_t) i;
            }
        }
        return -1;
    };
    // which: 0 n_kv, 1 n_outputs, 2 rs head, 3 rs length, 4 n_tokens
    auto mark = [&out](int32_t at, int dim, int which) {
        if (at < 0) {
            return;
        }
        llama_union_topology_excl & e = out[at];
        e.dim_mask |= (uint8_t) (1u << dim);
        switch (which) {
            case 0: e.dim      = dim; break;
            case 1: e.nout_dim = dim; break;
            case 2: e.head_dim = dim; break;
            case 3: e.nrs_dim  = dim; break;
            default: e.ntok_dim = dim; break;
        }
    };

    mark(find("union_nkv_k"), 2, 0);
    mark(find("union_nkv_v"), 2, 0);
    mark(find("attn_inp_kq_mask"), 0, 0);
    mark(find("union_out_ids"), 0, 1);
    mark(find("union_rs_copy"), 0, 3);
    mark(find("union_rs_copy_extra"), 0, 3);
    mark(find("union_rs_copy_dst"), 1, 3);
    mark(find("states"), 1, 3);              // GET_ROWS: out.ne[1] = s1.ne[0]
    const int32_t dst = find("union_rs_copy_dst");
    if (dst >= 0) {
        out[dst].head_dim = 1;               // the head selects the view offset
        out[dst].skip_view_data = true;
    }
    if (token_lane) {
        mark(find("attn_inp_kq_mask"), 1, 4);
        mark(find("union_inp_tokens"), 0, 4);
        mark(find("union_inp_pos"), 0, 4);
        mark(find("union_inp_k_idxs"), 0, 4);
        mark(find("union_inp_v_idxs"), 0, 4);
    }
    return out;
}

// The width table for `p`. `token_lane` false passes a token width of 0, which
// is what every non-token lane does: the token dimensions then stay hashed as
// fixed shapes.
static bool synth_table(const synth_graph & g, const synth_params & p,
                        llama_union_width_table & table, bool token_lane = true) {
    return table.build(g.nodes, synth_runtime_dims(g, p, token_lane),
                       (uint32_t) p.n_kv, (uint32_t) p.n_outputs,
                       (uint32_t) p.n_rs, (uint32_t) p.rs_head,
                       token_lane ? (uint32_t) p.n_tokens : 0,
                       (uint32_t) p.n_pos_per_embd, (uint32_t) p.n_embd_v_gqa_max);
}

static bool synth_hash(const synth_graph & g, int64_t n_tokens_excl, uint64_t & out) {
    std::vector<llama_union_topology_excl> excl(g.nodes.size());
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        excl[i].node = (int32_t) i;
        const char * name = g.nodes[i]->name;
        if (std::strcmp(name, "union_nkv_k") == 0 || std::strcmp(name, "union_nkv_v") == 0) {
            excl[i].dim = 2;
        } else if (std::strcmp(name, "attn_inp_kq_mask") == 0) {
            excl[i].dim = 0;
            if (n_tokens_excl != 0) {
                excl[i].ntok_dim = 1;
            }
        } else if (std::strcmp(name, "union_out_ids") == 0) {
            excl[i].nout_dim = 0;
        } else if (std::strcmp(name, "union_rs_copy") == 0 ||
                   std::strcmp(name, "union_rs_copy_extra") == 0) {
            excl[i].nrs_dim = 0;
        } else if (std::strcmp(name, "union_rs_copy_dst") == 0) {
            excl[i].nrs_dim = 1;
        } else if (n_tokens_excl != 0 &&
                (std::strcmp(name, "union_inp_tokens") == 0 ||
                 std::strcmp(name, "union_inp_pos") == 0 ||
                 std::strcmp(name, "union_inp_k_idxs") == 0 ||
                 std::strcmp(name, "union_inp_v_idxs") == 0)) {
            excl[i].ntok_dim = 0;
        }
        if (g.nodes[i]->view_src != nullptr) {
            excl[i].skip_view_data = true;
        }
    }
    return llama_union_topology_hash::hash_nodes(g.nodes, excl, out);
}

// The single graph cache key needs a real ggml_cgraph, because it walks the
// graph's node list. The skeleton above is a plain tensor list, so this builds
// the same shapes into a graph whose roots are the tensors the key must fold.
// The token inputs are graph nodes here too: each is consumed by a real op.
struct synth_key_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * gf  = nullptr;
};

static void synth_key_free(synth_key_graph & g) {
    if (g.ctx != nullptr) {
        ggml_free(g.ctx);
        g.ctx = nullptr;
    }
    g.gf = nullptr;
}

// The token-lane key graph must carry ALL FOUR token roots, because that is
// the case the exclusion has to cover: the two KV index inputs are graph nodes
// too, and a key that folded their ne[0] would move on every token-width bind.
// The live graph carries them as well (llama-graph.cpp
// build_attn_inp_kv_impl -> llama_kv_cache::build_input_k_idxs /
// build_input_v_idxs, src/llama-graph.cpp:3280-3281), so this builder mirrors
// the live shape rather than a simplified one.
static bool synth_key_build(synth_key_graph & g, const synth_params & p) {
    synth_key_free(g);

    ggml_init_params ip = {};
    ip.mem_size   = 16u * 1024u * 1024u;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    g.ctx = ggml_init(ip);
    if (g.ctx == nullptr) {
        return false;
    }

    const int64_t n_embd = 32;

    // fixed leaves: storage pre-exists the build, so the canonical walk stops
    // at them (they are marked with a non-null data sentinel, never read)
    ggml_tensor * w  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, n_embd);
    ggml_set_name(w, "w.weight");
    // kv is [n_embd, 1, n_kv, 1]: the cache row width is n_embd, so the 2-D
    // view below has the same ne[0] as the projection it multiplies. A cache
    // row narrower than the projection is not a graph ggml can build at all
    // (ggml_can_mul_mat: a->ne[0] == b->ne[0]), and the width the key must see
    // is n_kv, which lives in dimension 2 here.
    ggml_tensor * kv = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, n_embd, 1, p.n_kv, 1);
    ggml_set_name(kv, "k_cache");
    ggml_tensor * oid = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, p.n_outputs);
    ggml_set_name(oid, "out_ids");
    ggml_tensor * rs  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, p.n_rs);
    ggml_set_name(rs, "rs");
    for (ggml_tensor * t : { w, kv, oid, rs }) {
        t->data = &t;
    }

    // the token-lane inputs: graph NODES, each consumed by an op
    ggml_tensor * tokens = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, p.n_tokens);
    ggml_set_name(tokens, "union_inp_tokens");
    ggml_tensor * pos = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, p.n_tokens * p.n_pos_per_embd);
    ggml_set_name(pos, "union_inp_pos");

    // the two KV index inputs, as llama_kv_cache::build_input_k_idxs /
    // build_input_v_idxs create them: I64 [n_tokens], consumed by set_rows
    ggml_tensor * k_idxs = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I64, p.n_tokens);
    ggml_set_name(k_idxs, "union_inp_k_idxs");
    ggml_tensor * v_idxs = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I64, p.n_tokens);
    ggml_set_name(v_idxs, "union_inp_v_idxs");

    ggml_tensor * tok_rows = ggml_get_rows(g.ctx, w, tokens);   // [n_embd, n_tokens]
    ggml_set_name(tok_rows, "tok_rows");
    ggml_tensor * q = ggml_mul_mat(g.ctx, w, tok_rows);         // [n_embd, n_tokens]
    ggml_set_name(q, "q");
    // pos is consumed by an op so it is a node, not a leaf. A 1-D [n_tokens]
    // tensor cannot be added to [n_embd, n_tokens]: ggml broadcasts by
    // divisibility on every axis, and n_tokens does not divide n_embd. The
    // per-token vector is therefore reshaped to [1, n_tokens], which is the
    // shape ggml can repeat across the row axis. pos itself keeps ne[0] =
    // n_tokens, so it stays the token-lane root the key names.
    ggml_tensor * posf = ggml_cast(g.ctx, pos, GGML_TYPE_F32);   // [n_tokens*n_pos]
    ggml_set_name(posf, "posf");
    ggml_tensor * posb = ggml_reshape_2d(g.ctx, posf, 1, p.n_tokens);
    ggml_set_name(posb, "posb");
    ggml_tensor * sum = ggml_add(g.ctx, q, posb);                // [n_embd, n_tokens]

    // kv is [n_embd, 1, n_kv, 1]: reshaping to [n_embd, n_kv] would reorder the
    // dimensions, so the cache is read through a 2-D view whose row stride is
    // the third dimension's stride. The width stays n_kv, as in the real cache.
    ggml_tensor * k_all = ggml_view_2d(g.ctx, kv, n_embd, p.n_kv, kv->nb[2], 0);
    ggml_set_name(k_all, "k_all");
    ggml_tensor * kq = ggml_mul_mat(g.ctx, k_all, sum);         // [n_kv, n_tokens]
    ggml_set_name(kq, "kq");

    ggml_tensor * mask = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, p.n_kv, p.n_tokens, 1, 1);
    ggml_set_name(mask, "attn_inp_kq_mask");
    ggml_tensor * sm = ggml_soft_max_ext(g.ctx, kq, mask, 1.0f, 0.0f);
    ggml_set_name(sm, "kq_soft_max");

    // ggml_get_rows needs a 2-D source: sm is [n_kv, n_tokens, 1, 1]
    ggml_tensor * sm2 = ggml_reshape_2d(g.ctx, sm, p.n_kv, p.n_tokens);
    ggml_set_name(sm2, "kq_soft_max_2d");
    ggml_tensor * rows = ggml_get_rows(g.ctx, sm2, oid);
    ggml_set_name(rows, "rows");
    // The two index inputs are consumed by ggml_set_rows, exactly as the live
    // graph consumes them: the cache rows are the destination, the token rows
    // the source, the index input src[2]. That keeps them NODES (a tensor no op
    // consumes is a leaf, and a leaf is not in the hashed node set).
    //
    // ggml_set_rows asserts a->ne[0] == b->ne[0], a->ne[2] == b->ne[2],
    // a->ne[3] == b->ne[3] and b->ne[1] == c->ne[0]; the destination and the
    // source are therefore the same [n_embd, n_tokens] tensor and the index is
    // [n_tokens]. The shapes are a placeholder (nothing computes); the
    // NODE-ness is the point.
    ggml_tensor * k_set = ggml_set_rows(g.ctx, sum, sum, k_idxs);
    ggml_set_name(k_set, "k_set");
    ggml_tensor * v_set = ggml_set_rows(g.ctx, sum, sum, v_idxs);
    ggml_set_name(v_set, "v_set");

    // the rs_z lane: a zero row of the recurrent state, sized by n_rs
    ggml_tensor * z = ggml_view_2d(g.ctx, rs, n_embd, 1, rs->nb[1], (size_t) 0);
    ggml_set_name(z, "rs_zero");
    ggml_tensor * zs = ggml_scale(g.ctx, z, 0.0f);
    ggml_set_name(zs, "rs_zero_scaled");

    // The join. The four branches have four different shapes ([n_kv,
    // n_outputs], [n_embd, n_tokens] twice, [n_embd, 1]), and ggml_add
    // broadcasts only by divisibility, so they cannot be added to each other
    // directly. ggml_sum reduces each branch to [1] and the four scalars are
    // then the same shape. Every branch still reaches the root, so every node
    // above (and the two index inputs) is in the graph.
    ggml_tensor * s_rows = ggml_sum(g.ctx, rows);
    ggml_set_name(s_rows, "sum_rows");
    ggml_tensor * s_k = ggml_sum(g.ctx, k_set);
    ggml_set_name(s_k, "sum_k");
    ggml_tensor * s_v = ggml_sum(g.ctx, v_set);
    ggml_set_name(s_v, "sum_v");
    ggml_tensor * s_z = ggml_sum(g.ctx, zs);
    ggml_set_name(s_z, "sum_z");
    ggml_tensor * root = ggml_add(g.ctx,
            ggml_add(g.ctx, s_rows, s_k), ggml_add(g.ctx, s_v, s_z));
    ggml_set_name(root, "root");

    g.gf = ggml_new_graph_custom(g.ctx, 64, false);
    ggml_build_forward_expand(g.gf, root);
    return true;
}

// The name exclusion the binder derives from the shape map for this graph:
// every (tensor name, axis) whose ne[] differs between two builds at different
// token widths. That is exactly the set union_topology_key_excl(map, record_t)
// names, because the coefficient table's nonzero axes are the axes the token
// width moves.
//
// Matching is by NAME, not by pointer: the key is compared across two graph
// objects, so only the name survives. The fixture names every node it creates,
// as the live graph does (llama-graph.cpp cb()).
static llama_topology_key_excl synth_key_excl(const synth_key_graph & a, const synth_key_graph & b) {
    llama_topology_key_excl e;
    std::vector<const ggml_tensor *> na, nb;
    llama_topology_detail::topology_index ia, ib;
    std::vector<size_t> pa, pb;
    llama_topology_canonical_nodes(a.gf, na, ia, pa);
    llama_topology_canonical_nodes(b.gf, nb, ib, pb);
    for (const ggml_tensor * ta : na) {
        if (ta == nullptr) {
            continue;
        }
        const ggml_tensor * tb = nullptr;
        for (const ggml_tensor * t : nb) {
            if (t != nullptr && std::strcmp(t->name, ta->name) == 0) {
                tb = t;
                break;
            }
        }
        if (tb == nullptr) {
            continue;
        }
        // ne[] AND nb[]: the binder patches both (union_patch_value carries
        // both, and its add() applies the nb[] coefficients), and a 1-D
        // token root's nb[1..3] are derived from ne[0] by the allocation, so
        // an axis whose stride moves is a patched axis even when its own ne[]
        // does not. The coefficient table spells this the same way:
        // c.ne[d] != 0 || c.nb[d] != 0.
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (ta->ne[d] != tb->ne[d] || ta->nb[d] != tb->nb[d]) {
                e.add_name(ta->name, d);
            }
        }
    }
    return e;
}

static bool part1() {
    bool ok = true;
    int cases = 0;
    int key_cases = 0;
    uint64_t ref_key_hash = 0;

    // The token widths the sweep walks. Declared once, at function scope, so
    // (a), (b), (c) and the FOUR_HASH_ARM line all read the same array and
    // the printed width count cannot drift from what was swept.
    const int64_t tok_v[] = { 1, 3, 2, 4, 7, 5, 300 };
    const int     n_tok_v = (int) (sizeof(tok_v) / sizeof(tok_v[0]));

    // (a) every combination of n_kv x n_outputs x n_rs hashes the same.
    // The values include 1, a non-aligned value and a value past the 256-cell
    // bucket on each axis, so the exclusion is exercised on both sides of it.
    {
        const int64_t kv_v[]  = { 1, 8, 64, 256, 257, 300 };
        // n_outputs = 0 is a sweep value, not a negative: the table requires
        // union_out_ids->ne[0] == build_n_outputs exactly
        // (llama-union-width-table.h:207-213), so a zero-length tensor and a
        // zero parameter agree and the lane is recorded. The refusal half is
        // the length MISMATCH below, which is the assertion that can fail.
        const int64_t out_v[] = { 0, 1, 2, 257 };
        // 255/256/257 straddle the 256-cell bucket explicitly. 260 alone
        // implies the crossing; these three make it a boundary the sweep
        // steps across rather than a value that happens to be past it.
        const int64_t nrs_v[] = { 8, 9, 255, 256, 257, 260 };

        uint64_t ref = 0;
        bool have_ref = false;
        for (int64_t kv : kv_v) {
            for (int64_t no : out_v) {
                for (int64_t nr : nrs_v) {
                    for (int64_t nt : tok_v) {
                        synth_params p;
                        p.n_kv      = kv;
                        p.n_outputs = no;
                        p.n_rs      = nr;
                        p.n_tokens  = nt;
                        synth_graph g;
                        if (!synth_build(g, p, 0)) {
                            std::fprintf(stderr, "FAIL: synth build at n_kv=%" PRId64 "\n", kv);
                            ok = false;
                            synth_free(g);
                            continue;
                        }
                        llama_union_width_table table;
                        uint64_t h = 0;
                        if (!synth_table(g, p, table) || !table.topology_hash(g.nodes, h)) {
                            std::fprintf(stderr,
                                    "FAIL: width table or hash at n_kv=%" PRId64 " n_outputs=%" PRId64
                                    " n_rs=%" PRId64 " n_tokens=%" PRId64 "\n", kv, no, nr, nt);
                            ok = false;
                            synth_free(g);
                            continue;
                        }
                        if (!have_ref) {
                            ref = h;
                            have_ref = true;
                        } else if (h != ref) {
                            std::fprintf(stderr,
                                    "FAIL: the width-table hash moved with a runtime parameter at"
                                    " n_kv=%" PRId64 " n_outputs=%" PRId64 " n_rs=%" PRId64
                                    " n_tokens=%" PRId64 " (%016" PRIx64 " vs %016" PRIx64 ")\n",
                                    kv, no, nr, nt, h, ref);
                            ok = false;
                        }
                        ++cases;
                        synth_free(g);
                    }
                }
            }
        }

        // a non-runtime change must still move it with every exclusion on
        synth_graph a, b;
        synth_params p;
        if (!synth_build(a, p, 0) || !synth_build(b, p, 2 /* head + 1 */)) {
            std::fprintf(stderr, "FAIL: synth build for the head case\n");
            ok = false;
        } else {
            llama_union_width_table ta, tb;
            uint64_t ha = 0, hb = 0;
            if (!synth_table(a, p, ta) || !synth_table(b, p, tb) ||
                !ta.topology_hash(a.nodes, ha) || !tb.topology_hash(b.nodes, hb)) {
                std::fprintf(stderr, "FAIL: hash for the head case\n");
                ok = false;
            } else if (ha == hb) {
                std::fprintf(stderr, "FAIL: a non-runtime shape change did not move the hash\n");
                ok = false;
            }
        }
        synth_free(a);
        synth_free(b);
    }

    // (a2) n_kv = 0 is NOT a sweep value: it is fail-closed. The table cannot
    // record a runtime width of zero (llama-union-width-table.h:160-162:
    // `if (build_n_kv == 0) return fail();`), because zero means "no runtime
    // width is armed" and a consumer must then fall back to the shape baked
    // into the tensor rather than to a guessed value. The skeleton is built at
    // a real width and the table is asked for zero, so only the guard can make
    // this fail.
    {
        synth_graph g;
        synth_params p;
        if (!synth_build(g, p, 0)) {
            std::fprintf(stderr, "FAIL: nkv-zero: synth build\n");
            ok = false;
        } else {
            llama_union_width_table zero;
            if (zero.build(g.nodes, 0, (uint32_t) p.n_outputs,
                           (uint32_t) p.n_rs, (uint32_t) p.rs_head,
                           (uint32_t) p.n_tokens, 1, 1)) {
                std::fprintf(stderr, "FAIL: nkv-zero: build_n_kv = 0 was accepted\n");
                ok = false;
            } else if (!zero.excl.empty() || zero.has_nkv || zero.build_n_kv != 0) {
                // a failed build must leave an EMPTY table: a half-built one
                // would let topology_hash() succeed over partial data
                std::fprintf(stderr,
                        "FAIL: nkv-zero: the refused table was not cleared"
                        " (excl=%zu has_nkv=%d build_n_kv=%u)\n",
                        zero.excl.size(), int(zero.has_nkv), zero.build_n_kv);
                ok = false;
            } else {
                // the same skeleton at the real width must still be accepted,
                // so the refusal above is the guard and not a broken skeleton
                llama_union_width_table good;
                if (!synth_table(g, p, good) || !good.has_nkv) {
                    std::fprintf(stderr, "FAIL: nkv-zero: the real width was refused\n");
                    ok = false;
                }
            }
        }
        synth_free(g);
    }

    // (a3) n_outputs: zero is ACCEPTED and recorded, a length mismatch is
    // REFUSED. Both halves are asserted, because a sweep that crosses zero
    // only proves the hash did not move; it does not prove the table still
    // checks the length. The refusal is the assertion that can fail.
    {
        synth_params p;
        p.n_outputs = 0;
        synth_graph g;
        if (!synth_build(g, p, 0)) {
            std::fprintf(stderr, "FAIL: nout-zero: synth build\n");
            ok = false;
        } else {
            // (i) a zero-length union_out_ids against build_n_outputs = 0 is
            // recorded, and the zero dimension is excluded from the hash
            llama_union_width_table zero;
            if (!synth_table(g, p, zero)) {
                std::fprintf(stderr, "FAIL: nout-zero: build_n_outputs = 0 was refused\n");
                ok = false;
            } else if (!zero.has_nout || zero.build_n_outputs != 0) {
                std::fprintf(stderr,
                        "FAIL: nout-zero: the zero output lane was not recorded"
                        " (has_nout=%d build_n_outputs=%u)\n",
                        int(zero.has_nout), zero.build_n_outputs);
                ok = false;
            }

            // (ii) the SAME skeleton, asked for one output row, must be
            // refused: union_out_ids is [0] and cannot carry one row
            llama_union_width_table one;
            if (one.build(g.nodes, (uint32_t) p.n_kv, 1,
                          (uint32_t) p.n_rs, (uint32_t) p.rs_head,
                          (uint32_t) p.n_tokens, 1, 1)) {
                std::fprintf(stderr,
                        "FAIL: nout-zero: union_out_ids of length 0 was accepted"
                        " as build_n_outputs = 1\n");
                ok = false;
            }

            // (iii) the mirror: a skeleton with ONE output row asked for zero
            // must also be refused
            synth_params q = p;
            q.n_outputs = 1;
            synth_graph g1;
            if (!synth_build(g1, q, 0)) {
                std::fprintf(stderr, "FAIL: nout-zero: synth build for one output row\n");
                ok = false;
            } else {
                llama_union_width_table none;
                if (none.build(g1.nodes, (uint32_t) q.n_kv, 0,
                               (uint32_t) q.n_rs, (uint32_t) q.rs_head,
                               (uint32_t) q.n_tokens, 1, 1)) {
                    std::fprintf(stderr,
                            "FAIL: nout-zero: union_out_ids of length 1 was accepted"
                            " as build_n_outputs = 0\n");
                    ok = false;
                }
            }
            synth_free(g1);
        }
        synth_free(g);
    }

    // (b) the token dimension: equal with the exclusion, different without.
    // This is the negative half of the width-table hash: it proves (a) is not
    // true by construction.
    {
        uint64_t ref_on = 0, ref_off = 0;
        bool have_on = false, have_off = false;
        for (int64_t nt : tok_v) {
            synth_params p;
            p.n_tokens = nt;
            synth_graph g;
            if (!synth_build(g, p, 0)) {
                std::fprintf(stderr, "FAIL: ntok-hash: synth build at n_tokens=%" PRId64 "\n", nt);
                ok = false;
                synth_free(g);
                continue;
            }
            uint64_t on = 0, off = 0;
            if (!synth_hash(g, p.n_tokens, on) || !synth_hash(g, 0, off)) {
                std::fprintf(stderr, "FAIL: ntok-hash: hash at n_tokens=%" PRId64 "\n", nt);
                ok = false;
                synth_free(g);
                continue;
            }
            if (!have_on) {
                ref_on  = on;
                ref_off = off;
                have_on = have_off = true;
            } else {
                if (on != ref_on) {
                    std::fprintf(stderr,
                            "FAIL: ntok-hash: hash moved with the token dimension excluded"
                            " (n_tokens=%" PRId64 ": %016" PRIx64 " vs %016" PRIx64 ")\n",
                            nt, on, ref_on);
                    ok = false;
                }
                if (off == ref_off) {
                    std::fprintf(stderr,
                            "FAIL: ntok-hash: hash did NOT move with the token dimension hashed"
                            " (n_tokens=%" PRId64 "); the exclusion test proves nothing\n", nt);
                    ok = false;
                }
            }
            synth_free(g);
        }
    }

    // (c) the SINGLE GRAPH CACHE KEY, which is what the binder's reuse gate
    // compares. Same pair as (b), one level up: the key is taken over a real
    // ggml_cgraph and folds every ne[] unless the caller excludes a dimension.
    //
    // The exclusion is the one the product builds from the shape map
    // (union_topology_key_excl(map, record_t)): a NAME per patched axis,
    // because a key is compared across two graph OBJECTS and only the name
    // survives a rebuild. The four-name token_lane() set is NOT enough on this
    // graph: the token width also reaches derived nodes (tok_rows, q, sum, kq,
    // sm, rows, ...), which carry no runtime name at all. Here the patched set
    // is derived from two builds, which is exactly the set the coefficient
    // table names.
    {
        llama_topology_key_excl ident;
        bool have_ident = false;
        {
            synth_params p1, p5;
            p1.n_tokens = 1;
            p5.n_tokens = 5;
            synth_key_graph g1, g5;
            if (!synth_key_build(g1, p1) || !synth_key_build(g5, p5)) {
                std::fprintf(stderr, "FAIL: key: graph build for the exclusion\n");
                ok = false;
            } else {
                ident = synth_key_excl(g1, g5);
                have_ident = true;
                // The derived set must be non-empty: an empty one would make
                // the invariance below true by construction.
                if (ident.extra.empty()) {
                    std::fprintf(stderr,
                            "FAIL: key: the token width moved no dimension; the arm proves nothing\n");
                    ok = false;
                }
                // DIAGNOSTIC: the canonical order is derived from the node
                // shapes (llama_topology_node_key folds ne[]/nb[]), so two
                // builds that differ only in an excluded dimension can order
                // their nodes differently. Print whether they do, and whether
                // an exclusion that names EVERY axis of EVERY node still
                // separates the two keys -- if it does, no exclusion can make
                // the key invariant across builds and the cause is the order,
                // not a folded field.
                {
                    std::vector<const ggml_tensor *> na, nb_;
                    llama_topology_detail::topology_index ia, ib;
                    std::vector<size_t> pa, pb;
                    llama_topology_canonical_nodes(g1.gf, na, ia, pa);
                    llama_topology_canonical_nodes(g5.gf, nb_, ib, pb);
                    int first_diff = -1;
                    const size_t n = na.size() < nb_.size() ? na.size() : nb_.size();
                    for (size_t i = 0; i < n; ++i) {
                        if (std::strcmp(na[i]->name, nb_[i]->name) != 0) {
                            first_diff = (int) i;
                            break;
                        }
                    }
                    llama_topology_key_excl full;
                    for (const ggml_tensor * t : na) {
                        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                            full.add_name(t->name, d);
                        }
                    }
                    const uint64_t k1 = llama_topology_key_of(g1.gf, full).hash;
                    const uint64_t k5 = llama_topology_key_of(g5.gf, full).hash;
                    std::fprintf(stderr,
                            "KEY_DIAG nodes=%zu/%zu first_order_diff=%d full_excl_same=%d"
                            " (%016" PRIx64 " vs %016" PRIx64 ")\n",
                            na.size(), nb_.size(), first_diff, int(k1 == k5), k1, k5);
                }
            }
            synth_key_free(g1);
            synth_key_free(g5);
        }

        uint64_t ref_on = 0, ref_off = 0;
        bool have_on = false, have_off = false;
        for (int64_t nt : tok_v) {
            if (!have_ident) {
                break;
            }
            synth_params p;
            p.n_tokens = nt;
            synth_key_graph g;
            if (!synth_key_build(g, p)) {
                std::fprintf(stderr, "FAIL: key: graph build at n_tokens=%" PRId64 "\n", nt);
                ok = false;
                synth_key_free(g);
                continue;
            }
            const llama_topology_key on  = llama_topology_key_of(g.gf, ident);
            const llama_topology_key off = llama_topology_key_of(g.gf);
            if (!have_on) {
                ref_on  = on.hash;
                ref_off = off.hash;
                have_on = have_off = true;
            } else {
                if (on.hash != ref_on) {
                    std::fprintf(stderr,
                            "FAIL: key: the cache key moved with the token lane exclusion on"
                            " (n_tokens=%" PRId64 ": %016" PRIx64 " vs %016" PRIx64 ")\n",
                            nt, on.hash, ref_on);
                    ok = false;
                }
                if (off.hash == ref_off) {
                    std::fprintf(stderr,
                            "FAIL: key: the cache key did NOT move with the token dimension hashed"
                            " (n_tokens=%" PRId64 "); the exclusion test proves nothing\n", nt);
                    ok = false;
                }
            }
            ++key_cases;
            synth_key_free(g);
        }
        // The default argument is the no-exclusion form, so the two spellings
        // must agree. Fails if a default ever silently excluded something.
        synth_params p;
        synth_key_graph g;
        if (synth_key_build(g, p)) {
            if (llama_topology_key_of(g.gf).hash != llama_topology_key_of(g.gf, llama_topology_key_excl::none()).hash) {
                std::fprintf(stderr, "FAIL: key: the default exclusion is not none()\n");
                ok = false;
            }
        } else {
            std::fprintf(stderr, "FAIL: key: graph build for the default case\n");
            ok = false;
        }
        synth_key_free(g);
    }

    // (d) the width table records the token dimension and refuses a bad row
    // count, and leaves the lane unrecorded at width 0.
    {
        synth_graph g;
        synth_params p;
        if (!synth_build(g, p, 0)) {
            std::fprintf(stderr, "FAIL: ntok-table: synth build\n");
            ok = false;
        } else {
            llama_union_width_table table;
            if (!synth_table(g, p, table)) {
                std::fprintf(stderr, "FAIL: ntok-table: build\n");
                ok = false;
            } else if (!table.has_ntok || table.build_n_tokens != (uint32_t) p.n_tokens) {
                std::fprintf(stderr, "FAIL: ntok-table: token lane not recorded (has_ntok=%d build_n_tokens=%u)\n",
                        int(table.has_ntok), table.build_n_tokens);
                ok = false;
            } else {
                // the mask and the four token inputs must carry the token dim
                int n_ntok_dims = 0;
                for (const auto & e : table.excl) {
                    if (e.ntok_dim >= 0) {
                        ++n_ntok_dims;
                    }
                }
                if (n_ntok_dims != 5) {
                    std::fprintf(stderr, "FAIL: ntok-table: %d token dims recorded, want 5\n", n_ntok_dims);
                    ok = false;
                }
            }
            // a mask whose row count disagrees with the recorded token width
            // must be refused, not silently accepted
            if (ok) {
                llama_union_width_table bad;
                if (bad.build(g.nodes, (uint32_t) p.n_kv, (uint32_t) p.n_outputs,
                              (uint32_t) p.n_rs, (uint32_t) p.rs_head,
                              (uint32_t) p.n_tokens + 1, 1, 1)) {
                    std::fprintf(stderr, "FAIL: ntok-table: a token width the mask cannot hold was accepted\n");
                    ok = false;
                }
            }
            // a zero token width must leave the token lane unrecorded
            if (ok) {
                llama_union_width_table plain;
                if (!synth_table(g, p, plain, /*token_lane=*/false)) {
                    std::fprintf(stderr, "FAIL: ntok-table: build without the token lane\n");
                    ok = false;
                } else if (plain.has_ntok || plain.build_n_tokens != 0) {
                    std::fprintf(stderr, "FAIL: ntok-table: the token lane was recorded with width 0\n");
                    ok = false;
                }
            }
        }
        synth_free(g);
    }

    // (e) the token inputs are checked at their EXACT length. A multiple-of
    // check would accept both wrong pairings below.
    {
        synth_params p;
        p.n_tokens         = 4;
        p.n_pos_per_embd   = 2;   // M-RoPE: pos is [8]
        p.n_embd_v_gqa_max = 3;   // transposed V: v_idxs is [12]
        p.v_trans          = 1;

        synth_graph g;
        if (!synth_build(g, p, 0)) {
            std::fprintf(stderr, "FAIL: ntok-factor: synth build\n");
            ok = false;
        } else {
            // the matching factors are accepted
            llama_union_width_table good;
            if (!good.build(g.nodes, (uint32_t) p.n_kv, (uint32_t) p.n_outputs,
                            (uint32_t) p.n_rs, (uint32_t) p.rs_head,
                            (uint32_t) p.n_tokens, 2, 3) || !good.has_ntok) {
                std::fprintf(stderr, "FAIL: ntok-factor: the real layout was refused\n");
                ok = false;
            }
            // pos factor 1 against a pos tensor of [8] must be refused
            llama_union_width_table bad_pos;
            if (bad_pos.build(g.nodes, (uint32_t) p.n_kv, (uint32_t) p.n_outputs,
                              (uint32_t) p.n_rs, (uint32_t) p.rs_head,
                              (uint32_t) p.n_tokens, 1, 3)) {
                std::fprintf(stderr, "FAIL: ntok-factor: a pos tensor of n_tokens*2 was accepted "
                        "with factor 1\n");
                ok = false;
            }
            // v_idxs factor 2 against a tensor of [12] must be refused
            llama_union_width_table bad_v;
            if (bad_v.build(g.nodes, (uint32_t) p.n_kv, (uint32_t) p.n_outputs,
                            (uint32_t) p.n_rs, (uint32_t) p.rs_head,
                            (uint32_t) p.n_tokens, 2, 2)) {
                std::fprintf(stderr, "FAIL: ntok-factor: a v_idxs tensor of n_tokens*3 was accepted "
                        "with factor 2\n");
                ok = false;
            }
            // A non-transposed V index is exactly n_tokens, and the table
            // reads the LAYOUT from the tensor: a [n_tokens] v_idxs is the
            // plain layout at factor 1 whatever n_embd_v_gqa_max says, because
            // the declared factor is only used when the tensor is longer. The
            // refusal that exists is the LENGTH: a v_idxs of [n_tokens * 2]
            // against a declared transposed factor of 3 is neither layout, and
            // must be refused.
            //
            // (The previous arm here asked for a refusal of the plain tensor
            // under a transposed factor. That refusal cannot exist without
            // rejecting the legal plain layout, and it never ran before the
            // mul_mat abort above was fixed.)
            synth_params q = p;
            q.v_trans = 0;
            q.n_tokens = 4;
            synth_graph g2;
            if (!synth_build(g2, q, 0)) {
                std::fprintf(stderr, "FAIL: ntok-factor: synth build for the plain V layout\n");
                ok = false;
            } else {
                // widen the plain v_idxs to [n_tokens * 2]: neither n_tokens
                // nor n_tokens * n_embd_v_gqa_max
                ggml_tensor * v_idxs = nullptr;
                for (ggml_tensor * t : g2.nodes) {
                    if (std::strcmp(t->name, "union_inp_v_idxs") == 0) {
                        v_idxs = t;
                        break;
                    }
                }
                if (v_idxs == nullptr) {
                    std::fprintf(stderr, "FAIL: ntok-factor: no v_idxs in the plain V graph\n");
                    ok = false;
                } else {
                    v_idxs->ne[0] = q.n_tokens * 2;
                    v_idxs->nb[1] = v_idxs->nb[0] * v_idxs->ne[0];
                    llama_union_width_table bad_plain_v;
                    if (bad_plain_v.build(g2.nodes, (uint32_t) q.n_kv, (uint32_t) q.n_outputs,
                                          (uint32_t) q.n_rs, (uint32_t) q.rs_head,
                                          (uint32_t) q.n_tokens, 2, 3)) {
                        std::fprintf(stderr,
                                "FAIL: ntok-factor: a v_idxs of n_tokens*2 was accepted with"
                                " factor n_embd_v_gqa_max\n");
                        ok = false;
                    }
                }
            }
            synth_free(g2);
        }
        synth_free(g);
    }

    // (f) an armed token lane whose graph carries no token root must be
    // refused. Without this the table would report success while the hash
    // still counted the token dimension as fixed, and the bind would refuse
    // itself exactly as before the exclusion existed.
    {
        synth_graph g;
        synth_params p;
        if (!synth_build(g, p, 0)) {
            std::fprintf(stderr, "FAIL: ntok-lane: synth build\n");
            ok = false;
        } else {
            std::vector<ggml_tensor *> no_tokens;
            for (ggml_tensor * t : g.nodes) {
                if (std::strcmp(t->name, "union_inp_tokens") != 0 &&
                        std::strcmp(t->name, "union_inp_pos") != 0 &&
                        std::strcmp(t->name, "union_inp_k_idxs") != 0 &&
                        std::strcmp(t->name, "union_inp_v_idxs") != 0 &&
                        std::strcmp(t->name, "attn_inp_kq_mask") != 0) {
                    no_tokens.push_back(t);
                }
            }
            llama_union_width_table lane;
            if (lane.build(no_tokens, (uint32_t) p.n_kv, (uint32_t) p.n_outputs,
                           (uint32_t) p.n_rs, (uint32_t) p.rs_head,
                           (uint32_t) p.n_tokens, 1, 1)) {
                std::fprintf(stderr, "FAIL: ntok-lane: an armed token lane with no token root "
                        "was recorded\n");
                ok = false;
            }
        }
        synth_free(g);
    }

    // (g) THE PLUMBING, not the exclusion function. union_topology_key_excl()
    // is the function src/llama-context.cpp calls at the record build, at the
    // binder's topology gate and after every applied bind. This part calls the
    // SAME function, so a regression that made the call site pass none() - or
    // that inverted the lane test - turns it red. Part 1(c) alone would stay
    // green through such a regression: it calls the key directly.
    //
    // Fails when: the lane test is inverted, the token branch returns none(),
    // the token branch names fewer than the four roots the width table marks,
    // or the non-token branch returns anything but the empty set.
    {
        llama_union_shape_map token_lane_map;
        token_lane_map.ntok_valid = true;
        const llama_topology_key_excl token_excl = union_topology_key_excl(token_lane_map);
        if (token_excl.count != 4) {
            std::fprintf(stderr,
                    "FAIL: plumbing: the token lane excludes %d dimension(s), want 4\n",
                    token_excl.count);
            ok = false;
        }
        for (int i = 0; i < LLAMA_TOKEN_LANE_ROOTS_N; ++i) {
            if (!token_excl.names(LLAMA_TOKEN_LANE_ROOTS[i], 0)) {
                std::fprintf(stderr, "FAIL: plumbing: the token lane does not exclude %s\n",
                        LLAMA_TOKEN_LANE_ROOTS[i]);
                ok = false;
            }
        }
        // the four names are exactly the list, in both directions
        if (!token_excl.token_lane_roots_match(
                    std::vector<const char *>(LLAMA_TOKEN_LANE_ROOTS,
                                              LLAMA_TOKEN_LANE_ROOTS + LLAMA_TOKEN_LANE_ROOTS_N))) {
            std::fprintf(stderr, "FAIL: plumbing: the token lane's set is not the root list\n");
            ok = false;
        }

        llama_union_shape_map other_lane_map;
        other_lane_map.ntok_valid = false;
        const llama_topology_key_excl other_excl = union_topology_key_excl(other_lane_map);
        if (other_excl.count != 0) {
            std::fprintf(stderr,
                    "FAIL: plumbing: a non-token lane excludes %d dimension(s), want 0\n",
                    other_excl.count);
            ok = false;
        }
        // and a non-token lane's key is the plain key: the plumbing adds
        // nothing there
        synth_params p;
        synth_key_graph g;
        if (synth_key_build(g, p)) {
            if (llama_topology_key_of(g.gf, other_excl).hash != llama_topology_key_of(g.gf).hash) {
                std::fprintf(stderr, "FAIL: plumbing: a non-token lane moved the key\n");
                ok = false;
            }
        } else {
            std::fprintf(stderr, "FAIL: plumbing: graph build\n");
            ok = false;
        }
        synth_key_free(g);
    }

    // (h) THE INVARIANT: the key's exclusion set names every tensor the width
    // table marks ntok_dim. A graph that carries all four roots is the case
    // that would have failed while the exclusion had two slots.
    //
    // Fails when: the table records a root the key does not name, or the key
    // names a root the table did not record.
    {
        synth_params p;
        synth_graph g;
        if (!synth_build(g, p, 0)) {
            std::fprintf(stderr, "FAIL: invariant: synth build\n");
            ok = false;
        } else {
            llama_union_width_table table;
            if (!synth_table(g, p, table) || !table.has_ntok) {
                std::fprintf(stderr, "FAIL: invariant: the table recorded no token lane\n");
                ok = false;
            } else {
                // A table is rebuilt on every record build, so build() must
                // leave no trace of the previous one. ntok_roots is a vector,
                // and a vector that survived clear() would grow by one root
                // per build; token_lane_roots_match() would then see more
                // roots than the key names and the matrix would go red for a
                // reason that has nothing to do with the key.
                //
                // Fails when: clear() stops clearing ntok_roots.
                const size_t first_size = table.ntok_roots.size();
                if (!synth_table(g, p, table) || !table.has_ntok) {
                    std::fprintf(stderr, "FAIL: invariant: the second table build failed\n");
                    ok = false;
                } else if (table.ntok_roots.size() != first_size) {
                    std::fprintf(stderr,
                            "FAIL: invariant: a rebuilt table recorded %zu root(s), the first\n"
                            "      build recorded %zu; clear() left a stale root set\n",
                            table.ntok_roots.size(), first_size);
                    ok = false;
                }
                if (table.ntok_roots.size() != (size_t) LLAMA_TOKEN_LANE_ROOTS_N) {
                    std::fprintf(stderr,
                            "FAIL: invariant: the table recorded %zu root(s), want %d\n",
                            table.ntok_roots.size(), LLAMA_TOKEN_LANE_ROOTS_N);
                    ok = false;
                }
                const llama_topology_key_excl e = llama_topology_key_excl::token_lane();
                if (!e.token_lane_roots_match(table.ntok_roots)) {
                    std::fprintf(stderr,
                            "FAIL: invariant: the key's exclusion set and the table's recorded\n"
                            "      token roots differ (key=%d, table=%zu)\n",
                            e.count, table.ntok_roots.size());
                    ok = false;
                }
                // The consequence, measured: the key does not move when the
                // token width does, under the exclusion the record build
                // derives from the shape map. The four-root set alone is NOT
                // enough here -- the token width also reaches derived nodes
                // (tok_rows, q, sum, kq, sm, rows), which carry no runtime
                // name -- so the exclusion is extended with the axes the width
                // moves, derived from two builds exactly as the coefficient
                // table names them.
                llama_topology_key_excl e_full = e;
                {
                    synth_params p1 = p, p5 = p;
                    p1.n_tokens = 1;
                    p5.n_tokens = 5;
                    synth_key_graph k1, k5;
                    if (synth_key_build(k1, p1) && synth_key_build(k5, p5)) {
                        const llama_topology_key_excl extra = synth_key_excl(k1, k5);
                        for (const auto & nd : extra.extra) {
                            e_full.add_name(nd.name, nd.d);
                        }
                    } else {
                        std::fprintf(stderr, "FAIL: invariant: graph build for the exclusion\n");
                        ok = false;
                    }
                    synth_key_free(k1);
                    synth_key_free(k5);
                }
                for (int64_t nt : { 1, 5, 300 }) {
                    synth_params q = p;
                    q.n_tokens = nt;
                    synth_key_graph kg;
                    if (!synth_key_build(kg, q)) {
                        std::fprintf(stderr, "FAIL: invariant: graph build at n_tokens=%" PRId64 "\n", nt);
                        ok = false;
                        synth_key_free(kg);
                        continue;
                    }
                    const uint64_t h = llama_topology_key_of(kg.gf, e_full).hash;
                    if (nt == 1) {
                        ref_key_hash = h;
                    } else if (h != ref_key_hash) {
                        std::fprintf(stderr,
                                "FAIL: invariant: the key moved at n_tokens=%" PRId64 " with the\n"
                                "      map-derived exclusion on (%016" PRIx64 " vs %016" PRIx64 ")\n",
                                nt, h, ref_key_hash);
                        ok = false;
                    }
                    synth_key_free(kg);
                }
            }
        }
        synth_free(g);
    }

    // (i) THE NO-OP RULE'S LANE ANSWER. union_rebind_apply() does not compare
    // the live token width on a lane that cannot bind it; it compares the
    // recorded width. This part calls the SAME two functions the binder calls
    // (union_bound_ntok() and union_bind_is_noop(), both in
    // llama-nout-state.h) with a synthetic state, so removing the lane gate
    // turns it red.
    //
    // What would have been wrong without the gate: on a non-token lane the
    // graph's token inputs keep their BUILD shape, so a request at a different
    // token width is not a bind. Passing the live width would make the no-op
    // rule conclude "something to do" (false), let the call through, and the
    // patch would then be refused one branch later by the
    // `dT != 0 && !ntok_valid` guard - two places disagreeing about the lane,
    // and the key re-taken in between.
    {
        llama_nkv_ctx_state st;
        st.record_bound      = 100;
        st.record_bound_nout = 3;
        st.record_bound_ntok = 7;
        st.record_bound_rs_z = 0;
        st.record_bound_rs_head = 0;

        // non-token lane: the lane answer is the recorded width
        st.shape_map.ntok_valid = false;
        if (union_bound_ntok(st, 5) != 7) {
            std::fprintf(stderr,
                    "FAIL: no-op lane: a non-token lane answered %u for the token width, want 7 "
                    "(the recorded width)\n", (unsigned) union_bound_ntok(st, 5));
            ok = false;
        }
        // ... so a request that moves ONLY the token width is a no-op there
        if (!union_bind_is_noop(st, 100, 3, union_bound_ntok(st, 5), 0, 0)) {
            std::fprintf(stderr,
                    "FAIL: no-op lane: a non-token lane treated a token-width change as a bind\n");
            ok = false;
        }
        // ... and a request that moves n_kv is still a bind there
        if (union_bind_is_noop(st, 101, 3, union_bound_ntok(st, 5), 0, 0)) {
            std::fprintf(stderr, "FAIL: no-op lane: an n_kv change was called a no-op\n");
            ok = false;
        }

        // token lane: the lane answer is the live width
        st.shape_map.ntok_valid = true;
        if (union_bound_ntok(st, 5) != 5) {
            std::fprintf(stderr,
                    "FAIL: no-op lane: the token lane answered %u for the token width, want 5 "
                    "(the live width)\n", (unsigned) union_bound_ntok(st, 5));
            ok = false;
        }
        // ... so a token-width change IS a bind there
        if (union_bind_is_noop(st, 100, 3, union_bound_ntok(st, 5), 0, 0)) {
            std::fprintf(stderr,
                    "FAIL: no-op lane: the token lane called a token-width change a no-op\n");
            ok = false;
        }
        // ... and an unchanged width is still a no-op
        if (!union_bind_is_noop(st, 100, 3, union_bound_ntok(st, 7), 0, 0)) {
            std::fprintf(stderr, "FAIL: no-op lane: an unchanged request was called a bind\n");
            ok = false;
        }

        // THE DEFERRAL IS CAUGHT. Off the token lane the no-op rule reports
        // "nothing to do" for a token-width-only request (the assertion just
        // above). That is a deferral, not an acceptance: the request must not
        // be mistaken for an applied bind. union_rebind_apply's token-delta
        // guard is what turns it into a refusal, and this asserts the two
        // halves together.
        //
        // Fails when: the guard is removed, or it stops keying on ntok_valid.
        {
            // build base 7, request 5: dT = -2, and the lane cannot patch it
            const int64_t dT_non_token = (int64_t) 5 - (int64_t) 7;
            st.shape_map.ntok_valid = false;
            if (!union_bind_is_noop(st, 100, 3, union_bound_ntok(st, 5), 0, 0)) {
                std::fprintf(stderr,
                        "FAIL: deferral: the no-op rule did not defer a token-width change "
                        "off the token lane\n");
                ok = false;
            }
            if (!union_token_delta_refused(st, dT_non_token)) {
                std::fprintf(stderr,
                        "FAIL: deferral: the token-delta guard accepted dT=%" PRId64
                        " off the token lane; the deferral is uncaught\n", dT_non_token);
                ok = false;
            }
            // ... and a request that does not move the token width is not
            // refused by the guard, so the two rules do not overlap
            if (union_token_delta_refused(st, 0)) {
                std::fprintf(stderr,
                        "FAIL: deferral: the token-delta guard refused dT=0 off the token lane\n");
                ok = false;
            }

            // on the token lane the same delta is patchable: the guard passes
            st.shape_map.ntok_valid = true;
            if (union_token_delta_refused(st, dT_non_token)) {
                std::fprintf(stderr,
                        "FAIL: deferral: the token-delta guard refused dT=%" PRId64
                        " on the token lane\n", dT_non_token);
                ok = false;
            }
        }
    }


    // (j) FALSIFIABILITY: a runtime parameter that LEAKS into the hash must be
    // refused. Everything above asserts that the hash does not move; on its
    // own that is satisfiable by an exclusion that excludes too much, or by a
    // table that never checks anything. This arm drives the two ways the
    // token lane can leak and requires a refusal from each.
    //
    // The arm is not vacuous: it prints the observed value that made each case
    // fail, so a run where the guard silently disappeared shows the value it
    // would have had to reject.
    //
    // `leak_ok` alone would be vacuous in the other direction: it starts true
    // and is only ever cleared, so deleting this whole block would leave it
    // true and the gate green. The two `case_*` flags record that each case
    // REACHED its evidence line; the verdict printed below requires all three.
    bool leak_ok = true;
    bool case_mask_rows = false;
    bool case_token_dim = false;
    {
        // (j1) the mask's row count disagrees with the armed token width.
        // llama-union-width-table.h:200-206 refuses exactly this when the
        // lane is armed. Build at n_tokens = 4 with a mask of ne[1] = 5.
        synth_params p;
        p.n_tokens  = 4;
        p.mask_ntok = 5;
        synth_graph g;
        if (!synth_build(g, p, 0)) {
            std::fprintf(stderr, "FAIL: leak: synth build for the mask-row case\n");
            leak_ok = false;
        } else {
            llama_union_width_table t;
            const bool accepted = t.build(g.nodes, (uint32_t) p.n_kv,
                    (uint32_t) p.n_outputs, (uint32_t) p.n_rs, (uint32_t) p.rs_head,
                    (uint32_t) p.n_tokens, 1, 1);
            if (accepted) {
                std::fprintf(stderr,
                        "FAIL: leak: a mask of ne[1]=%" PRId64 " was accepted for an armed"
                        " token lane of width %" PRId64 "\n", p.mask_ntok, p.n_tokens);
                leak_ok = false;
            } else {
                std::fprintf(stderr, "FOUR_LEAK_NEG_CASE name=mask_rows observed=%" PRId64
                        " armed=%" PRId64 " refused=1\n", p.mask_ntok, p.n_tokens);
                case_mask_rows = true;
            }
        }
        synth_free(g);

        // (j2) the token dimension: the exclusion is what makes the widths
        // agree, and without it they must NOT agree.
        //
        // Two separate statements, each with its own control:
        //   - ONE skeleton at n_tokens = 4, hashed armed and disarmed. The
        //     two hashes must DIFFER. Same tensors, same graph, same
        //     everything except the exclusion, so the difference can only be
        //     the token dimension: this is the leak, isolated.
        //   - the pair (n_tokens = 4, 5): armed hashes must AGREE, disarmed
        //     hashes must DIFFER. That is the minimal claim — the exclusion
        //     is what makes two token widths hash the same.
        //
        // The exclusion argument of synth_hash() is a flag, not a width: it
        // sets ntok_dim when nonzero and does nothing when zero. Passing the
        // width itself would read as if the value mattered. It does not.
        synth_params q;
        q.n_tokens = 4;
        synth_params r = q;
        r.n_tokens = 5;

        synth_graph g4, g5;
        if (!synth_build(g4, q, 0)) {
            std::fprintf(stderr, "FAIL: leak: synth build at n_tokens=4\n");
            leak_ok = false;
        } else if (!synth_build(g5, r, 0)) {
            std::fprintf(stderr, "FAIL: leak: synth build at n_tokens=5\n");
            leak_ok = false;
        } else {
            const uint64_t arm_4 = 1, dis_4 = 0;   // the exclusion flag

            // (j2a) one skeleton, armed vs disarmed: they must differ
            uint64_t a4 = 0, d4 = 0;
            const bool h4_ok = synth_hash(g4, arm_4, a4) && synth_hash(g4, dis_4, d4);
            if (!h4_ok) {
                std::fprintf(stderr, "FAIL: leak: hash of the n_tokens=4 skeleton\n");
                leak_ok = false;
            } else if (a4 == d4) {
                std::fprintf(stderr,
                        "FAIL: leak: the exclusion changed nothing on one skeleton"
                        " (both %016" PRIx64 " at n_tokens=4)\n", a4);
                leak_ok = false;
            }

            // (j2b) the pair: armed agrees, disarmed differs. Only judged
            // when (g2a) produced two usable hashes; h4_ok false was already
            // reported and the pair would add nothing.
            uint64_t a5 = 0, d5 = 0;
            if (!synth_hash(g5, arm_4, a5) || !synth_hash(g5, dis_4, d5)) {
                std::fprintf(stderr, "FAIL: leak: hash of the n_tokens=5 skeleton\n");
                leak_ok = false;
            } else if (h4_ok) {
                const bool armed_agree     = a4 == a5;
                const bool disarmed_differ = d4 != d5;
                if (!armed_agree) {
                    std::fprintf(stderr,
                            "FAIL: leak: the token width reached the hash WITH the exclusion"
                            " (%016" PRIx64 " at n_tokens=4 vs %016" PRIx64 " at 5)\n", a4, a5);
                    leak_ok = false;
                }
                if (!disarmed_differ) {
                    std::fprintf(stderr,
                            "FAIL: leak: the token width did NOT reach the hash WITHOUT the"
                            " exclusion (both %016" PRIx64 "); the equality claim is vacuous\n", d4);
                    leak_ok = false;
                }
                if (armed_agree && disarmed_differ) {
                    std::fprintf(stderr,
                            "FOUR_LEAK_NEG_CASE name=token_dim armed_agree=1 disarmed_differ=1"
                            " armed=%016" PRIx64 " disarmed_4=%016" PRIx64
                            " disarmed_5=%016" PRIx64 "\n", a4, d4, d5);
                    case_token_dim = true;
                }
            }
        }
        synth_free(g4);
        synth_free(g5);
    }
    if (!leak_ok) {
        ok = false;
    }
    // The arm's own line, falsifiable in both directions: a case that never
    // ran leaves its flag false, and a case that ran and failed clears
    // leak_ok. Deleting the whole block clears both. The gate requires ok=1
    // here separately from FOUR_HASH_ARM, so the arm cannot be removed
    // without turning Gate FOURHASH red.
    const bool leak_verdict = leak_ok && case_mask_rows && case_token_dim;
    std::fprintf(stderr, "FOUR_LEAK_NEG ok=%d\n", int(leak_verdict));
    // widths is the token-width count (a), (b) and (c) actually walked. It is
    // n_tok_v, the size of the array they iterate, so the printed number
    // cannot drift from the sweep.
    std::fprintf(stderr, "FOUR_HASH_ARM ok=%d cases=%d widths=%d key_cases=%d\n",
            int(ok), cases, n_tok_v, key_cases);
    return ok;
}

// ---------------------------------------------------------------------------
// part 2: the four-parameter cross matrix
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

// one cell of the matrix: the tokens of this step, the output rows requested
// and the first position
struct four_cell {
    int      n_tokens  = 1;
    uint32_t n_outputs = 1;
    int64_t  pos0      = 0;
};

// n_batch and n_ubatch are the widest step: the graph is built at the active
// token width and bound in place afterwards, so the context must accept a
// batch as wide as the widest cell.
static llama_context * four_context(const live_model & m, int n_ctx, uint32_t n_ubatch) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx         = (uint32_t) n_ctx;
    cp.n_batch       = n_ubatch;
    cp.n_ubatch      = n_ubatch;
    cp.n_seq_max     = 1;
    cp.n_outputs_max = n_ubatch;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    // The token lane's own precondition (llama-context.cpp, token_candidate:
    // `cparams.kv_unified && cparams.n_seq_max == 1`). Without it the lane is
    // never armed, the width table records no token root, and the matrix
    // reports "missing runtime roots" for a graph that was never asked to
    // carry them. Both arms get the same params, so the comparison stays
    // apples-to-apples; with n_seq_max == 1 the unified cache is the same
    // single stream.
    cp.kv_unified      = true;
    return llama_init_from_model(m.model, cp);
}

// Decode one cell. `logits_out` receives every produced logits row, row-major,
// `n_outputs` rows of `n_vocab` floats.
static int four_decode(llama_context * ctx, const four_cell & c, int vocab, std::vector<float> & logits_out) {
    logits_out.assign((size_t) c.n_outputs * (size_t) vocab, 0.0f);

    llama_batch batch = llama_batch_init(c.n_tokens, 0, 1);
    batch.n_tokens = c.n_tokens;
    for (int i = 0; i < c.n_tokens; ++i) {
        batch.token[i]     = (llama_token) (1 + (c.pos0 + i) * 7 % (vocab - 1));
        batch.pos[i]       = (llama_pos) (c.pos0 + i);
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        // the LAST n_outputs tokens of the step are the output rows, so the
        // gather closure is a real subset of the batch
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

// The cells. The token widths are 1, 3, 2, 4, 7 and 5: not all equal, not a
// repeating power of two, and one of them is odd. n_kv grows monotonically
// (the KV span is the history length) and the output count moves independently
// of the token width and never exceeds it.
//
// The history must CROSS the 256-cell boundary, and the first schedule did
// not: 60 cells of at most 7 tokens total 220, so min_nkv was 1 and max_nkv
// 220 and the test's own `n_kv did not cross 256` assertion would have failed
// on every run. The schedule is therefore 60 cells of the six widths above
// (220 tokens) followed by 64 single-token cells, which puts the boundary at
// cell 60 and the last executed width at 284. The single-token tail also
// keeps n_tokens changing across the boundary, so the token axis is exercised
// on both sides of it.
// Three output counts are forced into the walk, because a schedule that never
// reaches them cannot prove anything about them:
//   - n_outputs = 0: no output row at all (every batch.logits[i] = 0). The
//     gather closure is empty and the graph must still execute;
//   - n_outputs = n_tokens: every token is an output, the full-gather case;
//   - n_outputs = n_ubatch: the widest gather the context was configured for
//     (n_outputs_max = n_ubatch). This only coincides with n_tokens on the
//     widest cell, so it is forced on its own.
// part2 counts the occurrences and fails if any is missing.
static std::vector<four_cell> four_cells() {
    std::vector<four_cell> cells;
    const int sizes[] = { 1, 3, 2, 4, 7, 5 };
    const int n_sizes = (int) (sizeof(sizes) / sizeof(sizes[0]));
    int pos = 0;
    for (int i = 0; i < 60; ++i) {
        four_cell c;
        c.n_tokens  = sizes[i % n_sizes];
        c.n_outputs = (uint32_t) (1 + (i % 3));
        if (c.n_outputs > (uint32_t) c.n_tokens) {
            c.n_outputs = (uint32_t) c.n_tokens;
        }
        // The three boundary counts are forced at fixed, distinct cells so
        // the walk stays deterministic and the stock arm sees the same order.
        // The indices are chosen against sizes[] = {1,3,2,4,7,5}: i % 6 == 4
        // is a 7-token cell, which is what the n_ubatch boundary needs.
        //   i = 4  (n_tokens = 7): n_outputs = 0
        //   i = 5  (n_tokens = 5): n_outputs = n_tokens, the full gather
        //   i = 10 (n_tokens = 7): n_outputs = 7 = n_ubatch = n_outputs_max
        if (i == 4) {
            c.n_outputs = 0;                        // no output row
        } else if (i == 5) {
            c.n_outputs = (uint32_t) c.n_tokens;    // every token is an output
        } else if (i == 10) {
            c.n_outputs = 7;                        // n_outputs == n_ubatch
            if (c.n_outputs > (uint32_t) c.n_tokens) {
                c.n_outputs = (uint32_t) c.n_tokens;
            }
        }
        c.pos0 = pos;
        pos += c.n_tokens;
        cells.push_back(c);
    }
    // the tail that crosses 256: 64 single-token cells, 220 -> 284
    for (int i = 0; i < 64; ++i) {
        four_cell c;
        c.n_tokens  = 1;
        c.n_outputs = 1;
        c.pos0      = pos;
        pos += c.n_tokens;
        cells.push_back(c);
    }
    return cells;
}

static bool part2(const live_model & m) {
    const int vocab = m.vocab;

    const std::vector<four_cell> cells = four_cells();
    int n_ctx = 0;
    int max_tokens = 1;
    for (const auto & c : cells) {
        n_ctx += c.n_tokens;
        if (c.n_tokens > max_tokens) {
            max_tokens = c.n_tokens;
        }
    }
    n_ctx += 8;
    const uint32_t n_ubatch = (uint32_t) max_tokens;

    // Environment audit for the two arms. Every variable below is read by the
    // llama_context CONSTRUCTOR, never by the runner, so setting it before a
    // context is built selects that context's arm for its whole lifetime.
    // Nothing here is unset while a context that read it is alive: an unset
    // would let one arm become the other and make the comparison meaningless.
    //   GGML_TP5_UNION_BINDER=1          binder arm (llama-context.cpp, the gate
    //                                    block after the CPU-default predicate).
    //   LLAMA_EXPERIMENTAL_VARLEN_NTOK=1 the token lane, which is what makes
    //                                    n_tokens a runtime parameter here.
    //   LLAMA_GRAPH_REUSE_DISABLE        unset for the binder arm, set for the
    //                                    stock arm (llama-context.cpp, the
    //                                    environment audit block).
    //   LLAMA_NKV_MASK_OVERRIDE          unset: it is part 3's hook, and an
    //                                    inherited value would change the mask
    //                                    width this matrix observes.
    setenv("GGML_TP5_UNION_BINDER", "1", 1);
    setenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK", "1", 1);
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    unsetenv("LLAMA_NKV_MASK_OVERRIDE");

    llama_context * ctx = four_context(m, n_ctx, n_ubatch);
    if (ctx == nullptr) {
        std::fprintf(stderr, "FAIL: four-param: binder context init\n");
        return false;
    }

    // The stock arm: one context, same history, every cell rebuilt. The flag
    // is set before this context is constructed and is not unset while it
    // lives; the constructor already read it.
    setenv("LLAMA_GRAPH_REUSE_DISABLE", "1", 1);
    llama_context * stock = four_context(m, n_ctx, n_ubatch);
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    if (stock == nullptr) {
        std::fprintf(stderr, "FAIL: four-param: stock context init\n");
        llama_free(ctx);
        return false;
    }

    bool ok = true;
    uint64_t ref_hash = 0;
    bool have_ref = false;
    uint32_t build_n_kv = 0, build_n_outputs = 0, build_n_rs = 0, build_n_tokens = 0;
    // Rebuild counts are measured, never guessed: the delta of each context's
    // own model.build_graph() counter. `binder_builds_first` is the count
    // after cell 0's decode, which is where the binder arm's initial build
    // happens; the binder's rebuild delta starts there. `stock_builds_first`
    // is the same point on the stock arm, used only to compare the two arms
    // over the same post-first-cell interval.
    uint64_t binder_builds0 = 0, binder_builds1 = 0;
    uint64_t stock_builds0  = 0, stock_builds1  = 0;
    uint64_t binder_builds_first = 0;
    uint64_t stock_builds_first  = 0;
    int bitwise = 0;
    int bitwise_cells = 0;
    int rs_z_moves = 0;
    int32_t last_rs_z = -3;
    int64_t  min_nkv = INT64_MAX, max_nkv = 0;
    int      distinct_tokens = 0;
    int      last_tokens = -1;
    // Boundary occurrences. A cell that a schedule change silently drops is
    // a claim that silently stops being tested, so each boundary is counted
    // and part2 fails when the count is zero.
    int      cells_nout_zero = 0;   // n_outputs == 0
    int      cells_nout_full = 0;   // n_outputs == n_tokens (full gather)
    int      cells_nout_max  = 0;   // n_outputs == n_ubatch == n_outputs_max
    int      cells_ntok_one  = 0;   // n_tokens == 1
    int      cells_ntok_ub   = 0;   // n_tokens == n_ubatch
    // The mask read-back's census. `mask_cells` counts the cells whose stats
    // reported a read-back at all; `mask_rows_checked` sums the rows compared,
    // `mask_rows_ok` the rows that agreed, and `mask_cells_all` the cells that
    // compared EVERY row of their batch. A cell with checked == 0, a refused
    // row, or mask_rows_all == 0 fails the per-cell assertion, so these
    // numbers can only agree with the cell count; they are printed so the
    // agreement is visible.
    //
    // `mask_rows_expect` is the denominator the census owes: every cell must
    // compare n_tokens rows for every mask it filled, so the total is the sum
    // of `n_tokens * mask_filled` over the cells. Without it `checked >= cells`
    // is the only lower bound, and a read-back that compares exactly one row
    // per cell satisfies it. A cell that compares fewer rows than it owns
    // fails below, so the printed `mask_rows=<ok>/<checked>` can no longer be
    // smaller than the batches actually decoded.
    int      mask_cells        = 0;
    int      mask_rows_checked = 0;
    int      mask_rows_ok      = 0;
    int      mask_rows_expect  = 0;
    int      mask_cells_all    = 0;

    // The axis census: how many DISTINCT values each axis actually took in
    // this run, counted from the cells the loop executed and from the values
    // the binder arm reported, not from the schedule the test intended. The
    // FOUR_AXES line reports them, so a run that crossed three axes cannot be
    // read as a four-axis pass.
    std::vector<uint32_t> seen_nout, seen_ntok;
    std::vector<int32_t>  seen_rsz;
    auto note = [](std::vector<uint32_t> & v, uint32_t x) {
        for (uint32_t y : v) { if (y == x) return; }
        v.push_back(x);
    };
    auto note_rsz = [](std::vector<int32_t> & v, int32_t x) {
        for (int32_t y : v) { if (y == x) return; }
        v.push_back(x);
    };

    // Baselines for the rebuild deltas. Taken after both contexts exist, so
    // the constructor/reserve builds are not counted as cell rebuilds. The
    // binder arm's initial build happens inside cell 0's decode, so its
    // rebuild interval starts at `binder_builds_first` below, not here.
    binder_builds0 = llama_graph_build_count(ctx);
    stock_builds0  = llama_graph_build_count(stock);
    binder_builds_first = binder_builds0;
    stock_builds_first  = stock_builds0;
    std::vector<float> got, want;

    for (size_t i = 0; i < cells.size() && ok; ++i) {
        const four_cell & c = cells[i];
        if (four_decode(ctx, c, vocab, got) != 0) {
            std::fprintf(stderr, "FAIL: four-param: decode cell=%zu\n", i);
            ok = false;
            break;
        }
        if (i == 0) {
            // The binder arm's initial graph build happens inside this
            // decode. Record the counter before the per-cell checks, so any
            // later build in this cell is counted as a rebuild.
            binder_builds_first = llama_graph_build_count(ctx);
        }
        llama_nkv_runtime_stats st = {};
        if (!llama_get_nkv_runtime_stats(ctx, &st)) {
            std::fprintf(stderr, "FAIL: four-param: no stats at cell=%zu\n", i);
            ok = false;
            break;
        }
        if (!st.topology_hashed) {
            std::fprintf(stderr, "FAIL: four-param: topology not hashed at cell=%zu\n", i);
            ok = false;
            break;
        }
        if (!st.has_nkv_dims || !st.has_nout_dims || !st.has_ntok_dims) {
            std::fprintf(stderr,
                    "FAIL: four-param: missing runtime roots at cell=%zu (nkv=%d nout=%d ntok=%d)\n",
                    i, int(st.has_nkv_dims), int(st.has_nout_dims), int(st.has_ntok_dims));
            ok = false;
            break;
        }
        if (!have_ref) {
            ref_hash         = st.topology_hash;
            build_n_kv       = st.build_n_kv;
            build_n_outputs  = st.build_n_outputs;
            build_n_rs       = st.bound_n_rs;
            build_n_tokens   = st.build_n_tokens;
            have_ref         = true;
        } else {
            if (st.topology_hash != ref_hash) {
                std::fprintf(stderr, "FAIL: four-param: hash moved at cell=%zu (%016" PRIx64 " -> %016" PRIx64 ")\n",
                        i, ref_hash, st.topology_hash);
                ok = false;
                break;
            }
            // A rebuild shows up as a changed build width on any axis. The
            // token axis is the one this test is about: a rebuild there is
            // exactly the defect the topology exclusion removes.
            if (st.build_n_kv != build_n_kv || st.build_n_outputs != build_n_outputs ||
                    st.bound_n_rs != build_n_rs || st.build_n_tokens != build_n_tokens) {
                std::fprintf(stderr,
                        "FAIL: four-param: rebuild at cell=%zu (n_kv %u->%u, n_outputs %u->%u, "
                        "n_rs %u->%u, n_tokens %u->%u)\n",
                        i, build_n_kv, st.build_n_kv, build_n_outputs, st.build_n_outputs,
                        build_n_rs, st.bound_n_rs, build_n_tokens, st.build_n_tokens);
                ok = false;
                break;
            }
        }
        // the executed parameters must equal what this cell asked for
        if (st.bound_n_outputs != c.n_outputs) {
            std::fprintf(stderr, "FAIL: four-param: n_outputs not applied at cell=%zu (want %u got %u)\n",
                    i, c.n_outputs, st.bound_n_outputs);
            ok = false;
            break;
        }
        if ((int64_t) st.runtime_n_kv != (int64_t) c.pos0 + c.n_tokens) {
            std::fprintf(stderr, "FAIL: four-param: n_kv not applied at cell=%zu (want %" PRId64 " got %u)\n",
                    i, (int64_t) c.pos0 + c.n_tokens, st.runtime_n_kv);
            ok = false;
            break;
        }
        // the token width must be the one this cell asked for, and it must be
        // the BOUND width, not the build width: a graph still at the build
        // width would compute the wrong number of rows
        if (st.bound_n_tokens != (uint32_t) c.n_tokens) {
            std::fprintf(stderr, "FAIL: four-param: token width not applied at cell=%zu (want %d got %u)\n",
                    i, c.n_tokens, st.bound_n_tokens);
            ok = false;
            break;
        }
        // the graph must execute at the width it reports, not only report it
        if (st.bound_n_kv != st.runtime_n_kv) {
            std::fprintf(stderr, "FAIL: four-param: bound n_kv %u != runtime n_kv %u at cell=%zu\n",
                    st.bound_n_kv, st.runtime_n_kv, i);
            ok = false;
            break;
        }
        // The mask observation, on EVERY cell including cell 0. Three facts
        // about the fill, then the read-back that makes the width claim
        // falsifiable:
        //   - the fill did not refuse;
        //   - it filled at least one mask;
        //   - the fill consumed exactly the width the graph executes at.
        // The fourth fact is the read-back: the row the last batch token
        // consumes was compared, column by column below that width, with the
        // visibility the KV cache's own cells imply. mask_n_kv == runtime_n_kv
        // alone is binder bookkeeping -- both numbers come from the width the
        // binder wrote -- so it is the read-back that makes the width claim
        // falsifiable. checked == 0 is a FAIL, not a skip: this lane builds an
        // F32 mask (flash attention is off in four_context()), so a graph that
        // checked nothing is a graph whose mask nobody looked at.
        if (st.mask_refused != 0) {
            std::fprintf(stderr, "FAIL: four-param: the mask fill refused at cell=%zu (refused=%u)\n",
                    i, st.mask_refused);
            ok = false;
            break;
        }
        if (st.mask_filled == 0) {
            std::fprintf(stderr, "FAIL: four-param: no KQ mask filled at cell=%zu\n", i);
            ok = false;
            break;
        }
        if (st.mask_n_kv != st.runtime_n_kv) {
            std::fprintf(stderr,
                    "FAIL: four-param: mask filled at %u, graph executes at %u (cell=%zu)\n",
                    st.mask_n_kv, st.runtime_n_kv, i);
            ok = false;
            break;
        }
        if (st.mask_row_checked == 0) {
            std::fprintf(stderr,
                    "FAIL: four-param: the mask row was not read back at cell=%zu "
                    "(filled=%u checked=0)\n", i, st.mask_filled);
            ok = false;
            break;
        }
        if (st.mask_row_ok != st.mask_row_checked) {
            std::fprintf(stderr,
                    "FAIL: four-param: the mask row disagrees with the KV cells at cell=%zu "
                    "(ok=%u checked=%u mask_n_kv=%u runtime_n_kv=%u)\n",
                    i, st.mask_row_ok, st.mask_row_checked, st.mask_n_kv, st.runtime_n_kv);
            ok = false;
            break;
        }
        // A refusal is its own failure: the check declined to model this
        // layout, so nothing was compared. Letting it ride on `checked == 0`
        // would make a layout change look like a mask that is wrong.
        if (st.mask_row_refused != 0) {
            std::fprintf(stderr,
                    "FAIL: four-param: the mask read-back refused to model %u row(s) at cell=%zu "
                    "(n_tokens=%d)\n", st.mask_row_refused, i, c.n_tokens);
            ok = false;
            break;
        }
        // Every row of this batch must have been compared. The schedule's
        // widest cell is 7 tokens, below the read-back's row cap of 8, so a
        // cell that reports mask_rows_all=0 means the cap or the batch changed
        // and the matrix is silently observing only the last row.
        if (st.mask_rows_all != 1) {
            std::fprintf(stderr,
                    "FAIL: four-param: only the last mask row was compared at cell=%zu "
                    "(n_tokens=%d, mask_rows_all=0)\n", i, c.n_tokens);
            ok = false;
            break;
        }
        // The row count is pinned per cell, which is the only strength that
        // makes the census falsifiable: this batch owns n_tokens rows for
        // every mask the fill registered, so the read-back must have compared
        // exactly that many. `checked >= cells` cannot see a read-back that
        // compares one row per cell; this can, and the cell is named.
        const int cell_rows_expect = c.n_tokens * (int) st.mask_filled;
        if ((int) st.mask_row_checked != cell_rows_expect) {
            std::fprintf(stderr,
                    "FAIL: four-param: the mask read-back compared %u rows at cell=%zu, "
                    "the batch owns %d (n_tokens=%d mask_filled=%u mask_rows_all=%u)\n",
                    st.mask_row_checked, i, cell_rows_expect, c.n_tokens,
                    st.mask_filled, st.mask_rows_all);
            ok = false;
            break;
        }
        ++mask_cells;
        mask_rows_checked += (int) st.mask_row_checked;
        mask_rows_ok      += (int) st.mask_row_ok;
        mask_rows_expect  += cell_rows_expect;
        if (st.mask_rows_all == 1) {
            ++mask_cells_all;
        }
        if (st.bound_rs_z != last_rs_z) {
            if (last_rs_z != -3) {
                ++rs_z_moves;
            }
            last_rs_z = st.bound_rs_z;
        }
        if ((int64_t) st.runtime_n_kv < min_nkv) min_nkv = (int64_t) st.runtime_n_kv;
        if ((int64_t) st.runtime_n_kv > max_nkv) max_nkv = (int64_t) st.runtime_n_kv;
        if (last_tokens != c.n_tokens) {
            ++distinct_tokens;
            last_tokens = c.n_tokens;
        }
        // Boundary occurrences, counted from the cell that was actually
        // decoded, not from the schedule that was intended.
        if (c.n_outputs == 0) {
            ++cells_nout_zero;
        }
        if (c.n_outputs == (uint32_t) c.n_tokens) {
            ++cells_nout_full;
        }
        if (c.n_outputs == n_ubatch) {
            ++cells_nout_max;
        }
        if (c.n_tokens == 1) {
            ++cells_ntok_one;
        }
        if ((uint32_t) c.n_tokens == n_ubatch) {
            ++cells_ntok_ub;
        }
        note(seen_nout, st.bound_n_outputs);
        note(seen_ntok, st.bound_n_tokens);
        note_rsz(seen_rsz, st.bound_rs_z);

        // --- stock arm: the SAME history, walked once -------------------
        //
        // One stock context lives for the whole loop and decodes each cell
        // exactly once, in the same order the binder arm decodes them, so its
        // state after cell i is the same history and the comparison is
        // unchanged. It is O(cells), not O(cells^2).
        const int rc_stock = four_decode(stock, c, vocab, want);
        if (rc_stock != 0) {
            std::fprintf(stderr, "FAIL: four-param: stock decode cell=%zu (rc=%d)\n", i, rc_stock);
            ok = false;
            break;
        }
        if (i == 0) {
            // The stock arm's cell-0 build is one of its per-cell builds;
            // record the post-decode count for the same-interval comparison.
            stock_builds_first = llama_graph_build_count(stock);
        }
        // The bitwise denominator counts only cells that REQUESTED an output
        // row. With n_outputs == 0 both buffers are empty and memcmp on two
        // empty ranges returns 0, so counting that cell would be a tautology:
        // it would "pass" without comparing anything. The zero-output cell is
        // proven separately, by its own line, from the decode return code and
        // the bound width.
        if (c.n_outputs == 0) {
            if (rc_stock != 0) {
                std::fprintf(stderr, "FAIL: four-param: stock decode at the zero-output cell=%zu\n", i);
                ok = false;
            } else {
                std::fprintf(stderr,
                        "FOUR_NOUT_ZERO cell=%zu rc=0 bound_n_outputs=%u n_tokens=%d\n",
                        i, st.bound_n_outputs, c.n_tokens);
            }
        } else {
            ++bitwise_cells;
            if (got.size() == want.size() && std::memcmp(got.data(), want.data(),
                        got.size() * sizeof(float)) == 0) {
                ++bitwise;
            } else {
                std::fprintf(stderr, "FAIL: four-param: logits differ from stock at cell=%zu (n_kv=%" PRId64
                        " n_tokens=%d n_outputs=%u)\n", i, (int64_t) c.pos0 + c.n_tokens, c.n_tokens, c.n_outputs);
                ok = false;
            }
        }
    }

    // the axes must actually have moved
    if (ok) {
        if (min_nkv >= 256 || max_nkv <= 256) {
            std::fprintf(stderr, "FAIL: four-param: n_kv did not cross 256 (min=%" PRId64 " max=%" PRId64 ")\n",
                    min_nkv, max_nkv);
            ok = false;
        }
        // The mask read-back must have run on EVERY cell the loop executed,
        // not only on the ones it did not break out of: a read-back that
        // silently stopped after the first cell would still print ok=1.
        if (mask_cells != (int) cells.size()) {
            std::fprintf(stderr,
                    "FAIL: four-param: the mask read-back ran on %d of %zu cells\n",
                    mask_cells, cells.size());
            ok = false;
        }
        if (mask_rows_checked == 0 || mask_rows_ok != mask_rows_checked) {
            std::fprintf(stderr,
                    "FAIL: four-param: the mask read-back agreed on %d of %d rows\n",
                    mask_rows_ok, mask_rows_checked);
            ok = false;
        }
        // Every cell of this schedule is at most 7 tokens, below the read-back's
        // row cap, so every cell must have compared all of its rows. A cell
        // that compared only the last row means the cap or the batch changed.
        if (mask_cells_all != mask_cells) {
            std::fprintf(stderr,
                    "FAIL: four-param: only %d of %d cells compared every mask row\n",
                    mask_cells_all, mask_cells);
            ok = false;
        }
        // The token axis must have moved. A matrix whose token width never
        // changed would prove nothing about the token lane.
        if (distinct_tokens < 2) {
            std::fprintf(stderr, "FAIL: four-param: the token width never moved (%d distinct value(s))\n",
                    distinct_tokens);
            ok = false;
        }
        // The boundary cells must have occurred. Each is a claim the matrix
        // makes; a schedule edit that drops one would otherwise leave the
        // claim untested while the run still printed ok=1.
        struct boundary {
            const char * name;
            int          seen;
            const char * want;
        };
        const boundary bounds[] = {
            { "n_outputs=0",       cells_nout_zero, "no output row at all" },
            { "n_outputs=n_tokens", cells_nout_full, "the full gather" },
            { "n_outputs=n_ubatch", cells_nout_max,  "the widest gather" },
            { "n_tokens=1",        cells_ntok_one,  "the shortest step" },
            { "n_tokens=n_ubatch", cells_ntok_ub,   "the widest step" },
        };
        for (const boundary & b : bounds) {
            if (b.seen == 0) {
                std::fprintf(stderr,
                        "FAIL: four-param: the boundary %s never occurred (%s)\n",
                        b.name, b.want);
                ok = false;
            }
        }
        // The census, read from the run. n_kv is reported as the SPAN the
        // executed width covered (max - min + 1), not as a distinct-value
        // count: the width is the cumulative token count, so the two are the
        // same number here and the span is the one that is cheap to state
        // honestly. n_outputs, rs_z and n_tokens are distinct-value counts of
        // what the binder arm reported.
        const int n_kv_distinct = (int) (max_nkv - min_nkv + 1);   // the span, not the count
        const int n_nout       = (int) seen_nout.size();
        const int n_rsz        = (int) seen_rsz.size();
        const int n_ntok       = (int) seen_ntok.size();
        if (last_rs_z == -2) {
            // no recurrent state: rs_z is INERT, and the line says so. The
            // three axes that did move are the ones the run may claim.
            std::fprintf(stderr,
                    "FOUR_AXES n_kv=%d n_outputs=%d rs_z=0 n_tokens=%d reason=dense_model\n",
                    n_kv_distinct, n_nout, n_ntok);
        } else if (rs_z_moves == 0) {
            std::fprintf(stderr, "FAIL: four-param: rs_z never moved on a recurrent model\n");
            ok = false;
        } else {
            std::fprintf(stderr,
                    "FOUR_AXES n_kv=%d n_outputs=%d rs_z=%d n_tokens=%d reason=recurrent_model\n",
                    n_kv_distinct, n_nout, n_rsz, n_ntok);
        }
    }

    // The rebuild deltas. This is the assertion that makes the two arms
    // genuinely different paths: the binder arm must perform exactly its
    // initial build and no rebuild after it, and the stock arm must rebuild
    // for every cell.
    //
    // The two printed counts use different intervals on purpose, because the
    // arms have different initial builds:
    //   - `rebuilds` counts the binder from AFTER cell 0's decode. Cell 0's
    //     build is the initial graph the later cells reuse, so it is not a
    //     rebuild; `binder_initial` below proves it happened exactly once.
    //   - `stock_rebuilds` counts the stock arm from the pre-loop baseline
    //     over every cell, including cell 0. With reuse disabled every cell
    //     builds, and the gate's `>= cells` check wants that full count.
    // The same-interval comparison at the end uses `stock_after_initial`,
    // which starts where `rebuilds` does.
    binder_builds1 = llama_graph_build_count(ctx);
    stock_builds1  = llama_graph_build_count(stock);
    const uint64_t binder_initial      = binder_builds_first - binder_builds0;
    const uint64_t rebuilds            = binder_builds1 - binder_builds_first;
    const uint64_t stock_rebuilds      = stock_builds1  - stock_builds0;
    const uint64_t stock_after_initial = stock_builds1  - stock_builds_first;
    if (ok) {
        if (binder_initial != 1) {
            std::fprintf(stderr,
                    "FAIL: four-param: the binder arm built %" PRIu64 " time(s) in its first cell; "
                    "expected exactly the initial build\n", binder_initial);
            ok = false;
        }
        if (rebuilds != 0) {
            std::fprintf(stderr,
                    "FAIL: four-param: the binder arm rebuilt %" PRIu64 " time(s) after its initial build\n",
                    rebuilds);
            ok = false;
        }
        if (stock_rebuilds < (uint64_t) cells.size()) {
            std::fprintf(stderr,
                    "FAIL: four-param: the stock arm rebuilt %" PRIu64 " time(s) for %zu cells "
                    "(LLAMA_GRAPH_REUSE_DISABLE did not take effect)\n",
                    stock_rebuilds, cells.size());
            ok = false;
        }
        if (rebuilds == stock_after_initial) {
            std::fprintf(stderr,
                    "FAIL: four-param: both arms rebuilt %" PRIu64 " time(s) after their first cell; "
                    "they took the same path\n", rebuilds);
            ok = false;
        }
    }

    // bitwise=<n>/<m>: m counts only the cells that requested an output row,
    // so m == cells - nout_zero. The zero-output cell is reported in its own
    // field instead of being counted as a comparison it did not make.
    //
    // mask_cells=, mask_rows=, mask_expect= and mask_cells_all= are the mask
    // observation's own census: how many cells had their mask rows read back
    // against the KV cells, how many rows agreed at every column out of how
    // many were compared, how many rows the batches owed, and how many cells
    // compared EVERY row of their batch rather than only the last one.
    // mask_expect is the denominator: n_tokens * mask_filled summed over the
    // cells. The per-cell assertion above already forces checked == expect, so
    // printing both is what lets a reader (and the gate) see that the equality
    // was not reached by comparing nothing.
    std::fprintf(stderr,
            "FOUR_MATRIX ok=%d cells=%zu hash=%016" PRIx64
            " rebuilds=%" PRIu64 " stock_rebuilds=%" PRIu64
            " bitwise=%d/%d nout_zero=%d"
            " mask_cells=%d mask_rows=%d/%d mask_expect=%d mask_cells_all=%d\n",
            int(ok), cells.size(), ref_hash, rebuilds, stock_rebuilds,
            bitwise, bitwise_cells, cells_nout_zero,
            mask_cells, mask_rows_ok, mask_rows_checked, mask_rows_expect, mask_cells_all);

    llama_free(stock);
    llama_free(ctx);
    return ok;
}

// ---------------------------------------------------------------------------
// part 3: the mask path MUST be able to fail
// ---------------------------------------------------------------------------
//
// The child: one decode with the runtime width forced above the mask's ne[0].
// It must abort. Reaching the return means the guard did not fire, and the exit
// code says so. GGML_ABORT kills the process, so the case runs in a child.
static int mask_negative_child(const char * model_path) {
    setenv("GGML_TP5_UNION_BINDER", "1", 1);
    setenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK", "1", 1);
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");

    live_model m;
    if (!live_open(model_path, m)) {
        return 3;
    }
    llama_context * ctx = four_context(m, 16, 4);
    if (ctx == nullptr) {
        live_close(m);
        return 4;
    }
    four_cell c;
    c.n_tokens  = 1;
    c.n_outputs = 1;
    c.pos0      = 0;
    std::vector<float> logits;
    (void) four_decode(ctx, c, m.vocab, logits);
    llama_free(ctx);
    live_close(m);
    std::fprintf(stderr, "FAIL: the mask guard did not fire on an over-wide runtime n_kv\n");
    return 5;
}

// The read-back's negative child: a fill that succeeds, then a mask row that
// does not match the cells. The corruption is written into the row the check
// reads (the last column of the last token's row) and the same public
// observation the matrix uses is taken again. The check must report the
// disagreement; reaching the end without it means the read-back cannot fail,
// and the exit code says so.
static int mask_row_negative_child(const char * model_path) {
    setenv("GGML_TP5_UNION_BINDER", "1", 1);
    setenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK", "1", 1);
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    unsetenv("LLAMA_NKV_MASK_OVERRIDE");

    live_model m;
    if (!live_open(model_path, m)) {
        return 3;
    }
    llama_context * ctx = four_context(m, 16, 4);
    if (ctx == nullptr) {
        live_close(m);
        return 4;
    }
    four_cell c;
    c.n_tokens  = 1;
    c.n_outputs = 1;
    c.pos0      = 0;
    std::vector<float> logits;
    if (four_decode(ctx, c, m.vocab, logits) != 0) {
        llama_free(ctx);
        live_close(m);
        return 6;
    }
    llama_nkv_runtime_stats st = {};
    if (!llama_get_nkv_runtime_stats(ctx, &st)) {
        llama_free(ctx);
        live_close(m);
        return 7;
    }
    if (st.mask_row_checked == 0 || st.mask_row_ok != st.mask_row_checked) {
        // The clean run must agree before the corruption means anything: a
        // check that fails on a clean row would pass this arm for the wrong
        // reason.
        std::fprintf(stderr, "FAIL: mask row negative: the clean row did not agree "
                "(ok=%u checked=%u)\n", st.mask_row_ok, st.mask_row_checked);
        llama_free(ctx);
        live_close(m);
        return 8;
    }
    const uint32_t width = st.mask_n_kv;
    if (width == 0) {
        llama_free(ctx);
        live_close(m);
        return 9;
    }
    float * mask_data = nullptr;
    int64_t ne0 = 0;
    // Row index 0 is the only row of a one-token batch, and it is the row the
    // check reads (tok = n_tokens - 1 = 0).
    if (!llama_union_kq_mask_row(ctx, 0, &mask_data, &ne0) || mask_data == nullptr) {
        llama_free(ctx);
        live_close(m);
        return 10;
    }
    // The fill masks every column at or beyond the runtime width with
    // -INFINITY, so a clean row must carry that value in [width, ne0). If it
    // does not, the row is not the shape this arm assumes and the corruption
    // below would prove nothing.
    for (int64_t i = (int64_t) width; i < ne0; ++i) {
        if (mask_data[i] != -INFINITY) {
            std::fprintf(stderr,
                    "FAIL: mask row negative: column %" PRId64 " of the row is not -INFINITY\n", i);
            llama_free(ctx);
            live_close(m);
            return 13;
        }
    }
    // The same public function the arm's verdict comes from, on the CLEAN
    // row, must agree first. The stats above are the fill's own counters; a
    // read-back that returned 0 unconditionally would satisfy the corruption
    // arm below for the wrong reason. Two-sided measurement, one process.
    if (llama_union_mask_row_check(ctx) != LLAMA_GATE_PASS) {
        std::fprintf(stderr, "FAIL: the read-back refused a clean mask row\n");
        llama_free(ctx);
        live_close(m);
        return 12;
    }
    // Corrupt the row the read-back reads. The check reads the LAST batch
    // token's row, and computes that row's offset itself (row i at
    // width*i in the fill's layout). The one-token decode here has exactly
    // one row, so the row is at offset 0 and the last column below the width
    // is `width - 1`. The corruption is a value the fill never writes: not
    // 0.0f, not -INFINITY. Keep the original so the row can be put back.
    const float original = mask_data[width - 1];
    mask_data[width - 1] = 1.0f;

    const enum llama_gate_verdict ok = llama_union_mask_row_check(ctx);
    if (ok != LLAMA_GATE_FAIL) {
        // The read-back accepted a row the fill never wrote, or refused to
        // model it. Neither is detection: the arm fails, and the exit code
        // carries it.
        std::fprintf(stderr,
                "FAIL: the read-back did not report the corrupt mask row (verdict=%d)\n", (int) ok);
        mask_data[width - 1] = original;
        llama_free(ctx);
        live_close(m);
        return 11;
    }
    // Put the column back: the same row, the same check, must agree again.
    mask_data[width - 1] = original;
    const enum llama_gate_verdict restored = llama_union_mask_row_check(ctx);
    if (restored != LLAMA_GATE_PASS) {
        std::fprintf(stderr,
                "FAIL: the read-back did not agree again after the row was restored (verdict=%d)\n",
                (int) restored);
        llama_free(ctx);
        live_close(m);
        return 14;
    }
    // Exit 0 means the corruption WAS detected and the restored row agrees.
    // The value line records both verdicts as numbers, so the pair is on the
    // record without reading prose; the sentence below is the phrase the
    // parent greps for, and without it the parent cannot tell detection from a
    // child that died before it checked anything.
    std::fprintf(stderr, "FOUR_MASK_ROW_NEG_CHILD clean=1 corrupt=0 restored=1\n");
    std::fprintf(stderr, "the mask row disagrees with the KV cells\n");
    llama_free(ctx);
    live_close(m);
    return 0;
}

// The read-back's tri-state child: the three verdicts must be distinct and
// reachable in one process. Before any decode there is no live mask, so the
// check must report UNPROVEN. After one decode it must report PASS. After the
// row is corrupted it must report FAIL. A collapse of any two states changes
// the printed line, so the parent's literal grep fails.
static int mask_unproven_child(const char * model_path) {
    setenv("GGML_TP5_UNION_BINDER", "1", 1);
    setenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK", "1", 1);
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    unsetenv("LLAMA_NKV_MASK_OVERRIDE");

    live_model m;
    if (!live_open(model_path, m)) {
        return 3;
    }
    llama_context * ctx = four_context(m, 16, 4);
    if (ctx == nullptr) {
        live_close(m);
        return 4;
    }

    // (a) no decode yet: nothing was filled, so nothing can be compared.
    const enum llama_gate_verdict before = llama_union_mask_row_check(ctx);
    if (before != LLAMA_GATE_UNPROVEN) {
        std::fprintf(stderr, "FAIL: the read-back before any decode reported %d, want UNPROVEN (%d)\n",
                (int) before, (int) LLAMA_GATE_UNPROVEN);
        llama_free(ctx);
        live_close(m);
        return 5;
    }

    // (b) one decode: the row must agree with the cells.
    four_cell c;
    c.n_tokens  = 1;
    c.n_outputs = 1;
    c.pos0      = 0;
    std::vector<float> logits;
    if (four_decode(ctx, c, m.vocab, logits) != 0) {
        llama_free(ctx);
        live_close(m);
        return 6;
    }
    const enum llama_gate_verdict clean = llama_union_mask_row_check(ctx);
    if (clean != LLAMA_GATE_PASS) {
        std::fprintf(stderr, "FAIL: the read-back on a clean row reported %d, want PASS (%d)\n",
                (int) clean, (int) LLAMA_GATE_PASS);
        llama_free(ctx);
        live_close(m);
        return 7;
    }

    // (c) corrupt the last column of the row the check reads.
    llama_nkv_runtime_stats st = {};
    if (!llama_get_nkv_runtime_stats(ctx, &st) || st.mask_n_kv == 0) {
        std::fprintf(stderr, "FAIL: no mask width to corrupt\n");
        llama_free(ctx);
        live_close(m);
        return 8;
    }
    const uint32_t width = st.mask_n_kv;
    float * mask_data = nullptr;
    int64_t ne0 = 0;
    if (!llama_union_kq_mask_row(ctx, 0, &mask_data, &ne0) || mask_data == nullptr) {
        llama_free(ctx);
        live_close(m);
        return 9;
    }
    mask_data[width - 1] = 1.0f;
    const enum llama_gate_verdict corrupt = llama_union_mask_row_check(ctx);
    llama_free(ctx);
    live_close(m);
    if (corrupt != LLAMA_GATE_FAIL) {
        std::fprintf(stderr, "FAIL: the read-back on a corrupt row reported %d, want FAIL (%d)\n",
                (int) corrupt, (int) LLAMA_GATE_FAIL);
        return 10;
    }

    // The three values, in the order the parent reads them. This is the only
    // success output: a child that died earlier prints a FAIL line and exits
    // non-zero instead.
    std::fprintf(stderr, "FOUR_MASK_VERDICTS unproven=%d pass=%d fail=%d\n",
            (int) LLAMA_GATE_UNPROVEN, (int) LLAMA_GATE_PASS, (int) LLAMA_GATE_FAIL);
    return 0;
}

static bool part3(const char * self_path, const char * model_path) {
    bool ok = true;

    // (a) positive: one decode with the binder on; the mask is filled at the
    // width the graph executes at, and nothing refused.
    {
        setenv("GGML_TP5_UNION_BINDER", "1", 1);
        setenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK", "1", 1);
        unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
        unsetenv("LLAMA_NKV_MASK_OVERRIDE");

        live_model m;
        if (!live_open(model_path, m)) {
            return false;
        }
        llama_context * ctx = four_context(m, 16, 4);
        bool pos_ok = ctx != nullptr;
        if (pos_ok) {
            four_cell c;
            c.n_tokens  = 1;
            c.n_outputs = 1;
            c.pos0      = 0;
            std::vector<float> logits;
            pos_ok = four_decode(ctx, c, m.vocab, logits) == 0;
            llama_nkv_runtime_stats st = {};
            pos_ok = pos_ok && llama_get_nkv_runtime_stats(ctx, &st);
            if (pos_ok) {
                if (st.mask_filled == 0 || st.mask_refused != 0 || st.mask_n_kv != st.runtime_n_kv) {
                    std::fprintf(stderr,
                            "FAIL: mask positive: filled=%u refused=%u mask_n_kv=%u runtime_n_kv=%u\n",
                            st.mask_filled, st.mask_refused, st.mask_n_kv, st.runtime_n_kv);
                    pos_ok = false;
                } else if (st.mask_row_checked == 0 || st.mask_row_ok != st.mask_row_checked) {
                    // The read-back is the independent half of this case: the
                    // rows the batch consumed must agree with the cells the
                    // cache holds at the width the fill was told.
                    std::fprintf(stderr,
                            "FAIL: mask positive: read-back ok=%u checked=%u (mask_n_kv=%u runtime_n_kv=%u)\n",
                            st.mask_row_ok, st.mask_row_checked, st.mask_n_kv, st.runtime_n_kv);
                    pos_ok = false;
                } else if (st.mask_row_refused != 0 || st.mask_rows_all != 1) {
                    std::fprintf(stderr,
                            "FAIL: mask positive: refused=%u rows_all=%u (n_tokens=%d)\n",
                            st.mask_row_refused, st.mask_rows_all, c.n_tokens);
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

    // (c) negative: a mask row that disagrees with the cells must be
    // REPORTED, not smoothed over. This is the read-back's own falsifiability
    // arm, and it is a different path from (b): (b) drives the fill's width
    // guard (llama-kv-cache.cpp, "runtime n_kv exceeds the KQ mask width");
    // (c) drives the public read-back (llama_union_mask_row_check, which runs
    // check_mask_row) after a fill that succeeded. The child decodes one
    // token, requires that read-back to return LLAMA_GATE_PASS on the clean
    // row, then overwrites the mask row's last column with a value the fill
    // never writes and requires LLAMA_GATE_FAIL. The verdict comes from that
    // function, not from the stats: the stats are the fill's own counters.
    {
        std::string cmd = std::string("'") + self_path + "' --mask-row-negative '" + model_path + "' 2>&1";
        FILE * pipe = popen(cmd.c_str(), "r");
        if (pipe == nullptr) {
            std::fprintf(stderr, "FAIL: mask row negative: cannot spawn the child\n");
            return false;
        }
        std::string out;
        char buf[512];
        size_t nread;
        while ((nread = fread(buf, 1, sizeof(buf), pipe)) > 0) {
            out.append(buf, nread);
        }
        const int rc = pclose(pipe);
        // The child exits 0 when the read-back REFUSED the corrupted row, and
        // prints the phrase for that verdict. A non-zero exit is the arm's
        // failure (the read-back accepted the row, or the child never got far
        // enough to check); requiring both the exit code and the phrase means
        // neither half can pass on its own.
        const bool detected = rc == 0;
        const bool named = out.find("the mask row disagrees with the KV cells") != std::string::npos;
        const bool row_ok = detected && named;
        if (!row_ok) {
            std::fprintf(stderr,
                    "FAIL: mask row negative did not report the corrupt row (rc=%d, message=%s)\n",
                    rc, named ? "present" : "MISSING");
        }
        // Its own record, on its own line. Folding this arm into
        // FOUR_MASK_NEG would let it be deleted without the gate noticing:
        // the conjunction would still print ok=1 from the other arm. The gate
        // requires this line separately.
        std::fprintf(stderr, "FOUR_MASK_ROW_NEG ok=%d\n", int(row_ok));
        ok = ok && row_ok;
    }

    // (d) the tri-state itself: UNPROVEN, PASS and FAIL must be three distinct
    // values and all three must be reachable. The child calls the read-back
    // before any decode (UNPROVEN), after one clean decode (PASS), and after
    // corrupting the row (FAIL), then prints the three values it observed. A
    // collapse of any two states changes the line, so this arm is the
    // falsifiability proof for the encoding, not only for the check.
    {
        std::string cmd = std::string("'") + self_path + "' --mask-unproven '" + model_path + "' 2>&1";
        FILE * pipe = popen(cmd.c_str(), "r");
        if (pipe == nullptr) {
            std::fprintf(stderr, "FAIL: mask unproven: cannot spawn the child\n");
            return false;
        }
        std::string out;
        char buf[512];
        size_t nread;
        while ((nread = fread(buf, 1, sizeof(buf), pipe)) > 0) {
            out.append(buf, nread);
        }
        const int rc = pclose(pipe);
        const bool exited_ok = rc == 0;
        const bool named = out.find("FOUR_MASK_VERDICTS unproven=2 pass=1 fail=0") != std::string::npos;
        const bool verdicts_ok = exited_ok && named;
        if (!verdicts_ok) {
            std::fprintf(stderr,
                    "FAIL: the three verdicts are not distinct and reachable (rc=%d, line=%s)\n",
                    rc, named ? "present" : "MISSING");
        }
        // Its own line, for the same reason as FOUR_MASK_ROW_NEG: the
        // conjunction must not hide a deleted arm.
        std::fprintf(stderr, "FOUR_MASK_UNPROVEN ok=%d\n", int(verdicts_ok));
        ok = ok && verdicts_ok;
    }

    std::fprintf(stderr, "FOUR_MASK_NEG ok=%d\n", int(ok));
    return ok;
}

int main(int argc, char ** argv) {
    // --hash-only runs part 1 alone: no model, no backend, no allocation.
    // It is the registration that always exists (test-union-four-param-hash).
    if (argc == 2 && std::strcmp(argv[1], "--hash-only") == 0) {
        const bool ok = part1();
        std::fprintf(stderr, "FOUR_END ok=%d\n", int(ok));
        return ok ? 0 : 1;
    }

    // --mask-negative is the child of part 3: one decode with the override
    // armed. It must abort; reaching its return means the guard did not fire.
    if (argc == 3 && std::strcmp(argv[1], "--mask-negative") == 0) {
        return mask_negative_child(argv[2]);
    }

    // --mask-row-negative is part 3's other child: a fill that succeeds, then
    // a corrupted mask row. It must report the disagreement; reaching the end
    // without it means the read-back cannot fail.
    if (argc == 3 && std::strcmp(argv[1], "--mask-row-negative") == 0) {
        return mask_row_negative_child(argv[2]);
    }

    // --mask-unproven is part 3's tri-state child: UNPROVEN before any decode,
    // PASS after one clean decode, FAIL after the row is corrupted.
    if (argc == 3 && std::strcmp(argv[1], "--mask-unproven") == 0) {
        return mask_unproven_child(argv[2]);
    }

    // --matrix-only is the registration that carries the
    // PASS_REGULAR_EXPRESSION on FOUR_MATRIX. It runs exactly the same parts as
    // the plain form, including part 3, so the regex covers part 2 only: a
    // part-3 failure leaves FOUR_MATRIX ok=1 on the log and the exit code
    // carries the failure. A caller that reads only the ctest result must also
    // read FOUR_MASK_NEG, FOUR_MASK_ROW_NEG, FOUR_MASK_UNPROVEN and FOUR_END.
    // A model path is required in both forms: a missing model is a FAIL, never
    // a skip.
    const char * model_path = argc == 2 ? argv[1]
                            : (argc == 3 && std::strcmp(argv[1], "--matrix-only") == 0 ? argv[2]
                                                                                      : nullptr);

    if (model_path == nullptr) {
        std::fprintf(stderr, "usage: %s model.gguf\n"
                             "       %s --matrix-only model.gguf\n"
                             "       %s --mask-negative model.gguf\n"
                             "       %s --mask-row-negative model.gguf\n"
                             "       %s --mask-unproven model.gguf\n"
                             "       %s --hash-only\n", argv[0], argv[0], argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    if (model_path[0] == '\0') {
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
    ok &= part2(m);
    // Part 3 re-opens the model in its own contexts (it must not inherit this
    // process's live context state), so it takes the path, not `m`.
    ok &= part3(argv[0], model_path);

    live_close(m);

    std::fprintf(stderr, "FOUR_END ok=%d\n", int(ok));
    return ok ? 0 : 1;
}
