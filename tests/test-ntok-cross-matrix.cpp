// CPU unit test: n_kv, n_outputs, rs_z and n_tokens are runtime parameters of
// ONE fixed-topology graph, not build-time constants.
//
// Pillar 1 asks for one graph topology reused across all four parameters, with
// their values supplied at execution time. This test makes that checkable
// instead of asserted, in three parts.
//
// Part 1 (always runs, no model): the topology hash and the width table over
// the n_tokens lane.
//   - a synthetic graph whose KQ mask and token inputs carry the token width
//     must hash the SAME at different token widths when the token dimension is
//     excluded (the token lane), and a DIFFERENT value when it is not (every
//     other lane). That pair is the whole contract: the exclusion is what makes
//     a token-width bind possible, and its absence is what refuses one;
//   - llama_union_width_table::build() must record the token dimension of the
//     mask and of the four token inputs, and must refuse a mask whose row count
//     disagrees with the recorded token width;
//   - a token width of 0 must leave every token dimension hashed, so the same
//     graph at two token widths hashes differently.
//
// Part 2 (needs a model, and a missing model is an ERROR, not a skip): the
// four-parameter cross matrix on the live llama_decode path.
//   - one binder context decodes a fixed token history as batches of 1, 3, 2,
//     4, 7 and 5 tokens, with the output count and (on a recurrent model) the
//     rs_z lane moving independently, so n_kv, n_outputs, rs_z and n_tokens all
//     move between cells;
//   - ONE stock context walks the same history in the same order with
//     LLAMA_GRAPH_REUSE_DISABLE=1 set for its whole lifetime, so it rebuilds
//     for every cell;
//   - every cell must report the FIRST cell's topology hash, an unchanged
//     build width on every axis (no rebuild), and the cell's own token width as
//     the bound width;
//   - every compared logits row must be byte-identical (memcmp) to the stock
//     row;
//   - the two arms must be measurably different paths: the binder arm's
//     model.build_graph() delta is 0 and the stock arm's is >= cells. If both
//     arms report the same count the test fails loudly, because then the
//     comparison proved nothing.
//
// Part 3 (needs a model): the mask refusal path must be able to fire. The
// child process decodes one token with LLAMA_NKV_MASK_OVERRIDE set above the
// mask's own ne[0] and must abort with the named message "runtime n_kv exceeds
// the KQ mask width". It is the same hook and the same guard the sibling test
// (tests/test-nkv-runtime-param.cpp part 4) drives; this test reuses it rather
// than inventing a second one, because a guard that cannot be driven is a
// guard nobody has proven.
//
// The rs_z axis needs a recurrent or hybrid model. A pure dense Llama has no
// recurrent state, so bound_rs_z stays -2; the test then prints
// NTOK_MATRIX_AXES with rs_z=0 and the reason, and crosses the other three
// axes. It never reports a four-axis pass it did not run.
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
// usage: test-ntok-cross-matrix <model.gguf>
//        test-ntok-cross-matrix --hash-only
//        test-ntok-cross-matrix --mask-negative <model.gguf>   (part 3 child)
//   --hash-only runs part 1 alone: no model, no backend, no allocation. That
//   registration always exists, so the hash contract is checked on every host.
//
// Output lines (parsed by tests/nkv-cpu-probe/run_ntok_cross_matrix_13022.sh):
//   NTOK_HASH_ARM ok=<0|1> cases=<n> widths=<n>
//   NTOK_MATRIX_AXES n_kv=<n> n_outputs=<n> rs_z=<0|1> n_tokens=<n> reason=<word>
//   NTOK_MATRIX ok=<0|1> cells=<n> hash=<hex> rebuilds=<binder>
//               stock_rebuilds=<stock> bitwise=<n>/<n>
//   NTOK_MASK_NEG ok=<0|1>
//   NTOK_END ok=<0|1>

#include "llama.h"

#include "llama-union-topology.h"
#include "llama-union-width-table.h"

#include <cinttypes>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// part 1: the token dimension in the topology hash and the width table
// ---------------------------------------------------------------------------
//
// A minimal skeleton of the token lane. Only metadata matters to the hash and
// the table, so the tensors live in a ggml context with no data buffer:
//
//   tokens  = new_tensor_1d(I32, n_tokens)              "union_inp_tokens"
//   pos     = new_tensor_1d(I32, n_tokens)              "union_inp_pos"
//   k_idxs  = new_tensor_1d(I64, n_tokens)              "union_inp_k_idxs"
//   v_idxs  = new_tensor_1d(I64, n_tokens)              "union_inp_v_idxs"
//   k_view  = view_4d(head_k, n_head_kv, n_kv, 1)       "union_nkv_k"
//   v_view  = view_4d(head_v, n_head_kv, n_kv, 1)       "union_nkv_v"
//   mask    = new_tensor_4d(F32, n_kv, n_tokens, 1, 1)  "attn_inp_kq_mask"
//   oids    = new_tensor_1d(I32, n_outputs)             "union_out_ids"

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
    int64_t head      = 8;
    int64_t n_head_kv = 2;
    // the two constant factors the token inputs can carry: the position input
    // is [n_tokens * n_pos_per_embd] and a transposed V index is
    // [n_tokens * n_embd_v_gqa_max]. 1 means "the plain layout".
    int64_t n_pos_per_embd  = 1;
    int64_t n_embd_v_gqa_max = 1;
    // 0: v_idxs is [n_tokens] (non-transposed); 1: [n_tokens * n_embd_v_gqa_max]
    int     v_trans = 0;
};

// Build the skeleton at `p`. `perturb` changes one thing the hash must see:
//   0: nothing
//   2: a non-runtime shape (the head dimension of the K view)
static bool synth_build(synth_graph & g, const synth_params & p, int perturb) {
    ggml_init_params ip = {};
    // one tensor per node plus the K/V roots, generously sized
    ip.mem_size = 16u * 1024u * 1024u;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    g.ctx = ggml_init(ip);
    if (g.ctx == nullptr) {
        return false;
    }

    const int64_t head = perturb == 2 ? p.head + 1 : p.head;

    ggml_tensor * tokens = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, p.n_tokens);
    ggml_set_name(tokens, "union_inp_tokens");
    ggml_tensor * pos = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, p.n_tokens * p.n_pos_per_embd);
    ggml_set_name(pos, "union_inp_pos");
    ggml_tensor * k_idxs = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I64, p.n_tokens);
    ggml_set_name(k_idxs, "union_inp_k_idxs");
    ggml_tensor * v_idxs = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I64,
            p.v_trans ? p.n_tokens * p.n_embd_v_gqa_max : p.n_tokens);
    ggml_set_name(v_idxs, "union_inp_v_idxs");

    // the KV views are views of one backing tensor, as in the real cache
    ggml_tensor * k_backing = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, head, p.n_head_kv, p.n_kv, 1);
    ggml_tensor * v_backing = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, head, p.n_head_kv, p.n_kv, 1);

    ggml_tensor * k_view = ggml_view_4d(g.ctx, k_backing, head, p.n_head_kv, p.n_kv, 1,
            k_backing->nb[1], k_backing->nb[2], k_backing->nb[3], 0);
    ggml_set_name(k_view, "union_nkv_k");
    ggml_tensor * v_view = ggml_view_4d(g.ctx, v_backing, head, p.n_head_kv, p.n_kv, 1,
            v_backing->nb[1], v_backing->nb[2], v_backing->nb[3], 0);
    ggml_set_name(v_view, "union_nkv_v");

    ggml_tensor * mask = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, p.n_kv, p.n_tokens, 1, 1);
    ggml_set_name(mask, "attn_inp_kq_mask");

    ggml_tensor * oids = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, p.n_outputs);
    ggml_set_name(oids, "union_out_ids");

    // graph order, sources before consumers
    g.nodes = { tokens, pos, k_idxs, v_idxs, k_backing, v_backing, k_view, v_view, mask, oids };
    return true;
}

// Hash `g` with the token dimension excluded when n_tokens_excl is nonzero.
static bool synth_hash(const synth_graph & g, int64_t n_kv, int64_t n_tokens_excl,
                       int64_t n_outputs, uint64_t & out) {
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
    (void) n_kv;
    (void) n_outputs;
    return llama_union_topology_hash::hash_nodes(g.nodes, excl, out);
}

static bool synth_table(const synth_graph & g, const synth_params & p, llama_union_width_table & table) {
    return table.build(g.nodes, (uint32_t) p.n_kv, (uint32_t) p.n_outputs,
                       0 /* n_rs */, 0 /* head */, (uint32_t) p.n_tokens,
                       (uint32_t) p.n_pos_per_embd, (uint32_t) p.n_embd_v_gqa_max);
}

static bool part1() {
    bool ok = true;

    // The token widths include 1, a non-power-of-two and a value above the
    // 256-cell bucket, so the exclusion is exercised on both sides of it.
    const int64_t token_widths[] = { 1, 3, 2, 4, 7, 5, 300 };
    int cases = 0;

    // (a) with the token dimension excluded, every token width hashes the same
    {
        uint64_t ref = 0;
        bool have_ref = false;
        for (int64_t nt : token_widths) {
            synth_graph g;
            synth_params p;
            p.n_tokens = nt;
            if (!synth_build(g, p, 0)) {
                std::fprintf(stderr, "FAIL: ntok-hash: synth build at n_tokens=%" PRId64 "\n", nt);
                ok = false;
                synth_free(g);
                break;
            }
            uint64_t h = 0;
            if (!synth_hash(g, p.n_kv, p.n_tokens, p.n_outputs, h)) {
                std::fprintf(stderr, "FAIL: ntok-hash: hash at n_tokens=%" PRId64 "\n", nt);
                ok = false;
                synth_free(g);
                break;
            }
            if (!have_ref) {
                ref = h;
                have_ref = true;
            } else if (h != ref) {
                std::fprintf(stderr,
                        "FAIL: ntok-hash: hash moved with the token dimension excluded "
                        "(n_tokens=%" PRId64 ": %016" PRIx64 " vs %016" PRIx64 ")\n",
                        nt, h, ref);
                ok = false;
                synth_free(g);
                break;
            }
            ++cases;
            synth_free(g);
        }
    }

    // (b) without the exclusion the same widths must hash differently. This is
    // the negative half: it proves (a) is not true by construction.
    if (ok) {
        uint64_t ref = 0;
        bool have_ref = false;
        for (int64_t nt : token_widths) {
            synth_graph g;
            synth_params p;
            p.n_tokens = nt;
            if (!synth_build(g, p, 0)) {
                std::fprintf(stderr, "FAIL: ntok-hash: synth build at n_tokens=%" PRId64 "\n", nt);
                ok = false;
                synth_free(g);
                break;
            }
            uint64_t h = 0;
            if (!synth_hash(g, p.n_kv, 0 /* no token exclusion */, p.n_outputs, h)) {
                std::fprintf(stderr, "FAIL: ntok-hash: unexcluded hash at n_tokens=%" PRId64 "\n", nt);
                ok = false;
                synth_free(g);
                break;
            }
            if (!have_ref) {
                ref = h;
                have_ref = true;
            } else if (h == ref) {
                std::fprintf(stderr,
                        "FAIL: ntok-hash: hash did NOT move with the token dimension hashed "
                        "(n_tokens=%" PRId64 "); the exclusion test proves nothing\n", nt);
                ok = false;
                synth_free(g);
                break;
            }
            synth_free(g);
        }
    }

    // (c) a non-runtime change must still move the hash with the exclusion on
    if (ok) {
        synth_graph a, b;
        synth_params p;
        if (!synth_build(a, p, 0) || !synth_build(b, p, 2 /* head + 1 */)) {
            std::fprintf(stderr, "FAIL: ntok-hash: synth build for the head case\n");
            ok = false;
        } else {
            uint64_t ha = 0, hb = 0;
            if (!synth_hash(a, p.n_kv, p.n_tokens, p.n_outputs, ha) ||
                !synth_hash(b, p.n_kv, p.n_tokens, p.n_outputs, hb)) {
                std::fprintf(stderr, "FAIL: ntok-hash: hash for the head case\n");
                ok = false;
            } else if (ha == hb) {
                std::fprintf(stderr, "FAIL: ntok-hash: a non-runtime shape change did not move the hash\n");
                ok = false;
            }
        }
        synth_free(a);
        synth_free(b);
    }

    // (d) the width table records the token dimension and refuses a bad row count
    if (ok) {
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
                if (bad.build(g.nodes, (uint32_t) p.n_kv, (uint32_t) p.n_outputs, 0, 0,
                              (uint32_t) p.n_tokens + 1, 1, 1)) {
                    std::fprintf(stderr, "FAIL: ntok-table: a token width the mask cannot hold was accepted\n");
                    ok = false;
                }
            }
            // a zero token width must leave the token lane unrecorded, and the
            // token dimensions hashed
            if (ok) {
                llama_union_width_table plain;
                if (!plain.build(g.nodes, (uint32_t) p.n_kv, (uint32_t) p.n_outputs, 0, 0, 0, 1, 1)) {
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

    // (e) the token inputs are checked at their EXACT length, not at a length
    // that merely divides by the token width. A position input of
    // [n_tokens * n_pos_per_embd] must be accepted only when the caller passes
    // that factor, and a V index of [n_tokens * n_embd_v_gqa_max] only when
    // the caller passes that one. A multiple-of check would accept both of the
    // wrong pairings, so each case here can fail.
    if (ok) {
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
            if (!good.build(g.nodes, (uint32_t) p.n_kv, (uint32_t) p.n_outputs, 0, 0,
                            (uint32_t) p.n_tokens, 2, 3) || !good.has_ntok) {
                std::fprintf(stderr, "FAIL: ntok-factor: the real layout was refused\n");
                ok = false;
            }
            // pos factor 1 against a pos tensor of [8] must be refused
            llama_union_width_table bad_pos;
            if (bad_pos.build(g.nodes, (uint32_t) p.n_kv, (uint32_t) p.n_outputs, 0, 0,
                              (uint32_t) p.n_tokens, 1, 3)) {
                std::fprintf(stderr, "FAIL: ntok-factor: a pos tensor of n_tokens*2 was accepted "
                        "with factor 1\n");
                ok = false;
            }
            // v_idxs factor 2 against a tensor of [12] must be refused
            llama_union_width_table bad_v;
            if (bad_v.build(g.nodes, (uint32_t) p.n_kv, (uint32_t) p.n_outputs, 0, 0,
                            (uint32_t) p.n_tokens, 2, 2)) {
                std::fprintf(stderr, "FAIL: ntok-factor: a v_idxs tensor of n_tokens*3 was accepted "
                        "with factor 2\n");
                ok = false;
            }
            // a non-transposed V index is exactly n_tokens; factor 3 must be
            // refused for it
            synth_params q = p;
            q.v_trans = 0;
            synth_graph g2;
            if (!synth_build(g2, q, 0)) {
                std::fprintf(stderr, "FAIL: ntok-factor: synth build for the plain V layout\n");
                ok = false;
            } else {
                llama_union_width_table bad_plain_v;
                if (bad_plain_v.build(g2.nodes, (uint32_t) q.n_kv, (uint32_t) q.n_outputs, 0, 0,
                                      (uint32_t) q.n_tokens, 2, 3)) {
                    std::fprintf(stderr, "FAIL: ntok-factor: a plain v_idxs of n_tokens was accepted "
                            "with factor n_embd_v_gqa_max\n");
                    ok = false;
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
    if (ok) {
        synth_graph g;
        synth_params p;
        if (!synth_build(g, p, 0)) {
            std::fprintf(stderr, "FAIL: ntok-lane: synth build\n");
            ok = false;
        } else {
            // rename the four token inputs so no root matches, and the mask's
            // row dimension too, by rebuilding the list without them
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
            if (lane.build(no_tokens, (uint32_t) p.n_kv, (uint32_t) p.n_outputs, 0, 0,
                           (uint32_t) p.n_tokens, 1, 1)) {
                std::fprintf(stderr, "FAIL: ntok-lane: an armed token lane with no token root "
                        "was recorded\n");
                ok = false;
            }
        }
        synth_free(g);
    }
    // (g) the CPU attention reduction length is the fifth runtime width, and
    // it lives in an op_param slot rather than in a dimension. The contract is
    // the same one: the slot is excluded positionally, so a live-length change
    // cannot move the hash, while a change to any other slot still does.
    //
    // The skeleton is the real attention closure, because the exclusion is
    // recorded per node from the tensor flags the binder sets:
    //   q      = new_tensor_4d(F32, head, n_head, n_tokens, 1)
    //   k, v   = new_tensor_4d(F32, head, n_head_kv, pad, 1)   "union_nkv_k/v"
    //   kq     = mul_mat(k, q)
    //   mask   = new_tensor_4d(F32, pad, n_tokens, 1, 1)       "attn_inp_kq_mask"
    //   scores = kq + mask
    //   probs  = soft_max(scores)      TRUE_LEN_REDUCE, slot 3 = live
    //   out    = mul_mat(probs, v)     TRUE_LEN_REDUCE, slot 3 = live
    if (ok) {
        struct fa_case {
            int32_t live;   // op_params[3], the runtime reduction length
            int32_t pad;    // op_params[2]/[4], the padded extent
        };
        // one padded extent at four live lengths, including both boundaries
        // (1 and the padded extent itself)
        const fa_case cases[] = { { 1, 64 }, { 7, 64 }, { 33, 64 }, { 64, 64 } };
        const int n_cases = (int) (sizeof(cases) / sizeof(cases[0]));
        int fa_checked = 0;

        uint64_t ref = 0;
        bool have_ref = false;
        for (int ci = 0; ci < n_cases && ok; ++ci) {
            const fa_case & c = cases[ci];
            ggml_init_params ip = {};
            ip.mem_size   = 16u * 1024u * 1024u;
            ip.mem_buffer = nullptr;
            ip.no_alloc   = true;
            ggml_context * ctx = ggml_init(ip);
            if (ctx == nullptr) {
                std::fprintf(stderr, "FAIL: fa-slot: ggml_init\n");
                ok = false;
                break;
            }

            const int64_t head      = 8;
            const int64_t n_head    = 2;
            const int64_t n_head_kv = 1;
            const int64_t n_tokens  = 4;

            ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head, n_head, n_tokens, 1);
            ggml_set_name(q, "fa_q");
            ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head, n_head_kv, c.pad, 1);
            ggml_set_name(k, "union_nkv_k");
            ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head, n_head_kv, c.pad, 1);
            ggml_set_name(v, "union_nkv_v");
            ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
            ggml_set_name(kq, "fa_kq");
            ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.pad, n_tokens, 1, 1);
            ggml_set_name(mask, "attn_inp_kq_mask");
            ggml_tensor * scores = ggml_add(ctx, kq, mask);
            ggml_set_name(scores, "fa_scores");
            ggml_tensor * probs = ggml_soft_max(ctx, scores);
            ggml_set_name(probs, "fa_probs");
            ggml_tensor * out = ggml_mul_mat(ctx, probs, v);
            ggml_set_name(out, "fa_out");

            // the bind's own writes: the padded extent in slots 2 and 4, the
            // runtime length in slot 3, and the two flags the kernels read
            for (ggml_tensor * t : { probs, out }) {
                t->flags |= GGML_TENSOR_FLAG_REDUCE_PAD_TAIL | GGML_TENSOR_FLAG_TRUE_LEN_REDUCE;
                t->op_params[2] = c.pad;
                t->op_params[3] = c.live;
                t->op_params[4] = c.pad;
            }

            std::vector<ggml_tensor *> nodes = { q, k, v, kq, mask, scores, probs, out };

            // the exclusion the width table records: n_kv in the K/V views, the
            // mask's key dimension, and slot 3 of every TRUE_LEN_REDUCE node
            std::vector<llama_union_topology_excl> excl(nodes.size());
            int n_skipped = 0;
            for (size_t i = 0; i < nodes.size(); ++i) {
                excl[i].node = (int32_t) i;
                const char * name = nodes[i]->name;
                if (std::strcmp(name, "union_nkv_k") == 0 ||
                        std::strcmp(name, "union_nkv_v") == 0) {
                    excl[i].dim = 2;
                } else if (std::strcmp(name, "attn_inp_kq_mask") == 0) {
                    excl[i].dim = 0;
                }
                if ((nodes[i]->flags & GGML_TENSOR_FLAG_TRUE_LEN_REDUCE) != 0) {
                    excl[i].skip_op_param = 3;
                    ++n_skipped;
                }
            }
            if (n_skipped != 2) {
                std::fprintf(stderr, "FAIL: fa-slot: %d reduction slots recorded, want 2\n", n_skipped);
                ok = false;
                ggml_free(ctx);
                break;
            }

            uint64_t h = 0;
            if (!llama_union_topology_hash::hash_nodes(nodes, excl, h)) {
                std::fprintf(stderr, "FAIL: fa-slot: hash at live=%d\n", c.live);
                ok = false;
                ggml_free(ctx);
                break;
            }
            if (!have_ref) {
                ref = h;
                have_ref = true;
            } else if (h != ref) {
                std::fprintf(stderr,
                        "FAIL: fa-slot: the hash moved with the reduction length excluded "
                        "(live=%d pad=%d: %016" PRIx64 " vs %016" PRIx64 ")\n",
                        c.live, c.pad, h, ref);
                ok = false;
                ggml_free(ctx);
                break;
            }

            // negative control: hashing the slot must make the same live
            // lengths disagree, so the equality above is not true by
            // construction. The reference case is the one that carries the
            // reference hash, so it cannot move.
            if (c.live != cases[0].live) {
                std::vector<llama_union_topology_excl> no_slot = excl;
                for (auto & e : no_slot) {
                    e.skip_op_param = -1;
                }
                uint64_t h_neg = 0;
                if (!llama_union_topology_hash::hash_nodes(nodes, no_slot, h_neg)) {
                    std::fprintf(stderr, "FAIL: fa-slot: unexcluded hash at live=%d\n", c.live);
                    ok = false;
                    ggml_free(ctx);
                    break;
                }
                if (h_neg == ref) {
                    std::fprintf(stderr,
                            "FAIL: fa-slot: the hash did NOT move with the reduction slot hashed "
                            "(live=%d); the slot exclusion test proves nothing\n", c.live);
                    ok = false;
                    ggml_free(ctx);
                    break;
                }
            }

            // positive control: the padded extent in slot 2 is still hashed, so
            // a padded-extent change moves the hash even with slot 3 excluded
            if (c.live == cases[0].live) {
                for (ggml_tensor * t : { probs, out }) {
                    t->op_params[2] = c.pad + 1;
                    t->op_params[4] = c.pad + 1;
                }
                uint64_t h_pad = 0;
                if (!llama_union_topology_hash::hash_nodes(nodes, excl, h_pad)) {
                    std::fprintf(stderr, "FAIL: fa-slot: hash for the padded-extent case\n");
                    ok = false;
                    ggml_free(ctx);
                    break;
                }
                if (h_pad == ref) {
                    std::fprintf(stderr,
                            "FAIL: fa-slot: a padded-extent (op_params[2]) change did not move "
                            "the hash; the exclusion swallowed a non-runtime slot\n");
                    ok = false;
                    ggml_free(ctx);
                    break;
                }
            }
            ++fa_checked;
            ggml_free(ctx);
        }

        std::fprintf(stderr, "NTOK_FA_SLOT ok=%d widths=%d/%d\n", int(ok), fa_checked, n_cases);
    }

    std::fprintf(stderr, "NTOK_HASH_ARM ok=%d cases=%d widths=%d\n",
            int(ok), cases, (int) (sizeof(token_widths) / sizeof(token_widths[0])));
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
struct ntok_cell {
    int      n_tokens  = 1;
    uint32_t n_outputs = 1;
    int64_t  pos0      = 0;
};

// n_batch and n_ubatch are the widest step: the graph is built at the active
// token width and bound in place afterwards, so the context must accept a
// batch as wide as the widest cell.
static llama_context * ntok_context(const live_model & m, int n_ctx, uint32_t n_ubatch) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx         = (uint32_t) n_ctx;
    cp.n_batch       = n_ubatch;
    cp.n_ubatch      = n_ubatch;
    cp.n_seq_max     = 1;
    cp.n_outputs_max = n_ubatch;
    cp.flash_attn    = false;
    return llama_init_from_model(m.model, cp);
}

// Decode one cell. `logits_out` receives every produced logits row, row-major,
// `n_outputs` rows of `n_vocab` floats.
static int ntok_decode(llama_context * ctx, const ntok_cell & c, int vocab, std::vector<float> & logits_out) {
    logits_out.assign((size_t) c.n_outputs * (size_t) vocab, 0.0f);

    llama_batch batch = llama_batch_init(c.n_tokens, 0, 1);
    if (!llama_batch_is_complete(&batch, c.n_tokens, false, true)) {
        llama_batch_free(batch);
        return -1;
    }
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
// (the KV span is the history length). The output count moves independently of
// the token width, and never exceeds it.
static std::vector<ntok_cell> ntok_cells() {
    std::vector<ntok_cell> cells;
    const int sizes[] = { 1, 3, 2, 4, 7, 5 };
    const int n_sizes = (int) (sizeof(sizes) / sizeof(sizes[0]));
    int pos = 0;
    for (int i = 0; i < 60; ++i) {
        ntok_cell c;
        c.n_tokens  = sizes[i % n_sizes];
        c.n_outputs = (uint32_t) (1 + (i % 3));
        if (c.n_outputs > (uint32_t) c.n_tokens) {
            c.n_outputs = (uint32_t) c.n_tokens;
        }
        c.pos0 = pos;
        pos += c.n_tokens;
        cells.push_back(c);
    }
    return cells;
}

static bool part2(const live_model & m) {
    const int vocab = m.vocab;

    const std::vector<ntok_cell> cells = ntok_cells();
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
    //   GGML_TP5_UNION_BINDER=1        binder arm (llama-context.cpp:508).
    //   LLAMA_EXPERIMENTAL_VARLEN_NTOK=1  the token lane (llama-context.cpp:517).
    //   LLAMA_GRAPH_REUSE_DISABLE      unset for the binder arm, set for the
    //                                  stock arm (llama-context.cpp:471).
    setenv("GGML_TP5_UNION_BINDER", "1", 1);
    setenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK", "1", 1);
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    unsetenv("LLAMA_NKV_MASK_OVERRIDE");

    llama_context * ctx = ntok_context(m, n_ctx, n_ubatch);
    if (ctx == nullptr) {
        std::fprintf(stderr, "FAIL: ntok: binder context init\n");
        return false;
    }

    // The stock arm: one context, same history, every cell rebuilt. The flag
    // is set before this context is constructed and is not unset while it
    // lives; the constructor already read it.
    setenv("LLAMA_GRAPH_REUSE_DISABLE", "1", 1);
    llama_context * stock = ntok_context(m, n_ctx, n_ubatch);
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    if (stock == nullptr) {
        std::fprintf(stderr, "FAIL: ntok: stock context init\n");
        llama_free(ctx);
        return false;
    }

    bool ok = true;
    uint64_t ref_hash = 0;
    bool have_ref = false;
    uint32_t build_n_kv = 0, build_n_outputs = 0, build_n_rs = 0, build_n_tokens = 0;
    // Rebuild counts are measured, never guessed: the delta of each context's
    // own model.build_graph() counter across the run.
    uint64_t binder_builds0 = 0, binder_builds1 = 0;
    uint64_t stock_builds0  = 0, stock_builds1  = 0;
    int bitwise = 0;
    int bitwise_cells = 0;
    int rs_z_moves = 0;
    int32_t last_rs_z = -3;
    int64_t  min_nkv = INT64_MAX, max_nkv = 0;
    int      distinct_tokens = 0;
    int      last_tokens = -1;

    // Baselines for the rebuild deltas. Taken after both contexts exist, so
    // the constructor/reserve builds are not counted as cell rebuilds.
    binder_builds0 = llama_graph_build_count(ctx);
    stock_builds0  = llama_graph_build_count(stock);
    std::vector<float> got, want;

    for (size_t i = 0; i < cells.size() && ok; ++i) {
        const ntok_cell & c = cells[i];
        if (ntok_decode(ctx, c, vocab, got) != 0) {
            std::fprintf(stderr, "FAIL: ntok: decode cell=%zu\n", i);
            ok = false;
            break;
        }
        llama_nkv_runtime_stats st = {};
        if (!llama_get_nkv_runtime_stats(ctx, &st)) {
            std::fprintf(stderr, "FAIL: ntok: no stats at cell=%zu\n", i);
            ok = false;
            break;
        }
        if (!st.topology_hashed) {
            std::fprintf(stderr, "FAIL: ntok: topology not hashed at cell=%zu\n", i);
            ok = false;
            break;
        }
        if (!st.has_nkv_dims || !st.has_nout_dims || !st.has_ntok_dims) {
            std::fprintf(stderr,
                    "FAIL: ntok: missing runtime roots at cell=%zu (nkv=%d nout=%d ntok=%d)\n",
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
                std::fprintf(stderr, "FAIL: ntok: hash moved at cell=%zu (%016" PRIx64 " -> %016" PRIx64 ")\n",
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
                        "FAIL: ntok: rebuild at cell=%zu (n_kv %u->%u, n_outputs %u->%u, "
                        "n_rs %u->%u, n_tokens %u->%u)\n",
                        i, build_n_kv, st.build_n_kv, build_n_outputs, st.build_n_outputs,
                        build_n_rs, st.bound_n_rs, build_n_tokens, st.build_n_tokens);
                ok = false;
                break;
            }
        }
        // the executed parameters must equal what this cell asked for
        if (st.bound_n_outputs != c.n_outputs) {
            std::fprintf(stderr, "FAIL: ntok: n_outputs not applied at cell=%zu (want %u got %u)\n",
                    i, c.n_outputs, st.bound_n_outputs);
            ok = false;
            break;
        }
        if ((int64_t) st.runtime_n_kv != (int64_t) c.pos0 + c.n_tokens) {
            std::fprintf(stderr, "FAIL: ntok: n_kv not applied at cell=%zu (want %" PRId64 " got %u)\n",
                    i, (int64_t) c.pos0 + c.n_tokens, st.runtime_n_kv);
            ok = false;
            break;
        }
        // the token width must be the one this cell asked for, and it must be
        // the BOUND width, not the build width: a graph still at the build
        // width would compute the wrong number of rows
        if (st.bound_n_tokens != (uint32_t) c.n_tokens) {
            std::fprintf(stderr, "FAIL: ntok: token width not applied at cell=%zu (want %d got %u)\n",
                    i, c.n_tokens, st.bound_n_tokens);
            ok = false;
            break;
        }
        // the graph must execute at the width it reports, not only report it
        if (st.bound_n_kv != st.runtime_n_kv) {
            std::fprintf(stderr, "FAIL: ntok: bound n_kv %u != runtime n_kv %u at cell=%zu\n",
                    st.bound_n_kv, st.runtime_n_kv, i);
            ok = false;
            break;
        }
        // The mask observation, on EVERY cell including cell 0: the fill must
        // have consumed exactly the width the graph executes at, and must not
        // have refused.
        if (st.mask_refused != 0) {
            std::fprintf(stderr, "FAIL: ntok: the mask fill refused at cell=%zu (refused=%u)\n",
                    i, st.mask_refused);
            ok = false;
            break;
        }
        if (st.mask_filled == 0) {
            std::fprintf(stderr, "FAIL: ntok: no KQ mask filled at cell=%zu\n", i);
            ok = false;
            break;
        }
        if (st.mask_n_kv != st.runtime_n_kv) {
            std::fprintf(stderr,
                    "FAIL: ntok: mask filled at %u, graph executes at %u (cell=%zu)\n",
                    st.mask_n_kv, st.runtime_n_kv, i);
            ok = false;
            break;
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

        // --- stock arm: the SAME history, walked once -------------------
        //
        // One stock context lives for the whole loop and decodes each cell
        // exactly once, in the same order the binder arm decodes them, so its
        // state after cell i is the same history and the comparison is
        // unchanged. It is O(cells), not O(cells^2).
        const int rc_stock = ntok_decode(stock, c, vocab, want);
        if (rc_stock != 0) {
            std::fprintf(stderr, "FAIL: ntok: stock decode cell=%zu (rc=%d)\n", i, rc_stock);
            ok = false;
            break;
        }
        // The denominator counts every cell whose stock decode completed, and
        // it is incremented BEFORE the comparison, so the numerator can never
        // exceed it and the pair is n/n only when every compared cell matched.
        // ntok_cells() always returns 60 cells, so the line cannot print
        // bitwise=0/0.
        ++bitwise_cells;
        if (got.size() == want.size() && std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0) {
            ++bitwise;
        } else {
            std::fprintf(stderr, "FAIL: ntok: logits differ from stock at cell=%zu (n_kv=%" PRId64
                    " n_tokens=%d n_outputs=%u)\n", i, (int64_t) c.pos0 + c.n_tokens, c.n_tokens, c.n_outputs);
            ok = false;
        }
    }

    // the axes must actually have moved
    if (ok) {
        if (min_nkv >= 256 || max_nkv <= 256) {
            std::fprintf(stderr, "FAIL: ntok: n_kv did not cross 256 (min=%" PRId64 " max=%" PRId64 ")\n",
                    min_nkv, max_nkv);
            ok = false;
        }
        // The token axis must have moved. A matrix whose token width never
        // changed would prove nothing about the token lane.
        if (distinct_tokens < 2) {
            std::fprintf(stderr, "FAIL: ntok: the token width never moved (%d distinct value(s))\n",
                    distinct_tokens);
            ok = false;
        }
        if (last_rs_z == -2) {
            std::fprintf(stderr,
                    "NTOK_MATRIX_AXES n_kv=1 n_outputs=1 rs_z=0 n_tokens=1 reason=dense_model\n");
        } else if (rs_z_moves == 0) {
            std::fprintf(stderr, "FAIL: ntok: rs_z never moved on a recurrent model\n");
            ok = false;
        } else {
            std::fprintf(stderr,
                    "NTOK_MATRIX_AXES n_kv=1 n_outputs=1 rs_z=1 n_tokens=1 reason=recurrent_model\n");
        }
    }

    // The rebuild deltas. This is the assertion that makes the two arms
    // genuinely different paths: the binder arm must not rebuild at all, and
    // the stock arm must rebuild for every cell. `>= cells` is the strict
    // form: a stock arm that replayed without rebuilding is a defect, not a
    // pass. Equal counts mean both arms took the same path.
    binder_builds1 = llama_graph_build_count(ctx);
    stock_builds1  = llama_graph_build_count(stock);
    const uint64_t rebuilds       = binder_builds1 - binder_builds0;
    const uint64_t stock_rebuilds = stock_builds1  - stock_builds0;
    if (ok) {
        if (rebuilds != 0) {
            std::fprintf(stderr, "FAIL: ntok: the binder arm rebuilt %" PRIu64 " time(s)\n", rebuilds);
            ok = false;
        }
        if (stock_rebuilds < (uint64_t) cells.size()) {
            std::fprintf(stderr,
                    "FAIL: ntok: the stock arm rebuilt %" PRIu64 " time(s) for %zu cells "
                    "(LLAMA_GRAPH_REUSE_DISABLE did not take effect)\n",
                    stock_rebuilds, cells.size());
            ok = false;
        }
        if (rebuilds == stock_rebuilds) {
            std::fprintf(stderr,
                    "FAIL: ntok: both arms rebuilt %" PRIu64 " time(s); they took the same path\n",
                    rebuilds);
            ok = false;
        }
    }

    std::fprintf(stderr,
            "NTOK_MATRIX ok=%d cells=%zu hash=%016" PRIx64
            " rebuilds=%" PRIu64 " stock_rebuilds=%" PRIu64 " bitwise=%d/%d\n",
            int(ok), cells.size(), ref_hash, rebuilds, stock_rebuilds, bitwise, bitwise_cells);

    llama_free(stock);
    llama_free(ctx);
    return ok;
}

// ---------------------------------------------------------------------------
// part 3: the mask path MUST be able to fail
// ---------------------------------------------------------------------------
//
// The child of part 3: one decode with the runtime width forced above the
// mask's ne[0]. It must abort. Reaching the return means the guard did not
// fire, and the exit code says so.
static int mask_negative_child(const char * model_path) {
    setenv("GGML_TP5_UNION_BINDER", "1", 1);
    setenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK", "1", 1);
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");

    live_model m;
    if (!live_open(model_path, m)) {
        return 3;
    }
    llama_context * ctx = ntok_context(m, 16, 4);
    if (ctx == nullptr) {
        live_close(m);
        return 4;
    }
    ntok_cell c;
    c.n_tokens  = 1;
    c.n_outputs = 1;
    c.pos0      = 0;
    std::vector<float> logits;
    (void) ntok_decode(ctx, c, m.vocab, logits);
    llama_free(ctx);
    live_close(m);
    std::fprintf(stderr, "FAIL: the mask guard did not fire on an over-wide runtime n_kv\n");
    return 5;
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
        llama_context * ctx = ntok_context(m, 16, 4);
        bool pos_ok = ctx != nullptr;
        if (pos_ok) {
            ntok_cell c;
            c.n_tokens  = 1;
            c.n_outputs = 1;
            c.pos0      = 0;
            std::vector<float> logits;
            pos_ok = ntok_decode(ctx, c, m.vocab, logits) == 0;
            llama_nkv_runtime_stats st = {};
            pos_ok = pos_ok && llama_get_nkv_runtime_stats(ctx, &st);
            if (pos_ok) {
                if (st.mask_filled == 0 || st.mask_refused != 0 || st.mask_n_kv != st.runtime_n_kv) {
                    std::fprintf(stderr,
                            "FAIL: mask positive: filled=%u refused=%u mask_n_kv=%u runtime_n_kv=%u\n",
                            st.mask_filled, st.mask_refused, st.mask_n_kv, st.runtime_n_kv);
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

    std::fprintf(stderr, "NTOK_MASK_NEG ok=%d\n", int(ok));
    return ok;
}

int main(int argc, char ** argv) {
    // --hash-only runs part 1 alone: no model, no backend, no allocation.
    // It is the registration that always exists (test-ntok-cross-matrix-hash).
    if (argc == 2 && std::strcmp(argv[1], "--hash-only") == 0) {
        const bool ok = part1();
        std::fprintf(stderr, "NTOK_END ok=%d\n", int(ok));
        return ok ? 0 : 1;
    }

    // --mask-negative is the child of part 3: one decode with the override
    // armed. It must abort; reaching its return means the guard did not fire.
    if (argc == 3 && std::strcmp(argv[1], "--mask-negative") == 0) {
        return mask_negative_child(argv[2]);
    }

    if (argc != 2) {
        std::fprintf(stderr, "usage: %s model.gguf\n"
                             "       %s --hash-only\n"
                             "       %s --mask-negative model.gguf\n", argv[0], argv[0], argv[0]);
        return 2;
    }
    if (argv[1][0] == '\0') {
        std::fprintf(stderr, "FAIL: empty model path\n");
        return 2;
    }

    live_model m;
    if (!live_open(argv[1], m)) {
        return 1;
    }

    // Every part runs even after one fails, so one invocation reports every
    // failure instead of only the first. The exit code is the conjunction.
    bool ok = true;
    ok &= part1();
    ok &= part2(m);
    // Part 3 re-opens the model in its own contexts (it must not inherit this
    // process's live context state), so it takes the path, not `m`.
    ok &= part3(argv[0], argv[1]);

    live_close(m);

    std::fprintf(stderr, "NTOK_END ok=%d\n", int(ok));
    return ok ? 0 : 1;
}
