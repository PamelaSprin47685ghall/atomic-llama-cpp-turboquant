#pragma once

//
// llama-union-topology.h
//
// Stable hash of a built graph's TOPOLOGY, independent of the runtime
// parameters the union binder binds (pillar 1: n_kv, n_outputs, rs_z and,
// on the token lane, n_tokens as true runtime parameters of ONE
// fixed-topology graph).
//
// Why this exists: the union binder reuses one built graph across runtime
// parameter values by patching tensor shapes in place. "The same graph serves
// every combination" is then a claim about the graph's node set and edges,
// not about its shapes. This hash makes that claim checkable instead of
// asserted: a caller hashes a graph built at one parameter combination and
// the same graph after it was bound to another; an equal hash proves the node
// set, the edge structure, the op types, every op parameter and every
// non-runtime shape dimension survived the change.
//
// What the hash covers (see llama_union_topology_hash::hash_nodes()):
//   - node count and, per node in graph order, the op type, the flags, ALL op
//     parameters, and the shape of every dimension that does not carry a
//     runtime parameter;
//   - every edge (src[j], view_src) as the index of the referenced node;
//   - the tensor NAME of each node, because the shape analysis keys its
//     runtime roots on names ("union_nkv_k", "attn_inp_kq_mask", ...).
//
// What the hash deliberately excludes:
//   - the dimensions that carry a runtime parameter. The caller names them
//     per node (llama_union_topology_excl); each named dimension is skipped,
//     so a parameter change cannot move the hash. The n_tokens dimension is
//     named only on the token lane; on every other lane it is hashed, so a
//     token-width change there still moves the hash and refuses the bind;
//   - the data pointer of a VIEW node. ggml copies the view offset into both
//     view_offs and op_params[0..1]; the binder patches view_offs, so the
//     op-param copy must be skipped for the same reason. The offset itself is
//     NOT hashed: it is a shape, and every offset-carrying node of the three
//     runtime lanes is either a view (the binder patches it) or a node whose
//     sources are all hashed. See llama_union_topology_excl::skip_view_data;
//   - one op_param slot of a node that carries a runtime VALUE there. The CPU
//     attention reduction length is the only such slot today: a
//     TRUE_LEN_REDUCE node keeps the number of keys to reduce in
//     op_params[3], and the bind rewrites it every step. See
//     llama_union_topology_excl::skip_op_param;
//   - data pointers, buffer pointers, backend placement and byte strides.
//     Those are allocation facts, not topology.
//
// The hash is FNV-1a 64-bit over a canonical little-endian byte encoding.
// It is a change detector, not a cryptographic digest.
//
// Edges to a tensor outside the hashed node list are FOLDED, not refused.
// The list need not be closed: an out-of-set target is hashed as the reserved
// marker EXTERNAL, a node's own self-edge as SELF, and a null source as ABSENT
// -- the same three markers, with the same values, that
// llama_topology_key_of() uses (llama-topology-hash.h, llama_topology_edge and
// canonical_edge()). Two hashes taken over the same graph must not disagree
// about what an edge to a non-member means.
//
// This is not a weakening. An out-of-set target is an ALLOCATION fact: after
// ggml_backend_sched_alloc_graph() the scheduler inserts split copies whose own
// view_src sits one level further out, and the canonical walk stops at fixed
// nodes, so such a target cannot be a member however late the list is captured.
// The header already excludes data/buffer pointers for the same reason. What
// stays hashed for every MEMBER is its op, type, flags, name, op params, every
// ne[]/nb[] (minus the excluded runtime dimensions), view_offs and all of its
// in-set edges -- including the KV-cache and recurrent-state leaves, which are
// members precisely so a width difference stays visible.
//
// Fail-closed rules (kept):
//   - an empty list, a null node, a negative ne[], or an exclusion index that
//     does not name its own position makes update() return false; the caller
//     must then treat the graph as unhashed rather than compare a partial hash.
//

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

// per-node exclusion of the dimensions that carry a runtime parameter.
// `dim` is the single dimension that carries n_kv (-1: none); `nout_dim`,
// `nrs_dim` and `head_dim` do the same for the n_outputs, recurrent-length
// and recurrent-head parameters. `ntok_dim` is the dimension that carries the
// token width n_tokens. `skip_view_data` drops the view-offset copy that ggml
// stores in op_params for a view node.
//
// n_tokens is a runtime parameter only on the token lane (the graph is built
// at the active token width and bound in place afterwards). On every other
// lane no entry sets `ntok_dim`, so the token dimension is hashed exactly as
// before and a token-width change still moves the hash.
//
// The same holds for the other three: a lane that binds a parameter sets its
// slot here and nowhere else sets it for that parameter. The exclusion is a
// property of the LANE, not of the parameter, and the lane is decided by the
// caller that passes the value (llama_union_width_table::build()).
//
// `skip_op_param` drops ONE op_param slot that carries a runtime VALUE rather
// than a topology fact (-1: none). It exists for the CPU attention reduction
// length: on a TRUE_LEN_REDUCE node the kernels read op_params[3] as the
// number of keys to reduce (ggml/src/ggml-cpu/ops.cpp:5543,
// ggml/src/ggml-cpu/ggml-cpu.c:1348), so that slot is a runtime value
// wherever it is written, not a topology fact. The union bind no longer
// writes it: the bind patches shapes in place (ne[], nb[], view_offs) and
// leaves every op_param alone.
//
// The exclusion stays because it is positional and value-independent -- a
// node carrying the flag hashes the same whether or not the slot has been
// written -- which is the same rule the n_kv dimensions follow, not a new
// one. Only slot 3 is excluded: slot 4 holds the build-time padded extent and
// nothing rewrites it at runtime, so hashing it is hashing a constant.
//
// The two exclusions are independent: a node may carry a token dimension, a
// reduction-length slot, both, or neither.
struct llama_union_topology_excl {
    int32_t node           = -1;
    int32_t dim            = -1;
    int32_t nout_dim       = -1;
    int32_t nrs_dim        = -1;
    int32_t head_dim       = -1;
    int32_t ntok_dim       = -1;
    // The exact set of runtime axes, one bit per ne[] index. The five named
    // slots above are the same fact spelled per parameter, and they stay for
    // the callers that name a parameter. A caller that has the binder's
    // coefficient table fills this mask as well, because the table can name
    // two axes of one tensor for the SAME parameter (a reshape that splits a
    // runtime width across two axes), which one named slot cannot hold.
    uint8_t dim_mask       = 0;
    bool    skip_view_data = false;
    int32_t skip_op_param  = -1;
};

class llama_union_topology_hash {
public:
    // FNV-1a 64-bit offset basis and prime
    static constexpr uint64_t basis = 0xcbf29ce484222325ULL;
    static constexpr uint64_t prime = 0x100000001b3ULL;

    llama_union_topology_hash() : h_(basis) {}

    uint64_t value() const { return h_; }

    void mix_u64(uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            h_ ^= (v >> (8 * i)) & 0xffULL;
            h_ *= prime;
        }
    }

    void mix_i64(int64_t v) {
        mix_u64((uint64_t) v);
    }

    void mix_str(const char * s) {
        if (s == nullptr) {
            mix_u64(0);
            return;
        }
        const size_t n = std::strlen(s);
        mix_u64((uint64_t) n);
        for (size_t i = 0; i < n; ++i) {
            h_ ^= (uint8_t) s[i];
            h_ *= prime;
        }
    }

    // Hash `nodes` (in graph order) as one topology.
    //
    // `excl` names, per node, the dimensions that carry a runtime parameter;
    // entries with node < 0 or dim < 0 mean "none". The vector may be shorter
    // than `nodes` (missing entries mean "none").
    // Returns false, with `out` untouched, when the graph cannot be hashed
    // under the rules above, or when `nodes` is empty.
    // The three edge markers of llama_topology_key_of() (llama-topology-hash.h,
    // enum llama_topology_edge), repeated here with the same values because
    // this header includes only ggml.h. They are reserved: no real position in
    // the hashed list can collide with them.
    static constexpr int64_t edge_absent   = -1;
    static constexpr int64_t edge_external = -2;
    static constexpr int64_t edge_self     = -3;

    static bool hash_nodes(const std::vector<ggml_tensor *> & nodes,
                           const std::vector<llama_union_topology_excl> & excl,
                           uint64_t & out) {
        // An empty list is not a topology. Hashing it would give every empty
        // graph the same value, and a comparison of two of them would pass
        // over nothing.
        if (nodes.empty()) {
            return false;
        }

        llama_union_topology_hash h;

        // index of every node in the hashed list; a source outside the list
        // means the caller handed us an incomplete graph
        std::vector<const ggml_tensor *> index(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) {
            index[i] = nodes[i];
        }
        auto find = [&index](const ggml_tensor * t) -> int64_t {
            for (size_t i = 0; i < index.size(); ++i) {
                if (index[i] == t) {
                    return (int64_t) i;
                }
            }
            return edge_external;
        };
        // The same three markers canonical_edge() folds, in the same order:
        // absent, self, external, in-set position.
        auto edge_marker = [&find](const ggml_tensor * self, const ggml_tensor * target) -> int64_t {
            if (target == nullptr) {
                return edge_absent;
            }
            if (target == self) {
                return edge_self;
            }
            return find(target);
        };

        h.mix_u64((uint64_t) nodes.size());

        for (size_t i = 0; i < nodes.size(); ++i) {
            const ggml_tensor * t = nodes[i];
            if (t == nullptr) {
                return false;
            }

            llama_union_topology_excl e;
            if (i < excl.size() && excl[i].node == (int32_t) i) {
                e = excl[i];
            }
            const int32_t dims[5] = { e.dim, e.nout_dim, e.nrs_dim, e.head_dim, e.ntok_dim };
            for (int32_t d : dims) {
                if (d < -1 || d > 3) {
                    return false;
                }
            }
            // only the four real axes can be masked; a wider mask is a caller
            // bug, not an exclusion
            if ((e.dim_mask & (uint8_t) ~0x0fu) != 0) {
                return false;
            }

            // op type and flags
            h.mix_i64((int64_t) t->op);
            h.mix_u64((uint64_t) t->flags);

            // op parameters, all of them except the view-offset copy and the
            // one slot that carries a runtime VALUE.
            //
            // Where the runtime widths live today: union_nkv_k / union_nkv_v
            // carry n_kv in ne[], attn_inp_kq_mask in ne[0], union_out_ids in
            // ne[0], the recurrent copy views in ne[] and view_offs, and the
            // FLASH_ATTN_EXT slot 4 (n_kv_max) is the CUDA sparse-mask bound,
            // not a width.
            //
            // The exception is the CPU attention reduction length: a
            // TRUE_LEN_REDUCE SOFT_MAX / MUL_MAT node stores the number of
            // keys to reduce in op_params[3], and the bind rewrites it every
            // step. That is the op-param form of the same runtime width the
            // n_kv dimensions carry, so it is excluded by the same rule. The
            // exclusion is positional and recorded at build time, when the
            // slot is still 0, so it cannot depend on the value.
            {
                const size_t n_params = sizeof(t->op_params) / sizeof(int32_t);
                const size_t skip     = e.skip_view_data ? sizeof(size_t) / sizeof(int32_t) : 0;
                for (size_t p = skip; p < n_params; ++p) {
                    if (e.skip_op_param >= 0 && p == (size_t) e.skip_op_param) {
                        continue;
                    }
                    h.mix_i64((int64_t) t->op_params[p]);
                }
            }

            // shape: every dimension except those that carry a runtime
            // parameter
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                if (d == e.dim || d == e.nout_dim || d == e.nrs_dim || d == e.head_dim ||
                        d == e.ntok_dim || ((e.dim_mask >> d) & 1u) != 0) {
                    continue;
                }
                if (t->ne[d] < 0) {
                    return false;
                }
                h.mix_i64(t->ne[d]);
            }

            // name (the shape analysis keys its runtime roots on names)
            h.mix_str(t->name);

            // edges: sources then the view source, as indices into this list
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                h.mix_i64(edge_marker(t, t->src[j]));
            }
            h.mix_i64(edge_marker(t, t->view_src));
        }

        out = h.value();
        return true;
    }

private:
    uint64_t h_;
};
