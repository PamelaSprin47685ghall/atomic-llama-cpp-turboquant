#pragma once

//
// llama-topology-hash.h
//
// The single graph cache key.
//
// llama_topology_key_of() is the one place that answers "what graph is this?".
// Its digest is the only topology identity the union binder compares; every
// other reuse test in the tree either feeds it or is an address-level check
// that the digest cannot express (see "Two identities" below).
//
// The key is invariant to the runtime length parameters as parameters:
// n_kv, n_outputs, rs_z, n_tokens and the replay length are never read from a
// cparams field, a cache width or a state offset. They reach the key only as
// tensor shapes and view offsets, i.e. as facts about the graph as built. Two
// graphs built at the same shapes therefore have the same key whatever those
// parameters were, and a graph built at a different shape has a different key.
//
// The one deliberate exception is the LANE exclusion
// (llama_topology_key_excl): a lane that binds a parameter in place passes the
// dimensions the bind moves, and those dimensions are folded as a constant
// instead of their value. The token lane does this for its four token inputs,
// because they are graph nodes whose ne[0] the bind patches; without the
// exclusion the re-taken key would move and the bind would refuse itself. Every
// other lane passes none(), so every dimension is hashed exactly as before.
//
// The binder's address record (llama_nkv_record_t, llama-context.cpp) remains
// the authority for "may this exact graph be patched": the key says the
// topology is the same, the record says the tensor objects are the same ones.
// A bind needs both.
//
// Two identities
// --------------
//
// The union binder (llama-union-binder.h) decides whether a live graph can be
// patched in place instead of rebuilt. Its record (llama-context.cpp,
// llama_nkv_record_t) proves that the *same tensor objects* are still wired
// together, and it is deliberately address-based: it compares `ggml_tensor *`
// identities. That is the right test for "is this still the graph I recorded",
// and the wrong test for "is this the same graph as the one another context,
// another shape parameter set, or another process built".
//
// This header adds the second, weaker, stable identity:
//
//   llama_topology_same(a, b)  <=>  the two graphs have the same node
//   operations, the same topology-relevant op parameters, the same tensor
//   types and shapes, and the same edge structure expressed in canonical node
//   indices.
//
// What is hashed (per node, then folded in canonical order):
//
//   - `op` (ggml_op). The operation itself.
//   - `ne[0..3]` and `nb[0..3]`. Shape and layout. `nb` is a pure function of
//     type + ne + padding, so it carries no address.
//   - `type` (ggml_type).
//   - `view_offs`. A view's byte offset inside its source. It is a graph
//     constant written at construction, not an address; the binder itself
//     patches it as a runtime shape (llama-context.cpp, union_patch_value), so
//     it is topology, not identity.
//   - the topology-relevant `op_params` words. See
//     llama_topology_op_params_word() for the exact per-op list and the reason
//     for each entry.
//   - the edge list: src[0..GGML_MAX_SRC-1] and view_src, each mapped to the
//     canonical index of the referenced node, or to a reserved value for
//     "absent" / "external" / "self".
//   - `fixed`: whether the node's storage pre-exists the build (weights, KV
//     cache, input buffers) or is graph-owned (an intermediate or a view made
//     by this build). This is the same predicate the binder record uses
//     (llama-context.cpp, llama_nkv_record_t::record), and it is what tells a
//     weight from an activation of identical shape.
//
// What is deliberately NOT hashed, and why:
//
//   - `data`, `buffer`, `extra`, and every `ggml_tensor *` value: addresses.
//     They change with the allocator, with ASLR, and between two contexts that
//     hold structurally identical graphs.
//   - `name`: a debug label. It carries layer numbers and build counters, not
//     structure, and it differs between otherwise identical graphs.
//   - `flags`: runtime marks. GGML_TENSOR_FLAG_INPUT/OUTPUT/PARAM/COMPUTE are
//     set by the build walk; GGML_TENSOR_FLAG_ATTN_MASK,
//     GGML_TENSOR_FLAG_ATTN_SOFT_MAX and GGML_TENSOR_FLAG_ATTN_SCORES are set
//     by the attention builders from the operand types. Folding them in would
//     make the hash depend on whether the binder ran, which is exactly the
//     runtime perturbation this hash exists to ignore.
//   - the graph's node *order* and the graph's node count as such: the node
//     list is re-sorted into a canonical order before folding (see below), so
//     two builds that emit the same nodes in a different order hash equal.
//   - the runtime shape parameters as parameters (n_kv, n_outputs, rs_z,
//     n_tokens, replay length): the hash never reads a cparams field, a
//     KV-cache width or a recurrent-state offset. It reads only the graph.
//
// The last point is the whole design, and it has two halves that must not be
// confused:
//
//   invariant to the parameter: two graphs built at the same shapes hash
//   equal whatever n_kv / n_outputs / rs_z were. The parameter value is not
//   an input.
//
//   sensitive to the graph as built: those parameters reach the graph as
//   tensor shapes and view offsets. n_kv sizes the KV cache tensors, whose
//   `ne`/`nb` are hashed even though they are fixed leaves; n_outputs sizes
//   out_ids; rs_z selects a view offset. A graph built at 256 cells and the
//   same graph built at 1000 cells therefore DIFFER in `ne`/`nb`/`view_offs`,
//   and that difference IS hashed, because it is a real difference in the
//   graph as built. A digest that ignored it would call two different graphs
//   the same, which is the failure the binder cannot afford.
//
// Canonical order:
//
//   1. Every node reachable from the graph's node list is collected exactly
//      once (the walk follows src[] and view_src, and stops at a node whose
//      storage is fixed: that node is a leaf of the graph-owned topology).
//      A graph-owned INPUT leaf is admitted even when the node that consumes
//      it is fixed: after allocation that node owns a buffer, so the src[]
//      edge is the only edge the input has, and the width table and the shape
//      map name these tensors as runtime roots.
//      Fixed leaves are still hashed - being a leaf of the walk is not being
//      outside the digest - and their shapes are what makes a KV-cache width
//      difference visible.
//   2. Each node gets a provisional key: a 64-bit FNV-1a digest of its own
//      fields (op, type, ne, nb, view_offs, masked op_params, fixed) - no
//      edges. Nodes with equal keys are structurally interchangeable leaves or
//      interchangeable interiors at this stage.
//   3. Each node is coloured by its own key plus the colours of its edge
//      targets (sorted, so operand order does not matter), and the colouring
//      is iterated until it stops splitting. This is colour refinement
//      (1-dimensional Weisfeiler-Lehman): after it settles, two nodes share a
//      colour only if they are structurally equivalent up to the refinement's
//      known limits (it cannot separate, for example, two nodes on a cycle of
//      identical nodes - a shape no ggml decoder graph has, because every
//      node's `ne`/`nb` differ along a layer stack).
//   4. Nodes are ordered by (colour, key). Ties that survive are symmetric
//      nodes, and their relative order cannot matter to the digest because
//      their folded field bytes are identical.
//   5. Edges are resolved to canonical indices through that order, and the
//      whole serialization is folded with FNV-1a 64.
//
// The digest is therefore a pure function of the graph structure: no address,
// no allocation order, no traversal order, no process state. It is stable
// across processes and across two builds that emit the same nodes in a
// different order, and it changes exactly when the graph as built changes.
//
// Cost: one hash-table build over the nodes during collection, then O(edges)
// hash lookups per refinement round, and at most n rounds on a DAG (in
// practice a handful). A decoder graph at n ~ 10^3 hashes in O(n * edges)
// lookups, not the O(n^2 * edges) that a linear scan per edge would cost.
//
// The operand order of a node's sources is NOT hashed: ggml_add(a, b) and
// ggml_add(b, a) are the same topology and hash equal. This is deliberate. It
// is also the one place where the digest is coarser than the graph: a
// non-commutative op whose operands were swapped (ggml_sub) hashes equal to
// its swap. The digest is an identity for reuse and diagnostics, and the
// binder's own address-based record remains the authority for "may this exact
// graph be patched".
//
// FNV-1a 64 is the in-tree choice (ggml/src/ggml-moe-cache-common.h:557-564,
// ggml/src/ggml-rpc/ggml-rpc.cpp:241-248). It is not cryptographic; the
// collision probability for a 64-bit digest over the ~10^3 nodes of a decoder
// graph is far below the probability of an unrelated build difference, and
// this hash is a diagnostic identity, never a security boundary.
//
// Header-only. It allocates only the vectors its own call stack owns, and
// depends on nothing outside ggml.h. Included by llama-context.cpp,
// llama-nout-state.h (which holds a llama_topology_key) and by
// tests/test-topology-hash.cpp and tests/test-topology-cache-key.cpp.

#include "ggml.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

// FNV-1a 64: offset basis and prime, the same constants the rest of the tree
// uses (ggml-moe-cache-common.h:558,561).
constexpr uint64_t LLAMA_TOPOLOGY_FNV_OFFSET = 0xcbf29ce484222325ULL;
constexpr uint64_t LLAMA_TOPOLOGY_FNV_PRIME  = 0x100000001b3ULL;

// Reserved canonical edge targets. Real node indices are >= 0, so these can
// never collide with one.
enum llama_topology_edge : int32_t {
    LLAMA_TOPOLOGY_EDGE_ABSENT   = -1, // src[k] == nullptr
    LLAMA_TOPOLOGY_EDGE_EXTERNAL = -2, // a real tensor that is not part of the graph topology
    LLAMA_TOPOLOGY_EDGE_SELF     = -3, // the node itself (ggml_cast sets src[1] = result)
};

// Forward declaration for edge_marker() below, which hashes an edge target's
// node key before the definition later in this file. It must be at global
// scope: a declaration inside llama_topology_detail would make the
// unqualified call ambiguous.
struct llama_topology_key_excl;
inline uint64_t llama_topology_node_key(const ggml_tensor * x,
                                        const llama_topology_key_excl * excl = nullptr);

namespace llama_topology_detail {

// Map from a tensor to its position in the collected node set. A target that
// is absent from the map is not part of the topology. The map is built once
// per hash; every edge lookup goes through it, so a hash costs O(nodes *
// edges) hash-table lookups rather than O(nodes^2 * edges) linear scans.
using topology_index = std::unordered_map<const ggml_tensor *, size_t>;

// Marker for an edge target during the first refinement step: a stable value
// that says which *kind* of target the edge has, without naming a node.
inline uint64_t edge_marker(const ggml_tensor * self, const ggml_tensor * target,
                            const topology_index & index,
                            const llama_topology_key_excl * excl = nullptr) {
    if (target == nullptr) {
        return (uint64_t) LLAMA_TOPOLOGY_EDGE_ABSENT;
    }
    if (target == self) {
        return (uint64_t) LLAMA_TOPOLOGY_EDGE_SELF;
    }
    if (index.find(target) == index.end()) {
        return (uint64_t) LLAMA_TOPOLOGY_EDGE_EXTERNAL;
    }
    return LLAMA_TOPOLOGY_FNV_OFFSET ^ llama_topology_node_key(target, excl);
}

// Marker for an edge target during a refinement round: the target's current
// dense colour id, or a reserved marker for a target outside the node set.
inline uint64_t colour_marker_id(const ggml_tensor * self, const ggml_tensor * target,
                                 const topology_index & index,
                                 const std::vector<uint32_t> & colour) {
    if (target == nullptr) {
        return (uint64_t) LLAMA_TOPOLOGY_EDGE_ABSENT;
    }
    if (target == self) {
        return (uint64_t) LLAMA_TOPOLOGY_EDGE_SELF;
    }
    const auto it = index.find(target);
    if (it == index.end()) {
        return (uint64_t) LLAMA_TOPOLOGY_EDGE_EXTERNAL;
    }
    return (uint64_t) colour[it->second];
}

// Canonical index of an edge target in the canonical order, or a reserved
// marker. The index is a position in `nodes`, never an address.
inline int32_t canonical_edge(const ggml_tensor * self, const ggml_tensor * target,
                              const topology_index & index,
                              const std::vector<size_t> & position) {
    if (target == nullptr) {
        return LLAMA_TOPOLOGY_EDGE_ABSENT;
    }
    if (target == self) {
        return LLAMA_TOPOLOGY_EDGE_SELF;
    }
    const auto it = index.find(target);
    if (it == index.end()) {
        return LLAMA_TOPOLOGY_EDGE_EXTERNAL;
    }
    return (int32_t) position[it->second];
}

inline void fnv_bytes(uint64_t & h, const void * data, size_t n) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t) p[i];
        h *= LLAMA_TOPOLOGY_FNV_PRIME;
    }
}

inline void fnv_u8(uint64_t & h, uint8_t v) {
    h ^= (uint64_t) v;
    h *= LLAMA_TOPOLOGY_FNV_PRIME;
}

inline void fnv_u32(uint64_t & h, uint32_t v) {
    uint8_t b[4];
    for (int i = 0; i < 4; ++i) {
        b[i] = (uint8_t) ((v >> (8 * i)) & 0xffu);
    }
    fnv_bytes(h, b, sizeof(b));
}

inline void fnv_u64(uint64_t & h, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) {
        b[i] = (uint8_t) ((v >> (8 * i)) & 0xffu);
    }
    fnv_bytes(h, b, sizeof(b));
}

inline void fnv_i64(uint64_t & h, int64_t v) {
    fnv_u64(h, (uint64_t) v);
}

// The storage predicate of the binder record, verbatim
// (llama-context.cpp, llama_nkv_record_t::record): a tensor whose storage
// pre-exists the build is a leaf; everything else was made by this build.
inline bool is_fixed(const ggml_tensor * x) {
    return x->view_src == nullptr && (x->data != nullptr || x->buffer != nullptr);
}

// A graph-owned INPUT leaf: a tensor the build created and the caller fills
// (op NONE, INPUT flag, no PARAM flag, no view source, and named).
//
// The walk below descends src[] only through non-fixed tensors, and after
// ggml_backend_sched_alloc_graph() every non-view node owns a buffer, so the
// descent stops at the node that CONSUMES an input. An input that no view node
// consumes is then reachable through no edge at all, even though the union
// binder patches its shape: the output-gather index, the KQ mask and the
// token/position inputs are all in that position.
//
// The name test is not decoration. The key's exclusion is by name
// (llama_topology_key_excl), so an unnamed tensor that carries a runtime width
// could never be excluded and every bind would refuse itself. A graph input
// the build did not name is therefore not admitted.
inline bool is_graph_input_leaf(const ggml_tensor * x) {
    return x->op == GGML_OP_NONE && x->view_src == nullptr &&
           (x->flags & GGML_TENSOR_FLAG_INPUT) != 0 &&
           (x->flags & GGML_TENSOR_FLAG_PARAM) == 0 &&
           x->name[0] != '\0';
}

} // namespace llama_topology_detail

// The k-th `op_params` word that carries topology for `op`, or -1.
//
// op_params is one 64-byte bag per op, and each op fills a documented prefix
// of it. The mask below is the set of words that change *what the node
// computes* or *which region it addresses*, i.e. the set that must differ when
// two graphs are genuinely different. Words outside the mask are either unused
// (zero-filled by ggml_new_tensor_impl) or carry runtime values that the union
// binder patches, and folding those in would make the hash depend on the
// binder's state rather than on the graph's structure.
//
// One word is 4 bytes. The layout of each entry below is taken from the ggml
// constructor that writes it, not from memory:
//
//   GGML_OP_SCALE    words 0-1: { s, b } as two floats
//                    (ggml/src/ggml.c, ggml_scale_impl: `float params[2]`).
//                    word 1 is 0 for ggml_scale and the bias for
//                    ggml_scale_bias.
//   GGML_OP_SOFT_MAX words 0-1: { scale, max_bias } as two floats
//                    (ggml/src/ggml.c, ggml_soft_max_impl: `float params[2]`).
//                    The mask tensor is src[1], i.e. an edge, not a parameter.
//   GGML_OP_ROPE     words 1, 2, 4, 5, 6, 7, 8, 9, 10
//                    (ggml/src/ggml.c, ggml_rope_impl: `int32_t params[15]`).
//                    Word 0 is `n_past` and word 3 is `n_ctx`; both are
//                    runtime positions the caller rewrites every step, so they
//                    are excluded. The hashed words are n_dims(1), mode(2),
//                    n_ctx_orig(4), freq_base(5), freq_scale(6),
//                    ext_factor(7), attn_factor(8), beta_fast(9),
//                    beta_slow(10). Words 11-14 hold the MROPE sections and
//                    are 0 for a plain rope; a MROPE graph is outside the
//                    binder's support range (llama-union-binder.h), so they
//                    are not hashed.
//   GGML_OP_MUL_MAT  word 0: precision, an `int32_t` ggml_prec
//                    (ggml/src/ggml.c, ggml_mul_mat_set_prec). It selects the
//                    arithmetic the node requests, so two graphs that differ
//                    only in precision are different graphs.
//   GGML_OP_VIEW     words 0-1: the view's byte offset, stored as a `size_t`
//                    (ggml/src/ggml.c, ggml_view_impl). Redundant with
//                    `view_offs`; see the switch below.
//
// Every other op contributes no op_params word. That is deliberate for:
//
//   - GGML_OP_CPY, GET_ROWS, ADD_ID: their constructors write none
//     (ggml_cpy_impl, ggml_get_rows, ggml_add_id). The destination type is a
//     `type` field and the source type is an edge, both already hashed.
//   - GGML_OP_MUL_MAT_ID: its expert counts are shapes the `ne` fields carry.
//   - the attention ops' ordered-attention flags: binder/graph state, see the
//     header comment.
//
// A new op whose parameters matter must be added here. An op that is not
// listed is hashed by op + type + shape + edges only, which is a coarser but
// never a wrong identity: it can fail to distinguish two graphs, never falsely
// distinguish two identical ones.
//
// The rope entry is not a leading prefix, so the mask is expressed as an
// explicit list of words rather than a count. `llama_topology_op_params_word`
// returns -1 past the end.
inline int llama_topology_op_params_word(enum ggml_op op, int k) {
    switch (op) {
        case GGML_OP_SCALE:
        case GGML_OP_SOFT_MAX:
            return k < 2 ? k : -1;
        case GGML_OP_ROPE: {
            static const int words[] = { 1, 2, 4, 5, 6, 7, 8, 9, 10 };
            return k < (int) (sizeof(words) / sizeof(words[0])) ? words[k] : -1;
        }
        case GGML_OP_MUL_MAT:
            return k < 1 ? 0 : -1;
        case GGML_OP_VIEW:
            // A view's offset is written to op_params as a size_t
            // (ggml/src/ggml.c, ggml_view_impl). The same value is also in
            // `view_offs`, so this word is redundant; it is hashed anyway
            // because a view whose two copies disagree is a graph bug worth
            // distinguishing.
            return k < 2 ? k : -1;
        default:
            return -1;
    }
}

// True when the hash reads at least one op_params word for `op`. False means
// the op's parameters, if it has any, are not hashed; the digest still works,
// it just hashes less. tests/test-topology-hash.cpp asserts the true cases and
// the false cases it relies on.
inline bool llama_topology_op_params_covered(enum ggml_op op) {
    return llama_topology_op_params_word(op, 0) != -1;
}

// Canonical node order of the graph reachable from `gf`'s node list, plus the
// two lookup tables the fold needs.
//
// `nodes` is the canonical order: every node is distinct, and the set is
// closed under src[]/view_src except at fixed nodes, which are leaves by
// construction. `index` maps a node to its position in `seen` (the collection
// order), and `position` maps that position to its place in `nodes`. Both are
// built once; every edge lookup in the refinement and in the fold is a hash
// lookup instead of a scan.
inline void llama_topology_canonical_nodes(ggml_cgraph * gf,
                                           std::vector<const ggml_tensor *> & nodes,
                                           llama_topology_detail::topology_index & index,
                                           std::vector<size_t> & position,
                                           const llama_topology_key_excl * excl = nullptr) {
    using namespace llama_topology_detail;

    nodes.clear();
    index.clear();
    position.clear();
    if (gf == nullptr) {
        return;
    }

    // 1. collect: walk from every graph node through src[]/view_src, stopping
    //    at fixed nodes. The walk is a stack, not recursion: a decoder graph
    //    is deep enough that recursion is a real risk. `index` is filled as
    //    nodes are first seen, so the walk itself is linear too.
    std::vector<const ggml_tensor *> seen;
    std::vector<const ggml_tensor *> stack;
    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; ++i) {
        stack.push_back(ggml_graph_node(gf, i));
    }
    while (!stack.empty()) {
        const ggml_tensor * x = stack.back();
        stack.pop_back();
        if (x == nullptr) {
            continue;
        }
        if (!index.emplace(x, seen.size()).second) {
            continue;
        }
        seen.push_back(x);
        // The descent stops at a fixed node: its storage pre-exists the build,
        // so its sources are weights or storage, not topology. One exception,
        // and it is not a relaxation of is_fixed: a graph-owned INPUT leaf is a
        // runtime root the binder patches, and after allocation every consumer
        // of one owns a buffer, so that src[] edge is the only edge the input
        // has. Without this the walk never reaches the output-gather index, the
        // KQ mask or the token/position inputs, and the width table - built
        // over this very set - records no n_outputs root even though the shape
        // map does. The leaf itself stays a leaf: it is pushed, hashed, and its
        // own (empty) src[] is not walked.
        const bool fixed = is_fixed(x);
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            const ggml_tensor * s = x->src[k];
            if (s == nullptr || s == x) {
                continue;
            }
            if (!fixed || is_graph_input_leaf(s)) {
                stack.push_back(s);
            }
        }
        if (!fixed && x->view_src != nullptr && x->view_src != x) {
            stack.push_back(x->view_src);
        }
    }

    // An empty graph is the documented empty key: llama_topology_key_of()
    // folds nodes.size() == 0 and returns LLAMA_TOPOLOGY_FNV_OFFSET with
    // n_nodes == 0. Return here, before the refinement: with no nodes the
    // round loop never runs, `converged` would stay false, and the DAG assert
    // below would fire on a graph that has nothing to converge. The assert
    // stays hard for every non-empty graph.
    if (seen.empty()) {
        return;
    }

    // 2. provisional keys
    std::vector<uint64_t> key(seen.size());
    for (size_t i = 0; i < seen.size(); ++i) {
        key[i] = llama_topology_node_key(seen[i], excl);
    }

    // 3. colour refinement. The colour of a node is a digest of its own key
    //    plus the colours of its edge targets, iterated until the partition
    //    stops splitting. One step is not enough: two nodes that share a key
    //    and whose sources also share a key would stay tied, and a later
    //    difference one level further down would not reach the sort key.
    //
    //    The refinement starts from the one-step colour (own key + the sorted
    //    multiset of the *keys* of the targets, so the first step already
    //    distinguishes edge kinds and target shapes) and then iterates the
    //    full neighbour colours.
    //
    //    Termination: each round ranks the signature (current id, sorted
    //    neighbour ids) to dense ids. The signature starts with the current
    //    id, so the new partition refines the old one and the number of
    //    classes is non-decreasing. It is bounded by n, so after at most n
    //    rounds the count stops growing: that is the fixed point, and the
    //    loop exits through it. Rehashing the previous colour (the previous
    //    form) kept changing the values after the partition was stable, so
    //    the fixed-point test never fired on a real graph.
    //
    //    Absent / external / self edges contribute a reserved marker, so an
    //    edge to a weight is distinguishable from no edge at all.
    std::vector<uint64_t> sig(seen.size());
    std::vector<uint64_t> targets;
    for (size_t i = 0; i < seen.size(); ++i) {
        const ggml_tensor * x = seen[i];
        targets.clear();
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            targets.push_back(edge_marker(x, x->src[k], index, excl));
        }
        targets.push_back(edge_marker(x, x->view_src, index, excl));
        std::sort(targets.begin(), targets.end());

        uint64_t h = LLAMA_TOPOLOGY_FNV_OFFSET;
        fnv_u64(h, key[i]);
        for (uint64_t t : targets) {
            fnv_u64(h, t);
        }
        sig[i] = h;
    }

    // Dense ranking of a signature vector: equal signatures share an id, and
    // the ids are ordered by the signature value. Deterministic for a given
    // graph, which is all the canonical order needs.
    auto rank = [](const std::vector<uint64_t> & s, std::vector<uint32_t> & out) -> size_t {
        std::vector<uint64_t> uniq = s;
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        for (size_t i = 0; i < s.size(); ++i) {
            out[i] = (uint32_t) (std::lower_bound(uniq.begin(), uniq.end(), s[i]) - uniq.begin());
        }
        return uniq.size();
    };

    std::vector<uint32_t> colour(seen.size());
    size_t n_colour = rank(sig, colour);

    std::vector<uint64_t> nsig(seen.size());
    std::vector<uint32_t> next(seen.size());
    bool converged = false;
    for (size_t round = 0; round < seen.size(); ++round) {
        for (size_t i = 0; i < seen.size(); ++i) {
            const ggml_tensor * x = seen[i];
            targets.clear();
            targets.push_back((uint64_t) colour[i]);
            for (int k = 0; k < GGML_MAX_SRC; ++k) {
                targets.push_back(colour_marker_id(x, x->src[k], index, colour));
            }
            targets.push_back(colour_marker_id(x, x->view_src, index, colour));
            // the current id stays first; the neighbour ids are order-free
            std::sort(targets.begin() + 1, targets.end());

            uint64_t h = LLAMA_TOPOLOGY_FNV_OFFSET;
            for (uint64_t t : targets) {
                fnv_u64(h, t);
            }
            nsig[i] = h;
        }
        const size_t n_next = rank(nsig, next);
        if (n_next == n_colour) {
            converged = true;
            break;
        }
        colour.swap(next);
        n_colour = n_next;
    }
    // The class count is non-decreasing and bounded by n, so the loop always
    // exits through the fixed-point test. The assert guards the invariant.
    GGML_ASSERT(converged);

    // 4. order by (colour, key). Stable ties are symmetric nodes; see the
    //    header comment.
    std::vector<size_t> order(seen.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (colour[a] != colour[b]) {
            return colour[a] < colour[b];
        }
        if (key[a] != key[b]) {
            return key[a] < key[b];
        }
        return a < b; // deterministic tie-break; identity of symmetric nodes only
    });

    nodes.reserve(seen.size());
    position.resize(seen.size());
    for (size_t k = 0; k < order.size(); ++k) {
        nodes.push_back(seen[order[k]]);
        position[order[k]] = k;
    }
}

// Which runtime dimensions the key must NOT see, because the binder patches
// them in place on the lane this graph was built for.
//
// The key hashes every ne[] of every canonical node (llama_topology_key_of,
// the loop over GGML_MAX_DIMS). That is correct for a graph that is only ever
// compared against another graph built at the same shapes, and it is WRONG for
// a graph the union binder patches: the bind moves exactly those ne[] values,
// so the re-taken key would move and the bind would refuse itself. This struct
// is how the caller says "this dimension is a runtime parameter of this lane,
// do not fold it".
//
// The set is per LANE, and a lane is decided by the caller, not by the key:
//   - the n_kv / n_outputs / rs_z lanes pass `none()`: their roots are fixed
//     leaves (the KV cache, the out_ids leaf, the recurrent state), the bind
//     patches those leaves' ne[] and view_offs, and the canonical walk stops
//     at them, so a patched leaf is not part of the hashed node set at all;
//   - the token lane passes `token_lane()`: its token inputs are graph NODES
//     (each is consumed by get_rows / rope / set_rows), so their ne[0] IS
//     folded, and the bind moves it.
//
// Every other lane keeps the default, so a token-width change there still
// moves the key and still refuses the bind: fail-closed by default.
//
// The exclusion names a dimension of a named tensor, not a node index: the
// canonical order is computed inside the key and is not the graph's order, so
// an index would be a guess. A name that is not in the graph excludes nothing.
// The token lane's roots, as one list. The width table and the key both read
// it, so the two cannot name different sets (see the invariant below).
//
//   union_inp_tokens  [n_tokens]                      factor 1
//   union_inp_pos     [n_tokens * n_pos_per_embd]     factor n_pos_per_embd
//   union_inp_k_idxs  [n_tokens]                      factor 1
//   union_inp_v_idxs  [n_tokens] or [n_tokens * n_embd_v_gqa_max]
//
// Every one of them is a graph NODE (each is consumed by ggml_get_rows, rope
// or ggml_set_rows), so the key folds its ne[0] unless the token lane excludes
// it. The width is always ne[0]: each tensor is 1-D, or 1-D times a constant
// factor, and the factor multiplies ne[0].
constexpr const char * LLAMA_TOKEN_LANE_ROOTS[] = {
    "union_inp_tokens",
    "union_inp_pos",
    "union_inp_k_idxs",
    "union_inp_v_idxs",
};
constexpr int LLAMA_TOKEN_LANE_ROOTS_N =
    (int) (sizeof(LLAMA_TOKEN_LANE_ROOTS) / sizeof(LLAMA_TOKEN_LANE_ROOTS[0]));

struct llama_topology_key_excl {
    struct dim {
        const char * name = nullptr; // tensor name the width lives in
        int          d    = -1;      // its dimension (0..3)
    };
    // One slot per token-lane root. The array is sized from the root list, so
    // adding a root to LLAMA_TOKEN_LANE_ROOTS cannot silently overflow it.
    dim dims[LLAMA_TOKEN_LANE_ROOTS_N];
    int count = 0;

    static llama_topology_key_excl none() {
        return llama_topology_key_excl{};
    }

    // The token lane: every root the width table marks `ntok_dim`.
    //
    // INVARIANT (the two headers must not drift): the key's exclusion set
    // names exactly the tensors llama_union_width_table::build() marks with
    // `ntok_dim = 0`, and both read LLAMA_TOKEN_LANE_ROOTS. The width table
    // records a root only after its exact-length check passes, so naming a
    // root here that the table would refuse is harmless (a refused table
    // clears itself and the key is never taken through it), and naming FEWER
    // roots here than the table records is the defect: the key would fold a
    // dimension the bind patches, the re-taken key would move, and
    // union_rebind_apply() would refuse the bind (c_refused_topology) and the
    // caller would rebuild. token_lane_roots_match() below is the runtime
    // form of the invariant, and the matrix asserts it.
    static llama_topology_key_excl token_lane() {
        llama_topology_key_excl e;
        for (int i = 0; i < LLAMA_TOKEN_LANE_ROOTS_N; ++i) {
            e.dims[i].name = LLAMA_TOKEN_LANE_ROOTS[i];
            e.dims[i].d    = 0;
        }
        e.count = LLAMA_TOKEN_LANE_ROOTS_N;
        return e;
    }

    // True when this set names `name` at dimension `d`.
    bool names(const char * name, int d) const {
        if (name == nullptr) {
            return false;
        }
        for (int i = 0; i < count; ++i) {
            if (dims[i].d == d && dims[i].name != nullptr &&
                    std::strcmp(name, dims[i].name) == 0) {
                return true;
            }
        }
        return false;
    }

    // True when `roots` (a list of names the width table recorded) is exactly
    // the set this exclusion names at dimension 0. Empty `roots` means the
    // table recorded no token root, which this set covers only when it is
    // empty too.
    bool token_lane_roots_match(const std::vector<const char *> & roots) const {
        int matched = 0;
        for (const char * r : roots) {
            if (!names(r, 0)) {
                return false;
            }
            ++matched;
        }
        return matched == count;
    }

    // The shape map's exclusion, in the same name form.
    //
    // The four slots above are the token lane's fixed root list. A graph the
    // binder patches carries DERIVED tensors as well -- the GET_ROWS result
    // whose ne[1] inherits the copy index's length, the matmul whose ne[0]
    // inherits the KV width -- and those carry no runtime name at all. The map
    // knows them by identity, but a key is compared across two graph OBJECTS
    // (the record's graph and a later build's), so an identity match cannot
    // carry over. Their NAMES can: the graph builder names every node it
    // creates (llama-graph.cpp cb()), so the same node has the same name in
    // both builds.
    //
    // This is the list union_topology_key_excl(map, record_t) fills from the
    // coefficient table: one (name, dim) pair per patched axis.
    struct name_dim {
        const char * name = nullptr;
        int          d    = -1;
    };
    std::vector<name_dim> extra;

    void add_name(const char * name, int d) {
        if (name == nullptr || name[0] == '\0' || d < 0 || d > 3) {
            return;
        }
        for (const name_dim & e : extra) {
            if (e.d == d && e.name != nullptr && std::strcmp(e.name, name) == 0) {
                return;
            }
        }
        extra.push_back(name_dim{name, d});
    }

    bool excludes(const ggml_tensor * x, int d) const {
        if (x == nullptr || x->name[0] == '\0') {
            return false;
        }
        if (names(x->name, d)) {
            return true;
        }
        for (const name_dim & e : extra) {
            if (e.d == d && e.name != nullptr && std::strcmp(e.name, x->name) == 0) {
                return true;
            }
        }
        return false;
    }
};


// The provisional sort key of the canonical order.
//
// It is a digest of one node's own fields, WITHOUT its edges. The exclusions
// are the caller's: an axis the binder patches is folded as a reserved
// constant instead of its value, so two graphs that differ only in a runtime
// width seed the colour refinement with the same key, sort the same way, and
// fold the same sequence. Without this the ORDER itself moved with the width,
// and no final-fold exclusion could make the key invariant: an exclusion
// naming every axis of every node still separated two builds.
//
// The constants are the ones llama_topology_key_of() folds: INT64_MIN for a
// shape (never below -1) and UINT64_MAX for a stride (never a real byte
// count). view_offs is NOT excluded: the exclusion set does not express a view
// offset, and the token lane's moving view offsets are handled by the width
// table's skip_view_data, not by the key.
inline uint64_t llama_topology_node_key(const ggml_tensor * x,
                                        const llama_topology_key_excl * excl) {
    using namespace llama_topology_detail;

    uint64_t h = LLAMA_TOPOLOGY_FNV_OFFSET;

    fnv_u32(h, (uint32_t) x->op);
    fnv_u32(h, (uint32_t) x->type);

    const bool have = excl != nullptr;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        fnv_i64(h, have && excl->excludes(x, i) ? (int64_t) INT64_MIN : x->ne[i]);
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        fnv_u64(h, have && excl->excludes(x, i) ? (uint64_t) UINT64_MAX : (uint64_t) x->nb[i]);
    }

    fnv_i64(h, (int64_t) x->view_offs);
    fnv_u8 (h, is_fixed(x) ? 1 : 0);

    for (int k = 0; ; ++k) {
        const int w = llama_topology_op_params_word(x->op, k);
        if (w < 0) {
            break;
        }
        fnv_u32(h, (uint32_t) x->op_params[w]);
    }

    return h;
}
// The single graph cache key.
//
// `hash` is the digest of llama_topology_key_of(). `n_nodes` is the size of
// the canonical node set the digest was folded over - the same number the
// digest already folds in as its first word (see llama_topology_key_of), so it
// is redundant by design: it is carried alongside so a caller can compare two
// keys without re-deriving it, and so a diagnostic can report the graph's size
// without a second graph walk.
//
// Nothing in this struct derives from n_kv, n_outputs, rs_z, n_tokens or the
// replay length as VALUES. Those reach the key only through the shapes and
// view offsets of the graph that was hashed, minus the dimensions the caller
// excluded (llama_topology_key_excl). A key is a pure function of a graph and
// of the exclusion the caller asked for.
struct llama_topology_key {
    uint64_t hash    = 0; // 0 means "no graph recorded"; see llama_topology_key_of
    uint32_t n_nodes = 0;
};

// The key of the topology reachable from `gf`'s node list.
//
// Empty graph (no nodes) returns LLAMA_TOPOLOGY_FNV_OFFSET with n_nodes == 0,
// i.e. the digest of the empty serialization. A null graph returns the same
// value. Neither is 0, so `hash == 0` is reserved for "no graph recorded" and
// never names a real graph.
//
// `excl` names the dimensions the caller's lane binds at runtime. The default
// (llama_topology_key_excl::none()) hashes every dimension of every node,
// which is the behaviour every lane had before the token lane existed.
//
// A node whose name matches an entry keeps its ne[d] out of the digest but
// still folds everything else: op, type, the other three dimensions, all four
// nb[], view_offs, the fixed flag, the op_params words and the edge structure.
// The excluded dimension is folded as one reserved constant, so the
// serialization keeps its shape and two nodes that differ ONLY in an excluded
// dimension hash equal - which is the point - while any other difference still
// separates them.
inline llama_topology_key llama_topology_key_of(ggml_cgraph * gf,
                                                const llama_topology_key_excl & excl =
                                                        llama_topology_key_excl::none()) {
    using namespace llama_topology_detail;

    std::vector<const ggml_tensor *> nodes;
    topology_index index;
    std::vector<size_t> position;
    // The SAME exclusion seeds the canonical order: an excluded axis must not
    // reach the provisional sort key, or two graphs that differ only in a
    // runtime width would colour and sort differently and the fold sequence
    // itself would move.
    llama_topology_canonical_nodes(gf, nodes, index, position, &excl);

    // An empty graph is the empty serialization: no bytes are hashed, so the
    // digest is the FNV offset and the node count is 0. Folding the count here
    // would hash eight zero bytes and produce a different value, so the empty
    // case returns before the fold. This is the contract the header states and
    // tests/test-topology-hash.cpp and test-topology-cache-key.cpp pin.
    if (nodes.empty()) {
        llama_topology_key key;
        key.hash    = LLAMA_TOPOLOGY_FNV_OFFSET;
        key.n_nodes = 0;
        return key;
    }

    uint64_t h = LLAMA_TOPOLOGY_FNV_OFFSET;

    // The node count is the first word of the serialization, so it is already
    // part of `hash`. llama_topology_same() compares it again as a cheap
    // guard; that is redundancy on purpose, not a second key.
    fnv_u64(h, (uint64_t) nodes.size());

    // An axis the caller excluded is folded as a reserved constant instead of
    // its value, in BOTH the ne[] and the nb[] pass. No real ne[] can carry
    // INT64_MIN (a shape is never below -1) and no real stride can carry
    // UINT64_MAX (a stride is a byte count of a live tensor), so an excluded
    // axis can never alias a real one, and the two passes use different
    // constants so a shape and a stride cannot alias each other either.
    //
    // The nb[] pass must honour the exclusion for the same reason the ne[]
    // pass does: union_rebind_apply() patches ne[] AND nb[] (union_patch_value
    // carries both, and its add() applies the nb[] coefficients), and a 1-D
    // token root's nb[1..3] are derived from ne[0] by the allocation. A key
    // that folded the strides would therefore move on every token-width bind
    // even with every ne[] excluded.
    //
    // none() excludes nothing, so every caller that passes it gets the same
    // digest it got before this rule existed.
    constexpr int64_t  excluded_dim    = INT64_MIN;
    constexpr uint64_t excluded_stride = UINT64_MAX;

    for (const ggml_tensor * x : nodes) {
        fnv_u32(h, (uint32_t) x->op);
        fnv_u32(h, (uint32_t) x->type);

        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            fnv_i64(h, excl.excludes(x, i) ? excluded_dim : x->ne[i]);
        }
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            fnv_u64(h, excl.excludes(x, i) ? excluded_stride : (uint64_t) x->nb[i]);
        }

        fnv_i64(h, (int64_t) x->view_offs);
        fnv_u8 (h, is_fixed(x) ? 1 : 0);

        int n_words = 0;
        for (int k = 0; ; ++k) {
            const int w = llama_topology_op_params_word(x->op, k);
            if (w < 0) {
                break;
            }
            fnv_u32(h, (uint32_t) x->op_params[w]);
            ++n_words;
        }
        // the count is part of the serialization so that two different masks
        // cannot alias by concatenation
        fnv_u8(h, (uint8_t) n_words);

        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            fnv_i64(h, (int64_t) canonical_edge(x, x->src[k], index, position));
        }
        fnv_i64(h, (int64_t) canonical_edge(x, x->view_src, index, position));
    }

    llama_topology_key key;
    key.hash    = h;
    key.n_nodes = (uint32_t) nodes.size();
    return key;
}

// True when two keys name the same graph.
//
// The node count is folded into `hash` already, so comparing it here is
// redundant by design: it is a cheap guard against a 64-bit digest collision
// between two graphs of different size, and it costs one integer compare. A
// caller that wants a single number uses `hash`; a caller that wants the
// binder's answer uses this.
//
// It does NOT say "may this exact graph be patched": that is the binder's
// address record (llama_nkv_record_t). Two keys can be equal for two graphs
// whose tensors are different objects.
inline bool llama_topology_same(const llama_topology_key & a, const llama_topology_key & b) {
    return a.hash == b.hash && a.n_nodes == b.n_nodes;
}

// Stable digest of the topology reachable from `gf`'s node list.
//
// This is the `hash` member of llama_topology_key_of(), and nothing else: one
// implementation, two spellings. Callers that need the node count or the
// comparison use the key; callers that only want the number (diagnostics,
// tests) use this.
inline uint64_t llama_topology_hash(ggml_cgraph * gf,
                                    const llama_topology_key_excl & excl =
                                            llama_topology_key_excl::none()) {
    return llama_topology_key_of(gf, excl).hash;
}
