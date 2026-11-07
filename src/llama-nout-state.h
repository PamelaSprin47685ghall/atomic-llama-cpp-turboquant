#pragma once

//
// llama-nout-state.h
//
// Per-context state for the union-graph runtime binder
// (env: GGML_TP5_UNION_BINDER). The legacy
// LLAMA_GRAPH_UNION_OUTPUTS_RUNTIME flag is retired.
//
// State placement, exactly as measured against the stock baseline (no
// blanket "layout unchanged" claim: several classes DO gain members below):
//
//   - llama_context (llama-context.h) DOES gain one opaque member: a
//     std::unique_ptr<llama_nkv_ctx_state> binder-state slot (see below),
//     holding the graph record + counters + the one-shot envelope-suppress
//     flag. Internal layout changes by design; the public llama.h ABI is
//     untouched.
//   - llm_graph_params DOES gain n_outputs_build (allocation envelope).
//   - llm_graph_input_out_ids DOES gain bound_outputs (executed width).
//   - llama_kv_cache_context DOES gain nkv_build_override (build width).
//   - llm_graph_result DOES gain an owner back-pointer member.
//
// This internal header defines one state struct plus free-function
// accessors; it is included by llama-context.h (for the forward declaration
// and the opaque member) and by the owning TUs.
//

#include "llama-topology-hash.h"
#include "llama-union-binder.h"
#include "llama-union-width-table.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

struct ggml_tensor;
struct llama_context;
struct llm_graph_params;
class  llm_graph_result;

// One tensor's shape sensitivity to one runtime parameter or the product of
// the n_kv and n_tokens parameter deltas.
//
// The coefficients are NOT sampled. They are derived at graph-record time by
// the symbolic shape pass in llama-context.cpp: every tagged runtime root
// (K/V view, KQ mask, out_ids, recurrent state view) starts with an explicit
// formula taken from its construction site, and every dependent consumer is
// derived by applying the exact ggml op shape rule to its sources. Any op
// whose shape rule is not implemented makes the analysis fail closed; the
// binder is then not used for that graph.
struct llama_union_shape_coeff {
    uint32_t tensor = 0;
    int64_t  ne[4] = {};
    int64_t  nb[4] = {};
    int64_t  view_offs = 0;
    int64_t  root_offs = 0;

    // Values at the record's base state. Each bind sums base, linear terms,
    // and the n_kv * n_tokens cross term.
    // so repeated binds are idempotent and a failed validation never leaves
    // a partially patched graph.
    int64_t  base_ne[4] = {};
    int64_t  base_nb[4] = {};
    int64_t  base_view_offs = 0;
    int64_t  base_root_offs = 0;
};

// Per-parameter dependent-tensor lists. A list is valid only when the
// symbolic pass completed for that parameter; invalid means "do not bind this
// parameter, fall back to a full rebuild".
struct llama_union_shape_map {
    std::vector<llama_union_shape_coeff> nkv;
    std::vector<llama_union_shape_coeff> nout;
    std::vector<llama_union_shape_coeff> rs_head;
    std::vector<llama_union_shape_coeff> rs_len;
    std::vector<llama_union_shape_coeff> ntok;
    std::vector<llama_union_shape_coeff> nkv_ntok;

    uint32_t base_nkv     = 0;   // build width of the live graph
    uint32_t base_nout    = 0;   // build envelope of the live graph
    uint32_t base_ntok    = 0;
    uint32_t base_rs_head = 0;   // build head of the live graph
    uint32_t base_n_rs    = 0;   // build recurrent state length of the live graph

    bool nkv_valid     = false;
    bool nout_valid    = false;
    bool ntok_valid    = false;
    bool rs_head_valid = false;
    bool nrs_valid     = false;

    // Set by union_shape_analyze() when the refusal came from a fused-QKV
    // projection slice (LLM_FUSED_QKV_SLICE_MARKER): the token lane cannot
    // bind the non-contiguous Q/K/V views of one fused matmul.
    bool fused_qkv_refused = false;

    bool any_valid() const {
        return nkv_valid || nout_valid || rs_head_valid || nrs_valid || ntok_valid;
    }
};

// Resolve the shape map's coefficient table into the width table's runtime
// dimensions.
//
// The coefficient table indexes the RECORD list (llama_nkv_ctx_state::
// record_t, the pre-allocation DFS order union_shape_analyze() walked). The
// width table is built over the CANONICAL node list
// (llama_topology_canonical_nodes(), the set the topology hash hashes). Those
// are NOT the same list -- the canonical set is the larger one, because
// allocation adds the scheduler's split copies -- so a coefficient's `tensor`
// is a record index and must never be used as a position in the hashed list.
//
// The lookup is by tensor IDENTITY: the record entry's pointer is found in the
// hashed list. A coefficient whose tensor is not in the hashed list is
// skipped: its shape does not enter the hash, so there is nothing to exclude.
// A fixed leaf is the common case -- the canonical walk stops at a fixed node,
// so the KV-cache storage and the recurrent state storage are not members,
// while the views built over them are.
//
// Every axis with a nonzero ne[] or nb[] coefficient is a runtime axis. A
// nonzero view_offs/root_offs coefficient means the view offset moves, which
// the hash skips through skip_view_data. Both spellings are filled: the named
// slot of the parameter the coefficient belongs to (`which`, see the call
// sites) and the exact dim_mask. This is the same set the patch loop in
// union_rebind_apply() walks, so the table and the bind cannot disagree about
// which dimensions are runtime.
//
// which: 0 n_kv, 1 n_outputs, 2 rs head, 3 rs length, 4 n_tokens, 5 n_kv*n_tokens
inline std::vector<llama_union_topology_excl> union_width_runtime_dims(
        const llama_union_shape_map & map,
        const std::vector<ggml_tensor *> & record_t,
        const std::vector<ggml_tensor *> & hashed_nodes) {
    std::vector<llama_union_topology_excl> acc(hashed_nodes.size());
    std::vector<uint8_t> used(hashed_nodes.size(), 0);
    for (size_t i = 0; i < hashed_nodes.size(); ++i) {
        acc[i].node = (int32_t) i;
    }

    auto take = [&](const std::vector<llama_union_shape_coeff> & list, int which) {
        for (const llama_union_shape_coeff & c : list) {
            if (c.tensor >= record_t.size()) {
                continue;   // the map names a tensor this record does not have
            }
            const ggml_tensor * t = record_t[c.tensor];
            if (t == nullptr) {
                continue;
            }
            for (size_t i = 0; i < hashed_nodes.size(); ++i) {
                if (hashed_nodes[i] != t) {
                    continue;
                }
                llama_union_topology_excl & e = acc[i];
                for (int d = 0; d < 4; ++d) {
                    if (c.ne[d] == 0 && c.nb[d] == 0) {
                        continue;
                    }
                    e.dim_mask |= (uint8_t) (1u << d);
                    switch (which) {
                        case 0: e.dim      = d; break;
                        case 1: e.nout_dim = d; break;
                        case 2: e.head_dim = d; break;
                        case 3: e.nrs_dim  = d; break;
                        case 4: e.ntok_dim = d; break;
                        default:
                            // the cross term: the axis follows both n_kv and
                            // n_tokens, so it is named on both
                            e.dim      = d;
                            e.ntok_dim = d;
                            break;
                    }
                }
                if (c.view_offs != 0 || c.root_offs != 0) {
                    e.skip_view_data = true;
                }
                used[i] = 1;
                break;
            }
        }
    };

    take(map.nkv,      0);
    take(map.nout,     1);
    take(map.rs_head,  2);
    take(map.rs_len,   3);
    take(map.ntok,     4);
    take(map.nkv_ntok, 5);

    std::vector<llama_union_topology_excl> out;
    for (size_t i = 0; i < acc.size(); ++i) {
        if (used[i] != 0) {
            out.push_back(acc[i]);
        }
    }
    return out;
}

// The dimensions the single graph cache key must not see for the lane a
// record was built on.
//
// The key folds every ne[] of every canonical node. That is right for a graph
// that is never patched, and wrong for one the binder patches: the bind moves
// exactly those ne[] values, so a key re-taken after the bind would move and
// the bind would refuse itself. The lane decides which dimensions those are,
// and the lane is a property of the record, not of the parameter:
//
//   - the n_kv / n_outputs / rs_z lanes: none(). Their roots are fixed leaves
//     (the KV cache, the out_ids leaf, the recurrent state). The canonical walk
//     stops at a fixed leaf, so a patched leaf is not in the hashed node set at
//     all, and no exclusion is needed. Passing one here would be dead weight.
//
//   - the token lane (shape_map.ntok_valid, i.e. the symbolic token shape map
//     was built for this graph): the token inputs. They ARE nodes (each is
//     consumed by ggml_get_rows / rope / ggml_set_rows), so their ne[0] is
//     folded, and the bind patches it. llama_topology_key_excl::token_lane()
//     names them, from the same LLAMA_TOKEN_LANE_ROOTS list the width table
//     records.
//
// A lane that is not the token lane keeps every dimension hashed, so a
// token-width change there still moves the key and still refuses the bind:
// fail-closed by default, as before this exclusion existed.
//
// This is a free function in a header so the test TU calls the SAME function
// the binder calls. A test that re-implemented the mapping would stay green
// through a regression that made the call site pass none().
//
// The parameter is the shape map, not the whole state: `ntok_valid` is the
// only field the lane depends on, and taking the smaller type keeps the
// function callable from a test that has a map but no context.
//
// `llama_union_shape_map` is defined above in this header, so this is both the
// declaration and the definition. `inline` so every TU that includes the header
// shares one definition (the test TU and llama-context.cpp both call it).
inline llama_topology_key_excl union_topology_key_excl(const llama_union_shape_map & shape_map) {
    return shape_map.ntok_valid ? llama_topology_key_excl::token_lane()
                                : llama_topology_key_excl::none();
}

// The same exclusion, extended with every axis the shape map's coefficient
// table patches, named by the tensor's own name.
//
// The one-argument form above names the four token-lane roots. That is not
// enough for a graph the binder patches: the patched set also contains DERIVED
// tensors (the GET_ROWS result whose ne[1] inherits the copy index's length,
// the matmul whose ne[0] inherits the KV width), and a key that folded those
// dimensions would move whenever the runtime parameter did. The map knows them
// by record index; the key needs them by name, because it is compared across
// two graph OBJECTS and only the name survives a rebuild.
//
// `record_t` is the list the map's `tensor` indices refer to.
inline llama_topology_key_excl union_topology_key_excl(
        const llama_union_shape_map & shape_map,
        const std::vector<ggml_tensor *> & record_t) {
    llama_topology_key_excl e = union_topology_key_excl(shape_map);
    auto take = [&](const std::vector<llama_union_shape_coeff> & list) {
        for (const llama_union_shape_coeff & c : list) {
            if (c.tensor >= record_t.size()) {
                continue;
            }
            const ggml_tensor * t = record_t[c.tensor];
            if (t == nullptr) {
                continue;
            }
            for (int d = 0; d < 4; ++d) {
                if (c.ne[d] != 0 || c.nb[d] != 0) {
                    e.add_name(t->name, d);
                }
            }
        }
    };
    take(shape_map.nkv);
    take(shape_map.nout);
    take(shape_map.rs_head);
    take(shape_map.rs_len);
    take(shape_map.ntok);
    take(shape_map.nkv_ntok);
    return e;
}

// ---- rs_z runtime lane: the two decision predicates -----------------------
//
// Both are pure functions of their arguments, so a model-free test can pin
// the contract without a context, a fixture or a backend. The owning code
// calls exactly these; there is no second copy of the rule.

// graph_params(): may this build carry the selection row?
//
//   gate_ok     the umbrella binder path is on and can carry the lane
//               (gate, no disabled reuse, no pipeline, no relay, no hybrid
//               iSWA, no attention-view readers, no hybrid_idx context)
//   has_recr    the live memory context has a recurrent sub-cache
//   n_rs_seq    the recurrent cache's rollback snapshot count
//   rs_z        the live zero-state selection (-1: no free cell)
//
// n_rs_seq > 0 widens the state tensors to mem_size*(1+n_rs_seq) rows and
// makes s_copy() read row idx*mem_size + src0, so a row index is no longer a
// cell index. The stock lane bakes rs_z into a view offset, so it refuses that
// case by name instead of silently clearing the wrong row (the proof is in
// build_rs(), llama-graph.cpp).
//
// The relay path carries the ring instead: there the row the selection names
// is rs_idx[seq]*mem_size + rs_z, i.e. one row of the group the gathers read
// (get_rs_zero_row). That is a property of the state tensor and the cell
// capacity, so it is decided by llama_rsz_rt_check_relay_row() from the
// recorded tensor geometry, not here. See the two predicates below.
inline bool llama_rsz_rt_armable(bool gate_ok, bool has_recr, uint32_t n_rs_seq, int32_t rs_z) {
    return gate_ok && has_recr && n_rs_seq == 0 && rs_z >= 0;
}

// graph_params(): the relay form of the same arming question.
//
//   relay_ok    the tp5-relay path is live AND its scheduler lane is a single
//               split, so the graph is one topology on every rank and no
//               relay channel is bound to a graph input (relay.hip:1020-1058
//               binds only COMPUTE terminals, prepare() rejects the rest)
//   has_recr    the live memory context has a recurrent sub-cache
//   n_rs_seq    the recurrent cache's rollback snapshot count (0 = stock)
//   rs_z        the live zero-state selection (-1: no free cell)
//
// n_rs_seq == 0 keeps the stock relay graph byte for byte: the selection is
// still a view offset and the build carries no index row. n_rs_seq > 0 is the
// ring: the state tensors hold (1 + n_rs_seq) groups of mem_size rows, the
// zero-row op clears the single row the gathers read, and no shape depends on
// the row. The ring is therefore carried, not refused: n_rs_seq is a named
// input for symmetry with llama_rsz_rt_armable() and is deliberately not a
// condition here.
inline bool llama_rsz_rt_relay_armable(bool relay_ok, bool has_recr, uint32_t n_rs_seq, int32_t rs_z) {
    GGML_UNUSED(n_rs_seq);
    return relay_ok && has_recr && rs_z >= 0;
}

// graph_params(): the relay lane's own gate -- everything the `relay_ok`
// argument of llama_rsz_rt_relay_armable() above means. Split out so that the
// conjunction production evaluates is the conjunction a model-free test pins:
// before this existed, graph_params() wrote the conjunction inline and the
// arming predicate was dead code in src/, so a test could pin a rule the
// context did not use.
//
//   relay_backend    a tp5-relay backend is bound to this context
//   lane_enabled     GGML_TP5_RSZ_RUNTIME_PARAM was exactly "1"
//   reuse_enabled    graph reuse is not disabled (LLAMA_GRAPH_REUSE_DISABLE)
//   pipeline_parallel  pipeline parallelism is off: a relay channel bound to a
//                    graph input is not carried
//   single_split     the scheduler has exactly one split, so the graph is one
//                    topology on every rank
inline bool llama_rsz_rt_relay_gate(bool relay_backend, bool lane_enabled, bool reuse_enabled,
                                    bool pipeline_parallel, bool single_split) {
    return relay_backend && lane_enabled && reuse_enabled && !pipeline_parallel && single_split;
}

// process_ubatch(): the decode-time check. A live lane may write the row only
// when it names a row of the state tensor. rs_z < 0 is the _cell refusal (no
// free zero-state cell); rs_z >= mem_size is the _envelope refusal (a cell
// index outside the capacity names no row of any group); _shape is the relay
// ring's refusal for a tensor that is not a whole number of groups (see
// llama_rsz_rt_check_relay_row()). The stock lane never returns _shape.
enum class llama_rsz_rt_row_check : uint8_t {
    ok = 0,
    cell,      // rs_z < 0
    envelope,  // rs_z >= mem_size
    shape,     // the state tensor is not a whole number of mem_size groups
};

// The relay ring row rule, decided from the state tensor and the cell capacity.
//
// There is no zero VIEW on this lane: the zero-row op clears ONE row of the
// state tensor and the row is a runtime value (union_rsz_bind_index). So the
// rule is a property of the tensor, not of a view:
//
//   ok        rs_z < mem_size, and mem_size divides the state tensor's row
//             count. rs_z is a CELL index in [0, mem_size) -- never a group
//             base -- and the tensor holds (1 + n_rs_seq) groups of mem_size
//             rows, so the division proves the tensor is an exact whole number
//             of groups. The row the op clears is rs_idx[seq]*mem_size + rs_z
//             (get_rs_zero_row), which is inside the tensor by construction.
//   envelope  rs_z >= mem_size: a cell index outside the capacity cannot name
//             a row of any group (the decode-time check refuses this by name)
//   shape     the ring is not representable: no rows, no cells, or a row count
//             that is not a whole number of mem_size groups
//
// rs_z < 0 is the _cell refusal, shared with llama_rsz_rt_check_row().
inline llama_rsz_rt_row_check llama_rsz_rt_check_relay_row(
        int32_t rs_z, uint32_t mem_size, uint32_t state_rows) {
    if (rs_z < 0) {
        return llama_rsz_rt_row_check::cell;
    }
    if (mem_size == 0 || state_rows == 0 || state_rows % mem_size != 0) {
        return llama_rsz_rt_row_check::shape;
    }
    if ((uint32_t) rs_z >= mem_size) {
        return llama_rsz_rt_row_check::envelope;
    }
    return llama_rsz_rt_row_check::ok;
}

inline llama_rsz_rt_row_check llama_rsz_rt_check_row(int32_t rs_z, uint32_t zero_view_rows) {
    if (rs_z < 0) {
        return llama_rsz_rt_row_check::cell;
    }
    if ((uint32_t) rs_z >= zero_view_rows) {
        return llama_rsz_rt_row_check::envelope;
    }
    return llama_rsz_rt_row_check::ok;
}

// per-llama_context state for the union-graph runtime parameter binder
// (env: GGML_TP5_UNION_BINDER). The runtime parameter kinds are table
// bindings: n_kv, n_outputs, and the recurrent state lane (n_rs length and
// head-selected copy views; the rs_z zero row itself is a runtime index input
// consumed by an explicit-dependency CPU op, not a shape parameter). The
// state also holds the live graph's recorded skeleton/envelope, the symbolic
// shape map and the fallback flags.
struct llama_nkv_ctx_state {
    // ---- union binder (GGML_TP5_UNION_BINDER) ----
    bool                     nkv_enabled    = false;  // was llama_context::graph_union_nkv_runtime
    bool                     ntok_enabled   = false;
    // The control-lane selector (env: GGML_TP5_UNION_BINDER_EXACT, default on
    // under the umbrella gate). True binds the true used span and keeps the
    // requested capacities; false is the stock-shape A/B control: the padded
    // 256-cell n_kv width and the stock 256-cell capacity buckets.
    //
    // Provenance, because it matters: this field is the pad-bucket run's
    // selector (r24j, e5-union-pad-bucket-delete-r24jj), re-added by the r24l
    // merge. The r24hh line never carried it, so merging the pad-bucket
    // deletion onto r24hh would have left GGML_TP5_UNION_BINDER_EXACT unread
    // and the control lane unreachable. It is NOT the deleted
    // GGML_TP5_UNION_PAD_BUCKET or GGML_TP5_NKV_FORCE_PAD_FALLBACK knob: those
    // selected the bucket and the forced fallback, and neither is read
    // anywhere.
    //
    // This field is the ONLY reader of GGML_TP5_UNION_BINDER_EXACT in the
    // library: the single getenv() for that name is llama-context.cpp:527,
    // which assigns this field. Every other site reads the field, not the
    // variable.
    bool                     nkv_exact_enabled = true;
    // set when the symbolic shape analysis (or the first bind) cannot cover
    // this graph: the envelope build is disabled and the context runs stock.
    // This prevents an envelope/binder retry loop on unsupported graphs.
    bool                     env_disabled   = false;
    llama_union_binder       union_binder;            // kind -> (tensor set provider, rebind)
    // (no scratch graph result: the bind path is a pure metadata patch)
    bool                     record_valid   = false;   // was nkv_record_valid
    // where the three runtime parameters live in the live graph, and the
    // width it must execute at (see llama-union-width-table.h). Rebuilt on
    // every record build and re-armed by every applied bind, so the executed
    // width is a value read at run time, not a constant frozen into the
    // built graph.
    llama_union_width_table  width_table;
    // topology hash of the live graph taken at the record build with the
    // n_kv, n_outputs and rs_z dimensions excluded (llama-union-topology.h);
    // hashed=false when the graph could not be hashed. Re-taken after every
    // applied bind, so one unchanged hash proves that a bind at any
    // (n_kv, n_outputs, rs_z) combination kept the topology.
    uint64_t                 topology_hash   = 0;
    bool                     topology_hashed = false;
    uint32_t                 record_cap     = 0;       // n_kv allocation envelope of the live graph
    uint32_t                 record_cap_nout = 0;      // n_outputs build envelope of the live graph
    bool                     record_has_out_ids = false; // live graph has an out_ids closure (even width 0)
    uint32_t                 record_cap_nrs  = 0;      // recurrent state length envelope of the live graph
    uint32_t                 record_bound    = 0;       // n_kv width the live graph is bound at
    // n_kv width the live graph's arena currently holds. It starts at the
    // build width (record_bound: the allocation envelope on the legacy lanes,
    // the exact live width on the token lane) and moves only with a
    // successful replan, the one operation that reallocates the arena without
    // rebuilding the graph. It is the bound the width table's runtime arm
    // uses, so an armed width can never exceed the allocation that actually
    // backs the patched shapes. record_cap is the KV cache's physical
    // capacity, not this: on the token lane the arena is built narrow and
    // widened by the replan, so the envelope is not yet allocated.
    uint32_t                 record_arena_n_kv = 0;
    uint32_t                 record_bound_nout = 0;    // n_outputs the live graph is bound at
    uint32_t                 record_bound_ntok = 0;
    int32_t                  record_bound_rs_z = -2;   // rs_z the live graph is bound at (-2: none)
    uint32_t                 record_bound_rs_head = 0; // recurrent head the live graph is bound at
    uint32_t                 record_bound_n_rs = 0;    // recurrent state length the live graph is bound at
    uint32_t                 record_bound_replay_len = 0; // replay_len baked into the live topology

    // The single graph identity of the live graph (llama-topology-hash.h).
    // Computed by nkv_record_snapshot() when the record is taken, refreshed by
    // union_rebind_apply() after every successful in-place bind, and cleared
    // by union_record_invalidate(). It always describes the graph as it
    // currently is: at record time that is the build shapes, after a bind that
    // is the bound shapes. "Same graph" in union_rebind_apply() therefore
    // compares like with like.
    //
    // Unlike record_t/record_src, which compare tensor addresses, the key is
    // the same for two graphs with the same operations, shapes and edges
    // regardless of allocation, traversal order, n_kv / n_outputs / rs_z /
    // n_tokens values at equal built shapes, or process.
    //
    // hash == 0 means "no recorded graph": llama_topology_key_of() never
    // returns 0 for a real graph (an empty graph returns the FNV offset basis).
    //
    // TWO hashes, two questions, both kept on purpose:
    //   record_topology_key  the single graph cache key of
    //                        llama-topology-hash.h. It is the binder's reuse
    //                        gate (union_rebind_apply) and it is invariant to
    //                        the runtime parameters as parameters.
    //   topology_hash        the width-table hash of llama-union-topology.h,
    //                        taken over record_nodes with the n_kv, n_outputs
    //                        and rs_z dimensions excluded. It is the pillar's
    //                        public observable: llama_get_nkv_runtime_stats()
    //                        reports it and llama_get_union_diagnostics()
    //                        exports it, and tests compare it across cells.
    // Neither replaces the other: the key answers "may this graph be patched",
    // the width-table hash answers "did the patch leave the topology alone".
    llama_topology_key       record_topology_key{};

    // High-water mark of the n_outputs build envelope. The graph is built at
    // max(active n_outputs, this value), never at cparams.n_outputs_max up
    // front: the envelope only ever covers widths that were actually
    // requested, so a large n_batch does not allocate a large logits tensor
    // for a one-token decode. A request above the current mark is the one
    // case that rebuilds; the rebuild raises the mark to that request and
    // every later width up to it binds in place. Not reset by
    // union_record_invalidate(): it is an observed maximum, not a graph
    // pointer, and preserving it keeps the envelope across reserve/rebuilds.
    uint32_t                 nout_high_water = 0;
    std::vector<ggml_tensor *> record_t;               // was nkv_record_t
    // the graph's node list in graph order at the record build; the width
    // table and the topology hash are expressed over it. Kept so the hash can
    // be re-taken after every bind without walking a graph result.
    std::vector<ggml_tensor *> record_nodes;
    std::vector<int32_t>      record_src;               // was nkv_record_src
    std::vector<uint8_t>     record_fixed;             // was nkv_record_fixed
    std::vector<size_t>      record_bytes;             // was nkv_record_bytes
    std::vector<int32_t>     record_root;              // ultimate non-view source per tensor
    std::vector<size_t>      record_root_offs;         // cumulative view offset from that root
    std::vector<size_t>      record_root_bytes;        // root allocation / envelope size

    // symbolic runtime shape map (see llama_union_shape_coeff above)
    llama_union_shape_map    shape_map;

    // c_nout_grow_binds counts successful in-place binds that RAISED the
    // executed n_outputs (growth inside the current high-water envelope).
    // c_nout_grow_rebuilds counts the rebuilds that raised the envelope
    // itself (one per new high-water, after a valid record). Together they
    // separate "reused" from "expanded", and c_patch_binds counts all binds.
    size_t c_attempts = 0, c_applied = 0, c_refused_align = 0, c_refused_skeleton = 0,
           c_refused_envelope = 0, c_refused_relay = 0, c_refused_pipeline = 0,
           c_refused_rs_z = 0, c_refused_reader = 0, c_refused_backend = 0,
           c_refused_iswa = 0, c_refused_nkv_unknown = 0, c_refused_nout = 0,
           c_refused_nrs = 0,
           c_refused_topology = 0,
           c_fallbacks = 0, c_env_builds = 0, c_graph_builds = 0,
           c_shape_analyses = 0, c_shape_fail = 0, c_fused_qkv_refused = 0, c_patch_binds = 0,
           c_nout_grow_binds = 0, c_nout_grow_rebuilds = 0; // 24 counters
    // failed whole-graph reuse or envelope build
    size_t c_rebuild_fallbacks = 0;
    // token-width replan refusals (strict line) and the recurrent/hybrid subset
    size_t c_ntok_fallbacks = 0, c_ntok_recurrent_hybrid_fallbacks = 0;


    // Padded-lane fallback count: the number of times the padded 256-cell n_kv
    // width was used while the exact (runtime-length) lane was enabled but
    // could not deliver the true used span. llama_context::process_ubatch
    // folds the KV batch context's once-per-batch flag into this counter when
    // it starts the next batch context, so one step counts at most once.
    //
    // It does NOT count: the graph-build envelope (deliberately built at full
    // capacity), a context whose exact lane was never armed, the legitimate
    // pre-record binder refusals, the rope long/short factor selection
    // (cparams.n_ctx_seq_rope). Each of those is either not a width or not a
    // fallback.
    size_t c_pad_fallbacks = 0;

    // ---- rs_z runtime parameter (env: GGML_TP5_RSZ_RUNTIME_PARAM) ----
    //
    // rs_z is the index of the recurrent-state CELL that build_rs() zeroes
    // before the copy step. Stock bakes it into the graph as a view offset,
    // so a moved rs_z is a rebuild. This lane makes it a runtime parameter:
    // the graph is built once with a zero-row op whose row is an I32 input
    // (union_rs_zero_index), so the node set no longer depends on rs_z, and
    // the per-decode row is written at execute time by union_rsz_bind_index().
    // The row written is the row the gathers read, rs_idx[seq]*mem_size + rs_z
    // (get_rs_zero_row), not the cell index.
    //
    // The lane engages when the build carries that index tensor
    // (llm_graph_params::rs_row_armed): on the umbrella binder gate, and
    // independently on the tp5-relay lane. Every other case is refused by
    // name below and falls back to the stock rebuild.
    bool                     rsz_rt_enabled = false; // gate read as exactly "1"
    bool                     rsz_rt_active  = false; // live graph carries the index tensor
    uint32_t                 rsz_rt_zero_view_rows = 0; // cell capacity (mem->size)
    // The state tensor's row count, recorded for the relay ring form.
    // llama_rsz_rt_check_relay_row() proves from this plus
    // rsz_rt_zero_view_rows (the cell capacity, mem->size) that the tensor is
    // a whole number of mem_size groups and that rs_z is a cell inside the
    // capacity. Unused by the stock lane.
    uint32_t                 rsz_rt_state_rows = 0;
    int32_t                  rsz_rt_bound_rs_z = -2;    // rs_z the live graph is bound at
    // Named fallbacks. INVARIANT, enforced structurally: every refusal of the
    // runtime path goes through rsz_rt_refuse(), which increments exactly one
    // named cause AND the lane total c_rsz_rt_fallbacks. Nothing else may
    // touch these counters, so "one refusal, one counter plus the lane total"
    // holds by construction, and a refusal can never be silent.
    //
    //   _gate      the lane was requested but the umbrella path is off or
    //              cannot carry it (graph reuse disabled, pipeline, relay,
    //              hybrid iSWA, attention-view readers, hybrid_idx)
    //   _arch      the graph has no recurrent-state input at all
    //   _no_rs_z   the build carried no zero-row index
    //   _cell      rs_z < 0: no free zero-state cell to name
    //   _envelope  rs_z outside the cell capacity (decode time)
    //   _shape     the selection index tensor is absent or not I32 [1]
    //   _n_rs_seq  n_rs_seq > 0 on the non-relay lane: the state rows are
    //              widened to mem_size*(1+n_rs_seq) and a row index is no
    //              longer a cell index (see build_rs() for the proof). The
    //              relay lane carries the ring instead and does not refuse
    //              here; it refuses a state tensor that is not a whole number
    //              of mem_size groups through _ring (below)
    //   _ring      the state tensor is not a whole number of mem_size groups
    //              (decode time)
    //   _mixed     the ubatch's sequences disagree on the zero row: rs_idx is
    //              per sequence, and one graph carries one row (decode time).
    //              Refused before set_inputs(), so no compute runs with a row
    //              that serves only some of the cells.
    mutable size_t c_rsz_rt_fallbacks = 0,  // total refusals of the runtime path
           c_rsz_rt_refused_gate = 0,
           c_rsz_rt_refused_arch = 0,
           c_rsz_rt_refused_no_rs_z = 0,
           c_rsz_rt_refused_cell = 0,
           c_rsz_rt_refused_envelope = 0,
           c_rsz_rt_refused_shape = 0,
           c_rsz_rt_refused_n_rs_seq = 0,
           c_rsz_rt_refused_ring = 0,
           c_rsz_rt_refused_mixed_depth = 0;

    // one refusal, one counter: the named cause plus the lane total. This is
    // the ONLY writer of the counters above. const because the const peek in
    // const graph_params() refuses through it; the counters are mutable for
    // that one reason, and the invariant above still holds.
    void rsz_rt_refuse(size_t & cause) const {
        ++cause;
        ++c_rsz_rt_fallbacks;
    }

    // one-shot: set when the n_outputs binding could not run/failed on an
    // envelope graph, so the next graph_params() call builds at the ACTIVE
    // count instead of the envelope. Consumed (cleared) by graph_params().
    // mutable: consumed through the const peek inside const graph_params()
    mutable bool nout_suppress_envelope = false;

    // out-of-line destructor (llama-nout-state.cpp): keeps the state's
    // destruction in the TU that owns the definition, while llama-context.h
    // holds it behind a unique_ptr to the incomplete type.
    ~llama_nkv_ctx_state();
};

// True when every runtime shape parameter the binder can patch is already at
// the value the live graph is bound at: the single "nothing to do" test.
//
// The rule is purely "are all five at their recorded values". It says nothing
// about which parameters are bindable: that is a property of the lane, and the
// lane states it through union_bound_ntok() below.
//
// In a header so the test TU calls the SAME predicate union_rebind_apply calls.
// A test that re-implemented it would stay green through a regression in the
// real one.
inline bool union_bind_is_noop(const llama_nkv_ctx_state & st,
                               uint32_t n_kv, uint32_t n_out, uint32_t n_tokens,
                               int32_t rs_z, uint32_t rs_head) {
    return n_kv == st.record_bound &&
           n_out == st.record_bound_nout &&
           n_tokens == st.record_bound_ntok &&
           rs_z == st.record_bound_rs_z &&
           rs_head == st.record_bound_rs_head;
}

// The token width the LANE is bound at, which is what union_bind_is_noop() must
// compare on that lane.
//
// On the token lane the graph's token inputs are patched by the bind, so the
// live width is the one to compare: a request at a new width is "something to
// do". On every other lane those inputs keep their BUILD shape, so a request at
// a different token width is not a bind at all - it is a shape the graph cannot
// execute - and the recorded width is the right answer. Passing the live width
// there would make the no-op rule conclude "something to do" and let the call
// through to a patch that the `dT != 0 && !ntok_valid` guard refuses one branch
// later; the two would then disagree about the lane.
inline uint32_t union_bound_ntok(const llama_nkv_ctx_state & st, uint32_t live_n_tokens) {
    return st.shape_map.ntok_valid ? live_n_tokens : st.record_bound_ntok;
}

// The token delta's own guard inside union_rebind_apply: a request whose token
// width differs from the graph's build width is patchable only on the token
// lane, where the symbolic map models the token dimension.
//
// This is the other half of union_bound_ntok(). Off the token lane the no-op
// rule compares the recorded width with itself, so it reports "nothing to do"
// for a token-width-only request; that is a DEFERRAL, not an acceptance, and
// this guard is what turns the deferral into a refusal. Together they mean a
// non-token-lane token-width change can never be mistaken for an applied bind:
// the no-op rule declines it, and if anything else moved in the same request
// so that the rule does not decline, this guard refuses the patch.
//
// In a header for the same reason as the other two: the test calls the
// function the binder calls.
inline bool union_token_delta_refused(const llama_nkv_ctx_state & st, int64_t dT) {
    return dT != 0 && !st.shape_map.ntok_valid;
}

// per-context state slot owned by llama_context (llama-context.h declares the
// member as a unique_ptr to this forward-declared type so the state struct
// definition stays out of that header; the member itself is a new
// llama_context data member - its internal layout changes by design).
//
// Lifetime: the constructor creates the slot only when the binder gate is on.
// ~llama_context destroys it; nothing is shared between contexts or threads.
llama_nkv_ctx_state & llama_nkv_ctx_state_get(llama_context * ctx);
