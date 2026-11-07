#pragma once

//
// llama-union-binder.h
//
// Table-driven union-graph runtime parameter binder.
//
// The union-graph patches let a live graph be reused across a runtime
// parameter change (n_kv width, output count, recurrent state) by binding the
// changed tensor layouts in place. Each such parameter is one
// `llama_union_param_kind` and one `llama_union_binding` entry in a fixed-size
// table: the binding names the provider that yields the affected tensors
// (`tensor_set`) and the operation that validates + patches them (`rebind`).
// `llama_union_binder::rebind_any()` is the single dispatch point.
//
// The three parameters are SIMULTANEOUS (pillar 1). A registered kind is a
// gate, not a lane: the first kind whose preconditions hold runs
// union_rebind_apply, and that one call evaluates the symbolic shape map at
// all three targets at once, so one bind applies every delta of that step. A
// kind whose own target did not move therefore cannot mask a delta in another
// kind, and a step that moves all three never needs a second bind.
//
// Fixed-topology contract (exact width by default, or the padded control):
//   - n_kv and recurrent graphs use their existing capacity bounds;
//   - the opt-in token graph starts at the active token, output and KV widths;
//     each shape change replans the live graph on one host backend. A new
//     arena is allocated before the old one is released, without rebuilding
//     topology, so allocation failure can restore the old graph;
//   - the per-tensor shape coefficients are derived symbolically at record
//     time from explicit construction-site formulas and exact ggml op shape
//     rules (see llama-nout-state.h / llama-context.cpp). There is no
//     finite-sample fitting: an op whose shape rule is not implemented makes
//     the analysis fail closed, and the context then runs stock.
//
// n_kv as a runtime parameter (pillar 1):
//   The n_kv lane does not merely patch shapes; it also supplies the width as
//   a VALUE. A binder build adds one I32 [1] graph input named
//   "union_nkv_runtime" (llm_graph_input_nkv_runtime, llama-graph.cpp). Its
//   value is rewritten before every compute from llama_kv_cache_context::
//   get_n_kv_runtime() -- the true used span, never the build envelope.
//   That value is CONSUMED by the host: the same input reads it back and fills
//   the registered KQ mask at that width
//   (llama_kv_cache::set_input_kq_mask(..., n_kv)), so the mask's fill width is
//   the runtime value and not the mask tensor's shape. Only the plain KV
//   attention builder (build_attn_inp_kv) creates the input and registers its
//   mask; the other attention builders register nothing. The K/V views are not
//   in this list: their width is the binder's in-place patch of the view
//   tensors, a separate mechanism.
//   The K/V WRITE path is the mirror image of the read views. The KV store
//   is one `ggml_set_rows(cache, cur, idxs)` per layer: the result is a view
//   of the cache storage, `src[0]` is the value tensor, `src[1]` the row
//   indices and `src[2]` the cache. The binder binds the value tensor's token
//   axis and the index input; it does NOT narrow the destination, whose row
//   dimension is the physical capacity. The symbolic shape pass refuses a
//   runtime coefficient on any destination ne/nb (the kernel's `i1 < ne1`
//   check would reject a legal cell, and `ne2 == ne02` / `ne3 == ne03` would
//   break), on the value tensor's plane axes (`ne[2]`/`ne[3]`, which would
//   make the kernel add one full cache span per token plane), and on the
//   index tensor's plane extents. It also re-checks the constructor and
//   kernel relations after copying the destination shape: `ne[1] ==
//   idx.ne[0]`, `idx.ne[3] == 1`, `ne[2] % idx.ne[1] == 0`, `ne[3] % idx.ne[2]
//   == 0`, `ne[0] == src0.ne[0]`, `ne[2] == src0.ne[2]`, `ne[3] == src0.ne[3]`,
//   and the contiguous-row strides of the destination. In the transposed-V
//   layout (flash attention off) `cpy_v` flattens `(n_embd_head, n_head,
//   n_tokens)` to `(1, n_embd_gqa*n_tokens)`; the token axis is axis 1 with
//   factor `n_embd_gqa`, and `cpy_v` tags it with `LLM_TOKEN_AXIS_MARKER | 1`
//   when the token lane is armed (`strict_ntok`). Without the tag the axis
//   search can pick the degenerate trailing axis 2 and the CPU kernel writes
//   outside the cache. Every other lane is byte-identical to before: the tag
//   is not applied and the shape pass sees the same graph.
//   `llama_get_nkv_runtime_stats()` reports the three parameters next to the
//   topology hash (llama-union-topology.h): the hash covers the node set, the
//   edges, the op types, every op parameter and every non-runtime shape
//   dimension, with the n_kv, n_outputs and rs_z dimensions excluded, so ONE
//   hash proves that a bind at any (n_kv, n_outputs, rs_z) combination kept
//   the topology. It is taken at the record build and RE-TAKEN after every
//   applied bind; a bind whose re-taken hash cannot be computed is refused and
//   the caller rebuilds. llama-union-width-table.h records which dimension of
//   which node carries each parameter.
//   What this does NOT prove: no ggml kernel reads the runtime width. The
//   consumer is the host-side mask fill. Flash attention on GPU/HIP still
//   needs a device-side width parameter and is outside this contract.
//
// Gate: when GGML_TP5_UNION_BINDER is unset, the default is limited to the
// currently supported LLAMA graph without ALiBi, MoE, relative attention or
// backend samplers, with one KV stream, graph reuse, CPU model placement,
// no device or tensor override, no registered ACCEL backend, and no relay.
// Explicit "1" also permits the recurrent and hybrid architectures listed
// in union_arch_supported(), subject to graph and backend guards. These
// preconditions do not prove safety: live split and envelope guards still
// apply. Every other explicit value, including "0", disables the binder.
// GPU and other backends stay opt-in.
// The legacy LLAMA_GRAPH_UNION_OUTPUTS_RUNTIME flag only logs a warning.
// Every explicit value follows llama_union_gate_exact_one(): exactly "1".
//
// Registered kinds:
//   - n_kv: the KQ-mask / KV-view / attention tensor set of a built graph;
//     provider `llama_context::union_tensors_n_kv()`, rebind
//     `llama_context::union_rebind_n_kv()`. The lane binds the true used span
//     from get_n_kv(). Execution shapes are patched to the bound width; the
//     allocation envelope is only an upper bound.
//     The lane does NOT mark any node and does NOT emulate any reduction order.
//     It does not need to: the attention products carry ATTN_SCORES /
//     ATTN_MASK / ATTN_SOFT_MAX from the graph build, those route to the
//     ordered CPU kernels, and the ordered kernels accumulate sequentially over
//     the surviving terms. A narrowed graph therefore runs the same kernels
//     with the same tree as a graph built at the same width. The earlier
//     REDUCE_PAD_TAIL / MASKED_TAIL_GEMM fixup was removed because its tree was
//     a function of the build envelope while stock's follows its own ne00, so
//     it could not agree with stock at any width (see ANALYSIS-EQUALITY.md).
//     Masked entries are skipped without reading their scores or values; an
//     unmasked zero probability does not exclude a nonfinite value. F16 values
//     use F16-rounded probabilities. Equality depends on identical visible
//     operands and floating-point behavior. F16/quantized V parity and flash
//     attention are outside this contract.
//     The patch requires exactly one scheduler split and validates split input
//     copies against their allocation envelopes before changing graph or copies.
//     Scheduler-rewritten edges stay intact; relay and pipeline are refused.
//     GGML_TP5_FA_MASK_KVLEN=1 must not be used on GPU/HIP before the
//     documented FA kernel issue is fixed.
//     Eligible CPU non-flash attention uses explicitly masked, ordered softmax
//     and F16/F32 value reductions on both binder ON and OFF arms; eligible
//     F16/F32 K by F32 Q products also use ordered masked CPU dots, avoiding
//     width-dependent KQ dispatch. Unmarked/quantized paths, flash attention,
//     other backends and their numerical or performance behavior are not
//     covered by this guarantee.
//   - n_outputs: the output-gather closure (out_ids and every tensor whose
//     row dimension derives from it); provider
//     `llama_context::union_tensors_n_outputs()`, rebind
//     `llama_context::union_rebind_n_outputs()`. The graph is built at
//     `n_outputs_build` = max(active, high-water) and bound to the active
//     count, including 0: a zero-output step is a shrink to zero rows, not a
//     refusal, and the closure executes over no rows. A request above the
//     high-water is the one rebuild that raises it. The input's
//     `bound_outputs` is the reuse key.
//   - rs_z: the zero row is a host runtime index input consumed by an
//     explicit-dependency in-place CPU op; -1 means no zero row. The recurrent
//     length n_rs and head-selected copy views are shape bindings. The state
//     envelope is the physical capacity, built at n_rs=size, head=0, rs_z=0.
//     Provider `llama_context::union_tensors_rs_z()`, rebind
//     `llama_context::union_rebind_rs_z()`; input runtime metadata is refreshed,
//     including the cached head/rs_z/replay_len values the next can_reuse()
//     reads.
//   - n_tokens: experimental total-token shape binding for the dense,
//     single-stream simple-batch CPU lane. The graph is constructed at the
//     active KV/token/output widths (never at a reserved width) and replanned
//     transactionally on a width change; it uses the same validated tensor
//     record as n_kv and can dispatch even when n_kv is unchanged. The
//     sequence-token width is fixed at one; other layouts, and unsupported or
//     recurrent/hybrid graphs, rebuild stock and count the fallback.
//
// rs_z runtime parameter (env: GGML_TP5_RSZ_RUNTIME_PARAM): the stronger
// form of the rs_z kind. Instead of moving a view offset, the graph is built
// once with an in-place zero-row op (union_rsz_zero_row) whose row is an
// I32 [1] input (`union_rs_zero_index`). union_rsz_bind_index() writes that
// row on every decode, so the node set never depends on rs_z. The value it
// writes is the row the gathers read, rs_idx[seq]*mem_size + rs_z
// (llama_memory_recurrent_context::get_rs_zero_row), not the cell index: the
// graph reads exactly one row per cell, and s_copy() names that row. A moved
// rs_z is therefore an input write: no tensor shape changes, no bind, no
// rebuild.
//
// Gate and interaction with the other knobs:
//   - GGML_TP5_RSZ_RUNTIME_PARAM must be exactly "1" (the contract of
//     llama_union_gate_exact_one() above; unset / "" / "0" / "true" / "01" /
//     " 1" / "2" are all OFF). It is read only when GGML_TP5_UNION_BINDER
//     opens the binder; with the umbrella gate off the value is ignored and
//     a warning names the reason.
//   - GGML_TP5_UNION_BINDER: unset keeps the qualified CPU default, "1"
//     opts in, every other explicit value is stock. The lane needs the
//     binder's per-context state and its record build, so it never runs
//     without it.
//
// The lane is armed per build when the build carries the row index
// (llm_graph_params::rs_row_armed): on the binder lane, and independently on
// the tp5-relay lane, which never passes the binder gate but must carry the
// row too. A context that cannot carry the row builds the stock graph and
// increments exactly one NAMED counter plus the lane total:
// rsz_rt_refused_gate (umbrella path off or cannot carry the lane), _arch
// (no recurrent-state input), _no_rs_z (the build carried no row index),
// _cell (rs_z < 0), _envelope (rs_z outside the cell capacity at decode
// time), _shape (index tensor absent or not I32 [1]), _n_rs_seq (the stock
// lane refuses the rollback ring; the relay lane carries it). The counters
// are printed in the NKV_BINDER_STATS line and exposed through
// llama_union_diagnostics (rsz_rt_*), so the lane is observable and
// controllable without reading logs. A refusal is never silent. With the gate
// unset or not "1" the build is byte-identical to stock.
//
//
// The binder fails closed for multi-split graphs, unsafe split copies,
// relay/pipeline, attention-view readers, hybrid iSWA, multi-stream KV,
// unknown widths, unsupported topology, changed replay length, or tensors
// outside their envelopes. Refusal falls back to a full rebuild.
//
// This internal header adds a header-only table class plus POD structs;
// it is included by llama-context.{h,cpp} and llama-nout-state.h and is not
// part of the public ABI.
//

#include <cstddef>
#include <cstring>
#include <cstdint>
#include <vector>

// The one contract for every GGML_TP5_* on/off knob in this file: the value
// must be exactly "1". Unset, "", "0", "true", "01", " 1", "2" and every
// other string are OFF. There is no atoi(), no prefix match and no
// whitespace trimming: a typo must never enable a lane.
inline bool llama_union_gate_exact_one(const char * value) {
    return value != nullptr && std::strcmp(value, "1") == 0;
}

// Umbrella gate (GGML_TP5_UNION_BINDER). Unset keeps the qualified CPU
// default; an explicit value uses the exact-one contract above, so "0" and
// every other non-"1" string select the stock path.
inline bool llama_union_binder_enabled(const char * value, bool cpu_default) {
    return value == nullptr ? cpu_default : llama_union_gate_exact_one(value);
}
struct ggml_tensor;
struct llama_context;
class  llm_graph_result;
struct llm_graph_params;

// runtime parameter lanes of the union-graph binder; `count` is the table size
enum class llama_union_param_kind : uint8_t {
    n_kv      = 0,
    n_outputs = 1,
    rs_z      = 2,
    n_tokens  = 3,
    count     = 4,
};

// the tensors a single rebind operation may touch, as recorded by a
// per-kind provider. n_kv/rs_z fill all vectors (DFS post-order tensor
// list, its src/view_src adjacency, the fixed-storage flags, the byte
// envelopes, the per-tensor root index, the cumulative root offset and the
// per-tensor root allocation size); a future kind may use only `tensors` and
// leave the auxiliary records null.
struct llama_union_tensor_set {
    const std::vector<ggml_tensor *> * tensors    = nullptr;
    const std::vector<int32_t>       * src        = nullptr;
    const std::vector<uint8_t>       * fixed      = nullptr;
    const std::vector<size_t>        * bytes      = nullptr;
    const std::vector<int32_t>       * root       = nullptr;
    const std::vector<size_t>        * root_offs  = nullptr;
    const std::vector<size_t>        * root_bytes = nullptr;
};

// What a kind's rebind() did, which is what the dispatch needs to decide
// whether the next kind may run. A bool cannot carry it: "declined" means the
// kind's own preconditions did not hold (its parameter did not move, or its
// lane is not present), so the next kind must be tried; "refused" means the
// kind RAN -- it passed its preconditions and called union_rebind_apply, which
// evaluated every runtime target at once -- and the bind failed, so the next
// kind would repeat the same patch and the same refusal.
enum class llama_union_rebind_result : uint8_t {
    declined = 0, // preconditions did not hold; nothing was evaluated
    refused  = 1, // ran, and union_rebind_apply refused the bind
    applied  = 2, // ran, and the graph was patched in place
};

// one table entry: kind -> (tensor set provider, rebind operation)
struct llama_union_binding {
    llama_union_param_kind kind = llama_union_param_kind::count;

    // diagnostic name of the lane ("n_kv", ...)
    const char * name = nullptr;

    // yields the tensors recorded for this kind on the given context
    llama_union_tensor_set (llama_context::*tensor_set)() = nullptr;

    // validates + patches the live graph in place
    llama_union_rebind_result (llama_context::*rebind)(llm_graph_result * res,
                                                       const llm_graph_params & params,
                                                       const llama_union_tensor_set & set) = nullptr;
};

// fixed-size table of registered bindings (no allocation, no ordered map:
// registration happens once in the llama_context constructor)
class llama_union_binder {
public:
    static constexpr size_t n_kinds = (size_t) llama_union_param_kind::count;

    // register (or replace) the binding for its kind; false for an invalid or
    // out-of-range kind
    bool add(const llama_union_binding & binding) {
        const size_t idx = (size_t) binding.kind;
        if (idx >= n_kinds) {
            return false;
        }
        slots_[idx]    = binding;
        registered_[idx] = true;
        return true;
    }

    bool has(llama_union_param_kind kind) const {
        const size_t idx = (size_t) kind;
        return idx < n_kinds && registered_[idx];
    }

    // registered binding for the kind, or nullptr
    const llama_union_binding * get(llama_union_param_kind kind) const {
        return has(kind) ? &slots_[(size_t) kind] : nullptr;
    }

    size_t size() const {
        size_t n = 0;
        for (size_t i = 0; i < n_kinds; ++i) {
            n += registered_[i] ? 1 : 0;
        }
        return n;
    }

    // dispatch: try every registered kind in enum order, obtain its tensor set
    // from the context and run its rebind. A kind that DECLINES (its own
    // preconditions did not hold, so its parameter did not move or its lane is
    // not present) hands the turn to the next kind. The first kind that RUNS
    // wins: it calls union_rebind_apply, which patches ALL THREE runtime
    // parameters to their current targets, not only its own kind, and its
    // verdict is the dispatch's verdict -- applied returns true, refused
    // returns false without trying the remaining kinds, because they would
    // repeat the same patch and meet the same refusal. false when no kind ran
    // (callers fall back to a full rebuild).
    bool rebind_any(llama_context * ctx,
                    llm_graph_result * res,
                    const llm_graph_params & params) const;

private:
    llama_union_binding slots_[n_kinds];
    bool                registered_[n_kinds] = {};
};
