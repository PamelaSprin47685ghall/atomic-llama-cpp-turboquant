#pragma once

//
// llama-union-binder.h
//
// Table-driven union-graph runtime parameter binder.
//
// parameter change (n_kv width, output count, ...) by rebuilding the graph
// in a scratch result and, when the only difference is the parameter's
// tensor set, copying the changed fields into the live graph in place.
// Each such parameter is one `llama_union_param_kind` and one
// `llama_union_binding` entry in a fixed-size table: the binding names the
// provider that yields the affected tensors (`tensor_set`) and the operation
// that validates + patches them (`rebind`). `llama_union_binder::rebind()`
// is the single dispatch point.
//
// Gates: two independent lane gates, each exact value "1" only (see
// llama-context.cpp): GGML_TP5_UNION_BINDER (the legacy outputs lane) and
// GGML_TP5_NKV_TRUEWIDTH (the r7 true-width nkv lane). Unset or any other
// value is stock for that lane; the legacy
// LLAMA_GRAPH_UNION_OUTPUTS_RUNTIME flag is subordinate to
// GGML_TP5_UNION_BINDER.
//
// Registered kinds:
//   - n_kv: the whole KQ-mask / KV-view / attention tensor set of a built
//     graph; provider `llama_context::union_tensors_n_kv()`, rebind
//     `llama_context::union_rebind_n_kv()`.
//   - n_outputs: the output-gather closure (out_ids + every GET_ROWS dst
//     reading it) of a graph built at the exact active count (no
//     pad-to-max); provider
//     `llama_context::union_tensors_n_outputs()`, rebind
//     `llama_context::union_rebind_outputs()`. Both kinds share the one
//     pre-allocation record of the latest fresh build (one live graph).
//
// Reserved kinds (no binding is registered; the enum value + the fixed-size
// `slots_` array + `add()` are the registration points):
//   - rs_z: `llama_memory_recurrent_context::get_rs_z()` feeds
//     `llm_graph_input_rs::rs_z` and the build_rs zero-state view; there is no
//     validated tensor registry / rebind operation for it yet.
//
// This internal header adds a header-only table class plus two POD structs;
// it is included by llama-context.{h,cpp} and llama-nout-state.h and is not
// part of the public ABI.
//

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_tensor;
struct llama_context;
class  llm_graph_result;
struct llm_graph_params;

// runtime parameter lanes of the union-graph binder; `count` is the table size
enum class llama_union_param_kind : uint8_t {
    n_kv      = 0,
    n_outputs = 1,
    rs_z      = 2,
    count     = 3,
};

// the tensors a single rebind operation may touch, as recorded by a
// per-kind provider. n_kv fills all four (DFS post-order tensor list, its
// src/view_src adjacency, the fixed-storage flags and the byte envelopes); a
// future kind may use only `tensors` and leave the auxiliary records null.
struct llama_union_tensor_set {
    const std::vector<ggml_tensor *> * tensors = nullptr;
    const std::vector<int32_t>       * src     = nullptr;
    const std::vector<uint8_t>       * fixed   = nullptr;
    const std::vector<size_t>        * bytes   = nullptr;
};

// one table entry: kind -> (tensor set provider, rebind operation)
struct llama_union_binding {
    llama_union_param_kind kind = llama_union_param_kind::count;

    // diagnostic name of the lane ("n_kv", ...)
    const char * name = nullptr;

    // yields the tensors recorded for this kind on the given context
    llama_union_tensor_set (llama_context::*tensor_set)() = nullptr;

    // validates + patches the live graph in place; returns true when the
    // graph can be reused
    bool (llama_context::*rebind)(llm_graph_result * res,
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

    // dispatch: look up the kind, obtain its tensor set from the context, run
    // the rebind. false when the kind is unregistered / incomplete or the
    // rebind refuses (callers fall back to a full rebuild).
    bool rebind(llama_union_param_kind kind,
                llama_context * ctx,
                llm_graph_result * res,
                const llm_graph_params & params) const;

private:
    llama_union_binding slots_[n_kinds];
    bool                registered_[n_kinds] = {};
};
