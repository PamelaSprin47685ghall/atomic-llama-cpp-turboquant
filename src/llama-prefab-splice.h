#pragma once

// ggml-only DAG splice kernel of the factor-prefab compose layers.
//
// A module ("A") is a graph emitted into its own ggml context: a list of `n_nodes` nodes in
// emission order (a topological order, because it was built with ggml_build_forward_expand),
// named cut tensors ("outputs") and formal placeholder inputs.
//
// Splice clones that DAG into a destination context/graph:
//   - a formal input is substituted by the destination tensor of the same name; the layout
//     (type, ne[0..3], nb[0..3]) must match exactly, otherwise it throws;
//   - a tensor that is not allocated in the module's own mem buffer is a shared weight (or any
//     other model-owned tensor): it is never cloned, the pointer is reused as-is;
//   - an unbound placeholder (op NONE, no view_src) throws: the DAG cannot be spliced without
//     one of its inputs;
//   - every other node is cloned into the destination context, its sources are cloned
//     recursively, and the clone is expanded into the destination graph in the module's node
//     order.
//
// This header depends only on ggml so that the kernel can be exercised outside llama.cpp
// (a standalone CPU harness covers the splice contract; it lives next to the workdir root).

#include "ggml.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>

struct llama_prefab_dag {
    ggml_context * ctx = nullptr; // context the module's tensors live in (ownership test)
    ggml_cgraph  * gf  = nullptr; // the module's graph (node order)

    int n_nodes = 0; // nodes to splice, in emission order

    std::map<std::string, ggml_tensor *> inputs;  // formal placeholders
    std::map<std::string, ggml_tensor *> outputs; // named cut tensors
};

// Clone `src` into dst_ctx/dst_gf, substituting `subs`. Returns the cloned outputs by name.
// `n_clones` (may be null) receives the number of cloned tensors; it is bumped as tensors are
// cloned, so a throw leaves it holding the clones created up to that point.
inline std::map<std::string, ggml_tensor *> llama_prefab_dag_splice(
        ggml_context * dst_ctx,
        ggml_cgraph  * dst_gf,
        const llama_prefab_dag & src,
        const std::map<std::string, ggml_tensor *> & subs,
        uint64_t * n_clones) {
    // ownership: tensor addresses inside the module's mem buffer belong to the module
    const char * buf_beg = (const char *) ggml_get_mem_buffer(src.ctx);
    const char * buf_end = buf_beg + ggml_get_mem_size(src.ctx);

    std::unordered_map<const ggml_tensor *, ggml_tensor *> map;

    for (const auto & [name, ph] : src.inputs) {
        auto it = subs.find(name);
        if (it == subs.end() || it->second == nullptr) {
            throw std::runtime_error("factor-prefab splice: unbound input '" + name + "'");
        }
        ggml_tensor * t = it->second;
        bool same = t->type == ph->type;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            same = same && t->ne[i] == ph->ne[i] && t->nb[i] == ph->nb[i];
        }
        if (!same) {
            throw std::runtime_error("factor-prefab splice: layout mismatch for input '" + name + "'");
        }
        map[ph] = t;
    }

    std::function<ggml_tensor * (ggml_tensor *)> clone = [&](ggml_tensor * t_src) -> ggml_tensor * {
        if (t_src == nullptr) {
            return nullptr;
        }
        auto it = map.find(t_src);
        if (it != map.end()) {
            return it->second;
        }
        const char * p = (const char *) t_src;
        if (p < buf_beg || p >= buf_end) {
            // weights: shared, never cloned
            map[t_src] = t_src;
            return t_src;
        }
        if (t_src->op == GGML_OP_NONE && t_src->view_src == nullptr) {
            // an unbound placeholder: the DAG registered an input the caller did not bind
            throw std::runtime_error(std::string("factor-prefab splice: leaf '") + t_src->name +
                    "' is not a formal input");
        }

        ggml_tensor * t = ggml_new_tensor(dst_ctx, t_src->type, GGML_MAX_DIMS, t_src->ne);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            t->nb[i] = t_src->nb[i];
        }
        t->op    = t_src->op;
        t->flags = t_src->flags;
        memcpy(t->op_params, t_src->op_params, sizeof(t->op_params));
        ggml_set_name(t, t_src->name);

        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            t->src[i] = clone(t_src->src[i]);
        }
        if (t_src->view_src != nullptr) {
            t->view_src  = clone(t_src->view_src);
            t->view_offs = t_src->view_offs;
            if (t->view_src->data != nullptr) {
                t->data = (char *) t->view_src->data + t->view_offs;
            }
        }

        map[t_src] = t;
        if (n_clones != nullptr) {
            (*n_clones)++;
        }
        return t;
    };

    for (int i = 0; i < src.n_nodes; ++i) {
        ggml_build_forward_expand(dst_gf, clone(ggml_graph_node(src.gf, i)));
    }

    std::map<std::string, ggml_tensor *> outs;
    for (const auto & [name, t] : src.outputs) {
        outs[name] = clone(t);
    }
    return outs;
}
