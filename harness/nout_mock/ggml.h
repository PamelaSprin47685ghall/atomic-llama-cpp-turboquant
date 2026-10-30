#pragma once
#include <cstddef>
#include <cstdint>
#define GGML_MAX_DIMS 4
#define GGML_MAX_SRC 10
enum ggml_type { GGML_TYPE_F32, GGML_TYPE_I32 };
enum ggml_op { GGML_OP_NONE, GGML_OP_GET_ROWS, GGML_OP_ADD, GGML_OP_MUL, GGML_OP_RMS_NORM, GGML_OP_NORM, GGML_OP_SCALE, GGML_OP_CONT, GGML_OP_MUL_MAT, GGML_OP_GLU };
struct ggml_tensor {
    ggml_type type;
    ggml_op op;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    ggml_tensor * src[GGML_MAX_SRC];
};
struct ggml_cgraph { int count; ggml_tensor ** nodes; };
inline int ggml_graph_n_nodes(const ggml_cgraph * g) { return g->count; }
inline ggml_tensor * ggml_graph_node(ggml_cgraph * g, int i) { return g->nodes[i]; }
inline size_t ggml_blck_size(ggml_type) { return 1; }
inline bool ggml_is_contiguous(const ggml_tensor * t) {
    return t->nb[0] == 4 && t->nb[1] == 4 * (size_t)t->ne[0] &&
           t->nb[2] == t->nb[1] * (size_t)t->ne[1] &&
           t->nb[3] == t->nb[2] * (size_t)t->ne[2];
}
inline bool ggml_are_same_shape(const ggml_tensor * a, const ggml_tensor * b) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) if (a->ne[i] != b->ne[i]) return false;
    return true;
}
