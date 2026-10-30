// Regression for envelope=1 operand provenance in the real NOUT planner.
#include "ggml.h"
#include "llama-nout-bind.h"
#include <cstdio>

static bool check(bool glu, bool independent_first, bool expected) {
    ggml_context * ctx = ggml_init({ 1024 * 1024, nullptr, false });
    if (!ctx) return false;
    ggml_tensor * table = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * gather = ggml_get_rows(ctx, table, ids);
    ggml_tensor * independent = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    ggml_tensor * a = independent_first ? independent : gather;
    ggml_tensor * b = independent_first ? gather : independent;
    ggml_tensor * out = glu ? ggml_swiglu_split(ctx, a, b) : ggml_add(ctx, a, b);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    llama_nout_plan plan = {};
    const bool result = llama_nout_plan_bind(graph, ids, 1, 0, &plan);
    const bool unchanged = ids->ne[0] == 1 && gather->ne[1] == 1 && out->ne[1] == 1;
    llama_nout_plan_free(&plan);
    ggml_free(ctx);
    return result == expected && unchanged;
}

int main() {
    // ADD src0 dictates output rows; src1 alone may be broadcast.
    // Split GLU requires both operands to track the planned rows.
    const bool ok = check(false, true, false) && check(false, false, true) &&
                    check(true, true, false) && check(true, false, false);
    std::fprintf(stderr, "nout_operand_regression: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
