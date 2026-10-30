// Model only: real planner, mocked ggml metadata and arithmetic, not a service.
#include "llama-nout-bind.h"
#include <cstdint>
#include <cstdio>
#include <cstring>

static ggml_tensor tensor(ggml_type type, int64_t cols, int64_t rows, ggml_op op) {
    ggml_tensor t = {};
    t.type = type; t.op = op;
    t.ne[0] = cols; t.ne[1] = rows; t.ne[2] = t.ne[3] = 1;
    t.nb[0] = 4; t.nb[1] = 4 * (size_t) cols;
    t.nb[2] = t.nb[1] * (size_t) rows; t.nb[3] = t.nb[2];
    return t;
}

int main() {
    constexpr int env = 8, cols = 4, rounds = 1200;
    ggml_tensor source = tensor(GGML_TYPE_F32, cols, env, GGML_OP_NONE);
    ggml_tensor ids = tensor(GGML_TYPE_I32, env, 1, GGML_OP_NONE);
    ggml_tensor gathered = tensor(GGML_TYPE_F32, cols, env, GGML_OP_GET_ROWS);
    ggml_tensor bias = tensor(GGML_TYPE_F32, cols, 1, GGML_OP_NONE);
    ggml_tensor out = tensor(GGML_TYPE_F32, cols, env, GGML_OP_ADD);
    gathered.src[0] = &source; gathered.src[1] = &ids;
    out.src[0] = &gathered; out.src[1] = &bias;
    ggml_tensor * nodes[] = { &gathered, &out };
    ggml_cgraph graph = { 2, nodes };
    const ggml_tensor env_ids = ids, env_gathered = gathered, env_out = out;
    float input[env][cols], weight[cols];
    for (int r = 0; r < env; ++r) for (int c = 0; c < cols; ++c) input[r][c] = (float)(r * cols + c) / 8.0f;
    for (int c = 0; c < cols; ++c) weight[c] = (float)c / 4.0f;
    const unsigned cycle[] = {0, 1, 3, 8, 1, 0, 7, 2};
    unsigned binds = 0, reused = 0, failures = 0;
    for (int round = 0; round < rounds; ++round) {
        const unsigned active = cycle[round % (sizeof(cycle) / sizeof(cycle[0]))];
        ids = env_ids; gathered = env_gathered; out = env_out;
        llama_nout_plan plan = {};
        if (!llama_nout_plan_bind(&graph, &ids, env, active, &plan)) { ++failures; continue; }
        llama_nout_plan_apply(&plan);
        ++binds; if (round > 0) ++reused;
        if (ids.ne[0] != active || gathered.ne[1] != active || out.ne[1] != active) ++failures;
        float actual[env][cols] = {}, intermediate[env][cols] = {}, reference[env][cols] = {};
        for (int64_t r = 0; r < gathered.ne[1]; ++r) {
            const unsigned idx = ((unsigned)r + (unsigned)round) % env;
            for (int c = 0; c < cols; ++c) intermediate[r][c] = input[idx][c];
        }
        for (int64_t r = 0; r < out.ne[1]; ++r) {
            for (int c = 0; c < cols; ++c) actual[r][c] = intermediate[r][c] + weight[c];
        }
        for (unsigned r = 0; r < active; ++r) {
            const unsigned idx = (r + (unsigned)round) % env;
            for (int c = 0; c < cols; ++c) reference[r][c] = input[idx][c] + weight[c];
        }
        if (std::memcmp(actual, reference, active * cols * sizeof(float)) != 0) ++failures;
        llama_nout_plan_free(&plan);
    }
    std::printf("rounds=%d binds=%u mock_same_graph_reuses=%u failures=%u\n", rounds, binds, reused, failures);
    return failures == 0 && binds == rounds && reused == rounds - 1 ? 0 : 1;
}
