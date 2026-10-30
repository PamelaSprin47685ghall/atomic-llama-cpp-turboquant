// Same-graph grow-back cycle for the GGML_TP5_NOUT_REUSE lane.
// One envelope graph (out_ids built at T), four binds in sequence
// 3 -> 7 -> 0 -> 16 (grow, grow, zero, grow), computing after each bind and
// comparing bit-exact against fresh exact-count builds.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama-nout-bind.h"
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    const int E = 96, V = 128, T = 32;
    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) { return 2; }
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    struct ggml_context * ctx = ggml_init({ 16*1024*1024, NULL, false });
    struct ggml_tensor * embs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, T);
    struct ggml_tensor * oids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    struct ggml_tensor * w    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, V);
    for (int i = 0; i < E*T; ++i) ((float*)embs->data)[i] = dist(rng);
    for (int i = 0; i < V*E; ++i) ((float*)w->data)[i]    = dist(rng);

    struct ggml_tensor * sel = ggml_get_rows(ctx, embs, oids);
    struct ggml_tensor * lg  = ggml_mul_mat(ctx, w, sel);
    struct ggml_cgraph * gf  = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(gf, lg);

    const int cycle[] = { 3, 7, 0, 16 };
    int failures = 0;
    struct llama_nout_plan prev = {}; // last successful plan (envelope restore)
    for (int step = 0; step < 4; ++step) {
        const int R = cycle[step];
        std::vector<int32_t> ids(T, 0);
        for (int i = 0; i < R; ++i) ids[i] = i;
        memcpy(oids->data, ids.data(), T * sizeof(int32_t));
        // restore the build-time envelope before re-planning (the contract
        // bind_outputs implements; the planner itself plans from the current
        // shapes and rejects a narrowed graph)
        for (size_t i = 0; i < prev.n_entries; ++i) {
            memcpy(prev.entries[i].t->ne, prev.entries[i].env_ne, sizeof(prev.entries[i].t->ne));
            memcpy(prev.entries[i].t->nb, prev.entries[i].env_nb, sizeof(prev.entries[i].t->nb));
        }
        oids->ne[0] = T;

        struct llama_nout_plan plan;
        if (!llama_nout_plan_bind(gf, oids, (uint32_t) T, (uint32_t) R, &plan)) {
            fprintf(stderr, "step %d (R=%d): bind failed\n", step, R);
            failures++;
            continue;
        }
        llama_nout_plan_apply(&plan);
        llama_nout_plan_free(&prev);
        prev = plan;
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "step %d (R=%d): narrowed graph compute failed\n", step, R);
            failures++;
            continue;
        }
        if (ggml_nrows(lg) != R) {
            fprintf(stderr, "step %d: rows %lld != %d\n", step, (long long) ggml_nrows(lg), R);
            failures++;
        }

        struct ggml_context * ctx2 = ggml_init({ 16*1024*1024, NULL, false });
        struct ggml_tensor * embs2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, E, T);
        struct ggml_tensor * oids2 = ggml_new_tensor_1d(ctx2, GGML_TYPE_I32, R);
        struct ggml_tensor * w2    = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, E, V);
        memcpy(embs2->data, embs->data, E*T*sizeof(float));

        memcpy(w2->data,    w->data,    V*E*sizeof(float));
        memcpy(oids2->data, ids.data(), R * sizeof(int32_t));
        struct ggml_tensor * lg2  = ggml_mul_mat(ctx2, w2, ggml_get_rows(ctx2, embs2, oids2));
        struct ggml_cgraph * gf2  = ggml_new_graph_custom(ctx2, GGML_DEFAULT_GRAPH_SIZE, false);
        ggml_build_forward_expand(gf2, lg2);
        if (ggml_backend_graph_compute(be, gf2) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "step %d (R=%d): reference graph compute failed\n", step, R);
            failures++;
            ggml_free(ctx2);
            continue;
        }
        fprintf(stderr, "[dbg] step %d R=%d: narrowed sel rows=%lld lg=%p lg2=%p oid0=%d ref0=%d\n", step, R, (long long) ggml_nrows(sel), (void*)lg->data, (void*)lg2->data, ((int32_t*)oids->data)[0], ((int32_t*)oids2->data)[0]);
        if (R > 0 && memcmp(lg->data, lg2->data, R * V * sizeof(float)) != 0) {
            fprintf(stderr, "step %d (R=%d): logits MISMATCH vs exact-count build\n", step, R);
            const float * a = (const float *) lg->data;
            const float * b = (const float *) lg2->data;
            for (int i = 0; i < R * V; ++i) {
                if (a[i] != b[i]) { fprintf(stderr, "  first diff at i=%d (row %d col %d): %g vs %g\n", i, i / V, i % V, a[i], b[i]); break; }
            }
            failures++;
        }
        ggml_free(ctx2);
    }
    ggml_free(ctx);
    ggml_backend_free(be);
    if (failures) { fprintf(stderr, "nout_growback: %d FAILURES\n", failures); return 1; }
    printf("nout_growback: PASS (bind 3->7->0->16 on one graph, bit-exact vs exact-count builds)\n");
    return 0;
}
