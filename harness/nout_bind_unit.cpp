// GGML-level unit harness for the GGML_TP5_NOUT_REUSE lane (src/llama-nout-bind.cpp).
//
// Fixture mirrors the real dense decoder output closure (llama.cpp:174-176
// onward): two GET_ROWS over out_ids -> ADD -> RMS_NORM -> MUL (broadcast
// gain) -> MUL_MAT (up) -> GLU split (gate mm) -> MUL_MAT (down) -> ADD
// residual -> RMS_NORM -> MUL_MAT (lm_head). Depth ~10; the envelope graph is
// narrowed via llama_nout_plan_bind and computed on the real CPU backend, then
// compared bitwise against a fresh exact-count build (stock semantics).
// R sweeps {32, 0, 1, 3, 7, 16, 32, 5}: full envelope, zero outputs, odd,
// shrink, grow-back, repeat.
//
// exit codes: 0 all passes bit-exact, 1 failure, 2 setup error.
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
    const int E = 96;   // n_embd
    const int F = 64;   // ffn hidden
    const int V = 128;  // n_vocab
    const int T = 32;   // token envelope

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> emb(E * T), gain(E), wup(F * E), wgate(F * E), wdown(E * F), whead(V * E);
    for (auto & v : emb)   v = dist(rng);
    for (auto & v : gain)  v = dist(rng);
    for (auto & v : wup)   v = dist(rng);
    for (auto & v : wgate) v = dist(rng);
    for (auto & v : wdown) v = dist(rng);
    for (auto & v : whead) v = dist(rng);

    const int Rs[] = { 32, 0, 1, 3, 7, 16, 32, 5 };
    int failures = 0;

    ggml_backend_t be = ggml_backend_cpu_init();
    if (be == nullptr) { fprintf(stderr, "no cpu backend\n"); return 2; }

    for (int pass = 0; pass < (int)(sizeof(Rs)/sizeof(Rs[0])); ++pass) {
        const int R = Rs[pass];

        // ---- envelope graph (narrowed in place) ----
        struct ggml_context * ctx = ggml_init({ 16*1024*1024, NULL, false });
        struct ggml_tensor * embs  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, T);
        struct ggml_tensor * oids  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        struct ggml_tensor * w1    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, F); // up
        struct ggml_tensor * w2    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, F); // gate
        struct ggml_tensor * w3    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F, E); // down
        struct ggml_tensor * w4    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, V); // head
        struct ggml_tensor * gn    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, E);    // gain
        memcpy(embs->data, emb.data(),  emb.size()  * sizeof(float));
        memcpy(w1->data,   wup.data(),  wup.size()  * sizeof(float));
        memcpy(w2->data,   wgate.data(),wgate.size()* sizeof(float));
        memcpy(w3->data,   wdown.data(),wdown.size()* sizeof(float));
        memcpy(w4->data,   whead.data(),whead.size()* sizeof(float));
        memcpy(gn->data,   gain.data(), gain.size() * sizeof(float));

        std::vector<int32_t> ids(T, 0);
        for (int i = 0; i < R; ++i) ids[i] = i;
        memcpy(oids->data, ids.data(), T * sizeof(int32_t));

        struct ggml_tensor * s1 = ggml_get_rows(ctx, embs, oids);
        struct ggml_tensor * s2 = ggml_get_rows(ctx, embs, oids);
        struct ggml_tensor * a1 = ggml_add(ctx, s1, s2);            // [E,R]
        struct ggml_tensor * n1 = ggml_rms_norm(ctx, a1, 1e-5f);    // [E,R]
        struct ggml_tensor * m1 = ggml_mul(ctx, n1, gn);            // [E,R]
        struct ggml_tensor * up = ggml_mul_mat(ctx, w1, m1);        // [F,R]
        struct ggml_tensor * gt = ggml_mul_mat(ctx, w2, m1);        // [F,R]
        struct ggml_tensor * gl = ggml_swiglu_split(ctx, gt, up);   // [F,R]
        struct ggml_tensor * dn = ggml_mul_mat(ctx, w3, gl);        // [E,R]
        struct ggml_tensor * a2 = ggml_add(ctx, dn, a1);            // [E,R]
        struct ggml_tensor * n2 = ggml_rms_norm(ctx, a2, 1e-5f);    // [E,R]
        struct ggml_tensor * m2 = ggml_mul(ctx, n2, gn);            // [E,R]
        struct ggml_tensor * lg  = ggml_mul_mat(ctx, w4, m2);       // [V,R]
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
        ggml_build_forward_expand(gf, lg);

        struct llama_nout_plan plan;
        if (!llama_nout_plan_bind(gf, oids, (uint32_t) T, (uint32_t) R, &plan)) {
            fprintf(stderr, "pass %d (R=%d): bind failed\n", pass, R);
            failures++;
        }
        llama_nout_plan_apply(&plan);
        llama_nout_plan_free(&plan);

        enum ggml_status st = ggml_backend_graph_compute(be, gf);
        fprintf(stderr, "[dbg] pass %d R=%d: compute status=%d lg rows=%lld lg[0]=%g lg[V]=%g sel_ne={%lld,%lld} lg_ne={%lld,%lld}\n",
            pass, R, (int) st, (long long) ggml_nrows(lg), ((float*)lg->data)[0], ((float*)lg->data)[V],
            (long long) s1->ne[0], (long long) s1->ne[1], (long long) lg->ne[0], (long long) lg->ne[1]);

        // ---- reference graph: exact count, no bind (stock semantics) ----
        struct ggml_context * ctx2 = ggml_init({ 16*1024*1024, NULL, false });
        struct ggml_tensor * embs2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, E, T);
        struct ggml_tensor * oids2 = ggml_new_tensor_1d(ctx2, GGML_TYPE_I32, R);
        struct ggml_tensor * w12   = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, E, F);
        struct ggml_tensor * w22   = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, E, F);
        struct ggml_tensor * w32   = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, F, E);
        struct ggml_tensor * w42   = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, E, V);
        struct ggml_tensor * gn2   = ggml_new_tensor_1d(ctx2, GGML_TYPE_F32, E);
        memcpy(embs2->data, emb.data(),  emb.size()  * sizeof(float));
        memcpy(w12->data,   wup.data(),  wup.size()  * sizeof(float));
        memcpy(w22->data,   wgate.data(),wgate.size()* sizeof(float));
        memcpy(w32->data,   wdown.data(),wdown.size()* sizeof(float));
        memcpy(w42->data,   whead.data(),whead.size()* sizeof(float));
        memcpy(gn2->data,   gain.data(), gain.size() * sizeof(float));
        memcpy(oids2->data, ids.data(), R * sizeof(int32_t));

        struct ggml_tensor * t1 = ggml_get_rows(ctx2, embs2, oids2);
        struct ggml_tensor * t2 = ggml_get_rows(ctx2, embs2, oids2);
        struct ggml_tensor * b1 = ggml_add(ctx2, t1, t2);
        struct ggml_tensor * c1 = ggml_rms_norm(ctx2, b1, 1e-5f);
        struct ggml_tensor * d1 = ggml_mul(ctx2, c1, gn2);
        struct ggml_tensor * e1 = ggml_mul_mat(ctx2, w12, d1);
        struct ggml_tensor * f1 = ggml_mul_mat(ctx2, w22, d1);
        struct ggml_tensor * g1 = ggml_swiglu_split(ctx2, f1, e1);
        struct ggml_tensor * h1 = ggml_mul_mat(ctx2, w32, g1);
        struct ggml_tensor * i1 = ggml_add(ctx2, h1, b1);
        struct ggml_tensor * j1 = ggml_rms_norm(ctx2, i1, 1e-5f);
        struct ggml_tensor * k1 = ggml_mul(ctx2, j1, gn2);
        struct ggml_tensor * lg2  = ggml_mul_mat(ctx2, w42, k1);
        struct ggml_cgraph * gf2  = ggml_new_graph_custom(ctx2, GGML_DEFAULT_GRAPH_SIZE, false);
        ggml_build_forward_expand(gf2, lg2);

        ggml_backend_graph_compute(be, gf2);

        if (ggml_nrows(lg) != R) {
            fprintf(stderr, "pass %d: narrowed rows %lld != %d\n", pass, (long long) ggml_nrows(lg), R);
            failures++;
        }
        if (R > 0 && memcmp(lg->data, lg2->data, R * V * sizeof(float)) != 0) {
            fprintf(stderr, "pass %d (R=%d): logits MISMATCH vs exact-count build\n", pass, R);
            failures++;
        }
        ggml_free(ctx);
        ggml_free(ctx2);
    }
    ggml_backend_free(be);

    if (failures) { fprintf(stderr, "nout_bind_unit: %d FAILURES\n", failures); return 1; }
    printf("nout_bind_unit: PASS (all binds bit-exact vs exact-count builds)\n");
    return 0;
}
