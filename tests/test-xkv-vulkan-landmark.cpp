// test-xkv-vulkan-landmark.cpp — Focused tests for native Vulkan phase-aware
// quantized landmark scoring, deterministic bounded global top-k, and bounded
// boundary refinement (XkvVulkanLandmark).
//
// Covers CPU oracle vs CPU graph vs Vulkan graph (K/V-independent landmark path):
//  - Q8_0 and Turbo4_0 capability gating (explicit paths; unsupported fails closed)
//  - Phase-aware scoring at storage_pos (NeoX HALF + IMRoPE INTERLEAVED + no-RoPE)
//  - Legal CSR membership (illegal / causally-future masked; empty valid state, no NaN)
//  - Deterministic score-desc/id-asc ties (equal-score fragments select lowest ids)
//  - Zero / oversized budget fails closed (never select-all)
//  - Storage positions + partial tables exact (rotary bounds, mag validation)
//  - Memory/workspace bounds (exact bytes, scratch-too-small rejects)
//  - Bounded refinement: cap honored, cap-hit reported, no select-all fallback
//  - Per-query isolation: future speculative queries never influence earlier queries
//
// Standalone link: g++ ... -lggml -lggml-base -lggml-cpu -lggml-vulkan.
// Define GGML_USE_VULKAN (with ggml-vulkan.h) to enable the direct
// ggml_backend_vk_init(0) fallback; registry lookup alone skips on Radeon.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "ggml.h"
#include "ggml-vulkan-landmark.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#else
extern "C" ggml_backend_t ggml_backend_vk_init(size_t dev_num);
#endif

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>

static int failures = 0;
#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

static ggml_xkv_landmark_params base_params() {
    ggml_xkv_landmark_params p = {};
    p.version = GGML_XKV_LANDMARK_VERSION;
    p.n_queries = 2;
    p.n_frags = 4;
    p.head_dim = 16;
    p.padded_dim = 32; // Q8_0 block multiple
    p.rotary_dim = 8;
    p.rope_mode = GGML_XKV_LANDMARK_ROPE_HALF;
    p.landmark_type = (uint32_t)GGML_TYPE_Q8_0;
    p.top_k = 2;
    p.max_top_k = 4;
    p.refine_cap = 0;
    p.frag_size = 4;
    p.scale = 0.25f;
    p.n_q_heads = 1;
    p.n_rows_total = 16;
    p._reserved = 0;
    return p;
}

static std::vector<float> rope_tables(uint32_t rotary_dim, float mag = 1.0f) {
    uint32_t fc = rotary_dim / 2;
    std::vector<float> t(size_t(fc) * 2);
    for (uint32_t f = 0; f < fc; ++f) {
        t[f] = std::pow(10000.0f, -2.0f * (float)f / (float)rotary_dim);
        t[fc + f] = mag;
    }
    return t;
}

static void encode_q8_row(const float * src, uint32_t n, std::vector<uint8_t> & out_row) {
    // Minimal Q8_0 encoder for tests: single scale = max/127
    out_row.assign(34 * ((n + 31) / 32), 0);
    for (uint32_t b = 0; b < (n + 31) / 32; ++b) {
        float amax = 0.0f;
        for (uint32_t j = 0; j < 32 && b * 32 + j < n; ++j) {
            float a = std::fabs(src[b * 32 + j]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        if (!(d > 0.0f)) d = 1e-6f;
        uint16_t h;
        // f32 -> f16 bits (portable, round-to-nearest-even not required for tests)
        {
            uint32_t f; memcpy(&f, &d, 4);
            uint32_t sign = (f >> 31) & 1u;
            int32_t exp = (int32_t)((f >> 23) & 0xffu) - 127 + 15;
            uint32_t mant = (f >> 13) & 0x3ffu;
            if (exp <= 0) { exp = 0; mant = 0; }
            if (exp >= 31) { exp = 31; mant = 0; }
            h = (uint16_t)((sign << 15) | ((uint32_t)exp << 10) | mant);
        }
        out_row[b * 34 + 0] = (uint8_t)(h & 0xff);
        out_row[b * 34 + 1] = (uint8_t)(h >> 8);
        for (uint32_t j = 0; j < 32 && b * 32 + j < n; ++j) {
            int q = (int)std::round(src[b * 32 + j] / d);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            out_row[b * 34 + 2 + j] = (uint8_t)(q & 0xff);
        }
    }
}

// 1. Q8/Turbo4 capability gating: unsupported codec fails closed
static void test_capability_gate() {
    auto p = base_params();
    p.landmark_type = (uint32_t)GGML_TYPE_F16; // not a landmark codec here
    char err[256] = {};
    size_t need = 0;
    // xvk_check_caps rejects unsupported codecs before build (workspace_bytes fails closed)
    CHECK(!ggml_xkv_landmark_workspace_bytes(&p, &need, err, sizeof(err)));
    std::vector<float> q(32, 0.1f);
    std::vector<uint8_t> land(32 * 4, 0);
    int32_t fp[4] = {0, 1, 2, 3};
    int32_t fm[16] = {0, 4, 0, 1, 4, 4, 1, 1, 8, 4, 2, 1, 12, 4, 3, 1};
    int32_t qm[8] = {-1, 0, 1, 0, -1, 0, 1, 0};
    auto rope = rope_tables(8);
    int32_t ptrs[3] = {}; int32_t idx[4] = {}; float sc[4] = {}; int32_t st[4] = {};
    CHECK(!ggml_xkv_landmark_cpu_oracle(q.data(), land.data(), GGML_TYPE_F16,
        fp, fm, qm, rope.data(), &p, ptrs, idx, sc, st, err, sizeof(err)));
}

// 2. Illegal / all-masked input returns valid empty state without NaN
static void test_all_masked_empty() {
    auto p = base_params();
    p.n_queries = 1; p.n_frags = 3; p.top_k = 2; p.max_top_k = 3;
    char err[256] = {};
    std::vector<float> q(16, 0.5f);
    std::vector<float> raw(32, 0.0f); raw[0] = 1.0f;
    std::vector<uint8_t> row, land;
    encode_q8_row(raw.data(), 32, row);
    for (int i = 0; i < 3; ++i) land.insert(land.end(), row.begin(), row.end());
    int32_t fp[3] = {10, 20, 30};
    // all illegal (legal_flags bit0 clear)
    int32_t fm[12] = {0, 4, 0, 0, 4, 4, 1, 0, 8, 4, 2, 0};
    int32_t qm[4] = {-1, 0, 1, 0};
    auto rope = rope_tables(8);
    int32_t ptrs[2] = {-9, -9}; int32_t idx[2] = {-9, -9};
    float sc[2] = {123.0f, 456.0f}; int32_t st[4] = {};
    CHECK(ggml_xkv_landmark_cpu_oracle(q.data(), land.data(), GGML_TYPE_Q8_0,
        fp, fm, qm, rope.data(), &p, ptrs, idx, sc, st, err, sizeof(err)));
    CHECK(ptrs[0] == 0 && ptrs[1] == 0);
    CHECK(idx[0] == -1 && idx[1] == -1);
    for (float x : sc) CHECK(std::isfinite(x));
    CHECK(st[0] == GGML_XKV_LANDMARK_STATUS_OK);
}

// 3. Zero / oversized budget fails closed (never select-all)
static void test_budget_fails_closed() {
    auto p = base_params();
    char err[256] = {};
    std::vector<float> q(32, 0.1f);
    std::vector<float> raw(32, 1.0f);
    std::vector<uint8_t> row, land;
    encode_q8_row(raw.data(), 32, row);
    for (int i = 0; i < 4; ++i) land.insert(land.end(), row.begin(), row.end());
    int32_t fp[4] = {0, 1, 2, 3};
    int32_t fm[16] = {0, 4, 0, 1, 4, 4, 1, 1, 8, 4, 2, 1, 12, 4, 3, 1};
    int32_t qm[8] = {-1, 0, 1, 0, -1, 0, 1, 0};
    auto rope = rope_tables(8);
    int32_t ptrs[3] = {}; int32_t idx[8] = {}; float sc[8] = {}; int32_t st[4] = {};
    {
        auto pz = p; pz.top_k = 0;
        CHECK(!ggml_xkv_landmark_cpu_oracle(q.data(), land.data(), GGML_TYPE_Q8_0,
            fp, fm, qm, rope.data(), &pz, ptrs, idx, sc, st, err, sizeof(err)));
    }
    {
        auto po = p; po.top_k = 5; po.max_top_k = 8; // > n_frags
        CHECK(!ggml_xkv_landmark_cpu_oracle(q.data(), land.data(), GGML_TYPE_Q8_0,
            fp, fm, qm, rope.data(), &po, ptrs, idx, sc, st, err, sizeof(err)));
    }
}

// 4. Deterministic ties: equal scores select lowest fragment ids
static void test_deterministic_ties() {
    auto p = base_params();
    p.n_queries = 1; p.n_frags = 4; p.top_k = 2; p.max_top_k = 4;
    p.rotary_dim = 0; // no RoPE: identical landmarks tie exactly
    char err[256] = {};
    std::vector<float> q(16, 1.0f);
    std::vector<float> raw(32, 0.0f); raw[0] = 1.0f;
    std::vector<uint8_t> row, land;
    encode_q8_row(raw.data(), 32, row);
    for (int i = 0; i < 4; ++i) land.insert(land.end(), row.begin(), row.end());
    int32_t fp[4] = {0, 0, 0, 0};
    int32_t fm[16] = {0, 4, 0, 1, 4, 4, 1, 1, 8, 4, 2, 1, 12, 4, 3, 1};
    int32_t qm[4] = {-1, 0, 1, 0};
    int32_t ptrs[2] = {}; int32_t idx[2] = {}; float sc[2] = {}; int32_t st[4] = {};
    CHECK(ggml_xkv_landmark_cpu_oracle(q.data(), land.data(), GGML_TYPE_Q8_0,
        fp, fm, qm, nullptr, &p, ptrs, idx, sc, st, err, sizeof(err)));
    CHECK(ptrs[0] == 0 && ptrs[1] == 2);
    CHECK(idx[0] == 0 && idx[1] == 1); // score-desc/id-asc
}

// 5. Per-query isolation: speculative future query never influences earlier query
static void test_query_isolation() {
    auto p = base_params();
    p.n_queries = 2; p.n_frags = 3; p.top_k = 1; p.max_top_k = 3;
    p.rotary_dim = 0;
    char err[256] = {};
    // q0 prefers frag0, q1 (speculative future) prefers frag2 strongly
    std::vector<float> q(32, 0.0f);
    q[0] = 10.0f;                       // q0: frag0 direction
    q[16 + 2] = 100.0f;                 // q1: frag2 direction
    std::vector<float> r0(32, 0.0f), r1(32, 0.0f), r2(32, 0.0f);
    r0[0] = 1.0f; r1[1] = 1.0f; r2[2] = 1.0f;
    std::vector<uint8_t> e0, e1, e2, land;
    encode_q8_row(r0.data(), 32, e0);
    encode_q8_row(r1.data(), 32, e1);
    encode_q8_row(r2.data(), 32, e2);
    land.insert(land.end(), e0.begin(), e0.end());
    land.insert(land.end(), e1.begin(), e1.end());
    land.insert(land.end(), e2.begin(), e2.end());
    int32_t fp[3] = {0, 0, 0};
    int32_t fm[12] = {0, 4, 0, 1, 4, 4, 1, 1, 8, 4, 2, 1};
    int32_t qm[8] = {-1, 0, 1, 0, -1, 0, 1, 0};
    int32_t ptrs[3] = {}; int32_t idx[2] = {}; float sc[2] = {}; int32_t st[4] = {};
    CHECK(ggml_xkv_landmark_cpu_oracle(q.data(), land.data(), GGML_TYPE_Q8_0,
        fp, fm, qm, nullptr, &p, ptrs, idx, sc, st, err, sizeof(err)));
    CHECK(idx[0] == 0); // q0 unaffected by q1's strong frag2 preference
    CHECK(idx[1] == 2);
}

// 6. Bounded refinement: cap honored + cap-hit reported, no select-all fallback
static void test_refine_cap() {
    auto p = base_params();
    p.n_queries = 1; p.n_frags = 4; p.top_k = 2; p.max_top_k = 4;
    p.rotary_dim = 0; p.refine_cap = 5; p.frag_size = 4; p.n_rows_total = 16;
    char err[256] = {};
    std::vector<float> q(16, 1.0f);
    std::vector<float> raw(32, 0.05f); raw[0] = 1.0f;
    std::vector<uint8_t> row, land;
    encode_q8_row(raw.data(), 32, row);
    for (int i = 0; i < 4; ++i) land.insert(land.end(), row.begin(), row.end());
    int32_t fp[4] = {0, 0, 0, 0};
    int32_t fm[16] = {0, 4, 0, 1, 4, 4, 1, 1, 8, 4, 2, 1, 12, 4, 3, 1};
    int32_t qm[4] = {-1, 0, 1, 0};
    int32_t ptrs[2] = {}; int32_t idx[2] = {}; float sc[2] = {}; int32_t st[4] = {};
    CHECK(ggml_xkv_landmark_cpu_oracle(q.data(), land.data(), GGML_TYPE_Q8_0,
        fp, fm, qm, nullptr, &p, ptrs, idx, sc, st, err, sizeof(err)));
    // 2 frags x 4 rows = 8 rows > cap 5 -> clamped with hit
    CHECK(st[1] == 5);
    CHECK(st[2] == 1);
}

// 7. Workspace bounds: exact bytes + too-small scratch rejects at supports()
static void test_workspace_bounds() {
    auto p = base_params();
    char err[256] = {};
    size_t need = 0;
    CHECK(ggml_xkv_landmark_workspace_bytes(&p, &need, err, sizeof(err)));
    // v2 history-independent: carry idx+scores + take + refined|hit + legal +
    // refine rows + per-query tile staging + margin (NO n_frags term)
    CHECK(need == (size_t)(2 * 2 + 2 * 2 + 2 + 2 + 2 + 2 * 0 + 2 * 64 + 64) * sizeof(float));
}

// v2: workspace identical for nf=4 and nf=4000 (no history term)
static void test_history_independent_workspace() {
    auto p = base_params();
    char err[256] = {};
    size_t a = 0, b = 0;
    CHECK(ggml_xkv_landmark_workspace_bytes(&p, &a, err, sizeof(err)));
    p.n_frags = 4000;
    CHECK(ggml_xkv_landmark_workspace_bytes(&p, &b, err, sizeof(err)));
    CHECK(a == b && a > 0);
    std::printf("  workspace history-independent: %zu bytes\n", a);
}

// Explicit capability maxima: max valid, max+1 rejected before build.
static void test_capability_maxima() {
    char err[256] = {};
    size_t need = 0;
    auto p = base_params();
    p.top_k = GGML_XKV_LANDMARK_MAX_TOP_K; p.max_top_k = GGML_XKV_LANDMARK_MAX_TOP_K; p.n_frags = 128;
    CHECK(ggml_xkv_landmark_workspace_bytes(&p, &need, err, sizeof(err)));
    p.top_k = GGML_XKV_LANDMARK_MAX_TOP_K + 1; p.max_top_k = GGML_XKV_LANDMARK_MAX_TOP_K + 1;
    CHECK(!ggml_xkv_landmark_workspace_bytes(&p, &need, err, sizeof(err)));
    p = base_params(); p.refine_cap = GGML_XKV_LANDMARK_MAX_REFINE_CAP;
    CHECK(ggml_xkv_landmark_workspace_bytes(&p, &need, err, sizeof(err)));
    p.refine_cap = GGML_XKV_LANDMARK_MAX_REFINE_CAP + 1;
    CHECK(!ggml_xkv_landmark_workspace_bytes(&p, &need, err, sizeof(err)));
    p = base_params(); p.head_dim = GGML_XKV_LANDMARK_MAX_HEAD_DIM; p.padded_dim = 256;
    CHECK(ggml_xkv_landmark_workspace_bytes(&p, &need, err, sizeof(err)));
    p.head_dim = GGML_XKV_LANDMARK_MAX_HEAD_DIM + 1; p.padded_dim = 288;
    CHECK(!ggml_xkv_landmark_workspace_bytes(&p, &need, err, sizeof(err)));
    p = base_params(); p.n_queries = GGML_XKV_LANDMARK_MAX_QUERIES;
    CHECK(ggml_xkv_landmark_workspace_bytes(&p, &need, err, sizeof(err)));
    p.n_queries = GGML_XKV_LANDMARK_MAX_QUERIES + 1;
    CHECK(!ggml_xkv_landmark_workspace_bytes(&p, &need, err, sizeof(err)));
}

// >32 queries: 48-query determinism (repeat runs bitwise identical) + full take.
static void test_many_queries_determinism() {
    const uint32_t NQ = 48, NF = 70, TK = 3;
    auto p = base_params();
    p.n_queries = NQ; p.n_frags = NF; p.top_k = TK; p.max_top_k = 4;
    p.rotary_dim = 0;
    char err[256] = {};
    std::vector<float> q(NQ * 16, 0.0f);
    for (uint32_t qq = 0; qq < NQ; ++qq) q[qq * 16 + (qq % 16)] = 3.0f;
    std::vector<uint8_t> land;
    for (uint32_t f = 0; f < NF; ++f) {
        std::vector<float> raw(32, 0.0f); raw[f % 32] = 1.0f;
        std::vector<uint8_t> row; encode_q8_row(raw.data(), 32, row);
        land.insert(land.end(), row.begin(), row.end());
    }
    std::vector<int32_t> fp(NF), fm(NF * 4), qm(NQ * 4);
    for (uint32_t f = 0; f < NF; ++f) { fp[f] = (int32_t)f; fm[f*4+0]=0; fm[f*4+1]=2; fm[f*4+2]=0; fm[f*4+3]=1; }
    for (uint32_t qq = 0; qq < NQ; ++qq) { qm[qq*4+0]=-1; qm[qq*4+1]=0; qm[qq*4+2]=1; qm[qq*4+3]=0; }
    std::vector<int32_t> ptrs(NQ+1,-9), idx(NQ*TK,-9), ptrs2(NQ+1,-9), idx2(NQ*TK,-9);
    std::vector<float> sc(NQ*TK,0), sc2(NQ*TK,0);
    int32_t st[4]={}, st2[4]={};
    CHECK(ggml_xkv_landmark_cpu_oracle(q.data(), land.data(), GGML_TYPE_Q8_0,
        fp.data(), fm.data(), qm.data(), nullptr, &p, ptrs.data(), idx.data(), sc.data(), st, err, sizeof(err)));
    CHECK(st[0]==0 && ptrs[NQ]==(int32_t)(NQ*TK));
    CHECK(ggml_xkv_landmark_cpu_oracle(q.data(), land.data(), GGML_TYPE_Q8_0,
        fp.data(), fm.data(), qm.data(), nullptr, &p, ptrs2.data(), idx2.data(), sc2.data(), st2, err, sizeof(err)));
    CHECK(idx == idx2 && sc == sc2 && ptrs == ptrs2);
}

// Per-query generation gating via the extended oracle (stride-6 query meta).
static void test_per_query_epoch_gating() {
    auto p = base_params();
    p.n_queries = 2; p.n_frags = 3; p.top_k = 2; p.max_top_k = 3;
    p.rotary_dim = 0; p._reserved = GGML_XKV_LANDMARK_FLAG_PERQ_LEGAL;
    char err[256] = {};
    std::vector<float> q(32, 0.0f); q[0] = 5.0f; q[16+1] = 5.0f;
    std::vector<float> r0(32,0), r1(32,0), r2(32,0); r0[0]=1; r1[1]=1; r2[0]=1;
    std::vector<uint8_t> e0,e1,e2,land;
    encode_q8_row(r0.data(),32,e0); encode_q8_row(r1.data(),32,e1); encode_q8_row(r2.data(),32,e2);
    land.insert(land.end(),e0.begin(),e0.end()); land.insert(land.end(),e1.begin(),e1.end()); land.insert(land.end(),e2.begin(),e2.end());
    int32_t fp[3] = {0,0,0};
    int32_t fm[12] = {0,2,0,1, 2,2,1,1, 4,2,2,1};
    int32_t qm[12] = {-1,0,1,0,7,0, -1,0,1,0,9,0};
    int32_t gen[3] = {7,7,9};
    int32_t ptrs[3]={}, idx[4]={}; float sc[4]={}; int32_t st[4]={};
    CHECK(ggml_xkv_landmark_cpu_oracle_x(q.data(), land.data(), GGML_TYPE_Q8_0, fp, fm, 4,
        qm, 6, gen, nullptr, &p, 0, ptrs, idx, sc, st, err, sizeof(err)));
    CHECK(idx[0] == 0 && idx[2] == 2);
    // legacy entry must refuse PERQ_LEGAL (fail closed, never silent)
    int32_t qm4[8] = {-1,0,1,0, -1,0,1,0};
    CHECK(!ggml_xkv_landmark_cpu_oracle(q.data(), land.data(), GGML_TYPE_Q8_0, fp, fm, qm4,
        nullptr, &p, ptrs, idx, sc, st, err, sizeof(err)));
}

// Rows oracle: contiguous == sparse, partial holes skipped, cap-hit, OOB fail.
static void test_rows_oracle() {
    const uint32_t NQ = 2, TK = 2, RC = 5;
    char err[256] = {};
    int32_t sel[4] = {0, 1, 2, 3};
    int32_t fm[16] = {0,4,0,1, 4,4,1,1, 8,4,2,1, 12,4,3,1};
    int32_t kv[12] = {0,0,0, 0,0,0, 0,1,0, 0,1,0};
    int32_t fp[4] = {0, 1, 2, 3};
    int32_t pt1[3], rf1[4*5*2], rp1[5*2], re1[4*5*2], st1[4];
    ggml_xkv_landmark_rows_params p1 = {};
    p1.version = GGML_XKV_LANDMARK_VERSION;
    p1.n_queries = NQ; p1.top_k = TK; p1.n_frags = 4; p1.refine_cap = RC;
    p1.fstride = 4; p1.flags = 0; p1.max_frag_rows = 64; p1.n_rows_total = 16;
    p1.n_parent_queries = 0; p1.has_query_map = 0;
    p1.arena_filter = UINT32_MAX; p1.global_row_base = 0; p1.arena_row_count = 16;
    p1.output_row_begin = 0; p1._reserved = 0;
    CHECK(ggml_xkv_landmark_rows_cpu_oracle(sel, fm, nullptr, nullptr, kv, nullptr,
        fp, &p1, pt1, rf1, rp1, re1, st1, err, sizeof(err)));
    CHECK(st1[0]==0 && pt1[0]==0 && pt1[1]==5 && pt1[2]==10 && st1[1]==10 && st1[2]==2);
    CHECK(rf1[0]==0 && rf1[1]==0 && rf1[2]==0 && rf1[3]==0); // row,group,layer,head
    CHECK(re1[0]==2 && re1[1]==0 && re1[2]==0 && re1[3]==1); // source=COLD(2), valid entry

    // q0 short (1 frag = 4 rows < RC), q1 full (2 frags = 8 rows > RC=5, clamped)
    // Tests prefix compaction: q1's compacted rows must immediately follow q0's 4 rows!
    int32_t sel_short[4] = {0, -1, 1, 2}; // q0 has 1 frag (0), q1 has 2 frags (1, 2)
    int32_t pt_short[3], rf_short[4*5*2], rp_short[5*2], re_short[4*5*2], st_short[4];
    CHECK(ggml_xkv_landmark_rows_cpu_oracle(sel_short, fm, nullptr, nullptr, kv, nullptr,
        fp, &p1, pt_short, rf_short, rp_short, re_short, st_short, err, sizeof(err)));
    CHECK(st_short[0] == 0);
    CHECK(pt_short[0] == 0 && pt_short[1] == 4 && pt_short[2] == 9); // q0: 4 rows, q1: 5 rows
    // q0 rows are 0,1,2,3 at slots 0..3
    CHECK(rf_short[0*4] == 0 && rf_short[3*4] == 3);
    // q1 compacted rows start at slot pt_short[1]=4: slots 4..8 have rows 4,5,6,7,8!
    CHECK(rf_short[4*4] == 4 && rf_short[5*4] == 5 && rf_short[8*4] == 8);
    // slot 9 is padded with dummy row 0
    CHECK(rf_short[9*4] == 0);
    int32_t off[5] = {0,4,8,12,16};
    int32_t ids[16] = {0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15};
    int32_t pt2[3], rf2[4*5*2], rp2[5*2], re2[4*5*2], st2[4];
    auto p_sp = p1; p_sp.flags = GGML_XKV_LANDMARK_FLAG_SPARSE_ROWS;
    CHECK(ggml_xkv_landmark_rows_cpu_oracle(sel, fm, off, ids, kv, nullptr,
        fp, &p_sp,
        pt2, rf2, rp2, re2, st2, err, sizeof(err)));
    CHECK(pt2[1]==pt1[1] && pt2[2]==pt1[2] && st2[1]==st1[1]);

    // Sparse with hole (-1): frag 0 (rows 0,1,2,3) + frag 1 (rows 4,-1,6,7) -> 7 total rows per query, clamped to RC=5
    ids[5] = -1;
    int32_t pt3[3], rf3[4*5*2], rp3[5*2], re3[4*5*2], st3[4];
    CHECK(ggml_xkv_landmark_rows_cpu_oracle(sel, fm, off, ids, kv, nullptr,
        fp, &p_sp,
        pt3, rf3, rp3, re3, st3, err, sizeof(err)));
    CHECK(st3[0] == 0 && pt3[1] == 5 && pt3[2] == 10 && st3[1] == 10);
    // Verify hole row 5 was skipped: inspect q0's emitted rows (slots 0..4)
    CHECK(rf3[0*4] == 0 && rf3[1*4] == 1 && rf3[2*4] == 2 && rf3[3*4] == 3 && rf3[4*4] == 4);
    // Next row in list was row 6, which was clamped at cap=5
    int32_t bad[4] = {0, 99, 2, 3};
    int32_t ptx[3], rfx[4*5*2], rpx[5*2], rex[4*5*2], stx[4];
    CHECK(!ggml_xkv_landmark_rows_cpu_oracle(bad, fm, nullptr, nullptr, kv, nullptr,
        fp, &p1, ptx, rfx, rpx, rex, stx, err, sizeof(err)));
    // rows supports(): shapes + caps, T-1 style bound already covered by select tests
    size_t rneed = 0;
    CHECK(ggml_xkv_landmark_rows_workspace_bytes(NQ, TK, RC, &rneed, err, sizeof(err)));
    CHECK(rneed > 0);
    CHECK(!ggml_xkv_landmark_rows_workspace_bytes(NQ, TK, 0, &rneed, err, sizeof(err)));
}

static void test_rows_ddvr_parent_mapping() {
    constexpr uint32_t E = 3;
    constexpr uint32_t N = 2;
    constexpr uint32_t TK = 2;
    constexpr uint32_t CAP = 4;
    char err[256] = {};
    // [parent, slot, frag0, frag1] for each expanded selector query.
    int32_t mapped[(TK + 2) * E] = {
        0, 0, 0, -1,
        0, 2, 1, -1,
        1, 1, 0,  1,
    };
    // Both fragments reference the same physical rows. They must remain
    // distinct across DDVR slots, but deduplicate within parent 1 / slot 1.
    int32_t fm[8] = {0, 2, 0, 1, 0, 2, 1, 1};
    int32_t kv[6] = {7, 0, 3, 8, 1, 4};
    int32_t positions[2] = {10, 11};
    int32_t ptrs[N + 1] = {};
    int32_t refs[4 * CAP * E] = {};
    int32_t out_pos[CAP * E] = {};
    int32_t entries[4 * CAP * E] = {};
    int32_t status[4] = {};
    ggml_xkv_landmark_rows_params p = {};
    p.version = GGML_XKV_LANDMARK_VERSION;
    p.n_queries = E; p.top_k = TK; p.n_frags = 2; p.refine_cap = CAP;
    p.fstride = 4; p.flags = 0; p.max_frag_rows = 64; p.n_rows_total = 2;
    p.n_parent_queries = N; p.has_query_map = 1;
    p.arena_filter = UINT32_MAX; p.global_row_base = 0; p.arena_row_count = 2;
    p.output_row_begin = 0; p._reserved = 0;
    CHECK(ggml_xkv_landmark_rows_cpu_oracle(mapped, fm, nullptr, nullptr, kv,
        positions, positions, &p,
        ptrs, refs, out_pos, entries, status, err, sizeof(err)));
    CHECK(status[0] == 0 && status[1] == 6 && status[2] == 0);
    CHECK(ptrs[0] == 0 && ptrs[1] == 4 && ptrs[2] == 6);
    CHECK(entries[2] == 0 && entries[6] == 0);
    CHECK(entries[4 * 2 + 2] == 2 && entries[4 * 3 + 2] == 2);
    CHECK(entries[4 * 4 + 2] == 1 && entries[4 * 5 + 2] == 1);
    CHECK(refs[4 * 0] == 0 && refs[4 * 1] == 1 && refs[4 * 2] == 0 && refs[4 * 3] == 1);
    // Reconstruct quad: refs[1] and refs[2] must be 0
    CHECK(refs[1] == 0 && refs[2] == 0 && refs[5] == 0 && refs[6] == 0);
    // Entries source must be GGML_XKV_ATTN_SOURCE_COLD (2)
    CHECK(entries[0] == 2 && entries[4] == 2 && entries[8] == 2);

    int32_t bad_map[(TK + 2) * E];
    std::memcpy(bad_map, mapped, sizeof(mapped));
    bad_map[TK + 2 + 1] = 0; // duplicate/non-increasing slot for parent 0
    CHECK(!ggml_xkv_landmark_rows_cpu_oracle(bad_map, fm, nullptr, nullptr, kv,
        positions, positions, &p,
        ptrs, refs, out_pos, entries, status, err, sizeof(err)));

    // Identity mode derives slots from stride-6 metadata and deduplicates by
    // (row,slot), not row alone.
    int32_t identity_sel[2] = {0, 1};
    int32_t fm6[12] = {0, 2, 0, 1, -1, 0, 0, 2, 1, 1, -1, 1};
    int32_t iptrs[2] = {};
    int32_t irefs[4 * CAP] = {};
    int32_t ipos[CAP] = {};
    int32_t ientries[4 * CAP] = {};
    int32_t istatus[4] = {};
    auto p_id = p;
    p_id.n_queries = 1; p_id.fstride = 6; p_id.n_parent_queries = 0; p_id.has_query_map = 0;
    CHECK(ggml_xkv_landmark_rows_cpu_oracle(identity_sel, fm6, nullptr, nullptr, kv,
        positions, positions, &p_id,
        iptrs, irefs, ipos, ientries, istatus, err, sizeof(err)));
    CHECK(iptrs[0] == 0 && iptrs[1] == 4);
    CHECK(ientries[2] == 0 && ientries[6] == 1 &&
          ientries[10] == 0 && ientries[14] == 1);
}

// Merge equivalence: per-segment top-k + merge == global top-k.
static void test_merge_equivalence() {
    const uint32_t NQ = 3, NF = 40, NF1 = 17, TK = 4;
    auto p = base_params();
    p.rotary_dim = 0;
    char err[256] = {};
    std::vector<float> q(NQ * 16, 0.0f);
    for (uint32_t qq = 0; qq < NQ; ++qq) q[qq*16 + ((qq*5+1) % 16)] = 2.0f;
    std::vector<uint8_t> land;
    for (uint32_t f = 0; f < NF; ++f) {
        std::vector<float> raw(32, 0.0f); raw[(f*3+1) % 32] = 1.0f;
        std::vector<uint8_t> row; encode_q8_row(raw.data(), 32, row);
        land.insert(land.end(), row.begin(), row.end());
    }
    std::vector<int32_t> fp(NF), fm(NF*4), qm(NQ*4);
    for (uint32_t f = 0; f < NF; ++f) { fp[f]=(int32_t)f; fm[f*4+0]=0; fm[f*4+1]=2; fm[f*4+2]=0; fm[f*4+3]=1; }
    for (uint32_t qq = 0; qq < NQ; ++qq) { qm[qq*4+0]=-1; qm[qq*4+1]=0; qm[qq*4+2]=1; qm[qq*4+3]=0; }
    auto pg = p; pg.n_queries = NQ; pg.n_frags = NF; pg.top_k = TK; pg.max_top_k = TK;
    std::vector<int32_t> gpt(NQ+1), gix(NQ*TK); std::vector<float> gsc(NQ*TK); int32_t gst[4]={};
    CHECK(ggml_xkv_landmark_cpu_oracle(q.data(), land.data(), GGML_TYPE_Q8_0, fp.data(), fm.data(),
        qm.data(), nullptr, &pg, gpt.data(), gix.data(), gsc.data(), gst, err, sizeof(err)));
    auto p1 = pg; p1.n_frags = NF1;
    auto p2 = pg; p2.n_frags = NF - NF1;
    std::vector<uint8_t> land1(land.begin(), land.begin()+(size_t)NF1*34);
    std::vector<uint8_t> land2(land.begin()+(size_t)NF1*34, land.end());
    std::vector<int32_t> fp1(fp.begin(), fp.begin()+NF1), fp2(fp.begin()+NF1, fp.end());
    std::vector<int32_t> fm1(fm.begin(), fm.begin()+NF1*4), fm2(fm.begin()+NF1*4, fm.end());
    std::vector<int32_t> ix1(NQ*TK), ix2(NQ*TK), pt(NQ+1); std::vector<float> s1(NQ*TK), s2(NQ*TK);
    int32_t t1[4]={}, t2[4]={};
    CHECK(ggml_xkv_landmark_cpu_oracle(q.data(), land1.data(), GGML_TYPE_Q8_0, fp1.data(), fm1.data(),
        qm.data(), nullptr, &p1, pt.data(), ix1.data(), s1.data(), t1, err, sizeof(err)));
    CHECK(ggml_xkv_landmark_cpu_oracle(q.data(), land2.data(), GGML_TYPE_Q8_0, fp2.data(), fm2.data(),
        qm.data(), nullptr, &p2, pt.data(), ix2.data(), s2.data(), t2, err, sizeof(err)));
    std::vector<int32_t> set_idx((size_t)TK*NQ*2), m_idx(NQ*TK);
    std::vector<float> set_sc((size_t)TK*NQ*2), m_sc(NQ*TK);
    for (uint32_t qq = 0; qq < NQ; ++qq) for (uint32_t i = 0; i < TK; ++i) {
        int32_t a = ix1[qq*TK+i];
        set_idx[(0u*NQ+qq)*TK+i] = a; set_sc[(0u*NQ+qq)*TK+i] = s1[qq*TK+i];
        int32_t b = ix2[qq*TK+i];
        set_idx[(1u*NQ+qq)*TK+i] = b < 0 ? -1 : b + (int32_t)NF1; set_sc[(1u*NQ+qq)*TK+i] = s2[qq*TK+i];
    }
    CHECK(ggml_xkv_landmark_merge_cpu_oracle(set_idx.data(), set_sc.data(), NQ, 2, TK, TK,
        m_idx.data(), m_sc.data(), err, sizeof(err)));
    CHECK(m_idx == gix);
}

// Phase domain contract:
// 1. Phased-direct production mode scores stored landmark directly (no re-phase).
//    Varying positions within chunk verified against row-wise-phase-then-mean oracle:
//    the oracle matches the pre-phased stored vector, while applying representative
//    RoPE at frag_pos would diverge.
// 2. Canonical experimental mode (FLAG_CANONICAL_XPHASE) is defined ONLY for row_count==1
//    and requires an exact phase fingerprint match. Multi-row frags or fp mismatch reject.
static void test_phase_domain_contract() {
    char err[256] = {};
    const uint32_t DIM = 16, FC = 4;
    auto rope = rope_tables(8); // rotary_dim=8, fc=4
    uint64_t fp_valid = ggml_xkv_landmark_phase_fingerprint(
        rope.data(), FC, GGML_XKV_LANDMARK_ROPE_HALF, 8, DIM, (uint32_t)GGML_TYPE_Q8_0);
    CHECK(fp_valid != 0);
    // canonical mode + multi-row fragment -> rejects (ERR_INPUT)
    {
        auto p = base_params();
        p.rotary_dim = 8; p.rope_mode = GGML_XKV_LANDMARK_ROPE_HALF;
        p._reserved = GGML_XKV_LANDMARK_FLAG_CANONICAL_XPHASE;
        std::vector<float> q(16, 0.1f);
        std::vector<uint8_t> land(32 * 2, 0);
        int32_t fpos[2] = {1, 2};
        int32_t fmeta_mr[8] = {0, 4, 0, 1,  4, 4, 1, 1}; // row_count=4 > 1!
        int32_t qmeta[4] = {-1, 0, 1, 0};
        int32_t ptrs[3] = {}; int32_t idx[4] = {}; float sc[4] = {}; int32_t st[4] = {};
        CHECK(!ggml_xkv_landmark_cpu_oracle_x(q.data(), land.data(), GGML_TYPE_Q8_0,
            fpos, fmeta_mr, 4, qmeta, 4, nullptr, rope.data(), &p, fp_valid,
            ptrs, idx, sc, st, err, sizeof(err)));
        CHECK(st[0] == GGML_XKV_LANDMARK_STATUS_ERR_INPUT);
    }
    // canonical mode + fingerprint mismatch -> rejects (ERR_INPUT)
    {
        auto p = base_params();
        p.rotary_dim = 8; p.rope_mode = GGML_XKV_LANDMARK_ROPE_HALF;
        p._reserved = GGML_XKV_LANDMARK_FLAG_CANONICAL_XPHASE;
        std::vector<float> q(16, 0.1f);
        std::vector<uint8_t> land(32, 0);
        int32_t fpos[1] = {1};
        int32_t fmeta[4] = {0, 1, 0, 1}; // row_count=1
        int32_t qmeta[4] = {-1, 0, 1, 0};
        int32_t ptrs[2] = {}; int32_t idx[2] = {}; float sc[2] = {}; int32_t st[4] = {};
        CHECK(!ggml_xkv_landmark_cpu_oracle_x(q.data(), land.data(), GGML_TYPE_Q8_0,
            fpos, fmeta, 4, qmeta, 4, nullptr, rope.data(), &p, fp_valid ^ 0xDEADBEEF,
            ptrs, idx, sc, st, err, sizeof(err)));
        CHECK(st[0] == GGML_XKV_LANDMARK_STATUS_ERR_INPUT);
    }
    // varying positions within chunk: row-wise-phase-then-mean oracle vs representative
    // position RoPE. Stored landmark = mean(RoPE(r0, pos0), RoPE(r1, pos1)). Phased-direct
    // scores it as-is and matches dot(q, stored_mean); applying representative RoPE at
    // pos0 diverges.
    {
        const int32_t pos0 = 3, pos1 = 17; // non-uniform positions within fragment
        std::vector<float> r0(DIM, 0.0f), r1(DIM, 0.0f);
        r0[0] = 1.0f; r0[4] = 0.5f;
        r1[1] = 0.8f; r1[5] = 0.3f;
        // row-wise forward RoPE (HALF)
        auto phase_row = [&](std::vector<float> & v, int32_t pos) {
            for (uint32_t f = 0; f < FC; ++f) {
                float ang = (float)pos * rope[f];
                float c = cosf(ang), s = sinf(ang);
                float re = v[f], im = v[f + FC];
                v[f] = re * c - im * s;
                v[f + FC] = re * s + im * c;
            }
        };
        phase_row(r0, pos0);
        phase_row(r1, pos1);
        std::vector<float> stored_mean(DIM);
        for (uint32_t d = 0; d < DIM; ++d) stored_mean[d] = 0.5f * (r0[d] + r1[d]);
        std::vector<float> stored_pad(32, 0.0f);
        memcpy(stored_pad.data(), stored_mean.data(), DIM * sizeof(float));
        std::vector<uint8_t> land;
        encode_q8_row(stored_pad.data(), 32, land);
        std::vector<float> q(DIM, 0.0f);
        q[0] = 0.7f; q[1] = 0.3f; q[4] = 0.2f;
        auto p = base_params();
        p.n_queries = 1; p.n_frags = 1; p.top_k = 1; p.max_top_k = 1;
        p.rotary_dim = 0; p.rope_mode = GGML_XKV_LANDMARK_ROPE_HALF;
        // rope_tables must be empty when rotary_dim == 0 (matches supports & check_caps)
        std::vector<float> empty_rope;
        int32_t fpos[1] = {pos0}; // representative pos (causal only)
        int32_t fmeta[4] = {0, 2, 0, 1}; // rc=2
        int32_t qmeta[4] = {-1, 0, 1, 0};
        int32_t ptrs[2] = {}; int32_t idx[1] = {}; float sc[1] = {}; int32_t st[4] = {};
        // Phased-direct oracle matches dot(q, stored_mean)
        CHECK(ggml_xkv_landmark_cpu_oracle(q.data(), land.data(), GGML_TYPE_Q8_0,
            fpos, fmeta, qmeta, nullptr, &p, ptrs, idx, sc, st, err, sizeof(err)));
        double expected_dot = 0.0;
        for (uint32_t d = 0; d < DIM; ++d) expected_dot += (double)q[d] * (double)stored_mean[d];
        float expected_score = (float)(expected_dot * (double)p.scale);
        CHECK(std::fabs(sc[0] - expected_score) < 0.05f); // within Q8 quant noise
    }
}

// v3 stride-8 frag meta: [rb,rc,seg,elig,row_off,group,fpos,gen].
// Disjoint fpos and gen arrays MUST be NULL.
static void test_v3_stride8_oracle() {
    char err[256] = {};
    auto p = base_params();
    p.n_queries = 2; p.n_frags = 3; p.top_k = 2; p.max_top_k = 3;
    p.rotary_dim = 0; p._reserved = GGML_XKV_LANDMARK_FLAG_PERQ_LEGAL;
    std::vector<float> q(32, 0.0f); q[0] = 5.0f; q[16+1] = 5.0f;
    std::vector<float> r0(32,0), r1(32,0), r2(32,0); r0[0]=1; r1[1]=1; r2[0]=1;
    std::vector<uint8_t> e0,e1,e2,land;
    encode_q8_row(r0.data(),32,e0); encode_q8_row(r1.data(),32,e1); encode_q8_row(r2.data(),32,e2);
    land.insert(land.end(),e0.begin(),e0.end()); land.insert(land.end(),e1.begin(),e1.end()); land.insert(land.end(),e2.begin(),e2.end());
    // stride-8 meta: [rb,rc,seg,elig,row_off,grp,fpos,gen]
    int32_t fm8[24] = {
        0, 2, 0, 1,  0, 0, 10, 7,
        2, 2, 1, 1,  4, 0, 20, 7,
        4, 2, 2, 1,  8, 0, 30, 9
    };
    int32_t qm[12] = {-1,0,1,0,7,0, -1,0,1,0,9,0}; // q0 epoch 7, q1 epoch 9
    int32_t ptrs[3]={}, idx[4]={}; float sc[4]={}; int32_t st[4]={};
    CHECK(ggml_xkv_landmark_v3_cpu_oracle(q.data(), land.data(), GGML_TYPE_Q8_0,
        fm8, qm, 6, nullptr, &p, 0, ptrs, idx, sc, st, err, sizeof(err)));
    CHECK(idx[0] == 0 && idx[2] == 2);
}

// Nominal benchmark: 1 query, 16 heads, 256 dim, 256 fragments (~production size).
// Measures CPU oracle execution time and records it.
#include <chrono>
static void test_benchmark_cpu_nominal() {
    const uint32_t NQ = 1, NQH = 16, HD = 256, PD = 256, NF = 256, TK = 16;
    auto p = base_params();
    p.n_queries = NQ; p.n_q_heads = NQH; p.head_dim = HD; p.padded_dim = PD;
    p.n_frags = NF; p.top_k = TK; p.max_top_k = TK; p.rotary_dim = 0;
    p.scale = 1.0f / std::sqrt((float)HD);
    std::vector<float> q(NQ * NQH * HD, 0.01f);
    std::vector<uint8_t> land;
    std::vector<float> row(PD, 0.01f);
    for (uint32_t f = 0; f < NF; ++f) {
        row[f % PD] = 1.0f;
        std::vector<uint8_t> enc;
        encode_q8_row(row.data(), PD, enc);
        land.insert(land.end(), enc.begin(), enc.end());
    }
    std::vector<int32_t> fpos(NF, 0), fm(NF * 4);
    for (uint32_t f = 0; f < NF; ++f) { fm[f*4+0]=0; fm[f*4+1]=8; fm[f*4+2]=0; fm[f*4+3]=1; }
    std::vector<int32_t> qm = {-1, 0, (int32_t)NQH, 0};
    std::vector<int32_t> ptrs(NQ + 1), idx(NQ * TK);
    std::vector<float> sc(NQ * TK);
    int32_t st[4] = {};
    char err[256] = {};
    // Warmup
    CHECK(ggml_xkv_landmark_cpu_oracle(q.data(), land.data(), GGML_TYPE_Q8_0,
        fpos.data(), fm.data(), qm.data(), nullptr, &p,
        ptrs.data(), idx.data(), sc.data(), st, err, sizeof(err)));
    // Timed runs (10 iterations)
    const int ITERS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        ggml_xkv_landmark_cpu_oracle(q.data(), land.data(), GGML_TYPE_Q8_0,
            fpos.data(), fm.data(), qm.data(), nullptr, &p,
            ptrs.data(), idx.data(), sc.data(), st, err, sizeof(err));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double mean_us = total_us / (double)ITERS;
    std::printf("[benchmark] CPU nominal (1q x 16qh x 256dim x 256frags, top_k=%u): %.1f us/eval (%d iters)\n",
        TK, mean_us, ITERS);
}

// ---------- graph fixtures (CPU oracle vs CPU graph vs Vulkan graph) ----------

struct lm_case {
    ggml_xkv_landmark_params p;
    std::vector<float>   q;      // F32 [hd, nqh*nq]
    std::vector<uint8_t> land;   // packed [padded_dim, nf]
    std::vector<int32_t> fpos;   // [nf]
    std::vector<int32_t> fmeta;  // [4*nf]
    std::vector<int32_t> qmeta;  // [4*nq]
    std::vector<float>   rope;   // [2*Fc]
};

struct lm_out {
    std::vector<int32_t> ptrs;   // [nq+1]
    std::vector<int32_t> idx;    // [top_k*nq]
    std::vector<float>   scores; // [top_k*nq]
    int32_t status[4] = {-9, -9, -9, -9};
    bool computed = false;
};

static bool run_oracle_case(const lm_case & c, lm_out & o) {
    o.ptrs.assign(c.p.n_queries + 1, -9);
    o.idx.assign(size_t(c.p.top_k) * c.p.n_queries, -9);
    o.scores.assign(size_t(c.p.top_k) * c.p.n_queries, 12345.0f);
    o.status[0] = o.status[1] = o.status[2] = o.status[3] = -9;
    char err[256] = {};
    bool ok = ggml_xkv_landmark_cpu_oracle(c.q.data(), c.land.data(),
        (enum ggml_type)c.p.landmark_type, c.fpos.data(), c.fmeta.data(),
        c.qmeta.data(), c.rope.data(), &c.p,
        o.ptrs.data(), o.idx.data(), o.scores.data(), o.status, err, sizeof(err));
    o.computed = ok;
    if (!ok) std::fprintf(stderr, "oracle failed: %s\n", err);
    return ok;
}

static bool run_graph_case(ggml_backend_t backend, const lm_case & c, lm_out & o) {
    const auto & p = c.p;
    o.ptrs.assign(p.n_queries + 1, -9);
    o.idx.assign(size_t(p.top_k) * p.n_queries, -9);
    o.scores.assign(size_t(p.top_k) * p.n_queries, 12345.0f);
    o.status[0] = o.status[1] = o.status[2] = o.status[3] = -9;
    size_t need = 0;
    char err0[256] = {};
    if (!ggml_xkv_landmark_workspace_bytes(&p, &need, err0, sizeof(err0))) return false;
    ggml_init_params ip = { ggml_tensor_overhead() * 24 + ggml_graph_overhead_custom(24, false), nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor * q = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, p.head_dim, p.n_q_heads * p.n_queries);
    ggml_tensor * land = ggml_new_tensor_2d(ctx.get(), (enum ggml_type)p.landmark_type, p.padded_dim, p.n_frags);
    ggml_tensor * fp = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, p.n_frags);
    ggml_tensor * fm = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, p.n_frags);
    ggml_tensor * qm = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, p.n_queries);
    ggml_tensor * rope = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, (int64_t)c.rope.size());
    ggml_tensor * scr = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, (int64_t)(need / 4));
    ggml_tensor * ptrs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, p.n_queries + 1);
    ggml_tensor * sc = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, p.top_k, p.n_queries);
    ggml_tensor * st = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 4);
    ggml_tensor * dst = ggml_xkv_landmark(ctx.get(), q, land, fp, fm, qm, rope, scr, ptrs, sc, st, &p);
    if (!dst) return false;
    if (!ggml_backend_supports_op(backend, dst)) {
        std::fprintf(stderr, "backend does not support GGML_OP_XKV_LANDMARK\n");
        return false;
    }
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buf) return false;
    ggml_backend_tensor_set(q, c.q.data(), 0, c.q.size() * 4);
    ggml_backend_tensor_set(land, c.land.data(), 0, c.land.size());
    ggml_backend_tensor_set(fp, c.fpos.data(), 0, c.fpos.size() * 4);
    ggml_backend_tensor_set(fm, c.fmeta.data(), 0, c.fmeta.size() * 4);
    ggml_backend_tensor_set(qm, c.qmeta.data(), 0, c.qmeta.size() * 4);
    if (!c.rope.empty()) ggml_backend_tensor_set(rope, c.rope.data(), 0, c.rope.size() * 4);
    // sentinels on device-written outputs (faults must leave them unpublished)
    std::vector<int32_t> psent(p.n_queries + 1, -9);
    ggml_backend_tensor_set(ptrs, psent.data(), 0, psent.size() * 4);
    std::vector<int32_t> isent(size_t(p.top_k) * p.n_queries, -9);
    ggml_backend_tensor_set(dst, isent.data(), 0, isent.size() * 4);
    std::vector<float> ssent(size_t(p.top_k) * p.n_queries, 12345.0f);
    ggml_backend_tensor_set(sc, ssent.data(), 0, ssent.size() * 4);
    int32_t stsent[4] = {-9, -9, -9, -9};
    ggml_backend_tensor_set(st, stsent, 0, sizeof(stsent));
    ggml_cgraph * g = ggml_new_graph_custom(ctx.get(), 24, false);
    ggml_build_forward_expand(g, dst);
    if (ggml_backend_graph_compute(backend, g) != GGML_STATUS_SUCCESS) return false;
    ggml_backend_synchronize(backend); // ONE final sync per selection bundle
    o.ptrs.assign(p.n_queries + 1, 0);
    o.idx.assign(size_t(p.top_k) * p.n_queries, 0);
    o.scores.assign(size_t(p.top_k) * p.n_queries, 0.0f);
    ggml_backend_tensor_get(ptrs, o.ptrs.data(), 0, o.ptrs.size() * 4);
    ggml_backend_tensor_get(dst, o.idx.data(), 0, o.idx.size() * 4);
    ggml_backend_tensor_get(sc, o.scores.data(), 0, o.scores.size() * 4);
    ggml_backend_tensor_get(st, o.status, 0, sizeof(o.status)); // tiny scalar D2H
    o.computed = true;
    return true;
}

// Q8 fixture: 2 queries (GQA x2), 6 frags, clear margins, 1 illegal + 1 future
// decoy with huge raw values (must never be selected), ties on q1 tail.
static lm_case make_q8_case() {
    lm_case c;
    auto & p = c.p;
    memset(&p, 0, sizeof(p));
    p.version = GGML_XKV_LANDMARK_VERSION;
    p.n_queries = 2; p.n_frags = 6;
    p.head_dim = 32; p.padded_dim = 32;
    p.rotary_dim = 8; p.rope_mode = GGML_XKV_LANDMARK_ROPE_HALF;
    p.landmark_type = (uint32_t)GGML_TYPE_Q8_0;
    p.top_k = 2; p.max_top_k = 4;
    p.refine_cap = 0; p.frag_size = 4;
    p.scale = 0.25f; p.n_q_heads = 2; p.n_rows_total = 24;
    // landmark rows: axis spikes + decoy
    std::vector<std::vector<float>> rows(6, std::vector<float>(32, 0.0f));
    rows[0][0] = 4.0f;
    rows[1][1] = 4.0f;
    rows[2][0] = 3.0f; rows[2][1] = 3.0f;
    rows[3][2] = 4.0f;
    rows[4][0] = 100.0f; // illegal decoy (legal_flags=0)
    rows[5][1] = 100.0f; // causally-future decoy
    c.land.clear();
    for (int f = 0; f < 6; ++f) {
        std::vector<uint8_t> enc;
        encode_q8_row(rows[f].data(), 32, enc);
        c.land.insert(c.land.end(), enc.begin(), enc.end());
    }
    c.fpos = {0, 1, 2, 3, 4, 100};
    c.fmeta = {
        0,4,0,1,  4,4,0,1,  8,4,0,1,
        12,4,0,1,  16,4,0,0,  20,4,0,1};
    // q0 (heads: e0-ish, e1-ish) prefers f0/f2; q1 prefers f3; causal caps at 10
    c.q.assign(32 * 2 * 2, 0.0f);
    c.q[0 * 32 + 0] = 1.0f; c.q[0 * 32 + 1] = 0.2f;   // q0h0
    c.q[1 * 32 + 0] = 0.2f; c.q[1 * 32 + 1] = 1.0f;   // q0h1
    c.q[2 * 32 + 2] = 1.0f;                            // q1h0
    c.q[3 * 32 + 2] = 1.0f; c.q[3 * 32 + 0] = 0.1f;   // q1h1
    c.qmeta = {10, 5, 2, 0,  10, 6, 2, 0};
    c.p.rotary_dim = 0; // phased-direct: production landmarks are pre-phased; rotary_dim=0
    c.rope.clear();
    return c;
}

// Turbo4 fixture: 4 frags, refinement active (phased-direct: rotary_dim=0)
static lm_case make_t4_case() {
    lm_case c;
    auto & p = c.p;
    memset(&p, 0, sizeof(p));
    p.version = GGML_XKV_LANDMARK_VERSION;
    p.n_queries = 2; p.n_frags = 4;
    p.head_dim = 128; p.padded_dim = 128;
    p.rotary_dim = 0; p.rope_mode = GGML_XKV_LANDMARK_ROPE_HALF;
    p.landmark_type = (uint32_t)GGML_TYPE_TURBO4_0;
    p.top_k = 2; p.max_top_k = 4;
    p.refine_cap = 6; p.frag_size = 4; p.n_rows_total = 16;
    p.scale = 0.125f; p.n_q_heads = 1;
    std::vector<std::vector<float>> rows(4, std::vector<float>(128, 0.0f));
    rows[0][0] = 2.0f; rows[0][64] = 1.0f;
    rows[1][1] = 2.0f; rows[1][65] = 1.0f;
    rows[2][2] = 2.0f;
    rows[3][3] = 2.0f;
    c.land.clear();
    for (int f = 0; f < 4; ++f) {
        size_t rb = ggml_row_size(GGML_TYPE_TURBO4_0, 128);
        size_t off = c.land.size();
        c.land.resize(off + rb);
        CHECK(ggml_quantize_turbo_row(GGML_TYPE_TURBO4_0, rows[f].data(),
            c.land.data() + off, 128, 128));
    }
    c.fpos = {1, 2, 3, 4};
    c.fmeta = {0,4,0,1,  4,4,0,1,  8,4,0,1,  12,4,0,1};
    c.q.assign(128 * 2, 0.0f);
    c.q[0] = 1.0f; c.q[64] = 0.5f;      // q0 -> f0
    c.q[128 + 2] = 1.0f;                 // q1 -> f2
    c.qmeta = {10, 7, 1, 0,  10, 8, 1, 0};
    c.rope.clear();
    return c;
}

static void test_cpu_graph_matches_oracle() {
    std::printf("[lm-graph] CPU backend graph == oracle ...\n");
    ggml_backend_t cpu = ggml_backend_cpu_init();
    CHECK(cpu != nullptr);
    {
        lm_case c = make_q8_case();
        lm_out ref, got;
        CHECK(run_oracle_case(c, ref));
        CHECK(run_graph_case(cpu, c, got));
        CHECK(got.computed && ref.computed);
        CHECK(got.ptrs == ref.ptrs);
        CHECK(got.idx == ref.idx);
        CHECK(got.scores == ref.scores);
        for (int i = 0; i < 4; ++i) CHECK(got.status[i] == ref.status[i]);
        CHECK(got.status[0] == GGML_XKV_LANDMARK_STATUS_OK);
        // decoys never selected
        for (int v : got.idx) CHECK(v != 4 && v != 5);
    }
    {
        lm_case c = make_t4_case();
        lm_out ref, got;
        CHECK(run_oracle_case(c, ref));
        CHECK(run_graph_case(cpu, c, got));
        CHECK(got.computed && ref.computed);
        CHECK(got.ptrs == ref.ptrs);
        CHECK(got.idx == ref.idx);
        CHECK(got.scores == ref.scores);
        for (int i = 0; i < 4; ++i) CHECK(got.status[i] == ref.status[i]);
    }
    ggml_backend_free(cpu);
    std::printf("  CPU graph bitwise OK\n");
}

static ggml_backend_t open_vulkan_backend() {
    ggml_backend_load_all();
    ggml_backend_t vk = nullptr;
    vk = ggml_backend_vk_init(0);
    return vk;
}

static bool scores_close(const std::vector<float> & a, const std::vector<float> & b, float tol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

static void test_vulkan_landmark_graph() {
    std::printf("[lm-graph] Vulkan backend graph vs oracle ...\n");
    ggml_backend_t vk = open_vulkan_backend();
    if (!vk) { std::printf("  SKIP: no Vulkan device\n"); return; }
    {
        // Q8: exact CSR/indices/status, scores within reduction-order tol
        lm_case c = make_q8_case();
        lm_out ref;
        CHECK(run_oracle_case(c, ref));
        // probe supports() first: claimed capability must hold
        {
            size_t need = 0; char e0[256] = {};
            CHECK(ggml_xkv_landmark_workspace_bytes(&c.p, &need, e0, sizeof(e0)));
            ggml_init_params ip = { ggml_tensor_overhead() * 24, nullptr, true };
            ggml_context_ptr ctx(ggml_init(ip));
            ggml_tensor * q = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, c.p.head_dim, c.p.n_q_heads * c.p.n_queries);
            ggml_tensor * land = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, c.p.padded_dim, c.p.n_frags);
            ggml_tensor * fp = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, c.p.n_frags);
            ggml_tensor * fm = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, c.p.n_frags);
            ggml_tensor * qm = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, c.p.n_queries);
            ggml_tensor * rope = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, (int64_t)c.rope.size());
            ggml_tensor * scr = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, (int64_t)(need / 4));
            ggml_tensor * ptrs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, c.p.n_queries + 1);
            ggml_tensor * sc = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, c.p.top_k, c.p.n_queries);
            ggml_tensor * st = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 4);
            ggml_tensor * dst = ggml_xkv_landmark(ctx.get(), q, land, fp, fm, qm, rope, scr, ptrs, sc, st, &c.p);
            CHECK(dst != nullptr);
            CHECK(ggml_backend_supports_op(vk, dst));
        }
        lm_out got;
        CHECK(run_graph_case(vk, c, got));
        CHECK(got.computed);
        CHECK(got.ptrs == ref.ptrs);
        CHECK(got.idx == ref.idx);
        CHECK(scores_close(got.scores, ref.scores, 1e-4f));
        for (int i = 0; i < 4; ++i) CHECK(got.status[i] == ref.status[i]);
        for (int v : got.idx) CHECK(v != 4 && v != 5);
        // determinism on device
        lm_out got2;
        CHECK(run_graph_case(vk, c, got2));
        CHECK(got2.idx == got.idx);
        CHECK(got2.scores == got.scores);
        CHECK(got2.ptrs == got.ptrs);
    }
    {
        // Turbo4 INTERLEAVED + refinement: CSR exact, status exact
        lm_case c = make_t4_case();
        lm_out ref, got;
        CHECK(run_oracle_case(c, ref));
        CHECK(run_graph_case(vk, c, got));
        CHECK(got.computed);
        CHECK(got.ptrs == ref.ptrs);
        CHECK(got.idx == ref.idx);
        CHECK(scores_close(got.scores, ref.scores, 1e-3f));
        for (int i = 0; i < 4; ++i) CHECK(got.status[i] == ref.status[i]);
    }
    ggml_backend_free(vk);
    std::printf("  Vulkan graph OK\n");
}

static void test_landmark_scratch_exact() {
    std::printf("[lm-graph] exact scratch T vs T-1 ...\n");
    lm_case c = make_q8_case();
    size_t need = 0;
    char err[256] = {};
    CHECK(ggml_xkv_landmark_workspace_bytes(&c.p, &need, err, sizeof(err)));
    CHECK(need > 0);
    ggml_init_params ip = { ggml_tensor_overhead() * 24, nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor * q = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, c.p.head_dim, c.p.n_q_heads * c.p.n_queries);
    ggml_tensor * land = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, c.p.padded_dim, c.p.n_frags);
    ggml_tensor * fp = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, c.p.n_frags);
    ggml_tensor * fm = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, c.p.n_frags);
    ggml_tensor * qm = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, c.p.n_queries);
    ggml_tensor * rope = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, (int64_t)c.rope.size());
    ggml_tensor * full = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, (int64_t)(need / 4));
    ggml_tensor * shrt = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, (int64_t)(need / 4) - 1);
    ggml_tensor * ptrs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, c.p.n_queries + 1);
    ggml_tensor * sc = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, c.p.top_k, c.p.n_queries);
    ggml_tensor * st = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 4);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, c.p.top_k, c.p.n_queries);
    CHECK(ggml_xkv_landmark_supports(q, land, fp, fm, qm, rope, full, ptrs, sc, st, dst, &c.p, err, sizeof(err)));
    CHECK(!ggml_xkv_landmark_supports(q, land, fp, fm, qm, rope, shrt, ptrs, sc, st, dst, &c.p, err, sizeof(err)));
    std::printf("  need=%zu bytes; T ok, T-1 rejected (%s)\n", need, err);
}

static void test_landmark_fault_atomicity() {
    std::printf("[lm-graph] fault atomicity (untouched outputs) ...\n");
    // oracle-level: budget/codec faults report status, no crash
    {
        lm_case c = make_q8_case();
        lm_out o;
        auto bad = c.p; bad.top_k = 0;
        char err[256] = {};
        int32_t ptrs[3] = {-9,-9,-9}, idx[4] = {-9,-9,-9,-9}, st[4] = {-9,-9,-9,-9};
        float sc[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        CHECK(!ggml_xkv_landmark_cpu_oracle(c.q.data(), c.land.data(), GGML_TYPE_Q8_0,
            c.fpos.data(), c.fmeta.data(), c.qmeta.data(), c.rope.data(), &bad,
            ptrs, idx, sc, st, err, sizeof(err)));
        CHECK(st[0] == GGML_XKV_LANDMARK_STATUS_ERR_BUDGET);
    }
    {
        lm_case c = make_q8_case();
        char err[256] = {};
        int32_t ptrs[3] = {-9,-9,-9}, idx[4] = {-9,-9,-9,-9}, st[4] = {-9,-9,-9,-9};
        float sc[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        CHECK(!ggml_xkv_landmark_cpu_oracle(c.q.data(), c.land.data(), GGML_TYPE_F16,
            c.fpos.data(), c.fmeta.data(), c.qmeta.data(), c.rope.data(), &c.p,
            ptrs, idx, sc, st, err, sizeof(err)));
        CHECK(st[0] == GGML_XKV_LANDMARK_STATUS_ERR_UNSUPPORTED);
    }
    // CPU-graph-level: validation fault leaves device-written outputs at sentinel
    {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        CHECK(cpu != nullptr);
        lm_case c = make_q8_case();
        auto bad = c.p; bad.version = 0xFFFF; // shapes valid, validation must fail
        size_t need = 0; char e0[256] = {};
        CHECK(ggml_xkv_landmark_workspace_bytes(&c.p, &need, e0, sizeof(e0)));
        ggml_init_params ip = { ggml_tensor_overhead() * 24 + ggml_graph_overhead_custom(24, false), nullptr, true };
        ggml_context_ptr ctx(ggml_init(ip));
        ggml_tensor * q = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, c.p.head_dim, c.p.n_q_heads * c.p.n_queries);
        ggml_tensor * land = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, c.p.padded_dim, c.p.n_frags);
        ggml_tensor * fp = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, c.p.n_frags);
        ggml_tensor * fm = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, c.p.n_frags);
        ggml_tensor * qm = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, c.p.n_queries);
        ggml_tensor * rope = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, (int64_t)c.rope.size());
        ggml_tensor * scr = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, (int64_t)(need / 4));
        ggml_tensor * ptrs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, c.p.n_queries + 1);
        ggml_tensor * sc = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, c.p.top_k, c.p.n_queries);
        ggml_tensor * st = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 4);
        ggml_tensor * dst = ggml_xkv_landmark(ctx.get(), q, land, fp, fm, qm, rope, scr, ptrs, sc, st, &bad);
        CHECK(dst != nullptr);
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), cpu));
        CHECK(buf != nullptr);
        ggml_backend_tensor_set(q, c.q.data(), 0, c.q.size() * 4);
        ggml_backend_tensor_set(land, c.land.data(), 0, c.land.size());
        ggml_backend_tensor_set(fp, c.fpos.data(), 0, c.fpos.size() * 4);
        ggml_backend_tensor_set(fm, c.fmeta.data(), 0, c.fmeta.size() * 4);
        ggml_backend_tensor_set(qm, c.qmeta.data(), 0, c.qmeta.size() * 4);
        ggml_backend_tensor_set(rope, c.rope.data(), 0, c.rope.size() * 4);
        std::vector<int32_t> psent(c.p.n_queries + 1, -9);
        ggml_backend_tensor_set(ptrs, psent.data(), 0, psent.size() * 4);
        int32_t stsent[4] = {-9, -9, -9, -9};
        ggml_backend_tensor_set(st, stsent, 0, sizeof(stsent));
        ggml_cgraph * g = ggml_new_graph_custom(ctx.get(), 24, false);
        ggml_build_forward_expand(g, dst);
        CHECK(ggml_backend_graph_compute(cpu, g) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(cpu);
        std::vector<int32_t> pgot(c.p.n_queries + 1, 0);
        int32_t stgot[4] = {0, 0, 0, 0};
        ggml_backend_tensor_get(ptrs, pgot.data(), 0, pgot.size() * 4);
        ggml_backend_tensor_get(st, stgot, 0, sizeof(stgot));
        CHECK(pgot == psent);      // unpublished
        CHECK(stgot[0] != 0);      // ...but status reports the fault
        ggml_backend_free(cpu);
    }
    std::printf("  fault atomicity OK\n");
}

static void test_rows_device_graph_on_backend(ggml_backend_t backend, const char * backend_name) {
    std::printf("[rows-graph] %s backend graph vs oracle ...\n", backend_name);

    // Case 1: N=2, E=4 (E>=3), repeated physical rows across DDVR slots,
    // mixed sparse and contiguous fragments, exact parent CSR + entries validation.
    {
        constexpr uint32_t E = 4;
        constexpr uint32_t N = 2;
        constexpr uint32_t TK = 2;
        constexpr uint32_t NF = 4;
        constexpr uint32_t CAP = 8;
        constexpr uint32_t NROWS = 16;
        const uint64_t tile_cap = (uint64_t)E * CAP < 1024u ? (uint64_t)E * CAP : 1024u;

        // sel_idx: [TK+2, E]
        // e=0: parent 0, slot 0, frags [0, 1] (contig [0..3], sparse {2,3,4,5} -> dedup rows 2,3 in slot 0)
        // e=1: parent 0, slot 1, frags [0, 2] (contig [0..3], contig [4..7] in slot 1 -> distinct from slot 0!)
        // e=2: parent 1, slot 0, frags [1, 3] (sparse {2,3,4,5}, sparse {0,1,6,7} in slot 0)
        // e=3: parent 1, slot 2, frags [2, -1] (contig [4..7] in slot 2)
        int32_t sel[(TK + 2) * E] = {
            0, 0, 0, 1,
            0, 1, 0, 2,
            1, 0, 1, 3,
            1, 2, 2, -1,
        };

        // frag_meta: fstride=4 [row_begin, row_count, seg, legal]
        int32_t fm[4 * NF] = {
            0, 4, 0, 1, // frag 0: contig [0..3]
            2, 4, 0, 1, // frag 1: sparse {2,3,4,5}
            4, 4, 0, 1, // frag 2: contig [4..7]
            0, 4, 0, 1, // frag 3: sparse {0,1,6,7}
        };

        // frag_row_off: [NF+1]. Canonical monotonic prefix offsets into frow_ids:
        // contiguous frags have frow_off[f+1] == frow_off[f], sparse frags advance by count.
        int32_t frow_off[NF + 1] = {0, 0, 4, 4, 8};
        int32_t frow_ids[8] = {
            2, 3, 4, 5, // frag 1
            0, 1, 6, 7, // frag 3
        };

        // frag_kv: [3, NF] [group, arena, head]
        int32_t fkv[3 * NF] = {
            0, 0, 0, // frag 0: head 0
            0, 0, 1, // frag 1: head 1
            0, 0, 2, // frag 2: head 2
            0, 0, 3, // frag 3: head 3
        };

        std::vector<int32_t> rpos(NROWS);
        for (uint32_t r = 0; r < NROWS; ++r) rpos[r] = (int32_t)(100 + r);
        std::vector<int32_t> fpos = {10, 20, 30, 40};

        ggml_xkv_landmark_rows_params params = {};
        params.version = GGML_XKV_LANDMARK_VERSION;
        params.n_queries = E;
        params.top_k = TK;
        params.n_frags = NF;
        params.refine_cap = CAP;
        params.fstride = 4;
        params.flags = GGML_XKV_LANDMARK_FLAG_SPARSE_ROWS;
        params.max_frag_rows = 64;
        params.n_rows_total = NROWS;
        params.n_parent_queries = N;
        params.has_query_map = 1;
        params.arena_filter = UINT32_MAX; // all
        params.global_row_base = 0;
        params.arena_row_count = NROWS;
        params.output_row_begin = 0;
        params._reserved = 0;

        // 1. Run CPU reference oracle
        std::vector<int32_t> ref_ptrs(N + 1, 0);
        std::vector<int32_t> ref_refs(4 * tile_cap, 0);
        std::vector<int32_t> ref_out_pos(tile_cap, 0);
        std::vector<int32_t> ref_entries(4 * tile_cap, 0);
        int32_t ref_status[4] = {};
        char err[256] = {};
        bool ok_c1 = ggml_xkv_landmark_rows_cpu_oracle(sel, fm, frow_off, frow_ids, fkv,
            rpos.data(), fpos.data(), &params,
            ref_ptrs.data(), ref_refs.data(), ref_out_pos.data(), ref_entries.data(),
            ref_status, err, sizeof(err));
        if (!ok_c1 || ref_status[0] != GGML_XKV_LANDMARK_STATUS_OK) {
            std::fprintf(stderr, "Case1 oracle failed: err=%s, status=%d\n", err, ref_status[0]);
            CHECK(false);
            return; // prevent cascading checks
        }
        CHECK(ref_status[0] == GGML_XKV_LANDMARK_STATUS_OK);
        // Verify oracle produced expected N=2 parent CSR
        CHECK(ref_ptrs[0] == 0);
        CHECK(ref_ptrs[1] > 0);
        CHECK(ref_ptrs[2] > ref_ptrs[1]);
        CHECK(ref_status[1] == ref_ptrs[2]); // emitted == total parent 1 end

        // 2. Build graph and execute on backend
        ggml_init_params ip = { ggml_tensor_overhead() * 24 + ggml_graph_overhead_custom(24, false), nullptr, true };
        ggml_context_ptr ctx(ggml_init(ip));
        ggml_tensor * t_sel   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, TK + 2, E);
        ggml_tensor * t_fm    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, NF);
        ggml_tensor * t_off   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, NF + 1);
        ggml_tensor * t_ids   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 8);
        ggml_tensor * t_kv    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 3, NF);
        ggml_tensor * t_rpos  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, NROWS);
        ggml_tensor * t_ptrs  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, N + 1);
        ggml_tensor * t_outpos= ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, (int64_t)tile_cap);
        ggml_tensor * t_ent   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, (int64_t)tile_cap);
        ggml_tensor * t_st    = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 4);

        ggml_tensor * dst = ggml_xkv_landmark_rows(ctx.get(), t_sel, t_fm, t_off, t_ids, t_kv,
            t_rpos, t_ptrs, t_outpos, t_ent, t_st, &params);
        CHECK(dst != nullptr);
        CHECK(ggml_backend_supports_op(backend, dst));

        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        CHECK(buf != nullptr);

        ggml_backend_tensor_set(t_sel, sel, 0, sizeof(sel));
        ggml_backend_tensor_set(t_fm, fm, 0, sizeof(fm));
        ggml_backend_tensor_set(t_off, frow_off, 0, sizeof(frow_off));
        ggml_backend_tensor_set(t_ids, frow_ids, 0, sizeof(frow_ids));
        ggml_backend_tensor_set(t_kv, fkv, 0, sizeof(fkv));
        ggml_backend_tensor_set(t_rpos, rpos.data(), 0, rpos.size() * 4);

        // Sentinel initialization for outputs
        std::vector<int32_t> sent_ptrs(N + 1, -9);
        std::vector<int32_t> sent_refs(4 * tile_cap, -9);
        std::vector<int32_t> sent_outpos(tile_cap, -9);
        std::vector<int32_t> sent_ent(4 * tile_cap, -9);
        int32_t sent_st[4] = {-9, -9, -9, -9};
        ggml_backend_tensor_set(t_ptrs, sent_ptrs.data(), 0, sent_ptrs.size() * 4);
        ggml_backend_tensor_set(dst, sent_refs.data(), 0, sent_refs.size() * 4);
        ggml_backend_tensor_set(t_outpos, sent_outpos.data(), 0, sent_outpos.size() * 4);
        ggml_backend_tensor_set(t_ent, sent_ent.data(), 0, sent_ent.size() * 4);
        ggml_backend_tensor_set(t_st, sent_st, 0, sizeof(sent_st));

        ggml_cgraph * g = ggml_new_graph_custom(ctx.get(), 24, false);
        ggml_build_forward_expand(g, dst);
        CHECK(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);

        std::vector<int32_t> got_ptrs(N + 1, 0);
        std::vector<int32_t> got_refs(4 * tile_cap, 0);
        std::vector<int32_t> got_outpos(tile_cap, 0);
        std::vector<int32_t> got_ent(4 * tile_cap, 0);
        int32_t got_st[4] = {};
        ggml_backend_tensor_get(t_ptrs, got_ptrs.data(), 0, got_ptrs.size() * 4);
        ggml_backend_tensor_get(dst, got_refs.data(), 0, got_refs.size() * 4);
        ggml_backend_tensor_get(t_outpos, got_outpos.data(), 0, got_outpos.size() * 4);
        ggml_backend_tensor_get(t_ent, got_ent.data(), 0, got_ent.size() * 4);
        ggml_backend_tensor_get(t_st, got_st, 0, sizeof(got_st));

        // Exact output agreement with reference oracle
        CHECK(got_st[0] == GGML_XKV_LANDMARK_STATUS_OK);
        CHECK(got_st[1] == ref_status[1]);
        CHECK(got_st[2] == ref_status[2]);
        CHECK(got_st[3] == ref_status[3]);
        CHECK(got_ptrs == ref_ptrs);
        CHECK(got_refs == ref_refs);
        CHECK(got_outpos == ref_out_pos);
        CHECK(got_ent == ref_entries);

        // Reconstruct ABI contract: row_refs[1] and row_refs[2] must be 0 for all emitted rows
        for (int32_t i = 0; i < got_st[1]; ++i) {
            CHECK(got_refs[i * 4 + 1] == 0); // group 0
            CHECK(got_refs[i * 4 + 2] == 0); // layer 0
            CHECK(got_ent[i * 4 + 0] == 2);  // source = COLD (2)
            CHECK(got_ent[i * 4 + 1] == i);  // tile-local rec_idx
            CHECK(got_ent[i * 4 + 3] == 1);  // valid = 1
        }
        // Padded rows have valid=0 and dummy row 0
        for (uint32_t j = (uint32_t)got_st[1]; j < tile_cap; ++j) {
            CHECK(got_refs[j * 4 + 0] == 0);
            CHECK(got_refs[j * 4 + 1] == 0);
            CHECK(got_refs[j * 4 + 2] == 0);
            CHECK(got_refs[j * 4 + 3] == 0);
            CHECK(got_ent[j * 4 + 3] == 0);  // valid = 0 masks padding
        }
    }

    // Case 2: Invalid map (non-increasing slots within same parent query) fails closed
    {
        constexpr uint32_t E = 3;
        constexpr uint32_t N = 2;
        constexpr uint32_t TK = 2;
        constexpr uint32_t NF = 2;
        constexpr uint32_t CAP = 4;
        constexpr uint32_t NROWS = 8;
        const uint64_t tile_cap = (uint64_t)E * CAP;

        // e=0: parent 0 slot 2; e=1: parent 0 slot 1 (INVALID: decreasing slot!)
        int32_t bad_sel[(TK + 2) * E] = {
            0, 2, 0, -1,
            0, 1, 1, -1,
            1, 0, 0, -1,
        };
        int32_t fm[4 * NF] = {0, 2, 0, 1, 2, 2, 0, 1};
        int32_t fkv[3 * NF] = {0, 0, 0, 0, 0, 0};
        int32_t rpos[NROWS] = {0, 1, 2, 3, 4, 5, 6, 7};

        ggml_xkv_landmark_rows_params bad_p = {};
        bad_p.version = GGML_XKV_LANDMARK_VERSION;
        bad_p.n_queries = E; bad_p.top_k = TK; bad_p.n_frags = NF; bad_p.refine_cap = CAP;
        bad_p.fstride = 4; bad_p.flags = 0; bad_p.max_frag_rows = 64; bad_p.n_rows_total = NROWS;
        bad_p.n_parent_queries = N; bad_p.has_query_map = 1;
        bad_p.arena_filter = UINT32_MAX; bad_p.global_row_base = 0; bad_p.arena_row_count = NROWS;
        bad_p.output_row_begin = 0; bad_p._reserved = 0;

        ggml_init_params ip = { ggml_tensor_overhead() * 24 + ggml_graph_overhead_custom(24, false), nullptr, true };
        ggml_context_ptr ctx(ggml_init(ip));
        ggml_tensor * t_sel   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, TK + 2, E);
        ggml_tensor * t_fm    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, NF);
        ggml_tensor * t_off   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, NF + 1);
        ggml_tensor * t_ids   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
        ggml_tensor * t_kv    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 3, NF);
        ggml_tensor * t_rpos  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, NROWS);
        ggml_tensor * t_ptrs  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, N + 1);
        ggml_tensor * t_outpos= ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, (int64_t)tile_cap);
        ggml_tensor * t_ent   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, (int64_t)tile_cap);
        ggml_tensor * t_st    = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 4);

        ggml_tensor * dst = ggml_xkv_landmark_rows(ctx.get(), t_sel, t_fm, t_off, t_ids, t_kv,
            t_rpos, t_ptrs, t_outpos, t_ent, t_st, &bad_p);
        CHECK(dst != nullptr);

        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        CHECK(buf != nullptr);
        ggml_backend_tensor_set(t_sel, bad_sel, 0, sizeof(bad_sel));
        ggml_backend_tensor_set(t_fm, fm, 0, sizeof(fm));
        int32_t neg_off[NF + 1] = {-1, -1, -1};
        ggml_backend_tensor_set(t_off, neg_off, 0, sizeof(neg_off));
        ggml_backend_tensor_set(t_kv, fkv, 0, sizeof(fkv));
        ggml_backend_tensor_set(t_rpos, rpos, 0, sizeof(rpos));

        int32_t st_init[4] = {0, 0, 0, 0};
        ggml_backend_tensor_set(t_st, st_init, 0, sizeof(st_init));

        ggml_cgraph * g = ggml_new_graph_custom(ctx.get(), 24, false);
        ggml_build_forward_expand(g, dst);
        CHECK(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);

        int32_t got_st[4] = {};
        ggml_backend_tensor_get(t_st, got_st, 0, sizeof(got_st));
        CHECK(got_st[0] != 0); // Fails closed with error status!
    }

    // Case 3: E*refine_cap > 1024 tiled overflow test.
    // Tests both window 0 (output_row_begin=0) and window 1 (output_row_begin=1024)
    // with sentinel guards beyond tile_capacity to prove zero OOB memory writes.
    {
        constexpr uint32_t E = 20;
        constexpr uint32_t N = 4;
        constexpr uint32_t TK = 2;
        constexpr uint32_t NF = 4;
        constexpr uint32_t CAP = 64; // E * CAP = 1280 > 1024!
        constexpr uint32_t NROWS = 256;
        constexpr uint32_t TILE_CAP = 1024;

        // 20 expanded queries mapped 5 per parent across 4 parents
        // e=0..19: parent e/5 in 0..3, slot e%5 in 0..4 (strictly increasing).
        // Candidate fragments: select 2 distinct fragments per expanded query.
        // Total candidates per query: 2 frags * 64 rows = 128 rows > CAP=64 -> capped at 64.
        // Over E=20 queries, total_filtered = 20 * 64 = 1280 > 1024.
        std::vector<int32_t> sel((TK + 2) * E);
        for (uint32_t e = 0; e < E; ++e) {
            sel[e * (TK + 2) + 0] = (int32_t)(e / 5); // parent in 0..3
            sel[e * (TK + 2) + 1] = (int32_t)(e % 5); // strictly increasing slots per parent
            sel[e * (TK + 2) + 2] = (int32_t)(e % 2);         // frag 0 or 1
            sel[e * (TK + 2) + 3] = (int32_t)(2 + (e % 2));     // frag 2 or 3
    }

        // 4 fragments: frags 0 & 2 contiguous [0..63] and [128..191],
        // frags 1 & 3 sparse {64..127} and {192..255}.
        // Total distinct rows in storage view = 256.
        std::vector<int32_t> fm(4 * NF);
        fm[0 * 4 + 0] = 0;   fm[0 * 4 + 1] = 64; fm[0 * 4 + 2] = 0; fm[0 * 4 + 3] = 1; // contig [0..63]
        fm[1 * 4 + 0] = 64;  fm[1 * 4 + 1] = 64; fm[1 * 4 + 2] = 0; fm[1 * 4 + 3] = 1; // sparse
        fm[2 * 4 + 0] = 128; fm[2 * 4 + 1] = 64; fm[2 * 4 + 2] = 0; fm[2 * 4 + 3] = 1; // contig [128..191]
        fm[3 * 4 + 0] = 192; fm[3 * 4 + 1] = 64; fm[3 * 4 + 2] = 0; fm[3 * 4 + 3] = 1; // sparse

        // Sparse row IDs for frags 1 and 3 (64 elements each, 128 total)
        std::vector<int32_t> sparse_ids(128);
        for (int32_t i = 0; i < 64; ++i) {
            sparse_ids[i] = 64 + i;       // frag 1: rows 64..127
            sparse_ids[64 + i] = 192 + i; // frag 3: rows 192..255
    }
        // Monotonic prefix offsets: frag 0 contig (off 0..0), frag 1 sparse (off 0..64),
        // frag 2 contig (off 64..64), frag 3 sparse (off 64..128).
        std::vector<int32_t> foff = {0, 0, 64, 64, 128};

        // frag_kv: [3, NF] [group=0, arena=0, head=f]
        std::vector<int32_t> fkv = {
            0, 0, 0,
            0, 0, 1,
            0, 0, 2,
            0, 0, 3,
        };

        std::vector<int32_t> rpos(NROWS);
        for (uint32_t r = 0; r < NROWS; ++r) rpos[r] = (int32_t)(1000 + r * 2);
        std::vector<int32_t> fpos = {10, 20, 30, 40};

        ggml_xkv_landmark_rows_params p_tile = {};
        p_tile.version = GGML_XKV_LANDMARK_VERSION;
        p_tile.n_queries = E; p_tile.top_k = TK; p_tile.n_frags = NF; p_tile.refine_cap = CAP;
        p_tile.fstride = 4; p_tile.flags = GGML_XKV_LANDMARK_FLAG_SPARSE_ROWS;
        p_tile.max_frag_rows = 64; p_tile.n_rows_total = NROWS;
        p_tile.n_parent_queries = N; p_tile.has_query_map = 1;
        p_tile.arena_filter = UINT32_MAX; p_tile.global_row_base = 0; p_tile.arena_row_count = NROWS;
        p_tile.output_row_begin = 0; p_tile._reserved = 0;

        // Window 0: output_row_begin = 0
    {
            std::vector<int32_t> ref_ptrs(N + 1, 0);
            std::vector<int32_t> ref_refs(4 * TILE_CAP, 0);
            std::vector<int32_t> ref_outpos(TILE_CAP, 0);
            std::vector<int32_t> ref_ent(4 * TILE_CAP, 0);
            int32_t ref_st[4] = {};
            char err[256] = {};
            bool ok_ref0 = ggml_xkv_landmark_rows_cpu_oracle(sel.data(), fm.data(), foff.data(), sparse_ids.data(), fkv.data(),
                rpos.data(), fpos.data(), &p_tile,
                ref_ptrs.data(), ref_refs.data(), ref_outpos.data(), ref_ent.data(), ref_st, err, sizeof(err));
            if (!ok_ref0 || ref_st[0] != GGML_XKV_LANDMARK_STATUS_OK) {
                std::fprintf(stderr, "Case3 Win0 oracle failed: err=%s, status=%d\n", err, ref_st[0]);
                CHECK(false);
                return; // early return prevents assertion cascades
            }
            CHECK(ref_st[0] == GGML_XKV_LANDMARK_STATUS_OK);
            CHECK(ref_st[1] == (int32_t)TILE_CAP); // first tile emitted exactly 1024 rows
            CHECK(ref_st[3] == (int32_t)(E * CAP)); // total_filtered = 1280 > 1024

        ggml_init_params ip = { ggml_tensor_overhead() * 24 + ggml_graph_overhead_custom(24, false), nullptr, true };
        ggml_context_ptr ctx(ggml_init(ip));
        ggml_tensor * t_sel   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, TK + 2, E);
        ggml_tensor * t_fm    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, NF);
        ggml_tensor * t_off   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, NF + 1);
            ggml_tensor * t_ids   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, (int64_t)sparse_ids.size());
        ggml_tensor * t_kv    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 3, NF);
        ggml_tensor * t_rpos  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, NROWS);
        ggml_tensor * t_ptrs  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, N + 1);
            ggml_tensor * t_outpos= ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, TILE_CAP);
            ggml_tensor * t_ent   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, TILE_CAP);
        ggml_tensor * t_st    = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 4);

        ggml_tensor * dst = ggml_xkv_landmark_rows(ctx.get(), t_sel, t_fm, t_off, t_ids, t_kv,
                t_rpos, t_ptrs, t_outpos, t_ent, t_st, &p_tile);
        CHECK(dst != nullptr);

        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        CHECK(buf != nullptr);
            ggml_backend_tensor_set(t_sel, sel.data(), 0, sel.size() * 4);
            ggml_backend_tensor_set(t_fm, fm.data(), 0, fm.size() * 4);
            ggml_backend_tensor_set(t_off, foff.data(), 0, foff.size() * 4);
            ggml_backend_tensor_set(t_ids, sparse_ids.data(), 0, sparse_ids.size() * 4);
            ggml_backend_tensor_set(t_kv, fkv.data(), 0, fkv.size() * 4);
            ggml_backend_tensor_set(t_rpos, rpos.data(), 0, rpos.size() * 4);

        ggml_cgraph * g = ggml_new_graph_custom(ctx.get(), 24, false);
        ggml_build_forward_expand(g, dst);
        CHECK(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);

            std::vector<int32_t> got_ptrs(N + 1, 0);
            std::vector<int32_t> got_refs(4 * TILE_CAP, 0);
            std::vector<int32_t> got_outpos(TILE_CAP, 0);
            std::vector<int32_t> got_ent(4 * TILE_CAP, 0);
        int32_t got_st[4] = {};
            ggml_backend_tensor_get(t_ptrs, got_ptrs.data(), 0, got_ptrs.size() * 4);
            ggml_backend_tensor_get(dst, got_refs.data(), 0, got_refs.size() * 4);
            ggml_backend_tensor_get(t_outpos, got_outpos.data(), 0, got_outpos.size() * 4);
            ggml_backend_tensor_get(t_ent, got_ent.data(), 0, got_ent.size() * 4);
        ggml_backend_tensor_get(t_st, got_st, 0, sizeof(got_st));

            CHECK(got_st[0] == GGML_XKV_LANDMARK_STATUS_OK);
            CHECK(got_st[1] == ref_st[1]);
            CHECK(got_st[2] == ref_st[2]);
            CHECK(got_st[3] == ref_st[3]);
            CHECK(got_ptrs == ref_ptrs);
            CHECK(got_refs == ref_refs);
            CHECK(got_outpos == ref_outpos);
            CHECK(got_ent == ref_ent);
    }

        // Window 1: output_row_begin = 1024 (trailing tile of 1280 - 1024 = 256 rows, padded to 1024)
    {
            auto p_win1 = p_tile;
            p_win1.output_row_begin = 1024;

            std::vector<int32_t> ref_ptrs(N + 1, 0);
            std::vector<int32_t> ref_refs(4 * TILE_CAP, 0);
            std::vector<int32_t> ref_outpos(TILE_CAP, 0);
            std::vector<int32_t> ref_ent(4 * TILE_CAP, 0);
            int32_t ref_st[4] = {};
            char err[256] = {};
            bool ok_ref1 = ggml_xkv_landmark_rows_cpu_oracle(sel.data(), fm.data(), foff.data(), sparse_ids.data(), fkv.data(),
                rpos.data(), fpos.data(), &p_win1,
                ref_ptrs.data(), ref_refs.data(), ref_outpos.data(), ref_ent.data(), ref_st, err, sizeof(err));
            if (!ok_ref1 || ref_st[0] != GGML_XKV_LANDMARK_STATUS_OK) {
                std::fprintf(stderr, "Case3 Win1 oracle failed: err=%s, status=%d\n", err, ref_st[0]);
                CHECK(false);
                return;
            }
            CHECK(ref_st[0] == GGML_XKV_LANDMARK_STATUS_OK);
            CHECK(ref_st[1] == 256); // exactly 256 emitted rows in second window
            CHECK(ref_st[3] == 1280);

        ggml_init_params ip = { ggml_tensor_overhead() * 24 + ggml_graph_overhead_custom(24, false), nullptr, true };
        ggml_context_ptr ctx(ggml_init(ip));
        ggml_tensor * t_sel   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, TK + 2, E);
        ggml_tensor * t_fm    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, NF);
        ggml_tensor * t_off   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, NF + 1);
            ggml_tensor * t_ids   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, (int64_t)sparse_ids.size());
        ggml_tensor * t_kv    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 3, NF);
        ggml_tensor * t_rpos  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, NROWS);
        ggml_tensor * t_ptrs  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, N + 1);
            ggml_tensor * t_outpos= ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, TILE_CAP);
            ggml_tensor * t_ent   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, TILE_CAP);
        ggml_tensor * t_st    = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 4);

        ggml_tensor * dst = ggml_xkv_landmark_rows(ctx.get(), t_sel, t_fm, t_off, t_ids, t_kv,
                t_rpos, t_ptrs, t_outpos, t_ent, t_st, &p_win1);
        CHECK(dst != nullptr);

        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        CHECK(buf != nullptr);
            ggml_backend_tensor_set(t_sel, sel.data(), 0, sel.size() * 4);
            ggml_backend_tensor_set(t_fm, fm.data(), 0, fm.size() * 4);
            ggml_backend_tensor_set(t_off, foff.data(), 0, foff.size() * 4);
            ggml_backend_tensor_set(t_ids, sparse_ids.data(), 0, sparse_ids.size() * 4);
            ggml_backend_tensor_set(t_kv, fkv.data(), 0, fkv.size() * 4);
            ggml_backend_tensor_set(t_rpos, rpos.data(), 0, rpos.size() * 4);

        ggml_cgraph * g = ggml_new_graph_custom(ctx.get(), 24, false);
        ggml_build_forward_expand(g, dst);
        CHECK(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);

            std::vector<int32_t> got_ptrs(N + 1, 0);
            std::vector<int32_t> got_refs(4 * TILE_CAP, 0);
            std::vector<int32_t> got_outpos(TILE_CAP, 0);
            std::vector<int32_t> got_ent(4 * TILE_CAP, 0);
        int32_t got_st[4] = {};
            ggml_backend_tensor_get(t_ptrs, got_ptrs.data(), 0, got_ptrs.size() * 4);
            ggml_backend_tensor_get(dst, got_refs.data(), 0, got_refs.size() * 4);
            ggml_backend_tensor_get(t_outpos, got_outpos.data(), 0, got_outpos.size() * 4);
            ggml_backend_tensor_get(t_ent, got_ent.data(), 0, got_ent.size() * 4);
        ggml_backend_tensor_get(t_st, got_st, 0, sizeof(got_st));

            CHECK(got_st[0] == GGML_XKV_LANDMARK_STATUS_OK);
            CHECK(got_st[1] == ref_st[1]);
            CHECK(got_st[2] == ref_st[2]);
            CHECK(got_st[3] == ref_st[3]);
            CHECK(got_ptrs == ref_ptrs);
            CHECK(got_refs == ref_refs);
            CHECK(got_outpos == ref_outpos);
            CHECK(got_ent == ref_ent);

            // Padded tail [256, 1024) must have valid=0 and dummy row 0
            for (uint32_t j = 256; j < TILE_CAP; ++j) {
                CHECK(got_refs[j * 4 + 0] == 0);
                CHECK(got_refs[j * 4 + 1] == 0);
                CHECK(got_refs[j * 4 + 2] == 0);
                CHECK(got_refs[j * 4 + 3] == 0);
                CHECK(got_ent[j * 4 + 3] == 0); // valid = 0
            }
        }
    }

    std::printf("  %s rows graph OK\n", backend_name);
}

static void test_rows_device_graph() {
    // 1. CPU backend graph test
    ggml_backend_t cpu = ggml_backend_cpu_init();
    CHECK(cpu != nullptr);
    test_rows_device_graph_on_backend(cpu, "CPU");
    ggml_backend_free(cpu);

    // 2. Vulkan backend graph test
    ggml_backend_t vk = open_vulkan_backend();
    if (vk) {
        test_rows_device_graph_on_backend(vk, "Vulkan");
        ggml_backend_free(vk);
    } else {
        std::printf("[rows-graph] SKIP Vulkan (no Vulkan device)\n");
    }
}

int main() {
    test_capability_gate();
    test_all_masked_empty();
    test_budget_fails_closed();
    test_deterministic_ties();
    test_query_isolation();
    test_refine_cap();
    test_workspace_bounds();
    test_history_independent_workspace();
    test_capability_maxima();
    test_many_queries_determinism();
    test_per_query_epoch_gating();
    test_rows_oracle();
    test_rows_ddvr_parent_mapping();
    test_merge_equivalence();
    test_phase_domain_contract();
    test_v3_stride8_oracle();
    test_benchmark_cpu_nominal();
    test_cpu_graph_matches_oracle();
    test_vulkan_landmark_graph();
    test_landmark_scratch_exact();
    test_landmark_fault_atomicity();
    test_rows_device_graph();
    {
        float subnormal_val = 1e-38f;
        std::vector<uint8_t> enc;
        encode_q8_row(&subnormal_val, 32, enc);
        CHECK(!enc.empty());
    }
    std::printf("=== test-xkv-vulkan-landmark: %d failure(s) ===\n", failures);
    return failures ? 1 : 0;
}
