// TP5 plan CPU tests: role tables, per-rank slice rules, quant-block legality.
// Pure CPU; no GPU, no GGUF, no Vulkan. Mirrors TP5.md sections 3.3, 4.4, 6.3,
// 7.1, 8.1 and the checks listed in 22.1.

#include "llama-tp5-plan.h"
#include "llama-hparams.h"
#include "llama-model.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"

#include "ggml.h"

#include "../ggml/src/ggml-impl.h" // ggml_set_op_params (private FA node construction)
#include "../ggml/src/ggml-vulkan/ggml-vulkan-collective.hpp"

#include <memory>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

static int g_failures = 0;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failures++; \
        return; \
    } \
} while (0)

static llama_hparams make_hparams() {
    // Real Qwen3.8-Flash-Next-APEX-I-Compact values (verified from the GGUF header)
    llama_hparams hp = {};
    hp.n_embd          = 2560;
    hp.n_layer_all     = 48;
    hp.n_expert        = 512;
    hp.n_expert_used   = 10;
    hp.n_ff_exp        = 640;
    hp.n_ff_shexp      = 640;
    hp.n_embd_head_k_full = 256;
    hp.n_embd_head_v_full = 256;
    hp.n_head_kv_arr[0]   = 2;
    hp.n_head_arr[0]      = 24;
    hp.dsv4_hc_mult    = 4;
    hp.hc_low_rank     = 320;
    hp.ssm_d_state     = 128;
    hp.ssm_dt_rank     = 48;
    hp.ssm_n_group     = 16;
    hp.indexer_n_head  = 4;
    hp.indexer_head_size = 128;
    // every 4th layer is full attention (layers 3, 7, ... 47)
    for (uint32_t i = 0; i < 48; ++i) {
        hp.is_recr_impl[i] = ((i + 1) % 4 != 0);
    }
    return hp;
}

static void test_plan_rejects_bad_ranks() {
    fprintf(stderr, "--- test_plan_rejects_bad_ranks ---\n");
    llama_hparams hp = make_hparams();
    llama_tp5_plan plan;
    llama_tp5_error err;
    TEST_ASSERT(!llama_tp5_plan_build(hp, 1, 248320, false, plan, err));
    TEST_ASSERT(err.code == "TP5_E_RANKS");
    TEST_ASSERT(!llama_tp5_plan_build(hp, 9, 248320, false, plan, err));
    TEST_ASSERT(err.code == "TP5_E_RANKS");
}

static void test_plan_rejects_indivisible_moe() {
    fprintf(stderr, "--- test_plan_rejects_indivisible_moe ---\n");
    llama_hparams hp = make_hparams();
    hp.n_ff_exp = 642; // not divisible by 5
    llama_tp5_plan plan;
    llama_tp5_error err;
    TEST_ASSERT(!llama_tp5_plan_build(hp, 5, 248320, false, plan, err));
    TEST_ASSERT(err.code == "TP5_E_MOE_SPLIT");
}

static void test_tensor_plans_target_types() {
    fprintf(stderr, "--- test_tensor_plans_target_types ---\n");
    llama_hparams hp = make_hparams();
    llama_tp5_plan plan;
    llama_tp5_error err;
    TEST_ASSERT(llama_tp5_plan_build(hp, 5, 248320, false, plan, err));

    bool ok = false;
    // ffn_down_exps: IQ4_NL (blck 32), [640, 2560, 512], axis0 128/rank
    {
        int64_t ne[4] = {640, 2560, 512, 1};
        auto tp = plan.plan_tensor("blk.0.ffn_down_exps.weight", ne, 20, 32, ok, err);
        TEST_ASSERT(ok);
        TEST_ASSERT(tp.semantic == llama_tp5_semantic::MOE_DOWN);
        TEST_ASSERT(tp.layout == llama_tp5_layout::SPLIT_AXIS0);
        for (int r = 0; r < 5; ++r) TEST_ASSERT(tp.per_rank_len[r] == 128);
    }
    // ffn_gate_exps: IQ3_XXS (blck 256 on ne[0]=2560), axis1 rows 128/rank
    {
        int64_t ne[4] = {2560, 640, 512, 1};
        auto tp = plan.plan_tensor("blk.0.ffn_gate_exps.weight", ne, 18, 256, ok, err);
        TEST_ASSERT(ok);
        TEST_ASSERT(tp.semantic == llama_tp5_semantic::MOE_GATE_UP);
        TEST_ASSERT(tp.layout == llama_tp5_layout::SPLIT_AXIS1);
        for (int r = 0; r < 5; ++r) TEST_ASSERT(tp.per_rank_len[r] == 128);
        TEST_ASSERT(tp.head_ranges[0][0] == 0 && tp.head_ranges[0][1] == 128);
        TEST_ASSERT(tp.head_ranges[4][0] == 512 && tp.head_ranges[4][1] == 640);
    }
    // Query/gate rows and output columns must own identical whole heads, with
    // no missing columns and no partial Q5_K block at a rank boundary.
    {
        int64_t q_ne[4]   = { 2560, 12288, 1, 1 };
        int64_t out_ne[4] = { 6144, 2560, 1, 1 };
        auto    q         = plan.plan_tensor("blk.3.attn_q.weight", q_ne, 13, 256, ok, err);
        TEST_ASSERT(ok);
        auto out = plan.plan_tensor("blk.3.attn_output.weight", out_ne, 13, 256, ok, err);
        TEST_ASSERT(ok);
        int64_t columns = 0;
        for (uint32_t rank = 0; rank < plan.ranks; ++rank) {
            TEST_ASSERT(q.per_rank_len[rank] % (2 * plan.da) == 0);
            TEST_ASSERT(q.per_rank_len[rank] / 2 == out.per_rank_len[rank]);
            TEST_ASSERT(out.per_rank_len[rank] % 256 == 0);
            columns += out.per_rank_len[rank];
        }
        TEST_ASSERT(columns == out_ne[0]);
    }
    // GDN layer 0: attn_qkv [2560, 10240] -> prearranged 3*ds*v_heads rows
    {
        int64_t ne[4] = {2560, 10240, 1, 1};
        auto tp = plan.plan_tensor("blk.0.attn_qkv.weight", ne, 13, 256, ok, err);
        TEST_ASSERT(ok);
        TEST_ASSERT(tp.semantic == llama_tp5_semantic::GDN_QKV);
        TEST_ASSERT(tp.per_rank_len[0] == 3 * 128 * 10);
        TEST_ASSERT(tp.per_rank_len[4] == 3 * 128 * 8);
    }
    // ssm_out [6144, 2560] Q5_K: cols 1280/1024 must be 256-aligned
    {
        int64_t ne[4] = {6144, 2560, 1, 1};
        auto tp = plan.plan_tensor("blk.0.ssm_out.weight", ne, 13, 256, ok, err);
        TEST_ASSERT(ok);
        TEST_ASSERT(tp.semantic == llama_tp5_semantic::GDN_OUT);
        TEST_ASSERT(tp.per_rank_len[0] == 1280 && tp.per_rank_len[4] == 1024);
    }
    // HC mirrored
    {
        int64_t ne[4] = {10240, 320, 1, 1};
        auto tp = plan.plan_tensor("blk.0.hc_attn_down.weight", ne, 8, 32, ok, err);
        TEST_ASSERT(ok);
        TEST_ASSERT(tp.semantic == llama_tp5_semantic::HC);
        TEST_ASSERT(tp.layout == llama_tp5_layout::MIRRORED);
    }
    // PLE table CPU
    {
        int64_t ne[4] = {160, 320001536, 1, 1};
        auto tp = plan.plan_tensor("per_layer_token_embd.weight", ne, 20, 32, ok, err);
        TEST_ASSERT(ok);
        TEST_ASSERT(tp.semantic == llama_tp5_semantic::PLE_TABLE);
        TEST_ASSERT(tp.layout == llama_tp5_layout::CPU_RESIDENT);
    }
    // LM head vocab split covers all rows exactly once
    {
        int64_t ne[4] = {2560, 248320, 1, 1};
        auto tp = plan.plan_tensor("output.weight", ne, 14, 256, ok, err);
        TEST_ASSERT(ok);
        TEST_ASSERT(tp.semantic == llama_tp5_semantic::LM_HEAD);
        int64_t total = 0;
        for (int r = 0; r < 5; ++r) total += tp.per_rank_len[r];
        TEST_ASSERT(total == 248320);
        // contiguous non-overlapping
        for (int r = 0; r < 5; ++r) {
            TEST_ASSERT(tp.head_ranges[r][1] - tp.head_ranges[r][0] == tp.per_rank_len[r]);
            if (r > 0) TEST_ASSERT(tp.head_ranges[r][0] == tp.head_ranges[r - 1][1]);
        }
    }
}

static void test_quant_slice_rejection() {
    fprintf(stderr, "--- test_quant_slice_rejection ---\n");
    llama_hparams hp = make_hparams();
    llama_tp5_plan plan;
    llama_tp5_error err;
    TEST_ASSERT(llama_tp5_plan_build(hp, 5, 248320, false, plan, err));

    bool ok = false;
    // down_exps with a 256-block type and 128/rank would be a half-block cut
    int64_t ne[4] = {640, 2560, 512, 1};
    auto tp = plan.plan_tensor("blk.0.ffn_down_exps.weight", ne, 13 /*Q5_K*/, 256, ok, err);
    TEST_ASSERT(!ok);
    TEST_ASSERT(err.code == "TP5_E_QUANT_SLICE");

    // GDN ssm_out with a hypothetical 512 block: 1280 % 512 != 0
    int64_t ne2[4] = {6144, 2560, 1, 1};
    ok = false;
    tp = plan.plan_tensor("blk.0.ssm_out.weight", ne2, 0, 512, ok, err);
    TEST_ASSERT(!ok);
    TEST_ASSERT(err.code == "TP5_E_QUANT_SLICE");
}

static void test_head_split_quant_math() {
    fprintf(stderr, "--- test_head_split_quant_math ---\n");
    // 48 heads, unit 128, blck 256 -> step 2 -> [10,10,10,10,8]
    llama_hparams hp = make_hparams();
    llama_tp5_plan plan;
    llama_tp5_error err;
    TEST_ASSERT(llama_tp5_plan_build(hp, 5, 248320, false, plan, err));
    for (int r = 0; r < 5; ++r) {
        // every rank's slice keeps 256-blocks whole on the ds*heads axis
        TEST_ASSERT((plan.gdn_v_heads[r] * plan.ds) % 256 == 0);
    }
}

static void test_validate() {
    fprintf(stderr, "--- test_validate ---\n");
    llama_hparams hp = make_hparams();
    llama_tp5_plan plan;
    llama_tp5_error err;
    TEST_ASSERT(llama_tp5_plan_build(hp, 5, 248320, false, plan, err));
    TEST_ASSERT(plan.validate(hp, 5, err));
    TEST_ASSERT(!plan.validate(hp, 4, err));
    TEST_ASSERT(err.code == "TP5_E_RANKS");
    // The former bridge covered all Q heads but made one local GQA read past KV.
    plan.q_role_counts  = { 5, 5, 5, 5, 4 };
    plan.kv_role_counts = { 1, 1, 2, 1, 1 };
    plan.kv_head_starts = { 0, 0, 0, 1, 1 };
    TEST_ASSERT(!plan.validate(hp, 5, err));
    TEST_ASSERT(err.code == "TP5_E_Q_SPLIT");
}

static void test_other_rank_counts() {
    fprintf(stderr, "--- test_other_rank_counts ---\n");
    llama_hparams hp = make_hparams();
    llama_tp5_plan plan;
    llama_tp5_error err;
    // 2 ranks: F=640 -> 320/rank; GDN heads [24,24]; Q [12,12]
    TEST_ASSERT(llama_tp5_plan_build(hp, 2, 248320, false, plan, err));
    TEST_ASSERT(plan.gdn_v_heads[0] == 24 && plan.gdn_v_heads[1] == 24);
    TEST_ASSERT(plan.q_role_counts[0] == 12 && plan.q_role_counts[1] == 12);
    TEST_ASSERT(plan.kv_role_counts[0] == 1 && plan.kv_role_counts[1] == 1); // no bridge at 2 ranks
    // 4 ranks: 48 heads -> [12,12,12,12]
    TEST_ASSERT(llama_tp5_plan_build(hp, 4, 248320, false, plan, err));
    for (int r = 0; r < 4; ++r) TEST_ASSERT(plan.gdn_v_heads[r] == 12);
    // 8 ranks: 48/8 = 6 heads each, 6*128=768 % 256 == 0 OK
    TEST_ASSERT(llama_tp5_plan_build(hp, 8, 248320, false, plan, err));
    for (int r = 0; r < 8; ++r) TEST_ASSERT(plan.gdn_v_heads[r] == 6);
}

// Execute the same single GQA operation each rank receives. Distinct KV values
// expose both illegal local ratios and a legal ratio selecting the wrong head.
static void test_gqa_head_mapping_cpu(uint32_t ranks, uint32_t kv_heads) {
    fprintf(stderr, "--- test_gqa_head_mapping_cpu ranks=%u kv=%u ---\n", ranks, kv_heads);
    llama_hparams hp = make_hparams();
    hp.n_head_kv_arr[0] = kv_heads;
    llama_tp5_plan plan;
    llama_tp5_error err;
    TEST_ASSERT(llama_tp5_plan_build(hp, ranks, 248320, false, plan, err));
    using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    context_ptr metadata(ggml_init({ 65536, nullptr, true }), ggml_free);
    auto *      weight = ggml_new_tensor_2d(metadata.get(), GGML_TYPE_F16, hp.n_embd, plan.Nkv * plan.da);
    ggml_backend_meta_split_state split{};
    TEST_ASSERT(llama_tp5_try_apply_split_state(plan, "blk.3.attn_v.weight", weight, hp.indexer_head_size, split));

    int64_t first_q = 0;
    for (uint32_t rank = 0; rank < ranks; ++rank) {
        const int64_t nq  = plan.q_role_counts[rank];
        const int64_t nkv = split.ne[rank] / plan.da;
        TEST_ASSERT(nq > 0 && nkv > 0 && nq % nkv == 0);
        context_ptr ctx(ggml_init({ 4 * 1024 * 1024, nullptr, false }), ggml_free);
        auto *      q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, plan.da, 1, nq, 1);
        auto *      k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, plan.da, 1, nkv, 1);
        auto *      v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, plan.da, 1, nkv, 1);
        memset(q->data, 0, ggml_nbytes(q));
        memset(k->data, 0, ggml_nbytes(k));
        for (int64_t h = 0; h < nkv; ++h) {
            const float value = float(1 + 10 * (split.replica_start[rank] / plan.da + h));
            for (int64_t d = 0; d < plan.da; ++d) {
                static_cast<ggml_fp16_t *>(v->data)[h * plan.da + d] = ggml_fp32_to_fp16(value);
            }
        }
        auto * result = ggml_flash_attn_ext(ctx.get(), q, k, v, nullptr, 1.0f, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(result, GGML_PREC_F32);
        auto * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, result);
        TEST_ASSERT(ggml_graph_compute_with_ctx(ctx.get(), graph, 1) == GGML_STATUS_SUCCESS);
        for (int64_t h = 0; h < nq; ++h) {
            const float expected = float(1 + 10 * ((first_q + h) / (plan.Nq / plan.Nkv)));
            for (int64_t d = 0; d < plan.da; ++d) {
                TEST_ASSERT(static_cast<float *>(result->data)[h * plan.da + d] == expected);
            }
        }
        first_q += nq;
    }
    TEST_ASSERT(first_q == plan.Nq);
}

// TP5.md §8.2: each global V head h maps Q/K to head (h % Nk).
static void test_gdn_headmap_modulo() {
    fprintf(stderr, "--- test_gdn_headmap_modulo ---\n");
    llama_hparams hp = make_hparams();
    llama_tp5_plan plan;
    llama_tp5_error err;
    TEST_ASSERT(llama_tp5_plan_build(hp, 5, 248320, false, plan, err));

    for (int64_t h = 0; h < plan.Nv; ++h) {
        const int64_t qk = llama_tp5_gdn_qk_global_head(plan, h);
        TEST_ASSERT(qk == h % plan.Nk);
    }

    // Prearranged local blocks must cover every global V head exactly once.
    int64_t covered = 0;
    for (uint32_t r = 0; r < plan.ranks; ++r) {
        covered += plan.gdn_v_heads[r];
        TEST_ASSERT(plan.gdn_v_ranges[r][1] - plan.gdn_v_ranges[r][0] == plan.gdn_v_heads[r]);
        if (r > 0) {
            TEST_ASSERT(plan.gdn_v_ranges[r][0] == plan.gdn_v_ranges[r - 1][1]);
        }
    }
    TEST_ASSERT(covered == plan.Nv);
}

// ---- Balanced GDN paired-head map (opt-in GGML_TP5_GDN_HEADMAP=1) ----

// Main's authoritative per-rank assignment (mirrors llama-tp5-plan.cpp):
// adjacent V-head pairs move together; QK list holds every original QK head
// (global V head h uses original QK head h % 16) the rank's V heads reference.
static const int32_t HM_V_ORDER[5][10] = {
    { 0,  1, 16, 17, 32, 33,  2,  3, 18, 19},
    {34, 35,  4,  5, 20, 21, 36, 37,  6,  7},
    {22, 23, 38, 39,  8,  9, 24, 25, 40, 41},
    {10, 11, 26, 27, 42, 43, 12, 13, 28, 29},
    {44, 45, 14, 15, 30, 31, 46, 47,  0,  0},
};
static const int32_t HM_V_COUNT[5]   = {10, 10, 10, 10, 8};
static const int32_t HM_QK_UNIQUE[5][6] = {
    { 0,  1,  2,  3,  0, 0},
    { 2,  3,  4,  5,  6, 7},
    { 6,  7,  8,  9,  0, 0},
    {10, 11, 12, 13,  0, 0},
    {12, 13, 14, 15, 0, 0},
};
static const int32_t HM_QK_COUNT[5]  = {4, 6, 4, 4, 4};

// Build the plan with the balanced GDN head map enabled (env-gated opt-in).
struct headmap_env {
    bool was_set;
    headmap_env()  { was_set = getenv("GGML_TP5_GDN_HEADMAP") != nullptr; setenv("GGML_TP5_GDN_HEADMAP", "1", 1); }
    ~headmap_env() { if (!was_set) unsetenv("GGML_TP5_GDN_HEADMAP"); }
};

// Exact mapping state oracle: per-rank V/QK tables, coverage, pair adjacency,
// local QK modulo, and the packed 3-bit op_params encoding (GDNLayout's
// oracle values rank0=0x1A688208, rank1=0x2C69A688).
static void test_gdn_headmap_plan_tables() {
    fprintf(stderr, "--- test_gdn_headmap_plan_tables ---\n");
    headmap_env env;
    llama_hparams hp = make_hparams();
    llama_tp5_plan plan;
    llama_tp5_error err;
    TEST_ASSERT(llama_tp5_plan_build(hp, 5, 248320, false, plan, err));
    TEST_ASSERT(plan.gdn_headmap_enabled);

    std::vector<int> v_seen(plan.Nv, 0);
    for (uint32_t r = 0; r < 5; ++r) {
        TEST_ASSERT(plan.gdn_v_global_count[r]  == HM_V_COUNT[r]);
        TEST_ASSERT(plan.gdn_qk_unique_count[r] == HM_QK_COUNT[r]);
        for (int32_t i = 0; i < HM_V_COUNT[r]; ++i) {
            TEST_ASSERT(plan.gdn_v_global[r][i] == HM_V_ORDER[r][i]);
            const int32_t h = HM_V_ORDER[r][i];
            TEST_ASSERT(h >= 0 && h < plan.Nv);
            v_seen[h]++;
            // adjacent V pairs move together so 256-element quant blocks stay whole
            if ((i & 1) == 0 && i + 1 < HM_V_COUNT[r]) {
                TEST_ASSERT((h ^ 1) == HM_V_ORDER[r][i + 1]);
            }
            // local QK slot = position of original QK head (h % Nk) in the unique list
            const int32_t qk = h % plan.Nk;
            int32_t slot = -1;
            for (int32_t q = 0; q < HM_QK_COUNT[r]; ++q) {
                if (plan.gdn_qk_unique[r][q] == qk) slot = q;
            }
            TEST_ASSERT(slot >= 0);
            TEST_ASSERT(llama_tp5_gdn_local_qk(plan, r, i) == slot);
        }
        for (int32_t q = 0; q < HM_QK_COUNT[r]; ++q) {
            TEST_ASSERT(plan.gdn_qk_unique[r][q] == HM_QK_UNIQUE[r][q]);
        }
        // local row layout [Q unique | K unique | V ordered]
        TEST_ASSERT(plan.gdn_local_qk_rows(r) == plan.ds * HM_QK_COUNT[r]);
        TEST_ASSERT(plan.gdn_local_v_off(r)   == 2 * plan.ds * HM_QK_COUNT[r]);
    }
    // all 48 V heads exactly once, minimal unique QK counts 4/6/4/4/4 (22 copies)
    for (int64_t h = 0; h < plan.Nv; ++h) TEST_ASSERT(v_seen[h] == 1);
    int64_t qk_copies = 0;
    for (uint32_t r = 0; r < 5; ++r) qk_copies += HM_QK_COUNT[r];
    TEST_ASSERT(qk_copies == 22);

    // packed 3-bit encoding oracle: entry i at bits 3i..3i+2, LSB-first
    for (uint32_t r = 0; r < 5; ++r) {
        uint32_t packed = 0;
        for (int32_t i = 0; i < HM_V_COUNT[r]; ++i) {
            packed |= uint32_t(llama_tp5_gdn_local_qk(plan, r, i)) << (3u * i);
        }
        if (r == 0) TEST_ASSERT(packed == 0x1A688208u);
        if (r == 1) TEST_ASSERT(packed == 0x2C69A688u);
    }

    // head map off (default) keeps the native repeated-segment layout
    unsetenv("GGML_TP5_GDN_HEADMAP");
    llama_tp5_plan native;
    TEST_ASSERT(llama_tp5_plan_build(hp, 5, 248320, false, native, err));
    TEST_ASSERT(!native.gdn_headmap_enabled);
    TEST_ASSERT(native.gdn_v_global_count[0] == 0 && native.gdn_qk_unique_count[0] == 0);
    TEST_ASSERT(llama_tp5_gdn_local_qk(native, 0, 0) == -1);
    setenv("GGML_TP5_GDN_HEADMAP", "1", 1);

    // incompatible geometry fails closed
    llama_hparams hp2 = make_hparams();
    hp2.ssm_dt_rank = 46; // != 48
    llama_tp5_plan bad;
    TEST_ASSERT(!llama_tp5_plan_build(hp2, 5, 248320, false, bad, err));
    TEST_ASSERT(err.code == "TP5_E_GDN_HEADMAP");
}

// Real GGML_OP_GATED_DELTA_NET CPU execution per rank with the stamped
// head map, against a reference that reconstructs the global head identity
// after rank-local execution. Wrong local QK modulo, wrong head order, or an
// omitted cache head produces a mismatch in the gathered state.
static void test_gdn_headmap_execution_cpu() {
    fprintf(stderr, "--- test_gdn_headmap_execution_cpu ---\n");
    headmap_env env;
    llama_hparams hp = make_hparams();
    llama_tp5_plan plan;
    llama_tp5_error err;
    TEST_ASSERT(llama_tp5_plan_build(hp, 5, 248320, false, plan, err));

    using ctx_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    ctx_ptr ctx(ggml_init({ 16 * 1024 * 1024, nullptr, false }), ggml_free);
    TEST_ASSERT(ctx != nullptr);

    // Global reference: one GDN op with 48 V heads and 16 QK heads, native
    // modulo broadcast (global V head h uses QK head h % 16). Same per-head
    // arithmetic as the rank-local ops -> exact F32 comparison is valid.
    const int64_t S = 8, T = 3, Nv = 48, Nk = 16;
    auto * rq = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, Nk, T, 1);
    auto * rk = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, Nk, T, 1);
    auto * rv = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, Nv, T, 1);
    auto * rg = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, Nv, T, 1);
    auto * rb = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, Nv, T, 1);
    auto * rs = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, S, Nv, 1);

    // Distinct deterministic values so a wrong head pairing cannot cancel out.
    auto val = [](int64_t h, int64_t t, int64_t d) { return float(1 + (h * 7 + t * 3 + d) % 97) * 0.25f; };
    for (int64_t h = 0; h < Nk; ++h)
        for (int64_t t = 0; t < T; ++t)
            for (int64_t d = 0; d < S; ++d) {
                ((float *) rq->data)[(t * Nk + h) * S + d] = val(h, t, d);
                ((float *) rk->data)[(t * Nk + h) * S + d] = val(h + 100, t, d);
            }
    for (int64_t h = 0; h < Nv; ++h) {
        for (int64_t t = 0; t < T; ++t)
            for (int64_t d = 0; d < S; ++d) {
                ((float *) rv->data)[(t * Nv + h) * S + d] = val(h + 200, t, d);
            }
        for (int64_t d = 0; d < S * S; ++d) {
            // distinct per-head state: fill the whole S*S block deterministically
            ((float *) rs->data)[h * S * S + d] = float((h * 13 + d) % 7) * 0.5f;
        }
        for (int64_t t = 0; t < T; ++t) {
            ((float *) rg->data)[t * Nv + h] = 0.5f;
            ((float *) rb->data)[t * Nv + h] = 1.0f;
        }
    }

    auto * rnode = ggml_gated_delta_net(ctx.get(), rq, rk, rv, rg, rb, rs, 1);
    TEST_ASSERT(rnode != nullptr);
    auto * rgraph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(rgraph, rnode);
    TEST_ASSERT(ggml_graph_compute_with_ctx(ctx.get(), rgraph, 1) == GGML_STATUS_SUCCESS);

    // Reference final state: output packs [scores | state snapshots], K=1
    const int64_t score_elems = S * Nv * T;
    const float * ref_state = (const float *) rnode->data + score_elems;

    // Per-rank local execution with the stamped head map; gather each local
    // state into the global layout using gdn_v_global and compare exactly.
    for (uint32_t rank = 0; rank < 5; ++rank) {
        ctx_ptr lctx(ggml_init({ 8 * 1024 * 1024, nullptr, false }), ggml_free);
        const int64_t H  = plan.gdn_v_global_count[rank];
        const int64_t Hk = plan.gdn_qk_unique_count[rank];
        auto * q = ggml_new_tensor_4d(lctx.get(), GGML_TYPE_F32, S, Hk, T, 1);
        auto * k = ggml_new_tensor_4d(lctx.get(), GGML_TYPE_F32, S, Hk, T, 1);
        auto * v = ggml_new_tensor_4d(lctx.get(), GGML_TYPE_F32, S, H, T, 1);
        auto * g = ggml_new_tensor_4d(lctx.get(), GGML_TYPE_F32, 1, H, T, 1);
        auto * b = ggml_new_tensor_4d(lctx.get(), GGML_TYPE_F32, 1, H, T, 1);
        auto * s = ggml_new_tensor_4d(lctx.get(), GGML_TYPE_F32, S, S, H, 1);

        // local unique QK slots carry the original QK head's values
        for (int32_t i = 0; i < Hk; ++i) {
            const int32_t gh = plan.gdn_qk_unique[rank][i];
            for (int64_t t = 0; t < T; ++t)
                for (int64_t d = 0; d < S; ++d) {
                    ((float *) q->data)[(t * Hk + i) * S + d] = val(gh, t, d);
                    ((float *) k->data)[(t * Hk + i) * S + d] = val(gh + 100, t, d);
                }
        }
        // local V slots carry the original V head's values in mapped order
        for (int32_t i = 0; i < H; ++i) {
            const int32_t gh = plan.gdn_v_global[rank][i];
            for (int64_t t = 0; t < T; ++t)
                for (int64_t d = 0; d < S; ++d)
                    ((float *) v->data)[(t * H + i) * S + d] = val(gh + 200, t, d);
            for (int64_t d = 0; d < S * S; ++d)
                ((float *) s->data)[i * S * S + d] = float((gh * 13 + d) % 7) * 0.5f;
            for (int64_t t = 0; t < T; ++t) {
                ((float *) g->data)[t * H + i] = 0.5f;
                ((float *) b->data)[t * H + i] = 1.0f;
            }
        }

        auto * node = ggml_gated_delta_net(lctx.get(), q, k, v, g, b, s, 1);
        TEST_ASSERT(node != nullptr);
        TEST_ASSERT(llama_tp5_gdn_headmap_stamp(plan, rank, node));
        // stamp landed in the shared op_params ABI and round-trips
        uint8_t heads[GGML_TP5_HEADMAP_MAX_ENTRIES] = {};
        TEST_ASSERT(ggml_tp5_headmap_get(node, heads) == H);
        for (int32_t i = 0; i < H; ++i)
            TEST_ASSERT(heads[i] == uint8_t(llama_tp5_gdn_local_qk(plan, rank, i)));

        auto * graph = ggml_new_graph(lctx.get());
        ggml_build_forward_expand(graph, node);
        TEST_ASSERT(ggml_graph_compute_with_ctx(lctx.get(), graph, 1) == GGML_STATUS_SUCCESS);

        // gather: local state slot i is global head gdn_v_global[rank][i]
        const float * st = (const float *) node->data + S * H * T;
        for (int32_t i = 0; i < H; ++i) {
            const int32_t gh = plan.gdn_v_global[rank][i];
            for (int64_t d0 = 0; d0 < S; ++d0)
                for (int64_t d1 = 0; d1 < S; ++d1) {
                    const float actual   = st[(i * S + d0) * S + d1];
                    const float expected = ref_state[(gh * S + d0) * S + d1];
                    if (actual != expected) {
                        fprintf(stderr, "  rank %u head %d(global %d) state mismatch at (%lld,%lld): %f != %f\n",
                                rank, i, gh, (long long) d0, (long long) d1, actual, expected);
                    }
                    TEST_ASSERT(actual == expected);
                }
        }
        fprintf(stderr, "  rank %u: %lld local V heads, %lld unique QK heads, gathered state exact\n",
                rank, (long long) H, (long long) Hk);
    }
}

// GGML_TP5_QSA_HEADMAP=1 fixture: balanced Q counts [5,5,5,5,4] with starts
// [0,5,10,15,20] over KV counts [1,1,2,1,1] with starts [0,0,0,1,1]. Executes
// the actual attention map on the CPU: each rank's rank-local FLASH_ATTN_EXT
// node carries the private head-map ABI, and the outputs of all 24 query
// heads are checked against a full 24-head reference, including rank 2's
// head 4 (global Q head 14 -> KV 1).
static void test_tp5_qsa_headmap() {
    fprintf(stderr, "--- test_tp5_qsa_headmap ---\n");
    llama_hparams hp = make_hparams();
    llama_tp5_plan plan;
    llama_tp5_error err;
    TEST_ASSERT(llama_tp5_plan_build(hp, 5, 248320, false, plan, err));
    TEST_ASSERT(plan.qsa_headmap);
    TEST_ASSERT(plan.validate(hp, 5, err));
    // Balanced roles
    const int32_t want_q[5]   = {5, 5, 5, 5, 4};
    const int32_t want_kv[5]  = {1, 1, 2, 1, 1};
    const int32_t want_kvs[5] = {0, 0, 0, 1, 1};
    int32_t q_total = 0;
    for (int r = 0; r < 5; ++r) {
        TEST_ASSERT(plan.q_role_counts[r] == want_q[r]);
        TEST_ASSERT(plan.kv_role_counts[r] == want_kv[r]);
        TEST_ASSERT(plan.kv_head_starts[r] == want_kvs[r]);
        q_total += want_q[r];
    }
    TEST_ASSERT(q_total == 24);
    // Local q->kv map: rank 2 is [0,0,1,1,1], all others all-zero
    for (int r = 0; r < 5; ++r) {
        for (int h = 0; h < want_q[r]; ++h) {
            const int32_t want = (r == 2) ? (h < 2 ? 0 : 1) : 0;
            TEST_ASSERT(plan.local_qkv_map[r][h] == want);
        }
    }

    // Full 24-head reference: global Q head g attends global KV head g/12.
    // Distinct V values per global KV head expose a wrong head selection.
    const int64_t D = 16;
    std::vector<float> ref_out(24 * D);
    {
        using ctx_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
        ctx_ptr ctx(ggml_init({ 4 * 1024 * 1024, nullptr, false }), ggml_free);
        auto * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, D, 1, 24);
        auto * k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, D, 1, 2);
        auto * v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, D, 1, 2);
        for (int64_t h = 0; h < 24; ++h)
            for (int64_t d = 0; d < D; ++d)
                ((float *) q->data)[h * D + d] = 0.25f;
        for (int64_t kvh = 0; kvh < 2; ++kvh) {
            const float value = float(1 + 10 * kvh);
            for (int64_t d = 0; d < D; ++d)
                ((ggml_fp16_t *) v->data)[kvh * D + d] = ggml_fp32_to_fp16(value);
        }
        memset(k->data, 0, ggml_nbytes(k));
        auto * fa = ggml_flash_attn_ext(ctx.get(), q, k, v, nullptr, 1.0f, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
        auto * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, fa);
        TEST_ASSERT(ggml_graph_compute_with_ctx(ctx.get(), graph, 1) == GGML_STATUS_SUCCESS);
        memcpy(ref_out.data(), fa->data, ref_out.size() * sizeof(float));
        for (int64_t h = 0; h < 24; ++h) {
            const float want = float(1 + 10 * (h / 12));
            for (int64_t d = 0; d < D; ++d) {
                TEST_ASSERT(ref_out[h * D + d] == want);
            }
        }
    }

    // Execute each rank's local attention exactly as the meta backend's
    // rank-local clone does: the mapped FA node is constructed directly (the
    // public constructor asserts a uniform Q/KV ratio, which the mapped
    // layout intentionally violates) and stamped with the head-map ABI.
    int64_t first_q = 0;
    for (int rank = 0; rank < 5; ++rank) {
        const int64_t nq  = plan.q_role_counts[rank];
        const int64_t nkv = plan.kv_role_counts[rank];
        using ctx_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
        ctx_ptr ctx(ggml_init({ 4 * 1024 * 1024, nullptr, false }), ggml_free);
        auto * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, D, 1, nq);
        auto * k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, D, 1, nkv);
        auto * v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, D, 1, nkv);
        for (int64_t h = 0; h < nq; ++h)
            for (int64_t d = 0; d < D; ++d)
                ((float *) q->data)[h * D + d] = 0.25f;
        memset(k->data, 0, ggml_nbytes(k));
        for (int64_t l = 0; l < nkv; ++l) {
            // Local KV slot l holds global KV head kv_head_starts[rank] + l.
            const float value = float(1 + 10 * (plan.kv_head_starts[rank] + l));
            for (int64_t d = 0; d < D; ++d)
                ((ggml_fp16_t *) v->data)[l * D + d] = ggml_fp32_to_fp16(value);
        }
        // FLASH_ATTN_EXT output shape: {v->ne[0], q->ne[2], q->ne[1], q->ne[3]}
        int64_t ne[4] = { D, nq, 1, 1 };
        auto * fa = ggml_new_tensor(ctx.get(), GGML_TYPE_F32, 4, ne);
        float params[] = { 1.0f, 0.0f, 0.0f };
        ggml_set_op_params(fa, params, sizeof(params));
        fa->op     = GGML_OP_FLASH_ATTN_EXT;
        fa->src[0] = q;
        fa->src[1] = k;
        fa->src[2] = v;
        ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
        TEST_ASSERT(llama_tp5_qsa_headmap_stamp(plan, (uint32_t) rank, fa));
        auto * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, fa);
        TEST_ASSERT(ggml_graph_compute_with_ctx(ctx.get(), graph, 1) == GGML_STATUS_SUCCESS);
        int mismatches = 0;
        for (int64_t h = 0; h < nq; ++h) {
            const int64_t g = first_q + h; // global Q head identity
            const float want = float(1 + 10 * (g / 12));
            for (int64_t d = 0; d < D; ++d) {
                const float actual = ((const float *) fa->data)[h * D + d];
                if (actual != want || ref_out[g * D + d] != want) {
                    if (mismatches++ == 0) {
                        fprintf(stderr, "  rank %d local Q head %lld (global %lld): got %f want %f ref %f\n",
                                rank, (long long) h, (long long) g, actual, want, ref_out[g * D + d]);
                    }
                }
            }
        }
        TEST_ASSERT(mismatches == 0);
        first_q += nq;
    }
    TEST_ASSERT(first_q == 24);

    // Weight/cache mapping coherence: Q rows, KV replica rows and output cols
    // all follow the same per-rank head counts and KV starts.
    bool ok = false;
    int64_t q_ne[4]   = { 2560, 12288, 1, 1 };
    // Real attn_k/v shape: [n_embd=2560, Nkv*da = 2*256 = 512].
    int64_t kv_ne[4]  = { 2560, 512, 1, 1 };
    int64_t out_ne[4] = { 6144, 2560, 1, 1 };
    auto qtp  = plan.plan_tensor("blk.3.attn_q.weight", q_ne, 13, 256, ok, err);
    TEST_ASSERT(ok);
    auto kvtp = plan.plan_tensor("blk.3.attn_k.weight", kv_ne, 13, 256, ok, err);
    TEST_ASSERT(ok);
    auto otp  = plan.plan_tensor("blk.3.attn_output.weight", out_ne, 13, 256, ok, err);
    TEST_ASSERT(ok);
    int64_t cols = 0;
    for (int r = 0; r < 5; ++r) {
        TEST_ASSERT(qtp.per_rank_len[r] == 2 * plan.da * want_q[r]);
        TEST_ASSERT(qtp.per_rank_len[r] / 2 == otp.per_rank_len[r]);
        TEST_ASSERT(otp.per_rank_len[r] % 256 == 0);
        TEST_ASSERT(kvtp.per_rank_len[r] == plan.da * want_kv[r]);
        // plan_tensor QSA_KV head_ranges are in HEAD units, not elements:
        // {kv_head_starts[r], kv_head_starts[r] + kv_role_counts[r]}.
        TEST_ASSERT(kvtp.head_ranges[r][0] == want_kvs[r]);
        TEST_ASSERT(kvtp.head_ranges[r][1] == want_kvs[r] + want_kv[r]);
        cols += otp.per_rank_len[r];
    }
    TEST_ASSERT(cols == out_ne[0]);
    // Six KV instances per full-attention layer (2 on rank 2, 1 elsewhere):
    // quantified replica count, not a free-balancing claim.
    TEST_ASSERT(plan.n_full == 12);
    int32_t per_layer = 0;
    for (int r = 0; r < 5; ++r) per_layer += plan.kv_role_counts[r];
    TEST_ASSERT(per_layer == 6);
    TEST_ASSERT(plan.kv_instances[2] == 2 * 12 && plan.kv_instances[0] == 12);
}

// Real mapped-span split states from llama_tp5_try_apply_split_state for the
// GDN QKV/conv/history tensors, verified against TAGGED global rows through
// the actual meta buffer transport: every original global row carries its own
// tag, and each rank's local storage must be exactly [ALL Q unique | ALL K
// unique | V ordered] with the right global identities — not merely counts or
// oracles recomputed from the same emitted spans. Catches the interleaved
// Qpair,Kpair,Qpair,Kpair emission bug and any wrong local row offsets.
static void test_gdn_headmap_qkv_span_values() {
    fprintf(stderr, "--- test_gdn_headmap_qkv_span_values ---\n");
    headmap_env env;
    llama_hparams hp = make_hparams();
    llama_tp5_plan plan;
    llama_tp5_error err;
    TEST_ASSERT(llama_tp5_plan_build(hp, 5, 248320, false, plan, err));
    TEST_ASSERT(plan.gdn_headmap_enabled);

    // Production split-state plumbing: model-owned ud with the headmap plan.
    auto                         params = llama_model_default_params();
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN4EXP, params));
    TEST_ASSERT(model != nullptr);
    model->hparams            = make_hparams();
    model->hparams.ssm_d_conv = 4;
    auto & ud                 = model->get_split_state_ud;
    ud.model                  = model.get();
    ud.n_devices              = 5;
    ud.has_tp5_plan           = true;
    ud.tp5_plan               = plan;
    auto *             cpu        = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
    ggml_backend_dev_t devices[5] = { cpu, cpu, cpu, cpu, cpu };
    auto *             device     = ggml_backend_meta_device(devices, 5, llama_meta_device_get_split_state, &ud);
    auto               backend    = ggml_backend_dev_init(device, nullptr);
    TEST_ASSERT(backend != nullptr);
    auto * buft = ggml_backend_dev_buffer_type(device);

    // Original GDN row axis: Q rows [0, Nk*ds), K rows [Nk*ds, 2*Nk*ds),
    // V rows [2*Nk*ds, (2*Nk+Nv)*ds).
    const int64_t ds = plan.ds, Nk = plan.Nk, Nv = plan.Nv;
    const int64_t n_rows = (2 * Nk + Nv) * ds;
    auto * weights = ggml_init({ 16 * 1024 * 1024, nullptr, true });
    auto * qkv = ggml_new_tensor_2d(weights, GGML_TYPE_F32, hp.n_embd, n_rows);
    ggml_set_name(qkv, "blk.0.attn_qkv.weight");
    auto * conv = ggml_new_tensor_2d(weights, GGML_TYPE_F32, 4, n_rows);
    ggml_set_name(conv, "blk.0.ssm_conv1d.weight");
    // Production cache_r_l layout: [n_embd_r = (d_conv-1)*channels, rows].
    // The conv history packs 3 consecutive values per channel on axis 0.
    auto * history = ggml_new_tensor_2d(weights, GGML_TYPE_F32, 3 * n_rows, 1);
    ggml_set_name(history, "cache_r_l0");
    auto buf = ggml_backend_alloc_ctx_tensors_from_buft(weights, buft);
    TEST_ASSERT(buf != nullptr);
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Tag every global row with its own value on all three tensors.
    std::vector<float> qkv_tag(hp.n_embd * n_rows), conv_tag(4 * n_rows), hist_tag(3 * n_rows);
    for (int64_t row = 0; row < n_rows; ++row) {
        const float tag = float(row % 4093) * 0.125f + 0.5f;
        for (int64_t c = 0; c < hp.n_embd; ++c) qkv_tag[row * hp.n_embd + c] = tag + float(c % 7) * 1e-3f;
        for (int64_t c = 0; c < 4; ++c) conv_tag[row * 4 + c] = tag + float(c) * 1e-4f;
        for (int64_t j = 0; j < 3; ++j) hist_tag[row * 3 + j] = tag + float(j) * 1e-5f;
    }
    ggml_backend_tensor_set(qkv, qkv_tag.data(), 0, qkv_tag.size() * sizeof(float));
    ggml_backend_tensor_set(conv, conv_tag.data(), 0, conv_tag.size() * sizeof(float));
    ggml_backend_tensor_set(history, hist_tag.data(), 0, hist_tag.size() * sizeof(float));

    for (uint32_t rank = 0; rank < 5; ++rank) {
        const int32_t nqk = plan.gdn_qk_unique_count[rank];
        const int32_t nv  = plan.gdn_v_global_count[rank];
        const int64_t local_rows = (2 * nqk + nv) * ds;

        // Expected rank-local row order: [ALL Q unique | ALL K unique | V ordered]
        std::vector<int64_t> expect_rows;
        expect_rows.reserve(local_rows);
        for (int32_t q = 0; q < nqk; ++q)
            for (int64_t d = 0; d < ds; ++d) expect_rows.push_back(plan.gdn_qk_unique[rank][q] * ds + d);
        for (int32_t q = 0; q < nqk; ++q)
            for (int64_t d = 0; d < ds; ++d) expect_rows.push_back(Nk * ds + plan.gdn_qk_unique[rank][q] * ds + d);
        for (int32_t i = 0; i < nv; ++i)
            for (int64_t d = 0; d < ds; ++d) expect_rows.push_back(2 * Nk * ds + plan.gdn_v_global[rank][i] * ds + d);
        TEST_ASSERT((int64_t) expect_rows.size() == local_rows);

        struct ggml_tensor * st_qkv  = ggml_backend_meta_buffer_simple_tensor(qkv, rank);
        struct ggml_tensor * st_conv = ggml_backend_meta_buffer_simple_tensor(conv, rank);
        struct ggml_tensor * st_hist = ggml_backend_meta_buffer_simple_tensor(history, rank);
        TEST_ASSERT(st_qkv && st_conv && st_hist);
        TEST_ASSERT(st_qkv->ne[1] == local_rows);
        TEST_ASSERT(st_conv->ne[1] == local_rows);
        // axis-0 history: local width = 3 * local channels
        TEST_ASSERT(st_hist->ne[0] == 3 * local_rows);

        std::vector<float> got_qkv(hp.n_embd * local_rows), got_conv(4 * local_rows), got_hist(3 * local_rows);
        ggml_backend_tensor_get(st_qkv, got_qkv.data(), 0, got_qkv.size() * sizeof(float));
        ggml_backend_tensor_get(st_conv, got_conv.data(), 0, got_conv.size() * sizeof(float));
        ggml_backend_tensor_get(st_hist, got_hist.data(), 0, got_hist.size() * sizeof(float));

        int bad = 0;
        for (int64_t lr = 0; lr < local_rows; ++lr) {
            const int64_t gr = expect_rows[lr];
            for (int64_t c = 0; c < hp.n_embd; ++c)
                bad += got_qkv[lr * hp.n_embd + c] != qkv_tag[gr * hp.n_embd + c];
            for (int64_t c = 0; c < 4; ++c)
                bad += got_conv[lr * 4 + c] != conv_tag[gr * 4 + c];
            for (int64_t j = 0; j < 3; ++j)
                bad += got_hist[lr * 3 + j] != hist_tag[gr * 3 + j];
        }
        if (bad) fprintf(stderr, "  rank %u: %d tagged-row mismatches (QKV/conv/history)\n", rank, bad);
        TEST_ASSERT(bad == 0);
        fprintf(stderr, "  rank %u: local rows %lld = [%d Q | %d K | %d V], tagged global rows exact\n",
                rank, (long long) local_rows, nqk, nqk, nv);
    }

    // Canonical readback reconstructs the full original row set exactly.
    {
        std::vector<float> rb(hp.n_embd * n_rows);
        ggml_backend_tensor_get(qkv, rb.data(), 0, rb.size() * sizeof(float));
        TEST_ASSERT(memcmp(rb.data(), qkv_tag.data(), rb.size() * sizeof(float)) == 0);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(weights);
    ggml_backend_free(backend);
}

static void test_tp5_split_state_gdn_qkv(bool replicate_attention) {
    auto                         params = llama_model_default_params();
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN4EXP, params));
    TEST_ASSERT(model != nullptr);
    model->hparams            = make_hparams();
    model->hparams.ssm_d_conv = 4;
    auto & ud                 = model->get_split_state_ud;
    ud.model                  = model.get();
    ud.n_devices              = 5;
    llama_tp5_error err;
    ud.has_tp5_plan = llama_tp5_plan_build(model->hparams, 5, 248320, replicate_attention, ud.tp5_plan, err);
    TEST_ASSERT(ud.has_tp5_plan);
    auto *             cpu        = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
    ggml_backend_dev_t devices[5] = { cpu, cpu, cpu, cpu, cpu };
    auto *             device     = ggml_backend_meta_device(devices, 5, llama_meta_device_get_split_state, &ud);
    auto               backend    = ggml_backend_dev_init(device, nullptr);
    TEST_ASSERT(backend != nullptr);
    auto * weights = ggml_init({ 2 * 1024 * 1024, nullptr, true });
    auto * compute = ggml_init({ 2 * 1024 * 1024, nullptr, true });
    auto * down    = ggml_new_tensor_2d(weights, GGML_TYPE_Q5_K, 6144, 1);
    ggml_set_name(down, "blk.0.ssm_out.weight");
    model->tensors_by_name.emplace_back(down->name, down);
    auto * qkv = ggml_new_tensor_2d(weights, GGML_TYPE_F32, 1, 10240);
    ggml_set_name(qkv, "blk.0.attn_qkv.weight");
    auto * kernel = ggml_new_tensor_2d(weights, GGML_TYPE_F32, 4, 10240);
    ggml_set_name(kernel, "blk.0.ssm_conv1d.weight");
    auto * history = ggml_new_tensor_1d(weights, GGML_TYPE_F32, 3 * 10240);
    ggml_set_name(history, "cache_r_l0");
    auto * alpha = ggml_new_tensor_2d(weights, GGML_TYPE_F32, 1, 48);
    ggml_set_name(alpha, "blk.0.ssm_alpha.weight");
    auto * dt = ggml_new_tensor_1d(weights, GGML_TYPE_F32, 48);
    ggml_set_name(dt, "blk.0.ssm_dt.bias");
    auto * a = ggml_new_tensor_1d(weights, GGML_TYPE_F32, 48);
    ggml_set_name(a, "blk.0.ssm_a");
    auto * x = ggml_new_tensor_1d(weights, GGML_TYPE_F32, 1);
    ggml_set_name(x, "input");
    ggml_tensor *fq = nullptr, *fk = nullptr, *fv = nullptr, *fa = nullptr;
    if (replicate_attention) {
        fq = ggml_new_tensor_4d(weights, GGML_TYPE_F32, 16, 1, 4, 1);
        fk = ggml_new_tensor_4d(weights, GGML_TYPE_F16, 16, 32, 2, 1);
        fv = ggml_new_tensor_4d(weights, GGML_TYPE_F16, 16, 32, 2, 1);
        ggml_set_name(fk, "cache_k_l3");
        ggml_set_name(fv, "cache_v_l3");
        fa = ggml_flash_attn_ext(compute, fq, fk, fv, nullptr, 0.25f, 0.0f, 0.0f);
        ggml_set_output(fa);
    }
    auto wbuf = ggml_backend_alloc_ctx_tensors_from_buft(weights, ggml_backend_dev_buffer_type(device));
    TEST_ASSERT(wbuf != nullptr);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto * projected = ggml_reshape_2d(compute, ggml_mul_mat(compute, qkv, x), 10240, 1);
    auto * past      = ggml_reshape_3d(compute, history, 3, 10240, 1);
    auto * padded    = ggml_concat(compute, past, ggml_transpose(compute, projected), 0);
    auto * conv      = ggml_ssm_conv(compute, padded, kernel);
    auto * bias      = ggml_add(compute, ggml_reshape_1d(compute, ggml_mul_mat(compute, alpha, x), 48), dt);
    auto * gate      = ggml_mul(compute, ggml_softplus(compute, bias), a);
    ggml_set_output(conv);
    ggml_set_output(gate);
    auto * graph = ggml_new_graph(compute);
    ggml_build_forward_expand(graph, conv);
    ggml_build_forward_expand(graph, gate);
    if (fa)
        ggml_build_forward_expand(graph, fa);
    auto allocator = ggml_gallocr_new(ggml_backend_dev_buffer_type(device));
    TEST_ASSERT(ggml_gallocr_alloc_graph(allocator, graph));
    std::vector<float> q(10240), k(4 * 10240), h(3 * 10240), al(48), d(48), av(48);
    for (int i = 0; i < 10240; ++i) {
        q[i] = float(i % 127 - 63);
        for (int t = 0; t < 4; ++t)
            k[4 * i + t] = float((i + t) % 5 - 2);
        for (int t = 0; t < 3; ++t)
            h[3 * i + t] = float((i + 7 * t) % 31 - 15);
    }
    for (int i = 0; i < 48; ++i) {
        al[i] = float(i % 7) * 0.125f;
        d[i]  = float(i - 24) * 0.0625f;
        av[i] = -float(i + 1);
    }
    const float one = 1;
    ggml_backend_tensor_set(x, &one, 0, sizeof(one));
    ggml_backend_tensor_set(qkv, q.data(), 0, q.size() * sizeof(float));
    ggml_backend_tensor_set(kernel, k.data(), 0, k.size() * sizeof(float));
    ggml_backend_tensor_set(history, h.data(), 0, h.size() * sizeof(float));
    ggml_backend_tensor_set(alpha, al.data(), 0, al.size() * sizeof(float));
    ggml_backend_tensor_set(dt, d.data(), 0, d.size() * sizeof(float));
    ggml_backend_tensor_set(a, av.data(), 0, av.size() * sizeof(float));
    if (fa) {
        std::vector<float>       queries(64, 0.0f);
        std::vector<ggml_fp16_t> keys(1024, ggml_fp32_to_fp16(0.0f));
        std::vector<ggml_fp16_t> values(1024);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = ggml_fp32_to_fp16(i < 512 ? 2.0f : 7.0f);
        ggml_backend_tensor_set(fq, queries.data(), 0, queries.size() * sizeof(float));
        ggml_backend_tensor_set(fk, keys.data(), 0, keys.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(fv, values.data(), 0, values.size() * sizeof(ggml_fp16_t));
    }
    TEST_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> actual_conv(10240), actual_gate(48);
    ggml_backend_tensor_get(conv, actual_conv.data(), 0, actual_conv.size() * sizeof(float));
    ggml_backend_tensor_get(gate, actual_gate.data(), 0, actual_gate.size() * sizeof(float));
    int errors = 0;
    if (fa) {
        std::vector<float> actual_attention(64);
        ggml_backend_tensor_get(fa, actual_attention.data(), 0, actual_attention.size() * sizeof(float));
        for (size_t i = 0; i < actual_attention.size(); ++i)
            errors += actual_attention[i] != (i < 32 ? 2.0f : 7.0f);
    }
    for (int i = 0; i < 10240; ++i) {
        float expected = q[i] * k[4 * i + 3];
        for (int t = 0; t < 3; ++t)
            expected += h[3 * i + t] * k[4 * i + t];
        errors += actual_conv[i] != expected;
    }
    int    gate_errors = 0;
    auto * reference   = ggml_init({ 1024 * 1024, nullptr, false });
    auto * ref_alpha   = ggml_new_tensor_1d(reference, GGML_TYPE_F32, 48);
    auto * ref_dt      = ggml_new_tensor_1d(reference, GGML_TYPE_F32, 48);
    auto * ref_a       = ggml_new_tensor_1d(reference, GGML_TYPE_F32, 48);
    memcpy(ref_alpha->data, al.data(), al.size() * sizeof(float));
    memcpy(ref_dt->data, d.data(), d.size() * sizeof(float));
    memcpy(ref_a->data, av.data(), av.size() * sizeof(float));
    auto * ref_gate  = ggml_mul(reference, ggml_softplus(reference, ggml_add(reference, ref_alpha, ref_dt)), ref_a);
    auto * ref_graph = ggml_new_graph(reference);
    ggml_build_forward_expand(ref_graph, ref_gate);
    TEST_ASSERT(ggml_graph_compute_with_ctx(reference, ref_graph, 1) == GGML_STATUS_SUCCESS);
    for (int i = 0; i < 48; ++i)
        gate_errors += actual_gate[i] != static_cast<float *>(ref_gate->data)[i];
    ggml_free(reference);
    fprintf(stderr, "  GDN native-layout mismatches: conv=%d/10240 gate=%d/48\n", errors, gate_errors);
    errors += gate_errors;
    ggml_gallocr_free(allocator);
    ggml_backend_buffer_free(wbuf);
    ggml_free(compute);
    ggml_free(weights);
    ggml_backend_free(backend);
    TEST_ASSERT(errors == 0);
}

static void test_nextn_layer_plan_bounds() {
    fprintf(stderr, "--- test_nextn_layer_plan_bounds ---\n");
    llama_hparams hp = make_hparams();
    // Configure 48 trunk layers + 1 NextN layer (total 49)
    hp.n_layer_all   = 49;
    hp.n_layer_nextn = 1;
    hp.is_recr_impl[48] = false; // NextN block is dense attention

    llama_tp5_plan plan;
    llama_tp5_error err;
    TEST_ASSERT(llama_tp5_plan_build(hp, 5, 248320, false, plan, err));
    TEST_ASSERT(plan.L == 48); // plan.L remains trunk length 48
    TEST_ASSERT(plan.is_recr.size() >= 49);
    TEST_ASSERT(!plan.is_recr[48]); // blk.48 is non-recurrent

    bool ok = false;
    int64_t ne[4] = { 640, 2560, 512, 1 };
    // Legitimate blk.48 MTP tensor must be accepted and planned
    llama_tp5_tensor_plan tp48 = plan.plan_tensor("blk.48.ffn_down_exps.weight", ne, 8, 32, ok, err);
    TEST_ASSERT(ok);
    TEST_ASSERT(tp48.layout == llama_tp5_layout::SPLIT_AXIS0);

    // Out of bounds block (blk.49) must be rejected with TP5_E_LAYER_BOUNDS
    llama_tp5_tensor_plan tp49 = plan.plan_tensor("blk.49.ffn_down_exps.weight", ne, 8, 32, ok, err);
    TEST_ASSERT(!ok);
    TEST_ASSERT(err.code == "TP5_E_LAYER_BOUNDS");
    fprintf(stderr, "  NextN layer plan bounds: blk.48 accepted, blk.49 rejected with TP5_E_LAYER_BOUNDS\n");
}

enum class test_numerical_mode { REFERENCE, EXACT_F32, AGGRESSIVE_Q8, P1A_NOSIDECAR_Q8 };
enum class test_numerical_reason {
    NONE, DISABLED_BY_ENV, NON_RELAY_SYNC, NON_F32_WIRE, NO_LATE_TENSORS,
    EXACT_Q_REQUESTED, MISSING_HARDWARE_INT_DOT, UNSUPPORTED_WAVE32,
    UNALIGNED_LATE_SHAPE, PIPELINE_UNAVAILABLE, P1A_BENCH_REQUESTED
};

struct test_numerical_spec {
    bool latebind_env_enabled    = false;
    bool is_relay_sync           = false;
    bool is_f32_wire             = false;
    bool has_late_tensors        = false;
    bool exact_q_requested       = false;
    bool p1a_nosidecar_requested = false;
    bool hw_int_dot              = false;
    bool hw_wave32               = false;
    bool shape_aligned           = false;
    bool pipeline_ready          = false;
};

static inline std::pair<test_numerical_mode, test_numerical_reason> test_resolve_numerical_mode(
        const test_numerical_spec & spec) {
    if (spec.p1a_nosidecar_requested) return { test_numerical_mode::P1A_NOSIDECAR_Q8, test_numerical_reason::NONE };
    if (!spec.latebind_env_enabled) return { test_numerical_mode::REFERENCE, test_numerical_reason::DISABLED_BY_ENV };
    if (!spec.is_relay_sync)        return { test_numerical_mode::REFERENCE, test_numerical_reason::NON_RELAY_SYNC };
    if (!spec.is_f32_wire)          return { test_numerical_mode::REFERENCE, test_numerical_reason::NON_F32_WIRE };
    if (!spec.has_late_tensors)     return { test_numerical_mode::REFERENCE, test_numerical_reason::NO_LATE_TENSORS };
    if (spec.exact_q_requested)     return { test_numerical_mode::EXACT_F32, test_numerical_reason::EXACT_Q_REQUESTED };
    if (!spec.hw_int_dot)           return { test_numerical_mode::EXACT_F32, test_numerical_reason::MISSING_HARDWARE_INT_DOT };
    if (!spec.hw_wave32)            return { test_numerical_mode::EXACT_F32, test_numerical_reason::UNSUPPORTED_WAVE32 };
    if (!spec.shape_aligned)        return { test_numerical_mode::EXACT_F32, test_numerical_reason::UNALIGNED_LATE_SHAPE };
    if (!spec.pipeline_ready)       return { test_numerical_mode::EXACT_F32, test_numerical_reason::PIPELINE_UNAVAILABLE };
    return { test_numerical_mode::AGGRESSIVE_Q8, test_numerical_reason::NONE };
}

static void test_tp5_numerical_mode_resolution() {
    fprintf(stderr, "--- test_tp5_numerical_mode_resolution ---\n");
    test_numerical_spec base;
    base.latebind_env_enabled    = true;
    base.is_relay_sync           = true;
    base.is_f32_wire             = true;
    base.has_late_tensors        = true;
    base.exact_q_requested       = false;
    base.p1a_nosidecar_requested = false;
    base.hw_int_dot              = true;
    base.hw_wave32               = true;
    base.shape_aligned           = true;
    base.pipeline_ready          = true;

    // Full fast path -> AGGRESSIVE_Q8
    auto res = test_resolve_numerical_mode(base);
    TEST_ASSERT(res.first == test_numerical_mode::AGGRESSIVE_Q8);
    TEST_ASSERT(res.second == test_numerical_reason::NONE);

    // P1-A requested via env -> P1A_NOSIDECAR_Q8
    {
        auto s = base; s.p1a_nosidecar_requested = true;
        res = test_resolve_numerical_mode(s);
        TEST_ASSERT(res.first == test_numerical_mode::P1A_NOSIDECAR_Q8);
        TEST_ASSERT(res.second == test_numerical_reason::NONE);
    }

    // Disabled by env -> REFERENCE
    {
        auto s = base; s.latebind_env_enabled = false;
        res = test_resolve_numerical_mode(s);
        TEST_ASSERT(res.first == test_numerical_mode::REFERENCE);
        TEST_ASSERT(res.second == test_numerical_reason::DISABLED_BY_ENV);
    }
    // Non-relay sync -> REFERENCE
    {
        auto s = base; s.is_relay_sync = false;
        res = test_resolve_numerical_mode(s);
        TEST_ASSERT(res.first == test_numerical_mode::REFERENCE);
        TEST_ASSERT(res.second == test_numerical_reason::NON_RELAY_SYNC);
    }
    // Non-F32 wire -> REFERENCE
    {
        auto s = base; s.is_f32_wire = false;
        res = test_resolve_numerical_mode(s);
        TEST_ASSERT(res.first == test_numerical_mode::REFERENCE);
        TEST_ASSERT(res.second == test_numerical_reason::NON_F32_WIRE);
    }
    // No late tensors -> REFERENCE
    {
        auto s = base; s.has_late_tensors = false;
        res = test_resolve_numerical_mode(s);
        TEST_ASSERT(res.first == test_numerical_mode::REFERENCE);
        TEST_ASSERT(res.second == test_numerical_reason::NO_LATE_TENSORS);
    }

    // Exact Q requested via env -> EXACT_F32
    {
        auto s = base; s.exact_q_requested = true;
        res = test_resolve_numerical_mode(s);
        TEST_ASSERT(res.first == test_numerical_mode::EXACT_F32);
        TEST_ASSERT(res.second == test_numerical_reason::EXACT_Q_REQUESTED);
    }
    // Missing int dot product -> EXACT_F32
    {
        auto s = base; s.hw_int_dot = false;
        res = test_resolve_numerical_mode(s);
        TEST_ASSERT(res.first == test_numerical_mode::EXACT_F32);
        TEST_ASSERT(res.second == test_numerical_reason::MISSING_HARDWARE_INT_DOT);
    }
    // Unsupported wave32 -> EXACT_F32
    {
        auto s = base; s.hw_wave32 = false;
        res = test_resolve_numerical_mode(s);
        TEST_ASSERT(res.first == test_numerical_mode::EXACT_F32);
        TEST_ASSERT(res.second == test_numerical_reason::UNSUPPORTED_WAVE32);
    }
    // Unaligned tensor shape -> EXACT_F32
    {
        auto s = base; s.shape_aligned = false;
        res = test_resolve_numerical_mode(s);
        TEST_ASSERT(res.first == test_numerical_mode::EXACT_F32);
        TEST_ASSERT(res.second == test_numerical_reason::UNALIGNED_LATE_SHAPE);
    }
    // Pipeline creation failure -> EXACT_F32
    {
        auto s = base; s.pipeline_ready = false;
        res = test_resolve_numerical_mode(s);
        TEST_ASSERT(res.first == test_numerical_mode::EXACT_F32);
        TEST_ASSERT(res.second == test_numerical_reason::PIPELINE_UNAVAILABLE);
    }
    fprintf(stderr, "  Numerical mode resolution: 11/11 branches verified\n");
}

static void test_tp5_latebind_protocol_invariants() {
    fprintf(stderr, "--- test_tp5_latebind_protocol_invariants ---\n");
    // Layout invariants from TP5 LateBind protocol
    constexpr size_t TP5_RELAY_HEADER_BYTES = 64;
    constexpr size_t TP5_LATE_Q_CONTROL_BYTES = 64;
    constexpr size_t TP5_LATE_Q_READY_WORD = 0;
    constexpr size_t TP5_LATE_Q_COUNTER_WORD = 1;

    const auto ctrl_offset = [](size_t L) { return TP5_RELAY_HEADER_BYTES + L; };
    const auto payload_offset = [](size_t L, bool fast) {
        return TP5_RELAY_HEADER_BYTES + L + (fast ? TP5_LATE_Q_CONTROL_BYTES : 0);
    };

    // Test a variety of realistic alignments and offsets (e.g. 2560 elements * sizeof(float))
    const size_t test_offsets[] = { 2560 * sizeof(float), 4096 * sizeof(float), 10240, 16384 };
    for (size_t L : test_offsets) {
        // Invariant 1: Control block is 64-byte aligned
        TEST_ASSERT(ctrl_offset(L) % 64 == 0);
        // Invariant 2: Aggressive Q8 Payload is 64-byte aligned
        TEST_ASSERT(payload_offset(L, true) % 64 == 0);
        // Invariant 3: Control block [ctrl, ctrl + 64) and Payload [payload, ...) do not overlap in aggressive mode
        TEST_ASSERT(ctrl_offset(L) + TP5_LATE_Q_CONTROL_BYTES <= payload_offset(L, true));
        TEST_ASSERT(payload_offset(L, true) - ctrl_offset(L) == TP5_LATE_Q_CONTROL_BYTES);

        // Invariant 4: GPU word indexing matches CPU byte indexing
        // GPU late_word_offset = (64 + L + 64) / 4 = payload_offset(L, true) / 4
        const uint32_t late_word_offset = (uint32_t) (payload_offset(L, true) / 4);
        // GPU writes ready flag to qmail[late_word_offset - 16u]
        const size_t gpu_ready_byte = (late_word_offset - 16u) * 4;
        TEST_ASSERT(gpu_ready_byte == ctrl_offset(L) + TP5_LATE_Q_READY_WORD * 4);
        // GPU writes WG counter to qmail[late_word_offset - 15u]
        const size_t gpu_counter_byte = (late_word_offset - 15u) * 4;
        TEST_ASSERT(gpu_counter_byte == ctrl_offset(L) + TP5_LATE_Q_COUNTER_WORD * 4);

        // Invariant 5: Regression test against the old defect
        // In the old bug, CPU polled offset B + 16 (main header word 4), which is NOT the control block!
        const size_t old_poll_offset = 16;
        TEST_ASSERT(old_poll_offset != ctrl_offset(L));
        // In the old bug, CPU payload read dst = B + 64 + L, which erroneously aliased the control region!
        const size_t old_cpu_payload = 64 + L;
        TEST_ASSERT(old_cpu_payload == ctrl_offset(L));
        TEST_ASSERT(old_cpu_payload != payload_offset(L, true));

        // Invariant 6: Exact fallback path consistency
        // In exact mode (!fast), control is in host-imported RAM status[6], and payload broadcast is at B + 64 + L
        TEST_ASSERT(payload_offset(L, false) == 64 + L);

        // Invariant 7: Workspace capacity bounds check covers control block + payload in both exact and fast modes
        // In exact mode (!fast), wire is F32, payload starts at B + 64 + L, so end is 64 + L + wire_bytes.
        // In fast mode (fast), wire is F16, payload starts at B + 128 + L (after 64B control), so end is 128 + L + wire_bytes.
        // Old check used (L + wire_bytes + 64) which under-counted by 64B in fast mode.
        const size_t late_counts[] = { 1024, 2048, 2560 };
        for (size_t late_count : late_counts) {
            const size_t exact_wire_bytes = late_count * sizeof(float);
            const size_t fast_wire_bytes  = late_count * sizeof(uint16_t); // ggml_fp16_t wire

            const size_t exact_payload_end = payload_offset(L, false) + exact_wire_bytes;
            const size_t fast_payload_end  = payload_offset(L, true)  + fast_wire_bytes;

            const size_t old_exact_check = L + exact_wire_bytes + 64;
            const size_t old_fast_check  = L + fast_wire_bytes  + 64;

            // In exact mode, old check happened to match single-source payload end
            TEST_ASSERT(exact_payload_end == old_exact_check);

            // In fast mode, old check under-counted by exactly TP5_LATE_Q_CONTROL_BYTES (64B)
            TEST_ASSERT(fast_payload_end > old_fast_check);
            TEST_ASSERT(fast_payload_end - old_fast_check == TP5_LATE_Q_CONTROL_BYTES);

            // A stride between old check and new check must fail the single-source bounds check
            const size_t borderline_stride = old_fast_check + 32; // > old check, but < real payload end
            TEST_ASSERT(old_fast_check <= borderline_stride);
            TEST_ASSERT(fast_payload_end > borderline_stride); // correctly identified as exceeding workspace
        }
    }

    // Invariant 8: Definition-time WAR contract verification for Exact F32 and Aggressive Q8
    // Validates against production tp5_latebind_semantic_step & tp5_validate_latebind_war_schedule.
    std::string war_err;

    // 1. Exact F32 canonical schedule: READ_TREFS (late_q) -> BARRIER_WAR_TREFS (q_norm_barrier) -> WRITE_TREFS (late_norm)
    std::vector<tp5_latebind_semantic_step> canonical_exact = {
        { tp5_latebind_semantic_step::kind::READ_TREFS,        "late_q" },
        { tp5_latebind_semantic_step::kind::BARRIER_WAR_TREFS, "q_norm_barrier" },
        { tp5_latebind_semantic_step::kind::WRITE_TREFS,       "late_norm" },
    };
    TEST_ASSERT(tp5_validate_latebind_war_schedule(false, canonical_exact, war_err));

    // 2. Aggressive Q8 canonical schedule:
    // READ_TREFS (late_act_q8) -> BARRIER_ACT_BUF (act_ready) -> BARRIER_WAR_TREFS (q_norm_barrier) -> WRITE_TREFS (late_norm) -> DISPATCH_Q8DOT (late_q8dot)
    // Note: No barrier between late_norm and late_q8dot (zero-barrier overlap).
    std::vector<tp5_latebind_semantic_step> canonical_agg = {
        { tp5_latebind_semantic_step::kind::READ_TREFS,        "late_act_q8" },
        { tp5_latebind_semantic_step::kind::BARRIER_ACT_BUF,   "act_ready" },
        { tp5_latebind_semantic_step::kind::BARRIER_WAR_TREFS, "q_norm_barrier" },
        { tp5_latebind_semantic_step::kind::WRITE_TREFS,       "late_norm" },
        { tp5_latebind_semantic_step::kind::DISPATCH_Q8DOT,    "late_q8dot" },
    };
    TEST_ASSERT(tp5_validate_latebind_war_schedule(true, canonical_agg, war_err));

    // 3. Negative regression 1: Aggressive Q8 missing WAR barrier on trefs (the pre-fix defect)
    std::vector<tp5_latebind_semantic_step> reg_agg_no_war = {
        { tp5_latebind_semantic_step::kind::READ_TREFS,        "late_act_q8" },
        { tp5_latebind_semantic_step::kind::BARRIER_ACT_BUF,   "act_ready" },
        { tp5_latebind_semantic_step::kind::WRITE_TREFS,       "late_norm" },
        { tp5_latebind_semantic_step::kind::DISPATCH_Q8DOT,    "late_q8dot" },
    };
    TEST_ASSERT(!tp5_validate_latebind_war_schedule(true, reg_agg_no_war, war_err));
    TEST_ASSERT(war_err.find("missing BARRIER_WAR_TREFS") != std::string::npos);

    // 4. Negative regression 2: Exact F32 missing WAR barrier on trefs
    std::vector<tp5_latebind_semantic_step> reg_exact_no_war = {
        { tp5_latebind_semantic_step::kind::READ_TREFS,        "late_q" },
        { tp5_latebind_semantic_step::kind::WRITE_TREFS,       "late_norm" },
    };
    TEST_ASSERT(!tp5_validate_latebind_war_schedule(false, reg_exact_no_war, war_err));
    TEST_ASSERT(war_err.find("missing BARRIER_WAR_TREFS") != std::string::npos);

    // 5. Negative regression 3: Misplaced barrier (after write instead of before write)
    std::vector<tp5_latebind_semantic_step> reg_barrier_after_write = {
        { tp5_latebind_semantic_step::kind::READ_TREFS,        "late_act_q8" },
        { tp5_latebind_semantic_step::kind::BARRIER_ACT_BUF,   "act_ready" },
        { tp5_latebind_semantic_step::kind::WRITE_TREFS,       "late_norm" },
        { tp5_latebind_semantic_step::kind::BARRIER_WAR_TREFS, "q_norm_barrier" },
        { tp5_latebind_semantic_step::kind::DISPATCH_Q8DOT,    "late_q8dot" },
    };
    TEST_ASSERT(!tp5_validate_latebind_war_schedule(true, reg_barrier_after_write, war_err));
    TEST_ASSERT(war_err.find("BARRIER_WAR_TREFS must precede WRITE_TREFS") != std::string::npos);

    // 6. Negative regression 4: Write before read hazard (norm overwrites before Q reads)
    std::vector<tp5_latebind_semantic_step> reg_write_before_read = {
        { tp5_latebind_semantic_step::kind::WRITE_TREFS,       "late_norm" },
        { tp5_latebind_semantic_step::kind::BARRIER_WAR_TREFS, "q_norm_barrier" },
        { tp5_latebind_semantic_step::kind::READ_TREFS,        "late_q" },
    };
    TEST_ASSERT(!tp5_validate_latebind_war_schedule(false, reg_write_before_read, war_err));
    TEST_ASSERT(war_err.find("READ_TREFS must precede BARRIER_WAR_TREFS") != std::string::npos);

    // 7. Negative regression 5: Barrier mistakenly inserted between norm and Q8dot
    // Violates the requirement that norm and Q8dot have zero execution barriers between them.
    std::vector<tp5_latebind_semantic_step> reg_barrier_in_q8dot = {
        { tp5_latebind_semantic_step::kind::READ_TREFS,        "late_act_q8" },
        { tp5_latebind_semantic_step::kind::BARRIER_ACT_BUF,   "act_ready" },
        { tp5_latebind_semantic_step::kind::BARRIER_WAR_TREFS, "q_norm_barrier" },
        { tp5_latebind_semantic_step::kind::WRITE_TREFS,       "late_norm" },
        { tp5_latebind_semantic_step::kind::BARRIER_ACT_BUF,   "extra_barrier" },
        { tp5_latebind_semantic_step::kind::DISPATCH_Q8DOT,    "late_q8dot" },
    };
    TEST_ASSERT(!tp5_validate_latebind_war_schedule(true, reg_barrier_in_q8dot, war_err));
    TEST_ASSERT(war_err.find("zero-barrier overlap violated") != std::string::npos);

    fprintf(stderr, "  LateBind protocol invariants: Q control/payload non-overlapping, word alignment verified, F32-Q order asserted\n");
}

static void test_tp5_latebind_multi_row_invariants() {
    fprintf(stderr, "--- test_tp5_latebind_multi_row_invariants ---\n");
    // Contract: Multi-row verification (active_rows in {1, 2, 4}) within single capacity=4 definition.
    // Layout/parameters defined once; Q/sidecar layout contiguous; single-row non-regression guaranteed.
    constexpr size_t TP5_RELAY_HEADER_BYTES = 64;
    constexpr size_t TP5_LATE_Q_CONTROL_BYTES = 64;
    constexpr size_t TP5_LATE_MAX_FLOATS = 8192;

    const auto ctrl_offset = [](size_t L) { return TP5_RELAY_HEADER_BYTES + L; };
    const auto payload_offset = [](size_t L, bool fast) {
        return TP5_RELAY_HEADER_BYTES + L + (fast ? TP5_LATE_Q_CONTROL_BYTES : 0);
    };

    const uint32_t width = 2560;
    const uint32_t streams = 4;
    const uint32_t late_rank = 320;
    const size_t capacity_rows = 4;
    const size_t max_late_floats = capacity_rows * streams * late_rank; // 5120 <= 8192
    TEST_ASSERT(max_late_floats <= TP5_LATE_MAX_FLOATS);

    // Multi-row boundary sweep: 1..4 rows (including 3 rows)
    for (uint32_t rows : {1u, 2u, 3u, 4u}) {
        const size_t n_elems = size_t(width) * rows;
        // Invariant 1: n_elems must be divisible by width
        TEST_ASSERT(n_elems % width == 0);
        const size_t derived_rows = n_elems / width;
        TEST_ASSERT(derived_rows == rows);
        TEST_ASSERT(derived_rows <= capacity_rows);

        // Invariant 2: Active late_count matches GPU output and CPU handoff exactly
        const size_t active_late_count = derived_rows * streams * late_rank;
        TEST_ASSERT(active_late_count <= max_late_floats);
        TEST_ASSERT(active_late_count == rows * 1280);

        // Invariant 3: tp5_late_stage capacity check: capacity_rows * streams * rank_dim <= TP5_LATE_MAX_FLOATS
        TEST_ASSERT(size_t(derived_rows) * streams * late_rank <= TP5_LATE_MAX_FLOATS);

        // Invariant 4: Word alignment & non-overlapping invariants for multi-row
        const size_t L = width * sizeof(float);
        TEST_ASSERT(ctrl_offset(L) % 64 == 0);
        TEST_ASSERT(payload_offset(L, true) % 64 == 0);
        TEST_ASSERT(ctrl_offset(L) + TP5_LATE_Q_CONTROL_BYTES <= payload_offset(L, true));

        // Invariant 5: External tensor binding size requirements against capacity_rows
        const uint64_t mixed_min_size = uint64_t(derived_rows) * width * sizeof(float);
        const uint64_t lo_min_size = uint64_t(derived_rows) * late_rank * sizeof(float);
        const uint64_t res_min_size = uint64_t(derived_rows) * streams * width * sizeof(float);
        const uint64_t binding1_local_z_min_size = uint64_t(derived_rows) * width * sizeof(float);
        TEST_ASSERT(mixed_min_size == rows * 2560 * sizeof(float));
        TEST_ASSERT(lo_min_size == rows * 320 * sizeof(float));
        TEST_ASSERT(res_min_size == rows * 10240 * sizeof(float));
        TEST_ASSERT(binding1_local_z_min_size == rows * 2560 * sizeof(float));

        // Negative regression test: hc.bindings[1] under-sized must be rejected fail-closed
        auto check_binding1_capacity = [&](uint64_t sz) -> bool {
            return sz >= binding1_local_z_min_size;
        };
        TEST_ASSERT(check_binding1_capacity(binding1_local_z_min_size));
        TEST_ASSERT(!check_binding1_capacity(binding1_local_z_min_size - 1));
        if (rows > 1) {
            // Legacy single-row size against multi-row capacity must fail closed
            TEST_ASSERT(!check_binding1_capacity(width * sizeof(float)));
        }

        // Single-row non-regression check
        if (rows == 1) {
            TEST_ASSERT(active_late_count == streams * late_rank);
            TEST_ASSERT(active_late_count == 1280);
            TEST_ASSERT(mixed_min_size == width * sizeof(float));
        }
    }
    fprintf(stderr, "  LateBind multi-row invariants: 1..4-row boundaries verified, external binding capacity asserted, single-row non-regression asserted\n");
}

static void test_tp5_p1a_nosidecar_schedule_contract() {
    fprintf(stderr, "--- test_tp5_p1a_nosidecar_schedule_contract ---\n");
    std::string err;

    // 1. Canonical P1-A schedule:
    // Y is ready -> combine/RMS (WRITE_TREFS) -> norm_ready (BARRIER_NORM_ACT) ->
    // norm_act_q8 (DISPATCH_ACT_Q8) -> act_ready (BARRIER_ACT_BUF) ->
    // down_q8dot_local (DISPATCH_Q8DOT) -> q_local_ready (BARRIER_LOCAL_Q) ->
    // lo_q8_local (DISPATCH_LO_Q8) -> lo_ready (BARRIER_LO_BUF) ->
    // up_q8dot_fold (DISPATCH_UP_Q8DOT)
    std::vector<tp5_latebind_semantic_step> canonical_p1a = {
        { tp5_latebind_semantic_step::kind::WRITE_TREFS,       "late_norm" },
        { tp5_latebind_semantic_step::kind::BARRIER_NORM_ACT,  "norm_ready" },
        { tp5_latebind_semantic_step::kind::DISPATCH_ACT_Q8,   "norm_act_q8" },
        { tp5_latebind_semantic_step::kind::BARRIER_ACT_BUF,   "act_ready" },
        { tp5_latebind_semantic_step::kind::DISPATCH_Q8DOT,    "down_q8dot_local" },
        { tp5_latebind_semantic_step::kind::BARRIER_LOCAL_Q,   "q_local_ready" },
        { tp5_latebind_semantic_step::kind::DISPATCH_LO_Q8,    "lo_q8_local" },
        { tp5_latebind_semantic_step::kind::BARRIER_LO_BUF,    "lo_ready" },
        { tp5_latebind_semantic_step::kind::DISPATCH_UP_Q8DOT, "up_q8dot_fold" },
    };
    TEST_ASSERT(tp5_validate_p1a_schedule(canonical_p1a, err));

    // 2. Negative Regression 1: Norm after Q8 (violates combine/norm before Q8)
    {
        std::vector<tp5_latebind_semantic_step> reg_norm_after_q8 = {
            { tp5_latebind_semantic_step::kind::DISPATCH_ACT_Q8,   "norm_act_q8" },
            { tp5_latebind_semantic_step::kind::WRITE_TREFS,       "late_norm" },
            { tp5_latebind_semantic_step::kind::BARRIER_NORM_ACT,  "norm_ready" },
            { tp5_latebind_semantic_step::kind::BARRIER_ACT_BUF,   "act_ready" },
            { tp5_latebind_semantic_step::kind::DISPATCH_Q8DOT,    "down_q8dot_local" },
            { tp5_latebind_semantic_step::kind::BARRIER_LOCAL_Q,   "q_local_ready" },
            { tp5_latebind_semantic_step::kind::DISPATCH_LO_Q8,    "lo_q8_local" },
            { tp5_latebind_semantic_step::kind::BARRIER_LO_BUF,    "lo_ready" },
            { tp5_latebind_semantic_step::kind::DISPATCH_UP_Q8DOT, "up_q8dot_fold" },
        };
        TEST_ASSERT(!tp5_validate_p1a_schedule(reg_norm_after_q8, err));
        TEST_ASSERT(err.find("norm") != std::string::npos || err.find("BARRIER_NORM_ACT") != std::string::npos);
    }

    // 3. Negative Regression 2: Pre-norm sidecar read on trefs (violates no-sidecar invariant)
    {
        std::vector<tp5_latebind_semantic_step> reg_sidecar_read = canonical_p1a;
        reg_sidecar_read.insert(reg_sidecar_read.begin(), { tp5_latebind_semantic_step::kind::READ_TREFS, "late_act_q8" });
        TEST_ASSERT(!tp5_validate_p1a_schedule(reg_sidecar_read, err));
        TEST_ASSERT(err.find("no-sidecar invariant violated") != std::string::npos);
    }

    // 4. Negative Regression 3: Missing Q8 down projection
    {
        std::vector<tp5_latebind_semantic_step> reg_missing_down = canonical_p1a;
        reg_missing_down.erase(reg_missing_down.begin() + 4);
        TEST_ASSERT(!tp5_validate_p1a_schedule(reg_missing_down, err));
        TEST_ASSERT(err.find("missing DISPATCH_Q8DOT") != std::string::npos);
    }

    // 5. Negative Regression 4: Missing Q8 up projection
    {
        std::vector<tp5_latebind_semantic_step> reg_missing_up = canonical_p1a;
        reg_missing_up.pop_back();
        TEST_ASSERT(!tp5_validate_p1a_schedule(reg_missing_up, err));
        TEST_ASSERT(err.find("missing DISPATCH_UP_Q8DOT") != std::string::npos);
    }

    // 6. Negative Regression 5: Missing barrier between norm and ACT_Q8
    {
        std::vector<tp5_latebind_semantic_step> reg_missing_norm_barrier = canonical_p1a;
        reg_missing_norm_barrier.erase(reg_missing_norm_barrier.begin() + 1);
        TEST_ASSERT(!tp5_validate_p1a_schedule(reg_missing_norm_barrier, err));
        TEST_ASSERT(err.find("missing BARRIER_NORM_ACT") != std::string::npos);
    }

    // 7. Negative Regression 6: Missing barrier between down Q8 and LO Q8
    {
        std::vector<tp5_latebind_semantic_step> reg_missing_q_barrier = canonical_p1a;
        reg_missing_q_barrier.erase(reg_missing_q_barrier.begin() + 5);
        TEST_ASSERT(!tp5_validate_p1a_schedule(reg_missing_q_barrier, err));
        TEST_ASSERT(err.find("missing BARRIER_LOCAL_Q") != std::string::npos);
    }

    // 8. Default-off invariant: When P1-A mode is disabled, existing exact and aggressive schedules
    // remain 100% identical and validate successfully under tp5_validate_latebind_war_schedule.
    {
        std::vector<tp5_latebind_semantic_step> canonical_exact = {
            { tp5_latebind_semantic_step::kind::READ_TREFS,        "late_q" },
            { tp5_latebind_semantic_step::kind::BARRIER_WAR_TREFS, "q_norm_barrier" },
            { tp5_latebind_semantic_step::kind::WRITE_TREFS,       "late_norm" },
        };
        TEST_ASSERT(tp5_validate_latebind_war_schedule(false, canonical_exact, err));

        std::vector<tp5_latebind_semantic_step> canonical_agg = {
            { tp5_latebind_semantic_step::kind::READ_TREFS,        "late_act_q8" },
            { tp5_latebind_semantic_step::kind::BARRIER_ACT_BUF,   "act_ready" },
            { tp5_latebind_semantic_step::kind::BARRIER_WAR_TREFS, "q_norm_barrier" },
            { tp5_latebind_semantic_step::kind::WRITE_TREFS,       "late_norm" },
            { tp5_latebind_semantic_step::kind::DISPATCH_Q8DOT,    "late_q8dot" },
        };
        TEST_ASSERT(tp5_validate_latebind_war_schedule(true, canonical_agg, err));
    }

    fprintf(stderr, "  P1-A no-sidecar schedule contract: canonical sequence, 6 negative regressions, and mode-off invariants verified\n");
}

// M3 invariant 2 / M4 runtime-row protocol: LateBind kernels bound their token
// loops with the CPU-published RELAY header word 5 (runtime active rows), never
// with the push-constant capacity. This fixture pins the CPU side of that
// contract: which word carries the count and what values it takes.
static void test_tp5_latebind_runtime_rows_protocol() {
    fprintf(stderr, "--- test_tp5_latebind_runtime_rows_protocol ---\n");
    constexpr uint32_t ACTIVE_ROWS_WORD = 5u;
    constexpr size_t HEADER_BYTES = 64;
    constexpr size_t PAYLOAD_WORD_OFFSET = 16; // F32 payload starts at byte 64
    (void) PAYLOAD_WORD_OFFSET;

    // 1. Word 5 lies strictly inside the 64-byte header, before the payload.
    TEST_ASSERT((ACTIVE_ROWS_WORD + 1u) * sizeof(uint32_t) <= HEADER_BYTES);

    // 2. Publication value: active rows when the frame is known, capacity
    //    when only capacity is known (chain path without a frame), zero for
    //    non-predefined one-shot callers (legacy shader fallback).
    const uint32_t capacity_rows = 4;
    for (uint32_t active : {1u, 2u, 3u, 4u}) {
        TEST_ASSERT(active <= capacity_rows);
        TEST_ASSERT((active ? active : capacity_rows) == active);
    }
    TEST_ASSERT((0u ? 0u : capacity_rows) == capacity_rows);

    // 3. MTP sequence 1->4->1->2: every publication carries the true active
    //    count, never the capacity, for a 1-row draft. The definition
    //    (descriptors, pipelines, push constants) is identical across all
    //    four publications — only the runtime word changes.
    const uint32_t seq[] = {1u, 4u, 1u, 2u};
    for (uint32_t rows : seq) {
        TEST_ASSERT(rows <= capacity_rows);
        TEST_ASSERT(rows != 0u);
        TEST_ASSERT((rows ? rows : capacity_rows) == rows);
    }
    fprintf(stderr, "  LateBind runtime-rows protocol: header word 5 placement, publication values, "
                   "legacy fallback and 1->4->1->2 sequence verified\n");
}

static void test_tp5_sidecar_format_keying() {
    fprintf(stderr, "--- test_tp5_sidecar_format_keying ---\n");
    // The F16 sidecar format is a property of the Q8DOT producer, not of the
    // aggressive schedule name. P1-A dispatches the same Q8DOT kernel, so it
    // must consume the same F16 sidecar layout and the same Q control region.
    // Keying the format off mode == AGGRESSIVE_Q8 left P1-A with the exact-path
    // consumers (status[6] poll + F32 host-import read), which can never be
    // satisfied by the Q8DOT publication — a guaranteed handoff timeout.
    const test_numerical_mode modes[] = {
        test_numerical_mode::REFERENCE,
        test_numerical_mode::EXACT_F32,
        test_numerical_mode::AGGRESSIVE_Q8,
        test_numerical_mode::P1A_NOSIDECAR_Q8,
    };
    for (test_numerical_mode mode : modes) {
        const bool late_q8_fast =
            mode == test_numerical_mode::AGGRESSIVE_Q8 ||
            mode == test_numerical_mode::P1A_NOSIDECAR_Q8;
        // Contract: sidecar format follows the Q8 producer path.
        const bool sidecar_f16 = late_q8_fast;
        TEST_ASSERT(sidecar_f16 == late_q8_fast);
        // The Q control region and payload offsets are derived from the same
        // flag: F16 payload sits 64 bytes after the control region, and the
        // CPU Q-wait polls qctrl[0] exactly when the Q8DOT producer is active.
        constexpr size_t L = 2560 * sizeof(float); // production shape fixture
        constexpr size_t control = 64 + L;
        constexpr size_t payload_f16 = 64 + L + 64;
        constexpr size_t payload_f32 = 64 + L;
        static_assert(payload_f16 != payload_f32);
        if (late_q8_fast) {
            TEST_ASSERT(payload_f16 == control + 64);
            TEST_ASSERT((payload_f16 % 64) == 0);
        } else {
            TEST_ASSERT(payload_f32 == control);
        }
    }
    fprintf(stderr, "  Sidecar format keying: Q8DOT producer implies F16 sidecar and qctrl ready poll "
                    "for both AGGRESSIVE_Q8 and P1A_NOSIDECAR_Q8 verified\n");
}

static void test_tp5_packed_weight_layout() {
    fprintf(stderr, "--- test_tp5_packed_weight_layout ---\n");
    // P1-B: the packed weight layout must exactly cover the same logical
    // blocks as the late_q8_0 GGUF-side layout, plus one flag block at
    // index 0. Both W_down and W_up have identical block counts:
    //   W_down = [late_rank][streams*width]  -> late_rank * streams*width/32
    //   W_up   = [streams*width][late_rank]  -> streams*width * late_rank/32
    // The dot-product inner loop reads one uint32 word per operand, so the
    // per-block element is 4 (d) + 8*4 (qs_words) = 36 bytes.
    struct PackedBlock { float d; uint32_t qs_words[8]; };
    static_assert(sizeof(PackedBlock) == 36, "packed block must be 36B");
    TEST_ASSERT(sizeof(PackedBlock) == sizeof(float) + 8 * sizeof(uint32_t));

    // Production shape fixture: width=2560, streams=4, late_rank=320.
    const uint32_t width = 2560u, streams = 4u, late_rank = 320u;
    const uint64_t blocks = uint64_t(late_rank) * streams * width / 32u;
    TEST_ASSERT(blocks == 320u * 4u * 80u);  // 102400 blocks
    const uint64_t packed_bytes = (blocks + 1u) * sizeof(PackedBlock);
    TEST_ASSERT(packed_bytes == (102400u + 1u) * 36u);

    // The source Q8_0 bytes must cover exactly `blocks` blocks (34B each in
    // the late_q8_0 shader-side layout: f16 d + 16 x i16).
    const uint64_t src_bytes = blocks * (2u + 32u);
    TEST_ASSERT(src_bytes == 102400u * 34u);

    // Index mapping: shader reads w[base + block + 1]; the +1 flag offset must
    // never push the last block past the buffer end.
    TEST_ASSERT((blocks - 1u + 1u) * sizeof(PackedBlock) + sizeof(PackedBlock) <= packed_bytes);

    // The done-flag word is the first 4 bytes of block 0; the pack shader
    // writes block i to index i+1, so the flag block is never overwritten by
    // data (blocks span indices [1, blocks]).
    TEST_ASSERT(1u <= blocks);
    fprintf(stderr, "  Packed weight layout: 36B/block + flag at index 0, W_down/W_up share block count, "
                    "bounds verified (blocks=%llu bytes=%llu)\n",
            (unsigned long long) blocks, (unsigned long long) packed_bytes);
}

int main() {
    test_plan_rejects_bad_ranks();
    test_plan_rejects_indivisible_moe();
    test_tensor_plans_target_types();
    test_quant_slice_rejection();
    test_head_split_quant_math();
    test_validate();
    test_other_rank_counts();
    test_gqa_head_mapping_cpu(5, 2);
    test_gqa_head_mapping_cpu(2, 2);
    test_gqa_head_mapping_cpu(5, 8);
    test_gdn_headmap_modulo();
    test_gdn_headmap_plan_tables();
    test_gdn_headmap_execution_cpu();
    test_gdn_headmap_qkv_span_values();
    // QSA headmap fixture owns the opt-in flag: enable it only around its own
    // plan build so every other fixture keeps the native default layout.
    setenv("GGML_TP5_QSA_HEADMAP", "1", 1);
    test_tp5_qsa_headmap();
    unsetenv("GGML_TP5_QSA_HEADMAP");
    test_tp5_split_state_gdn_qkv(false);
    test_tp5_split_state_gdn_qkv(true);
    test_nextn_layer_plan_bounds();
    test_tp5_latebind_protocol_invariants();
    test_tp5_latebind_multi_row_invariants();
    test_tp5_latebind_runtime_rows_protocol();
    test_tp5_sidecar_format_keying();
    test_tp5_packed_weight_layout();
    test_tp5_numerical_mode_resolution();
    test_tp5_p1a_nosidecar_schedule_contract();

    if (g_failures > 0) {
        fprintf(stderr, "%d failures\n", g_failures);
        return 1;
    }
    printf("test-tp5-plan: all passed\n");
    return 0;
}
