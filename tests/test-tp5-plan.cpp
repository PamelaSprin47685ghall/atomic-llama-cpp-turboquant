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

    if (g_failures > 0) {
        fprintf(stderr, "%d failures\n", g_failures);
        return 1;
    }
    printf("test-tp5-plan: all passed\n");
    return 0;
}
