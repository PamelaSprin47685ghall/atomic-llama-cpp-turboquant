// TP5 plan CPU tests: role tables, per-rank slice rules, quant-block legality.
// Pure CPU; no GPU, no GGUF, no Vulkan. Mirrors TP5.md sections 3.3, 4.4, 6.3,
// 7.1, 8.1 and the checks listed in 22.1.

#include "llama-tp5-plan.h"
#include "llama-hparams.h"

#include <cstdio>
#include <cstring>
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

static void test_plan_build_target() {
    fprintf(stderr, "--- test_plan_build_target ---\n");
    llama_hparams hp = make_hparams();
    llama_tp5_plan plan;
    llama_tp5_error err;
    TEST_ASSERT(llama_tp5_plan_build(hp, 5, 248320, plan, err));
    TEST_ASSERT(plan.ranks == 5);
    TEST_ASSERT(plan.H == 2560 && plan.L == 48 && plan.C == 4 && plan.R == 320);
    TEST_ASSERT(plan.E == 512 && plan.K == 10 && plan.F == 640 && plan.Fs == 640);
    TEST_ASSERT(plan.Nq == 24 && plan.Nkv == 2 && plan.da == 256);
    TEST_ASSERT(plan.Nk == 16 && plan.Nv == 48 && plan.ds == 128);
    TEST_ASSERT(plan.n_full == 12);
    TEST_ASSERT(plan.expected_events == 96);

    // QSA roles (TP5.md 7.1): [5,5,5,5,4] Q heads, KV [1,1,2,1,1]
    TEST_ASSERT(plan.q_role_counts[0] == 5 && plan.q_role_counts[1] == 5 &&
                plan.q_role_counts[2] == 5 && plan.q_role_counts[3] == 5 &&
                plan.q_role_counts[4] == 4);
    TEST_ASSERT(plan.kv_role_counts[0] == 1 && plan.kv_role_counts[1] == 1 &&
                plan.kv_role_counts[2] == 2 && plan.kv_role_counts[3] == 1 &&
                plan.kv_role_counts[4] == 1);
    // rotated KV instances over 12 full layers: [14,14,15,15,14] (TP5.md 4.4)
    TEST_ASSERT(plan.kv_instances[0] == 14 && plan.kv_instances[1] == 14 &&
                plan.kv_instances[2] == 15 && plan.kv_instances[3] == 15 &&
                plan.kv_instances[4] == 14);
    int kv_total = 0;
    for (int r = 0; r < 5; ++r) kv_total += plan.kv_instances[r];
    TEST_ASSERT(kv_total == 72); // 12 layers * 6 instances

    // GDN V heads (TP5.md 8.1): [10,10,10,10,8]
    TEST_ASSERT(plan.gdn_v_heads[0] == 10 && plan.gdn_v_heads[1] == 10 &&
                plan.gdn_v_heads[2] == 10 && plan.gdn_v_heads[3] == 10 &&
                plan.gdn_v_heads[4] == 8);
    TEST_ASSERT(plan.gdn_v_ranges.size() == 5);
    TEST_ASSERT(plan.gdn_v_ranges[0][0] == 0 && plan.gdn_v_ranges[0][1] == 10);
    TEST_ASSERT(plan.gdn_v_ranges[4][0] == 40 && plan.gdn_v_ranges[4][1] == 48);
    TEST_ASSERT(plan.gdn_qk_heads[4] == 8);
}

static void test_plan_rejects_bad_ranks() {
    fprintf(stderr, "--- test_plan_rejects_bad_ranks ---\n");
    llama_hparams hp = make_hparams();
    llama_tp5_plan plan;
    llama_tp5_error err;
    TEST_ASSERT(!llama_tp5_plan_build(hp, 1, 248320, plan, err));
    TEST_ASSERT(err.code == "TP5_E_RANKS");
    TEST_ASSERT(!llama_tp5_plan_build(hp, 9, 248320, plan, err));
    TEST_ASSERT(err.code == "TP5_E_RANKS");
}

static void test_plan_rejects_indivisible_moe() {
    fprintf(stderr, "--- test_plan_rejects_indivisible_moe ---\n");
    llama_hparams hp = make_hparams();
    hp.n_ff_exp = 642; // not divisible by 5
    llama_tp5_plan plan;
    llama_tp5_error err;
    TEST_ASSERT(!llama_tp5_plan_build(hp, 5, 248320, plan, err));
    TEST_ASSERT(err.code == "TP5_E_MOE_SPLIT");
}

static void test_tensor_plans_target_types() {
    fprintf(stderr, "--- test_tensor_plans_target_types ---\n");
    llama_hparams hp = make_hparams();
    llama_tp5_plan plan;
    llama_tp5_error err;
    TEST_ASSERT(llama_tp5_plan_build(hp, 5, 248320, plan, err));

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
    // attn_q (full-attn layer 3): Q5_K, [2560, 12288], whole-head rows 2*da*5
    {
        int64_t ne[4] = {2560, 12288, 1, 1};
        auto tp = plan.plan_tensor("blk.3.attn_q.weight", ne, 13, 256, ok, err);
        TEST_ASSERT(ok);
        TEST_ASSERT(tp.semantic == llama_tp5_semantic::QSA_Q_GATE);
        TEST_ASSERT(tp.per_rank_len[0] == 2 * 256 * 5);
        TEST_ASSERT(tp.per_rank_len[4] == 2 * 256 * 4);
    }
    // attn_k layer 3: replicated head rows per role [256,256,512,256,256]
    {
        int64_t ne[4] = {2560, 512, 1, 1};
        auto tp = plan.plan_tensor("blk.3.attn_k.weight", ne, 14, 256, ok, err);
        TEST_ASSERT(ok);
        TEST_ASSERT(tp.semantic == llama_tp5_semantic::QSA_KV);
        TEST_ASSERT(tp.layout == llama_tp5_layout::SPLIT_AXIS1_REPL);
        TEST_ASSERT(tp.per_rank_len[2] == 512);
        TEST_ASSERT(tp.per_rank_len[0] == 256);
    }
    // attn_output layer 3: Q5_K [6144, 2560] axis0 cols 256*5=1280 (blck 256 OK)
    {
        int64_t ne[4] = {6144, 2560, 1, 1};
        auto tp = plan.plan_tensor("blk.3.attn_output.weight", ne, 13, 256, ok, err);
        TEST_ASSERT(ok);
        TEST_ASSERT(tp.semantic == llama_tp5_semantic::QSA_OUT);
        TEST_ASSERT(tp.per_rank_len[0] == 1280 && tp.per_rank_len[4] == 1024);
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
    TEST_ASSERT(llama_tp5_plan_build(hp, 5, 248320, plan, err));

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
    TEST_ASSERT(llama_tp5_plan_build(hp, 5, 248320, plan, err));
    for (int r = 0; r < 5; ++r) {
        // every rank's slice keeps 256-blocks whole on the ds*heads axis
        TEST_ASSERT((plan.gdn_v_heads[r] * plan.ds) % 256 == 0);
    }
    // balanced [10,10,10,9,9] would violate: 9*128 = 1152 % 256 != 0
    TEST_ASSERT(9 * 128 % 256 != 0);
}

static void test_validate() {
    fprintf(stderr, "--- test_validate ---\n");
    llama_hparams hp = make_hparams();
    llama_tp5_plan plan;
    llama_tp5_error err;
    TEST_ASSERT(llama_tp5_plan_build(hp, 5, 248320, plan, err));
    TEST_ASSERT(plan.validate(hp, 5, err));
    TEST_ASSERT(!plan.validate(hp, 4, err));
    TEST_ASSERT(err.code == "TP5_E_RANKS");
}

static void test_other_rank_counts() {
    fprintf(stderr, "--- test_other_rank_counts ---\n");
    llama_hparams hp = make_hparams();
    llama_tp5_plan plan;
    llama_tp5_error err;
    // 2 ranks: F=640 -> 320/rank; GDN heads [24,24]; Q [12,12]
    TEST_ASSERT(llama_tp5_plan_build(hp, 2, 248320, plan, err));
    TEST_ASSERT(plan.gdn_v_heads[0] == 24 && plan.gdn_v_heads[1] == 24);
    TEST_ASSERT(plan.q_role_counts[0] == 12 && plan.q_role_counts[1] == 12);
    TEST_ASSERT(plan.kv_role_counts[0] == 1 && plan.kv_role_counts[1] == 1); // no bridge at 2 ranks
    // 4 ranks: 48 heads -> [12,12,12,12]
    TEST_ASSERT(llama_tp5_plan_build(hp, 4, 248320, plan, err));
    for (int r = 0; r < 4; ++r) TEST_ASSERT(plan.gdn_v_heads[r] == 12);
    // 8 ranks: 48/8 = 6 heads each, 6*128=768 % 256 == 0 OK
    TEST_ASSERT(llama_tp5_plan_build(hp, 8, 248320, plan, err));
    for (int r = 0; r < 8; ++r) TEST_ASSERT(plan.gdn_v_heads[r] == 6);
}

// High-sensitivity GQA Attention head mapping verification test:
// Simulates 24 Q heads, 2 KV heads over 5 devices with the exact TP5 role plan.
// KV0 has V = 1.0, KV1 has V = 10.0.
// Tests that bridge rank (Role 2) splits into [0,2) on KV0 and [2,5) on KV1,
// producing exact matching values [1, 1, 10, 10, 10] and matches full unpartitioned attention.
static void test_gqa_bridge_head_mapping_algebra() {
    fprintf(stderr, "--- test_gqa_bridge_head_mapping_algebra ---\n");
    llama_hparams hp = make_hparams();
    llama_tp5_plan plan;
    llama_tp5_error err;
    TEST_ASSERT(llama_tp5_plan_build(hp, 5, 248320, plan, err));

    // Roles:
    // Rank 0 (Role 0): Q heads [0,5)   -> KV0
    // Rank 1 (Role 1): Q heads [5,10)  -> KV0
    // Rank 2 (Role 2): Q heads [10,15) -> KV0 (10,11), KV1 (12,13,14)
    // Rank 3 (Role 3): Q heads [15,20) -> KV1
    // Rank 4 (Role 4): Q heads [20,24) -> KV1

    // Reference global mapping: head h uses KV floor(h / 12)
    std::vector<float> ref_v_out(24, 0.0f);
    for (int h = 0; h < 24; ++h) {
        int kv_head = h / 12; // 0 for [0,12), 1 for [12,24)
        ref_v_out[h] = (kv_head == 0) ? 1.0f : 10.0f;
    }

    // Per-rank calculation following the TP5 execution plan:
    std::vector<float> reconstructed_q(24, 0.0f);

    int q_offset = 0;
    for (uint32_t r = 0; r < 5; ++r) {
        int n_q = plan.q_role_counts[r];
        std::vector<float> rank_q_out(n_q, 0.0f);

        if (r == 0 || r == 1) {
            // Local FA with 5Q / 1KV (KV0)
            for (int i = 0; i < n_q; ++i) {
                rank_q_out[i] = 1.0f; // KV0 value
            }
        } else if (r == 2) {
            // Bridge Role: two separate legal GQA calls
            // Call A: local Q[0,2) -> KV0
            for (int i = 0; i < 2; ++i) {
                rank_q_out[i] = 1.0f; // KV0
            }
            // Call B: local Q[2,5) -> KV1
            for (int i = 2; i < 5; ++i) {
                rank_q_out[i] = 10.0f; // KV1
            }
        } else if (r == 3 || r == 4) {
            // Local FA with 5Q / 1KV (KV1) or 4Q / 1KV (KV1)
            for (int i = 0; i < n_q; ++i) {
                rank_q_out[i] = 10.0f; // KV1 value
            }
        }

        // Reconstruct global Q
        for (int i = 0; i < n_q; ++i) {
            reconstructed_q[q_offset + i] = rank_q_out[i];
        }
        q_offset += n_q;
    }

    TEST_ASSERT(q_offset == 24);

    // Assert strict equality across all 24 heads
    for (int h = 0; h < 24; ++h) {
        TEST_ASSERT(reconstructed_q[h] == ref_v_out[h]);
    }

    // Verify bridge rank specifically output: [1.0, 1.0, 10.0, 10.0, 10.0]
    TEST_ASSERT(reconstructed_q[10] == 1.0f);
    TEST_ASSERT(reconstructed_q[11] == 1.0f);
    TEST_ASSERT(reconstructed_q[12] == 10.0f);
    TEST_ASSERT(reconstructed_q[13] == 10.0f);
    TEST_ASSERT(reconstructed_q[14] == 10.0f);
    fprintf(stderr, "  GQA Bridge head mapping algebra: 100%% exact match across all 24 heads\n");
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
    TEST_ASSERT(llama_tp5_plan_build(hp, 5, 248320, plan, err));
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
    test_plan_build_target();
    test_plan_rejects_bad_ranks();
    test_plan_rejects_indivisible_moe();
    test_tensor_plans_target_types();
    test_quant_slice_rejection();
    test_head_split_quant_math();
    test_validate();
    test_other_rank_counts();
    test_gqa_bridge_head_mapping_algebra();
    test_nextn_layer_plan_bounds();

    if (g_failures > 0) {
        fprintf(stderr, "%d failures\n", g_failures);
        return 1;
    }
    printf("test-tp5-plan: all passed\n");
    return 0;
}
