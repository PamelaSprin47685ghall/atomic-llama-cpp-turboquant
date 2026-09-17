// TP5 plan CPU tests: role tables, per-rank slice rules, quant-block legality.
// Pure CPU; no GPU, no GGUF, no Vulkan. Mirrors TP5.md sections 3.3, 4.4, 6.3,
// 7.1, 8.1 and the checks listed in 22.1.

#include "llama-tp5-plan.h"
#include "llama-hparams.h"
#include "llama-model.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"

#include "ggml.h"

#include <memory>
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

static void test_tp5_split_state_gdn_qkv(bool replicate_attention) {
    fprintf(stderr, "--- test_tp5_split_state_gdn_qkv (replicate=%d) ---\n", replicate_attention);
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
