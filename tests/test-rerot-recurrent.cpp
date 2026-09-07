#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "llama-graph.h"
#include "models/models.h"
#include "llama-batch.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <new>
#include <vector>

// RERoT hybrid-recurrent parking COW (planning guide Stage 3 / §6):
//
//   parent S -> seq_cp_recurrent -> parked A/B/C (+ high-id H/D)
//   A admission (find_slot) must not disturb B/C/D/H/S state bytes, and vice
//   versa. Retiring the parent exec id must not disturb parked siblings, the
//   freed id must be reusable, and a recursive fork (A -> A1) must keep its
//   own lineage. Logical seq ids span [0, LLAMA_MAX_SEQ) while the physical
//   state tensors stay at mem_size cells.
//
// Hermetic: a stub model provides hparams only; no model file is needed.

static int g_failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        ++g_failures; \
    } \
} while (0)

struct stub_model : public llama_model {
    stub_model() : llama_model(llama_model_default_params()) {}
    void load_stats(llama_model_loader &) override {}
    void load_hparams(llama_model_loader &) override {}
    void load_vocab(llama_model_loader &) override {}
    bool load_tensors(llama_model_loader &) override { return true; }
    void load_arch_hparams(llama_model_loader &) override {}
    void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override {
        return nullptr;
    }
};

// Build a single-sequence ubatch with explicit backing storage. The ubatch
// borrows its table storage from balloc (via ub.data) and the seq ids from
// id_store; both must outlive the find_slot() call.
static llama_ubatch make_ubatch(
        llama_batch_allocr & balloc,
        llama_seq_id seq,
        const std::vector<llama_pos> & positions,
        std::vector<llama_seq_id> & id_store) {
    llama_ubatch ub = balloc.ubatch_reserve((uint32_t) positions.size(), 1);
    id_store.assign(positions.size(), seq);
    for (size_t i = 0; i < positions.size(); ++i) {
        ub.token[i] = 1;
        ub.pos[i] = positions[i];
        ub.n_seq_id[i] = 1;
        ub.seq_id[i] = &id_store[i];
        ub.output[i] = 0;
    }
    return ub;
}

// Admission path: exactly what llama_memory_recurrent_context::apply() runs.
static bool admit(llama_memory_recurrent & mem, llama_seq_id seq, llama_pos pos0, uint32_t n_tokens) {
    llama_batch_allocr balloc(1);
    std::vector<llama_seq_id> ids;
    std::vector<llama_pos> positions;
    for (uint32_t k = 0; k < n_tokens; ++k) {
        positions.push_back(pos0 + (llama_pos) k);
    }
    llama_ubatch ub = make_ubatch(balloc, seq, positions, ids);
    return mem.find_slot(ub);
}

static int32_t resolved_row(const llama_memory_recurrent & mem, llama_seq_id seq, llama_pos & pos_out) {
    pos_out = -1;
    if (seq < 0 || (size_t) seq >= mem.tails.size()) {
        return -1;
    }
    const int32_t tail = mem.tails[(size_t) seq];
    if (tail < 0) {
        return -1;
    }
    const auto & cell = mem.cells[(size_t) tail];
    pos_out = cell.pos;
    return cell.src >= 0 ? cell.src : tail;
}

// Full observable state of one logical sequence: tail position plus the
// resolved R/S tensor row bytes on every layer (same resolution rule as
// state_write, so equality means the bytes a checkpoint would capture).
static std::vector<uint8_t> snap_seq(const llama_memory_recurrent & mem, llama_seq_id seq) {
    std::vector<uint8_t> out;
    auto append = [&](const void * p, size_t n) {
        const auto * b = (const uint8_t *) p;
        out.insert(out.end(), b, b + n);
    };
    llama_pos pos = -1;
    const int32_t row = resolved_row(mem, seq, pos);
    append(&pos, sizeof(pos));
    append(&row, sizeof(row));
    if (row < 0) {
        return out;
    }
    for (size_t il = 0; il < mem.r_l.size(); ++il) {
        if (mem.r_l[il] != nullptr) {
            const size_t row_size = ggml_row_size(mem.r_l[il]->type, mem.r_l[il]->ne[0]);
            std::vector<uint8_t> buf(row_size);
            ggml_backend_tensor_get(mem.r_l[il], buf.data(), (size_t) row * row_size, row_size);
            append(buf.data(), buf.size());
        }
        if (mem.s_l[il] != nullptr) {
            const size_t row_size = ggml_row_size(mem.s_l[il]->type, mem.s_l[il]->ne[0]);
            std::vector<uint8_t> buf(row_size);
            ggml_backend_tensor_get(mem.s_l[il], buf.data(), (size_t) row * row_size, row_size);
            append(buf.data(), buf.size());
        }
    }
    return out;
}

// Simulate a committed write to one lane's exclusive state.
static void stamp_seq(llama_memory_recurrent & mem, llama_seq_id seq, uint8_t fill_r, uint8_t fill_s) {
    llama_pos pos = -1;
    const int32_t row = resolved_row(mem, seq, pos);
    CHECK(row >= 0);
    if (row < 0) {
        return;
    }
    for (size_t il = 0; il < mem.r_l.size(); ++il) {
        if (mem.r_l[il] != nullptr) {
            const size_t row_size = ggml_row_size(mem.r_l[il]->type, mem.r_l[il]->ne[0]);
            const std::vector<uint8_t> buf(row_size, fill_r);
            ggml_backend_tensor_set(mem.r_l[il], buf.data(), (size_t) row * row_size, row_size);
        }
        if (mem.s_l[il] != nullptr) {
            const size_t row_size = ggml_row_size(mem.s_l[il]->type, mem.s_l[il]->ne[0]);
            const std::vector<uint8_t> buf(row_size, fill_s);
            ggml_backend_tensor_set(mem.s_l[il], buf.data(), (size_t) row * row_size, row_size);
        }
    }
}

// --- Ordinary grouped prefill regression (cold --rerot, request rerot:false) ---
//
// A grouped context with no episode write tag must produce native
// single-sequence recurrence: the two ordinary public/private mirror brain
// rows and the F16 hand must not change the readout. Uses the production
// graph inputs (build_rs_inp + set_inputs) and the production state prep /
// build_recurrent_attn path, not copied forward equations. S=16 hits the
// supported fused-GDN kernels so the same comparison can run on Vulkan
// where available (graph ordering/backend), with a CPU-only hermetic core.
static float ord_det(uint64_t x) {
    return float(int64_t(x * 2654435761ULL % 17) - 8) * 0.05f;
}

static float ord_max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float m = 0.0f;
    const size_t n = std::min(a.size(), b.size());
    CHECK(a.size() == b.size());
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, std::abs(a[i] - b[i]));
    }
    return m;
}

struct ord_step_out {
    std::vector<float> output;
    std::vector<float> eff; // effective S state: plain row, or grouped brain(pub)+hand
    std::vector<float> brain_pub;
    std::vector<float> brain_priv;
    std::vector<float> hand_f32;
    int32_t brain_row = -1;
    int32_t hand_row = -1;
};

// One production recurrent step for mem+ubatch on CPU. Builds the graph via
// build_rs_inp (production rbb_groups/brain_copy topology) + production
// state prep (build_rs_shared + F16 get_rows + cast for shared, build_rs
// for native) + build_recurrent_attn, then set_inputs + compute. Commits
// land in the real mem tensors as a side effect; outputs + effective state
// are read back for grouped-vs-plain comparison.
static ord_step_out ord_run_mem_step(
        stub_model & model,
        llama_memory_recurrent & mem,
        const llama_ubatch & ubatch,
        int il,
        int64_t S, int64_t H,
        const std::vector<float> & qd,
        const std::vector<float> & kd,
        const std::vector<float> & vd,
        const std::vector<float> & gd,
        const std::vector<float> & bd) {
    ord_step_out out;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs = ubatch.n_seqs;
    const int64_t D = (int64_t) model.hparams.n_embd_s();
    CHECK((int64_t) qd.size() == S * H * n_seq_tokens * n_seqs);
    CHECK((int64_t) kd.size() == S * H * n_seq_tokens * n_seqs);
    CHECK((int64_t) vd.size() == S * H * n_seq_tokens * n_seqs);
    CHECK((int64_t) gd.size() == H * n_seq_tokens * n_seqs);
    CHECK((int64_t) bd.size() == H * n_seq_tokens * n_seqs);

    llama_memory_recurrent_context mctx(&mem, {ubatch});
    CHECK(mctx.apply());

    llm_graph_result res(4096);
    llm_graph_params params{};
    params.hparams = model.hparams;
    params.cparams.fused_gdn_ar = true;
    params.cparams.fused_gdn_ch = true;
    params.ubatch = ubatch;
    params.mctx = &mctx;
    params.res = &res;
    params.n_outputs = ubatch.n_tokens;
    llm_build_delta_net_base builder(params);
    llm_graph_input_rs * inp = builder.build_rs_inp();
    if (mctx.is_grouped() && mem.seq_episode[(size_t) ubatch.seq_id[0][0]] == 0) {
        // Ordinary root (episode 0): ready private-planner mirror keeps two
        // single-row groups; the fused all-writer fast path must stay off.
        CHECK(inp->rbb_groups.size() == 2);
        CHECK(mctx.public_brain_groups().size() == 2);
        CHECK(!builder.uses_parallel_delta(inp, il));
    } else if (!mctx.is_grouped()) {
        CHECK(inp->rbb_groups.empty());
    }

    ggml_context * ctx0 = res.get_ctx();
    ggml_tensor * ssm = mctx.get_s_l(il);
    CHECK(ssm != nullptr);
    ggml_tensor * state_base = nullptr;
    ggml_tensor * hand_all = nullptr;
    ggml_tensor * state4d = nullptr;
    if (mctx.is_s_shared(il)) {
        hand_all = mctx.get_d_l(il);
        CHECK(hand_all != nullptr);
        state_base = builder.build_rs_shared(inp, ssm, (int32_t) D, (int32_t) n_seqs);
        ggml_tensor * hand_echo = builder.build_rs(inp, hand_all, (int32_t) D, (int32_t) n_seqs);
        ggml_tensor * state = ggml_add(ctx0, state_base, ggml_cast(ctx0, hand_echo, GGML_TYPE_F32));
        state4d = ggml_reshape_4d(ctx0, state, S, S, H, n_seqs);
    } else {
        ggml_tensor * gathered = builder.build_rs(inp, ssm, (int32_t) D, (int32_t) n_seqs);
        state4d = ggml_reshape_4d(ctx0, gathered, S, S, H, n_seqs);
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, S, H, n_seq_tokens, n_seqs);
    ggml_tensor * k = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, S, H, n_seq_tokens, n_seqs);
    ggml_tensor * v = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, S, H, n_seq_tokens, n_seqs);
    ggml_tensor * g = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, H, n_seq_tokens, n_seqs);
    ggml_tensor * b = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, H, n_seq_tokens, n_seqs);
    ggml_tensor * output = builder.build_recurrent_attn(
        inp, ssm, state_base, hand_all, q, k, v, g, b, state4d, il);
    ggml_build_forward_expand(res.get_gf(), output);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx0, cpu);
    CHECK(buf != nullptr);
    if (!buf) {
        ggml_backend_free(cpu);
        return out;
    }
    res.set_inputs(&ubatch);
    ggml_backend_tensor_set(q, qd.data(), 0, qd.size() * sizeof(float));
    ggml_backend_tensor_set(k, kd.data(), 0, kd.size() * sizeof(float));
    ggml_backend_tensor_set(v, vd.data(), 0, vd.size() * sizeof(float));
    ggml_backend_tensor_set(g, gd.data(), 0, gd.size() * sizeof(float));
    ggml_backend_tensor_set(b, bd.data(), 0, bd.size() * sizeof(float));
    CHECK(ggml_backend_graph_compute(cpu, res.get_gf()) == GGML_STATUS_SUCCESS);

    out.output.resize((size_t) ggml_nelements(output));
    ggml_backend_tensor_get(output, out.output.data(), 0, out.output.size() * sizeof(float));

    const llama_seq_id seq = ubatch.seq_id[0][0];
    if (mctx.is_s_shared(il)) {
        const int32_t brain_pub = mctx.brain_copy(0);
        CHECK(brain_pub >= 0);
        const size_t row_size = ggml_row_size(ssm->type, ssm->ne[0]);
        out.brain_pub.resize((size_t) D);
        ggml_backend_tensor_get(ssm, out.brain_pub.data(), (size_t) brain_pub * row_size, (size_t) D * sizeof(float));
        const int32_t brain_priv = brain_pub + (int32_t) mem.n_brain_rows;
        out.brain_priv.resize((size_t) D);
        ggml_backend_tensor_get(ssm, out.brain_priv.data(), (size_t) brain_priv * row_size, (size_t) D * sizeof(float));
        llama_pos pos = -1;
        const int32_t hand_row = resolved_row(mem, seq, pos);
        CHECK(hand_row >= 0);
        ggml_tensor * d = mem.d_l[(size_t) il];
        const size_t hrow_size = ggml_row_size(d->type, d->ne[0]);
        std::vector<ggml_fp16_t> hraw((size_t) D);
        ggml_backend_tensor_get(d, hraw.data(), (size_t) hand_row * hrow_size, hrow_size);
        out.hand_f32.resize((size_t) D);
        for (int64_t i = 0; i < D; ++i) {
            out.hand_f32[i] = ggml_fp16_to_fp32(hraw[i]);
        }
        out.eff.resize((size_t) D);
        for (int64_t i = 0; i < D; ++i) {
            out.eff[i] = out.brain_pub[i] + out.hand_f32[i];
        }
        out.brain_row = brain_pub;
        out.hand_row = hand_row;
    } else {
        llama_pos pos = -1;
        const int32_t row = resolved_row(mem, seq, pos);
        CHECK(row >= 0);
        const size_t row_size = ggml_row_size(ssm->type, ssm->ne[0]);
        out.eff.resize((size_t) D);
        ggml_backend_tensor_get(ssm, out.eff.data(), (size_t) row * row_size, (size_t) D * sizeof(float));
        out.brain_row = row;
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(cpu);
    return out;
}

// Standalone fused-GDN parity for backend ordering (no mem tensors, so the
// same math can run on CPU and on Vulkan where available). Returns empty +
// supported=false when the backend lacks the fused op.
static std::vector<float> ord_run_standalone_gdn(
        stub_model & model,
        int64_t S, int64_t H, int64_t N,
        const std::vector<float> & qd,
        const std::vector<float> & kd,
        const std::vector<float> & vd,
        const std::vector<float> & gd,
        const std::vector<float> & bd,
        const std::vector<float> & sd,
        ggml_backend_t backend, bool & supported) {
    supported = false;
    std::vector<float> out;
    llm_graph_result res(4096);
    llm_graph_params params{};
    params.hparams = model.hparams;
    params.cparams.fused_gdn_ar = true;
    params.cparams.fused_gdn_ch = true;
    params.res = &res;
    params.n_outputs = (uint32_t) N;
    llm_build_delta_net_base builder(params);
    ggml_context * ctx0 = res.get_ctx();
    ggml_tensor * q = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, S, H, N, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, S, H, N, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, S, H, N, 1);
    ggml_tensor * g = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, H, N, 1);
    ggml_tensor * b = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, H, N, 1);
    ggml_tensor * s = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, S, S, H, 1);
    auto pr = builder.build_delta_net(q, k, v, g, b, s, 3);
    ggml_build_forward_expand(res.get_gf(), pr.first);
    ggml_build_forward_expand(res.get_gf(), pr.second);
    if (!ggml_backend_supports_op(backend, pr.first)) {
        return out;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx0, backend);
    if (!buf) {
        return out;
    }
    ggml_backend_tensor_set(q, qd.data(), 0, qd.size() * sizeof(float));
    ggml_backend_tensor_set(k, kd.data(), 0, kd.size() * sizeof(float));
    ggml_backend_tensor_set(v, vd.data(), 0, vd.size() * sizeof(float));
    ggml_backend_tensor_set(g, gd.data(), 0, gd.size() * sizeof(float));
    ggml_backend_tensor_set(b, bd.data(), 0, bd.size() * sizeof(float));
    ggml_backend_tensor_set(s, sd.data(), 0, sd.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, res.get_gf()) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        return out;
    }
    supported = true;
    out.resize((size_t) ggml_nelements(pr.first));
    ggml_backend_tensor_get(pr.first, out.data(), 0, out.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    return out;
}

// Execute the production mixed-group graph on a single backend. Two people,
// two public writers for person A, one for B, plus PRIVATE and PENDING rows.
// Orthogonal public keys give an independent scalar oracle. Current-token
// readout must remain the native per-Lane transition while the shared block
// commit is stored only for future frontiers.
static void test_mixed_frontier_graph(ggml_backend_t backend, bool reversed) {
    constexpr int S = 16, H = 2, N = 5, D = S * S * H;
    stub_model model;
    model.hparams.n_layer_all = 4;
    model.hparams.n_embd = 8;
    model.hparams.n_embd_r_impl = 16;
    model.hparams.ssm_d_state = S;
    model.hparams.ssm_d_inner = S * H;
    llama_memory_recurrent mem(model, GGML_TYPE_F32, GGML_TYPE_F32,
        false, N, 16, 0, 2, N, nullptr);
    llama_batch_allocr alloc(1);
    llama_ubatch ub = alloc.ubatch_reserve(1, N);
    llama_seq_id ids[N];
    std::vector<int32_t> groups[2];
    for (int row = 0; row < N; ++row) {
        const int origin = reversed ? N - 1 - row : row;
        ids[row] = origin;
        ub.token[row] = 1;
        ub.pos[row] = 0;
        ub.n_seq_id[row] = 1;
        ub.seq_id[row] = &ids[row];
        ub.output[row] = 1;
        if (origin < 2) groups[0].push_back(row);
        if (origin == 3) groups[1].push_back(row);
    }
    llama_memory_recurrent_context mctx(&mem, {ub});
    llm_graph_result res(2048);
    llm_graph_params params{};
    params.hparams = model.hparams;
    params.ubatch = ub;
    params.res = &res;
    params.cparams.fused_gdn_ar = true;
    params.n_outputs = N;
    llm_build_delta_net_base builder(params);
    ggml_context * ctx = res.get_ctx();
    llm_graph_input_rs inp(&mctx);
    inp.brain_copy = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    for (int group = 0; group < 2; ++group) {
        inp.rbb_groups.push_back({group,
            ggml_new_tensor_1d(ctx, GGML_TYPE_I32, groups[group].size())});
    }
    auto * brain = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 4);
    auto * hand = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D, N);
    auto * base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, N);
    auto * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, N);
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, 1, N);
    auto * k = ggml_dup_tensor(ctx, q);
    auto * v = ggml_dup_tensor(ctx, q);
    auto * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, 1, N);
    auto * beta = ggml_dup_tensor(ctx, g);
    const char * old_mode = std::getenv("LLAMA_REROT_RBB_ABLATION");
    const std::string old_mode_copy = old_mode ? old_mode : "";
    setenv("LLAMA_REROT_RBB_ABLATION", "shared-rbb", 1);
    auto * output = builder.build_recurrent_attn(
        &inp, brain, base, hand, q, k, v, g, beta, state, 3);
    if (old_mode) setenv("LLAMA_REROT_RBB_ABLATION", old_mode_copy.c_str(), 1);
    else unsetenv("LLAMA_REROT_RBB_ABLATION");
    ggml_build_forward_expand(res.get_gf(), output);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(buffer != nullptr);
    if (!buffer) return;

    std::vector<float> bdata(D * 4, 9.0f), bases(D * N), states(D * N);
    std::vector<float> qs(S * H * N, 0), ks(qs), vs(qs), gs(H * N), bs(H * N);
    auto old_brain = [](int person, int head) { return (person == 0 ? 0.75 : 2.0) + 0.01 * head; };
    auto gate = [](int origin, int head) { return float(-0.1 * (origin + 1) - 0.02 * head); };
    auto rate = [](int origin, int head) { return float(0.2 + 0.1 * origin + 0.01 * head); };
    auto value = [](int origin, int col) { return float(origin + 1 + 0.1 * col + (origin == 2 || origin == 4 ? 100 : 0)); };
    auto key_row = [](int origin) { return origin == 3 ? 0 : (origin == 4 ? 3 : origin); };
    auto query_row = [&](int origin) { return origin < 2 ? 1 - origin : key_row(origin); };
    for (int person = 0; person < 2; ++person) {
        for (int h = 0; h < H; ++h) {
            std::fill_n(bdata.data() + person * D + h * S * S, S * S, float(old_brain(person, h)));
        }
    }
    std::vector<float> expected_brain = bdata;
    for (int row = 0; row < N; ++row) {
        const int origin = ids[row], person = origin < 3 ? 0 : 1;
        for (int h = 0; h < H; ++h) {
            const int off = (row * H + h) * S;
            qs[off + query_row(origin)] = 1.0f;
            ks[off + key_row(origin)] = 1.0f;
            gs[row * H + h] = gate(origin, h);
            bs[row * H + h] = rate(origin, h);
            for (int col = 0; col < S; ++col) {
                vs[off + col] = value(origin, col);
                for (int d = 0; d < S; ++d) {
                    const int index = row * D + (h * S + col) * S + d;
                    bases[index] = float(old_brain(person, h));
                    states[index] = bases[index] + 0.125f * (origin + 1);
                }
            }
        }
    }
    for (int person = 0; person < 2; ++person) {
        for (int h = 0; h < H; ++h) {
            const double decay = std::exp(person == 0
                ? (double(gate(0, h)) + gate(1, h)) / 2.0 : double(gate(3, h)));
            for (int col = 0; col < S; ++col) {
                for (int d = 0; d < S; ++d) {
                    double next = decay * float(old_brain(person, h));
                    for (int origin : (person == 0 ? std::vector<int>{0, 1} : std::vector<int>{3})) {
                        if (d == key_row(origin)) {
                            const double r = rate(origin, h);
                            next += r / (person == 0 ? 1.0 + 1.0e-4 * r : 1.0) *
                                (value(origin, col) - decay * float(old_brain(person, h)));
                        }
                    }
                    expected_brain[person * D + (h * S + col) * S + d] = float(next);
                }
            }
        }
    }
    auto set = [](ggml_tensor * t, const std::vector<float> & data) {
        ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
    };
    set(brain, bdata); set(base, bases); set(state, states);
    set(q, qs); set(k, ks); set(v, vs); set(g, gs); set(beta, bs);
    for (int group = 0; group < 2; ++group) {
        ggml_backend_tensor_set(inp.rbb_groups[group].public_rows, groups[group].data(),
            0, groups[group].size() * sizeof(int32_t));
    }
    CHECK(ggml_backend_graph_compute(backend, res.get_gf()) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    std::vector<float> actual_brain(D * 4), actual_output(S * H * N);
    std::vector<ggml_fp16_t> actual_hand(D * N);
    ggml_backend_tensor_get(brain, actual_brain.data(), 0, actual_brain.size() * sizeof(float));
    ggml_backend_tensor_get(output, actual_output.data(), 0, actual_output.size() * sizeof(float));
    ggml_backend_tensor_get(hand, actual_hand.data(), 0, actual_hand.size() * sizeof(ggml_fp16_t));
    const float brain_err = ord_max_abs_diff(actual_brain, expected_brain);
    double output_err = 0, hand_err = 0;
    for (int row = 0; row < N; ++row) {
        const int origin = ids[row], person = origin < 3 ? 0 : 1;
        const bool is_public = origin < 2 || origin == 3;
        for (int h = 0; h < H; ++h) {
            const double alpha = std::exp(double(gate(origin, h)));
            const double r = rate(origin, h);
            for (int col = 0; col < S; ++col) {
                for (int d = 0; d < S; ++d) {
                    const int offset = (h * S + col) * S + d;
                    const int index = row * D + offset;
                    const double B = bases[index], E = states[index] - B;
                    const double native = alpha * states[index] + (d == key_row(origin)
                        ? r * (value(origin, col) - alpha * states[index]) : 0.0);
                    double expected_hand = native - B;
                    if (is_public) {
                        const int n_public = person == 0 ? 2 : 1;
                        const double decay = std::exp(person == 0
                            ? (double(gate(0, h)) + gate(1, h)) / 2.0 : double(gate(3, h)));
                        const double local_hand = alpha * E *
                            (d == key_row(origin) ? 1.0 - r : 1.0);
                        const double shared_write =
                            expected_brain[person * D + offset] - decay * old_brain(person, h);
                        const double self_write = d == key_row(origin)
                            ? r / (person == 0 ? 1.0 + 1.0e-4 * r : 1.0) *
                                (value(origin, col) - decay * old_brain(person, h))
                            : 0.0;
                        const double self_echo = n_public == 1 ? 0.0 :
                            self_write - shared_write / n_public;
                        expected_hand = local_hand + self_echo;
                    }
                    CHECK(std::isfinite(ggml_fp16_to_fp32(actual_hand[index])));
                    hand_err = std::max(hand_err, std::abs(ggml_fp16_to_fp32(actual_hand[index]) - expected_hand));
                    if (d == query_row(origin)) {
                        const double expected = native / std::sqrt(double(S));
                        const float actual = actual_output[(row * H + h) * S + col];
                        CHECK(std::isfinite(actual));
                        output_err = std::max(output_err, std::abs(actual - expected));
                    }
                }
            }
        }
    }
    CHECK(brain_err < 2e-5f);
    CHECK(output_err < 2e-5);
    CHECK(hand_err < 0.04); // FP16 private deltas contain values around 60.
    std::printf("mixed frontier %s reversed=%d: brain=%g output=%g hand=%g\n",
        ggml_backend_name(backend), reversed, brain_err, output_err, hand_err);
    ggml_backend_buffer_free(buffer);
}

// Non-grouped graph inputs have no brain_copy tensor. Force nonzero recycled
// storage so this regression does not depend on whether malloc returns zeros.
static void test_native_input_recycled_storage() {
    stub_model model;
    model.hparams.n_layer_all = 1;
    model.hparams.n_embd = 8;
    model.hparams.n_embd_r_impl = 16;
    model.hparams.ssm_d_state = 4;
    model.hparams.ssm_d_inner = 8;
    llama_memory_recurrent mem(model, GGML_TYPE_F32, GGML_TYPE_F32,
        false, 2, 2, 0, 0, 0, nullptr);
    llama_batch_allocr alloc(1);
    std::vector<llama_seq_id> ids;
    const auto ub = make_ubatch(alloc, 0, {0}, ids);
    llama_memory_recurrent_context mctx(&mem, {ub});
    CHECK(mctx.apply());
    alignas(llm_graph_input_rs) unsigned char storage[sizeof(llm_graph_input_rs)];
    std::fill_n(storage, sizeof(storage), 0x45);
    auto * inp = new (storage) llm_graph_input_rs(&mctx);
    // Exercise the real setter, which used to dereference 0x45454545... .
    inp->set_input(&ub);
    inp->~llm_graph_input_rs();
    std::puts("native input recycled storage: PASS");
}

// A hand checkpoint is an exact local frontier, not just some tensor bytes.
// Rewind must restore its position and detach a shared tail before any write;
// malformed input must leave both bytes and ownership untouched.
static void test_fence_replay_recurrent_oracle() {
    stub_model model;
    model.hparams.n_layer_all = 4;
    model.hparams.n_embd = 8;
    model.hparams.n_embd_r_impl = 16;
    model.hparams.ssm_d_state = 16;
    model.hparams.ssm_d_inner = 32;
    constexpr int S = 16, H = 2, D = S * S * H, IL = 3;
    llama_memory_recurrent mem(model, GGML_TYPE_F32, GGML_TYPE_F32,
        false, 4, 8, 0, 2, 4, nullptr);
    std::vector<float> q(S * H, 0.0f), k(q), v(S * H, 0.25f);
    std::vector<float> g(H, std::log(0.7f)), b(H, 0.4f);
    for (int h = 0; h < H; ++h) {
        q[h * S] = k[h * S] = 1.0f;
    }
    auto step = [&](llama_pos pos) {
        llama_batch_allocr alloc(1);
        std::vector<llama_seq_id> ids;
        const auto ub = make_ubatch(alloc, 0, {pos}, ids);
        return ord_run_mem_step(model, mem, ub, IL, S, H, q, k, v, g, b);
    };
    step(0);
    llama_kv_rerot_meta tag;
    tag.episode_id = 999;
    tag.node_id = 1;
    tag.visibility = llama_rerot_visibility::private_control;
    CHECK(mem.rerot_set_write_tag(0, tag));
    std::vector<ggml_fp16_t> hand(D);
    for (int i = 0; i < D; ++i) {
        hand[i] = ggml_fp32_to_fp16(0.03125f * float(i % 7 - 3));
    }
    ggml_backend_tensor_set(mem.d_l[IL], hand.data(),
        size_t(mem.tails[0]) * D * sizeof(ggml_fp16_t), D * sizeof(ggml_fp16_t));
    const auto checkpoint = mem.capture_hand_seed(1, 0);
    CHECK(checkpoint && checkpoint->source_pos == 0);
    step(1);
    step(2); // Original closing sequence has already advanced the local state.

    // A sibling's last PUBLIC commit changes B while this child closes.
    // Replay must use this new B and the PRE-marker H, not old B or post-marker H.
    std::vector<float> brain(D);
    for (int i = 0; i < D; ++i) {
        brain[i] = 0.125f + 0.001f * float(i % 13);
    }
    const size_t brain_off = size_t(mem.episode_brain.at(999)) * D * sizeof(float);
    ggml_backend_tensor_set(mem.s_l[IL], brain.data(), brain_off, D * sizeof(float));
    CHECK(mem.apply_hand_seed(0, checkpoint));
    CHECK(mem.cells[mem.tails[0]].pos == 0);
    std::vector<double> expected(D);
    for (int i = 0; i < D; ++i) {
        expected[i] = double(brain[i]) + ggml_fp16_to_fp32(hand[i]);
    }
    double max_output_error = 0.0, max_state_error = 0.0;
    for (int t = 1; t <= 2; ++t) {
        const auto actual = step(t);
        for (int h = 0; h < H; ++h) {
            for (int col = 0; col < S; ++col) {
                const int base = (h * S + col) * S;
                const double decay = std::exp(double(g[h]));
                const double delta = b[h] * (v[h * S + col] - decay * expected[base]);
                for (int row = 0; row < S; ++row) {
                    const double next = decay * expected[base + row] + (row == 0 ? delta : 0.0);
                    if (row == 0) {
                        max_output_error = std::max(max_output_error,
                            std::abs(double(actual.output[h * S + col]) - next / std::sqrt(double(S))));
                    }
                    // The persistent hand plane is FP16; each real step rounds
                    // its overlay before reconstructing the next B+H input.
                    expected[base + row] = double(brain[base + row]) + ggml_fp16_to_fp32(
                        ggml_fp32_to_fp16(float(next - brain[base + row])));
                    max_state_error = std::max(max_state_error,
                        std::abs(double(actual.eff[base + row]) - expected[base + row]));
                }
            }
        }
        CHECK(actual.brain_pub == brain);
    }
    CHECK(max_output_error < 2e-5);
    CHECK(max_state_error < 2e-4);
    CHECK(mem.cells[mem.tails[0]].pos == 2);
    std::printf("fence recurrent replay: output=%g state=%g; public brain unchanged\n",
        max_output_error, max_state_error);
}

static void test_root_visibility_state_continuity() {
    stub_model model;
    model.hparams.n_layer_all = 4;
    model.hparams.n_embd = 8;
    model.hparams.n_embd_r_impl = 16;
    model.hparams.ssm_d_state = 4;
    model.hparams.ssm_d_inner = 8;
    llama_memory_recurrent mem(model, GGML_TYPE_F32, GGML_TYPE_F32,
        false, 4, 8, 0, 2, 4, nullptr);
    CHECK(admit(mem, 0, 0, 1));
    llama_kv_rerot_meta tag;
    tag.episode_id = 808;
    tag.node_id = 0;
    tag.visibility = llama_rerot_visibility::private_control;
    CHECK(mem.rerot_set_write_tag(0, tag));
    auto * brain = mem.s_l[3];
    auto * hand = mem.d_l[3];
    const size_t n = size_t(brain->ne[0]);
    const size_t public_off = size_t(mem.episode_brain.at(808)) * n * sizeof(float);
    const size_t private_off = public_off + mem.get_brain_capacity() * n * sizeof(float);
    const size_t hand_off = size_t(mem.tails[0]) * n * sizeof(ggml_fp16_t);
    const std::vector<float> public_values(n, 1.0f), private_values(n, 4.0f);
    const std::vector<ggml_fp16_t> initial_hand(n, ggml_fp32_to_fp16(2.0f));
    ggml_backend_tensor_set(brain, public_values.data(), public_off, n * sizeof(float));
    ggml_backend_tensor_set(brain, private_values.data(), private_off, n * sizeof(float));
    ggml_backend_tensor_set(hand, initial_hand.data(), hand_off, n * sizeof(ggml_fp16_t));
    auto check_hand = [&](float expected) {
        std::vector<ggml_fp16_t> values(n);
        ggml_backend_tensor_get(hand, values.data(), hand_off, n * sizeof(ggml_fp16_t));
        CHECK(std::all_of(values.begin(), values.end(), [=](ggml_fp16_t x) {
            return ggml_fp16_to_fp32(x) == expected;
        }));
    };
    tag.visibility = llama_rerot_visibility::public_live;
    CHECK(mem.rerot_set_write_tag(0, tag));
    check_hand(5.0f); // 4+2 = 1+5; switching a base is not a model transition.
    CHECK(mem.rerot_set_write_tag(0, tag));
    check_hand(5.0f); // Idempotent: reinstalling a tag cannot apply the offset twice.
    tag.visibility = llama_rerot_visibility::pending_record;
    CHECK(mem.rerot_set_write_tag(0, tag));
    check_hand(2.0f); // Switch back: 1+5 = 4+2.
    std::vector<float> actual(n);
    ggml_backend_tensor_get(brain, actual.data(), public_off, n * sizeof(float));
    CHECK(actual == public_values);
    ggml_backend_tensor_get(brain, actual.data(), private_off, n * sizeof(float));
    CHECK(actual == private_values);
}

static void test_hand_checkpoint_snapshot_capture() {
    stub_model model;
    model.hparams.n_layer_all = 4;
    model.hparams.n_embd = 8;
    model.hparams.n_embd_r_impl = 16;
    model.hparams.ssm_d_state = 4;
    model.hparams.ssm_d_inner = 8;
    llama_memory_recurrent mem(model, GGML_TYPE_F32, GGML_TYPE_F32,
        false, 4, 8, 2, 2, 4, nullptr);
    CHECK(admit(mem, 0, 0, 3));
    llama_kv_rerot_meta tag;
    tag.episode_id = 818;
    tag.node_id = 1;
    tag.visibility = llama_rerot_visibility::public_live;
    CHECK(mem.rerot_set_write_tag(0, tag));
    const size_t row = size_t(mem.tails[0]);
    const size_t saved_row = mem.size + row;
    for (size_t il = 0; il < mem.r_l.size(); ++il) {
        const size_t bytes = ggml_row_size(mem.r_l[il]->type, mem.r_l[il]->ne[0]);
        const std::vector<uint8_t> saved(bytes, 0x31);
        ggml_backend_tensor_set(mem.r_l[il], saved.data(), saved_row * bytes, bytes);
    }
    for (int il = 0; il < 3; ++il) {
        const size_t bytes = ggml_row_size(mem.s_l[il]->type, mem.s_l[il]->ne[0]);
        const std::vector<uint8_t> saved(bytes, 0x41);
        ggml_backend_tensor_set(mem.s_l[il], saved.data(), saved_row * bytes, bytes);
    }
    const size_t n = size_t(mem.s_l[3]->ne[0]);
    const size_t br = size_t(mem.episode_brain.at(818));
    const std::vector<float> current(n, 10.0f), saved(n, 4.0f);
    const std::vector<ggml_fp16_t> delta(n, ggml_fp32_to_fp16(2.0f));
    ggml_backend_tensor_set(mem.s_l[3], current.data(), br * n * sizeof(float), n * sizeof(float));
    ggml_backend_tensor_set(mem.s_l[3], saved.data(),
        (2 * mem.get_brain_capacity() + br) * n * sizeof(float), n * sizeof(float));
    ggml_backend_tensor_set(mem.d_l[3], delta.data(), saved_row * n * sizeof(ggml_fp16_t), n * sizeof(ggml_fp16_t));
    CHECK(mem.seq_rm(0, 2, -1));
    const auto checkpoint = mem.capture_hand_seed(1, 0);
    CHECK(checkpoint && checkpoint->source_pos == 1);
    CHECK(std::all_of(checkpoint->conv_tail_bytes[0].begin(), checkpoint->conv_tail_bytes[0].end(),
        [](uint8_t x) { return x == 0x31; }));
    CHECK(std::all_of(checkpoint->state_bytes[0].begin(), checkpoint->state_bytes[0].end(),
        [](uint8_t x) { return x == 0x41; }));
    CHECK(mem.apply_hand_seed(0, checkpoint));
    std::vector<ggml_fp16_t> actual(n);
    ggml_backend_tensor_get(mem.d_l[3], actual.data(), row * n * sizeof(ggml_fp16_t), n * sizeof(ggml_fp16_t));
    CHECK(std::all_of(actual.begin(), actual.end(), [](ggml_fp16_t x) {
        return ggml_fp16_to_fp32(x) == -4.0f; // saved 4+2 = current 10+(-4)
    }));
    CHECK(mem.rs_idx[0] == 0);
}

static void test_hand_checkpoint_restore() {
    stub_model model;
    model.hparams.n_layer_all = 2;
    model.hparams.n_embd = 8;
    model.hparams.n_embd_r_impl = 16;
    model.hparams.ssm_d_state = 4;
    model.hparams.ssm_d_inner = 8;
    llama_memory_recurrent mem(model, GGML_TYPE_F32, GGML_TYPE_F32,
        false, 4, 8, 0, 0, 0, nullptr);
    CHECK(admit(mem, 0, 0, 4));
    stamp_seq(mem, 0, 0x11, 0x22);
    auto seed = mem.capture_hand_seed(1, 0);
    CHECK(seed && seed->source_pos == 3);
    CHECK(admit(mem, 0, 4, 3));
    stamp_seq(mem, 0, 0x33, 0x44);
    mem.seq_cp(0, 1, -1, -1);
    const auto sibling = snap_seq(mem, 1);
    CHECK(mem.apply_hand_seed(0, seed));
    CHECK(snap_seq(mem, 1) == sibling);
    CHECK(mem.tails[0] != mem.tails[1]);
    CHECK(mem.cells[mem.tails[0]].pos == 3);
    const auto restored = mem.capture_hand_seed(2, 0);
    CHECK(restored->conv_tail_bytes == seed->conv_tail_bytes);
    CHECK(restored->state_bytes == seed->state_bytes);

    auto bad = std::make_shared<llama_memory_recurrent::hand_seed>(*seed);
    bad->conv_tail_bytes[0].assign(bad->conv_tail_bytes[0].size(), 0x55);
    bad->state_bytes.back().pop_back();
    const auto before = snap_seq(mem, 0);
    const auto used = mem.get_recurrent_used();
    CHECK(!mem.apply_hand_seed(0, bad));
    CHECK(snap_seq(mem, 0) == before);
    CHECK(!mem.apply_hand_seed(2, bad));
    CHECK(mem.tails[2] == -1);
    CHECK(mem.get_recurrent_used() == used);

    std::vector<uint8_t> blob;
    CHECK(mem.rerot_capture_hand_seed(0, blob));
    blob.push_back(0x7f);
    CHECK(!mem.rerot_apply_hand_seed(0, blob));
    CHECK(snap_seq(mem, 0) == before);
}

int main() {
    std::fprintf(stderr, "=== RERoT Recurrent Parking COW Tests ===\n");
    test_hand_checkpoint_restore();
    test_fence_replay_recurrent_oracle();
    test_root_visibility_state_continuity();
    test_hand_checkpoint_snapshot_capture();

    test_native_input_recycled_storage();

    ggml_backend_t cpu_oracle = ggml_backend_cpu_init();
    test_mixed_frontier_graph(cpu_oracle, false);
    test_mixed_frontier_graph(cpu_oracle, true);
    ggml_backend_free(cpu_oracle);
    ggml_backend_load_all();
    if (auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
        if (auto * gpu = ggml_backend_dev_init(device, nullptr)) {
            test_mixed_frontier_graph(gpu, false);
            test_mixed_frontier_graph(gpu, true);
            ggml_backend_free(gpu);
        }
    }

    stub_model model;
    model.hparams.n_layer_all = 2;
    model.hparams.n_embd = 8;
    model.hparams.n_embd_r_impl = 16;
    model.hparams.ssm_d_state = 4;
    model.hparams.ssm_d_inner = 8;

    // n_seq_max is intentionally tiny: parked/high logical ids must still work.
    llama_memory_recurrent mem(model, GGML_TYPE_F32, GGML_TYPE_F32,
        false, /*mem_size=*/ 8, /*n_seq_max=*/ 4, /*n_rs_seq=*/ 0,
        /*n_brain_max=*/ 0, /*n_hand_max=*/ 0, nullptr);

    CHECK(mem.get_recurrent_capacity() == 8);
    CHECK(mem.tails.size() == LLAMA_MAX_SEQ);
    CHECK(mem.r_l.size() == 2 && mem.r_l[0] != nullptr && mem.r_l[1] != nullptr);
    CHECK(mem.s_l.size() == 2 && mem.s_l[0] != nullptr && mem.s_l[1] != nullptr);

    const llama_seq_id S = 0;
    const llama_seq_id A = 1;
    const llama_seq_id B = 2;
    const llama_seq_id C = 3;
    const llama_seq_id A1 = 4;
    const llama_seq_id H = 200; // parked id far beyond the ctor n_seq_max
    const llama_seq_id D = 255; // LLAMA_MAX_SEQ boundary

    // 1. parent S decodes its prefix.
    CHECK(admit(mem, S, 0, 4));
    CHECK(mem.seq_pos_max(S) == 3);
    CHECK(mem.get_recurrent_used() == 1);
    stamp_seq(mem, S, 0x53, 0x73);
    const auto snapS0 = snap_seq(mem, S);

    // 2. fork A/B/C (+ high parked ids): all share the parent tail, no new
    // cell, no tensor bytes touched.
    mem.seq_cp_recurrent(S, A, -1, -1);
    mem.seq_cp_recurrent(S, B, -1, -1);
    mem.seq_cp_recurrent(S, C, -1, -1);
    mem.seq_cp_recurrent(S, H, -1, -1);
    mem.seq_cp_recurrent(S, D, -1, -1);
    CHECK(mem.tails[(size_t) A] == mem.tails[(size_t) S]);
    CHECK(mem.tails[(size_t) B] == mem.tails[(size_t) S]);
    CHECK(mem.tails[(size_t) C] == mem.tails[(size_t) S]);
    CHECK(mem.tails[(size_t) H] == mem.tails[(size_t) S]);
    CHECK(mem.tails[(size_t) D] == mem.tails[(size_t) S]);
    CHECK(mem.get_recurrent_used() == 1);
    CHECK(snap_seq(mem, S) == snapS0);
    CHECK(snap_seq(mem, A) == snapS0);
    CHECK(snap_seq(mem, H) == snapS0);

    // 3. admit A: COW isolates it, B/C/H/D/S bytes unchanged.
    const auto snapB = snap_seq(mem, B);
    const auto snapC = snap_seq(mem, C);
    const auto snapH = snap_seq(mem, H);
    const auto snapD = snap_seq(mem, D);
    CHECK(admit(mem, A, 4, 1));
    CHECK(mem.seq_pos_max(A) == 4);
    CHECK(mem.tails[(size_t) A] != mem.tails[(size_t) B]);
    CHECK(mem.tails[(size_t) B] == mem.tails[(size_t) S]);
    CHECK(mem.tails[(size_t) C] == mem.tails[(size_t) S]);
    CHECK(mem.tails[(size_t) H] == mem.tails[(size_t) S]);
    CHECK(mem.tails[(size_t) D] == mem.tails[(size_t) S]);
    CHECK(mem.get_recurrent_used() == 2);
    CHECK(snap_seq(mem, B) == snapB);
    CHECK(snap_seq(mem, C) == snapC);
    CHECK(snap_seq(mem, H) == snapH);
    CHECK(snap_seq(mem, D) == snapD);
    CHECK(snap_seq(mem, S) == snapS0);
    stamp_seq(mem, A, 0xA1, 0xA2);
    const auto snapA1 = snap_seq(mem, A);
    CHECK(snap_seq(mem, B) == snapB);
    CHECK(snap_seq(mem, C) == snapC);
    CHECK(snap_seq(mem, H) == snapH);
    CHECK(snap_seq(mem, D) == snapD);

    // 4. admit B (vice versa): A/C/H/D/S bytes unchanged.
    CHECK(admit(mem, B, 4, 1));
    CHECK(mem.get_recurrent_used() == 3);
    CHECK(snap_seq(mem, A) == snapA1);
    CHECK(snap_seq(mem, C) == snapC);
    CHECK(snap_seq(mem, H) == snapH);
    CHECK(snap_seq(mem, D) == snapD);
    CHECK(snap_seq(mem, S) == snapS0);
    stamp_seq(mem, B, 0xB1, 0xB2);
    const auto snapB1 = snap_seq(mem, B);
    CHECK(snap_seq(mem, A) == snapA1);
    CHECK(snap_seq(mem, C) == snapC);
    CHECK(snap_seq(mem, H) == snapH);

    // 5. admit high-id parked child H: logical capacity, siblings isolated.
    CHECK(admit(mem, H, 4, 1));
    CHECK(mem.get_recurrent_used() == 4);
    CHECK(mem.tails[(size_t) H] != mem.tails[(size_t) C]);
    CHECK(snap_seq(mem, A) == snapA1);
    CHECK(snap_seq(mem, B) == snapB1);
    CHECK(snap_seq(mem, C) == snapC);
    CHECK(snap_seq(mem, D) == snapD);
    stamp_seq(mem, H, 0xC1, 0xC2);
    const auto snapH1 = snap_seq(mem, H);
    CHECK(snap_seq(mem, A) == snapA1);
    CHECK(snap_seq(mem, B) == snapB1);
    CHECK(snap_seq(mem, C) == snapC);

    // 6. retire the parent exec id: parked C/D keep their tail, no cell freed
    // for a still-referenced tail; the id is then reusable without pollution.
    CHECK(mem.seq_rm_recurrent(S, -1, -1));
    CHECK(mem.tails[(size_t) S] == -1);
    CHECK(mem.get_recurrent_used() == 4);
    CHECK(mem.tails[(size_t) C] == mem.tails[(size_t) D]);
    CHECK(snap_seq(mem, C) == snapC);
    CHECK(snap_seq(mem, D) == snapD);
    mem.seq_cp_recurrent(A, S, -1, -1); // reuse S as a fresh child of A
    CHECK(mem.tails[(size_t) S] == mem.tails[(size_t) A]);
    CHECK(mem.get_recurrent_used() == 4);
    CHECK(snap_seq(mem, A) == snapA1);
    CHECK(snap_seq(mem, B) == snapB1);
    CHECK(snap_seq(mem, C) == snapC);
    CHECK(snap_seq(mem, H) == snapH1);

    // 7. recursive fork: admitted A forks A1, A1 admission isolates again.
    mem.seq_cp_recurrent(A, A1, -1, -1);
    CHECK(mem.tails[(size_t) A1] == mem.tails[(size_t) A]);
    CHECK(mem.get_recurrent_used() == 4);
    CHECK(admit(mem, A1, 5, 1));
    CHECK(mem.tails[(size_t) A1] != mem.tails[(size_t) A]);
    CHECK(snap_seq(mem, A) == snapA1);
    stamp_seq(mem, A1, 0xD1, 0xD2);
    CHECK(snap_seq(mem, A) == snapA1);
    CHECK(snap_seq(mem, S) == snapA1); // S still shares A's tail: same lineage

    // 8. component-selective isolation on recurrent-only memory: attention
    // ops are vacuous no-ops that never touch recurrent bytes.
    CHECK(mem.seq_rm_attention(A, 0, 1000000) == true);
    CHECK(snap_seq(mem, A) == snapA1);
    mem.seq_cp_attention(A, B, 0, 1000000);
    CHECK(snap_seq(mem, A) == snapA1);
    CHECK(snap_seq(mem, B) == snapB1);
    CHECK(mem.tails[(size_t) B] != mem.tails[(size_t) A]);

    // 9. cleanup of parked refs releases exactly their cells.
    CHECK(mem.seq_rm_recurrent(C, -1, -1));
    CHECK(mem.seq_rm_recurrent(D, -1, -1));
    CHECK(mem.tails[(size_t) C] == -1 && mem.tails[(size_t) D] == -1);
    CHECK(snap_seq(mem, A) == snapA1);
    CHECK(snap_seq(mem, B) == snapB1);
    CHECK(snap_seq(mem, H) == snapH1);

    // 10. Grouped recurrent allocation keeps only B shared brain rows while
    // retaining P hand rows for convolution and the first three private states.
    {
        model.hparams.n_layer_all = 5;
        llama_memory_recurrent gmem(model, GGML_TYPE_F32, GGML_TYPE_F32,
            false, /*mem_size=*/ 12, /*n_seq_max=*/ 16, /*n_rs_seq=*/ 0,
            /*n_brain_max=*/ 3, /*n_hand_max=*/ 12, nullptr);

        gmem.set_grouped_layout(3, 12);
        CHECK(gmem.is_grouped_layout());
        CHECK(gmem.get_brain_capacity() == 3);
        CHECK(gmem.get_hand_capacity() == 12);
        for (int il = 0; il < 5; ++il) {
            CHECK(gmem.r_l[il]->ne[1] == 12);
            CHECK(gmem.s_l[il]->ne[1] == (il < 3 ? 12 : 6));
            CHECK(gmem.is_s_shared(il) == (il >= 3));
            CHECK((gmem.d_l[il] != nullptr) == (il >= 3));
            if (gmem.d_l[il]) {
                CHECK(gmem.d_l[il]->type == GGML_TYPE_F16);
                CHECK(gmem.d_l[il]->ne[1] == 12);
            }
        }

        CHECK(admit(gmem, 1, 0, 1));
        stamp_seq(gmem, 1, 0x11, 0x12);
        llama_pos pos = -1;
        const int32_t r1 = resolved_row(gmem, 1, pos);
        CHECK(r1 >= 0);
        for (size_t il = 0; il < gmem.d_l.size(); ++il) {
            if (!gmem.d_l[il]) {
                continue;
            }
            const size_t row_size =
                ggml_row_size(gmem.d_l[il]->type, gmem.d_l[il]->ne[0]);
            const std::vector<uint8_t> hand(row_size, uint8_t(0x60 + il));
            ggml_backend_tensor_set(
                gmem.d_l[il],
                hand.data(),
                (size_t) r1 * row_size,
                row_size);
        }

        // Shared fork hand seed (§B.6.4):
        // Capture parent hand seed from sequence 1
        auto seed = gmem.capture_hand_seed(999, 1);
        CHECK(seed != nullptr);
        CHECK(seed->fork_id == 999);
        CHECK(!seed->conv_tail_bytes.empty());

        // Apply shared seed to sequence 8 and 9
        CHECK(admit(gmem, 8, 0, 1));
        CHECK(admit(gmem, 9, 0, 1));
        CHECK(gmem.apply_hand_seed(8, seed));
        CHECK(gmem.apply_hand_seed(9, seed));

        const int32_t r8 = resolved_row(gmem, 8, pos);
        const int32_t r9 = resolved_row(gmem, 9, pos);
        CHECK(r8 >= 0 && r9 >= 0 && r8 != r9);

        // Verify both received identical conv state matching the parent seed
        const size_t r_row_size = ggml_row_size(gmem.r_l[0]->type, gmem.r_l[0]->ne[0]);
        std::vector<uint8_t> conv8(r_row_size);
        std::vector<uint8_t> conv9(r_row_size);
        ggml_backend_tensor_get(gmem.r_l[0], conv8.data(), (size_t) r8 * r_row_size, r_row_size);
        ggml_backend_tensor_get(gmem.r_l[0], conv9.data(), (size_t) r9 * r_row_size, r_row_size);
        CHECK(conv8 == seed->conv_tail_bytes[0]);
        CHECK(conv9 == seed->conv_tail_bytes[0]);
        CHECK(conv8 == conv9);

        // Serialized blob capture/apply roundtrip (§B.6.4)
        std::vector<uint8_t> seed_blob;
        CHECK(gmem.rerot_capture_hand_seed(1, seed_blob));
        CHECK(!seed_blob.empty());
        CHECK(gmem.rerot_hand_seed_size(1) == seed_blob.size());

        // A parked child has no recurrent row. Applying its serialized fork
        // seed must allocate one physical hand row directly, without retaining
        // the parent row as a hidden capacity consumer.
        const uint32_t used_before_restore = gmem.get_recurrent_used();
        CHECK(gmem.tails[10] == -1);
        CHECK(gmem.rerot_apply_hand_seed(10, seed_blob));
        const int32_t r10 = resolved_row(gmem, 10, pos);
        CHECK(r10 >= 0);
        CHECK(pos == 0);
        CHECK(gmem.get_recurrent_used() == used_before_restore + 1);
        std::vector<uint8_t> conv10(r_row_size);
        ggml_backend_tensor_get(gmem.r_l[0], conv10.data(), (size_t) r10 * r_row_size, r_row_size);
        CHECK(conv10 == conv8);
        for (size_t il = 0; il < gmem.d_l.size(); ++il) {
            if (!gmem.d_l[il]) {
                continue;
            }
            const size_t row_size =
                ggml_row_size(gmem.d_l[il]->type, gmem.d_l[il]->ne[0]);
            std::vector<uint8_t> restored(row_size);
            ggml_backend_tensor_get(
                gmem.d_l[il],
                restored.data(),
                (size_t) r10 * row_size,
                row_size);
            CHECK(seed->state_bytes[il].size() == row_size);
            CHECK(restored == seed->state_bytes[il]);
        }

        // A PRIVATE root planner writes into its private brain row. Forking
        // must preserve the effective state as a child hand delta relative to
        // the still-unmodified public brain.
        llama_kv_rerot_meta planner_tag;
        planner_tag.episode_id = 888;
        planner_tag.node_id = 0;
        planner_tag.visibility = llama_rerot_visibility::private_control;
        CHECK(gmem.rerot_set_write_tag(1, planner_tag));
        const int32_t planner_brain = gmem.episode_brain.at(888);
        for (size_t il = 0; il < gmem.s_l.size(); ++il) {
            if (!gmem.is_s_shared((int32_t) il)) {
                continue;
            }
            const size_t n = (size_t) gmem.s_l[il]->ne[0];
            const size_t brain_row_size =
                ggml_row_size(gmem.s_l[il]->type, gmem.s_l[il]->ne[0]);
            const size_t hand_row_size =
                ggml_row_size(gmem.d_l[il]->type, gmem.d_l[il]->ne[0]);
            const std::vector<float> public_state(n, 1.0f);
            const std::vector<float> private_state(n, 4.0f);
            const std::vector<ggml_fp16_t> planner_delta(
                n, ggml_fp32_to_fp16(2.0f));
            ggml_backend_tensor_set(
                gmem.s_l[il],
                public_state.data(),
                (size_t) planner_brain * brain_row_size,
                brain_row_size);
            ggml_backend_tensor_set(
                gmem.s_l[il],
                private_state.data(),
                ((size_t) gmem.get_brain_capacity() +
                    (size_t) planner_brain) * brain_row_size,
                brain_row_size);
            ggml_backend_tensor_set(
                gmem.d_l[il],
                planner_delta.data(),
                (size_t) r1 * hand_row_size,
                hand_row_size);
        }

        auto planner_seed = gmem.capture_hand_seed(1000, 1);
        CHECK(planner_seed != nullptr);
        CHECK(gmem.apply_hand_seed(11, planner_seed));
        const int32_t r11 = resolved_row(gmem, 11, pos);
        CHECK(r11 >= 0);
        for (size_t il = 0; il < gmem.d_l.size(); ++il) {
            if (!gmem.d_l[il]) {
                continue;
            }
            const size_t n = (size_t) gmem.d_l[il]->ne[0];
            const size_t row_size =
                ggml_row_size(gmem.d_l[il]->type, gmem.d_l[il]->ne[0]);
            std::vector<ggml_fp16_t> restored(n);
            ggml_backend_tensor_get(
                gmem.d_l[il],
                restored.data(),
                (size_t) r11 * row_size,
                row_size);
            CHECK(std::all_of(
                restored.begin(), restored.end(),
                [](ggml_fp16_t value) {
                    return std::abs(
                        ggml_fp16_to_fp32(value) - 5.0f) < 1.0e-6f;
                }));
        }

        // Child PUBLIC writes must reach the shared brain just like root
        // writes. A PRIVATE sibling must not be part of that commit.
        llama_batch_allocr frontier_alloc(1);
        llama_ubatch frontier = frontier_alloc.ubatch_reserve(1, 3);
        llama_seq_id frontier_ids[] = {4, 5, 6};
        for (uint32_t i = 0; i < 3; ++i) {
            frontier.token[i] = 1;
            frontier.pos[i] = 0;
            frontier.n_seq_id[i] = 1;
            frontier.seq_id[i] = &frontier_ids[i];
            frontier.output[i] = 0;
            llama_kv_rerot_meta tag;
            tag.episode_id = 777;
            tag.node_id = i + 1;
            tag.visibility = i < 2
                ? llama_rerot_visibility::public_live
                : llama_rerot_visibility::private_control;
            CHECK(gmem.rerot_set_write_tag(frontier_ids[i], tag));
        }
        llama_memory_recurrent_context frontier_ctx(&gmem, {frontier});
        const auto public_groups = frontier_ctx.public_brain_groups();
        CHECK(public_groups.size() == 1);
        if (public_groups.size() == 1) {
            CHECK(public_groups.begin()->second == std::vector<int32_t>({0, 1}));
        }
        CHECK(frontier_ctx.is_public_write(0));
        CHECK(frontier_ctx.is_public_write(1));
        CHECK(!frontier_ctx.is_public_write(2));

        // PENDING bytes stay out of the global recurrent commit for every
        // node. The root planner additionally stays on its private brain
        // overlay while constructing the decomposition.
        llama_batch_allocr root_pending_alloc(1);
        llama_ubatch root_pending = root_pending_alloc.ubatch_reserve(1, 2);
        llama_seq_id pending_ids[] = {7, 8};
        for (uint32_t i = 0; i < 2; ++i) {
            root_pending.token[i] = 1;
            root_pending.pos[i] = 0;
            root_pending.n_seq_id[i] = 1;
            root_pending.seq_id[i] = &pending_ids[i];
            root_pending.output[i] = 0;
            llama_kv_rerot_meta pending_tag;
            pending_tag.episode_id = 777;
            pending_tag.node_id = i == 0 ? 1 : 0;
            pending_tag.visibility = llama_rerot_visibility::pending_record;
            CHECK(gmem.rerot_set_write_tag(pending_ids[i], pending_tag));
        }
        llama_memory_recurrent_context root_pending_ctx(&gmem, {root_pending});
        const auto root_pending_groups = root_pending_ctx.public_brain_groups();
        CHECK(root_pending_groups.size() == 1);
        if (root_pending_groups.size() == 1) {
            const int32_t private_brain =
                (int32_t) gmem.get_brain_capacity() + gmem.episode_brain.at(777);
            CHECK(root_pending_groups.begin()->first == private_brain);
            CHECK(root_pending_groups.begin()->second == std::vector<int32_t>({1}));
            CHECK(root_pending_ctx.brain_copy(1) == private_brain);
        }
        CHECK(!root_pending_ctx.is_public_write(0));
        CHECK(!root_pending_ctx.is_public_write(1));

        // Exercise the actual graph builder with two PUBLIC writers and a
        // PRIVATE sibling. PUBLIC writers start from the shared brain, not
        // retained private-control overlays. PUBLIC decay is the geometric
        // mean of alpha (mean log-decay); a PRIVATE hand is T(B+H)-B,
        // not T(B+H)-T(B). Check the mathematical contract, not the old mean
        // fallback's output.
        {
            llama_memory_recurrent graph_mem(model, GGML_TYPE_F32, GGML_TYPE_F32,
                false, 3, 16, 0, 1, 3, nullptr);
            llama_memory_recurrent_context graph_mctx(&graph_mem, {frontier});
            llm_graph_result graph_result(256);
            llm_graph_params graph_params{};
            graph_params.hparams = model.hparams;
            graph_params.ubatch = frontier;
            graph_params.n_outputs = 3;
            graph_params.res = &graph_result;
            llm_build_delta_net_base builder(graph_params);
            auto * graph_ctx = graph_result.get_ctx();
            llm_graph_input_rs graph_input(&graph_mctx);
            graph_input.brain_copy = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_I32, 3);
            auto * public_rows = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_I32, 2);
            graph_input.rbb_groups.push_back({0, public_rows});

            constexpr int64_t S = 4;
            constexpr int64_t H = 2;
            constexpr int64_t D = S * S * H;
            auto * base = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, D, 3);
            auto * state = ggml_new_tensor_4d(graph_ctx, GGML_TYPE_F32, S, S, H, 3);
            auto * q = ggml_new_tensor_4d(graph_ctx, GGML_TYPE_F32, S, H, 1, 3);
            auto * k = ggml_dup_tensor(graph_ctx, q);
            auto * v = ggml_dup_tensor(graph_ctx, q);
            auto * g = ggml_new_tensor_4d(graph_ctx, GGML_TYPE_F32, 1, H, 1, 3);
            auto * beta = ggml_dup_tensor(graph_ctx, g);
            auto * output = builder.build_recurrent_attn(
                &graph_input, graph_mem.s_l[3], base, graph_mem.d_l[3],
                q, k, v, g, beta, state, 3);
            ggml_build_forward_expand(graph_result.get_gf(), output);

            ggml_backend_t cpu = ggml_backend_cpu_init();
            ggml_backend_buffer_t graph_buffer =
                ggml_backend_alloc_ctx_tensors(graph_ctx, cpu);
            CHECK(graph_buffer != nullptr);
            const int32_t indices[] = {0, 1};
            ggml_backend_tensor_set(public_rows, indices, 0, sizeof(indices));
            auto fill = [](ggml_tensor * tensor, float value) {
                const std::vector<float> values(ggml_nelements(tensor), value);
                ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
            };
            fill(base, 2.0f);
            fill(graph_mem.s_l[3], 2.0f);
            std::vector<float> states(D * 3);
            std::fill_n(states.data(), D, 2.0f);
            std::fill_n(states.data() + D, D, 4.0f);
            std::fill_n(states.data() + 2 * D, D, 8.0f);
            ggml_backend_tensor_set(state, states.data(), 0, states.size() * sizeof(float));
            fill(q, 0.0f);
            fill(k, 0.0f);
            fill(v, 0.0f);
            const float decay[] = {
                std::log(0.5f), std::log(0.5f),
                std::log(0.25f), std::log(0.25f),
                std::log(0.5f), std::log(0.5f),
            };
            ggml_backend_tensor_set(g, decay, 0, sizeof(decay));
            fill(beta, 0.0f);
            CHECK(ggml_backend_graph_compute(cpu, graph_result.get_gf()) == GGML_STATUS_SUCCESS);

            std::vector<float> brain(D);
            ggml_backend_tensor_get(
                graph_mem.s_l[3], brain.data(), 0, brain.size() * sizeof(float));
            CHECK(std::all_of(brain.begin(), brain.end(), [](float value) {
                return std::abs(value - 2.0f * std::sqrt(0.5f * 0.25f)) < 1.0e-6f;
            }));
            std::vector<ggml_fp16_t> hand(D * 3);
            ggml_backend_tensor_get(
                graph_mem.d_l[3], hand.data(), 0, hand.size() * sizeof(ggml_fp16_t));
            CHECK(std::all_of(hand.begin(), hand.begin() + D, [](ggml_fp16_t value) {
                return ggml_fp16_to_fp32(value) == 0.0f;
            }));
            CHECK(std::all_of(hand.begin() + D, hand.begin() + 2 * D, [](ggml_fp16_t value) {
                return std::abs(ggml_fp16_to_fp32(value) - 0.5f) < 1.0e-6f;
            }));
            CHECK(std::all_of(hand.begin() + 2 * D, hand.end(), [](ggml_fp16_t value) {
                return std::abs(ggml_fp16_to_fp32(value) - 2.0f) < 1.0e-6f;
            }));
            ggml_backend_buffer_free(graph_buffer);
            ggml_backend_free(cpu);
        }

        // Rollback snapshots retain one plane per public brain, not one full
        // shared-state row per pen or a duplicate private-planner plane.
        llama_memory_recurrent rollback_mem(model, GGML_TYPE_F32, GGML_TYPE_F32,
            false, /*mem_size=*/ 12, /*n_seq_max=*/ 16, /*n_rs_seq=*/ 3,
            /*n_brain_max=*/ 3, /*n_hand_max=*/ 12, nullptr);
        for (int il = 0; il < 5; ++il) {
            CHECK(rollback_mem.r_l[il]->ne[1] == 48);
            CHECK(rollback_mem.s_l[il]->ne[1] == (il < 3 ? 48 : 15));
            CHECK((rollback_mem.d_l[il] != nullptr) == (il >= 3));
            if (rollback_mem.d_l[il]) {
                CHECK(rollback_mem.d_l[il]->ne[1] == 48);
            }
        }
    }

    // 11. Ordinary grouped prefill+decode (cold --rerot, request rerot:false)
    // must match native single-sequence recurrence before any episode tag.
    // The two ordinary mirror brain rows + zero F16 hand must not change the
    // readout; catches aliasing/wrong gather type/double-commit bugs that
    // misread e.g. 29x31 as 29x23. Prefill uses n_seq_tokens>1, then decode.
    // Reused physical row after reset covers the missing rs_zero clear in
    // build_rs_shared. Standalone Vulkan parity runs the same S=16 fused
    // math on GPU where available (graph ordering/backend).
    {
        std::fprintf(stderr, "--- Ordinary grouped prefill vs native ---\n");
        model.hparams.n_layer_all = 4;
        model.hparams.n_embd = 8;
        model.hparams.n_embd_r_impl = 16;
        model.hparams.ssm_d_state = 16;
        model.hparams.ssm_d_inner = 32;
        constexpr int64_t S = 16;
        constexpr int64_t H = 2;
        constexpr int64_t D = S * S * H;
        CHECK((int64_t) model.hparams.n_embd_s() == D);
        constexpr int IL = 3; // first shared layer (0-2 private, 3 shared)

        llama_memory_recurrent plain(model, GGML_TYPE_F32, GGML_TYPE_F32,
            false, /*mem_size=*/ 4, /*n_seq_max=*/ 16, /*n_rs_seq=*/ 0,
            /*n_brain_max=*/ 0, /*n_hand_max=*/ 0, nullptr);
        llama_memory_recurrent grouped(model, GGML_TYPE_F32, GGML_TYPE_F32,
            false, /*mem_size=*/ 4, /*n_seq_max=*/ 16, /*n_rs_seq=*/ 0,
            /*n_brain_max=*/ 2, /*n_hand_max=*/ 4, nullptr);
        grouped.set_grouped_layout(2, 4);
        CHECK(!plain.is_grouped_layout());
        CHECK(grouped.is_grouped_layout());
        CHECK(!plain.is_s_shared(IL));
        CHECK(grouped.is_s_shared(IL));
        CHECK(grouped.d_l[IL] != nullptr);
        CHECK(grouped.d_l[IL]->type == GGML_TYPE_F16);

        auto make_vec = [](int64_t n, uint64_t seed, bool is_gate, bool is_beta) {
            std::vector<float> v((size_t) n);
            for (int64_t i = 0; i < n; ++i) {
                const float d = ord_det(seed + (uint64_t) i);
                if (is_gate) {
                    v[(size_t) i] = -0.5f + 0.2f * d; // log-decay in [-0.58,-0.42]
                } else if (is_beta) {
                    v[(size_t) i] = 0.5f + d; // (0.1,0.9)
                } else {
                    v[(size_t) i] = d;
                }
            }
            return v;
        };

        // Prefill: n_seq_tokens=3 (>1), single ordinary seq 0, no episode tag.
        const int64_t NP = 3;
        const std::vector<float> pq = make_vec(S * H * NP, 100, false, false);
        const std::vector<float> pk = make_vec(S * H * NP, 200, false, false);
        const std::vector<float> pv = make_vec(S * H * NP, 300, false, false);
        const std::vector<float> pg = make_vec(H * NP, 400, true, false);
        const std::vector<float> pb = make_vec(H * NP, 500, false, true);
        llama_batch_allocr pre_alloc(1);
        std::vector<llama_seq_id> pre_ids;
        const llama_ubatch pre_ub = make_ubatch(pre_alloc, 0, {0, 1, 2}, pre_ids);
        CHECK(pre_ub.n_seq_tokens == 3 && pre_ub.n_seqs == 1);
        const ord_step_out plain_pre = ord_run_mem_step(
            model, plain, pre_ub, IL, S, H, pq, pk, pv, pg, pb);
        const ord_step_out group_pre = ord_run_mem_step(
            model, grouped, pre_ub, IL, S, H, pq, pk, pv, pg, pb);
        CHECK(plain_pre.output.size() == group_pre.output.size());
        CHECK(plain_pre.eff.size() == (size_t) D && group_pre.eff.size() == (size_t) D);
        if (plain_pre.output.size() == group_pre.output.size()) {
            const float e = ord_max_abs_diff(plain_pre.output, group_pre.output);
            if (e >= 1.0e-5f) {
                std::fprintf(stderr, "prefill output diff = %.9g\n", e);
            }
            CHECK(e < 1.0e-5f);
        }
        {
            const float e = ord_max_abs_diff(plain_pre.eff, group_pre.eff);
            if (e >= 1.0e-5f) {
                std::fprintf(stderr, "prefill effective-state diff = %.9g\n", e);
            }
            CHECK(e < 1.0e-5f);
        }
        // Zero-hand + mirror invariant: ordinary readout is brain-only.
        CHECK(group_pre.hand_f32.size() == (size_t) D);
        CHECK(group_pre.brain_pub.size() == (size_t) D);
        CHECK(group_pre.brain_priv.size() == (size_t) D);
        for (int64_t i = 0; i < D; ++i) {
            CHECK(std::abs(group_pre.hand_f32[(size_t) i]) < 1.0e-6f);
        }
        CHECK(ord_max_abs_diff(group_pre.brain_pub, group_pre.brain_priv) < 1.0e-6f);

        // Decode: n_seq_tokens=1 continuation at pos 3.
        const int64_t ND = 1;
        const std::vector<float> dq = make_vec(S * H * ND, 600, false, false);
        const std::vector<float> dk = make_vec(S * H * ND, 700, false, false);
        const std::vector<float> dv = make_vec(S * H * ND, 800, false, false);
        const std::vector<float> dg = make_vec(H * ND, 900, true, false);
        const std::vector<float> db = make_vec(H * ND, 1000, false, true);
        llama_batch_allocr dec_alloc(1);
        std::vector<llama_seq_id> dec_ids;
        const llama_ubatch dec_ub = make_ubatch(dec_alloc, 0, {3}, dec_ids);
        CHECK(dec_ub.n_seq_tokens == 1 && dec_ub.n_seqs == 1);
        const ord_step_out plain_dec = ord_run_mem_step(
            model, plain, dec_ub, IL, S, H, dq, dk, dv, dg, db);
        const ord_step_out group_dec = ord_run_mem_step(
            model, grouped, dec_ub, IL, S, H, dq, dk, dv, dg, db);
        CHECK(plain_dec.output.size() == group_dec.output.size());
        if (plain_dec.output.size() == group_dec.output.size()) {
            const float e = ord_max_abs_diff(plain_dec.output, group_dec.output);
            if (e >= 2.0e-4f) {
                std::fprintf(stderr, "decode output diff = %.9g\n", e);
            }
            CHECK(e < 2.0e-4f);
        }
        {
            const float e = ord_max_abs_diff(plain_dec.eff, group_dec.eff);
            if (e >= 1.0e-2f) {
                std::fprintf(stderr, "decode effective-state diff = %.9g\n", e);
            }
            CHECK(e < 1.0e-2f);
        }
        for (int64_t i = 0; i < D; ++i) {
            CHECK(std::abs(group_dec.hand_f32[(size_t) i]) < 1.0e-6f);
        }
        CHECK(ord_max_abs_diff(group_dec.brain_pub, group_dec.brain_priv) < 1.0e-6f);

        // Reused root after reset: same seq id re-admitted at pos 0 must
        // still match native (build_rs_shared lacks the rs_zero clear that
        // build_rs performs, so stale brain bytes would leak here).
        CHECK(plain.seq_rm_recurrent(0, -1, -1));
        CHECK(grouped.seq_rm_recurrent(0, -1, -1));
        const std::vector<float> rq = make_vec(S * H * ND, 1100, false, false);
        const std::vector<float> rk = make_vec(S * H * ND, 1200, false, false);
        const std::vector<float> rv = make_vec(S * H * ND, 1300, false, false);
        const std::vector<float> rg = make_vec(H * ND, 1400, true, false);
        const std::vector<float> rb = make_vec(H * ND, 1500, false, true);
        llama_batch_allocr re_alloc(1);
        std::vector<llama_seq_id> re_ids;
        const llama_ubatch re_ub = make_ubatch(re_alloc, 0, {0}, re_ids);
        const ord_step_out plain_re = ord_run_mem_step(
            model, plain, re_ub, IL, S, H, rq, rk, rv, rg, rb);
        const ord_step_out group_re = ord_run_mem_step(
            model, grouped, re_ub, IL, S, H, rq, rk, rv, rg, rb);
        CHECK(plain_re.output.size() == group_re.output.size());
        if (plain_re.output.size() == group_re.output.size()) {
            const float e = ord_max_abs_diff(plain_re.output, group_re.output);
            if (e >= 1.0e-5f) {
                std::fprintf(stderr, "reused-root output diff = %.9g\n", e);
            }
            CHECK(e < 1.0e-5f);
        }
        {
            const float e = ord_max_abs_diff(plain_re.eff, group_re.eff);
            if (e >= 1.0e-5f) {
                std::fprintf(stderr, "reused-root effective-state diff = %.9g\n", e);
            }
            CHECK(e < 1.0e-5f);
        }

        // Backend parity: same fused S=16 math on Vulkan where available.
        // CPU-only hosts SKIP (no failure); a real ordering/dtype bug shows
        // as a large output diff, not tolerance noise.
        {
            ggml_backend_load_all();
            ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            if (!gpu) {
                std::fprintf(stderr, "SKIP: no GPU backend for ordinary GDN parity\n");
            } else {
                ggml_backend_t cpu_be = ggml_backend_cpu_init();
                ggml_backend_t gpu_be = ggml_backend_dev_init(gpu, nullptr);
                CHECK(gpu_be != nullptr);
                if (gpu_be) {
                    const std::vector<float> s0((size_t) D, 0.0f);
                    bool cpu_ok = false;
                    bool gpu_ok = false;
                    const std::vector<float> cpu_out = ord_run_standalone_gdn(
                        model, S, H, NP, pq, pk, pv, pg, pb, s0, cpu_be, cpu_ok);
                    const std::vector<float> gpu_out = ord_run_standalone_gdn(
                        model, S, H, NP, pq, pk, pv, pg, pb, s0, gpu_be, gpu_ok);
                    if (!cpu_ok) {
                        std::fprintf(stderr, "SKIP: CPU lacks fused GDN S=16 op\n");
                    } else if (!gpu_ok) {
                        std::fprintf(stderr, "SKIP: GPU lacks fused GDN S=16 op\n");
                    } else {
                        CHECK(cpu_out.size() == gpu_out.size());
                        if (cpu_out.size() == gpu_out.size()) {
                            const float e = ord_max_abs_diff(cpu_out, gpu_out);
                            if (e >= 1.0e-3f) {
                                std::fprintf(stderr, "CPU/GPU GDN parity diff = %.9g\n", e);
                            }
                            CHECK(e < 1.0e-3f);
                        }
                    }
                    ggml_backend_free(gpu_be);
                }
                ggml_backend_free(cpu_be);
            }
        }
    }

    std::fprintf(stderr, "=== Results: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
