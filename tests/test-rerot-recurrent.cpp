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

int main() {
    std::fprintf(stderr, "=== RERoT Recurrent Parking COW Tests ===\n");

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
            CHECK(seed->state_bytes[il].empty());
            CHECK(std::all_of(
                restored.begin(), restored.end(),
                [](uint8_t value) { return value == 0; }));
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

        // Exercise the actual graph builder with two PUBLIC writers and a
        // PRIVATE sibling. PUBLIC writers start from the shared brain, not
        // retained private-control overlays. The mean must neither amplify
        // their updates nor erase the PRIVATE sibling's state delta.
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
                return std::abs(value - 0.75f) < 1.0e-6f;
            }));
            std::vector<ggml_fp16_t> hand(D * 3);
            ggml_backend_tensor_get(
                graph_mem.d_l[3], hand.data(), 0, hand.size() * sizeof(ggml_fp16_t));
            CHECK(std::all_of(hand.begin(), hand.begin() + 2 * D, [](ggml_fp16_t value) {
                return ggml_fp16_to_fp32(value) == 0.0f;
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

    std::fprintf(stderr, "=== Results: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
