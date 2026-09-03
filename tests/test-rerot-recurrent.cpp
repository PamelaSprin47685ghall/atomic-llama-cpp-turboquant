#include "ggml-backend.h"
#include "ggml.h"
#include "llama-batch.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"
#include "llama.h"

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
        false, /*mem_size=*/ 8, /*n_seq_max=*/ 4, /*n_rs_seq=*/ 0, nullptr);

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

    std::fprintf(stderr, "=== Results: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
