#include "../common/speculative-mtp-workspace.h"
#include "../src/llama-predefined-hidden.h"

#include <cstdio>
#include <stdexcept>

#define CHECK(x) do { if (!(x)) throw std::runtime_error("failed: " #x); } while (0)

static void accepted_target_hidden_is_not_draft_hidden() {
    common_mtp_workspace workspace(2, 8, 2);
    const size_t capacity_bytes = workspace.hidden_storage_bytes();
    CHECK(capacity_bytes == (8 + 2 + 2) * 2 * sizeof(float));
    float * const carry = workspace.pending(0);
    carry[0] = 10;
    carry[1] = 11;
    workspace.pending(1)[0] = 20;
    workspace.pending(1)[1] = 21;

    const float verified[] = {100, 101, 110, 111, 120, 121, 200, 201};
    const llama_token tokens[] = {7, 8, 9, 17};
    const llama_pos positions[] = {40, 41, 42, 90};
    CHECK(workspace.begin(4, verified, tokens, positions));
    CHECK(workspace.sequence(0, 0, 3, true, true));
    CHECK(!workspace.sequence(1, 2, 2, true, false)); // overlap: zero mutation
    CHECK(workspace.sequence(1, 3, 1, true, false));
    CHECK(workspace.total_commit_rows() == 1); // deferred seq 0 not accepted yet
    CHECK(!workspace.begin(4, verified, tokens, positions)); // cannot overwrite staged state
    CHECK(workspace.accept(0, 1));
    CHECK(workspace.commit_rows(0) == 2 && workspace.total_commit_rows() == 3);
    CHECK(workspace.pending(0)[0] == 110); // row 1 from TARGET verification

    llama_token token = -1;
    llama_pos position = -1;
    const float * hidden = nullptr;
    CHECK(workspace.commit_row(0, 0, token, position, hidden));
    CHECK(token == 7 && position == 40 && hidden[0] == 10 && hidden[1] == 11);
    CHECK(workspace.commit_row(0, 1, token, position, hidden));
    CHECK(token == 8 && position == 41 && hidden[0] == 100 && hidden[1] == 101);
    CHECK(!workspace.commit_row(0, 2, token, position, hidden)); // rejected suffix
    CHECK(workspace.commit_row(1, 0, token, position, hidden));
    CHECK(token == 17 && position == 90 && hidden[0] == 20);
    workspace.clear_staged();

    for (uint32_t rows : {1u, 4u, 2u, 3u, 1u}) {
        CHECK(workspace.begin(rows, verified, tokens, positions));
        CHECK(workspace.sequence(0, 0, rows, true, true));
        CHECK(workspace.accept(0, 0));
        CHECK(workspace.commit_rows(0) == 1); // zero accepted candidates still catches sampled row
        CHECK(!workspace.commit_row(0, 1, token, position, hidden));
        CHECK(workspace.pending(0) == carry);
        CHECK(workspace.hidden_storage_bytes() == capacity_bytes);
        workspace.clear_staged();
    }
    CHECK(!workspace.begin(9, verified, tokens, positions));
    CHECK(workspace.pending(0) == carry && workspace.pending(0)[0] == 100);
    workspace.reset_sequence(0);
    CHECK(workspace.pending(0)[0] == 0 && workspace.pending(1)[0] == 200);
    CHECK(!workspace.accept(0, 0));
    workspace.reset();
    CHECK(workspace.pending(1)[0] == 0);
    CHECK(workspace.hidden_storage_bytes() == capacity_bytes);
}

static void batch_owns_both_inputs() {
    common_mtp_batch_storage storage(8, 4);
    auto batch = storage.view();
    CHECK(batch.token && batch.embd && batch.pos && batch.n_seq_id && batch.seq_id && batch.logits);
    for (int i = 0; i < 8; ++i) {
        CHECK(batch.seq_id[i]);
        batch.token[i] = i;
        batch.embd[4 * i] = float(i);
        batch.seq_id[i][0] = i & 1;
    }
    CHECK(batch.seq_id[8] == nullptr);
    auto next = storage.view();
    CHECK(next.n_tokens == 0 && next.token == batch.token && next.embd == batch.embd);
    CHECK(next.token[7] == 7 && next.embd[28] == 7);

    bool refused = false;
    try {
        common_mtp_batch_storage invalid(0, 1);
    } catch (const std::length_error &) {
        refused = true;
    }
    CHECK(refused);
    refused = false;
    try {
        common_mtp_workspace invalid(UINT32_MAX, UINT32_MAX, UINT32_MAX);
    } catch (const std::length_error &) {
        refused = true;
    }
    CHECK(refused);
}

static void device_workspace_keeps_only_row_metadata() {
    common_mtp_workspace workspace(1, 8, 10240, false);
    CHECK(workspace.hidden_storage_bytes() == 0 && workspace.pending(0) == nullptr);
    const llama_token tokens[] = {7, 8, 9, 10};
    const llama_pos positions[] = {40, 41, 42, 43};
    for (uint32_t rows : {4u, 1u, 3u, 2u, 4u}) {
        CHECK(workspace.begin(rows, nullptr, tokens, positions));
        CHECK(workspace.sequence(0, 0, rows, true, true));
        CHECK(workspace.first_row(0) == 0 && workspace.verified_rows(0) == rows);
        CHECK(!workspace.begin(1, nullptr, tokens, positions));
        CHECK(workspace.accept(0, rows - 1));
        CHECK(workspace.commit_rows(0) == rows);
        llama_token token = -1;
        llama_pos position = -1;
        const float * hidden = reinterpret_cast<const float *>(1);
        CHECK(workspace.commit_row(0, rows - 1, token, position, hidden));
        CHECK(token == tokens[rows - 1] && position == positions[rows - 1] && hidden == nullptr);
        CHECK(workspace.hidden_storage_bytes() == 0);
        workspace.clear_staged();
    }
    workspace.reset_sequence(0);
    workspace.reset();
    CHECK(!workspace.accept(0, 0) && workspace.hidden_storage_bytes() == 0);
    common_mtp_batch_storage batch_storage(8, 10240, false);
    const auto a = batch_storage.view(), b = batch_storage.view();
    CHECK(a.token && a.pos && a.seq_id && a.embd == nullptr && a.token == b.token);
}

static void device_hidden_slices_use_actual_rows() {
    ggml_tensor seed{}, target{};
    const size_t row_bytes = 10240 * sizeof(float);
    llama_predefined_hidden_store::bound_range plan[] = {
        {&seed, 0, 0, 1}, {&target, 0, 1, 3},
    };
    llama_device_hidden_input out{};
    const int32_t first[] = {0, 1};
    CHECK(llama_predefined_hidden_slice(plan, 2, 4, 10240, first, 2, nullptr, out));
    CHECK(out.rows == 2 && out.count == 2);
    CHECK(out.ranges[0].src == &seed && out.ranges[0].bytes == row_bytes);
    CHECK(out.ranges[1].src == &target && out.ranges[1].src_offset == 0 &&
          out.ranges[1].dst_offset == row_bytes && out.ranges[1].bytes == row_bytes);
    const int32_t tail[] = {2, 3};
    CHECK(llama_predefined_hidden_slice(plan, 2, 4, 10240, tail, 2, nullptr, out));
    CHECK(out.count == 1 && out.ranges[0].src == &target &&
          out.ranges[0].src_offset == row_bytes && out.ranges[0].dst_offset == 0 &&
          out.ranges[0].bytes == 2 * row_bytes);
    const auto previous = out;
    const int32_t reversed[] = {1, 0}, outside[] = {4};
    CHECK(!llama_predefined_hidden_slice(plan, 2, 4, 10240, reversed, 2, nullptr, out));
    CHECK(out.rows == previous.rows && out.count == previous.count);
    CHECK(!llama_predefined_hidden_slice(plan, 2, 4, 10240, outside, 1, nullptr, out));
    CHECK(!llama_predefined_hidden_slice(plan, 2, 4, 10240, nullptr, 1, nullptr, out));
    plan[1].dst_row = 0; // overlap is not repaired by treating capacity as rows
    CHECK(!llama_predefined_hidden_slice(plan, 2, 4, 10240, first, 2, nullptr, out));
    plan[1].dst_row = 1;
    plan[1].src_offset = SIZE_MAX - 7;
    CHECK(!llama_predefined_hidden_slice(plan, 2, 4, 10240, first, 2, nullptr, out));
}

static void device_hidden_rejects_stale_generation() {
    llama_predefined_hidden_store source;
    source.generation = 7;
    source.valid_rows = 4;
    CHECK(llama_predefined_hidden_generation_matches(source, LLAMA_PREDEFINED_H_RESULT, 7));
    CHECK(!llama_predefined_hidden_generation_matches(source, LLAMA_PREDEFINED_H_RESULT, 6));
    CHECK(!llama_predefined_hidden_generation_matches(source, LLAMA_PREDEFINED_H_RESULT, 0));
    CHECK(llama_predefined_hidden_generation_matches(source, LLAMA_PREDEFINED_H_SEED, 0));
    CHECK(!llama_predefined_hidden_generation_matches(source, LLAMA_PREDEFINED_H_SEED, 7));
    CHECK(!llama_predefined_hidden_generation_matches(source, LLAMA_PREDEFINED_H_INPUT, 0));
    source.valid_rows = 0; // failed/partial capture must not become a snapshot
    CHECK(!llama_predefined_hidden_generation_matches(source, LLAMA_PREDEFINED_H_RESULT, 7));
}

int main() {
    try {
        accepted_target_hidden_is_not_draft_hidden();
        batch_owns_both_inputs();
        device_workspace_keeps_only_row_metadata();
        device_hidden_slices_use_actual_rows();
        device_hidden_rejects_stale_generation();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    std::puts("fixed MTP workspace/accepted-row checks passed");
    return 0;
}
