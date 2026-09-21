#include "../common/speculative-mtp-workspace.h"

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

int main() {
    try {
        accepted_target_hidden_is_not_draft_hidden();
        batch_owns_both_inputs();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    std::puts("fixed MTP workspace/accepted-row checks passed");
    return 0;
}
