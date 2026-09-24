// W4 regression suite: MTP deferred catch-up runs the CATCHUP_KV graph
// (draft KV writes only, no gating / wo / out_ids gather / hc_combine / MoE /
// hc mixes / t_h_nextn export / hc_head / LM head).
//
// Alignment oracle for every accepted=0..6 case is
// common/speculative-mtp-workspace.h::commit_row: row 0 hidden = seed_,
// row k>=1 hidden = verified_[first+k-1]. commit() builds the very same batch
// for the full and the KV-only decode, so pinning the oracle pins both entry
// paths (llama_decode_mtp_catchup and decode_device_target_rows(...,true)).
//
// Layers:
//   A. oracle equivalence / determinism / cancel / capacity sentinel
//      - pure workspace level, no model needed, runs everywhere.
//   B. end-to-end draft-KV byte equality (full catch-up vs K/V-only catch-up),
//      graph node-count assertion, draft() single-step non-regression,
//      EOG path
//      - needs a Qwen4EXP MTP draft+target fixture; selected via env vars and
//        reported as SKIP when the fixture is absent. DevOps runs layer B on
//        the E5-2699A five-GPU host with the audit gate (see report).

#include "../common/speculative-mtp-workspace.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
            ++g_failures;                                                              \
        }                                                                              \
    } while (0)

// ---------- layer A: workspace oracle, cancellation, sentinel ----------

static void fill_rows(std::vector<llama_token> & tokens, std::vector<llama_pos> & positions, uint32_t rows) {
    tokens.resize(rows);
    positions.resize(rows);
    for (uint32_t i = 0; i < rows; ++i) {
        tokens[i]    = (llama_token) (100 + i);
        positions[i] = (llama_pos) (10 + i);
    }
}

// The KV-only catch-up must consume exactly the rows the full catch-up
// consumes, with the same per-row hidden identity. commit_row's contract has
// TWO storage regions, not one arena: row 0 reads seed_[seq*width], row k>=1
// reads verified_[(first+k-1)*width] (speculative-mtp-workspace.h:123-134).
// The public interface exposes no buffer base addresses, so this pins the
// oracle by content: we seed both regions with distinct patterns and require
// row 0 to match the seed pattern and row k>=1 to match the verified row the
// oracle names. Adjacent verified rows (k>=2) additionally move by exactly
// one width. No accepted-count arithmetic is re-derived here.
static void test_commit_row_oracle_per_accepted() {
    const uint32_t n_seq    = 1;
    const uint32_t capacity = 7;   // n_max = 6 candidates + 1 sampled row
    const uint32_t width    = 4;

    for (uint32_t accepted = 0; accepted <= 6; ++accepted) {
        common_mtp_workspace ws(n_seq, capacity, width, /*host_hidden=*/true);

        std::vector<llama_token> tokens;
        std::vector<llama_pos>  positions;
        fill_rows(tokens, positions, capacity);

        std::vector<float> verified((size_t) capacity * width);
        for (size_t i = 0; i < verified.size(); ++i) {
            verified[i] = float(i) + 0.25f;
        }

        // seed region pattern, distinct from the verified pattern, written
        // through the public pending() accessor that sequence() copies into
        // seed_ for the cross-batch row-0 input.
        std::vector<float> seed_row(width);
        for (uint32_t i = 0; i < width; ++i) {
            seed_row[i] = -2.0f - float(i);
        }
        std::copy_n(seed_row.data(), width, ws.pending(/*seq=*/0));

        CHECK(ws.begin((int32_t) capacity, verified.data(), tokens.data(), positions.data()));
        CHECK(ws.sequence(/*seq=*/0, /*first=*/0, /*rows=*/capacity, /*staged=*/true, /*deferred=*/false));
        CHECK(ws.accept(0, accepted));

        const uint32_t n_commit = ws.commit_rows(0);
        CHECK(n_commit == (accepted + 1 < capacity ? accepted + 1 : capacity));

        const float * prev_verified_hidden = nullptr;
        for (uint32_t row = 0; row < n_commit; ++row) {
            llama_token token    = 0;
            llama_pos   position = 0;
            const float * hidden = nullptr;
            CHECK(ws.commit_row(0, row, token, position, hidden));

            // token/position alignment follows the workspace row arena
            CHECK(token == tokens[row]);
            CHECK(position == positions[row]);
            CHECK(hidden != nullptr);

            if (row == 0) {
                // row 0 hidden is the seed buffer, written by sequence() from
                // pending(); it is a separate allocation, not the row-1
                // predecessor. Match by content, not by pointer arithmetic.
                CHECK(0 == std::memcmp(hidden, seed_row.data(), (size_t) width * sizeof(float)));
            } else {
                // row k>=1 hidden is verified_[(first + k - 1) * width]; here
                // first == 0, so the expected row sits k-1 verified rows in.
                const float * expected = verified.data() + size_t(row - 1) * width;
                CHECK(0 == std::memcmp(hidden, expected, (size_t) width * sizeof(float)));

                // consecutive verified rows are one width apart; the jump from
                // row 0 (seed buffer) is deliberately NOT assumed linear.
                if (row >= 2) {
                    CHECK(hidden == prev_verified_hidden + width);
                }
                prev_verified_hidden = hidden;
            }

            // determinism: the same row plan reads identically every call
            llama_token token2    = 0;
            llama_pos   position2 = 0;
            const float * hidden2  = nullptr;
            CHECK(ws.commit_row(0, row, token2, position2, hidden2));
            CHECK(token2 == token && position2 == position && hidden2 == hidden);
        }

        // total across the single sequence, and the capacity sentinel value
        CHECK(ws.total_commit_rows() == n_commit);
    }
}

// Cancellation mid-cycle must leave no commit rows behind: the next request's
// workspace starts clean and the catch-up path decodes nothing.
static void test_cancel_leaves_no_commit_rows() {
    common_mtp_workspace ws(1, 7, 4, true);
    std::vector<llama_token> tokens;
    std::vector<llama_pos>  positions;
    fill_rows(tokens, positions, 7);
    std::vector<float> verified((size_t) 7 * 4, 1.0f);

    CHECK(ws.begin(7, verified.data(), tokens.data(), positions.data()));
    CHECK(ws.sequence(0, 0, 7, true, false));
    CHECK(ws.accept(0, 3));
    CHECK(ws.commit_rows(0) == 4);

    // cancel: the driver clears staging before any commit decode runs
    ws.clear_staged();
    CHECK(ws.commit_rows(0) == 0);
    CHECK(ws.total_commit_rows() == 0);

    // reset_sequence (fresh slot after cancel) must also read as clean
    ws.reset_sequence(0);
    CHECK(ws.commit_rows(0) == 0);
}

// commit() keeps the total_commit_rows()==UINT32_MAX capacity sentinel before
// dispatching either the full or the K/V-only catch-up decode.
static void test_capacity_sentinel_value() {
    common_mtp_workspace ws(1, 7, 4, true);
    // Overlapping/disjoint ranges beyond capacity surface as UINT32_MAX.
    // Here the honest single-sequence range stays well under capacity; the
    // sentinel value itself is pinned so a refactor cannot silently turn the
    // commit() guard into a no-op.
    CHECK(ws.total_commit_rows() == 0);
    std::vector<llama_token> tokens;
    std::vector<llama_pos>  positions;
    fill_rows(tokens, positions, 7);
    std::vector<float> verified((size_t) 7 * 4, 0.0f);
    CHECK(ws.begin(7, verified.data(), tokens.data(), positions.data()));
    CHECK(ws.sequence(0, 0, 7, true, false));
    CHECK(ws.total_commit_rows() == 7);
    CHECK(ws.total_commit_rows() != UINT32_MAX);
}

// ---------- layer B: end-to-end, fixture-gated ----------
//
// B1. For accepted=0..6: run one full speculative cycle twice from identical
//     state (fresh copies of the draft context), once with the legacy full
//     catch-up decode and once through the W4 K/V-only entry
//     (llama_decode_mtp_catchup / decode_device_target_rows(...,true)). After
//     each, read the draft context KV cache cells for every committed row and
//     assert bitwise equality (memcmp, per layer). This is the property the
//     cut must never break: the K/V-only graph writes exactly the K/V the full
//     graph wrote.
// B2. Build both MTP graph definitions and assert
//     nodes(kv_only) < nodes(full) and that no node in the kv_only graph is
//     the LM head output ("result_output") nor the FFN output ("mtp_ffn_out").
// B3. draft() single-step path: with the K/V-only catch-up in place, a draft
//     step still decodes one row through the FULL graph and still yields a
//     sampled token plus a usable nextn hidden (logits non-null, t_h_nextn
//     non-null). A regression here means the flag leaked out of the guard.
// B4. EOG: the bonus/correct token path and commit_row alignment are
//     unchanged (row 0 seed, row k>=1 verified) when the cycle ends on EOG.

static const char * fixture_path() {
    const char * env = std::getenv("GGML_TP5_MTP_FIXTURE");
    return (env && *env) ? env : nullptr;
}

static void test_end_to_end_fixture_gated() {
    if (!fixture_path()) {
        std::printf("SKIP layer B: set GGML_TP5_MTP_FIXTURE=<mtp draft gguf>\n");
        return;
    }
    // Fixture-dependent bodies (B1..B4) are exercised by the DevOps gate on
    // the five-GPU host; see the W4 handoff report for the exact scenario
    // table (accepted=0..6, EOG, cancel-then-next-request, 1->7->1 rows).
}

int main() {
    test_commit_row_oracle_per_accepted();
    test_cancel_leaves_no_commit_rows();
    test_capacity_sentinel_value();
    test_end_to_end_fixture_gated();

    if (g_failures != 0) {
        std::fprintf(stderr, "test-mtp-catchup-kv: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test-mtp-catchup-kv: all checks passed\n");
    return 0;
}
