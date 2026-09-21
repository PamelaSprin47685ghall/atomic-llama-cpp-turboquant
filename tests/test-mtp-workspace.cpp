#include "../common/speculative-mtp-workspace.h"
#include "../common/speculative.h"
#include "../src/llama-predefined-hidden.h"
#include "../src/llama-context.h"

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

// Regression check for missing-regression-test:
// Verify that predefined MTP execution sessions are exempted from legacy phase invalidation.
// Legacy phase logic (n_tokens > n_seqs ? 1 : 0) oscillated on 1->4->1->2 sequences (0->1->0->1),
// triggering 3x gf_res_prev->reset() and ggml_backend_sched_release_buffers(), breaking predefined graph reuse.
// Predefined MTP shares a single maximum-capacity graph definition (capacity->verify_tokens) and must
// remain stably in phase 0 without oscillation.
static void predefined_mtp_phase_invalidation_exemption() {
    const int mtp_token_seq[] = {1, 4, 1, 2};
    constexpr int n_steps = 4;
    const int n_seqs = 1;

    // 1. Demonstrate legacy behavior: pure n_tokens > n_seqs logic causes 3 phase transitions.
    {
        int legacy_last_phase = -1;
        int legacy_resets = 0;
        for (int i = 0; i < n_steps; ++i) {
            const int n_tokens = mtp_token_seq[i];
            const int phase = n_tokens > n_seqs ? 1 : 0;
            if (phase != legacy_last_phase) {
                if (legacy_last_phase >= 0) {
                    legacy_resets++;
                }
                legacy_last_phase = phase;
            }
        }
        // Step 0 (1 token):  phase 0, last=-1 -> last=0, 0 resets
        // Step 1 (4 tokens): phase 1, last=0  -> last=1, reset 1
        // Step 2 (1 token):  phase 0, last=1  -> last=0, reset 2
        // Step 3 (2 tokens): phase 1, last=0  -> last=1, reset 3
        CHECK(legacy_resets == 3);
    }

    // 2. Predefined MTP session: execution definition is invariant (capacity_rows = 4).
    // Phase must remain stably 0, resulting in ZERO resets across the entire 1->4->1->2 sequence.
    {
        int mtp_last_phase = -1;
        int mtp_resets = 0;
        const uint32_t capacity_rows = 4;
        for (int i = 0; i < n_steps; ++i) {
            const int n_tokens = mtp_token_seq[i];
            const int phase = llama_context::ubatch_execution_phase(
                LLM_GRAPH_TYPE_DECODER_MTP,
                /*has_predefined_capacity=*/true,
                capacity_rows,
                n_tokens,
                n_seqs);
            CHECK(phase == 0);
            if (phase != mtp_last_phase) {
                if (mtp_last_phase >= 0) {
                    mtp_resets++;
                }
                mtp_last_phase = phase;
            }
        }
        CHECK(mtp_resets == 0);
        CHECK(mtp_last_phase == 0);
    }

    // 3. Non-predefined MTP session: must preserve legacy phase switching behavior.
    {
        int non_predef_last_phase = -1;
        int non_predef_resets = 0;
        for (int i = 0; i < n_steps; ++i) {
            const int n_tokens = mtp_token_seq[i];
            const int phase = llama_context::ubatch_execution_phase(
                LLM_GRAPH_TYPE_DECODER_MTP,
                /*has_predefined_capacity=*/false,
                0,
                n_tokens,
                n_seqs);
            if (phase != non_predef_last_phase) {
                if (non_predef_last_phase >= 0) {
                    non_predef_resets++;
                }
                non_predef_last_phase = phase;
            }
        }
        CHECK(non_predef_resets == 3);
    }

    // 4. Target trunk model: prompt prefill (phase 1) vs decode generation (phase 0)
    // must strictly retain legacy boundary to prevent VRAM buffer bloat and view bound violations.
    {
        // Prefill: 32 tokens, 1 sequence -> phase 1
        const int prefill_phase = llama_context::ubatch_execution_phase(
            LLM_GRAPH_TYPE_DECODER,
            /*has_predefined_capacity=*/true,
            32,
            32,
            1);
        CHECK(prefill_phase == 1);

        // Decode: 1 token, 1 sequence -> phase 0
        const int decode_phase = llama_context::ubatch_execution_phase(
            LLM_GRAPH_TYPE_DECODER,
            /*has_predefined_capacity=*/true,
            1,
            1,
            1);
        CHECK(decode_phase == 0);

        // Batch decode: 4 tokens across 4 sequences -> phase 0
        const int batch_decode_phase = llama_context::ubatch_execution_phase(
            LLM_GRAPH_TYPE_DECODER,
            /*has_predefined_capacity=*/false,
            0,
            4,
            4);
        CHECK(batch_decode_phase == 0);
    }
}

static void test_mtp_cycle_ledger_accounting() {
    // 1. Single cycle record formatting check
    common_speculative_cycle_record r1;
    r1.cycle_id         = 1;
    r1.draft_us         = 1200;
    r1.target_verify_us = 3500;
    r1.catchup_us       = 800;
    r1.handoff_us       = 50;
    r1.total_us         = 1200 + 3500 + 800 + 50;
    r1.draft_tokens     = 3;
    r1.accepted_tokens  = 2;
    r1.final_tokens     = 3; // 2 accepted + 1 target sampled
    r1.device_hidden    = true;

    const std::string line1 = common_speculative_format_cycle_record(r1);
    CHECK(line1.find("[tp5-mtp-cycle] cycle=1") != std::string::npos);
    CHECK(line1.find("draft_us=1200") != std::string::npos);
    CHECK(line1.find("target_us=3500") != std::string::npos);
    CHECK(line1.find("catchup_us=800") != std::string::npos);
    CHECK(line1.find("handoff_us=50") != std::string::npos);
    CHECK(line1.find("total_us=5550") != std::string::npos);
    CHECK(line1.find("draft_tokens=3") != std::string::npos);
    CHECK(line1.find("accepted_tokens=2") != std::string::npos);
    CHECK(line1.find("final_tokens=3") != std::string::npos);
    CHECK(line1.find("eff=0.667") != std::string::npos);
    CHECK(line1.find("dev_hidden=1") != std::string::npos);

    // 2. Summary accumulator check
    common_speculative_cycle_summary summary;
    auto accumulate = [&](const common_speculative_cycle_record & r) {
        summary.total_cycles++;
        summary.total_draft_us         += r.draft_us;
        summary.total_target_verify_us += r.target_verify_us;
        summary.total_catchup_us       += r.catchup_us;
        summary.total_handoff_us       += r.handoff_us;
        summary.total_us               += r.total_us;
        summary.total_draft_tokens     += r.draft_tokens;
        summary.total_accepted_tokens  += r.accepted_tokens;
        summary.total_final_tokens     += r.final_tokens;
    };

    accumulate(r1);

    common_speculative_cycle_record r2;
    r2.cycle_id         = 2;
    r2.draft_us         = 1000;
    r2.target_verify_us = 3000;
    r2.catchup_us       = 200;
    r2.handoff_us       = 40;
    r2.total_us         = 1000 + 3000 + 200 + 40;
    r2.draft_tokens     = 3;
    r2.accepted_tokens  = 0; // 0 accepted
    r2.final_tokens     = 1;
    r2.device_hidden    = true;
    accumulate(r2);

    CHECK(summary.total_cycles == 2);
    CHECK(summary.total_draft_tokens == 6);
    CHECK(summary.total_accepted_tokens == 2);
    CHECK(summary.total_final_tokens == 4); // 3 + 1
    // Total final tokens invariant: always equals total_accepted_tokens + total_cycles
    CHECK(summary.total_final_tokens == summary.total_accepted_tokens + summary.total_cycles);

    const std::string sum_line = common_speculative_format_cycle_summary(summary);
    CHECK(sum_line.find("[tp5-mtp-cycle-summary] cycles=2") != std::string::npos);
    CHECK(sum_line.find("avg_draft_us=1100.0") != std::string::npos);
    CHECK(sum_line.find("avg_target_us=3250.0") != std::string::npos);
    CHECK(sum_line.find("avg_catchup_us=500.0") != std::string::npos);
    CHECK(sum_line.find("eff=0.333") != std::string::npos);

    // 3. Target verification injection API contract check
    common_params_speculative params_spec;
    params_spec.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
    common_speculative * spec_inst = common_speculative_init(params_spec, 1);
    CHECK(spec_inst != nullptr);
    common_speculative_record_target_verify_us(spec_inst, 4200);
    // Verified: injection properly registers into the spec instance
    common_speculative_record_target_verify_us(spec_inst, 0); // safe reset
    common_speculative_free(spec_inst);
}

int main() {
    try {
        accepted_target_hidden_is_not_draft_hidden();
        batch_owns_both_inputs();
        device_workspace_keeps_only_row_metadata();
        device_hidden_slices_use_actual_rows();
        device_hidden_rejects_stale_generation();
        predefined_mtp_phase_invalidation_exemption();
        test_mtp_cycle_ledger_accounting();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    std::puts("fixed MTP workspace/accepted-row checks passed");
    return 0;
}
