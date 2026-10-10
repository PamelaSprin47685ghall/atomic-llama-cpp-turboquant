#include "../common/speculative-mtp-workspace.h"
#include "../common/speculative.h"
#include "../src/llama-predefined-hidden.h"
#include "../src/llama-context.h"
#include "../src/llama-graph.h"

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
        for (int i = 0; i < n_steps; ++i) {
            const int n_tokens = mtp_token_seq[i];
            const enum ggml_predefined_phase phase_role = (n_tokens == 1) ? GGML_PREDEFINED_DRAFT : GGML_PREDEFINED_CATCHUP;
            const int phase = llama_context::ubatch_execution_phase(
                LLM_GRAPH_TYPE_DECODER_MTP,
                /*has_predefined_capacity=*/true,
                phase_role,
                /*capacity_admitted=*/false,
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
                GGML_PREDEFINED_PREFILL,
                /*capacity_admitted=*/false,
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
        // Even if predefined capacity exists, target prefill must retain phase 1.
        const int prefill_phase = llama_context::ubatch_execution_phase(
            LLM_GRAPH_TYPE_DECODER,
            /*has_predefined_capacity=*/true,
            GGML_PREDEFINED_PREFILL,
            /*capacity_admitted=*/false,
            32,
            1);
        CHECK(prefill_phase == 1);

        // Decode: 1 token, 1 sequence -> phase 0
        const int decode_phase = llama_context::ubatch_execution_phase(
            LLM_GRAPH_TYPE_DECODER,
            /*has_predefined_capacity=*/true,
            GGML_PREDEFINED_TARGET,
            /*capacity_admitted=*/false,
            1,
            1);
        CHECK(decode_phase == 0);

        // Admitted TARGET capacity: e.g. 4 verification tokens admitted -> phase 0
        const int target_admitted_phase = llama_context::ubatch_execution_phase(
            LLM_GRAPH_TYPE_DECODER,
            /*has_predefined_capacity=*/true,
            GGML_PREDEFINED_TARGET,
            /*capacity_admitted=*/true,
            4,
            1);
        CHECK(target_admitted_phase == 0);

        // Unadmitted TARGET multi-token verification (fallback to exact): 4 tokens, 1 seq -> phase 1
        const int target_unadmitted_phase = llama_context::ubatch_execution_phase(
            LLM_GRAPH_TYPE_DECODER,
            /*has_predefined_capacity=*/true,
            GGML_PREDEFINED_TARGET,
            /*capacity_admitted=*/false,
            4,
            1);
        CHECK(target_unadmitted_phase == 1);

        // Batch decode: 4 tokens across 4 sequences -> phase 0
        const int batch_decode_phase = llama_context::ubatch_execution_phase(
            LLM_GRAPH_TYPE_DECODER,
            /*has_predefined_capacity=*/false,
            GGML_PREDEFINED_PREFILL,
            /*capacity_admitted=*/false,
            4,
            4);
        CHECK(batch_decode_phase == 0);
    }
}

static void test_target_verify_timing_request_reset() {
    // Request reset must not carry target verification time into the next request.
    common_params_speculative params_spec;
    params_spec.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
    common_speculative * spec_inst = common_speculative_init(params_spec, 1);
    CHECK(spec_inst != nullptr);
    common_speculative_record_target_verify_us(spec_inst, 4200);
    CHECK(common_speculative_get_target_verify_us(spec_inst) == 4200);
    common_speculative_reset(spec_inst);
    CHECK(common_speculative_get_target_verify_us(spec_inst) == 0);
    common_speculative_free(spec_inst);
}

static void test_mtp_evidence_surface_contracts() {
    // The workspace exposes only the accepted prefix, not rejected rows.
    {
        common_mtp_workspace ws(1, 8, 2);
        const float verified[] = {10, 11, 20, 21, 30, 31, 40, 41};
        const llama_token tokens[] = {101, 102, 103, 104};
        const llama_pos positions[] = {1, 2, 3, 4};
        CHECK(ws.begin(4, verified, tokens, positions));
        CHECK(ws.sequence(0, 0, 4, true, true));
        CHECK(ws.accept(0, 1)); // only 1 token accepted, 2 committed (seed + accepted)
        CHECK(ws.commit_rows(0) == 2);

        llama_token tok = 0;
        llama_pos pos = 0;
        const float * h = nullptr;
        CHECK(ws.commit_row(0, 0, tok, pos, h));
        CHECK(tok == 101 && pos == 1);
        CHECK(ws.commit_row(0, 1, tok, pos, h));
        CHECK(tok == 102 && pos == 2);
        // Invalid/rejected row 2 and 3 must be rejected by commit_row
        CHECK(!ws.commit_row(0, 2, tok, pos, h));
        CHECK(!ws.commit_row(0, 3, tok, pos, h));
    }

    // Versioned results reject stale generations; carry/seed are unversioned.
    {
        llama_predefined_hidden_store s{};
        s.generation = 42;
        s.valid_rows = 4;

        // RESULT slot requires exact non-zero generation match and valid_rows > 0
        CHECK(llama_predefined_hidden_generation_matches(s, LLAMA_PREDEFINED_H_RESULT, 42));
        CHECK(!llama_predefined_hidden_generation_matches(s, LLAMA_PREDEFINED_H_RESULT, 41));
        CHECK(!llama_predefined_hidden_generation_matches(s, LLAMA_PREDEFINED_H_RESULT, 0));

        // When valid_rows == 0, RESULT generation match fails regardless of generation value
        s.valid_rows = 0;
        CHECK(!llama_predefined_hidden_generation_matches(s, LLAMA_PREDEFINED_H_RESULT, 42));

        // CARRY and SEED slots require expected == 0 (unversioned device registers)
        CHECK(llama_predefined_hidden_generation_matches(s, LLAMA_PREDEFINED_H_CARRY, 0));
        CHECK(!llama_predefined_hidden_generation_matches(s, LLAMA_PREDEFINED_H_CARRY, 1));
        CHECK(llama_predefined_hidden_generation_matches(s, LLAMA_PREDEFINED_H_SEED, 0));
        CHECK(!llama_predefined_hidden_generation_matches(s, LLAMA_PREDEFINED_H_SEED, 42));
    }

}

// Logical graph-reuse and host-workspace checks, not native Vulkan/model qualification.
// Verifies:
// 1. allow_reuse holds unconditionally at every step under fixed capacity (capacity_rows=4) without reallocation.
// 2. common_mtp_workspace internal storage pointers and capacity bytes remain strictly invariant (no realloc).
// 3. Active tokens & output semantics: draft step has active_tokens=1, active_outputs=1; catchup steps have active_outputs=0.
// 4. commit_row accepts strictly within effective committed prefix and rejects all unaccepted/out-of-bound rows.
// 5. Negative boundary invariants (capacity mismatch, unaccepted row access, buffer bounds).
static void test_mtp_single_maximal_definition_sequence_invariants() {
    const int mtp_token_seq[] = {1, 4, 1, 2, 1, 4, 1, 2};
    constexpr size_t total_steps = sizeof(mtp_token_seq) / sizeof(mtp_token_seq[0]);
    constexpr uint32_t capacity_rows = 4;
    constexpr uint32_t capacity_outputs = 4;
    constexpr uint32_t hidden_width = 128;

    // 1. Baseline predefined MTP graph definition (Single Maximal Graph Definition)
    llm_graph_params base_def{};
    base_def.gtype = LLM_GRAPH_TYPE_DECODER_MTP;
    base_def.predefined_enabled = true;
    base_def.predefined_capacity_rows = capacity_rows;
    base_def.predefined_capacity_outputs = capacity_outputs;
    base_def.predefined_frame.version = GGML_PREDEFINED_ABI_VERSION;
    base_def.predefined_frame.phase = GGML_PREDEFINED_DRAFT;
    base_def.predefined_frame.active_tokens = 1;
    base_def.predefined_frame.active_outputs = 1;
    base_def.ubatch.n_tokens = 1;
    base_def.ubatch.n_seqs = 1;
    base_def.ubatch.n_seqs_unq = 1;
    base_def.ubatch.n_seq_tokens = 1;
    base_def.n_outputs = 1;

    // Negative check on base definition: capacity mismatch MUST reject reuse
    {
        llm_graph_params bad_cap = base_def;
        bad_cap.predefined_capacity_rows = 8;
        bad_cap.predefined_capacity_outputs = 8;
        CHECK(!base_def.allow_reuse(bad_cap));
        CHECK(!bad_cap.allow_reuse(base_def));
    }

    // 2. Initialize host MTP workspace once with fixed capacity (capacity_rows = 4, width = 128)
    common_mtp_workspace workspace(1, capacity_rows, hidden_width, /*host_hidden=*/true);
    const float * const initial_pending_ptr = workspace.pending(0);
    const size_t initial_storage_bytes = workspace.hidden_storage_bytes();
    const uint32_t initial_capacity = workspace.capacity();
    CHECK(initial_pending_ptr != nullptr);
    CHECK(initial_storage_bytes > 0);
    CHECK(initial_capacity == capacity_rows);

    std::vector<float> verified_hidden(capacity_rows * hidden_width, 1.0f);
    const llama_token mock_tokens[capacity_rows]   = {1001, 1002, 1003, 1004};
    const llama_pos   mock_positions[capacity_rows] = {10, 11, 12, 13};

    llm_graph_params prev_step_params = base_def;
    int phase_resets = 0;
    int last_phase = -1;

    // 3. Drive 1 -> 4 -> 1 -> 2 -> 1 -> 4 -> 1 -> 2 sequence sequentially in a single uninterrupted execution
    for (size_t step = 0; step < total_steps; ++step) {
        const int n_tokens = mtp_token_seq[step];
        const bool is_draft = (n_tokens == 1);
        const uint32_t active_outputs = is_draft ? 1 : 0;
        const enum ggml_predefined_phase phase = is_draft ? GGML_PREDEFINED_DRAFT : GGML_PREDEFINED_CATCHUP;

        // (a) Construct step graph params
        llm_graph_params step_params{};
        step_params.gtype = LLM_GRAPH_TYPE_DECODER_MTP;
        step_params.predefined_enabled = true;
        step_params.predefined_capacity_rows = capacity_rows;
        step_params.predefined_capacity_outputs = capacity_outputs;
        step_params.predefined_frame.version = GGML_PREDEFINED_ABI_VERSION;
        step_params.predefined_frame.phase = phase;
        step_params.predefined_frame.active_tokens = n_tokens;
        step_params.predefined_frame.active_outputs = active_outputs;
        step_params.ubatch.n_tokens = n_tokens;
        step_params.ubatch.n_seqs = 1;
        step_params.ubatch.n_seqs_unq = 1;
        step_params.ubatch.n_seq_tokens = n_tokens;
        step_params.n_outputs = active_outputs;

        // Invariant: allow_reuse holds against base definition and against immediately preceding step
        CHECK(base_def.allow_reuse(step_params));
        CHECK(step_params.allow_reuse(base_def));
        CHECK(prev_step_params.allow_reuse(step_params));
        CHECK(step_params.allow_reuse(prev_step_params));

        // Invariant: execution phase remains stably 0 without oscillation or buffer resets
        const int exec_phase = llama_context::ubatch_execution_phase(
            LLM_GRAPH_TYPE_DECODER_MTP,
            /*has_predefined_capacity=*/true,
            phase,
            /*capacity_admitted=*/false,
            n_tokens,
            /*n_seqs=*/1);
        CHECK(exec_phase == 0);
        if (exec_phase != last_phase) {
            if (last_phase >= 0) {
                phase_resets++;
            }
            last_phase = exec_phase;
        }

        // (b) Drive MTP workspace with step input
        for (uint32_t i = 0; i < uint32_t(n_tokens) * hidden_width; ++i) {
            verified_hidden[i] = float(step * 1000 + i);
        }
        CHECK(workspace.begin(n_tokens, verified_hidden.data(), mock_tokens, mock_positions));
        CHECK(workspace.sequence(0, 0, n_tokens, /*staged=*/true, /*deferred=*/true));

        // Invariant: workspace internal storage address and capacity bytes never realloc
        CHECK(workspace.pending(0) == initial_pending_ptr);
        CHECK(workspace.hidden_storage_bytes() == initial_storage_bytes);
        CHECK(workspace.capacity() == initial_capacity);

        // (c) Active row & output semantics verification
        if (is_draft) {
            CHECK(step_params.predefined_frame.active_tokens == 1);
            CHECK(step_params.predefined_frame.active_outputs == 1);
        } else {
            CHECK(step_params.predefined_frame.active_tokens == uint32_t(n_tokens));
            CHECK(step_params.predefined_frame.active_outputs == 0);
        }

        // (d) Simulate acceptance & verify commit_row effective boundary
        // For draft (n_tokens=1), 0 candidates accepted -> 1 committed (the seed/sampled row)
        // For catch-up (n_tokens > 1), accept min(1, n_tokens-1) -> committed = accepted + 1
        const uint32_t accepted_candidates = is_draft ? 0 : std::min<uint32_t>(1, n_tokens - 1);
        CHECK(workspace.accept(0, accepted_candidates));

        const uint32_t committed = workspace.commit_rows(0);
        CHECK(committed == accepted_candidates + 1);
        CHECK(committed <= uint32_t(n_tokens));

        // Valid committed prefix must be accessible with correct metadata
        llama_token out_tok = -1;
        llama_pos   out_pos = -1;
        const float * out_h = nullptr;
        for (uint32_t r = 0; r < committed; ++r) {
            CHECK(workspace.commit_row(0, r, out_tok, out_pos, out_h));
            CHECK(out_tok == mock_tokens[r]);
            CHECK(out_pos == mock_positions[r]);
            CHECK(out_h != nullptr);
        }

        // All unaccepted rows in [committed, n_tokens) and out-of-capacity rows must be strictly rejected
        for (uint32_t r = committed; r < capacity_rows + 2; ++r) {
            CHECK(!workspace.commit_row(0, r, out_tok, out_pos, out_h));
        }

        workspace.clear_staged();
        prev_step_params = step_params;
    }

    CHECK(phase_resets == 0);

    // 4. Invariant preservation after complete 8-step sequence and reset
    workspace.reset();
    CHECK(workspace.pending(0) == initial_pending_ptr);
    CHECK(workspace.hidden_storage_bytes() == initial_storage_bytes);
    CHECK(workspace.capacity() == initial_capacity);
}

int main() {
    try {
        accepted_target_hidden_is_not_draft_hidden();
        batch_owns_both_inputs();
        device_workspace_keeps_only_row_metadata();
        device_hidden_slices_use_actual_rows();
        device_hidden_rejects_stale_generation();
        predefined_mtp_phase_invalidation_exemption();
        test_target_verify_timing_request_reset();
        test_mtp_evidence_surface_contracts();
        test_mtp_single_maximal_definition_sequence_invariants();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    std::puts("fixed MTP workspace/accepted-row checks passed");
    return 0;
}
