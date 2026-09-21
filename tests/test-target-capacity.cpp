// CPU contract tests for TARGET verification capacity path (Stage 1).
// Pure CPU metadata & tensor shell tests; no GPU, no model weight load.
// Verifies dual-length contract:
// 1. Frame generation: active <= capacity, outputs == tokens for TARGET.
// 2. out_ids: capacity-sized allocation, active prefix written, inactive suffix zeroed, overflow throws.
// 3. input shell: tokens & pos bounds check throws when active > capacity.
// 4. Reuse key predefined_target_dynamic: capacity equal & active variable allows reuse.
// 5. Switch OFF: strictly preserves status quo exact-matching behavior.

#include "ggml-predefined.h"
#include "llama.h"
#include "llama-hparams.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        throw std::runtime_error("contract violation: " #cond); \
    } \
} while (0)

static ggml_predefined_limits make_test_limits() {
    // verify_tokens = 4, tokens = 32, wire_bytes = 4
    return {GGML_PREDEFINED_ABI_VERSION, sizeof(ggml_predefined_limits), 1, 3, 32, 256,
            4, 2560, 10240, 248320, 4, 5};
}

static void test_target_frame_contract() {
    fprintf(stderr, "--- test_target_frame_contract ---\n");
    char error[192] = {0};
    auto lim = make_test_limits();
    ggml_predefined_capacity cap{};
    CHECK(ggml_predefined_make_capacity(&lim, &cap, error, sizeof(error)));
    CHECK(cap.verify_tokens == 4);

    // TARGET valid frames: active <= verify_tokens (4), outputs == tokens
    for (uint32_t active = 1; active <= 4; ++active) {
        ggml_predefined_request req{};
        req.phase = GGML_PREDEFINED_TARGET;
        req.sequences = 1;
        req.tokens = active;
        req.outputs = active;
        req.context_tokens = 100;
        req.draft_tokens = (active >= 1) ? (active - 1) : 0;
        req.accepted_tokens = 0;
        req.draft_step = 0;

        ggml_predefined_frame frame{};
        CHECK(ggml_predefined_make_frame(&cap, &req, 100 + active, 0, &frame, error, sizeof(error)));
        CHECK(frame.active_tokens == active);
        CHECK(frame.active_outputs == active);
        CHECK(frame.active_tokens <= cap.verify_tokens);
        CHECK(frame.payload_elements == active * 2560);
    }

    // Invalid TARGET frames:
    // 1. active tokens > verify_tokens (e.g. 5 > 4)
    {
        ggml_predefined_request req{GGML_PREDEFINED_TARGET, 1, 5, 5, 100, 4, 0, 0};
        ggml_predefined_frame frame{};
        CHECK(!ggml_predefined_make_frame(&cap, &req, 200, 0, &frame, error, sizeof(error)));
    }
    // 2. outputs != tokens (TARGET requires outputs == tokens)
    {
        ggml_predefined_request req{GGML_PREDEFINED_TARGET, 1, 4, 3, 100, 3, 0, 0};
        ggml_predefined_frame frame{};
        CHECK(!ggml_predefined_make_frame(&cap, &req, 201, 0, &frame, error, sizeof(error)));
    }
    // 3. 0 tokens
    {
        ggml_predefined_request req{GGML_PREDEFINED_TARGET, 1, 0, 0, 100, 0, 0, 0};
        ggml_predefined_frame frame{};
        CHECK(!ggml_predefined_make_frame(&cap, &req, 202, 0, &frame, error, sizeof(error)));
    }
    fprintf(stderr, "  PASSED: test_target_frame_contract\n");
}

static void test_target_out_ids_capacity() {
    fprintf(stderr, "--- test_target_out_ids_capacity ---\n");
    llama_hparams hparams{};
    llama_cparams cparams{};

    const uint32_t capacity_outputs = 4;
    llm_graph_input_out_ids inp_out_ids(hparams, cparams, capacity_outputs, /*capacity_mode=*/true);

    struct ggml_init_params gparams = { 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(gparams);
    CHECK(ctx != nullptr);

    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, capacity_outputs);
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), ggml_nbytes(tensor));
    CHECK(buf != nullptr);
    ggml_backend_buffer_init_tensor(buf, tensor);
    tensor->data = ggml_backend_buffer_get_base(buf);
    tensor->buffer = buf;
    inp_out_ids.out_ids = tensor;

    // Case A: active = 4 (full capacity)
    {
        llama_ubatch ubatch{};
        ubatch.n_tokens = 4;
        ubatch.output = nullptr; // all tokens output
        inp_out_ids.set_input(&ubatch);

        int32_t * data = (int32_t *) tensor->data;
        for (int i = 0; i < 4; ++i) {
            CHECK(data[i] == i);
        }
    }

    // Case B: active = 2 with garbage in tail -> verify active prefix [0, 1] and tail [2, 4) zeroed
    {
        int32_t * data = (int32_t *) tensor->data;
        data[0] = 99; data[1] = 88; data[2] = 77; data[3] = 66;

        llama_ubatch ubatch{};
        ubatch.n_tokens = 2;
        ubatch.output = nullptr;
        inp_out_ids.set_input(&ubatch);

        CHECK(data[0] == 0);
        CHECK(data[1] == 1);
        CHECK(data[2] == 0);
        CHECK(data[3] == 0);
    }

    // Case C: active = 1 (degenerate single-token decode) -> verify data[0] == 0 and [1, 4) zeroed
    {
        int32_t * data = (int32_t *) tensor->data;
        data[0] = 42; data[1] = 42; data[2] = 42; data[3] = 42;

        llama_ubatch ubatch{};
        ubatch.n_tokens = 1;
        ubatch.output = nullptr;
        inp_out_ids.set_input(&ubatch);

        CHECK(data[0] == 0);
        CHECK(data[1] == 0);
        CHECK(data[2] == 0);
        CHECK(data[3] == 0);
    }

    // Case D: overflow guard: active = 5 > capacity (4) -> must throw runtime_error
    {
        bool threw = false;
        try {
            llama_ubatch ubatch{};
            ubatch.n_tokens = 5;
            ubatch.output = nullptr;
            inp_out_ids.set_input(&ubatch);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
    }

    // can_reuse check: matches when capacity_outputs matches
    {
        llm_graph_params gp{};
        gp.predefined_enabled = true;
        gp.predefined_capacity_outputs = 4;
        CHECK(inp_out_ids.can_reuse(gp));

        gp.predefined_capacity_outputs = 8;
        CHECK(!inp_out_ids.can_reuse(gp));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    fprintf(stderr, "  PASSED: test_target_out_ids_capacity\n");
}

static void test_target_input_shell_bounds() {
    fprintf(stderr, "--- test_target_input_shell_bounds ---\n");
    struct ggml_init_params gparams = { 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(gparams);
    CHECK(ctx != nullptr);

    // 1. llm_graph_input_embd
    {
        llm_graph_input_embd inp(2560);
        ggml_tensor * tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), ggml_nbytes(tokens));
        ggml_backend_buffer_init_tensor(buf, tokens);
        tokens->data = ggml_backend_buffer_get_base(buf);
        tokens->buffer = buf;
        inp.tokens = tokens;

        llama_token tok_data[5] = {1, 2, 3, 4, 5};
        llama_ubatch ubatch{};
        ubatch.n_tokens = 3;
        ubatch.token = tok_data;
        inp.set_input(&ubatch); // active = 3 <= 4, succeeds

        // active = 5 > 4, must throw
        bool threw = false;
        try {
            llama_ubatch ubatch_overflow{};
            ubatch_overflow.n_tokens = 5;
            ubatch_overflow.token = tok_data;
            inp.set_input(&ubatch_overflow);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
        ggml_backend_buffer_free(buf);
    }

    // 2. llm_graph_input_pos
    {
        llm_graph_input_pos inp(1);
        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), ggml_nbytes(pos));
        ggml_backend_buffer_init_tensor(buf, pos);
        pos->data = ggml_backend_buffer_get_base(buf);
        pos->buffer = buf;
        inp.pos = pos;

        llama_pos pos_data[5] = {10, 11, 12, 13, 14};
        llama_ubatch ubatch{};
        ubatch.n_tokens = 2;
        ubatch.pos = pos_data;
        inp.set_input(&ubatch); // active = 2 <= 4, succeeds

        bool threw = false;
        try {
            llama_ubatch ubatch_overflow{};
            ubatch_overflow.n_tokens = 5;
            ubatch_overflow.pos = pos_data;
            inp.set_input(&ubatch_overflow);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
        ggml_backend_buffer_free(buf);
    }

    ggml_free(ctx);
    fprintf(stderr, "  PASSED: test_target_input_shell_bounds\n");
}

static void test_target_reuse_key_dynamic() {
    fprintf(stderr, "--- test_target_reuse_key_dynamic ---\n");
    llm_graph_params p1{};
    p1.gtype = LLM_GRAPH_TYPE_DECODER;
    p1.predefined_enabled = true;
    p1.predefined_target_enabled = true;
    p1.predefined_capacity_rows = 4;
    p1.predefined_capacity_outputs = 4;
    p1.predefined_frame.version = GGML_PREDEFINED_ABI_VERSION;
    p1.predefined_frame.phase = GGML_PREDEFINED_TARGET;
    p1.predefined_frame.active_tokens = 4;
    p1.predefined_frame.active_outputs = 4;
    p1.ubatch.n_tokens = 4;
    p1.ubatch.n_seqs = 1;
    p1.ubatch.n_seqs_unq = 1;
    p1.ubatch.n_seq_tokens = 4;
    p1.n_outputs = 4;

    llm_graph_params p2 = p1;
    // Active token count changes to 2 (e.g. 2 candidate verification tokens)
    p2.ubatch.n_tokens = 2;
    p2.ubatch.n_seq_tokens = 2;
    p2.n_outputs = 2;
    p2.predefined_frame.active_tokens = 2;
    p2.predefined_frame.active_outputs = 2;

    // Both have capacity == 4, phase == TARGET, predefined_target_enabled == true
    CHECK(p1.allow_reuse(p2)); // Dynamic reuse succeeds!
    CHECK(p2.allow_reuse(p1));

    // Active token count changes to 1 (single token decode)
    llm_graph_params p3 = p1;
    p3.ubatch.n_tokens = 1;
    p3.ubatch.n_seq_tokens = 1;
    p3.n_outputs = 1;
    p3.predefined_frame.active_tokens = 1;
    p3.predefined_frame.active_outputs = 1;
    CHECK(p1.allow_reuse(p3));
    CHECK(p2.allow_reuse(p3));

    // Invariant: capacity_rows mismatch fails reuse
    llm_graph_params p_bad_cap = p2;
    p_bad_cap.predefined_capacity_rows = 8;
    p_bad_cap.predefined_capacity_outputs = 8;
    CHECK(!p1.allow_reuse(p_bad_cap));

    // Invariant: capacity_outputs mismatch fails reuse
    llm_graph_params p_bad_out_cap = p2;
    p_bad_out_cap.predefined_capacity_outputs = 2;
    CHECK(!p1.allow_reuse(p_bad_out_cap));

    // Invariant: phase mismatch (e.g. PREFILL) fails reuse
    llm_graph_params p_prefill = p2;
    p_prefill.predefined_frame.phase = GGML_PREDEFINED_PREFILL;
    CHECK(!p1.allow_reuse(p_prefill));

    // Invariant: active > capacity fails reuse
    llm_graph_params p_overflow = p2;
    p_overflow.ubatch.n_tokens = 5;
    p_overflow.predefined_frame.active_tokens = 5;
    CHECK(!p1.allow_reuse(p_overflow));

    // Invariant: gtype mismatch fails reuse
    llm_graph_params p_mtp = p2;
    p_mtp.gtype = LLM_GRAPH_TYPE_DECODER_MTP;
    CHECK(!p1.allow_reuse(p_mtp));

    fprintf(stderr, "  PASSED: test_target_reuse_key_dynamic\n");
}

static void test_target_switch_off_status_quo() {
    fprintf(stderr, "--- test_target_switch_off_status_quo ---\n");
    llm_graph_params p1{};
    p1.gtype = LLM_GRAPH_TYPE_DECODER;
    p1.predefined_enabled = true;
    p1.predefined_target_enabled = false; // Switch OFF (default)
    p1.predefined_capacity_rows = 4;
    p1.predefined_capacity_outputs = 4;
    p1.predefined_frame.version = GGML_PREDEFINED_ABI_VERSION;
    p1.predefined_frame.phase = GGML_PREDEFINED_TARGET;
    p1.predefined_frame.active_tokens = 4;
    p1.predefined_frame.active_outputs = 4;
    p1.ubatch.n_tokens = 4;
    p1.ubatch.n_seqs = 1;
    p1.ubatch.n_seqs_unq = 1;
    p1.ubatch.n_seq_tokens = 4;
    p1.n_outputs = 4;

    llm_graph_params p2 = p1;
    p2.ubatch.n_tokens = 2;
    p2.ubatch.n_seq_tokens = 2;
    p2.n_outputs = 2;
    p2.predefined_frame.active_tokens = 2;
    p2.predefined_frame.active_outputs = 2;

    // When switch is OFF, n_tokens mismatch (4 vs 2) MUST reject reuse!
    CHECK(!p1.allow_reuse(p2));

    // Only exact n_tokens == 4 allows reuse under switch OFF
    llm_graph_params p3 = p1;
    CHECK(p1.allow_reuse(p3));

    fprintf(stderr, "  PASSED: test_target_switch_off_status_quo\n");
}

int main() {
    try {
        test_target_frame_contract();
        test_target_out_ids_capacity();
        test_target_input_shell_bounds();
        test_target_reuse_key_dynamic();
        test_target_switch_off_status_quo();
        fprintf(stderr, "\nALL TARGET CAPACITY CONTRACT TESTS PASSED (100%% CPU verified)\n");
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
}
