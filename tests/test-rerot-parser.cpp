#include "server-rerot.h"

#include "../src/llama-grammar.h"
#include "../src/unicode.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        ++g_failures; \
    } \
} while (0)

static void test_plain_public_text() {
    server_rerot_planner_parser parser;
    const auto step = parser.consume("Analyze the dependency graph first.");
    CHECK(step.write_visibility == llama_rerot_visibility::public_live);
    CHECK(!step.release_previous_pending);
    CHECK(!step.record_opened);
    CHECK(!parser.complete());
}

static void test_split_open_and_close() {
    server_rerot_planner_parser parser;

    auto step = parser.consume("analysis <");
    CHECK(step.write_visibility == llama_rerot_visibility::pending_record);
    CHECK(parser.state() == server_rerot_parser_state::opening_candidate);

    step = parser.consume("o");
    CHECK(step.write_visibility == llama_rerot_visibility::pending_record);
    CHECK(!step.record_opened);

    step = parser.consume("l><li> Alpha </li><li>Beta");
    CHECK(step.write_visibility == llama_rerot_visibility::pending_record);
    CHECK(step.record_opened);
    CHECK(!step.record_closed);

    step = parser.consume(" branch</li></o");
    CHECK(step.write_visibility == llama_rerot_visibility::pending_record);
    CHECK(!step.record_closed);

    step = parser.consume("l>\n");
    CHECK(step.record_closed);
    CHECK(!step.malformed);
    CHECK(step.items == std::vector<std::string>({"Alpha", "Beta branch"}));
    CHECK(parser.complete());
}

static void test_byte_by_byte_record() {
    // Tokenization boundaries may fall anywhere, including inside every tag.
    server_rerot_planner_parser parser;
    const std::string bytes = "pre <ol><li>A</li><li>B</li></ol>";
    server_rerot_parser_step step;
    for (size_t i = 0; i < bytes.size(); ++i) {
        step = parser.consume(bytes.substr(i, 1));
        CHECK(!step.malformed);
        if (i + 1 < bytes.size()) {
            CHECK(!step.record_closed);
        }
    }
    CHECK(step.record_closed);
    CHECK(step.items == std::vector<std::string>({"A", "B"}));
    CHECK(parser.complete());
}

static void test_false_prefix_release() {
    server_rerot_planner_parser parser;

    auto step = parser.consume("text <");
    CHECK(step.write_visibility == llama_rerot_visibility::pending_record);
    CHECK(!step.release_previous_pending);

    step = parser.consume("x is ordinary");
    CHECK(step.release_previous_pending);
    CHECK(step.write_visibility == llama_rerot_visibility::public_live);
    CHECK(parser.state() == server_rerot_parser_state::public_text);
}

static void test_false_prefix_then_new_record_same_token() {
    server_rerot_planner_parser parser;

    auto step = parser.consume("<o");
    CHECK(step.write_visibility == llama_rerot_visibility::pending_record);

    step = parser.consume("ops <ol><li>A</li></ol>");
    CHECK(step.release_previous_pending);
    CHECK(step.write_visibility == llama_rerot_visibility::pending_record);
    CHECK(step.record_opened);
    CHECK(step.record_closed);
    CHECK(step.items == std::vector<std::string>({"A"}));
}

static void test_complete_record_one_token() {
    server_rerot_planner_parser parser;
    const auto step = parser.consume("prefix <ol>\n<li>One lane</li>\n</ol>\t");
    CHECK(step.write_visibility == llama_rerot_visibility::pending_record);
    CHECK(step.record_opened);
    CHECK(step.record_closed);
    CHECK(step.items == std::vector<std::string>({"One lane"}));
}

static void test_non_candidate_angle_text_stays_public() {
    server_rerot_planner_parser parser;
    const auto step = parser.consume("x <other> y <ol nope");
    CHECK(step.write_visibility == llama_rerot_visibility::public_live);
    CHECK(parser.state() == server_rerot_parser_state::public_text);
}

static void test_malformed_records() {
    {
        server_rerot_planner_parser parser;
        const auto step = parser.consume("<ol><li>A</li><ol><li>B</li></ol></ol>");
        CHECK(step.malformed);
        CHECK(parser.failed());
    }
    {
        server_rerot_planner_parser parser;
        const auto step = parser.consume("<ol><li>   </li></ol>");
        CHECK(step.malformed);
        CHECK(parser.failed());
    }
    {
        server_rerot_planner_parser parser;
        const auto step = parser.consume("<ol><li>A</li></ol>not-planner-whitespace");
        CHECK(step.malformed);
        CHECK(parser.failed());
    }
    {
        server_rerot_planner_parser parser;
        const auto step = parser.consume("<ol>free text<li>A</li></ol>");
        CHECK(step.malformed);
        CHECK(parser.failed());
    }
}

static void test_bytes_after_complete_are_malformed() {
    // The planner grammar disarms at </ol>: emitting more planner bytes is a
    // protocol violation, never a second fork record.
    server_rerot_planner_parser parser;
    const auto closed = parser.consume("<ol><li>A</li></ol>");
    CHECK(closed.record_closed);
    CHECK(parser.complete());
    const auto extra = parser.consume("trailing");
    CHECK(extra.malformed);
    CHECK(!extra.error.empty());
}

static bool grammar_accepts(
        const std::string & grammar_text,
        const std::string & input) {
    llama_grammar * grammar = llama_grammar_init_impl(
        nullptr, grammar_text.c_str(), "root", false, nullptr, 0, nullptr, 0);
    CHECK(grammar != nullptr);
    if (!grammar) {
        return false;
    }

    auto & stacks = llama_grammar_get_stacks(grammar);
    for (const auto cpt : unicode_cpts_from_utf8(input)) {
        llama_grammar_accept(grammar, cpt);
        if (stacks.empty()) {
            llama_grammar_free_impl(grammar);
            return false;
        }
    }

    const bool complete = std::any_of(
        stacks.begin(), stacks.end(), [](const auto & stack) { return stack.empty(); });
    llama_grammar_free_impl(grammar);
    return complete;
}

static bool planner_grammar_accepts(const std::string & input) {
    return grammar_accepts(
        std::string(server_rerot_planner_grammar()), input);
}

static void test_planner_grammar_rejects_empty_items() {
    CHECK(planner_grammar_accepts("目标</li>\n</ol>"));
    CHECK(planner_grammar_accepts("\n  目标  \n</li>\n<li>另一个目标</li>\n</ol>"));
    CHECK(!planner_grammar_accepts("\n</li>\n</ol>"));
    CHECK(!planner_grammar_accepts(" \t </li>\n</ol>"));
    CHECK(!planner_grammar_accepts("目标</li>\n<li>\n</li>\n</ol>"));
}

static constexpr const char * TEST_CHILD_CLOSE = "</AbCdEfG0>";

static void test_child_grammar_requires_exact_random_close() {
    const std::string grammar = server_rerot_child_grammar(TEST_CHILD_CLOSE);
    CHECK(!grammar.empty());
    CHECK(grammar_accepts(
        grammar, "原生标签只是正文</think>\n</AbCdEfG0>"));
    CHECK(grammar_accepts(grammar, "事实清单。</AbCdEfG0>"));
    CHECK(grammar_accepts(grammar, "事实清单。包含更多标点符号，任意写！\n</AbCdEfG0>"));
    CHECK(grammar_accepts(grammar, "one sentence.</AbCdEfG0>"));
    CHECK(grammar_accepts(grammar, "one sentence. with more words.\n</AbCdEfG0>"));
    CHECK(grammar_accepts(grammar, "推导正文\n</AbCdEfG0>"));
    CHECK(grammar_accepts(grammar, "<ol><li>递归列表</li></ol>\n</AbCdEfG0>"));
    CHECK(grammar_accepts(grammar, "<ol><li>递归列表</li></ol></AbCdEfG0>"));
    CHECK(!grammar_accepts(grammar, "</AbCdEfG0>"));
    CHECK(!grammar_accepts(grammar, " \n\n</AbCdEfG0>"));
    CHECK(!grammar_accepts(grammar, "推导正文"));
    CHECK(!grammar_accepts(grammar, "推导正文</AbCdEfG1>"));
    CHECK(grammar_accepts(grammar, "推导第一行\n第二行继续推导\n</AbCdEfG0>"));
    CHECK(grammar_accepts(grammar, "第一段。\n\n第二段。\r\n```cpp\nint x = 1;\n```\n</AbCdEfG0>"));
    CHECK(grammar_accepts(grammar, "<ol>\n<li>递归任务一</li>\n<li>递归任务二</li>\n</ol>\n</AbCdEfG0>"));
    CHECK(!grammar_accepts(grammar, "第一行\n第二行\n"));
    CHECK(!grammar_accepts(grammar, "第一行\n第二行</AbCdEfG1>"));
    CHECK(!grammar_accepts(grammar, "正文</AbCdEfG0>闭合后不能续写"));
    std::string long_body = "长推导";
    for (int i = 0; i < 256; ++i) {
        long_body += "\n继续核对另一项事实，不以行数或长度代替任务完成。";
    }
    CHECK(grammar_accepts(grammar, long_body + TEST_CHILD_CLOSE));
    CHECK(server_rerot_child_grammar("</think>").empty());
}

static void test_private_marker_split() {
    server_rerot_marker_parser parser(TEST_CHILD_CLOSE);

    auto step = parser.consume("conclusion </AbCd");
    CHECK(step.write_visibility == llama_rerot_visibility::pending_record);
    CHECK(step.public_prefix_bytes == std::string("conclusion ").size());
    CHECK(!step.marker_closed);
    CHECK(parser.state() == server_rerot_marker_state::marker_candidate);

    step = parser.consume("EfG0>\n");
    CHECK(step.write_visibility == llama_rerot_visibility::pending_record);
    CHECK(step.public_prefix_bytes == 0);
    CHECK(step.marker_closed);
    CHECK(!step.malformed);
    CHECK(parser.complete());
}

static void test_private_marker_false_prefix() {
    server_rerot_marker_parser parser(TEST_CHILD_CLOSE);

    auto step = parser.consume("ordinary <");
    CHECK(step.write_visibility == llama_rerot_visibility::pending_record);

    step = parser.consume("x remains public");
    CHECK(step.release_previous_pending);
    CHECK(step.write_visibility == llama_rerot_visibility::public_live);
    CHECK(!step.marker_closed);
}

static void test_private_marker_closes_with_trailing_body() {
    server_rerot_marker_parser parser(TEST_CHILD_CLOSE);
    const auto step = parser.consume("</AbCdEfG0>answer in same tokenizer token");
    CHECK(step.marker_closed);
    CHECK(!step.malformed);
    CHECK(parser.complete());
}

static void test_private_marker_after_complete_is_ignored() {
    server_rerot_marker_parser parser(TEST_CHILD_CLOSE);
    const auto closed = parser.consume("done </AbCdEfG0>");
    CHECK(closed.marker_closed);
    CHECK(parser.complete());
    const auto extra = parser.consume("more");
    CHECK(extra.marker_closed);
    CHECK(!extra.malformed);
}

static void test_unarmed_root_does_not_parse_native_thought_tags() {
    server_rerot_marker_parser parser;
    const auto step = parser.consume("</think>");
    CHECK(step.write_visibility == llama_rerot_visibility::public_live);
    CHECK(!step.marker_closed);
    CHECK(!step.malformed);
    CHECK(!parser.complete());
}

static void test_private_marker_snapshot_preserves_id_and_split() {
    server_rerot_marker_parser parser(TEST_CHILD_CLOSE);
    CHECK(!parser.consume("</AbCd").malformed);

    const auto snapshot = parser.snapshot();
    server_rerot_marker_parser restored;
    std::string error;
    CHECK(restored.restore(snapshot, &error));
    CHECK(restored.marker() == TEST_CHILD_CLOSE);
    CHECK(restored.state() == server_rerot_marker_state::marker_candidate);
    CHECK(restored.consume("EfG0>").marker_closed);

    auto obsolete = snapshot;
    obsolete.marker = "</think>";
    CHECK(!restored.restore(obsolete, &error));
    CHECK(error.find("obsolete") != std::string::npos);
}

static void test_rejects_obsolete_or_malformed_delimiters() {
    for (const std::string marker : {
            "</think>", "</blockquote>", "</short>", "</AbCd-Ef0>",
            "</AbCdEfG01>", "<AbCdEfG0>"}) {
        server_rerot_marker_parser parser(marker);
        CHECK(parser.failed());
        CHECK(!parser.error().empty());
    }
}

static void test_routing_decision_parser() {
    // §02.3 single-task passthrough: one line without dependency edges maps
    // to the ordinary single-stream continuation. Trailing blank lines are
    // optional for the parser (the server additionally requires the grammar's
    // "\n\n" terminator before deciding — see rerot_try_finish_probe).
    const auto dec_simple = server_rerot_parse_routing_decision("A: do something\n\n");
    CHECK(dec_simple.is_simple());
    CHECK(!dec_simple.is_dag());
    CHECK(dec_simple.error.empty());
    CHECK(dec_simple.questions.size() == 1);
    CHECK(dec_simple.questions[0].id == "A");
    CHECK(dec_simple.questions[0].intent == "do something");
    CHECK(dec_simple.questions[0].plan_rank == 0);
    CHECK(server_rerot_parse_routing_decision("A: do something").is_simple());
    CHECK(server_rerot_parse_routing_decision("A: do something\n").is_simple());

    // One node stays dag only when the caller explicitly forces it
    // (Stage-5 single-worker DAG tests).
    const auto dec_forced = server_rerot_parse_routing_decision(
        "A: Fact A\n\n", /*force_single_node_dag=*/true);
    CHECK(dec_forced.is_dag());
    CHECK(!dec_forced.is_simple());

    // Two independent nodes are already a DAG (parallel decomposition).
    const auto dec_pair = server_rerot_parse_routing_decision("A: task1\nB: task2\n\n");
    CHECK(dec_pair.is_dag());
    CHECK(dec_pair.error.empty());
    CHECK(dec_pair.questions.size() == 2);
    CHECK(dec_pair.dependencies.empty());
    CHECK(dec_pair.questions[0].plan_rank == 0);
    CHECK(dec_pair.questions[1].plan_rank == 1);

    // Classic dependency chain A <- ... with rank in line order.
    const auto dec_chain = server_rerot_parse_routing_decision(
        "A: step one\nB <- A: step two\nC <- B: step three\n\n");
    CHECK(dec_chain.is_dag());
    CHECK(dec_chain.questions.size() == 3);
    CHECK(dec_chain.dependencies.size() == 2);
    CHECK(dec_chain.dependencies[0].from_id == "A");
    CHECK(dec_chain.dependencies[0].to_id == "B");
    CHECK(dec_chain.dependencies[1].from_id == "B");
    CHECK(dec_chain.dependencies[1].to_id == "C");

    // Diamond join with multi-dependency syntax.
    const auto dec_diamond = server_rerot_parse_routing_decision(
        "A: base\nB <- A: left\nC <- A: right\nD <- B, C: merge\n\n");
    CHECK(dec_diamond.is_dag());
    CHECK(dec_diamond.questions.size() == 4);
    CHECK(dec_diamond.dependencies.size() == 4);

    // Forward reference: ids are validated against the complete set, so a
    // line may depend on an id declared later.
    const auto dec_forward = server_rerot_parse_routing_decision(
        "B <- A: second\nA: first\n\n");
    CHECK(dec_forward.is_dag());
    CHECK(dec_forward.dependencies.size() == 1);
    CHECK(dec_forward.questions[0].id == "B");

    // Complex ids: underscores and hyphens are legal (grammar id class).
    const auto dec_ids = server_rerot_parse_routing_decision(
        "step_1: do step one\nsub-task-2 <- step_1: do step two\n\n");
    CHECK(dec_ids.is_dag());
    CHECK(dec_ids.questions.size() == 2);
    CHECK(dec_ids.questions[0].id == "step_1");
    CHECK(dec_ids.questions[1].id == "sub-task-2");

    // An intent may itself contain ':' and '<-' — only the head is split.
    const auto dec_intent = server_rerot_parse_routing_decision(
        "algebra: derive x <- y: expand (a+b)^2\n\n");
    CHECK(dec_intent.is_simple());
    CHECK(dec_intent.questions[0].intent == "derive x <- y: expand (a+b)^2");
}

static void test_routing_decision_rejections() {
    // Control-plane fail-closed matrix (RERoT.md 15.1): every rejection the
    // peer parser must enforce. Invalid inputs must be explicitly invalid —
    // nonempty error, neither dag nor simple — since a silent fallback to
    // simple is forbidden by contract.
    const char * invalid_cases[] = {
        // missing ':' separator
        "A task without colon\n\n",
        // empty intent
        "A: \n\n",
        // self-loop
        "A <- A: self loop\n\n",
        // duplicate id
        "A: task1\nA: task2\n\n",
        // unknown dependency endpoint (Z never declared)
        "B <- Z: unknown dep\n\n",
        // two-node cycle
        "A <- B: first\nB <- A: second\n\n",
        // reserved id "0" (main synthesis)
        "0: reserved task\n\n",
        // whitespace-only id
        "   : Fact A\n\n",
        // interior blank line (the terminator may only end the plan)
        "A: one\n\nB: two\n\n",
        // duplicate dependency edge within one head
        "A: base\nC <- A, A: join\n\n",
        // id outside the grammar id class
        "A B: spaced id\n\n",
        // empty input / terminator-only input
        "",
        "\n\n",
    };
    for (const char * text : invalid_cases) {
        const auto decision = server_rerot_parse_routing_decision(text);
        CHECK(!decision.is_dag());
        CHECK(!decision.is_simple());
        CHECK(!decision.error.empty());
    }
}

static void test_native_source_end_marker() {
    // Tests standard XML think marker
    server_rerot_marker_parser parser("</think>", true);
    CHECK(!parser.failed());
    auto step = parser.consume("done ");
    CHECK(!step.marker_closed);
    CHECK(step.write_visibility == llama_rerot_visibility::public_live);
    step = parser.consume("</think>");
    CHECK(step.marker_closed);
    CHECK(parser.complete());

    const auto snapshot = parser.snapshot();
    CHECK(snapshot.native_end);
    CHECK(snapshot.marker == "</think>");
    server_rerot_marker_parser restored;
    std::string error;
    CHECK(restored.restore(snapshot, &error));
    CHECK(restored.marker() == "</think>");
    CHECK(restored.complete());

    // Non-XML template markers (e.g. [/THINK], <|im_end|>) must also be valid
    server_rerot_marker_parser alt_parser("[/THINK]", true);
    CHECK(!alt_parser.failed());
    auto alt_step = alt_parser.consume("answer [/THINK]");
    CHECK(alt_step.marker_closed);
    CHECK(alt_parser.complete());
}

static void test_multi_template_source_end_boundary_certification() {
    // Stage 2 (§12.3 item 9 & AGENTS.md §04.7.1) Verification:
    // Certify lossless token-level / byte-level reasoning-end boundary handling
    // across major model chat templates:
    // 1. Standard XML-style (Qwen, DeepSeek-R1, Ornith): "</think>"
    // 2. Command-R / Peg-native bracketed: "[/THINK]"
    // 3. ChatML / Specialized special-token closers: "<|im_end|>", "<|close|>think<|sep|>", "<|END_THINKING|>", "</mm:think>"
    //
    // For each template, assert:
    // a) Precise public prefix boundary preservation when token contains trailing body before marker
    // b) Multi-token split candidate holding in PENDING until final closing token
    // c) False candidate release without dropping ordinary text
    // d) Correct snapshot serialization and restoration
    const std::vector<std::string> target_markers = {
        "</think>",
        "[/THINK]",
        "<|im_end|>",
        "<|close|>think<|sep|>",
        "<|END_THINKING|>",
        "</mm:think>",
        "<|channel|>"
    };

    for (const auto & marker : target_markers) {
        server_rerot_marker_parser parser(marker, true);
        CHECK(!parser.failed());
        CHECK(parser.marker() == marker);

        // Case 1: Pure body bytes remain public_live
        auto step = parser.consume("Reasoning step 1: 42 * 2 = 84. ");
        CHECK(!step.marker_closed);
        CHECK(!step.malformed);
        CHECK(step.write_visibility == llama_rerot_visibility::public_live);
        CHECK(step.public_prefix_bytes == 0);

        // Case 2: Token containing body and start of marker (partial split)
        // e.g. "answer </thi" for "</think>"
        const std::string split_head = marker.substr(0, marker.size() / 2);
        const std::string split_tail = marker.substr(marker.size() / 2);
        const std::string prefix_text = "Conclusion established. ";

        step = parser.consume(prefix_text + split_head);
        CHECK(!step.marker_closed);
        CHECK(!step.malformed);
        CHECK(step.write_visibility == llama_rerot_visibility::pending_record);
        CHECK(step.public_prefix_bytes == prefix_text.size());
        CHECK(parser.state() == server_rerot_marker_state::marker_candidate);

        // Case 3: Next token completes the marker
        step = parser.consume(split_tail);
        CHECK(step.marker_closed);
        CHECK(!step.malformed);
        CHECK(step.write_visibility == llama_rerot_visibility::pending_record);
        CHECK(parser.complete());

        // Case 4: Snapshot / Restore fidelity
        const auto snap = parser.snapshot();
        CHECK(snap.native_end);
        CHECK(snap.marker == marker);
        server_rerot_marker_parser restored;
        std::string restore_err;
        CHECK(restored.restore(snap, &restore_err));
        CHECK(restored.marker() == marker);
        CHECK(restored.complete());

        // Case 5: False candidate test (prefix matches marker start then diverges)
        server_rerot_marker_parser false_test(marker, true);
        step = false_test.consume("text " + split_head);
        CHECK(step.write_visibility == llama_rerot_visibility::pending_record);
        CHECK(false_test.state() == server_rerot_marker_state::marker_candidate);
        // Diverges:
        step = false_test.consume("xyz false alarm");
        CHECK(step.release_previous_pending);
        CHECK(step.write_visibility == llama_rerot_visibility::public_live);
        CHECK(!step.marker_closed);
        CHECK(false_test.state() == server_rerot_marker_state::public_text);
    }
}

static void test_format_fixed_entry() {
    const std::string frame = server_rerot_format_fixed_entry(
        "A", "compute A", false, "</think>", "<think>");
    CHECK(frame.find("</think>") == 0);
    CHECK(frame.find("lane:A") != std::string::npos);
    CHECK(frame.find("intent:compute A") != std::string::npos);
    CHECK(frame.find("<think>") != std::string::npos);
    CHECK(frame.rfind("<think>") > frame.find("</think>"));
    const std::string synth = server_rerot_format_fixed_entry(
        "0", "0.synthesize", true, "</think>", "<think>");
    CHECK(synth.find("synthesis:0") != std::string::npos);
    CHECK(synth.find("lane:") == std::string::npos);

    // Intent line breaks and raw think markers must be sanitized to disarm forgery
    const std::string sanitized = server_rerot_format_fixed_entry(
        "B", "line 1\nline 2 </think> evil <think>", false, "</think>", "<think>");
    CHECK(sanitized.find("line 1 line 2") != std::string::npos);
    CHECK(sanitized.find("</think> evil") == std::string::npos);
    CHECK(sanitized.find("[end] evil [start]") != std::string::npos);
}

static void test_format_plan_prefix() {
    // PUBLIC 0.plan body: id+intent in plan_rank order. It must not close
    // reasoning and must not emit worker lane: frames.
    const auto decision = server_rerot_parse_routing_decision(
        "A: Fact A\nB: Fact B\nC <- A: Fact C <think>\n\n");
    CHECK(decision.is_dag());
    const std::string think_end = "</think>";
    const std::string think_start = "<think>";
    const std::string prefix = server_rerot_format_plan_prefix(decision, think_start);
    CHECK(prefix.find("plan:") != std::string::npos);
    CHECK(prefix.find("Fact A") != std::string::npos);
    CHECK(prefix.find("Fact B") != std::string::npos);
    CHECK(prefix.find("Fact C [start]") != std::string::npos);
    CHECK(prefix.find("A:") != std::string::npos);
    CHECK(prefix.find(think_end) == std::string::npos);
    CHECK(prefix.find("</think>") == std::string::npos);
    CHECK(prefix.find("lane:") == std::string::npos);
    CHECK(prefix.find("<think>") == std::string::npos);
}

static void test_dag_protocol_does_not_require_source_end_grammar() {
    // The source_end grammar helper may still exist, but DAG workers close on
    // native source-end detection. Installing that grammar is not required.
    server_rerot_marker_parser native("</think>", true);
    auto step = native.consume("task body ");
    CHECK(!step.marker_closed);
    step = native.consume("</think>");
    CHECK(step.marker_closed);
    CHECK(native.complete());

    const std::string optional_grammar = server_rerot_source_end_grammar("</think>");
    (void) optional_grammar;

    // Single-worker DAG kept explicitly (§02.3 default maps one node to
    // simple).
    const auto decision = server_rerot_parse_routing_decision(
        "A: Fact A\n\n", /*force_single_node_dag=*/true);
    CHECK(decision.is_dag());
    CHECK(decision.error.empty());
}

static void test_routing_dsl_grammar() {
    // The probe prompt must end on a fresh line so the first sampled token
    // opens the first task line directly (no glued preamble).
    const std::string_view probe_prompt = server_rerot_routing_probe_prompt();
    CHECK(!probe_prompt.empty());
    CHECK(probe_prompt.back() == '\n');

    const std::string grammar = server_rerot_routing_grammar();
    CHECK(!grammar.empty());
    // Legal DSL outputs.
    CHECK(grammar_accepts(
        grammar, "A: do something\n\n"));
    CHECK(grammar_accepts(
        grammar, "A: task1\nB: task2\n\n"));
    CHECK(grammar_accepts(
        grammar, "A: one\nB <- A: two\nC <- B: three\n\n"));
    CHECK(grammar_accepts(
        grammar, "A: base\nB <- A: left\nC <- A: right\nD <- B, C: merge\n\n"));
    CHECK(grammar_accepts(
        grammar, "step_1: do it\nsub-task-2 <- step_1: then it\n\n"));
    // No preamble: the first character must be an id character.
    CHECK(!grammar_accepts(grammar, "Sure, here is a plan:\nA: x\n\n"));
    CHECK(!grammar_accepts(grammar, " A: leading space\n\n"));
    // Malformed lines and missing/early terminator.
    CHECK(!grammar_accepts(grammar, "A task without colon\n\n"));
    CHECK(!grammar_accepts(grammar, "A: \n\n"));
    CHECK(!grammar_accepts(grammar, "A: one\n\nB: two\n\n"));
    CHECK(!grammar_accepts(grammar, "A: one"));
    CHECK(!grammar_accepts(grammar, "A: one\n"));
    CHECK(!grammar_accepts(grammar, ""));
    // Unknown ids/self-loops/cycles are semantically rejected by the parser,
    // not the grammar — the grammar only guarantees well-formed lines.
    CHECK(grammar_accepts(grammar, "A <- A: self loop\n\n"));
    // ': ' needs the literal space; intent may not contain CR.
    CHECK(!grammar_accepts(grammar, "A:Fact missing space\n\n"));
    CHECK(!grammar_accepts(grammar, "A: carriage\rreturn\n\n"));
}

int main() {
    std::fprintf(stderr, "=== RERoT Planner Parser Tests ===\n");
    test_routing_decision_parser();
    test_routing_decision_rejections();
    test_native_source_end_marker();
    test_multi_template_source_end_boundary_certification();
    test_format_fixed_entry();
    test_format_plan_prefix();
    test_dag_protocol_does_not_require_source_end_grammar();
    test_routing_dsl_grammar();
    test_plain_public_text();
    test_split_open_and_close();
    test_byte_by_byte_record();
    test_false_prefix_release();
    test_false_prefix_then_new_record_same_token();
    test_complete_record_one_token();
    test_non_candidate_angle_text_stays_public();
    test_malformed_records();
    test_bytes_after_complete_are_malformed();
    test_planner_grammar_rejects_empty_items();
    test_child_grammar_requires_exact_random_close();
    test_private_marker_split();
    test_private_marker_false_prefix();
    test_private_marker_closes_with_trailing_body();
    test_private_marker_after_complete_is_ignored();
    test_unarmed_root_does_not_parse_native_thought_tags();
    test_private_marker_snapshot_preserves_id_and_split();
    test_rejects_obsolete_or_malformed_delimiters();
    std::fprintf(stderr, "=== Results: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
