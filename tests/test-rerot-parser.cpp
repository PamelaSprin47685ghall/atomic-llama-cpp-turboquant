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
    CHECK(grammar_accepts(grammar, "one sentence.</AbCdEfG0>"));
    CHECK(grammar_accepts(grammar, "推导正文\n</AbCdEfG0>"));
    CHECK(!grammar_accepts(grammar, "</AbCdEfG0>"));
    CHECK(!grammar_accepts(grammar, " \n\n</AbCdEfG0>"));
    CHECK(!grammar_accepts(grammar, "推导正文"));
    CHECK(!grammar_accepts(grammar, "推导正文</AbCdEfG1>"));
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

int main() {
    std::fprintf(stderr, "=== RERoT Planner Parser Tests ===\n");
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
