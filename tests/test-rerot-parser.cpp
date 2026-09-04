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

static void test_planner_prompt_shape() {
    const std::string prompt(server_rerot_planner_prompt());
    CHECK(!prompt.empty());
    CHECK(prompt.find("ol") != std::string::npos);
    CHECK(prompt.find("li") != std::string::npos);
    CHECK(prompt.find("/ol") != std::string::npos);
    CHECK(prompt.find("请先") == std::string::npos);
    CHECK(prompt.find("禁止") == std::string::npos);
    CHECK(prompt.find("目标") != std::string::npos);
    CHECK(prompt.find("子答案") == std::string::npos);
    CHECK(prompt.find("简单问题") == std::string::npos);
    CHECK(prompt.size() < 300);
}

static bool planner_grammar_accepts(const std::string & input) {
    const std::string grammar_text(server_rerot_planner_grammar());
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

static void test_planner_grammar_rejects_empty_items() {
    CHECK(planner_grammar_accepts("目标</li>\n</ol>"));
    CHECK(planner_grammar_accepts("\n  目标  \n</li>\n<li>另一个目标</li>\n</ol>"));
    CHECK(!planner_grammar_accepts("\n</li>\n</ol>"));
    CHECK(!planner_grammar_accepts(" \t </li>\n</ol>"));
    CHECK(!planner_grammar_accepts("目标</li>\n<li>\n</li>\n</ol>"));
}

static void test_private_marker_split() {
    server_rerot_marker_parser parser;

    auto step = parser.consume("conclusion </blockquo");
    CHECK(step.write_visibility == llama_rerot_visibility::pending_record);
    CHECK(!step.marker_closed);
    CHECK(parser.state() == server_rerot_marker_state::marker_candidate);

    step = parser.consume("te>\n");
    CHECK(step.write_visibility == llama_rerot_visibility::pending_record);
    CHECK(step.marker_closed);
    CHECK(!step.malformed);
    CHECK(parser.complete());
}

static void test_private_marker_false_prefix() {
    server_rerot_marker_parser parser;

    auto step = parser.consume("ordinary <");
    CHECK(step.write_visibility == llama_rerot_visibility::pending_record);

    step = parser.consume("x remains public");
    CHECK(step.release_previous_pending);
    CHECK(step.write_visibility == llama_rerot_visibility::public_live);
    CHECK(!step.marker_closed);
}

static void test_private_marker_closes_with_trailing_body() {
    server_rerot_marker_parser parser;
    const auto step = parser.consume("</blockquote>answer in same tokenizer token");
    CHECK(step.marker_closed);
    CHECK(!step.malformed);
    CHECK(parser.complete());
}

static void test_private_marker_after_complete_is_ignored() {
    server_rerot_marker_parser parser;
    const auto closed = parser.consume("done </blockquote>");
    CHECK(closed.marker_closed);
    CHECK(parser.complete());
    const auto extra = parser.consume("more");
    CHECK(extra.marker_closed);
    CHECK(!extra.malformed);
}

static void test_control_tag_filter_across_tokens() {
    server_rerot_control_tag_filter filter;

    std::string bytes = "<";
    filter.consume(bytes);
    CHECK(bytes.empty());

    bytes = "blockquote";
    filter.consume(bytes);
    CHECK(bytes.empty());

    bytes = ">answer</block";
    filter.consume(bytes);
    CHECK(bytes == "answer");

    bytes = "quote>tail";
    filter.consume(bytes);
    CHECK(bytes == "tail");

    bytes = "one<blockquote>two</blockquote>three";
    filter.consume(bytes);
    CHECK(bytes == "onetwothree");
}

static void test_control_tag_filter_false_prefix() {
    server_rerot_control_tag_filter filter;

    std::string bytes = "prefix <";
    filter.consume(bytes);
    CHECK(bytes == "prefix ");

    bytes = "b> remains";
    filter.consume(bytes);
    CHECK(bytes == "<b> remains");

    bytes = "</block";
    filter.consume(bytes);
    CHECK(bytes.empty());
    filter.reset();

    bytes = "safe";
    filter.consume(bytes);
    CHECK(bytes == "safe");
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
    test_planner_prompt_shape();
    test_planner_grammar_rejects_empty_items();
    test_private_marker_split();
    test_private_marker_false_prefix();
    test_private_marker_closes_with_trailing_body();
    test_private_marker_after_complete_is_ignored();
    test_control_tag_filter_across_tokens();
    test_control_tag_filter_false_prefix();
    std::fprintf(stderr, "=== Results: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
