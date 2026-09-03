#include "server-rerot.h"

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

static void test_private_marker_split() {
    server_rerot_marker_parser parser;

    auto step = parser.consume("conclusion </thi");
    CHECK(step.write_visibility == llama_rerot_visibility::pending_record);
    CHECK(!step.marker_closed);
    CHECK(parser.state() == server_rerot_marker_state::marker_candidate);

    step = parser.consume("nk>\n");
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

static void test_private_marker_rejects_trailing_body() {
    server_rerot_marker_parser parser;
    const auto step = parser.consume("</think>answer in same tokenizer token");
    CHECK(step.malformed);
    CHECK(parser.failed());
}

int main() {
    std::fprintf(stderr, "=== RERoT Planner Parser Tests ===\n");
    test_plain_public_text();
    test_split_open_and_close();
    test_false_prefix_release();
    test_false_prefix_then_new_record_same_token();
    test_complete_record_one_token();
    test_non_candidate_angle_text_stays_public();
    test_malformed_records();
    test_private_marker_split();
    test_private_marker_false_prefix();
    test_private_marker_rejects_trailing_body();
    std::fprintf(stderr, "=== Results: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}

