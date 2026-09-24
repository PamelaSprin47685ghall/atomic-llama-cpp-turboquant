#include "server-rerot-todo.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(condition)                                                              \
    do {                                                                              \
        if (!(condition)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            ++g_failures;                                                             \
        }                                                                             \
    } while (0)

using namespace server_todo;

static void test_stop_offset_incremental() {
    // Exact stop rule: returns string_view::npos while incomplete else offset of FIRST byte at line start not '-'.
    std::string text = "- Task 1\n- Task 2\n";
    CHECK(stop_offset(text, 0) == std::string_view::npos);

    // Incremental stop detection across token boundaries
    size_t prev_len = text.size();
    text += "Done";
    // Now line 3 starts with 'D' (not '-'), offset is prev_len
    CHECK(stop_offset(text, prev_len) == prev_len);
    CHECK(stop_offset(text, 0) == prev_len);

    // Token-boundary split newline and '-'
    std::string split_text = "- A\n";
    size_t      len1       = split_text.size();
    CHECK(stop_offset(split_text, 0) == std::string_view::npos);
    split_text += "-";
    CHECK(stop_offset(split_text, len1) == std::string_view::npos);
    split_text += " B\n";
    CHECK(stop_offset(split_text, len1 + 1) == std::string_view::npos);

    // Split newline and non-'-'
    split_text += "X";
    CHECK(stop_offset(split_text, len1 + 1) == split_text.size() - 1);
}

static void test_discard_terminator_and_suffix() {
    // Single token holding label end + newline + sentinel + discarded later bullets
    std::string text = "- Step A\nDone with thinking\n- Step B\n- Step C\n";
    size_t      term = stop_offset(text, 0);
    CHECK(term == 9);  // offset of 'D'

    auto res = parse(text);
    CHECK(res.complete);
    CHECK(res.accepted_bytes == 9);
    CHECK(res.items.size() == 1);
    CHECK(res.items[0] == "Step A");

    // All bytes starting from terminator (including suffix of same token) dropped
    std::string single_token = "- Item 1\nTermination suffix and - Item 2\n";
    auto        res2         = parse(single_token);
    CHECK(res2.complete);
    CHECK(res2.items.size() == 1);
    CHECK(res2.items[0] == "Item 1");
}

static void test_two_chunk_boundaries() {
    const std::string wire = "- 甲任务\n- Beta\nStop\n- must not become a task\n";
    for (size_t split = 0; split <= wire.size(); ++split) {
        std::string accumulated = wire.substr(0, split);
        size_t      stop        = stop_offset(accumulated);
        if (stop == std::string_view::npos) {
            accumulated += wire.substr(split);
            stop = stop_offset(accumulated, split);
        }
        CHECK(stop != std::string_view::npos);
        if (stop == std::string_view::npos) {
            continue;
        }
        accumulated.resize(stop);
        const auto accepted = parse(accumulated, true);
        CHECK(accepted.complete);
        CHECK(accepted.items == std::vector<std::string>({ "甲任务", "Beta" }));
    }
}

static void test_immediate_blank_line_stop() {
    // Blank line immediately after newline: offset of '\n' is not '-', so it terminates
    std::string text = "- First\n\n- Second\n";
    size_t      term = stop_offset(text, 0);
    CHECK(term == 8);  // offset of second '\n'

    auto res = parse(text);
    CHECK(res.complete);
    CHECK(res.accepted_bytes == 8);
    CHECK(res.items.size() == 1);
    CHECK(res.items[0] == "First");

    // Immediate blank line at offset 0
    std::string text0 = "\n- First\n";
    CHECK(stop_offset(text0, 0) == 0);
    auto res0 = parse(text0);
    CHECK(!res0.complete);
    CHECK(res0.items.empty());
    CHECK(!res0.error.empty());
}

static void test_indented_bullet_terminates() {
    // Indented bullet ('  - Subtask') starts with ' ' at line start, not '-'.
    // Must terminate rather than become child / hierarchy.
    std::string text = "- Parent task\n  - Subtask\n";
    size_t      term = stop_offset(text, 0);
    CHECK(term == 14);  // offset of ' ' at line 2

    auto res = parse(text);
    CHECK(res.complete);
    CHECK(res.accepted_bytes == 14);
    CHECK(res.items.size() == 1);
    CHECK(res.items[0] == "Parent task");
}

static void test_free_labels_and_unicode() {
    // Free punctuation / Unicode / Mermaid reserved characters
    std::vector<std::string> items = { "Fix bug in C++: #include <vector> & std::string?",
                                       "Compute 25 × 12 = 300; [test] (nested) {braces}!",
                                       "总目标与计划: Ω 完成 100% 测试",
                                       "Mermaid symbols: graph TD; A-->B; mindmap root",
                                       "Quotes: \"double\" 'single' `backtick` \\backslash /slash" };

    std::string wire = serialize(items);
    wire += "Thinking concluded here.\n";

    auto res = parse(wire);
    CHECK(res.complete);
    CHECK(res.items.size() == items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        CHECK(res.items[i] == items[i]);
    }
}

static void test_eog_finish() {
    // eof=true finalizes trailing item without requiring newline
    std::string text           = "- Incomplete line without trailing newline";
    auto        res_incomplete = parse(text, false);
    CHECK(!res_incomplete.complete);
    CHECK(res_incomplete.items.empty());

    auto res_eof = parse(text, true);
    CHECK(res_eof.complete);
    CHECK(res_eof.items.size() == 1);
    CHECK(res_eof.items[0] == "Incomplete line without trailing newline");
    CHECK(res_eof.accepted_bytes == text.size());

    // Multiple items with last one lacking newline under eof=true
    std::string text2 = "- Task 1\n- Task 2";
    auto        res2  = parse(text2, true);
    CHECK(res2.complete);
    CHECK(res2.items.size() == 2);
    CHECK(res2.items[0] == "Task 1");
    CHECK(res2.items[1] == "Task 2");
    CHECK(res2.accepted_bytes == text2.size());

    const auto empty = parse("", true);
    CHECK(!empty.complete);
    CHECK(!empty.error.empty());
}

static void test_empty_task_rejection() {
    // Empty task: "-" alone or "- " or "-   \n"
    std::string text1 = "-\n";
    auto        res1  = parse(text1);
    CHECK(!res1.complete);
    CHECK(!res1.error.empty());

    std::string text2 = "- \n";
    auto        res2  = parse(text2);
    CHECK(!res2.complete);
    CHECK(!res2.error.empty());

    std::string text3 = "-   \t \r\n";
    auto        res3  = parse(text3);
    CHECK(!res3.complete);
    CHECK(!res3.error.empty());

    // Empty task in the middle
    std::string text4 = "- Valid 1\n- \n- Valid 2\n";
    auto        res4  = parse(text4);
    CHECK(!res4.complete);
    CHECK(!res4.error.empty());
}

static void test_crlf_handling() {
    // Optional CR before LF may be removed
    std::string text = "- Task 1\r\n- Task 2\r\nDone\r\n";
    auto        res  = parse(text);
    CHECK(res.complete);
    CHECK(res.items.size() == 2);
    CHECK(res.items[0] == "Task 1");
    CHECK(res.items[1] == "Task 2");
}

static void test_optional_single_space() {
    // Item content starts after '-'. Optional single space removed.
    // If no space, e.g. "-Task", item is "Task".
    // If two spaces, e.g. "-  Task", only the single space after '-' is removed -> " Task".
    std::string text = "-Task 1\n-  Task 2\nDone\n";
    auto        res  = parse(text);
    CHECK(res.complete);
    CHECK(res.items.size() == 2);
    CHECK(res.items[0] == "Task 1");
    CHECK(res.items[1] == " Task 2");
}

int main() {
    test_stop_offset_incremental();
    test_discard_terminator_and_suffix();
    test_two_chunk_boundaries();
    test_immediate_blank_line_stop();
    test_indented_bullet_terminates();
    test_free_labels_and_unicode();
    test_eog_finish();
    test_empty_task_rejection();
    test_crlf_handling();
    test_optional_single_space();

    if (g_failures > 0) {
        std::fprintf(stderr, "FAILED: %d tests failed\n", g_failures);
        return 1;
    }
    std::printf("PASS: test-rerot-todo\n");
    return 0;
}
