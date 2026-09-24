// MM-R1 mindmap wire parser tests (strict, incremental, structural checks).
//
// The parser is the protocol gate for the mindmap research line: it must accept
// every legal document (including every truncation of one), reject every
// illegal byte sequence with a classified diagnosis, and never silently repair.
// These tests are byte-level and model-free.

#include "server-rerot-mindmap.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        ++g_failures; \
    } \
} while (0)

using namespace server_mindmap;

static const char * HEADER = "```mermaid\nmindmap\n";

static std::string wire(const std::string & body) {
    return std::string(HEADER) + body + "```\n";
}

// Every legal document is accepted, round-trips through serialize(), and its
// canonical form is idempotent.
static void test_valid_documents() {
    std::vector<std::string> valid = {
        wire("  root\n    总目标\n"),
        wire("  root\n    Leaf\n"),
        wire("  root\n    A\n      B\n        C\n"),
        wire("  root\n    A\n      A1\n      A2\n    B\n      B1\n      B2\n"),
        wire("  root\n    重复\n    重复\n"),
        wire("  root\n    求 25 × 12\n    求 15 × 16\n"),
        wire("  root\n    テスト\n    검증\n"),
        wire("  root\n    A\n      X\n    B\n"),
        wire("  root\n    A\n      B\n        C\n          D\n            E\n              F\n"), // deeper than the old default
        wire("  root\n    42\n"), // label starting with a digit
        wire("  root\n    N/2-1\n    1! 2+3,4\n"), // punctuation admitted by the grammar
    };
    std::string many_leaves = "  root\n";
    for (int i = 0; i < 2048; ++i) {
        many_leaves += "    L" + std::to_string(i) + "\n";
    }
    CHECK(wire(many_leaves).size() > 16384);
    const size_t many_index = valid.size();
    valid.push_back(wire(many_leaves));
    valid.push_back(wire("  root\n    " + std::string(256, 'A') + "\n"));
    std::string deep = "  root\n";
    for (uint32_t d = 2; d <= 64; ++d) {
        deep.append(d * 2, ' ');
        deep += "N" + std::to_string(d) + "\n";
    }
    valid.push_back(wire(deep));
    for (const auto & text : valid) {
        auto res = parse(text);
        if (!res.is_complete()) {
            std::fprintf(stderr, "FAIL %s:%d: rejected legal wire (line %u col %u): %s\n  wire=%s\n",
                __FILE__, __LINE__, res.line, res.column, res.error.c_str(), text.c_str());
            ++g_failures;
            continue;
        }
        const std::string canon = serialize(res.tree);
        CHECK(canon == text);
        auto again = parse(canon);
        CHECK(again.is_complete());
        CHECK(again.tree.tree_hash == res.tree.tree_hash);
        if (&text == &valid[many_index]) {
            CHECK(res.tree.leaf_count == 2048);
            CHECK(res.tree.node_count == 2048);
        }
    }
}

static void expect_reject(const std::string & text, error_class cls, const char * why) {
    auto res = parse(text);
    if (!res.is_invalid()) {
        std::fprintf(stderr, "FAIL %s:%d: accepted illegal wire (%s): %s\n",
            __FILE__, __LINE__, why, text.c_str());
        ++g_failures;
        return;
    }
    if (res.err != cls) {
        std::fprintf(stderr, "FAIL %s:%d: wrong class for %s (got %d want %d) wire=%s\n",
            __FILE__, __LINE__, why, (int) res.err, (int) cls, text.c_str());
        ++g_failures;
    }
    // A rejected document must never expose a tree.
    CHECK(res.tree.nodes.empty());
}

static void test_invalid_structure() {
    // An empty input is a valid prefix of the header, not a rejection:
    // a legal completion still exists.
    CHECK(parse("").is_incomplete());
    // A truncated header is not a rejection either.
    CHECK(parse("```mermaid\nmindmap").is_incomplete());
    CHECK(parse("```").is_incomplete());
    CHECK(parse("```mermaid\nmind").is_incomplete());
    expect_reject(std::string("text\n") + wire("  root\n    A\n"), error_class::envelope, "prose before header");
    // A header with no body is still completable (the root may follow).
    CHECK(parse(std::string(HEADER)).is_incomplete());
    CHECK(parse(std::string(probe_prefix)).is_incomplete());
    CHECK(parse(std::string(HEADER) + "  ro").is_incomplete());
    expect_reject(std::string(HEADER) + "  R", error_class::structure, "impossible root prefix");
    expect_reject(std::string(HEADER) + "  root\n    A\n" + "```junk\n", error_class::envelope, "fence with junk");
    expect_reject(wire("  root\n    A\n") + "trailing", error_class::envelope, "trailing bytes after fence");
    // Same for a fence whose final LF has not arrived yet.
    CHECK(parse(std::string(HEADER) + "  root\n    A\n```").is_incomplete());
    expect_reject(std::string(HEADER) + "  root\n    A\n```\n```\n", error_class::envelope, "two fences");
    expect_reject(wire("  root\n\ttab\n"), error_class::envelope, "tab indent");
    expect_reject(wire("  root\n  \ttab\n"), error_class::envelope, "tab after spaces");
    expect_reject(wire("  root\r\n"), error_class::envelope, "CRLF");
    expect_reject(wire("  root\n   X\n"), error_class::structure, "odd indent");
    expect_reject(wire("  root\n  S\n"), error_class::structure, "second root");
    expect_reject(wire("  root\n      X\n"), error_class::structure, "skipped level");
    expect_reject(std::string(HEADER) + "    root\n" + "```\n", error_class::structure, "root not at level 1");
    expect_reject(std::string(HEADER) + "  root\n\n" + "```\n", error_class::structure, "blank line");
    expect_reject(std::string(HEADER) + "  root\n   \n" + "```\n", error_class::structure, "odd-indent whitespace line");
    expect_reject(wire("  root\n"), error_class::structure, "root without a child");
    expect_reject(wire("  R\n    A\n"), error_class::structure, "nonliteral root");
    expect_reject(wire("  root\n    A \n"), error_class::label, "trailing space in label");
    expect_reject(wire("  root\n    [leaf]\n"), error_class::label, "brackets");
    expect_reject(wire("  root\n    a\"b\n"), error_class::label, "quote");
    expect_reject(wire("  root\n    `code`\n"), error_class::label, "backtick");
}

// An incomplete prefix must stay incomplete as long as a legal completion
// exists, and turn complete exactly when the fence is closed. Feeding the
// document byte by byte must never produce `invalid` for a legal sample.
static void test_incremental_prefixes() {
    const std::vector<std::string> samples = {
        wire("  root\n    总目标\n"),
        wire("  root\n    A\n      B\n"),
        wire("  root\n    计算分量\n      求 25 × 12\n      求 15 × 16\n    独立核验\n      检查分量计算和最终总和\n"),
        wire("  root\n    A\n    B\n"),
    };
    for (const auto & text : samples) {
        for (size_t n = 0; n <= text.size(); ++n) {
            const std::string prefix = text.substr(0, n);
            auto res = parse(prefix);
            const bool complete_expected = (n == text.size());
            if (complete_expected) {
                CHECK(res.is_complete());
            } else if (res.is_invalid()) {
                std::fprintf(stderr, "FAIL %s:%d: legal prefix turned invalid at %zu/%zu: %s\n",
                    __FILE__, __LINE__, n, text.size(), res.error.c_str());
                ++g_failures;
            }
            // byte_offset / consumed_bytes must stay inside the input.
            CHECK(res.consumed_bytes <= prefix.size());
        }
    }
    // Half a UTF-8 scalar at the tail stays incomplete (no decisive verdict).
    {
        const std::string text = wire("  root\n    总目标\n");
        for (size_t n = 1; n <= 3; ++n) {
            const std::string cut = text.substr(0, text.size() - n);
            // Only the trailing fence/newline bytes produce these shapes; a
            // cut inside body must not become invalid.
            auto res = parse(cut);
            CHECK(!res.is_invalid());
        }
    }
    // A prefix that ends inside a multi-byte label is incomplete, not invalid.
    {
        const std::string label = "任"; // 3 UTF-8 bytes
        const std::string text = wire("  root\n    " + label + "\n");
        const size_t label_end = text.find(label) + label.size();
        for (size_t n = 0; n < label.size(); ++n) {
            auto res = parse(text.substr(0, label_end - n));
            CHECK(res.is_incomplete());
        }
    }
    // Prefixes of an ILLEGAL document may become invalid before the end, but
    // must never become complete.
    {
        const std::string bad = wire("  root\n      X\n"); // skipped level
        for (size_t n = 0; n <= bad.size(); ++n) {
            auto res = parse(bad.substr(0, n));
            CHECK(!res.is_complete());
        }
    }
}

static void test_plan_shape() {
    const std::string text = wire("  root\n    A\n      A1\n      A2\n    B\n      B1\n      B2\n");
    auto res = parse(text);
    CHECK(res.is_complete());
    CHECK(res.tree.node_count == 6);
    CHECK(res.tree.root == 0);
    CHECK(res.tree.leaf_count == 4);
    CHECK(res.tree.depth == 3);
    CHECK(res.tree.children[0].size() == 2);
    CHECK(res.tree.children[1].size() == 2);
    CHECK(res.tree.nodes[1].label == "A");
    CHECK(res.tree.nodes[0].parent == UINT32_MAX);
    CHECK(res.tree.nodes[1].parent == 0);
    CHECK(res.tree.nodes[2].parent == 1);
    CHECK(res.tree.nodes[4].parent == 0);
    CHECK(res.tree.nodes[4].label == "B");
    CHECK(res.tree.nodes[0].subtree_eleaves == 4);
    CHECK(res.tree.nodes[2].subtree_eleaves == 1);
    std::vector<uint32_t> leaves;
    CHECK(res.tree.leaves(leaves) == 4);
    CHECK(leaves == std::vector<uint32_t>({2, 3, 5, 6}));
    CHECK(!res.tree.is_leaf(0));
    auto one_leaf = parse(wire("  root\n    唯一任务\n"));
    CHECK(one_leaf.is_complete());
    CHECK(one_leaf.tree.node_count == 1);
    CHECK(one_leaf.tree.leaf_count == 1);
    CHECK(one_leaf.tree.depth == 2);
    CHECK(one_leaf.tree.nodes[0].subtree_eleaves == 1);
}

// The canonical serialization is a fixed point: parse(serialize(p)) == p and
// the tree hash is structural, not id-dependent.
static void test_roundtrip_and_hash() {
    const std::string text = wire("  root\n    计算分量\n      求 25 × 12\n    独立核验\n      检查总和\n");
    auto a = parse(text);
    CHECK(a.is_complete());
    auto b = parse(serialize(a.tree));
    CHECK(b.is_complete());
    CHECK(a.tree.tree_hash == b.tree.tree_hash);
    // Different labels hash differently; identical structure+labels hash equal.
    auto c = parse(wire("  root\n    独立核验\n      检查总和\n    计算分量\n      求 25 × 12\n"));
    CHECK(c.is_complete());
    CHECK(c.tree.tree_hash != a.tree.tree_hash);
    auto d = parse(text);
    CHECK(d.tree.tree_hash == a.tree.tree_hash);
}

int main() {
    test_valid_documents();
    test_invalid_structure();
    test_incremental_prefixes();
    test_plan_shape();
    test_roundtrip_and_hash();

    if (g_failures == 0) {
        std::printf("test-rerot-mindmap-parser: all tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "test-rerot-mindmap-parser: %d failure(s)\n", g_failures);
    return 1;
}
