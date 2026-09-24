// MM-R1 mindmap wire parser tests (strict, incremental, resource budgets).
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
        wire("  总目标\n"),
        wire("  Root\n    Leaf\n"),
        wire("  R\n    A\n      B\n        C\n"),
        wire("  R\n    A\n      A1\n      A2\n    B\n      B1\n      B2\n"),
        wire("  R\n    重复\n    重复\n"),
        wire("  求总和\n    求 25 × 12\n    求 15 × 16\n"),
        wire("  R\n    テスト\n    검증\n"),
        wire("  R\n    A\n      X\n    B\n"),
        wire("  R\n    A\n      B\n        C\n          D\n            E\n"), // depth 6
        wire("  R\n    42\n"), // label starting with a digit
    };
    std::string sixteen_leaves = "  R\n";
    for (int i = 0; i < 16; ++i) {
        sixteen_leaves += "    L" + std::to_string(i) + "\n";
    }
    valid.push_back(wire(sixteen_leaves));
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
    expect_reject(std::string("text\n") + wire("  R\n"), error_class::envelope, "prose before header");
    // A header with no body is still completable (the root may follow).
    CHECK(parse(std::string(HEADER)).is_incomplete());
    expect_reject(std::string(HEADER) + "  R\n" + "```junk\n", error_class::envelope, "fence with junk");
    expect_reject(wire("  R\n") + "trailing", error_class::envelope, "trailing bytes after fence");
    // Same for a fence whose final LF has not arrived yet.
    CHECK(parse(std::string(HEADER) + "  R\n```").is_incomplete());
    expect_reject(std::string(HEADER) + "  R\n    A\n```\n```\n", error_class::envelope, "two fences");
    expect_reject(wire("  R\n\ttab\n"), error_class::envelope, "tab indent");
    expect_reject(wire("  R\n  \ttab\n"), error_class::envelope, "tab after spaces");
    expect_reject(wire("  R\r\n"), error_class::envelope, "CRLF");
    expect_reject(wire("  R\n   X\n"), error_class::structure, "odd indent");
    expect_reject(wire("  R\n  S\n"), error_class::structure, "second root");
    expect_reject(wire("  R\n      X\n"), error_class::structure, "skipped level");
    expect_reject(std::string(HEADER) + "    R\n" + "```\n", error_class::structure, "root not at level 1");
    expect_reject(std::string(HEADER) + "  R\n\n" + "```\n", error_class::structure, "blank line");
    expect_reject(std::string(HEADER) + "  R\n   \n" + "```\n", error_class::structure, "odd-indent whitespace line");
    expect_reject(std::string(HEADER) + "  R\n    A\n      B\n        C\n          D\n            E\n              F\n" + "```\n",
        error_class::budget, "depth 7");
    expect_reject(wire("  R \n"), error_class::label, "trailing space in label");
    expect_reject(wire("  R\n    [leaf]\n"), error_class::label, "brackets");
    expect_reject(wire("  R\n    a\"b\n"), error_class::label, "quote");
    expect_reject(wire("  R\n    `code`\n"), error_class::label, "backtick");
    expect_reject(std::string(HEADER) + "  " + std::string(97, 'A') + "\n```\n",
        error_class::budget, "label bytes");
    std::string seventeen_leaves = "  R\n";
    for (int i = 0; i < 17; ++i) {
        seventeen_leaves += "    L" + std::to_string(i) + "\n";
    }
    expect_reject(wire(seventeen_leaves), error_class::budget, "17 leaves");
    {
        // Isolate the NODE budget from the leaf budget: a 65-node chain is
        // one leaf but 65 nodes. The budget fires on the node that would be
        // number 65, not on the closing fence.
        // A single 65-node chain hits the DEPTH budget first (node 17 is
        // already past max_depth 16) -- budgets fire in encounter order, so
        // the diagnosis names the binding one, not an unrelated limit.
        limits lim;
        lim.max_leaves = 8;
        lim.max_depth = 16;
        std::string chain = std::string(HEADER);
        for (uint32_t d = 1; d <= 65; ++d) {
            chain.append(d * 2, ' ');
            chain += "N" + std::to_string(d) + "\n";
        }
        chain += "```\n";
        auto res = parse(chain, lim);
        CHECK(res.is_invalid() && res.budget_hit());
        CHECK(res.error.find("depth") != std::string::npos);
    }
    {
        limits lim;
        lim.max_bytes = 10;
        auto res = parse(wire("  R\n"), lim);
        CHECK(res.is_invalid() && res.budget_hit());
    }
    {
        // Four nodes is exactly at budget (legal); five is over. Raise the
        // depth budget so the node budget is what binds, not chain depth.
        limits lim;
        lim.max_nodes = 4;
        lim.max_depth = 8;
        auto at = parse(wire("  R\n    A\n      B\n        C\n"), lim);
        CHECK(at.is_complete());
        auto over = parse(wire("  R\n    A\n      B\n        C\n          D\n"), lim);
        CHECK(over.is_invalid() && over.budget_hit());
        CHECK(over.error.find("node count") != std::string::npos);
    }
    {
        // One node (root only) is exactly at budget.
        limits lim;
        lim.max_nodes = 1;
        auto at = parse(wire("  R\n"), lim);
        CHECK(at.is_complete());
        auto over = parse(wire("  R\n    A\n"), lim);
        CHECK(over.is_invalid() && over.budget_hit());
        CHECK(over.error.find("node count") != std::string::npos);
    }
    {
        // The leaf budget is only knowable once the tree closes, and it
        // counts LEAVES, not children: "R -> A" is one leaf (A has no child),
        // while "R -> {A, B}" is two.
        limits lim;
        lim.max_leaves = 1;
        auto single = parse(wire("  R\n    A\n"), lim);
        CHECK(single.is_complete());
        CHECK(single.tree.leaf_count == 1);
        auto two = parse(wire("  R\n    A\n    B\n"), lim);
        CHECK(two.is_invalid() && two.budget_hit());
        CHECK(two.error.find("leaves") != std::string::npos);
    }
}

// An incomplete prefix must stay incomplete as long as a legal completion
// exists, and turn complete exactly when the fence is closed. Feeding the
// document byte by byte must never produce `invalid` for a legal sample.
static void test_incremental_prefixes() {
    const std::vector<std::string> samples = {
        wire("  总目标\n"),
        wire("  R\n    A\n      B\n"),
        wire("  求总和\n    计算分量\n      求 25 × 12\n      求 15 × 16\n    独立核验\n      检查分量计算和最终总和\n"),
        wire("  R\n    A\n    B\n"),
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
        const std::string text = wire("  总目标\n");
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
        const std::string text = wire("  R\n    " + label + "\n");
        const size_t label_end = text.find(label) + label.size();
        for (size_t n = 0; n < label.size(); ++n) {
            auto res = parse(text.substr(0, label_end - n));
            CHECK(res.is_incomplete());
        }
    }
    // Prefixes of an ILLEGAL document may become invalid before the end, but
    // must never become complete.
    {
        const std::string bad = wire("  R\n      X\n"); // skipped level
        for (size_t n = 0; n <= bad.size(); ++n) {
            auto res = parse(bad.substr(0, n));
            CHECK(!res.is_complete());
        }
    }
}

static void test_plan_shape() {
    const std::string text = wire("  R\n    A\n      A1\n      A2\n    B\n      B1\n      B2\n");
    auto res = parse(text);
    CHECK(res.is_complete());
    CHECK(res.tree.node_count == 7);
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
    // Root-only tree: node count, depth and leaf accounting agree with "one
    // leaf, no children".
    auto root_only = parse(wire("  R\n"));
    CHECK(root_only.is_complete());
    CHECK(root_only.tree.node_count == 1);
    CHECK(root_only.tree.leaf_count == 1);
    CHECK(root_only.tree.depth == 1);
    CHECK(root_only.tree.nodes[0].subtree_eleaves == 1);
}

// Deep and wide trees must stay inside their declared budgets and report the
// budget class (not a syntax error) when they cross one.
static void test_limits_are_budgets() {
    limits lim;
    lim.max_depth = 4;
    {
        const std::string text = wire("  R\n    A\n      B\n        C\n");
        auto res = parse(text, lim);
        CHECK(res.is_complete());
        CHECK(res.tree.depth == 4);
    }
    {
        const std::string text = wire("  R\n    A\n      B\n        C\n          D\n");
        auto res = parse(text, lim);
        CHECK(res.is_invalid() && res.budget_hit());
    }
    lim.max_depth = 16;
    {
        // A single chain: root at 2 spaces, then 4, 6, ... spaces. Each level
        // adds exactly one child, so the tree stays legal at depth 16.
        std::string body;
        for (uint32_t d = 1; d <= 16; ++d) {
            body.append(d * 2, ' ');
            body += "L" + std::to_string(d) + "\n";
        }
        auto res = parse(wire(body), lim);
        if (!res.is_complete()) {
            std::fprintf(stderr, "FAIL %s:%d: depth16 rejected: %s\n",
                __FILE__, __LINE__, res.error.c_str());
            ++g_failures;
        }
        CHECK(res.tree.depth == 16);
        CHECK(res.tree.leaf_count == 1);
    }
}

// The canonical serialization is a fixed point: parse(serialize(p)) == p and
// the tree hash is structural, not id-dependent.
static void test_roundtrip_and_hash() {
    const std::string text = wire("  求总和\n    计算分量\n      求 25 × 12\n    独立核验\n      检查总和\n");
    auto a = parse(text);
    CHECK(a.is_complete());
    auto b = parse(serialize(a.tree));
    CHECK(b.is_complete());
    CHECK(a.tree.tree_hash == b.tree.tree_hash);
    // Different labels hash differently; identical structure+labels hash equal.
    auto c = parse(wire("  求总和\n    独立核验\n      检查总和\n    计算分量\n      求 25 × 12\n"));
    CHECK(c.is_complete());
    CHECK(c.tree.tree_hash != a.tree.tree_hash);
    auto d = parse(text);
    CHECK(d.tree.tree_hash == a.tree.tree_hash);
}

static void test_grammar_text() {
    const std::string g = grammar_g0();
    CHECK(g.find("root ::= \"```mermaid") != std::string::npos);
    CHECK(g.find("node-4 ::= \"        \" label \"\\n\"") != std::string::npos);
    // The G0 grammar must not assert budgets it cannot express.
    CHECK(g.find("node-5") == std::string::npos);
}

int main() {
    test_valid_documents();
    test_invalid_structure();
    test_incremental_prefixes();
    test_plan_shape();
    test_limits_are_budgets();
    test_roundtrip_and_hash();
    test_grammar_text();

    if (g_failures == 0) {
        std::printf("test-rerot-mindmap-parser: all tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "test-rerot-mindmap-parser: %d failure(s)\n", g_failures);
    return 1;
}
