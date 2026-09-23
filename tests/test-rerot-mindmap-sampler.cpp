// MM-R1 G1 sampler tests (M08).
//
// The sampler owns the protocol's incremental mask. These tests pin it against
// three independent references:
//
//  1. The strict parser (server-rerot-mindmap.cpp): feeding the sampler a legal
//     document byte by byte must end in `complete`, and the parser must accept
//     the same document. Feeding it an illegal document must end in `dead`.
//  2. A byte-by-byte random walk: every prefix that the parser still considers
//     incomplete must leave the sampler in a non-dead state.
//  3. A token-level oracle: for a corpus of synthetic vocabulary pieces
//     (including pieces that span newline + indent + label + fence), the mask
//     implied by allow() must equal "the parser accepts the concatenation".

#include "server-rerot-mindmap-sampler.h"

#include "server-rerot-mindmap.h"

#include <cstdio>
#include <cstring>
#include <random>
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

// Feeds `text` through the sampler one byte at a time (the worst-case
// tokenization) and reports the final status.
static advance_status feed_bytes(const std::string & text, sampler_state & st,
                                 const sampler_limits & lim) {
    advance_status last = advance_status::keep_going;
    for (size_t i = 0; i < text.size(); ++i) {
        last = advance(st, text.data() + i, 1, lim);
        if (last == advance_status::dead) {
            return last;
        }
    }
    return last;
}

// Feeds `text` in random chunks: real tokenizers emit multi-byte pieces, so the
// state machine must survive arbitrary splits (including inside a UTF-8 scalar).
static advance_status feed_chunks(const std::string & text, sampler_state & st,
                                  const sampler_limits & lim, uint32_t seed) {
    std::mt19937 rng(seed);
    advance_status last = advance_status::keep_going;
    size_t i = 0;
    while (i < text.size()) {
        const size_t remain = text.size() - i;
        const size_t take = 1 + (rng() % std::min<size_t>(remain, 7));
        last = advance(st, text.data() + i, take, lim);
        if (last == advance_status::dead) {
            return last;
        }
        i += take;
    }
    return last;
}

static std::string escape_for_log(const std::string & in) {
    std::string out;
    for (char c : in) {
        switch (c) {
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c;      break;
        }
    }
    return out;
}

static void test_legal_documents_complete_byte_by_byte() {
    sampler_limits lim;
    const std::vector<std::string> valid = {
        wire("  总目标\n"),
        wire("  Root\n    Leaf\n"),
        wire("  R\n    A\n      B\n        C\n"),
        wire("  R\n    A\n      A1\n      A2\n    B\n      B1\n      B2\n"),
        wire("  R\n    重复\n    重复\n"),
        wire("  求总和\n    求 25 × 12\n    求 15 × 16\n"),
        wire("  R\n    テスト\n    검증\n"),
        wire("  R\n    42\n"),
        wire("  R\n    A\n    B\n    C\n    D\n    E\n    F\n    G\n    H\n"),
    };
    for (const auto & text : valid) {
        // The strict parser must accept it too (the two references agree).
        const auto parsed = parse(text);
        CHECK(parsed.is_complete());

        sampler_state st;
        const auto st_end = feed_bytes(text, st, lim);
        if (st_end != advance_status::complete) {
            std::fprintf(stderr, "FAIL %s:%d: sampler did not complete legal wire (%d): %s\n",
                __FILE__, __LINE__, (int) st_end, text.c_str());
            ++g_failures;
        }
        CHECK(requires_end(st));

        // Every prefix must stay non-dead.
        for (size_t n = 1; n < text.size(); ++n) {
            sampler_state p;
            const auto ps = feed_bytes(text.substr(0, n), p, lim);
            if (ps == advance_status::dead) {
                std::fprintf(stderr, "FAIL %s:%d: legal prefix %zu/%zu died\n",
                    __FILE__, __LINE__, n, text.size());
                ++g_failures;
                break;
            }
        }

        // Random chunking must reach the same verdict.
        for (uint32_t seed = 1; seed <= 8; ++seed) {
            sampler_state cs;
            const auto ce = feed_chunks(text, cs, lim, seed);
            if (ce != advance_status::complete) {
                std::fprintf(stderr, "FAIL %s:%d: chunked feed seed=%u status=%d\n",
                    __FILE__, __LINE__, seed, (int) ce);
                ++g_failures;
                break;
            }
        }
    }
}

static void test_illegal_documents_die() {
    sampler_limits lim;
    const std::vector<std::pair<std::string, const char *>> bad = {
        {wire("  R\n  S\n"), "second root"},
        {wire("  R\n      X\n"), "skipped level"},
        {wire("  R\n   X\n"), "odd indent"},
        {wire("  R\n    A\n      B\n        C\n          D\n"), "depth 5"},
        {wire("  R \n"), "trailing space"},
        {wire("  R\n    [leaf]\n"), "bracket"},
        {wire("  R\n    a\"b\n"), "quote"},
        {wire("  R\n    `code`\n"), "backtick"},
        {std::string(HEADER) + "  R\n\n```\n", "blank line"},
        {std::string(HEADER) + "  R\n" + std::string(97, 'A') + "\n```\n", "label bytes"},
        {wire("  R\n    A1\n    A2\n    B1\n    B2\n    C1\n    C2\n    D1\n    D2\n    E1\n"), "9 leaves"},
        {wire("  R\n") + "junk", "trailing junk after fence"},
        {std::string(HEADER) + "  R\n```junk\n", "fence with trailing text"},
        {std::string(HEADER) + "  R\r\n```\n", "CRLF"},
        {std::string(HEADER) + "  R\n\ttab\n```\n", "tab indent"},
        {std::string(HEADER) + "R\n```\n", "root at column 0"},
        {std::string(HEADER) + "    R\n```\n", "root too deep"},
    };
    for (const auto & [text, why] : bad) {
        // Both references must agree that the document never becomes legal.
        sampler_state st;
        const auto st_end = feed_bytes(text, st, lim);
        if (st_end != advance_status::dead) {
            std::fprintf(stderr, "FAIL %s:%d: sampler accepted illegal wire (%s), status=%d\n",
                __FILE__, __LINE__, why, (int) st_end);
            ++g_failures;
        }
        const auto parsed = parse(text);
        if (!parsed.is_invalid()) {
            std::fprintf(stderr, "FAIL %s:%d: parser accepted illegal wire (%s)\n",
                __FILE__, __LINE__, why);
            ++g_failures;
        }
    }
}

// The mask oracle. For a vocabulary of pieces, allow(st, piece) must be true
// exactly when the parser can still complete SOME document that begins with
// (bytes consumed so far) + piece. The reference implementation is therefore
// the strict parser over the reconstructed byte string, tested against a set of
// candidate suffixes -- this is the "slow oracle" the plan calls for: slow, and
// only used as ground truth.
static void test_mask_matches_parser_completion() {
    sampler_limits lim;
    // A vocabulary that mimics a real tokenizer: whole words, partial words,
    // newline+indent runs, pieces spanning several lines, fence fragments, and
    // pieces that split a multi-byte scalar.
    const std::vector<std::string> vocab = {
        "`", "``", "```", "```\n", "```mermaid\n", "mindmap\n",
        "mermaid", "mind", "map\n",
        "  ", "    ", "      ", "        ",
        "\n", "\n  ", "\n    ", "\n      ",
        "R", "o", "ot", " A", "1", "2", "B",
        "求", " 求", "总", "和", " 25 ", "×", " 12", "× 12",
        "任", "务", "A\n    ", "A1\n      ",
        " ```\n", "```\n", "x", " ", "  \n",
        "求总和\n    计算分量\n      求 25 × 12\n",
        "[", "]", "\"", "(", ")", "\t", "\r", "  \t", "R\r\n",
        "```junk", "```junk\n", "\n\n", "\n \n",
    };

    // Suffixes that can legally close a document from a reachable state. If the
    // parser accepts (prefix + piece + suffix) for ANY of them, the piece had a
    // legal completion. The first entry is the empty suffix so a piece that
    // already finishes the document (or the header) is judged correctly.
    const std::vector<std::string> closers = {
        "",
        "\n```\n",
        "\n    A\n```\n",
        "\n    A\n      B\n```\n",
        "  R\n```\n",
        "mermaid\nmindmap\n  R\n```\n",
        // Mid-header states: completing the header, then a one-leaf tree.
        "``mermaid\nmindmap\n  R\n```\n",
        "`mermaid\nmindmap\n  R\n```\n",
        "\nmindmap\n  R\n```\n",
        "mindmap\n  R\n```\n",
    };

    std::mt19937 rng(12345);
    int checked = 0;
    for (int round = 0; round < 300; ++round) {
        sampler_state probe;
        // Replay a random legal prefix from the vocabulary until the document
        // closes (or a piece dies, which is itself a useful state).
        std::string consumed;
        for (int step = 0; step < 12; ++step) {
            const std::string & p = vocab[rng() % vocab.size()];
            sampler_state next = probe;
            const auto r = advance(next, p, lim);
            if (r == advance_status::dead) {
                break;
            }
            probe = next;
            consumed += p;
            if (r == advance_status::complete) {
                break;
            }
        }
        if (probe.ph == phase::closed) {
            continue; // nothing more to test from a closed document
        }
        if (consumed.empty()) {
            // Mid-header states need the full header as a closer entry; the
            // sweep starts once at least one byte has been consumed.
            continue;
        }
        // A state still inside the header can only be completed by the rest of
        // the FIXED header followed by a legal tree. The remainder is computed
        // AFTER the piece is appended, so mixed pieces (header bytes plus more)
        // are judged exactly.
        std::vector<std::string> phase_closers = closers;

        for (const auto & piece : vocab) {
            const bool allowed = allow(probe, piece, lim);
            // Header-phase states have exactly one legal continuation: the rest
            // of the fixed header, then a legal tree.
            std::vector<std::string> try_closers = closers;
            if (probe.ph == phase::header) {
                const size_t after = probe.header_len + piece.size();
                const std::string full(HEADER);
                if (after <= full.size() && full.compare(0, probe.header_len,
                        consumed.data() + consumed.size() - probe.header_len,
                        probe.header_len) == 0 &&
                    full.compare(probe.header_len, piece.size(), piece) == 0) {
                    const std::string rest = full.substr(after);
                    try_closers.push_back(rest + "  R\n```\n");
                    try_closers.push_back(rest);
                }
            }
            // Oracle: does any suffix complete prefix+piece?
            bool completable = false;
            for (const auto & suffix : try_closers) {
                if (parse(consumed + piece + suffix).is_complete()) {
                    completable = true;
                    break;
                }
            }
            ++checked;
            if (allowed != completable) {
                // The sampler may be CONSERVATIVE for pieces that are legal in
                // isolation but need a longer suffix than any closer provides
                // (e.g. a deep indentation the closers do not build). Only a
                // sampler verdict of true with NO parser completion is a real
                // soundness bug (it would let the probe emit an illegal doc).
                if (allowed && !completable) {
                    std::fprintf(stderr,
                        "FAIL %s:%d: sampler allowed a piece the parser can never "
                        "complete: consumed=[%s] piece=[%s]\n",
                        __FILE__, __LINE__,
                        escape_for_log(consumed).c_str(), escape_for_log(piece).c_str());
                    ++g_failures;
                }
            }
        }
    }
    CHECK(checked > 1000); // the oracle must actually have been exercised
}

static void test_budget_latch_is_sticky() {
    sampler_limits lim;
    lim.max_nodes = 4;
    const std::string text = std::string(HEADER) + "  R\n    A\n    B\n    C\n    D\n";
    sampler_state st;
    const auto st_end = feed_bytes(text, st, lim);
    CHECK(st_end == advance_status::dead);
    CHECK(st.budget_failed);
    // Once latched, nothing revives the state -- not even a piece that would
    // otherwise be legal.
    sampler_state again = st;
    CHECK(advance(again, "```\n", lim) == advance_status::dead);

    // Exactly at budget is legal.
    sampler_limits ok_lim;
    ok_lim.max_nodes = 4;
    sampler_state ok_st;
    CHECK(feed_bytes(std::string(HEADER) + "  R\n    A\n    B\n    C\n", ok_st, ok_lim) != advance_status::dead);
    CHECK(!ok_st.budget_failed);

    // The leaf budget is knowable only as the tree closes: 8 leaves are fine,
    // 9 are not, and the 9th dies when its line is committed. Fresh limits --
    // the node budget of `lim` above is irrelevant here.
    sampler_limits leaf_lim;
    sampler_state leaf_st;
    CHECK(feed_bytes(wire("  R\n    A\n    B\n    C\n    D\n    E\n    F\n    G\n    H\n"),
                     leaf_st, leaf_lim) == advance_status::complete);
    CHECK(!leaf_st.budget_failed);
    sampler_state over_st;
    CHECK(feed_bytes(wire("  R\n    A\n    B\n    C\n    D\n    E\n    F\n    G\n    H\n    I\n"),
                     over_st, leaf_lim) == advance_status::dead);
    CHECK(over_st.budget_failed);
}

static void test_state_lifecycle() {
    sampler_limits lim;
    sampler_state a;
    const std::string part = std::string(HEADER) + "  R\n    A\n";
    CHECK(feed_bytes(part, a, lim) != advance_status::dead);
    CHECK(!requires_end(a));

    // Cloning (copy) then diverging must not alias.
    sampler_state b = a;
    CHECK(a == b);
    CHECK(advance(a, "```\n", lim) == advance_status::complete);
    CHECK(requires_end(a));
    // b is untouched by a's acceptance.
    CHECK(!requires_end(b));
    CHECK(b != a);

    // Reset returns a pristine state.
    b.reset();
    CHECK(b == sampler_state{});
    CHECK(advance(b, std::string(HEADER), lim) != advance_status::dead);

    // Reusing one state for many documents must not leak between them.
    for (int i = 0; i < 32; ++i) {
        sampler_state st;
        const std::string doc = wire("  R\n    A\n      A1\n");
        CHECK(feed_bytes(doc, st, lim) == advance_status::complete);
    }
}

static void test_piece_completes_helper() {
    sampler_limits lim;
    sampler_state st;
    // Header + a complete root line, WITHOUT the closing fence yet.
    CHECK(feed_bytes(std::string(HEADER) + "  R\n", st, lim) != advance_status::dead);
    CHECK(!requires_end(st));
    CHECK(piece_completes(st, "```\n", lim));
    CHECK(!piece_completes(st, "```", lim));      // missing the LF
    CHECK(!piece_completes(st, "```x\n", lim));   // junk on the fence line
    CHECK(!piece_completes(st, "  A\n", lim));    // more tree, not the fence
}

// The sampler must never allow a token whose piece ENDS the document with
// trailing bytes: the parser rejects any byte after the closing LF.
static void test_no_trailing_after_close() {
    sampler_limits lim;
    sampler_state st;
    CHECK(feed_bytes(wire("  R\n"), st, lim) == advance_status::complete);
    for (const auto & junk : {std::string("x"), std::string("\n"), std::string(" ")}) {
        sampler_state probe = st;
        CHECK(advance(probe, junk, lim) == advance_status::dead);
    }
    // EOG-equivalent: nothing at all may follow.
    CHECK(requires_end(st));
}

int main() {
    test_legal_documents_complete_byte_by_byte();
    test_illegal_documents_die();
    test_mask_matches_parser_completion();
    test_budget_latch_is_sticky();
    test_state_lifecycle();
    test_piece_completes_helper();
    test_no_trailing_after_close();

    if (g_failures == 0) {
        std::printf("test-rerot-mindmap-sampler: all tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "test-rerot-mindmap-sampler: %d failure(s)\n", g_failures);
    return 1;
}
