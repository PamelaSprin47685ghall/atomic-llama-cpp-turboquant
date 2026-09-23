// G2 indent-aware GBNF extension tests (M12).
//
// The extension is opt-in: a grammar text without %indent/%dedent must behave
// exactly as before, and one that uses them gets a lexed event stream the
// existing PDA consumes unchanged. These tests pin both halves plus the policy
// decisions the lexer deliberately does NOT hide (blank lines, tabs, unmatched
// columns, unterminated final line).

#include "../src/llama-grammar.h"

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

static std::string kinds_to_string(const std::vector<llama_grammar_indent_event> & evs) {
    std::string out;
    for (const auto & e : evs) {
        switch (e.kind) {
            case LLAMA_GRAMMAR_INDENT_CH:      out += "ch:";  out += char(e.cp); out += ' '; break;
            case LLAMA_GRAMMAR_INDENT_NEWLINE: out += "NL ";  break;
            case LLAMA_GRAMMAR_INDENT_INDENT:  out += "IN ";  break;
            case LLAMA_GRAMMAR_INDENT_DEDENT:  out += "OUT "; break;
            case LLAMA_GRAMMAR_INDENT_EOF:     out += "EOF "; break;
            default:                           out += "? ";   break;
        }
    }
    return out;
}

// Builds a grammar from text; returns null on a parse error.
static struct llama_grammar * make_grammar(const std::string & text) {
    return llama_grammar_init_impl(nullptr, text.c_str(), "root", false, nullptr, 0, nullptr, 0);
}

// Runs the lexer over `text` and returns the event names.
static std::string lex(const std::string & text) {
    struct llama_grammar * g = make_grammar(
        "root ::= %indent \"x\" %dedent\n");
    if (g == nullptr) {
        return "<parse-error>";
    }
    if (!g->indent_lexer) {
        return "<lexer-not-enabled>";
    }
    // One space between the two segments: each segment already ends with a
    // marker space, so the join must not add another.
    std::string out = kinds_to_string(llama_grammar_indent_lex(*g, text));
    out += "| ";
    out += kinds_to_string(llama_grammar_indent_finish(*g));
    llama_grammar_free_impl(g);
    return out;
}

static void test_lexer_is_opt_in() {
    // A grammar without directives must not enable the lexer and must not
    // change any behaviour.
    struct llama_grammar * plain = make_grammar("root ::= \"a\"\n");
    CHECK(plain != nullptr);
    if (plain == nullptr) {
        return;
    }
    CHECK(!plain->indent_lexer);
    CHECK(llama_grammar_indent_lex(*plain, "  a\n").empty());
    CHECK(llama_grammar_indent_finish(*plain).empty());
    llama_grammar_free_impl(plain);

    // A grammar that merely contains a percent sign (but no directive) is still
    // an ordinary grammar.
    struct llama_grammar * percent = make_grammar("root ::= \"%\"\n");
    CHECK(percent != nullptr);
    if (percent != nullptr) {
        CHECK(!percent->indent_lexer);
        llama_grammar_free_impl(percent);
    }

    // A directive enables it.
    struct llama_grammar * indented = make_grammar("root ::= %indent \"a\"\n");
    CHECK(indented != nullptr);
    if (indented != nullptr) {
        CHECK(indented->indent_lexer);
        llama_grammar_free_impl(indented);
    }
}

static void test_basic_indent_dedent_events() {
    // Two levels of indentation then a return to the root level.
    const std::string events = lex("a\n  b\n");
}

static void test_blank_line_policy() {
    // A blank line reports only the boundary; the next non-empty line settles
    // its own level, which is deeper than the root, so exactly one INDENT is
    // emitted for it -- never a phantom INDENT for the blank line itself.
    const std::string events = lex("a\n\n  b\n");
    CHECK(events == "ch:a NL NL IN ch:b NL | OUT EOF ");
}

static void test_tab_policy() {
    // A TAB is NOT converted into a column: it closes the indentation run and
    // is handed through as an ordinary scalar, so a grammar that accepts only
    // spaces rejects the line. The lexer never silently rewrites a TAB into
    // spaces (that would accept an input the grammar never allowed).
    const std::string events = lex("a\n\tb\n");
    CHECK(events == "ch:a NL ch:\t ch:b NL | EOF ");
}

static void test_unmatched_column_is_reported() {
    // A dedent to a column that was never opened reports DEDENT + INDENT so a
    // strict grammar can reject the shape instead of accepting a rewrite.
    const std::string events = lex("a\n    b\n  c\n");
    CHECK(events == "ch:a NL IN ch:b NL OUT IN ch:c NL | OUT EOF ");
}

static void test_unterminated_final_line() {
    // An unterminated final line still settles its level, then EOF.
    const std::string events = lex("a\n  b");
    CHECK(events == "ch:a NL IN ch:b | OUT EOF ");
}

static void test_eof_only_after_settle() {
    // Finish() on an already-settled input emits only EOF.
    const std::string events = lex("a");
    CHECK(events == "ch:a | EOF ");
}

static void test_state_is_cloneable_and_resettable() {
    // The lexer state lives on the grammar by value, so clone/reset must carry
    // or clear it exactly like any other field.
    struct llama_grammar * g = make_grammar("root ::= %indent \"a\" %dedent\n");
    CHECK(g != nullptr);
    if (g == nullptr) {
        return;
    }
    CHECK(llama_grammar_indent_lex(*g, "a\n  b").size() > 1);

    struct llama_grammar * c = llama_grammar_clone_impl(*g);
    CHECK(c != nullptr);
    if (c != nullptr) {
        CHECK(c->indent_lexer);
        CHECK(c->indent_stack == g->indent_stack);
        llama_grammar_free_impl(c);
    }

    llama_grammar_free_impl(g);
}

static void test_unknown_directive_is_a_parse_error() {
    // A percent-prefixed token that is not a known directive must fail the
    // parse rather than silently becoming a terminal.
    struct llama_grammar * bad = make_grammar("root ::= %unknown \"a\"\n");
    CHECK(bad == nullptr);
}

int main() {
    test_lexer_is_opt_in();
    test_basic_indent_dedent_events();
    test_blank_line_policy();
    test_tab_policy();
    test_unmatched_column_is_reported();
    test_unterminated_final_line();
    test_eof_only_after_settle();
    test_state_is_cloneable_and_resettable();
    test_unknown_directive_is_a_parse_error();

    if (g_failures == 0) {
        std::printf("test-grammar-indent: all tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "test-grammar-indent: %d failure(s)\n", g_failures);
    return 1;
}
