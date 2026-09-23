#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "server-rerot-mindmap.h"

// MM-R1 G1 incremental mindmap sampler (M08).
//
// The Mermaid wire cannot be expressed by a bounded-depth GBNF alone: an
// arbitrary token may carry newline + indentation + label bytes + part of the
// closing fence at once, and the budgets (nodes, leaves, label length) are
// structural, not lexical. G1 therefore simulates the WHOLE piece of every
// candidate token against the protocol's incremental state machine.
//
// The state machine is an independent re-implementation of the same contract the
// strict parser (server-rerot-mindmap.cpp) enforces on complete documents. The
// correctness contract between the two is the point: the sampler must never
// allow a token after which no legal MM-R1 document can be completed, and the
// parser must accept exactly the documents the sampler admits.
//
// Nothing here depends on llama_context, the KV cache, or the scheduler.

namespace server_mindmap {

// Sampler-side resource limits. Identical defaults to the parser's so a probe
// armed with one and judged by the other can never disagree.
struct sampler_limits {
    uint32_t max_depth      = 4;
    uint32_t max_nodes      = 64;
    uint32_t max_leaves     = 8;
    uint32_t max_label_utf8 = 96;
    size_t   max_bytes      = 16384;

    static sampler_limits from_limits(const limits & lim);
};

// (envelope_phase, grammar_states are intentionally absent: G1 owns the whole
// protocol language, so the product state IS the protocol state.)
enum class phase : uint8_t {
    // Before the fixed header "```mermaid\nmindmap\n" is complete.
    header = 0,
    // Inside the tree body: node lines and the closing fence.
    body,
    // The closing fence line "```\n" has been fully consumed. Only EOF is legal.
    closed,
};

struct sampler_state {
    phase ph = phase::header;

    // Header prefix consumed so far (0..HEADER_LEN).
    uint32_t header_len = 0;

    // Body bookkeeping, mirrored from the strict parser.
    std::vector<uint32_t> stack; // open ancestor chain, depth-indexed (node ids)
    uint32_t node_count = 0;
    uint32_t leaf_count = 0;   // current leaves (a parent can still be promoted)
    uint32_t depth = 0;        // deepest committed level

    // Line-level state for the line currently being scanned.
    bool in_line = false;
    uint32_t line_spaces = 0;   // leading spaces seen on this line
    uint32_t line_depth = 0;     // committed indentation level of this line
    bool line_depth_known = false;
    bool line_level_committed = false;
    std::string line_label;      // label bytes of this line so far
    bool line_has_nonspace = false; // a non-space byte has been seen on this line

    // Closing-fence scanning state for the line being consumed as the fence.
    uint32_t fence_len = 0;
    bool fence_active = false;

    // Partial UTF-8 sequence carried across token boundaries. 0 when none.
    uint32_t utf8_partial_len = 0;
    uint8_t  utf8_partial[4] = {0, 0, 0, 0};
    uint32_t utf8_partial_need = 0;

    // Whole-document byte count (for the wire budget).
    size_t bytes = 0;

    // Has the root been committed? (A node at level 1.)
    bool has_root = false;

    // Budget-failure latch: once a budget is exceeded the state machine has no
    // legal completion, so every later allow() must be false.
    bool budget_failed = false;

    void reset();

    bool operator==(const sampler_state & other) const;
    bool operator!=(const sampler_state & other) const { return !(*this == other); }
};

// The result of advancing a state by one complete token piece.
enum class advance_status : uint8_t {
    ok = 0,
    // The piece keeps a legal completion available.
    keep_going,
    // The piece completes the document exactly.
    complete,
    // No legal completion exists after this piece.
    dead,
};

// Advances `st` by the bytes of `piece`. Pure w.r.t. the input state: `st` is
// only mutated on a non-dead outcome, so a caller that wants to probe a token
// can pass a copy and decide afterwards.
//
// Byte-level contract: `piece` is decoded as UTF-8 exactly as the tokenizer
// produced it, so a candidate that splits a multi-byte scalar is handled by
// carrying the partial sequence in the state.
advance_status advance(sampler_state & st, const char * piece, size_t n,
                       const sampler_limits & lim);

// Overload for std::string pieces.
advance_status advance(sampler_state & st, const std::string & piece,
                       const sampler_limits & lim);

// Would appending `piece` leave a legal completion available? Never mutates
// `st`. This is the predicate the production mask needs; `advance` is kept
// separate so the slow oracle and the fast path share one definition.
bool allow(const sampler_state & st, const std::string & piece,
           const sampler_limits & lim);

// True when the state can only be completed by EOF (the document is closed and
// nothing else is legal). Used to decide when the prompt must stop.
bool requires_end(const sampler_state & st);

// Convenience for the slow oracle: apply the piece to a copy and report whether
// the document was completed exactly by it.
bool piece_completes(const sampler_state & st, const std::string & piece,
                     const sampler_limits & lim);

} // namespace server_mindmap
