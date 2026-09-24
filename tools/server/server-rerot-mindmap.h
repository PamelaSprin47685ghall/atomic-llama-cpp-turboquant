#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Strict MM-R1 Mermaid mindmap wire parser (host side).
//
// Protocol (see the mindmap experiment design doc, section 3):
//   ```mermaid\nmindmap\n  root\n    <first child>\n      ...\n  ```\n
//   - LF only; CRLF, TAB, blank lines and surrounding prose are rejected.
//   - The literal root line is a synthetic container, not a logical node or
//     worker. Children start at 4 spaces; each level adds 2 spaces; no level
//     skipping or second root is allowed.
//   - The document is only atomic after the whole closing line is consumed and
//     the entire input is exhausted. Partial input yields status::incomplete
//     with the guarantee that a legal completion still exists.
//
// The parser deliberately knows nothing about llama_context, KV cache or the
// scheduler: it converts bytes into a tree plus diagnostics.

namespace server_mindmap {

inline constexpr std::string_view probe_prefix = "```mermaid\nmindmap\n  root\n    ";

enum class status : uint8_t {
    // A legal completion of the document still exists.
    incomplete = 0,
    // The document is complete and legal (envelope closed, no trailing junk).
    complete,
    // No legal completion exists; the error class identifies the defect.
    invalid,
};

enum class error_class : uint8_t {
    none = 0,
    envelope,       // header/fence/trailing garbage/CR/Tab
    structure,      // root, indentation, skipped level, blank line
    label,          // empty/whitespace/bad scalar label
};

struct node {
    // Host-assigned identity: document order index (root == 0).
    uint32_t host_node_id = 0;
    uint32_t parent       = UINT32_MAX;
    uint32_t depth        = 0;   // root == 1 (matches the wire indentation level)
    uint64_t subtree_eleaves = 0;
    uint32_t preorder_rank = 0;
    std::string label;
};

struct plan {
    // Node id 0 is the synthetic wire root; node_count excludes it. Other
    // ids are dense document-order indices for serialization and scheduling.
    std::vector<node> nodes;
    std::vector<std::vector<uint32_t>> children; // parallel to nodes
    uint32_t root = UINT32_MAX;
    uint64_t tree_hash = 0;

    uint32_t leaf_count = 0;
    uint32_t node_count = 0;
    uint32_t depth      = 0;

    size_t leaves(std::vector<uint32_t> & out) const;
    bool   is_leaf(uint32_t id) const { return id != root && id < nodes.size() && id < children.size() && children[id].empty(); }
};

struct result {
    status st = status::incomplete;
    error_class err = error_class::none;
    std::string error;
    plan tree;

    // Byte offset of the first offending byte (0 when unknown) and the 1-based
    // line/column of that offset for human diagnostics.
    size_t byte_offset = 0;
    uint32_t line = 0;
    uint32_t column = 0;
    size_t consumed_bytes = 0;

    bool is_incomplete() const { return st == status::incomplete; }
    bool is_complete()   const { return st == status::complete; }
    bool is_invalid()    const { return st == status::invalid; }
};

// Incremental verdict over `text` (the whole probe output so far). `text` may
// end mid-token, mid-line, or mid-fence. Purely functional: no state is kept
// between calls, so callers may feed any prefix.
result parse(const std::string & text);

// Canonical serialization of an accepted plan: the wire round-trips
// (parse(serialize(p)) == p) and is the identity recorded as canonical_wire.
std::string serialize(const plan & p);

// Stable structural hash over the accepted tree (labels included). Node
// identity (host ids) is NOT part of it: two plans differing only in id
// assignment hash equal only when labels/structure match.
uint64_t hash_tree(const plan & p);

// Per-label scalar check used by the parser. Returns true for a non-empty label made of the MM-R1
// character set (ASCII words plus the CJK/kana/hangul and operator scalars).
bool label_ok(const std::string & label);

// G0 constrains the wire's line syntax. The companion sampler constrains
// each line's depth relative to the preceding line; neither has a depth cap.
std::string grammar_g0();

// State starts after probe_prefix, at the first child's label. consume() is
// transactional: callers may try a candidate token without mutating the
// accepted prefix. The GBNF supplies labels/fences; this state supplies the
// cross-line nesting constraint that an unbounded static GBNF cannot express.
struct indent_state {
    size_t last_depth = 1;
    size_t spaces = 4;
    bool at_indent = false;
    bool has_label = false;
    uint8_t fence_bytes = 0;
    bool closed = false;

    bool consume(std::string_view bytes);
};

// Composed after grammar_g0() in the probe grammar sampler only. Owns no
// vocabulary; its immutable token-piece table is shared across clones.
llama_sampler * init_indent_sampler(const llama_vocab * vocab);

} // namespace server_mindmap
