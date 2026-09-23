#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Strict MM-R1 Mermaid mindmap wire parser (host side).
//
// Protocol (see the mindmap experiment design doc, section 3):
//   ```mermaid\nmindmap\n  <root label>\n    <child>\n      ...\n  ```\n
//   - LF only; CRLF, TAB, blank lines and surrounding prose are rejected.
//   - Root is indented by exactly 2 spaces; each level adds 2 spaces; no level
//     skipping; a second root is rejected.
//   - The document is only atomic after the whole closing line is consumed and
//     the entire input is exhausted. Partial input yields status::incomplete
//     with the guarantee that a legal completion still exists.
//
// The parser deliberately knows nothing about llama_context, KV cache or the
// scheduler: it converts bytes into a tree plus diagnostics.

namespace server_mindmap {

// Resource limits. They are explicit budget checks, never syntax errors.
struct limits {
    uint32_t max_depth     = 6;
    uint32_t max_nodes     = 96;
    uint32_t max_leaves    = 16;
    uint32_t max_label_utf8 = 96;  // UTF-8 bytes of one label
    size_t   max_bytes     = 16384;

    static limits defaults() { return limits{}; }
};

enum class status : uint8_t {
    // A legal completion of the document still exists.
    incomplete = 0,
    // The document is complete and legal (envelope closed, no trailing junk,
    // all budgets respected).
    complete,
    // No completion exists (protocol defect) or a budget was exceeded. The
    // distinction is carried by error_class.
    invalid,
};

enum class error_class : uint8_t {
    none = 0,
    envelope,       // header/fence/trailing garbage/CR/Tab
    structure,      // root, indentation, skipped level, blank line
    label,          // empty/whitespace/bad scalar label
    budget,         // depth/nodes/leaves/label/bytes exceeded
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
    // Canonical plan tree. Node ids are dense document-order indices so the
    // structure round-trips through serialization without a second mapping.
    std::vector<node> nodes;
    std::vector<std::vector<uint32_t>> children; // parallel to nodes
    uint32_t root = UINT32_MAX;
    uint64_t tree_hash = 0;

    uint32_t leaf_count = 0;
    uint32_t node_count = 0;
    uint32_t depth      = 0;

    size_t leaves(std::vector<uint32_t> & out) const;
    bool   is_leaf(uint32_t id) const { return id < nodes.size() && children[id].empty(); }
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
    bool budget_hit()    const { return st == status::invalid && err == error_class::budget; }
};

// Incremental verdict over `text` (the whole probe output so far). `text` may
// end mid-token, mid-line, or mid-fence. Purely functional: no state is kept
// between calls, so callers may feed any prefix.
result parse(const std::string & text, const limits & lim = limits::defaults());

// Canonical serialization of an accepted plan: the wire round-trips
// (parse(serialize(p)) == p) and is the identity recorded as canonical_wire.
std::string serialize(const plan & p);

// Stable structural hash over the accepted tree (labels included). Node
// identity (host ids) is NOT part of it: two plans differing only in id
// assignment hash equal only when labels/structure match.
uint64_t hash_tree(const plan & p);

// Per-label scalar check used by the parser and exposed for the incremental
// sampler oracle. Returns true for one non-empty label made of the MM-R1
// character set (ASCII words plus the CJK/kana/hangul and operator scalars).
bool label_ok(const std::string & label);

// G0 grammar text (bounded depth). Node/leaf/label/byte budgets remain the
// parser's job; the grammar only fixes depth and the character subset.
std::string grammar_g0();

} // namespace server_mindmap
