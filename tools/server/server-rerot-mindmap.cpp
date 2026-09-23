#include "server-rerot-mindmap.h"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace server_mindmap {

namespace {

constexpr char HEADER[] = "```mermaid\nmindmap\n";
constexpr const char * FENCE_CLOSE = "```";

constexpr size_t HEADER_LEN = sizeof(HEADER) - 1; // 19 bytes
constexpr size_t CLOSE_LEN  = 3;  // strlen(FENCE_CLOSE)

// MM-R1 label charset. Mirrors the G0 grammar's `char` production exactly:
// ASCII alphanumerics plus a small operator set, the CJK/kana/hangul ranges
// and the CJK punctuation a natural-language task label needs. Everything
// else (parens, ASCII quotes, backticks, HTML, emoji, tabs, control bytes)
// is rejected -- the grammar is the contract, not an approximation.
struct scalar_range {
    uint32_t lo;
    uint32_t hi;
};

constexpr scalar_range kScalarRanges[] = {
    { 0x2A, 0x2A },      // *
    { 0x2B, 0x2B },      // +
    { 0x2D, 0x2D },      // -
    { 0x2E, 0x2E },      // .
    { 0x30, 0x39 },      // 0-9
    { 0x3A, 0x3A },      // :
    { 0x3B, 0x3B },      // ;
    { 0x3D, 0x3D },      // =
    { 0x3F, 0x3F },      // ?
    { 0x41, 0x5A },      // A-Z
    { 0x5F, 0x5F },      // _
    { 0x61, 0x7A },      // a-z
    { 0xA1, 0xA1 },      // inverted exclamation
    { 0xA8, 0xA8 },
    { 0xB0, 0xB0 },      // degree
    { 0xB7, 0xB7 },      // middle dot
    { 0xD7, 0xD7 },      // multiplication
    { 0xF7, 0xF7 },      // division
    { 0x2018, 0x201D },  // single/double curly quotes
    { 0x3001, 0x3002 },  // CJK comma / full stop
    { 0x300C, 0x300F },  // CJK quotes
    { 0x3040, 0x30FF },  // hiragana + katakana
    { 0x3400, 0x4DBF },  // CJK ext A
    { 0x4E00, 0x9FFF },  // CJK unified
    { 0xAC00, 0xD7AF },  // hangul syllables
    { 0xFF01, 0xFF01 },  // fullwidth exclamation
    { 0xFF0C, 0xFF0C },  // fullwidth comma
    { 0xFF1A, 0xFF1A },  // fullwidth colon
    { 0xFF1B, 0xFF1B },  // fullwidth semicolon
    { 0xFF1F, 0xFF1F },  // fullwidth question
    { 0xFF3F, 0xFF5E },  // fullwidth letters / digits / symbols
};

bool scalar_allowed(uint32_t cp) {
    for (const auto & r : kScalarRanges) {
        if (cp >= r.lo && cp <= r.hi) {
            return true;
        }
    }
    return false;
}

// Decode one UTF-8 scalar at s[i].
//   > 0 : byte length, cp set
//   == 0: the bytes present so far are a valid but incomplete sequence
//   UINT32_MAX: decisively invalid sequence (cp set to 0)
uint32_t decode_one(const std::string & s, size_t i, uint32_t & cp) {
    const unsigned char c0 = (unsigned char) s[i];
    size_t   need  = 0;
    uint32_t value = 0;
    uint32_t min_cp = 0;
    if (c0 < 0x80) {
        cp = c0;
        return 1;
    } else if ((c0 & 0xE0) == 0xC0) {
        need = 1; value = c0 & 0x1F; min_cp = 0x80;
    } else if ((c0 & 0xF0) == 0xE0) {
        need = 2; value = c0 & 0x0F; min_cp = 0x800;
    } else if ((c0 & 0xF8) == 0xF0) {
        need = 3; value = c0 & 0x07; min_cp = 0x10000;
    } else {
        cp = 0;
        return UINT32_MAX;
    }
    for (size_t k = 1; k <= need; ++k) {
        if (i + k >= s.size()) {
            return 0; // truncated continuation: not decisive yet
        }
        const unsigned char ck = (unsigned char) s[i + k];
        if ((ck & 0xC0) != 0x80) {
            cp = 0;
            return UINT32_MAX;
        }
        value = (value << 6) | (ck & 0x3F);
    }
    if (value < min_cp || value > 0x10FFFF || (value >= 0xD800 && value <= 0xDFFF)) {
        cp = 0;
        return UINT32_MAX;
    }
    cp = value;
    return need + 1;
}

// 0 = still a valid prefix of the header, 1 = complete, UINT32_MAX = impossible.
uint32_t header_state(const std::string & s, size_t pos) {
    size_t k = 0;
    while (k < HEADER_LEN && pos + k < s.size()) {
        if (s[pos + k] != HEADER[k]) {
            return UINT32_MAX;
        }
        ++k;
    }
    return k == HEADER_LEN ? 1u : 0u;
}

bool bytes_equal(const std::string & s, size_t pos, const char * lit, size_t n) {
    if (s.size() - pos < n) {
        return false;
    }
    return s.compare(pos, n, lit, n) == 0;
}

// True when the bytes at `pos` are the beginning of a legal closing fence
// line: an exact "```" (terminated by LF, end-of-input, or nothing yet), or a
// truncated prefix of it. A line starting with a fence-shaped prefix but a
// different third byte is NOT a fence prefix (e.g. "``x").
bool fence_prefix_at(const std::string & s, size_t pos) {
    const size_t avail = s.size() - pos;
    if (avail == 0) {
        return false;
    }
    if (s[pos] != '`') {
        return false;
    }
    if (avail == 1) {
        return true; // truncated "`"
    }
    if (s[pos + 1] != '`') {
        return false;
    }
    if (avail == 2) {
        return true; // truncated "``"
    }
    return s[pos + 2] == '`';
}

} // namespace

bool label_ok(const std::string & label) {
    if (label.empty()) {
        return false;
    }
    // MM-R1 labels are `word (" " word)*`: a single ASCII space separates
    // words and may never lead or trail the label.
    if (label.front() == ' ' || label.back() == ' ') {
        return false;
    }
    size_t i = 0;
    while (i < label.size()) {
        uint32_t cp = 0;
        const uint32_t n = decode_one(label, i, cp);
        if (n == 0 || n == UINT32_MAX) {
            return false;
        }
        if (!scalar_allowed(cp)) {
            if (cp != 0x20) {
                return false;
            }
            // Reject runs of more than one space.
            if (i + 1 < label.size() && label[i + 1] == ' ') {
                return false;
            }
        }
        i += n;
    }
    return true;
}

size_t plan::leaves(std::vector<uint32_t> & out) const {
    out.clear();
    for (uint32_t i = 0; i < nodes.size(); ++i) {
        if (children[i].empty()) {
            out.push_back(i);
        }
    }
    return out.size();
}

std::string serialize(const plan & p) {
    if (p.root == UINT32_MAX || p.root >= p.nodes.size()) {
        return {};
    }
    std::string out = HEADER;
    std::vector<uint32_t> order;
    order.reserve(p.nodes.size());
    std::vector<uint32_t> stack;
    stack.push_back(p.root);
    while (!stack.empty()) {
        const uint32_t u = stack.back();
        stack.pop_back();
        order.push_back(u);
        const auto & kids = p.children[u];
        for (size_t k = kids.size(); k-- > 0;) {
            stack.push_back(kids[k]);
        }
    }
    for (uint32_t u : order) {
        out.append((size_t) p.nodes[u].depth * 2, ' ');
        out += p.nodes[u].label;
        out += '\n';
    }
    out += FENCE_CLOSE;
    out += '\n';
    return out;
}

uint64_t hash_tree(const plan & p) {
    // FNV-1a over the canonical serialization (structure + labels): two plans
    // hash equal iff their canonical wire is identical, which is exactly the
    // identity the experiment ledger records.
    const std::string canonical = serialize(p);
    uint64_t h = 1469598103934665603ULL;
    for (const char c : canonical) {
        h ^= (uint64_t) (unsigned char) c;
        h *= 1099511628211ULL;
    }
    return h;
}

std::string grammar_g0() {
    // Depth-bounded (4 levels) GBNF. Node/leaf/label/byte budgets stay the
    // parser's job: a character grammar cannot express them, and pretending
    // otherwise would misreport a budget failure as a syntax error.
    return
        "root ::= \"```mermaid\\nmindmap\\n\" node-1 \"```\\n\"\n"
        "node-1 ::= \"  \" label \"\\n\" node-2*\n"
        "node-2 ::= \"    \" label \"\\n\" node-3*\n"
        "node-3 ::= \"      \" label \"\\n\" node-4*\n"
        "node-4 ::= \"        \" label \"\\n\"\n"
        "label ::= word (\" \" word)*\n"
        "word ::= char+\n"
        "char ::= [A-Za-z0-9\\u3400-\\u4DBF\\u4E00-\\u9FFF\\u3040-\\u30FF"
        "\\uAC00-\\uD7AF\\u3001-\\u3002\\u300C-\\u300F\\u2018-\\u201D"
        "\\uFF01\\uFF0C\\uFF1A\\uFF1B\\uFF1F\\uFF3F-\\uFF5E"
        "\\u00A1\\u00A8\\u00B0\\u00B7\\u00D7\\u00F7"
        "*/=.,?!;:_\\u002D]\n";
}

result parse(const std::string & text, const limits & lim) {
    result res;


    if (text.size() > lim.max_bytes) {
        res.st = status::invalid;
        res.err = error_class::budget;
        res.error = "resource_limit: wire bytes";
        res.byte_offset = lim.max_bytes;
        res.consumed_bytes = text.size();
        return res;
    }

    const uint32_t hs = header_state(text, 0);
    if (hs == UINT32_MAX) {
        res.st = status::invalid;
        res.err = error_class::envelope;
        res.error = "envelope: expected ```mermaid\\nmindmap\\n header";
        res.byte_offset = 0;
        res.consumed_bytes = 0;
        return res;
    }
    if (hs == 0) {
        res.st = status::incomplete;
        res.consumed_bytes = text.size();
        return res;
    }

    size_t pos = HEADER_LEN;

    std::vector<uint32_t> stack; // open ancestor chain, depth-indexed
    std::vector<node> nodes;
    std::vector<std::vector<uint32_t>> children;

    auto incomplete = [&]() {
        res.st = status::incomplete;
        res.consumed_bytes = text.size();
        return res;
    };
    auto fail = [&](error_class cls, const std::string & msg, size_t off) {
        res.st = status::invalid;
        res.err = cls;
        res.error = msg;
        res.byte_offset = off;
        res.consumed_bytes = off;
        return res;
    };

    while (true) {
        if (pos >= text.size()) {
            return incomplete();
        }

        // Line start: closing fence, or one node line.
        if (fence_prefix_at(text, pos)) {
            const size_t avail = text.size() - pos;
            if (avail < CLOSE_LEN) {
                return incomplete(); // truncated fence: still completable
            }
            // Exact fence bytes. Legal only as a bare "```\n" at column 0
            // with nothing after it, and only for a document that has a root.
            size_t after = pos + CLOSE_LEN;
            if (after >= text.size()) {
                return incomplete(); // need the LF to decide
            }
            if (text[after] != '\n') {
                return fail(error_class::envelope,
                            "envelope: closing fence must be at column 0 with no trailing text",
                            pos);
            }
            if (after + 1 != text.size()) {
                return fail(error_class::envelope,
                            "envelope: trailing content after closing fence",
                            after + 1);
            }
            if (nodes.empty()) {
                return fail(error_class::structure,
                            "structure: document has no root node", pos);
            }
            res.st = status::complete;
            res.consumed_bytes = text.size();
            res.tree.nodes = std::move(nodes);
            res.tree.children = std::move(children);
            res.tree.root = 0;
            res.tree.node_count = (uint32_t) res.tree.nodes.size();
            uint32_t depth_max = 0;
            std::vector<uint64_t> subtree_leaves(res.tree.nodes.size(), 0);
            for (size_t i = res.tree.nodes.size(); i-- > 0;) {
                uint64_t l = res.tree.children[i].empty() ? 1 : 0;
                for (uint32_t c : res.tree.children[i]) {
                    l += subtree_leaves[c];
                }
                subtree_leaves[i] = l;
                depth_max = std::max(depth_max, res.tree.nodes[i].depth);
                res.tree.nodes[i].subtree_eleaves = l;
            }
            res.tree.leaf_count = (uint32_t) subtree_leaves[0];
            res.tree.depth = depth_max;
            res.tree.tree_hash = hash_tree(res.tree);
            if (res.tree.leaf_count > lim.max_leaves) {
                // The leaf budget is only knowable once the tree closes: a
                // late parent-to-internal promotion changes the count.
                res.st = status::invalid;
                res.err = error_class::budget;
                res.error = "resource_limit: leaves";
                res.byte_offset = pos;
                res.consumed_bytes = pos;
                res.tree = plan{};
                return res;
            }
            return res;
        }

        // ---- node line ----
        // Indentation can only be judged once the line is terminated: an
        // unterminated tail of spaces may still grow to a legal indent (or
        // to a legal label separator), so the verdict waits for the LF.
        if (text[pos] == '\t') {
            return fail(error_class::envelope, "envelope: TAB is not allowed", pos);
        }
        uint32_t spaces = 0;
        while (pos + spaces < text.size() && text[pos + spaces] == ' ') {
            ++spaces;
        }
        const size_t content = pos + spaces;
        if (content < text.size() && text[content] == '\t') {
            return fail(error_class::envelope, "envelope: TAB is not allowed", content);
        }
        if (spaces % 2 != 0 && content < text.size()) {
            // A non-space byte is already visible: the odd run is decisive.
            return fail(error_class::structure,
                        "structure: indentation must be a multiple of 2 spaces", pos);
        }
        if (content >= text.size()) {
            return incomplete(); // only spaces so far: still completable
        }
        // Root is at exactly 2 spaces => level 1; each level adds 2 spaces.
        const uint32_t depth = spaces / 2;
        if (depth < 1) {
            return fail(error_class::structure,
                        "structure: node line must be indented by at least 2 spaces", pos);
        }
        if (depth > lim.max_depth) {
            return fail(error_class::budget, "resource_limit: depth", pos);
        }
        if (nodes.empty()) {
            if (depth != 1) {
                return fail(error_class::structure,
                            "structure: first node must be the root at indent 2", pos);
            }
        } else if (depth == 1) {
            return fail(error_class::structure,
                        "structure: second root node is not allowed", pos);
        }
        if (depth > (uint32_t) stack.size() + 1) {
            return fail(error_class::structure,
                        "structure: skipped an indentation level", pos);
        }

        size_t i = content;
        bool line_end = false;
        while (i < text.size()) {
            const char c = text[i];
            if (c == '\n') {
                line_end = true;
                break;
            }
            if (c == '\r') {
                return fail(error_class::envelope, "envelope: CR is not allowed", i);
            }
            if (c == '\t') {
                return fail(error_class::envelope, "envelope: TAB is not allowed", i);
            }
            ++i;
        }
        const std::string label = text.substr(content, i - content);
        if (label.size() > lim.max_label_utf8) {
            return fail(error_class::budget, "resource_limit: label bytes", content);
        }
        if (!server_mindmap::label_ok(label)) {
            if (line_end || i + 1 < text.size()) {
                return fail(error_class::label, "label: illegal scalar in node label", content);
            }
            return incomplete();
        }
        if (!line_end) {
            return incomplete();
        }
        if (label.empty()) {
            return fail(error_class::label, "label: empty node label", content);
        }

        if (nodes.size() >= lim.max_nodes) {
            // Resource verdict, never a syntax one. The check sits AFTER the
            // structure verdicts so a legal document that is merely over
            // budget is never misreported as a protocol defect.
            return fail(error_class::budget, "resource_limit: node count", pos);
        }
        const uint32_t parent = depth > 1 ? stack[depth - 2] : UINT32_MAX;
        const uint32_t nid = (uint32_t) nodes.size();
        node n;
        n.host_node_id = nid;
        n.parent = parent;
        n.depth = depth;
        n.label = label;
        nodes.push_back(std::move(n));
        children.push_back({});
        if (parent != UINT32_MAX) {
            children[parent].push_back(nid);
        }
        stack.resize(depth - 1);
        stack.push_back(nid);

        pos = i + 1;
    }
}

} // namespace server_mindmap
