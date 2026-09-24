#include "server-rerot-mindmap-sampler.h"

#include <algorithm>
#include <cstring>

namespace server_mindmap {

namespace {

constexpr char HEADER[] = "```mermaid\nmindmap\n";
constexpr size_t HEADER_LEN = sizeof(HEADER) - 1;
constexpr char FENCE_CLOSE[] = "```";
constexpr size_t CLOSE_LEN = sizeof(FENCE_CLOSE) - 1;

// Per-scalar subset of the strict parser's `label_ok`. A token-level mask must
// reject a forbidden byte the moment it arrives, so this check runs per scalar
// AND the assembled label is re-validated at line commit.
// Single source of truth for the label charset. This must agree with the
// strict parser's `label_ok` -- the two are cross-checked by the differential
// test (test-rerot-mindmap-sampler).
bool scalar_allowed(uint32_t cp) {
    if (cp == 0x20) {
        return true; // separator, validated separately by the label scanner
    }
    if (cp >= 0x21 && cp <= 0x7E) {
        // The G0 charset is a SUBSET of printable ASCII: letters, digits and
        // the operator scalars the grammar names. Everything else printable
        // (brackets, quotes, backticks, angle brackets, ...) is rejected so a
        // label can never smuggle a structural or template boundary.
        static const char * ascii_ok =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            "*/=.,?!;:_-";
        const char c = (char) cp;
        for (const char * q = ascii_ok; *q; ++q) {
            if (*q == c) {
                return true;
            }
        }
        return false;
    }
    if (cp == 0xA1 || cp == 0xA8 || cp == 0xB0 || cp == 0xB7 || cp == 0xD7 || cp == 0xF7) {
        return true;
    }
    if (cp >= 0x2018 && cp <= 0x201D) {
        return true;
    }
    if (cp >= 0x3001 && cp <= 0x3002) {
        return true;
    }
    if (cp >= 0x300C && cp <= 0x300F) {
        return true;
    }
    if (cp >= 0x3040 && cp <= 0x30FF) {
        return true;
    }
    if (cp >= 0x3400 && cp <= 0x4DBF) {
        return true;
    }
    if (cp >= 0x4E00 && cp <= 0x9FFF) {
        return true;
    }
    if (cp >= 0xAC00 && cp <= 0xD7AF) {
        return true;
    }
    if (cp >= 0xFF01 && cp <= 0xFF5E) {
        return true;
    }
    return false;
}

struct utf8_step {
    uint32_t cp = 0;
    uint32_t len = 0;   // bytes consumed for a full scalar
    uint32_t need = 0;  // bytes still required for a partial scalar
};

// Decode one scalar starting at s[i], honouring a partial prefix carried in
// `st`. Sets cp/len/need. `len == 0 && need > 0` means "incomplete".
utf8_step decode_scalar(const uint8_t * s, size_t i, size_t n, const sampler_state & st) {
    utf8_step out;

    // Assemble the sequence: any partial prefix carried in the state first,
    // then as many bytes of this piece as available.
    uint8_t buf[4];
    size_t total = 0;
    if (st.utf8_partial_len > 0 && st.utf8_partial_len <= 4) {
        total = st.utf8_partial_len;
        std::memcpy(buf, st.utf8_partial, total);
    }

    size_t consumed = 0; // bytes taken from this piece

    // Read the lead byte first (from the piece or the carried prefix), then
    // decide how many continuation bytes the sequence needs.
    if (total == 0) {
        if (i >= n) {
            out.len = 0; // nothing to decode at all
            return out;
        }
        buf[0] = s[i];
        ++total;
        ++consumed;
    }
    {
        const uint8_t c0 = buf[0];
        if (c0 < 0x80) {
            out.need = 0;
        } else if ((c0 & 0xE0) == 0xC0) {
            out.need = 1;
        } else if ((c0 & 0xF0) == 0xE0) {
            out.need = 2;
        } else if ((c0 & 0xF8) == 0xF0) {
            out.need = 3;
        } else {
            out.need = 0xFFFFFFFFu;
            return out;
        }
    }

    const size_t required = (size_t) out.need + 1;
    while (total < required) {
        if (i + consumed >= n) {
            // Incomplete: the caller carries the partial bytes forward.
            out.len = 0;
            return out;
        }
        buf[total] = s[i + consumed];
        ++total;
        ++consumed;
    }

    if (out.need == 0) {
        out.cp = buf[0];
        out.len = consumed;
        return out;
    }

    uint32_t cp = buf[0] & (out.need == 1 ? 0x1Fu : (out.need == 2 ? 0x0Fu : 0x07u));
    for (size_t k = 1; k <= (size_t) out.need; ++k) {
        const uint8_t ck = buf[k];
        if ((ck & 0xC0) != 0x80) {
            out.need = 0xFFFFFFFFu;
            return out;
        }
        cp = (cp << 6) | (ck & 0x3F);
    }
    out.cp = cp;
    out.len = consumed;
    return out;
}

// Commits the current line's node (or its promotion of a parent to internal).
// Returns false when the line's indentation is illegal for the open stack.
struct line_commit {
    bool ok = false;
    bool dead = false;
    bool node_added = false;
};

line_commit commit_line(sampler_state & st, const sampler_limits & lim) {
    line_commit out;
    if (!st.has_root) {
        if (st.line_depth != 1) {
            out.dead = true;
            return out;
        }
        st.has_root = true;
        st.node_count = 1;
        st.leaf_count = 1;
        st.depth = 1;
        st.stack.clear();
        st.stack.push_back(0);
        out.node_added = true;
        out.ok = true;
        return out;
    }
    if (st.line_depth == 1) {
        out.dead = true; // second root
        return out;
    }
    if (st.line_depth > (uint32_t) st.stack.size() + 1) {
        out.dead = true; // skipped level
        return out;
    }
    if (st.line_depth > lim.max_depth) {
        st.budget_failed = true;
        out.dead = true;
        return out;
    }
    if (st.node_count >= lim.max_nodes) {
        st.budget_failed = true;
        out.dead = true;
        return out;
    }
    const uint32_t new_id = st.node_count;
    ++st.node_count;
    // Leaf accounting, decided BEFORE the stack is rewritten: the parent is the
    // node currently at level line_depth-1. If that slot is occupied, the parent
    // already had a child (so this line is a new sibling leaf: +1). If it is
    // empty, the parent was itself a leaf (promotion: net zero).
    const size_t stack_before = st.stack.size();
    const bool parent_was_leaf = (st.line_depth > stack_before);
    st.stack.resize(st.line_depth - 1);
    st.stack.push_back(new_id);
    st.depth = std::max(st.depth, st.line_depth);
    if (parent_was_leaf) {
        // Promotion: the parent stops being a leaf, the child becomes one.
    } else {
        ++st.leaf_count;
    }
    if (st.leaf_count > lim.max_leaves) {
        st.budget_failed = true;
        out.dead = true;
        return out;
    }
    out.ok = true;
    out.node_added = true;
    (void) new_id;
    return out;
}

} // namespace

sampler_limits sampler_limits::from_limits(const limits & lim) {
    sampler_limits out;
    out.max_depth      = lim.max_depth;
    out.max_nodes      = lim.max_nodes;
    out.max_leaves     = lim.max_leaves;
    out.max_label_utf8 = lim.max_label_utf8;
    out.max_bytes      = lim.max_bytes;
    return out;
}

void sampler_state::reset() {
    *this = sampler_state{};
}

bool sampler_state::operator==(const sampler_state & other) const {
    return ph == other.ph &&
           header_len == other.header_len &&
           stack == other.stack &&
           node_count == other.node_count &&
           leaf_count == other.leaf_count &&
           depth == other.depth &&
           in_line == other.in_line &&
           line_spaces == other.line_spaces &&
           line_depth == other.line_depth &&
           line_depth_known == other.line_depth_known &&
           line_level_committed == other.line_level_committed &&
           line_label == other.line_label &&
           line_has_nonspace == other.line_has_nonspace &&
           fence_len == other.fence_len &&
           fence_active == other.fence_active &&
           utf8_partial_len == other.utf8_partial_len &&
           utf8_partial_need == other.utf8_partial_need &&
           bytes == other.bytes &&
           has_root == other.has_root &&
           budget_failed == other.budget_failed &&
           std::memcmp(utf8_partial, other.utf8_partial, sizeof(utf8_partial)) == 0;
}

advance_status advance(sampler_state & st, const char * piece, size_t n,
                       const sampler_limits & lim) {
    if (st.budget_failed || st.utf8_partial_len >= sizeof(st.utf8_partial)) {
        return advance_status::dead;
    }
    const uint8_t * data = reinterpret_cast<const uint8_t *>(piece);
    size_t i = 0;
    while (i < n) {
        if (st.bytes >= lim.max_bytes) {
            return advance_status::dead;
        }
        ++st.bytes;

        // ---- closing-fence line -------------------------------------------
        if (st.fence_active) {
            if (st.fence_len < CLOSE_LEN) {
                if (data[i] == (uint8_t) FENCE_CLOSE[st.fence_len]) {
                    ++st.fence_len;
                    ++i;
                    continue;
                }
                if (data[i] == '\n') {
                    // "```\n": the document is complete.
                    st.ph = phase::closed;
                    st.fence_active = false;
                    st.in_line = false;
                    ++i;
                    continue;
                }
                return advance_status::dead;
            }
            // Fence bytes are all present: only the LF may follow.
            if (data[i] == '\n') {
                st.ph = phase::closed;
                st.fence_active = false;
                st.in_line = false;
                ++i;
                continue;
            }
            return advance_status::dead;
        }

        // ---- closed: nothing may follow -----------------------------------
        if (st.ph == phase::closed) {
            return advance_status::dead;
        }

        // ---- header -------------------------------------------------------
        if (st.ph == phase::header) {
            if (data[i] == (uint8_t) HEADER[st.header_len]) {
                ++st.header_len;
                ++i;
                if (st.header_len == HEADER_LEN) {
                    st.ph = phase::body;
                }
                continue;
            }
            return advance_status::dead;
        }

        // ---- body: line scanning ------------------------------------------
        const uint8_t c = data[i];

        if (!st.in_line) {
            // Line start: it is either a fence line or a node line.
            if (c == '`') {
                st.fence_active = true;
                st.fence_len = 1;
                st.in_line = true;
                ++i;
                continue;
            }
            if (c == ' ') {
                ++st.line_spaces;
                ++i;
                continue;
            }
            if (c == '\n') {
                // Blank line: no legal label can be produced from it.
                return advance_status::dead;
            }
            if (c == '\t' || c == '\r') {
                return advance_status::dead;
            }
            // First non-space byte: the indentation level is now known, but it
            // is only judged legal once the line is committed.
            st.in_line = true;
            if (st.line_spaces % 2 != 0) {
                return advance_status::dead;
            }
            st.line_depth = st.line_spaces / 2;
            st.line_depth_known = true;
            st.line_has_nonspace = true;
            // fall through to the scalar handling below
        }

        // Scanning a node line.
        if (c == '\n') {
            // End of line: commit or promote.
            if (st.line_label.empty()) {
                return advance_status::dead; // empty label
            }
            if (st.line_label.back() == ' ') {
                return advance_status::dead; // trailing space in the label
            }
            if (!label_ok(st.line_label)) {
                return advance_status::dead; // the strict parser would reject it
            }
            // No rollback: a dead verdict means the document can never be
            // completed, and a budget failure must stay latched so later
            // pieces cannot resurrect the state.
            line_commit res = commit_line(st, lim);
            if (res.dead || !res.ok) {
                return advance_status::dead;
            }
            st.in_line = false;
            st.line_spaces = 0;
            st.line_depth = 0;
            st.line_depth_known = false;
            st.line_has_nonspace = false;
            st.line_label.clear();
            ++i;
            continue;
        }
        if (c == '\t' || c == '\r') {
            return advance_status::dead;
        }
        if (c == '`') {
            // A backtick inside a label is outside the MM-R1 charset: it could
            // otherwise be mistaken for the fence.
            return advance_status::dead;
        }

        // Decode the scalar (honouring a partial prefix from earlier pieces).
        const utf8_step s = decode_scalar(data, i, n, st);
        if (s.need == 0xFFFFFFFFu) {
            return advance_status::dead;
        }
        if (s.len == 0) {
            // Incomplete: carry the bytes of THIS piece forward.
            const size_t have = st.utf8_partial_len;
            const size_t take2 = n - i;
            if (have >= sizeof(st.utf8_partial) ||
                take2 > sizeof(st.utf8_partial) - have) {
                return advance_status::dead; // cannot be a valid sequence
            }
            std::memcpy(st.utf8_partial + have, data + i, take2);
            st.utf8_partial_len = have + take2;
            st.utf8_partial_need = s.need;
            i = n;
            continue;
        }
        if (!scalar_allowed(s.cp)) {
            return advance_status::dead;
        }
        // Space is the only permitted separator; anything else printable that
        // the strict parser's `label_ok` would reject must die here too.
        if (s.cp == 0x20) {
            if (st.line_label.empty()) {
                return advance_status::dead; // leading space inside the label
            }
            if (st.line_label.back() == ' ') {
                return advance_status::dead; // double space
            }
            st.line_label.push_back(' ');
        } else {
            // Append the WHOLE scalar: the bytes carried in the state's partial
            // buffer first, then the `s.len` bytes this piece contributed.
            uint8_t whole[4];
            const size_t carried = st.utf8_partial_len;
            if (s.len > sizeof(whole) - carried) {
                return advance_status::dead;
            }
            for (size_t k = 0; k < carried; ++k) {
                whole[k] = st.utf8_partial[k];
            }
            for (uint32_t k = 0; k < s.len; ++k) {
                whole[carried + k] = data[i + k];
            }
            const size_t total_len = carried + s.len;
            if (st.line_label.size() + total_len > lim.max_label_utf8) {
                st.budget_failed = true;
                return advance_status::dead;
            }
            for (size_t k = 0; k < total_len; ++k) {
                st.line_label.push_back((char) whole[k]);
            }
        }
        st.utf8_partial_len = 0;
        st.utf8_partial_need = 0;
        i += s.len;
    }

    // The piece may end mid-scalar but never mid-line: a partially consumed
    // line keeps its own state, so the document still has a legal completion.
    if (st.ph == phase::closed) {
        return advance_status::complete;
    }
    return advance_status::keep_going;
}

advance_status advance(sampler_state & st, const std::string & piece,
                       const sampler_limits & lim) {
    return advance(st, piece.data(), piece.size(), lim);
}

bool allow(const sampler_state & st, const std::string & piece,
           const sampler_limits & lim) {
    sampler_state probe = st;
    return advance(probe, piece, lim) != advance_status::dead;
}

bool requires_end(const sampler_state & st) {
    return st.ph == phase::closed;
}

bool piece_completes(const sampler_state & st, const std::string & piece,
                     const sampler_limits & lim) {
    sampler_state probe = st;
    return advance(probe, piece, lim) == advance_status::complete;
}

} // namespace server_mindmap
