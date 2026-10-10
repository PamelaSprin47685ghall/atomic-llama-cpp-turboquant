#include "server-rerot-todo.h"

#include <cctype>

namespace server_todo {

size_t stop_offset(std::string_view text, size_t begin) {
    if (text.empty()) {
        return std::string_view::npos;
    }

    // Determine starting index. If begin is 0, check offset 0.
    // If begin > 0, we might need to check if begin-1 was a '\n' (which would mean offset `begin` is line start).
    // Or if previous scan stopped inside a line, we search for '\n' starting from begin > 0 ? begin - 1 : 0.
    size_t scan_pos = (begin > 0) ? (begin - 1) : 0;

    // First check offset 0 if scan_pos == 0
    if (scan_pos == 0) {
        if (text[0] != '-') {
            return 0;
        }
    }

    while (scan_pos < text.size()) {
        size_t nl = text.find('\n', scan_pos);
        if (nl == std::string_view::npos) {
            break;
        }
        size_t next_line_start = nl + 1;
        if (next_line_start < text.size()) {
            if (text[next_line_start] != '-') {
                return next_line_start;
            }
            scan_pos = next_line_start;
        } else {
            // Newline is the last character in text; whether next line starts with '-' is not yet known
            break;
        }
    }

    return std::string_view::npos;
}

static bool is_whitespace_only(std::string_view s) {
    for (char c : s) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

result parse(std::string_view text, bool eof) {
    result res;

    size_t           term       = stop_offset(text, 0);
    std::string_view valid_text = text;
    bool             terminated = (term != std::string_view::npos);

    if (terminated) {
        valid_text = text.substr(0, term);
    }

    // Now process valid_text line by line
    size_t pos = 0;
    while (pos < valid_text.size()) {
        // Line start must be '-' because non-dash lines would have been stopped by stop_offset
        if (valid_text[pos] != '-') {
            res.complete       = false;
            res.error          = "Line does not start with '-'";
            res.accepted_bytes = pos;
            return res;
        }

        size_t nl = valid_text.find('\n', pos);
        if (nl == std::string_view::npos) {
            if (!eof) {
                break;
            }
            nl = valid_text.size();
        }

        size_t line_end = nl;
        size_t next_pos = (nl < valid_text.size() && valid_text[nl] == '\n') ? (nl + 1) : nl;

        // Content of item starts after '-'
        size_t item_start = pos + 1;
        // Optional single space after '-' removed
        if (item_start < line_end && valid_text[item_start] == ' ') {
            item_start++;
        }

        size_t item_end = line_end;
        // Optional CR before LF may be removed
        if (item_end > item_start && valid_text[item_end - 1] == '\r') {
            item_end--;
        }

        std::string_view item_view = valid_text.substr(item_start, item_end - item_start);
        if (item_view.empty() || is_whitespace_only(item_view)) {
            res.complete       = false;
            res.error          = "Empty or whitespace-only item";
            res.accepted_bytes = pos;
            return res;
        }

        res.items.emplace_back(item_view);
        pos = next_pos;
    }

    res.accepted_bytes = pos;
    if ((terminated || eof) && res.items.empty()) {
        res.error = "Todo plan has no items";
        return res;
    }
    res.complete = (terminated || eof) && res.accepted_bytes == valid_text.size();

    return res;
}

std::string serialize(const std::vector<std::string> & items) {
    std::string out;
    for (const auto & item : items) {
        out += "- ";
        out += item;
        out += '\n';
    }
    return out;
}

}  // namespace server_todo
