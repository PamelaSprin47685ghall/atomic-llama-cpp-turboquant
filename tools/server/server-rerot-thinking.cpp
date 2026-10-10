#include "server-rerot-thinking.h"

#include <cctype>

namespace server_thinking {

std::string grammar() {
    // Generation starts inside the FIRST item, right after the injected
    // "<ul>...<li>", so the root begins with that item's text. No head region: see
    // the header for the three head designs that were measured and dropped.
    //
    //   text  words, spaces and commas only -- no sentence punctuation, no newline.
    //         This is the only thing that makes an item terminate: '</li>' starts
    //         with '<', which min_p removes from the pool once prose is under way
    //         (measured on this model: '<' at rank 2, p=0.086, 0.28 of the top token
    //         at the start of a free region; below 1.2e-3 and 2.7e-3 of the top after
    //         one sentence). An item that can hold prose never closes: one ran 1670
    //         tokens and 6061 chars without a single '</li>'. Without punctuation the
    //         lane has to say it in one breath and runs out of breath, which ends the
    //         item on its own. It also removes '&lt;' escaping and stray newlines as
    //         failure modes. Path and identifier characters (/ - _) stay legal so an
    //         aspect can still name a file.
    //   item  bare <li>...</li>; no class attribute to drop.
    //   gap   at most a single newline between items -- never free text, or the lane
    //         thinks between every pair of items and never closes the list.
    //   end   </ul> is reachable only after at least one complete item, so the first
    //         '<' the lane types cannot close an empty list.
    return "root ::= sp? text \"</li>\" (nl? item)* nl? \"</ul>\"\n"
           "item ::= \"<li>\" sp? text \"</li>\"\n"
           "text ::= wchar tchar*\n"
           "tchar ::= wchar | sp | comma\n"
           "wchar ::= [0-9A-Za-z/_\\u00C0-\\u024F\\u0370-\\u03FF\\u0400-\\u04FF"
           "\\u3040-\\u30FA\\u3400-\\u4DBF\\u4E00-\\u9FFF\\uAC00-\\uD7AF-]\n"
           "sp ::= \" \"\n"
           "comma ::= \",\" | \"\\uFF0C\"\n"
           "nl ::= \"\\n\"\n";
}

size_t stop_offset(std::string_view text, size_t begin) {
    if (text.empty()) {
        return std::string_view::npos;
    }
    // Overlap by terminator-1 bytes so a marker split across token pieces is
    // still found exactly once, at its true offset.
    const size_t overlap = terminator.size() - 1;
    const size_t from    = begin > overlap ? begin - overlap : 0;
    return text.find(terminator, from);
}

static bool is_space(char c) {
    return std::isspace(static_cast<unsigned char>(c)) != 0;
}

static std::string_view trim(std::string_view s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && is_space(s[b])) {
        ++b;
    }
    while (e > b && is_space(s[e - 1])) {
        --e;
    }
    return s.substr(b, e - b);
}

static size_t skip_space(std::string_view s, size_t pos) {
    while (pos < s.size() && is_space(s[pos])) {
        ++pos;
    }
    return pos;
}

result parse(std::string_view text, bool eof) {
    result res;

    const size_t term = stop_offset(text, 0);
    if (term == std::string_view::npos) {
        if (eof) {
            // The list is the plan; a source that stops early has not produced
            // one. Never synthesize the missing close.
            res.error = "question list ended without " + std::string(terminator);
        }
        return res;
    }

    const std::string_view body = text.substr(0, term);

    size_t pos = skip_space(body, 0);
    if (body.compare(pos, open_list.size(), open_list) != 0) {
        res.error = "question list does not open with " + std::string(open_list);
        return res;
    }
    pos += open_list.size();

    // Head: the probe's own thinking, kept out of the plan. It ends at the first
    // '<', which the grammar allows only as the start of an item. The head is
    // discarded together with the probe KV, so it is only validated, never used.
    const size_t head_end = body.find('<', pos);
    if (head_end == std::string_view::npos) {
        res.error = "question list has no questions";
        res.accepted_bytes = pos;
        return res;
    }
    pos = head_end;

    while (true) {
        if (pos >= body.size()) {
            break;
        }
        if (body.compare(pos, open_item.size(), open_item) != 0) {
            // Includes the "free text between items" case: after </li> only a
            // single newline and then <li> or the terminator are legal.
            res.error = "expected " + std::string(open_item) + " before a question";
            res.accepted_bytes = pos;
            return res;
        }
        pos += open_item.size();

        const size_t close = body.find(close_item, pos);
        if (close == std::string_view::npos) {
            // Inside the terminator the item must be closed; a dangling item is
            // a protocol defect, not a truncation (the terminator already came).
            res.error = "question not closed by " + std::string(close_item);
            res.accepted_bytes = pos;
            return res;
        }
        const std::string_view item = trim(body.substr(pos, close - pos));
        if (item.empty()) {
            res.error = "empty question";
            res.accepted_bytes = pos;
            return res;
        }
        if (item.find('<') != std::string_view::npos) {
            res.error = "markup inside a question";
            res.accepted_bytes = pos;
            return res;
        }
        res.items.emplace_back(item);
        pos = skip_space(body, close + close_item.size());
    }

    if (res.items.empty()) {
        res.error = "question list has no questions";
        return res;
    }

    res.accepted_bytes = term;
    res.complete       = true;
    return res;
}

std::string serialize(const std::vector<std::string> & items) {
    std::string out(open_list);
    out += '\n';
    for (const auto & item : items) {
        out += open_item;
        out += item;
        out += close_item;
        out += '\n';
    }
    out += terminator;
    out += '\n';
    return out;
}

}  // namespace server_thinking
