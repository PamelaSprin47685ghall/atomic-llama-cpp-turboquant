#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace server_todo {

// Same contract as the mindmap probe prompt (server_rerot_mindmap_probe_prompt),
// minus the nesting clause: this wire is flat by construction.
inline constexpr std::string_view prompt       =
    "Let me split the work into parallel parts, one per line, each a concrete task "
    "that makes sense on its own: no overlap, no repeats.\n";
inline constexpr std::string_view probe_prefix = "- ";

struct result {
    bool                     complete       = false;
    size_t                   accepted_bytes = 0;
    std::vector<std::string> items;
    std::string              error;
};

// Returns string_view::npos while incomplete, else offset of the FIRST byte at line start not '-'.
// Line start means offset 0 or preceded by LF ('\n').
// begin is previous buffer length for O(new bytes) incremental scan, including boundaries across token pieces.
size_t stop_offset(std::string_view text, size_t begin = 0);

// Parses text. Stops/discards non-dash line suffix starting at terminator offset.
// If eof == true, finalizes any trailing item without requiring a newline.
// Items are opaque free text; optional single space after '-' is removed; CR ('\r') before LF ('\n') may be removed;
// empty or whitespace-only item is an error.
result parse(std::string_view text, bool eof = false);

// Emits "- " + item + "\n" for each item.
std::string serialize(const std::vector<std::string> & items);

}  // namespace server_todo
