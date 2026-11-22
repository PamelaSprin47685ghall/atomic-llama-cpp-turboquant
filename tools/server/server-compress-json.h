#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace server_compress_json {

// All intervals use byte offsets and exclude the end offset.
struct range {
    size_t begin = 0;
    size_t end = 0;
};

enum class kind { object, array, string, primitive };

struct string_unit {
    size_t text_begin = 0;
    size_t text_end = 0;
    range source;
};

struct node {
    kind type = kind::primitive;
    range source;
    // Strings contain decoded UTF-8. Primitives contain their original spelling.
    std::string text;
    // Object children are values, in the same order as keys. Arrays have no keys.
    std::vector<node> children;
    std::vector<std::string> keys;
    // Each unit is one Unicode code point, including a complete surrogate pair.
    std::vector<string_unit> units;

    const node * find(const std::string & key) const;
};

// Reject invalid JSON, invalid Unicode, duplicate object keys, and more than
// 128 nested containers. Errors use std::invalid_argument.
node parse(const std::string & raw);

// Map decoded UTF-8 byte offsets to the original JSON spelling. Both offsets
// must be code point boundaries. The result excludes the string's quotes.
range map_string_range(const node & value, size_t decoded_begin, size_t decoded_end);

// Delete the union of the ranges. Reject invalid ranges before changing bytes.
// This function preserves source bytes; it does not validate the resulting JSON.
std::string erase_ranges(const std::string & raw, std::vector<range> ranges);

} // namespace server_compress_json
