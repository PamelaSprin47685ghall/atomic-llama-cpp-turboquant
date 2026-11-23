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

struct deletion_result {
    std::string text;
    // Output byte ranges for markers inserted by this operation only. Original
    // literal or escaped U+2421 characters are not included.
    std::vector<range> markers;
};

// Replace each continuous string deletion with UTF-8 U+2421 (the delete symbol).
// document must be the tree parsed from raw. String ranges must use complete
// code point/escape boundaries within a JSON string value, never a key.
// Map nested argument ranges through the outer string before calling this.
// Structural deletions are plain erasure and suppress markers they fully cover.
// Partial overlap between structural and string deletions is an error.
// Empty ranges do not insert markers. Errors use std::invalid_argument.
deletion_result apply_deletions(const std::string & raw, const node & document,
                               std::vector<range> structural, std::vector<range> string_ranges);

} // namespace server_compress_json
