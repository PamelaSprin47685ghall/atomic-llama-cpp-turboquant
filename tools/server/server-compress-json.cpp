#include "server-compress-json.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace server_compress_json {
namespace {

constexpr size_t max_container_depth = 128;

class parser {
public:
    explicit parser(const std::string & raw) : raw(raw) {}

    node run() {
        skip_space();
        auto result = value(0);
        skip_space();
        if (pos != raw.size()) {
            fail("unexpected data after JSON value");
        }
        return result;
    }

private:
    const std::string & raw;
    size_t pos = 0;

    [[noreturn]] void fail(const char * message) const {
        throw std::invalid_argument(std::string(message) + " at byte " + std::to_string(pos));
    }

    void skip_space() {
        while (pos < raw.size()) {
            const char byte = raw[pos];
            if (byte != ' ' && byte != '\t' && byte != '\r' && byte != '\n') {
                break;
            }
            ++pos;
        }
    }

    bool take(char byte) {
        if (pos < raw.size() && raw[pos] == byte) {
            ++pos;
            return true;
        }
        return false;
    }

    void expect(char byte) {
        if (!take(byte)) {
            fail("unexpected JSON character");
        }
    }

    static bool digit(char byte) {
        return byte >= '0' && byte <= '9';
    }

    uint32_t hex4() {
        uint32_t result = 0;
        for (size_t i = 0; i < 4; ++i) {
            if (pos == raw.size()) {
                fail("incomplete Unicode escape");
            }
            const char byte = raw[pos++];
            result <<= 4;
            if (byte >= '0' && byte <= '9') {
                result += uint32_t(byte - '0');
            } else if (byte >= 'a' && byte <= 'f') {
                result += uint32_t(byte - 'a' + 10);
            } else if (byte >= 'A' && byte <= 'F') {
                result += uint32_t(byte - 'A' + 10);
            } else {
                fail("invalid Unicode escape");
            }
        }
        return result;
    }

    static void append_utf8(std::string & text, uint32_t codepoint) {
        if (codepoint <= 0x7f) {
            text += char(codepoint);
        } else if (codepoint <= 0x7ff) {
            text += char(0xc0 | (codepoint >> 6));
            text += char(0x80 | (codepoint & 0x3f));
        } else if (codepoint <= 0xffff) {
            text += char(0xe0 | (codepoint >> 12));
            text += char(0x80 | ((codepoint >> 6) & 0x3f));
            text += char(0x80 | (codepoint & 0x3f));
        } else {
            text += char(0xf0 | (codepoint >> 18));
            text += char(0x80 | ((codepoint >> 12) & 0x3f));
            text += char(0x80 | ((codepoint >> 6) & 0x3f));
            text += char(0x80 | (codepoint & 0x3f));
        }
    }

    void escaped_codepoint(std::string & text) {
        if (pos == raw.size()) {
            fail("incomplete string escape");
        }
        const char escape = raw[pos++];
        switch (escape) {
            case '"': text += '"';  return;
            case '\\': text += '\\'; return;
            case '/': text += '/';  return;
            case 'b': text += '\b'; return;
            case 'f': text += '\f'; return;
            case 'n': text += '\n'; return;
            case 'r': text += '\r'; return;
            case 't': text += '\t'; return;
            case 'u': break;
            default: fail("invalid string escape");
        }
        uint32_t codepoint = hex4();
        if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
            if (!take('\\') || !take('u')) {
                fail("missing low Unicode surrogate");
            }
            const uint32_t low = hex4();
            if (low < 0xdc00 || low > 0xdfff) {
                fail("invalid low Unicode surrogate");
            }
            codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
        } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
            fail("unpaired low Unicode surrogate");
        }
        append_utf8(text, codepoint);
    }

    void literal_codepoint(std::string & text) {
        const size_t begin = pos;
        const auto lead = static_cast<unsigned char>(raw[pos++]);
        size_t continuation = 0;
        uint32_t codepoint = 0;
        uint32_t minimum = 0;
        if (lead < 0x20) {
            fail("unescaped control character");
        } else if (lead <= 0x7f) {
            text += char(lead);
            return;
        } else if (lead >= 0xc2 && lead <= 0xdf) {
            continuation = 1;
            codepoint = lead & 0x1f;
            minimum = 0x80;
        } else if (lead >= 0xe0 && lead <= 0xef) {
            continuation = 2;
            codepoint = lead & 0x0f;
            minimum = 0x800;
        } else if (lead >= 0xf0 && lead <= 0xf4) {
            continuation = 3;
            codepoint = lead & 0x07;
            minimum = 0x10000;
        } else {
            fail("invalid UTF-8 lead byte");
        }
        for (size_t i = 0; i < continuation; ++i) {
            if (pos == raw.size()) {
                fail("incomplete UTF-8 code point");
            }
            const auto byte = static_cast<unsigned char>(raw[pos++]);
            if ((byte & 0xc0) != 0x80) {
                fail("invalid UTF-8 continuation byte");
            }
            codepoint = (codepoint << 6) | (byte & 0x3f);
        }
        if (codepoint < minimum || codepoint > 0x10ffff || (codepoint >= 0xd800 && codepoint <= 0xdfff)) {
            fail("invalid Unicode code point");
        }
        text.append(raw, begin, pos - begin);
    }

    node string() {
        node result;
        result.type = kind::string;
        result.source.begin = pos;
        expect('"');
        while (pos < raw.size() && raw[pos] != '"') {
            const size_t source_begin = pos;
            const size_t text_begin = result.text.size();
            if (take('\\')) {
                escaped_codepoint(result.text);
            } else {
                literal_codepoint(result.text);
            }
            result.units.push_back({text_begin, result.text.size(), {source_begin, pos}});
        }
        expect('"');
        result.source.end = pos;
        return result;
    }

    node container(size_t depth, bool object) {
        if (depth >= max_container_depth) {
            fail("JSON nesting exceeds 128 containers");
        }
        node result;
        result.type = object ? kind::object : kind::array;
        result.source.begin = pos++;
        const char close = object ? '}' : ']';
        skip_space();
        std::unordered_set<std::string> seen_keys;
        if (!take(close)) {
            for (;;) {
                if (object) {
                    auto key = string();
                    if (!seen_keys.insert(key.text).second) {
                        fail("duplicate object key");
                    }
                    result.keys.push_back(std::move(key.text));
                    skip_space();
                    expect(':');
                    skip_space();
                }
                result.children.push_back(value(depth + 1));
                skip_space();
                if (take(close)) {
                    break;
                }
                expect(',');
                skip_space();
            }
        }
        result.source.end = pos;
        return result;
    }

    node primitive() {
        node result;
        result.source.begin = pos;
        if (raw[pos] == 't' || raw[pos] == 'f' || raw[pos] == 'n') {
            const char * literal = raw[pos] == 't' ? "true" : raw[pos] == 'f' ? "false" : "null";
            for (; *literal; ++literal) {
                expect(*literal);
            }
        } else {
            take('-');
            if (!take('0')) {
                if (pos == raw.size() || raw[pos] < '1' || raw[pos] > '9') {
                    fail("invalid JSON value");
                }
                do {
                    ++pos;
                } while (pos < raw.size() && digit(raw[pos]));
            }
            if (take('.')) {
                if (pos == raw.size() || !digit(raw[pos])) {
                    fail("missing fractional digits");
                }
                while (pos < raw.size() && digit(raw[pos])) {
                    ++pos;
                }
            }
            if (take('e') || take('E')) {
                if (!take('+')) {
                    take('-');
                }
                if (pos == raw.size() || !digit(raw[pos])) {
                    fail("missing exponent digits");
                }
                while (pos < raw.size() && digit(raw[pos])) {
                    ++pos;
                }
            }
        }
        result.source.end = pos;
        result.text = raw.substr(result.source.begin, pos - result.source.begin);
        return result;
    }

    node value(size_t depth) {
        if (pos == raw.size()) {
            fail("missing JSON value");
        }
        switch (raw[pos]) {
            case '{': return container(depth, true);
            case '[': return container(depth, false);
            case '"': return string();
            default: return primitive();
        }
    }
};

size_t source_boundary(const node & value, size_t offset) {
    if (offset == value.text.size()) {
        return value.source.end - 1;
    }
    const auto unit = std::lower_bound(value.units.begin(), value.units.end(), offset,
                                      [](const string_unit & item, size_t position) {
                                          return item.text_begin < position;
                                      });
    if (unit == value.units.end() || unit->text_begin != offset) {
        throw std::invalid_argument("string range must use Unicode code point boundaries");
    }
    return unit->source.begin;
}

} // namespace

const node * node::find(const std::string & key) const {
    if (type == kind::object) {
        for (size_t i = 0; i < keys.size(); ++i) {
            if (keys[i] == key) {
                return &children.at(i);
            }
        }
    }
    return nullptr;
}

node parse(const std::string & raw) {
    return parser(raw).run();
}

range map_string_range(const node & value, size_t decoded_begin, size_t decoded_end) {
    if (value.type != kind::string || decoded_begin > decoded_end || decoded_end > value.text.size() ||
        value.source.end < value.source.begin || value.source.end - value.source.begin < 2) {
        throw std::invalid_argument("invalid string range");
    }
    return {source_boundary(value, decoded_begin), source_boundary(value, decoded_end)};
}

std::string erase_ranges(const std::string & raw, std::vector<range> ranges) {
    for (const auto & span : ranges) {
        if (span.begin > span.end || span.end > raw.size()) {
            throw std::invalid_argument("deletion range is outside the JSON source");
        }
    }
    std::sort(ranges.begin(), ranges.end(), [](const range & left, const range & right) {
        return left.begin < right.begin || (left.begin == right.begin && left.end < right.end);
    });
    std::string result;
    result.reserve(raw.size());
    size_t next = 0;
    for (const auto & span : ranges) {
        if (span.begin > next) {
            result.append(raw, next, span.begin - next);
        }
        next = std::max(next, span.end);
    }
    result.append(raw, next, raw.size() - next);
    return result;
}

} // namespace server_compress_json
