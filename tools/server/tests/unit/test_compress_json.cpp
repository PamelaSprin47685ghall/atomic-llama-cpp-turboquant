#include "server-compress-json.h"

#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace server_compress_json;

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void rejects(const std::function<void()> & operation) {
    try {
        operation();
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error("expected std::invalid_argument");
}

static bool is_subsequence(const std::string & shorter, const std::string & original) {
    size_t next = 0;
    for (char byte : original) {
        if (next < shorter.size() && shorter[next] == byte) {
            ++next;
        }
    }
    return next == shorter.size();
}

static const node & member(const node & value, const char * key) {
    const auto * result = value.find(key);
    require(result != nullptr, "missing object member");
    return *result;
}

static void test_tree_and_source() {
    const std::string raw = " \r\n {\"a\" : [true, null, -12.50e+2, \"x\"], \"\\u0062\": {}} \t";
    const auto root = parse(raw);
    require(root.type == kind::object, "root object type");
    require(raw.substr(root.source.begin, root.source.end - root.source.begin) ==
            "{\"a\" : [true, null, -12.50e+2, \"x\"], \"\\u0062\": {}}", "root source range");
    require(root.keys == std::vector<std::string>({"a", "b"}), "decoded object keys");
    require(root.children.size() == 2, "object values");
    const auto & list = member(root, "a");
    require(list.type == kind::array && list.children.size() == 4, "array values");
    require(list.children[0].type == kind::primitive && list.children[0].text == "true", "boolean spelling");
    require(list.children[1].text == "null", "null spelling");
    require(list.children[2].text == "-12.50e+2", "number spelling");
    require(list.children[3].text == "x", "string text");
    require(member(root, "b").children.empty(), "empty object");
    require(root.find("absent") == nullptr && list.find("a") == nullptr, "missing member");
    require(parse("[ ]").children.empty(), "empty array");
    require(parse("1e9999999999999999999").text == "1e9999999999999999999", "no numeric conversion");
}

static void test_string_units() {
    const std::vector<std::string> spelling = {
        "A", R"(\")", R"(\\)", R"(\/)", R"(\b)", R"(\f)", R"(\n)", R"(\r)", R"(\t)",
        R"(\u0042)", R"(\u03b1)", R"(\uD83D\uDE00)", "é", R"(\u0000)", R"(\uDBFF\uDFFF)",
    };
    const std::vector<std::string> decoded = {
        "A", "\"", "\\", "/", "\b", "\f", "\n", "\r", "\t", "B", "α", "😀", "é",
        std::string(1, '\0'), "\xf4\x8f\xbf\xbf",
    };
    std::string raw = "\"";
    std::string text;
    for (size_t i = 0; i < spelling.size(); ++i) {
        raw += spelling[i];
        text += decoded[i];
    }
    raw += '"';
    const auto value = parse(raw);
    require(value.type == kind::string && value.text == text, "decoded string");
    require(value.units.size() == spelling.size(), "one unit per Unicode code point");
    size_t source_offset = 1;
    size_t text_offset = 0;
    for (size_t i = 0; i < spelling.size(); ++i) {
        const auto & unit = value.units[i];
        require(unit.text_begin == text_offset && unit.text_end == text_offset + decoded[i].size(), "decoded unit bounds");
        require(unit.source.begin == source_offset && unit.source.end == source_offset + spelling[i].size(), "source unit bounds");
        const auto span = map_string_range(value, unit.text_begin, unit.text_end);
        require(raw.substr(span.begin, span.end - span.begin) == spelling[i], "mapped unit spelling");
        const auto empty = map_string_range(value, text_offset, text_offset);
        require(empty.begin == source_offset && empty.end == source_offset, "empty mapped range");
        source_offset += spelling[i].size();
        text_offset += decoded[i].size();
    }
    const auto all = map_string_range(value, 0, value.text.size());
    require(all.begin == 1 && all.end == raw.size() - 1, "whole string excludes quotes");
    const auto end = map_string_range(value, text.size(), text.size());
    require(end.begin == raw.size() - 1 && end.end == end.begin, "end boundary");
    const auto empty = parse("\"\"");
    const auto empty_span = map_string_range(empty, 0, 0);
    require(empty_span.begin == 1 && empty_span.end == 1, "empty string mapping");

    const auto emoji = parse(R"("a\ud83d\ude00b")");
    rejects([&] { map_string_range(emoji, 2, 5); });
    rejects([&] { map_string_range(emoji, 1, 4); });
    rejects([&] { map_string_range(emoji, 4, 3); });
    rejects([&] { map_string_range(emoji, 0, 7); });
    rejects([&] { map_string_range(parse("[]"), 0, 0); });
}

static void test_nested_arguments_mapping() {
    const std::string raw = R"json( { "messages" : [{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"write","arguments":"{ \"path\": \"x\", \"content\" : \"A\\n\\uD83D\\uDE00\\\"B\\\\C\", \"mode\": 1 }"}}]}], "stream":false } )json";
    const auto root = parse(raw);
    const auto & message = member(root, "messages").children.at(0);
    const auto & call = member(message, "tool_calls").children.at(0);
    const auto & arguments = member(member(call, "function"), "arguments");
    const auto inner = parse(arguments.text);
    const auto & content = member(inner, "content");
    require(content.text == "A\n😀\"B\\C", "nested decoded argument");

    // Remove the newline, emoji, and quote from the inner value. Map the raw
    // inner JSON interval through the outer JSON string without serializing it.
    const auto inner_span = map_string_range(content, 1, 7);
    const auto outer_span = map_string_range(arguments, inner_span.begin, inner_span.end);
    const auto result = erase_ranges(raw, {outer_span});
    require(is_subsequence(result, raw), "HTTP body byte subsequence");
    const auto result_root = parse(result);
    const auto & result_call = member(member(result_root, "messages").children.at(0), "tool_calls").children.at(0);
    const auto & result_arguments = member(member(result_call, "function"), "arguments");
    const auto result_inner = parse(result_arguments.text);
    require(member(result_inner, "content").text == "AB\\C", "nested deletion keeps valid JSON escapes");
    require(member(result_inner, "path").text == "x" && member(result_inner, "mode").text == "1", "other arguments unchanged");
    require(is_subsequence(result_arguments.text, arguments.text), "inner JSON spelling subsequence");
    require(result.substr(0, outer_span.begin) == raw.substr(0, outer_span.begin), "prefix byte identity");
    require(result.substr(outer_span.begin) == raw.substr(outer_span.end), "suffix byte identity");
}

static void test_erase_ranges() {
    const std::string raw = "0123456789";
    require(erase_ranges(raw, {{7, 9}, {2, 4}, {3, 6}, {6, 7}, {4, 4}, {2, 4}}) == "019", "merge overlapping and adjacent ranges");
    require(erase_ranges(raw, {}) == raw, "empty deletion");
    require(erase_ranges(raw, {{0, 0}, {10, 10}}) == raw, "empty boundary ranges");
    require(erase_ranges(raw, {{0, 10}}).empty(), "whole deletion");
    require(erase_ranges("", {{0, 0}}).empty(), "empty source");
    rejects([&] { erase_ranges(raw, {{3, 2}}); });
    rejects([&] { erase_ranges(raw, {{0, 11}}); });
    rejects([&] { erase_ranges(raw, {{11, 11}}); });
    rejects([&] { erase_ranges(raw, {{0, std::numeric_limits<size_t>::max()}}); });
}

static void check_marker_provenance(const deletion_result & result, const std::string & raw) {
    size_t previous_end = 0;
    for (const auto & marker : result.markers) {
        require(marker.begin >= previous_end && marker.end <= result.text.size(), "ordered marker provenance");
        require(result.text.substr(marker.begin, marker.end - marker.begin) == "␡", "inserted marker spelling");
        previous_end = marker.end;
    }
    require(is_subsequence(erase_ranges(result.text, result.markers), raw),
            "removing only inserted markers leaves original source bytes");
}

static void test_marked_deletions() {
    const std::string raw = R"json( { "text" : "ABCDEFGHIJ", "untouched":"\u2421" } )json";
    const auto root = parse(raw);
    const auto & value = member(root, "text");
    const auto result = apply_deletions(raw, root, {}, {
        map_string_range(value, 6, 8), map_string_range(value, 2, 4),
        map_string_range(value, 3, 6), map_string_range(value, 2, 4),
    });
    require(result.text == R"json( { "text" : "AB␡IJ", "untouched":"\u2421" } )json",
            "overlapping and adjacent string deletions use one marker");
    require(result.markers.size() == 1, "one continuous deletion has one inserted marker");
    check_marker_provenance(result, raw);

    const auto unchanged = apply_deletions(raw, root, {}, {{0, 0}, {raw.size(), raw.size()}});
    require(unchanged.text == raw && unchanged.markers.empty(), "empty edits keep exact bytes");
    const auto whole = apply_deletions(raw, root, {}, {map_string_range(value, 0, value.text.size())});
    require(member(parse(whole.text), "text").text == "␡", "whole string deletion retains a marker value");
    check_marker_provenance(whole, raw);
}

static void test_marker_collision() {
    const std::string raw = R"json({"text":"left␡right","escaped":"\u2421"})json";
    const auto root = parse(raw);
    const auto & value = member(root, "text");
    const auto result = apply_deletions(raw, root, {}, {
        map_string_range(value, 0, 4), map_string_range(value, 7, value.text.size()),
    });
    require(member(parse(result.text), "text").text == "␡␡␡", "literal user marker remains between inserted markers");
    require(result.markers.size() == 2, "provenance excludes the original marker");
    const auto restored = erase_ranges(result.text, result.markers);
    require(restored == R"json({"text":"␡","escaped":"\u2421"})json",
            "provenance retains literal and escaped user markers");
    check_marker_provenance(result, raw);

    const std::string escaped_raw = R"json({"text":"left\u2421right"})json";
    const auto escaped_root = parse(escaped_raw);
    const auto & escaped_value = member(escaped_root, "text");
    const auto escaped_result = apply_deletions(escaped_raw, escaped_root, {}, {
        map_string_range(escaped_value, 0, 4), map_string_range(escaped_value, 7, escaped_value.text.size()),
    });
    require(escaped_result.text == R"json({"text":"␡\u2421␡"})json",
            "adjacent markers do not change the user's original escape spelling");
    require(escaped_result.markers.size() == 2 &&
            erase_ranges(escaped_result.text, escaped_result.markers) == R"json({"text":"\u2421"})json",
            "provenance excludes an escaped user marker in the same string");
    check_marker_provenance(escaped_result, escaped_raw);
}

static void test_nested_marked_deletions() {
    const std::string raw = R"json({"arguments":"{ \"path\":\"src/x.cpp\", \"content\":\"keep ␡ A\\n\\uD83D\\uDE00\\\" drop tail\", \"mode\":1 }"})json";
    const auto root = parse(raw);
    const auto & outer = member(root, "arguments");
    const auto inner = parse(outer.text);
    const auto & value = member(inner, "content");
    const size_t begin = value.text.find("A\n");
    const size_t end = value.text.find(" tail");
    require(begin != std::string::npos && end != std::string::npos, "nested test boundaries");
    const auto inner_span = map_string_range(value, begin, end);
    const auto span = map_string_range(outer, inner_span.begin, inner_span.end);
    const auto result = apply_deletions(raw, root, {}, {span});
    const auto final_root = parse(result.text);
    const auto final_inner = parse(member(final_root, "arguments").text);
    require(member(final_inner, "content").text == "keep ␡ ␡ tail", "nested marker keeps both JSON layers valid");
    require(member(final_inner, "path").text == "src/x.cpp" && member(final_inner, "mode").text == "1",
            "nested marker leaves other arguments unchanged");
    require(result.markers.size() == 1, "one nested marker has one outer provenance span");
    check_marker_provenance(result, raw);
}

static void test_structural_and_marked_deletions() {
    const std::string raw = R"json({"messages":[{"content":"old text"},{"content":"keep more tail"}]})json";
    const auto root = parse(raw);
    const auto & messages = member(root, "messages").children;
    const auto & old_text = member(messages[0], "content");
    const auto & new_text = member(messages[1], "content");
    const range old_message = {messages[0].source.begin, messages[1].source.begin};
    const auto result = apply_deletions(raw, root, {old_message}, {
        map_string_range(old_text, 0, old_text.text.size()), map_string_range(new_text, 5, 9),
    });
    require(result.text == R"json({"messages":[{"content":"keep ␡ tail"}]})json",
            "structural deletion does not leave a marker outside a string");
    require(result.markers.size() == 1, "structural deletion suppresses covered markers");
    check_marker_provenance(result, raw);
    const auto structural_only = apply_deletions(raw, root, {old_message}, {});
    require(structural_only.text == erase_ranges(raw, {old_message}) && structural_only.markers.empty(),
            "structural deletions remain pure erasure");
    const auto span = map_string_range(new_text, 5, 9);
    rejects([&] { apply_deletions(raw, root, {{span.begin + 1, span.end}}, {span}); });
    rejects([&] { apply_deletions(raw, root, {{span.begin, span.end - 1}}, {span}); });
}

static void test_marked_range_validation() {
    const std::string raw = R"json({"text":"a\uD83D\uDE00éz","other":"tail","number":17})json";
    const auto root = parse(raw);
    const auto & value = member(root, "text");
    const auto & other = member(root, "other");
    const auto & number = member(root, "number");
    const auto adjacent = apply_deletions(raw, root, {}, {value.units[2].source, value.units[1].source});
    require(member(parse(adjacent.text), "text").text == "a␡z" && adjacent.markers.size() == 1,
            "adjacent UTF-8 and surrogate-pair units merge into one deletion");
    check_marker_provenance(adjacent, raw);
    rejects([&] { apply_deletions(raw, root, {}, {value.source}); });
    rejects([&] { apply_deletions(raw, root, {}, {{2, 5}}); }); // object key
    rejects([&] { apply_deletions(raw, root, {}, {number.source}); });
    rejects([&] { apply_deletions(raw, root, {}, {{value.units[0].source.begin, other.units[0].source.end}}); });
    rejects([&] { apply_deletions(raw, root, {}, {{value.units[1].source.begin, value.units[1].source.end - 1}}); });
    rejects([&] { apply_deletions(raw, root, {}, {{value.units[1].source.begin + 6, value.units[1].source.end}}); });
    rejects([&] { apply_deletions(raw, root, {}, {
        {value.units[1].source.begin, value.units[1].source.begin + 6},
        {value.units[1].source.begin + 6, value.units[1].source.end},
    }); });
    rejects([&] { apply_deletions(raw, root, {}, {{value.units[2].source.begin + 1, value.units[2].source.end}}); });
    rejects([&] { apply_deletions(raw, root, {}, {{3, 2}}); });
    rejects([&] { apply_deletions(raw, root, {}, {{0, raw.size() + 1}}); });
    rejects([&] { apply_deletions(raw, root, {{0, raw.size() + 1}}, {}); });
}

static void test_invalid_json() {
    const std::vector<std::string> invalid = {
        "", " ", "{}{}", "[] true", "nul", "TRUE", "undefined", "NaN", "Infinity",
        "+1", "01", "-01", "-", ".5", "1.", "1e", "1e+", "1e-", "--1", "0x10",
        "[", "{", "[1,]", "[,1]", "[1 2]", "{a:1}", "{\"a\"}", "{\"a\":}", "{\"a\":1,}",
        R"({"a":1,"a":2})", R"({"a":1,"\u0061":2})", R"({"\u0000":1,"\u0000":2})",
        "\"", R"("\x")", R"("\u123")", R"("\u12xz")", R"("\ud800")", R"("\udfff")",
        R"("\ud800x")", R"("\ud800\u0041")", R"("\ud800\ud800")", std::string("\"line\nline\""),
        std::string("\"a") + char(0) + "b\"", std::string("\"") + char(0x1f) + "\"",
    };
    for (const auto & raw : invalid) {
        rejects([&] { parse(raw); });
    }
    const std::vector<std::string> invalid_utf8 = {
        "\x80", "\xc0\x80", "\xc1\xbf", "\xc2", "\xc2\x20", "\xe0\x80\x80", "\xed\xa0\x80",
        "\xf0\x80\x80\x80", "\xf4\x90\x80\x80", "\xf5\x80\x80\x80", "\xff", "\xef\xbf",
    };
    for (const auto & bytes : invalid_utf8) {
        rejects([&] { parse("\"" + bytes + "\""); });
    }
    require(parse(R"({"a":1,"A":2})").children.size() == 2, "distinct object keys");
    require(parse(R"({"a":{"a":1}})").children.size() == 1, "keys are scoped to each object");
}

static void test_depth_bound() {
    const std::string allowed = std::string(128, '[') + "0" + std::string(128, ']');
    require(parse(allowed).type == kind::array, "supported depth");
    rejects([&] { parse(std::string(129, '[') + "0" + std::string(129, ']')); });
    rejects([&] { parse(std::string(129, '[') + std::string(129, ']')); });
}

int main() {
    try {
        test_tree_and_source();
        test_string_units();
        test_nested_arguments_mapping();
        test_erase_ranges();
        test_marked_deletions();
        test_marker_collision();
        test_nested_marked_deletions();
        test_structural_and_marked_deletions();
        test_marked_range_validation();
        test_invalid_json();
        test_depth_bound();
        std::cout << "compression JSON tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "compression JSON test failed: " << error.what() << '\n';
        return 1;
    }
}
