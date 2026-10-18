#include <tq/text/stream.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using tq::text::generation_stream;
using tq::text::stream_delta;

namespace {

std::size_t checked_runs = 0;

void check(bool condition, const std::string & message) {
    if (!condition) { throw std::runtime_error(message); }
}

template<class Exception, class Function>
void rejects(Function action) {
    bool rejected = false;
    try { action(); } catch (const Exception &) { rejected = true; }
    check(rejected, "operation did not reject the input");
}

std::string normalize(std::string_view input) {
    std::string output;
    for (std::size_t i = 0; i < input.size();) {
        const auto lead = static_cast<unsigned char>(input[i]);
        if (lead < 128) { output.push_back(static_cast<char>(lead)); ++i; continue; }
        const auto count = lead >= 0xc2 && lead <= 0xdf ? 2u : lead >= 0xe0 && lead <= 0xef ? 3u : lead >= 0xf0 && lead <= 0xf4 ? 4u : 0u;
        std::uint32_t scalar = lead & (count == 2 ? 31u : count == 3 ? 15u : 7u);
        bool valid = count && count <= input.size() - i;
        for (unsigned j = 1; valid && j < count; ++j) {
            const auto byte = static_cast<unsigned char>(input[i + j]);
            valid = (byte & 0xc0) == 0x80;
            scalar = (scalar << 6) | (byte & 63u);
        }
        valid = valid && scalar >= (count == 2 ? 0x80u : count == 3 ? 0x800u : 0x10000u) &&
                scalar <= 0x10ffff && !(scalar >= 0xd800 && scalar <= 0xdfff);
        if (valid) { output.append(input.substr(i, count)); i += count; }
        else { output += "\xef\xbf\xbd"; ++i; }
    }
    return output;
}

struct reference_result {
    std::string text;
    std::optional<std::size_t> stop_index;
};

reference_result reference(std::string_view input, const std::vector<std::string> & stops) {
    const auto text = normalize(input);
    for (std::size_t end = 1; end <= text.size(); ++end) {
        std::optional<std::size_t> selected;
        std::size_t length = 0;
        for (std::size_t i = 0; i < stops.size(); ++i) {
            if (stops[i].size() <= end && stops[i].size() > length &&
                text.compare(end - stops[i].size(), stops[i].size(), stops[i]) == 0) {
                selected = i;
                length = stops[i].size();
            }
        }
        if (selected) { return {text.substr(0, end - length), selected}; }
    }
    return {text, std::nullopt};
}

void run_chunks(const std::string & input, const std::vector<std::string> & stops, const std::vector<std::size_t> & ends) {
    generation_stream stream(stops);
    const auto expected = reference(input, stops);
    std::string output;
    std::optional<std::size_t> matched;
    const auto consume = [&](const stream_delta & delta) {
        check(delta.offset == output.size(), "output cursor lost or repeated bytes");
        check(normalize(delta.text) == delta.text, "delta ends with invalid UTF-8");
        output += delta.text;
        check(output.size() <= expected.text.size() && expected.text.compare(0, output.size(), output) == 0,
              "a byte escaped before the stop decision");
        check(delta.finished == stream.finished(), "result lost terminal state");
        if (delta.stop_index) { check(delta.finished, "matched stop did not finish the stream"); }
        matched = delta.stop_index;
    };
    consume(stream.push({}));
    std::size_t first = 0;
    for (const auto end : ends) {
        check(first <= end && end <= input.size(), "invalid test partition");
        if (!stream.finished()) {
            auto fragment = input.substr(first, end - first);
            const auto delta = stream.push(fragment);
            fragment.assign(fragment.size(), 'x');
            consume(delta);
        }
        first = end;
    }
    check(first == input.size(), "test did not submit its complete input");
    consume(stream.finish());
    check(output == expected.text && matched == expected.stop_index, "stream differs from the complete-input oracle");
    const auto again = stream.finish();
    check(again.finished && again.text.empty() && again.offset == output.size() && again.stop_index == matched,
          "repeated finish changed the result");
    rejects<std::logic_error>([&] { stream.push("late"); });
    rejects<std::logic_error>([&] { stream.push({}); });
    ++checked_runs;
}

void all_partitions(const std::string & input, const std::vector<std::string> & stops = {}) {
    check(input.size() <= 12, "exhaustive fixture is too large");
    const auto variants = input.empty() ? 1u : 1u << (input.size() - 1);
    for (unsigned mask = 0; mask < variants; ++mask) {
        std::vector<std::size_t> ends;
        for (std::size_t i = 1; i < input.size(); ++i) {
            if (mask & (1u << (i - 1))) { ends.push_back(i); }
        }
        ends.push_back(input.size());
        run_chunks(input, stops, ends);
    }
}

std::vector<std::size_t> fixed_chunks(std::size_t size, std::size_t width) {
    std::vector<std::size_t> ends;
    for (std::size_t end = 0; end < size;) {
        end += std::min(width, size - end);
        ends.push_back(end);
    }
    if (!size) { ends.push_back(0); }
    return ends;
}

void test_partitions() {
    all_partitions("");
    all_partitions("plain text");
    all_partitions(std::string("a\0STOPb", 7), {std::string("\0S", 2)});
    all_partitions("aba", {"a"});
    all_partitions("preENDtail", {"END"});
    all_partitions("xababaz", {"aba", "bab"});
    all_partitions("abc", {"abc", "b"});
    all_partitions("zabc", {"bc", "abc"});
    all_partitions("xEND", {"END", "END"});
    all_partitions("zabcd", {"ab", "abc"});
    all_partitions("hi<END", {"<END>"});
    all_partitions("Z\xe2\x82\xac" "ENDx", {"\xe2\x82\xac" "END"});
    all_partitions("X\xf0\x9f\x98\x80" "Y", {"\xf0\x9f\x98\x80"});
    all_partitions("x\xe2\x82");
    all_partitions("\xe0\xa0\x80" "!\xed\x9f\xbf");
    all_partitions("\xf0\x90\x80\x80\xf4\x8f\xbf\xbf");
    all_partitions("\xc0\x80\xe0\x80\x80");
    all_partitions("\xed\xa0\x80\xf4\x90\x80\x80");
    all_partitions("x\xe2\x82Z", {"\xef\xbf\xbd\xef\xbf\xbd"});
    all_partitions("x\xe2\x82Z", {"\xef\xbf\xbdZ"});
    all_partitions("x\xe2\x82", {"\xef\xbf\xbd"});
    all_partitions("\xff\xc1\xf5\x80\x80\x80");
}

std::uint32_t next(std::uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

void test_generated_cases() {
    const std::vector<unsigned char> alphabet = {
        'a', 'b', 'c', '<', '>', 0, 0xc2, 0x80, 0xe2, 0x82, 0xac, 0xed, 0xa0, 0xf0, 0x9f, 0x98, 0xff};
    const std::vector<std::string> candidates = {
        "ab", "aba", "bc", "b", "<>", std::string("\0", 1), "\xc2\x80", "\xe2\x82\xac", "\xef\xbf\xbd", "\xef\xbf\xbd" "a"};
    std::uint32_t seed = 0x1d038ab7;
    for (std::size_t trial = 0; trial < 300; ++trial) {
        std::string input;
        const auto count = next(seed) % 49;
        for (std::uint32_t i = 0; i < count; ++i) { input.push_back(static_cast<char>(alphabet[next(seed) % alphabet.size()])); }
        std::vector<std::string> stops;
        const auto patterns = next(seed) % 5;
        for (std::uint32_t i = 0; i < patterns; ++i) { stops.push_back(candidates[next(seed) % candidates.size()]); }
        run_chunks(input, stops, {input.size()});
        run_chunks(input, stops, fixed_chunks(input.size(), 1));
        run_chunks(input, stops, fixed_chunks(input.size(), 1 + next(seed) % 11));
    }
    std::string every_byte;
    for (unsigned value = 0; value < 256; ++value) { every_byte.push_back(static_cast<char>(value)); }
    run_chunks(every_byte, {}, {every_byte.size()});
    run_chunks(every_byte, {}, fixed_chunks(every_byte.size(), 1));
}

void test_limits_and_long_prefixes() {
    const std::string longest(generation_stream::max_stop_bytes, 'a');
    generation_stream at_count(std::vector<std::string>(generation_stream::max_stops, "x"));
    const auto same_end = at_count.push("x");
    check(same_end.finished && same_end.stop_index == 0 && same_end.text.empty(), "duplicate stops lost registration order");
    generation_stream at_total(std::vector<std::string>(16, longest));
    check(at_total.push("Z").text == "Z", "valid maximum configuration failed");
    rejects<std::invalid_argument>([&] { generation_stream too_many(std::vector<std::string>(generation_stream::max_stops + 1, "x")); });
    rejects<std::invalid_argument>([&] { generation_stream too_long({longest + 'a'}); });
    rejects<std::invalid_argument>([&] { generation_stream too_much(std::vector<std::string>(17, longest)); });
    for (const auto & value : std::vector<std::string>{"", "\x80", "\xc0\x80", "\xe2\x82", "\xed\xa0\x80", "\xf4\x90\x80\x80"}) {
        rejects<std::invalid_argument>([&] { generation_stream invalid({value}); });
    }
    const auto stop = std::string(generation_stream::max_stop_bytes - 1, 'a') + 'b';
    const auto adversarial = std::string(131072, 'a') + 'c';
    run_chunks(adversarial, {stop}, fixed_chunks(adversarial.size(), 257));
    const auto wrapped = "prefix" + std::string(8192, 'a') + "bignored";
    run_chunks(wrapped, {stop}, fixed_chunks(wrapped.size(), 31));
}

void test_ownership_and_move() {
    std::vector<std::string> stops = {"END"};
    generation_stream stream(stops);
    stops[0] = "NEW";
    stops.clear();
    std::string input = "safeE";
    auto first = stream.push(input);
    input.assign(input.size(), 'x');
    check(first.text == "safe" && first.offset == 0, "first output was incorrect");
    first.text = "changed externally";
    auto final = stream.push("NDignored");
    check(final.finished && final.stop_index == 0 && final.text.empty() && final.offset == 4,
          "input or result aliases changed the stream");
    check(first.text == "changed externally", "later input changed an owned result");

    generation_stream one({"STOP"});
    generation_stream two({"HALT"});
    check(one.push("oneST").text == "one", "first instance lost its prefix");
    check(two.push("twoHA").text == "two", "second instance lost its prefix");
    check(one.push("OP").finished && !two.finished(), "one instance ended the other");
    check(two.push("LT").finished, "second instance lost its stop state");

    generation_stream original({"E\xe2\x82\xac!"});
    check(original.push("startE\xe2").text == "start", "move fixture did not retain both prefixes");
    generation_stream moved(std::move(original));
    check(original.finished() && original.finish().text.empty(), "moved-from stream is live");
    rejects<std::logic_error>([&] { original.push("late"); });
    generation_stream destination({"unused"});
    destination.push("old state");
    destination = std::move(moved);
    check(moved.finished(), "move assignment left a second owner");
    auto * self = &destination;
    destination = std::move(*self);
    const auto done = destination.push("\x82\xac!tail");
    check(done.finished && done.stop_index == 0 && done.offset == 5 && done.text.empty(),
          "move lost text, UTF-8, stop, or cursor state");
}

} // namespace

static_assert(!std::is_copy_constructible<generation_stream>::value, "streams have one owner");
static_assert(!std::is_copy_assignable<generation_stream>::value, "streams have one owner");
static_assert(std::is_nothrow_move_constructible<generation_stream>::value, "moving a stream transfers ownership");
static_assert(std::is_nothrow_move_assignable<generation_stream>::value, "moving a stream transfers ownership");

int main() {
    try {
        test_partitions();
        test_generated_cases();
        test_limits_and_long_prefixes();
        test_ownership_and_move();
        std::cout << "{\"status\":\"pass\",\"partition_and_adversarial_runs\":" << checked_runs
                  << ",\"generated_inputs\":300,\"long_prefix_input_bytes\":131073}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "text stream test: " << error.what() << '\n';
        return 1;
    }
}
