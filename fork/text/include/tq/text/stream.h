#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tq::text {

namespace detail {

// Remaining bytes a destination can accept. The string limit is its
// max_size() minus what it already holds; the cursor limit is the room left
// in the 64-bit output counter. The smaller one wins.
inline std::size_t output_room(std::size_t used, std::size_t max_size, std::uint64_t emitted) {
    const std::size_t string_room = used <= max_size ? max_size - used : 0;
    const std::uint64_t cursor = std::numeric_limits<std::uint64_t>::max() - emitted;
    const std::size_t cursor_room = cursor > std::numeric_limits<std::size_t>::max()
        ? std::numeric_limits<std::size_t>::max()
        : static_cast<std::size_t>(cursor);
    return std::min(string_room, cursor_room);
}

// Geometric capacity target, clamped to the destination's max_size().
inline std::size_t growth_target(std::size_t need, std::size_t capacity, std::size_t max_size) {
    std::size_t target = need;
    if (capacity <= std::numeric_limits<std::size_t>::max() / 2) {
        target = std::max(target, capacity * 2);
    }
    return std::min(target, max_size);
}

} // namespace detail

struct stream_delta {
    std::uint64_t offset = 0;
    std::string text;
    bool finished = false;
    std::optional<std::size_t> stop_index;
};

struct stream_event {
    std::uint64_t offset = 0;
    bool finished = false;
    std::optional<std::size_t> stop_index;
};

class generation_stream {
public:
    static constexpr std::size_t max_stops = 32;
    static constexpr std::size_t max_stop_bytes = 4096;
    static constexpr std::size_t max_total_stop_bytes = 65536;

    explicit generation_stream(const std::vector<std::string> & stops = {});
    generation_stream(const generation_stream &) = delete;
    generation_stream & operator=(const generation_stream &) = delete;
    generation_stream(generation_stream && other) noexcept;
    generation_stream & operator=(generation_stream && other) noexcept;

    // Each result owns valid UTF-8 text. Its offset counts prior output bytes.
    // A stop ends the stream and discards the rest of this input chunk.
    stream_delta push(std::string_view bytes);
    // Flush unmatched text and incomplete UTF-8. Repeated finish emits no text.
    stream_delta finish();
    // Same as push(), but append the text to out and return no text.
    // bytes must not refer into out.
    stream_event push_into(std::string_view bytes, std::string & out);
    // Same as finish(), but append the text to out and return no text.
    stream_event finish_into(std::string & out);
    bool finished() const noexcept;

private:
    struct pattern {
        std::string bytes;
        std::vector<std::size_t> prefix;
        std::size_t matched = 0;
    };

    stream_event prepare_event(std::size_t incoming, std::string & out) const;
    stream_event complete_event(stream_event event, std::size_t appended);
    void consume_byte(unsigned char byte, std::string & output);
    void consume_text(std::string_view text, std::string & output);
    void emit_pending(std::size_t count, std::string & output);

    std::vector<pattern> patterns_;
    std::vector<char> pending_;
    std::size_t pending_head_ = 0;
    std::size_t pending_size_ = 0;
    std::array<unsigned char, 4> utf8_{};
    std::size_t utf8_size_ = 0;
    std::uint64_t emitted_ = 0;
    bool finished_ = false;
    std::optional<std::size_t> stop_index_;
};

} // namespace tq::text
