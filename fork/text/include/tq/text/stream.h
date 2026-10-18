#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tq::text {

struct stream_delta {
    std::uint64_t offset = 0;
    std::string text;
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
    bool finished() const noexcept;

private:
    struct pattern {
        std::string bytes;
        std::vector<std::size_t> prefix;
        std::size_t matched = 0;
    };

    stream_delta prepare_delta(std::size_t incoming) const;
    stream_delta complete_delta(stream_delta delta);
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
