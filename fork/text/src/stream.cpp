#include <tq/text/stream.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tq::text {
namespace {

constexpr char replacement[] = "\xef\xbf\xbd";

std::size_t sequence_size(unsigned char byte) {
    if (byte < 0x80) { return 1; }
    if (byte >= 0xc2 && byte <= 0xdf) { return 2; }
    if (byte >= 0xe0 && byte <= 0xef) { return 3; }
    if (byte >= 0xf0 && byte <= 0xf4) { return 4; }
    return 0;
}

bool continuation(unsigned char lead, std::size_t index, unsigned char byte) {
    if (byte < 0x80 || byte > 0xbf) { return false; }
    if (index != 1) { return true; }
    if (lead == 0xe0) { return byte >= 0xa0; }
    if (lead == 0xed) { return byte <= 0x9f; }
    if (lead == 0xf0) { return byte >= 0x90; }
    if (lead == 0xf4) { return byte <= 0x8f; }
    return true;
}

bool valid_utf8(std::string_view text) {
    for (std::size_t first = 0; first < text.size();) {
        const auto lead = static_cast<unsigned char>(text[first]);
        const auto count = sequence_size(lead);
        if (!count || count > text.size() - first) { return false; }
        for (std::size_t i = 1; i < count; ++i) {
            if (!continuation(lead, i, static_cast<unsigned char>(text[first + i]))) { return false; }
        }
        first += count;
    }
    return true;
}

} // namespace

generation_stream::generation_stream(const std::vector<std::string> & stops) {
    if (stops.size() > max_stops) { throw std::invalid_argument("too many stop strings"); }
    std::size_t total = 0;
    std::size_t longest = 0;
    for (const auto & stop : stops) {
        if (stop.empty() || stop.size() > max_stop_bytes) { throw std::invalid_argument("stop string is empty or too large"); }
        if (stop.size() > max_total_stop_bytes - total) { throw std::invalid_argument("stop strings exceed the total byte limit"); }
        if (!valid_utf8(stop)) { throw std::invalid_argument("stop string is not valid UTF-8"); }
        total += stop.size();
        longest = std::max(longest, stop.size());
    }
    patterns_.reserve(stops.size());
    for (const auto & stop : stops) {
        pattern item;
        item.bytes = stop;
        item.prefix.resize(stop.size());
        for (std::size_t i = 1; i < stop.size(); ++i) {
            auto length = item.prefix[i - 1];
            while (length && stop[i] != stop[length]) { length = item.prefix[length - 1]; }
            if (stop[i] == stop[length]) { ++length; }
            item.prefix[i] = length;
        }
        patterns_.push_back(std::move(item));
    }
    pending_.resize(longest);
}

generation_stream::generation_stream(generation_stream && other) noexcept {
    *this = std::move(other);
}

generation_stream & generation_stream::operator=(generation_stream && other) noexcept {
    if (this != &other) {
        patterns_ = std::move(other.patterns_);
        pending_ = std::move(other.pending_);
        pending_head_ = std::exchange(other.pending_head_, 0);
        pending_size_ = std::exchange(other.pending_size_, 0);
        utf8_ = other.utf8_;
        utf8_size_ = std::exchange(other.utf8_size_, 0);
        emitted_ = std::exchange(other.emitted_, 0);
        finished_ = std::exchange(other.finished_, true);
        stop_index_ = std::exchange(other.stop_index_, std::nullopt);
    }
    return *this;
}

stream_delta generation_stream::prepare_delta(std::size_t incoming) const {
    stream_delta result;
    result.offset = emitted_;
    auto room = result.text.max_size();
    const auto cursor_room = std::numeric_limits<std::uint64_t>::max() - emitted_;
    if (cursor_room < room) { room = static_cast<std::size_t>(cursor_room); }
    if (pending_size_ > room || utf8_size_ > (room - pending_size_) / 3) {
        throw std::length_error("text output exceeds its byte counter");
    }
    const auto held = pending_size_ + 3 * utf8_size_;
    if (incoming > (room - held) / 3) { throw std::length_error("text message exceeds output capacity"); }
    // Reserve before changing state. One invalid input byte can produce three output bytes.
    result.text.reserve(held + 3 * incoming);
    return result;
}

stream_delta generation_stream::complete_delta(stream_delta result) {
    emitted_ += result.text.size();
    result.finished = finished_;
    result.stop_index = stop_index_;
    return result;
}

void generation_stream::emit_pending(std::size_t count, std::string & output) {
    if (!count) { return; }
    const auto first = std::min(count, pending_.size() - pending_head_);
    output.append(pending_.data() + pending_head_, first);
    output.append(pending_.data(), count - first);
    pending_head_ = (pending_head_ + count) % pending_.size();
    pending_size_ -= count;
}

void generation_stream::consume_text(std::string_view text, std::string & output) {
    if (patterns_.empty()) { output.append(text); return; }
    for (const auto raw : text) {
        pending_[(pending_head_ + pending_size_) % pending_.size()] = raw;
        ++pending_size_;
        std::size_t keep = 0;
        std::size_t stop_length = 0;
        std::optional<std::size_t> stop_index;
        for (std::size_t i = 0; i < patterns_.size(); ++i) {
            auto & item = patterns_[i];
            while (item.matched && item.bytes[item.matched] != raw) { item.matched = item.prefix[item.matched - 1]; }
            if (item.bytes[item.matched] == raw) { ++item.matched; }
            if (item.matched == item.bytes.size() && item.matched > stop_length) {
                stop_length = item.matched;
                stop_index = i;
            }
            keep = std::max(keep, item.matched);
        }
        if (stop_index) {
            emit_pending(pending_size_ - stop_length, output);
            pending_head_ = 0;
            pending_size_ = 0;
            finished_ = true;
            stop_index_ = stop_index;
            return;
        }
        emit_pending(pending_size_ - keep, output);
    }
}

void generation_stream::consume_byte(unsigned char byte, std::string & output) {
    if (utf8_size_) {
        if (continuation(utf8_[0], utf8_size_, byte)) {
            utf8_[utf8_size_++] = byte;
            if (utf8_size_ == sequence_size(utf8_[0])) {
                const auto count = utf8_size_;
                utf8_size_ = 0;
                consume_text({reinterpret_cast<const char *>(utf8_.data()), count}, output);
            }
            return;
        }
        const auto count = utf8_size_;
        utf8_size_ = 0;
        for (std::size_t i = 0; i < count && !finished_; ++i) { consume_text(replacement, output); }
        if (finished_) { return; }
    }
    const auto count = sequence_size(byte);
    if (count == 1) {
        const auto character = static_cast<char>(byte);
        consume_text({&character, 1}, output);
    } else if (count) {
        utf8_[0] = byte;
        utf8_size_ = 1;
    } else {
        consume_text(replacement, output);
    }
}

stream_delta generation_stream::push(std::string_view bytes) {
    if (finished_) { throw std::logic_error("text stream is finished"); }
    auto result = prepare_delta(bytes.size());
    for (const auto byte : bytes) {
        consume_byte(static_cast<unsigned char>(byte), result.text);
        if (finished_) { break; }
    }
    return complete_delta(std::move(result));
}

stream_delta generation_stream::finish() {
    if (finished_) {
        stream_delta result;
        result.offset = emitted_;
        result.finished = true;
        result.stop_index = stop_index_;
        return result;
    }
    auto result = prepare_delta(0);
    const auto count = utf8_size_;
    utf8_size_ = 0;
    for (std::size_t i = 0; i < count && !finished_; ++i) { consume_text(replacement, result.text); }
    if (!finished_) { emit_pending(pending_size_, result.text); }
    finished_ = true;
    return complete_delta(std::move(result));
}

bool generation_stream::finished() const noexcept {
    return finished_;
}

} // namespace tq::text
