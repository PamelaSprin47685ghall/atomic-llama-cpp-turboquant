#include <tq/calibration/reader.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace tq::calibration {
namespace {

std::uint64_t decode_u64(const unsigned char * bytes, unsigned count = 8) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < count; ++i) { value |= std::uint64_t(bytes[i]) << (8 * i); }
    return value;
}

bool little_endian() {
    const std::uint32_t value = 1;
    return *reinterpret_cast<const unsigned char *>(&value) == 1;
}

std::uint64_t add(std::uint64_t left, std::uint64_t right) {
    if (right > UINT64_MAX - left) { throw std::invalid_argument("calibration size overflows"); }
    return left + right;
}

std::uint64_t multiply(std::uint64_t left, std::uint64_t right) {
    if (right && left > UINT64_MAX / right) { throw std::invalid_argument("calibration size overflows"); }
    return left * right;
}

void check_zero_rows(const std::vector<std::uint64_t> & counts, const std::vector<double> & values, std::uint32_t width) {
    for (std::size_t expert = 0; expert < counts.size(); ++expert) {
        if (counts[expert]) { continue; }
        for (std::uint32_t column = 0; column < width; ++column) {
            if (values[expert * width + column] != 0.0) {
                throw std::invalid_argument("zero sample count has nonzero statistics");
            }
        }
    }
}

} // namespace

reader::reader(std::istream & input, read_limits limits) : input_(input), limits_(limits) {
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559, "WXQCAL03 requires IEEE binary64");
    if (!limits.max_layers || limits.max_layer_bytes < 40 || limits.max_file_bytes < 64) {
        throw std::invalid_argument("invalid calibration read limits");
    }
    unsigned char magic[8];
    read_bytes(magic, sizeof(magic));
    if (std::memcmp(magic, "WXQCAL03", 8) != 0) { throw std::invalid_argument("not a WXQCAL03 file"); }
    header_.layers = read_u32();
    header_.max_context = read_u32();
    header_.activation_stride = read_u32();
    header_.activation_tokens = read_u32();
    header_.route_stride = read_u32();
    header_.flags = read_u32();
    header_.decoded = read_u64();
    header_.total = read_u64();
    header_.context_resets = read_u64();
    unsigned char tail[8];
    read_bytes(tail, sizeof(tail));
    if (tail[0] > 1 || std::any_of(tail + 1, tail + 8, [](unsigned char value) { return value != 0; })) {
        throw std::invalid_argument("invalid complete flag or header padding");
    }
    header_.complete = tail[0] != 0;
    if (header_.layers > limits_.max_layers || (header_.flags & ~(COVARIANCE | HIDDEN | EXPERT_INPUT))) {
        throw std::invalid_argument("unsupported calibration flags or layer count");
    }
}

void reader::read_bytes(void * output, std::uint64_t count) {
    if (count > limits_.max_file_bytes - bytes_) { throw std::invalid_argument("calibration exceeds the file byte limit"); }
    auto * next = static_cast<char *>(output);
    while (count) {
        const auto chunk = static_cast<std::streamsize>(std::min<std::uint64_t>(count, 65536));
        input_.read(next, chunk);
        if (input_.gcount() != chunk || !input_) { throw std::invalid_argument("truncated or unreadable calibration at byte " + std::to_string(bytes_)); }
        next += chunk;
        bytes_ += static_cast<std::uint64_t>(chunk);
        count -= static_cast<std::uint64_t>(chunk);
    }
}

std::uint32_t reader::read_u32() {
    unsigned char bytes[4];
    read_bytes(bytes, sizeof(bytes));
    return static_cast<std::uint32_t>(decode_u64(bytes, 4));
}

std::uint64_t reader::read_u64() {
    unsigned char bytes[8];
    read_bytes(bytes, sizeof(bytes));
    return decode_u64(bytes);
}

std::vector<std::uint64_t> reader::read_counts(std::uint64_t count, std::uint64_t maximum) {
    std::vector<std::uint64_t> values(static_cast<std::size_t>(count));
    read_bytes(values.data(), multiply(count, 8));
    for (auto & value : values) {
        if (!little_endian()) { value = decode_u64(reinterpret_cast<const unsigned char *>(&value)); }
        if (value > maximum) { throw std::invalid_argument("expert count exceeds its observed positions"); }
    }
    return values;
}

std::vector<double> reader::read_values(std::uint64_t count, bool nonnegative) {
    std::vector<double> values(static_cast<std::size_t>(count));
    read_bytes(values.data(), multiply(count, 8));
    for (auto & value : values) {
        if (!little_endian()) {
            const auto bits = decode_u64(reinterpret_cast<const unsigned char *>(&value));
            std::memcpy(&value, &bits, sizeof(value));
        }
        if (!std::isfinite(value) || (nonnegative && value < 0)) { throw std::invalid_argument("invalid calibration moment"); }
    }
    return values;
}

void reader::read_covariance(std::uint32_t width, std::uint64_t positions) {
    const auto count = multiply(width, width);
    std::array<unsigned char, 65536> buffer;
    for (std::uint64_t first = 0; first < count;) {
        const auto chunk = std::min<std::uint64_t>(count - first, buffer.size() / 8);
        read_bytes(buffer.data(), chunk * 8);
        for (std::uint64_t i = 0; i < chunk; ++i) {
            const auto bits = decode_u64(buffer.data() + i * 8);
            double value;
            std::memcpy(&value, &bits, sizeof(value));
            if (!std::isfinite(value) || ((first + i) % (std::uint64_t(width) + 1) == 0 && value < 0) || (!positions && value != 0)) {
                throw std::invalid_argument("invalid covariance value");
            }
        }
        first += chunk;
    }
}

void reader::check_end() {
    if (input_.peek() != std::char_traits<char>::eof()) { throw std::invalid_argument("trailing data after calibration layers"); }
    if (input_.bad()) { throw std::runtime_error("cannot check calibration end"); }
    finished_ = true;
}

std::optional<layer_statistics> reader::next() {
    if (failed_) { throw std::logic_error("calibration reader has failed"); }
    if (finished_) { return std::nullopt; }
    try {
        if (read_layers_ == header_.layers) { check_end(); return std::nullopt; }
        layer_statistics layer;
        layer.id = read_u32();
        layer.experts = read_u32();
        layer.embedding = read_u32();
        layer.hidden = read_u32();
        layer.positions = read_u64();
        layer.routes = read_u64();
        layer.covariance_positions = read_u64();
        if (!layer.experts || !layer.embedding || ((header_.flags & HIDDEN) && !layer.hidden) || !ids_.insert(layer.id).second) {
            throw std::invalid_argument("invalid or repeated calibration layer");
        }
        if (layer.positions > header_.decoded || layer.routes > header_.decoded || layer.covariance_positions > layer.positions ||
            (!(header_.flags & COVARIANCE) && layer.covariance_positions)) {
            throw std::invalid_argument("layer observations exceed the declared decoded range");
        }
        const auto expert_values = multiply(layer.experts, layer.embedding);
        const auto hidden_values = multiply(layer.experts, layer.hidden);
        std::uint64_t values = add(layer.experts, layer.embedding);
        if (header_.flags & EXPERT_INPUT) { values = add(values, add(layer.experts, multiply(expert_values, 2))); }
        if (header_.flags & HIDDEN) { values = add(values, add(layer.experts, hidden_values)); }
        if (header_.flags & COVARIANCE) { values = add(values, multiply(layer.embedding, layer.embedding)); }
        const auto payload = multiply(values, 8);
        if (add(payload, 40) > limits_.max_layer_bytes || payload > limits_.max_file_bytes - bytes_ || payload > SIZE_MAX) {
            throw std::invalid_argument("calibration layer exceeds the byte limit");
        }
        layer.route_counts = read_counts(layer.experts, layer.routes);
        layer.input_sum2 = read_values(layer.embedding, true);
        if (!layer.positions && std::any_of(layer.input_sum2.begin(), layer.input_sum2.end(), [](double value) { return value != 0; })) {
            throw std::invalid_argument("zero input count has nonzero statistics");
        }
        if (header_.flags & EXPERT_INPUT) {
            layer.expert_counts = read_counts(layer.experts, layer.positions);
            layer.expert_sum = read_values(expert_values, false);
            layer.expert_sum2 = read_values(expert_values, true);
            check_zero_rows(layer.expert_counts, layer.expert_sum, layer.embedding);
            check_zero_rows(layer.expert_counts, layer.expert_sum2, layer.embedding);
        }
        if (header_.flags & HIDDEN) {
            layer.hidden_counts = read_counts(layer.experts, layer.positions);
            layer.hidden_sum2 = read_values(hidden_values, true);
            check_zero_rows(layer.hidden_counts, layer.hidden_sum2, layer.hidden);
        }
        if (header_.flags & COVARIANCE) { read_covariance(layer.embedding, layer.covariance_positions); }
        ++read_layers_;
        if (read_layers_ == header_.layers) { check_end(); }
        return layer;
    } catch (...) {
        failed_ = true;
        throw;
    }
}

} // namespace tq::calibration
