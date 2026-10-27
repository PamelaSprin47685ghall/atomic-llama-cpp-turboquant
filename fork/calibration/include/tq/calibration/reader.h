#pragma once

#include <cstdint>
#include <istream>
#include <optional>
#include <set>
#include <vector>

namespace tq::calibration {

enum statistic_flags : std::uint32_t {
    COVARIANCE = 1u,
    HIDDEN = 2u,
    EXPERT_INPUT = 4u,
};

struct read_limits {
    std::uint32_t max_layers = 4096;
    std::uint64_t max_layer_bytes = 256ULL * 1024 * 1024;
    std::uint64_t max_file_bytes = 16ULL * 1024 * 1024 * 1024;
};

struct file_header {
    std::uint32_t layers = 0;
    std::uint32_t max_context = 0;
    std::uint32_t activation_stride = 0;
    std::uint32_t activation_tokens = 0;
    std::uint32_t route_stride = 0;
    std::uint32_t flags = 0;
    std::uint64_t decoded = 0;
    std::uint64_t total = 0;
    std::uint64_t context_resets = 0;
    bool complete = false;
};

struct layer_statistics {
    std::uint32_t id = 0;
    std::uint32_t experts = 0;
    std::uint32_t embedding = 0;
    std::uint32_t hidden = 0;
    std::uint64_t positions = 0;
    std::uint64_t routes = 0;
    std::uint64_t covariance_positions = 0;
    std::vector<std::uint64_t> route_counts;
    std::vector<double> input_sum2;
    std::vector<std::uint64_t> expert_counts;
    std::vector<double> expert_sum;
    std::vector<double> expert_sum2;
    std::vector<std::uint64_t> hidden_counts;
    std::vector<double> hidden_sum2;
};

// The stream must outlive the reader. Each returned layer owns its arrays.
class reader {
public:
    explicit reader(std::istream & input, read_limits limits = {});
    reader(const reader &) = delete;
    reader & operator=(const reader &) = delete;
    reader(reader &&) = delete;
    reader & operator=(reader &&) = delete;

    file_header header() const { return header_; }
    std::optional<layer_statistics> next();
    std::uint64_t bytes_read() const { return bytes_; }

private:
    void read_bytes(void * output, std::uint64_t count);
    std::uint32_t read_u32();
    std::uint64_t read_u64();
    std::vector<std::uint64_t> read_counts(std::uint64_t count, std::uint64_t maximum);
    std::vector<double> read_values(std::uint64_t count, bool nonnegative);
    void read_covariance(std::uint32_t width, std::uint64_t positions);
    void check_end();

    std::istream & input_;
    read_limits limits_;
    file_header header_;
    std::set<std::uint32_t> ids_;
    std::uint32_t read_layers_ = 0;
    std::uint64_t bytes_ = 0;
    bool finished_ = false;
    bool failed_ = false;
};

} // namespace tq::calibration
