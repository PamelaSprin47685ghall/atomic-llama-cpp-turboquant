#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace tq::rerot_compose {

// Public keys can enter the final reader. Each stage also owns one private
// source-end key. Only the current reader's source end enters its view.
class episode_capacity {
public:
    explicit episode_capacity(std::uint64_t base_tokens) : public_tokens_(base_tokens) {}

    void add_stage(std::uint64_t frame_tokens, std::uint64_t body_limit) {
        const auto next_public = add(add(public_tokens_, frame_tokens), body_limit);
        const auto next_ends = add(source_ends_, 1);
        // Check both totals before changing the budget.
        static_cast<void>(add(next_public, next_ends));
        public_tokens_ = next_public;
        source_ends_ = next_ends;
    }

    std::uint64_t physical_tokens() const { return add(public_tokens_, source_ends_); }
    std::uint64_t reader_tokens() const { return add(public_tokens_, source_ends_ ? 1 : 0); }

    void require(std::uint64_t physical_limit, std::uint64_t reader_limit) const {
        if (physical_tokens() > physical_limit) {
            throw std::invalid_argument("the full task graph needs " + std::to_string(physical_tokens()) +
                " physical KV tokens, but --total-kv supplies " + std::to_string(physical_limit));
        }
        if (reader_tokens() > reader_limit) {
            throw std::invalid_argument("the full task graph needs " + std::to_string(reader_tokens()) +
                " reader tokens, but --ctx-size supplies " + std::to_string(reader_limit));
        }
    }

private:
    static std::uint64_t add(std::uint64_t first, std::uint64_t second) {
        if (second > std::numeric_limits<std::uint64_t>::max() - first) {
            throw std::overflow_error("task graph capacity overflows");
        }
        return first + second;
    }

    std::uint64_t public_tokens_;
    std::uint64_t source_ends_ = 0;
};

} // namespace tq::rerot_compose
