#pragma once

#include <cstdint>
#include <memory>

namespace tq::compose {

struct sampling_options {
    int32_t top_k = 40;
    float top_p = 0.95f;
    float min_p = 0.05f;
    float temperature = 0.0f;
    uint32_t seed = 42;
    bool compact = true;
};

struct sampling_stats {
    uint64_t compact_rows = 0;
    uint64_t full_rows = 0;
};

// Owns the sampler chain and its random state. Calls do not retain input rows.
class sampler {
public:
    explicit sampler(sampling_options options = {});
    ~sampler();
    sampler(sampler &&) noexcept;
    sampler & operator=(sampler &&) noexcept;
    sampler(const sampler &) = delete;
    sampler & operator=(const sampler &) = delete;

    // Samples and accepts one token. Rejects NaN, positive infinity, and empty support.
    int32_t sample(const float * logits, int32_t count);
    sampler clone() const;
    void reset();
    void set_temperature(float temperature);
    sampling_stats stats() const;

private:
    struct impl;
    std::unique_ptr<impl> state_;
};

} // namespace tq::compose
