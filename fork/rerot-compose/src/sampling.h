#pragma once

#include <llama.h>
#include <tq/sampling/shortlist.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tq::rerot_compose {

struct sampling_stats {
    std::uint64_t compact_rows = 0;
    std::uint64_t full_rows = 0;
};

class stage_sampler {
public:
    stage_sampler(std::uint32_t base_seed, std::uint32_t stage, std::uint32_t top_k, float temperature)
        : sampler_(llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free),
          top_k_(top_k), greedy_(temperature <= 0.0f) {
        if (!sampler_) { throw std::runtime_error("could not create a sampler"); }
        if (temperature > 0.0f) {
            // The plan stage, never a physical row or pen, chooses the stream.
            // UINT32_MAX has a special random-seed meaning in the native API.
            const auto seed = std::uint32_t((std::uint64_t(base_seed) +
                std::uint64_t(stage) * UINT64_C(0x9e3779b9)) % UINT64_C(0xffffffff));
            llama_sampler_chain_add(sampler_.get(), llama_sampler_init_top_k(std::int32_t(top_k)));
            llama_sampler_chain_add(sampler_.get(), llama_sampler_init_temp(temperature));
            llama_sampler_chain_add(sampler_.get(), llama_sampler_init_dist(seed));
        } else { llama_sampler_chain_add(sampler_.get(), llama_sampler_init_greedy()); }
    }

    llama_token sample(const std::vector<float> & logits) {
        if (logits.empty() || logits.size() > std::size_t(INT32_MAX)) {
            throw std::runtime_error("invalid output vocabulary");
        }
        // Keep the existing finite-only contract before either route can touch
        // the native sampler. B also permits negative infinity masks; C does not.
        for (const auto value : logits) {
            if (!std::isfinite(value)) { throw std::runtime_error("A returned non-finite logits"); }
        }
        const bool compact = prepare_compact(logits);
        if (!compact) {
            // Reuse storage, but reset every native field. A may sort the last
            // row in place. Original token order preserves ties and RNG use.
            candidates_.resize(logits.size());
            for (std::size_t token = 0; token < logits.size(); ++token) {
                candidates_[token] = {llama_token(token), logits[token], 0.0f};
            }
        }
        llama_token_data_array array{candidates_.data(), candidates_.size(), -1, compact};
        llama_sampler_apply(sampler_.get(), &array);
        if (array.selected < 0 || std::size_t(array.selected) >= array.size) {
            throw std::runtime_error("sampler returned no selected token");
        }
        const auto token = array.data[array.selected].id;
        if (token < 0 || std::size_t(token) >= logits.size()) {
            throw std::runtime_error("sampler returned a token outside the vocabulary");
        }
        llama_sampler_accept(sampler_.get(), token);
        if (compact) { ++stats_.compact_rows; }
        else { ++stats_.full_rows; }
        return token;
    }

    void accept(llama_token token) { llama_sampler_accept(sampler_.get(), token); }
    sampling_stats stats() const noexcept { return stats_; }

private:
    bool prepare_compact(const std::vector<float> & logits) {
        if (!greedy_ && (top_k_ == 0 || top_k_ > 128)) { return false; }
        // Greedy ignores top-k. B needs two candidates to detect a tied maximum.
        const auto count = greedy_ ? 2 : std::max(2, std::int32_t(top_k_));
        const auto selected = tq::sampling::select(logits.data(), std::int32_t(logits.size()), count);
        if (!selected.usable) { return false; }
        const auto kept = greedy_ ? selected.candidates.size() :
            std::min(logits.size(), std::size_t(top_k_));
        if (!greedy_) {
            // Native stochastic sampling reads candidates in order. B does not
            // specify tie order, so require a unique retained order and cutoff.
            for (std::size_t index = 1; index < kept; ++index) {
                if (selected.candidates[index - 1].logit == selected.candidates[index].logit) { return false; }
            }
            const auto cutoff = selected.candidates[kept - 1].logit;
            bool found = false;
            for (const auto value : logits) {
                if (value == cutoff) {
                    if (found) { return false; }
                    found = true;
                }
            }
        }
        candidates_.resize(kept);
        for (std::size_t index = 0; index < kept; ++index) {
            const auto & entry = selected.candidates[index];
            candidates_[index] = {entry.id, entry.logit, 0.0f};
        }
        return true;
    }

    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler_;
    std::uint32_t top_k_;
    bool greedy_;
    std::vector<llama_token_data> candidates_;
    sampling_stats stats_;
};

} // namespace tq::rerot_compose
