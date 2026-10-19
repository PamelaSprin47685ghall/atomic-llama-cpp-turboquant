#pragma once

#include <llama.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tq::rerot_compose {

class stage_sampler {
public:
    stage_sampler(std::uint32_t base_seed, std::uint32_t stage, std::uint32_t top_k, float temperature)
        : sampler_(llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free) {
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
        std::vector<llama_token_data> candidates;
        candidates.reserve(logits.size());
        for (std::size_t token = 0; token < logits.size(); ++token) {
            if (!std::isfinite(logits[token])) { throw std::runtime_error("A returned non-finite logits"); }
            candidates.push_back({llama_token(token), logits[token], 0.0f});
        }
        llama_token_data_array array{candidates.data(), candidates.size(), -1, false};
        llama_sampler_apply(sampler_.get(), &array);
        if (array.selected < 0 || std::size_t(array.selected) >= array.size) {
            throw std::runtime_error("sampler returned no selected token");
        }
        const auto token = array.data[array.selected].id;
        if (token < 0 || std::size_t(token) >= logits.size()) {
            throw std::runtime_error("sampler returned a token outside the vocabulary");
        }
        llama_sampler_accept(sampler_.get(), token);
        return token;
    }

    void accept(llama_token token) { llama_sampler_accept(sampler_.get(), token); }

private:
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler_;
};

} // namespace tq::rerot_compose
