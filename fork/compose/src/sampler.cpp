#include "tq/compose/sampler.h"
#include "tq/sampling/shortlist.h"
#include "llama.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tq::compose {
namespace {

struct sampler_deleter {
    void operator()(llama_sampler * value) const { llama_sampler_free(value); }
};
using sampler_ptr = std::unique_ptr<llama_sampler, sampler_deleter>;

void check_temperature(float value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("temperature must be finite");
    }
}

void check_options(const sampling_options & options) {
    check_temperature(options.temperature);
    if (!std::isfinite(options.top_p) || options.top_p < 0 || options.top_p > 1 ||
        !std::isfinite(options.min_p) || options.min_p < 0 || options.min_p > 1) {
        throw std::invalid_argument("top_p and min_p must be between zero and one");
    }
}

sampler_ptr make_chain(const sampling_options & options) {
    sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(options.top_k));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_top_p(options.top_p, 1));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_min_p(options.min_p, 1));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(options.temperature));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(options.seed));
    return chain;
}

} // namespace

struct sampler::impl {
    explicit impl(sampling_options value) : options(value), chain(make_chain(value)) {}

    sampling_options options;
    sampler_ptr chain;
    std::vector<llama_token_data> candidates;
    sampling_stats counts;
};

sampler::sampler(sampling_options options) {
    check_options(options);
    state_ = std::make_unique<impl>(options);
}

sampler::~sampler() = default;
sampler::sampler(sampler &&) noexcept = default;
sampler & sampler::operator=(sampler &&) noexcept = default;

int32_t sampler::sample(const float * logits, int32_t count) {
    if (!logits || count <= 0) {
        throw std::invalid_argument("logits must contain at least one value");
    }
    bool compact = false;
    auto & candidates = state_->candidates;
    if (state_->options.compact && state_->options.temperature <= 0) {
        const auto selected = sampling::select(logits, count, state_->options.top_k);
        if (selected.usable) {
            candidates.clear();
            candidates.reserve(selected.candidates.size());
            for (const auto & entry : selected.candidates) {
                candidates.push_back({entry.id, entry.logit, 0.0f});
            }
            compact = true;
        }
    }
    if (!compact) {
        candidates.resize(static_cast<size_t>(count));
        bool has_finite = false;
        for (int32_t i = 0; i < count; ++i) {
            const float value = logits[i];
            if (std::isnan(value) || value == std::numeric_limits<float>::infinity()) {
                throw std::invalid_argument("logits contain NaN or positive infinity");
            }
            has_finite = has_finite || std::isfinite(value);
            candidates[static_cast<size_t>(i)] = {i, value, 0.0f};
        }
        if (!has_finite) {
            throw std::invalid_argument("logits have no finite candidate");
        }
    }

    llama_token_data_array row{candidates.data(), candidates.size(), -1, compact};
    llama_sampler_apply(state_->chain.get(), &row);
    if (row.selected < 0 || static_cast<size_t>(row.selected) >= row.size) {
        throw std::runtime_error("upstream sampler did not select a token");
    }
    const auto token = row.data[row.selected].id;
    llama_sampler_accept(state_->chain.get(), token);
    if (compact) {
        ++state_->counts.compact_rows;
    } else {
        ++state_->counts.full_rows;
    }
    return token;
}

sampler sampler::clone() const {
    sampler result(state_->options);
    result.state_->chain.reset(llama_sampler_clone(state_->chain.get()));
    if (!result.state_->chain) {
        throw std::runtime_error("upstream sampler clone failed");
    }
    result.state_->counts = state_->counts;
    return result;
}

void sampler::reset() {
    llama_sampler_reset(state_->chain.get());
    state_->counts = {};
}

void sampler::set_temperature(float temperature) {
    check_temperature(temperature);
    sampler_ptr replacement(llama_sampler_init_temp(temperature));
    // Keep the distribution sampler, including the current random state.
    sampler_ptr distribution(llama_sampler_chain_remove(state_->chain.get(), 4));
    sampler_ptr previous(llama_sampler_chain_remove(state_->chain.get(), 3));
    llama_sampler_chain_add(state_->chain.get(), replacement.release());
    llama_sampler_chain_add(state_->chain.get(), distribution.release());
    state_->options.temperature = temperature;
}

sampling_stats sampler::stats() const { return state_->counts; }

} // namespace tq::compose
