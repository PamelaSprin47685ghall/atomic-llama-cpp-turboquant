#pragma once

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <vector>

// Fast top-k preselection helper for common_sampler.
// Evaluates raw unmasked logits, pre-masks model-suppressed tokens to -INFINITY,
// and extracts top k candidates in descending order, matching llama_sampler_init_top_k(k).
// Declines (returns false) on invalid inputs, non-finite logits, or ties.

namespace common_sampler_detail {

struct lazy_candidate_iterator {
    using iterator_category = std::input_iterator_tag;
    using value_type        = llama_token_data;
    using difference_type   = std::ptrdiff_t;
    using pointer           = const llama_token_data *;
    using reference         = llama_token_data;

    const float *                    logits;
    const std::vector<llama_token> & supp;
    llama_token                      curr_id;
    size_t                           supp_idx;
    bool *                           flag_nonfinite;

    lazy_candidate_iterator(const float *                    logits,
                            const std::vector<llama_token> & supp,
                            llama_token                      curr_id,
                            bool *                           flag_nonfinite) :
        logits(logits),
        supp(supp),
        curr_id(curr_id),
        supp_idx(0),
        flag_nonfinite(flag_nonfinite) {}

    llama_token_data operator*() const {
        const float val = logits[curr_id];
        if (!std::isfinite(val)) {
            *flag_nonfinite = true;
            return llama_token_data{ curr_id, -INFINITY, 0.0f };
        }
        const bool is_supp = (supp_idx < supp.size() && supp[supp_idx] == curr_id);
        return llama_token_data{ curr_id, is_supp ? -INFINITY : val, 0.0f };
    }

    lazy_candidate_iterator & operator++() {
        ++curr_id;
        while (supp_idx < supp.size() && supp[supp_idx] < curr_id) {
            ++supp_idx;
        }
        return *this;
    }

    lazy_candidate_iterator operator++(int) {
        lazy_candidate_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    bool operator==(const lazy_candidate_iterator & other) const { return curr_id == other.curr_id; }

    bool operator!=(const lazy_candidate_iterator & other) const { return curr_id != other.curr_id; }
};

}  // namespace common_sampler_detail

inline bool common_sampler_compact_top_k(const float *                    logits,
                                         int32_t                          n_vocab,
                                         int32_t                          k,
                                         const std::vector<llama_token> & sorted_suppressed,
                                         std::vector<llama_token_data> &  candidates) {
    if (!logits || n_vocab <= 0 || k <= 0 || k > 128 || k >= n_vocab) {
        return false;
    }

    bool nonfinite_encountered = false;
    bool tie_encountered       = false;

    // Partial sort comparator flags equal logits to prevent stdlib tie discrepancies.
    // Evicted cutoff ties are observed at insertion/tail comparison; retained ties trigger equality.
    auto comp = [&tie_encountered](const llama_token_data & a, const llama_token_data & b) {
        if (a.logit == b.logit && a.logit != -INFINITY) {
            tie_encountered = true;
        }
        return a.logit > b.logit;
    };

    candidates.resize(k);

    common_sampler_detail::lazy_candidate_iterator first(logits, sorted_suppressed, 0, &nonfinite_encountered);
    common_sampler_detail::lazy_candidate_iterator last(logits, sorted_suppressed, n_vocab, &nonfinite_encountered);

    std::partial_sort_copy(first, last, candidates.begin(), candidates.end(), comp);

    if (nonfinite_encountered || tie_encountered) {
        candidates.clear();
        return false;
    }

    // Cutoff must be finite; non-finite implies all/mostly suppressed or invalid tokens.
    if (!std::isfinite(candidates[k - 1].logit)) {
        candidates.clear();
        return false;
    }

    // Strict descending check across top-k guarantees no adjacent ties.
    for (int32_t i = 1; i < k; ++i) {
        if (candidates[i - 1].logit <= candidates[i].logit) {
            candidates.clear();
            return false;
        }
    }

    return true;
}
