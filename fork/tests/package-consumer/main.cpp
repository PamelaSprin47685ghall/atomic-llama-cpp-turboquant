#include "tq/compose/sampler.h"
#include "tq/corpus/trie.h"
#include "tq/sampling/shortlist.h"

#include <stdexcept>
#include <vector>

int main() {
    std::vector<float> values{1.0f, 3.0f, 2.0f, -1.0f};
    auto result = tq::sampling::select(values.data(), static_cast<int32_t>(values.size()), 2);
    if (!result.usable || result.candidates.size() != 2 || result.candidates.front().id != 1) {
        throw std::runtime_error("installed sampling library failed");
    }
    tq::corpus::token_trie trie(4);
    const auto leaf = trie.append({1, 2, 3});
    auto path = trie.path_to(leaf);
    path[0] = 0;
    if (trie.path_to(leaf).front() != 1 || trie.request_count() != 1) {
        throw std::runtime_error("installed corpus library failed");
    }
    tq::compose::sampling_options options;
    options.top_k = 2;
    tq::compose::sampler sample(options);
    if (sample.sample(values.data(), static_cast<int32_t>(values.size())) != 1 ||
        sample.stats().compact_rows != 1) {
        throw std::runtime_error("installed composer failed");
    }
}
