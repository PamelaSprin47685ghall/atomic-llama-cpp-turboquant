#include <llama.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool valid, const std::string & message) {
    if (!valid) { throw std::runtime_error(message); }
}

} // namespace

int main() {
    try {
        constexpr std::int32_t top_k = 129;
        std::vector<llama_token_data> candidates;
        for (llama_token id = 0; id < top_k; ++id) {
            candidates.push_back({id, float(id), 0.0f});
        }
        candidates.push_back({129, 1.0e10f, 0.0f});
        candidates.push_back({130, std::numeric_limits<float>::max(), 0.0f});
        candidates.push_back({131, -std::numeric_limits<float>::max(), 0.0f});

        // A full comparison sort supplies the expected order and set. It has
        // no buckets, float-to-int conversion, or native sampling dependency.
        auto expected = candidates;
        std::sort(expected.begin(), expected.end(), [](const auto & a, const auto & b) {
            return a.logit > b.logit;
        });
        expected.resize(top_k);

        using sampler_owner = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
        sampler_owner top(llama_sampler_init_top_k(top_k), llama_sampler_free);
        sampler_owner greedy(llama_sampler_init_greedy(), llama_sampler_free);
        require(bool(top) && bool(greedy), "could not create native samplers");
        llama_token_data_array row{candidates.data(), candidates.size(), -1, false};
        llama_sampler_apply(top.get(), &row);
        require(row.size == std::size_t(top_k) && row.sorted, "native top-k returned an invalid row");
        llama_sampler_apply(greedy.get(), &row);
        require(row.selected >= 0 && std::size_t(row.selected) < row.size, "native greedy did not select a token");
        const auto selected = row.data[row.selected].id;
        require(selected == expected.front().id,
                "native top-k lost the maximum finite logit: selected " + std::to_string(selected) +
                ", expected " + std::to_string(expected.front().id));
        for (std::size_t index = 0; index < expected.size(); ++index) {
            require(row.data[index].id == expected[index].id && row.data[index].logit == expected[index].logit,
                    "native top-k retained the wrong ordered set at index " + std::to_string(index));
        }
        std::cout << "PASS: native top-k 129 retains the exact ordered set and maximum across extreme finite logits\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
