#include "sampling.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using tq::rerot_compose::stage_sampler;

void require(bool valid, const char * message) {
    if (!valid) { throw std::runtime_error(message); }
}

// Preserve the original RERoT full-row chain and ascending token order.
// This reference never calls TQSampling or uses the candidate's row buffer.
class full_reference {
public:
    full_reference(std::uint32_t base_seed, std::uint32_t stage,
                   std::uint32_t top_k, float temperature)
        : sampler_(llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free) {
        require(bool(sampler_), "could not create the reference sampler");
        if (temperature > 0.0f) {
            const auto seed = std::uint32_t((std::uint64_t(base_seed) +
                std::uint64_t(stage) * UINT64_C(0x9e3779b9)) % UINT64_C(0xffffffff));
            llama_sampler_chain_add(sampler_.get(), llama_sampler_init_top_k(std::int32_t(top_k)));
            llama_sampler_chain_add(sampler_.get(), llama_sampler_init_temp(temperature));
            llama_sampler_chain_add(sampler_.get(), llama_sampler_init_dist(seed));
        } else { llama_sampler_chain_add(sampler_.get(), llama_sampler_init_greedy()); }
    }

    llama_token sample(const std::vector<float> & logits) {
        std::vector<llama_token_data> candidates;
        candidates.reserve(logits.size());
        for (std::size_t token = 0; token < logits.size(); ++token) {
            candidates.push_back({llama_token(token), logits[token], 0.0f});
        }
        llama_token_data_array row{candidates.data(), candidates.size(), -1, false};
        llama_sampler_apply(sampler_.get(), &row);
        require(row.selected >= 0 && std::size_t(row.selected) < row.size,
                "reference sampler returned no selected token");
        const auto token = row.data[row.selected].id;
        llama_sampler_accept(sampler_.get(), token);
        return token;
    }

    void accept(llama_token token) { llama_sampler_accept(sampler_.get(), token); }

private:
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler_;
};

std::uint64_t comparisons = 0;

void compare(stage_sampler & candidate, full_reference & reference, const std::vector<float> & logits) {
    const auto original = logits;
    require(candidate.sample(logits) == reference.sample(logits), "token or sampler RNG sequence differs");
    require(logits == original, "sampling changed the input logits");
    ++comparisons;
}

std::vector<float> make_row(std::mt19937 & random, std::uint32_t top_k, std::size_t trial) {
    constexpr std::size_t sizes[] = {1, 2, 3, 7, 8, 9, 31, 128, 129, 257, 513};
    std::vector<float> row(sizes[trial % (sizeof(sizes) / sizeof(sizes[0]))]);
    std::uniform_real_distribution<float> value(-20.0f, 20.0f);
    for (auto & logit : row) { logit = value(random); }
    if (trial % 4 == 0) {
        std::fill(row.begin(), row.end(), 0.0f);
        if (row.size() > 1) { row.front() = -0.0f; }
    } else if (trial % 4 == 1) {
        // Ties at the cutoff can involve candidates outside B's returned row.
        const auto count = std::min(row.size(), std::size_t(top_k));
        std::fill(row.begin(), row.end(), -50.0f);
        for (std::size_t i = 0; i + 1 < count; ++i) { row[i] = 1000.0f - float(i); }
    } else if (trial % 4 == 2 && row.size() >= 3) {
        // A unique maximum does not make the other retained logits unique.
        row[0] = 1000.0f;
        row[1] = row[2] = 500.0f;
    }
    return row;
}

void parity() {
    std::mt19937 random(7321);
    for (const float temperature : {0.0f, 0.125f, 0.8f, 1.7f}) {
        for (const std::uint32_t top_k : {1u, 2u, 7u, 40u, 128u, 129u, 4096u}) {
            for (const std::uint32_t seed : {0u, UINT32_MAX - 1}) {
                const std::uint32_t stage = seed == 0 ? 1 : 17;
                stage_sampler candidate(seed, stage, top_k, temperature);
                full_reference reference(seed, stage, top_k, temperature);
                for (const llama_token token : {0, 2, 1, 7}) {
                    candidate.accept(token);
                    reference.accept(token);
                }
                for (std::size_t trial = 0; trial < 128; ++trial) {
                    compare(candidate, reference, make_row(random, top_k, trial));
                }
                const auto stats = candidate.stats();
                require(stats.compact_rows + stats.full_rows == 128, "sampling routes did not account for every row");
                if (temperature == 0.0f || top_k <= 128) {
                    require(stats.compact_rows > 0 && stats.full_rows > 0,
                            "eligible rows and tied fallback rows must both execute");
                } else {
                    require(stats.compact_rows == 0, "an unsupported stochastic top-k used a compact row");
                }
            }
        }
    }
}

void invalid_rows() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<std::vector<float>> invalid = {
        {}, {nan, 1.0f}, {1.0f, nan}, {1.0f, inf}, {1.0f, -inf}, {-inf, -inf}};
    std::mt19937 random(8321);
    for (const float temperature : {0.0f, 0.8f}) {
        stage_sampler candidate(123, 5, 7, temperature);
        full_reference reference(123, 5, 7, temperature);
        for (std::size_t trial = 0; trial < 20; ++trial) {
            compare(candidate, reference, make_row(random, 7, trial));
        }
        const auto before = candidate.stats();
        for (const auto & row : invalid) {
            bool rejected = false;
            try { static_cast<void>(candidate.sample(row)); }
            catch (const std::runtime_error &) { rejected = true; }
            require(rejected, "invalid logits were accepted");
        }
        const auto after = candidate.stats();
        require(after.compact_rows == before.compact_rows && after.full_rows == before.full_rows,
                "invalid input changed sampling counters");
        for (std::size_t trial = 0; trial < 100; ++trial) {
            compare(candidate, reference, make_row(random, 7, trial));
        }
    }
}

void tie_routes() {
    for (const std::uint32_t top_k : {1u, 2u, 3u}) {
        stage_sampler candidate(17, 3, top_k, 0.8f);
        full_reference reference(17, 3, top_k, 0.8f);
        const std::vector<float> tied = top_k == 1 ? std::vector<float>{10.0f, 10.0f, 0.0f, -1.0f} :
                                                                   std::vector<float>{10.0f, 0.0f, -0.0f, -1.0f};
        compare(candidate, reference, tied);
        require(candidate.stats().full_rows == 1 && candidate.stats().compact_rows == 0,
                "a stochastic retained or cutoff tie did not use the full row");
        compare(candidate, reference, {10.0f, 0.5f, 0.0f, -1.0f});
        require(candidate.stats().compact_rows == 1, "an ordered stochastic shortlist did not execute");
    }
    stage_sampler greedy(17, 3, 2, 0.0f);
    full_reference reference(17, 3, 2, 0.0f);
    compare(greedy, reference, {10.0f, 0.0f, -0.0f, -1.0f});
    require(greedy.stats().compact_rows == 1, "a lower greedy tie disabled a valid unique maximum");
}

void separate_stages() {
    stage_sampler first(31, 1, 7, 0.8f), second(31, 2, 40, 0.8f), noise(31, 1, 7, 0.8f);
    full_reference first_reference(31, 1, 7, 0.8f), second_reference(31, 2, 40, 0.8f);
    std::mt19937 random(9321);
    for (std::size_t trial = 0; trial < 128; ++trial) {
        const auto row = make_row(random, 7, trial);
        static_cast<void>(noise.sample(row));
        if (trial % 2 == 0) {
            compare(first, first_reference, row);
            compare(second, second_reference, row);
        } else {
            compare(second, second_reference, row);
            compare(first, first_reference, row);
        }
    }
}

} // namespace

int main() {
    try {
        parity();
        invalid_rows();
        tie_routes();
        separate_stages();
        std::cout << "PASS: " << comparisons
                  << " native sampler comparisons; compact/full routes, ties, cutoff ties, RNG, changing vocabulary,"
                     " finite-only rejection, input ownership, and separate stages\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
