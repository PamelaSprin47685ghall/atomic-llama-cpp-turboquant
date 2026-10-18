#include "tq/compose/sampler.h"
#include "llama.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

template<class F>
void rejects(F action) {
    bool failed = false;
    try { action(); } catch (const std::invalid_argument &) { failed = true; }
    check(failed, "invalid input was not rejected");
}

struct reference {
    explicit reference(tq::compose::sampling_options options) {
        chain.reset(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(options.top_k));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_p(options.top_p, 1));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_min_p(options.min_p, 1));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(options.temperature));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(options.seed));
    }

    int32_t sample(const std::vector<float> & logits) {
        data.resize(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) data[i] = {static_cast<llama_token>(i), logits[i], 0};
        llama_token_data_array row{data.data(), data.size(), -1, false};
        llama_sampler_apply(chain.get(), &row);
        check(row.selected >= 0 && static_cast<size_t>(row.selected) < row.size, "reference did not select a token");
        const auto token = row.data[row.selected].id;
        llama_sampler_accept(chain.get(), token);
        return token;
    }

    void temperature(float value) {
        auto * distribution = llama_sampler_chain_remove(chain.get(), 4);
        llama_sampler_free(llama_sampler_chain_remove(chain.get(), 3));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(value));
        llama_sampler_chain_add(chain.get(), distribution);
    }

    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> chain{nullptr, llama_sampler_free};
    std::vector<llama_token_data> data;
};

void parity() {
    std::mt19937 random(123);
    std::uniform_real_distribution<float> value(-20, 20);
    const float inf = std::numeric_limits<float>::infinity();
    size_t comparisons = 0;
    for (int32_t k : {0, 1, 2, 7, 40, 128, 129}) {
        tq::compose::sampling_options options;
        options.top_k = k;
        tq::compose::sampler candidate(options);
        reference ordinary(options);
        for (int trial = 0; trial < 240; ++trial) {
            std::vector<float> logits(1 + random() % 2048);
            for (auto & x : logits) x = value(random);
            if (trial % 4 == 0 && logits.size() > 1) logits[0] = logits[1] = 100;
            if (trial % 4 == 1 && logits.size() > 1) logits[0] = -inf;
            check(candidate.sample(logits.data(), static_cast<int32_t>(logits.size())) == ordinary.sample(logits), "greedy token differs");
            ++comparisons;
        }
        const auto before = candidate.stats();
        check(before.compact_rows + before.full_rows == 240, "route counters are incomplete");
        if (k >= 2 && k <= 128) {
            check(before.compact_rows > 0 && before.full_rows > 0, "both compact and fallback must run");
        } else {
            check(before.compact_rows == 0, "unsupported top-k used compact rows");
        }
        candidate.set_temperature(0.8f);
        ordinary.temperature(0.8f);
        for (int trial = 0; trial < 180; ++trial) {
            std::vector<float> logits(17, 0.0f);
            if (trial % 2) for (auto & x : logits) x = value(random) / 10;
            check(candidate.sample(logits.data(), 17) == ordinary.sample(logits), "random state differs after compact sampling");
            ++comparisons;
        }
        check(candidate.stats().compact_rows == before.compact_rows, "stochastic rows used compact mode");
    }
    std::cout << comparisons << " real upstream token comparisons passed\n";
}

void ownership_and_errors() {
    tq::compose::sampling_options options;
    options.temperature = 0.7f;
    tq::compose::sampler original(options);
    options.temperature = 0;
    auto peer = original.clone();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (const auto & row : std::vector<std::vector<float>>{{nan, 1}, {1, nan}, {1, inf}, {-inf, -inf}}) {
        rejects([&] { original.sample(row.data(), static_cast<int32_t>(row.size())); });
    }
    rejects([&] { original.sample(nullptr, 1); });
    rejects([&] { original.sample(&inf, 0); });
    rejects([&] { original.set_temperature(nan); });
    check(original.stats().full_rows == 0 && original.stats().compact_rows == 0, "invalid input changed counters");
    std::vector<float> logits(31, 0.0f);
    for (int i = 0; i < 100; ++i) check(original.sample(logits.data(), 31) == peer.sample(logits.data(), 31), "invalid input changed random state");
    auto untouched = original.clone();
    auto advanced = original.clone();
    advanced.sample(logits.data(), 31);
    for (int i = 0; i < 100; ++i) check(original.sample(logits.data(), 31) == untouched.sample(logits.data(), 31), "one clone changed another");
    auto snapshot = original.stats();
    snapshot.full_rows = 999999;
    check(original.stats().full_rows != snapshot.full_rows, "stats expose internal state");
    original.reset();
    tq::compose::sampling_options fresh_options;
    fresh_options.temperature = 0.7f;
    tq::compose::sampler fresh(fresh_options);
    for (int i = 0; i < 100; ++i) check(original.sample(logits.data(), 31) == fresh.sample(logits.data(), 31), "reset did not restore seed");
    fresh_options.min_p = -1;
    rejects([&] { tq::compose::sampler invalid(fresh_options); });
    std::cout << "clone, reset, input ownership, and invalid-input state checks passed\n";
}

void benchmark() {
    std::mt19937 random(456);
    std::uniform_real_distribution<float> value(-20, 20);
    std::vector<float> logits(248320);
    for (auto & x : logits) x = value(random);
    logits[12345] = 100;
    for (int pair = 0; pair < 4; ++pair) {
        for (int arm = 0; arm < 2; ++arm) {
            const bool compact = (pair % 2 == 0) ? arm == 1 : arm == 0;
            tq::compose::sampling_options options;
            options.compact = compact;
            tq::compose::sampler sample(options);
            for (int i = 0; i < 10; ++i) sample.sample(logits.data(), static_cast<int32_t>(logits.size()));
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 200; ++i) check(sample.sample(logits.data(), static_cast<int32_t>(logits.size())) == 12345, "benchmark token differs");
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 200;
            std::cout << "{\"scope\":\"CPU_SAMPLING_MICRO_NOT_MODEL\",\"pair\":" << pair
                      << ",\"compact\":" << (compact ? "true" : "false") << ",\"us_per_row\":" << us << "}\n";
        }
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        parity();
        ownership_and_errors();
        if (argc == 2 && std::string(argv[1]) == "--benchmark") benchmark();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
