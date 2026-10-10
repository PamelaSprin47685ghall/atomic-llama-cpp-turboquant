#include "sampling-unique-argmax.h"
#include <cassert>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

static llama_sampler * chain() {
    auto * s = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(s, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(s, llama_sampler_init_top_p(.95f, 1));
    llama_sampler_chain_add(s, llama_sampler_init_min_p(.05f, 1));
    llama_sampler_chain_add(s, llama_sampler_init_temp(0));
    llama_sampler_chain_add(s, llama_sampler_init_dist(42));
    return s;
}

static llama_token sample(llama_sampler * s, const std::vector<float> & row) {
    std::vector<llama_token_data> data;
    for (size_t i = 0; i < row.size(); ++i) data.push_back({(llama_token)i, row[i], 0});
    llama_token_data_array a{data.data(), data.size(), -1, false};
    llama_sampler_apply(s, &a);
    return a.data[a.selected].id;
}

int main() {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    assert(common_unique_finite_argmax(nullptr, 3) == LLAMA_TOKEN_NULL);
    for (const auto & row : std::vector<std::vector<float>>{{1,1}, {-inf,-inf}, {0,inf}, {nan,1}, {1,nan}})
        assert(common_unique_finite_argmax(row.data(), row.size()) == LLAMA_TOKEN_NULL);
    assert(common_unique_finite_argmax(std::vector<float>{-inf, 1}.data(), 2) == 1);
    assert(common_unique_finite_argmax(std::vector<float>{1, 1, 2}.data(), 3) == 2);
    auto * reference = chain();
    auto * candidate = chain();
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> value(-20, 20);
    for (int trial = 0; trial < 400; ++trial) {
        std::vector<float> row(1 + rng() % 2048);
        for (auto & x : row) x = value(rng);
        if (trial % 4 == 0 && row.size() > 1) row[0] = row[1] = 100; // tie fallback
        std::vector<llama_token_data> compact;
        llama_token selected;
        if (common_greedy_compact_top_k(row.data(), row.size(), 40, compact)) {
            llama_token_data_array a{compact.data(), compact.size(), -1, true};
            llama_sampler_apply(candidate, &a);
            selected = a.data[a.selected].id;
        } else selected = sample(candidate, row);
        assert(selected == sample(reference, row));
        llama_sampler_accept(candidate, selected);
        llama_sampler_accept(reference, selected);
    }
    // Check the actual dist state after bypasses, not just deterministic IDs.
    auto * rd = llama_sampler_chain_get(reference, 4);
    auto * cd = llama_sampler_chain_get(candidate, 4);
    for (int i = 0; i < 100; ++i) assert(sample(rd, {0,0,0,0}) == sample(cd, {0,0,0,0}));
    llama_sampler_free(reference); llama_sampler_free(candidate);
    std::puts("400 mixed unique/tie rows matched; 100 subsequent RNG-dependent samples matched; non-finite guards passed");
    // Independent CPU cost calibration at the production vocabulary size.
    // Reuse candidate storage like common_sampler::set_logits, not allocation per row.
    std::vector<float> logits(248320);
    for (auto & x : logits) x = value(rng);
    logits[12345] = 100;
    std::vector<llama_token_data> data(logits.size());
    auto * ordinary = chain();
    auto * fast = chain();
    const int repeats = 200;
    const auto start = ggml_time_us();
    for (int k = 0; k < repeats; ++k) {
        for (size_t i = 0; i < data.size(); ++i) data[i] = {(llama_token)i, logits[i], 0};
        llama_token_data_array a{data.data(), data.size(), -1, false};
        llama_sampler_apply(ordinary, &a);
        assert(a.data[a.selected].id == 12345);
    }
    const auto middle = ggml_time_us();
    std::vector<llama_token_data> compact;
    for (int k = 0; k < repeats; ++k) {
        assert(common_greedy_compact_top_k(logits.data(), logits.size(), 40, compact));
        llama_token_data_array a{compact.data(), compact.size(), -1, true};
        llama_sampler_apply(fast, &a);
        assert(a.data[a.selected].id == 12345);
    }
    const auto end = ggml_time_us();
    std::printf("{\"scope\":\"CPU_SAMPLING_MICRO_NOT_MODEL\",\"vocabulary\":248320,\"repeats\":%d,\"ordinary_us\":%.3f,\"unique_us\":%.3f}\n",
            repeats, double(middle-start)/repeats, double(end-middle)/repeats);
    llama_sampler_free(ordinary); llama_sampler_free(fast);
}
