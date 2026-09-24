#include "ggml.h"
#include "llama.h"
#include "sampling-top-k.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

extern struct llama_sampler * llama_sampler_init_dry_testing(int32_t context_size, float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const std::vector<std::vector<llama_token>>& seq_breakers);

static void dump(const llama_token_data_array * cur_p) {
    for (size_t i = 0; i < cur_p->size; i++) {
        printf("%d: %f (%f)\n", cur_p->data[i].id, cur_p->data[i].p, cur_p->data[i].logit);
    }
}

#define DUMP(__cur_p) do { printf("%s:%d (%s)\n", __FILE__, __LINE__, __func__); dump((__cur_p)); printf("-\n"); } while(0)

struct sampler_tester {
    sampler_tester(size_t n_vocab) {
        cur.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
            const float logit = logf(token_id);
            cur.emplace_back(llama_token_data{token_id, logit, 0.0f});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    sampler_tester(const std::vector<float> & probs, const std::vector<float> & probs_expected) : probs_expected(probs_expected) {
        cur.reserve(probs.size());
        for (llama_token token_id = 0; token_id < (llama_token)probs.size(); token_id++) {
            const float logit = logf(probs[token_id]);
            cur.emplace_back(llama_token_data{token_id, logit, probs[token_id]});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    void apply(llama_sampler * sampler) {
        llama_sampler_apply(sampler, &cur_p);
        llama_sampler_free(sampler);
    }

    void check() {
        GGML_ASSERT(cur_p.size == probs_expected.size());
        for (size_t i = 0; i < cur_p.size; i++) {
            GGML_ASSERT(fabs(cur_p.data[i].p - probs_expected[i]) < 1e-5);
        }
    }

    llama_token_data_array cur_p;

private:
    const std::vector<float> probs_expected;

    std::vector<llama_token_data> cur;
};

static void test_temp(const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_temp(temp));
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_temp_ext(const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp, float delta, float exponent) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_temp_ext(temp, delta, exponent));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_top_k(const std::vector<float> & probs, const std::vector<float> & probs_expected, int k) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_k(k));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_compact_top_k_behavioral(void) {
    printf("Testing compact top-k candidate preselection:\n");

    // Direct regression vs legacy llama_sampler_init_top_k over full ascending IDs after suppression
    {
        const int          n_vocab = 32007;  // non-aligned vocab length
        std::vector<float> logits(n_vocab);
        for (int i = 0; i < n_vocab; i++) {
            // Coprime multiplier gives exact unique integer permutation scaled by power of two
            const uint32_t perm = ((uint32_t) i * 17977u) % (uint32_t) n_vocab;
            logits[i]           = (float) perm / 256.0f;
        }

        std::vector<llama_token> suppressed = { 13, 42, 100, 512, 1024, 7777, 12345, 20000, 30000, 32006 };

        const std::vector<int> test_ks = { 1, 2, 5, 32, 64, 128 };
        for (int k : test_ks) {
            std::vector<llama_token_data> legacy_cur;
            legacy_cur.reserve(n_vocab);
            for (llama_token id = 0; id < n_vocab; id++) {
                legacy_cur.push_back({ id, logits[id], 0.0f });
            }
            for (llama_token s_id : suppressed) {
                if (s_id >= 0 && s_id < n_vocab) {
                    legacy_cur[s_id].logit = -INFINITY;
                }
            }
            llama_token_data_array legacy_p = { legacy_cur.data(), legacy_cur.size(), -1, false };
            struct llama_sampler * smp_k    = llama_sampler_init_top_k(k);
            llama_sampler_apply(smp_k, &legacy_p);
            llama_sampler_free(smp_k);

            std::vector<llama_token_data> compact_cur;
            bool ok = common_sampler_compact_top_k(logits.data(), n_vocab, k, suppressed, compact_cur);
            GGML_ASSERT(ok);
            GGML_ASSERT(compact_cur.size() == (size_t) k);
            GGML_ASSERT(legacy_p.size == (size_t) k);

            for (size_t i = 0; i < (size_t) k; i++) {
                GGML_ASSERT(compact_cur[i].id == legacy_p.data[i].id);
                GGML_ASSERT(compact_cur[i].logit == legacy_p.data[i].logit);
                GGML_ASSERT(compact_cur[i].p == legacy_p.data[i].p);
                if (i > 0) {
                    GGML_ASSERT(compact_cur[i - 1].logit > compact_cur[i].logit);
                }
            }
        }
    }

    // Full chain regression with reused chains across >=32 rows verifying RNG advancement and observable probabilities
    {
        const int      n_vocab = 4096;
        const int      k       = 40;
        const float    temp    = 0.8f;
        const float    top_p   = 0.9f;
        const float    min_p   = 0.05f;
        const uint32_t seed    = 424242;
        const int      n_rows  = 36;

        std::vector<llama_token> suppressed = { 0, 10, 50, 200, 1000, 3000 };

        // Chains created once before the loop and reused across rows
        struct llama_sampler * chain_compact = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(chain_compact, llama_sampler_init_top_k(k));
        llama_sampler_chain_add(chain_compact, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(chain_compact, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(chain_compact, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(chain_compact, llama_sampler_init_dist(seed));

        struct llama_sampler * chain_full = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(chain_full, llama_sampler_init_top_k(k));
        llama_sampler_chain_add(chain_full, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(chain_full, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(chain_full, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(chain_full, llama_sampler_init_dist(seed));

        for (int row = 0; row < n_rows; row++) {
            std::vector<float> logits(n_vocab);
            for (int i = 0; i < n_vocab; i++) {
                // Exact integer permutation scaled by power of two
                const uint32_t perm = ((uint32_t) i * 1583u + (uint32_t) row * 37u) % (uint32_t) n_vocab;
                logits[i]           = (float) perm / 256.0f;
            }

            std::vector<llama_token_data> compact_cur;
            bool ok = common_sampler_compact_top_k(logits.data(), n_vocab, k, suppressed, compact_cur);
            GGML_ASSERT(ok);
            llama_token_data_array compact_p = { compact_cur.data(), compact_cur.size(), -1, true };

            std::vector<llama_token_data> full_cur;
            full_cur.reserve(n_vocab);
            for (llama_token id = 0; id < n_vocab; id++) {
                full_cur.push_back({ id, logits[id], 0.0f });
            }
            for (llama_token s_id : suppressed) {
                full_cur[s_id].logit = -INFINITY;
            }
            llama_token_data_array full_p = { full_cur.data(), full_cur.size(), -1, false };

            llama_sampler_apply(chain_compact, &compact_p);
            llama_sampler_apply(chain_full, &full_p);

            GGML_ASSERT(compact_p.selected >= 0 && (size_t) compact_p.selected < compact_p.size);
            GGML_ASSERT(full_p.selected >= 0 && (size_t) full_p.selected < full_p.size);

            llama_token sample_compact = compact_p.data[compact_p.selected].id;
            llama_token sample_full    = full_p.data[full_p.selected].id;

            GGML_ASSERT(sample_compact == sample_full);

            // Observable candidate view and exact probabilities
            GGML_ASSERT(compact_p.size == full_p.size);
            for (size_t i = 0; i < compact_p.size; i++) {
                GGML_ASSERT(compact_p.data[i].id == full_p.data[i].id);
                GGML_ASSERT(compact_p.data[i].logit == full_p.data[i].logit);
                GGML_ASSERT(compact_p.data[i].p == full_p.data[i].p);
            }
        }

        llama_sampler_free(chain_compact);
        llama_sampler_free(chain_full);
    }

    // Fallback cases: must decline without applying sampler
    {
        std::vector<llama_token_data> cur;
        std::vector<llama_token>      no_suppression;

        // Invalid k: <= 0, > 128, >= n_vocab
        {
            float logits[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            GGML_ASSERT(!common_sampler_compact_top_k(logits, 10, 0, no_suppression, cur));
            GGML_ASSERT(!common_sampler_compact_top_k(logits, 10, -1, no_suppression, cur));
            GGML_ASSERT(!common_sampler_compact_top_k(logits, 10, 10, no_suppression, cur));
            GGML_ASSERT(!common_sampler_compact_top_k(logits, 10, 15, no_suppression, cur));

            std::vector<float> big_logits(200, 1.0f);
            GGML_ASSERT(!common_sampler_compact_top_k(big_logits.data(), 200, 129, no_suppression, cur));
        }

        // Nullptr or zero n_vocab
        {
            GGML_ASSERT(!common_sampler_compact_top_k(nullptr, 100, 10, no_suppression, cur));
            float logits[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            GGML_ASSERT(!common_sampler_compact_top_k(logits, 0, 10, no_suppression, cur));
            GGML_ASSERT(!common_sampler_compact_top_k(logits, -5, 10, no_suppression, cur));
        }

        // NaN and Inf handling
        {
            float logits_nan[8] = { 1.0f, 2.0f, NAN, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
            GGML_ASSERT(!common_sampler_compact_top_k(logits_nan, 8, 4, no_suppression, cur));

            float logits_inf[8] = { 1.0f, 2.0f, INFINITY, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
            GGML_ASSERT(!common_sampler_compact_top_k(logits_inf, 8, 4, no_suppression, cur));

            float logits_neginf[8] = { 1.0f, 2.0f, -INFINITY, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
            GGML_ASSERT(!common_sampler_compact_top_k(logits_neginf, 8, 4, no_suppression, cur));
        }

        // Ambiguous ties within selection
        {
            float logits_tie[8] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 7.0f };
            GGML_ASSERT(!common_sampler_compact_top_k(logits_tie, 8, 4, no_suppression, cur));
        }

        // Ambiguous ties at cutoff / boundary with outside elements
        {
            float logits_cutoff_tie[8] = { 1.0f, 2.0f, 3.0f, 5.0f, 5.0f, 6.0f, 7.0f, 8.0f };
            GGML_ASSERT(!common_sampler_compact_top_k(logits_cutoff_tie, 8, 4, no_suppression, cur));
        }

        // All or mostly suppressed: selected -inf ties must decline
        {
            float                    logits[8]      = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
            // Suppress all 8 tokens
            std::vector<llama_token> all_suppressed = { 0, 1, 2, 3, 4, 5, 6, 7 };
            GGML_ASSERT(!common_sampler_compact_top_k(logits, 8, 4, all_suppressed, cur));

            // Suppress 6 tokens leaving only 2 finite tokens when k=4
            std::vector<llama_token> mostly_suppressed = { 0, 1, 2, 3, 4, 5 };
            GGML_ASSERT(!common_sampler_compact_top_k(logits, 8, 4, mostly_suppressed, cur));
        }
    }

    printf("Compact top-k candidate preselection tests OK\n");
}

static void test_top_p(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_p(p, 0));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_min_p(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_min_p(p, 0));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_xtc(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p, float t) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_xtc(p, t, 0, 0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_typical(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_typical(p, 0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_penalties(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & probs_expected, float repeat_penalty, float alpha_frequency, float alpha_presence
) {
    GGML_ASSERT(probs.size() == probs_expected.size());

    sampler_tester tester(probs, probs_expected);

    auto * sampler = llama_sampler_init_penalties((int32_t) probs.size(), (int32_t) last_tokens.size(), repeat_penalty, alpha_frequency, alpha_presence);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        llama_sampler_accept(sampler, last_tokens[i]);
    }

    DUMP(&tester.cur_p);
    tester.apply(sampler);
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_dry(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & expected_probs, float dry_multiplier, float dry_base,
    int dry_allowed_length, int dry_penalty_last_n,
    const std::vector<std::vector<llama_token>> & seq_breakers
) {
    GGML_ASSERT(probs.size() == expected_probs.size());

    sampler_tester tester(probs, expected_probs);

    auto * sampler = llama_sampler_init_dry_testing(1024, dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, seq_breakers);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        llama_sampler_accept(sampler, last_tokens[i]);
    }

    DUMP(&tester.cur_p);
    tester.apply(sampler);
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);
    tester.check();
}

static void test_top_n_sigma(const std::vector<float> & probs, const std::vector<float> & probs_expected, int n) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_n_sigma(n));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_sampler_queue(const size_t n_vocab, const std::string & samplers_sequence, const int top_k, const float top_p, const float min_p
) {
    sampler_tester tester(n_vocab);

          llama_token min_token_id = 0;
    const llama_token max_token_id = n_vocab - 1;

    for (auto s : samplers_sequence) {
        switch (s) {
            case 'k': tester.apply(llama_sampler_init_top_k(top_k)); break;
            case 'y': GGML_ABORT("typical test not implemented");
            case 'p': tester.apply(llama_sampler_init_top_p(top_p, 1)); break;
            case 'm': tester.apply(llama_sampler_init_min_p(min_p, 1)); break;
            case 't': GGML_ABORT("temperature test not implemented");
            default : GGML_ABORT("Unknown sampler");
        }

        tester.apply(llama_sampler_init_dist(0));

        auto & cur_p = tester.cur_p;

        const int size = cur_p.size;

        if (s == 'k') {
            const int expected_size = std::min(size, top_k);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - top_k));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(cur_p.data[0].id == max_token_id);
            GGML_ASSERT(cur_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'p') {
            const int softmax_divisor = n_vocab * (n_vocab-1) / 2 - min_token_id * (min_token_id-1) / 2;
            const int softmax_numerator_target = ceilf(top_p * softmax_divisor);

                min_token_id  = n_vocab;
            int expected_size = 0;
            int cumsum        = 0;
            do { // do-while because always at least one token is sampled
                min_token_id--;
                expected_size++;

                cumsum += min_token_id;
            } while (cumsum < softmax_numerator_target);

            // token 0 has p == 0, need special consideration for cumsum because top_p immediately returns
            if (min_token_id == 1) {
                min_token_id--;
                expected_size += 1;
            }

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[0].id == max_token_id);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'm') {
            int expected_size = ceilf((1.0f - min_p) * n_vocab);
            expected_size = std::max(expected_size, 1);
            expected_size = std::min(expected_size, size);

            min_token_id = floorf(min_p * n_vocab);
            min_token_id = std::max(min_token_id, 1);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - size));
            min_token_id = std::min(min_token_id, (llama_token)(n_vocab - 1));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[0].id == max_token_id);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
        } else {
            GGML_ABORT("fatal error");
        }
    }

    printf("Sampler queue %3s OK with n_vocab=%05zu top_k=%5d top_p=%f min_p=%f\n",
           samplers_sequence.c_str(), n_vocab, top_k, top_p, min_p);
}

static void bench(llama_sampler * cnstr, const char * cnstr_name, const std::vector<llama_token_data> & data, int n_iter) {
    std::vector<llama_token_data> cur(data.size());
    std::copy(data.begin(), data.end(), cur.begin());
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(cnstr, &cur_p);
    llama_sampler_reset(cnstr);
    const int64_t t_start = ggml_time_us();
    for (int i = 0; i < n_iter; i++) {
        std::copy(data.begin(), data.end(), cur.begin());
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(cnstr, &cur_p);
        llama_sampler_reset(cnstr);
    }
    const int64_t t_end = ggml_time_us();
    llama_sampler_free(cnstr);
    printf("%-43s: %8.3f us/iter\n", cnstr_name, (t_end - t_start) / (float)n_iter);
}

#define BENCH(__cnstr, __data, __n_iter) bench((__cnstr), #__cnstr, (__data), (__n_iter))

static void test_perf() {
    const int n_vocab = 1 << 17;

    std::vector<llama_token_data> data;

    data.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        const float logit = 2.0f*((double)(rand())/RAND_MAX - 0.5);
        data.emplace_back(llama_token_data{i, logit, 0.0f});
    }

    BENCH(llama_sampler_init_top_k  (40),                     data, 32);
    BENCH(llama_sampler_init_top_p  (0.8f, 1),                data, 32);
    BENCH(llama_sampler_init_min_p  (0.2f, 1),                data, 32);
    BENCH(llama_sampler_init_typical(0.5f, 1),                data, 32);
    BENCH(llama_sampler_init_xtc    (1.0f, 0.1f, 1, 1),       data, 32);
}

int main(void) {
    ggml_time_init();

    test_compact_top_k_behavioral();

    test_temp({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);
    test_temp({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f);

    test_temp_ext({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f, 0.0f, 1.0f);
    test_temp_ext({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f, 0.0f, 1.0f);

    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 1);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 3);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f, 0.1f}, 4);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0);

    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 0);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.571429f, 0.428571f}, 0.7f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 0.8f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);

    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.24f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.26f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.49f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.51f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.74f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  0.76f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.05f);

    printf("XTC should:\n");
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.1f},                                0.99f, 0.09f);
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.2f, 0.1f},                          0.99f, 0.19f);
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.3f, 0.2f, 0.1f},                    0.99f, 0.29f);

    printf("XTC should not:\n");
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.4f, 0.3f, 0.2f, 0.1f},              0.99f, 0.39f);

    test_typical({0.97f, 0.01f, 0.01f, 0.01f}, {0.97f},            0.5f);
    test_typical({0.4f, 0.2f, 0.2f, 0.2f},     {0.2f, 0.2f, 0.2f}, 0.5f);

    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0}, {0, 0.25f, 0.25f, 0.25f, 0.25f},   50.0f, 0.0f, 0.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2}, {0, 0, 0, 0.5f, 0.5f},       50.0f, 0.0f, 0.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0, 0, 0, 0.5f, 0.5f}, 50.0f, 0.0f, 0.0f);

    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0},             {0.000011f, 0.249997f, 0.249997f, 0.249997f, 0.249997f}, 1.0f, 5.0f, 5.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2},       {0.000023f, 0.000023f, 0.000023f, 0.499966f, 0.499966f}, 1.0f, 5.0f, 5.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0.000000f, 0.000023f, 0.000023f, 0.499977f, 0.499977f}, 1.0f, 5.0f, 5.0f);


    test_dry({0.25f, 0.25f, 0.25f, 0.25f}, {0, 1}, {0.25f, 0.25f, 0.25f, 0.25f}, 1.0f, 1.1f, 2, 4, {});
    test_dry({0.25f, 0.25f, 0.25f, 0.25f}, {0, 1, 2, 0, 1}, {0.296923f, 0.296923f, 0.109232f, 0.296923f}, 1.0f, 1.1f, 2, 5, {});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 2, 6, {{3}});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 1}, {0.241818f, 0.241818f, 0.032727f, 0.241818f, 0.241818f}, 2.0f, 1.1f, 2, 5, {});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 4, 7, {});

    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.428571f, 0.571429f}, 1.00f);
    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0.00f); // top_n_sigma == 0 now represents a no-op rather than greedy decoding as of PR#13345
    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 3.00f);

    test_sampler_queue(10000, "k", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "k",     1, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1e-12);

    test_sampler_queue(10000, "k",   100, 1.0000f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0003f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.8000f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0000f, 9997.9f/9999.0f);
    test_sampler_queue(10000, "m", 10000, 1.0000f, 0.1f);

    test_sampler_queue(10000, "kp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "km", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mp", 100, 0.8f, 9997.9f/9999.0f);
    test_sampler_queue(10000, "mp", 100, 0.8f, 0.1f);

    test_sampler_queue(10000, "kpm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "kmp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pkm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pmk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mkp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mpk", 100, 0.8f, 0.1f);

    printf("OK\n");

    test_perf();

    return 0;
}
