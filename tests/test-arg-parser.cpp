#include "arg.h"
#include "common.h"
#include "download.h"
#include "llama.h"

#include <string>
#include <vector>
#include <sstream>
#include <unordered_set>

#undef NDEBUG
#include <cassert>

static void test(void) {
    common_params params;

    printf("test-arg-parser: make sure there is no duplicated arguments in any examples\n\n");
    for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
        try {
            auto ctx_arg = common_params_parser_init(params, (enum llama_example)ex);
            common_params_add_preset_options(ctx_arg.options);
            std::unordered_set<std::string> seen_args;
            std::unordered_set<std::string> seen_env_vars;
            for (const auto & opt : ctx_arg.options) {
                // check for args duplications
                for (const auto & arg : opt.get_args()) {
                    if (seen_args.find(arg) == seen_args.end()) {
                        seen_args.insert(arg);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same argument: %s", arg.c_str());
                        exit(1);
                    }
                }
                // check for env var duplications
                for (const auto & env : opt.get_env()) {
                    if (seen_env_vars.find(env) == seen_env_vars.end()) {
                        seen_env_vars.insert(env);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same env var: %s", env.c_str());
                        exit(1);
                    }
                }

                // exclude spec args from this check
                // ref: https://github.com/ggml-org/llama.cpp/pull/22397
                const bool skip = opt.is_spec;

                // ensure shorter argument precedes longer argument
                if (!skip && opt.args.size() > 1) {
                    const std::string first(opt.args.front());
                    const std::string last(opt.args.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }

                // same check for negated arguments
                if (opt.args_neg.size() > 1) {
                    const std::string first(opt.args_neg.front());
                    const std::string last(opt.args_neg.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter negated argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }
            }
        } catch (std::exception & e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

    std::vector<std::string> argv;

    printf("test-arg-parser: test invalid usage\n\n");

    // missing value
    argv = {"binary_name", "-m"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (int)
    argv = {"binary_name", "-ngl", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (enum)
    argv = {"binary_name", "-sm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    {
        common_params penalty_params;

        argv = {"binary_name", "--repeat-penalty", "0"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "nan"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        const char * penalty_options[] = {"--frequency-penalty", "--presence-penalty"};
        const char * nonfinite_values[] = {"nan", "inf", "-inf"};
        for (const char * option : penalty_options) {
            for (const char * value : nonfinite_values) {
                argv = {"binary_name", option, value};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));
            }
        }
    }

    // non-existence arg in specific example (--draft cannot be used outside llama-speculative)
    argv = {"binary_name", "--draft", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_EMBEDDING));

    argv = {"binary_name", "-lm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "model_file.gguf");

    argv = {"binary_name", "-t", "1234"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cpuparams.n_threads == 1234);

    argv = {"binary_name", "--verbose"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.verbosity > 1);

    argv = {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "abc.gguf");
    assert(params.n_predict == 6789);
    assert(params.n_batch == 9090);

    // --draft cannot be used outside llama-speculative
    argv = {"binary_name", "--spec-draft-n-max", "123"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_max == 123);

    argv = {"binary_name", "-lm", "none"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);

    argv = {"binary_name", "-lm", "mmap"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    argv = {"binary_name", "-lm", "mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    argv = {"binary_name", "-lm", "mmap+mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    {
        common_params tri_params;
        assert(tri_params.triattention_ratio == 3.0 / 32.0);

        argv = {"binary_name", "--triattention-ratio", "0.125"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), tri_params, LLAMA_EXAMPLE_SERVER));
        assert(tri_params.triattention_ratio == 0.125);

        const char * invalid_ratios[] = {"0", "-0.1", "1.1", "nan", "inf", "0.125x"};
        for (const char * ratio : invalid_ratios) {
            common_params invalid_tri_params;
            argv = {"binary_name", "--triattention-ratio", ratio};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), invalid_tri_params, LLAMA_EXAMPLE_SERVER));
        }
    }

    {
        common_params rerot_params;
        assert(!rerot_params.rerot_enabled);
        assert(rerot_params.rerot_frontier == LLAMA_REROT_FRONTIER_STRONG);

        argv = {"binary_name", "--rerot"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), rerot_params, LLAMA_EXAMPLE_SERVER));
        assert(rerot_params.rerot_enabled);
        assert(rerot_params.kv_unified);
        assert(rerot_params.rerot_frontier == LLAMA_REROT_FRONTIER_STRONG);
        rerot_params.n_parallel = 6;
        const auto rerot_cparams = common_context_params_to_llama(rerot_params);
        assert(rerot_cparams.n_seq_max == LLAMA_MAX_SEQ);
        assert(rerot_cparams.n_seq_recurrent == 6);
        assert(rerot_cparams.kv_unified);

        common_params lag1_params;
        argv = {"binary_name", "--rerot-frontier", "lag1"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), lag1_params, LLAMA_EXAMPLE_SERVER));
        assert(lag1_params.rerot_enabled);
        assert(lag1_params.kv_unified);
        assert(lag1_params.rerot_frontier == LLAMA_REROT_FRONTIER_LAG1);

        common_params invalid_rerot_params;
        argv = {"binary_name", "--rerot-frontier", "eventual"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), invalid_rerot_params, LLAMA_EXAMPLE_SERVER));

        // Phase 0: full-auto contract and fail-fast checks (§B.3.1, §B.13)
        common_params full_auto_params;
        argv = {"binary_name", "--rerot", "--total-kv", "auto"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), full_auto_params, LLAMA_EXAMPLE_SERVER));
        assert(full_auto_params.rerot_enabled);
        assert(full_auto_params.n_ctx_kv_auto);
        assert(!full_auto_params.n_parallel_explicit);

        // Conflicting explicit -np with --total-kv auto must fail-fast
        common_params conflict_params;
        argv = {"binary_name", "--rerot", "--total-kv", "auto", "-np", "6"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), conflict_params, LLAMA_EXAMPLE_SERVER));

        // Manual KV size with explicit -np remains supported
        common_params manual_params;
        argv = {"binary_name", "--rerot", "--total-kv", "2048", "-np", "6"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), manual_params, LLAMA_EXAMPLE_SERVER));
        assert(manual_params.rerot_enabled);
        assert(!manual_params.n_ctx_kv_auto);
        assert(manual_params.n_parallel_explicit);
        assert(manual_params.n_parallel == 6);

        // RERoT OFF: -np with --total-kv auto is legal (ordinary server auto-fit)
        common_params off_params;
        argv = {"binary_name", "--total-kv", "auto", "-np", "6"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), off_params, LLAMA_EXAMPLE_SERVER));
        assert(!off_params.rerot_enabled);
        assert(off_params.n_parallel == 6);
    }

    {
        // FlashPrefill V2 policy defaults (PREFILL.md §12): OFF with frozen v1 values.
        common_params fp_defaults;
        assert(fp_defaults.flashprefill.mode == LLAMA_FLASHPREFILL_MODE_OFF);
        assert(fp_defaults.flashprefill.tail_scope == LLAMA_FLASHPREFILL_TAIL_LOGICAL_PROMPT);
        assert(fp_defaults.flashprefill.alpha == LLAMA_FLASHPREFILL_DEFAULT_ALPHA);
        assert(fp_defaults.flashprefill.block_q == LLAMA_FLASHPREFILL_DEFAULT_BLOCK_Q);
        assert(fp_defaults.flashprefill.block_k == LLAMA_FLASHPREFILL_DEFAULT_BLOCK_K);
        assert(fp_defaults.flashprefill.sink_blocks == LLAMA_FLASHPREFILL_DEFAULT_SINK_BLOCKS);
        assert(fp_defaults.flashprefill.window_blocks == LLAMA_FLASHPREFILL_DEFAULT_WINDOW_BLOCKS);
        assert(fp_defaults.flashprefill.dense_tail_tiles == LLAMA_FLASHPREFILL_DEFAULT_DENSE_TAIL_TILES);
        assert(fp_defaults.flashprefill.min_kv == LLAMA_FLASHPREFILL_DEFAULT_MIN_KV);
        assert(fp_defaults.flashprefill.full_attn_layers == LLAMA_FLASHPREFILL_DEFAULT_FULL_ATTN_LAYERS);
        assert(fp_defaults.flashprefill.mean_correction == true);
        assert(fp_defaults.flashprefill.exact_all == false);

        // Default OFF transfers verbatim into versioned context params.
        const auto fp_default_cparams = common_context_params_to_llama(fp_defaults);
        assert(fp_default_cparams.flashprefill.mode == LLAMA_FLASHPREFILL_MODE_OFF);
        assert(fp_default_cparams.flashprefill.version == LLAMA_FLASHPREFILL_CONFIG_VERSION);
        assert(fp_default_cparams.flashprefill.struct_size == sizeof(struct llama_flashprefill_config));
        assert(fp_default_cparams.flashprefill.alpha == LLAMA_FLASHPREFILL_DEFAULT_ALPHA);

        // Mode enum parsing.
        common_params fp_mode;
        argv = {"binary_name", "--flashprefill", "auto"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_mode, LLAMA_EXAMPLE_SERVER));
        assert(fp_mode.flashprefill.mode == LLAMA_FLASHPREFILL_MODE_AUTO);
        // No switch implicitly enables Tri/RERoT or disables MTP routing.
        assert(!fp_mode.triattention_enabled);
        assert(!fp_mode.rerot_enabled);
        assert(fp_mode.speculative.types.size() == 1 && fp_mode.speculative.types[0] == COMMON_SPECULATIVE_TYPE_NONE);

        argv = {"binary_name", "--flashprefill", "required"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_mode, LLAMA_EXAMPLE_SERVER));
        assert(fp_mode.flashprefill.mode == LLAMA_FLASHPREFILL_MODE_REQUIRED);

        argv = {"binary_name", "--flashprefill", "off"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_mode, LLAMA_EXAMPLE_SERVER));
        assert(fp_mode.flashprefill.mode == LLAMA_FLASHPREFILL_MODE_OFF);

        const char * invalid_modes[] = {"", "AUTO", "on", "exact", "sparse", "autox"};
        for (const char * mode : invalid_modes) {
            common_params fp_invalid;
            argv = {"binary_name", "--flashprefill", mode};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_invalid, LLAMA_EXAMPLE_SERVER));
        }

        // Alpha: finite and in (0, 1].
        common_params fp_alpha;
        argv = {"binary_name", "--flashprefill-alpha", "0.5"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_alpha, LLAMA_EXAMPLE_SERVER));
        assert(fp_alpha.flashprefill.alpha == 0.5f);
        argv = {"binary_name", "--flashprefill-alpha", "1"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_alpha, LLAMA_EXAMPLE_SERVER));
        assert(fp_alpha.flashprefill.alpha == 1.0f);

        const char * invalid_alphas[] = {"0", "-0.1", "1.1", "2", "nan", "inf", "-inf", "abc", "", "0.1x"};
        for (const char * alpha : invalid_alphas) {
            common_params fp_invalid;
            argv = {"binary_name", "--flashprefill-alpha", alpha};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_invalid, LLAMA_EXAMPLE_SERVER));
        }

        // BM: positive integers <= 256 (safe signed parse with overflow checks).
        common_params fp_bq;
        for (const char * valid : {"1", "128", "256"}) {
            argv = {"binary_name", "--flashprefill-block-q", valid};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_bq, LLAMA_EXAMPLE_SERVER));
        }
        assert(fp_bq.flashprefill.block_q == 256u);
        const char * invalid_bq[] = {"0", "-1", "257", "1024", "128x", "abc", "", "9999999999999999999999"};
        for (const char * bq : invalid_bq) {
            common_params fp_invalid;
            argv = {"binary_name", "--flashprefill-block-q", bq};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_invalid, LLAMA_EXAMPLE_SERVER));
        }

        // BN: multiples of 64 in [64, 1024].
        common_params fp_bk;
        for (const char * valid : {"64", "128", "1024"}) {
            argv = {"binary_name", "--flashprefill-block-k", valid};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_bk, LLAMA_EXAMPLE_SERVER));
        }
        assert(fp_bk.flashprefill.block_k == 1024u);
        const char * invalid_bk[] = {"0", "1", "32", "63", "65", "100", "1088", "-128", "128x", "abc", "", "9999999999999999999999"};
        for (const char * bk : invalid_bk) {
            common_params fp_invalid;
            argv = {"binary_name", "--flashprefill-block-k", bk};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_invalid, LLAMA_EXAMPLE_SERVER));
        }

        // Non-negative counts: sink, window, dense tail, min-kv, full-attn-layers.
        common_params fp_counts;
        argv = {"binary_name", "--flashprefill-sink-blocks", "0", "--flashprefill-window-blocks", "6",
                "--flashprefill-dense-tail-tiles", "0", "--flashprefill-min-kv", "0",
                "--flashprefill-full-attn-layers", "3"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_counts, LLAMA_EXAMPLE_SERVER));
        assert(fp_counts.flashprefill.sink_blocks == 0u);
        assert(fp_counts.flashprefill.window_blocks == 6u);
        assert(fp_counts.flashprefill.dense_tail_tiles == 0u);
        assert(fp_counts.flashprefill.min_kv == 0u);
        assert(fp_counts.flashprefill.full_attn_layers == 3u);
        // Valid policy values without a mode still preserve OFF.
        assert(fp_counts.flashprefill.mode == LLAMA_FLASHPREFILL_MODE_OFF);

        const char * count_opts[] = {"--flashprefill-sink-blocks", "--flashprefill-window-blocks",
            "--flashprefill-dense-tail-tiles", "--flashprefill-min-kv", "--flashprefill-full-attn-layers"};
        const char * invalid_counts[] = {"-1", "-128", "4x", "abc", "", "99999999999", "9999999999999999999999"};
        for (const char * opt : count_opts) {
            for (const char * count : invalid_counts) {
                common_params fp_invalid;
                argv = {"binary_name", opt, count};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_invalid, LLAMA_EXAMPLE_SERVER));
            }
        }

        // Tail scope enum.
        common_params fp_tail;
        argv = {"binary_name", "--flashprefill-tail-scope", "call"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_tail, LLAMA_EXAMPLE_SERVER));
        assert(fp_tail.flashprefill.tail_scope == LLAMA_FLASHPREFILL_TAIL_CALL);
        argv = {"binary_name", "--flashprefill-tail-scope", "logical-prompt"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_tail, LLAMA_EXAMPLE_SERVER));
        assert(fp_tail.flashprefill.tail_scope == LLAMA_FLASHPREFILL_TAIL_LOGICAL_PROMPT);
        const char * invalid_scopes[] = {"", "prompt", "CALL", "logical_prompt", "callx"};
        for (const char * scope : invalid_scopes) {
            common_params fp_invalid;
            argv = {"binary_name", "--flashprefill-tail-scope", scope};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_invalid, LLAMA_EXAMPLE_SERVER));
        }

        // Mean correction on/off (off is ablation only).
        common_params fp_mean;
        argv = {"binary_name", "--flashprefill-mean-correction", "off"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_mean, LLAMA_EXAMPLE_SERVER));
        assert(fp_mean.flashprefill.mean_correction == false);
        argv = {"binary_name", "--flashprefill-mean-correction", "on"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_mean, LLAMA_EXAMPLE_SERVER));
        assert(fp_mean.flashprefill.mean_correction == true);
        const char * invalid_means[] = {"", "yes", "true", "1", "ON", "offx"};
        for (const char * mean : invalid_means) {
            common_params fp_invalid;
            argv = {"binary_name", "--flashprefill-mean-correction", mean};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_invalid, LLAMA_EXAMPLE_SERVER));
        }

        // Exact-all debug gate.
        common_params fp_exact;
        argv = {"binary_name", "--flashprefill-exact-all"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_exact, LLAMA_EXAMPLE_SERVER));
        assert(fp_exact.flashprefill.exact_all == true);
        assert(fp_exact.flashprefill.mode == LLAMA_FLASHPREFILL_MODE_OFF);

        // Invalid combo: enabled mode with an out-of-range block still fails clearly.
        {
            common_params fp_invalid;
            argv = {"binary_name", "--flashprefill", "required", "--flashprefill-block-k", "100"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_invalid, LLAMA_EXAMPLE_SERVER));
        }
        {
            common_params fp_invalid;
            argv = {"binary_name", "--flashprefill", "auto", "--flashprefill-alpha", "nan"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_invalid, LLAMA_EXAMPLE_SERVER));
        }

        // Enabled config transfers by value into context params.
        {
            common_params fp_on;
            argv = {"binary_name", "--flashprefill", "required", "--flashprefill-alpha", "0.5",
                    "--flashprefill-block-q", "64", "--flashprefill-block-k", "256",
                    "--flashprefill-tail-scope", "call", "--flashprefill-mean-correction", "off",
                    "--flashprefill-exact-all"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), fp_on, LLAMA_EXAMPLE_SERVER));
            const auto fp_on_cparams = common_context_params_to_llama(fp_on);
            assert(fp_on_cparams.flashprefill.mode == LLAMA_FLASHPREFILL_MODE_REQUIRED);
            assert(fp_on_cparams.flashprefill.alpha == 0.5f);
            assert(fp_on_cparams.flashprefill.block_q == 64u);
            assert(fp_on_cparams.flashprefill.block_k == 256u);
            assert(fp_on_cparams.flashprefill.tail_scope == LLAMA_FLASHPREFILL_TAIL_CALL);
            assert(fp_on_cparams.flashprefill.mean_correction == false);
            assert(fp_on_cparams.flashprefill.exact_all == true);
            assert(!fp_on.triattention_enabled);
            assert(!fp_on.rerot_enabled);
        }
    }

    argv = {"binary_name", "-lm", "dio"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    // multi-value args (CSV)
    argv = {"binary_name", "--lora", "file1.gguf,\"file2,2.gguf\",\"file3\"\"3\"\".gguf\",file4\".gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.lora_adapters.size() == 4);
    assert(params.lora_adapters[0].path == "file1.gguf");
    assert(params.lora_adapters[1].path == "file2,2.gguf");
    assert(params.lora_adapters[2].path == "file3\"3\".gguf");
    assert(params.lora_adapters[3].path == "file4\".gguf");

// skip this part on windows, because setenv is not supported
#ifdef _WIN32
    printf("test-arg-parser: skip on windows build\n");
#else
    printf("test-arg-parser: test environment variables (valid + invalid usages)\n\n");

    setenv("LLAMA_ARG_THREADS", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "blah.gguf");
    assert(params.cpuparams.n_threads == 1010);

    setenv("LLAMA_ARG_LOAD_MODE", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_LOAD_MODE", "mmap", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    setenv("LLAMA_ARG_LOAD_MODE", "mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "mmap+mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "dio", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    printf("test-arg-parser: test negated environment variables\n\n");

    setenv("LLAMA_ARG_LOAD_MODE", "none", true);
    setenv("LLAMA_ARG_NO_PERF", "1", true); // legacy format
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);
    assert(params.no_perf == true);

    printf("test-arg-parser: test environment variables being overwritten\n\n");

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name", "-m", "overwritten.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "overwritten.gguf");
    assert(params.cpuparams.n_threads == 1010);
#endif // _WIN32

    printf("test-arg-parser: test download functions\n\n");
    const char * GOOD_URL = "http://ggml.ai/";
    const char * BAD_URL  = "http://ggml.ai/404";

    {
        printf("test-arg-parser: test good URL\n\n");
        auto res = common_remote_get_content(GOOD_URL, {});
        assert(res.first == 200);
        assert(res.second.size() > 0);
        std::string str(res.second.data(), res.second.size());
        assert(str.find("llama.cpp") != std::string::npos);
    }

    {
        printf("test-arg-parser: test bad URL\n\n");
        auto res = common_remote_get_content(BAD_URL, {});
        assert(res.first == 404);
    }

    {
        printf("test-arg-parser: test max size error\n");
        common_remote_params params;
        params.max_size = 1;
        try {
            common_remote_get_content(GOOD_URL, params);
            assert(false && "it should throw an error");
        } catch (std::exception & e) {
            printf("  expected error: %s\n\n", e.what());
        }
    }

    printf("test-arg-parser: all tests OK\n\n");
}

int main(void) {
    try {
        test();
    } catch (std::exception & e) {
        fprintf(stderr, "test-arg-parser: exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
