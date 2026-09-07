#include "arg.h"
#include "common.h"
#include "download.h"
#include "llama.h"

#include <string>
#include <cmath>
#include <cstdlib>
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

        // Phase 7: Auto-fit reserve accounting and worst-case scratch (§A.12, Gate 22 of DoD A.30)
        // Invariant:
        // - RERoT OFF: common_rerot_scratch_reserve_bytes returns 0 (zero overhead, no reserve).
        // - RERoT ON: common_rerot_scratch_reserve_bytes returns worst-case runtime scratch (>= 8 MiB)
        //   covering span-table/DDVR inputs, frontier query rows, parked recurrent metadata,
        //   host episode metadata.
        assert(common_rerot_scratch_reserve_bytes(off_params) == 0);
        assert(common_rerot_scratch_reserve_bytes(full_auto_params) >= 8ull * 1024ull * 1024ull);
        assert(common_rerot_scratch_reserve_bytes(manual_params) >= 8ull * 1024ull * 1024ull);

        // Auto-fit gate validation
        const auto gate_auto = common_rerot_validate_stage0(full_auto_params);
        assert(gate_auto.ok);

        common_params manual_conflict = full_auto_params;
        manual_conflict.n_parallel_explicit = true;
        manual_conflict.n_parallel = 6;
        const auto gate_conflict = common_rerot_validate_stage0(manual_conflict);
        assert(!gate_conflict.ok);
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
        printf("test-arg-parser: test XKV parameters\n\n");

        // 1. Confirm default OFF values and preservation of legacy fields
        {
            common_params p_default;
            assert(p_default.xkv_mode == LLAMA_XKV_MODE_OFF);
            assert(!llama_xkv_is_enabled(p_default.xkv_mode));
            assert(p_default.xkv_storage_profile == LLAMA_XKV_STORAGE_PROFILE_REFERENCE);
            assert(p_default.xkv_group_size == 4);
            assert(p_default.xkv_rank_k == 384);
            assert(p_default.xkv_rank_v == 576);
            assert(p_default.xkv_segment_tokens == 4096);
            assert(p_default.xkv_chunk_tokens == 8);
            assert(p_default.xkv_sr_budget == 0);
            assert(p_default.xkv_source == LLAMA_XKV_SOURCE_DECODED_HOT);
            assert(p_default.xkv_factor_a_k == GGML_TYPE_TURBO4_0);
            assert(p_default.xkv_factor_b_k == GGML_TYPE_TURBO4_0);
            assert(p_default.xkv_factor_a_v == GGML_TYPE_TURBO4_0);
            assert(p_default.xkv_factor_b_v == GGML_TYPE_TURBO4_0);
            assert(p_default.xkv_factor_balance == LLAMA_XKV_FACTOR_BALANCE_UPSTREAM);
            assert(p_default.xkv_landmark_type == GGML_TYPE_Q8_0);
            assert(p_default.xkv_landmark_refine == LLAMA_XKV_LANDMARK_REFINE_NONE);
            assert(p_default.xkv_landmark_refine_max_rows == 64);
            assert(p_default.xkv_workspace_mib == 256);
            assert(p_default.xkv_decode_cache_mib == 64);
            assert(p_default.xkv_store_mib == 0);
            assert(p_default.xkv_seed == 6362273814452121649ULL);
            assert(std::abs(p_default.xkv_min_saving - 0.10) < 1e-6);
            assert(std::abs(p_default.xkv_min_factor_coverage - 0.50) < 1e-6);
            assert(p_default.xkv_factorizer == LLAMA_XKV_FACTORIZER_CPU_REFERENCE);

            // Legacy fields remain unperturbed
            assert(!p_default.kv_unified);
            assert(!p_default.triattention_enabled);
            assert(!p_default.rerot_enabled);
            assert(p_default.cache_type_k == GGML_TYPE_F16);
            assert(p_default.cache_type_v == GGML_TYPE_F16);
            assert(p_default.n_ctx_kv == 0);
            assert(!p_default.n_ctx_kv_auto);

            // Parsing with no XKV options maintains legacy defaults
            common_params p_none;
            argv = {"binary_name"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_none, LLAMA_EXAMPLE_SERVER));
            assert(p_none.xkv_mode == LLAMA_XKV_MODE_OFF);
            assert(!p_none.kv_unified);
            assert(!p_none.triattention_enabled);
            assert(!p_none.rerot_enabled);
            assert(p_none.cache_type_k == GGML_TYPE_F16);
            assert(p_none.cache_type_v == GGML_TYPE_F16);

            // Parsing explicit --xkv off maintains legacy defaults
            common_params p_off;
            argv = {"binary_name", "--xkv", "off"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_off, LLAMA_EXAMPLE_SERVER));
            assert(p_off.xkv_mode == LLAMA_XKV_MODE_OFF);
            assert(!p_off.kv_unified);
            assert(!p_off.triattention_enabled);
            assert(!p_off.rerot_enabled);
            assert(p_off.cache_type_k == GGML_TYPE_F16);
            assert(p_off.cache_type_v == GGML_TYPE_F16);
        }

        // 2. Enabled modes force unified KV
        {
            const char * enabled_modes[] = {"sr", "shadow", "dense"};
            const enum llama_xkv_mode expected_modes[] = {
                LLAMA_XKV_MODE_SR,
                LLAMA_XKV_MODE_SHADOW,
                LLAMA_XKV_MODE_DENSE
            };
            for (size_t i = 0; i < 3; ++i) {
                common_params p_en;
                assert(!p_en.kv_unified);
                argv = {"binary_name", "--xkv", enabled_modes[i]};
                // SR success cases must supply a positive --xkv-sr-budget: the static
                // gate rejects SR with a zero budget. Shadow/dense need none.
                if (expected_modes[i] == LLAMA_XKV_MODE_SR) {
                    argv.push_back("--xkv-sr-budget");
                    argv.push_back("64");
                }
                assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_en, LLAMA_EXAMPLE_SERVER));
                assert(p_en.xkv_mode == expected_modes[i]);
                assert(p_en.kv_unified);

                // Even if --no-kv-unified is explicitly specified, enabling XKV enforces unified KV
                common_params p_no_kvu;
                argv = {"binary_name", "--xkv", enabled_modes[i], "--no-kv-unified"};
                if (expected_modes[i] == LLAMA_XKV_MODE_SR) {
                    argv.push_back("--xkv-sr-budget");
                    argv.push_back("64");
                }
                assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_no_kvu, LLAMA_EXAMPLE_SERVER));
                assert(p_no_kvu.kv_unified);
            }

            // CLI example with -m dummy.gguf also enables and forces unified KV
            common_params p_cli;
            argv = {"binary_name", "-m", "dummy.gguf", "--xkv", "sr", "--xkv-sr-budget", "64"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_cli, LLAMA_EXAMPLE_CLI));
            assert(p_cli.xkv_mode == LLAMA_XKV_MODE_SR);
            assert(p_cli.kv_unified);
        }

        // 3. Last-option behavior: --xkv sr --xkv off leaves no XKV-induced unified side effect
        {
            common_params p_sr_off;
            // NB: no --xkv-sr-budget here: OFF with any non-default flag is
            // fail-closed, so the side-effect probe uses defaults only
            argv = {"binary_name", "--xkv", "sr", "--xkv", "off"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_sr_off, LLAMA_EXAMPLE_SERVER));
            assert(p_sr_off.xkv_mode == LLAMA_XKV_MODE_OFF);
            assert(!p_sr_off.kv_unified);

            common_params p_shadow_off;
            argv = {"binary_name", "--xkv", "shadow", "--xkv", "off"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_shadow_off, LLAMA_EXAMPLE_SERVER));
            assert(p_shadow_off.xkv_mode == LLAMA_XKV_MODE_OFF);
            assert(!p_shadow_off.kv_unified);

            common_params p_dense_off;
            argv = {"binary_name", "--xkv", "dense", "--xkv", "off"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_dense_off, LLAMA_EXAMPLE_SERVER));
            assert(p_dense_off.xkv_mode == LLAMA_XKV_MODE_OFF);
            assert(!p_dense_off.kv_unified);

            // --xkv-off option resets mode and leaves no unified KV side effect
            common_params p_xkv_off_flag;
            argv = {"binary_name", "--xkv", "sr", "--xkv-off"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_xkv_off_flag, LLAMA_EXAMPLE_SERVER));
            assert(p_xkv_off_flag.xkv_mode == LLAMA_XKV_MODE_OFF);
            assert(!p_xkv_off_flag.kv_unified);

            common_params p_xkv_off_flag2;
            argv = {"binary_name", "--xkv-off"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_xkv_off_flag2, LLAMA_EXAMPLE_SERVER));
            assert(p_xkv_off_flag2.xkv_mode == LLAMA_XKV_MODE_OFF);
            assert(!p_xkv_off_flag2.kv_unified);

            // Inverted order: --xkv off --xkv sr leaves XKV enabled and unified KV active
            common_params p_off_sr;
            argv = {"binary_name", "--xkv", "off", "--xkv", "sr", "--xkv-sr-budget", "64"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_off_sr, LLAMA_EXAMPLE_SERVER));
            assert(p_off_sr.xkv_mode == LLAMA_XKV_MODE_SR);
            assert(p_off_sr.kv_unified);

            common_params p_off_flag_sr;
            argv = {"binary_name", "--xkv-off", "--xkv", "sr", "--xkv-sr-budget", "64"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_off_flag_sr, LLAMA_EXAMPLE_SERVER));
            assert(p_off_flag_sr.xkv_mode == LLAMA_XKV_MODE_SR);
            assert(p_off_flag_sr.kv_unified);
        }

        // 4. Explicit --kv-unified surviving OFF
        {
            // Explicit --kv-unified preceding --xkv off
            common_params p_kvu_off;
            argv = {"binary_name", "--kv-unified", "--xkv", "off"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_kvu_off, LLAMA_EXAMPLE_SERVER));
            assert(p_kvu_off.xkv_mode == LLAMA_XKV_MODE_OFF);
            assert(p_kvu_off.kv_unified);

            // Explicit --kv-unified following --xkv off
            common_params p_off_kvu;
            argv = {"binary_name", "--xkv", "off", "--kv-unified"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_off_kvu, LLAMA_EXAMPLE_SERVER));
            assert(p_off_kvu.xkv_mode == LLAMA_XKV_MODE_OFF);
            assert(p_off_kvu.kv_unified);

            // Explicit --kv-unified following --xkv sr --xkv off
            common_params p_sr_off_kvu;
            argv = {"binary_name", "--xkv", "sr", "--xkv", "off", "--kv-unified"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_sr_off_kvu, LLAMA_EXAMPLE_SERVER));
            assert(p_sr_off_kvu.xkv_mode == LLAMA_XKV_MODE_OFF);
            assert(p_sr_off_kvu.kv_unified);

            // Explicit --kv-unified preceding --xkv sr --xkv off
            common_params p_kvu_sr_off;
            argv = {"binary_name", "--kv-unified", "--xkv", "sr", "--xkv", "off"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_kvu_sr_off, LLAMA_EXAMPLE_SERVER));
            assert(p_kvu_sr_off.xkv_mode == LLAMA_XKV_MODE_OFF);
            assert(p_kvu_sr_off.kv_unified);
        }

        // 5. Valid argument parsing for all XKV enums, strings, and codecs
        {
            // Storage profiles
            common_params p_prof1;
            argv = {"binary_name", "--xkv-storage-profile", "reference"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_prof1, LLAMA_EXAMPLE_SERVER));
            assert(p_prof1.xkv_storage_profile == LLAMA_XKV_STORAGE_PROFILE_REFERENCE);

            common_params p_prof2;
            argv = {"binary_name", "--xkv-storage-profile", "tq-factors"};
            // OFF with a non-default profile is fail-closed (§16: no silent ignores)
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_prof2, LLAMA_EXAMPLE_SERVER));

            common_params p_prof3;
            argv = {"binary_name", "--xkv-storage-profile", "tq-factors-landmarks"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_prof3, LLAMA_EXAMPLE_SERVER));

            // Sources
            common_params p_src1;
            argv = {"binary_name", "--xkv-source", "decoded-hot"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_src1, LLAMA_EXAMPLE_SERVER));
            assert(p_src1.xkv_source == LLAMA_XKV_SOURCE_DECODED_HOT);

            common_params p_src2;
            argv = {"binary_name", "--xkv-source", "prerope-capture"};
            // OFF with a non-default source is fail-closed (and prerope-capture
            // is unfinished when enabled: both reject)
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_src2, LLAMA_EXAMPLE_SERVER));

            // Factor balance
            common_params p_bal1;
            argv = {"binary_name", "--xkv-factor-balance", "upstream"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_bal1, LLAMA_EXAMPLE_SERVER));
            assert(p_bal1.xkv_factor_balance == LLAMA_XKV_FACTOR_BALANCE_UPSTREAM);

            common_params p_bal2;
            argv = {"binary_name", "--xkv-factor-balance", "sqrt"};
            // OFF with a non-default balance is fail-closed
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_bal2, LLAMA_EXAMPLE_SERVER));

            common_params p_bal3;
            argv = {"binary_name", "--xkv-factor-balance", "diagonal"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_bal3, LLAMA_EXAMPLE_SERVER));

            // Landmark refine
            common_params p_ref1;
            argv = {"binary_name", "--xkv-landmark-refine", "none"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_ref1, LLAMA_EXAMPLE_SERVER));
            assert(p_ref1.xkv_landmark_refine == LLAMA_XKV_LANDMARK_REFINE_NONE);

            common_params p_ref2;
            argv = {"binary_name", "--xkv-landmark-refine", "boundary"};
            // OFF with a non-default refine mode is fail-closed
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_ref2, LLAMA_EXAMPLE_SERVER));

            // Factorizer backends
            common_params p_fct1;
            argv = {"binary_name", "--xkv-factorizer", "cpu-reference"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_fct1, LLAMA_EXAMPLE_SERVER));
            assert(p_fct1.xkv_factorizer == LLAMA_XKV_FACTORIZER_CPU_REFERENCE);

            common_params p_fct2;
            argv = {"binary_name", "--xkv-factorizer", "vulkan"};
            // OFF with a non-default factorizer is fail-closed (vulkan acceptance
            // with production profiles is covered in §9)
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_fct2, LLAMA_EXAMPLE_SERVER));

            common_params p_fct3;
            argv = {"binary_name", "--xkv-factorizer", "vulkan-hybrid"};
            // Unfinished hybrid backend: fail-closed both OFF (non-default) and enabled
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_fct3, LLAMA_EXAMPLE_SERVER));

            common_params p_fct4;
            argv = {"binary_name", "--xkv-factorizer", "cuda"};
            // Unsupported CUDA backend: fail-closed both OFF (non-default) and enabled
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_fct4, LLAMA_EXAMPLE_SERVER));

            // Codecs for factor streams and landmarks
            common_params p_codecs;
            argv = {"binary_name",
                "--xkv-factor-a-k", "turbo2_0",
                "--xkv-factor-b-k", "turbo3_0",
                "--xkv-factor-a-v", "turbo4_0",
                "--xkv-factor-b-v", "turbo2_0",
                "--xkv-landmark-type", "turbo4_0"
            };
            // OFF with non-default codecs is fail-closed
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_codecs, LLAMA_EXAMPLE_SERVER));

            // Enabled dense + reference: the same codec strings map to parsed types
            common_params p_codecs_en;
            argv = {"binary_name",
                "--xkv", "dense",
                "--xkv-storage-profile", "reference",
                "--xkv-factor-a-k", "turbo2_0",
                "--xkv-factor-b-k", "turbo3_0",
                "--xkv-factor-a-v", "turbo4_0",
                "--xkv-factor-b-v", "turbo2_0",
                "--xkv-landmark-type", "turbo4_0"
            };
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_codecs_en, LLAMA_EXAMPLE_SERVER));
            assert(p_codecs_en.xkv_factor_a_k == GGML_TYPE_TURBO2_0);
            assert(p_codecs_en.xkv_factor_b_k == GGML_TYPE_TURBO3_0);
            assert(p_codecs_en.xkv_factor_a_v == GGML_TYPE_TURBO4_0);
            assert(p_codecs_en.xkv_factor_b_v == GGML_TYPE_TURBO2_0);
            assert(p_codecs_en.xkv_landmark_type == GGML_TYPE_TURBO4_0);

            common_params p_codecs_q8;
            argv = {"binary_name",
                "--xkv-factor-a-k", "q8_0",
                "--xkv-factor-b-k", "f16",
                "--xkv-factor-a-v", "q8_0",
                "--xkv-factor-b-v", "f16",
                "--xkv-landmark-type", "q8_0"
            };
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_codecs_q8, LLAMA_EXAMPLE_SERVER));

            // Test both standard ggml_type names ("turbo4") and aliases ("turbo4_0")
            common_params p_codecs_names;
            argv = {"binary_name",
                "--xkv-factor-a-k", "turbo2",
                "--xkv-factor-b-k", "turbo3",
                "--xkv-factor-a-v", "turbo4",
                "--xkv-factor-b-v", "turbo4",
                "--xkv-landmark-type", "turbo2"
            };
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_codecs_names, LLAMA_EXAMPLE_SERVER));

            // Enabled dense + reference: standard names map identically to aliases
            common_params p_codecs_names_en;
            argv = {"binary_name",
                "--xkv", "dense",
                "--xkv-storage-profile", "reference",
                "--xkv-factor-a-k", "turbo2",
                "--xkv-factor-b-k", "turbo3",
                "--xkv-factor-a-v", "turbo4",
                "--xkv-factor-b-v", "turbo4",
                "--xkv-landmark-type", "turbo4"
            };
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_codecs_names_en, LLAMA_EXAMPLE_SERVER));
            assert(p_codecs_names_en.xkv_factor_a_k == GGML_TYPE_TURBO2_0);
            assert(p_codecs_names_en.xkv_factor_b_k == GGML_TYPE_TURBO3_0);
            assert(p_codecs_names_en.xkv_factor_a_v == GGML_TYPE_TURBO4_0);
            assert(p_codecs_names_en.xkv_factor_b_v == GGML_TYPE_TURBO4_0);
            assert(p_codecs_names_en.xkv_landmark_type == GGML_TYPE_TURBO4_0);

            // Invalid XKV codec name rejected
            common_params p_bad_codec;
            argv = {"binary_name", "--xkv-factor-a-k", "unsupported_codec_xyz"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_bad_codec, LLAMA_EXAMPLE_SERVER));

            // Rejected factor types: q4, q5, bf16
            const char * disallowed_factor_types[] = {"q4_0", "q4_1", "q5_0", "q5_1", "bf16", "iq4_nl"};
            for (const char * dft : disallowed_factor_types) {
                common_params p_disallow_fct;
                argv = {"binary_name", "--xkv-factor-a-k", dft};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_disallow_fct, LLAMA_EXAMPLE_SERVER));
            }

            common_params p_bad_landmark_codec;
            argv = {"binary_name", "--xkv-landmark-type", "unsupported_codec_xyz"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_bad_landmark_codec, LLAMA_EXAMPLE_SERVER));

            // Disallowed landmark types: q4, q5, bf16, turbo2, turbo3
            const char * disallowed_landmark_types[] = {"q4_0", "q4_1", "q5_0", "q5_1", "bf16", "turbo2", "turbo3", "turbo2_0", "turbo3_0"};
            for (const char * dlt : disallowed_landmark_types) {
                common_params p_disallow_lm;
                argv = {"binary_name", "--xkv-landmark-type", dlt};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_disallow_lm, LLAMA_EXAMPLE_SERVER));
            }
        }

        // 6. Valid numeric options
        {
            common_params p_num;
            argv = {"binary_name",
                "--xkv", "dense",
                "--xkv-group-size", "8",
                "--xkv-rank-k", "256",
                "--xkv-rank-v", "512",
                "--xkv-segment-tokens", "2048",
                "--xkv-chunk-tokens", "16",
                "--xkv-sr-budget", "128",
                "--xkv-landmark-refine-max-rows", "32",
                "--xkv-workspace-mib", "512",
                "--xkv-decode-cache-mib", "128",
                "--xkv-store-mib", "256",
                "--xkv-min-saving", "0.25",
                "--xkv-min-factor-coverage", "0.75"
            };
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_num, LLAMA_EXAMPLE_SERVER));
            assert(p_num.xkv_group_size == 8);
            assert(p_num.xkv_rank_k == 256);
            assert(p_num.xkv_rank_v == 512);
            assert(p_num.xkv_segment_tokens == 2048);
            assert(p_num.xkv_chunk_tokens == 16);
            assert(p_num.xkv_sr_budget == 128);
            assert(p_num.xkv_landmark_refine_max_rows == 32);
            assert(p_num.xkv_workspace_mib == 512);
            assert(p_num.xkv_decode_cache_mib == 128);
            assert(p_num.xkv_store_mib == 256);
            assert(std::abs(p_num.xkv_min_saving - 0.25) < 1e-6);
            assert(std::abs(p_num.xkv_min_factor_coverage - 0.75) < 1e-6);

            // Fraction boundary values 0.0 and 1.0
            common_params p_frac_bounds;
            argv = {"binary_name", "--xkv", "dense", "--xkv-min-saving", "0.0", "--xkv-min-factor-coverage", "1.0"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_frac_bounds, LLAMA_EXAMPLE_SERVER));
            assert(std::abs(p_frac_bounds.xkv_min_saving - 0.0) < 1e-6);
            assert(std::abs(p_frac_bounds.xkv_min_factor_coverage - 1.0) < 1e-6);

            // SR budget 0 with XKV off: ignored, parses fine (fail-closed only in sr mode)
            common_params p_budget0;
            argv = {"binary_name", "--xkv-sr-budget", "0"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_budget0, LLAMA_EXAMPLE_SERVER));
            assert(p_budget0.xkv_sr_budget == 0);

            // Persistent store budget rules: OFF or SHADOW with nonzero fails;
            // DENSE/SR with zero parses (auto-derived placeholder).
            common_params p_off_store;
            argv = {"binary_name", "--xkv", "off", "--xkv-store-mib", "256"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_off_store, LLAMA_EXAMPLE_SERVER));
            common_params p_shadow_store;
            argv = {"binary_name", "--xkv", "shadow", "--xkv-store-mib", "256"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_shadow_store, LLAMA_EXAMPLE_SERVER));
            common_params p_dense_auto;
            argv = {"binary_name", "--xkv", "dense", "--xkv-store-mib", "0"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_dense_auto, LLAMA_EXAMPLE_SERVER));
            assert(p_dense_auto.xkv_store_mib == 0);
            common_params p_dense_explicit;
            argv = {"binary_name", "--xkv", "dense", "--xkv-store-mib", "512"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_dense_explicit, LLAMA_EXAMPLE_SERVER));
            assert(p_dense_explicit.xkv_store_mib == 512);
            // SR budget stays fail-closed: 0 in sr mode fails, 0 ignored otherwise.
            common_params p_sr0;
            argv = {"binary_name", "--xkv", "sr", "--xkv-sr-budget", "128", "--xkv-store-mib", "0"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_sr0, LLAMA_EXAMPLE_SERVER));
            common_params p_sr_fail;
            argv = {"binary_name", "--xkv", "sr", "--xkv-sr-budget", "0"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_sr_fail, LLAMA_EXAMPLE_SERVER));
        }

        // 7. Strict rejection of negative / overflow / trailing junk / NaN / inf
        {
            const char * u32_options[] = {
                "--xkv-group-size",
                "--xkv-rank-k",
                "--xkv-rank-v",
                "--xkv-segment-tokens",
                "--xkv-chunk-tokens",
                "--xkv-sr-budget",
                "--xkv-landmark-refine-max-rows",
                "--xkv-workspace-mib",
                "--xkv-decode-cache-mib",
                "--xkv-store-mib",
            };
            const char * invalid_u32_values[] = {
                "-1",
                "-42",
                "+4",
                "4x",
                "128tokens",
                "10.5",
                "nan",
                "inf",
                "-inf",
                "4294967296", // UINT32_MAX + 1
                "99999999999999999999",
            };
            for (const char * opt : u32_options) {
                for (const char * val : invalid_u32_values) {
                    common_params p_bad;
                    argv = {"binary_name", opt, val};
                    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_bad, LLAMA_EXAMPLE_SERVER));
                }
            }

            const char * fraction_options[] = {
                "--xkv-min-saving",
                "--xkv-min-factor-coverage",
            };
            const char * invalid_fraction_values[] = {
                "-0.01",
                "-1.0",
                "1.0001",
                "2.0",
                "0.5x",
                "0.10foo",
                "nan",
                "inf",
                "-inf",
            };
            for (const char * opt : fraction_options) {
                for (const char * val : invalid_fraction_values) {
                    common_params p_bad;
                    argv = {"binary_name", opt, val};
                    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_bad, LLAMA_EXAMPLE_SERVER));
                }
            }

            // Invalid enum values
            const char * invalid_enums[][2] = {
                {"--xkv", "invalid_mode"},
                {"--xkv", "on"},
                {"--xkv", "1"},
                {"--xkv-storage-profile", "invalid_profile"},
                {"--xkv-storage-profile", "turbo"},
                {"--xkv-source", "invalid_source"},
                {"--xkv-source", "hot"},
                {"--xkv-factor-balance", "invalid_balance"},
                {"--xkv-factor-balance", "svd"},
                {"--xkv-landmark-refine", "invalid_refine"},
                {"--xkv-landmark-refine", "all"},
                {"--xkv-factorizer", "invalid_factorizer"},
                {"--xkv-factorizer", "metal"},
            };
            for (const auto & pair : invalid_enums) {
                common_params p_bad;
                argv = {"binary_name", pair[0], pair[1]};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_bad, LLAMA_EXAMPLE_SERVER));
            }
        }

        // 8. Invalid production codec / profile / CUDA combinations (Stage 0 validation)
        {
            // CUDA factorizer rejected when XKV is enabled
            common_params p_cuda_sr;
            argv = {"binary_name", "--xkv", "sr", "--xkv-sr-budget", "128", "--xkv-factorizer", "cuda"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_cuda_sr, LLAMA_EXAMPLE_SERVER));

            common_params p_cuda_shadow;
            argv = {"binary_name", "--xkv", "shadow", "--xkv-factorizer", "cuda"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_cuda_shadow, LLAMA_EXAMPLE_SERVER));

            common_params p_cuda_dense;
            argv = {"binary_name", "--xkv", "dense", "--xkv-factorizer", "cuda"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_cuda_dense, LLAMA_EXAMPLE_SERVER));

            // Production profile tq-factors requires four Turbo factor streams
            const char * non_turbo_types[] = {"f16", "q8_0", "f32"};
            for (const char * t : non_turbo_types) {
                common_params p_bad_ak;
                argv = {"binary_name", "--xkv", "sr", "--xkv-sr-budget", "128", "--xkv-storage-profile", "tq-factors", "--xkv-factor-a-k", t};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_bad_ak, LLAMA_EXAMPLE_SERVER));

                common_params p_bad_bk;
                argv = {"binary_name", "--xkv", "sr", "--xkv-sr-budget", "128", "--xkv-storage-profile", "tq-factors", "--xkv-factor-b-k", t};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_bad_bk, LLAMA_EXAMPLE_SERVER));

                common_params p_bad_av;
                argv = {"binary_name", "--xkv", "sr", "--xkv-sr-budget", "128", "--xkv-storage-profile", "tq-factors", "--xkv-factor-a-v", t};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_bad_av, LLAMA_EXAMPLE_SERVER));

                common_params p_bad_bv;
                argv = {"binary_name", "--xkv", "sr", "--xkv-sr-budget", "128", "--xkv-storage-profile", "tq-factors", "--xkv-factor-b-v", t};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_bad_bv, LLAMA_EXAMPLE_SERVER));
            }

            // Production profile tq-factors-landmarks requires Q8 or Turbo landmark type
            const char * invalid_landmark_types[] = {"f16", "f32"};
            for (const char * lm : invalid_landmark_types) {
                common_params p_bad_lm;
                argv = {"binary_name", "--xkv", "sr", "--xkv-sr-budget", "128", "--xkv-storage-profile", "tq-factors-landmarks", "--xkv-landmark-type", lm};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_bad_lm, LLAMA_EXAMPLE_SERVER));
            }

            // Valid landmark types for tq-factors-landmarks: exactly q8_0 or turbo4_0
            const char * valid_landmark_types[] = {"q8_0", "turbo4_0", "turbo4"};
            for (const char * lm : valid_landmark_types) {
                common_params p_ok_lm;
                argv = {"binary_name", "--xkv", "sr", "--xkv-sr-budget", "128", "--xkv-storage-profile", "tq-factors-landmarks", "--xkv-landmark-type", lm};
                assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_ok_lm, LLAMA_EXAMPLE_SERVER));
                assert(p_ok_lm.xkv_storage_profile == LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS);
            }

            // Profile 'reference' allows non-turbo factor streams
            common_params p_ref_ok;
            argv = {"binary_name", "--xkv", "sr", "--xkv-sr-budget", "128", "--xkv-storage-profile", "reference",
                "--xkv-factor-a-k", "f16", "--xkv-factor-b-k", "f16",
                "--xkv-factor-a-v", "f16", "--xkv-factor-b-v", "f16",
                "--xkv-landmark-type", "f16"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_ref_ok, LLAMA_EXAMPLE_SERVER));
            assert(p_ref_ok.xkv_storage_profile == LLAMA_XKV_STORAGE_PROFILE_REFERENCE);
            assert(p_ref_ok.xkv_factor_a_k == GGML_TYPE_F16);

            // Validate valid profiles R0, R1, R2, R3, R4 (§8.5, §15.4)
            // R0: reference profile, FP32/FP16 A/B, FP16 landmark
            common_params p_r0;
            argv = {"binary_name", "--xkv", "dense", "--xkv-storage-profile", "reference",
                "--xkv-factor-a-k", "f16", "--xkv-factor-b-k", "f16",
                "--xkv-factor-a-v", "f16", "--xkv-factor-b-v", "f16",
                "--xkv-landmark-type", "f16"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_r0, LLAMA_EXAMPLE_SERVER));
            assert(common_xkv_validate_stage0(p_r0).ok);

            // R1: reference profile with mixed A=TQ4, B=FP16/Q8 (ablation)
            common_params p_r1;
            argv = {"binary_name", "--xkv", "dense", "--xkv-storage-profile", "reference",
                "--xkv-factor-a-k", "turbo4_0", "--xkv-factor-b-k", "f16",
                "--xkv-factor-a-v", "turbo4_0", "--xkv-factor-b-v", "q8_0",
                "--xkv-landmark-type", "f16"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_r1, LLAMA_EXAMPLE_SERVER));
            assert(common_xkv_validate_stage0(p_r1).ok);

            // R2: tq-factors profile, all four Turbo4, FP16 landmark allowed
            common_params p_r2;
            argv = {"binary_name", "--xkv", "dense", "--xkv-storage-profile", "tq-factors",
                "--xkv-factor-a-k", "turbo4_0", "--xkv-factor-b-k", "turbo4_0",
                "--xkv-factor-a-v", "turbo4_0", "--xkv-factor-b-v", "turbo4_0",
                "--xkv-landmark-type", "f16"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_r2, LLAMA_EXAMPLE_SERVER));
            assert(common_xkv_validate_stage0(p_r2).ok);

            // R3: tq-factors-landmarks profile, all four Turbo4, Q8_0 landmark
            common_params p_r3;
            argv = {"binary_name", "--xkv", "sr", "--xkv-sr-budget", "64", "--xkv-storage-profile", "tq-factors-landmarks",
                "--xkv-factor-a-k", "turbo4_0", "--xkv-factor-b-k", "turbo4_0",
                "--xkv-factor-a-v", "turbo4_0", "--xkv-factor-b-v", "turbo4_0",
                "--xkv-landmark-type", "q8_0"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_r3, LLAMA_EXAMPLE_SERVER));
            assert(common_xkv_validate_stage0(p_r3).ok);

            // R4: tq-factors-landmarks profile, four independent Turbo2/3/4 streams, Turbo4 landmark
            common_params p_r4;
            argv = {"binary_name", "--xkv", "sr", "--xkv-sr-budget", "32", "--xkv-storage-profile", "tq-factors-landmarks",
                "--xkv-factor-a-k", "turbo2_0", "--xkv-factor-b-k", "turbo3_0",
                "--xkv-factor-a-v", "turbo4_0", "--xkv-factor-b-v", "turbo2_0",
                "--xkv-landmark-type", "turbo4_0"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_r4, LLAMA_EXAMPLE_SERVER));
            assert(common_xkv_validate_stage0(p_r4).ok);

            // Programmatic validation gate edge cases:
        {
                common_params prog_p = p_r3;
                // Invalid enum modes
                prog_p.xkv_mode = (enum llama_xkv_mode) 99;
                assert(!common_xkv_validate_stage0(prog_p).ok);
                prog_p = p_r3;
                prog_p.xkv_storage_profile = (enum llama_xkv_storage_profile) 99;
                assert(!common_xkv_validate_stage0(prog_p).ok);
                prog_p = p_r3;
                prog_p.xkv_source = (enum llama_xkv_source) 99;
                assert(!common_xkv_validate_stage0(prog_p).ok);
                prog_p = p_r3;
                prog_p.xkv_factor_balance = (enum llama_xkv_factor_balance) 99;
                assert(!common_xkv_validate_stage0(prog_p).ok);
                prog_p = p_r3;
                prog_p.xkv_landmark_refine = (enum llama_xkv_landmark_refine) 99;
                assert(!common_xkv_validate_stage0(prog_p).ok);
                prog_p = p_r3;
                prog_p.xkv_factorizer = (enum llama_xkv_factorizer) 99;
                assert(!common_xkv_validate_stage0(prog_p).ok);

                // Programmatic factor type validation
                prog_p = p_r3;
                prog_p.xkv_factor_a_k = (ggml_type) 999;
                assert(!common_xkv_validate_stage0(prog_p).ok);
                prog_p = p_r3;
                prog_p.xkv_factor_a_k = GGML_TYPE_Q4_0;
                assert(!common_xkv_validate_stage0(prog_p).ok);
                prog_p = p_r3;
                prog_p.xkv_landmark_type = GGML_TYPE_Q4_0;
                assert(!common_xkv_validate_stage0(prog_p).ok);

                // Unified KV missing
                prog_p = p_r3;
                prog_p.kv_unified = false;
                assert(!common_xkv_validate_stage0(prog_p).ok);

                // Boundary conditions: chunk > segment
                prog_p = p_r3;
                prog_p.xkv_chunk_tokens = prog_p.xkv_segment_tokens + 1;
                assert(!common_xkv_validate_stage0(prog_p).ok);

                // SR mode sr_budget == 0
                prog_p = p_r3;
                prog_p.xkv_sr_budget = 0;
                assert(!common_xkv_validate_stage0(prog_p).ok);

                // Landmark boundary refinement max_rows == 0
                prog_p = p_r3;
                prog_p.xkv_landmark_refine = LLAMA_XKV_LANDMARK_REFINE_BOUNDARY;
                prog_p.xkv_landmark_refine_max_rows = 0;
                assert(!common_xkv_validate_stage0(prog_p).ok);

                // Decode cache > workspace
                prog_p = p_r3;
                prog_p.xkv_decode_cache_mib = prog_p.xkv_workspace_mib + 1;
                assert(!common_xkv_validate_stage0(prog_p).ok);

                // Non-finite fractions
                prog_p = p_r3;
                prog_p.xkv_min_saving = NAN;
                assert(!common_xkv_validate_stage0(prog_p).ok);
                prog_p = p_r3;
                prog_p.xkv_min_factor_coverage = INFINITY;
                assert(!common_xkv_validate_stage0(prog_p).ok);

                // Round-trip to llama_context_params via common_context_params_to_llama
                prog_p = p_r3;
                struct llama_context_params cparams = common_context_params_to_llama(prog_p);
                assert(cparams.kv_unified);
                assert(cparams.xkv_mode == prog_p.xkv_mode);
                assert(cparams.xkv_storage_profile == prog_p.xkv_storage_profile);
                assert(cparams.xkv_group_size == prog_p.xkv_group_size);
                assert(cparams.xkv_rank_k == prog_p.xkv_rank_k);
                assert(cparams.xkv_rank_v == prog_p.xkv_rank_v);
                assert(cparams.xkv_segment_tokens == prog_p.xkv_segment_tokens);
                assert(cparams.xkv_chunk_tokens == prog_p.xkv_chunk_tokens);
                assert(cparams.xkv_sr_budget == prog_p.xkv_sr_budget);
                assert(cparams.xkv_source == prog_p.xkv_source);
                assert(cparams.xkv_factor_a_k == prog_p.xkv_factor_a_k);
                assert(cparams.xkv_factor_b_k == prog_p.xkv_factor_b_k);
                assert(cparams.xkv_factor_a_v == prog_p.xkv_factor_a_v);
                assert(cparams.xkv_factor_b_v == prog_p.xkv_factor_b_v);
                assert(cparams.xkv_factor_balance == prog_p.xkv_factor_balance);
                assert(cparams.xkv_landmark_type == prog_p.xkv_landmark_type);
                assert(cparams.xkv_landmark_refine == prog_p.xkv_landmark_refine);
                assert(cparams.xkv_landmark_refine_max_rows == prog_p.xkv_landmark_refine_max_rows);
                assert(cparams.xkv_workspace_mib == prog_p.xkv_workspace_mib);
                assert(cparams.xkv_decode_cache_mib == prog_p.xkv_decode_cache_mib);
                assert(cparams.xkv_store_mib == prog_p.xkv_store_mib);
                assert(cparams.xkv_seed == prog_p.xkv_seed);
                assert(std::abs(cparams.xkv_min_saving - prog_p.xkv_min_saving) < 1e-6);
                assert(std::abs(cparams.xkv_min_factor_coverage - prog_p.xkv_min_factor_coverage) < 1e-6);
                assert(cparams.xkv_factorizer == prog_p.xkv_factorizer);
            }

            // Chunk tokens must be <= segment tokens
            common_params p_chunk_gt_seg;
            argv = {"binary_name", "--xkv", "dense", "--xkv-segment-tokens", "128", "--xkv-chunk-tokens", "129"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_chunk_gt_seg, LLAMA_EXAMPLE_SERVER));

            common_params p_chunk_eq_seg;
            argv = {"binary_name", "--xkv", "dense", "--xkv-segment-tokens", "128", "--xkv-chunk-tokens", "128"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_chunk_eq_seg, LLAMA_EXAMPLE_SERVER));
            assert(p_chunk_eq_seg.xkv_chunk_tokens == 128);

            // SR mode requires sr_budget > 0
            common_params p_sr_zero_budget;
            argv = {"binary_name", "--xkv", "sr", "--xkv-sr-budget", "0"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_sr_zero_budget, LLAMA_EXAMPLE_SERVER));

            common_params p_dense_zero_budget;
            argv = {"binary_name", "--xkv", "dense", "--xkv-sr-budget", "0"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_dense_zero_budget, LLAMA_EXAMPLE_SERVER));

            // Landmark boundary refinement requires max_rows > 0
            common_params p_refine_zero_rows;
            argv = {"binary_name", "--xkv", "dense", "--xkv-landmark-refine", "boundary", "--xkv-landmark-refine-max-rows", "0"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_refine_zero_rows, LLAMA_EXAMPLE_SERVER));

            common_params p_refine_none_zero_rows;
            argv = {"binary_name", "--xkv", "dense", "--xkv-landmark-refine", "none", "--xkv-landmark-refine-max-rows", "0"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_refine_none_zero_rows, LLAMA_EXAMPLE_SERVER));

            // Enabled sizes must be positive (> 0)
            const char * positive_only_opts[] = {
                "--xkv-group-size",
                "--xkv-rank-k",
                "--xkv-rank-v",
                "--xkv-segment-tokens",
                "--xkv-chunk-tokens",
                "--xkv-workspace-mib",
            };
            for (const char * opt : positive_only_opts) {
                common_params p_zero;
                argv = {"binary_name", "--xkv", "dense", opt, "0"};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_zero, LLAMA_EXAMPLE_SERVER));
            }

            // Decode cache size must be <= workspace size
            common_params p_cache_too_large;
            argv = {"binary_name", "--xkv", "dense", "--xkv-workspace-mib", "64", "--xkv-decode-cache-mib", "128"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_cache_too_large, LLAMA_EXAMPLE_SERVER));

            common_params p_cache_equal;
            argv = {"binary_name", "--xkv", "dense", "--xkv-workspace-mib", "64", "--xkv-decode-cache-mib", "64"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_cache_equal, LLAMA_EXAMPLE_SERVER));
            assert(p_cache_equal.xkv_workspace_mib == 64);
            assert(p_cache_equal.xkv_decode_cache_mib == 64);

            // Scoping: XKV options are scoped to SERVER and CLI, not available in EMBEDDING
            common_params p_embed;
            argv = {"binary_name", "--xkv", "dense"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_embed, LLAMA_EXAMPLE_EMBEDDING));
        }

        // 9. Focused profile/backend/source/residency/OFF-default/boundary gates (§16)
        {
            // SR + tq-factors rejected even with all-Turbo codecs: SR selects via
            // landmarks, and tq-factors has none
            common_params p_sr_tqf;
            argv = {"binary_name", "--xkv", "sr", "--xkv-sr-budget", "64",
                "--xkv-storage-profile", "tq-factors",
                "--xkv-factor-a-k", "turbo4_0", "--xkv-factor-b-k", "turbo4_0",
                "--xkv-factor-a-v", "turbo4_0", "--xkv-factor-b-v", "turbo4_0"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_sr_tqf, LLAMA_EXAMPLE_SERVER));
            {
                common_params prog = p_sr_tqf;
                prog.xkv_mode = LLAMA_XKV_MODE_SR;
                prog.xkv_sr_budget = 64;
                prog.xkv_storage_profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS;
                prog.kv_unified = true;
                const common_xkv_gate gate = common_xkv_validate_stage0(prog);
                assert(!gate.ok);
                assert(gate.error == "XKV SR mode requires tq-factors-landmarks storage profile (tq-factors has no landmarks)");
            }

            // Reference SR + FP16 accepted (nominal §2.3 integer payload math path)
            common_params p_sr_ref_fp16;
            argv = {"binary_name", "--xkv", "sr", "--xkv-sr-budget", "64",
                "--xkv-storage-profile", "reference",
                "--xkv-factor-a-k", "f16", "--xkv-factor-b-k", "f16",
                "--xkv-factor-a-v", "f16", "--xkv-factor-b-v", "f16",
                "--xkv-landmark-type", "f16"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_sr_ref_fp16, LLAMA_EXAMPLE_SERVER));
            assert(p_sr_ref_fp16.xkv_mode == LLAMA_XKV_MODE_SR);
            assert(p_sr_ref_fp16.kv_unified);
            assert(common_xkv_validate_stage0(p_sr_ref_fp16).ok);

            // Vulkan production profiles accepted: dense tq-factors + vulkan
            common_params p_vk_tqf;
            argv = {"binary_name", "--xkv", "dense",
                "--xkv-storage-profile", "tq-factors",
                "--xkv-factorizer", "vulkan",
                "--xkv-factor-a-k", "turbo4_0", "--xkv-factor-b-k", "turbo4_0",
                "--xkv-factor-a-v", "turbo4_0", "--xkv-factor-b-v", "turbo4_0"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_vk_tqf, LLAMA_EXAMPLE_SERVER));
            assert(p_vk_tqf.xkv_factorizer == LLAMA_XKV_FACTORIZER_VULKAN);
            assert(common_xkv_validate_stage0(p_vk_tqf).ok);

            // Vulkan production profiles accepted: SR tq-factors-landmarks + vulkan
            common_params p_vk_sr_lm;
            argv = {"binary_name", "--xkv", "sr", "--xkv-sr-budget", "32",
                "--xkv-storage-profile", "tq-factors-landmarks",
                "--xkv-factorizer", "vulkan",
                "--xkv-factor-a-k", "turbo2_0", "--xkv-factor-b-k", "turbo3_0",
                "--xkv-factor-a-v", "turbo4_0", "--xkv-factor-b-v", "turbo2_0",
                "--xkv-landmark-type", "q8_0"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_vk_sr_lm, LLAMA_EXAMPLE_SERVER));
            assert(common_xkv_validate_stage0(p_vk_sr_lm).ok);

            // Reference + vulkan residency mismatch rejected
            common_params p_ref_vk;
            argv = {"binary_name", "--xkv", "dense",
                "--xkv-storage-profile", "reference", "--xkv-factorizer", "vulkan"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_ref_vk, LLAMA_EXAMPLE_SERVER));
            {
                common_params prog = p_vk_tqf;
                prog.xkv_storage_profile = LLAMA_XKV_STORAGE_PROFILE_REFERENCE;
                const common_xkv_gate gate = common_xkv_validate_stage0(prog);
                assert(!gate.ok);
                assert(gate.error == "XKV reference profile requires cpu-reference factorizer (device residency not supported for reference)");
            }

            // Unfinished vulkan-hybrid backend fail-closed when enabled
            common_params p_hybrid;
            argv = {"binary_name", "--xkv", "dense", "--xkv-factorizer", "vulkan-hybrid"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_hybrid, LLAMA_EXAMPLE_SERVER));
            {
                common_params prog = p_vk_tqf;
                prog.xkv_factorizer = LLAMA_XKV_FACTORIZER_VULKAN_HYBRID;
                const common_xkv_gate gate = common_xkv_validate_stage0(prog);
                assert(!gate.ok);
                assert(gate.error == "XKV vulkan-hybrid factorizer is not yet implemented; rejected");
            }

            // Unfinished CUDA backend fail-closed when enabled (exact error pinned)
            {
                common_params prog = p_vk_tqf;
                prog.xkv_factorizer = LLAMA_XKV_FACTORIZER_CUDA;
                const common_xkv_gate gate = common_xkv_validate_stage0(prog);
                assert(!gate.ok);
                assert(gate.error == "XKV CUDA factorizer is not supported; rejected");
            }

            // Unfinished prerope-capture source fail-closed when enabled
            common_params p_prerope;
            argv = {"binary_name", "--xkv", "dense", "--xkv-source", "prerope-capture"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_prerope, LLAMA_EXAMPLE_SERVER));
            {
                common_params prog = p_vk_tqf;
                prog.xkv_factorizer = LLAMA_XKV_FACTORIZER_CPU_REFERENCE;
                prog.xkv_source = LLAMA_XKV_SOURCE_PREROPE_CAPTURE;
                const common_xkv_gate gate = common_xkv_validate_stage0(prog);
                assert(!gate.ok);
                assert(gate.error == "XKV prerope-capture source is not yet implemented; rejected");
            }

            // OFF with explicit defaults accepted (no-op, no allocation)
            common_params p_off_defaults;
            argv = {"binary_name", "--xkv", "off",
                "--xkv-storage-profile", "reference",
                "--xkv-group-size", "4",
                "--xkv-rank-k", "384", "--xkv-rank-v", "576",
                "--xkv-segment-tokens", "4096", "--xkv-chunk-tokens", "8",
                "--xkv-sr-budget", "0",
                "--xkv-source", "decoded-hot",
                "--xkv-factor-balance", "upstream",
                "--xkv-landmark-refine", "none",
                "--xkv-landmark-refine-max-rows", "64",
                "--xkv-workspace-mib", "256", "--xkv-decode-cache-mib", "64",
                "--xkv-store-mib", "0",
                "--xkv-seed", "6362273814452121649",
                "--xkv-min-saving", "0.10", "--xkv-min-factor-coverage", "0.50",
                "--xkv-factorizer", "cpu-reference"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_off_defaults, LLAMA_EXAMPLE_SERVER));
            assert(p_off_defaults.xkv_mode == LLAMA_XKV_MODE_OFF);
            assert(!p_off_defaults.kv_unified);
            assert(common_xkv_validate_stage0(p_off_defaults).ok);

            // OFF with any non-default option rejected (representative flags)
            const char * off_non_default[][2] = {
                {"--xkv-storage-profile", "tq-factors"},
                {"--xkv-group-size", "8"},
                {"--xkv-rank-k", "256"},
                {"--xkv-rank-v", "256"},
                {"--xkv-segment-tokens", "2048"},
                {"--xkv-chunk-tokens", "16"},
                {"--xkv-sr-budget", "32"},
                {"--xkv-source", "prerope-capture"},
                {"--xkv-factor-a-k", "q8_0"},
                {"--xkv-factor-balance", "sqrt"},
                {"--xkv-landmark-type", "turbo4_0"},
                {"--xkv-landmark-refine", "boundary"},
                {"--xkv-landmark-refine-max-rows", "32"},
                {"--xkv-workspace-mib", "512"},
                {"--xkv-decode-cache-mib", "128"},
                {"--xkv-min-saving", "0.25"},
                {"--xkv-min-factor-coverage", "0.75"},
                {"--xkv-seed", "12345"},
                {"--xkv-factorizer", "vulkan"},
            };
            for (const auto & pair : off_non_default) {
                common_params p_off_bad;
                argv = {"binary_name", "--xkv", "off", pair[0], pair[1]};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_off_bad, LLAMA_EXAMPLE_SERVER));
            }

            // Enabled UINT32_MAX boundary parses (positive-size gate passes)
            common_params p_u32max;
            argv = {"binary_name", "--xkv", "dense", "--xkv-group-size", "4294967295"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_u32max, LLAMA_EXAMPLE_SERVER));
            assert(p_u32max.xkv_group_size == 4294967295u);

            // Enabled workspace/cache Residency boundary: off-by-one over fails
            common_params p_ws_plus_one;
            argv = {"binary_name", "--xkv", "dense", "--xkv-workspace-mib", "64", "--xkv-decode-cache-mib", "65"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_ws_plus_one, LLAMA_EXAMPLE_SERVER));

            // Enabled decode cache zero stays allowed (only cache > workspace fails)
            common_params p_cache_zero;
            argv = {"binary_name", "--xkv", "dense", "--xkv-workspace-mib", "64", "--xkv-decode-cache-mib", "0"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_cache_zero, LLAMA_EXAMPLE_SERVER));
            assert(p_cache_zero.xkv_decode_cache_mib == 0);

            // SR chunk == segment boundary accepted
            common_params p_sr_chunk_eq;
            argv = {"binary_name", "--xkv", "sr", "--xkv-sr-budget", "16",
                "--xkv-storage-profile", "tq-factors-landmarks",
                "--xkv-segment-tokens", "128", "--xkv-chunk-tokens", "128"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_sr_chunk_eq, LLAMA_EXAMPLE_SERVER));
            assert(p_sr_chunk_eq.xkv_chunk_tokens == 128);
        }

        // 10. Focused --xkv-seed tests (default, boundaries, strict parsing, OFF refusal, roundtrip)
        {
            // Exact default value is 6362273814452121649ULL (0x584b565352303031 / "XKVSR001")
            common_params p_seed_def;
            argv = {"binary_name", "--xkv", "dense"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_seed_def, LLAMA_EXAMPLE_SERVER));
            assert(p_seed_def.xkv_seed == 6362273814452121649ULL);

            // Explicit seed roundtrip under enabled mode
            common_params p_seed_custom;
            argv = {"binary_name", "--xkv", "dense", "--xkv-seed", "133742"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_seed_custom, LLAMA_EXAMPLE_SERVER));
            assert(p_seed_custom.xkv_seed == 133742ULL);
            struct llama_context_params cparams_custom = common_context_params_to_llama(p_seed_custom);
            assert(cparams_custom.xkv_seed == 133742ULL);

            // Boundary 0 is valid policy
            common_params p_seed_zero;
            argv = {"binary_name", "--xkv", "dense", "--xkv-seed", "0"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_seed_zero, LLAMA_EXAMPLE_SERVER));
            assert(p_seed_zero.xkv_seed == 0ULL);

            // Boundary UINT64_MAX (18446744073709551615) is valid policy
            common_params p_seed_max;
            argv = {"binary_name", "--xkv", "dense", "--xkv-seed", "18446744073709551615"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_seed_max, LLAMA_EXAMPLE_SERVER));
            assert(p_seed_max.xkv_seed == 18446744073709551615ULL);

            // Exact default passed explicitly is accepted even under OFF
            common_params p_seed_off_default;
            argv = {"binary_name", "--xkv", "off", "--xkv-seed", "6362273814452121649"};
            assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_seed_off_default, LLAMA_EXAMPLE_SERVER));
            assert(p_seed_off_default.xkv_seed == 6362273814452121649ULL);

            // Non-default seed under OFF is refused with exact error
            common_params p_seed_off_bad;
            argv = {"binary_name", "--xkv", "off", "--xkv-seed", "42"};
            assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_seed_off_bad, LLAMA_EXAMPLE_SERVER));
            {
                common_params prog;
                prog.xkv_mode = LLAMA_XKV_MODE_OFF;
                prog.xkv_seed = 42ULL;
                const common_xkv_gate gate = common_xkv_validate_stage0(prog);
                assert(!gate.ok);
                assert(gate.error == "XKV is OFF but non-default --xkv-seed was provided");
            }

            // Strict parsing rejections: negative, signed positive, trailing junk, overflow, float, non-digit, empty
            const char * invalid_seed_values[] = {
                "-1",
                "-42",
                "+1",
                "+6362273814452121649",
                "18446744073709551616", // UINT64_MAX + 1
                "99999999999999999999999999",
                "42x",
                "seed",
                "10.5",
                "1e10",
                "nan",
                "inf",
                "-inf",
                " 42",
                "42 ",
                "",
            };
            for (const char * bad_val : invalid_seed_values) {
                common_params p_bad_seed;
                argv = {"binary_name", "--xkv", "dense", "--xkv-seed", bad_val};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_bad_seed, LLAMA_EXAMPLE_SERVER));
            }
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

    printf("test-arg-parser: test XKV environment variables (valid + invalid usages)\n\n");

    // All XKV environment variables set valid configuration
    setenv("LLAMA_ARG_XKV", "sr", true);
    setenv("LLAMA_ARG_XKV_STORAGE_PROFILE", "tq-factors-landmarks", true);
    setenv("LLAMA_ARG_XKV_GROUP_SIZE", "8", true);
    setenv("LLAMA_ARG_XKV_RANK_K", "256", true);
    setenv("LLAMA_ARG_XKV_RANK_V", "512", true);
    setenv("LLAMA_ARG_XKV_SEGMENT_TOKENS", "2048", true);
    setenv("LLAMA_ARG_XKV_CHUNK_TOKENS", "16", true);
    setenv("LLAMA_ARG_XKV_SR_BUDGET", "32", true);
    setenv("LLAMA_ARG_XKV_SOURCE", "decoded-hot", true);
    setenv("LLAMA_ARG_XKV_FACTOR_A_K", "turbo2_0", true);
    setenv("LLAMA_ARG_XKV_FACTOR_B_K", "turbo4_0", true);
    setenv("LLAMA_ARG_XKV_FACTOR_A_V", "turbo3_0", true);
    setenv("LLAMA_ARG_XKV_FACTOR_B_V", "turbo2_0", true);
    setenv("LLAMA_ARG_XKV_FACTOR_BALANCE", "sqrt", true);
    setenv("LLAMA_ARG_XKV_LANDMARK_TYPE", "turbo4_0", true);
    setenv("LLAMA_ARG_XKV_LANDMARK_REFINE", "boundary", true);
    setenv("LLAMA_ARG_XKV_LANDMARK_REFINE_MAX_ROWS", "32", true);
    setenv("LLAMA_ARG_XKV_WORKSPACE_MIB", "512", true);
    setenv("LLAMA_ARG_XKV_DECODE_CACHE_MIB", "128", true);
    setenv("LLAMA_ARG_XKV_STORE_MIB", "256", true);
    setenv("LLAMA_ARG_XKV_SEED", "42424242", true);
    setenv("LLAMA_ARG_XKV_MIN_SAVING", "0.20", true);
    setenv("LLAMA_ARG_XKV_MIN_FACTOR_COVERAGE", "0.60", true);
    setenv("LLAMA_ARG_XKV_FACTORIZER", "vulkan", true);

    argv = {"binary_name"};
    common_params xkv_env_params;
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), xkv_env_params, LLAMA_EXAMPLE_SERVER));
    assert(xkv_env_params.xkv_mode == LLAMA_XKV_MODE_SR);
    assert(xkv_env_params.kv_unified); // Environment-enabled XKV forces unified KV
    assert(xkv_env_params.xkv_storage_profile == LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS);
    assert(xkv_env_params.xkv_group_size == 8);
    assert(xkv_env_params.xkv_rank_k == 256);
    assert(xkv_env_params.xkv_rank_v == 512);
    assert(xkv_env_params.xkv_segment_tokens == 2048);
    assert(xkv_env_params.xkv_chunk_tokens == 16);
    assert(xkv_env_params.xkv_sr_budget == 32);
    assert(xkv_env_params.xkv_source == LLAMA_XKV_SOURCE_DECODED_HOT);
    assert(xkv_env_params.xkv_factor_a_k == GGML_TYPE_TURBO2_0);
    assert(xkv_env_params.xkv_factor_b_k == GGML_TYPE_TURBO4_0);
    assert(xkv_env_params.xkv_factor_a_v == GGML_TYPE_TURBO3_0);
    assert(xkv_env_params.xkv_factor_b_v == GGML_TYPE_TURBO2_0);
    assert(xkv_env_params.xkv_factor_balance == LLAMA_XKV_FACTOR_BALANCE_SQRT);
    assert(xkv_env_params.xkv_landmark_type == GGML_TYPE_TURBO4_0);
    assert(xkv_env_params.xkv_landmark_refine == LLAMA_XKV_LANDMARK_REFINE_BOUNDARY);
    assert(xkv_env_params.xkv_landmark_refine_max_rows == 32);
    assert(xkv_env_params.xkv_workspace_mib == 512);
    assert(xkv_env_params.xkv_decode_cache_mib == 128);
    assert(xkv_env_params.xkv_store_mib == 256);
    assert(xkv_env_params.xkv_seed == 42424242ULL);
    assert(std::abs(xkv_env_params.xkv_min_saving - 0.20) < 1e-6);
    assert(std::abs(xkv_env_params.xkv_min_factor_coverage - 0.60) < 1e-6);
    assert(xkv_env_params.xkv_factorizer == LLAMA_XKV_FACTORIZER_VULKAN);

    // Overwriting environment variables via CLI
    // CLI overrides an env-provided value under an enabled mode (OFF with any
    // non-default flag is fail-closed, so the override target stays enabled)
    argv = {"binary_name", "--xkv", "dense", "--xkv-group-size", "16"};
    common_params xkv_ovr_params;
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), xkv_ovr_params, LLAMA_EXAMPLE_SERVER));
    assert(xkv_ovr_params.xkv_mode == LLAMA_XKV_MODE_DENSE);
    assert(xkv_ovr_params.kv_unified);
    assert(xkv_ovr_params.xkv_group_size == 16);

    // Invalid environment variable rejection
    setenv("LLAMA_ARG_XKV", "invalid_mode", true);
    argv = {"binary_name"};
    common_params xkv_bad_env;
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), xkv_bad_env, LLAMA_EXAMPLE_SERVER));

    setenv("LLAMA_ARG_XKV", "off", true);
    setenv("LLAMA_ARG_XKV_GROUP_SIZE", "bad_int", true);
    argv = {"binary_name"};
    common_params xkv_bad_env2;
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), xkv_bad_env2, LLAMA_EXAMPLE_SERVER));

    // Clean up all XKV environment variables
    unsetenv("LLAMA_ARG_XKV");
    unsetenv("LLAMA_ARG_XKV_STORAGE_PROFILE");
    unsetenv("LLAMA_ARG_XKV_GROUP_SIZE");
    unsetenv("LLAMA_ARG_XKV_RANK_K");
    unsetenv("LLAMA_ARG_XKV_RANK_V");
    unsetenv("LLAMA_ARG_XKV_SEGMENT_TOKENS");
    unsetenv("LLAMA_ARG_XKV_CHUNK_TOKENS");
    unsetenv("LLAMA_ARG_XKV_SR_BUDGET");
    unsetenv("LLAMA_ARG_XKV_SOURCE");
    unsetenv("LLAMA_ARG_XKV_FACTOR_A_K");
    unsetenv("LLAMA_ARG_XKV_FACTOR_B_K");
    unsetenv("LLAMA_ARG_XKV_FACTOR_A_V");
    unsetenv("LLAMA_ARG_XKV_FACTOR_B_V");
    unsetenv("LLAMA_ARG_XKV_FACTOR_BALANCE");
    unsetenv("LLAMA_ARG_XKV_LANDMARK_TYPE");
    unsetenv("LLAMA_ARG_XKV_LANDMARK_REFINE");
    unsetenv("LLAMA_ARG_XKV_LANDMARK_REFINE_MAX_ROWS");
    unsetenv("LLAMA_ARG_XKV_WORKSPACE_MIB");
    unsetenv("LLAMA_ARG_XKV_DECODE_CACHE_MIB");
    unsetenv("LLAMA_ARG_XKV_STORE_MIB");
    unsetenv("LLAMA_ARG_XKV_SEED");
    unsetenv("LLAMA_ARG_XKV_MIN_SAVING");
    unsetenv("LLAMA_ARG_XKV_MIN_FACTOR_COVERAGE");
    unsetenv("LLAMA_ARG_XKV_FACTORIZER");
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
