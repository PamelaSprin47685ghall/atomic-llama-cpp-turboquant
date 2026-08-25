#include "arg-options.h"

#include "../../common/common.h"

#include <stdexcept>

std::vector<common_arg> wanxiangqi_common_args(common_params & params) {
    std::vector<common_arg> args;

    args.push_back(common_arg(
        {"--run-dump", "--request-token-dump"}, "PATH",
        "while serving, dump every canonical request token sequence into a persistent prefix trie",
        [](common_params & p, const std::string & value) {
            p.request_token_dump_path = value;
        }
    ).set_env("LLAMA_ARG_REQUEST_TOKEN_DUMP").set_examples({LLAMA_EXAMPLE_SERVER}));

    args.push_back(common_arg(
        {"--request-token-dump-shard"}, "N",
        string_format("maximum node/request records per token-dump shard (default: %llu)",
            (unsigned long long) params.request_token_dump_shard_records),
        [](common_params & p, int value) {
            if (value <= 0) {
                throw std::invalid_argument("request-token-dump-shard must be greater than zero");
            }
            p.request_token_dump_shard_records = (uint64_t) value;
        }
    ).set_env("LLAMA_ARG_REQUEST_TOKEN_DUMP_SHARD").set_examples({LLAMA_EXAMPLE_SERVER}));

    args.push_back(common_arg(
        {"--prune-qk-gram"},
        "collect Qwen3.5 GDN Q/K Gram matrices for llama-qwen35-prune",
        [](common_params & p) { p.prune_qk_gram = true; }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));

    args.push_back(common_arg(
        {"--prune-vocab-counts"},
        "collect tokenizer token counts for llama-qwen35-prune",
        [](common_params & p) { p.prune_vocab_counts = true; }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));

    args.push_back(common_arg(
        {"--prune-mtp"},
        "execute the Qwen3.5 MTP block while collecting prune calibration data",
        [](common_params & p) { p.prune_mtp = true; }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));

    args.push_back(common_arg(
        {"--prune-enp"},
        "collect task-blind geometric coreset data for standard Expert Neuron Pruning (ENP)",
        [](common_params & p) { p.prune_enp = true; }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));

    args.push_back(common_arg(
        {"--prune-enp-only"},
        "collect only standard-ENP hidden-state geometry; skip generic imatrix, GDN, MTP, and vocab calibration",
        [](common_params & p) {
            p.prune_enp = true;
            p.prune_enp_only = true;
            p.prune_qk_gram = false;
            p.prune_vocab_counts = false;
            p.prune_mtp = false;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));

    args.push_back(common_arg(
        {"--prune-enp-cert-only"},
        "append only an independent task-blind uniform-WOR ENP certification sample; intended with --in-file geometric-imatrix.gguf",
        [](common_params & p) {
            p.prune_enp = true;
            p.prune_enp_only = true;
            p.prune_enp_cert_only = true;
            p.prune_qk_gram = false;
            p.prune_vocab_counts = false;
            p.prune_mtp = false;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));

    args.push_back(common_arg(
        {"--prune-enp-downstream-lens"},
        "collect only a 256D task-blind post-MoE-to-final hidden-state lens for downstream-aware ENP reranking",
        [](common_params & p) {
            p.prune_enp_downstream_lens = true;
            p.prune_enp_only = true;
            p.prune_qk_gram = false;
            p.prune_vocab_counts = false;
            p.prune_mtp = false;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));

    args.push_back(common_arg(
        {"--prune-expert-replacement"},
        "collect a single-pass expert-replacement calibration: the 256D downstream lens plus a uniform reservoir of exact MoE input hidden states for replacement-aware expert-count pruning",
        [](common_params & p) {
            p.prune_enp_downstream_lens = true;
            p.prune_expert_replacement = true;
            p.prune_enp_only = true;
            p.prune_qk_gram = false;
            p.prune_vocab_counts = false;
            p.prune_mtp = false;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));

    args.push_back(common_arg(
        {"--prune-expert-replacement-points"}, "N",
        string_format("uniform exact MoE-input states per layer retained for replacement-aware expert pruning (default: %d)",
            params.prune_expert_replacement_points),
        [](common_params & p, int value) {
            if (value <= 0) throw std::invalid_argument("prune-expert-replacement-points must be greater than zero");
            p.prune_expert_replacement_points = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));

    args.push_back(common_arg(
        {"--prune-enp-legacy-uniform"},
        "use the published legacy 512-position uniform-without-replacement ENP sampler (reproduction only)",
        [](common_params & p) {
            p.prune_enp = true;
            p.prune_enp_legacy_uniform = true;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));

    args.push_back(common_arg(
        {"--prune-enp-coreset-points"}, "N",
        string_format("maximum task-blind ENP geometric medoids per layer (default: %d)", params.prune_enp_coreset_points),
        [](common_params & p, int value) {
            if (value <= 0) throw std::invalid_argument("prune-enp-coreset-points must be greater than zero");
            p.prune_enp_coreset_points = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));

    args.push_back(common_arg(
        {"--prune-enp-cert-points"}, "N",
        string_format("independent uniform-WOR ENP certification points per layer (default: %d)", params.prune_enp_cert_points),
        [](common_params & p, int value) {
            if (value <= 0) throw std::invalid_argument("prune-enp-cert-points must be greater than zero");
            p.prune_enp_cert_points = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));

    args.push_back(common_arg(
        {"--prune-enp-cert-tail-points"}, "N",
        string_format("highest-hidden-norm ENP certification states evaluated exactly before uniform-WOR certification (default: %d)", params.prune_enp_cert_tail_points),
        [](common_params & p, int value) {
            if (value < 0) throw std::invalid_argument("prune-enp-cert-tail-points must be non-negative");
            p.prune_enp_cert_tail_points = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));

    args.push_back(common_arg(
        {"--request-token-trie"}, "DIR",
        "traverse a llama-server request-token trie once for imatrix calibration; shared prefixes are evaluated once and branch states are restored from CPU checkpoints",
        [](common_params & p, const std::string & value) {
            p.imatrix_request_token_trie = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));

    return args;
}

