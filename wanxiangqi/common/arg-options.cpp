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

    return args;
}
