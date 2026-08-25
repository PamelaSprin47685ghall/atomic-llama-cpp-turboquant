#pragma once

#include <cstdint>
#include <string>

// Fork-only common_params extension. common_params inherits this tiny POD so
// upstream parameter layout/source stays otherwise unchanged.
struct wanxiangqi_common_params {
    std::string request_token_dump_path = "";
    uint64_t request_token_dump_shard_records = 1000000;

    bool prune_qk_gram      = false;
    bool prune_vocab_counts = false;
    bool prune_mtp          = false;
    bool prune_enp          = false;
    bool prune_enp_only     = false;
    bool prune_enp_cert_only = false;
    bool prune_enp_downstream_lens = false;
    bool prune_expert_replacement = false;
    bool prune_enp_legacy_uniform = false;
    int32_t prune_enp_coreset_points = 1024;
    int32_t prune_enp_cert_points = 8192;
    int32_t prune_enp_cert_tail_points = 512;
    int32_t prune_expert_replacement_points = 2048;
    std::string imatrix_request_token_trie;
};

