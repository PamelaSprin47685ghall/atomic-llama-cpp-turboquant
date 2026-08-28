#pragma once

#include <cstdint>
#include <string>

// Fork-only common_params extension. common_params inherits this tiny POD so
// upstream parameter layout/source stays otherwise unchanged.
struct wanxiangqi_common_params {
    std::string request_token_dump_path = "";
    uint64_t request_token_dump_shard_records = 1000000;
};
