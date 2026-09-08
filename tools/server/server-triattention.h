#pragma once

#include "llama-ext.h"

#include <cstdint>
#include <utility>

inline bool server_triattention_compressed_after_load(
        bool previous, bool success, bool loaded_state, uint64_t resident, uint64_t logical) {
    return success && loaded_state ? resident < logical : previous;
}

// Draft pressure is independent of target progress. In particular a target
// already at its floor must not suppress reclaim of a separate draft cache.
template <typename Target, typename Draft>
std::pair<llama_memory_kv_reclaim_result, llama_memory_kv_reclaim_result>
server_triattention_reclaim_pair(bool separate_draft, Target target, Draft draft) {
    auto result_tgt = target();
    auto result_dft = separate_draft ? draft() : llama_memory_kv_reclaim_result{};
    return {result_tgt, result_dft};
}
