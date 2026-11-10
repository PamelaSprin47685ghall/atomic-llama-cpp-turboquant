#pragma once

#include <cstddef>

// Host-side KV journal/prepare accounting. This is deliberately separate from
// llama_memory_breakdown: that API reports physical buffers by backend buffer
// type and keeps its existing device-buffer meaning.
//
// The fields are payload capacities, not allocator bookkeeping, and not a
// resident-memory total. They cover the KV journal and the prepare/decode
// scratch only. See llama-ext.h for the query contract and the excluded
// resident state.
struct llama_host_memory_breakdown_data {
    size_t journal      = 0; // cell_journal arrays, owned set/map nodes, entries, ranges, discards
    size_t trial_index  = 0; // flat per-stream trial index blocks
    size_t snapshots    = 0; // prepare snapshot cells/heads, slot infos, and shift backup
    size_t scratch      = 0; // decode scratch control vectors/objects and slot infos

    size_t total() const {
        return journal + trial_index + snapshots + scratch;
    }
};
