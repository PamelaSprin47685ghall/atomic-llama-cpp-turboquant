#pragma once

#include "ggml.h"

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <mutex>

// Cross-layer TP5 timing ledger (TP5.md §五 P0). Enabled with GGML_TP5_PROFILE=1.
struct ggml_tp5_profile {
    uint64_t graph_exec_id = 0;
    bool     is_decode     = false;

    std::atomic<uint64_t> queue_submits{0};
    std::atomic<uint64_t> submit_batches{0};
    std::atomic<uint64_t> host_wait_us{0};
    std::atomic<uint64_t> host_wait_count{0};
    std::atomic<uint64_t> backpressure_us{0};
    std::atomic<uint64_t> collective_calls{0};
    std::atomic<uint64_t> fd_exports{0};
    std::atomic<uint64_t> fd_imports{0};
    std::atomic<uint64_t> compute_replay_hits{0};
    std::atomic<uint64_t> compute_replay_misses{0};
    std::atomic<uint64_t> collective_plan_hits{0};
    std::atomic<uint64_t> collective_plan_misses{0};
    std::atomic<uint64_t> drm_wait_hits{0};

    void reset(uint64_t new_id, bool decode);
    void print_summary() const;
};

ggml_tp5_profile * ggml_tp5_profile_active();
void ggml_tp5_profile_begin(uint64_t exec_id, bool decode);
void ggml_tp5_profile_end();
