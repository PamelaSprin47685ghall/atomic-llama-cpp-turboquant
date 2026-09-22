#pragma once

#include "ggml.h"

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <mutex>

enum class ggml_tp5_rerot_reject_reason : uint32_t {
    none = 0,
    shmem_tune = 1,
    unsupported_shape = 2,
    unsupported_type = 3,
    subgroup = 4,
    unsupported_quant_pq2_0 = 5,
    count = 6,
};
const char * ggml_tp5_rerot_reject_reason_name(ggml_tp5_rerot_reject_reason reason);

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
    // RELAY full-chain soft-scheduling ledger. All values are accumulated
    // only while GGML_TP5_PROFILE is active; hot-path code does not sample
    // clocks or counters otherwise.
    std::atomic<uint64_t> relay_chains{0};
    std::atomic<uint64_t> relay_direct_stages{0};
    std::atomic<uint64_t> relay_fallback_stages{0};
    std::atomic<uint64_t> relay_route_patch_us{0};
    std::atomic<uint64_t> relay_submit_us{0};
    std::atomic<uint64_t> relay_ready_wait_us{0};
    std::atomic<uint64_t> relay_ready_wait_max_us{0};
    std::atomic<uint64_t> relay_ready_skew_us{0};
    std::atomic<uint64_t> relay_ready_skew_max_us{0};
    std::atomic<uint64_t> relay_arm_us{0};
    std::atomic<uint64_t> relay_cpu_data_us{0};
    std::atomic<uint64_t> relay_sidecar_wait_us{0};
    std::atomic<uint64_t> relay_sidecar_data_us{0};
    std::atomic<uint64_t> relay_y_publish_us{0};
    std::atomic<uint64_t> relay_q_publish_us{0};
    std::atomic<uint64_t> relay_q_spin_iters{0};
    std::atomic<uint64_t> relay_q_spin_samples{0};
    std::atomic<uint64_t> relay_q_spin_max{0};
    std::atomic<uint64_t> relay_generation_us{0};
    std::atomic<uint64_t> relay_handoff_total_us{0};
    std::atomic<uint64_t> relay_poll_iters{0};
    std::atomic<uint64_t> relay_gpu_spin_iters{0};
    std::atomic<uint64_t> relay_gpu_spin_samples{0};
    std::atomic<uint64_t> relay_gpu_spin_max{0};

    // Vulkan RERoT indexed dispatch & route metrics
    std::atomic<uint64_t> vk_rerot_dispatch{0};
    std::atomic<uint64_t> vk_rerot_split_k_total{0};
    std::atomic<uint64_t> vk_rerot_split_k_max{0};
    std::atomic<uint64_t> vk_rerot_queries{0};
    std::atomic<uint64_t> vk_rerot_entries{0};
    // P9 live-length split-K evidence: entries counted at the live clamp vs
    // the capacity-padded entries->ne[1] the old average used.
    std::atomic<uint64_t> vk_rerot_live_entries{0};
    std::atomic<uint64_t> vk_rerot_cap_entries{0};
    std::atomic<uint64_t> vk_rerot_shmem_reject{0};
    std::atomic<uint64_t> vk_rerot_reject_reasons[(size_t) ggml_tp5_rerot_reject_reason::count]{};

    void reset(uint64_t new_id, bool decode);
    void print_summary() const;
};

ggml_tp5_profile * ggml_tp5_profile_active();
void ggml_tp5_profile_begin(uint64_t exec_id, bool decode);
void ggml_tp5_profile_end();
