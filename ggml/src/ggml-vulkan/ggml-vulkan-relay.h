#pragma once

#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>

// Quarantined after an unsafe RADV/Navi21 reset. Not a production collective
// or a sync-mode alias; calls fail closed without submitting GPU work.
struct ggml_vk_relay_config {
    uint32_t stages = 2;
    uint32_t replays = 4;
    uint32_t elements = 2560;
    uint32_t spin_max = 200000;
    uint32_t delay_us = 0;
    uint32_t withhold_stage = 0;
    uint32_t partial_submit_ranks = 0;
    bool stale_doorbell = false;
};

// Always returns 1 while RELAY remains quarantined.
GGML_BACKEND_API int ggml_vk_tp5_relay_probe(ggml_backend_t * backends, size_t n,
                                           const ggml_vk_relay_config & config);
