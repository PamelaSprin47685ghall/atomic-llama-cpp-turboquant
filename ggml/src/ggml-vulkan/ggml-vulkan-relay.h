#pragma once

#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>

// Explicit opt-in hardware probe for the bounded VRAM-doorbell RELAY path.
// It is not part of unattended tests. A timeout or stale generation is a
// terminal collective failure; callers must not retry by enlarging spin_max.
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

// Returns zero only after every bounded stage completes and native draining
// succeeds. A failure leaves resources retained when completion is unproven.
GGML_BACKEND_API int ggml_vk_tp5_relay_probe(ggml_backend_t * backends, size_t n,
                                           const ggml_vk_relay_config & config);
