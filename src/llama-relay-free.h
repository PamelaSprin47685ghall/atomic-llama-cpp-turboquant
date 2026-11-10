#pragma once

#include "ggml-relay.h"

#include <cstdint>

// Pure decision for llama_free's relay retain path. The caller has already
// queried ggml_backend_relay_snapshot() and, when available,
// ggml_backend_relay_cpu_snapshot(). Keeping the policy here lets a CPU-only
// test cover every branch without a llama_context or a relay provider.
struct llama_relay_free_facts {
    bool     have_snapshot   = false;
    bool     status_failed   = false;
    bool     status_finished = false;
    uint64_t status_epoch    = 0;
    uint64_t status_channels = 0;
    // 1: cpu filled, 0: contended, -1: unsupported (private provider ABI < 5).
    int                  cpu_result = -1;
    ggml_relay_cpu_status cpu        = {};
};

struct llama_relay_free_decision {
    bool         retain = false;
    const char * reason = "an async/native submission is still unproven";
};

inline llama_relay_free_decision llama_relay_free_decide(const llama_relay_free_facts & facts) {
    llama_relay_free_decision decision;

    bool borrow_outstanding   = false;
    bool tokens_outstanding   = false;
    bool releases_outstanding = false;
    bool cpu_unproven         = false;

    if (facts.cpu_result == 1) {
        if (facts.cpu.epoch == facts.status_epoch && facts.cpu.channels == facts.status_channels) {
            borrow_outstanding   = facts.cpu.borrowed_channel != UINT32_MAX;
            tokens_outstanding   = facts.cpu.active_token != 0 || facts.cpu.tokens_outstanding != 0;
            // A failed provider's destroy does not require output release, and
            // epoch 0 has no begun epoch whose outputs could be owed.
            releases_outstanding = facts.cpu.epoch != 0 && !facts.status_failed &&
                                   facts.cpu.outputs_released != facts.cpu.channels;
        } else {
            // The provider moved between the two queries, or a prepare changed
            // the channel count. Do not compare stale facts.
            cpu_unproven = true;
        }
    } else if (facts.cpu_result == 0) {
        // The provider mutex stayed contended for the whole window.
        cpu_unproven = true;
    }

    if (borrow_outstanding) {
        decision.retain = true;
        decision.reason = "a relay CPU borrow is still outstanding";
    } else if (tokens_outstanding) {
        decision.retain = true;
        decision.reason = "relay completion tokens are still outstanding";
    } else if (releases_outstanding) {
        decision.retain = true;
        decision.reason = "relay output releases are still outstanding";
    } else if (cpu_unproven) {
        decision.retain = true;
        decision.reason = "relay CPU state could not be proven";
    } else if (!facts.have_snapshot ||
               (facts.status_epoch != 0 && !facts.status_finished)) {
        decision.retain = true;
    }
    return decision;
}
