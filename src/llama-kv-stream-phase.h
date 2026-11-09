#pragma once

// KV-stream phase-arena state contract.
//
// A phase transition changes the physical KV/compute split before the new
// scheduler and its reserved graph exist. A phase may be reused only when its
// scheduler is ready and the committed split is the one it was built for.
//
// The transition itself lives in llama_context::kv_stream_switch_phase and
// calls the device ops directly; this header owns only the state so the
// commit/ready distinction stays explicit.
//
// Failure isolation:
//   * the caller checks every scheduler owner (main, prefab, per-shape) for an
//     unproven submission before destroying or replacing any of them;
//   * a failed physical step is rolled back; if the rollback fails, the state
//     becomes inconsistent and every later transition is refused;
//   * a successful physical step is committed as a fact even if the scheduler
//     or graph reserve fails; phase_ready stays false, so the next call retries
//     from the committed split instead of claiming the old phase.

#include <cstddef>
#include <cstdint>

struct llama_kv_stream_phase_layout {
    std::size_t   kv_bytes       = 0;
    std::size_t   compute_offset = 0;
    std::size_t   compute_bytes  = 0;
    std::uint32_t ring_slots     = 0;

    bool operator==(const llama_kv_stream_phase_layout & other) const noexcept {
        return kv_bytes == other.kv_bytes && compute_offset == other.compute_offset &&
               compute_bytes == other.compute_bytes && ring_slots == other.ring_slots;
    }
};

struct llama_kv_stream_phase_state {
    llama_kv_stream_phase_layout current{};
    bool decode       = false;
    bool phase_ready  = false;
    bool inconsistent = false;

    bool can_reuse(bool requested) const noexcept {
        return !inconsistent && phase_ready && decode == requested;
    }
};
