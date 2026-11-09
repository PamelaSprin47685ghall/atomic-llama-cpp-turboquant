#pragma once

// Legacy relay channel map: one entry per (plan generation, channel).
//
// The default compose path uses this map when neither
// GGML_TP5_PLAN_UNION=1 nor GGML_TP5_RELAY_SLOT_CHANNELS=1 selects the
// slotized ownership lane. A graph rebuild can advance the provider plan
// generation (for example the MTP 1-token / n+1-token alternation), and each
// generation inserts its channels into this map. Without reclamation the map
// grows for the provider lifetime.
//
// Reclamation is allowed only from the next generation's INPUT_READY, after
// the provider reported idle (prepare requires it) and after every older
// OUTPUT_RELEASED was serviced by the same courier. If any older entry still
// has a held lease, the whole reclamation is refused: no entry is erased and
// the caller must fail closed. This is deliberately all-or-nothing so a
// partially reclaimed generation cannot hide a live reader.

#include <tq/relay/ledger.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <utility>

namespace tq::tp5 {

struct channel_state {
    relay::channel_ledger       ledger;
    std::optional<relay::lease> held;
    std::uint64_t               epoch = 0;
    std::uint64_t               elements = 0;
    std::uint64_t               wire_bytes = 0;
    std::uint32_t               computed_mask = 0;

    channel_state(std::uint64_t run, std::uint64_t plan, std::uint64_t channel) : ledger(run, plan, channel, 5, 1) {}
};

using channel_map = std::map<std::pair<std::uint64_t, std::uint32_t>, std::unique_ptr<channel_state>>;

enum class channel_map_reclaim {
    none,       // the generation did not advance; nothing to do
    reclaimed,  // every entry older than the new generation was erased
    live_lease, // an older entry still has a held lease; nothing was erased
};

// Reclaim entries older than new_generation. `map_generation` is the highest
// generation already admitted by this window. On live_lease, neither the map
// nor map_generation is modified. `removed` (when non-null) receives the
// number of erased entries, or 0 on refusal.
template <typename HasHeld>
channel_map_reclaim reclaim_older_channels(
        channel_map & channels,
        std::uint64_t new_generation,
        std::uint64_t & map_generation,
        HasHeld && has_held,
        std::size_t * removed = nullptr) {
    if (removed != nullptr) {
        *removed = 0;
    }
    if (new_generation <= map_generation) {
        return channel_map_reclaim::none;
    }
    for (const auto & entry : channels) {
        if (entry.first.first < new_generation && entry.second && has_held(*entry.second)) {
            return channel_map_reclaim::live_lease;
        }
    }
    std::size_t erased = 0;
    for (auto it = channels.begin(); it != channels.end();) {
        if (it->first.first < new_generation) {
            it = channels.erase(it);
            ++erased;
        } else {
            ++it;
        }
    }
    map_generation = new_generation;
    if (removed != nullptr) {
        *removed = erased;
    }
    return channel_map_reclaim::reclaimed;
}

} // namespace tq::tp5
