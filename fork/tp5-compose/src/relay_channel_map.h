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
// Reclamation is state-driven from the next generation's INPUT_READY, after
// the provider reported idle (prepare requires it) and after every older
// OUTPUT_RELEASED was serviced by the same courier. Entries with a held lease
// are retained, never erased; an unheld older entry is retired through the
// caller's callback before it is erased. This preserves a held generation for
// its later OUTPUT_RELEASED instead of aborting a run or dropping an event.
//
// A generation older than the admitted one is a protocol violation: input_ready
// reports stale and creates no entry, so a stale INPUT_READY cannot rebuild an
// old ledger. OUTPUT_RELEASED still finds a retained old entry and never
// triggers reclamation.

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

struct legacy_input {
    channel_state * state = nullptr;
    bool            stale = false;
    std::size_t     retired = 0;
    std::size_t     retired_bank = 0;
    std::size_t     retired_no_reservations = 0;
    std::size_t     retained_held = 0;
    std::size_t     retained_not_ready = 0;
};

class legacy_channel_map {
  public:
    // One INPUT_READY event. `on_retire` is called once for every unheld older
    // entry immediately before it is erased. A held older entry is retained.
    // A stale generation returns stale without touching the map.
    template <typename OnRetire>
    legacy_input input_ready(std::uint64_t run, std::uint64_t generation, std::uint32_t channel,
                             OnRetire && on_retire) {
        legacy_input result;
        if (generation < map_generation_) {
            result.stale = true;
            return result;
        }
        if (generation > map_generation_) {
            for (auto it = channels_.begin(); it != channels_.end();) {
                if (it->first.first >= generation) {
                    ++it;
                    continue;
                }
                if (it->second && it->second->held.has_value()) {
                    ++result.retained_held;
                    ++it;
                    continue;
                }
                if (it->second) {
                    const auto proof = it->second->ledger.retirement();
                    if (proof == relay::retirement_state::not_ready) {
                        ++result.retained_not_ready;
                        ++it;
                        continue;
                    }
                    on_retire(*it->second, proof);
                    if (proof == relay::retirement_state::bank_released) {
                        ++result.retired_bank;
                        ++retired_bank_;
                    } else {
                        ++result.retired_no_reservations;
                        ++retired_no_reservations_;
                    }
                }
                it = channels_.erase(it);
                ++result.retired;
                ++retired_;
            }
            map_generation_ = generation;
        }
        const auto key = std::make_pair(generation, channel);
        auto       it  = channels_.find(key);
        if (it == channels_.end()) {
            it = channels_.emplace(key, std::make_unique<channel_state>(run, generation, channel)).first;
        }
        result.state = it->second.get();
        return result;
    }

    // One OUTPUT_RELEASED event. A retained old generation is found; an
    // already reclaimed one returns nullptr (the caller treats that as a
    // protocol violation). This never reclaims.
    channel_state * output_released(std::uint64_t generation, std::uint32_t channel) {
        const auto it = channels_.find({generation, channel});
        return it == channels_.end() ? nullptr : it->second.get();
    }

    // End-of-run iteration over the remaining entries.
    template <typename F>
    void for_each(F && f) {
        for (auto & entry : channels_) {
            if (entry.second) {
                f(*entry.second);
            }
        }
    }

    std::uint64_t map_generation() const noexcept { return map_generation_; }
    std::uint64_t retired() const noexcept { return retired_; }
    std::uint64_t retired_bank() const noexcept { return retired_bank_; }
    std::uint64_t retired_no_reservations() const noexcept { return retired_no_reservations_; }
    std::size_t   size() const noexcept { return channels_.size(); }

  private:
    channel_map   channels_;
    std::uint64_t map_generation_ = 0;
    std::uint64_t retired_ = 0;
    std::uint64_t retired_bank_ = 0;
    std::uint64_t retired_no_reservations_ = 0;
};

} // namespace tq::tp5
