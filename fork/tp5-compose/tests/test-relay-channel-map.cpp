// TP5 legacy relay channel-map reclamation test. CPU only, no HIP, no GPU.
//
// The default compose path owns channel_state entries in a map keyed by
// (plan generation, channel). A graph rebuild can advance the plan generation;
// without reclamation the map grows for the provider lifetime. The shared
// reclaim_older_channels() helper is the production helper used by runtime.cpp.
//
// This test observes the real channel_ledger state: a held entry is a real
// reserved lease with an active bank. Reclamation must refuse the whole older
// generation while that lease is held, leave the map untouched, and succeed
// only after the ledger completes and the held optional is cleared.

#include "relay_channel_map.h"

#include <tq/relay/ledger.h>

#include <cstdint>
#include <cstdio>
#include <memory>

namespace {

int failures = 0;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            ++failures;                                                                  \
        }                                                                                \
    } while (0)

using tq::relay::ledger_result;
using tq::relay::reserve_code;
using tq::tp5::channel_map;
using tq::tp5::channel_map_reclaim;
using tq::tp5::channel_state;
using tq::tp5::reclaim_older_channels;

std::unique_ptr<channel_state> make_state(std::uint64_t generation, std::uint32_t channel, bool hold) {
    auto state = std::make_unique<channel_state>(1, generation, channel);
    if (hold) {
        auto reservation = state->ledger.reserve(1, 1, 0, 1);
        CHECK(reservation.code == reserve_code::accepted);
        CHECK(reservation.granted.has_value());
        if (reservation.granted.has_value()) {
            state->held = *reservation.granted;
            state->epoch = 1;
            CHECK(state->ledger.active_bank_count() == 1);
        }
    }
    return state;
}

void add_generation(channel_map & channels, std::uint64_t generation, std::uint32_t count,
                    std::int32_t held_channel = -1) {
    for (std::uint32_t channel = 0; channel < count; ++channel) {
        channels.emplace(std::make_pair(generation, channel),
                         make_state(generation, channel + 1, held_channel == std::int32_t(channel)));
    }
}

bool has_held(const channel_state & state) {
    return state.held.has_value();
}

channel_map_reclaim reclaim(channel_map & channels, std::uint64_t generation,
                            std::uint64_t & map_generation, std::size_t * removed = nullptr) {
    return reclaim_older_channels(channels, generation, map_generation, has_held, removed);
}

void complete_and_release(channel_state & state) {
    CHECK(state.held.has_value());
    if (!state.held.has_value()) {
        return;
    }
    const auto & held = *state.held;
    const std::uint32_t slot = held.slot();
    for (std::uint32_t rank = 0; rank < 5; ++rank) {
        CHECK(state.ledger.producer_ready(held, rank) == ledger_result::ok);
    }
    CHECK(state.ledger.reduce_done(held) == ledger_result::ok);
    CHECK(state.ledger.output_published(held) == ledger_result::ok);
    for (std::uint32_t rank = 0; rank < 5; ++rank) {
        CHECK(state.ledger.consumer_done(held, rank) == ledger_result::ok);
    }
    const auto bank = state.ledger.bank(slot);
    CHECK(bank.occupied);
    CHECK(bank.input_released);
    CHECK(bank.output_published);
    CHECK(bank.consumed_ranks == 0x1fu);
    state.held.reset();
}

void test_generation_does_not_advance() {
    channel_map channels;
    add_generation(channels, 1, 3);
    std::uint64_t map_generation = 1;
    std::size_t removed = 99;

    CHECK(reclaim(channels, 1, map_generation, &removed) == channel_map_reclaim::none);
    CHECK(channels.size() == 3);
    CHECK(map_generation == 1);
    CHECK(removed == 0);
}

void test_reclaims_older_generations() {
    channel_map channels;
    add_generation(channels, 1, 181);
    add_generation(channels, 2, 181);
    add_generation(channels, 3, 181);
    std::uint64_t map_generation = 0;
    std::size_t removed = 0;

    CHECK(reclaim(channels, 4, map_generation, &removed) == channel_map_reclaim::reclaimed);
    CHECK(channels.empty());
    CHECK(map_generation == 4);
    CHECK(removed == 543);

    add_generation(channels, 4, 181);
    CHECK(reclaim(channels, 5, map_generation, &removed) == channel_map_reclaim::reclaimed);
    CHECK(channels.empty());
    CHECK(map_generation == 5);
    CHECK(removed == 181);
}

void test_keeps_newer_entries() {
    channel_map channels;
    add_generation(channels, 1, 2);
    add_generation(channels, 2, 2);
    std::uint64_t map_generation = 1;
    std::size_t removed = 0;

    CHECK(reclaim(channels, 2, map_generation, &removed) == channel_map_reclaim::reclaimed);
    CHECK(channels.size() == 2);
    CHECK(channels.count({2, 0}) == 1);
    CHECK(channels.count({2, 1}) == 1);
    CHECK(map_generation == 2);
    CHECK(removed == 2);

    CHECK(reclaim(channels, 2, map_generation, &removed) == channel_map_reclaim::none);
    CHECK(channels.size() == 2);
    CHECK(removed == 0);
}

void test_live_lease_refuses_whole_reclaim() {
    channel_map channels;
    add_generation(channels, 1, 2, 1); // channel index 1 holds a real lease
    add_generation(channels, 2, 1);
    std::uint64_t map_generation = 1;
    std::size_t removed = 99;

    auto held_it = channels.find({1, 1});
    CHECK(held_it != channels.end());
    if (held_it == channels.end()) {
        return;
    }
    channel_state * held_state = held_it->second.get();
    CHECK(held_state->ledger.active_bank_count() == 1);

    CHECK(reclaim(channels, 2, map_generation, &removed) == channel_map_reclaim::live_lease);
    CHECK(channels.size() == 3);
    CHECK(map_generation == 1);
    CHECK(removed == 0);
    CHECK(channels.count({1, 0}) == 1);
    CHECK(channels.count({1, 1}) == 1);
    CHECK(channels.count({2, 0}) == 1);
    CHECK(held_state->held.has_value());

    complete_and_release(*held_state);
    CHECK(reclaim(channels, 2, map_generation, &removed) == channel_map_reclaim::reclaimed);
    CHECK(channels.size() == 1);
    CHECK(channels.count({2, 0}) == 1);
    CHECK(map_generation == 2);
    CHECK(removed == 2);
}

} // namespace

int main() {
    test_generation_does_not_advance();
    test_reclaims_older_generations();
    test_keeps_newer_entries();
    test_live_lease_refuses_whole_reclaim();
    if (failures != 0) {
        std::fprintf(stderr, "RELAY_CHANNEL_MAP_TEST FAIL %d\n", failures);
        return 1;
    }
    std::printf("RELAY_CHANNEL_MAP_TEST PASS\n");
    return 0;
}
