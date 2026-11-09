// TP5 legacy relay channel-map reclamation test. CPU only, no HIP, no GPU.
//
// The default compose path owns channel_state entries in a map keyed by
// (plan generation, channel). A graph rebuild can advance the plan generation;
// without reclamation the map grows for the provider lifetime.
//
// This test drives the production event chain used by runtime.cpp:
//   INPUT_READY -> (held lease) -> OUTPUT_RELEASED -> newer INPUT_READY
// through legacy_channel_map, with real channel_ledger leases. It observes the
// map generation/size, the held retention, the retired drain facts, and the
// explicit stale-generation refusal. It is not a pure helper-counter test.

#include "relay_channel_map.h"

#include <tq/relay/ledger.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            ++failures;                                                                  \
        }                                                                                \
    } while (0)

using tq::relay::bank_state;
using tq::relay::ledger_result;
using tq::relay::reserve_code;
using tq::tp5::channel_state;
using tq::tp5::legacy_channel_map;

struct retire_log {
    int                     calls = 0;
    std::vector<bank_state> banks;
};

void complete_and_release(channel_state & state) {
    CHECK(state.held.has_value());
    if (!state.held.has_value()) {
        return;
    }
    const auto &        held = *state.held;
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

bool hold(channel_state & state) {
    auto reservation = state.ledger.reserve(1, 1, 0, 1);
    CHECK(reservation.code == reserve_code::accepted);
    CHECK(reservation.granted.has_value());
    if (!reservation.granted.has_value()) {
        return false;
    }
    state.held = *reservation.granted;
    state.epoch = 1;
    CHECK(state.ledger.active_bank_count() == 1);
    return true;
}

void test_input_output_input_chain() {
    legacy_channel_map map;
    retire_log        log;
    auto on_retire = [&](channel_state & old) {
        CHECK(old.ledger.mark_drained(true));
        log.banks.push_back(old.ledger.bank(0));
        ++log.calls;
    };

    // Generation 1 input, with a real held lease.
    auto in1 = map.input_ready(1, 1, 1, on_retire);
    CHECK(!in1.stale);
    CHECK(in1.state != nullptr);
    if (in1.state == nullptr) {
        return;
    }
    CHECK(hold(*in1.state));
    CHECK(map.map_generation() == 1);
    CHECK(map.size() == 1);

    // Generation 2 input while generation 1 is still held: retain, do not
    // abort and do not erase. This is the state-driven availability path.
    auto in2 = map.input_ready(1, 2, 1, on_retire);
    CHECK(!in2.stale);
    CHECK(in2.state != nullptr);
    CHECK(in2.retained_held == 1);
    CHECK(in2.retired == 0);
    CHECK(map.map_generation() == 2);
    CHECK(map.size() == 2);
    CHECK(map.retired() == 0);
    CHECK(log.calls == 0);

    // OUTPUT_RELEASED for the retained generation 1: the entry is still
    // findable, and the real ledger completes before held is cleared.
    channel_state * out1 = map.output_released(1, 1);
    CHECK(out1 == in1.state);
    CHECK(out1 != nullptr && out1->held.has_value());
    if (out1 != nullptr) {
        complete_and_release(*out1);
    }
    CHECK(map.size() == 2); // OUTPUT_RELEASED never reclaims

    // Generation 3 input retires the now-unheld generation 1 and the unheld
    // generation 2 through the callback, and keeps only generation 3.
    auto in3 = map.input_ready(1, 3, 1, on_retire);
    CHECK(!in3.stale);
    CHECK(in3.state != nullptr);
    CHECK(in3.retained_held == 0);
    CHECK(in3.retired == 2);
    CHECK(map.map_generation() == 3);
    CHECK(map.size() == 1);
    CHECK(map.retired() == 2);
    CHECK(log.calls == 2);
    bool saw_completed_bank = false;
    for (const auto & bank : log.banks) {
        if (bank.occupied && bank.input_released && bank.output_published && bank.consumed_ranks == 0x1fu) {
            saw_completed_bank = true;
        }
    }
    CHECK(saw_completed_bank); // the retired ledger drain fact is observable

    // A stale INPUT_READY for an already reclaimed generation is refused and
    // must not recreate an old ledger.
    auto stale = map.input_ready(1, 1, 1, on_retire);
    CHECK(stale.stale);
    CHECK(stale.state == nullptr);
    CHECK(stale.retired == 0);
    CHECK(stale.retained_held == 0);
    CHECK(map.size() == 1);
    CHECK(map.map_generation() == 3);
    CHECK(log.calls == 2);

    // The same admitted generation may still introduce a new channel.
    auto same = map.input_ready(1, 3, 2, on_retire);
    CHECK(!same.stale);
    CHECK(same.state != nullptr);
    CHECK(map.size() == 2);
    CHECK(map.map_generation() == 3);

    // OUTPUT_RELEASED for a reclaimed generation has no entry: the runtime
    // treats this as a protocol violation. It never recreates the ledger.
    CHECK(map.output_released(2, 1) == nullptr);
}

void test_output_release_is_not_reclaim() {
    legacy_channel_map map;
    retire_log        log;
    auto on_retire = [&](channel_state & old) {
        CHECK(old.ledger.mark_drained(true));
        ++log.calls;
    };

    auto in1 = map.input_ready(1, 1, 1, on_retire);
    CHECK(!in1.stale);
    CHECK(in1.state != nullptr);
    if (in1.state == nullptr) {
        return;
    }
    CHECK(hold(*in1.state));
    channel_state * out = map.output_released(1, 1);
    CHECK(out == in1.state);
    CHECK(map.size() == 1);
    CHECK(map.retired() == 0);
    CHECK(log.calls == 0);
}

} // namespace

int main() {
    test_input_output_input_chain();
    test_output_release_is_not_reclaim();
    if (failures != 0) {
        std::fprintf(stderr, "RELAY_CHANNEL_MAP_TEST FAIL %d\n", failures);
        return 1;
    }
    std::printf("RELAY_CHANNEL_MAP_TEST PASS\n");
    return 0;
}
