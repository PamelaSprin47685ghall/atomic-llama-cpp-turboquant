// TP5 legacy relay channel-map reclamation test. CPU only, no HIP, no GPU.
//
// The default compose path owns channel_state entries in a map keyed by
// (plan generation, channel). A graph rebuild can advance the plan generation;
// without reclamation the map grows for the provider lifetime.
//
// This test drives the production event chain used by runtime.cpp:
//   INPUT_READY -> held lease -> OUTPUT_RELEASED -> newer INPUT_READY
// through legacy_channel_map, with real channel_ledger receipts. An entry is
// retired only when the ledger proves a complete bank or proves that no bank
// was ever reserved. It never calls mark_drained or uses a bool as a proof.

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
using tq::relay::channel_ledger;
using tq::relay::lease;
using tq::relay::ledger_result;
using tq::relay::reserve_code;
using tq::relay::retirement_state;
using tq::tp5::channel_state;
using tq::tp5::legacy_channel_map;

struct retire_log {
    int                           calls = 0;
    std::vector<retirement_state> proofs;
    std::vector<bank_state>       banks;
};

bool reserve_lease(channel_state & state) {
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

void complete_lease(channel_ledger & ledger, const lease & held) {
    const std::uint32_t slot = held.slot();
    for (std::uint32_t rank = 0; rank < 5; ++rank) {
        CHECK(ledger.producer_ready(held, rank) == ledger_result::ok);
    }
    CHECK(ledger.reduce_done(held) == ledger_result::ok);
    CHECK(ledger.output_published(held) == ledger_result::ok);
    for (std::uint32_t rank = 0; rank < 5; ++rank) {
        CHECK(ledger.consumer_done(held, rank) == ledger_result::ok);
    }
    const auto bank = ledger.bank(slot);
    CHECK(bank.occupied);
    CHECK(bank.input_released);
    CHECK(bank.output_published);
    CHECK(bank.consumed_ranks == 0x1fu);
    CHECK(ledger.retirement() == retirement_state::bank_released);
}

void complete_and_release(channel_state & state) {
    CHECK(state.held.has_value());
    if (!state.held.has_value()) {
        return;
    }
    const lease held = *state.held;
    complete_lease(state.ledger, held);
    state.held.reset();
}

void test_input_output_input_chain() {
    legacy_channel_map map;
    retire_log        log;
    auto on_retire = [&](channel_state & old, retirement_state proof) {
        log.proofs.push_back(proof);
        log.banks.push_back(old.ledger.bank(0));
        ++log.calls;
    };

    // Generation 1 input with a real held lease.
    auto in1 = map.input_ready(1, 1, 1, on_retire);
    CHECK(!in1.stale);
    CHECK(in1.state != nullptr);
    if (in1.state == nullptr) {
        return;
    }
    CHECK(reserve_lease(*in1.state));
    CHECK(map.map_generation() == 1);
    CHECK(map.size() == 1);

    // Generation 2 input while generation 1 is held: retain, do not abort.
    auto in2 = map.input_ready(1, 2, 1, on_retire);
    CHECK(!in2.stale);
    CHECK(in2.state != nullptr);
    CHECK(in2.retained_held == 1);
    CHECK(in2.retained_not_ready == 0);
    CHECK(in2.retired == 0);
    CHECK(map.map_generation() == 2);
    CHECK(map.size() == 2);
    CHECK(map.retired() == 0);
    CHECK(log.calls == 0);

    // OUTPUT_RELEASED for retained generation 1: find it, complete the real
    // ledger, then clear held. OUTPUT_RELEASED never reclaims.
    channel_state * out1 = map.output_released(1, 1);
    CHECK(out1 == in1.state);
    CHECK(out1 != nullptr && out1->held.has_value());
    if (out1 != nullptr) {
        complete_and_release(*out1);
    }
    CHECK(map.size() == 2);

    // Generation 3 retires generation 1 with the complete-bank proof and
    // generation 2 with the no-bank proof; only generation 3 remains.
    auto in3 = map.input_ready(1, 3, 1, on_retire);
    CHECK(!in3.stale);
    CHECK(in3.state != nullptr);
    CHECK(in3.retained_held == 0);
    CHECK(in3.retained_not_ready == 0);
    CHECK(in3.retired == 2);
    CHECK(in3.retired_bank == 1);
    CHECK(in3.retired_no_work == 1);
    CHECK(map.map_generation() == 3);
    CHECK(map.size() == 1);
    CHECK(map.retired() == 2);
    CHECK(map.retired_bank() == 1);
    CHECK(map.retired_no_work() == 1);
    CHECK(log.calls == 2);
    bool saw_bank_released = false;
    bool saw_no_work = false;
    for (std::size_t i = 0; i < log.proofs.size(); ++i) {
        if (log.proofs[i] == retirement_state::bank_released) {
            saw_bank_released = true;
            const auto & bank = log.banks[i];
            CHECK(bank.occupied);
            CHECK(bank.input_released);
            CHECK(bank.output_published);
            CHECK(bank.consumed_ranks == 0x1fu);
        } else if (log.proofs[i] == retirement_state::no_native_work) {
            saw_no_work = true;
        } else {
            CHECK(false);
        }
    }
    CHECK(saw_bank_released);
    CHECK(saw_no_work);

    // Stale INPUT_READY for a reclaimed generation is refused and creates no
    // old ledger.
    auto stale = map.input_ready(1, 1, 1, on_retire);
    CHECK(stale.stale);
    CHECK(stale.state == nullptr);
    CHECK(stale.retired == 0);
    CHECK(stale.retained_held == 0);
    CHECK(stale.retained_not_ready == 0);
    CHECK(map.size() == 1);
    CHECK(map.map_generation() == 3);
    CHECK(log.calls == 2);

    // The same admitted generation may introduce a new channel.
    auto same = map.input_ready(1, 3, 2, on_retire);
    CHECK(!same.stale);
    CHECK(same.state != nullptr);
    CHECK(map.size() == 2);
    CHECK(map.map_generation() == 3);

    // OUTPUT_RELEASED for a reclaimed generation has no entry.
    CHECK(map.output_released(2, 1) == nullptr);
}

void test_incomplete_bank_is_not_retired() {
    legacy_channel_map map;
    retire_log        log;
    auto on_retire = [&](channel_state & old, retirement_state proof) {
        log.proofs.push_back(proof);
        log.banks.push_back(old.ledger.bank(0));
        ++log.calls;
    };

    auto in1 = map.input_ready(1, 1, 1, on_retire);
    CHECK(!in1.stale);
    CHECK(in1.state != nullptr);
    if (in1.state == nullptr) {
        return;
    }
    CHECK(reserve_lease(*in1.state));
    const lease held1 = *in1.state->held;
    in1.state->held.reset(); // simulate an error/partial state with no held mirror
    CHECK(in1.state->ledger.producer_ready(held1, 0) == ledger_result::ok);
    CHECK(in1.state->ledger.retirement() == retirement_state::not_ready);

    auto in2 = map.input_ready(1, 2, 1, on_retire);
    CHECK(!in2.stale);
    CHECK(in2.retained_held == 0);
    CHECK(in2.retained_not_ready == 1);
    CHECK(in2.retired == 0);
    CHECK(map.size() == 2);
    CHECK(map.map_generation() == 2);
    CHECK(log.calls == 0);

    // Finish the bank receipts, then a newer generation retires generation 1
    // with the complete-bank proof and generation 2 with the no-bank proof.
    for (std::uint32_t rank = 1; rank < 5; ++rank) {
        CHECK(in1.state->ledger.producer_ready(held1, rank) == ledger_result::ok);
    }
    CHECK(in1.state->ledger.reduce_done(held1) == ledger_result::ok);
    CHECK(in1.state->ledger.output_published(held1) == ledger_result::ok);
    for (std::uint32_t rank = 0; rank < 5; ++rank) {
        CHECK(in1.state->ledger.consumer_done(held1, rank) == ledger_result::ok);
    }
    CHECK(in1.state->ledger.retirement() == retirement_state::bank_released);

    auto in3 = map.input_ready(1, 3, 1, on_retire);
    CHECK(!in3.stale);
    CHECK(in3.retired == 2);
    CHECK(in3.retired_bank == 1);
    CHECK(in3.retired_no_work == 1);
    CHECK(map.map_generation() == 3);
    CHECK(map.size() == 1);
    CHECK(log.calls == 2);
}

void test_output_release_is_not_reclaim() {
    legacy_channel_map map;
    retire_log        log;
    auto on_retire = [&](channel_state &, retirement_state) { ++log.calls; };

    auto in1 = map.input_ready(1, 1, 1, on_retire);
    CHECK(!in1.stale);
    CHECK(in1.state != nullptr);
    if (in1.state == nullptr) {
        return;
    }
    CHECK(reserve_lease(*in1.state));
    channel_state * out = map.output_released(1, 1);
    CHECK(out == in1.state);
    CHECK(map.size() == 1);
    CHECK(map.retired() == 0);
    CHECK(log.calls == 0);
}

} // namespace

int main() {
    test_input_output_input_chain();
    test_incomplete_bank_is_not_retired();
    test_output_release_is_not_reclaim();
    if (failures != 0) {
        std::fprintf(stderr, "RELAY_CHANNEL_MAP_TEST FAIL %d\n", failures);
        return 1;
    }
    std::printf("RELAY_CHANNEL_MAP_TEST PASS\n");
    return 0;
}
