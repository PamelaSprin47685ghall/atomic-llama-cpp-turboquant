// C++ port of the eight scenario groups of `tp5_protocol_spec.py`
// (SHA-256 4cc04d96...abba2, guide 50.4) plus additional boundary
// counterexamples. Groups 5 and 6 exercise the admission budget, whose
// contract already lives in TQScheduling (`admission-test.cpp`); this file
// covers the channel ledger groups 1-4, 7, 8 and the ledger-only edges.
//
// Scope: single-threaded, trusted, single-issuer CPU ownership model. No
// GPU memory, no transport, no concurrency qualification (guide 50.5).

#include <tq/relay/detail/arithmetic.h>
#include <tq/relay/ledger.h>

#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using tq::relay::channel_ledger;
using tq::relay::lease;
using tq::relay::ledger_result;
using tq::relay::reserve_code;
using tq::relay::reserve_reply;

namespace {

void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void must_fail(const std::function<void()> & action, const char * message) {
    try {
        action();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error(message);
}

void expect_result(ledger_result actual, ledger_result expected, const char * message) {
    check(actual == expected, message);
}

reserve_reply accepted_or_die(channel_ledger & ledger,
                              std::uint64_t    window_id,
                              std::uint64_t    window_generation,
                              std::uint64_t    segment_id,
                              std::uint64_t    allocation_generation = 1) {
    auto reply = ledger.reserve(window_id, window_generation, segment_id, allocation_generation);
    check(reply.code == reserve_code::accepted, "reserve unexpectedly refused");
    return reply;
}

void complete_inputs(channel_ledger & ledger, const lease & held) {
    for (std::uint32_t rank = 0; rank < ledger.rank_count(); ++rank) {
        expect_result(ledger.producer_ready(held, rank), ledger_result::ok, "producer rejected");
    }
    expect_result(ledger.reduce_done(held), ledger_result::ok, "reduce rejected");
    expect_result(ledger.output_published(held), ledger_result::ok, "publish rejected");
}

void complete_readers(channel_ledger & ledger, const lease & held) {
    for (std::uint32_t rank = 0; rank < ledger.rank_count(); ++rank) {
        expect_result(ledger.consumer_done(held, rank), ledger_result::ok, "consumer rejected");
    }
}

void group_1_two_banks_protect_last_reader() {
    channel_ledger ledger(7, 1, 2, 5, 2);
    const auto     first  = accepted_or_die(ledger, 10, 7, 0).granted.value();
    const auto     second = accepted_or_die(ledger, 11, 4, 0).granted.value();
    complete_inputs(ledger, first);
    complete_inputs(ledger, second);
    for (std::uint32_t rank = 0; rank < 4; ++rank) {
        expect_result(ledger.consumer_done(first, rank), ledger_result::ok, "consumer rejected");
    }
    check(ledger.reserve(12, 1, 0, 1).code == reserve_code::wait_for_capacity, "last reader was not protected");
    expect_result(ledger.consumer_done(first, 4), ledger_result::ok, "consumer rejected");
    const auto third = accepted_or_die(ledger, 12, 1, 0).granted.value();
    check(third.slot() == first.slot(), "reuse must land on the retired bank");
    check(third.message().ticket != first.message().ticket, "reuse must issue a new ticket");
}

void group_2_stale_and_foreign_tickets() {
    channel_ledger ledger(7, 1, 2, 5, 2);
    const auto     first  = accepted_or_die(ledger, 10, 7, 0).granted.value();
    const auto     second = accepted_or_die(ledger, 11, 4, 0).granted.value();
    complete_inputs(ledger, first);
    complete_inputs(ledger, second);
    for (std::uint32_t rank = 0; rank < 4; ++rank) {
        expect_result(ledger.consumer_done(first, rank), ledger_result::ok, "consumer rejected");
    }
    check(ledger.reserve(12, 1, 0, 1).code == reserve_code::wait_for_capacity, "last reader was not protected");
    expect_result(ledger.consumer_done(first, 4), ledger_result::ok, "consumer rejected");
    const auto third = accepted_or_die(ledger, 12, 1, 0).granted.value();
    check(third.slot() == first.slot(), "reuse must land on the retired bank");
    check(third.message().ticket != first.message().ticket, "reuse must issue a new ticket");
    // The retired lease names the old generation: exact-ticket matching
    // must reject it instead of consuming the new data (guide 32.6).
    expect_result(ledger.consumer_done(first, 0), ledger_result::stale_ticket, "stale ticket accepted");
    // A lease minted by a different ledger cannot name this channel.
    channel_ledger other(99, 1, 2, 5, 2);
    const auto     foreign = accepted_or_die(other, 11, 4, 0).granted.value();
    expect_result(ledger.producer_ready(foreign, 0), ledger_result::wrong_identity, "foreign issuer accepted");
    // The new tenant is untouched by the rejected calls.
    check(ledger.bank(third.slot()).occupied, "reuse disappeared");
    check(ledger.bank(third.slot()).ready_ranks == 0, "foreign ready polluted the bank");
}

void group_3_partial_inputs_and_unpublished_reads() {
    channel_ledger ledger(8, 1, 0, 5, 2);
    const auto     item = accepted_or_die(ledger, 0, 0, 0).granted.value();
    for (std::uint32_t rank = 0; rank < 4; ++rank) {
        expect_result(ledger.producer_ready(item, rank), ledger_result::ok, "producer rejected");
    }
    expect_result(ledger.reduce_done(item), ledger_result::not_ready, "partial reduction accepted");
    expect_result(ledger.consumer_done(item, 0), ledger_result::not_ready, "unpublished data read");
    expect_result(ledger.output_published(item), ledger_result::not_ready, "publish before reduce accepted");
}

void group_4_sticky_failure_and_drain() {
    channel_ledger ledger(8, 1, 0, 5, 2);
    const auto     item = accepted_or_die(ledger, 0, 0, 0).granted.value();
    ledger.fail("rank 4 timeout");
    ledger.fail("later stdout failure");
    check(ledger.failure_reason() == "rank 4 timeout", "first failure was overwritten");
    check(!ledger.may_release_storage(), "timeout must not prove storage release");
    check(!ledger.mark_drained(false), "false drain proof must not take effect");
    check(ledger.reserve(1, 0, 0, 1).code == reserve_code::stopped, "failed channel accepted new work");
    expect_result(ledger.consumer_done(item, 0), ledger_result::stopped, "failed channel accepted completions");
    check(ledger.mark_drained(true), "true drain proof must take effect");
    check(ledger.may_release_storage(), "drained channel must allow storage release");
}

void group_7_serial_reuse() {
    channel_ledger ledger(9, 1, 1, 5, 2);
    std::uint64_t  last_ticket = 0;
    for (int step = 0; step < 10000; ++step) {
        const auto item = accepted_or_die(ledger, static_cast<std::uint64_t>(step % 2), 0, step % 3).granted.value();
        complete_inputs(ledger, item);
        complete_readers(ledger, item);
        last_ticket = item.message().ticket;
    }
    check(last_ticket == 10000, "reuse sequence drifted");
}

void group_8_ticket_exhaustion_is_refused() {
    // The wrap guard is pure arithmetic; drive it directly (guide 50.5:
    // direct field poking is a teaching-model artifact, so the production
    // seam is a tested helper instead of a public setter).
    using tq::relay::detail::advance_ticket;
    check(advance_ticket(1) == 2, "ticket advance broken");
    check(advance_ticket(std::numeric_limits<std::uint64_t>::max()) == 0,
          "ticket wrap must land on the exhausted sentinel");
    check(advance_ticket(0) == 0, "exhausted state must stay exhausted");
}

void duplicate_message_rejected() {
    channel_ledger ledger(3, 1, 0, 5, 3);
    const auto     first = accepted_or_die(ledger, 20, 9, 4).granted.value();
    check(ledger.reserve(20, 9, 4, 1).code == reserve_code::invalid_message,
          "duplicate (window, generation, segment) accepted");
    complete_inputs(ledger, first);
    complete_readers(ledger, first);
}

void duplicate_completion_facts_rejected() {
    channel_ledger ledger(3, 1, 0, 5, 2);
    const auto     item = accepted_or_die(ledger, 0, 0, 0).granted.value();
    expect_result(ledger.producer_ready(item, 0), ledger_result::ok, "producer rejected");
    expect_result(ledger.producer_ready(item, 0), ledger_result::duplicate, "duplicate ready accepted");
    for (std::uint32_t rank = 1; rank < 5; ++rank) {
        expect_result(ledger.producer_ready(item, rank), ledger_result::ok, "producer rejected");
    }
    expect_result(ledger.reduce_done(item), ledger_result::ok, "reduce rejected");
    expect_result(ledger.reduce_done(item), ledger_result::duplicate, "duplicate reduce accepted");
    expect_result(ledger.producer_ready(item, 2), ledger_result::duplicate, "write after input retirement accepted");
    expect_result(ledger.output_published(item), ledger_result::ok, "publish rejected");
    expect_result(ledger.output_published(item), ledger_result::duplicate, "duplicate publish accepted");
    expect_result(ledger.consumer_done(item, 3), ledger_result::ok, "consumer rejected");
    expect_result(ledger.consumer_done(item, 3), ledger_result::duplicate, "duplicate consumer accepted");
}

void invalid_rank_and_ctor_rejected() {
    channel_ledger ledger(3, 1, 0, 5, 2);
    const auto     item = accepted_or_die(ledger, 0, 0, 0).granted.value();
    expect_result(ledger.producer_ready(item, 5), ledger_result::invalid_rank, "out-of-range producer accepted");
    expect_result(ledger.consumer_done(item, 5), ledger_result::invalid_rank, "out-of-range consumer accepted");
    must_fail([]() { channel_ledger bad(0, 1, 0, 5, 2); }, "zero run identity accepted");
    must_fail([]() { channel_ledger bad(1, 0, 0, 5, 2); }, "zero plan identity accepted");
    must_fail([]() { channel_ledger bad(1, 1, 0, 0, 2); }, "zero ranks accepted");
    must_fail([]() { channel_ledger bad(1, 1, 0, 33, 2); }, "33 ranks accepted");
    must_fail([]() { channel_ledger bad(1, 1, 0, 5, 0); }, "zero banks accepted");
}

void bank_ring_and_capacity() {
    channel_ledger ledger(4, 1, 0, 5, 3);
    for (std::uint32_t expect_bank = 0; expect_bank < 3; ++expect_bank) {
        const auto item = accepted_or_die(ledger, expect_bank, 0, 0).granted.value();
        check(item.slot() == expect_bank, "ring order broken");
        complete_inputs(ledger, item);
        complete_readers(ledger, item);
    }
    // All three banks live: no reuse is possible.
    const auto first  = accepted_or_die(ledger, 10, 1, 0).granted.value();
    const auto second = accepted_or_die(ledger, 11, 1, 0).granted.value();
    const auto third  = accepted_or_die(ledger, 12, 1, 0).granted.value();
    check(ledger.reserve(13, 1, 0, 1).code == reserve_code::wait_for_capacity, "full ring accepted a fourth tenant");
    check(ledger.active_bank_count() == 3, "active count drifted");
    // Retire the oldest; the ring must wrap back to it.
    complete_inputs(ledger, first);
    complete_readers(ledger, first);
    const auto fourth = accepted_or_die(ledger, 13, 1, 0).granted.value();
    check(fourth.slot() == first.slot(), "ring wrap landed on the wrong bank");
    (void) second;
    (void) third;
}

void identity_fields_bind_the_lease() {
    channel_ledger ledger(5, 2, 7, 5, 2);
    const auto     item = accepted_or_die(ledger, 30, 6, 2, /*allocation_generation=*/8).granted.value();
    // A lease from the same ledger with a different allocation generation
    // must not pass identity checking even if every id field matches.
    channel_ledger twin(5, 2, 7, 5, 2);
    const auto     same_ids = accepted_or_die(twin, 30, 6, 2, /*allocation_generation=*/9).granted.value();
    expect_result(ledger.consumer_done(same_ids, 0), ledger_result::wrong_identity,
                  "allocation generation mismatch accepted");
    // Cross-channel lease: same window, wrong channel.
    channel_ledger other_channel(5, 2, 8, 5, 2);
    const auto     wrong_channel = accepted_or_die(other_channel, 30, 6, 2, 8).granted.value();
    expect_result(ledger.consumer_done(wrong_channel, 0), ledger_result::wrong_identity,
                  "cross-channel lease accepted");
    check(ledger.bank(item.slot()).allocation_generation == 8, "bank lost its binding");
}

}  // namespace

int main() {
    group_1_two_banks_protect_last_reader();
    group_2_stale_and_foreign_tickets();
    group_3_partial_inputs_and_unpublished_reads();
    group_4_sticky_failure_and_drain();
    group_7_serial_reuse();
    group_8_ticket_exhaustion_is_refused();
    duplicate_message_rejected();
    duplicate_completion_facts_rejected();
    invalid_rank_and_ctor_rejected();
    bank_ring_and_capacity();
    identity_fields_bind_the_lease();
    std::cout << "PASS: channel ledger ownership checks; CPU model only\n";
    return 0;
}
