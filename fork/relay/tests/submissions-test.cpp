// Native submission accounting: partial five-rank submits, unknown outcomes,
// the submission-sealed gate (guide 52.4, 52.12), and drain coverage from
// actual completion facts only (guide 35.2, 35.3, 52.5).

#include <tq/relay/submissions.h>

#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

using tq::relay::attempt_id;
using tq::relay::run_submissions;
using tq::relay::submit_state;

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

attempt_id begin(run_submissions & submissions, std::uint64_t window_id, std::uint32_t rank) {
    auto id = submissions.begin_attempt(window_id, rank);
    check(id.has_value(), "begin_attempt unexpectedly refused");
    return id.value();
}

void sealed_only_when_all_ranks_accepted_and_resolved() {
    run_submissions     submissions(70, 5);
    const std::uint64_t window = 7;
    attempt_id          ids[5];
    for (std::uint32_t rank = 0; rank < 5; ++rank) {
        ids[rank] = begin(submissions, window, rank);
    }
    check(!submissions.sealed(window), "unresolved attempts cannot seal");
    check(submissions.unknown_rank_mask(window) == 0x1F, "pre-call registration must be unknown, not absent");
    for (std::uint32_t rank = 0; rank < 5; ++rank) {
        check(submissions.resolve_accepted(ids[rank]), "resolve_accepted rejected");
    }
    check(submissions.attempts_resolved(window), "all attempts should be resolved");
    check(submissions.sealed(window), "five accepted ranks must seal the frame");
    check(!submissions.drained(window), "accepted work is still in flight");
}

void partial_submit_tracks_unknown_rank() {
    run_submissions     submissions(71, 5);
    const std::uint64_t window = 8;
    const auto          rank0  = begin(submissions, window, 0);
    const auto          rank1  = begin(submissions, window, 1);
    const auto          rank2  = begin(submissions, window, 2);  // provider call failed, outcome unclear
    check(submissions.resolve_accepted(rank0), "resolve_accepted rejected");
    check(submissions.resolve_accepted(rank1), "resolve_accepted rejected");
    check(submissions.resolve_unknown(rank2), "resolve_unknown rejected");
    check(submissions.accepted_rank_mask(window) == 0x03, "accepted mask drifted");
    check(submissions.unknown_rank_mask(window) == 0x04, "unknown rank not recorded");
    check(submissions.in_flight_rank_mask(window) == 0x07, "in-flight mask drifted");
    check(!submissions.sealed(window), "unknown outcome must block sealing");
    check(!submissions.drained(window), "in-flight work must block drain");
    // Ranks 3 and 4 were never attempted; they must not be counted as
    // accepted or unknown by accident.
    check((submissions.accepted_rank_mask(window) & 0x18) == 0, "unattempted rank marked accepted");
    check((submissions.unknown_rank_mask(window) & 0x18) == 0, "unattempted rank marked unknown");
}

void completion_proof_shrinks_the_in_flight_set() {
    run_submissions     submissions(71, 5);
    const std::uint64_t window = 8;
    const auto          rank0  = begin(submissions, window, 0);
    const auto          rank1  = begin(submissions, window, 1);
    const auto          rank2  = begin(submissions, window, 2);
    check(submissions.resolve_accepted(rank0), "resolve_accepted rejected");
    check(submissions.resolve_accepted(rank1), "resolve_accepted rejected");
    check(submissions.resolve_unknown(rank2), "resolve_unknown rejected");
    check(submissions.completed(rank0), "completed on accepted attempt rejected");
    check(submissions.in_flight_rank_mask(window) == 0x06, "completed rank still in flight");
    check(!submissions.drained(window), "drain before full coverage");
    // Guide 52.12 F10: a later completion fact may cover an attempt whose
    // immediate outcome was unknown.
    check(submissions.completed(rank2), "completed on unknown attempt rejected");
    check(submissions.completed(rank1), "completed on accepted attempt rejected");
    check(submissions.in_flight_rank_mask(window) == 0, "completion did not clear in-flight");
    check(submissions.drained(window), "full coverage must drain");
}

void no_effect_rejections_leave_no_flight() {
    run_submissions     submissions(71, 5);
    const std::uint64_t window = 8;
    const auto          rank3  = begin(submissions, window, 3);
    const auto          rank4  = begin(submissions, window, 4);
    check(submissions.resolve_no_effect(rank3), "resolve_no_effect rejected");
    check(submissions.resolve_no_effect(rank4), "resolve_no_effect rejected");
    check(submissions.in_flight_rank_mask(window) == 0, "no-effect rejection left flight");
    check(submissions.drained(window), "no-effect rejection must not block drain");
    check(!submissions.completed(rank3), "completed accepted a not-submitted attempt");
    check(!submissions.sealed(window), "rejections alone must not seal a frame");
}

void double_resolution_and_stale_ids_rejected() {
    run_submissions submissions(72, 5);
    const auto      item = begin(submissions, 9, 1);
    check(submissions.resolve_accepted(item), "first resolution rejected");
    check(!submissions.resolve_accepted(item), "double resolution accepted");
    check(!submissions.resolve_no_effect(item), "cross-state resolution accepted");
    check(!submissions.resolve_unknown(item), "cross-state resolution accepted");
    check(!submissions.completed(attempt_id{ 0 }), "zero attempt id accepted");
    check(!submissions.completed(attempt_id{ 999 }), "out-of-range attempt id accepted");
    check(!submissions.attempt(attempt_id{ 999 }).has_value(), "out-of-range attempt state returned");
    const auto state = submissions.attempt(item);
    check(state.has_value() && state->state == submit_state::accepted && state->resolved, "attempt state drifted");
    check(state->window_id == 9 && state->rank == 1, "attempt identity drifted");
}

void multiple_attempts_per_rank() {
    run_submissions     submissions(72, 5);
    const std::uint64_t window   = 9;
    const auto          graph    = begin(submissions, window, 0);
    const auto          readback = begin(submissions, window, 0);
    check(submissions.resolve_accepted(graph), "resolve_accepted rejected");
    check(submissions.resolve_accepted(readback), "resolve_accepted rejected");
    // Both belong to rank 0: completing the graph attempt must not clear
    // the rank while the readback attempt still flies.
    check(submissions.completed(graph), "completed rejected");
    check(submissions.in_flight_rank_mask(window) == 0x01, "readback flight lost");
    check(!submissions.drained(window), "second attempt ignored by drain");
    check(submissions.completed(readback), "completed rejected");
    check(submissions.drained(window), "both attempts must drain");
}

void failure_stops_new_attempts_and_blocks_sealing() {
    run_submissions     submissions(73, 5);
    const std::uint64_t window = 10;
    attempt_id          ids[5];
    for (std::uint32_t rank = 0; rank < 5; ++rank) {
        ids[rank] = begin(submissions, window, rank);
        check(submissions.resolve_accepted(ids[rank]), "resolve_accepted rejected");
    }
    check(submissions.sealed(window), "five accepted ranks must seal before failure");
    submissions.fail("rank 2 submit failed");
    submissions.fail("later drain timeout");
    check(submissions.failure_reason() == "rank 2 submit failed", "first failure overwritten");
    check(!submissions.sealed(window), "failed run must not seal");
    check(!submissions.begin_attempt(window + 1, 0).has_value(), "failed run accepted a new attempt");
    // Completion facts still flow after failure: drain remains observable.
    for (const auto id : ids) {
        check(submissions.completed(id), "completed rejected after failure");
    }
    check(submissions.drained(window), "drain proof lost after failure");
}

void ctor_and_rank_validation() {
    must_fail([]() { run_submissions bad(0, 5); }, "zero run identity accepted");
    must_fail([]() { run_submissions bad(1, 0); }, "zero ranks accepted");
    must_fail([]() { run_submissions bad(1, 33); }, "33 ranks accepted");
    run_submissions submissions(1, 5);
    must_fail([&]() { submissions.begin_attempt(0, 5); }, "out-of-range rank accepted");
}

void vacuous_drain_is_documented() {
    // A window that was never attempted has no in-flight work. The ledger
    // reports drained; the caller decides whether an unsubmitted window is
    // releasable (guide 35.5).
    run_submissions submissions(1, 5);
    check(submissions.drained(999), "empty window should be drain-vacuous");
    check(!submissions.sealed(999), "empty window must not seal");
}

}  // namespace

int main() {
    sealed_only_when_all_ranks_accepted_and_resolved();
    partial_submit_tracks_unknown_rank();
    completion_proof_shrinks_the_in_flight_set();
    no_effect_rejections_leave_no_flight();
    double_resolution_and_stale_ids_rejected();
    multiple_attempts_per_rank();
    failure_stops_new_attempts_and_blocks_sealing();
    ctor_and_rank_validation();
    vacuous_drain_is_documented();
    std::cout << "PASS: submission accounting checks; CPU model only\n";
    return 0;
}
