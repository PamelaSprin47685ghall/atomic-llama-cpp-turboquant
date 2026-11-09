#include "commit_stop.h"

#include <tq/text/stream.h>

#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using tq::tp5::apply_commit_trims;
using tq::tp5::classify_commit_stop;
using tq::tp5::commit_stop;
using tq::tp5::commit_stop_reason;
using tq::tp5::commit_stop_state;
using tq::tp5::has_later_row;
using tq::tp5::replay_token_must_match;
using tq::tp5::stopped_source_position;

void require(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

void check_stop_classification() {
    require(classify_commit_stop(false, false, false, 1, 99, 99) == commit_stop::none,
            "plain row produced a stop");
    require(classify_commit_stop(true, false, false, 1, 99, 99) == commit_stop::eos,
            "EOG row did not stop");
    require(classify_commit_stop(true, true, false, 1, 99, 99) == commit_stop::none,
            "ignore_eos did not suppress EOG");
    require(classify_commit_stop(false, false, true, 2, 2, 99) == commit_stop::cancelled,
            "cancel row did not stop");
    require(classify_commit_stop(true, false, true, 1, 1, 99) == commit_stop::cancelled,
            "cancel did not take precedence over EOG");
    require(classify_commit_stop(false, false, false, 3, 99, 3) == commit_stop::budget,
            "budget row did not stop");
    require(commit_stop_reason(commit_stop::cancelled) == "cancelled", "cancel reason differs");
    require(commit_stop_reason(commit_stop::eos) == "eos", "EOG reason differs");
    require(commit_stop_reason(commit_stop::budget) == "budget", "budget reason differs");
    require(commit_stop_reason(commit_stop::none).empty(), "none reason is not empty");
}

void check_stop_state() {
    commit_stop_state with_later(1);
    require(with_later.note_stop(0, 100, true, commit_stop::eos) == "eos",
            "state stop reason differs");
    require(with_later.trims()[0] == std::optional<std::int64_t>(101),
            "state did not record the target trim position");

    commit_stop_state last_row(1);
    require(last_row.note_stop(0, 100, false, commit_stop::eos) == "eos",
            "last-row stop reason differs");
    require(!last_row.trims()[0].has_value(),
            "last-row stop recorded an unnecessary target trim");

    commit_stop_state no_stop(1);
    require(no_stop.note_stop(0, 100, true, commit_stop::none).empty(),
            "no-stop row produced a reason");
    require(!no_stop.trims()[0].has_value(), "no-stop row recorded a target trim");

    commit_stop_state reset_state(1);
    reset_state.note_stop(0, 100, true, commit_stop::budget);
    reset_state.reset();
    require(!reset_state.trims()[0].has_value(), "reset did not clear the target trim");
}

void check_has_later_row() {
    require(has_later_row({ 0, 0, 0 }, 0, 0), "first row did not see the same request later");
    require(has_later_row({ 0, 0, 0 }, 1, 0), "second row did not see the same request later");
    require(!has_later_row({ 0, 0, 0 }, 2, 0), "last row saw a nonexistent later row");
    require(!has_later_row({ 0, 1 }, 0, 0), "different-request row was treated as later");
    require(!has_later_row({ 0, 1 }, 1, 1), "last row of a request saw a later row");
    require(has_later_row({ 0, 1, 1 }, 0, 1), "later same-request row was not found");
    require(!has_later_row({}, 0, 0), "empty batch reported a later row");
}

void check_stopped_source_position() {
    const std::vector<std::size_t>  locals    = { 0, 1, 1, 2 };
    const std::vector<std::int64_t> positions = { 10, 20, 21, 30 };
    const auto at = [&](std::size_t row) { return positions[row]; };

    require(stopped_source_position(locals, 0, 0, at) == std::optional<std::int64_t>(10),
            "first request source position differs");
    require(stopped_source_position(locals, 1, 1, at) == std::optional<std::int64_t>(20),
            "second request first row position differs");
    require(stopped_source_position(locals, 2, 1, at) == std::optional<std::int64_t>(20),
            "second request later row used its own position instead of the source");
    require(stopped_source_position(locals, 3, 2, at) == std::optional<std::int64_t>(30),
            "third request source position differs");
    require(!stopped_source_position(locals, 1, 0, at).has_value(),
            "row from another request was accepted as the stopped local");
    require(!stopped_source_position(locals, 4, 0, at).has_value(),
            "out-of-range row was accepted");
}

void check_replay_contract() {
    // Original contract: accepted draft rows are exact. The correction row
    // (row == accepted without a stop) is allowed to differ inside the
    // KL/rel/overlap gate; do not turn it into an extra rejection.
    require(replay_token_must_match(0, 2, false), "accepted draft row was not exact");
    require(replay_token_must_match(1, 2, false), "accepted draft row was not exact");
    require(!replay_token_must_match(2, 2, false), "correction row became an extra rejection");
    require(!replay_token_must_match(3, 2, false), "row after correction became exact");

    // A stop row must be exact because its token decides the stop.
    require(replay_token_must_match(2, 2, true), "stop correction row was not exact");

    // Full accept does not replay the extra row n; the commit loop reads the
    // original target logits there.
    require(replay_token_must_match(0, 3, false), "full-accept accepted row was not exact");
    require(replay_token_must_match(2, 3, false), "full-accept accepted row was not exact");
    require(!replay_token_must_match(3, 3, false), "full-accept extra row was treated as a replay row");
}

void check_apply_commit_trims() {
    const std::vector<std::optional<std::int64_t>> trims = {
        std::int64_t(101),
        std::nullopt,
        std::int64_t(203),
    };
    std::vector<std::pair<std::size_t, std::int64_t>> calls;
    apply_commit_trims(trims, [&](std::size_t local, std::int64_t trim_from) {
        calls.emplace_back(local, trim_from);
    });
    require(calls.size() == 2, "trim application count differs");
    require(calls[0] == std::make_pair(std::size_t(0), std::int64_t(101)),
            "first trim application differs");
    require(calls[1] == std::make_pair(std::size_t(2), std::int64_t(203)),
            "second trim application differs");
}

void check_generation_stream() {
    tq::text::generation_stream stream;
    std::string                 text;
    stream.push_into("a", text);
    stream.push_into("b", text);
    stream.finish_into(text);
    require(text == "ab", "stream text differs");

    bool rejected = false;
    try {
        stream.push_into("c", text);
    } catch (const std::logic_error &) {
        rejected = true;
    }
    require(rejected, "finished stream accepted a later push");
}

void check_stop_state_with_stream() {
    // The runtime's observable stop sequence: commit one piece, record the
    // stop/trim through the production state, finish the real stream once.
    commit_stop_state state(1);
    tq::text::generation_stream stream;
    std::string text;

    stream.push_into("a", text);
    const auto reason = state.note_stop(0, 200, true, commit_stop::eos);
    stream.finish_into(text);

    require(reason == "eos", "stop sequence reason differs");
    require(state.trims()[0] == std::optional<std::int64_t>(201),
            "stop sequence trim position differs");
    require(text == "a", "stop sequence text differs");

    std::vector<std::pair<std::size_t, std::int64_t>> applied;
    apply_commit_trims(state.trims(), [&](std::size_t local, std::int64_t trim_from) {
        applied.emplace_back(local, trim_from);
    });
    require(applied.size() == 1, "stop sequence did not apply one target trim");
    require(applied[0] == std::make_pair(std::size_t(0), std::int64_t(201)),
            "stop sequence applied the wrong target trim");

    // A later row for the same request must be skipped by the commit loop;
    // the stream itself correctly rejects it.
    bool rejected = false;
    try {
        stream.push_into("b", text);
    } catch (const std::logic_error &) {
        rejected = true;
    }
    require(rejected, "stop sequence accepted a later push");
}

} // namespace

int main() {
    try {
        check_stop_classification();
        check_stop_state();
        check_has_later_row();
        check_stopped_source_position();
        check_replay_contract();
        check_apply_commit_trims();
        check_generation_stream();
        check_stop_state_with_stream();
        std::cout << "commit stop tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "commit stop test failed: " << error.what() << '\n';
        return 1;
    }
}
