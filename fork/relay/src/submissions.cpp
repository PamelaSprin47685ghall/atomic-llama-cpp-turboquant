#include <tq/relay/submissions.h>

#include <stdexcept>

namespace tq::relay {

namespace {

const char * submit_names[] = { "not_submitted", "accepted", "unknown_in_flight", "completed" };

const char * name_of(std::size_t index) noexcept {
    constexpr std::size_t count = sizeof(submit_names) / sizeof(submit_names[0]);
    return index < count ? submit_names[index] : "unknown";
}

}  // namespace

const char * to_string(submit_state state) noexcept {
    return name_of(static_cast<std::size_t>(state));
}

run_submissions::run_submissions(std::uint64_t run_id, std::uint32_t ranks) : run_id_(run_id), ranks_(ranks) {
    if (run_id == 0) {
        throw std::invalid_argument("run identity must be positive");
    }
    if (ranks == 0 || ranks > 32) {
        throw std::invalid_argument("rank count must be in [1, 32]");
    }
}

std::optional<attempt_id> run_submissions::begin_attempt(std::uint64_t window_id, std::uint32_t rank) {
    if (rank >= ranks_) {
        throw std::invalid_argument("attempt rank is outside the run membership");
    }
    if (failed_) {
        return std::nullopt;
    }
    attempts_.push_back({ window_id, rank, submit_state::unknown_in_flight, false });
    return attempt_id{ attempts_.size() };
}

bool run_submissions::resolve_accepted(attempt_id id) noexcept {
    if (id.value == 0 || id.value > attempts_.size()) {
        return false;
    }
    auto & item = attempts_[id.value - 1];
    if (item.resolved) {
        return false;
    }
    item.resolved = true;
    item.state    = submit_state::accepted;
    return true;
}

bool run_submissions::resolve_no_effect(attempt_id id) noexcept {
    if (id.value == 0 || id.value > attempts_.size()) {
        return false;
    }
    auto & item = attempts_[id.value - 1];
    if (item.resolved) {
        return false;
    }
    item.resolved = true;
    item.state    = submit_state::not_submitted;
    return true;
}

bool run_submissions::resolve_unknown(attempt_id id) noexcept {
    if (id.value == 0 || id.value > attempts_.size()) {
        return false;
    }
    auto & item = attempts_[id.value - 1];
    if (item.resolved) {
        return false;
    }
    item.resolved = true;
    item.state    = submit_state::unknown_in_flight;
    return true;
}

bool run_submissions::completed(attempt_id id) noexcept {
    if (id.value == 0 || id.value > attempts_.size()) {
        return false;
    }
    auto & item = attempts_[id.value - 1];
    if (item.state != submit_state::accepted && item.state != submit_state::unknown_in_flight) {
        return false;
    }
    item.state = submit_state::completed;
    return true;
}

void run_submissions::fail(const std::string & reason) {
    if (reason.empty()) {
        throw std::invalid_argument("failure needs a reason");
    }
    if (!failed_) {
        failed_  = true;
        failure_ = reason;
    }
}

std::uint32_t run_submissions::accepted_rank_mask(std::uint64_t window_id) const noexcept {
    std::uint32_t mask = 0;
    for (const auto & item : attempts_) {
        if (item.window_id == window_id && item.state == submit_state::accepted) {
            mask |= (1u << item.rank);
        }
    }
    return mask;
}

std::uint32_t run_submissions::unknown_rank_mask(std::uint64_t window_id) const noexcept {
    std::uint32_t mask = 0;
    for (const auto & item : attempts_) {
        if (item.window_id == window_id && item.state == submit_state::unknown_in_flight) {
            mask |= (1u << item.rank);
        }
    }
    return mask;
}

std::uint32_t run_submissions::in_flight_rank_mask(std::uint64_t window_id) const noexcept {
    return accepted_rank_mask(window_id) | unknown_rank_mask(window_id);
}

bool run_submissions::attempts_resolved(std::uint64_t window_id) const noexcept {
    for (const auto & item : attempts_) {
        if (item.window_id == window_id && !item.resolved) {
            return false;
        }
    }
    return true;
}

bool run_submissions::sealed(std::uint64_t window_id) const noexcept {
    if (failed_) {
        return false;
    }
    std::uint32_t member_mask = ranks_ == 32 ? 0xFFFFFFFFu : ((1u << ranks_) - 1u);
    return attempts_resolved(window_id) && unknown_rank_mask(window_id) == 0 &&
           accepted_rank_mask(window_id) == member_mask;
}

bool run_submissions::drained(std::uint64_t window_id) const noexcept {
    return in_flight_rank_mask(window_id) == 0;
}

std::optional<attempt_state> run_submissions::attempt(attempt_id id) const noexcept {
    if (id.value == 0 || id.value > attempts_.size()) {
        return std::nullopt;
    }
    const auto & item = attempts_[id.value - 1];
    return attempt_state{ item.window_id, item.rank, item.state, item.resolved };
}

}  // namespace tq::relay
