#include <tq/relay/detail/arithmetic.h>
#include <tq/relay/ledger.h>

#include <stdexcept>
#include <utility>

namespace tq::relay {

namespace {

const char * reserve_names[] = { "accepted", "wait_for_capacity", "sequence_exhausted", "invalid_message", "stopped" };
const char * result_names[]  = { "ok",        "stale_ticket", "wrong_identity", "not_ready",
                                 "duplicate", "invalid_rank", "stopped" };

template <std::size_t N> const char * name_of(const char * const (&names)[N], std::size_t index) noexcept {
    return index < N ? names[index] : "unknown";
}

}  // namespace

const char * to_string(reserve_code code) noexcept {
    return name_of(reserve_names, static_cast<std::size_t>(code));
}

const char * to_string(ledger_result result) noexcept {
    return name_of(result_names, static_cast<std::size_t>(result));
}

lease::lease(std::shared_ptr<const domain> owner,
             message_id                    message,
             std::uint64_t                 allocation_generation,
             std::uint32_t                 slot) noexcept :
    owner_(std::move(owner)),
    message_(message),
    allocation_generation_(allocation_generation),
    slot_(slot) {}

channel_ledger::channel_ledger(std::uint64_t run_id,
                               std::uint64_t plan_id,
                               std::uint64_t channel_id,
                               std::uint32_t ranks,
                               std::uint32_t banks) :
    run_id_(run_id),
    plan_id_(plan_id),
    channel_id_(channel_id),
    ranks_(ranks) {
    if (run_id == 0 || plan_id == 0) {
        throw std::invalid_argument("run and plan identity must be positive");
    }
    if (ranks == 0 || ranks > 32) {
        throw std::invalid_argument("rank count must be in [1, 32]");
    }
    if (banks == 0) {
        throw std::invalid_argument("bank count must be positive");
    }
    member_mask_ = ranks == 32 ? 0xFFFFFFFFu : ((1u << ranks) - 1u);
    banks_.resize(banks);
    owner_ = std::make_shared<const lease::domain>();
}

reserve_reply channel_ledger::reserve(std::uint64_t window_id,
                                      std::uint64_t window_generation,
                                      std::uint64_t segment_id,
                                      std::uint64_t allocation_generation) {
    if (failed_ || drained_) {
        return { reserve_code::stopped, std::nullopt };
    }
    if (next_ticket_ == 0) {
        return { reserve_code::sequence_exhausted, std::nullopt };
    }
    if (active_duplicate(window_id, window_generation, segment_id)) {
        return { reserve_code::invalid_message, std::nullopt };
    }
    const std::uint64_t ticket     = next_ticket_;
    const std::uint32_t slot_index = static_cast<std::uint32_t>((ticket - 1) % banks_.size());
    auto &              slot       = banks_[slot_index];
    if (slot.occupied && !(slot.input_released && slot.output_published && slot.consumed == member_mask_)) {
        return { reserve_code::wait_for_capacity, std::nullopt };
    }
    slot                       = {};
    slot.occupied              = true;
    slot.ticket                = ticket;
    slot.window_id             = window_id;
    slot.window_generation     = window_generation;
    slot.segment_id            = segment_id;
    slot.allocation_generation = allocation_generation;
    next_ticket_               = detail::advance_ticket(next_ticket_);
    message_id message{ run_id_, plan_id_, window_id, window_generation, channel_id_, segment_id, ticket };
    return { reserve_code::accepted, lease(owner_, message, allocation_generation, slot_index) };
}

ledger_result channel_ledger::check_identity(const lease & held, const slot_state *& slot) {
    if (failed_ || drained_) {
        return ledger_result::stopped;
    }
    if (held.owner_ != owner_ || held.slot_ >= banks_.size()) {
        return ledger_result::wrong_identity;
    }
    slot = &banks_[held.slot_];
    if (!slot->occupied || held.message_.ticket != slot->ticket) {
        return ledger_result::stale_ticket;
    }
    if (held.message_.run_id != run_id_ || held.message_.plan_id != plan_id_ ||
        held.message_.channel_id != channel_id_ || held.message_.window_id != slot->window_id ||
        held.message_.window_generation != slot->window_generation || held.message_.segment_id != slot->segment_id ||
        held.allocation_generation_ != slot->allocation_generation) {
        return ledger_result::wrong_identity;
    }
    return ledger_result::ok;
}

bool channel_ledger::active_duplicate(std::uint64_t window_id,
                                      std::uint64_t window_generation,
                                      std::uint64_t segment_id) const noexcept {
    for (const auto & slot : banks_) {
        if (slot.occupied && slot.window_id == window_id && slot.window_generation == window_generation &&
            slot.segment_id == segment_id) {
            return true;
        }
    }
    return false;
}

ledger_result channel_ledger::producer_ready(const lease & held, std::uint32_t rank) {
    const slot_state *  slot     = nullptr;
    const ledger_result identity = check_identity(held, slot);
    if (identity != ledger_result::ok) {
        return identity;
    }
    if (rank >= ranks_) {
        return ledger_result::invalid_rank;
    }
    if (slot->input_released || (slot->ready & (1u << rank)) != 0) {
        return ledger_result::duplicate;
    }
    auto & live = banks_[held.slot_];
    live.ready |= (1u << rank);
    return ledger_result::ok;
}

ledger_result channel_ledger::reduce_done(const lease & held) {
    const slot_state *  slot     = nullptr;
    const ledger_result identity = check_identity(held, slot);
    if (identity != ledger_result::ok) {
        return identity;
    }
    if (slot->ready != member_mask_) {
        return ledger_result::not_ready;
    }
    if (slot->input_released) {
        return ledger_result::duplicate;
    }
    banks_[held.slot_].input_released = true;
    return ledger_result::ok;
}

ledger_result channel_ledger::output_published(const lease & held) {
    const slot_state *  slot     = nullptr;
    const ledger_result identity = check_identity(held, slot);
    if (identity != ledger_result::ok) {
        return identity;
    }
    if (!slot->input_released) {
        return ledger_result::not_ready;
    }
    if (slot->output_published) {
        return ledger_result::duplicate;
    }
    banks_[held.slot_].output_published = true;
    return ledger_result::ok;
}

ledger_result channel_ledger::consumer_done(const lease & held, std::uint32_t rank) {
    const slot_state *  slot     = nullptr;
    const ledger_result identity = check_identity(held, slot);
    if (identity != ledger_result::ok) {
        return identity;
    }
    if (rank >= ranks_) {
        return ledger_result::invalid_rank;
    }
    if (!slot->output_published) {
        return ledger_result::not_ready;
    }
    if ((slot->consumed & (1u << rank)) != 0) {
        return ledger_result::duplicate;
    }
    banks_[held.slot_].consumed |= (1u << rank);
    return ledger_result::ok;
}

void channel_ledger::fail(const std::string & reason) {
    if (reason.empty()) {
        throw std::invalid_argument("failure needs a reason");
    }
    if (!failed_) {
        failed_  = true;
        failure_ = reason;
    }
}

bool channel_ledger::mark_drained(bool all_native_work_done) {
    if (!all_native_work_done) {
        return false;
    }
    drained_ = true;
    return true;
}

std::uint32_t channel_ledger::active_bank_count() const noexcept {
    std::uint32_t count = 0;
    for (const auto & slot : banks_) {
        if (slot.occupied) {
            ++count;
        }
    }
    return count;
}

bank_state channel_ledger::bank(std::uint32_t slot) const noexcept {
    if (slot >= banks_.size() || !banks_[slot].occupied) {
        return {};
    }
    const auto & live = banks_[slot];
    return { true,
             live.ticket,
             live.window_id,
             live.window_generation,
             live.segment_id,
             live.allocation_generation,
             live.input_released,
             live.output_published,
             live.ready,
             live.consumed };
}

}  // namespace tq::relay
