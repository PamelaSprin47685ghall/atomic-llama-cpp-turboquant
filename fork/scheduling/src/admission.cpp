#include <tq/scheduling/admission.h>

#include <limits>
#include <stdexcept>
#include <utility>

namespace tq::scheduling {

reservation::reservation(std::shared_ptr<const domain> owner, std::uint32_t slot, std::uint64_t request,
                         std::uint64_t generation, std::uint64_t tokens) noexcept :
    owner_(std::move(owner)), slot_(slot), request_(request), generation_(generation), tokens_(tokens) {}

admission_budget::admission_budget(std::uint64_t tokens, std::uint32_t slots) : capacity_(tokens) {
    if (tokens == 0 || slots == 0) {
        throw std::invalid_argument("token capacity and slot count must be positive");
    }
    owner_ = std::make_shared<const reservation::domain>();
    entries_.resize(slots);
}

std::optional<reservation> admission_budget::try_acquire(std::uint64_t request, std::uint64_t tokens) {
    if (tokens == 0 || tokens > capacity_) {
        throw std::invalid_argument("request token demand must be positive and fit the total capacity");
    }
    for (const auto & item : entries_) {
        if (item.tokens != 0 && item.request == request) {
            throw std::invalid_argument("request already has an active reservation");
        }
    }
    if (tokens > available_tokens() || active_ == slot_count()) {
        return std::nullopt;
    }
    if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("reservation generation is exhausted");
    }
    for (std::uint32_t slot = 0; slot < slot_count(); ++slot) {
        auto & item = entries_[slot];
        if (item.tokens != 0) {
            continue;
        }
        reservation result(owner_, slot, request, generation_ + 1, tokens);
        item = {request, ++generation_, tokens};
        reserved_ += tokens;
        ++active_;
        return result;
    }
    throw std::logic_error("active count does not match occupied slots");
}

bool admission_budget::release(const reservation & held) noexcept {
    if (held.owner_ != owner_ || held.slot_ >= slot_count()) {
        return false;
    }
    auto & item = entries_[held.slot_];
    if (item.tokens == 0 || item.generation != held.generation_ || item.request != held.request_ ||
        item.tokens != held.tokens_) {
        return false;
    }
    reserved_ -= item.tokens;
    --active_;
    item = {};
    return true;
}

} // namespace tq::scheduling
