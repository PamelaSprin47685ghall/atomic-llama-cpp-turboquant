#include <tq/scheduling/prefill.h>

#include <stdexcept>
#include <unordered_set>

namespace tq::scheduling {

selection prefill_selector::choose(const std::vector<slot_state> & slots) {
    std::unordered_set<slot_id> seen;
    seen.reserve(slots.size());
    for (const auto & slot : slots) {
        if (!seen.insert(slot.id).second) {
            throw std::invalid_argument("duplicate slot id");
        }
        if (slot.state != stage::idle && slot.state != stage::prompt && slot.state != stage::generation) {
            throw std::invalid_argument("invalid slot stage");
        }
    }

    std::optional<slot_id> first_prompt;
    bool current_still_prompt = false;
    selection result;
    for (const auto & slot : slots) {
        if (slot.state == stage::prompt) {
            if (!first_prompt) {
                first_prompt = slot.id;
            }
            if (current_prompt_ && slot.id == *current_prompt_) {
                current_still_prompt = true;
            }
        } else if (slot.state == stage::generation) {
            result.generation_slots.push_back(slot.id);
        }
    }

    if (!current_still_prompt) {
        current_prompt_ = first_prompt;
    }
    if (current_prompt_) {
        result.next = stage::prompt;
        result.prompt_slot = current_prompt_;
        result.generation_slots.clear();
    }
    return result;
}

void prefill_selector::reset() noexcept {
    current_prompt_.reset();
}

} // namespace tq::scheduling
