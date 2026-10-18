#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace tq::scheduling {

using slot_id = std::uint32_t;

enum class stage {
    idle,
    prompt,
    generation,
};

struct slot_state {
    slot_id id;
    stage state;
};

// Results own their values. They do not refer to the input or selector.
struct selection {
    stage next = stage::generation;
    std::optional<slot_id> prompt_slot;
    std::vector<slot_id> generation_slots;
};

// Keep the current prompt until it finishes. Then use the first prompt in
// caller-supplied slot order. Generation can starve while prompts keep arriving.
class prefill_selector {
public:
    // Duplicate ids or invalid stage values throw std::invalid_argument and
    // leave the current prompt unchanged. No input reference is retained.
    selection choose(const std::vector<slot_state> & slots);
    void reset() noexcept;

private:
    std::optional<slot_id> current_prompt_;
};

} // namespace tq::scheduling
