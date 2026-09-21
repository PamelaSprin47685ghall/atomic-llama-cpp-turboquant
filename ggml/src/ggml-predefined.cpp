#include "ggml-predefined.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <iterator>
#include <limits>
#include <new>

namespace {

bool fail(char * error, size_t size, const char * message) {
    if (error != nullptr && size != 0) {
        std::snprintf(error, size, "%s", message);
    }
    return false;
}

void success(char * error, size_t size) {
    if (error != nullptr && size != 0) {
        error[0] = '\0';
    }
}

bool multiply(uint64_t a, uint64_t b, uint64_t & out) {
    if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b) {
        return false;
    }
    out = a * b;
    return true;
}

bool align_up(uint64_t value, uint64_t alignment, uint64_t & out) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0 ||
        value > std::numeric_limits<uint64_t>::max() - (alignment - 1)) {
        return false;
    }
    out = (value + alignment - 1) & ~(alignment - 1);
    return true;
}

bool same_ticket(uint32_t slot, uint64_t epoch) {
    return slot < GGML_PREDEFINED_FRAME_SLOTS && epoch != 0;
}

} // namespace

bool ggml_predefined_make_capacity(const ggml_predefined_limits * p, ggml_predefined_capacity * out,
                                   char * error, size_t error_size) {
    if (p == nullptr || out == nullptr) {
        return fail(error, error_size, "null predefined capacity argument");
    }
    if (p->version != GGML_PREDEFINED_ABI_VERSION || p->struct_size != sizeof(*p)) {
        return fail(error, error_size, "unsupported predefined limits ABI");
    }
    if (p->sequences == 0 || p->ubatch_tokens == 0 || p->context_tokens == 0 || p->output_rows == 0 ||
        p->model_width == 0 || p->hidden_width == 0 || p->vocabulary == 0 ||
        p->ranks == 0 || p->ranks > GGML_PREDEFINED_MAX_RANKS || (p->wire_bytes != 2 && p->wire_bytes != 4)) {
        return fail(error, error_size, "invalid predefined capacity limit");
    }
    // Add in 64 bits: UINT32_MAX drafts must not wrap to an empty verification.
    const uint64_t verify = uint64_t(p->sequences) * (uint64_t(p->draft_tokens) + 1);
    if (verify > p->ubatch_tokens || verify > p->output_rows ||
        uint64_t(p->draft_tokens) + 1 > p->context_tokens || p->sequences > p->ubatch_tokens) {
        return fail(error, error_size, "maximum verification must fit one microbatch, its outputs, and context");
    }
    if (p->output_rows > p->ubatch_tokens) {
        return fail(error, error_size, "output capacity exceeds input row capacity");
    }
    ggml_predefined_capacity c{};
    c.limits = *p;
    c.verify_tokens = (uint32_t) verify;
    c.tokens = p->ubatch_tokens;
    uint64_t hidden_elements, logits_elements, payload_elements;
    if (!multiply(c.tokens, p->hidden_width, hidden_elements) ||
        !multiply(p->output_rows, p->vocabulary, logits_elements) ||
        !multiply(c.tokens, p->model_width, payload_elements) ||
        !multiply(hidden_elements, sizeof(float), c.hidden_bytes) ||
        !multiply(logits_elements, sizeof(float), c.logits_bytes) ||
        !multiply(payload_elements, p->wire_bytes, c.payload_bytes) ||
        payload_elements > UINT32_MAX || c.payload_bytes > UINT32_MAX ||
        c.hidden_bytes > SIZE_MAX || c.logits_bytes > SIZE_MAX || c.payload_bytes > SIZE_MAX) {
        return fail(error, error_size, "predefined capacity arithmetic exceeds the address/dispatch ABI");
    }
    *out = c;
    success(error, error_size);
    return true;
}

bool ggml_predefined_make_frame(const ggml_predefined_capacity * cap, const ggml_predefined_request * r,
                                uint64_t epoch, uint32_t slot, ggml_predefined_frame * out,
                                char * error, size_t error_size) {
    if (cap == nullptr || r == nullptr || out == nullptr) {
        return fail(error, error_size, "null predefined frame argument");
    }
    const auto & p = cap->limits;
    if (p.version != GGML_PREDEFINED_ABI_VERSION || p.struct_size != sizeof(p) ||
        !same_ticket(slot, epoch) || unsigned(r->phase) >= GGML_PREDEFINED_PHASE_COUNT) {
        return fail(error, error_size, "invalid predefined frame ABI, phase, or ticket");
    }
    if (r->sequences == 0 || r->sequences > p.sequences || r->tokens == 0 ||
        r->tokens < r->sequences || r->tokens > cap->tokens || r->outputs > r->tokens ||
        r->outputs > p.output_rows || r->context_tokens == 0 || r->context_tokens > p.context_tokens ||
        r->draft_tokens > uint64_t(r->sequences) * p.draft_tokens ||
        r->accepted_tokens > r->draft_tokens) {
        return fail(error, error_size, "active frame exceeds its immutable capacity");
    }
    switch (r->phase) {
        case GGML_PREDEFINED_PREFILL:
            if (r->draft_tokens || r->accepted_tokens || r->draft_step) {
                return fail(error, error_size, "prefill cannot carry speculative state");
            }
            break;
        case GGML_PREDEFINED_TARGET:
            if (r->tokens != uint64_t(r->sequences) + r->draft_tokens ||
                r->outputs != r->tokens || r->accepted_tokens || r->draft_step) {
                return fail(error, error_size, "target verification requires sampled rows plus candidates and all logits");
            }
            break;
        case GGML_PREDEFINED_DRAFT:
            if (r->tokens != r->sequences || r->outputs != r->sequences ||
                r->draft_step >= p.draft_tokens || r->draft_tokens || r->accepted_tokens) {
                return fail(error, error_size, "a draft step has one useful row per active sequence");
            }
            break;
        case GGML_PREDEFINED_CATCHUP:
            if (r->tokens != uint64_t(r->sequences) + r->accepted_tokens || r->outputs || r->draft_step) {
                return fail(error, error_size, "catch-up requires only sampled plus accepted rows and no logits");
            }
            break;
        default:
            return fail(error, error_size, "unsupported predefined phase");
    }
    const uint64_t elements = uint64_t(r->tokens) * p.model_width;
    const uint64_t bytes = elements * p.wire_bytes;
    if (elements > UINT32_MAX || bytes > UINT32_MAX || bytes > cap->payload_bytes) {
        return fail(error, error_size, "active payload exceeds its fixed bank");
    }
    ggml_predefined_frame f{};
    f.version          = GGML_PREDEFINED_ABI_VERSION;
    f.phase            = r->phase;
    f.epoch_lo         = (uint32_t) epoch;
    f.epoch_hi         = (uint32_t) (epoch >> 32);
    f.active_sequences = r->sequences;
    f.active_tokens    = r->tokens;
    f.active_outputs   = r->outputs;
    f.context_tokens   = r->context_tokens;
    f.draft_tokens     = r->draft_tokens;
    f.accepted_tokens  = r->accepted_tokens;
    f.draft_step       = r->draft_step;
    f.payload_elements = (uint32_t) elements;
    f.payload_bytes    = (uint32_t) bytes;
    f.slot            = slot;
    *out = f;
    success(error, error_size);
    return true;
}

bool ggml_predefined_dispatch_arguments(const ggml_predefined_frame * f, const ggml_predefined_dispatch * d,
                                        ggml_predefined_dispatch_args * out, char * error, size_t error_size) {
    if (f == nullptr || d == nullptr || out == nullptr || f->version != GGML_PREDEFINED_ABI_VERSION ||
        f->phase >= GGML_PREDEFINED_PHASE_COUNT || d->phase_mask >> GGML_PREDEFINED_PHASE_COUNT) {
        return fail(error, error_size, "invalid predefined indirect dispatch definition");
    }
    uint32_t groups[3]{};
    const uint32_t extents[] = {1, f->active_tokens, f->active_outputs, f->active_sequences,
                               f->context_tokens, f->payload_elements};
    for (size_t i = 0; i < 3; ++i) {
        const auto & a = d->axis[i];
        if (a.extent >= sizeof(extents) / sizeof(extents[0]) || a.divisor == 0 || a.limit == 0) {
            return fail(error, error_size, "invalid predefined dispatch axis");
        }
        const uint64_t n = uint64_t(extents[a.extent]) * a.scale;
        const uint64_t count = n / a.divisor + (n % a.divisor != 0);
        if (count > a.limit) {
            return fail(error, error_size, "predefined dispatch exceeds device workgroup limit");
        }
        groups[i] = (uint32_t) count;
    }
    // Disabled entries and empty outputs create no workgroups. No fake token,
    // dummy KV destination, or per-lane early return before a barrier is needed.
    if (!(d->phase_mask & (1u << f->phase)) || f->active_tokens == 0) {
        groups[0] = 0;
        groups[1] = groups[2] = 1;
    }
    *out = {groups[0], groups[1], groups[2]};
    success(error, error_size);
    return true;
}

bool ggml_predefined_plan_workspace(const ggml_predefined_workspace * r, ggml_predefined_workspace_layout * out,
                                    char * error, size_t error_size) {
    if (r == nullptr || out == nullptr) {
        return fail(error, error_size, "null predefined workspace argument");
    }
    ggml_predefined_workspace_layout layout{};
    const uint64_t scratch = *std::max_element(std::begin(r->scratch_bytes), std::end(r->scratch_bytes));
    if (!align_up(scratch, r->alignment, layout.scratch_bytes) ||
        !align_up(r->persistent_bytes, r->alignment, layout.persistent_bytes) ||
        layout.scratch_bytes > UINT64_MAX - layout.persistent_bytes) {
        return fail(error, error_size, "predefined workspace size or alignment overflow");
    }
    layout.persistent_offset = layout.scratch_bytes;
    layout.total_bytes = layout.scratch_bytes + layout.persistent_bytes;
    if (layout.total_bytes > SIZE_MAX) {
        return fail(error, error_size, "predefined workspace exceeds host address space");
    }
    *out = layout;
    success(error, error_size);
    return true;
}

struct ggml_predefined_slots {
    struct slot_state {
        uint64_t epoch = 0;
        std::array<uint64_t, GGML_PREDEFINED_MAX_RANKS> submitted{};
    };
    uint32_t ranks = 0;
    uint64_t last_epoch = 0;
    std::array<uint64_t, GGML_PREDEFINED_MAX_RANKS> last_native{};
    std::array<slot_state, GGML_PREDEFINED_FRAME_SLOTS> slots{};
};

ggml_predefined_slots_t ggml_predefined_slots_new(uint32_t ranks) {
    if (ranks == 0 || ranks > GGML_PREDEFINED_MAX_RANKS) {
        return nullptr;
    }
    auto * slots = new (std::nothrow) ggml_predefined_slots;
    if (slots) {
        slots->ranks = ranks;
    }
    return slots;
}

bool ggml_predefined_slots_pending(ggml_predefined_slots_t slots) {
    if (!slots) {
        return false;
    }
    for (const auto & s : slots->slots) {
        if (s.epoch != 0) {
            return true;
        }
    }
    return false;
}

bool ggml_predefined_slots_free(ggml_predefined_slots_t slots) {
    // An unretired parameter owner is deliberately preserved. This is not a
    // native wait and must not turn a partial submit into fictitious completion.
    if (ggml_predefined_slots_pending(slots)) {
        return false;
    }
    delete slots;
    return true;
}

bool ggml_predefined_slots_acquire(ggml_predefined_slots_t state, uint64_t epoch, uint32_t * out) {
    if (!state || !out || epoch == 0 || epoch <= state->last_epoch) {
        return false;
    }
    for (uint32_t i = 0; i < GGML_PREDEFINED_FRAME_SLOTS; ++i) {
        auto & s = state->slots[i];
        if (s.epoch == 0) {
            s.epoch = epoch;
            s.submitted.fill(0);
            state->last_epoch = epoch;
            *out = i;
            return true;
        }
    }
    return false;
}

bool ggml_predefined_slots_record_submit(ggml_predefined_slots_t state, uint32_t slot, uint64_t epoch,
                                        uint32_t rank, uint64_t native_value) {
    if (!state || !same_ticket(slot, epoch) || rank >= state->ranks || native_value == 0 ||
        state->slots[slot].epoch != epoch) {
        return false;
    }
    auto & value = state->slots[slot].submitted[rank];
    if (native_value <= state->last_native[rank]) {
        return false;
    }
    value = native_value;
    state->last_native[rank] = native_value;
    return true;
}

bool ggml_predefined_slots_retire(ggml_predefined_slots_t state, uint32_t slot, uint64_t epoch,
                                 const uint64_t * observed, size_t count) {
    if (!state || !same_ticket(slot, epoch) || !observed || count != state->ranks ||
        state->slots[slot].epoch != epoch) {
        return false;
    }
    const auto & submitted = state->slots[slot].submitted;
    bool any_submit = false;
    for (uint32_t i = 0; i < state->ranks; ++i) {
        any_submit |= submitted[i] != 0;
        if (observed[i] < submitted[i]) {
            return false;
        }
    }
    // An acquired-but-unsubmitted frame must use cancel, never pretend that a
    // GPU receipt retired it. Partial submission waits ONLY on submitted ranks.
    if (!any_submit) {
        return false;
    }
    state->slots[slot] = {};
    return true;
}

bool ggml_predefined_slots_cancel(ggml_predefined_slots_t state, uint32_t slot, uint64_t epoch) {
    if (!state || !same_ticket(slot, epoch) || state->slots[slot].epoch != epoch) {
        return false;
    }
    for (uint64_t value : state->slots[slot].submitted) {
        if (value != 0) {
            return false;
        }
    }
    state->slots[slot] = {};
    return true;
}
