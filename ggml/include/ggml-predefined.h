#pragma once

#include "ggml.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Capacity is immutable after definition. A frame describes useful work, not
// padded work: inactive rows must never enter KV/recurrent state or collectives.
// This ABI is backend-independent. It does NOT by itself make an arbitrary
// ggml graph capacity-aware; a backend must lower every stateful operator before
// advertising support for executing such a graph.
#define GGML_PREDEFINED_ABI_VERSION 1u
#define GGML_PREDEFINED_FRAME_SLOTS 2u
#define GGML_PREDEFINED_MAX_RANKS 8u

enum ggml_predefined_phase {
    GGML_PREDEFINED_PREFILL = 0,
    GGML_PREDEFINED_TARGET  = 1,
    GGML_PREDEFINED_DRAFT   = 2,
    GGML_PREDEFINED_CATCHUP = 3,
    GGML_PREDEFINED_PHASE_COUNT = 4,
};

struct ggml_predefined_limits {
    uint32_t version;
    uint32_t struct_size;
    uint32_t sequences;
    uint32_t draft_tokens;    // maximum draft length PER sequence
    uint32_t ubatch_tokens;   // maximum total rows in one execution
    uint32_t context_tokens;  // maximum logical position + 1 PER sequence
    uint32_t output_rows;     // independent of input/hidden capacity
    uint32_t model_width;
    uint32_t hidden_width;    // e.g. four residual streams, not model_width
    uint32_t vocabulary;
    uint32_t wire_bytes;      // 2 or 4; reductions/hidden remain F32
    uint32_t ranks;
};

struct ggml_predefined_capacity {
    struct ggml_predefined_limits limits;
    uint32_t verify_tokens;
    uint32_t tokens;
    uint64_t hidden_bytes;
    uint64_t logits_bytes;
    uint64_t payload_bytes;   // PER rank, PER bank; excludes protocol headers
};

struct ggml_predefined_request {
    enum ggml_predefined_phase phase;
    uint32_t sequences;
    uint32_t tokens;
    uint32_t outputs;
    uint32_t context_tokens;
    uint32_t draft_tokens;    // total candidate rows, excluding sampled rows
    uint32_t accepted_tokens; // total accepted candidates, excluding sampled rows
    uint32_t draft_step;      // zero based; used only for DRAFT
};

// std430-compatible 64-byte frame. The CPU prepares an inactive slot; submitted
// slots remain immutable until ALL recorded native completion values retire.
// No shader derives physical strides from active_tokens: strides belong to the
// immutable capacity layout. uint64_t epoch is split for baseline GLSL support.
struct ggml_predefined_frame {
    uint32_t version;
    uint32_t phase;
    uint32_t epoch_lo;
    uint32_t epoch_hi;
    uint32_t active_sequences;
    uint32_t active_tokens;
    uint32_t active_outputs;
    uint32_t context_tokens;
    uint32_t draft_tokens;
    uint32_t accepted_tokens;
    uint32_t draft_step;
    uint32_t payload_elements;
    uint32_t payload_bytes;
    uint32_t slot;
    uint32_t reserved[2];
};

enum ggml_predefined_extent {
    GGML_PREDEFINED_EXTENT_ONE       = 0,
    GGML_PREDEFINED_EXTENT_TOKENS    = 1,
    GGML_PREDEFINED_EXTENT_OUTPUTS   = 2,
    GGML_PREDEFINED_EXTENT_SEQUENCES = 3,
    GGML_PREDEFINED_EXTENT_CONTEXT   = 4,
    GGML_PREDEFINED_EXTENT_PAYLOAD   = 5,
};

// ceil((frame[extent] * scale) / divisor). All arithmetic is checked before
// publishing arguments. Static extents use ONE. There is no shape lookup table.
struct ggml_predefined_axis {
    uint32_t extent;
    uint32_t scale;
    uint32_t divisor;
    uint32_t limit;          // physical-device maxComputeWorkGroupCount[axis]
};

struct ggml_predefined_dispatch {
    uint32_t phase_mask;
    uint32_t reserved[3];
    struct ggml_predefined_axis axis[3];
};

struct ggml_predefined_dispatch_args {
    uint32_t x, y, z;
};

// Serial entries share temporary storage. Persistent hidden/KV/rollback data
// must be budgeted separately; it is never aliased with the temporary arena.
struct ggml_predefined_workspace {
    uint64_t scratch_bytes[GGML_PREDEFINED_PHASE_COUNT];
    uint64_t persistent_bytes;
    uint64_t alignment;
};

struct ggml_predefined_workspace_layout {
    uint64_t scratch_offset;
    uint64_t scratch_bytes;
    uint64_t persistent_offset;
    uint64_t persistent_bytes;
    uint64_t total_bytes;
};

// Pure planning helpers. False leaves the output unchanged and writes a bounded
// explanation to error, when supplied. They allocate no memory and touch no GPU.
GGML_API bool ggml_predefined_make_capacity(
        const struct ggml_predefined_limits * limits,
        struct ggml_predefined_capacity * capacity, char * error, size_t error_size);
GGML_API bool ggml_predefined_make_frame(
        const struct ggml_predefined_capacity * capacity,
        const struct ggml_predefined_request * request, uint64_t epoch, uint32_t slot,
        struct ggml_predefined_frame * frame, char * error, size_t error_size);
GGML_API bool ggml_predefined_dispatch_arguments(
        const struct ggml_predefined_frame * frame,
        const struct ggml_predefined_dispatch * dispatch,
        struct ggml_predefined_dispatch_args * args, char * error, size_t error_size);
GGML_API bool ggml_predefined_plan_workspace(
        const struct ggml_predefined_workspace * request,
        struct ggml_predefined_workspace_layout * layout, char * error, size_t error_size);

// Two small parameter slots, NOT two graphs or two workspaces. There is no
// waiting in this API. A busy result is backpressure for the owner; it must use
// the existing bounded native retirement path, never overwrite an in-flight
// frame or fabricate completion. Calls are serialized by the session owner.
typedef struct ggml_predefined_slots * ggml_predefined_slots_t;
GGML_API ggml_predefined_slots_t ggml_predefined_slots_new(uint32_t ranks);
// False preserves the metadata owner; the caller must also preserve its GPU
// frame/argument allocations until native retirement has actually completed.
GGML_API bool ggml_predefined_slots_free(ggml_predefined_slots_t slots);
GGML_API bool ggml_predefined_slots_acquire(
        ggml_predefined_slots_t slots, uint64_t epoch, uint32_t * slot);
GGML_API bool ggml_predefined_slots_record_submit(
        ggml_predefined_slots_t slots, uint32_t slot, uint64_t epoch,
        uint32_t rank, uint64_t native_value);
GGML_API bool ggml_predefined_slots_retire(
        ggml_predefined_slots_t slots, uint32_t slot, uint64_t epoch,
        const uint64_t * observed_native_values, size_t count);
GGML_API bool ggml_predefined_slots_cancel(
        ggml_predefined_slots_t slots, uint32_t slot, uint64_t epoch);
GGML_API bool ggml_predefined_slots_pending(ggml_predefined_slots_t slots);

#ifdef __cplusplus
}
static_assert(sizeof(ggml_predefined_frame) == 64, "predefined frame ABI");
static_assert(sizeof(ggml_predefined_axis) == 16, "predefined axis ABI");
static_assert(sizeof(ggml_predefined_dispatch) == 64, "predefined dispatch ABI");
static_assert(sizeof(ggml_predefined_dispatch_args) == 12, "Vulkan indirect dispatch ABI");
#endif
