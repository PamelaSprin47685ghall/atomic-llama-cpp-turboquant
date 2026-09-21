#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Small, device-only handoffs. Unlike ggml_backend_tensor_copy_async, this
// API must never fall back to a host read/write. All ranges are preflighted
// before any command is recorded. Unsupported/invalid ranges are rejected
// during preflight. A backend execution error may leave recorded work and is
// fatal to the handoff: retain/retire its owners, never retry as a host copy.
#define GGML_DEVICE_COPY_MAX_RANGES 8

struct ggml_device_copy_range {
    const struct ggml_tensor * src;
    struct ggml_tensor * dst;
    size_t src_offset;
    size_t dst_offset;
    size_t bytes;
};

// Executor owns the destination queue. Before calling, the source writes must
// already be ordered before that queue (same queue, or an existing completed
// source dependency). No new cross-device synchronization is implied here.
// The tensor allocations must outlive the queued commands. Only contiguous
// F32 ranges are supported by this initial handoff contract.
GGML_API bool ggml_backend_device_copy_ranges(
        ggml_backend_t executor, const struct ggml_device_copy_range * ranges,
        size_t n_ranges, bool dry_run);

// Shared side-effect-free validation. Backends additionally check physical
// device ownership, buffer bounds and aliased byte ranges.
GGML_API bool ggml_device_copy_ranges_valid(
        const struct ggml_device_copy_range * ranges, size_t n_ranges);

typedef bool (*ggml_backend_device_copy_ranges_t)(
        ggml_backend_t, const struct ggml_device_copy_range *, size_t, bool);

#ifdef __cplusplus
}
#endif
