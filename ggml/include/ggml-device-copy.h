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

// Host DFlash/DSpark feature sanitization uses this exact replacement:
// NaN -> 0, +Inf -> 65504, -Inf -> -65504, every finite value unchanged.
// 65504 is the largest finite f16 value. Do not clamp finite values above it:
// the host path leaves those massive activations intact.
#define GGML_SANITIZE_NONFINITE_F16_MAX 65504.0f

static inline float ggml_sanitize_nonfinite_f32(float v) {
    if (v != v) {
        return 0.0f;
    }
    // Inf compares equal to itself after halving. A finite value does not.
    if (v > 0.0f && v * 0.5f == v) {
        return GGML_SANITIZE_NONFINITE_F16_MAX;
    }
    if (v < 0.0f && v * 0.5f == v) {
        return -GGML_SANITIZE_NONFINITE_F16_MAX;
    }
    return v;
}

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

// In-place replacement of non-finite F32 values, matching
// ggml_sanitize_nonfinite_f32. offset and bytes are float-aligned and must
// lie inside the contiguous tensor. Work is queued on the executor compute
// stream; there is no host readback. dry_run validates and records nothing.
// A missing backend implementation returns false.
GGML_API bool ggml_backend_device_sanitize_nonfinite(
        ggml_backend_t executor, struct ggml_tensor * tensor,
        size_t offset, size_t bytes, bool dry_run);

typedef bool (*ggml_backend_device_sanitize_nonfinite_t)(
        ggml_backend_t, struct ggml_tensor *, size_t, size_t, bool);

#ifdef __cplusplus
}
#endif
