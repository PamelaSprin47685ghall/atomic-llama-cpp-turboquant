#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>

namespace qwen35_prune {

static constexpr int Q2_OPT_BLOCK_SIZE = 64;
static constexpr int Q2_OPT_PACKED_SIZE = 18;

struct q2_opt_block {
    ggml_fp16_t d;
    uint8_t qs[16];
};

static_assert(sizeof(q2_opt_block) == Q2_OPT_PACKED_SIZE, "Q2_0 block layout mismatch");

struct q2_opt_metrics {
    double weighted_sse = 0.0;
    double unweighted_sse = 0.0;
    double weighted_power = 0.0;
};

enum class q2_opt_mode {
    positive,
    signed_scale,
};

// Deterministic weighted Q2_0 encoder. The emitted 18 bytes are valid stock
// GGML_TYPE_Q2_0 and decode as (code - 1) * d. signed_scale permits d < 0.
q2_opt_metrics encode_q2_opt(
        const float * w,
        const float * h,
        q2_opt_block & out,
        q2_opt_mode mode = q2_opt_mode::signed_scale);

// Planner-only scoring path: runs the identical FP16-scale search but skips
// final Q2 byte materialization. The returned metrics still include
// weighted_power=sum(h_i*w_i^2), so retention scoring needs no second pass.
q2_opt_metrics score_q2_opt(
        const float * w,
        const float * h,
        q2_opt_mode mode = q2_opt_mode::signed_scale);

// Same signed search as encode_q2_opt(...signed_scale), but also returns the
// positive-scale optimum found during the +d half of that same sweep. This is
// used for manifest conformance without running the expensive optimizer twice.
q2_opt_metrics encode_q2_opt_signed_compare(
        const float * w,
        const float * h,
        q2_opt_block & out,
        q2_opt_metrics & positive_metrics);

q2_opt_metrics encode_q2_stock(const float * w, const float * h, q2_opt_block & out);

void decode_q2_opt(const q2_opt_block & block, float * out64);

// Slow test oracle: scans every finite FP16 scale. Not for production paths.
q2_opt_metrics encode_q2_oracle(const float * w, const float * h, q2_opt_block & out, bool allow_negative);

bool q2_opt_block_equal(const q2_opt_block & a, const q2_opt_block & b);

} // namespace qwen35_prune
