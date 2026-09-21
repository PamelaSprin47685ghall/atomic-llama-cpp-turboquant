#pragma once

#include "ggml.h"

// Subtraction-based range checks do not overflow for a malformed large view.
static constexpr bool ggml_vk_tp5_range_fits(uint64_t capacity, uint64_t base,
                                            uint64_t view_offset, uint64_t bytes) {
    return base <= capacity && view_offset <= capacity - base &&
           bytes <= capacity - base - view_offset;
}

static constexpr uint64_t ggml_vk_tp5_checked_address(uint64_t base, uint64_t offset) {
    return base != 0 && offset <= UINT64_MAX - base ? base + offset : 0;
}

// A transport route may ask a terminal producer not to materialize z_p in
// local VRAM. That request is not itself a proof that z_p has no reader.
// Analyze the concrete graph once, when recording the producer dispatch, and
// bake permission into that dispatch's push constants. LateBind can still
// require the local value through the dynamic route bit.
//
// 'overlaps' must compare actual tensor storage ranges, including aliased views
// and tensors with different identities but the same backing allocation.
// The caller separately validates that target/producer cover the exact same
// contiguous transport range. No allocation or per-replay scan is needed.
template <typename Overlaps>
static bool ggml_vk_tp5_output_can_elide_local(const ggml_tensor * const * nodes,
                                             int n_nodes,
                                             const ggml_tensor * producer,
                                             const ggml_tensor * target,
                                             Overlaps overlaps) {
    if (!nodes || n_nodes <= 0 || !producer || !target || nodes[n_nodes - 1] != target) {
        return false;
    }

    // Follow only the metadata aliases accepted by the producer-wire API.
    // A malformed/cyclic metadata chain must not grant elision permission.
    const ggml_tensor * alias = target;
    int remaining = n_nodes;
    while (alias != producer && remaining-- > 0) {
        if (!alias || (alias->op != GGML_OP_VIEW && alias->op != GGML_OP_RESHAPE)) {
            return false;
        }
        alias = alias->src[0];
    }
    if (alias != producer) {
        return false;
    }

    int producer_index = -1;
    for (int i = 0; i < n_nodes; ++i) {
        if (!nodes[i]) {
            return false;
        }
        if (nodes[i] == producer) {
            if (producer_index != -1) {
                return false;
            }
            producer_index = i;
        }
    }
    if (producer_index < 0) {
        return false;
    }

    for (int i = producer_index + 1; i < n_nodes; ++i) {
        const ggml_tensor * node = nodes[i];
        switch (node->op) {
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                // Metadata is not a GPU memory read. Consumers of these
                // aliases are checked below using physical range overlap.
                continue;
            default:
                break;
        }
        for (const ggml_tensor * input : node->src) {
            if (input && overlaps(input, producer)) {
                return false;
            }
        }
        // A different writer reusing this range means that the identified
        // producer is not the sole last writer of the transport output.
        if (overlaps(node, producer)) {
            return false;
        }
    }
    return true;
}
