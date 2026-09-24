#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

enum llama_predefined_hidden_slot {
    LLAMA_PREDEFINED_H_RESULT = 0,
    LLAMA_PREDEFINED_H_CARRY  = 1,
    LLAMA_PREDEFINED_H_SEED   = 2,
    LLAMA_PREDEFINED_H_INPUT  = 3,
};

struct llama_predefined_hidden_range {
    struct llama_context * source;
    enum llama_predefined_hidden_slot src_slot;
    enum llama_predefined_hidden_slot dst_slot;
    uint32_t src_row;
    uint32_t dst_row;
    uint32_t rows;
    // Required for RESULT sources; zero for CARRY/SEED. Prevents a stale
    // accepted-row plan from reading a newer verification with the same shape.
    uint64_t generation;
};

// Startup-only, single-sequence Qwen4EXP TP5 with matching local devices.
// Retains hidden states on-device, not in the shared temporary compute arena.
// This does not imply that maximum-shape GPU graph lowering is complete.
LLAMA_API bool llama_predefined_hidden_enable(struct llama_context * target, struct llama_context * draft);
LLAMA_API uint32_t llama_predefined_hidden_rows(struct llama_context * ctx);
LLAMA_API uint64_t llama_predefined_hidden_generation(const struct llama_context * ctx);
// Explicit persistent CARRY/SEED updates. H_INPUT is handled by decode_hidden,
// and RESULT may only be written by successful graph output capture.
LLAMA_API bool llama_predefined_hidden_copy(struct llama_context * dst,
        const struct llama_predefined_hidden_range * ranges, size_t n_ranges);

// H_INPUT is virtual: the ranges cover [0,batch.n_tokens) without gaps/overlap
// and are copied directly into the graph input. No packed input arena exists.
// Consumes a token batch and device hidden rows; does not upload batch.embd.
LLAMA_API int llama_predefined_decode_hidden(struct llama_context * ctx, struct llama_batch batch,
        const struct llama_predefined_hidden_range * ranges, size_t n_ranges);

// W4 MTP catch-up variants. Same H_SEED/H_RESULT range contract as
// llama_predefined_decode_hidden / llama_decode; they build the MTP head's
// CATCHUP_KV graph (draft KV writes only) instead of the full draft graph.
// Used exclusively by the deferred catch-up in common/speculative.cpp::commit().
LLAMA_API int llama_predefined_decode_hidden_kv(struct llama_context * ctx, struct llama_batch batch,
        const struct llama_predefined_hidden_range * ranges, size_t n_ranges);
LLAMA_API int llama_decode_mtp_catchup(struct llama_context * ctx, struct llama_batch batch);

// Explicit checkpoint/reset I/O only. Normal drafting/verification never uses
// these host paths. bytes must be exactly one hidden row.
LLAMA_API bool llama_predefined_hidden_carry_get(struct llama_context * ctx, float * data, size_t bytes);
LLAMA_API bool llama_predefined_hidden_carry_set(struct llama_context * ctx, const float * data, size_t bytes);
LLAMA_API bool llama_predefined_hidden_reset(struct llama_context * ctx);

#ifdef __cplusplus
}
#endif
