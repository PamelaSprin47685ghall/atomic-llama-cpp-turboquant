#pragma once

#include "llama.h"
#include "ggml-predefined.h"

#ifdef __cplusplus
extern "C" {
#endif

// Startup-only TP5/Qwen4EXP MTP maximum-capacity resource preparation. Target
// and draft share one high-water compute pool and one immutable capacity record.
// It reserves graphs for sizing only; it does not execute a model or a shader.
// Existing non-TP5/non-Qwen MTP paths are unchanged (success, no session attached).
//
// This is RESOURCE preparation, not an assertion that every Vulkan operator
// has been lowered to the capacity-frame ABI. Logical shape checks remain in
// force until that lowering is complete. Never turn `ne == rows` into `<=`
// solely because this call succeeded.
LLAMA_API bool llama_predefined_mtp_reserve(
        struct llama_context * target, struct llama_context * draft, uint32_t max_draft_tokens);

// Returns false if no maximum-capacity session is attached. Copies a small
// immutable record only; no GPU calls, graph scan, or allocation.
LLAMA_API bool llama_predefined_get_capacity(
        const struct llama_context * ctx, struct ggml_predefined_capacity * capacity);

#ifdef __cplusplus
}
#endif
