#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"
#include "llama-host-memory.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

struct llama_moe_tensor_info {
    enum ggml_type type;
    size_t expert_size;
    int64_t n_input;
    int64_t n_output;
    int64_t n_expert;
    // Transformer block index parsed from the tensor name ("blk.<N>..."),
    // or -1 when the layer cannot be determined.
    int64_t layer;
};

LLAMA_API size_t llama_model_get_moe_tensor_info(
        const struct llama_model * model,
        struct llama_moe_tensor_info * info,
        size_t capacity);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Host-only KV journal/prepare accounting (C++ extension). This is not part
// of the buffer-type map above and does not change llama_get_memory_breakdown's
// device-buffer meaning.
//
// The values are capacity accounts of the cache's host containers, not a
// resident-memory total. The query is not a thread-safe snapshot and takes no
// locks. The caller must serialize it with every memory mutation: decode and
// encode, set_input, llama_memory_clear, the seq_* family,
// state_read/state_write, update/shift, prepare/replay, and pool
// acquire/release. A const context pointer is not a read-only guarantee: the
// reported containers are mutated in place, and a concurrent mutation can
// reallocate them while the query walks them. llama_synchronize() alone does
// not make the query safe.
//
// A batch context returned by init_batch owns the split ubatches and slot
// infos. On the pooled decode-scratch path they are moved out of the cache
// scratch; on the non-pooled or pool-exhausted path they are local
// allocations. In both cases the cache account cannot see them while the
// context is alive, so scratch is a lower bound. An init_update or init_full
// context does not take the decode scratch payload. Do not query while any
// memory context is in use, and serialize the query with the context's
// creation and destruction.
//
// The scope is the journal/prepare high-water payload capacity. It excludes
// allocator overhead, resident live v_cells/v_heads, recurrent state,
// compressor state, PLE history, rotation matrices, and every host container
// that no journal/prepare owner reports. A memory implementation without this
// account (for example recurrent) contributes zero. Capacities can stay high
// after clear or replay.
LLAMA_API llama_host_memory_breakdown_data llama_get_host_memory_breakdown(const struct llama_context * ctx);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// Marks the entries that a joint decision head (clef) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0, // not read by the head
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1, // text of a question
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4, // text of an option
};
// The embeddings output has one value per entry: row i is the score of option i
// fork: upstream sets the order per entry with llama_batch_ext_set_decision_order(); this fork has no llama_batch_ext yet,
// so the order of all n_tokens tokens of the next llama_encode() / llama_decode() batch is set here (n_tokens == 0 clears it)
LLAMA_API void llama_set_decision_order(struct llama_context * ctx, const int32_t * order, int32_t n_tokens);

LLAMA_API bool llama_model_supports_mtp_chain(const struct llama_model * model);

// True only for architectures whose "draft" is a same-position early-exit of the
// target's own trunk (gemma4-assistant), where every drafted token in a round shares
// llama_pos n_past. A dedicated trained NextN/MTP head (qwen35, qwen4exp, deepseek, ...)
// still needs an incrementing position per draft step even when it shares KV memory
// with the target (no separate -md model given).
LLAMA_API bool llama_model_uses_shared_position_draft(const struct llama_model * model);

// Run the DECODER_MTP graph in chained mode: the batch's first row carries the
// real (token, h) inputs and each following row's inputs come from the previous
// row's in-graph argmax and hidden state. One decode drafts n_tokens tokens.
LLAMA_API void llama_set_mtp_chain(struct llama_context * ctx, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// Same row resolution without the implicit synchronize. The caller must already
// hold a FULL synchronize of this context (a stock syncing getter or
// llama_synchronize); the recorded output-copy events cover only the logits D2H
// copy, not the separate wide NextN copy.
LLAMA_API float * llama_get_embeddings_nextn_ith_no_sync(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);
