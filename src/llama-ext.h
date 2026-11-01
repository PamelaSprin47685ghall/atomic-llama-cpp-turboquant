#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

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

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

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

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Row count of a batch accepted by llama_decode_n3_verify_logits().
#define LLAMA_N3_VERIFY_MAX_ROWS 3

// Token-major description of one speculative-verify step for a DEFAULT context of a model
// whose draft head is a NextN/MTP block (n_layer_nextn == 1, e.g. qwen4exp).
//
// The batch is [current, draft_0, draft_1] at positions [p, p+1, p+2] of ONE sequence. Row i
// is executed by the same N3 graph an MTP verify step executes (ubatch shape 3 x 1 sequence),
// so every row is a target row, not an N1 row of the same sequence decoded at another time.
//
// When out_logits != NULL (needs_logits[i] != 0) the call additionally copies that row's FULL
// vocabulary output-head vector (llama_vocab_n_tokens floats) after the graph retired. The
// copies never feed the acceptance decision, so turning them on cannot change the trajectory
// of the run. needs_logits == NULL copies every row.
//
// The caller must have called llama_set_embeddings_nextn(ctx, true, false) first: the target
// graph exports the wide NextN residual per row (llama_get_embeddings_nextn_ith), which is
// what the draft head consumes. A context that does not export it is refused.
//
// State after a failed call - read this before retrying anything.
//
//   The call decodes [p, p + 3) of ONE sequence in one ubatch. If anything fails AFTER the batch
//   was submitted (the decode itself, or a row that has no logits), the call tries to remove
//   exactly [p, p + 3) from the sequence's memory. That removal is a PARTIAL removal of the
//   recurrent (GDN) cache, and a recurrent cache can only perform one when the context has a
//   per-step state plane:
//
//     * llama_memory_recurrent::seq_rm() needs 1 <= (tail.pos - p + 1) <= n_rs_seq. The default
//       context has n_rs_seq == 0 (llama_context_default_params, and include/llama.h documents
//       rollback/replay as unqualified), so at n_rs_seq == 0 the removal ALWAYS FAILS and
//       mutates nothing.
//     * the hybrid memories try the recurrent cache first and return false without touching the
//       attention cache, the indexer cache or the PLE history. The caches then DISAGREE: the
//       GDN state describes p + 2 while the other stores may still describe p.
//
//   Consequently this call does NOT promise that a failed step can be retried from p. It only
//   promises:
//
//     1. positions before p are never touched by the removal;
//     2. when the removal succeeds (only possible with a recurrent state plane, i.e. n_rs_seq > 0
//        or the replay lane), the sequence is back at p and the context stays usable;
//     3. when the removal fails, the context is marked unusable and EVERY later decode on it is
//        refused with -1, both through this function and through llama_decode: the rejection
//        happens BEFORE anything is submitted, so it is a refusal, not a third failure mode.
//        The caller MUST NOT decode on this context again. The only recovery is a full sequence
//        state restore (llama_state_seq_set_data_ext) into a context that has not been used
//        since, or destroying the context.
//
// Return value: 0 on success, negative otherwise.
//   -1  refused, nothing was submitted and no state was written. Two cases produce it: the
//       argument/shape checks before the batch is built, and a context already marked unusable
//       by an earlier failed removal (see 3 above). The reason is logged.
//   -2  the step failed after submission (decode error, wrong output-row count, or a missing
//       row) and the removal of [p, p + 3) SUCCEEDED. Only reachable with a recurrent state
//       plane (n_rs_seq > 0); at n_rs_seq == 0 the removal always fails, so -2 never occurs
//       there. The sequence is back at p, but this call makes NO claim that a later
//       state_seq_get on this context is safe or meaningful - that is not verified.
//   -3  the step failed after submission and the removal failed too: the context is UNUSABLE
//       (see 3 above); later calls return -1 as described there.
// The underlying decode status is logged verbatim; it is not folded into the return code, so a
// caller can never mistake "decode failed, recovered" for "refused, nothing happened".
LLAMA_API int llama_decode_n3_verify_logits(
        struct llama_context * ctx,
        llama_seq_id           seq_id,
        llama_pos              p,
        const llama_token      tokens[LLAMA_N3_VERIFY_MAX_ROWS],
        const int32_t          needs_logits[LLAMA_N3_VERIFY_MAX_ROWS],
        float * const          out_logits[LLAMA_N3_VERIFY_MAX_ROWS]);

// Row count of a batch accepted by llama_decode_n3_verify_logits_pair().
#define LLAMA_N3_VERIFY_PAIR_MAX_ROWS 3

// Same verify step as llama_decode_n3_verify_logits(), for the QUALIFIED two-model layout:
// a trunk context over a model with n_layer_nextn == 0, whose draft head is a SEPARATE model
// loaded into its own LLAMA_CONTEXT_TYPE_MTP context (the layout fork/tp5-compose uses:
// runtime.cpp loads the trunk and the head as two models and reads the head's input residual
// from the trunk context).
//
// Differences from the single-model entry point, and only these:
//   * the model is accepted when it has NO NextN block, provided the trunk graph exports the
//     wide NextN residual for the output rows (cparams.embeddings_nextn, unmasked). That export
//     is the head's input; without it the rows are not the input of any draft head and the call
//     is refused.
//   * the context's recurrent rollback capability is REPORTED, never assumed: with
//     n_rs_seq >= 1 a partial removal of the speculative rows can succeed; with n_rs_seq == 0 it
//     cannot, and a failed step leaves the context unusable exactly as in the single-model case.
//     The two models are never merged, aliased, or loaded as one; the head model is not touched
//     by this call.
//   * the vocabulary of the trunk model is the vocabulary of the captured rows; the caller is
//     responsible for the head model sharing it (a row compared across vocabularies is
//     meaningless and the harness refuses that pairing).
//
// Everything else - the three-row contract, the acceptance of the returned rows, the ordering
// of the output rows, the full-vocabulary copies after the graph retired, and the return codes -
// is identical to llama_decode_n3_verify_logits() and is documented there.
LLAMA_API int llama_decode_n3_verify_logits_pair(
        struct llama_context * ctx,
        llama_seq_id           seq_id,
        llama_pos              p,
        const llama_token      tokens[LLAMA_N3_VERIFY_PAIR_MAX_ROWS],
        const int32_t          needs_logits[LLAMA_N3_VERIFY_PAIR_MAX_ROWS],
        float * const          out_logits[LLAMA_N3_VERIFY_PAIR_MAX_ROWS]);

// The recurrent rollback capability of a context, as the loader/context reports it:
// 0 means a partial removal of the recurrent cache is impossible (a failed step cannot be
// retried), >= 1 means up to that many steps can be removed. Never inferred by the caller.
LLAMA_API uint32_t llama_n3_verify_rollback_steps(struct llama_context * ctx);

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
