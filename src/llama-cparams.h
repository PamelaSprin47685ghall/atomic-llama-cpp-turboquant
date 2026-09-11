#pragma once

#include "llama.h"

// Public FlashPrefill policy contract (PolicyCore owner). Resolves via the
// include path to include/llama-flashprefill.h, or via the src-internal
// forwarding header when built from src/. Never includes llama.h (no cycle).
#include "llama-flashprefill.h"

#include <cstdint>
#include <memory>
#include <vector>

#ifndef LLAMA_MAX_SEQ
#define LLAMA_MAX_SEQ 256
#endif

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_ctx_kv;        // number of cells in the attention KV cache
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_seq_max_pp;
    uint32_t n_seq_recurrent; // physical recurrent-state slots
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool kv_size_explicit;
    bool triattention_enabled = false;
    double triattention_ratio = 3.0 / 32.0;
    uint32_t triattention_recent_window = 128;
    bool rerot_enabled = false;
    llama_rerot_frontier_mode rerot_frontier = LLAMA_REROT_FRONTIER_STRONG;
    uint32_t n_person_max = 0; // auto-selected B (people / independent brains, B.3.2)
    uint32_t n_pen_max    = 0; // auto-selected P (pens / execution states, B.3.2)
    uint32_t n_brain_rows = 0; // derived from B
    uint32_t n_hand_rows  = 0; // derived from P
    // FlashPrefill V2 policy (PolicyCore: struct llama_flashprefill_config).
    // Immutable for the context lifetime; validated on construction; default
    // OFF (no scratch, no new graph nodes, no routing work when disabled).
    // Copied verbatim into llm_graph_params, so the graph slice always sees
    // the same policy without extra plumbing. No per-row arrays here: the
    // per-call/per-ubatch row snapshot lives owned by llama_context.
    struct llama_flashprefill_config flashprefill;
    // FlashPrefill per-ubatch owned row snapshot for the graph slice.
    // Shared ownership: llm_graph_params copies cparams by value, so every
    // graph copy keeps its own snapshot alive across async execution — never
    // a borrowed pointer, no dangling arrays. Null means "route dense"
    // (OFF, bypassed, legacy decode, or no source map for the ubatch).
    // Populated enabled-only: set per ubatch in the decode loop from the
    // context-owned snapshot, and to a validated synthetic eligible snapshot
    // in graph_reserve (sizing needs eligible shape, not dense). Consumers
    // read gparams.cparams.flashprefill_rows; GraphIntegration owns any
    // further graph.h-side interpretation.
    std::shared_ptr<const std::vector<struct llama_flashprefill_row>> flashprefill_rows;
    bool pipeline_parallel;

    // XKV (§16) parameters
    enum llama_xkv_mode            xkv_mode                 = LLAMA_XKV_MODE_OFF;
    enum llama_xkv_storage_profile xkv_storage_profile      = LLAMA_XKV_STORAGE_PROFILE_REFERENCE;
    uint32_t                       xkv_group_size           = 4;
    uint32_t                       xkv_rank_k               = 384;
    uint32_t                       xkv_rank_v               = 576;
    uint32_t                       xkv_segment_tokens       = 4096;
    uint32_t                       xkv_chunk_tokens         = 8;
    uint32_t                       xkv_sr_budget            = 0;
    enum llama_xkv_source          xkv_source               = LLAMA_XKV_SOURCE_DECODED_HOT;
    enum ggml_type                 xkv_factor_a_k           = GGML_TYPE_TURBO4_0;
    enum ggml_type                 xkv_factor_b_k           = GGML_TYPE_TURBO4_0;
    enum ggml_type                 xkv_factor_a_v           = GGML_TYPE_TURBO4_0;
    enum ggml_type                 xkv_factor_b_v           = GGML_TYPE_TURBO4_0;
    enum llama_xkv_factor_balance  xkv_factor_balance       = LLAMA_XKV_FACTOR_BALANCE_UPSTREAM;
    enum ggml_type                 xkv_landmark_type        = GGML_TYPE_Q8_0;
    enum llama_xkv_landmark_refine xkv_landmark_refine      = LLAMA_XKV_LANDMARK_REFINE_NONE;
    uint32_t                       xkv_landmark_refine_max_rows = 64;
    uint32_t                       xkv_workspace_mib        = 256;
    uint32_t                       xkv_decode_cache_mib     = 64;
    uint32_t                       xkv_store_mib            = 0;
    uint64_t                       xkv_seed                 = 6362273814452121649ULL;
    double                         xkv_min_saving           = 0.10;
    double                         xkv_min_factor_coverage  = 0.50;
    enum llama_xkv_factorizer      xkv_factorizer           = LLAMA_XKV_FACTORIZER_CPU_REFERENCE;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer
    std::vector<bool> attention_q_pre_rope; // [n_layer()] extract normalized Q before RoPE

    enum llama_context_type ctx_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;
};
