#ifndef LLAMA_H
#define LLAMA_H

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-opt.h"
#include "gguf.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define LLAMA_API __declspec(dllexport)
#        else
#            define LLAMA_API __declspec(dllimport)
#        endif
#    else
#        define LLAMA_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LLAMA_API
#endif

#ifdef __GNUC__
#    define DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define DEPRECATED(func, hint) func
#endif

#define LLAMA_DEFAULT_SEED 0xFFFFFFFF

#define LLAMA_TOKEN_NULL -1

#define LLAMA_FILE_MAGIC_GGLA 0x67676c61u // 'ggla'
#define LLAMA_FILE_MAGIC_GGSN 0x6767736eu // 'ggsn'
#define LLAMA_FILE_MAGIC_GGSQ 0x67677371u // 'ggsq'

#define LLAMA_SESSION_MAGIC   LLAMA_FILE_MAGIC_GGSN
#define LLAMA_SESSION_VERSION 10

#define LLAMA_STATE_SEQ_MAGIC   LLAMA_FILE_MAGIC_GGSQ
#define LLAMA_STATE_SEQ_VERSION 3

#define LLAMA_MEMORY_HYBRID_IDX_TRACE_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct llama_vocab;
    struct llama_model;
    struct llama_context;
    struct llama_sampler;

    typedef struct llama_memory_i * llama_memory_t;

    typedef int32_t llama_pos;
    typedef int32_t llama_token;
    typedef int32_t llama_seq_id;

    enum llama_vocab_type {
        LLAMA_VOCAB_TYPE_NONE   = 0, // For models without vocab
        LLAMA_VOCAB_TYPE_SPM    = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
        LLAMA_VOCAB_TYPE_BPE    = 2, // GPT-2 tokenizer based on byte-level BPE
        LLAMA_VOCAB_TYPE_WPM    = 3, // BERT tokenizer based on WordPiece
        LLAMA_VOCAB_TYPE_UGM    = 4, // T5 tokenizer based on Unigram
        LLAMA_VOCAB_TYPE_RWKV   = 5, // RWKV tokenizer based on greedy tokenization
        LLAMA_VOCAB_TYPE_PLAMO2 = 6, // PLaMo-2 tokenizer based on Aho-Corasick with dynamic programming
    };

    enum llama_rope_type {
        LLAMA_ROPE_TYPE_NONE   = -1,
        LLAMA_ROPE_TYPE_NORM   = 0,
        LLAMA_ROPE_TYPE_NEOX   = GGML_ROPE_TYPE_NEOX,
        LLAMA_ROPE_TYPE_MROPE  = GGML_ROPE_TYPE_MROPE,
        LLAMA_ROPE_TYPE_IMROPE = GGML_ROPE_TYPE_IMROPE,
        LLAMA_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION,
    };

    enum llama_token_type { //TODO: remove, required until per token attributes are available from GGUF file
        LLAMA_TOKEN_TYPE_UNDEFINED    = 0,
        LLAMA_TOKEN_TYPE_NORMAL       = 1,
        LLAMA_TOKEN_TYPE_UNKNOWN      = 2,
        LLAMA_TOKEN_TYPE_CONTROL      = 3,
        LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
        LLAMA_TOKEN_TYPE_UNUSED       = 5,
        LLAMA_TOKEN_TYPE_BYTE         = 6,
    };

    enum llama_token_attr {
        LLAMA_TOKEN_ATTR_UNDEFINED    = 0,
        LLAMA_TOKEN_ATTR_UNKNOWN      = 1 << 0,
        LLAMA_TOKEN_ATTR_UNUSED       = 1 << 1,
        LLAMA_TOKEN_ATTR_NORMAL       = 1 << 2,
        LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3,  // SPECIAL?
        LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4,
        LLAMA_TOKEN_ATTR_BYTE         = 1 << 5,
        LLAMA_TOKEN_ATTR_NORMALIZED   = 1 << 6,
        LLAMA_TOKEN_ATTR_LSTRIP       = 1 << 7,
        LLAMA_TOKEN_ATTR_RSTRIP       = 1 << 8,
        LLAMA_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
    };

    // model file types
    enum llama_ftype {
        LLAMA_FTYPE_ALL_F32              = 0,
        LLAMA_FTYPE_MOSTLY_F16           = 1,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_0          = 2,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_1          = 3,  // except 1d tensors
        // LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
        // LLAMA_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
        // LLAMA_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
        LLAMA_FTYPE_MOSTLY_Q8_0          = 7,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_0          = 8,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_1          = 9,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q2_K          = 10, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q6_K          = 18, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_XXS       = 19, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_XS        = 20, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q2_K_S        = 21, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_XS        = 22, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_XXS       = 23, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_S         = 24, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_NL        = 25, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_S         = 26, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_M         = 27, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_S         = 28, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_M         = 29, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_XS        = 30, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_M         = 31, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_BF16          = 32, // except 1d tensors
        //LLAMA_FTYPE_MOSTLY_Q4_0_4_4      = 33, // removed from gguf files, use Q4_0 and runtime repack
        //LLAMA_FTYPE_MOSTLY_Q4_0_4_8      = 34, // removed from gguf files, use Q4_0 and runtime repack
        //LLAMA_FTYPE_MOSTLY_Q4_0_8_8      = 35, // removed from gguf files, use Q4_0 and runtime repack
        LLAMA_FTYPE_MOSTLY_TQ1_0         = 36, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_TQ2_0         = 37, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_MXFP4_MOE     = 38, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_NVFP4         = 39, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q1_0          = 40, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q2_0          = 41, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q8_CR         = 42, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_TQ3_1S        = 43, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_TQ4_1S        = 44, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_CR         = 45, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q6_CR         = 46, // except 1d tensors

        LLAMA_FTYPE_GUESSED = 1024, // not specified in the model file
    };

    // Get the model file type (quantization) as a string, e.g. "Q8_0" or "Q4_K - Medium"
    LLAMA_API const char * llama_ftype_name(enum llama_ftype ftype);

    enum llama_rope_scaling_type {
        LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
        LLAMA_ROPE_SCALING_TYPE_NONE        = 0,
        LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1,
        LLAMA_ROPE_SCALING_TYPE_YARN        = 2,
        LLAMA_ROPE_SCALING_TYPE_LONGROPE    = 3,
        LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = LLAMA_ROPE_SCALING_TYPE_LONGROPE,
    };

    enum llama_pooling_type {
        LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
        LLAMA_POOLING_TYPE_NONE = 0,
        LLAMA_POOLING_TYPE_MEAN = 1,
        LLAMA_POOLING_TYPE_CLS  = 2,
        LLAMA_POOLING_TYPE_LAST = 3,
        LLAMA_POOLING_TYPE_RANK = 4, // used by reranking models to attach the classification head to the graph
    };

    enum llama_attention_type {
        LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1,
        LLAMA_ATTENTION_TYPE_CAUSAL      = 0,
        LLAMA_ATTENTION_TYPE_NON_CAUSAL  = 1,
    };

    enum llama_flash_attn_type {
        LLAMA_FLASH_ATTN_TYPE_AUTO     = -1,
        LLAMA_FLASH_ATTN_TYPE_DISABLED = 0,
        LLAMA_FLASH_ATTN_TYPE_ENABLED  = 1,
    };

    LLAMA_API const char * llama_flash_attn_type_name(enum llama_flash_attn_type flash_attn_type);

    enum llama_split_mode {
        LLAMA_SPLIT_MODE_NONE   = 0, // single GPU
        LLAMA_SPLIT_MODE_LAYER  = 1, // split layers and KV across GPUs
        LLAMA_SPLIT_MODE_ROW    = 2, // split layers and KV across GPUs, use tensor parallelism if supported
        LLAMA_SPLIT_MODE_TENSOR = 3,
    };

    enum llama_load_mode {
        LLAMA_LOAD_MODE_AUTO       = -1, // auto-detect based on device capabilities
        LLAMA_LOAD_MODE_NONE       =  0, // no special loading mode
        LLAMA_LOAD_MODE_MMAP       = 1, // memory map the model
        LLAMA_LOAD_MODE_MLOCK      = 2, // force system to keep model in RAM rather than swapping or compressing
        LLAMA_LOAD_MODE_MMAP_MLOCK = 3, // mmap + force system to keep model in RAM rather than swapping or compressing
        LLAMA_LOAD_MODE_DIRECT_IO  = 4, // use direct I/O if available
    };

    LLAMA_API const char * llama_load_mode_name(enum llama_load_mode load_mode);
    LLAMA_API enum llama_load_mode llama_load_mode_from_str(const char * str);

    enum llama_tensor_read_lazy {
        LLAMA_TENSOR_READ_LAZY_OFF  = 0, // always read the whole tensor up front
        LLAMA_TENSOR_READ_LAZY_AUTO = 1, // lazy only for marked tensors larger than 4 GiB (requires mmap)
        LLAMA_TENSOR_READ_LAZY_ON   = 2, // read the rows of tensors marked by the arch on demand (requires mmap)
    };

    enum llama_context_type {
        LLAMA_CONTEXT_TYPE_DEFAULT = 0,
        LLAMA_CONTEXT_TYPE_MTP     = 1,
    };

    enum llama_moe_cache_mode {
        LLAMA_MOE_CACHE_MODE_UNSPECIFIED = -1,
        LLAMA_MOE_CACHE_MODE_OFF = 0,
        LLAMA_MOE_CACHE_MODE_AUTO = 1,
        LLAMA_MOE_CACHE_MODE_ON = 2,
    };

    // TODO: simplify (https://github.com/ggml-org/llama.cpp/pull/9294#pullrequestreview-2286561979)
    typedef struct llama_token_data {
        llama_token id; // token id
        float logit;    // log-odds of the token
        float p;        // probability of the token
    } llama_token_data;

    typedef struct llama_token_data_array {
        // TODO: consider SoA
        // NOTE: this pointer can be modified by the samplers
        llama_token_data * data;
        size_t size;
        int64_t selected; // this is the index in the data array (i.e. not the token id)
        bool sorted;      // note: do not assume the data is sorted - always check this flag
    } llama_token_data_array;

    typedef bool (*llama_progress_callback)(float progress, void * user_data);

    // Input data for llama_encode/llama_decode
    // A llama_batch object can contain input about one or many sequences
    // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    //
    // - token  : the token ids of the input (used when embd is NULL)
    // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    // - pos    : the positions of the respective token in the sequence
    //            (if set to NULL, the token position will be tracked automatically by llama_encode/llama_decode)
    // - seq_id : the sequence to which the respective token belongs
    //            (if set to NULL, the sequence ID will be assumed to be 0)
    // - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
    //            (if set to NULL:
    //               - if embeddings: all tokens are output
    //               - if not:        only the last token is output
    //            )
    //
    typedef struct llama_batch {
        int32_t n_tokens;

        llama_token  *  token;
        float        *  embd;
        llama_pos    *  pos;
        int32_t      *  n_seq_id;
        llama_seq_id ** seq_id;
        int8_t       *  logits;   // TODO: rename this to "output"
    } llama_batch;

    enum llama_model_kv_override_type {
        LLAMA_KV_OVERRIDE_TYPE_INT,
        LLAMA_KV_OVERRIDE_TYPE_FLOAT,
        LLAMA_KV_OVERRIDE_TYPE_BOOL,
        LLAMA_KV_OVERRIDE_TYPE_STR,
    };

    enum llama_model_meta_key {
        LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE,
        LLAMA_MODEL_META_KEY_SAMPLING_TOP_K,
        LLAMA_MODEL_META_KEY_SAMPLING_TOP_P,
        LLAMA_MODEL_META_KEY_SAMPLING_MIN_P,
        LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY,
        LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD,
        LLAMA_MODEL_META_KEY_SAMPLING_TEMP,
        LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N,
        LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT,
        LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT,
        LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU,
        LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA,
    };

    struct llama_model_kv_override {
        enum llama_model_kv_override_type tag;

        char key[128];

        union {
            int64_t val_i64;
            double  val_f64;
            bool    val_bool;
            char    val_str[128];
        };
    };

    struct llama_model_tensor_buft_override {
        const char * pattern;
        ggml_backend_buffer_type_t buft;
    };

    struct llama_model_params {
        // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
        ggml_backend_dev_t * devices;

        // NULL-terminated list of buffer types to use for tensors that match a pattern
        const struct llama_model_tensor_buft_override * tensor_buft_overrides;

        int32_t n_gpu_layers; // number of layers to store in VRAM, a negative value means all layers
        enum llama_split_mode split_mode; // how to split the model across multiple GPUs
        enum llama_load_mode  load_mode;  // how to load the model

        enum llama_tensor_read_lazy tensor_read_lazy; // on-demand reading of tensors marked by the arch

        // the GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE
        int32_t main_gpu;

        // proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
        const float * tensor_split;

        // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
        // If the provided progress_callback returns true, model loading continues.
        // If it returns false, model loading is immediately aborted.
        llama_progress_callback progress_callback;

        // context pointer passed to the progress callback
        void * progress_callback_user_data;

        // override key-value pairs of the model meta data
        const struct llama_model_kv_override * kv_overrides;

        // Keep the booleans together to avoid misalignment during copy-by-value.
        bool vocab_only;      // only load the vocabulary, no weights
        bool check_tensors;   // validate model tensor data
        bool use_extra_bufts; // use extra buffer types (used for weight repacking)
        bool no_host;         // bypass host buffer allowing extra buffers to be used
        bool no_alloc;        // only load metadata and simulate memory allocations
        bool load_mtp;        // whether to load MTP layers
        // Place input-layer weights on the first offloaded device, without a
        // CPU buffer fallback. Requires real GPU layers and allocated weights.
        // Tensor Meta placement keeps one read-only embedding copy per rank.
        bool offload_input;
    };

    struct llama_sampler_seq_config {
        llama_seq_id           seq_id;
        struct llama_sampler * sampler;
    };

    // NOTE: changing the default values of parameters marked as [EXPERIMENTAL] may cause crashes or incorrect results in certain configurations
    //       https://github.com/ggml-org/llama.cpp/pull/7544
    struct llama_context_params {
        uint32_t n_ctx;             // text context, 0 = from model
        uint32_t n_batch;           // logical maximum batch size that can be submitted to llama_decode
        uint32_t n_ubatch;          // physical maximum batch size
        uint32_t n_seq_max;         // max number of sequences (i.e. distinct states for recurrent models)
        uint32_t n_rs_seq;          // number of recurrent-state snapshots per seq for rollback (0 = no rollback) [EXPERIMENTAL]
        bool     gdn_replay;        // ingredient-replay rollback instead of full K-snapshots when n_rs_seq > 0 [EXPERIMENTAL]
        uint32_t n_outputs_max;     // max outputs in a ubatch (0 = n_batch)
        int32_t  n_threads;         // number of threads to use for generation
        int32_t  n_threads_batch;   // number of threads to use for batch processing

        enum llama_context_type      ctx_type;          // set the context type (e.g. MTP)
        enum llama_rope_scaling_type rope_scaling_type; // RoPE scaling type, from `enum llama_rope_scaling_type`
        enum llama_pooling_type      pooling_type;      // whether to pool (sum) embedding results by sequence id
        enum llama_attention_type    attention_type;    // attention type to use for embeddings
        enum llama_flash_attn_type   flash_attn_type;   // when to enable Flash Attention

        // ref: https://github.com/ggml-org/llama.cpp/pull/2054
        float    rope_freq_base;   // RoPE base frequency, 0 = from model
        float    rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
        float    yarn_ext_factor;  // YaRN extrapolation mix factor, negative = from model
        float    yarn_attn_factor; // YaRN magnitude scaling factor
        float    yarn_beta_fast;   // YaRN low correction dim
        float    yarn_beta_slow;   // YaRN high correction dim
        uint32_t yarn_orig_ctx;    // YaRN original context size
        float    defrag_thold;     // [DEPRECATED] defragment the KV cache if holes/size > thold, <= 0 disabled (default)

        ggml_backend_sched_eval_callback cb_eval;
        void * cb_eval_user_data;

        enum ggml_type type_k; // data type for K cache [EXPERIMENTAL]
        enum ggml_type type_v; // data type for V cache [EXPERIMENTAL]

        enum llama_moe_cache_mode moe_cache_mode; // runtime MoE expert cache mode
        size_t moe_cache_budget_mib;               // 0 uses the provider's available-memory budget

        // Abort callback
        // if it returns true, execution of llama_decode() will be aborted
        // currently works only with CPU execution
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;

        // Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
        bool embeddings;  // if true, extract embeddings (together with logits)
        bool offload_kqv; // offload the KQV ops (including the KV cache) to GPU
        bool no_perf;     // measure performance timings
        bool op_offload;  // offload host tensor operations to device
        bool swa_full;    // use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
                          // NOTE: setting to false when n_seq_max > 1 can cause bad performance in some cases
                          //       ref: https://github.com/ggml-org/llama.cpp/pull/13845#issuecomment-2924800573
        bool kv_unified;  // use a unified buffer across the input sequences when computing the attention
                          // try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix
                          // ref: https://github.com/ggml-org/llama.cpp/pull/14363

        // [EXPERIMENTAL]
        // backend sampler chain configuration (make sure the caller keeps the sampler chains alive)
        // note: the samplers must be sampler chains (i.e. use llama_sampler_chain_init)
        struct llama_sampler_seq_config * samplers;
        size_t                            n_samplers;

        // a source/target/parent context
        // can be utilized in various ways, for example by sharing results or llama_memory between 2 contexts
        struct llama_context * ctx_other;

        // [EXPERIMENTAL] block-granular KV cache streaming: total shared CUDA
        // arena (compute workspace + resident KV pages + transfer ring), in
        // MiB. 0 disables streaming.
        // Appended here (not with the other [EXPERIMENTAL] fields above) so
        // a caller built against a pre-streaming header keeps the same
        // offsets for every field before it.
        uint32_t kv_stream_arena_mib;

        // Physical tokens in the unified KV pool. 0 keeps upstream sizing.
        // The logical per-sequence window remains n_ctx. Rebuild callers when
        // changing this versioned struct; a stable binary ABI is not promised.
        uint32_t n_ctx_kv;

        // Physical recurrent states. 0 uses n_seq_max. Logical sequence IDs
        // keep their n_seq_max range. A new live state fails if this pool is full.
        uint32_t n_rs_slots;

        // Request native single-query arithmetic for direct and expert-indexed
        // quantized matrix projections across query batch sizes. CPU/Vulkan
        // qualification is limited to tested profiles. Vulkan requires contiguous
        // quantized weights and F32 activations; its native Q8 activation policy
        // still applies. Expert-indexed profiles allow shared or per-slot input
        // rows and padded I32 ID rows, with descriptor-aligned storage.
        // While enabled, this policy takes precedence over a projection builder's
        // requested accumulator precision. This does not request true-F32
        // arithmetic or qualify every model operation/backend/layout.
        // Set before creating the context, including any prefix prefill.
        // false keeps the optimized native quantized matrix dispatch.
        bool batch_invariant_matmul;

        // Decode Q8_0 weights using their stored scales and multiply by F32
        // activations without activation requantization, with F32 multiply-adds
        // and partial sums (and F32 CPU reduction for tp5_relay).
        // Qualified only for dense Qwen3-8B Q8_0 with all layers, output and
        // Q8_0 input weights on one HIP GPU, or the five-HIP-device tp5_relay
        // profile below. Requires 1..4 sequences/rows and GPU operation/KQV
        // offload; load the model with offload_input=true. Context
        // initialization rejects CPU, mixed offload and unsupported profiles;
        // quantized projections never silently fall back to CPU execution.
        // For this RDNA2 profile, Flash Attention requested with GGML_PREC_F32
        // keeps F32 cache views, probabilities and V-weighted accumulation in
        // F32 through the fused online-softmax attention implementation.
        // Separate from batch_invariant_matmul, which must remain false.
        // Set before creating the context. false preserves native dispatch.
        bool quantized_matmul_f32;

        // Strict Qwen3-8B Q8_0 tensor-parallel CPU relay on five HIP RX6800
        // devices. Unsupported models, shapes or providers fail context init.
        // Requires quantized_matmul_f32=true.
        // The caller drives ggml-relay.h events from an independent worker,
        // uses one ubatch per decode, and serializes all access per context.
        // Supports 1..4 independent sequence IDs and F16/F32 KV with native
        // ordinary or flash attention; no host-routed compute is admitted.
        // Two contexts may share one read-only model; no adapters are admitted.
        bool tp5_relay;
    };

    struct llama_model_tensor_override {
        const char * pattern;
        enum ggml_type type;
    };

    struct llama_model_imatrix_data {
        const char * name;
        const float * data;
        size_t size;
    };

    // model quantization parameters
    typedef struct llama_model_quantize_params {
        int32_t nthread;                                            // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        enum llama_ftype ftype;                                     // quantize to this llama_ftype
        enum ggml_type output_tensor_type;                          // output tensor type
        enum ggml_type token_embedding_type;                        // token embeddings tensor type
        bool allow_requantize;                                      // allow quantizing non-f32/f16 tensors
        bool quantize_output_tensor;                                // quantize output.weight
        bool only_copy;                                             // only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        bool pure;                                                  // quantize all tensors to the default type
        bool keep_split;                                            // quantize to the same number of shards
        bool dry_run;                                               // calculate and show the final quantization size without performing quantization
        const struct llama_model_imatrix_data * imatrix;            // pointer to importance matrix data
        const struct llama_model_kv_override * kv_overrides;        // pointer to kv overrides
        const struct llama_model_tensor_override * tt_overrides;    // pointer to tensor overrides
        const int32_t * prune_layers;                               // pointer to layer indices to prune
    } llama_model_quantize_params;

    typedef struct llama_logit_bias {
        llama_token token;
        float bias;
    } llama_logit_bias;

    typedef struct llama_sampler_chain_params {
        bool no_perf; // whether to measure performance timings
    } llama_sampler_chain_params;

    // used in chat template
    typedef struct llama_chat_message {
        const char * role;
        const char * content;
    } llama_chat_message;

    // lora adapter
    struct llama_adapter_lora;

    // Helpers for getting default parameters
    // TODO: update API to start accepting pointers to params structs (https://github.com/ggml-org/llama.cpp/discussions/9172)
    LLAMA_API struct llama_model_params          llama_model_default_params(void);
    LLAMA_API struct llama_context_params        llama_context_default_params(void);
    LLAMA_API struct llama_sampler_chain_params  llama_sampler_chain_default_params(void);
    LLAMA_API struct llama_model_quantize_params llama_model_quantize_default_params(void);

    // Initialize the llama + ggml backend
    // If numa is true, use NUMA optimizations
    // Call once at the start of the program
    LLAMA_API void llama_backend_init(void);

    // Call once at the end of the program - currently only used for MPI
    LLAMA_API void llama_backend_free(void);

    //optional:
    LLAMA_API void llama_numa_init(enum ggml_numa_strategy numa);

    // Optional: an auto threadpool gets created in ggml if not passed explicitly
    LLAMA_API void llama_attach_threadpool(
            struct llama_context * ctx,
               ggml_threadpool_t   threadpool,
               ggml_threadpool_t   threadpool_batch);

    LLAMA_API void llama_detach_threadpool(struct llama_context * ctx);

    typedef void (*llama_model_set_tensor_data_t)(struct ggml_tensor * tensor, void * userdata);

    // Create a new model from GGUF metadata as well as a function to set the tensor data
    //   - tensors are created as GGML_TYPE_F32 by default,
    //     override by adding a tensor with the same name but a different name to the context
    LLAMA_API struct llama_model * llama_model_init_from_user(
                    struct gguf_context * metadata,
          llama_model_set_tensor_data_t   set_tensor_data,    // function to initialize tensor data with
                                   void * set_tensor_data_ud, // userdata for function
              struct llama_model_params   params);

    DEPRECATED(LLAMA_API struct llama_model * llama_load_model_from_file(
                             const char * path_model,
              struct llama_model_params   params),
            "use llama_model_load_from_file instead");

    // Load a model from a file
    // If the file is split into multiple parts, the file name must follow this pattern: <name>-%05d-of-%05d.gguf
    // If the split file name does not follow this pattern, use llama_model_load_from_splits
    LLAMA_API struct llama_model * llama_model_load_from_file(
                             const char * path_model,
              struct llama_model_params   params);

    // Load a model from an open FILE pointer
    LLAMA_API struct llama_model * llama_model_load_from_file_ptr(
                                   FILE * file,
              struct llama_model_params   params);

    // Load a model from multiple splits (support custom naming scheme)
    // The paths must be in the correct order
    LLAMA_API struct llama_model * llama_model_load_from_splits(
                             const char ** paths,
                                 size_t    n_paths,
              struct llama_model_params    params);

    LLAMA_API void llama_model_save_to_file(
            const struct llama_model * model,
                        const char * path_model);

    DEPRECATED(LLAMA_API void llama_free_model(struct llama_model * model),
            "use llama_model_free instead");

    // Frees the model. Deletion is deferred while any clean or retained
    // context still references it; the last clean llama_free performs it. A
    // second free request is ignored. Creating a context after a free request
    // fails, and the caller must serialize the final free with context
    // creation. A retained pending context keeps the model process-owned.
    LLAMA_API void llama_model_free(struct llama_model * model);

    LLAMA_API struct llama_context * llama_init_from_model(
                     struct llama_model * model,
            struct llama_context_params   params);

    DEPRECATED(LLAMA_API struct llama_context * llama_new_context_with_model(
                     struct llama_model * model,
            struct llama_context_params   params),
            "use llama_init_from_model instead");

    // Frees all allocated memory. Strict tp5-relay contexts must first prove
    // native completion using ggml-relay.h and drain result/event readers.
    // Unproved native completion retains the context rather than freeing live
    // device owners; the shared model must also remain alive in that case.
    // A retained context is process-owned: a duplicate llama_free is ignored,
    // and the caller must not use the context again. Caller-owned samplers may
    // be freed after that return. A LoRA adapter released after retention is
    // erased from the model but kept process-owned by llama_adapter_lora_free;
    // a duplicate adapter free is ignored. Cvec data is copied, not borrowed.
    LLAMA_API void llama_free(struct llama_context * ctx);

    // Borrow the one strict relay Meta backend, or NULL outside relay mode.
    // Its lifetime is the context's. Only the thread-safe ggml-relay.h event
    // API may run concurrently with llama_decode/llama_synchronize.
    LLAMA_API struct ggml_backend * llama_tp5_relay_backend(struct llama_context * ctx);

    // DUALWIN-COMPOSE-EVENT (GGML_TP5_DW_COMPOSE_EVENT=1): tp5-relay only.
    // Record one device event per child stream that the asynchronous logits D2H
    // copy of the last decoded epoch used (one stream for a mirrored output,
    // every contributing rank for a split output). The record is ordered behind
    // that copy, so the paired wait proves the host logits row is readable
    // without a scheduler-wide synchronize. Returns the number of streams
    // covered; 0 means no sound barrier exists (non-relay context, no graph,
    // non-Meta/partial/host-resident output, missing event interface, gate off)
    // and the caller must keep llama_synchronize. No event is left allocated
    // on 0.
    LLAMA_API size_t llama_tp5_output_copy_record(struct llama_context * ctx);

    // Host-complete and release the events recorded by
    // llama_tp5_output_copy_record, then run the context's synchronize
    // bookkeeping — only when at least one recorded event was actually
    // synchronized, so an epoch with nothing to prove never reports its queued
    // tokens as synchronized. Returns the number of streams proven complete.
    // Never synchronizes the scheduler's backends.
    LLAMA_API size_t llama_tp5_output_event_wait(struct llama_context * ctx);

    // Resolve one output row of the FULL logits buffer without synchronizing
    // the context. The caller must already hold this epoch's output-copy
    // completion proof (see llama_tp5_output_event_wait, or
    // llama_synchronize); that proof covers exactly the full-logits copy
    // streams recorded by llama_tp5_output_copy_record. Device-sampled
    // outputs resolve on streams no recorded event orders, so callers that
    // need them must keep the synchronizing getters.
    LLAMA_API float * llama_get_logits_ith_no_sync(struct llama_context * ctx, int32_t i);

    LLAMA_API int64_t llama_time_us(void);

    LLAMA_API size_t llama_max_devices(void);
    LLAMA_API size_t llama_max_parallel_sequences(void);
    LLAMA_API size_t llama_max_tensor_buft_overrides(void);

    LLAMA_API bool llama_supports_mmap       (void);
    LLAMA_API bool llama_supports_mlock      (void);
    LLAMA_API bool llama_supports_gpu_offload(void);
    LLAMA_API bool llama_supports_rpc        (void);

    // NOTE: After creating a llama_context, it is recommended to query the actual values using these functions
    //       In some cases the requested values via llama_context_params may differ from the actual values used by the context
    //       ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    LLAMA_API uint32_t llama_n_ctx      (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_ctx_seq  (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_ctx_kv   (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_batch    (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_ubatch   (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_seq_max  (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_rs_seq   (const struct llama_context * ctx);

    // Diagnostics for the active CPU token graph. Plan bytes are allocator
    // requests for token/KV/output-dependent roots; arena bytes are reusable
    // backing capacity and may be larger. Returns false when unavailable.
    struct llama_ntok_plan_stats {
        uint64_t graph_builds;
        uint64_t planned_bytes;
        uint64_t active_bytes;
        uint64_t arena_bytes;
        uint64_t token_input_bytes;
        uint64_t kv_mask_bytes;
        uint64_t output_ids_bytes;
        uint64_t checked_tensors;
        uint64_t mismatched_tensors;
        uint32_t active_tokens;
        uint32_t active_kv;
        uint32_t active_outputs;
    };
    LLAMA_API bool llama_get_ntok_plan_stats(const struct llama_context * ctx, struct llama_ntok_plan_stats * stats);

    // LLAMA_UNION_KERNEL_SEL_LEN_INDEP: the CPU kernel-selection decoupling the
    // exact n_kv lane depends on. It is defined here, in the only llama header
    // a consumer includes, because the gate that reads it
    // (tests/test-union-binder-stock-equiv.cpp) includes llama.h and no
    // internal header; a definition in src/ could not turn that gate green,
    // and that is the property the probe wants.
    //
    // What it asserts: on the non-flash CPU path a narrowed graph and a graph
    // built at the same width run the same kernels with the same accumulation
    // tree, because the binder patches the live graph's shapes in place and
    // marks nothing. No REDUCE_PAD_TAIL, no MASKED_TAIL_GEMM and no
    // op_params[2]/[4] write is left on the lane, so the only selection
    // predicates that read ne00/ne01/ne11 read exactly n in both arms. Landed
    // as c78df85 (the llamafile gate visible in both conditional regions),
    // ce9ea33 (the reduction-order emulation cut from the n_kv lane) and
    // 209ed25 (the in-place-narrowing test, with falsifiers that can fail).
    //
    // What it does NOT assert: flash attention. cparams.flash_attn is excluded
    // from the token lane and the n_kv lane no longer marks FA at all, so a
    // CPU FA graph keeps its length-dependent dispatch; that is outside this
    // lane's contract and src/llama-union-binder.h states it there. Quantized
    // K/V, GPU/HIP and end-to-end logits are outside the contract too.
    //
    // The evidence is not this macro. It is the selection-site table in
    // ANALYSIS-EQUALITY.md section C (every predicate on the non-FA CPU path
    // is a type, precision, flag-constant or output-shape test, except the
    // three that read ne00/ne01/ne11, which the patch sets to exactly n in
    // both arms) and the KERNEL_SEL_END ok=1 line the gate requires. The macro
    // says only that the mechanism is present; it must be removed the day the
    // mechanism is not.
#define LLAMA_UNION_KERNEL_SEL_LEN_INDEP 1

    // Pillar 1: n_kv, n_outputs and rs_z as runtime parameters of one graph.
    //
    // `runtime_n_kv` is the width the live graph must execute at, as a VALUE:
    // the same value written into the graph's runtime n_kv input before each
    // compute. It is 0 when no width is armed, and it is armed only within
    // the width the live arena actually holds. `build_n_kv` is the width the
    // graph was constructed at: the allocation envelope on the legacy lanes,
    // the exact live width on the token lane (whose arena is widened by a
    // replan as the width moves, so the envelope is not yet allocated).
    // `bound_n_kv` is the width the last applied bind set.
    //
    // `build_n_outputs` is the out_ids row count the graph was built at
    // (the output high-water envelope), `bound_n_outputs` the width the
    // gather closure executes at, and `runtime_n_outputs` the active output
    // count of the last ubatch. A build that did not move between two steps
    // is the proof that the graph was not rebuilt.
    //
    // `bound_n_rs`, `bound_rs_head` and `bound_rs_z` are the recurrent state
    // length, head and zero row the live graph is bound at. n_rs and head are
    // shape bindings; rs_z is a runtime index consumed by the host zero-row
    // op (-1: no zero row, -2: this graph has no recurrent state).
    //
    // `topology_hash` hashes the live graph's node set, edges, op types and
    // non-runtime op parameters, with the n_kv, n_outputs and rs_z dimensions
    // excluded (see src/llama-union-topology.h). Two combinations that report
    // the same hash ran the same topology. `topology_hashed` is false when the
    // graph could not be hashed; the hash is then 0 and must not be compared.
    // This is the same value llama_get_union_diagnostics() reports as
    // `topology_hash`, and it is NOT the binder's reuse key
    // (src/llama-topology-hash.h), which is internal.
    //
    // `mask_n_kv`, `mask_filled` and `mask_refused` observe the last KQ-mask
    // fill: the width the fill consumed (read back from the graph's runtime
    // n_kv input, not taken from the width the caller asked for), how many
    // registered masks were filled at that width, and how many fills refused.
    // A step whose mask is stale cannot report mask_filled > 0 together with
    // mask_refused == 0.
    //
    // `mask_row_checked`, `mask_row_ok`, `mask_row_refused` and `mask_rows_all`
    // are the read-back that makes the three above falsifiable. After the fill,
    // the rows it wrote are compared, column by column below the same width,
    // with the visibility the KV cache's own cells imply (visible exactly when
    // the cell carries the token's sequence and no causal/SWA rule drops it).
    // The fill compares every row of the batch when the batch is at most 8
    // tokens and the last row otherwise; `mask_rows_all` says which happened.
    // The counters are rows, not masks: `mask_row_checked` rows compared,
    // `mask_row_ok` rows that agreed, `mask_row_refused` rows the check
    // declined to model. The per-field comment below states each one.
    //
    // The public test surface (`llama_union_mask_row_check`) re-runs the same
    // check on ONE row, the last batch token's, and returns a single verdict.
    // It is the test surface, not the fill's sweep.
    //
    // `graph_builds` is the context's monotonic model.build_graph() count at
    // the time of the call (the value llama_graph_build_count() returns). A
    // test compares it between two steps: an unchanged value across a run is
    // proof that the graph was never rebuilt.
    //
    // Returns false when the union binder is off or no graph is recorded.
    struct llama_nkv_runtime_stats {
        uint64_t topology_hash;
        uint32_t build_n_kv;
        uint32_t runtime_n_kv;
        uint32_t bound_n_kv;
        uint32_t build_n_outputs;
        uint32_t bound_n_outputs;
        uint32_t runtime_n_outputs;
        uint32_t bound_n_rs;
        uint32_t bound_rs_head;
        int32_t  bound_rs_z;
        bool     topology_hashed;
        bool     has_nkv_dims;
        bool     has_nout_dims;
        bool     has_nrs_dims;
        // appended after the original fields: the mask observation, the
        // build counter and the token-width observation. Appended, not
        // inserted, so every existing initializer keeps its order.
        uint32_t mask_n_kv;
        uint32_t mask_filled;
        uint32_t mask_refused;
        // The read-back of the fill against the KV cells: `mask_row_checked`
        // counts the ROWS compared with the cell table and `mask_row_ok` the
        // rows that agreed at every column below the width. This is the
        // independent observation: `mask_n_kv` is the width the fill was
        // told, `mask_row_ok` is the mask the graph will execute agreeing
        // with the cells it will attend. A stale mask is checked > 0 with
        // ok < checked; a graph with no F32 mask checked nothing.
        //
        // `mask_row_refused` counts rows the check declined to model (a
        // layout it cannot derive). It is separate from a disagreement on
        // purpose: a layout change that makes every check refuse must not
        // look like a wrong mask.
        //
        // `mask_rows_all` is 1 when EVERY row of the batch was compared and 0
        // when only the last row was (the batch was wider than the row cap).
        uint32_t mask_row_checked;
        uint32_t mask_row_ok;
        uint32_t mask_row_refused;
        uint32_t mask_rows_all;
        uint64_t graph_builds;
        // the token lane: the token width the live graph was built at, the
        // width it is bound at, and whether the token dimensions are recorded
        // as runtime dimensions (token_map_valid). build_n_tokens == 0 means
        // the token lane is not recorded for this graph.
        uint32_t build_n_tokens;
        uint32_t bound_n_tokens;
        bool     has_ntok_dims;
        // Appended again: the CPU attention work the last graph actually did,
        // measured inside the kernels from the loop bounds they executed.
        // `attn_live_elems` counts the elements the measured reductions
        // processed, `attn_pad_elems` the extent those same rows cover, and
        // `attn_rows` the number of executed runtime-length reduction rows.
        // They are monotonic for the context's lifetime, so a caller takes
        // deltas.
        //
        // The identity a test checks is attn_rows * (runtime n_kv) ==
        // attn_live_elems: it fails if the two counters drift. Rows come from
        // the ordered attention softmax and from a TRUE_LEN_REDUCE mul_mat.
        // On the ordered n_kv lane the bind narrows the mask and the K/V
        // views, so the executed length IS the extent and live == pad; a
        // TRUE_LEN_REDUCE node shaped at a padded extent reports live < pad.
        // A generic (unflagged) softmax still adds its elements to live and
        // pad but no rows.
        //
        // All three stay 0 when the context ran no CPU attention (GPU-only
        // placement, or no graph yet).
        uint64_t attn_live_elems;
        uint64_t attn_pad_elems;
        uint64_t attn_rows;
        // FIX-A: existing binder counter, appended for the zero-output arm.
        // c_fallbacks is already exposed as llama_union_diagnostics::lane_declines.
        uint64_t refused_nout;
    };
    LLAMA_API bool llama_get_nkv_runtime_stats(const struct llama_context * ctx, struct llama_nkv_runtime_stats * stats);

    // The verdict of a gate check. One encoding, used by the API, the logs and
    // the shell gates.
    //
    //   LLAMA_GATE_FAIL      the check ran and the observation did not hold.
    //   LLAMA_GATE_PASS      the check ran and the observation held.
    //   LLAMA_GATE_UNPROVEN  the check did not run, or refused to model its
    //                        input. Nothing was compared.
    //
    // A refusal is UNPROVEN, never PASS. A disagreement is FAIL.
    //
    // Do NOT test a verdict for truthiness: `v != 0` is true for UNPROVEN.
    // Compare against the named value.
    enum llama_gate_verdict {
        LLAMA_GATE_FAIL     = 0,
        LLAMA_GATE_PASS     = 1,
        LLAMA_GATE_UNPROVEN = 2,
    };

    // The read-back's test surface, for the same lane as the stats above.
    //
    // `llama_union_kq_mask_row` returns true when it set `*data` to row `row`
    // of the KQ mask the live graph's runtime n_kv input filled (row-major
    // over the mask's own ne[0] stride: the fill writes the first `width`
    // columns and pads the rest), and set `*ne0` to the mask's own
    // ne[0]. It returns false, and leaves `*data` null and `*ne0` zero, when
    // the graph has no such mask, the mask is not F32, or `row` is outside it.
    // The pointer is the live host buffer: it is valid only until the next
    // decode, and only a test that drives the read-back's failure arm may
    // write through it.
    //
    // `llama_union_mask_row_check` re-runs the read-back on the LAST row of
    // the last batch (token n_tokens - 1; one row, because that row covers the
    // whole step and because multi-row coverage is the fill's job, reported
    // through the stats above). Every column below the fill width must be
    // exactly 0.0f or -INFINITY and must agree with the visibility the KV
    // cache's cells imply. It returns LLAMA_GATE_PASS when the row agrees,
    // LLAMA_GATE_FAIL when a column disagrees, and LLAMA_GATE_UNPROVEN when
    // there is nothing to check or the layout cannot be modelled.
    //
    // The answer depends on live state, all of it invalidated by the next
    // decode: the mask row pointer the fill wrote (`llama_union_kq_mask_row`),
    // the batch the inputs were last filled from, and the context's own KV
    // cache, from which the cell table and the SWA rule are read. It is the
    // same code path the fill runs (`llama_kv_cache::check_mask_row`); it
    // exists so a test can prove the check can fail, not as a second
    // implementation.
    LLAMA_API bool llama_union_kq_mask_row  (const struct llama_context * ctx, uint32_t row, float ** data, int64_t * ne0);
    LLAMA_API enum llama_gate_verdict llama_union_mask_row_check(const struct llama_context * ctx);

    // Reports the requested projection policy, not model-wide qualification
    // or proof that every backend kernel honored it. NULL returns false.
    LLAMA_API bool llama_get_batch_invariant_matmul(const struct llama_context * ctx);

    // Read the current scheduler and attention configuration. These values can
    // differ from the requested model/context parameters. Graph reservation
    // can disable pipeline parallelism after a compute allocation failure.
    // They do not prove kernel use, device overlap, or a performance gain.
    // The caller must serialize context access. NULL returns false.
    LLAMA_API bool llama_get_pipeline_parallel(const struct llama_context * ctx);
    LLAMA_API bool llama_get_flash_attention  (const struct llama_context * ctx);

    DEPRECATED(LLAMA_API int32_t llama_n_ctx_train(const struct llama_model * model), "use llama_model_n_ctx_train instead");
    DEPRECATED(LLAMA_API int32_t llama_n_embd     (const struct llama_model * model), "use llama_model_n_embd instead");
    DEPRECATED(LLAMA_API int32_t llama_n_layer    (const struct llama_model * model), "use llama_model_n_layer instead");
    DEPRECATED(LLAMA_API int32_t llama_n_head     (const struct llama_model * model), "use llama_model_n_head instead");

    DEPRECATED(LLAMA_API int32_t llama_n_vocab    (const struct llama_vocab * vocab), "use llama_vocab_n_tokens instead");

    LLAMA_API const struct llama_model * llama_get_model   (const struct llama_context * ctx);
    LLAMA_API           llama_memory_t   llama_get_memory  (const struct llama_context * ctx);
    LLAMA_API  enum llama_pooling_type   llama_pooling_type(const struct llama_context * ctx); // TODO: rename to llama_get_pooling_type

    // Release this context's transient compute workspace at a caller-selected
    // phase boundary. Synchronizes pending computation and invalidates reusable
    // graphs; the next decode/encode allocates its actual graph shape again.
    // Model weights, KV/recurrent state, output buffers, and the scheduler's
    // MoE cache session are preserved. Does not change llama_set_decode_phase.
    // Returns false without changing state for NULL, metadata-only contexts,
    // an active training optimizer, or a shared CUDA KV-stream phase arena.
    // Otherwise returns true (including when no compute buffers are owned).
    // The caller must serialize access to the context. Do not call from an
    // evaluation callback or while another thread is using the same context.
    LLAMA_API bool llama_memory_release_compute(struct llama_context * ctx);

    // Current bytes owned by this context's graph allocator, counted once per
    // shared buffer type. This is not peak/reserved-size estimation or device
    // free/resident memory. Model/KV/output/provider-cache buffers are excluded.
    // Does not synchronize; requires serialized context access. NULL returns 0.
    LLAMA_API uint64_t llama_memory_compute_bytes(const struct llama_context * ctx);

    // Return the *effective* K/V cache tensor type currently used by the context's memory.
    // This can differ from the type_k/type_v requested via llama_context_params: some memory
    // implementations silently rewrite the requested type at construction time (e.g. TurboQuant
    // "auto-asymmetric" upgrades K to q8_0 for models with GQA ratio >= 6 when a symmetric turbo
    // K+V cache was requested, to avoid catastrophic quality loss - see llama_kv_cache's ctor and
    // its "auto-asymmetric" LLAMA_LOG_WARN).
    //
    // Returns GGML_TYPE_COUNT if ctx is NULL, if the memory has no K/V cache at all (e.g. pure
    // recurrent/Mamba-style memory), or if the memory is a composite of multiple sub-caches that
    // can legitimately hold different effective types (e.g. DSV4's raw/CSA/HCA/indexer caches).
    LLAMA_API enum ggml_type llama_get_kv_cache_type_k(const struct llama_context * ctx);
    LLAMA_API enum ggml_type llama_get_kv_cache_type_v(const struct llama_context * ctx);

    LLAMA_API const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
    LLAMA_API enum llama_rope_type       llama_model_rope_type(const struct llama_model * model);

    // Devices selected by the model loader. This list does not describe which
    // kernels ran, or prove that every listed device holds model layers.
    // The caller must keep the model and backend registry alive. The index
    // must be in [0, llama_model_n_devices(model)).
    LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);
    LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

    LLAMA_API int32_t llama_model_n_ctx_train  (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_embd       (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_embd_inp   (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_embd_out   (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_layer      (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_layer_nextn(const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_head       (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_head_kv    (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_swa        (const struct llama_model * model);

    // Get the model's RoPE frequency scaling factor
    LLAMA_API float llama_model_rope_freq_scale_train(const struct llama_model * model);

    // Returns the number of classifier outputs (only valid for classifier models)
    // Undefined behavior for non-classifier models
    LLAMA_API uint32_t llama_model_n_cls_out(const struct llama_model * model);

    // Returns label of classifier output by index (<n_cls_out). Returns nullptr if no label provided
    LLAMA_API const char * llama_model_cls_label(const struct llama_model * model, uint32_t i);

    LLAMA_API enum llama_vocab_type llama_vocab_type(const struct llama_vocab * vocab);

    LLAMA_API int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab);

    // Functions to access the model's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - When retrieving a string, an extra byte must be allocated to account for the null terminator
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    LLAMA_API int32_t llama_model_meta_val_str(const struct llama_model * model, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    LLAMA_API int32_t llama_model_meta_count(const struct llama_model * model);

    // Get sampling metadata key name. Returns nullptr if the key is invalid
    LLAMA_API const char * llama_model_meta_key_str(enum llama_model_meta_key key);

    // Get metadata key name by index
    LLAMA_API int32_t llama_model_meta_key_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    LLAMA_API int32_t llama_model_meta_val_str_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);

    // Get a string describing the model type
    LLAMA_API int32_t llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size);

    // Get the model file type (quantization), e.g. LLAMA_FTYPE_MOSTLY_Q8_0
    LLAMA_API enum llama_ftype llama_model_ftype(const struct llama_model * model);

    // Returns the total size of all the tensors in the model in bytes
    LLAMA_API uint64_t llama_model_size(const struct llama_model * model);

    // Get the default chat template. Returns nullptr if not available
    // If name is NULL, returns the default chat template
    LLAMA_API const char * llama_model_chat_template(const struct llama_model * model, const char * name);

    // Returns the total number of parameters in the model
    LLAMA_API uint64_t llama_model_n_params(const struct llama_model * model);

    // Returns true if the model contains an encoder that requires llama_encode() call
    LLAMA_API bool llama_model_has_encoder(const struct llama_model * model);

    // Returns true if the model contains a decoder that requires llama_decode() call
    LLAMA_API bool llama_model_has_decoder(const struct llama_model * model);

    // For encoder-decoder models, this function returns id of the token that must be provided
    // to the decoder to start generating output sequence. For other models, it returns -1.
    LLAMA_API llama_token llama_model_decoder_start_token(const struct llama_model * model);

    // Returns true if the model is recurrent (like Mamba, RWKV, etc.)
    LLAMA_API bool llama_model_is_recurrent(const struct llama_model * model);

    // Returns true if the model is hybrid (like Jamba, Granite, etc.)
    LLAMA_API bool llama_model_is_hybrid(const struct llama_model * model);

    // Returns true if the model is diffusion-based (like LLaDA, Dream, etc.)
    LLAMA_API bool llama_model_is_diffusion(const struct llama_model * model);

    // Returns 0 on success
    LLAMA_API uint32_t llama_model_quantize(
            const char * fname_inp,
            const char * fname_out,
            const llama_model_quantize_params * params);

    //
    // Adapters
    //

    // Load a LoRA adapter from file
    // The adapter is valid as long as the associated model is not freed
    LLAMA_API struct llama_adapter_lora * llama_adapter_lora_init(
            struct llama_model * model,
            const char * path_lora);

    // Functions to access the adapter's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - When retrieving a string, an extra byte must be allocated to account for the null terminator
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    LLAMA_API int32_t llama_adapter_meta_val_str(const struct llama_adapter_lora * adapter, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    LLAMA_API int32_t llama_adapter_meta_count(const struct llama_adapter_lora * adapter);

    // Get metadata key name by index
    LLAMA_API int32_t llama_adapter_meta_key_by_index(const struct llama_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    LLAMA_API int32_t llama_adapter_meta_val_str_by_index(const struct llama_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);

    // Manually free a LoRA adapter.
    // NOTE: loaded adapters that are not manually freed will be freed when the associated model is deleted.
    // When the associated model has a retained pending context, the adapter is
    // erased from the model and kept process-owned instead of deleted; a
    // duplicate call is ignored. This avoids freeing tensors that submitted
    // native work may still read. The decision uses the model's retained-
    // context count, not a per-adapter reference: once any context on the model
    // is retained, every released adapter is process-owned.
    LLAMA_API void llama_adapter_lora_free(struct llama_adapter_lora * adapter);

    // Get the invocation tokens if the current lora is an alora
    LLAMA_API uint64_t            llama_adapter_get_alora_n_invocation_tokens(const struct llama_adapter_lora * adapter);
    LLAMA_API const llama_token * llama_adapter_get_alora_invocation_tokens  (const struct llama_adapter_lora * adapter);

    // The following functions operate on a llama_context, hence the naming: llama_verb_...

    // Set LoRa adapters on the context. Will only modify if the adapters currently in context are different.
    LLAMA_API int32_t llama_set_adapters_lora(
            struct llama_context * ctx,
            struct llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales);

    // Apply a loaded control vector to a llama_context, or if data is NULL, clear
    // the currently loaded vector.
    // n_embd should be the size of a single layer's control, and data should point
    // to an n_embd x n_layers buffer starting from layer 1.
    // il_start and il_end are the layer range the vector should apply to (both inclusive)
    // See llama_control_vector_load in common to load a control vector.
    LLAMA_API int32_t llama_set_adapter_cvec(
            struct llama_context * ctx,
                     const float * data,
                          size_t   len,
                         int32_t   n_embd,
                         int32_t   il_start,
                         int32_t   il_end);

    //
    // Memory
    //

    // An explicit reader view of the same context's stored attention keys.
    // Positions are text RoPE positions, not byte or physical KV offsets.
    struct llama_attention_entry {
        llama_seq_id key_seq;
        llama_pos key_pos;
        llama_pos virtual_pos;
    };

    struct llama_attention_view {
        llama_seq_id query_seq;
        llama_pos query_pos;
        llama_pos query_virtual_pos;
        size_t n_entries;
        const struct llama_attention_entry * entries;
    };

    // This reference route uses stored K at writer positions and rotates Q for
    // each reader span. All visible entries share one softmax per query/head.
    // It requires a supported text RoPE model and a unified, resident KV cache.
    // Both K and V must use GGML_TYPE_F32, and Flash Attention must be disabled.
    // Current supported architectures use n_rs_seq=0; rollback/replay is unqualified.
    // Returns nullptr when supported, or the first unsupported requirement.
    // The reason is a static string; the caller must not modify or free it.
    // A null context returns a reason. Query again after changing context modes.
    LLAMA_API const char * llama_attention_views_unsupported_reason(const struct llama_context * ctx);

    LLAMA_API bool llama_attention_views_supported(const struct llama_context * ctx);

    // Copies the views. Failure preserves the previous views and all model state.
    // Each view must include its own current key exactly once. Foreign entries
    // must already exist; another query's pending key is not visible. The next
    // decode must have exactly these one-token, one-sequence query rows. These
    // APIs share the context's single-caller contract with llama_decode.
    LLAMA_API bool llama_attention_set_views(
            struct llama_context * ctx,
            const struct llama_attention_view * views,
            size_t n_views);

    // Clear the view before returning to ordinary decoding. Synchronizes work
    // that reads the current view before releasing its owned metadata.
    LLAMA_API void llama_attention_clear_views(struct llama_context * ctx);

    // Release only the native recurrent state. Stored attention keys keep their
    // logical sequence owner and remain visible through explicit reader views.
    LLAMA_API bool llama_memory_seq_rm_recurrent(llama_memory_t mem, llama_seq_id seq_id);

    // Versioned, read-only diagnostic view of a Qwen4Exp hybrid-index memory.
    // The caller sets version = LLAMA_MEMORY_HYBRID_IDX_TRACE_VERSION and
    // struct_size = sizeof(*out); a mismatch returns false without writing. The query
    // also returns false when the memory is not llama_memory_hybrid_idx or
    // seq_id is negative. The fields are host-visible diagnostics only: they
    // do not alter memory state and do not prove device execution. The query
    // follows the same single-caller contract as the other memory accessors.
    struct llama_memory_hybrid_idx_trace {
        uint32_t version;
        uint32_t struct_size;
        int32_t  gdn_tail_pos;
        uint32_t replay_len;
        uint32_t rs_idx;
        int32_t  ple_next_pos;
        uint64_t ple_count;
        int32_t  ple_recent[2];
        int32_t  attention_pos_max;
        int32_t  indexer_pos_max;
    };
    LLAMA_API bool llama_memory_hybrid_idx_get_trace(
            llama_memory_t mem,
              llama_seq_id seq_id,
        struct llama_memory_hybrid_idx_trace * out);

    // Clear the memory contents
    // If data == true, the data buffers will also be cleared together with the metadata
    LLAMA_API void llama_memory_clear(
            llama_memory_t mem,
                      bool data);

    // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    // Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
    // seq_id < 0 : match any sequence
    // p0 < 0     : [0,  p1]
    // p1 < 0     : [p0, inf)
    LLAMA_API bool llama_memory_seq_rm(
            llama_memory_t mem,
              llama_seq_id seq_id,
                 llama_pos p0,
                 llama_pos p1);

    // Copy all tokens that belong to the specified sequence to another sequence
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_memory_seq_cp(
            llama_memory_t mem,
              llama_seq_id seq_id_src,
              llama_seq_id seq_id_dst,
                 llama_pos p0,
                 llama_pos p1);

    // Removes all tokens that do not belong to the specified sequence
    LLAMA_API void llama_memory_seq_keep(
            llama_memory_t mem,
              llama_seq_id seq_id);

    // Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_memory_seq_add(
            llama_memory_t mem,
              llama_seq_id seq_id,
                 llama_pos p0,
                 llama_pos p1,
                 llama_pos delta);

    // Integer division of the positions by factor of `d > 1`
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_memory_seq_div(
            llama_memory_t mem,
              llama_seq_id seq_id,
                 llama_pos p0,
                 llama_pos p1,
                       int d);

    // Returns the smallest position present in the memory for the specified sequence
    // This is typically non-zero only for SWA caches
    // Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    // Return -1 if the sequence is empty
    LLAMA_API llama_pos llama_memory_seq_pos_min(
            llama_memory_t mem,
              llama_seq_id seq_id);

    // Returns the largest position present in the memory for the specified sequence
    // Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    // Return -1 if the sequence is empty
    LLAMA_API llama_pos llama_memory_seq_pos_max(
            llama_memory_t mem,
              llama_seq_id seq_id);

    // Check if the memory supports shifting
    LLAMA_API bool llama_memory_can_shift(llama_memory_t mem);

    //
    // State / sessions
    //

    // Returns the *actual* size in bytes of the state
    // (logits, embedding and memory)
    // Only use when saving the state, not when restoring it, otherwise the size may be too small.
    LLAMA_API size_t llama_state_get_size(struct llama_context * ctx);
    LLAMA_API DEPRECATED(size_t llama_get_state_size(struct llama_context * ctx),
        "use llama_state_get_size instead");

    // Copies the state to the specified destination address.
    // Destination needs to have allocated enough memory.
    // Returns the number of bytes copied
    LLAMA_API size_t llama_state_get_data(
            struct llama_context * ctx,
                         uint8_t * dst,
                          size_t   size);
    LLAMA_API DEPRECATED(size_t llama_copy_state_data(
            struct llama_context * ctx,
                         uint8_t * dst),
        "use llama_state_get_data instead");

    // Set the state reading from the specified address
    // Returns the number of bytes read
    LLAMA_API size_t llama_state_set_data(
            struct llama_context * ctx,
                   const uint8_t * src,
                          size_t   size);
    LLAMA_API DEPRECATED(size_t llama_set_state_data(
            struct llama_context * ctx,
                   const uint8_t * src),
        "use llama_state_set_data instead");

    // Save/load session file
    LLAMA_API bool llama_state_load_file(
            struct llama_context * ctx,
                      const char * path_session,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);
    LLAMA_API DEPRECATED(bool llama_load_session_file(
            struct llama_context * ctx,
                      const char * path_session,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out),
        "use llama_state_load_file instead");

    LLAMA_API bool llama_state_save_file(
            struct llama_context * ctx,
                      const char * path_session,
               const llama_token * tokens,
                          size_t   n_token_count);
    LLAMA_API DEPRECATED(bool llama_save_session_file(
            struct llama_context * ctx,
                      const char * path_session,
               const llama_token * tokens,
                          size_t   n_token_count),
        "use llama_state_save_file instead");

    // Get the exact size needed to copy the state of a single sequence
    LLAMA_API size_t llama_state_seq_get_size(
            struct llama_context * ctx,
                    llama_seq_id   seq_id);

    // Copy the state of a single sequence into the specified buffer
    LLAMA_API size_t llama_state_seq_get_data(
            struct llama_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    llama_seq_id   seq_id);

    // Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence
    // Returns:
    //  - Positive: Ok
    //  - Zero: Failed to load
    LLAMA_API size_t llama_state_seq_set_data(
            struct llama_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    llama_seq_id   dest_seq_id);

    LLAMA_API size_t llama_state_seq_save_file(
            struct llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   seq_id,
               const llama_token * tokens,
                          size_t   n_token_count);

    LLAMA_API size_t llama_state_seq_load_file(
            struct llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   dest_seq_id,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);

#define LLAMA_STATE_SEQ_FLAGS_NONE 0

// for backwards-compat
#define LLAMA_STATE_SEQ_FLAGS_SWA_ONLY 1

// work only with partial states, such as SWA KV cache or recurrent cache (e.g. Mamba)
#define LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY 1

// Keeps the tensor data on device buffers (i.e. not accessible in host memory, but faster save/load).
// Getting the state for a seq_id with this flag invalidates all prior states gotten for that seq_id with this flag.
#define LLAMA_STATE_SEQ_FLAGS_ON_DEVICE 2

    typedef uint32_t llama_state_seq_flags;

    LLAMA_API size_t llama_state_seq_get_size_ext(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
           llama_state_seq_flags   flags);

    LLAMA_API size_t llama_state_seq_get_data_ext(
            struct llama_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    llama_seq_id   seq_id,
           llama_state_seq_flags   flags);

    LLAMA_API size_t llama_state_seq_set_data_ext(
            struct llama_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    llama_seq_id   dest_seq_id,
           llama_state_seq_flags   flags);

    //
    // Decoding
    //

    // Return batch for single sequence of tokens
    // The sequence ID will be fixed to 0
    // The position of the tokens will be tracked automatically by llama_decode
    //
    // NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
    //
    LLAMA_API struct llama_batch llama_batch_get_one(
                  llama_token * tokens,
                      int32_t   n_tokens);

    // Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
    // Each token can be assigned up to n_seq_max sequence ids
    // The batch has to be freed with llama_batch_free()
    // If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
    // Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
    // The rest of the llama_batch members are allocated with size n_tokens
    // n_tokens must be positive, embd non-negative, and n_seq_max positive;
    // an invalid extent returns an empty batch.
    // On success, the selected token or embd array and every common array are
    // allocated. Their contents are uninitialized except that seq_id[i] is
    // non-null for i < n_tokens and seq_id[n_tokens] is the null sentinel.
    // On allocation failure, every field that could not be allocated is null;
    // the function never dereferences a failed allocation and never throws.
    // The caller must check every field it uses and call llama_batch_free() on
    // the partial batch before discarding it.
    LLAMA_API struct llama_batch llama_batch_init(
            int32_t n_tokens,
            int32_t embd,
            int32_t n_seq_max);

    // Frees a batch of tokens allocated with llama_batch_init()
    // Safe for a partially allocated batch returned after an allocation failure.
    // The seq_id pointer array must end with a null sentinel; llama_batch_init
    // always sets it.
    LLAMA_API void llama_batch_free(struct llama_batch batch);

    // Returns true when every array needed for n_tokens rows is present.
    // needs_embd selects the embedding input; needs_token selects the token
    // input. A batch returned by llama_batch_init can be incomplete after an
    // allocation failure; check it before use and call llama_batch_free() when
    // this returns false.
    // n_tokens must equal the capacity passed to llama_batch_init (or the
    // number of initialized seq_id entries). llama_batch does not store the
    // capacity, so a larger value reads past seq_id.
    static inline bool llama_batch_is_complete(
            const struct llama_batch * batch,
            int32_t n_tokens,
            bool needs_embd,
            bool needs_token) {
        if (batch == NULL || n_tokens < 0) {
            return false;
        }
        if (needs_embd && batch->embd == NULL) {
            return false;
        }
        if (needs_token && batch->token == NULL) {
            return false;
        }
        if (batch->pos == NULL || batch->n_seq_id == NULL ||
            batch->seq_id == NULL || batch->logits == NULL) {
            return false;
        }
        for (int32_t i = 0; i < n_tokens; ++i) {
            if (batch->seq_id[i] == NULL) {
                return false;
            }
        }
        return true;
    }

    // Process a batch of tokens.
    // In contrast to llama_decode() - this call does not use KV cache.
    // For encode-decoder contexts, processes the batch using the encoder.
    // Can store the encoder output internally for later use by the decoder's cross-attention layers.
    //   0 - success
    // < 0 - error. the memory state is restored to the state before this call
    LLAMA_API int32_t llama_encode(
            struct llama_context * ctx,
              struct llama_batch   batch);

    // Process a batch of tokens.
    // Requires the context to have a memory.
    // For encode-decoder contexts, processes the batch using the decoder.
    // Positive return values does not mean a fatal error, but rather a warning.
    // Upon fatal-error or abort, the ubatches that managed to be been processed will remain in the memory state of the context
    //   To handle this correctly, query the memory state using llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
    // Upon other return values, the memory state is restored to the state before this call
    //    0 - success
    //    1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    //    2 - aborted     (processed ubatches will remain in the context's memory)
    //   -1 - invalid input batch
    // < -1 - fatal error (processed ubatches will remain in the context's memory)
    LLAMA_API int32_t llama_decode(
            struct llama_context * ctx,
              struct llama_batch   batch);

    // Describes the role of subsequent llama_decode() batches for
    // phase-specialized memory allocators. It does not change model math.
    enum llama_decode_phase {
        LLAMA_DECODE_PHASE_AUTOMATIC  = 0,
        LLAMA_DECODE_PHASE_PROMPT     = 1,
        LLAMA_DECODE_PHASE_GENERATION = 2,
    };

    // The selected phase remains active until changed. AUTOMATIC preserves
    // the traditional token-count heuristic for callers without phase state.
    LLAMA_API void llama_set_decode_phase(
        struct llama_context * ctx, enum llama_decode_phase phase);

    // Set the number of threads used for decoding
    // n_threads is the number of threads used for generation (single token)
    // n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    LLAMA_API void llama_set_n_threads(struct llama_context * ctx, int32_t n_threads, int32_t n_threads_batch);

    // Get the number of threads used for generation of a single token.
    LLAMA_API int32_t llama_n_threads(struct llama_context * ctx);

    // Get the number of threads used for prompt and batch processing (multiple token).
    LLAMA_API int32_t llama_n_threads_batch(struct llama_context * ctx);

    // Set whether the context outputs embeddings or not
    // TODO: rename to avoid confusion with llama_get_embeddings()
    LLAMA_API void llama_set_embeddings(struct llama_context * ctx, bool embeddings);

    // Set whether to use causal attention or not
    // If set to true, the model will only attend to the past tokens
    LLAMA_API void llama_set_causal_attn(struct llama_context * ctx, bool causal_attn);

    // Set whether the model is in warmup mode or not
    // If true, all model tensors are activated during llama_decode() to load and cache their weights.
    //
    // note: using this can cause extra graph reallocations because it changes the graph topology with MoE models,
    //       so it is generally not recommended to use in practice. will be removed in the future
    DEPRECATED(LLAMA_API void llama_set_warmup(struct llama_context * ctx, bool warmup),
            "user code should do warmup runs manually [TAG_LLAMA_GRAPH_NO_WARMUP]");

    // Set abort callback
    LLAMA_API void llama_set_abort_callback(struct llama_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);

    // Wait until all computations are finished
    // This is automatically done when using one of the functions below to obtain the computation results
    // and is not necessary to call it explicitly in most cases
    LLAMA_API void llama_synchronize(struct llama_context * ctx);

    // Token logits obtained from the last call to llama_decode()
    // The logits for which llama_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // Rows: number of tokens for which llama_batch.logits[i] != 0
    // Cols: n_vocab
    // TODO: deprecate in favor of llama_get_logits_ith() (ref: https://github.com/ggml-org/llama.cpp/pull/14853#issuecomment-3113143522)
    LLAMA_API float * llama_get_logits(struct llama_context * ctx);

    // Logits for the ith token. For positive indices, Equivalent to:
    // llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab
    // Negative indices can be used to access logits in reverse order, -1 is the last logit.
    // returns NULL for invalid ids.
    LLAMA_API float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);

    // Get all output token embeddings.
    // when pooling_type == LLAMA_POOLING_TYPE_NONE or when using a generative model,
    // the embeddings for which llama_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // shape: [n_outputs*n_embd]
    // Otherwise, returns NULL.
    // TODO: deprecate in favor of llama_get_embeddings_ith() (ref: https://github.com/ggml-org/llama.cpp/pull/14853#issuecomment-3113143522)
    LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);

    // Get the embeddings for the ith token. For positive indices, Equivalent to:
    // llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
    // Negative indices can be used to access embeddings in reverse order, -1 is the last embedding.
    // shape: [n_embd] (1-dimensional)
    // returns NULL for invalid ids.
    LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);

    // Get the embeddings for a sequence id
    // Returns NULL if pooling_type is LLAMA_POOLING_TYPE_NONE
    // when pooling_type == LLAMA_POOLING_TYPE_RANK, returns float[n_cls_out] with the rank(s) of the sequence
    // otherwise: float[n_embd] (1-dimensional)
    LLAMA_API float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id);

    //
    // backend sampling API [EXPERIMENTAL]
    // note: use only if the llama_context was created with at least one llama_sampler_seq_config
    //

    // Get the backend sampled token for the ith token.
    // Returns LLAMA_TOKEN_NULL if no token was sampled.
    LLAMA_API llama_token llama_get_sampled_token_ith(struct llama_context * ctx, int32_t i);

    // Get the backend sampled probabilities for the ith token
    // The index matches llama_get_sampled_token_ith().
    // Returns NULL if no probabilities were generated.
    LLAMA_API float *  llama_get_sampled_probs_ith      (struct llama_context * ctx, int32_t i);
    LLAMA_API uint32_t llama_get_sampled_probs_count_ith(struct llama_context * ctx, int32_t i);

    // Get the backend sampled logits for the ith token
    // Returns NULL if no logits were sampled.
    LLAMA_API float *  llama_get_sampled_logits_ith      (struct llama_context * ctx, int32_t i);
    LLAMA_API uint32_t llama_get_sampled_logits_count_ith(struct llama_context * ctx, int32_t i);

    // Get the backend sampled candidates (token ids) for the ith token
    // These are needed to map probability/logit indices to vocab token ids.
    // Returns NULL if no candidates were sampled.
    LLAMA_API llama_token * llama_get_sampled_candidates_ith      (struct llama_context * ctx, int32_t i);
    LLAMA_API uint32_t      llama_get_sampled_candidates_count_ith(struct llama_context * ctx, int32_t i);

    //
    // Vocab
    //

    LLAMA_API const char * llama_vocab_get_text(const struct llama_vocab * vocab, llama_token token);

    LLAMA_API float llama_vocab_get_score(const struct llama_vocab * vocab, llama_token token);

    LLAMA_API enum llama_token_attr llama_vocab_get_attr(const struct llama_vocab * vocab, llama_token token);

    // Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
    LLAMA_API bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token);

    // Identify if Token Id is a control token or a render-able token
    LLAMA_API bool llama_vocab_is_control(const struct llama_vocab * vocab, llama_token token);

    // Special tokens
    LLAMA_API llama_token llama_vocab_bos(const struct llama_vocab * vocab); // beginning-of-sentence
    LLAMA_API llama_token llama_vocab_eos(const struct llama_vocab * vocab); // end-of-sentence
    LLAMA_API llama_token llama_vocab_eot(const struct llama_vocab * vocab); // end-of-turn
    LLAMA_API llama_token llama_vocab_sep(const struct llama_vocab * vocab); // sentence separator
    LLAMA_API llama_token llama_vocab_nl (const struct llama_vocab * vocab); // next-line
    LLAMA_API llama_token llama_vocab_pad(const struct llama_vocab * vocab); // padding
    LLAMA_API llama_token llama_vocab_mask(const struct llama_vocab * vocab); // mask

    LLAMA_API bool llama_vocab_get_add_bos(const struct llama_vocab * vocab);
    LLAMA_API bool llama_vocab_get_add_eos(const struct llama_vocab * vocab);
    LLAMA_API bool llama_vocab_get_add_sep(const struct llama_vocab * vocab);

    // model-specific suppress tokens (gguf key: tokenizer.ggml.suppress_tokens)
    LLAMA_API const llama_token * llama_vocab_get_suppress_tokens(const struct llama_vocab * vocab, int32_t * n_suppress_tokens);

    LLAMA_API llama_token llama_vocab_fim_pre(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_suf(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_mid(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_pad(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_rep(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_sep(const struct llama_vocab * vocab);

    DEPRECATED(LLAMA_API const char * llama_token_get_text(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_get_text instead");
    DEPRECATED(LLAMA_API float llama_token_get_score(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_get_score instead");
    DEPRECATED(LLAMA_API enum llama_token_attr llama_token_get_attr(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_get_attr instead");
    DEPRECATED(LLAMA_API bool llama_token_is_eog(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_is_eog instead");
    DEPRECATED(LLAMA_API bool llama_token_is_control(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_is_control instead");
    DEPRECATED(LLAMA_API llama_token llama_token_bos(const struct llama_vocab * vocab), "use llama_vocab_bos instead");
    DEPRECATED(LLAMA_API llama_token llama_token_eos(const struct llama_vocab * vocab), "use llama_vocab_eos instead");
    DEPRECATED(LLAMA_API llama_token llama_token_eot(const struct llama_vocab * vocab), "use llama_vocab_eot instead");
    DEPRECATED(LLAMA_API llama_token llama_token_cls(const struct llama_vocab * vocab), "use llama_vocab_cls instead");
    DEPRECATED(LLAMA_API llama_token llama_token_sep(const struct llama_vocab * vocab), "use llama_vocab_sep instead");
    DEPRECATED(LLAMA_API llama_token llama_token_nl (const struct llama_vocab * vocab), "use llama_vocab_nl instead");
    DEPRECATED(LLAMA_API llama_token llama_token_pad(const struct llama_vocab * vocab), "use llama_vocab_pad instead");
    DEPRECATED(LLAMA_API bool llama_add_bos_token(const struct llama_vocab * vocab), "use llama_vocab_get_add_bos instead");
    DEPRECATED(LLAMA_API bool llama_add_eos_token(const struct llama_vocab * vocab), "use llama_vocab_get_add_eos instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_pre(const struct llama_vocab * vocab), "use llama_vocab_fim_pre instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_suf(const struct llama_vocab * vocab), "use llama_vocab_fim_suf instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_mid(const struct llama_vocab * vocab), "use llama_vocab_fim_mid instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_pad(const struct llama_vocab * vocab), "use llama_vocab_fim_pad instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_rep(const struct llama_vocab * vocab), "use llama_vocab_fim_rep instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_sep(const struct llama_vocab * vocab), "use llama_vocab_fim_sep instead");

    // CLS is equivalent to BOS
    DEPRECATED(LLAMA_API llama_token llama_vocab_cls(const struct llama_vocab * vocab), // classification
            "use llama_vocab_bos instead");

    //
    // Tokenization
    //
    // The API is thread-safe.
    //

    /// @details Convert the provided text into tokens.
    /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
    /// @return Returns the number of tokens on success, no more than n_tokens_max
    /// @return Returns a negative number on failure - the number of tokens that would have been returned
    /// @return Returns INT32_MIN on overflow (e.g., tokenization result size exceeds int32_t limit)
    /// @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
    /// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
    ///                      as plaintext. Does not insert a leading space.
    LLAMA_API int32_t llama_tokenize(
        const struct llama_vocab * vocab,
                      const char * text,
                         int32_t   text_len,
                     llama_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);

    // Token Id -> Piece.
    // Uses the vocabulary in the provided context.
    // Does not write null terminator to the buffer.
    // User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
    // @param special If true, special tokens are rendered in the output.
    LLAMA_API int32_t llama_token_to_piece(
              const struct llama_vocab * vocab,
                           llama_token   token,
                                  char * buf,
                               int32_t   length,
                               int32_t   lstrip,
                                  bool   special);

    /// @details Convert the provided tokens into text (inverse of llama_tokenize()).
    /// @param text The char pointer must be large enough to hold the resulting text.
    /// @return Returns the number of chars/bytes on success, no more than text_len_max.
    /// @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
    /// @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
    /// @param unparse_special If true, special tokens are rendered in the output.
    LLAMA_API int32_t llama_detokenize(
        const struct llama_vocab * vocab,
               const llama_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special);

    //
    // Chat templates
    //

    /// Apply chat template. Inspired by hf apply_chat_template() on python.
    ///
    /// NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
    /// @param tmpl A Jinja template to use for this chat.
    /// @param chat Pointer to a list of multiple llama_chat_message
    /// @param n_msg Number of llama_chat_message in this chat
    /// @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
    /// @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
    /// @param length The size of the allocated buffer
    /// @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
    LLAMA_API int32_t llama_chat_apply_template(
                            const char * tmpl,
       const struct llama_chat_message * chat,
                                size_t   n_msg,
                                  bool   add_ass,
                                  char * buf,
                               int32_t   length);

    // Get list of built-in chat templates
    LLAMA_API int32_t llama_chat_builtin_templates(const char ** output, size_t len);

    //
    // Sampling API
    //
    // Sample usage:
    //
    //    // prepare the sampling chain at the start
    //    auto sparams = llama_sampler_chain_default_params();
    //
    //    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    //
    //    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));
    //    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9, 1));
    //    llama_sampler_chain_add(smpl, llama_sampler_init_temp (0.8));
    //
    //    // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
    //    // this sampler will be responsible to select the actual token
    //    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    //
    //    ...
    //
    //    // decoding loop:
    //    while (...) {
    //        ...
    //
    //        llama_decode(ctx, batch);
    //
    //        // sample from the logits of the last token in the batch
    //        const llama_token id = llama_sampler_sample(smpl, ctx, -1);
    //
    //        ...
    //    }
    //
    //    llama_sampler_free(smpl);
    //

    typedef void * llama_sampler_context_t;

    struct llama_sampler_data {
        struct ggml_tensor * logits;
        struct ggml_tensor * probs;
        struct ggml_tensor * sampled;
        struct ggml_tensor * candidates;
    };

    // user code can implement the interface below in order to create custom llama_sampler
    struct llama_sampler_i {
        const char *           (*name)  (const struct llama_sampler * smpl);                                 // can be NULL
        void                   (*accept)(      struct llama_sampler * smpl, llama_token token);              // can be NULL
        void                   (*apply) (      struct llama_sampler * smpl, llama_token_data_array * cur_p); // required
        void                   (*reset) (      struct llama_sampler * smpl);                                 // can be NULL
        struct llama_sampler * (*clone) (const struct llama_sampler * smpl);                                 // can be NULL if ctx is NULL
        void                   (*free)  (      struct llama_sampler * smpl);                                 // can be NULL if ctx is NULL

        // [EXPERIMENTAL]
        // backend sampling interface:

        // return true if the backend supports all ops needed by the sampler
        // note: call once per sampler
        bool (*backend_init)(struct llama_sampler * smpl, ggml_backend_buffer_type_t buft);

        // call after .backend_apply()
        void (*backend_accept)(
                struct llama_sampler * smpl,
                struct ggml_context  * ctx,
                struct ggml_cgraph   * gf,
                struct ggml_tensor   * selected_token);

        // call after .backend_init()
        void (*backend_apply)(
                struct llama_sampler      * smpl,
                struct ggml_context       * ctx,
                struct ggml_cgraph        * gf,
                struct llama_sampler_data * data);

        // called before graph execution to set inputs for the current ubatch
        void (*backend_set_input)(struct llama_sampler * smpl);
    };

    struct llama_sampler {
        struct llama_sampler_i * iface;

        llama_sampler_context_t ctx;
    };

    // [EXPERIMENTAL]
    // attach a sampler to the context
    // note: prefer initializing the context with llama_context_params.samplers when possible
    LLAMA_API bool llama_set_sampler(struct llama_context * ctx, llama_seq_id seq_id, struct llama_sampler * smpl);

    // mirror of llama_sampler_i:
    LLAMA_API struct llama_sampler * llama_sampler_init  (      struct llama_sampler_i * iface, llama_sampler_context_t ctx);
    LLAMA_API const char *           llama_sampler_name  (const struct llama_sampler * smpl);
    LLAMA_API void                   llama_sampler_accept(      struct llama_sampler * smpl, llama_token token);
    LLAMA_API void                   llama_sampler_apply (      struct llama_sampler * smpl, llama_token_data_array * cur_p);
    LLAMA_API void                   llama_sampler_reset (      struct llama_sampler * smpl);
    LLAMA_API struct llama_sampler * llama_sampler_clone (const struct llama_sampler * smpl);
    // important: do not free if the sampler has been added to a llama_sampler_chain (via llama_sampler_chain_add)
    LLAMA_API void                   llama_sampler_free  (      struct llama_sampler * smpl);

    // llama_sampler_chain
    // a type of llama_sampler that can chain multiple samplers one after another

    LLAMA_API struct llama_sampler * llama_sampler_chain_init(struct llama_sampler_chain_params params);

    // important: takes ownership of the sampler object and will free it when llama_sampler_free is called
    LLAMA_API void                   llama_sampler_chain_add(      struct llama_sampler * chain, struct llama_sampler * smpl);

    // return NULL if:
    //   - the sampler is NULL
    //   - the sampler is not a llama_sampler_chain
    //   - the index is out of bounds, unless i == -1
    //   - if i == -1, returns the chain itself (can be used to check if the sampler is a chain)
    LLAMA_API struct llama_sampler * llama_sampler_chain_get(      struct llama_sampler * chain, int32_t i);

    // the total number of samplers in the chain
    LLAMA_API int                    llama_sampler_chain_n  (const struct llama_sampler * chain);

    // after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
    LLAMA_API struct llama_sampler * llama_sampler_chain_remove(   struct llama_sampler * chain, int32_t i);

    // available samplers:

    LLAMA_API struct llama_sampler * llama_sampler_init_greedy(void);

    /// seed == LLAMA_DEFAULT_SEED to use a random seed.
    LLAMA_API struct llama_sampler * llama_sampler_init_dist(uint32_t seed);

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    /// Setting k <= 0 makes this a noop
    LLAMA_API struct llama_sampler * llama_sampler_init_top_k      (int32_t k);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LLAMA_API struct llama_sampler * llama_sampler_init_top_p      (float   p, size_t min_keep);

    /// @details Minimum P sampling as described in https://github.com/ggml-org/llama.cpp/pull/3841
    LLAMA_API struct llama_sampler * llama_sampler_init_min_p      (float   p, size_t min_keep);

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    LLAMA_API struct llama_sampler * llama_sampler_init_typical    (float   p, size_t min_keep);

    /// #details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
    LLAMA_API struct llama_sampler * llama_sampler_init_temp       (float   t);

    /// @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
    LLAMA_API struct llama_sampler * llama_sampler_init_temp_ext   (float   t, float   delta, float exponent);

    /// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
    LLAMA_API struct llama_sampler * llama_sampler_init_xtc        (float   p, float   t,     size_t min_keep, uint32_t seed);

    /// @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
    LLAMA_API struct llama_sampler * llama_sampler_init_top_n_sigma(float   n);

    /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLAMA_API struct llama_sampler * llama_sampler_init_mirostat(
                             int32_t   n_vocab,
                            uint32_t   seed,
                               float   tau,
                               float   eta,
                             int32_t   m);

    /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLAMA_API struct llama_sampler * llama_sampler_init_mirostat_v2(
                            uint32_t   seed,
                               float   tau,
                               float   eta);

    /// @details Initializes a GBNF grammar, see grammars/README.md for details.
    /// @param vocab The vocabulary that this grammar will be used with.
    /// @param grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty. Returns NULL if parsing of grammar_str fails.
    /// @param grammar_root The name of the start symbol for the grammar.
    LLAMA_API struct llama_sampler * llama_sampler_init_grammar(
            const struct llama_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root);

    DEPRECATED(LLAMA_API struct llama_sampler * llama_sampler_init_grammar_lazy(
            const struct llama_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root,
                         const char ** trigger_words,
                                size_t num_trigger_words,
                   const llama_token * trigger_tokens,
                                size_t num_trigger_tokens),
        "use llama_sampler_init_grammar_lazy_patterns instead");


    /// @details Lazy grammar sampler, introduced in https://github.com/ggml-org/llama.cpp/pull/9639
    /// @param trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.
    /// @param trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token included.
    LLAMA_API struct llama_sampler * llama_sampler_init_grammar_lazy_patterns(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens);


    /// NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k or top-p sampling first.
    LLAMA_API struct llama_sampler * llama_sampler_init_penalties(
                             int32_t   n_vocab,
                             int32_t   penalty_last_n,   // last n tokens to penalize (0 = disable penalty)
                               float   penalty_repeat,   // must be > 0.0, 1.0 = disabled
                               float   penalty_freq,     // must be finite, 0.0 = disabled
                               float   penalty_present); // must be finite, 0.0 = disabled

    ///  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
    LLAMA_API struct llama_sampler * llama_sampler_init_dry(
            const struct llama_vocab *  vocab,
                               float    dry_multiplier,
                               float    dry_base,
                             int32_t    dry_allowed_length,
                             int32_t    dry_penalty_last_n, // last n tokens to penalize (0 = disable penalty)
                          const char ** seq_breakers,
                              size_t    num_breakers);

    /// adaptive-p: select tokens near a configurable target probability over time.
    ///
    /// the adaptive-p sampler transforms the token probability distribution to favor tokens
    /// that fall near a user-configurable probability target.
    ///
    /// internally, the sampler maintains an exponential moving average of the *ORIGINAL*
    /// probabilities of selected tokens at each sampling step. it uses this EMA to compute an
    /// adapted target probability at each sampling step, thus maintaining the desired target
    /// probability over time.
    ///
    /// adaptive-p selects a token ID rather than just mutating candidates, so it must be last
    /// in the sampler chain (like mirostat, dist, greedy).
    ///
    /// only mild truncation before this sampler is recommended. we suggest applying min-p
    /// before adaptive-p as the only other active sampler in the chain.
    ///
    /// @param target select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
    /// @param decay  EMA decay for adaptation; history ≈ 1/(1-decay) tokens (valid range 0.0 - 0.99)
    /// @param seed   RNG seed
    ///
    /// ref: https://github.com/ggml-org/llama.cpp/pull/17927
    ///
    LLAMA_API struct llama_sampler * llama_sampler_init_adaptive_p(
                               float   target,
                               float   decay,
                            uint32_t   seed);

    LLAMA_API struct llama_sampler * llama_sampler_init_logit_bias(
                             int32_t   n_vocab,
                             int32_t   n_logit_bias,
              const llama_logit_bias * logit_bias);

    // this sampler is meant to be used for fill-in-the-middle infilling
    // it's supposed to be used after top_k + top_p sampling
    //
    // 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
    // 2. combine probs of tokens that have the same prefix
    //
    // example:
    //
    // - before:
    //   "hel":   0.5
    //   "hell":  0.2
    //   "hello": 0.1
    //   "dummy": 0.1
    //
    // - after:
    //   "hel":   0.8
    //   "dummy": 0.1
    //
    // 3. discard non-EOG tokens with low prob
    // 4. if no tokens are left -> pick EOT
    //
    LLAMA_API struct llama_sampler * llama_sampler_init_infill(const struct llama_vocab * vocab);

    // Returns the seed used by the sampler if applicable, LLAMA_DEFAULT_SEED otherwise
    LLAMA_API uint32_t llama_sampler_get_seed(const struct llama_sampler * smpl);

    /// @details Sample and accept a token from the idx-th output of the last evaluation
    //
    // Shorthand for:
    //    const auto * logits = llama_get_logits_ith(ctx, idx);
    //    llama_token_data_array cur_p = { ... init from logits ... };
    //    llama_sampler_apply(smpl, &cur_p);
    //    auto token = cur_p.data[cur_p.selected].id;
    //    llama_sampler_accept(smpl, token);
    //    return token;
    // Returns the sampled token
    LLAMA_API llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx);

    // TODO: extend in the future
    //LLAMA_API void llama_decode_with_sampler(struct llama_context * ctx, struct llama_sampler * smpl, struct llama_batch batch, ...);

    //
    // Model split
    //

    /// @details Build a split GGUF final path for this chunk.
    ///          llama_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
    //  Returns the split_path length.
    LLAMA_API int32_t llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int32_t split_no, int32_t split_count);

    /// @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
    ///          llama_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
    //  Returns the split_prefix length.
    LLAMA_API int32_t llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int32_t split_no, int32_t split_count);

    // Print system information
    LLAMA_API const char * llama_print_system_info(void);

    // Set callback for all future logging events.
    // If this is not called, or NULL is supplied, everything is output on stderr.
    // The logger state is global so these functions are NOT thread safe.
    LLAMA_API void llama_log_get(ggml_log_callback * log_callback, void ** user_data);
    LLAMA_API void llama_log_set(ggml_log_callback   log_callback, void *  user_data);

    //
    // Performance utils
    //
    // NOTE: Used by llama.cpp examples/tools, avoid using in third-party apps. Instead, do your own performance measurements.
    //

    struct llama_perf_context_data {
        // ms == milliseconds
        double t_start_ms;  // absolute start time
        double t_load_ms;   // time needed for loading the model
        double t_p_eval_ms; // time needed for processing the prompt
        double t_eval_ms;   // time needed for generating tokens

        int32_t n_p_eval;   // number of prompt tokens
        int32_t n_eval;     // number of generated tokens
        int32_t n_reused;   // number of times a ggml compute graph had been reused
    };

    struct llama_perf_sampler_data {
        double t_sample_ms; // time needed for sampling in ms

        int32_t n_sample;   // number of sampled tokens
    };

    LLAMA_API struct llama_perf_context_data llama_perf_context      (const struct llama_context * ctx);
    // Monotonic count of model graph build calls, including reserve and fallback builds.
    LLAMA_API uint64_t llama_graph_build_count(const struct llama_context * ctx);
    // Diagnostics for the experimental single-stream CPU graph. Lane declines
    // count per-kind dispatch misses; rebuild_fallbacks count failed reuse or
    // envelope attempts that require a stock/replacement graph.
    //
    // bound_n_kv is the n_kv width the live graph is currently bound at, i.e.
    // the width the next compute executes at: the true used span on the exact
    // lane, the stock padded width when the lane is not armed. 0 before the
    // first bind.
    //
    // pad_fallbacks counts the times the exact lane was armed and its width
    // query refused, so the graph had to execute at the stock padded width
    // instead of the width the binder chose. It is 0 on a run that stayed on
    // the exact lane. Since the pad bucket and its forced-fallback hook were
    // deleted (the padded lane is no longer a terminal state), the counter can
    // only move on a layout refusal, and the exact lane treats that refusal as
    // a hard failure; the field stays exported so a caller can see it is 0.
    //
    // exact_lane reports the GGML_TP5_UNION_BINDER_EXACT selection: 1 when the
    // runtime-length lane is active (the default under the binder gate), 0
    // when the stock padded control lane is selected. It is a configuration
    // value, not an observation of what a step executed at.
    //
    // topology_hash is the width-table hash of the live graph
    // (src/llama-union-topology.h): node set, edges, op types, every op
    // parameter and every shape dimension EXCEPT the ones that carry the
    // runtime parameters (n_kv, n_outputs, rs_z and, on the token lane,
    // n_tokens). It is re-taken after every applied bind, so an unchanged
    // value across a parameter sweep is evidence that the bind changed shapes
    // only. 0 means no graph is hashed.
    //
    // It is NOT the binder's reuse key (src/llama-topology-hash.h,
    // llama_topology_key). That key is invariant to the runtime parameters by
    // construction and is internal; exporting it here would report a number
    // that cannot move when the parameter does. This field is the pillar's
    // public observable and reports the same value as
    // llama_get_nkv_runtime_stats().topology_hash. It is a diagnostic
    // identity, not a security boundary.
    struct llama_union_diagnostics {
        uint64_t lane_declines;
        uint64_t rebuild_fallbacks;
        uint64_t patch_binds;
        uint64_t pad_fallbacks;
        uint32_t bound_tokens;
        uint32_t bound_n_kv;
        bool enabled;
        bool envelope_disabled;
        bool record_valid;
        bool token_map_valid;
        uint64_t topology_hash;
        int  exact_lane;
        // rs_z runtime parameter (GGML_TP5_RSZ_RUNTIME_PARAM): the gate as
        // read, whether the live graph carries the selection row, the named
        // fallback counters (a refusal is never silent), and the rs_z the
        // live graph is bound at.
        //
        // Every refusal of the lane increments exactly one named counter AND
        // rsz_rt_fallbacks, so
        //   rsz_rt_fallbacks == refused_gate + refused_arch + refused_no_rs_z
        //                      + refused_cell + refused_envelope
        //                      + refused_shape + refused_n_rs_seq
        //                      + refused_ring + refused_mixed_depth
        // holds for any context. A counter that moves is the reason the lane
        // is not active; nothing falls back silently.
        bool     rsz_rt_enabled;        // GGML_TP5_RSZ_RUNTIME_PARAM was exactly "1"
        bool     rsz_rt_active;         // the live graph carries the selection row
        uint64_t rsz_rt_fallbacks;      // total refusals of the lane
        uint64_t rsz_rt_refused_gate;   // umbrella path off or cannot carry the lane
        uint64_t rsz_rt_refused_arch;   // the graph has no recurrent-state input
        uint64_t rsz_rt_refused_no_rs_z;// the build carried no zero-row index
        uint64_t rsz_rt_refused_cell;   // rs_z < 0: no free zero-state cell
        uint64_t rsz_rt_refused_envelope;// rs_z outside the cell capacity
        uint64_t rsz_rt_refused_shape;  // index tensor absent or not I32 [1]
        uint64_t rsz_rt_refused_n_rs_seq;// n_rs_seq > 0 on the non-relay lane
        uint64_t rsz_rt_refused_ring;   // state rows are not a whole number of groups
        uint64_t rsz_rt_refused_mixed_depth; // the ubatch's sequences disagree on the zero row
        int32_t  rsz_rt_bound_rs_z;     // rs_z the live graph is bound at (-1: none)
        // The cell capacity (mem->size), i.e. the per-group height of the state
        // tensor. Named "zero_view_rows" for the field's original meaning; the
        // zero view no longer exists, so read it as mem_size.
        uint32_t rsz_rt_zero_view_rows;
    };
    LLAMA_API bool llama_get_union_diagnostics(const struct llama_context * ctx, struct llama_union_diagnostics * out);

    // Copy a final normalized activation row after a successful single-
    // microbatch dense LLAMA decode with all rows in batch order. Read before
    // any other graph or memory operation on this context; output accessors
    // and synchronization are safe. No eval callback or embeddings mode is
    // enabled. Returns false for unsupported layouts or a wrong size.
    LLAMA_API bool llama_copy_last_llama_activation(struct llama_context * ctx, int32_t row, float * dst, size_t count);
    LLAMA_API void                           llama_perf_context_print(const struct llama_context * ctx);
    LLAMA_API void                           llama_perf_context_reset(      struct llama_context * ctx);

    //
    // factor-prefab counters (llama-prefab lane; see src/llama-prefab.h)
    //
    // Organic per-context counters for the graph-build/reuse lifecycle of the
    // prefab and stock builder paths. Every write is armed only when the env
    // gate GGML_TP5_PREFAB_STATS is exactly "1"; with the gate off (the
    // default) all fields stay zero and the hot path is untouched.
    //
    struct llama_prefab_stats {
        uint64_t n_ubatch_prefab;       // ubatches executed on an envelope graph
        uint64_t n_ubatch_stock;        // ubatches executed by the stock builder path
        uint64_t n_stock_builds;        // stock model.build_graph() invocations
        uint64_t n_stock_reuses;        // stock graph reuses (can_reuse == true)
        uint64_t n_envelope_builds;     // envelope graph compositions
        uint64_t n_envelope_reuses;     // envelope graph reused without rebuild
        uint64_t n_envelope_allocs;     // ggml_backend_sched_alloc_graph on an envelope sched
        uint64_t n_sub_envelope;        // ubatches with n_tokens < n_env (runtime padded)
        uint64_t n_pad_rows;            // total pad rows executed
        uint64_t n_factor_compiles;     // factor modules compiled
        uint64_t n_factor_cache_hits;   // factor modules taken from the cache while composing
        uint64_t n_factor_splices;      // factor modules spliced into an envelope
        uint64_t n_factor_clones;       // tensors cloned by splicing
        uint64_t n_cache_resets;        // topology-key invalidations of the factor cache
        uint64_t n_fallback_no_scratch; // runtime fallback to stock: no free scratch KV cell
        uint64_t n_fallback_runtime;    // runtime fallback to stock: adapters / samplers / attention views / ...
        uint64_t last_n_env;            // envelope width of the last prefab ubatch
        uint64_t last_n_tokens;         // real n_tokens of the last prefab ubatch
        uint64_t last_graph_nodes;      // node count of the last executed envelope graph
        uint64_t last_inp_tokens_ne0;   // ne[0] of the token input of the executed envelope graph (== last_n_env)
        uint64_t n_decouple_declines;   // decouple iface: ubatches declined by the compose layer (gate on)
        uint64_t n_decouple_handled;    // decouple iface: ubatches executed by the compose layer (gate on)
        // MTP factor compose (GGML_TP5_PREFAB_IN_GRAPH=1), all organic:
        uint64_t n_mtp_compose_hits;    // DECODER_MTP ubatches executed on a composed A x B envelope graph
        uint64_t n_mtp_a_compiles;      // A (head prologue) modules built
        uint64_t n_mtp_a_reuses;        // A modules taken from the cache while composing
        uint64_t n_mtp_b_builds;        // B (block + epilogue) parts emitted into an envelope
        uint64_t n_mtp_a_clones;        // tensors cloned from A while splicing it into an envelope
        uint64_t n_mtp_declines;        // DECODER_MTP ubatches the compose layer declined (stock fallback)
        uint64_t build_ns;              // wall time in instrumented request graph builds, excluding allocation/compute
        uint64_t n_nodes_built;         // nodes in successfully emitted request graphs, plus compiled MTP A graphs
        uint64_t template_hits;         // MTP A template cache hits, not whole-graph hits
    };

    LLAMA_API bool   llama_prefab_is_enabled      (const struct llama_context * ctx);
    LLAMA_API void   llama_prefab_get_stats       (const struct llama_context * ctx, struct llama_prefab_stats * out);
    LLAMA_API void   llama_prefab_reset_stats     (      struct llama_context * ctx);
    LLAMA_API size_t llama_prefab_n_cached_factors(const struct llama_context * ctx);

    // NOTE: the following work only with samplers constructed via llama_sampler_chain_init
    LLAMA_API struct llama_perf_sampler_data llama_perf_sampler      (const struct llama_sampler * chain);
    LLAMA_API void                           llama_perf_sampler_print(const struct llama_sampler * chain);
    LLAMA_API void                           llama_perf_sampler_reset(      struct llama_sampler * chain);

    //
    // training
    //

    // function that returns whether or not a given tensor contains trainable parameters
    typedef bool (*llama_opt_param_filter)(const struct ggml_tensor * tensor, void * userdata);

    // always returns true
    LLAMA_API bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata);

    struct llama_opt_params {
        uint32_t n_ctx_train; // assumed context size post training, use context size specified in llama_context if 0

        llama_opt_param_filter param_filter; // callback for determining which tensors contain trainable parameters
        void * param_filter_ud;              // userdata for determining which tensors contain trainable parameters

        ggml_opt_get_optimizer_params get_opt_pars; // callback for calculating optimizer parameters
        void * get_opt_pars_ud;                     // userdata for calculating optimizer parameters

        enum ggml_opt_optimizer_type optimizer_type;
    };

    LLAMA_API void llama_opt_init(struct llama_context * lctx, struct llama_model * model, struct llama_opt_params lopt_params);

    LLAMA_API void llama_opt_epoch(
            struct llama_context    * lctx,
            ggml_opt_dataset_t        dataset,
            ggml_opt_result_t         result_train,
            ggml_opt_result_t         result_eval,
            int64_t                   idata_split,
            ggml_opt_epoch_callback   callback_train,
            ggml_opt_epoch_callback   callback_eval);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_H
