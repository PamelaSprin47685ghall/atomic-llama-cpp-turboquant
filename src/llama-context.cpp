#include "llama-context.h"

#include "../ggml/src/ggml-backend-moe-cache.h"
#include "ggml.h"
#include "ggml-relay.h"
#include "llama-arch.h"
#include "llama-batch.h"
#include "llama-ext.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-nout-state.h"
#include "llama-io.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-msa.h"
#include "llama-kv-cache.h"
#include "llama-kv-stream-config.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

//
// llama_context
//

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t     n_tokens_per_seq;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/"Flash Attention",
    /*.n_tokens_per_seq =*/1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ar_probe = {
    /*.op               =*/LLM_FUSED_OP_GDN_AR,
    /*.name             =*/"fused Gated Delta Net (autoregressive)",
    /*.n_tokens_per_seq =*/1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ch_probe = {
    /*.op               =*/LLM_FUSED_OP_GDN_CH,
    /*.name             =*/"fused Gated Delta Net (chunked)",
    /*.n_tokens_per_seq =*/16,
};

static const llm_fused_op_probe llm_fused_op_lid_probe = {
    /*.op               =*/LLM_FUSED_OP_LIGHTNING_INDEXER,
    /*.name             =*/"Lightning Indexer",
    /*.n_tokens_per_seq =*/1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_pre_probe = {
    /*.op               =*/LLM_FUSED_OP_DSV4_HC_PRE,
    /*.name             =*/"fused DeepSeek V4 HC pre",
    /*.n_tokens_per_seq =*/1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_comb_probe = {
    /*.op               =*/LLM_FUSED_OP_DSV4_HC_COMB,
    /*.name             =*/"fused DeepSeek V4 HC comb",
    /*.n_tokens_per_seq =*/1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_post_probe = {
    /*.op               =*/LLM_FUSED_OP_DSV4_HC_POST,
    /*.name             =*/"fused DeepSeek V4 HC post",
    /*.n_tokens_per_seq =*/1,
};

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT:
            return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP:
            return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

llama_context::llama_context(const llama_model & model, llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    // per-context union-binder state slot (llama-nout-state.h): NOT created
    // here eagerly - the constructor reads the env gate below and creates the
    // slot only when the gate is on, so the OFF path allocates no binder
    // state. Unique_ptr ownership in llama-context.h: a constructor-body
    // throw still destroys fully-initialized members, so the slot cannot
    // leak, and ~llama_context releases it with the context.

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    // Relay ownership is independent of the backend arithmetic policy. Keep
    // the same model/layout admission for both native and decoded-F32 kernels.
    if (params.quantized_matmul_f32 || params.tp5_relay) {
        const bool dense_q8 = model.arch == LLM_ARCH_QWEN3 && model.ftype() == LLAMA_FTYPE_MOSTLY_Q8_0 &&
            hparams.n_embd == 4096 && hparams.n_layer() == 36 && hparams.n_expert == 0 && hparams.n_expert_used == 0;
        const bool swift = model.arch == LLM_ARCH_QWEN4EXP && model.ftype() == LLAMA_FTYPE_MOSTLY_Q2_0 &&
            hparams.n_embd == 2560 && hparams.n_layer() == 48 && hparams.n_expert == 512 && hparams.n_expert_used == 10 &&
            hparams.n_ff_exp == 640 && hparams.n_ff_shexp == 640 && hparams.ssm_d_conv == 4 &&
            hparams.ssm_d_state == 128 && hparams.ssm_n_group == 16 && hparams.ssm_dt_rank == 48 &&
            hparams.ssm_d_inner == 6144 && hparams.dsv4_hc_mult == 4 && hparams.hc_low_rank == 320 &&
            hparams.ple_n_heads == 16 && hparams.ple_head_dim == 160 && hparams.ple_ngram_size == 3 &&
            hparams.indexer_n_head == 4 && hparams.indexer_head_size == 128 && hparams.indexer_top_k == 2048;
        if ((!dense_q8 && !swift) || hparams.vocab_only || hparams.no_alloc || hparams.n_layer_nextn != 0 ||
            hparams.n_swa != 0 || !hparams.causal_attn || !model.loras.empty() || model.has_tensor_overrides()) {
            throw std::runtime_error("quantized_matmul_f32 requires the Qwen3-8B Q8_0 or Swift Qwen4Exp mixed-Q2_0 profile without adapters or overrides");
        }
        for (uint32_t layer = 0; layer < hparams.n_layer(); ++layer) {
            if (dense_q8 ?
                (hparams.n_head(layer) != 32 || hparams.n_head_kv(layer) != 8 || hparams.n_ff(layer) != 12288 ||
                 hparams.n_embd_head_k(layer) != 128 || hparams.n_embd_head_v(layer) != 128 ||
                 hparams.is_recr(layer) || hparams.is_swa(layer)) :
                (hparams.n_head(layer) != 24 || hparams.n_head_kv(layer) != 2 ||
                 hparams.n_embd_head_k(layer) != 256 || hparams.n_embd_head_v(layer) != 256 ||
                 hparams.is_recr(layer) != ((layer + 1) % 4 != 0) || hparams.is_ple(layer) != (layer == 1) ||
                 hparams.is_swa(layer))) {
                throw std::runtime_error("quantized_matmul_f32: unsupported layer geometry");
            }
        }
        if (model.devices.size() != 1 || model.n_gpu_layers() <= hparams.n_layer_all) {
            throw std::runtime_error("quantized_matmul_f32 requires full GPU layer/output offload on one execution device");
        }
        const auto device = model.devices.front().dev;
        const bool meta = ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_META;
        if (swift && (!meta || model.split_mode() != LLAMA_SPLIT_MODE_TENSOR ||
                      model.get_split_state_ud.n_devices != GGML_RELAY_RANKS)) {
            throw std::runtime_error("Swift decoded-F32 requires five tensor-split HIP ranks, including the staged reference");
        }
        if (!params.tp5_relay && !meta) {
            const auto reg = ggml_backend_dev_backend_reg(device);
            if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU || reg == nullptr ||
                std::strcmp(ggml_backend_reg_name(reg), "ROCm") != 0 ||
                (model.split_mode() != LLAMA_SPLIT_MODE_NONE && model.split_mode() != LLAMA_SPLIT_MODE_LAYER)) {
                throw std::runtime_error("quantized_matmul_f32 reference requires one HIP GPU without tensor/row split; CPU reference is unsupported");
            }
        }
        if (model.dev_output() != device) {
            throw std::runtime_error("quantized_matmul_f32 requires the output projection on the selected GPU device");
        }
        for (uint32_t layer = 0; layer < hparams.n_layer(); ++layer) {
            if (model.dev_layer(layer) != device) {
                throw std::runtime_error("quantized_matmul_f32 refuses mixed CPU/GPU layer offload");
            }
        }
        for (const auto & entry : model.tensors_by_name) {
            const auto * tensor = entry.second;
            if (tensor != nullptr && ggml_is_quantized(tensor->type)) {
                if (swift && entry.first == "per_layer_token_embd.weight") {
                    if (tensor->type != GGML_TYPE_IQ4_NL || tensor->ne[0] != 160 ||
                        tensor->ne[1] != 320001536 || tensor->buffer == nullptr ||
                        !ggml_backend_buffer_is_host(tensor->buffer)) {
                        throw std::runtime_error("Swift PLE requires its original CPU-resident IQ4_NL input table");
                    }
                    continue;
                }
                const bool supported_type = tensor->type == GGML_TYPE_Q8_0 || (swift &&
                    (tensor->type == GGML_TYPE_Q2_0 || tensor->type == GGML_TYPE_Q3_K ||
                     tensor->type == GGML_TYPE_Q4_K || tensor->type == GGML_TYPE_Q5_K || tensor->type == GGML_TYPE_Q6_K ||
                     tensor->type == GGML_TYPE_IQ4_XS || tensor->type == GGML_TYPE_IQ4_NL ||
                     tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q5_0));
                if (!supported_type || tensor->buffer == nullptr ||
                    ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer)) != device) {
                    throw std::runtime_error("quantized_matmul_f32 requires supported original weight blocks on the selected GPU device");
                }
            }
        }
        if (params.batch_invariant_matmul || !params.op_offload || !params.offload_kqv ||
            params.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || params.ctx_other != nullptr ||
            params.n_batch == 0 || params.n_batch > 4 || params.n_seq_max == 0 || params.n_seq_max > 4) {
            throw std::runtime_error("quantized_matmul_f32 requires the default decoder, 1..4 sequences/rows, GPU operation/KQV offload and batch_invariant_matmul=false");
        }
    }

    const bool tp5_native_q4_kv =
        params.tp5_relay && model.arch == LLM_ARCH_QWEN4EXP &&
        !params.quantized_matmul_f32 && params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED &&
        params.type_k == GGML_TYPE_Q4_0 && params.type_v == GGML_TYPE_Q4_0;

    if (params.tp5_relay) {
        if (model.split_mode() != LLAMA_SPLIT_MODE_TENSOR || model.devices.size() != 1 ||
            ggml_backend_dev_type(model.devices.front().dev) != GGML_BACKEND_DEVICE_TYPE_META ||
            model.get_split_state_ud.n_devices != GGML_RELAY_RANKS || model.n_gpu_layers() <= hparams.n_layer_all) {
            throw std::runtime_error("tp5-relay requires tensor split across five HIP devices and full GPU layer/output offload");
        }
        const auto input_device = model.tok_embd != nullptr && model.tok_embd->buffer != nullptr ?
            ggml_backend_buft_get_device(ggml_backend_buffer_get_type(model.tok_embd->buffer)) : nullptr;
        if (input_device == nullptr || ggml_backend_dev_type(input_device) != GGML_BACKEND_DEVICE_TYPE_META) {
            throw std::runtime_error("tp5-relay requires offload_input=true when loading the model");
        }
        if (params.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || params.ctx_other != nullptr ||
            params.n_seq_max == 0 || params.n_seq_max > 4 || params.n_batch == 0 || params.n_batch > 4 ||
            params.n_ctx == 0 || params.n_ctx > (model.arch == LLM_ARCH_QWEN4EXP ? 1048576u : 2048u) ||
            (params.n_ubatch != 0 && params.n_ubatch < params.n_batch) ||
            (!tp5_native_q4_kv &&
             ((params.type_k != GGML_TYPE_F16 && params.type_k != GGML_TYPE_F32) ||
              (params.type_v != GGML_TYPE_F16 && params.type_v != GGML_TYPE_F32))) ||
            !params.offload_kqv || !params.op_offload || params.embeddings || params.n_samplers != 0 ||
            params.cb_eval != nullptr || params.kv_stream_arena_mib != 0 || params.n_rs_seq != 0 ||
            params.attention_type == LLAMA_ATTENTION_TYPE_NON_CAUSAL || params.batch_invariant_matmul) {
            throw std::runtime_error("tp5-relay requires 1..4 sequences/rows, bounded profile KV capacity, one ubatch, F16/F32 KV or native Swift FA Q4_0 KV, no sampler, callback or alternate compute policy");
        }
    }

    cparams.n_seq_max     = std::max(1u, params.n_seq_max);
    cparams.n_outputs_max = params.n_outputs_max;
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }
    cparams.n_rs_slots = params.n_rs_slots == 0 ? cparams.n_seq_max : params.n_rs_slots;
    if (cparams.n_rs_slots > cparams.n_seq_max) {
        throw std::runtime_error("n_rs_slots must be <= n_seq_max");
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        LLAMA_LOG_DEBUG(
            "%s: n_rs_seq=%u requested but model arch does not support recurrent partial rollback; clamping to 0\n",
            __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }
    cparams.gdn_replay = params.gdn_replay;
    cparams.batch_invariant_matmul = params.batch_invariant_matmul;
    cparams.quantized_matmul_f32 = params.quantized_matmul_f32;

    cparams.n_threads            = params.n_threads;
    cparams.n_threads_batch      = params.n_threads_batch;
    cparams.moe_cache_mode       = params.moe_cache_mode;
    cparams.moe_cache_budget_mib = params.moe_cache_budget_mib;
    cparams.yarn_ext_factor      = params.yarn_ext_factor >= 0.0f ? params.yarn_ext_factor : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor     = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast       = params.yarn_beta_fast >= 0.0f ? params.yarn_beta_fast : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow       = params.yarn_beta_slow >= 0.0f ? params.yarn_beta_slow : hparams.yarn_beta_slow;
    cparams.embeddings           = params.embeddings;
    cparams.embeddings_nextn     = false;
    cparams.embeddings_nextn_masked = false;
    cparams.mtp_chain               = false;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.kv_stream_arena_mib     = params.kv_stream_arena_mib;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    // +1: id n_layer() taps the output of the last layer ("input" of the head)
    cparams.embeddings_layer_inp.resize(hparams.n_layer() + 1, false);
    embd_layer_inp.resize(hparams.n_layer() + 1);

    cparams.ctx_type     = params.ctx_type;
    cparams.pooling_type = params.pooling_type;

    cparams.n_ctx           = params.n_ctx == 0 ? hparams.n_ctx_train : params.n_ctx;
    cparams.rope_freq_base  = params.rope_freq_base == 0.0f ? hparams.rope_freq_base_train : params.rope_freq_base;
    cparams.rope_freq_scale = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn = params.yarn_orig_ctx != 0    ? params.yarn_orig_ctx :
                              hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                             hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error(
                "Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH || model.arch == LLM_ARCH_QWEN4EXP) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw std::runtime_error(
                    model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        }
    }

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id,
                               n_samplers);
            }
        }
    }

    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f;  // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) {  // negative indicates 'not set'
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
            float       mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                           __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.fused_lid  = true;
    cparams.auto_flid  = true;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = true;

    cparams.fused_dsv4_hc_pre  = true;
    cparams.fused_dsv4_hc_comb = true;
    cparams.fused_dsv4_hc_post = true;
    cparams.auto_fhc           = true;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    if (cparams.n_rs_seq > 0) {
        const uint32_t n_batch_min = cparams.n_rs_seq + 2;
        if (cparams.n_ctx < n_batch_min) {
            throw std::runtime_error(
                format("n_ctx (%u) must be at least n_rs_seq + 2 (%u)", cparams.n_ctx, n_batch_min));
        }
        if (cparams.n_batch < n_batch_min) {
            LLAMA_LOG_WARN("%s: n_batch (%u) is too small for n_rs_seq=%u; increasing to %u\n", __func__,
                           cparams.n_batch, cparams.n_rs_seq, n_batch_min);
            cparams.n_batch = n_batch_min;
        }
        if (cparams.n_ubatch < n_batch_min) {
            LLAMA_LOG_WARN("%s: n_ubatch (%u) is too small for n_rs_seq=%u; increasing to %u\n", __func__,
                           cparams.n_ubatch, cparams.n_rs_seq, n_batch_min);
            cparams.n_ubatch = n_batch_min;
        }
    }

    cparams.n_outputs_max =
        params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;

    // initialized later
    cparams.pipeline_parallel = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // Lane gates (independent, each exact "1"; unset or any other value is
    // stock for that lane):
    //  - GGML_TP5_UNION_BINDER: the legacy outputs lane (n_outputs binding).
    //    Kept for old-gate compatibility: with only this gate on, the nkv
    //    lane is off and n_kv behavior is stock.
    //  - GGML_TP5_NKV_TRUEWIDTH: the r7 true-width nkv lane. With only this
    //    gate on, the outputs lane is off and n_outputs behavior is stock.
    //    With this gate OFF there is NO fallback to any envelope/pad
    //    behavior: the envelope machinery was removed from the tree, not
    //    gated, so OFF is stock, not r5.
    const char * union_binder_env     = getenv("GGML_TP5_UNION_BINDER");
    const bool   union_binder_gate    = union_binder_env != nullptr && strcmp(union_binder_env, "1") == 0;
    const char * nkv_truewidth_env    = getenv("GGML_TP5_NKV_TRUEWIDTH");
    const bool   nkv_truewidth_gate   = nkv_truewidth_env != nullptr && strcmp(nkv_truewidth_env, "1") == 0;

    {
        // The state slot exists only while at least one lane gate is exactly
        // "1"; with both gates off the context holds no binder state at all
        // (OFF = stock, zero alloc).
        if (union_binder_gate || nkv_truewidth_gate) {
            auto & nkv_st = llama_nkv_ctx_state_get(this);

            // r7 true-width nkv lane: registered only under its own gate.
            if (nkv_truewidth_gate) {
                nkv_st.nkv_enabled = true;
                if (!nkv_st.union_binder.add({
                        llama_union_param_kind::n_kv, "n_kv",
                        &llama_context::union_tensors_n_kv,
                        &llama_context::union_rebind_n_kv,
                    })) {
                    // cannot happen for a valid kind; stay fail-closed anyway
                    nkv_st.nkv_enabled = false;
                    LLAMA_LOG_ERROR("%s: failed to register the n_kv union binding; runtime graph patching disabled\n", __func__);
                }
            }

            // n_outputs binding: the legacy outputs lane under the old gate.
            // Bounded binding: only the output-gather closure (out_ids + every
            // GET_ROWS dst reading it) is reconciled, and only when the graph
            // builds WITHOUT backend samplers: build_sampling() consumes the
            // gather output through a PAD (+1 row), which the binding does not
            // cover. Under the binding, decoder graphs build at the exact
            // active count and the rebind narrows/restores the executed shapes
            // within that built width; 0 active outputs is a supported bound
            // (the stock zero-output topology).
            if (union_binder_gate) {
                if (!sampling.samplers.empty()) {
                    LLAMA_LOG_WARN("%s: n_outputs union binding NOT registered: backend samplers attached; the gather output feeds build_sampling (PAD +1 row), which the binding does not cover - graphs rebuild at the exact count instead\n", __func__);
                } else if (!nkv_st.union_binder.add({
                        llama_union_param_kind::n_outputs, "n_outputs",
                        &llama_context::union_tensors_n_outputs,
                        &llama_context::union_rebind_outputs,
                    })) {
                    // cannot happen for a valid kind; stay fail-closed anyway
                    LLAMA_LOG_ERROR("%s: failed to register the n_outputs union binding; runtime n_outputs patching disabled\n", __func__);
                }
                nkv_st.nout_enabled = nkv_st.union_binder.has(llama_union_param_kind::n_outputs);
            }

            // r8 coverage lane: exact "1"; the coverage audit and the
            // binder_bind/graph_rebuild/refused_view counters are active only
            // with it. The gate alone (both lane gates off) creates no slot
            // and does nothing.
            nkv_st.cov_enabled = getenv("GGML_TP5_NKV_COV") != nullptr && strcmp(getenv("GGML_TP5_NKV_COV"), "1") == 0;

            if (nkv_st.nkv_enabled || nkv_st.nout_enabled) {
                LLAMA_LOG_WARN("%s: union runtime binder enabled (%zu param kind(s) registered: n_kv=%d n_outputs=%d)\n",
                               __func__, nkv_st.union_binder.size(),
                               (int) nkv_st.nkv_enabled, (int) nkv_st.nout_enabled);
            }
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    // The legacy LLAMA_GRAPH_UNION_OUTPUTS_RUNTIME lane was replaced by the
    // n_outputs binding registered above (umbrella gate GGML_TP5_UNION_BINDER
    // only); the legacy env var is no longer read. A stale value in the
    // environment is reported, never silently ignored:
    if (const char * legacy_env = getenv("LLAMA_GRAPH_UNION_OUTPUTS_RUNTIME")) {
        LLAMA_LOG_WARN("%s: LLAMA_GRAPH_UNION_OUTPUTS_RUNTIME=%s ignored: the legacy outputs lane was replaced by the n_outputs union binding (GGML_TP5_UNION_BINDER=1)\n",
                       __func__, legacy_env);
    }
    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx = cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    cparams.n_ctx_kv = cparams.n_ctx_seq;
    if (params.n_ctx_kv != 0) {
        if (!cparams.kv_unified) {
            throw std::runtime_error("n_ctx_kv requires a unified KV cache");
        }
        if (params.n_ctx_kv > uint32_t(INT32_MAX) - 255) {
            throw std::runtime_error("n_ctx_kv exceeds the supported physical KV range");
        }
        if (model.arch == LLM_ARCH_DEEPSEEK4 || llm_arch_is_recurrent(model.arch)) {
            throw std::runtime_error("this model does not support a separate physical KV capacity");
        }
        if (params.kv_stream_arena_mib != 0) {
            throw std::runtime_error("separate physical KV capacity is not supported with a shared CUDA phase arena");
        }
        cparams.n_ctx_kv = GGML_PAD(params.n_ctx_kv, 256);
        if (cparams.n_ctx_kv < cparams.n_ctx_seq) {
            throw std::runtime_error("physical KV capacity must cover one complete logical window");
        }
    }

    LLAMA_LOG_INFO("%s: n_ctx_kv      = %u\n", __func__, cparams.n_ctx_kv);
    LLAMA_LOG_INFO("%s: n_seq_max     = %u\n", __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx         = %u\n", __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq     = %u\n", __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch       = %u\n", __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch      = %u\n", __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn   = %d\n", __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn    = %s\n", __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified    = %s\n", __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base     = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale    = %g\n", __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq      = %u\n", __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max = %u\n", __func__, cparams.n_outputs_max);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                       __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n", __func__,
                       cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            const char * backend_params = params.tp5_relay ?
                (params.quantized_matmul_f32 ? "tp5-relay" : "tp5-relay-native") :
                (params.quantized_matmul_f32 ? "quantized-f32" : nullptr);
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, backend_params);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
            if (params.quantized_matmul_f32) {
                backend_quantized_matmul_f32 = backend;
            }
            if (params.tp5_relay) {
                ggml_relay_status status = {};
                if (!ggml_backend_relay_snapshot(backend, &status) || status.failed || status.ranks != GGML_RELAY_RANKS) {
                    throw std::runtime_error("tp5-relay capability is unavailable on the initialized Meta backend");
                }
                backend_tp5_relay = backend;
            }
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!params.quantized_matmul_f32 && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn =
                    (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                           ggml_backend_buffer_name(buf_output.get()),
                           ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        const uint64_t kv_stream_arena_bytes         = uint64_t(cparams.kv_stream_arena_mib) * 1024ULL * 1024ULL;
        uint64_t       kv_stream_stage_bytes         = kv_stream_arena_bytes;
        uint64_t       kv_stream_minimum_stage_bytes = 0;

        // TurboQuant: this fork's MLA/indexer caches (DSA/DSV4/MSA) and the
        // recurrent memory implementation compress or index the KV store in
        // ways block KV streaming's uniform-page-size pool does not model -
        // same exclusion set as this fork's now-removed PR #326 attempt.
        // DFLASH only needs the exclusion when its DSpark stages hold a
        // single MLA-style K per position (dsv4_hc_mult > 0); the plain
        // multi-head-K variant behaves like an ordinary unified cache.
        // TurboQuant: SWA (iSWA) architectures are no longer blanket-excluded
        // here - llama_kv_cache_iswa/llama_memory_hybrid_iswa now forward the
        // streaming params to their base cache (see llama-kv-cache-iswa.cpp),
        // and llama_context::kv_stream_switch_phase/process_ubatch resolve
        // targets via llama_memory_i::get_kv_stream_targets() rather than a
        // fixed set of dynamic_casts, so an un-wired memory type (e.g.
        // llama_memory_hybrid_idx) fails closed with a clean error instead
        // of the hard crash this exclusion used to prevent.
        //
        // LLM_ARCH_DEEPSEEK4 (llama_kv_cache_dsv4) stays excluded, along with
        // DEEPSEEK32/GLM_DSA (llama_kv_cache_dsa) and MINIMAX_M3
        // (llama_kv_cache_msa). kv_csa has streaming plumbing wired up (see
        // llama-kv-cache-dsv4.cpp), but real-model KL-divergence testing
        // against a real DeepSeek-V4 model found a genuine, large, and
        // arena-size-independent correctness regression (mean KLD 0.07, max
        // 4.3, vs ~0.000000 for every other architecture validated this
        // way) when streaming is enabled. The initial hypothesis - that the
        // shared page-geometry function in fattn.cu mis-sizes the V region
        // for kv_csa's K-only shape, since DSV4's compressed-cache attention
        // (models/deepseek4.cpp, build_attn_mha with k_all passed as both K
        // and V) always needs a real V-shaped region even though the
        // persistent cache doesn't store V - was tested and RULED OUT: the
        // KLD was bit-identical whether or not that geometry path reserved a
        // V region at all. The actual root cause is still unidentified;
        // likely candidates are DSV4's online-compression bookkeeping
        // (state_pos/state_persist_*/state_snapshot_* row addressing in
        // llama_kv_cache_dsv4_context::comp_plan) not surviving the
        // resident-cache/ring-buffer model correctly, but this needs
        // dedicated investigation into that machinery before re-enabling.
        // The same K-only cache shape is used by DSA's kv_mla and by plain
        // MLA, so - although neither has been empirically tested against a
        // real model - both should be treated as similarly unverified until
        // this is root-caused, not assumed safe by extension of Phase 4's
        // (inspection-only, never-tested) conclusion that MLA "just works"
        // via the generalized single-cache path.
        const bool kv_stream_unified_kv_cache = !llm_arch_is_recurrent(model.arch) &&
                                                model.arch != LLM_ARCH_MINIMAX_M3 && model.arch != LLM_ARCH_GLM_DSA &&
                                                model.arch != LLM_ARCH_DEEPSEEK32 && model.arch != LLM_ARCH_DEEPSEEK4 &&
                                                (model.arch != LLM_ARCH_DFLASH || hparams.dsv4_hc_mult == 0) &&
                                                // without SWA, qwen4exp builds llama_memory_hybrid_idx (a per-token
                                                // indexer cache beside the attention one), which does not carry the
                                                // streaming parameters. with SWA it takes the hybrid-iswa path like
                                                // any other hybrid model, which does stream.
                                                (model.arch != LLM_ARCH_QWEN4EXP ||
                                                 hparams.swa_type != LLAMA_SWA_TYPE_NONE);

        const llama_kv_stream_config stream_config = {
            /*.arena_bytes         =*/kv_stream_arena_bytes,
            /*.minimum_arena_bytes =*/kv_stream_arena_bytes == 0 ? uint64_t(0) : uint64_t(1),
            /*.unified_kv_cache    =*/kv_stream_unified_kv_cache,
            /*.context_default     =*/cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT,
            /*.single_sequence     =*/cparams.n_seq_max == 1,
            /*.flash_attention     =*/cparams.flash_attn,
            /*.kv_offload          =*/cparams.offload_kqv,
            /*.swa_full_conflict   =*/params.swa_full && hparams.is_swa_any(),
        };
        const auto stream_validation = llama_kv_stream_config_validate(stream_config);
        if (!stream_validation.valid) {
            throw std::runtime_error(stream_validation.error);
        }
        // TurboQuant: a no_alloc context (e.g. common_get_device_memory_data's
        // no_alloc=true measurement pass, used to size the parent model for
        // MTP) never actually allocates backend buffers, so llama_kv_cache's
        // own streaming-attach loop skips it (see the matching !hparams.no_alloc
        // guard in llama-kv-cache.cpp). Building the phase arena here anyway
        // would leave it with no cache ever attached, and kv_stream_switch_phase
        // would then fail closed with a confusing "requires a memory type with
        // a streamable KV cache" error on every no_alloc measurement pass.
        if (stream_validation.enabled && !hparams.no_alloc) {
            using page_bytes_fn_t = bool (*)(ggml_type, ggml_type, uint32_t, uint32_t, uint32_t, uint32_t, size_t *);
            using arena_new_fn_t  = void * (*) (ggml_backend_dev_t, size_t);
            using arena_free_fn_t = void (*)(void *);
            using arena_set_compute_fn_t = bool (*)(void *, size_t, size_t);
            using arena_buffer_type_fn_t = ggml_backend_buffer_type_t (*)(void *);
            using graph_reset_fn_t       = bool (*)(ggml_backend_t);

            // llama_kv_cache's constructor may itself upgrade K away from the
            // requested type (turbo auto-asymmetric on high-GQA-ratio models -
            // see llama-kv-cache.cpp). Size this bootstrap pre-scan off the
            // same effective type it will actually construct with, or the
            // streaming runtime's internal page-count arithmetic can end up
            // inconsistent with what gets allocated.
            const ggml_type stream_type_k =
                llama_kv_cache_resolve_stream_type_k(model, hparams, params.type_k, params.type_v);

            // TURBO_LAYER_ADAPTIVE (see llama-kv-cache.cpp) can give individual
            // layers a different K/V precision than the rest of the cache -
            // including auto-enabling itself with no env var at all whenever
            // type_v is turbo2 on an >=8-layer model. The streamed page pool
            // has one page size and one buffer type for the whole arena, so a
            // model that would actually get non-uniform per-layer types must
            // be refused here, before ever touching the CUDA runtime, rather
            // than silently packing wrongly-shaped rows into shared pages.
            const int kv_stream_adaptive_mode =
                llama_kv_cache_turbo_layer_adaptive_mode(params.type_v, hparams.n_layer());
            if (kv_stream_adaptive_mode != 0) {
                for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
                    if (!hparams.has_kv(il) || hparams.is_recr(il) || (hparams.is_swa(il) && !params.swa_full)) {
                        continue;
                    }
                    const ggml_type layer_type_k = llama_kv_cache_turbo_layer_adaptive_type_k(
                        kv_stream_adaptive_mode, stream_type_k, params.type_v, il, hparams.n_layer());
                    const ggml_type layer_type_v = llama_kv_cache_turbo_layer_adaptive_type_v(
                        kv_stream_adaptive_mode, stream_type_k, params.type_v, il, hparams.n_layer());
                    if (layer_type_k != stream_type_k || layer_type_v != params.type_v) {
                        throw std::runtime_error(
                            "block KV streaming does not support per-layer mixed KV precision "
                            "(TURBO_LAYER_ADAPTIVE=" +
                            std::to_string(kv_stream_adaptive_mode) +
                            "); the streamed page pool assumes uniform per-layer geometry");
                    }
                }
            }

            ggml_backend_dev_t stream_device       = nullptr;
            size_t             page_bytes          = 0;
            size_t             conversion_bytes    = 0;
            uint32_t           layer_count         = 0;
            uint32_t           stream_head_count_q = 0;
            uint32_t           stream_head_dim_v   = 0;
            for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
                // For iSWA-shaped models (llama_kv_cache_iswa), SWA layers
                // land in kv_swa, not kv_base - and kv_swa only gets a
                // streaming runtime attached when swa_full makes it
                // full-context-length too (see llama-kv-cache-iswa.cpp).
                // Some models (e.g. Gemma-family) have genuinely different
                // per-layer geometry between their SWA and full-attention
                // layers (separate key_length/key_length_swa in the GGUF),
                // so including SWA layers in this scan's uniform-geometry
                // check would wrongly reject models whose actually-streamed
                // layer set (kv_base's non-SWA layers) IS uniform.
                if (!hparams.has_kv(il) || hparams.is_recr(il) || (hparams.is_swa(il) && !params.swa_full)) {
                    continue;
                }
                auto * dev = model.dev_layer(il);
                if (stream_device != nullptr && stream_device != dev) {
                    throw std::runtime_error("block KV streaming requires every attention layer on one CUDA device");
                }
                stream_device = dev;
                ++layer_count;

                auto reg = ggml_backend_dev_backend_reg(dev);
                auto page_bytes_fn =
                    (page_bytes_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_stream_page_bytes");
                auto workspace_bytes_fn = (page_bytes_fn_t) ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_cuda_kv_stream_workspace_bytes");
                if (page_bytes_fn == nullptr || workspace_bytes_fn == nullptr) {
                    throw std::runtime_error("block KV streaming requires the CUDA backend");
                }
                size_t layer_page_bytes       = 0;
                size_t layer_conversion_bytes = 0;
                if (!page_bytes_fn(stream_type_k, params.type_v, hparams.n_embd_head_k(il), hparams.n_embd_head_v(il),
                                   hparams.n_head_kv(il), 256, &layer_page_bytes) ||
                    !workspace_bytes_fn(stream_type_k, params.type_v, hparams.n_embd_head_k(il),
                                        hparams.n_embd_head_v(il), hparams.n_head_kv(il), 256,
                                        &layer_conversion_bytes)) {
                    throw std::runtime_error("invalid block KV streaming page geometry");
                }
                if ((page_bytes != 0 && page_bytes != layer_page_bytes) ||
                    (conversion_bytes != 0 && conversion_bytes != layer_conversion_bytes)) {
                    throw std::runtime_error("block KV streaming requires uniform attention-layer geometry");
                }
                page_bytes          = layer_page_bytes;
                conversion_bytes    = layer_conversion_bytes;
                stream_head_count_q = hparams.n_head(il);
                stream_head_dim_v   = hparams.n_embd_head_v(il);
            }
            if (stream_device == nullptr || page_bytes == 0 || layer_count == 0) {
                throw std::runtime_error("block KV streaming found no attention layers");
            }

            const uint64_t bootstrap_pages = uint64_t(layer_count) + kv_stream_phase_arena.minimum_ring_slots;
            if (bootstrap_pages > std::numeric_limits<uint64_t>::max() / page_bytes ||
                conversion_bytes > std::numeric_limits<uint64_t>::max() - bootstrap_pages * page_bytes) {
                throw std::runtime_error("block KV streaming bootstrap size overflow");
            }
            const uint64_t bootstrap_raw  = conversion_bytes + bootstrap_pages * page_bytes;
            kv_stream_minimum_stage_bytes = (bootstrap_raw + 127ULL) & ~127ULL;
            if (kv_stream_minimum_stage_bytes >= kv_stream_arena_bytes) {
                throw std::runtime_error("block KV streaming arena is too small for bootstrap KV and compute slices");
            }

            // The streamed attention kernel's chunked partial-attention
            // reduction (fattn.cu's ggml_cuda_flash_attn_ext_streamed) draws
            // its scratch buffers from the device's general CUDA memory
            // pool, not from this arena - an arena that fits by itself can
            // still leave too little free VRAM for that pool to grow into
            // during actual inference, which aborts the whole process
            // instead of failing cleanly (see docs/kv-stream.md's "Sizing
            // the arena"). Confirm enough headroom survives reserving the
            // arena for that pool's worst case before ever allocating it.
            // This uses free VRAM as measured right now; other allocations
            // later in context construction (e.g. the compute buffer) are
            // not yet accounted for, so this catches the clearly-too-tight
            // case, not every possible one - the empirical probing in
            // benchmarks/benchmark_kv_stream.py remains the reliable way to
            // size an arena for production.
            {
                using transient_workspace_fn_t = bool (*)(uint32_t, uint32_t, uint32_t, size_t *);
                auto transient_workspace_fn    = (transient_workspace_fn_t) ggml_backend_reg_get_proc_address(
                    ggml_backend_dev_backend_reg(stream_device),
                    "ggml_backend_cuda_kv_stream_transient_workspace_bytes");
                if (transient_workspace_fn == nullptr) {
                    throw std::runtime_error("block KV streaming requires the CUDA backend");
                }
                size_t transient_workspace_bytes = 0;
                if (!transient_workspace_fn(stream_head_count_q, stream_head_dim_v, cparams.n_ubatch,
                                            &transient_workspace_bytes)) {
                    throw std::runtime_error("invalid block KV streaming transient workspace geometry");
                }
                size_t dev_free_bytes  = 0;
                size_t dev_total_bytes = 0;
                ggml_backend_dev_memory(stream_device, &dev_free_bytes, &dev_total_bytes);
                if (kv_stream_arena_bytes > dev_free_bytes ||
                    transient_workspace_bytes > dev_free_bytes - kv_stream_arena_bytes) {
                    throw std::runtime_error(
                        "block KV streaming arena leaves insufficient free VRAM for the streamed "
                        "attention kernel's transient workspace (needs roughly " +
                        std::to_string((transient_workspace_bytes + 1024 * 1024 - 1) / (1024 * 1024)) +
                        " MiB of headroom beyond the arena) - reduce --kv-stream-arena-mib or "
                        "--ubatch-size, or free more device memory");
                }
            }

            auto reg = ggml_backend_dev_backend_reg(stream_device);
            auto arena_new_fn =
                (arena_new_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_phase_arena_new_for_device");
            auto arena_free_fn =
                (arena_free_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_phase_arena_free");
            auto arena_set_compute_fn = (arena_set_compute_fn_t) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_phase_arena_set_compute");
            auto arena_buffer_type_fn = (arena_buffer_type_fn_t) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_phase_arena_buffer_type");
            auto graph_reset_fn =
                (graph_reset_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_graph_reset");
            if (arena_new_fn == nullptr || arena_free_fn == nullptr || arena_set_compute_fn == nullptr ||
                arena_buffer_type_fn == nullptr || graph_reset_fn == nullptr) {
                throw std::runtime_error("block KV streaming phase arena is unavailable");
            }

            kv_stream_phase_arena.arena                  = arena_new_fn(stream_device, kv_stream_arena_bytes);
            kv_stream_phase_arena.free_fn                = arena_free_fn;
            kv_stream_phase_arena.set_compute_fn         = arena_set_compute_fn;
            kv_stream_phase_arena.buffer_type_fn         = arena_buffer_type_fn;
            kv_stream_phase_arena.graph_reset_fn         = graph_reset_fn;
            kv_stream_phase_arena.device                 = stream_device;
            kv_stream_phase_arena.arena_bytes            = kv_stream_arena_bytes;
            kv_stream_phase_arena.page_bytes             = page_bytes;
            kv_stream_phase_arena.conversion_bytes       = conversion_bytes;
            kv_stream_phase_arena.layer_count            = layer_count;
            kv_stream_phase_arena.current_kv_bytes       = kv_stream_minimum_stage_bytes;
            kv_stream_phase_arena.current_compute_offset = kv_stream_minimum_stage_bytes;
            kv_stream_phase_arena.current_compute_bytes  = kv_stream_arena_bytes - kv_stream_minimum_stage_bytes;
            kv_stream_phase_arena.current_ring_slots     = kv_stream_phase_arena.minimum_ring_slots;
            if (kv_stream_phase_arena.arena == nullptr ||
                !arena_set_compute_fn(kv_stream_phase_arena.arena, kv_stream_minimum_stage_bytes,
                                      kv_stream_arena_bytes - kv_stream_minimum_stage_bytes)) {
                throw std::runtime_error("failed to create CUDA block KV streaming phase arena");
            }
            kv_stream_phase_arena.buffer_type = arena_buffer_type_fn(kv_stream_phase_arena.arena);
            if (kv_stream_phase_arena.buffer_type == nullptr) {
                throw std::runtime_error("failed to obtain phase arena CUDA buffer type");
            }
            kv_stream_stage_bytes = kv_stream_minimum_stage_bytes;
            LLAMA_LOG_INFO("%s: experimental block KV streaming enabled, arena = %.2f MiB, bootstrap KV = %.2f MiB\n",
                           __func__, kv_stream_arena_bytes / 1024.0 / 1024.0, kv_stream_stage_bytes / 1024.0 / 1024.0);
        }

        // Amortize relay graph capture over more decode steps without changing
        // physical KV capacity or the causal mask. Other contexts keep pad 1.
        const uint32_t attn_n_pad =
            params.tp5_relay && !cparams.quantized_matmul_f32 &&
            model.arch == LLM_ARCH_QWEN4EXP && cparams.flash_attn && cparams.causal_attn &&
            ((params.type_k == GGML_TYPE_F16 && params.type_v == GGML_TYPE_F16) || tp5_native_q4_kv) &&
            cparams.n_ctx_kv >= 1024 && cparams.n_ctx_kv % 1024 == 0 ? 1024 : 1;

        llama_memory_params params_mem = {
            /*.type_k                =*/params.type_k,
            /*.type_v                =*/params.type_v,
            /*.kv_stream_stage_bytes =*/kv_stream_stage_bytes,
            /*.kv_stream_phase_arena =*/kv_stream_phase_arena.arena,
            /*.kv_stream_maximum_pool_bytes =*/kv_stream_arena_bytes,
            /*.swa_full              =*/params.swa_full,
            /*.ctx_type              =*/cparams.ctx_type,
            /*.mem_other             =*/llama_get_memory(cparams.ctx_other),
            /*.canonical_v           =*/params.tp5_relay,
            /*.attn_n_pad            =*/attn_n_pad,
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft         = ggml_backend_get_default_buffer_type(backend.get());
            auto   backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev       = model.devices[0];
                auto *       host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel = model.n_devices() > 1 && model.n_gpu_layers() > model.hparams.n_layer_all &&
                                 model.split_mode() == LLAMA_SPLIT_MODE_LAYER && cparams.offload_kqv &&
                                 !model.has_tensor_overrides();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto *                 dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        sched_reserve();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

llama_context::~llama_context() {
    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();

    if (!model.hparams.no_alloc) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                                __func__, ggml_backend_buft_name(buft), size_act / (1024.0 * 1024.0),
                                size_exp / (1024.0 * 1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                               __func__, ggml_backend_buft_name(buft), size_act / (1024.0 * 1024.0),
                               size_exp / (1024.0 * 1024.0));
            }
        }
    }
    // one-line summary at context destruction (runbook contract: exact prefix
    // NKV_BINDER_STATS, greppable by the HIP gate harness; printed when the
    // state slot exists and at least one lane gate is exactly "1" - the nkv
    // counters are 0 when only the legacy outputs gate is on)
    // (state note: the counters live in the context's own state slot, not in
    // class data members)
    {
        // peek only: never create state during destruction (with the gate off
        // no slot exists and no stats line prints - stock OFF)
        const llama_nkv_ctx_state * st = union_binder_state_peek();
        if (st != nullptr && (st->nkv_enabled || st->nout_enabled)) {
            if (st->cov_enabled) {
                // r8 coverage lane: the r7 fields verbatim, then the r8
                // acceptance counters (binder_bind / graph_rebuild /
                // refused_view) and the per-category coverage counts
                // (present in the live record / patched by binds)
                LLAMA_LOG_INFO("NKV_BINDER_STATS attempts=%zu applied=%zu refused_width=%zu refused_envelope=%zu refused_skeleton=%zu refused_grow=%zu refused_relay=%zu refused_pipeline=%zu fallbacks=%zu record_builds=%zu binder_bind=%zu graph_rebuild=%zu refused_view=%zu\n",
                               st->c_attempts, st->c_applied, st->c_refused_width, st->c_refused_envelope,
                               st->c_refused_skeleton, st->c_refused_grow, st->c_refused_relay,
                               st->c_refused_pipeline, st->c_fallbacks, st->c_record_builds,
                               st->c_binder_bind, st->c_graph_rebuild, st->c_refused_view);
                LLAMA_LOG_INFO("NKV_BINDER_COV present(mask/view/rope/kvwrite/fa/other)=%u/%u/%u/%u/%u/%u patched(mask/view/rope/kvwrite/fa/other)=%u/%u/%u/%u/%u/%u\n",
                               st->cov_present[LLAMA_NKV_COV_MASK], st->cov_present[LLAMA_NKV_COV_VIEW],
                               st->cov_present[LLAMA_NKV_COV_ROPE], st->cov_present[LLAMA_NKV_COV_KVWRITE],
                               st->cov_present[LLAMA_NKV_COV_FA], st->cov_present[LLAMA_NKV_COV_OTHER],
                               st->c_cov_patched[LLAMA_NKV_COV_MASK], st->c_cov_patched[LLAMA_NKV_COV_VIEW],
                               st->c_cov_patched[LLAMA_NKV_COV_ROPE], st->c_cov_patched[LLAMA_NKV_COV_KVWRITE],
                               st->c_cov_patched[LLAMA_NKV_COV_FA], st->c_cov_patched[LLAMA_NKV_COV_OTHER]);
            } else {
                // r7 line format verbatim (byte-identical without the cov gate)
                LLAMA_LOG_INFO("NKV_BINDER_STATS attempts=%zu applied=%zu refused_width=%zu refused_envelope=%zu refused_skeleton=%zu refused_grow=%zu refused_relay=%zu refused_pipeline=%zu fallbacks=%zu record_builds=%zu\n",
                               st->c_attempts, st->c_applied, st->c_refused_width, st->c_refused_envelope,
                               st->c_refused_skeleton, st->c_refused_grow, st->c_refused_relay,
                               st->c_refused_pipeline, st->c_fallbacks, st->c_record_builds);
            }
    }
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const char * func    = __func__;
    auto         resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
        if (!enabled) {
            return;
        }

        const uint32_t n_tokens_probe = probe.n_tokens_per_seq * n_seqs;

        auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx, true);
        if (!gf) {
            throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
        }

        bool device_mismatch = false;
        for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t     backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
            ggml_backend_dev_t device_fused  = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);

            if (device_fused != device_layer) {
                LLAMA_LOG_WARN(
                    "%s: layer %d is assigned to device %s but %s "
                            "is assigned to device %s (usually due to missing support)\n",
                    func, node.il, device_layer ? ggml_backend_dev_name(device_layer) : "none", probe.name,
                    device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }
        }

        if (device_mismatch) {
            enabled = false;
            LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", func, probe.name);
        } else {
            enabled = true;
            LLAMA_LOG_INFO("%s: %s enabled\n", func, probe.name);
        }
    };

    if (cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", func);
        resolve(llm_fused_op_gdn_ar_probe, cparams.fused_gdn_ar);
        resolve(llm_fused_op_gdn_ch_probe, cparams.fused_gdn_ch);
        cparams.auto_fgdn = false;
    }

    if (cparams.auto_flid) {
        LLAMA_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", func);
        resolve(llm_fused_op_lid_probe, cparams.fused_lid);
        cparams.auto_flid = false;
    }

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", func);
        resolve(llm_fused_op_dsv4_hc_pre_probe, cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        cparams.auto_fhc = false;
    }
}

static bool llama_model_has_cacheable_moe_weights(const llama_model &                 model,
                                                  llama_moe_cache_mode                mode,
                                                  size_t                              budget_mib,
                                                  const std::vector<ggml_backend_t> & backends) {
    // Probe the provider that owns the model's backends; fall back to the
    // thread's active provider (first registered).
    ggml_moe_cache_api api = ggml_moe_cache_active();
    for (ggml_backend_t backend : backends) {
        if (!backend) {
            continue;
        }
        const ggml_moe_cache_api owned =
            ggml_moe_cache_get(ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)));
        if (owned.owner) {
            api = owned;
            break;
        }
    }
    if (mode == LLAMA_MOE_CACHE_MODE_OFF) {
        LLAMA_LOG_INFO("%s: MoE cache disabled (mode=off)\n", __func__);
        return false;
    }
    if (!api.query_config || !api.query_device || !api.query_shape) {
        LLAMA_LOG_INFO("%s: MoE cache disabled (no provider registered)\n", __func__);
        return false;
    }

    ggml_moe_cache_config config    = {};
    const int             automatic = mode == LLAMA_MOE_CACHE_MODE_UNSPECIFIED ? -1 : mode == LLAMA_MOE_CACHE_MODE_AUTO;
    if (!api.query_config(automatic, budget_mib, &config)) {
        LLAMA_LOG_INFO("%s: MoE cache disabled (provider query_config returned false)\n", __func__);
        return false;
    }

    std::vector<int32_t> physical_devices;
    size_t               min_expert_bytes = 0;
    for (ggml_backend_t backend : backends) {
        if (!backend) {
            continue;
        }
        ggml_moe_cache_device_caps caps = {};
        if (!api.query_device(ggml_backend_get_device(backend), &config, &caps) ||
            std::find(physical_devices.begin(), physical_devices.end(), caps.physical_device) !=
                physical_devices.end()) {
            continue;
        }
        physical_devices.push_back(caps.physical_device);
        min_expert_bytes = std::max(min_expert_bytes, caps.min_expert_bytes);
    }
    if ((int) physical_devices.size() < config.min_devices) {
        LLAMA_LOG_INFO("%s: MoE cache disabled (eligible devices=%d, need=%d, min_cc=%d)\n", __func__,
                       (int) physical_devices.size(), config.min_devices, config.min_compute_capability);
        return false;
    }

    size_t largest_expert_bytes = 0;
    for (const auto & entry : model.tensors_by_name) {
        const std::string & name   = entry.first;
        const ggml_tensor * tensor = entry.second;
        if (!tensor || (name.find("_exps") == std::string::npos &&
                        name.find("_chexps") == std::string::npos) ||
            ggml_n_dims(tensor) != 3 || tensor->ne[0] <= 0 ||
            tensor->ne[1] <= 0 || tensor->ne[2] <= 0) {
            continue;
        }
        largest_expert_bytes = std::max(largest_expert_bytes, tensor->nb[2]);
        if (tensor->nb[2] < min_expert_bytes) {
            continue;
        }

        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        if (!buffer || !ggml_backend_buffer_is_host(buffer) ||
            ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            continue;
        }

        ggml_moe_cache_shape_caps shape = {};
        if (api.query_shape(tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->nb[2], &shape)) {
            const size_t slab_bytes = std::max(shape.pool_bytes, config.minimum_slab_bytes);
            if (config.budget_bytes > 0 &&
                (shape.scratch_bytes > config.budget_bytes || slab_bytes > config.budget_bytes - shape.scratch_bytes)) {
                continue;
            }
            return true;
        }
    }
    LLAMA_LOG_INFO("%s: MoE cache disabled (no cacheable expert tensors found; largest expert slab=%zu KiB, minimum=%zu KiB)\n",
            __func__, largest_expert_bytes >> 10, min_expert_bytes >> 10);
    return false;
}

bool llama_context::kv_stream_switch_phase(bool decode, uint32_t active_tokens) {
    auto & arena = kv_stream_phase_arena;
    if (!arena.configured) {
        return true;
    }
    if (sched && arena.decode == decode) {
        return true;
    }
    if (arena.backend_index >= backend_ptrs.size()) {
        LLAMA_LOG_ERROR("%s: invalid phase-arena backend index\n", __func__);
        return false;
    }

    // TurboQuant: memory->get_kv_stream_targets() reports which sub-cache(s)
    // of this memory object (a plain llama_kv_cache, llama_memory_hybrid's
    // attention sub-cache, or - once wired up - one of iSWA/DSA/MSA/DSV4's
    // sub-caches) have a streaming runtime attached. Only one target is
    // ever produced today; a second concurrent target needs the CUDA phase
    // arena's single-lease design generalized first (not yet done - see the
    // multi-cache streaming plan), so more than one here is a hard error
    // rather than silently only switching the first.
    const auto stream_targets = memory->get_kv_stream_targets();
    if (stream_targets.empty()) {
        LLAMA_LOG_ERROR("%s: phase arena requires a memory type with a streamable KV cache\n", __func__);
        return false;
    }
    if (stream_targets.size() > 1) {
        LLAMA_LOG_ERROR("%s: phase arena does not yet support more than one concurrent streaming target\n", __func__);
        return false;
    }
    llama_kv_cache * kv = stream_targets[0].cache;
    if (kv == nullptr) {
        LLAMA_LOG_ERROR("%s: phase arena requires a memory type with a streamable KV cache\n", __func__);
        return false;
    }
    const auto & target = decode ? arena.token_generation : arena.prefill;

    synchronize();
    if (sched && !arena.graph_reset_fn(backend_ptrs[arena.backend_index])) {
        LLAMA_LOG_ERROR("%s: failed to invalidate CUDA graphs before arena repartition\n", __func__);
        return false;
    }

    // Releasing the scheduler returns the only borrowed compute lease.
    sched.reset();
    gf_res_prev->reset();

    const size_t   old_kv_bytes       = arena.current_kv_bytes;
    const size_t   old_compute_offset = arena.current_compute_offset;
    const size_t   old_compute_bytes  = arena.current_compute_bytes;
    const uint32_t old_ring_slots     = arena.current_ring_slots;

    auto resize_kv = [&](size_t bytes, uint32_t ring_slots) {
        return kv->kv_stream_resize_pool(bytes, active_tokens, ring_slots);
    };
    auto restore_compute = [&] {
        return arena.set_compute_fn(arena.arena, old_compute_offset, old_compute_bytes);
    };

    if (target.kv_bytes > old_kv_bytes) {
        // Moving compute upward frees the bytes before KV claims them.
        if (!arena.set_compute_fn(arena.arena, target.compute_offset, target.compute_bytes)) {
            LLAMA_LOG_ERROR("%s: failed to move arena compute slice upward\n", __func__);
            return false;
        }
        if (!resize_kv(target.kv_bytes, target.ring_slots)) {
            (void) restore_compute();
            LLAMA_LOG_ERROR("%s: failed to grow arena KV slice\n", __func__);
            return false;
        }
    } else if (target.kv_bytes < old_kv_bytes) {
        // KV releases the bytes before compute expands downward.
        if (!resize_kv(target.kv_bytes, target.ring_slots)) {
            LLAMA_LOG_ERROR("%s: failed to shrink arena KV slice\n", __func__);
            return false;
        }
        if (!arena.set_compute_fn(arena.arena, target.compute_offset, target.compute_bytes)) {
            (void) resize_kv(old_kv_bytes, old_ring_slots);
            LLAMA_LOG_ERROR("%s: failed to move arena compute slice downward\n", __func__);
            return false;
        }
    } else {
        if (!arena.set_compute_fn(arena.arena, target.compute_offset, target.compute_bytes)) {
            LLAMA_LOG_ERROR("%s: failed to update arena compute slice\n", __func__);
            return false;
        }
        if (!resize_kv(target.kv_bytes, target.ring_slots)) {
            (void) restore_compute();
            LLAMA_LOG_ERROR("%s: failed to update arena KV layout\n", __func__);
            return false;
        }
    }

    arena.current_kv_bytes       = target.kv_bytes;
    arena.current_compute_offset = target.compute_offset;
    arena.current_compute_bytes  = target.compute_bytes;
    arena.current_ring_slots     = target.ring_slots;

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), arena.max_nodes,
                                       cparams.pipeline_parallel, cparams.op_offload));
    if (!sched) {
        LLAMA_LOG_ERROR("%s: failed to recreate scheduler\n", __func__);
        return false;
    }

    auto reserve_mctx = memory->init_full();
    if (!reserve_mctx) {
        LLAMA_LOG_ERROR("%s: failed to initialize full memory for phase reserve\n", __func__);
        return false;
    }

    const uint32_t n_seqs    = n_seq_max_active();
    const uint32_t n_tokens  = decode ? n_seqs : std::min(cparams.n_ctx, cparams.n_ubatch);
    const uint32_t n_outputs = decode ? n_seqs : std::min(n_tokens, cparams.n_outputs_max);
    auto *         gf        = graph_reserve(n_tokens, n_seqs, n_outputs, reserve_mctx.get());
    if (gf == nullptr) {
        LLAMA_LOG_ERROR("%s: failed to reserve %s compute graph in phase arena\n", __func__,
                        decode ? "decode" : "prefill");
        return false;
    }

    backend_buf_exp_size = target.backend_sizes;
    arena.decode         = decode;
    LLAMA_LOG_INFO("%s: activated %s phase: KV %.2f MiB, compute %.2f MiB, active tokens %u\n", __func__,
                   decode ? "decode" : "prefill", target.kv_bytes / 1024.0 / 1024.0,
                   target.compute_bytes / 1024.0 / 1024.0, active_tokens);
    return true;
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();
    if (kv_stream_phase_arena.configured && sched && kv_stream_phase_arena.backend_index < backend_ptrs.size()) {
        if (!kv_stream_phase_arena.graph_reset_fn(backend_ptrs[kv_stream_phase_arena.backend_index])) {
            throw std::runtime_error("failed to reset CUDA graphs before measuring phase arena");
        }
        sched.reset();
    }

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs   = n_seq_max_active();
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    gf_res_prev.reset(new llm_graph_result(max_nodes));
    // the new long-lived result object is owned by this context: the union
    // binder state uses this back-pointer (llama-nout-state.cpp) instead of a
    // global result -> context reverse map
    gf_res_prev->set_owner(this);
    gf_res_reserve.reset(new llm_graph_result(max_nodes));
    gf_res_reserve->set_owner(this); // same owner invariant for the reserve result

    const bool moe_cache_eligible = llama_model_has_cacheable_moe_weights(
        model, (llama_moe_cache_mode) cparams.moe_cache_mode, cparams.moe_cache_budget_mib, backend_ptrs);
    const ggml_moe_cache_mode moe_cache_mode =
        moe_cache_eligible ? (ggml_moe_cache_mode) cparams.moe_cache_mode : GGML_MOE_CACHE_MODE_OFF;
    const char * moe_cache_requested = "provider";
    switch (cparams.moe_cache_mode) {
        case LLAMA_MOE_CACHE_MODE_OFF:
            moe_cache_requested = "off";
            break;
        case LLAMA_MOE_CACHE_MODE_AUTO:
            moe_cache_requested = "auto";
            break;
        case LLAMA_MOE_CACHE_MODE_ON:
            moe_cache_requested = "on";
            break;
        case LLAMA_MOE_CACHE_MODE_UNSPECIFIED:
            break;
    }
    LLAMA_LOG_INFO("%s: MoE cache requested=%s resolved=%s\n", __func__, moe_cache_requested,
                   moe_cache_eligible ? moe_cache_requested : "off");

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes,
                                       cparams.pipeline_parallel, cparams.op_offload));
    ggml_backend_sched_set_moe_cache(sched.get(), moe_cache_mode, cparams.moe_cache_budget_mib);

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs,
                    n_outputs);

    resolve_fused_ops(mctx.get(), n_seqs);

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", __func__);
        auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
            if (!enabled) {
                return;
            }

            const uint32_t n_tokens_probe = probe.n_tokens_per_seq * n_seqs;

            auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx.get(), true);
            if (!gf) {
                throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
            }

            bool device_mismatch = false;
            for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
                if (node.op != probe.op) {
                    continue;
                }

                GGML_ASSERT(node.il >= 0);

                ggml_backend_t     backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
                ggml_backend_dev_t device_fused  = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

                ggml_backend_dev_t device_layer = model.dev_layer(node.il);

                if (device_fused != device_layer) {
                    LLAMA_LOG_WARN(
                        "%s: layer %d is assigned to device %s but %s "
                        "is assigned to device %s (usually due to missing support)\n",
                        __func__, node.il, device_layer ? ggml_backend_dev_name(device_layer) : "none", probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                    device_mismatch = true;
                    break;
                }
            }

            if (device_mismatch) {
                enabled = false;
                LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", __func__, probe.name);
            } else {
                enabled = true;
                LLAMA_LOG_INFO("%s: %s enabled\n", __func__, probe.name);
            }
        };

        resolve(llm_fused_op_dsv4_hc_pre_probe, cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        cparams.auto_fhc = false;
    }

    if (kv_stream_phase_arena.arena != nullptr) {
        size_t backend_index = SIZE_MAX;
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            if (ggml_backend_get_device(backend_ptrs[i]) == kv_stream_phase_arena.device) {
                backend_index = i;
                break;
            }
        }
        if (backend_index == SIZE_MAX) {
            throw std::runtime_error("failed to match phase arena to its CUDA scheduler backend");
        }

        const uint32_t      n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);
        std::vector<size_t> sizes_pp(backend_ptrs.size(), 0);
        std::vector<size_t> sizes_tg(backend_ptrs.size(), 0);

        auto * gf_pp = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), true, sizes_pp.data());
        if (gf_pp == nullptr) {
            throw std::runtime_error("failed to measure compute pp buffers");
        }
        const int n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
        const int n_nodes_pp  = ggml_graph_n_nodes(gf_pp);

        auto * gf_tg = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), true, sizes_tg.data());
        if (gf_tg == nullptr) {
            throw std::runtime_error("failed to measure compute tg buffers");
        }
        const int n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
        const int n_nodes_tg  = ggml_graph_n_nodes(gf_tg);

        const uint64_t compute_alignment = ggml_backend_buft_get_alignment(backend_buft[backend_index]);
        const auto     plan_pp           = llama_kv_stream_phase_plan_make({
            /*.arena_bytes        =*/kv_stream_phase_arena.arena_bytes,
            /*.compute_bytes      =*/sizes_pp[backend_index],
            /*.compute_alignment  =*/compute_alignment,
            /*.page_bytes         =*/kv_stream_phase_arena.page_bytes,
            /*.conversion_bytes   =*/kv_stream_phase_arena.conversion_bytes,
            /*.layer_count        =*/kv_stream_phase_arena.layer_count,
            /*.minimum_ring_pages =*/kv_stream_phase_arena.minimum_ring_slots,
        });
        const auto     plan_tg           = llama_kv_stream_phase_plan_make({
            /*.arena_bytes        =*/kv_stream_phase_arena.arena_bytes,
            /*.compute_bytes      =*/sizes_tg[backend_index],
            /*.compute_alignment  =*/compute_alignment,
            /*.page_bytes         =*/kv_stream_phase_arena.page_bytes,
            /*.conversion_bytes   =*/kv_stream_phase_arena.conversion_bytes,
            /*.layer_count        =*/kv_stream_phase_arena.layer_count,
            /*.minimum_ring_pages =*/kv_stream_phase_arena.minimum_ring_slots,
        });
        if (!plan_pp.valid) {
            throw std::runtime_error(std::string("prefill phase does not fit shared CUDA arena: ") + plan_pp.error);
        }
        if (!plan_tg.valid) {
            throw std::runtime_error(std::string("token-generation phase does not fit shared CUDA arena: ") +
                                     plan_tg.error);
        }

        auto make_layout = [](const llama_kv_stream_phase_plan & plan, std::vector<size_t> sizes) {
            kv_stream_phase_arena_owner::layout result;
            result.kv_bytes                 = plan.kv_bytes;
            result.compute_offset           = plan.compute_offset;
            result.compute_bytes            = plan.compute_bytes;
            result.ring_slots               = plan.ring_pages;
            result.resident_pages_per_layer = plan.resident_pages_per_layer;
            result.backend_sizes            = std::move(sizes);
            return result;
        };

        kv_stream_phase_arena.prefill          = make_layout(plan_pp, std::move(sizes_pp));
        kv_stream_phase_arena.token_generation = make_layout(plan_tg, std::move(sizes_tg));
        kv_stream_phase_arena.backend_index    = backend_index;
        kv_stream_phase_arena.max_nodes        = max_nodes;
        kv_stream_phase_arena.configured       = true;

        sched.reset();
        backend_buft[backend_index] = kv_stream_phase_arena.buffer_type;
        if (!kv_stream_switch_phase(false, 0)) {
            throw std::runtime_error("failed to activate shared CUDA arena prefill phase");
        }

        LLAMA_LOG_INFO(
            "%s: phase arena prefill: KV %.2f MiB, compute %.2f MiB, resident %u pages/layer, ring %u pages\n",
            __func__, kv_stream_phase_arena.prefill.kv_bytes / 1024.0 / 1024.0,
            kv_stream_phase_arena.prefill.compute_bytes / 1024.0 / 1024.0,
            kv_stream_phase_arena.prefill.resident_pages_per_layer, kv_stream_phase_arena.prefill.ring_slots);
        LLAMA_LOG_INFO(
            "%s: phase arena decode:  KV %.2f MiB, compute %.2f MiB, resident %u pages/layer, ring %u pages\n",
            __func__, kv_stream_phase_arena.token_generation.kv_bytes / 1024.0 / 1024.0,
            kv_stream_phase_arena.token_generation.compute_bytes / 1024.0 / 1024.0,
            kv_stream_phase_arena.token_generation.resident_pages_per_layer,
            kv_stream_phase_arena.token_generation.ring_slots);

        if (n_nodes_pp == n_nodes_tg) {
            LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
        } else {
            LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens,
                           n_nodes_tg);
        }
        if (n_splits_pp == n_splits_tg) {
            LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
        } else {
            LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens,
                           n_splits_tg);
        }

        const int64_t t_end_us = ggml_time_us();
        LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n", __func__, (t_end_us - t_start_us) / 1000.0,
                       ggml_backend_sched_get_n_copies(sched.get()));
        return;
    }

    // reserve worst-case graph
    int n_splits_pp = -1;
    int n_nodes_pp  = -1;

    int n_splits_tg = -1;
    int n_nodes_tg  = -1;

    const uint32_t n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc,
                                  model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n",
                               __func__);
                cparams.pipeline_parallel = false;
                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                                                   max_nodes, false, cparams.op_offload));
                ggml_backend_sched_set_moe_cache(sched.get(), moe_cache_mode, cparams.moe_cache_budget_mib);
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp  = ggml_graph_n_nodes(gf);
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg  = ggml_graph_n_nodes(gf);
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: not sure if the following graph would be worst case for multi-stream KV caches:
        //
        // auto * gf = graph_reserve(n_tokens, 1, n_tokens, mctx.get());
        //
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__, ggml_backend_buft_name(buft),
                           backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    if (n_nodes_pp == n_nodes_tg) {
        LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens,
                       n_nodes_tg);
    }

    if (n_splits_pp == n_splits_tg) {
        LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens,
                       n_splits_tg);
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n", __func__, (t_end_us - t_start_us) / 1000.0,
                   ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us          = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens    = 0;
    t_compute_start_us = 0;
}

bool llama_context::release_compute() {
    // A phase arena owns a joint KV/compute layout; releasing its allocator
    // without the arena protocol would invalidate that layout. Optimizer
    // graphs also have a separate lifetime. Reject both before any mutation.
    if (!sched || model.hparams.no_alloc || opt_ctx != nullptr ||
            kv_stream_phase_arena.arena != nullptr || kv_stream_phase_arena.configured) {
        return false;
    }

    synchronize();
    // These own the only inference graph metadata that may reuse allocator
    // addresses. Reset both before the allocator is freed, retaining their
    // graph-result objects for the normal next graph build.
    if (gf_res_prev) {
        gf_res_prev->reset();
    }
    if (gf_res_reserve) {
        gf_res_reserve->reset();
    }
    ggml_backend_sched_release_buffers(sched.get());
    return true;
}

uint64_t llama_context::compute_bytes() const {
    if (!sched) {
        return 0;
    }
    uint64_t total = 0;
    const int n_backends = ggml_backend_sched_get_n_backends(sched.get());
    for (int i = 0; i < n_backends; ++i) {
        const auto backend = ggml_backend_sched_get_backend(sched.get(), i);
        const auto buft = ggml_backend_sched_get_buffer_type(sched.get(), backend);
        bool already_counted = false;
        for (int j = 0; j < i; ++j) {
            const auto previous = ggml_backend_sched_get_backend(sched.get(), j);
            if (ggml_backend_sched_get_buffer_type(sched.get(), previous) == buft) {
                already_counted = true;
                break;
            }
        }
        if (!already_counted) {
            total += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return total;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_ctx_kv() const {
    return cparams.n_ctx_kv;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_seq_max_active() const {
    return llm_arch_is_recurrent(model.arch) || llm_arch_is_hybrid(model.arch) ?
        cparams.n_rs_slots : cparams.n_seq_max;
}

llama_kv_cache * llama_context::attention_view_cache() const {
    if (auto * cache = dynamic_cast<llama_kv_cache *>(memory.get())) {
        return cache;
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get())) {
        return hybrid->get_mem_attn();
    }
    return nullptr;
}

const char * llama_context::attention_views_unsupported_reason() const {
    if (cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT) {
        return "reader views require the default context role";
    }
    if (!cparams.causal_attn) {
        return "reader views require causal attention";
    }
    if (cparams.embeddings) {
        return "reader views are unavailable while embedding output is enabled";
    }
    if (!cparams.kv_unified) {
        return "reader views require a unified KV cache";
    }
    if (cparams.kv_stream_arena_mib != 0) {
        return "reader views do not support the shared KV streaming arena";
    }
    if (cparams.flash_attn) {
        return "Flash Attention must be disabled for reader-view attention";
    }
    if (model.hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        return "reader views do not support sliding-window attention";
    }
    if (model.hparams.f_max_alibi_bias != 0) {
        return "reader views do not support ALiBi attention bias";
    }
    if (model.hparams.n_pos_per_embd() != 1) {
        return "reader views require one text RoPE position per token";
    }
    if (model.hparams.use_kq_norm) {
        return "reader views do not support post-RoPE K/Q normalization";
    }
    const auto * cache = attention_view_cache();
    if (!cache) {
        return "context has no supported resident attention cache";
    }
    if (cache->type_k() != GGML_TYPE_F32) {
        return "reader-view K cache must use GGML_TYPE_F32";
    }
    if (cache->type_v() != GGML_TYPE_F32) {
        return "reader-view V cache must use GGML_TYPE_F32";
    }
    switch (model.arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_LFM2:
        case LLM_ARCH_LFM2MOE:
            return nullptr;
        default:
            return "model architecture is not supported by reader-view attention";
    }
}

bool llama_context::attention_views_supported() const {
    return attention_views_unsupported_reason() == nullptr;
}

bool llama_context::set_attention_views(const llama_attention_view * views, size_t count) {
    if (!attention_views_supported() || !views || count == 0 || count > cparams.n_batch ||
        count > n_seq_max_active()) {
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        if (views[i].query_seq < 0 || uint32_t(views[i].query_seq) >= cparams.n_seq_max) {
            return false;
        }
    }
    auto * cache = attention_view_cache();
    std::vector<llama_attention_view_owned> owned;
    if (!cache->prepare_attention_views(views, count, owned)) {
        return false;
    }
    uint64_t extra = 0;
    for (const auto & view : owned) {
        int64_t previous = -1;
        extra += 24;
        for (const auto & entry : view.entries) {
            const int64_t effective = int64_t(view.query_virtual_pos) + entry.key_pos - entry.virtual_pos;
            if (effective != previous) {
                extra += 16;
                previous = effective;
            }
        }
    }
    const uint64_t budget = extra * model.hparams.n_layer() + 64ull * model.n_tensors() +
                            160ull * std::min(cparams.n_ctx, cparams.n_ubatch);
    if (budget > UINT32_MAX) {
        return false;
    }
    synchronize();
    cache->attention_views = std::move(owned);
    if (budget > attention_node_budget) {
        attention_node_budget = uint32_t(budget);
        sched_need_reserve = true;
    }
    return true;
}

void llama_context::clear_attention_views() {
    if (auto * cache = attention_view_cache()) {
        synchronize();
        cache->attention_views.clear();
    }
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

// Resolve the effective K/V cache type of a memory object.
//
// llama_memory_i has many concrete implementations (plain KV cache, iSWA,
// DSA, DSV4, MSA, recurrent, hybrid, hybrid-iSWA, ...) and none of them
// carry a common "type_k()/type_v()" accessor on the base interface, so we
// have to downcast to figure out which one we're holding. This mirrors how
// llama-model.cpp::create_memory() picks the concrete type in the first
// place - see the switch over `arch` there.
//
// `want_k` selects K (true) or V (false). Returns GGML_TYPE_COUNT when the
// memory object has no single representative K/V type (pure recurrent
// memory, or a composite cache like DSV4 whose sub-caches - raw token
// attention, CSA/HCA compressed blocks, lightning-indexer keys - are
// configured independently and need not share a single type. Note that
// auto-asymmetric is not the reason here: it skips MLA architectures, DSV4
// included, so it never rewrites these.
static enum ggml_type llama_memory_get_kv_type(const llama_memory_i * mem, bool want_k) {
    if (!mem) {
        return GGML_TYPE_COUNT;
    }

    // plain unified/non-SWA KV cache
    if (const auto * kv = dynamic_cast<const llama_kv_cache *>(mem)) {
        return want_k ? kv->type_k() : kv->type_v();
    }

    // interleaved-SWA cache: base and swa sub-caches are built with the same
    // requested type_k/type_v and the same hparams-derived GQA ratio, so any
    // auto-asymmetric rewrite is consistent between them - report the base.
    if (const auto * kv = dynamic_cast<const llama_kv_cache_iswa *>(mem)) {
        const llama_kv_cache * base = kv->get_base();
        return base ? (want_k ? base->type_k() : base->type_v()) : GGML_TYPE_COUNT;
    }

    // DeepSeek sparse-attention (DSA): the MLA cache is the real K/V cache
    // (MLA models are always symmetric and skip auto-asymmetric); the
    // lightning-indexer cache is a separate, unrelated key-only cache.
    if (const auto * kv = dynamic_cast<const llama_kv_cache_dsa *>(mem)) {
        const llama_kv_cache * mla = kv->get_mla();
        return mla ? (want_k ? mla->type_k() : mla->type_v()) : GGML_TYPE_COUNT;
    }

    // MiniMax-style sparse attention (MSA): report the base attention cache,
    // not the separate indexer key cache.
    if (const auto * kv = dynamic_cast<const llama_kv_cache_msa *>(mem)) {
        const llama_kv_cache * base = kv->get_base();
        return base ? (want_k ? base->type_k() : base->type_v()) : GGML_TYPE_COUNT;
    }

    // DSV4: raw token attention (iSWA), plus CSA/HCA compressed block caches
    // and a lightning-indexer key cache. These are independent llama_kv_cache
    // instances configured independently, so there is no single type that
    // represents "the" cache - be honest about it rather than picking one.
    if (dynamic_cast<const llama_kv_cache_dsv4 *>(mem)) {
        return GGML_TYPE_COUNT;
    }

    // hybrid (recurrent + attention): report the attention sub-cache; the
    // recurrent sub-cache has no K/V concept (type_r/type_s instead).
    if (const auto * hy = dynamic_cast<const llama_memory_hybrid *>(mem)) {
        const llama_kv_cache * attn = hy->get_mem_attn();
        return attn ? (want_k ? attn->type_k() : attn->type_v()) : GGML_TYPE_COUNT;
    }

    if (const auto * hy = dynamic_cast<const llama_memory_hybrid_iswa *>(mem)) {
        const llama_kv_cache_iswa * attn = hy->get_mem_attn();
        const llama_kv_cache *      base = attn ? attn->get_base() : nullptr;
        return base ? (want_k ? base->type_k() : base->type_v()) : GGML_TYPE_COUNT;
    }

    // pure recurrent memory (Mamba/RWKV-style): no K/V cache at all.
    if (dynamic_cast<const llama_memory_recurrent *>(mem)) {
        return GGML_TYPE_COUNT;
    }

    // unknown memory implementation - don't guess.
    return GGML_TYPE_COUNT;
}

enum ggml_type llama_context::get_kv_type_k() const {
    return llama_memory_get_kv_type(memory.get(), /* want_k */ true);
}

enum ggml_type llama_context::get_kv_type_v() const {
    return llama_memory_get_kv_type(memory.get(), /* want_k */ false);
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                }
                break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        if (backend_tp5_relay != nullptr) {
            // This update can discard retained input storage before process_ubatch.
            // NO_UPDATE returns above without adding a fence to the hot path.
            ggml_backend_sched_synchronize(sched.get());
        }

        // reset the previous graph result to make sure that it won't be reused
        // TODO: change the mctx->apply() to return information if a graph reserve is needed
        //       reset the graph result only if the memory module did reset the scheduler
        gf_res_prev->reset();

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs   = n_seq_max_active();
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j * model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens() const {
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t  j          = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j * n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t) (i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j * n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row * model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row * model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() && (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row * model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

void llama_context::attach_threadpool(ggml_threadpool_t threadpool, ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_decode_phase(enum llama_decode_phase phase) {
    switch (phase) {
        case LLAMA_DECODE_PHASE_AUTOMATIC:
        case LLAMA_DECODE_PHASE_PROMPT:
        case LLAMA_DECODE_PHASE_GENERATION:
            decode_phase = phase;
            return;
    }
    decode_phase = LLAMA_DECODE_PHASE_AUTOMATIC;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // TODO: not sure yet if we want to reserve here
    //sched_need_reserve = true;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

void llama_context::set_decision_order(const int32_t * order, int32_t n_tokens) {
    if (order == nullptr || n_tokens <= 0) {
        decision_order_next.clear();
        return;
    }
    decision_order_next.assign(order, order + n_tokens);
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_mtp_chain(bool value) {
    cparams.mtp_chain = value;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    sched_need_reserve = true;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler && sampler->iface->backend_init && sampler->iface->backend_apply && llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__,
                       llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i++) {
        if (scales[i] != 0.0f) {
            loras->insert({ adapters[i], scales[i] });
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(const float * data, size_t len, int32_t n_embd, int32_t il_start, int32_t il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

// ---- per-context union-binder state slot (llama-nout-state.h) ----

llama_nkv_ctx_state & llama_context::union_binder_state_get() {
    if (union_binder_state == nullptr) {
        // one slot per context, owned by the context (no process-global map);
        // only the constructor's gate blocks create it - OFF allocates nothing
        union_binder_state.reset(new llama_nkv_ctx_state);
    }
    return *union_binder_state;
}

const llama_nkv_ctx_state * llama_context::union_binder_state_peek() const {
    return union_binder_state.get();
}

// union-graph runtime parameter binder: n_kv lane.
//
// 1. when the live graph is (re)built, process_ubatch builds it at the KV
//    cache's CURRENT true n_kv (no envelope, no headroom, no bucket sizing on
//    this lane's side) and records the graph's pre-allocation skeleton and
//    per-tensor byte extents in the context's union-binder state slot; the
//    scheduler allocates exactly that extent;
// 2. when the KV cache later moves (can_reuse fails only on the stale
//    n_kv-coupled shapes), union_rebind_n_kv builds a scratch graph at the
//    cache's CURRENT n_kv and requires an exact skeleton match against the
//    recorded live graph;
// 3. a move to a smaller n_kv fits inside the recorded extents (the
//    nkv-coupled shapes shrink with n_kv) and is patched in place; a move
//    past the built width has no allocated backing and is refused - the
//    caller full-rebuilds at the cache's exact current n_kv and the fresh
//    build re-records. The binder never pads or rounds: it binds exactly the
//    value get_n_kv() reports, so any bucket alignment still present is the
//    stock cache-level pad (llama_kv_cache::get_n_kv), unchanged for OFF and
//    ON alike;
// 4. the changed fields (ne/nb/op_params/view_offs + view data pointers) are
//    copied into the live graph in place, then set_inputs refills the KQ mask
//    at the live width (cells beyond the applied n_kv are empty -> -INF
//    masked).
// Any mismatch -> false and the caller falls through to the normal full
// rebuild at the cache's exact current n_kv.


namespace {

struct llama_nkv_record_t {
    // DFS post-order: nodes and, before each, its not-yet-seen sources / view source
    std::vector<ggml_tensor *> t;
    std::vector<int32_t>       src;   // (GGML_MAX_SRC + 1) entries per tensor, view source last
    std::vector<uint8_t>       fixed; // pre-existing storage (weights, KV cache)
    std::vector<size_t>        bytes; // byte envelope of each graph-owned tensor

    void record(ggml_cgraph * gf) {
        t.clear();
        src.clear();
        fixed.clear();
        bytes.clear();

        std::unordered_map<const ggml_tensor *, int32_t> idx;
        std::unordered_map<const ggml_tensor *, uint8_t> seen;
        idx.reserve(4 * (size_t) ggml_graph_n_nodes(gf) + 16);
        seen.reserve(4 * (size_t) ggml_graph_n_nodes(gf) + 16);

        std::vector<std::pair<ggml_tensor *, int>> stack;

        auto add = [&](ggml_tensor * root) {
            if (root == nullptr || seen.count(root)) {
                return;
            }
            stack.clear();
            stack.push_back({ root, 0 });
            seen[root] = 1;
            while (!stack.empty()) {
                ggml_tensor * x = stack.back().first;
                // fixed = pre-existing storage (weights, KV cache); views created by
                // this build are graph-owned
                const bool is_fixed = x->view_src == nullptr && (x->data != nullptr || x->buffer != nullptr);

                bool pushed = false;
                while (!is_fixed && stack.back().second <= GGML_MAX_SRC) {
                    const int k = stack.back().second++;
                    ggml_tensor * s = k < GGML_MAX_SRC ? x->src[k] : x->view_src;
                    if (s != nullptr && !seen.count(s)) {
                        seen[s] = 1;
                        stack.push_back({ s, 0 });
                        pushed = true;
                        break;
                    }
                }
                if (pushed) {
                    continue;
                }
                stack.pop_back();

                const int32_t id = (int32_t) t.size();
                idx[x] = id;
                t.push_back(x);
                fixed.push_back(is_fixed ? 1 : 0);

                for (int j = 0; j <= GGML_MAX_SRC; ++j) {
                    ggml_tensor * s = is_fixed ? nullptr : (j < GGML_MAX_SRC ? x->src[j] : x->view_src);
                    int32_t v = -1;
                    if (s != nullptr) {
                        auto it = idx.find(s);
                        v = it != idx.end() ? it->second : -2; // -2: back-edge (on DFS stack)
                    }
                    src.push_back(v);
                }
            }
        };

        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            add(ggml_graph_node(gf, i));
        }
    }

    void record_bytes() {
        bytes.assign(t.size(), 0);
        for (size_t k = 0; k < t.size(); ++k) {
            if (!fixed[k] && t[k]->view_src == nullptr) {
                bytes[k] = ggml_nbytes(t[k]);
            }
        }
    }
};
} // namespace

// record the pre-allocation skeleton + byte extents of a freshly built (not
// yet sched-placed) graph into the context's union-binder state slot; ONE
// record serves both lanes (nkv + n_outputs) of the latest fresh build
// (state note: was llama_context::nkv_record_* data members; now lives in the
// context's own state slot)
void llama_context::binder_record_snapshot(ggml_cgraph * gf, uint32_t nkv_built) {
    llama_nkv_record_t rec;
    rec.record(gf);
    rec.record_bytes();

    auto & st = llama_nkv_ctx_state_get(this);
    st.record_t     = std::move(rec.t);
    st.record_src   = std::move(rec.src);
    st.record_fixed = std::move(rec.fixed);
    st.record_bytes = std::move(rec.bytes);

    // r8 coverage lane: classify the recorded nkv-coupled tensor families
    // (structural: op / flags / name / first dim vs the built width; see
    // llama_nkv_cov_classify in llama-nout-state.h). ACCOUNTING ONLY: the
    // counts describe which nkv-coupled families the LIVE record contains;
    // they are acceptance bookkeeping, not per-op correctness or coverage
    // evidence. Reset per record: cov_present always describes the CURRENT
    // live record, never an accumulation. OFF (cov gate not exactly "1"):
    // no walk - the r7 record is untouched.
    if (st.cov_enabled) {
        st.cov_nkv_built = nkv_built;   // the classifier's width reference for this record
        memset(st.cov_present, 0, sizeof(st.cov_present));
        for (size_t k = 0; k < st.record_t.size(); ++k) {
            const ggml_tensor * t = st.record_t[k];
            const int32_t s0 = st.record_src[k * (GGML_MAX_SRC + 1) + 0];
            const int32_t vs = st.record_src[k * (GGML_MAX_SRC + 1) + GGML_MAX_SRC];
            ++st.cov_present[llama_nkv_cov_classify((int32_t) t->op, t->name,
                                                    (t->flags & GGML_TENSOR_FLAG_INPUT) != 0,
                                                    st.record_fixed[k] != 0,
                                                    s0 >= 0 && st.record_fixed[s0] != 0,
                                                    vs >= 0 && st.record_fixed[vs] != 0,
                                                    t->ne[0], nkv_built)];
        }
    }
}

// table-driven dispatch: kind -> tensor set provider -> rebind operation
bool llama_union_binder::rebind(llama_union_param_kind kind,
                                llama_context * ctx,
                                llm_graph_result * res,
                                const llm_graph_params & params) const {
    const llama_union_binding * binding = get(kind);
    if (binding == nullptr || binding->tensor_set == nullptr || binding->rebind == nullptr) {
        return false;
    }
    const llama_union_tensor_set set = (ctx->*(binding->tensor_set))();
    return (ctx->*(binding->rebind))(res, params, set);
}

// n_kv binding: the recorded skeleton + byte extents of the latest fresh build
llama_union_tensor_set llama_context::union_tensors_n_kv() {
    auto & st = llama_nkv_ctx_state_get(this);
    return { &st.record_t, &st.record_src, &st.record_fixed, &st.record_bytes };
}

// reader-audit: does this graph result carry attention-view reader inputs?
// The plain attn_kv input (and the msa derivative sharing the class) holds
// build-time captured spans; the hybrid wrapper embeds one. Checked against
// the LIVE graph's captured input state so a graph built under attention
// views stays refused even after the views are cleared from the cache. All
// other input variants carry no reader state (verified: no other input class
// has a readers member, and the msa/hybrid builders never populate one).
static bool llama_graph_result_has_attn_readers(const llm_graph_result * res) {
    if (res == nullptr) {
        return false;
    }
    for (const auto & input : res->inputs) {
        if (const auto * kv = dynamic_cast<const llm_graph_input_attn_kv *>(input.get())) {
            if (!kv->readers.empty()) {
                return true;
            }
        } else if (const auto * hy = dynamic_cast<const llm_graph_input_mem_hybrid *>(input.get())) {
            if (hy->inp_attn != nullptr && !hy->inp_attn->readers.empty()) {
                return true;
            }
        }
    }
    return false;
}

bool llama_context::union_rebind_n_kv(llm_graph_result * res, const llm_graph_params & params, const llama_union_tensor_set & set) {
    // the record vectors come from the binding's tensor set provider; alias them
    // once (all later reads/writes go through these names)
    const std::vector<ggml_tensor *> & record_t     = *set.tensors;
    const std::vector<int32_t>       & record_src   = *set.src;
    const std::vector<uint8_t>       & record_fixed = *set.fixed;
    const std::vector<size_t>        & record_bytes = *set.bytes;

    auto & st = llama_nkv_ctx_state_get(this);

    // NKV_BINDER_STATS: every binder invocation counts as an attempt
    ++st.c_attempts;

    if (!res || res->get_gf() == nullptr) {
        ++st.c_fallbacks;
        return false;
    }
    // the binder needs the pre-allocation record of the latest fresh build
    if (!st.record_valid || record_t.empty()) {
        ++st.c_fallbacks;
        return false;
    }
    // tp5-relay pins host graph buffers and validates every node's scheduler
    // placement after the rebuild; in-place n_kv patching is out of its contract
    if (backend_tp5_relay != nullptr || cparams.pipeline_parallel) {
        if (backend_tp5_relay != nullptr) {
            ++st.c_refused_relay;
        }
        if (cparams.pipeline_parallel) {
            ++st.c_refused_pipeline;
        }
        ++st.c_fallbacks;
        return false;
    }

    // collect every KV-cache context whose n_kv shapes this graph
    const llama_kv_cache_context * kvs[2] = { nullptr, nullptr };
    int nkvs = 0;
    if (const auto * h = dynamic_cast<const llama_memory_hybrid_context *>(params.mctx)) {
        kvs[nkvs++] = h->get_attn();
    } else if (const auto * k = dynamic_cast<const llama_kv_cache_context *>(params.mctx)) {
        kvs[nkvs++] = k;
    }
    if (nkvs == 0) {
        ++st.c_fallbacks;
        return false;
    }

    uint32_t nkv_now = kvs[0]->get_n_kv();
    for (int i = 1; i < nkvs; ++i) {
        if (kvs[i]->get_n_kv() != nkv_now) {
            nkv_now = 0;
        }
    }
    // one well-defined width: nkv_now was zeroed above when the sub-caches
    // disagree; an empty cache has no width to bind either. No alignment or
    // rounding here - the binder binds exactly the value the cache reports
    // (any bucket padding is the stock cache-level pad, unchanged for OFF
    // and ON alike)
    if (nkv_now == 0) {
        ++st.c_refused_width;
        ++st.c_fallbacks;
        return false;
    }
    // Fail-closed on attention-view readers (reader-audit fix; was gated on
    // the r8 cov lane). The in-place patch copies tensor layouts only: it
    // cannot re-resolve build-time captured reader spans, so patching a
    // reader graph would execute stale spans against the patched widths.
    // Two detectors, both must miss: the cache's CURRENT attention-view
    // state, and the live graph's captured input reader state (the latter
    // covers graphs built under views that were cleared before this rebind).
    // Refuse unconditionally and fall back to the caller's full rebuild,
    // which re-resolves the readers at the current frontier.
    // c_refused_view stays r8 cov-family (never written with the gate off);
    // c_fallbacks (r7 field) accounts every refusal.
    bool attn_readers = false;
    for (int i = 0; i < nkvs; ++i) {
        attn_readers |= kvs[i]->has_attention_views();
    }
    attn_readers |= llama_graph_result_has_attn_readers(res);
    if (attn_readers) {
        if (st.cov_enabled) {
            ++st.c_refused_view;
        }
        ++st.c_fallbacks;
        return false;
    }
    // non-nkv graph keys must be identical: allow_reuse compares every graph-
    // shaping input EXCEPT the n_kv-coupled shapes (ubatch layout, sequence
    // ids, output flags, samplers, adapters, cross, cparams flags, arch,
    // gtype). Without this gate, a non-nkv difference whose node skeleton
    // happens to match could be patched while the live graph's recorded
    // params (and its set_inputs closures) still reflect the old keys.
    // NOTE: the caller only invokes the binder when can_reuse already failed,
    // so this rejects precisely the "reuse failed for a NON-nkv reason" case.
    // (encapsulation: llm_graph_result::params stays private; the
    // get_params() accessor adds no data member)
    if (!res->get_params().allow_reuse(params)) {
        ++st.c_refused_skeleton;
        ++st.c_fallbacks;
        return false;
    }
    // true-width window: the live graph was built and sched-allocated AT
    // record_cap (the cache's exact n_kv at build time - no envelope), so
    // only n_kv <= record_cap has real backing. record_bound tracks the width
    // the live graph's nkv-coupled shapes are CURRENTLY bound at (it moves on
    // every successful patch and equals record_cap right after a fresh
    // build), so:
    // - equality with the bound width is a true no-op (a patch would be an
    //   exact no-op; the normal reuse path usually takes this case first, but
    //   stay fail-closed);
    // - anything above the built width has no allocated backing -> full
    //   rebuild at the cache's exact current n_kv (which re-records).
    if (nkv_now > st.record_cap) {
        ++st.c_refused_grow;
        ++st.c_fallbacks;
        return false;
    }
    if (nkv_now == st.record_bound) {
        // the live graph's nkv-coupled shapes are already bound at exactly
        // this width: a patch would be an exact no-op. This compares against
        // record_bound (the current bound width); a grow-back to the built
        // width after a shrink patch is a real bind, not a no-op
        ++st.c_fallbacks;
        return false;
    }

    // scratch build at the cache's CURRENT n_kv. Everything except the
    // nkv-coupled shapes must match; the shared scratch-validate-patch tail
    // is the definitive gate: a skeleton match proves the only difference is
    // nkv-coupled shapes (the scratch is built at the live graph's CURRENT
    // bound output width), and the byte-extent checks prove the placement
    // still fits.
    llm_graph_params gp = params;
    // build the scratch at the live graph's CURRENT bound output width
    // (nout_bound), never at this ubatch's active count (params.n_outputs
    // holds the requested count) - so the patch leaves every output shape
    // untouched
    gp.n_outputs = st.nout_bound;
    if (!union_binder_scratch_patch(res, std::move(gp), set, st.cov_enabled ? st.cov_pending : nullptr)) {
        return false;
    }
    st.record_bound = nkv_now;   // the live graph's nkv-coupled shapes are now bound at nkv_now
    ++st.c_applied;
    // Coverage remains staged until output reconciliation and scheduler allocation succeed.
    return true;
}

// n_outputs binding: shares the nkv lane's pre-allocation record (one live
// graph, one fresh-build record)
llama_union_tensor_set llama_context::union_tensors_n_outputs() {
    auto & st = llama_nkv_ctx_state_get(this);
    return { &st.record_t, &st.record_src, &st.record_fixed, &st.record_bytes };
}

// n_outputs binding: narrow/restore the executed output shapes of the live
// build to this ubatch's active count. The live graph was built and
// sched-allocated at the EXACT active count of its build round (no
// pad-to-max, no envelope - the stock out_ids topology); its out_ids input
// and every GET_ROWS dst reading it currently hold nout_bound (the LAST
// bound active count). This rebind:
//   1. requires the same placement contracts as the nkv lane (fail-closed);
//   2. requires the kv side of the live graph to be consistent (the scratch
//      build must NOT touch nkv-coupled shapes - that is the n_kv lane's job);
//   3. builds a scratch graph at the active count (plain n_outputs - no
//      envelope);
//   4. requires an exact skeleton match against the recorded live build
//      (any topology difference -> caller rebuilds);
//   5. checks every changed tensor fits its recorded byte extent;
//   6. copies the changed fields into the live graph in place.
// n_outputs == 0 is a first-class bound: the scratch build is the stock
// zero-output topology (the gather over an empty out_ids), the skeleton
// matches, and every narrowed tensor has nbytes 0 - the graph computes zero
// output rows, exactly like a stock zero-output build.
bool llama_context::union_rebind_outputs(llm_graph_result * res, const llm_graph_params & params, const llama_union_tensor_set & set) {
    auto & st = llama_nkv_ctx_state_get(this);

    // NKV_BINDER_STATS: every binder invocation counts as an attempt
    ++st.c_attempts;

    if (!res || res->get_gf() == nullptr) {
        ++st.c_fallbacks;
        return false;
    }
    // the binder needs the pre-allocation record of a live decoder build
    if (!st.nout_record_valid || set.tensors->empty()) {
        ++st.c_fallbacks;
        return false;
    }
    // the live graph must actually carry an out_ids gather input (pooling /
    // embedding-only graphs have none): nothing to reconcile -> fail closed
    {
        bool has_out_ids = false;
        for (const auto & input : res->inputs) {
            if (dynamic_cast<const llm_graph_input_out_ids *>(input.get()) != nullptr) {
                has_out_ids = true;
                break;
            }
        }
        if (!has_out_ids) {
            ++st.c_refused_skeleton;
            ++st.c_fallbacks;
            return false;
        }
    }
    // tp5-relay pins host graph buffers and validates every node's scheduler
    // placement after the rebuild; in-place n_outputs patching is out of its
    // contract (same as the nkv lane)
    if (backend_tp5_relay != nullptr || cparams.pipeline_parallel) {
        if (backend_tp5_relay != nullptr) {
            ++st.c_refused_relay;
        }
        if (cparams.pipeline_parallel) {
            ++st.c_refused_pipeline;
        }
        ++st.c_fallbacks;
        return false;
    }
    // non-nkv graph keys must be identical: normalize the n_outputs key (the
    // graph is built at the exact active count; the ACTIVE count is exactly
    // what THIS rebind reconciles)
    {
        llm_graph_params pk = params;
        pk.n_outputs = res->get_params().n_outputs;
        if (!res->get_params().allow_reuse(pk)) {
            ++st.c_refused_skeleton;
            ++st.c_fallbacks;
            return false;
        }
    }
    // no-op: the executed output shapes already hold this active count
    if ((uint32_t) n_outputs == st.nout_bound) {
        ++st.c_fallbacks;
        return false;
    }
    // growth past the built width has no allocated backing; cannot normally
    // happen (the live graph was built at a previous active count and this
    // rebind only moves the executed shapes within [0, nout_cap]) but stay
    // fail-closed
    if ((uint32_t) n_outputs > st.nout_cap) {
        ++st.c_refused_grow;
        ++st.c_fallbacks;
        return false;
    }

    // kv consistency: when the nkv lane has a live record, the scratch build
    // runs at the cache's CURRENT n_kv and the live graph's nkv-coupled
    // shapes are bound at record_bound - they must agree (the nkv lane binds
    // first when the cache moved, or nothing moved), otherwise output shapes
    // would be patched onto a kv side bound at another width: fail closed,
    // caller rebuilds. This also refuses the combined "kv grew past the
    // built width + outputs changed" round.
    // When the nkv lane has NO live record (lane off, or no KV-cache
    // context in this graph), no in-place nkv patching can have happened:
    // the live kv shapes are exactly what the last fresh build produced and
    // can_reuse (live-ne keys, llama-graph.cpp) already refused any mismatch
    // with the cache - the check is skipped, not forced to a 0-bound
    // rejection (an unconditional check would refuse every round when the
    // legacy outputs gate runs alone, since record_bound stays 0 there).
    const llama_kv_cache_context * kvs[2] = { nullptr, nullptr };
    int nkvs = 0;
    if (const auto * h = dynamic_cast<const llama_memory_hybrid_context *>(params.mctx)) {
        kvs[nkvs++] = h->get_attn();
    } else if (const auto * k = dynamic_cast<const llama_kv_cache_context *>(params.mctx)) {
        kvs[nkvs++] = k;
    }
    if (nkvs > 0 && st.nkv_enabled && st.record_valid) {

        uint32_t nkv_now = kvs[0]->get_n_kv();
        for (int i = 1; i < nkvs; ++i) {
            if (kvs[i]->get_n_kv() != nkv_now) {
                nkv_now = 0;
            }
        }
        if (nkv_now != st.record_bound) {
            ++st.c_refused_width;
            ++st.c_fallbacks;
            return false;
        }
    }
    // Fail-closed on attention-view readers (reader-audit fix): the n_outputs
    // rebind's shared pre-allocation record covers reader input tensors too,
    // and a patch copies layout only - it cannot re-resolve build-time
    // captured reader spans. Two detectors, both must miss: the cache's
    // CURRENT attention-view state, and the live graph's captured input
    // reader state (the latter covers graphs built under views that were
    // cleared before this rebind). Refuse and fall back to the caller's full
    // rebuild, which re-resolves the readers at the current frontier.
    // c_refused_view stays r8 cov-family (never written with the gate off);
    // c_fallbacks (r7 field) accounts every refusal.
    bool attn_readers = false;
    for (int i = 0; i < nkvs; ++i) {
        attn_readers |= kvs[i]->has_attention_views();
    }
    attn_readers |= llama_graph_result_has_attn_readers(res);
    if (attn_readers) {
        if (st.cov_enabled) {
            ++st.c_refused_view;
        }
        ++st.c_fallbacks;
        return false;
    }

    // scratch build at the active count (plain n_outputs - no envelope)
    llm_graph_params gp = params;
    gp.n_outputs = (uint32_t) n_outputs;

    const bool ok = union_binder_scratch_patch(res, std::move(gp), set, nullptr);
    if (!ok) {
        return false;
    }

    st.nout_bound = (uint32_t) n_outputs;   // the executed output shapes now hold this count
    // r8: the shared scratch tail no longer bumps c_applied (the r7 tail bump
    // double-counted the nkv lane); each caller counts its own success. This
    // is the FINAL success point for the n_outputs lane: everything after it
    // can only fail via the caller's alloc-failure rollback, which unwinds
    // this increment (--c_applied per rolled-back lane).
    ++st.c_applied;
    ++st.nout_binds;
    return true;
}

// shared scratch-validate-patch tail of every union binding (nkv + n_outputs):
// build the scratch graph from gp, require an exact skeleton match against the
// recorded fresh build, check every changed tensor fits its recorded byte
// extent, and copy the changed fields into the live graph in place. Any
// mismatch leaves the live graph untouched and returns false (the caller falls
// back to a full rebuild). Takes gp by value: the caller sets res/n_outputs
// before dispatching. cov_patched_out (r8 coverage lane, may be null): when
// non-null, the apply loop classifies every changed tensor into its
// nkv-coupled coverage category (llama_nkv_cov_classify) and accumulates the
// per-category count into a LOCAL array that is published into the caller's
// array only at the final success return - a refused patch can never leak
// counts (no bump-on-fail); null keeps the accounting off.
bool llama_context::union_binder_scratch_patch(llm_graph_result * res, llm_graph_params gp, const llama_union_tensor_set & set, uint32_t * cov_patched_out) {
    auto & st = llama_nkv_ctx_state_get(this);

    uint32_t cov_local[LLAMA_NKV_COV_N] = {};   // r8: staged per-category counts (published at the final return)

    // the record vectors come from the binding's tensor set provider; alias them
    // once (all later reads/writes go through these names)
    const std::vector<ggml_tensor *> & record_t     = *set.tensors;
    const std::vector<int32_t>       & record_src   = *set.src;
    const std::vector<uint8_t>       & record_fixed = *set.fixed;
    const std::vector<size_t>        & record_bytes = *set.bytes;

    llm_graph_result_ptr & nkv_scratch = *reinterpret_cast<llm_graph_result_ptr *>(&st.nkv_scratch);
    if (!nkv_scratch) {
        nkv_scratch.reset(new llm_graph_result(res->get_max_nodes()));
    }
    nkv_scratch->reset();

    // naming-only callback: the real callback pins tensors on the scheduler, which a
    // never-placed scratch graph must not touch

    gp.res = nkv_scratch.get();
    gp.cb = [](const llama_ubatch &, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }
    };

    ggml_cgraph * gs = model.build_graph(gp);
    if (gs == nullptr) {
        ++st.c_refused_skeleton;
        ++st.c_fallbacks;
        return false;
    }

    llama_nkv_record_t rec_new;
    rec_new.record(gs);
    rec_new.record_bytes(); // envelope sizes for the shrink-only byte checks below

    const size_t n = record_t.size();
    if (rec_new.t.size() != n || rec_new.src != record_src ||
            rec_new.fixed != record_fixed || rec_new.bytes.size() != record_bytes.size()) {
        ++st.c_refused_skeleton;
        ++st.c_fallbacks;
        return false;
    }

    // exact skeleton + every changed field fits: copy in place
    for (size_t k = 0; k < n; ++k) {
        ggml_tensor * s = rec_new.t[k];
        ggml_tensor * d = record_t[k];

        if (record_fixed[k]) {
            // pre-existing storage (weights, KV cache): the scratch build must
            // reference the very same tensor object
            if (s != d) {
                ++st.c_refused_skeleton;
                ++st.c_fallbacks;
                return false;
            }
            continue;
        }

        if (s->op != d->op || s->type != d->type ||
                strncmp(s->name, d->name, GGML_MAX_NAME) != 0 ||
                (s->view_src == nullptr) != (d->view_src == nullptr)) {
            ++st.c_refused_skeleton;
            ++st.c_fallbacks;
            return false;
        }

        if (s->view_src == nullptr) {
            // graph-owned tensor: must fit inside the envelope the scheduler
            // allocated the live tensor at (shrink-only)
            if (ggml_nbytes(s) > record_bytes[k]) {
                ++st.c_refused_envelope;
                ++st.c_fallbacks;
                return false;
            }
        } else {
            // 0-byte view can have a nonzero view_offs (static slicing); no bytes
            // are accessed, so skip the range check (avoids false OOB rejections)
            if (ggml_nbytes(s) == 0) {
                continue;
            }
            const int32_t vs = rec_new.src[k * (GGML_MAX_SRC + 1) + GGML_MAX_SRC];
            if (vs < 0) {
                ++st.c_refused_skeleton;
                ++st.c_fallbacks;
                return false;
            }
            // the limit is the full byte size of the view's source: a fixed source
            // (KV cache) has no recorded envelope, its allocation is exact
            const size_t lim = record_fixed[vs] ? ggml_nbytes(record_t[vs]) : record_bytes[vs];
            if (s->view_offs + ggml_nbytes(s) > lim) {
                ++st.c_refused_envelope;
                ++st.c_fallbacks;
                return false;
            }
        }
    }

    // apply: copy every changed tensor's layout into the live graph
    for (size_t k = 0; k < n; ++k) {
        if (record_fixed[k]) {
            continue;
        }
        ggml_tensor * s = rec_new.t[k];
        ggml_tensor * d = record_t[k];

        const bool diff = memcmp(s->ne, d->ne, sizeof(d->ne)) != 0 ||
                          memcmp(s->nb, d->nb, sizeof(d->nb)) != 0 ||
                          memcmp(s->op_params, d->op_params, sizeof(d->op_params)) != 0 ||
                          s->view_offs != d->view_offs;
        if (!diff) {
            continue;
        }

        // r8 coverage lane: classify BEFORE the copy (the mask rule keys on
        // the record-time first dim == the recorded built width); the copy
        // below overwrites d->ne with the scratch's shapes
        int cov_cat = LLAMA_NKV_COV_OTHER;
        if (cov_patched_out != nullptr) {
            const int32_t s0 = record_src[k * (GGML_MAX_SRC + 1) + 0];
            const int32_t vs = record_src[k * (GGML_MAX_SRC + 1) + GGML_MAX_SRC];
            cov_cat = llama_nkv_cov_classify((int32_t) s->op, s->name,
                                             (s->flags & GGML_TENSOR_FLAG_INPUT) != 0,
                                             record_fixed[k] != 0,
                                             s0 >= 0 && record_fixed[s0] != 0,
                                             vs >= 0 && record_fixed[vs] != 0,
                                             d->ne[0], st.cov_nkv_built);
        }
        memcpy(d->ne, s->ne, sizeof(d->ne));
        memcpy(d->nb, s->nb, sizeof(d->nb));
        memcpy(d->op_params, s->op_params, sizeof(d->op_params));
        d->view_offs = s->view_offs;
        if (d->view_src != nullptr && d->view_src->data != nullptr) {
            d->data = (char *) d->view_src->data + d->view_offs;
        }

        if (cov_patched_out != nullptr) {
            ++cov_local[cov_cat];   // staged: published only at the final return
        }
    }

    // restore: the previous alloc_graph's split_graph pass 5 (ggml-backend.cpp,
    // "node->src[j] = tensor_id_copy(...)") rewrote the LIVE graph's node src[]
    // to point at sched->ctx-owned split-input copies. ggml_backend_sched_reset
    // + alloc_graph re-runs split_graph on this same live graph, and the freed
    // copy structs' stale buffer/data fields then poison the re-derivation (the
    // freed input_cpy is treated as a pre-allocated tensor of the GPU backend,
    // so no new copy is created for it and the node keeps a freed struct as its
    // source -> garbage ids read at compute -> OOB). The record was taken
    // pre-alloc (binder_record_snapshot, before the first alloc_graph), so it
    // holds the pristine build-time topology: restore every non-fixed recorded
    // tensor's src[]/view_src from it. Fixed tensors (weights, KV cache) are
    // never rewritten by split_graph and are skipped. -1 = null src (unchanged),
    // -2 = back-edge (impossible in a build DAG; leave the pointer untouched).
    // Pure pointer fix-up from the same vectors the skeleton compare validated:
    // no new failure mode, and the OFF path never enters this function.
    for (size_t k = 0; k < n; ++k) {
        if (record_fixed[k]) {
            continue;
        }
        const int32_t vsrc = record_src[k * (GGML_MAX_SRC + 1) + GGML_MAX_SRC];
        if (vsrc >= 0) {
            record_t[k]->view_src = record_t[vsrc];
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const int32_t v = record_src[k * (GGML_MAX_SRC + 1) + j];
            if (v >= 0) {
                record_t[k]->src[j] = record_t[v];
            }
        }
    }

    // clear stale allocation state: the previous alloc_graph's gallocr wrote
    // buffer/data (and, for CPU-side buffers, extra) into every graph-owned
    // tensor of this LIVE graph. The re-plan below (sched_reset +
    // alloc_graph) re-runs split_graph on the same graph, and pass 1
    // (ggml-backend.cpp, ggml_backend_sched_backend_id_from_cur ->
    // ggml_backend_sched_backend_from_buffer) treats any tensor with a
    // non-NULL buffer (directly or via view_src) as PRE-ALLOCATED: it derives
    // the backend from the stale buffer instead of the op, and when no
    // backend both owns that buffer type and supports the op it GGML_ABORTs
    // ("pre-allocated tensor ... cannot run the operation"). The stale
    // buffer/data also poison supports_op's src[] scan (a freed split-copy
    // struct's buffer/buft is undefined). Clearing buffer/data/extra on the
    // graph-owned tensors restores the fresh-build state the re-plan expects:
    //   - split_graph pass 1 then assigns backends by op, as on a fresh graph;
    //   - gallocr re-assigns storage: ggml_gallocr_init_tensor (ggml-alloc.c)
    //     re-runs ggml_backend_view_init for views (buffer==NULL) and
    //     ggml_vbuffer_tensor_alloc for base tensors (data==NULL), and
    //     ggml_gallocr_needs_realloc falls back to a full reserve when a
    //     cleared tensor's size no longer fits its recorded size_max - so no
    //     cleared tensor can reach compute unallocated;
    //   - extra is only ever written by CPU-side buffer init_tensor hooks
    //     (repack/amx) and never read on GPU paths; clearing it is safe and
    //     keeps the tensor byte-identical to a fresh build.
    // Scope (matches the GPU-verified probe semantics): every recorded tensor
    // that is NOT fixed (fixed = pre-existing storage: weights, KV cache) and
    // NOT a view of a fixed tensor (KV-cache views keep their fixed source's
    // live buffer/data; their own buffer/data were also gallocr-assigned but
    // their view_src is never rewritten, and the re-plan re-inits them via
    // ggml_backend_view_init only if their buffer is NULL - so leave them
    // alone, exactly like the probe). Fixed tensors are never touched.
    for (size_t k = 0; k < n; ++k) {
        if (record_fixed[k]) {
            continue;
        }
        const int32_t vsrc = record_src[k * (GGML_MAX_SRC + 1) + GGML_MAX_SRC];
        if (vsrc >= 0 && record_fixed[vsrc]) {
            // view of a fixed tensor (KV view): keep its buffer/data
            continue;
        }
        ggml_tensor * d = record_t[k];
        d->buffer = nullptr;
        d->data   = nullptr;
        d->extra  = nullptr;
    }
    nkv_scratch->reset();
    // r8: success counting moved to the CALLERS (union_rebind_n_kv bumps
    // c_applied for the nkv lane, union_rebind_outputs for the n_outputs
    // lane). The r7 tail bump double-counted every nkv bind (the caller
    // bumps too), which broke attempts == applied + fallbacks on every
    // successful bind; the r8 counters fix the identity.
    // r8 coverage lane: publish the local per-category counts only now - the
    // patch is committed past every failure path (skeleton/envelope checks
    // all ran before the apply loop; the restore/clear loops cannot fail), so
    // a refused patch can never leak counts into the caller's array.
    if (cov_patched_out != nullptr) {
        for (int c = 0; c < LLAMA_NKV_COV_N; ++c) {
            cov_patched_out[c] = cov_local[c];
        }
    }
    return true;
}

// GGML_TP5_FA_NKV_PARAM (exact "1", r8): per-request true runtime KV length.
// The FA op param is written at graph build, so the gate must keep every
// reuse/patch/envelope path off: a reused or patched graph would carry the
// previous eval's param value and the kernels would read a stale bound.
static bool llama_ctx_fa_nkv_param_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_TP5_FA_NKV_PARAM");
        return env != nullptr && strcmp(env, "1") == 0;
    }();
    return enabled;
}
llm_graph_result * llama_context::process_ubatch(const llama_ubatch &     ubatch,
                                                 llm_graph_type           gtype,
                                                 llama_memory_context_i * mctx,
                                                 ggml_status &            ret) {
    if (backend_tp5_relay != nullptr) {
        // A retained input batch owns this context's host graph buffers until
        // completion. Drain before memory updates, graph reset, or set_inputs.
        // Do not account the current, not-yet-submitted token as synchronized.
        ggml_backend_sched_synchronize(sched.get());
    }

    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    // Consume feedback from the previous completed decode before building the
    // next graph. Repartition itself synchronizes only when hysteresis selects
    // a new boundary, so steady-state token generation stays asynchronous.
    //
    // TurboQuant: this used to be gated on dynamic_cast<llama_memory_hybrid*>
    // with no fallback, so for a PLAIN (non-hybrid) llama_kv_cache - what
    // every model validated on this fork so far actually uses - the phase
    // never switched past its initial construction-time prefill layout, and
    // kv_stream_adapt() never ran at all. mctx->get_kv_stream_active_targets()
    // covers both cases (and, once wired up, iSWA/DSA/MSA/DSV4's sub-caches)
    // uniformly.
    if (mctx != nullptr && mctx->has_kv_stream_targets()) {
        for (const auto & active_target : mctx->get_kv_stream_active_targets()) {
            if (active_target.cache == nullptr) {
                continue;
            }
            llama_kv_stream_phase phase = LLAMA_KV_STREAM_PHASE_AUTOMATIC;
            if (decode_phase == LLAMA_DECODE_PHASE_PROMPT) {
                phase = LLAMA_KV_STREAM_PHASE_PROMPT;
            } else if (decode_phase == LLAMA_DECODE_PHASE_GENERATION) {
                phase = LLAMA_KV_STREAM_PHASE_GENERATION;
            }
            const bool generation = llama_kv_stream_phase_is_generation(phase, ubatch.n_tokens);
            if (kv_stream_phase_arena.configured && generation && ubatch.n_tokens != cparams.n_seq_max) {
                LLAMA_LOG_ERROR("%s: phase arena currently supports TG1 without speculative batches\n", __func__);
                ret = GGML_STATUS_FAILED;
                return nullptr;
            }
            if (!kv_stream_switch_phase(generation, active_target.n_kv)) {
                LLAMA_LOG_ERROR("%s: failed to switch shared CUDA arena phase\n", __func__);
                ret = GGML_STATUS_ALLOC_FAILED;
                return nullptr;
            }
            (void) active_target.cache->kv_stream_adapt(active_target.n_kv, ubatch.n_tokens);
        }
    }

    auto * res = gf_res_prev.get();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    bool reused = !graph_reuse_disable && !llama_ctx_fa_nkv_param_enabled() && res->can_reuse(gparams);

    // set only when an in-place patch (below) succeeded: the scheduler's
    // split-input copies were created during the PREVIOUS alloc_graph and still
    // carry the pre-patch layout, so they must be re-planned before compute.
    bool nkv_patched  = false;
    bool nout_patched = false;

    // n_outputs lane: the live graph is bound at nout_bound (the LAST bound
    // active count); when this ubatch needs a different count and the
    // n_outputs binding is registered, the binder must reconcile it
    // (can_reuse keys the out_ids input on the count at BUILD time, so a
    // changed count fails the stock key and the bound-vs-active difference
    // forces the dispatch; a narrowed graph reused at its built count is
    // re-narrowed here before any compute).
    auto * binder_st = union_binder_state_peek();
    const bool nout_lane = binder_st != nullptr && binder_st->nout_enabled;
    const bool nout_pending = nout_lane && binder_st->nout_bound != (uint32_t) n_outputs;
    // stale-bound guard: the build-time can_reuse keys mix live tensor shapes
    // (can_reuse_kq_mask compares kq_mask->ne[0] against the cache's CURRENT
    // get_n_kv(), llama-graph.cpp:95-101) with build-time members
    // (llm_graph_input_out_ids::can_reuse compares the BUILD count,
    // llama-graph.cpp:277). An in-place patch moves the live shapes without
    // touching the build record, so a "reusable" verdict can coexist with a
    // bound state that differs from this round's values: e.g. n_outputs
    // returns to the build count after a narrow bind - the out_ids key
    // passes, while the executed gather shapes still hold the narrowed width
    // (missing outputs / OOB gather). Reuse is trusted only when every
    // enabled lane's bound state equals the current value; otherwise fall
    // through to the binder (restore within the built width) or a fail-
    // closed full rebuild. Covers nkv_now == 0 (sub-cache disagreement) too.
    if (reused && binder_st != nullptr) {
        bool stale = false;
        if (binder_st->nout_enabled && binder_st->nout_bound != (uint32_t) n_outputs) {
            stale = true;
        }
        if (!stale && binder_st->nkv_enabled && binder_st->record_valid) {
            const llama_kv_cache_context * kvs[2] = { nullptr, nullptr };
            int nkvs = 0;
            if (const auto * h = dynamic_cast<const llama_memory_hybrid_context *>(mctx)) {
                kvs[nkvs++] = h->get_attn();
            } else if (const auto * k = dynamic_cast<const llama_kv_cache_context *>(mctx)) {
                kvs[nkvs++] = k;
            }
            if (nkvs > 0) {
                uint32_t nkv_now = kvs[0]->get_n_kv();
                for (int i = 1; i < nkvs; ++i) {
                    if (kvs[i]->get_n_kv() != nkv_now) {
                        nkv_now = 0;
                    }
                }
                if (nkv_now != binder_st->record_bound) {
                    stale = true;
                }
            }
        }
        if (stale) {
            reused = false;
        }
    }
    if (!reused && !graph_reuse_disable && binder_st != nullptr && binder_st->nkv_enabled &&
            !llama_ctx_fa_nkv_param_enabled()) {
        // union-graph runtime parameter binding: when the previous graph cannot
        // be reused only because the KV cache moved (width change, bucket
        // crossing, ...), rebuild the graph in a scratch result at the cache's
        // CURRENT n_kv, verify the skeleton matches, and patch the live graph
        // in place only when every changed shape fits the built extents.
        // Table-driven: the kind -> (tensor set provider, rebind) entry lives
        // in union_binder and n_kv and n_outputs are registered today (rs_z is
        // reserved). Fail-closed: an unregistered kind, any mismatch or any
        // growth past the built width falls back to a full rebuild.
        // GGML_TP5_FA_NKV_PARAM (exact "1"): disabled - the FA op param is
        // written at graph build only, so an in-place patch would leave the
        // previous eval's param value in the live graph; the FA gate keeps
        // every reuse/patch path off and every eval full-rebuilds at the
        // true width.
        auto & union_st = llama_nkv_ctx_state_get(this);
        reused = nkv_patched = union_st.union_binder.rebind(llama_union_param_kind::n_kv, this, res, gparams);
    }

    // outputs-only change (or a pending reconcile after an nkv refusal): the
    // n_outputs binding narrows/restores the executed gather shapes in place.
    // Skipped when the nkv patch already reconciled the whole param set, or
    // when the lane is unregistered (nout_pending false). GGML_TP5_FA_NKV_PARAM
    // (exact "1"): disabled like every in-place patch - a patched-then-reused
    // graph skips the rebuild that refreshes the FA op param.
    if (nout_pending && (nkv_patched || !reused) && !graph_reuse_disable && binder_st != nullptr &&
            binder_st->nout_enabled && !llama_ctx_fa_nkv_param_enabled()) {
        if (binder_st->union_binder.rebind(llama_union_param_kind::n_outputs, this, res, gparams)) {
            nout_patched = true;
            reused = true;
        } else if (nkv_patched) {
            // the kv patch applied but the output reconcile failed: the live
            // graph is half-patched (kv shapes at the new bucket, output shapes
            // at the OLD bound) - discard it (the rebuild below resets the
            // result and re-records a fresh build) and keep the counters
            // honest
            auto & nkv_st = llama_nkv_ctx_state_get(this);
            --nkv_st.c_applied;
            ++nkv_st.c_fallbacks;
            nkv_patched = false;
            reused      = false;
        }
    }

    if (reused) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // an in-place patch (union_rebind_n_kv / union_rebind_outputs) mutated
        // the live graph tensors' ne/nb/op_params/view_offs AFTER the previous alloc_graph
        // created the scheduler's split-input copies at the OLD layout; at
        // compute time ggml_backend_sched_compute_splits would copy graph
        // tensor -> stale split copy and the ggml_are_same_layout assert fires.
        // Force a re-plan: alloc_graph unconditionally re-runs split_graph,
        // which frees the whole copy-tensor context and re-creates every
        // split-input copy at the CURRENT (patched) layout, and re-assigns
        // input tensor data pointers (incl. views), so set_inputs() must run
        // AFTER it. Fail-closed: on alloc failure fall through to the normal
        // full-rebuild path below (the rebuild's res->reset() + build_graph
        // discards the in-place mutation; gparams already reflects the new
        // parameters) and keep the NKV_BINDER_STATS counters honest
        // (--c_applied per patched lane, ++c_fallbacks).
        if (nkv_patched || nout_patched) {
            ggml_backend_sched_reset(sched.get());
            ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
            if (!ggml_backend_sched_alloc_graph(sched.get(), res->get_gf())) {
                auto & nkv_st = llama_nkv_ctx_state_get(this);
                if (nkv_patched) {
                    --nkv_st.c_applied;
                    ++nkv_st.c_fallbacks;   // the unwound bind becomes a fallback (identity attempts == applied + fallbacks)
                }
                if (nout_patched) {
                    --nkv_st.c_applied;
                    --nkv_st.nout_binds;
                    ++nkv_st.c_fallbacks;   // same, per rolled-back lane
                }
                reused       = false;
                nkv_patched  = false;
                nout_patched = false;
            }
            if (nkv_patched && binder_st->cov_enabled) {
                auto & nkv_st = llama_nkv_ctx_state_get(this);
                ++nkv_st.c_binder_bind;
                for (int c = 0; c < LLAMA_NKV_COV_N; ++c) {
                    nkv_st.c_cov_patched[c] += nkv_st.cov_pending[c];
                }
            }
        }

        if (reused) {
            // with pipeline parallelism, the previous graph_compute_async may still be running
            // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
            // that the previous compute is still reading.
            if (cparams.pipeline_parallel) {
                ggml_backend_sched_synchronize(sched.get());
            }

            n_reused++;
        }
    }
    // fresh-build bookkeeping shared by the build branch below (record +
    // per-lane validity/cap/bound); also read by nothing else - the narrow
    // that used to live further down was folded into the dispatch above
    bool     rec_nkv = false;
    bool     env_out = false;
    uint32_t nkv_built = 0;
    const llama_kv_cache_context * kv_ctx = nullptr;
    if (!reused) {
        res->reset();

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
        // union-graph runtime nkv record: build the graph at the KV cache's
        // CURRENT true n_kv (no envelope, no headroom, no bucket sizing on
        // this lane's side - the value get_n_kv() reports is bound as-is), so
        // a later move to a SMALLER n_kv can be patched in place while a move
        // to a LARGER n_kv full-rebuilds at the exact new width.
        if (!graph_reuse_disable && union_binder_state_peek() != nullptr &&
                union_binder_state_peek()->nkv_enabled && !cparams.pipeline_parallel &&
                backend_tp5_relay == nullptr) {
            if (const auto * h = dynamic_cast<const llama_memory_hybrid_context *>(mctx)) {
                kv_ctx = h->get_attn();
            } else if (const auto * k = dynamic_cast<const llama_kv_cache_context *>(mctx)) {
                kv_ctx = k;
            }
            if (kv_ctx != nullptr) {
                // the cache's current width: what the fresh build uses and
                // what the record binds (no +1 bucket, no padding)
                nkv_built = kv_ctx->get_n_kv();
                rec_nkv   = true;
            }
        }
        // n_outputs record: taken on EVERY fresh decoder build under the lane
        // (no pad-to-max - the graph is built at the exact active count and
        // nout_cap records that built width; the rebind only moves the
        // executed shapes within [0, nout_cap])
        env_out = !graph_reuse_disable && union_binder_state_peek() != nullptr &&
            union_binder_state_peek()->nout_enabled && gtype == LLM_GRAPH_TYPE_DECODER;

        gf = model.build_graph(gparams);

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        // one pre-allocation record serves both lanes (one live graph):
        //  - the nkv lane's record_valid stays tied to its true-width fresh
        //    build (r7 semantics: the graph is built and sched-allocated at
        //    the cache's exact current n_kv - no envelope, no headroom; a
        //    later move to a LARGER n_kv full-rebuilds and re-records);
        //  - the n_outputs lane records EVERY fresh decoder build (no
        //    pad-to-max): nout_cap is the BUILT width, nout_bound the width
        //    the executed shapes currently hold (== nout_cap on a fresh
        //    build).
        // peek first: with the gate off no state slot exists and this whole
        // block is skipped - the OFF path allocates nothing (stock). The
        // get() below then reuses the existing slot (no allocation).
        if (union_binder_state_peek() != nullptr) {
            auto & st = llama_nkv_ctx_state_get(this);

            if (rec_nkv || env_out) {
                binder_record_snapshot(gf, nkv_built);
            }
            st.record_valid = rec_nkv;
            if (rec_nkv) {
                st.record_cap   = nkv_built;   // exact built n_kv (no envelope)
                st.record_bound = nkv_built;   // fresh build: bound == cap
                ++st.c_record_builds;
            } else {
                st.record_bound = 0;   // no nkv lane record: the nout consistency check is gated off (no in-place kv patching possible)
            }
            st.nout_record_valid = env_out;
            if (env_out) {
                st.nout_cap = (uint32_t) n_outputs;   // the BUILT width (no pad-to-max)
            }
            st.nout_bound = (uint32_t) n_outputs;
        }

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }

        // r8 coverage lane: the graph was (re)built AND sched-allocated under
        // the enabled nkv lane - count it (mock mirror: the contract harness
        // counts the first build as a rebuild too). Counted only after the
        // allocation succeeded: a failed alloc abandons the build (the caller
        // returns above) and must not count as a completed rebuild.
        if (union_binder_state_peek() != nullptr) {
            auto & st = llama_nkv_ctx_state_get(this);
            if (st.cov_enabled && st.nkv_enabled) {
                ++st.c_graph_rebuild;
            }
        }
    }

    // Validate the actual scheduler placement before submitting any compute.
    // The reference may use CPU shape/index operations, never CPU quantized
    // projections: CPU's native activation policy is not this F32 policy.
    if (backend_quantized_matmul_f32 != nullptr) {
        if (backend_tp5_relay != nullptr && ggml_backend_sched_get_n_splits(sched.get()) != 1) {
            LLAMA_LOG_ERROR("%s: tp5-relay requires a single native Meta graph, not host-routed splits\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
        for (int node = 0; node < ggml_graph_n_nodes(gf); ++node) {
            ggml_tensor * tensor = ggml_graph_node(gf, node);
            if ((tensor->op == GGML_OP_MUL_MAT || tensor->op == GGML_OP_MUL_MAT_ID) &&
                tensor->src[0] != nullptr && ggml_is_quantized(tensor->src[0]->type) &&
                ((tensor->op == GGML_OP_MUL_MAT_ID && tensor->src[0]->type != GGML_TYPE_Q2_0) ||
                 tensor->src[1] == nullptr || tensor->src[1]->type != GGML_TYPE_F32 ||
                 ggml_backend_sched_get_tensor_backend(sched.get(), tensor) != backend_quantized_matmul_f32)) {
                LLAMA_LOG_ERROR("%s: quantized_matmul_f32 refuses unsupported or non-GPU quantized projection %s\n", __func__, tensor->name);
                ret = GGML_STATUS_FAILED;
                return nullptr;
            }
            if (backend_tp5_relay != nullptr &&
                (tensor->flags & GGML_TENSOR_FLAG_COMPUTE) != 0 && tensor->op != GGML_OP_NONE &&
                tensor->op != GGML_OP_VIEW && tensor->op != GGML_OP_RESHAPE &&
                tensor->op != GGML_OP_PERMUTE && tensor->op != GGML_OP_TRANSPOSE &&
                ggml_backend_sched_get_tensor_backend(sched.get(), tensor) != backend_tp5_relay) {
                LLAMA_LOG_ERROR("%s: tp5-relay refuses non-Meta compute node %s\n", __func__, tensor->name);
                ret = GGML_STATUS_FAILED;
                return nullptr;
            }
        }
    }
    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        // NOTE: no narrow block here. Fresh builds run at the exact active
        // count (stock build shape, no pad-to-max), so a fresh graph needs no
        // narrow; every shrink/restore goes through the binder dispatch above
        // (nout_pending -> union_rebind_outputs), which re-plans the scheduler
        // and leaves set_inputs to the shared call below/above. A narrow
        // failure there falls through to the normal full rebuild at the exact
        // count, which re-records the state - no one-shot suppress flag and no
        // stale record can survive.

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch & batch_inp) {
    if (backend_tp5_relay != nullptr) {
        LLAMA_LOG_ERROR("%s: tp5-relay supports decoder-only calls\n", __func__);
        return -1;
    }
    // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd);

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_embd  = hparams.n_embd_inp_enc();
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always pass the full sequence starting from pos = 0
    if (!balloc->init(batch_inp, model.vocab, nullptr, n_embd, cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max,
                      true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        decision_order_next.clear();
        return -1;
    }

    {
        const bool ok = balloc->set_decision_order(decision_order_next);
        decision_order_next.clear();
        if (!ok) {
            return -1;
        }
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status  status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:
                return 2;
            case GGML_STATUS_ALLOC_FAILED:
                return -2;
            case GGML_STATUS_FAILED:
                return -3;
            case GGML_STATUS_SUCCESS:
                GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        // chain mode: logits are [2, n_chain] (token_id, prob) pairs, not [n_vocab, n_tokens]
        const size_t logits_bytes =
            cparams.mtp_chain ? ggml_nbytes(t_logits) : (size_t) n_tokens * n_vocab * sizeof(float);
        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, logits_bytes);
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens * n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0,
                                                  n_tokens * n_embd_out * sizeof(float));
                }
                break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(),
                                                      (n_embd_out * seq_idx) * sizeof(float),
                                                      n_embd_out * sizeof(float));
                    }
                }
                break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(),
                                                      (n_cls_out * seq_idx) * sizeof(float), n_cls_out * sizeof(float));
                    }
                }
                break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_tokens * n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_tokens * n_embd * sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd * cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

static std::map<llama_seq_id, uint32_t> build_seq_to_output_row(const llama_ubatch & ubatch, uint32_t row_offset) {
    std::map<llama_seq_id, uint32_t> seq_to_row;
    // how many output tokens we have seen so far for this ubatch.
    uint32_t                         local = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        // skip tokens that are not output.
        if (!ubatch.output[i]) {
            continue;
        }

        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        // row_offset is the number of output tokens before this ubatch.
        seq_to_row[seq_id]        = row_offset + local;
        ++local;
    }
    return seq_to_row;
}

static void copy_tensor_async_ints(const std::map<llama_seq_id, ggml_tensor *> & tensor_map,
                                   const buffer_view<llama_token> &              sampled,
                                   const std::map<llama_seq_id, uint32_t> &      seq_to_row,
                                   ggml_backend_sched_t                          sched) {
    if (!sampled.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < sampled.size);

        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampled tokens tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        ggml_backend_tensor_get_async(backend, tensor, sampled.data + row, 0, sizeof(sampled.data[row]));
    }
}

static void copy_tensor_async_floats(const std::map<llama_seq_id, ggml_tensor *> & tensor_map,
                                     const buffer_view<float> &                    dst,
                                     size_t                                        stride,
                                     std::vector<uint32_t> &                       counts,
                                     const std::map<llama_seq_id, uint32_t> &      seq_to_row,
                                     ggml_backend_sched_t                          sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "logits/probs tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        float *        row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of logits/probabilities that were written for this row.
        counts[row] = ggml_nelements(tensor);
    }
}

static void copy_tensor_async_candidates(const std::map<llama_seq_id, ggml_tensor *> & tensor_map,
                                         const buffer_view<llama_token> &              dst,
                                         size_t                                        stride,
                                         std::vector<uint32_t> &                       counts,
                                         const std::map<llama_seq_id, uint32_t> &      seq_to_row,
                                         ggml_backend_sched_t                          sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "candidates tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        llama_token *  row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of candidates that were written.
        counts[row] = ggml_nelements(tensor);
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false;  // all sequences use backend sampling
}

int llama_context::decode(const llama_batch & batch_inp) {
    if (backend_tp5_relay != nullptr &&
        (batch_inp.n_tokens <= 0 || uint32_t(batch_inp.n_tokens) > cparams.n_ubatch ||
         batch_inp.token == nullptr || batch_inp.embd != nullptr || !loras->empty() ||
         cparams.embeddings || !cparams.causal_attn || !sampling.samplers.empty())) {
        LLAMA_LOG_ERROR("%s: tp5-relay requires an adapter-free token batch fitting one ubatch\n", __func__);
        return -1;
    }
    if (backend_tp5_relay != nullptr) {
        for (int32_t token = 0; token < batch_inp.n_tokens; ++token) {
            const int32_t count = batch_inp.n_seq_id != nullptr ? batch_inp.n_seq_id[token] : 1;
            const llama_seq_id seq = batch_inp.seq_id != nullptr && count == 1 ? batch_inp.seq_id[token][0] : 0;
            if (count != 1 || seq < 0 || uint32_t(seq) >= cparams.n_seq_max) {
                LLAMA_LOG_ERROR("%s: tp5-relay requires independent sequence IDs within the context range\n", __func__);
                return -1;
            }
        }
    }
    // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd);

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    if (const auto * cache = attention_view_cache()) {
        if ((!cache->attention_views.empty() && !attention_views_supported()) ||
            !cache->validate_attention_batch(batch_inp)) {
            LLAMA_LOG_ERROR("%s: batch does not match the installed reader views\n", __func__);
            return -1;
        }
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab  = vocab.n_tokens();
    const bool    mtp_embd = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && batch_inp.embd;
    const int64_t n_embd   = mtp_embd ? hparams.n_embd_out() : hparams.n_embd_inp();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    if (has_samplers && batch_inp.logits) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
            if (batch_inp.logits[i] == 0) {
                continue;
            }

            const int ns = batch_inp.n_seq_id ? batch_inp.n_seq_id[i] : 1;

            for (int32_t s = 0; s < ns; ++s) {
                const llama_seq_id seq_id = batch_inp.seq_id ? batch_inp.seq_id[i][s] : 0;

                seq_output_count[seq_id]++;
                if (seq_output_count[seq_id] > 1) {
                    LLAMA_LOG_ERROR(
                        "%s: backend sampling requires at most one output token per sequence (seq_id %d had %d)\n",
                        __func__, seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        decision_order_next.clear();
        return -1;
    }

    {
        const bool ok = balloc->set_decision_order(decision_order_next);
        decision_order_next.clear();
        if (!ok) {
            return -1;
        }
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR(
                "%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) &&
                "non-causal attention requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    output_swaps.clear();

    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                }
                break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__,
                                            balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__,
                                   balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__,
                                    balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    if (backend_tp5_relay != nullptr && mctx->get_ubatch().n_tokens != uint32_t(n_tokens_all)) {
        LLAMA_LOG_ERROR("%s: tp5-relay refuses a decode partitioned into multiple ubatches\n", __func__);
        return -1;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;

        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s,
                               pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:
                    return 2;
                case GGML_STATUS_ALLOC_FAILED:
                    return -2;
                case GGML_STATUS_FAILED:
                    return -3;
                case GGML_STATUS_SUCCESS:
                    GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings ? res->get_embd() : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev * n_vocab;

            if (n_outputs) {
                // chain mode: logits are [2, n_chain] pairs, not [n_vocab, n_outputs]
                const size_t extract_bytes =
                    cparams.mtp_chain ? ggml_nbytes(t_logits) : (size_t) n_outputs * n_vocab * sizeof(float);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, extract_bytes);
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float *        embd_out   = embd.data + n_outputs_prev * n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT(n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs) * n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0,
                                                          n_outputs * n_embd_out * sizeof(float));
                        }
                    }
                    break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(),
                                                          (n_embd_out * seq_idx) * sizeof(float),
                                                          n_embd_out * sizeof(float));
                        }
                    }
                    break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(),
                                                          (n_cls_out * seq_idx) * sizeof(float),
                                                          n_cls_out * sizeof(float));
                        }
                    }
                    break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool    masked = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev : n_tokens_prev;

            if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd         = hparams.n_embd_out();
                float *        embd_nextn_out = embd_nextn.data + offset * n_embd;

                GGML_ASSERT((offset + n_rows) * n_embd <= (int64_t) embd_nextn.size);
                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows * n_embd * sizeof(float));
            }
        }

        // Copy backend sampling output if this ubatch produced any sampling tensors.
        if (has_samplers &&
            (!res->t_sampled.empty() || !res->t_sampled_probs.empty() || !res->t_sampled_logits.empty())) {
            const auto seq_to_output_row = build_seq_to_output_row(ubatch, n_outputs_prev);
            const auto stride            = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_ints(res->t_sampled, sampling.sampled, seq_to_output_row, sched.get());

            copy_tensor_async_floats(res->t_sampled_logits, sampling.logits, stride, sampling.logits_count,
                                     seq_to_output_row, sched.get());
            copy_tensor_async_floats(res->t_sampled_probs, sampling.probs, stride, sampling.probs_count,
                                     seq_to_output_row, sched.get());
            copy_tensor_async_candidates(res->t_candidates, sampling.candidates, stride, sampling.candidates_count,
                                         seq_to_output_row, sched.get());
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev += ubatch.n_tokens;
    } while (mctx->next());

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id     = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    // the clef graph has no logits: n_vocab floats per token would be allocated and never written
    bool has_logits     = model.arch != LLM_ARCH_CLEF;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count        = 0;
    size_t backend_token_count        = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size     = has_logits ? n_vocab * n_outputs_max : 0;
    embd.size       = has_embd ? n_embd_out * n_outputs_max : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out * n_outputs_max : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;    // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;  // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size =
        (logits.size + embd.size + embd_nextn.size + embd_layer_inp_float_count + backend_float_count) * sizeof(float) +
        (backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__,
                            prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output      = nullptr;
            logits.data     = nullptr;
            embd.data       = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = { nullptr, 0 };
            }
        }

        auto * buft                 = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev           = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__,
                            new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t    offset = 0;
    uint8_t * base   = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{ output_base, logits.size } : buffer_view<float>{ nullptr, 0 };
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{ (float *) (base + offset), embd.size } : buffer_view<float>{ nullptr, 0 };
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{ (float *) (base + offset), embd_nextn.size } :
                                  buffer_view<float>{ nullptr, 0 };
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            embd_layer_inp[il] = buffer_view<float>{ (float *) (base + offset), (size_t) n_embd * n_batch };
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{ nullptr, 0 };
        }
    }

    if (has_sampling) {
        sampling.logits = { (float *) (base + offset), (size_t) (n_vocab * n_outputs_max) };
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = { (float *) (base + offset), (size_t) (n_vocab * n_outputs_max) };
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = { (llama_token *) (base + offset), (size_t) n_outputs_max };
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = { (llama_token *) (base + offset), (size_t) (n_vocab * n_outputs_max) };
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(), sampling.logits_count.end(), 0);
        std::fill(sampling.probs_count.begin(), sampling.probs_count.end(), 0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = { nullptr, 0 };
        sampling.probs      = { nullptr, 0 };
        sampling.sampled    = { nullptr, 0 };
        sampling.candidates = { nullptr, 0 };

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;
    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);
    return n_outputs_max;
}

void llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes  = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
    }
}

void llama_context::output_reorder() {
    const uint64_t n_vocab    = model.vocab.n_tokens();
    const uint64_t n_embd     = model.hparams.n_embd;
    const uint64_t n_embd_out = model.hparams.n_embd_out();

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0 * n_vocab + k], logits.data[i1 * n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd.data[i0 * n_embd_out + k], embd.data[i1 * n_embd_out + k]);
            }
        }

        if (embd_nextn.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd_nextn.data[i0 * n_embd_out + k], embd_nextn.data[i1 * n_embd_out + k]);
            }
        }

        if (embd_layer_inp.size() > 0) {
            for (int lid = 0; lid < (int) embd_layer_inp.size(); ++lid) {
                if (embd_layer_inp[lid].size > 0) {
                    for (uint64_t k = 0; k < n_embd; ++k) {
                        std::swap(embd_layer_inp[lid].data[i0 * n_embd + k], embd_layer_inp[lid].data[i1 * n_embd + k]);
                    }
                }
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0 * n_vocab + k], sampling.logits.data[i1 * n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0 * n_vocab + k], sampling.probs.data[i1 * n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0 * n_vocab + k], sampling.candidates.data[i1 * n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0], sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0], sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0], sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    if (attention_node_budget != 0) {
        return attention_node_budget;
    }
    if (model.arch == LLM_ARCH_DFLASH && model.hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost about 8.6 nodes per tensor against 5.9 for plain DFlash.
        return std::max<uint32_t>(1024u, 12u * model.n_tensors());
    }

    if (model.arch == LLM_ARCH_KIMI_K3) {
        // the n_tokens*40 budget below is exhausted at ubatch 3840
        return std::max<uint32_t>(n_tokens * 160, 64u * model.n_tensors());
    }
    if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_CLEF ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_QWEN4EXP ||
        model.arch == LLM_ARCH_DEEPSEEK4 ||
        model.arch == LLM_ARCH_DFLASH ||
        model.arch == LLM_ARCH_NANBEIGE ||
        model.arch == LLM_ARCH_MINIMAX_M3) {
        return std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    }
    uint32_t res = std::max<uint32_t>(1024u, 8u * model.n_tensors());
    for (const auto & lora : model.loras) {
        res += lora->get_n_nodes();
    }
    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

ggml_cgraph * llama_context::graph_reserve(uint32_t                       n_tokens,
                                           uint32_t                       n_seqs,
                                           uint32_t                       n_outputs,
                                           const llama_memory_context_i * mctx,
                                           bool                           split_only,
                                           size_t *                       sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__,
                    n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs;  // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n",
                        __func__, n_tokens, n_seqs, n_outputs);
    }

    if (backend_tp5_relay != nullptr) {
        // graph_reserve can discard host input storage from a prior submission.
        ggml_backend_sched_synchronize(sched.get());
    }
    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse the old graph, so we reset the previous graph result to prevent that
    gf_res_prev->reset();

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch       ubatch = balloc.ubatch_reserve(n_tokens / n_seqs, n_seqs);

    // set one output token per sequence in order to activate all backend samplers
    std::vector<llama_seq_id> seq_ids(n_seqs);
    for (uint32_t i = 0; i < n_seqs; ++i) {
        seq_ids[i]         = i;
        ubatch.n_seq_id[i] = 1;
        ubatch.seq_id[i]   = &seq_ids[i];
        ubatch.output[i]   = true;
    }

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    res->reset();

    auto * gf = model.build_graph(gparams);

    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(llm_graph_result *             res,
                                             const llama_ubatch &           ubatch,
                                             const llama_memory_context_i * mctx,
                                             llm_graph_type                 gtype) const {
    // NOTE: stock. The n_outputs binding does NOT build over a padded
    // envelope: every graph builds at the exact active count (the pre-r4
    // behaviour) and the binder's n_outputs rebind only moves the executed
    // shapes WITHIN the built width (shrink or restore); growth beyond the
    // built width is refused (refused_envelope) and falls back to a full
    // rebuild at the new count - no pad-to-max, no headroom.
    return {
        /*.arch        =*/model.arch,
        /*.hparams     =*/model.hparams,
        /*.cparams     =*/cparams,
        /*.ubatch      =*/ubatch,
        /*.gtype       =*/gtype,
        /*.sched       =*/sched.get(),
        /*.backend_cpu =*/backend_cpu,
        /*.cvec        =*/cvec.get(),
        /*.loras       =*/loras.get(),
        /*.mctx        =*/mctx,
        /*.cross       =*/&cross,
        /*.samplers    =*/sampling.samplers,
        /*.n_outputs   =*/n_outputs,

        /*.cb          =*/graph_get_cb(),
        /*.res         =*/res,
    };
}

ggml_status llama_context::graph_compute(ggml_cgraph * gf, bool batched) {
    int               n_threads = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp        = batched ? threadpool_batch : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg               = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    // Maintenance graphs can reuse the reserve arena between submissions.
    // Only the inference result owns inputs through the fences below.
    const bool retain_inputs = backend_tp5_relay != nullptr && gf_res_prev &&
            gf == gf_res_prev->get_gf() && !batched &&
            cparams.n_batch == 1 && cparams.n_ubatch == 1;
    auto status = retain_inputs ?
            ggml_backend_sched_graph_compute_async_retained_inputs(sched.get(), gf) :
            ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
  public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override { size_written += size; }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override { return size_written; }

  private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
  public:
    llama_io_write_host(uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({ tensor, ptr, size, offset });

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override { return size_written; }

  private:
    uint8_t * ptr;
    size_t    buf_size     = 0;
    size_t    size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t *     ptr;
        size_t        size;
        size_t        offset;
    };

    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
  public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({ tensor, ptr, size, offset });

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    size_t n_bytes() override { return size_read; }

  private:
    const uint8_t * ptr;
    size_t          buf_size  = 0;
    size_t          size_read = 0;

    struct read_info {
        ggml_tensor *   tensor;
        const uint8_t * ptr;
        size_t          size;
        size_t          offset;
    };

    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
  public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override { return size_written; }

  private:
    llama_file *         file;
    size_t               size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
  public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override { return size_read; }

  private:
    llama_file *         file;
    size_t               size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
  public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) :
        ptr(p),
        buf_size(len),
        mbufs(mbufs) {}

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/2 * mbuf.n_tensors * ggml_tensor_overhead(),
                /*.mem_buffer =*/NULL,
                /*.no_alloc   =*/true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            const int64_t n = winfo.size / ggml_element_size(winfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft),
                                   mbuf.total_size / 1024.0 / 1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({ tensor, ptr, size, offset });
    }

    size_t n_bytes() override { return size_written; }

  private:
    uint8_t * ptr;
    size_t    buf_size     = 0;
    size_t    size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t *     ptr;
        size_t        size;
        size_t        offset;
    };

    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
  public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) :
        ptr(p),
        buf_size(len),
        mbufs(mbufs) {}

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/mbuf.n_tensors * ggml_tensor_overhead(),
                /*.mem_buffer =*/NULL,
                /*.no_alloc   =*/true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            const int64_t n = rinfo.size / ggml_element_size(rinfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.n_tensors != mbuf.n_tensors || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf.org[i]);
            }
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({ tensor, ptr, size, offset });
    }

    size_t n_bytes() override { return size_read; }

  private:
    const uint8_t * ptr;
    size_t          buf_size  = 0;
    size_t          size_read = 0;

    struct read_info {
        ggml_tensor *   tensor;
        const uint8_t * ptr;
        size_t          size;
        size_t          offset;
    };

    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id          seq_id,
                                         const uint8_t *       src,
                                         size_t                size,
                                         llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

bool llama_context::state_load_file(const char *  filepath,
                                    llama_token * tokens_out,
                                    size_t        n_token_capacity,
                                    size_t *      n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count,
                            n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io(&file);
        const size_t       n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__,
                            n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id  seq_id,
                                          const char *  filepath,
                                          llama_token * tokens_out,
                                          size_t        n_token_capacity,
                                          size_t *      n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic,
                            version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__,
                            n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t       state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t       nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id        seq_id,
                                          const char *        filepath,
                                          const llama_token * tokens,
                                          size_t              n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(
                format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us = ggml_time_us();
    t_eval_us = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused               = 0;
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return;  // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return;  // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(), model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch == 0);
    GGML_ASSERT(n_batch % n_ubatch == 0);

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx                    = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter    = lopt_params.param_filter;
    void *                 param_filter_ud = lopt_params.param_filter_ud;

    //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd, param_filter, param_filter_ud);
    llama_set_param(model->pos_embd, param_filter, param_filter_ud);
    llama_set_param(model->tok_norm, param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b, param_filter, param_filter_ud);
    llama_set_param(model->output_norm, param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b, param_filter, param_filter_ud);
    llama_set_param(model->output, param_filter, param_filter_ud);
    llama_set_param(model->output_b, param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls, param_filter, param_filter_ud);
    llama_set_param(model->cls_b, param_filter, param_filter_ud);
    llama_set_param(model->cls_out, param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b, param_filter, param_filter_ud);
    llama_set_param(model->cls_norm, param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer) / sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(ggml_opt_dataset_t               dataset,
                                   ggml_opt_result_t                result,
                                   const std::vector<llama_token> & tokens,
                                   const std::vector<llama_token> & labels_sparse,
                                   llama_batch &                    batch,
                                   ggml_opt_epoch_callback          callback,
                                   bool                             train,
                                   int64_t                          idata_in_loop,
                                   int64_t                          ndata_in_loop,
                                   int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(), n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token[pos_batch]     = tokens[pos_ctx + pos_batch];
            batch.pos[pos_batch]       = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]  = 1;
            batch.seq_id[pos_batch][0] = 0;
            batch.logits[pos_batch]    = true;
        }

        if (!balloc->init(batch, model.vocab, nullptr, model.hparams.n_embd_inp(),
                          cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
            LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
            return;
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = gf_res_prev.get();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta =
                    4 * size_gf * ggml_tensor_overhead() + 2 * ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/size_meta,
                    /*.mem_buffer =*/nullptr,
                    /*.no_alloc   =*/true,
                };
                ctx_compute_opt = ggml_init(params);
            }

            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef,
                                            (pos_ubatch * labels->ne[0] + labels_sparse[ilabel]) * sizeof(float),
                                            sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch) / n_ubatch + 1,
                         ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(ggml_opt_dataset_t      dataset,
                              ggml_opt_result_t       result_train,
                              ggml_opt_result_t       result_eval,
                              int64_t                 idata_split,
                              ggml_opt_epoch_callback callback_train,
                              ggml_opt_epoch_callback callback_eval) {
    if (backend_tp5_relay != nullptr) {
        // The optimizer clears memory and rebuilds inputs outside process_ubatch.
        ggml_backend_sched_synchronize(sched.get());
    }
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch, n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const int64_t  ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch       batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token> tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start  = ggml_time_us();
    int64_t ndata_in_loop = idata_split * ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train         = true;
        const int64_t  idata_in_loop = idata * ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx * sizeof(llama_token), labels_sparse.data(),
                                        idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch, callback_train, train, idata_in_loop,
                       ndata_in_loop, t_loop_start);
    }

    t_loop_start  = ggml_time_us();
    ndata_in_loop = (ndata - idata_split) * ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train         = false;
        const int64_t  idata_in_loop = (idata - idata_split) * ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx * sizeof(llama_token), labels_sparse.data(),
                                        idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch, callback_eval, train, idata_in_loop,
                       ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/512,
        /*.n_batch                     =*/2048,
        /*.n_ubatch                    =*/512,
        /*.n_seq_max                   =*/1,
        /*.n_rs_seq                    =*/0,
        /*.gdn_replay                  =*/false,
        /*.n_outputs_max               =*/0,
        /*.n_threads                   =*/GGML_DEFAULT_N_THREADS,  // TODO: better default
        /*.n_threads_batch             =*/GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/0.0f,
        /*.rope_freq_scale             =*/0.0f,
        /*.yarn_ext_factor             =*/-1.0f,
        /*.yarn_attn_factor            =*/-1.0f,
        /*.yarn_beta_fast              =*/-1.0f,
        /*.yarn_beta_slow              =*/-1.0f,
        /*.yarn_orig_ctx               =*/0,
        /*.defrag_thold                =*/-1.0f,
        /*.cb_eval                     =*/nullptr,
        /*.cb_eval_user_data           =*/nullptr,
        /*.type_k                      =*/GGML_TYPE_F16,
        /*.type_v                      =*/GGML_TYPE_F16,
        /*.moe_cache_mode              =*/LLAMA_MOE_CACHE_MODE_UNSPECIFIED,
        /*.moe_cache_budget_mib        =*/0,
        /*.abort_callback              =*/nullptr,
        /*.abort_callback_data         =*/nullptr,
        /*.embeddings                  =*/false,
        /*.offload_kqv                 =*/true,
        /*.no_perf                     =*/true,
        /*.op_offload                  =*/true,
        /*.swa_full                    =*/true,
        /*.kv_unified                  =*/false,
        /*.sampler                     =*/nullptr,
        /*.n_sampler                   =*/0,
        /*.ctx_other                   =*/nullptr,
        /*.kv_stream_arena_mib         =*/0,
        /*.n_ctx_kv                    =*/0,
        /*.n_rs_slots                  =*/0,
        /*.batch_invariant_matmul      =*/false,
        /*.quantized_matmul_f32         =*/false,
        /*.tp5_relay                   =*/false,
    };

    return result;
}

llama_context * llama_init_from_model(llama_model * model, llama_context_params params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR && !params.tp5_relay) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if ((model->hparams.is_mla() || model->arch == LLM_ARCH_DEEPSEEK4) && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("%s: model does not support different K (%s) and V (%s) cache types\n", __func__,
                        ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        return nullptr;
    }

    // TurboQuant cache types require flash attention — auto-enable if disabled
    if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED &&
        (params.type_k == GGML_TYPE_TURBO2_0 || params.type_k == GGML_TYPE_TURBO3_0 ||
         params.type_k == GGML_TYPE_TURBO4_0 || params.type_v == GGML_TYPE_TURBO2_0 ||
         params.type_v == GGML_TYPE_TURBO3_0 || params.type_v == GGML_TYPE_TURBO4_0)) {
        LLAMA_LOG_WARN("%s: turbo cache types require flash_attn — enabling automatically\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size  = ggml_blck_size(params.type_k);
        const bool     k_is_turbo = (params.type_k == GGML_TYPE_TURBO2_0 || params.type_k == GGML_TYPE_TURBO3_0 ||
                                 params.type_k == GGML_TYPE_TURBO4_0);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            uint32_t head_k = model->hparams.n_embd_head_k(il);
            // Turbo types zero-pad heads to next multiple of 128 in llama-kv-cache.cpp
            if (k_is_turbo && head_k % 128 != 0) {
                head_k = ((head_k + 127) / 128) * 128;
            }
            if (head_k % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n", __func__,
                                ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size  = ggml_blck_size(params.type_v);
        const bool     v_is_turbo = (params.type_v == GGML_TYPE_TURBO2_0 || params.type_v == GGML_TYPE_TURBO3_0 ||
                                 params.type_v == GGML_TYPE_TURBO4_0);
        const bool     is_mla     = model->hparams.is_mla();
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            uint32_t head_v = model->hparams.n_embd_head_v(il);
            // Turbo types zero-pad; MLA has no separate V cache (V = view of K)
            if (v_is_turbo && !is_mla && head_v % 128 != 0) {
                head_v = ((head_v + 127) / 128) * 128;
            }
            if (head_v % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n", __func__,
                                ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP && model->hparams.n_layer_nextn == 0) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(llama_model * model, llama_context_params params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    if (ctx != nullptr && ctx->get_tp5_relay_backend() != nullptr) {
        ggml_relay_status status = {};
        ggml_backend_t backend = ctx->get_tp5_relay_backend();
        if (!ggml_backend_relay_snapshot(backend, &status) || (status.epoch != 0 && !status.finished)) {
            ggml_backend_relay_abort(backend, GGML_STATUS_ABORTED);
            // Scheduler buffers and memory are also native readers' owners.
            // Do not enter an unbounded synchronize or release them on failure.
            LLAMA_LOG_ERROR("%s: retaining tp5-relay context: native completion is unproved\n", __func__);
            return;
        }
    }
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_ctx_kv(const llama_context * ctx) {
    return ctx->n_ctx_kv();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

bool llama_get_batch_invariant_matmul(const llama_context * ctx) {
    return ctx && ctx->get_cparams().batch_invariant_matmul;
}

bool llama_get_pipeline_parallel(const llama_context * ctx) {
    return ctx && ctx->get_cparams().pipeline_parallel;
}

bool llama_get_flash_attention(const llama_context * ctx) {
    return ctx && ctx->get_cparams().flash_attn;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(llama_context * ctx, ggml_threadpool_t threadpool, ggml_threadpool_t threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

void llama_set_decode_phase(llama_context * ctx, enum llama_decode_phase phase) {
    ctx->set_decode_phase(phase);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

bool llama_memory_release_compute(llama_context * ctx) {
    return ctx != nullptr && ctx->release_compute();
}

uint64_t llama_memory_compute_bytes(const llama_context * ctx) {
    return ctx != nullptr ? ctx->compute_bytes() : 0;
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

void llama_set_decision_order(llama_context * ctx, const int32_t * order, int32_t n_tokens) {
    ctx->set_decision_order(order, n_tokens);
}

bool llama_model_supports_mtp_chain(const llama_model * model) {
    return model != nullptr && model->arch == LLM_ARCH_QWEN35;
}

bool llama_model_uses_shared_position_draft(const llama_model * model) {
    return model != nullptr && model->arch == LLM_ARCH_GEMMA4_ASSISTANT;
}

void llama_set_mtp_chain(llama_context * ctx, bool value) {
    ctx->set_mtp_chain(value);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

enum ggml_type llama_get_kv_cache_type_k(const struct llama_context * ctx) {
    if (!ctx) {
        return GGML_TYPE_COUNT;
    }

    return ctx->get_kv_type_k();
}

enum ggml_type llama_get_kv_cache_type_v(const struct llama_context * ctx) {
    if (!ctx) {
        return GGML_TYPE_COUNT;
    }

    return ctx->get_kv_type_v();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(struct llama_context * ctx,
                                         uint32_t               n_tokens,
                                         uint32_t               n_seqs,
                                         uint32_t               n_outputs) {
    auto                     memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

ggml_backend_t llama_tp5_relay_backend(llama_context * ctx) {
    return ctx != nullptr ? ctx->get_tp5_relay_backend() : nullptr;
}

int32_t llama_set_adapters_lora(llama_context *       ctx,
                                llama_adapter_lora ** adapters,
                                size_t                n_adapters,
                                float *               scales) {
    if (ctx->get_quantized_matmul_f32_backend() != nullptr && n_adapters != 0) {
        LLAMA_LOG_ERROR("%s: quantized_matmul_f32 does not support adapters\n", __func__);
        return -1;
    }
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(llama_context * ctx,
                               const float *   data,
                               size_t          len,
                               int32_t         n_embd,
                               int32_t         il_start,
                               int32_t         il_end) {
    if (ctx->get_quantized_matmul_f32_backend() != nullptr && data != nullptr) {
        LLAMA_LOG_ERROR("%s: quantized_matmul_f32 does not support control vectors\n", __func__);
        return -1;
    }
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

const char * llama_attention_views_unsupported_reason(const llama_context * ctx) {
    return ctx ? ctx->attention_views_unsupported_reason() : "context is null";
}

bool llama_attention_views_supported(const llama_context * ctx) {
    return llama_attention_views_unsupported_reason(ctx) == nullptr;
}

bool llama_attention_set_views(llama_context * ctx, const llama_attention_view * views, size_t count) {
    if (!ctx) {
        return false;
    }
    try {
        return ctx->set_attention_views(views, count);
    } catch (const std::exception & error) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, error.what());
        return false;
    }
}

void llama_attention_clear_views(llama_context * ctx) {
    if (ctx) {
        ctx->clear_attention_views();
    }
}

bool llama_memory_seq_rm_recurrent(llama_memory_t mem, llama_seq_id seq_id) {
    return mem && seq_id >= 0 && mem->seq_rm_recurrent(seq_id);
}

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(llama_memory_t mem,
                         llama_seq_id   seq_id_src,
                         llama_seq_id   seq_id_dst,
                         llama_pos      p0,
                         llama_pos      p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(llama_memory_t mem, llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(llama_memory_t mem, llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(llama_memory_t mem, llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx,
                             const char *    path_session,
                             llama_token *   tokens_out,
                             size_t          n_token_capacity,
                             size_t *        n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context *     ctx,
                             const char *        path_session,
                             const llama_token * tokens,
                             size_t              n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx,
                           const char *    path_session,
                           llama_token *   tokens_out,
                           size_t          n_token_capacity,
                           size_t *        n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context *     ctx,
                           const char *        path_session,
                           const llama_token * tokens,
                           size_t              n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context *       ctx,
                                    uint8_t *             dst,
                                    size_t                size,
                                    llama_seq_id          seq_id,
                                    llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}

size_t llama_state_seq_set_data_ext(llama_context *       ctx,
                                    const uint8_t *       src,
                                    size_t                size,
                                    llama_seq_id          seq_id,
                                    llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context *     ctx,
                                 const char *        filepath,
                                 llama_seq_id        seq_id,
                                 const llama_token * tokens,
                                 size_t              n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx,
                                 const char *    filepath,
                                 llama_seq_id    dest_seq_id,
                                 llama_token *   tokens_out,
                                 size_t          n_token_capacity,
                                 size_t *        n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

///

int32_t llama_encode(llama_context * ctx, llama_batch batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(llama_context * ctx, llama_batch batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                   __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval,
                   1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                   __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval,
                   1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms),
                   (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(struct llama_context *  ctx,
                     ggml_opt_dataset_t      dataset,
                     ggml_opt_result_t       result_train,
                     ggml_opt_result_t       result_eval,
                     int64_t                 idata_split,
                     ggml_opt_epoch_callback callback_train,
                     ggml_opt_epoch_callback callback_eval) {
    ctx->opt_epoch(dataset, result_train, result_eval, idata_split, callback_train, callback_eval);
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}
