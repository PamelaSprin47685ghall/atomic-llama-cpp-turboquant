#include "llama-memory-hybrid-idx.h"
#include "llama-context.h"

#include "../ggml/src/ggml-backend-moe-cache.h"
#include "../ggml/src/ggml-impl.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-relay.h"
#include "llama-arch.h"
#include "llama-batch.h"
#include "llama-ext.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-nout-state.h"
#include "llama-prefab-env.h"
#include "llama-prefab-iface.h"
#include "llama-io.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-msa.h"
#include "llama-kv-cache.h"
#include "llama-kv-stream-config.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-topology-hash.h"
#include "llama-union-shape-guard.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
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
// Pillar 5: a graph object is built once and reused in place, so this helper
// assigns a uid only when the object has none. A rebuild allocates a fresh
// graph (llm_graph_result::reset) and therefore a fresh uid.
//
// GGML_TP5_GRAPH_UID is not a relay fast-path switch. With the switch unset,
// ggml_backend_sched_split_graph() still assigns a uid to every graph whose
// uid is 0, and the relay provider receives the Meta backend's internal child
// graphs, whose uids are assigned on every Meta rebuild regardless of this
// switch. The relay uid fast path therefore still applies when the switch is
// unset. Use GGML_TP5_RELAY_VERIFY_UID=1 to require the signature describe on
// a fast hit.
static void tp5_assign_graph_uid(ggml_backend_t relay, ggml_cgraph * gf) {
    static const bool enabled = [] {
        const char * value = getenv("GGML_TP5_GRAPH_UID");
        return value != nullptr && strcmp(value, "1") == 0;
    }();
    if (relay != nullptr && enabled && gf != nullptr && gf->uid == 0) {
        gf->uid = ggml_graph_next_uid();
    }
}

// Apply the context's MoE cache policy to one scheduler. A scheduler born
// from ggml_backend_sched_new carries an environment-default session; this
// call replaces it with the resolved policy. A dormant outcome is not a
// fallback: the default session was destroyed and no session remains. A
// refusal keeps the previous session and must fail the caller closed.
static_assert(int(LLAMA_MOE_CACHE_MODE_UNSPECIFIED) == int(GGML_MOE_CACHE_MODE_UNSPECIFIED),
              "llama/ggml MoE cache mode values diverged");
static_assert(int(LLAMA_MOE_CACHE_MODE_OFF) == int(GGML_MOE_CACHE_MODE_OFF),
              "llama/ggml MoE cache mode values diverged");
static_assert(int(LLAMA_MOE_CACHE_MODE_AUTO) == int(GGML_MOE_CACHE_MODE_AUTO),
              "llama/ggml MoE cache mode values diverged");
static_assert(int(LLAMA_MOE_CACHE_MODE_ON) == int(GGML_MOE_CACHE_MODE_ON),
              "llama/ggml MoE cache mode values diverged");

static ggml_moe_cache_mode llama_to_ggml_moe_cache_mode(llama_moe_cache_mode mode) {
    switch (mode) {
        case LLAMA_MOE_CACHE_MODE_UNSPECIFIED:
            return GGML_MOE_CACHE_MODE_UNSPECIFIED;
        case LLAMA_MOE_CACHE_MODE_OFF:
            return GGML_MOE_CACHE_MODE_OFF;
        case LLAMA_MOE_CACHE_MODE_AUTO:
            return GGML_MOE_CACHE_MODE_AUTO;
        case LLAMA_MOE_CACHE_MODE_ON:
            return GGML_MOE_CACHE_MODE_ON;
    }
    LLAMA_LOG_ERROR("%s: unknown MoE cache mode %d\n", __func__, int(mode));
    return GGML_MOE_CACHE_MODE_UNSPECIFIED;
}

static bool llama_apply_moe_cache_config(
        ggml_backend_sched_t sched, enum ggml_moe_cache_mode mode, size_t budget_mib) {
    if (sched == nullptr) {
        return false;
    }
    switch (ggml_backend_sched_set_moe_cache(sched, mode, budget_mib)) {
        case GGML_MOE_CACHE_SET_APPLIED:
            return true;
        case GGML_MOE_CACHE_SET_DORMANT:
            LLAMA_LOG_WARN("%s: MoE cache mode=%d budget=%zu MiB is dormant; "
                           "the scheduler has no cache session\n",
                           __func__, int(mode), budget_mib);
            return true;
        case GGML_MOE_CACHE_SET_REFUSED:
            LLAMA_LOG_ERROR("%s: refused to change an unproven MoE cache session; "
                            "the scheduler must not run with its default session\n",
                            __func__);
            return false;
    }
    return false;
}

static bool union_model_shape_supported(const llama_model & model) {
    return model.arch == LLM_ARCH_LLAMA && !model.hparams.use_alibi &&
        model.hparams.n_expert == 0 && model.hparams.n_rel_attn_bkts == 0;
}

// Head dimensions used by the existing CUDA/HIP-derived width guard. This
// list does not establish support on other backends or prove GPU correctness.
// GGML_TP5_FA_MASK_KVLEN=1 remains prohibited on GPU/HIP until the documented
// FA kernel issue is fixed.
static bool union_fa_head_dim_generic(uint32_t dk, uint32_t dv) {
    if (dk != dv) {
        return false;
    }
    switch (dk) {
        case 40:
        case 64:
        case 72:
        case 80:
        case 96:
        case 112:
        case 128:
        case 256:
            return true;
        default:
            return false;
    }
}

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
            params.cb_eval != nullptr || params.kv_stream_arena_mib != 0 ||
            // Opt-in, fully bounded Qwen4Exp MTP N=2 target verifier rollback.
            // The model has native (1+n_rs_seq) recurrent state planes, and
            // llama_memory_recurrent::seq_rm checks rollback distance <=2.
            // Keep ALL other tp5-relay admissions exactly as before; never
            // permit an unqualified arch/seq count/batch or switch the default.
            //
            // pillar-1 note: the union tree removed the raw n_rs_seq read here
            // and left the ring to the post-clamp "tp5-relay ring" check below,
            // on the grounds that the clamp is what knows whether the arch can
            // carry the ring. The integration tree keeps this stricter, shape-
            // bounded admission as well: it admits exactly the one MTP ring the
            // relay path has been verified with, and the post-clamp check below
            // still refuses the gdn_replay ring. Both gates are kept; the
            // stricter one decides.
            (params.n_rs_seq != 0 &&
             !(params.n_rs_seq == 2 && model.arch == LLM_ARCH_QWEN4EXP &&
               params.n_seq_max == 1 && params.n_batch == 4 && params.n_ubatch == 4 &&
               std::getenv("GGML_TP5_MTP_RS_RING") &&
               std::strcmp(std::getenv("GGML_TP5_MTP_RS_RING"), "1") == 0)) ||
            params.attention_type == LLAMA_ATTENTION_TYPE_NON_CAUSAL || params.batch_invariant_matmul) {
            throw std::runtime_error("tp5-relay requires 1..4 sequences/rows, bounded profile KV capacity, one ubatch, F16/F32 KV or native Swift FA Q4_0 KV, no sampler, callback or alternate compute policy");
        }
        // The recurrent rollback ring (n_rs_seq) is read AFTER the arch clamp
        // below, on purpose. The clamp is the only place that knows whether
        // the architecture can carry the ring at all
        // (llm_arch_supports_rs_rollback), and a request on an arch without
        // rollback is not a ring: it is clamped to 0 and the relay graph is
        // the stock graph. Reading the raw parameter here would refuse that
        // case as if the ring existed, i.e. it would refuse a configuration
        // the clamp already reduced to stock. The ring check therefore lives
        // in the post-clamp block (see the "tp5-relay ring" check below),
        // which refuses exactly the ring shape the relay path has not been
        // proven to carry. The raw read was ordered wrong, not load-bearing:
        // the user's intent is preserved by the clamp's debug log, and a ring
        // the arch supports reaches the post-clamp check unchanged.
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

    // tp5-relay ring: the relay path carries the recurrent rollback ring only
    // when the ring keeps the stock state layout, i.e. one authoritative
    // state per cell. gdn_replay replaces that layout with a per-seq
    // ingredient ring plus an oldest-edge checkpoint
    // (llama-memory-recurrent.cpp:121-145), so the state tensor the relay
    // graph clears would no longer be the state the model reads. Refuse by
    // name rather than build a graph whose cleared rows are not the rows the
    // model uses. This is the only ring shape the relay path has not been
    // proven to carry; n_rs_seq itself is carried (see graph_params()).
    if (cparams.n_rs_seq > 0 && params.tp5_relay && cparams.gdn_replay) {
        throw std::runtime_error(
            "tp5-relay with n_rs_seq > 0 requires the snapshot ring; gdn_replay (ingredient ring) is not carried by the relay path");
    }
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

    // ENVIRONMENT AUDIT - the arm-distinguishing variables, read here and
    // nowhere else, once per context, before any graph exists:
    //
    //   LLAMA_GRAPH_REUSE_DISABLE  this block. Nonzero selects the stock arm:
    //       no graph reuse and no binder, so every ubatch builds. It is the
    //       one variable that makes the stock arm a genuinely different path,
    //       and the tests that use it keep it set for the context's whole
    //       lifetime (tests/test-nkv-runtime-param.cpp part 3,
    //       tests/test-topology-key-sweep.cpp) because unsetting it after
    //       construction would not change this already-read value but would
    //       mislead a reader about which arm a later context gets.
    //
    //   GGML_TP5_UNION_BINDER      the gate block below. Unset selects the
    //       qualified CPU default; "1" opts in explicitly; "0" is stock.
    //
    //   LLAMA_EXPERIMENTAL_VARLEN_NTOK  the same gate block, read into
    //       ntok_enabled. It selects the token-width lane, which is a
    //       different contract (arena replan) from the n_kv / n_outputs /
    //       rs_z lanes; part 3 unsets it so the matrix measures those three.
    //
    //   LLAMA_NKV_MASK_OVERRIDE    a test hook read per set_input in
    //       llama-graph.cpp; it is not read here and cannot select an arm.
    //
    // None of these is re-read after construction, so a test that sets one
    // before llama_init_from_model() selects the arm for that context's whole
    // lifetime. The tests assert the resulting difference (the build-count
    // deltas) instead of trusting the variable.
    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    const auto model_is_cpu_only = [&model]() {
        const auto is_cpu = [](ggml_backend_dev_t dev) {
            return dev != nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
        };
        if (!is_cpu(model.dev_output())) {
            return false;
        }
        for (int il = 0; il < (int) model.hparams.n_layer_all; ++il) {
            if (!is_cpu(model.dev_layer(il))) {
                return false;
            }
        }
        return true;
    };

    // Default only where the scheduler can have one CPU backend. The live
    // split count and all tensor envelopes are still checked before binding.
    bool cpu_default = model.devices.empty() && !model.has_tensor_overrides() &&
        !params.tp5_relay && !params.quantized_matmul_f32 && !graph_reuse_disable &&
        union_model_shape_supported(model) && sampling.samplers.empty() &&
        (cparams.kv_unified || cparams.n_seq_max == 1) && model_is_cpu_only();
    for (size_t i = 0; cpu_default && i < ggml_backend_dev_count(); ++i) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            cpu_default = false;
        }
    }

    // Unset enables the qualified CPU path only. An explicit value enables
    // the existing opt-in path only when it is exactly "1"; "0" is stock.
    const char * union_binder_env = getenv("GGML_TP5_UNION_BINDER");
    const bool union_binder_gate = llama_union_binder_enabled(union_binder_env, cpu_default);

    {
        // With the gate off, the context allocates no binder state.
        if (union_binder_gate) {
            auto & nkv_st = llama_nkv_ctx_state_get(this);
            nkv_st.nkv_enabled = true;
            const char * ntok_env = getenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK");
            nkv_st.ntok_enabled = ntok_env != nullptr && strcmp(ntok_env, "1") == 0;

            // runtime-length n_kv lane: bind the true used span instead of
            // the stock 256-cell padded width. The 256-cell floor exists only
            // to keep the graph width constant across batches so a stock graph
            // can be reused; the binder provides that constancy by patching
            // the width in place, so the floor is not needed here. This is the
            // default under the umbrella gate; the explicit value "0" selects
            // the padded lane for the A/B control.
            const char * exact_env = getenv("GGML_TP5_UNION_BINDER_EXACT");
            nkv_st.nkv_exact_enabled = exact_env == nullptr || strcmp(exact_env, "1") == 0;

            // n_kv execution width: the graph is built once at the physical
            // per-stream capacity and the true used span is bound in place.
            // The lane is fail-closed: a width the cache cannot prove
            // (streaming arena, reserve context, block-misaligned V) keeps
            // the width the cache applied for that ubatch.
            const bool added_nkv = nkv_st.union_binder.add({
                llama_union_param_kind::n_kv, "n_kv",
                &llama_context::union_tensors_n_kv,
                &llama_context::union_rebind_n_kv,
            });
            const bool added_ntok = nkv_st.union_binder.add({
                llama_union_param_kind::n_tokens, "n_tokens",
                &llama_context::union_tensors_n_kv,
                &llama_context::union_rebind_n_tokens,
            });
            const bool added_rs_z = nkv_st.union_binder.add({
                llama_union_param_kind::rs_z, "rs_z",
                &llama_context::union_tensors_rs_z,
                &llama_context::union_rebind_rs_z,
            });

            // rs_z runtime parameter (GGML_TP5_RSZ_RUNTIME_PARAM): exactly
            // "1" enables it, and only under this umbrella gate. With the
            // lane on, the graph carries the I32 state-selection row and a
            // moved rs_z is an input write; the rs_z table binding above
            // stays registered for the arm-refused case. The value contract
            // is llama_union_gate_exact_one() (llama-union-binder.h), the
            // same one the umbrella gate uses for explicit values.
            {
                nkv_st.rsz_rt_enabled = llama_union_gate_exact_one(getenv("GGML_TP5_RSZ_RUNTIME_PARAM"));
                if (nkv_st.rsz_rt_enabled) {
                    LLAMA_LOG_WARN("%s: rs_z runtime parameter enabled (GGML_TP5_RSZ_RUNTIME_PARAM=1): "
                                   "the recurrent zero-state selection is an input write, not a graph key\n", __func__);
                }
            }

            // backend samplers make build_sampling() derive extra nodes from
            // the gather width; the generic patch can reproduce them, but that
            // path is not proven, so keep the stock exact-count build instead
            const bool nout_ok = sampling.samplers.empty();
            const bool added_nout = nout_ok ? nkv_st.union_binder.add({
                llama_union_param_kind::n_outputs, "n_outputs",
                &llama_context::union_tensors_n_outputs,
                &llama_context::union_rebind_n_outputs,
            }) : true; // not registered: the stock exact-count build stays
            if (!added_nkv || !added_ntok || !added_rs_z || !added_nout) {
                // cannot happen for valid kinds; stay fail-closed anyway
                nkv_st.nkv_enabled = false;
                LLAMA_LOG_ERROR("%s: failed to register the union bindings; runtime graph patching disabled\n", __func__);
            } else {
                LLAMA_LOG_WARN("%s: union runtime binder enabled (%s, %zu param kind(s) registered)\n",
                               __func__, union_binder_env == nullptr ? "CPU default" : "explicit opt-in",
                               nkv_st.union_binder.size());
                if (!nout_ok) {
                    LLAMA_LOG_WARN("%s: n_outputs binding disabled: backend samplers attached (build_sampling derives nodes from the gather width)\n", __func__);
                }
            }
        } else if (llama_union_gate_exact_one(getenv("GGML_TP5_RSZ_RUNTIME_PARAM"))) {
            // rs_z runtime parameter asked for under a gate that is off. The
            // umbrella gate decides whether the binder state exists at all,
            // so there is no per-context slot to count in; the refusal is
            // named here and the lane stays off (stock graph).
            LLAMA_LOG_WARN("%s: GGML_TP5_RSZ_RUNTIME_PARAM=1 ignored: GGML_TP5_UNION_BINDER is off; "
                           "the rs_z runtime lane requires the union binder\n", __func__);
        }
    }

    // the legacy per-lane flag is retired: n_outputs is now a table binding
    // under the umbrella gate. Keep one warning for stale environments.
    {
        const char * legacy = getenv("LLAMA_GRAPH_UNION_OUTPUTS_RUNTIME");
        if (legacy != nullptr && strcmp(legacy, "1") == 0) {
            LLAMA_LOG_WARN("%s: LLAMA_GRAPH_UNION_OUTPUTS_RUNTIME=1 is retired; n_outputs binding is controlled by GGML_TP5_UNION_BINDER=1\n", __func__);
        }
    }
    // stock rounds n_ctx up to the 256-cell KV block. The binder gate keeps
    // the raw value only when the exact sub-lane is active; with the gate off
    // or the exact sub-lane explicitly disabled (GGML_TP5_UNION_BINDER_EXACT=0)
    // this must stay byte-identical to stock (the original patch dropped the
    // pad unconditionally, which changed OFF for any n_ctx that is not a
    // multiple of 256).
    if (!union_binder_gate || !llama_nkv_ctx_state_get(this).nkv_exact_enabled) {
        // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
        cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);
    }

    // The stock capacity sites pad to a 256-cell bucket. The runtime-length
    // lane uses the requested values instead: the KV cache only requires
    // n_pad alignment (1 for a plain cache), and the n_kv graph width is
    // bound to the true used span by the union binder, so the physical
    // capacity is not a graph-width contract. The model's device assignment
    // is fixed at this point (dev_layer/dev_output).
    // The aligned-capacity guard is kept for GPU flash-attention head
    // dimensions outside the CUDA/HIP-derived list. It is not a proof for
    // other backends.
    const bool gpu_fa_needs_aligned_capacity = [&]() {
        if (!cparams.flash_attn || model_is_cpu_only()) {
            return false;
        }
        for (uint32_t il = 0; il < model.hparams.n_layer_all; ++il) {
            if (!model.hparams.has_kv(il)) {
                continue;
            }
            if (!union_fa_head_dim_generic(model.hparams.n_embd_head_k(il), model.hparams.n_embd_head_v(il))) {
                return true;
            }
        }
        return false;
    }();
    // One sub-lane controls every runtime-length replacement: the exact n_kv
    // width and the two capacity sites. GGML_TP5_UNION_BINDER_EXACT=0 is
    // therefore a full stock-shape control (padded n_kv + stock 256 capacity),
    // not just an n_kv control.
    const bool runtime_len_capacity = union_binder_gate &&
        llama_nkv_ctx_state_get(this).nkv_exact_enabled && !gpu_fa_needs_aligned_capacity;

    // rope long/short factor selection is keyed on the stock 256-cell bucket
    // of the per-sequence window (a model-training property). Keep that value
    // even when the capacity bucket itself is not applied, so the runtime-
    // length lane cannot change logits through the rope factors.
    const uint32_t n_ctx_stock = GGML_PAD(cparams.n_ctx, 256);
    cparams.n_ctx_seq_rope = cparams.kv_unified ? n_ctx_stock
                                                : GGML_PAD(n_ctx_stock / cparams.n_seq_max, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        if (runtime_len_capacity) {
            // runtime length: ceil division keeps the requested n_ctx covered
            // by the per-sequence windows; n_ctx itself is never reduced.
            cparams.n_ctx_seq = (cparams.n_ctx + cparams.n_seq_max - 1) / cparams.n_seq_max;
            if (cparams.n_ctx_seq == 0) {
                throw std::runtime_error("n_ctx_seq == 0");
            }
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
        // Runtime length: the physical KV capacity is a runtime parameter,
        // not a 256-cell bucket. The binder envelope is built at the exact
        // capacity and every width up to it binds in place. The KV cache only
        // requires n_pad alignment, and attn_n_pad == 1024 is selected only
        // when the capacity is already 1024-aligned, so the exact value is
        // allocatable. A GPU FA consumer with a hard-gated head dim keeps the
        // stock 256 pad: the padded lane clamps its last bucket to the
        // capacity, so a non-256 capacity would hand it a non-256 K length.
        cparams.n_ctx_kv = runtime_len_capacity ? params.n_ctx_kv : GGML_PAD(params.n_ctx_kv, 256);
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
            // Pillar 5: a relay context reserves the full output envelope here,
            // before the first decode. Every decode passes n_outputs <=
            // cparams.n_outputs_max (process_ubatch refuses more), so the
            // first decode can no longer grow the host output buffer. That
            // grow freed the old buffer with hipHostFree while another
            // window's kernel launch held the ROCm runtime mutex, and the
            // HSA signal wait deadlocked (H2e). The stock path keeps its
            // original lazy initial size.
            // GGML_TP5_RELAY_OUTPUT_ROWS lets the owner name the first
            // catch-up row count (the compose runtime knows it before context
            // creation). Without it the legacy bound is used; a later late
            // set_embeddings_nextn/sampler change can still grow once, which
            // is safe because the old buffer is retired, not freed.
            const uint32_t n_outputs_reserve = backend_tp5_relay != nullptr
                ? llama_shape_cache::output_rows_bound(
                      cparams.n_outputs_max, n_seq_max(), getenv("GGML_TP5_RELAY_OUTPUT_ROWS"))
                : params.n_seq_max;
            if (output_reserve(int32_t(n_outputs_reserve), backend_tp5_relay != nullptr) < n_outputs_reserve) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                           ggml_backend_buffer_name(buf_output.get()),
                           ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }

        // Pillar 5: per-shape scheduler/graph cache. The flag value is the
        // slot count; 0 keeps the single-scheduler path. Relay contexts get a
        // two-slot default because the stock target graph alternates between
        // the 1-row decode shape and the 2-row MTP verify shape (1 accepted +
        // 1 draft), and a single gf_res_prev slot rebuilds and re-allocates on
        // every switch. Two slots are enough for that working set; larger
        // working sets can raise the value explicitly. The cache is no longer
        // relay-only: an explicit value also enables it on non-relay stock
        // contexts (make_shape_plan fails closed and falls back if the context
        // cannot host a per-shape scheduler).
        {
            const char * plan_env = getenv("GGML_TP5_LLAMA_PLAN_CACHE");
            const size_t cap      = llama_shape_cache::parse_capacity(plan_env);
            const bool   plan_default = plan_env == nullptr && backend_tp5_relay != nullptr;
            shape_plan_active = cap != 0 || plan_default;
            shape_plan_cache  = llama_shape_cache::cache(plan_default ? 2 : cap);
            if (plan_env != nullptr && cap == 0) {
                LLAMA_LOG_WARN("%s: GGML_TP5_LLAMA_PLAN_CACHE value is invalid or 0; "
                               "per-shape plan cache disabled\n", __func__);
            } else if (shape_plan_active) {
                LLAMA_LOG_INFO("%s: per-shape plan cache enabled, slots=%zu%s\n", __func__,
                               shape_plan_cache.capacity(), plan_default ? " (relay default)" : "");
            }
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
            if (kv_stream_phase_arena.arena == nullptr ||
                !arena_set_compute_fn(kv_stream_phase_arena.arena, kv_stream_minimum_stage_bytes,
                                      kv_stream_arena_bytes - kv_stream_minimum_stage_bytes)) {
                throw std::runtime_error("failed to create CUDA block KV streaming phase arena");
            }
            kv_stream_phase_arena.phase.current = {
                kv_stream_minimum_stage_bytes,
                kv_stream_minimum_stage_bytes,
                kv_stream_arena_bytes - kv_stream_minimum_stage_bytes,
                kv_stream_phase_arena.minimum_ring_slots,
            };
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

    // factor-prefab envelopes (default off; throws -> context creation fails closed)
    prefab_init();

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
    if (has_unproven()) {
        // llama_free retains the whole context (and increments the model's
        // retained-context count); a direct delete cannot keep the external
        // model and context-owned buffers alive. This covers the main sched
        // and every prefab envelope scheduler. Fail loudly instead of freeing
        // behind an in-flight copy.
        LLAMA_LOG_ERROR("%s: pending context destroyed directly; llama_free must retain it\n", __func__);
        GGML_ABORT("pending llama_context destroyed directly");
    }

    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();
    // Per-shape schedulers own native work too; drain and release them before
    // the backends go away.
    flush_shape_plans();

    // envelope scheds/graphs and the factor cache reference this context's backends and memory:
    // drop them first (the factor cache is context-owned, nothing outlives the context)
    prefab.reset();

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
    // One-line summary at context destruction, only when the binder is enabled.
    // The NKV_BINDER_STATS prefix is also used by external gate harnesses.
    // (state note: the counters live in the context's own state slot, not in
    // class data members)
    {
        // peek only: never create state during destruction (with the gate off
        // no slot exists and no stats line prints - stock OFF)
        const llama_nkv_ctx_state * st = union_binder_state_peek();
        if (st != nullptr && st->nkv_enabled) {
            LLAMA_LOG_INFO("NKV_BINDER_STATS attempts=%zu applied=%zu refused_align=%zu refused_skeleton=%zu refused_envelope=%zu refused_relay=%zu refused_pipeline=%zu refused_rs_z=%zu refused_nrs=%zu refused_reader=%zu refused_backend=%zu refused_iswa=%zu refused_nkv_unknown=%zu refused_nout=%zu refused_topology=%zu fallbacks=%zu env_builds=%zu graph_builds=%zu shape_analyses=%zu shape_fail=%zu fused_qkv_refused=%zu patch_binds=%zu cap=%u cap_nout=%u cap_nrs=%u env_disabled=%d nout_grow_binds=%zu nout_grow_rebuilds=%zu ntok_fallbacks=%zu ntok_recurrent_hybrid_fallbacks=%zu rsz_rt=%d rsz_rt_active=%d rsz_rt_fallbacks=%zu rsz_rt_refused_gate=%zu rsz_rt_refused_arch=%zu rsz_rt_refused_no_rs_z=%zu rsz_rt_refused_cell=%zu rsz_rt_refused_envelope=%zu rsz_rt_refused_shape=%zu rsz_rt_refused_n_rs_seq=%zu rsz_rt_refused_ring=%zu rsz_rt_refused_mixed_depth=%zu rsz_rt_bound_rs_z=%d rsz_rt_zero_view_rows=%u\n",
                           st->c_attempts, st->c_applied, st->c_refused_align, st->c_refused_skeleton,
                           st->c_refused_envelope, st->c_refused_relay, st->c_refused_pipeline,
                           st->c_refused_rs_z, st->c_refused_nrs, st->c_refused_reader, st->c_refused_backend,
                           st->c_refused_iswa, st->c_refused_nkv_unknown, st->c_refused_nout,
                           st->c_refused_topology,
                           st->c_fallbacks, st->c_env_builds, st->c_graph_builds,
                           st->c_shape_analyses, st->c_shape_fail, st->c_fused_qkv_refused, st->c_patch_binds,
                           st->record_cap, st->record_cap_nout, st->record_cap_nrs, st->env_disabled ? 1 : 0,
                           st->c_nout_grow_binds, st->c_nout_grow_rebuilds,
                           st->c_ntok_fallbacks, st->c_ntok_recurrent_hybrid_fallbacks,
                           st->rsz_rt_enabled ? 1 : 0, st->rsz_rt_active ? 1 : 0,
                           st->c_rsz_rt_fallbacks, st->c_rsz_rt_refused_gate,
                           st->c_rsz_rt_refused_arch, st->c_rsz_rt_refused_no_rs_z,
                           st->c_rsz_rt_refused_cell, st->c_rsz_rt_refused_envelope,
                           st->c_rsz_rt_refused_shape, st->c_rsz_rt_refused_n_rs_seq,
                           st->c_rsz_rt_refused_ring, st->c_rsz_rt_refused_mixed_depth,
                           st->rsz_rt_bound_rs_z, st->rsz_rt_zero_view_rows);
        }
    }
    ggml_opt_free(opt_ctx);

    // Members destroy in reverse declaration order, so the backends would be
    // freed before the scheduler. Free the scheduler first, while the backends
    // are still alive; sched_free never dereferences them afterwards.
    sched.reset();
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
    // The factor-prefab lane rejects a context with KV-stream targets at
    // init. Keep the invariant explicit: a prefab slot scheduler would own
    // backends and graph buffers the phase arena is about to repartition.
    if (prefab) {
        LLAMA_LOG_ERROR("%s: prefab and the KV-stream phase arena are mutually exclusive\n", __func__);
        return false;
    }
    if (arena.phase.inconsistent) {
        LLAMA_LOG_ERROR("%s: phase arena is inconsistent after a failed rollback; "
                        "refusing further switches\n", __func__);
        return false;
    }
    // Any unproven owner is refused before the early reuse. The early return
    // does not replace owners, but the same rule must hold on both paths:
    // has_unproven() covers the main scheduler, prefab slots/pending and every
    // per-shape scheduler.
    if (has_unproven()) {
        LLAMA_LOG_ERROR("%s: refusing to switch phases while a scheduler has an unproven submission\n", __func__);
        arena.phase.phase_ready = false;
        return false;
    }
    if (sched && arena.phase.can_reuse(decode)) {
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
    const llama_kv_stream_phase_layout target_layout = {
        target.kv_bytes, target.compute_offset, target.compute_bytes, target.ring_slots,
    };
    auto & phase = arena.phase;

    // From here the old phase is not reusable: its scheduler is about to be
    // dropped. Any failure leaves phase_ready false so the next call rebuilds.
    phase.phase_ready = false;

    synchronize();
    if (sched && !arena.graph_reset_fn(backend_ptrs[arena.backend_index])) {
        LLAMA_LOG_ERROR("%s: failed to invalidate CUDA graphs before arena repartition\n", __func__);
        return false;
    }

    // Releasing the scheduler returns the only borrowed compute lease.
    // Per-shape schedulers were placed in the old generation; drop them.
    flush_shape_plans();
    sched.reset();
    ++alloc_gen_; // Placement is rebuilt from scratch below.
    gf_res_prev->reset();
    union_record_invalidate();

    const llama_kv_stream_phase_layout old = phase.current;
    const auto rollback_compute = [&]() {
        return arena.set_compute_fn(arena.arena, old.compute_offset, old.compute_bytes);
    };

    if (target_layout.kv_bytes > old.kv_bytes) {
        // Moving compute upward frees the bytes before KV claims them.
        if (!arena.set_compute_fn(arena.arena, target_layout.compute_offset, target_layout.compute_bytes)) {
            LLAMA_LOG_ERROR("%s: failed to move arena compute slice upward\n", __func__);
            return false;
        }
        if (!kv->kv_stream_resize_pool(target_layout.kv_bytes, active_tokens, target_layout.ring_slots)) {
            if (!rollback_compute()) {
                phase.inconsistent = true;
                LLAMA_LOG_ERROR("%s: failed to restore the old arena compute slice after a failed grow; "
                                "marking the arena inconsistent\n", __func__);
            }
            LLAMA_LOG_ERROR("%s: failed to grow arena KV slice\n", __func__);
            return false;
        }
    } else if (target_layout.kv_bytes < old.kv_bytes) {
        // KV releases the bytes before compute expands downward.
        if (!kv->kv_stream_resize_pool(target_layout.kv_bytes, active_tokens, target_layout.ring_slots)) {
            LLAMA_LOG_ERROR("%s: failed to shrink arena KV slice\n", __func__);
            return false;
        }
        if (!arena.set_compute_fn(arena.arena, target_layout.compute_offset, target_layout.compute_bytes)) {
            if (!kv->kv_stream_resize_pool(old.kv_bytes, active_tokens, old.ring_slots)) {
                phase.inconsistent = true;
                LLAMA_LOG_ERROR("%s: failed to restore the old arena KV slice after a failed shrink; "
                                "marking the arena inconsistent\n", __func__);
            }
            LLAMA_LOG_ERROR("%s: failed to move arena compute slice downward\n", __func__);
            return false;
        }
    } else {
        if (!arena.set_compute_fn(arena.arena, target_layout.compute_offset, target_layout.compute_bytes)) {
            LLAMA_LOG_ERROR("%s: failed to update arena compute slice\n", __func__);
            return false;
        }
        if (!kv->kv_stream_resize_pool(target_layout.kv_bytes, active_tokens, target_layout.ring_slots)) {
            if (!rollback_compute()) {
                phase.inconsistent = true;
                LLAMA_LOG_ERROR("%s: failed to restore the old arena compute slice after a failed layout update; "
                                "marking the arena inconsistent\n", __func__);
            }
            LLAMA_LOG_ERROR("%s: failed to update arena KV layout\n", __func__);
            return false;
        }
    }

    // The physical split is the target. Record that fact, but keep the phase
    // not-ready until its scheduler and reserved graph exist.
    phase.current = target_layout;

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), arena.max_nodes,
                                       cparams.pipeline_parallel, cparams.op_offload));
    if (!sched) {
        LLAMA_LOG_ERROR("%s: failed to recreate scheduler\n", __func__);
        return false;
    }
    // ggml_backend_sched_new created an environment-default MoE session.
    // Re-apply the context's resolved policy so a phase switch cannot silently
    // change the user's off/budget or fall back to that default.
    if (!moe_cache_configured) {
        LLAMA_LOG_ERROR("%s: no resolved MoE cache configuration for the phase scheduler; "
                        "refusing the environment default\n", __func__);
        sched.reset();
        return false;
    }
    if (!llama_apply_moe_cache_config(sched.get(), llama_to_ggml_moe_cache_mode(moe_cache_mode_resolved),
                                      moe_cache_budget_mib)) {
        LLAMA_LOG_ERROR("%s: failed to apply the context's MoE cache configuration to the phase scheduler\n", __func__);
        sched.reset();
        return false;
    }

    auto reserve_mctx = memory->init_full();
    if (!reserve_mctx) {
        LLAMA_LOG_ERROR("%s: failed to initialize full memory for phase reserve\n", __func__);
        sched.reset();
        return false;
    }

    const uint32_t n_seqs    = n_seq_max_active();
    const uint32_t n_tokens  = decode ? n_seqs : std::min(cparams.n_ctx, cparams.n_ubatch);
    const uint32_t n_outputs = decode ? n_seqs : std::min(n_tokens, cparams.n_outputs_max);
    auto *         gf        = graph_reserve(n_tokens, n_seqs, n_outputs, reserve_mctx.get());
    if (gf == nullptr) {
        LLAMA_LOG_ERROR("%s: failed to reserve %s compute graph in phase arena\n", __func__,
                        decode ? "decode" : "prefill");
        sched.reset();
        return false;
    }

    backend_buf_exp_size = target.backend_sizes;
    phase.decode         = decode;
    phase.phase_ready    = true;
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
    ++alloc_gen_; // A new reserve may recreate the scheduler and its placement.

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    // drain every envelope still in flight before the main sched is touched or any slot is
    // reset; synchronize() below then only has the main sched left to wait for
    if (prefab) {
        prefab->sync_pending();
    }
    synchronize();

    if (prefab) {
        // cparams may have changed (warmup, causal, ...): envelopes are recomposed on next use,
        // factor modules survive iff the topology key still matches
        prefab->reset_slots();
    }

    if (kv_stream_phase_arena.configured && sched && kv_stream_phase_arena.backend_index < backend_ptrs.size()) {
        if (!kv_stream_phase_arena.graph_reset_fn(backend_ptrs[kv_stream_phase_arena.backend_index])) {
            throw std::runtime_error("failed to reset CUDA graphs before measuring phase arena");
        }
        kv_stream_phase_arena.phase.phase_ready = false;
        sched.reset();
    }

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs   = n_seq_max_active();
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    flush_shape_plans();
    gf_res_prev.reset(new llm_graph_result(max_nodes));
    union_record_invalidate();
    // the new long-lived result object is owned by this context: the union
    // binder state uses this back-pointer (llama-nout-state.cpp) instead of a
    // global result -> context reverse map
    gf_res_prev->set_owner(this);
    gf_res_reserve.reset(new llm_graph_result(max_nodes));
    gf_res_reserve->set_owner(this); // same owner invariant for the reserve result

    const bool moe_cache_eligible = llama_model_has_cacheable_moe_weights(
        model, cparams.moe_cache_mode, cparams.moe_cache_budget_mib, backend_ptrs);
    // Persist the resolved policy before any scheduler is created. Every
    // recreated scheduler (reserve retry, per-shape plan, KV-stream phase
    // switch) re-applies these values instead of keeping the default session.
    moe_cache_mode_resolved = moe_cache_eligible ? cparams.moe_cache_mode : LLAMA_MOE_CACHE_MODE_OFF;
    moe_cache_budget_mib    = cparams.moe_cache_budget_mib;
    moe_cache_configured    = true;
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
    if (!llama_apply_moe_cache_config(sched.get(), llama_to_ggml_moe_cache_mode(moe_cache_mode_resolved),
                                      moe_cache_budget_mib)) {
        sched.reset();
        throw std::runtime_error("failed to apply the resolved MoE cache configuration");
    }

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

        kv_stream_phase_arena.phase.phase_ready = false;
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

    // TP5 max-footprint share (GGML_TP5_MAXFP_SHARE=1, exact "1"): reserve the
    // compute buffers ONCE at the max footprint of the {pp, tg} shape set
    // instead of the stock pp -> tg -> pp triple. The stock triple re-places
    // tensors on every reserve and relies on the allocator's grow-only chunk
    // logic to (hopefully) cover the other shape; the max-footprint reserve
    // measures both shapes, takes the per-backend max, and reserves the final
    // pp graph with that total-size floor. A total-size floor alone does not
    // guarantee that a different graph's chunk placement will fit.
    const char * maxfp_env = getenv("GGML_TP5_MAXFP_SHARE");
    const bool   maxfp_share = maxfp_env != nullptr && strcmp(maxfp_env, "1") == 0;
    if (maxfp_share && !model.hparams.no_alloc) {
        std::vector<size_t> sizes_pp(backend_ptrs.size(), 0);
        std::vector<size_t> sizes_tg(backend_ptrs.size(), 0);

        // A new scheduler has different graph copies and placement. Remeasure
        // both shapes after disabling pipeline parallelism before reserving.
        for (;;) {
            // Measure both shapes without releasing the live buffers.
            {
                auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), true, sizes_pp.data(), nullptr, true);
                if (!gf) {
                    throw std::runtime_error("failed to measure compute pp buffers");
                }
                n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
                n_nodes_pp  = ggml_graph_n_nodes(gf);
            }
            {
                auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), true, sizes_tg.data(), nullptr, true);
                if (!gf) {
                    throw std::runtime_error("failed to measure compute tg buffers");
                }
                n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
                n_nodes_tg  = ggml_graph_n_nodes(gf);
            }

            std::vector<size_t> sizes_max(backend_ptrs.size(), 0);
            for (size_t i = 0; i < backend_ptrs.size(); ++i) {
                sizes_max[i] = std::max(sizes_pp[i], sizes_tg[i]);
            }

            auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), false, nullptr, sizes_max.data());
            if (!gf) {
                if (!cparams.pipeline_parallel) {
                    throw std::runtime_error("failed to allocate compute pp buffers");
                }
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n",
                               __func__);
                cparams.pipeline_parallel = false;
                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                                                   max_nodes, false, cparams.op_offload));
                if (!llama_apply_moe_cache_config(sched.get(), llama_to_ggml_moe_cache_mode(moe_cache_mode_resolved),
                                                  moe_cache_budget_mib)) {
                    sched.reset();
                    throw std::runtime_error("failed to apply the resolved MoE cache configuration after scheduler retry");
                }
                continue;
            }
            break;
        }

        LLAMA_LOG_INFO("%s: max-footprint share enabled (GGML_TP5_MAXFP_SHARE=1): single reserve at max(pp, tg) per backend\n",
                       __func__);
    } else {
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
                if (!llama_apply_moe_cache_config(sched.get(), llama_to_ggml_moe_cache_mode(moe_cache_mode_resolved),
                                                  moe_cache_budget_mib)) {
                    sched.reset();
                    throw std::runtime_error("failed to apply the resolved MoE cache configuration after scheduler retry");
                }
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

    if (prefab) {
        // every envelope executed since the last synchronize, not just the last active one: a
        // ubatch may have switched slots, or fallen back to stock and cleared `active`
        prefab->sync_pending();
    }
    ggml_backend_sched_synchronize(sched.get());

    synchronize_accounting();
}

void llama_context::synchronize_accounting() {
    if (!sched) {
        return;
    }

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

size_t llama_context::tp5_output_copy_record() {
    const llm_graph_result * current = gf_res_current ? gf_res_current : gf_res_prev.get();
    if (backend_tp5_relay == nullptr || current == nullptr) {
        return 0;
    }
    const ggml_tensor * t_logits = current->get_logits();
    if (t_logits == nullptr || t_logits->buffer == nullptr) {
        return 0;
    }
    // A host-resident output is not copied by a device at all; no event on a
    // device stream can be its barrier. Refuse and keep the full synchronize.
    if (ggml_backend_buffer_is_host(t_logits->buffer)) {
        return 0;
    }
    // ggml records on exactly the streams its D2H readback uses (MIRRORED reads
    // the first child, a split axis copies from every contributing child) and
    // fails closed for PARTIAL/unknown output or a missing event interface.
    return ggml_backend_meta_output_record(backend_tp5_relay, t_logits);
}

size_t llama_context::tp5_output_copy_wait() {
    if (backend_tp5_relay == nullptr) {
        return 0;
    }
    const size_t complete = ggml_backend_meta_output_wait(backend_tp5_relay);
    // Only a wait that actually released recorded events is this epoch's
    // completion barrier. With no event (gate off, refused record, or an
    // already-drained epoch) nothing was synchronized, so the eval bookkeeping
    // of synchronize() must NOT run: that would claim the queued tokens were
    // synchronized when no host wait happened.
    if (complete > 0) {
        synchronize_accounting();
    }
    return complete;
}

bool llama_context::release_compute() {
    // A phase arena owns a joint KV/compute layout; releasing its allocator
    // without the arena protocol would invalidate that layout. Optimizer
    // graphs also have a separate lifetime. Reject both before any mutation.
    if (!sched || model.hparams.no_alloc || opt_ctx != nullptr ||
            kv_stream_phase_arena.arena != nullptr || kv_stream_phase_arena.configured) {
        return false;
    }
    if (ggml_backend_sched_has_unproven(sched.get())) {
        LLAMA_LOG_ERROR("%s: refusing to release compute buffers: an async submission is still unproven\n", __func__);
        return false;
    }

    synchronize();
    if (prefab) {
        prefab->reset_slots();
        for (auto & s : prefab->slots) {
            s.sched.reset();
        }
        prefab->cache.clear();
    }
    // Per-shape entries hold their own scheduler and graph metadata; release
    // them before the shared allocator/buffers they reference are freed.
    flush_shape_plans();
    // These own the only inference graph metadata that may reuse allocator
    // addresses. Reset both before the allocator is freed, retaining their
    // graph-result objects for the normal next graph build.
    if (gf_res_prev) {
        gf_res_prev->reset();
        union_record_invalidate();
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

bool llama_context::nkv_runtime_stats(llama_nkv_runtime_stats * stats) const {
    if (!stats) return false;
    *stats = {};
    const auto * st = union_binder_state_peek();
    if (!st || !st->nkv_enabled || !st->record_valid) return false;
    stats->topology_hash     = st->topology_hash;
    stats->topology_hashed   = st->topology_hashed;
    stats->build_n_kv        = st->width_table.build_n_kv;
    stats->runtime_n_kv      = st->width_table.runtime_n_kv;
    stats->bound_n_kv        = st->record_bound;
    stats->build_n_outputs   = st->width_table.build_n_outputs;
    stats->bound_n_outputs   = st->record_bound_nout;
    stats->runtime_n_outputs = n_outputs;
    stats->bound_n_rs        = st->record_bound_n_rs;
    stats->bound_rs_head     = st->record_bound_rs_head;
    stats->bound_rs_z        = st->record_bound_rs_z;
    stats->has_nkv_dims      = st->width_table.has_nkv;
    stats->has_nout_dims     = st->width_table.has_nout;
    stats->has_nrs_dims      = st->width_table.has_nrs;
    // The n_outputs refusal counter. It moves on the one refusal that is left
    // - a request above the recorded envelope on a lane that cannot replan
    // (llama-context.cpp, union_rebind_n_outputs) - and not on a zero-output
    // step, which is an ordinary shrink the bind applies. It stays exported,
    // not just logged, because test-ntok-onegraph reads it to prove that a
    // zero-output step binds instead of rebuilding.
    stats->refused_nout      = st->c_refused_nout;
    // The mask observation belongs to the live graph's runtime n_kv input.
    // When the graph has none (the binder built no runtime input for this
    // topology) the five counters stay 0, which reads as "nothing was
    // filled", never as "filled correctly".
    if (const llm_graph_result * res = gf_res_prev.get()) {
        if (const llm_graph_input_nkv_runtime * rt = res->get_nkv_runtime_input()) {
            stats->mask_n_kv        = rt->mask_n_kv;
            stats->mask_filled      = rt->mask_filled;
            stats->mask_refused     = rt->mask_refused;
            stats->mask_row_checked = rt->mask_row_checked;
            stats->mask_row_ok      = rt->mask_row_ok;
            stats->mask_row_refused = rt->mask_row_refused;
            stats->mask_rows_all    = rt->mask_rows_all;
        }
    }
    stats->graph_builds = n_graph_builds;
    // The token lane's observation. build_n_tokens is 0 on every other lane,
    // which reads as "no token-width runtime dimension", never as "bound at
    // width 0".
    stats->build_n_tokens = st->width_table.build_n_tokens;
    stats->bound_n_tokens = st->record_bound_ntok;
    stats->has_ntok_dims  = st->width_table.has_ntok;
    // The CPU attention work, read from the backend that owns the counters.
    // A non-CPU backend reports zeros, which reads as "no CPU attention ran".
    if (backend_cpu != nullptr) {
        ggml_backend_cpu_attn_reduce_stats(backend_cpu,
                &stats->attn_live_elems, &stats->attn_pad_elems, &stats->attn_rows);
    }
    return true;
}

bool llama_context::ntok_plan_stats(llama_ntok_plan_stats * stats) const {
    if (!stats) return false;
    *stats = {};
    const auto * st = union_binder_state_peek();
    if (!st || !st->record_valid || !st->shape_map.ntok_valid || !union_sched_cpu_only()) return false;
    stats->graph_builds = st->c_graph_builds;
    stats->active_tokens = st->record_bound_ntok;
    stats->active_kv = st->record_bound;
    stats->active_outputs = st->record_bound_nout;
    stats->arena_bytes = ggml_backend_sched_get_buffer_size(sched.get(), backend_cpu);
    auto buft = ggml_backend_sched_get_buffer_type(sched.get(), backend_cpu);
    std::vector<uint8_t> seen(st->record_t.size(), 0);
    auto account = [&](const std::vector<llama_union_shape_coeff> & coeffs) {
        for (const auto & coeff : coeffs) {
            const size_t i = coeff.tensor;
            if (i >= st->record_t.size()) return false;
            ggml_tensor * t = st->record_t[i];
            if (seen[i] || st->record_fixed[i] || t->view_src) continue;
            seen[i] = 1;
            size_t planned = 0;
            if (!ggml_backend_sched_get_tensor_plan_size(sched.get(), t, &planned)) return false;
            stats->planned_bytes += planned;
            const size_t active = ggml_backend_buft_get_alloc_size(buft, t);
            stats->active_bytes += active;
            if (planned != active) ++stats->mismatched_tensors;
            ++stats->checked_tensors;
        }
        return true;
    };
    if (!account(st->shape_map.ntok) || !account(st->shape_map.nout) ||
            !account(st->shape_map.nkv) || !account(st->shape_map.nkv_ntok)) return false;
    bool have_mask = false, have_outputs = false;
    for (ggml_tensor * tensor : st->record_t) {
        if (strcmp(tensor->name, "attn_inp_kq_mask") == 0) {
            size_t bytes = 0;
            if (!ggml_backend_sched_get_tensor_plan_size(sched.get(), tensor, &bytes)) return false;
            stats->kv_mask_bytes = bytes;
            have_mask = true;
        } else if (strcmp(tensor->name, "union_out_ids") == 0) {
            size_t bytes = 0;
            if (!ggml_backend_sched_get_tensor_plan_size(sched.get(), tensor, &bytes)) return false;
            stats->output_ids_bytes = bytes;
            have_outputs = true;
        }
    }
    for (const auto & input : gf_res_prev->inputs) {
        const auto * embd = dynamic_cast<const llm_graph_input_embd *>(input.get());
        if (embd && embd->tokens) {
            size_t bytes = 0;
            if (!ggml_backend_sched_get_tensor_plan_size(sched.get(), embd->tokens, &bytes)) return false;
            stats->token_input_bytes = bytes;
            break;
        }
    }
    return stats->checked_tensors > 0 && stats->token_input_bytes > 0 && have_mask && have_outputs;
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
    memory_update_failed = false;
    memory_update_error  = 0;

    if (!memory) {
        return false;
    }
    if (has_unproven()) {
        LLAMA_LOG_ERROR("%s: refusing memory update while an async submission is unproven\n", __func__);
        memory_update_failed = true;
        memory_update_error  = -3;
        return false;
    }

    {
        llama_memory_context_ptr mctx;
        try {
            mctx = memory->init_update(this, optimize);
        } catch (const std::exception & err) {
            LLAMA_LOG_ERROR("%s: failed to initialize memory update: %s\n", __func__, err.what());
            memory_update_failed = true;
            memory_update_error  = -2;
            return false;
        } catch (...) {
            LLAMA_LOG_ERROR("%s: failed to initialize memory update: unknown exception\n", __func__);
            memory_update_failed = true;
            memory_update_error  = -2;
            return false;
        }
        if (!mctx) {
            LLAMA_LOG_ERROR("%s: failed to initialize memory update\n", __func__);
            memory_update_failed = true;
            memory_update_error  = -2;
            return false;
        }
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
                    memory_update_failed = true;
                    memory_update_error  = -2;
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
        union_record_invalidate();

        bool apply_attempted = false;
        bool apply_failed    = false;
        bool no_update       = false;
        apply_attempted = true;
        try {
            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
                apply_failed = true;
            } else {
                const auto update_status = mctx->get_update_status();
                if (update_status == LLAMA_MEMORY_APPLY_FAILED) {
                    LLAMA_LOG_ERROR("%s: memory update reported FAILED\n", __func__);
                    apply_failed = true;
                } else if (update_status == LLAMA_MEMORY_APPLY_NO_UPDATE) {
                    // no state change; no graph reserve is needed
                    no_update = true;
                } else if (has_unproven()) {
                    LLAMA_LOG_ERROR("%s: memory update left an unproven submission\n", __func__);
                    apply_failed = true;
                }
            }
        } catch (const std::exception & err) {
            LLAMA_LOG_ERROR("%s: memory update threw: %s\n", __func__, err.what());
            apply_failed = true;
        } catch (...) {
            LLAMA_LOG_ERROR("%s: memory update threw: unknown exception\n", __func__);
            apply_failed = true;
        }

        // A failed apply can already have submitted an async K-shift graph.
        // Wait on the main scheduler on every apply exit, before returning or
        // entering graph_reserve. The void wait is a completion barrier only:
        // it is not an async-success receipt, and has_unproven() is not a
        // complete GPU completion proof.
        if (apply_attempted && sched) {
            ggml_backend_sched_synchronize(sched.get());
        }
        if (apply_failed) {
            memory_update_failed = true;
            memory_update_error  = -3;
            return false;
        }
        if (no_update) {
            return false;
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        llama_memory_context_ptr mctx;
        try {
            mctx = memory->init_full();
        } catch (const std::exception & err) {
            LLAMA_LOG_ERROR("%s: failed to initialize memory context for reserve: %s\n", __func__, err.what());
            memory_update_failed = true;
            memory_update_error  = -2;
            return false;
        } catch (...) {
            LLAMA_LOG_ERROR("%s: failed to initialize memory context for reserve: unknown exception\n", __func__);
            memory_update_failed = true;
            memory_update_error  = -2;
            return false;
        }
        if (!mctx) {
            LLAMA_LOG_ERROR("%s: failed to initialize memory context for reserve\n", __func__);
            memory_update_failed = true;
            memory_update_error  = -2;
            return false;
        }

        const uint32_t n_seqs   = n_seq_max_active();
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
            memory_update_failed = true;
            memory_update_error  = -2;
            return false;
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

bool llama_context::has_unproven() const {
    if (sched && ggml_backend_sched_has_unproven(sched.get())) {
        return true;
    }
    // Envelope slots own independent schedulers over the same backends. A
    // backend with per-scheduler state can be unproven there while the main
    // scheduler is clean, so every slot and every recorded in-flight slot is
    // part of the context's state.
    if (prefab) {
        for (const llama_prefab_slot & slot : prefab->slots) {
            if (slot.sched && ggml_backend_sched_has_unproven(slot.sched.get())) {
                return true;
            }
        }
        for (ggml_backend_sched_t pending : prefab->pending) {
            if (pending && ggml_backend_sched_has_unproven(pending)) {
                return true;
            }
        }
    }
    // Per-shape schedulers are independent owners too. A shape plan can be the
    // actual submitter, and the destructor/flush paths must not synchronize an
    // unproven submission.
    for (const auto & plan : shape_plans) {
        if (plan && plan->sched && ggml_backend_sched_has_unproven(plan->sched.get())) {
            return true;
        }
    }
    return false;
}

bool llama_context::retain_for_process() {
    // A duplicate llama_free must not increment the retained-context count
    // again. The context is already process-owned.
    if (retained.exchange(true)) {
        return false;
    }
    // The threadpools are caller-owned; detach so the caller may free them.
    // The CPU backend holds its own raw pointer, so clear that too; clearing
    // never pauses the caller's pool.
    // The context no longer computes after this point.
    detach_threadpool();
    set_backend_cpu_threadpool(nullptr);
    model.retain_context();
    return true;
}

void llama_context::debug_mark_unproven() {
    if (sched) {
        ggml_backend_sched_mark_unproven(sched.get());
    }
}

bool llama_context::debug_mark_prefab_slot_unproven(size_t index) {
    if (!prefab || index >= prefab->slots.size() || !prefab->slots[index].sched) {
        return false;
    }
    ggml_backend_sched_mark_unproven(prefab->slots[index].sched.get());
    return true;
}

void llama_context::set_backend_cpu_threadpool(ggml_threadpool_t threadpool) {
    if (backend_cpu == nullptr) {
        return;
    }
    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
    auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(
        reg, "ggml_backend_cpu_set_threadpool");
    if (set_threadpool_fn) {
        set_threadpool_fn(backend_cpu, threadpool);
    }
}

ggml_threadpool_t llama_context::debug_cpu_backend_threadpool() const {
    if (backend_cpu == nullptr) {
        return nullptr;
    }
    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
    auto * get_threadpool_fn = (decltype(ggml_backend_cpu_get_threadpool) *) ggml_backend_reg_get_proc_address(
        reg, "ggml_backend_cpu_get_threadpool");
    return get_threadpool_fn ? get_threadpool_fn(backend_cpu) : nullptr;
}

void llama_context::debug_set_cpu_backend_threadpool(ggml_threadpool_t threadpool) {
    set_backend_cpu_threadpool(threadpool);
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

    if (prefab && llama_prefab_mtp_lifetime_enabled()) {
        prefab->reset_slots();
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

bool llama_context::nkv_binder_record_valid() const {
    return union_binder_state != nullptr && union_binder_state->record_valid;
}

// union-graph runtime parameter binder: n_kv lane.
//
// 1. when the live graph is (re)built, process_ubatch builds it once at the
//    full legal n_kv range (physical per-stream KV capacity) and at the
//    n_outputs high-water mark for legacy lanes. The opt-in token lane builds
//    at active KV, token and output widths and replans on changes. The
//    pre-allocation topology and per-tensor bounds are recorded;
// 2. union_shape_analyze derives, from the tagged roots and the exact ggml op
//    shape rules, a symbolic map: every dependent tensor's ne/nb/view_offs/
//    root_offs uses linear terms plus an n_kv * n_tokens cross term.
//    There is no finite-sample fitting and no scratch graph;
// 3. when a runtime parameter moves (can_reuse fails only on the stale coupled
//    shapes), union_rebind_apply evaluates the map at the new values and
//    validates every changed graph-owned tensor against its recorded byte
//    envelope; views are checked against the ultimate source allocation, so
//    nested views (the recurrent state zero view) are covered as well;
// 4. the changed fields (ne/nb/view_offs + view data pointers) are written into
//    the live graph in place, then set_inputs refills the KQ mask at the live
//    width (cells beyond the applied n_kv are empty -> -INF masked).
// Any mismatch -> false and the caller falls through to the normal full rebuild.
// The n_kv lane binds the true used span (runtime length). The patch requires
// exactly one live split. The scheduler verifies its split copies against
// their allocation envelopes before the patch. The existing FA width guard is
// retained; it is not a GPU correctness proof.
//
// The two capacity sites are runtime parameters too: the physical KV capacity
// (n_ctx_kv) and the per-sequence window (n_ctx_seq) use the requested length
// instead of a 256-cell bucket. The only remaining pad is the GPU FA capacity
// bucket: a GPU FA consumer with a hard-gated head dim keeps the stock 256
// capacity. The per-sequence window uses ceil division, so the requested
// n_ctx is never reduced. The rope long/short selection keeps the stock
// bucket (cparams.n_ctx_seq_rope), so removing the capacity padding does not
// change logits.


namespace {

struct llama_nkv_record_t {
    // DFS post-order: nodes and, before each, its not-yet-seen sources / view source
    std::vector<ggml_tensor *> t;
    std::vector<int32_t>       src;       // (GGML_MAX_SRC + 1) entries per tensor, view source last
    std::vector<uint8_t>       fixed;     // pre-existing storage (weights, KV cache)
    std::vector<size_t>        bytes;     // byte envelope of each graph-owned tensor
    std::vector<int32_t>       root;      // ultimate non-view source per tensor
    std::vector<size_t>        root_offs; // cumulative view offset from that root
    std::vector<size_t>        root_bytes; // root allocation / envelope size

    void record(ggml_cgraph * gf) {
        t.clear();
        src.clear();
        fixed.clear();
        bytes.clear();
        root.clear();
        root_offs.clear();
        root_bytes.clear();

        std::unordered_map<const ggml_tensor *, int32_t> idx;
        std::unordered_map<const ggml_tensor *, uint8_t> seen;
        idx.reserve(4 * (size_t) ggml_graph_n_nodes(gf) + 16);
        seen.reserve(4 * (size_t) ggml_graph_n_nodes(gf) + 16);

        std::vector<std::pair<ggml_tensor *, int>> stack;

        auto add = [&](ggml_tensor * root_t) {
            if (root_t == nullptr || seen.count(root_t)) {
                return;
            }
            stack.clear();
            stack.push_back({ root_t, 0 });
            seen[root_t] = 1;
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

                // resolve the view chain to its ultimate non-view source and
                // accumulate the byte offset from that source
                int32_t r = id;
                size_t roff = 0;
                if (x->view_src != nullptr) {
                    const auto it = idx.find(x->view_src);
                    GGML_ASSERT(it != idx.end());
                    const int32_t vs = it->second;
                    r = root[vs];
                    if (__builtin_add_overflow(root_offs[vs], x->view_offs, &roff)) roff = SIZE_MAX;
                }
                root.push_back(r);
                root_offs.push_back(roff);

                for (int j = 0; j <= GGML_MAX_SRC; ++j) {
                    ggml_tensor * s = is_fixed ? nullptr : (j < GGML_MAX_SRC ? x->src[j] : x->view_src);
                    int32_t v = -1;
                    if (s != nullptr) {
                        if (s == x) {
                            // ggml_cast sets src[1] = result (self-reference).
                            // Record the tensor's own index, not a cycle marker:
                            // the topology is valid and the bind restores it.
                            v = id;
                        } else {
                            auto it = idx.find(s);
                            v = it != idx.end() ? it->second : -2; // -2: back-edge (on DFS stack)
                        }
                    }
                    src.push_back(v);
                }
            }
        };

        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            add(ggml_graph_node(gf, i));
        }

        // The split graph the scheduler re-plans carries the graph's leaf list,
        // and ggml_backend_sched_replan_graph() refuses a leaf that is not in
        // the record. A leaf that is nobody's source never enters the node walk
        // above: build_inp_embd() selects one of two inputs with
        // ggml_build_forward_select(), which visits both candidates, so the
        // unselected input (inp_embd in a token batch) is a leaf of the graph
        // but not of the data path. Record the leaf list too; add() dedups
        // through `seen`, so every node-walk index keeps its value.
        for (int i = 0; i < ggml_graph_n_leafs(gf); ++i) {
            add(ggml_graph_leaf(gf, i));
        }
    }

    void record_bytes() {
        bytes.assign(t.size(), 0);
        for (size_t k = 0; k < t.size(); ++k) {
            if (!fixed[k] && t[k]->view_src == nullptr) {
                bytes[k] = ggml_nbytes(t[k]);
            }
        }
        // the limit for any view is the allocation of its ultimate source:
        // a fixed source's logical byte size, or a graph-owned source's
        // sched-allocated envelope
        root_bytes.assign(t.size(), 0);
        for (size_t k = 0; k < t.size(); ++k) {
            const int32_t r = root[k];
            root_bytes[k] = fixed[r] ? ggml_nbytes(t[r]) : bytes[r];
        }
    }
};
} // namespace

// record the pre-allocation topology + byte envelopes of a freshly built (not
// yet sched-placed) graph into the context's union-binder state slot
// (state note: was llama_context::nkv_record_* data members; now lives in the
// context's own state slot)
void llama_context::nkv_record_snapshot(ggml_cgraph * gf) {
    llama_nkv_record_t rec;
    rec.record(gf);
    rec.record_bytes();

    auto & st = llama_nkv_ctx_state_get(this);
    // The single graph identity of the graph as built. Computed here, once,
    // while the graph is still at its build shapes and before the binder
    // patches anything: the key must describe the recorded topology, not
    // whichever runtime width the first bind happens to choose. Every later
    // bind refreshes it with the bound state (union_rebind_apply), so it
    // always describes the graph as it currently is. See llama-topology-hash.h
    // for what is and is not folded in.
    //
    // The exclusion is the record's LANE (see union_topology_key_excl() in
    // llama-nout-state.h): a graph built with the token lane armed has token
    // inputs whose ne[0] the bind patches, so those dimensions must stay out
    // of the key or the bind would refuse itself. Every other lane passes
    // none() and keeps every dimension hashed.
    //
    // This assignment uses the shape map of the PREVIOUS record, because the
    // analysis for this one has not run yet. It is overwritten below, right
    // after union_shape_analyze(), which is the assignment the binder reads.
    // The exclusion is the map's, so it names every patched axis by tensor
    // name: the four token roots AND the derived tensors the coefficient table
    // knows (a name survives a rebuild, a pointer does not).
    st.record_topology_key = llama_topology_key_of(gf, union_topology_key_excl(st.shape_map, st.record_t));
    st.record_t          = std::move(rec.t);
    st.record_src        = std::move(rec.src);
    st.record_fixed      = std::move(rec.fixed);
    st.record_bytes      = std::move(rec.bytes);
    st.record_root       = std::move(rec.root);
    st.record_root_offs  = std::move(rec.root_offs);
    st.record_root_bytes = std::move(rec.root_bytes);
}

void llama_context::union_record_invalidate() {
    if (union_binder_state == nullptr) {
        return; // gate off: no state was ever allocated
    }
    auto & st = *union_binder_state;
    st.record_valid = false;
    st.record_t.clear();
    st.record_src.clear();
    st.record_fixed.clear();
    st.record_bytes.clear();
    st.record_root.clear();
    st.record_root_offs.clear();
    st.record_root_bytes.clear();
    st.record_cap = 0;
    st.record_cap_nout = 0;
    st.record_has_out_ids = false;
    st.record_cap_nrs = 0;
    st.record_bound = 0;
    st.record_arena_n_kv = 0;
    st.record_bound_nout = 0;
    st.record_bound_n_rs = 0;
    st.record_bound_ntok = 0;
    st.record_bound_rs_z = -2;
    st.record_bound_rs_head = 0;
    st.record_bound_replay_len = 0;
    st.record_topology_key = llama_topology_key{};
    st.live_topology_key   = llama_topology_key{};
    st.live_topology_gf    = nullptr;
    st.shape_map = llama_union_shape_map{};
    st.record_nodes.clear();
    st.width_table.clear();
    st.topology_hash   = 0;
    st.topology_hashed = false;
}

// Table-driven dispatch: one bind applies ALL THREE runtime parameters.
//
// n_kv, n_outputs and rs_z are simultaneous parameters of one fixed-topology
// graph, so a dispatch that returned on the first kind which "applied" would
// leave the other two deltas unapplied whenever a step moves more than one of
// them. The registered kinds therefore only GATE the bind: the first one whose
// preconditions hold for this ubatch (single split, envelope, replay length,
// rs_z present, ...) runs union_rebind_apply, and that single call evaluates
// the symbolic shape map at all three targets at once. Every gating kind calls
// apply with the same targets, so the enum order selects the precondition set,
// never the set of parameters that move. A kind whose own target did not move
// does not mask a delta in another kind: the delta test lives in apply, not in
// the gate. The n_tokens kind keeps its own gate (ntok_valid, token width
// moved), which is the token lane's own contract.
bool llama_union_binder::rebind_any(llama_context * ctx,
                                    llm_graph_result * res,
                                    const llm_graph_params & params) const {
    for (size_t i = 0; i < n_kinds; ++i) {
        if (!registered_[i]) {
            continue;
        }
        const llama_union_binding & binding = slots_[i];
        if (binding.tensor_set == nullptr || binding.rebind == nullptr) {
            continue;
        }
        const llama_union_tensor_set set = (ctx->*(binding.tensor_set))();
        switch ((ctx->*(binding.rebind))(res, params, set)) {
            case llama_union_rebind_result::applied:
                return true;
            case llama_union_rebind_result::refused:
                // This kind RAN: it passed its preconditions and
                // union_rebind_apply evaluated every runtime target at once.
                // A later kind would repeat the same patch and meet the same
                // refusal, so the first kind that ran decides the dispatch
                // (llama-union-binder.h's contract). The caller rebuilds.
                return false;
            case llama_union_rebind_result::declined:
                break;
        }
        // A failed replan can invalidate the live allocation. Do not let a
        // later kind inspect or reuse that graph; the caller must rebuild it.
        if (!ctx->nkv_binder_record_valid()) {
            return false;
        }
    }
    return false;
}

// n_kv / rs_z bindings share the recorded topology / envelope vectors of the
// live envelope build; the kind selects the rebind preconditions
llama_union_tensor_set llama_context::union_tensors_n_kv() {
    auto & st = llama_nkv_ctx_state_get(this);
    return { &st.record_t, &st.record_src, &st.record_fixed, &st.record_bytes,
             &st.record_root, &st.record_root_offs, &st.record_root_bytes };
}

llama_union_tensor_set llama_context::union_tensors_rs_z() {
    auto & st = llama_nkv_ctx_state_get(this);
    return { &st.record_t, &st.record_src, &st.record_fixed, &st.record_bytes,
             &st.record_root, &st.record_root_offs, &st.record_root_bytes };
}

llama_union_tensor_set llama_context::union_tensors_n_outputs() {
    auto & st = llama_nkv_ctx_state_get(this);
    return { &st.record_t, &st.record_src, &st.record_fixed, &st.record_bytes,
             &st.record_root, &st.record_root_offs, &st.record_root_bytes };
}

namespace {

// width a KV-cache context contributes to the graph: the true used span when
// the cache layout supports it, 0 when it does not. The width is bound in
// place by the symbolic shape map, so a width change does not rebuild the
// graph. True kv_len on GPU still needs the flash-attention device parameter.
//
// 0 is not a width: a caller that needs a width must treat it as "the exact
// lane cannot deliver here" and refuse the bind.
//
// One query, no second switch: get_n_kv_exact() is the single place that knows
// whether the layout can address the true span, and get_n_kv() is the single
// place that decides what the graph executes at. This function stays a pure
// pass-through so no call site can disagree with the width the graph inputs
// use.
//
// The `exact` parameter records the caller's intent -- "I want the lane's
// width, not a clamp" -- and defaults to true. It cannot select a width: the
// selection is nkv_exact_target's, and get_n_kv() is the one place that reads
// it. A caller that passes false is saying "this is a reduction-order clamp,
// not a lane choice", and on this line both values reach the same width
// because the exact lane no longer degrades to the padded one.
uint32_t union_nkv_width(const llama_kv_cache_context * kv, bool exact = true) {
    (void) exact;
    return kv->get_n_kv_exact();
}

// current n_kv of the KV-cache contexts that shape this graph; 0 when the
// graph has no KV cache, when a context has no exact width, or when the
// contexts disagree. A disagreement fails closed instead of silently using
// one of the widths.
//
// `exact` defaults to true because every caller wants the lane's width, and
// the one caller that is explicit about it (the reduction-order clamp in
// union_rebind_apply) passes false to say "this is not a lane choice, it is a
// clamp". Both values reach the same width on this line, so the default
// cannot change a bind.
uint32_t union_nkv_now(const llama_memory_context_i * mctx, bool exact = true) {
    const llama_kv_cache_context * kvs[2] = { nullptr, nullptr };
    int nkvs = 0;
    if (const auto * h = dynamic_cast<const llama_memory_hybrid_context *>(mctx)) {
        if (h->get_attn() == nullptr) {
            return 0; // pure recurrent hybrid: no attention sub-cache to bind
        }
        kvs[nkvs++] = h->get_attn();
    } else if (const auto * k = dynamic_cast<const llama_kv_cache_context *>(mctx)) {
        kvs[nkvs++] = k;
    } else if (const auto * h = dynamic_cast<const llama_memory_hybrid_iswa_context *>(mctx)) {
        // the two sub-caches (base + SWA) can have different widths; report
        // both so a disagreement fails closed instead of silently using one
        const auto * attn = h->get_attn();
        if (attn == nullptr || attn->get_base() == nullptr || attn->get_swa() == nullptr) {
            return 0;
        }
        kvs[nkvs++] = attn->get_base();
        kvs[nkvs++] = attn->get_swa();
    }
    if (nkvs == 0) {
        return 0;
    }
    uint32_t nkv = union_nkv_width(kvs[0]);
    for (int i = 1; i < nkvs; ++i) {
        if (union_nkv_width(kvs[i]) != nkv) {
            return 0;
        }
    }
    return nkv;
}

// a hybrid iSWA graph builds two attention masks (base + SWA) whose widths
// and readers the generic single-width patch does not cover; refuse it until
// a dedicated dual-mask binding exists
bool union_is_hybrid_iswa(const llama_memory_context_i * mctx) {
    return dynamic_cast<const llama_memory_hybrid_iswa_context *>(mctx) != nullptr;
}

const llama_memory_recurrent_context * union_recr_ctx(const llama_memory_context_i * mctx) {
    if (mctx == nullptr) {
        return nullptr;
    }
    if (const auto * r = dynamic_cast<const llama_memory_recurrent_context *>(mctx)) {
        return r;
    }
    if (const auto * h = dynamic_cast<const llama_memory_hybrid_context *>(mctx)) {
        return h->get_recr();
    }
    if (const auto * h = dynamic_cast<const llama_memory_hybrid_iswa_context *>(mctx)) {
        return h->get_recr();
    }
    return nullptr;
}

int32_t union_rs_z_now(const llama_memory_context_i * mctx) {
    const auto * r = union_recr_ctx(mctx);
    return r != nullptr ? r->get_rs_z() : -2;
}

uint32_t union_rs_head_now(const llama_memory_context_i * mctx) {
    const auto * r = union_recr_ctx(mctx);
    return r != nullptr ? r->get_head() : 0;
}

uint32_t union_n_rs_now(const llama_memory_context_i * mctx) {
    const auto * r = union_recr_ctx(mctx);
    return r != nullptr ? r->get_n_rs() : 0;
}
uint32_t union_replay_len_now(const llama_memory_context_i * mctx) {
    const auto * r = union_recr_ctx(mctx);
    return r != nullptr ? r->get_replay_len() : 0;
}

// attention-view readers cache per-frontier metadata (reader indices/spans)
// that the generic tensor patch does not refresh; the binder refuses such
// graphs instead of patching stale reader state.
bool union_has_attention_views(const llama_memory_context_i * mctx) {
    if (const auto * k = dynamic_cast<const llama_kv_cache_context *>(mctx)) {
        return k->has_attention_views();
    }
    if (const auto * h = dynamic_cast<const llama_memory_hybrid_context *>(mctx)) {
        const auto * attn = h->get_attn();
        return attn != nullptr && attn->has_attention_views();
    }
    if (const auto * h = dynamic_cast<const llama_memory_hybrid_iswa_context *>(mctx)) {
        const auto * attn = h->get_attn();
        if (attn == nullptr) {
            return false;
        }
        const auto * base = attn->get_base();
        const auto * swa  = attn->get_swa();
        return (base != nullptr && base->has_attention_views()) ||
               (swa  != nullptr && swa->has_attention_views());
    }
    return false;
}


// attention KV-cache context that shapes this graph (plain or hybrid); the
// hybrid iSWA context is deliberately not returned (its dual-mask graph is
// refused by union_rebind_apply)
llama_kv_cache_context * union_kv_ctx(const llama_memory_context_i * mctx) {
    if (const auto * h = dynamic_cast<const llama_memory_hybrid_context *>(mctx)) {
        return const_cast<llama_kv_cache_context *>(h->get_attn());
    }
    if (const auto * k = dynamic_cast<const llama_kv_cache_context *>(mctx)) {
        return const_cast<llama_kv_cache_context *>(k);
    }
    return nullptr;
}

// The long-lived attention cache this context's memory owns, for the mask
// read-back. The per-decode memory CONTEXT is owned by the unique_ptr inside
// llama_context::decode and dies when decode returns, so nothing may be
// resolved from it after the call. The memory object itself lives as long as
// the context, and the cache is a sub-object of it.
//
// Same shape as union_kv_ctx above: plain KV cache, or the attention half of
// a hybrid memory. The iSWA / DSA / MSA / DSV4 wrappers are deliberately not
// returned -- their dual-mask graphs are refused by union_rebind_apply, and a
// read-back against a cache the graph does not execute is worse than none.
static llama_kv_cache * union_kv_cache_of(llama_memory_i * mem) {
    if (mem == nullptr) {
        return nullptr;
    }
    if (auto * h = dynamic_cast<llama_memory_hybrid *>(mem)) {
        return h->get_mem_attn();
    }
    if (auto * k = dynamic_cast<llama_kv_cache *>(mem)) {
        return k;
    }
    return nullptr;
}

// RAII build-width override: while alive, get_n_kv() reports `width` so the
// graph under construction is sized at that width. Used for the envelope
// build (allocation width) and for the scratch build at the target true
// width. Exception-safe: an aborted build cannot leave the override armed.
class union_nkv_build_guard {
public:
    union_nkv_build_guard(llama_kv_cache_context * kv, uint32_t width) : kv(kv) {
        if (kv != nullptr && width != 0) {
            kv->nkv_envelope_begin(width);
            armed = true;
        }
    }

    ~union_nkv_build_guard() {
        if (armed) {
            kv->nkv_envelope_end();
        }
    }

    union_nkv_build_guard(const union_nkv_build_guard &) = delete;
    union_nkv_build_guard & operator=(const union_nkv_build_guard &) = delete;

private:
    llama_kv_cache_context * kv = nullptr;
    bool armed = false;
};

// RAII recurrent-state envelope override: while alive, get_n_rs()/get_head()/
// get_rs_z() report the full state envelope (n_rs = size, head = 0, rs_z = 0)
// so the graph is sized and sched-allocated for every legal live width. Used
// only around model.build_graph(); the live values are bound in place after.
class union_nrs_build_guard {
public:
    union_nrs_build_guard(llama_memory_recurrent_context * recr, uint32_t n_rs)
        : recr(recr) {
        if (recr != nullptr && n_rs != 0) {
            recr->nrs_envelope_begin(n_rs, 0, 0);
            armed = true;
        }
    }

    ~union_nrs_build_guard() {
        if (armed) {
            recr->nrs_envelope_end();
        }
    }

    union_nrs_build_guard(const union_nrs_build_guard &) = delete;
    union_nrs_build_guard & operator=(const union_nrs_build_guard &) = delete;

private:
    llama_memory_recurrent_context * recr = nullptr;
    bool armed = false;
};
// byte span actually addressed by a tensor: the last addressed byte minus the
// base, plus one unit. For a contiguous tensor this equals ggml_nbytes; for a
// permuted/strided view ggml_nbytes (ne product) underestimates the span, so
// the envelope check uses this for views.
//
// Quantized types are blocked: ggml_element_size()/nb[0] are the size of one
// block, not one element, so the first dimension contributes
// type_size * (ne[0] / blck_size) blocks, not (ne[0] - 1) * nb[0]. For
// blck_size == 1 this reduces to the plain formula.
size_t union_tensor_span(const ggml_tensor * t) {
    if (t == nullptr) {
        return 0;
    }
    // a zero-size dimension addresses no bytes; the data pointer may legally
    // sit one past the root allocation
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] == 0) {
            return 0;
        }
    }
    const size_t blck = ggml_blck_size(t->type);
    if (blck == 0 || t->ne[0] < 0 || t->ne[0] % (int64_t) blck != 0) return SIZE_MAX;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] < 0) return SIZE_MAX;
        if (t->ne[i] == 0) return 0;
    }
    size_t span = ggml_type_size(t->type);
    size_t first_stride;
    if (__builtin_mul_overflow((size_t) (t->ne[0] / (int64_t) blck - 1), t->nb[0], &first_stride) ||
            __builtin_add_overflow(span, first_stride, &span)) return SIZE_MAX;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        size_t contribution;
        if (__builtin_mul_overflow((size_t) (t->ne[i] - 1), t->nb[i], &contribution) ||
            __builtin_add_overflow(span, contribution, &span)) return SIZE_MAX;
    }
    return span;
}

// ---- symbolic runtime shape analysis (fixed-topology binder) ----
//
// The bind path must not build a graph. To patch the live graph in place we
// need the exact layout of every tensor that depends on a runtime parameter.
// This pass derives those layouts symbolically:
//
//   - tagged roots (K/V view, KQ mask, out_ids, recurrent state views) start
//     with the formula from their construction site;
//   - every consumer is derived from its sources with the exact ggml op shape
//     rule (the same ne/nb arithmetic the op constructor uses);
//   - an op outside the implemented rule set, an ambiguous view, or a
//     metadata mismatch makes the whole analysis fail closed.
//
// There is no finite-sample fitting: the coefficients are the op rules
// themselves. The supported range is the standard dense Llama decoder path
// with a plain or hybrid (non-iSWA) KV cache, no attention views, no
// ALiBi/relative-position inputs, no MoE, and no top-k attention. Anything
// else runs stock.

// contiguous nb from ne for a block-size-1 type; false for quantized types
// (a runtime-dependent quantized tensor is outside the supported range)
static bool union_contiguous_nb(enum ggml_type type, const union_dim_expr ne[4], union_dim_expr nb[4]) {
    if (ggml_blck_size(type) != 1) {
        return false;
    }
    nb[0] = union_dim_const((int64_t) ggml_type_size(type));
    for (int i = 1; i < 4; ++i) {
        if (!union_dim_mul(nb[i - 1], ne[i - 1], nb[i])) {
            return false;
        }
    }
    return true;
}

// exact symbolic shape map for the live graph. Returns false when the graph is
// outside the supported range or any dependent tensor cannot be derived by an
// exact rule. On success, map.*_valid says which parameters are bindable.
// Architectures the symbolic rules below cover. Dense Llama binds n_kv,
// n_outputs and n_tokens; its token lane also needs an attention KV cache,
// Flash Attention off and one RoPE position per token. The pure recurrent
// families have no attention KV cache (env false, token_runtime false) and
// reach the record build on the n_rs lane (n_rs length and head-selected
// copy views). Qwen3-Next is excluded by n_expert > 0 and Qwen3.5 by the
// one-position-per-token ROPE check; hybrid iSWA, hybrid_idx/QSA and
// ingredient-replay rollback groups (gdn_replay) stay fail-closed.
static bool union_arch_supported(llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_MAMBA:
        case LLM_ARCH_MAMBA2:
        case LLM_ARCH_RWKV6:
        case LLM_ARCH_RWKV6QWEN2:
        case LLM_ARCH_RWKV7:
        case LLM_ARCH_ARWKV7:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN35:
            return true;
        default:
            return false;
    }
}
static bool union_shape_analyze(const llm_graph_params & params,
                                const std::vector<ggml_tensor *> & t,
                                const std::vector<int32_t> & src,
                                const std::vector<uint8_t> & fixed,
                                const std::vector<int32_t> & root,
                                const std::vector<size_t> & root_offs,
                                uint32_t base_nkv, uint32_t base_nout,
                                uint32_t base_rs_head, uint32_t base_n_rs, bool token_runtime,
                                llama_union_shape_map & map) {
    map = llama_union_shape_map{};
    map.base_nkv     = base_nkv;
    map.base_nout    = base_nout;
    map.base_ntok    = params.ubatch.n_tokens;
    map.base_rs_head = base_rs_head;
    map.base_n_rs    = base_n_rs;

    // strict support range: the exact rules below are proven only for the
    // architectures in union_arch_supported(); everything else runs stock.
    if (!union_arch_supported(params.arch)) {
        return false;
    }
    if (const auto * recr = union_recr_ctx(params.mctx); recr != nullptr && recr->gdn_replay()) {
        // ingredient-replay state views select a group by the pending replay
        // length (a runtime shape); keep stock. The K-snapshot rollback path
        // (gdn_replay false, n_rs_seq > 0) is shape-stable and supported.
        return false;
    }
    if (dynamic_cast<const llama_memory_hybrid_idx_context *>(params.mctx) != nullptr) {
        // qwen4exp hybrid_idx/QSA has a second KV cache (idx) whose envelope
        // build and QSA input runtime metadata are not covered by this map
        return false;
    }
    if (params.hparams.use_alibi || params.hparams.n_expert > 0 || params.hparams.n_rel_attn_bkts > 0) {
        return false;
    }
    if (!params.samplers.empty()) {
        return false;
    }
    if (params.mctx == nullptr || union_is_hybrid_iswa(params.mctx) || union_has_attention_views(params.mctx)) {
        return false;
    }

    const size_t n = t.size();
    if (n == 0 || src.size() != n * (GGML_MAX_SRC + 1) || fixed.size() != n ||
            root.size() != n || root_offs.size() != n) {
        return false;
    }

    std::vector<union_shape_expr> expr(n);
    std::vector<uint8_t> is_root(n, 0);
    bool have_k = false, have_v = false, have_mask = false;
    bool have_out_ids = false;
    bool have_rs_zero_index = false;
    bool have_copy = false, have_copy_extra = false, have_copy_dst = false, have_head_view = false;

    for (size_t k = 0; k < n; ++k) {
        for (int i = 0; i < 4; ++i) {
            if (t[k]->ne[i] < 0 || t[k]->nb[i] > INT64_MAX) return false;
            expr[k].ne[i].base = t[k]->ne[i];
            expr[k].nb[i].base = t[k]->nb[i];
        }
        if (t[k]->view_offs > INT64_MAX || root_offs[k] > INT64_MAX) return false;
        expr[k].view_offs.base = (int64_t) t[k]->view_offs;
        expr[k].root_offs.base = (int64_t) root_offs[k];
    }

    for (size_t k = 0; k < n; ++k) {
        if (root[k] < 0 || root[k] >= (int32_t) n) {
            return false;
        }
    }

    auto src_idx = [&](size_t k, int j) -> int32_t {
        return src[k * (GGML_MAX_SRC + 1) + j];
    };
    std::unordered_map<const ggml_tensor *, int> token_roots;
    if (token_runtime) {
        for (const auto & input : params.res->inputs) {
            if (const auto * embd = dynamic_cast<const llm_graph_input_embd *>(input.get())) {
                if (embd->tokens) token_roots[embd->tokens] = 0;
                if (embd->embd) token_roots[embd->embd] = 1;
            } else if (const auto * pos = dynamic_cast<const llm_graph_input_pos *>(input.get())) {
                if (pos->pos) token_roots[pos->pos] = 0;
            } else if (const auto * attn = dynamic_cast<const llm_graph_input_attn_kv *>(input.get())) {
                if (attn->get_k_idxs()) token_roots[attn->get_k_idxs()] = 0;
                if (attn->get_v_idxs()) token_roots[attn->get_v_idxs()] = 0;
            }
        }
    }
    size_t found_token_roots = 0;


    for (size_t k = 0; k < n; ++k) {
        ggml_tensor * x = t[k];
        if (strcmp(x->name, "union_nkv_k") == 0) {
            // get_k: view_4d(head_k_eff, n_head_kv, n_kv, ns)
            if (x->ne[2] != (int64_t) base_nkv) {
                return false;
            }
            expr[k].ne[2].d_nkv = 1;
            is_root[k] = 1;
            have_k = true;
        } else if (strcmp(x->name, "union_nkv_v") == 0) {
            // get_v: non-transposed view_4d(head_v_eff, n_head_kv, n_kv, ns);
            // transposed view_4d(n_kv, n_head_kv, head_v_eff, ns)
            const int dim = x->nb[1] > x->nb[2] ? 0 : 2;
            if (x->ne[dim] != (int64_t) base_nkv) {
                return false;
            }
            expr[k].ne[dim].d_nkv = 1;
            is_root[k] = 1;
            have_v = true;
        } else if (strcmp(x->name, "attn_inp_kq_mask") == 0) {
            // ggml_new_tensor_4d(type, n_kv, n_tokens/n_stream, 1, n_stream)
            if (x->ne[0] != (int64_t) base_nkv || ggml_blck_size(x->type) != 1) {
                return false;
            }
            expr[k].ne[0].d_nkv = 1;
            if (token_runtime) {
                if (x->ne[1] != (int64_t) map.base_ntok) return false;
                expr[k].ne[1].d_ntok = 1;
            }
            if (!union_contiguous_nb(x->type, expr[k].ne, expr[k].nb)) return false;
            is_root[k] = 1;
            have_mask = true;
        } else if (strcmp(x->name, "union_out_ids") == 0) {
            if (x->ne[0] != (int64_t) base_nout) {
                return false;
            }
            expr[k].ne[0].d_nout = 1;
            // The row count is the width, and the strides follow it exactly as
            // they do for every other 1-D runtime root (the KQ mask, the token
            // inputs, the GET_ROWS result): the bind writes ne[0] and the
            // derived nb[1..3], so the live tensor stays a well-formed
            // contiguous 1-D tensor at every bound width, and the key's
            // name-based exclusion covers the strides it moves. A width-only
            // coefficient would leave nb[1..3] frozen at their build value and
            // the tensor non-contiguous (ggml_is_contiguous() false) after the
            // first bind.
            if (!union_contiguous_nb(x->type, expr[k].ne, expr[k].nb)) return false;
            is_root[k] = 1;
            have_out_ids = true;
        } else if (token_roots.count(x)) {
            const int axis = token_roots.at(x);
            if (!map.base_ntok || x->ne[axis] % map.base_ntok != 0) return false;
            expr[k].ne[axis].d_ntok = x->ne[axis] / map.base_ntok;
            if (!union_contiguous_nb(x->type, expr[k].ne, expr[k].nb)) return false;
            is_root[k] = 1;
            ++found_token_roots;
        } else if (strcmp(x->name, "union_rs_copy") == 0) {
            // build_rs_inp_impl: I32 [n_rs] copy-index tensor
            if (x->ne[0] != (int64_t) base_n_rs) {
                return false;
            }
            expr[k].ne[0].d_nrs = 1;
            is_root[k] = 1;
            have_copy = true;
        } else if (strcmp(x->name, "union_rs_copy_main") == 0) {
            // constant-length prefix view of union_rs_copy; it must not inherit
            // the source's n_rs coefficient (the main gather is fixed n_seqs)
            if (x->view_src == nullptr || x->view_offs != 0) {
                return false;
            }
            is_root[k] = 1;
        } else if (strcmp(x->name, "union_rs_copy_extra") == 0) {
            // view_1d(s_copy, n_rs - n_seqs, n_seqs * nb[0])
            if (x->view_src == nullptr || x->view_src->ne[0] != (int64_t) base_n_rs) {
                return false;
            }
            const int64_t n_seqs = x->view_src->ne[0] - x->ne[0];
            if (n_seqs < 0 || x->view_offs != (uint64_t) (n_seqs * x->view_src->nb[0])) {
                return false;
            }
            expr[k].ne[0].d_nrs = 1;
            is_root[k] = 1;
            have_copy_extra = true;
        } else if (strcmp(x->name, "union_rs_copy_dst") == 0) {
            // build_rs: view_2d(s, state_size, n_rs-n_seqs, s->nb[1], (head+n_seqs)*s->nb[1])
            if (x->view_src == nullptr || x->ne[1] > (int64_t) base_n_rs) {
                return false;
            }
            const int64_t row = (int64_t) x->view_src->nb[1];
            const int64_t n_seqs = (int64_t) base_n_rs - x->ne[1];
            if (x->view_offs != (uint64_t) (n_seqs * row)) {
                return false;
            }
            expr[k].ne[1].d_nrs = 1;
            expr[k].view_offs.d_head = row;
            expr[k].root_offs.d_head = row;
            is_root[k] = 1;
            have_copy_dst = true;
        } else if (strcmp(x->name, "union_rs_head_view") == 0) {
            // model state-store view: a constant base slot plus head * row
            // stride. The shape is constant; only the offset follows the head.
            // The base may be nonzero (rollback group s_slot * mem_size).
            if (x->view_src == nullptr) {
                return false;
            }
            const int64_t row = (int64_t) x->view_src->nb[1];
            expr[k].view_offs.d_head = row;
            expr[k].root_offs.d_head = row;
            is_root[k] = 1;
            have_head_view = true;
        } else if (strcmp(x->name, "union_rs_zero_index") == 0) {
            // rs_z runtime parameter: I32 [1] zero-row index written by
            // union_rsz_bind_index() on every set_inputs(). Its shape never
            // depends on the row, so it is a fixed root (no coefficients) and
            // it feeds no fill() list: the row is a value, not a shape. A
            // tensor with this name and another shape is not the input this
            // analysis understands, so the whole analysis fails closed.
            if (x->ne[0] != 1 || x->type != GGML_TYPE_I32) {
                return false;
            }
            is_root[k] = 1;
            have_rs_zero_index = true;
        }
    }

    const bool has_kv = union_kv_ctx(params.mctx) != nullptr;
    const bool has_recr = union_recr_ctx(params.mctx) != nullptr;
    // The zero-row index is a fixed root: it must not carry coefficients, and
    // a graph that carries it must have a recurrent state to clear.
    if (have_rs_zero_index && !has_recr) {
        return false;
    }

    // topological propagation (the record is DFS post-order: sources first)
    for (size_t k = 0; k < n; ++k) {
        if (is_root[k] || fixed[k]) {
            continue;
        }
        ggml_tensor * x = t[k];

        int dep_src = -1;
        for (int j = 0; j <= GGML_MAX_SRC; ++j) {
            const int32_t v = src_idx(k, j);
            if (v >= 0 && expr[v].dep()) {
                dep_src = v;
                break;
            }
        }
        if (dep_src < 0) {
            continue; // invariant
        }

        union_shape_expr out;
        for (int i = 0; i < 4; ++i) {
            out.ne[i] = union_dim_const(x->ne[i]);
            out.nb[i] = union_dim_const(x->nb[i]);
        }
        out.view_offs = union_dim_const((int64_t) x->view_offs);
        out.root_offs = union_dim_const((int64_t) root_offs[k]);

        const int32_t s0i = src_idx(k, 0);
        const int32_t s1i = src_idx(k, 1);
        const int32_t s2i = src_idx(k, 2);
        const int32_t vsi = src_idx(k, GGML_MAX_SRC);

        auto need = [&](int32_t v) -> bool { return v >= 0 && v < (int32_t) n; };
        auto shape_preserving = [&](const union_shape_expr & s) -> bool {
            for (int i = 0; i < 4; ++i) {
                out.ne[i] = s.ne[i];
            }
            if (x->view_src != nullptr) {
                for (int i = 0; i < 4; ++i) {
                    out.nb[i] = s.nb[i];
                }
                out.view_offs = union_dim_const((int64_t) x->view_offs);
                out.root_offs = union_dim_add(s.root_offs, out.view_offs);
            } else {
                if (!union_contiguous_nb(x->type, out.ne, out.nb)) {
                    return false;
                }
                out.view_offs = union_dim_const(0);
                out.root_offs = union_dim_const(0);
            }
            return true;
        };
        // A full reshape may split or merge constant dimensions around one
        // token dimension. Map that dimension by its linear element stride.
        auto token_reshape = [&](int32_t si) -> bool {
            if (!need(si)) return false;
            const auto & source = expr[si];
            int token_axis = -1;
            int64_t source_total = 1, token_stride = 1, output_total = 1;
            for (int i = 0; i < 4; ++i) {
                const auto & d = source.ne[i];
                if (d.invalid || d.d_nkv || d.d_nout || d.d_nrs || d.d_head || d.d_nkv_ntok) return false;
                if (d.d_ntok) {
                    if (token_axis >= 0 || d.base <= 0) return false;
                    token_axis = i;
                    token_stride = source_total;
                }
                if (__builtin_mul_overflow(source_total, d.base, &source_total) ||
                    __builtin_mul_overflow(output_total, x->ne[i], &output_total)) return false;
            }
            if (token_axis < 0 || source_total != output_total) return false;
            const int32_t tag = x->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1];
            if ((tag & ~3) == LLM_TOKEN_AXIS_MARKER) {
                const int axis = tag & 3;
                const int64_t base = source.ne[token_axis].base;
                if (base <= 0 || x->ne[axis] % base != 0) {
                    return false;
                }
                const int64_t factor = x->ne[axis] / base;
                // The marked axis carries the token axis when one token's
                // elements span the axis exactly once: the axis's byte stride
                // times the number of tokens it folds equals the source token
                // stride in bytes. This is the factor-1 case of the old
                // `prefix == token_stride` test, generalized to a marked axis
                // that folds the token axis into a larger extent. The
                // transposed V cache write flattens (n_embd, n_tokens) to
                // (1, n_embd*n_tokens), where the token stride lies inside
                // axis 1 instead of at its start.
                int64_t token_bytes, axis_bytes;
                if (__builtin_mul_overflow(token_stride, (int64_t) ggml_type_size(x->type), &token_bytes) ||
                        __builtin_mul_overflow((int64_t) x->nb[axis], factor, &axis_bytes) ||
                        token_bytes != axis_bytes) {
                    return false;
                }
                return union_dim_mul(source.ne[token_axis], union_dim_const(factor), out.ne[axis]);
            }
            // Equal layouts preserve the source axis even when its build
            // width is one and several dimensions have the same value.
            bool identity = true;
            for (int i = 0; i < 4; ++i) identity &= x->ne[i] == t[si]->ne[i];
            if (identity) {
                for (int i = 0; i < 4; ++i) out.ne[i] = source.ne[i];
                return true;
            }
            int64_t stride = 1;
            int output_axis = -1;
            int fallback_axis = -1;
            int64_t fallback_stride = 1;
            for (int i = 0; i < 4; ++i) {
                int64_t next;
                if (__builtin_mul_overflow(stride, x->ne[i], &next)) return false;
                // The token axis starts at `token_stride` in the source. An
                // output axis carries it when the token axis lies strictly
                // inside the axis. A merge with a trailing extent of one
                // instead leaves the token axis at the axis start: a 1-token
                // build folds it onto the axis that begins exactly at its
                // stride, e.g. the 2-D cache-write view (n_embd_gqa,
                // n_tokens) of a (n_embd_head, n_head, n_tokens) source at
                // n_tokens == 1. That equality is only a fallback, kept so an
                // extent-1 axis before the real token axis cannot win; the
                // widest candidate (earliest on a tie) is used when no axis
                // holds the token stride strictly.
                if (token_stride >= stride && token_stride < next) {
                    output_axis = i;
                    break;
                }
                if (stride == token_stride &&
                        (fallback_axis < 0 || x->ne[i] > x->ne[fallback_axis])) {
                    fallback_axis = i;
                    fallback_stride = stride;
                }
                stride = next;
            }
            if (output_axis < 0) {
                output_axis = fallback_axis;
                stride = fallback_stride;
            }
            int64_t token_end, output_end;
            if (output_axis < 0 || token_stride % stride ||
                    __builtin_mul_overflow(token_stride, source.ne[token_axis].base, &token_end) ||
                    __builtin_mul_overflow(stride, x->ne[output_axis], &output_end) || token_end > output_end ||
                    x->ne[output_axis] % source.ne[token_axis].base) return false;
            const int64_t factor = x->ne[output_axis] / source.ne[token_axis].base;
            return union_dim_mul(source.ne[token_axis], union_dim_const(factor), out.ne[output_axis]);
        };


        switch (x->op) {
            case GGML_OP_VIEW: {
                // A fused-QKV projection slice is marked at build time. Each
                // is a non-contiguous row slice of one fused matmul, so the
                // general path below cannot prove its fused row stride. Every
                // refusal of this view names that reason; the analysis still
                // fails closed.
                const bool fused_qkv_slice =
                    x->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] == LLM_FUSED_QKV_SLICE_MARKER;
                auto view_refuse = [&]() -> bool {
                    if (fused_qkv_slice) map.fused_qkv_refused = true;
                    return false;
                };
                if (!need(vsi)) {
                    return view_refuse();
                }
                if (token_runtime && need(s0i) && ggml_is_contiguous(t[s0i]) &&
                        ggml_is_contiguous(x) && x->view_offs == t[s0i]->view_offs) {
                    bool token_dep = false;
                    for (const auto & dim : expr[s0i].ne) token_dep |= dim.d_ntok != 0;
                    if (token_dep) {
                        if (!token_reshape(s0i) || !union_contiguous_nb(x->type, out.ne, out.nb)) return view_refuse();
                        out.view_offs = expr[s0i].view_offs;
                        out.root_offs = expr[s0i].root_offs;
                        break;
                    }
                }
                const union_shape_expr & s = expr[vsi];
                for (int i = 0; i < 4; ++i) {
                    int match = -1;
                    for (int j = 0; j < 4; ++j) {
                        if (x->ne[i] == t[vsi]->ne[j]) {
                            if (match >= 0 && !union_dim_equal(s.ne[match], s.ne[j])) {
                                return view_refuse(); // ambiguous structural match
                            }
                            match = j;
                        }
                    }
                    if (match >= 0) {
                        out.ne[i] = s.ne[match];
                    } else {
                        if (s.dep()) {
                            // no structural match on a dependent source: the
                            // view may encode a product/slice we cannot prove
                            return view_refuse();
                        }
                        if (x->ne[i] == (int64_t) base_nkv || x->ne[i] == (int64_t) base_nout ||
                                x->ne[i] == (int64_t) base_n_rs || x->ne[i] == (int64_t) base_rs_head) {
                            return view_refuse();
                        }
                        out.ne[i] = union_dim_const(x->ne[i]);
                    }
                    match = -1;
                    for (int j = 0; j < 4; ++j) {
                        if (x->nb[i] == t[vsi]->nb[j]) {
                            if (match >= 0 && !union_dim_equal(s.nb[match], s.nb[j])) {
                                return view_refuse(); // ambiguous structural match
                            }
                            match = j;
                        }
                    }
                    if (match >= 0) {
                        out.nb[i] = s.nb[match];
                    } else {
                        if (s.dep()) {
                            return view_refuse();
                        }
                        out.nb[i] = union_dim_const((int64_t) x->nb[i]);
                    }
                }
                out.view_offs = x->view_offs == t[vsi]->view_offs
                    ? s.view_offs
                    : union_dim_const((int64_t) x->view_offs);
                out.root_offs = union_dim_add(s.root_offs, out.view_offs);
            } break;
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE: {
                if (!need(s0i)) {
                    return false;
                }
                const union_shape_expr & s = expr[s0i];
                int32_t axes[4] = { 0, 1, 2, 3 };
                if (x->op == GGML_OP_PERMUTE) {
                    memcpy(axes, x->op_params, sizeof(axes));
                } else {
                    axes[0] = 1;
                    axes[1] = 0;
                }
                for (int i = 0; i < 4; ++i) {
                    if (axes[i] < 0 || axes[i] > 3) {
                        return false;
                    }
                    out.ne[axes[i]] = s.ne[i];
                    out.nb[axes[i]] = s.nb[i];
                }
                // ggml_permute/ggml_transpose use ggml_view_tensor: the
                // result aliases the source with view_offs == 0
                if (x->view_offs != 0) {
                    return false;
                }
                out.view_offs = union_dim_const(0);
                out.root_offs = s.root_offs;
            } break;
            case GGML_OP_RESHAPE: {
                if (!need(s0i)) {
                    return false;
                }
                const union_shape_expr & s = expr[s0i];
                bool token_dep = false;
                for (const auto & dim : s.ne) token_dep |= dim.d_ntok != 0;
                if (token_dep) {
                    if (!token_reshape(s0i) || !union_contiguous_nb(x->type, out.ne, out.nb)) return false;
                    out.view_offs = union_dim_const(0);
                    out.root_offs = x->view_src != nullptr ? s.root_offs : union_dim_const(0);
                    break;
                }
                for (int i = 0; i < 4; ++i) {
                    int match = -1;
                    for (int j = 0; j < 4; ++j) {
                        if (x->ne[i] == t[s0i]->ne[j]) {
                            if (match >= 0 && !union_dim_equal(s.ne[match], s.ne[j])) {
                                return false; // ambiguous structural match
                            }
                            match = j;
                        }
                    }
                    if (match >= 0) {
                        out.ne[i] = s.ne[match];
                    } else {
                        if (s.dep()) {
                            // a reshape can fold several dims into a product;
                            // without a single structural match, fail closed
                            return false;
                        }
                        if (x->ne[i] == (int64_t) base_nkv || x->ne[i] == (int64_t) base_nout ||
                                x->ne[i] == (int64_t) base_n_rs || x->ne[i] == (int64_t) base_rs_head) {
                            return false;
                        }
                        out.ne[i] = union_dim_const(x->ne[i]);
                    }
                }
                if (!union_contiguous_nb(x->type, out.ne, out.nb)) {
                    return false;
                }
                out.view_offs = union_dim_const(0);
                out.root_offs = x->view_src != nullptr ? s.root_offs : union_dim_const(0);
            } break;
            case GGML_OP_CONT: {
                if (!need(s0i)) {
                    return false;
                }
                bool same_shape = true;
                for (int i = 0; i < 4; ++i) same_shape &= x->ne[i] == t[s0i]->ne[i];
                const int32_t tag = x->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1];
                if (token_runtime && (tag & ~3) == LLM_TOKEN_AXIS_MARKER) {
                    if (!token_reshape(s0i)) return false;
                } else if (same_shape) {
                    for (int i = 0; i < 4; ++i) out.ne[i] = expr[s0i].ne[i];
                } else if (!token_runtime || !token_reshape(s0i)) {
                    return false;
                }
                if (!union_contiguous_nb(x->type, out.ne, out.nb)) {
                    return false;
                }
                out.view_offs = union_dim_const(0);
                out.root_offs = union_dim_const(0);
            } break;
            case GGML_OP_CPY: {
                if (t[k]->src[1] == t[k]) {
                    // ggml_cast: new contiguous tensor with the source shape
                    if (!need(s0i)) {
                        return false;
                    }
                    for (int i = 0; i < 4; ++i) {
                        out.ne[i] = expr[s0i].ne[i];
                    }
                    if (!union_contiguous_nb(x->type, out.ne, out.nb)) {
                        return false;
                    }
                    out.view_offs = union_dim_const(0);
                    out.root_offs = union_dim_const(0);
                } else {
                    // ggml_cpy: ggml_cpy_impl() returns ggml_view_tensor(b),
                    // i.e. a view of the destination flattened to the
                    // destination's ultimate source. Its shape is the
                    // destination's shape, its view_offs is the destination's
                    // (absolute) view offset and its root is the destination's
                    // root, so both offsets follow the destination view's
                    // symbolic expressions -- not the build-time constants,
                    // which would leave a head-moved copy destination stale.
                    if (!need(s1i)) {
                        return false;
                    }
                    for (int i = 0; i < 4; ++i) {
                        out.ne[i] = expr[s1i].ne[i];
                        out.nb[i] = expr[s1i].nb[i];
                    }
                    out.view_offs = expr[s1i].view_offs;
                    out.root_offs = expr[s1i].root_offs;
                }
            } break;
            case GGML_OP_ROPE:
                if (token_runtime) {
                    if (!need(s0i) || !need(s1i) || !union_dim_equal(expr[s1i].ne[0], expr[s0i].ne[2])) return false;
                    if (!shape_preserving(expr[s0i])) return false;
                    break;
                }
                /* fall through */
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SCALE:
            case GGML_OP_ADD:
            case GGML_OP_MUL:
            case GGML_OP_SUB:
            case GGML_OP_DIV:
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_UNARY: {
                int dep = need(s0i) ? s0i : (need(s1i) ? s1i : -1);
                if (dep < 0) {
                    return false;
                }
                if (need(s0i) && need(s1i)) {
                    // broadcast: a constant-1 dim yields to the other source;
                    // two different dependent shapes cannot be represented
                    int winner = -1;
                    for (int i = 0; i < 4; ++i) {
                        const union_dim_expr & a = expr[s0i].ne[i];
                        const union_dim_expr & b = expr[s1i].ne[i];
                        int w = -1;
                        if (union_dim_equal(a, b)) {
                            w = -1;
                        } else if (a.base == 1 && !a.dep()) {
                            w = s1i;
                        } else if (b.base == 1 && !b.dep()) {
                            w = s0i;
                        } else {
                            return false;
                        }
                        if (w >= 0) {
                            if (winner >= 0 && winner != w) {
                                return false;
                            }
                            winner = w;
                        }
                    }
                    if (winner >= 0) {
                        dep = winner;
                    }
                }
                if (!shape_preserving(expr[dep])) {
                    return false;
                }
            } break;
            case GGML_OP_GLU: {
                if (!need(s0i)) {
                    return false;
                }
                const union_shape_expr & s = expr[s0i];
                for (int i = 0; i < 4; ++i) {
                    out.ne[i] = s.ne[i];
                }
                if (x->src[1] == nullptr) {
                    // merged GLU: ne[0] = a->ne[0] / 2
                    union_dim_expr half = s.ne[0];
                    if (half.base % 2 != 0 || half.d_nkv % 2 != 0 || half.d_nout % 2 != 0 ||
                            half.d_head % 2 != 0 || half.d_nrs % 2 != 0 ||
                            half.d_ntok % 2 != 0 || half.d_nkv_ntok % 2 != 0) {
                        return false;
                    }
                    half.base /= 2;
                    half.d_nkv /= 2;
                    half.d_nout /= 2;
                    half.d_head /= 2;
                    half.d_nrs /= 2;
                    half.d_ntok /= 2;
                    half.d_nkv_ntok /= 2;
                    out.ne[0] = half;
                }
                if (!union_contiguous_nb(x->type, out.ne, out.nb)) {
                    return false;
                }
                out.view_offs = union_dim_const(0);
                out.root_offs = union_dim_const(0);
            } break;
            case GGML_OP_MUL_MAT: {
                out.ne[0] = expr[s0i].ne[1];
                out.ne[1] = expr[s1i].ne[1];
                out.ne[2] = expr[s1i].ne[2];
                out.ne[3] = expr[s1i].ne[3];
                if (!union_contiguous_nb(x->type, out.ne, out.nb)) {
                    return false;
                }
                out.view_offs = union_dim_const(0);
                out.root_offs = union_dim_const(0);
            } break;
            case GGML_OP_GET_ROWS: {
                out.ne[0] = expr[s0i].ne[0];
                out.ne[1] = expr[s1i].ne[0];
                out.ne[2] = expr[s1i].ne[1];
                out.ne[3] = expr[s1i].ne[2];
                if (!union_contiguous_nb(x->type, out.ne, out.nb)) {
                    return false;
                }
                out.view_offs = union_dim_const(0);
                out.root_offs = union_dim_const(0);
            } break;
            case GGML_OP_FLASH_ATTN_EXT: {
                out.ne[0] = expr[s2i].ne[0];
                out.ne[1] = expr[s0i].ne[2];
                out.ne[2] = expr[s0i].ne[1];
                out.ne[3] = expr[s0i].ne[3];
                if (!union_contiguous_nb(x->type, out.ne, out.nb)) {
                    return false;
                }
                out.view_offs = union_dim_const(0);
                out.root_offs = union_dim_const(0);
            } break;
            case GGML_OP_PAD: {
                if (!need(s0i)) {
                    return false;
                }
                const int32_t * pads = (const int32_t *) x->op_params;
                for (int i = 0; i < 4; ++i) {
                    // ggml_pad_ext stores (left, right) per dimension
                    out.ne[i] = union_dim_add(expr[s0i].ne[i],
                            union_dim_const(pads[2*i] + pads[2*i + 1]));
                }
                if (!union_contiguous_nb(x->type, out.ne, out.nb)) {
                    return false;
                }
                out.view_offs = union_dim_const(0);
                out.root_offs = union_dim_const(0);
            } break;
            case GGML_OP_SUM_ROWS: {
                if (!need(s0i)) {
                    return false;
                }
                // ggml_sum_rows: ne = {1, a->ne[1], a->ne[2], a->ne[3]}
                out.ne[0] = union_dim_const(1);
                out.ne[1] = expr[s0i].ne[1];
                out.ne[2] = expr[s0i].ne[2];
                out.ne[3] = expr[s0i].ne[3];
                if (!union_contiguous_nb(x->type, out.ne, out.nb)) {
                    return false;
                }
                out.view_offs = union_dim_const(0);
                out.root_offs = union_dim_const(0);
            } break;
            case GGML_OP_CONCAT: {
                const int axis = ((const int32_t *) x->op_params)[0];
                if (axis < 0 || axis > 3 || !need(s0i)) {
                    return false;
                }
                for (int i = 0; i < 4; ++i) {
                    if (i == axis) {
                        continue;
                    }
                    out.ne[i] = expr[s0i].ne[i];
                }
                union_dim_expr sum = union_dim_const(0);
                for (int j = 0; j < GGML_MAX_SRC; ++j) {
                    const int32_t v = src_idx(k, j);
                    if (v < 0) {
                        continue;
                    }
                    if (!need(v)) {
                        return false;
                    }
                    for (int i = 0; i < 4; ++i) {
                        if (i != axis && !union_dim_equal(out.ne[i], expr[v].ne[i])) {
                            return false;
                        }
                    }
                    sum = union_dim_add(sum, expr[v].ne[axis]);
                }
                out.ne[axis] = sum;
                if (!union_contiguous_nb(x->type, out.ne, out.nb)) {
                    return false;
                }
                out.view_offs = union_dim_const(0);
                out.root_offs = union_dim_const(0);
            } break;
            case GGML_OP_SET_ROWS: {
                // ggml_set_rows(cache, cur, idxs): src[0] is the value tensor,
                // src[1] the row indices, src[2] the destination. The result
                // is a view of the destination, so it carries the
                // destination's layout.
                const int32_t rs = src_idx(k, 0);
                const int32_t is = src_idx(k, 1);
                const int32_t ds = src_idx(k, 2);
                if (!need(rs) || !need(is) || !need(ds)) {
                    return false;
                }
                // The SET_ROWS shape contract (destination frozen, index/value
                // cross-check, post-copy kernel invariants) is one pure
                // predicate shared with tests/test-union-shape-guard.cpp.
                if (!union_set_rows_shape_ok(expr[ds], expr[rs], expr[is],
                                             (int64_t) ggml_type_size(x->type),
                                             ggml_blck_size(x->type), out)) {
                    return false;
                }
                out.view_offs = union_dim_const((int64_t) x->view_offs);
                out.root_offs = union_dim_add(expr[ds].root_offs, out.view_offs);
            } break;
            case GGML_OP_MAP_CUSTOM2: {
                // in-place op: the output aliases src[0]; the zero-row op
                // carries no shape coefficients of its own.
                const int32_t s0i = src_idx(k, 0);
                if (!need(s0i)) {
                    return false;
                }
                for (int i = 0; i < 4; ++i) {
                    out.ne[i] = expr[s0i].ne[i];
                    out.nb[i] = expr[s0i].nb[i];
                }
                out.view_offs = expr[s0i].view_offs;
                out.root_offs = expr[s0i].root_offs;
            } break;
            case GGML_OP_SSM_CONV:
            case GGML_OP_SSM_SCAN: {
                // the recurrent state and token axes are constant at the
                // envelope; the ids input is a root with zero coefficients.
                for (int i = 0; i < 4; ++i) {
                    out.ne[i] = union_dim_const(x->ne[i]);
                    out.nb[i] = union_dim_const(x->nb[i]);
                }
                out.view_offs = union_dim_const(0);
                out.root_offs = union_dim_const(0);
            } break;
            default:
                LLAMA_LOG_ERROR("%s: unsupported op %s tensor %s\n",
                        __func__, ggml_op_name(x->op), x->name);
                return false;
        }

        // the rule must reproduce the live metadata exactly
        for (int i = 0; i < 4; ++i) {
            if (out.ne[i].invalid || out.nb[i].invalid || out.ne[i].base != x->ne[i] ||
                    out.nb[i].base != (int64_t) x->nb[i]) {
                return false;
            }
        }
        if (out.view_offs.invalid || out.root_offs.invalid ||
                out.view_offs.base != (int64_t) x->view_offs ||
                out.root_offs.base != (int64_t) root_offs[k]) {
            LLAMA_LOG_ERROR("%s: offset mismatch tensor %s op %s: view %lld vs %lld root %lld vs %lld\n",
                    __func__, x->name, ggml_op_name(x->op),
                    (long long) out.view_offs.base, (long long) x->view_offs,
                    (long long) out.root_offs.base, (long long) root_offs[k]);
            return false;
        }

        expr[k] = out;
    }

    auto coefficient = [](const union_dim_expr & e, int which) -> int64_t {
        switch (which) {
            case 0: return e.d_nkv;
            case 1: return e.d_nout;
            case 2: return e.d_head;
            case 3: return e.d_nrs;
            case 4: return e.d_ntok;
            default: return e.d_nkv_ntok;
        }
    };
    // materialize the per-parameter coefficient lists
    auto fill = [&](int which, bool valid, std::vector<llama_union_shape_coeff> & list, bool & out_valid) {
        out_valid = valid;
        if (!valid) {
            return;
        }
        for (size_t k = 0; k < n; ++k) {
            const union_shape_expr & e = expr[k];
            llama_union_shape_coeff c;
            c.tensor = (uint32_t) k;
            bool any = false;
            for (int i = 0; i < 4; ++i) {
                c.ne[i] = coefficient(e.ne[i], which);
                c.nb[i] = coefficient(e.nb[i], which);
                c.base_ne[i] = e.ne[i].base;
                c.base_nb[i] = e.nb[i].base;
                any = any || c.ne[i] != 0 || c.nb[i] != 0;
            }
            c.view_offs = coefficient(e.view_offs, which);
            c.root_offs = coefficient(e.root_offs, which);
            c.base_view_offs = e.view_offs.base;
            c.base_root_offs = e.root_offs.base;
            any = any || c.view_offs != 0 || c.root_offs != 0;
            if (any) {
                list.push_back(c);
            }
        }
    };

    fill(0, has_kv && have_k && have_v && have_mask, map.nkv, map.nkv_valid);
    fill(1, have_out_ids, map.nout, map.nout_valid);
    fill(2, has_recr && have_copy_dst && have_head_view, map.rs_head, map.rs_head_valid);
    fill(3, has_recr && have_copy && have_copy_extra && have_copy_dst, map.rs_len, map.nrs_valid);
    fill(4, token_runtime && found_token_roots >= 3, map.ntok, map.ntok_valid);
    bool mixed_valid = false;
    fill(5, map.ntok_valid && map.nkv_valid, map.nkv_ntok, mixed_valid);
    if (token_runtime && !map.ntok_valid) return false;

    // A KV graph whose n_kv closure could not be derived must not keep the
    // full-capacity build: the live mask would stay at cap while the reuse
    // target is the applied width, and every ubatch would rebuild. Fail the
    // whole analysis so the caller rebuilds at the stock width.
    if (has_kv && !map.nkv_valid) {
        return false;
    }

    return map.any_valid();
}

} // namespace

// FA widths bound by the union binder use a CUDA/ROCm-specific head-dimension
// guard. Other non-CPU backends and unknown placements must not use that
// whitelist.
bool llama_context::union_graph_gpu_fa_accepts_width(llm_graph_result * res, uint32_t n_kv) const {
    ggml_cgraph * gf = res != nullptr ? res->get_gf() : nullptr;
    if (gf == nullptr) {
        return false;
    }
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->op != GGML_OP_FLASH_ATTN_EXT) {
            continue;
        }
        ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched.get(), node);
        if (b == backend_cpu) {
            continue;
        }
        if (b == nullptr) {
            return false;
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(b);
        ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
        const char * name = reg != nullptr ? ggml_backend_reg_name(reg) : nullptr;
        if (name == nullptr || (std::strcmp(name, "CUDA") != 0 && std::strcmp(name, "ROCm") != 0)) {
            return false;
        }
        const ggml_tensor * K = node->src[1];
        const ggml_tensor * V = node->src[2];
        if (K == nullptr || V == nullptr ||
                !union_fa_head_dim_generic((uint32_t) K->ne[0], (uint32_t) V->ne[0])) {
            return false;
        }
    }
    return true;
}

// true when the scheduler runs the whole graph on the single CPU backend.
// Used as the first-build proof for the n_outputs envelope: the split count is
// only known after alloc_graph, while a single-backend CPU scheduler is
// patch-safe by construction.
bool llama_context::union_sched_cpu_only() const {
    if (sched == nullptr || backend_cpu == nullptr) {
        return false;
    }
    if (ggml_backend_sched_get_n_backends(sched.get()) != 1) {
        return false;
    }
    return ggml_backend_sched_get_backend(sched.get(), 0) == backend_cpu;
}

// The patch requires one live split and rejects relay and pipeline contexts.
// Split input copies are checked against their allocation envelopes by the
// scheduler before any tensor is changed.
bool llama_context::union_sched_single_split() const {
    if (sched == nullptr || backend_cpu == nullptr) {
        return false;
    }
    if (backend_tp5_relay != nullptr || cparams.pipeline_parallel) {
        return false;
    }
    return ggml_backend_sched_get_n_splits(sched.get()) == 1;
}
// in-place runtime-parameter bind. The live graph is never rebuilt here: the
// symbolic shape map (union_shape_analyze) derives each dependent layout as
// linear terms and an n_kv * n_tokens cross term. The bind validates and
// applies those values.
// Bind-trace switch: GGML_TP5_UNION_BIND_TRACE=1 (the exact-one contract of
// llama-union-binder.h) turns on the per-attempt trace lines below. OFF by
// default, and deliberately so: a decode step binds once, so a 300-step run
// would emit 300 ERROR lines, and a shipped lane must not. The COUNTERS are
// the always-on observable; the trace only explains a refusal a counter
// cannot attribute (several rules share c_fallbacks), and the named
// site=<slug> in each refusal line is what makes a refusal reproducible.
static bool union_bind_trace_on() {
    static const bool on = llama_union_gate_exact_one(std::getenv("GGML_TP5_UNION_BIND_TRACE"));
    return on;
}

bool llama_context::union_rebind_apply(llm_graph_result * res,
                                       const llm_graph_params & params,
                                       const llama_union_tensor_set & set,
                                       uint32_t new_nkv,
                                       uint32_t new_nout,
                                       uint32_t new_n_rs,
                                       uint32_t new_rs_head) {
    auto & st = llama_nkv_ctx_state_get(this);
    // Every envelope refusal names its rule and what it saw. The counters say
    // how many binds were refused; a refusal nobody can attribute is a refusal
    // nobody can fix, so the one line below is part of the contract, not
    // debug output. It fires only on a refusal (never on a bind), and the
    // counter semantics are exactly the ones the sites had before.
    auto refuse_envelope = [&st](const char * site, const ggml_tensor * x,
                                 const char * what, long long a, long long b) {
        ++st.c_refused_envelope;
        ++st.c_fallbacks;
        if (x != nullptr) {
            if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: envelope refusal %s: tensor %s op %s "
                            "ne=[%lld %lld %lld %lld] nb=[%lld %lld %lld %lld] "
                            "view_offs=%lld %s (%lld vs %lld)\n",
                            "union_rebind_apply", site, x->name, ggml_op_name(x->op),
                            (long long) x->ne[0], (long long) x->ne[1],
                            (long long) x->ne[2], (long long) x->ne[3],
                            (long long) x->nb[0], (long long) x->nb[1],
                            (long long) x->nb[2], (long long) x->nb[3],
                            (long long) x->view_offs,
                            what, a, b);
        } else {
            if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: envelope refusal %s: %s (%lld vs %lld)\n",
                            "union_rebind_apply", site, what, a, b);
        }
    };
    // A step may mark no token as output, and the output-gather closure IS
    // bindable to width 0. ggml_get_rows over a zero-row index yields a
    // zero-row result (its ne[0] is the SOURCE width, its ne[1] the index
    // length), every tensor of the closure derives its row axis from that
    // result, and union_tensor_span() already treats a zero-size dimension as
    // addressing no bytes, so the patch is a shrink like any other. The one
    // thing that must not happen is a write past the live tensor:
    // llm_graph_input_out_ids::set_input() bounds its write by out_ids->ne[0]
    // for exactly this case.
    //
    // Refusing here is what this function did before the zero-output cell
    // existed, and it cost two builds plus a build width the step did not ask
    // for (nout_suppress_envelope, then graph_params() at the active count).
    // Every registered kind passes the current n_outputs here, so this is the
    // one place the rule lives.

    if (set.tensors == nullptr || set.src == nullptr || set.fixed == nullptr ||
            set.bytes == nullptr || set.root == nullptr || set.root_offs == nullptr ||
            set.root_bytes == nullptr) {
        if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: bind refusal site=set_vectors\n", "union_rebind_apply");
        ++st.c_fallbacks;
        return false;
    }

    const std::vector<ggml_tensor *> & record_t          = *set.tensors;
    const std::vector<int32_t>       & record_src        = *set.src;
    const std::vector<uint8_t>       & record_fixed      = *set.fixed;
    const std::vector<size_t>        & record_bytes      = *set.bytes;
    const std::vector<size_t>        & record_root_bytes = *set.root_bytes;
    // Bind decision trace: one line per attempt, naming every input the
    // decision reads. A refusal that cannot be attributed to a rule is not
    // diagnosable from the counters alone (several rules share a counter).
    if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: bind attempt new_nkv=%u new_nout=%u new_n_rs=%u new_head=%u | "
                    "record_cap=%u record_bound=%u cap_nout=%u bound_nout=%u has_out_ids=%d "
                    "bound_ntok=%u | base_nkv=%u base_nout=%u base_ntok=%u ntok_valid=%d "
                    "nkv_valid=%d nout_valid=%d | ubatch n_tokens=%u\n",
                    "union_rebind_apply", new_nkv, new_nout, new_n_rs, new_rs_head,
                    st.record_cap, st.record_bound, st.record_cap_nout, st.record_bound_nout,
                    st.record_has_out_ids ? 1 : 0, st.record_bound_ntok,
                    st.shape_map.base_nkv, st.shape_map.base_nout, st.shape_map.base_ntok,
                    st.shape_map.ntok_valid ? 1 : 0, st.shape_map.nkv_valid ? 1 : 0,
                    st.shape_map.nout_valid ? 1 : 0, params.ubatch.n_tokens);

    if (!res || res->get_gf() == nullptr || !st.record_valid || record_t.empty()) {
        if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: bind refusal site=no_record res=%d gf=%d valid=%d n=%zu\n",
                        "union_rebind_apply", res != nullptr ? 1 : 0,
                        (res != nullptr && res->get_gf() != nullptr) ? 1 : 0,
                        st.record_valid ? 1 : 0, record_t.size());
        ++st.c_fallbacks;
        return false;
    }
    // The single graph cache key, as the first gate: the live graph must be
    // the graph the record describes. The record is taken from this graph at
    // its build shapes, and every successful bind refreshes the key to the
    // bound shapes (see the refresh next to st.record_bound* below), so the
    // two sides always describe the same moment. A graph that was rebuilt, or
    // whose shapes were changed behind the binder's back, fails here before
    // any address is trusted.
    //
    // This gate is strictly additive: it can refuse a bind the address walk
    // would have accepted, never accept one the walk would refuse. The walk
    // below remains the authority for the tensor-level identity - the key says
    // "same topology", the walk says "same tensor objects".
    llama_topology_key live_topology{};
    {
        // The key is invariant to every runtime dimension this lane binds
        // (union_topology_key_excl() removes them), and record_topology_key is
        // refreshed to the same value after every successful bind. Re-walking
        // and re-hashing the whole graph on every decode step therefore
        // recomputes the same bytes. Cache the live key per graph object; the
        // record invalidation and the record snapshot (which re-arms the
        // record after a rebuild, including workspace address reuse) reset it.
        if (st.live_topology_gf == res->get_gf()) {
            live_topology = st.live_topology_key;
        } else {
            live_topology = llama_topology_key_of(res->get_gf(),
                    union_topology_key_excl(st.shape_map, st.record_t));
            st.live_topology_key = live_topology;
            st.live_topology_gf  = res->get_gf();
        }
        if (union_bind_trace_on()) {
            LLAMA_LOG_ERROR("%s: topology gate live hash=%016llx nodes=%u | "
                            "record hash=%016llx nodes=%u | gf=%p res_gf=%p same=%d\n",
                            "union_rebind_apply",
                            (unsigned long long) live_topology.hash, live_topology.n_nodes,
                            (unsigned long long) st.record_topology_key.hash,
                            st.record_topology_key.n_nodes,
                            (const void *) res->get_gf(), (const void *) res->get_gf(),
                            (int) (live_topology.hash == st.record_topology_key.hash &&
                                   live_topology.n_nodes == st.record_topology_key.n_nodes));
        }
        if (!llama_topology_same(live_topology, st.record_topology_key)) {
            ++st.c_refused_topology;
            ++st.c_fallbacks;
            return false;
        }
    }
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
    if (!union_sched_single_split()) {
        ++st.c_refused_backend;
        ++st.c_fallbacks;
        return false;
    }
    // Scheduler split copies are validated together with the proposed graph
    // layouts below, after the complete tensor patch has been computed.
    auto compatible = params;
    if (st.shape_map.ntok_valid && !params.ubatch.equal_seqs() &&
            params.ubatch.n_seqs_unq == 1 && params.ubatch.n_seq_tokens == 1 &&
            params.ubatch.n_seqs == params.ubatch.n_tokens &&
            params.ubatch.n_tokens > 0 && params.ubatch.n_tokens <= cparams.n_ubatch) {
        compatible.ubatch.n_tokens = res->get_params().ubatch.n_tokens;
        compatible.ubatch.n_seqs = res->get_params().ubatch.n_seqs;
    }
    if (!res->get_params().allow_reuse(compatible, /*ignore_runtime=*/true)) {
        if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: bind refusal site=allow_reuse\n", "union_rebind_apply");
        ++st.c_refused_skeleton;
        ++st.c_fallbacks;
        return false;
    }
    if (union_is_hybrid_iswa(params.mctx)) {
        ++st.c_refused_iswa;
        ++st.c_fallbacks;
        return false;
    }
    if (dynamic_cast<const llama_memory_hybrid_idx_context *>(params.mctx) != nullptr) {
        ++st.c_refused_iswa;
        ++st.c_fallbacks;
        return false;
    }
    if (union_has_attention_views(params.mctx)) {
        ++st.c_refused_reader;
        ++st.c_fallbacks;
        return false;
    }
    for (const auto & input : res->inputs) {
        if (input->union_has_readers()) {
            ++st.c_refused_reader;
            ++st.c_fallbacks;
            return false;
        }
    }
    if (union_kv_ctx(params.mctx) != nullptr && new_nkv == 0) {
        ++st.c_refused_nkv_unknown;
        ++st.c_fallbacks;
        return false;
    }
    if (const llama_kv_cache_context * kv_ctx = union_kv_ctx(params.mctx)) {
        // The K/V view offset encodes the build-time stream base (sinfo.s0);
        // the symbolic map does not model a stream switch. A multi-stream
        // cache therefore fails closed to a rebuild.
        if (kv_ctx->get_n_stream() > 1) {
            if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: bind refusal site=skeleton@4725\n", "union_rebind_apply");
            ++st.c_refused_skeleton;
            ++st.c_fallbacks;
            return false;
        }
    }
    if (new_nkv != 0) {
        // The non-x86 F16 masked-dot refusal is gone with the marking it
        // guarded: with no REDUCE_PAD_TAIL / MASKED_TAIL_GEMM on this lane the
        // value product takes the same kernel stock takes, so there is no
        // x86-only parity to protect.
        if (union_sched_cpu_only() && new_nkv > INT32_MAX - 255) {
            refuse_envelope("nkv_int32", nullptr, "new_nkv > INT32_MAX - 255",
                            (long long) new_nkv, (long long) (INT32_MAX - 255));
            return false;
        }
        if (!union_graph_gpu_fa_accepts_width(res, new_nkv)) {
            ++st.c_refused_align;
            ++st.c_fallbacks;
            return false;
        }
        if (new_nkv > st.record_cap) {
            refuse_envelope("nkv_cap", nullptr, "new_nkv > record_cap",
                            (long long) new_nkv, (long long) st.record_cap);
            return false;
        }
    }
    uint32_t stock_nkv = 0;
    if (new_nkv != 0) {
        const llama_kv_cache_context * kv_ctx = union_kv_ctx(params.mctx);
        stock_nkv = kv_ctx != nullptr ? kv_ctx->get_n_kv_stock() : 0;
        // The applied width must be known before any tensor is changed.
        if (stock_nkv < new_nkv || stock_nkv > INT32_MAX) {
            ++st.c_refused_nkv_unknown;
            ++st.c_fallbacks;
            return false;
        }
    }
    if (!st.shape_map.ntok_valid && st.record_has_out_ids && new_nout > st.record_cap_nout) {
        if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: bind refusal site=nout_over_cap new_nout=%u cap_nout=%u\n",
                        "union_rebind_apply", new_nout, st.record_cap_nout);
        ++st.c_fallbacks;
        return false;
    }
    const auto * recr = union_recr_ctx(params.mctx);
    if (recr != nullptr) {
        if (new_n_rs == 0 || new_n_rs > st.record_cap_nrs ||
                new_rs_head >= st.record_cap_nrs ||
                (uint64_t) new_rs_head + new_n_rs > st.record_cap_nrs) {
            ++st.c_refused_nrs;
            ++st.c_fallbacks;
            return false;
        }
    } else if (new_n_rs != 0 || new_rs_head != 0) {
        ++st.c_refused_nrs;
        ++st.c_fallbacks;
        return false;
    }
    if (union_replay_len_now(params.mctx) != st.record_bound_replay_len) {
        if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: bind refusal site=skeleton@4783\n", "union_rebind_apply");
        ++st.c_refused_skeleton;
        ++st.c_fallbacks;
        return false;
    }
    if (record_src.size() != record_t.size() * (GGML_MAX_SRC + 1)) {
        if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: bind refusal site=skeleton@4788\n", "union_rebind_apply");
        ++st.c_refused_skeleton;
        ++st.c_fallbacks;
        return false;
    }
    for (size_t k = 0; k < record_t.size(); ++k) {
        if (record_fixed[k]) {
            continue;
        }
        const ggml_tensor * x = record_t[k];
        for (int j = 0; j <= GGML_MAX_SRC; ++j) {
            const int32_t idx = record_src[k * (GGML_MAX_SRC + 1) + j];
            const ggml_tensor * actual = j == GGML_MAX_SRC ? x->view_src : x->src[j];
            const ggml_tensor * original = idx >= 0 && (size_t) idx < record_t.size() ? record_t[idx] : nullptr;
            if (idx < -1 || (idx >= 0 && original == nullptr) ||
                    (actual != original && (j == GGML_MAX_SRC || original == nullptr ||
                     !ggml_backend_sched_is_tensor_copy(sched.get(), original, actual)))) {
                if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: bind refusal site=address_walk k=%zu j=%d tensor=%s idx=%d actual=%s original=%s\n",
                        "union_rebind_apply", k, j, x->name, idx,
                        actual != nullptr ? actual->name : "(null)",
                        original != nullptr ? original->name : "(null)");
                if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: bind refusal site=skeleton@4808\n", "union_rebind_apply");
                ++st.c_refused_skeleton;
                ++st.c_fallbacks;
                return false;
            }
        }
    }

    const int64_t dW = (int64_t) new_nkv     - (int64_t) st.shape_map.base_nkv;
    const int64_t dN = st.record_has_out_ids
        ? (int64_t) new_nout - (int64_t) st.shape_map.base_nout
        : 0;
    const int64_t dL = (int64_t) new_n_rs - (int64_t) st.shape_map.base_n_rs;
    const int64_t dH = (int64_t) new_rs_head - (int64_t) st.shape_map.base_rs_head;
    const int64_t dT = (int64_t) params.ubatch.n_tokens - (int64_t) st.shape_map.base_ntok;
    if (union_token_delta_refused(st, dT)) {
        if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: bind refusal site=skeleton@4823\n", "union_rebind_apply");
        ++st.c_refused_skeleton;
        ++st.c_fallbacks;
        return false;
    }

    if (dW != 0 && !st.shape_map.nkv_valid) {
        if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: bind refusal site=skeleton@4829\n", "union_rebind_apply");
        ++st.c_refused_skeleton;
        ++st.c_fallbacks;
        return false;
    }
    if (dN != 0 && !st.shape_map.nout_valid) {
        if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: bind refusal site=skeleton@4834\n", "union_rebind_apply");
        ++st.c_refused_skeleton;
        ++st.c_fallbacks;
        return false;
    }
    if (dL != 0 && !st.shape_map.nrs_valid) {
        ++st.c_refused_nrs;
        ++st.c_fallbacks;
        return false;
    }
    if (dH != 0 && !st.shape_map.rs_head_valid) {
        ++st.c_refused_rs_z;
        ++st.c_fallbacks;
        return false;
    }
    // "nothing to do" is the live bound state, not the delta from the build
    // base: a request equal to the build base after an earlier bind moved the
    // graph away from it has dW/dN == 0 yet must be re-applied (the patch
    // values below are absolute, so re-applying restores the base shapes).
    //
    // This is the one gate, and the per-kind rebinds apply the same rule with
    // their own parameter live. rs_z is passed as the recorded value because
    // it is a runtime INDEX input, not a shape this function patches: it is
    // rebound on every set_inputs() by union_rsz_bind_index, so a row-only
    // move must not reach this function at all. The kinds that own a row
    // (rs_z kind) pass it live to decide whether to call here.
    // The token width is a bindable parameter only on the token lane. On every
    // other lane the recorded token width is the build width and the graph's
    // token inputs are fixed shapes, so a request that differs there is NOT a
    // bind: it is a rebuild, and the caller must see a refusal. Passing the
    // live width unconditionally would let a non-token-lane bind "succeed" on a
    // graph whose token inputs keep their build shape, i.e. compute with the
    // wrong number of rows. The symbolic map's own guard below (dT != 0 &&
    // !ntok_valid) refuses such a patch, but it is reached only after the
    // no-op rule; stating the lane here keeps the two consistent.
    const uint32_t bound_ntok = union_bound_ntok(st, params.ubatch.n_tokens);
    if (union_bind_is_noop(st, new_nkv, new_nout, bound_ntok,
                           st.record_bound_rs_z, new_rs_head)) {
        if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: bind refusal site=noop new_nkv=%u bound=%u new_nout=%u bound_nout=%u "
                        "ntok=%u bound_ntok=%u\n",
                        "union_rebind_apply", new_nkv, st.record_bound, new_nout,
                        st.record_bound_nout, bound_ntok, st.record_bound_ntok);
        ++st.c_fallbacks;
        return false;
    }

    struct union_patch_value {
        int64_t ne[4] = {};
        int64_t nb[4] = {};
        int64_t view_offs = 0;
        int64_t root_offs = 0;
    };
    std::unordered_map<uint32_t, union_patch_value> patch;
    patch.reserve(st.shape_map.nkv.size() + st.shape_map.nout.size() +
                  st.shape_map.rs_head.size() + st.shape_map.rs_len.size() +
                  st.shape_map.ntok.size() + st.shape_map.nkv_ntok.size());

    // Add all terms, including zero deltas, to restore build-base layouts.
    auto add = [&](const std::vector<llama_union_shape_coeff> & list, int64_t delta) -> bool {
        for (const auto & c : list) {
            auto result = patch.emplace(c.tensor, union_patch_value {});
            auto & v = result.first->second;
            if (result.second) {
                for (int i = 0; i < 4; ++i) {
                    v.ne[i] = c.base_ne[i];
                    v.nb[i] = c.base_nb[i];
                }
                v.view_offs = c.base_view_offs;
                v.root_offs = c.base_root_offs;
            }
            auto term = [delta](int64_t & value, int64_t coeff) {
                int64_t product;
                return !__builtin_mul_overflow(coeff, delta, &product) &&
                       !__builtin_add_overflow(value, product, &value);
            };
            for (int i = 0; i < 4; ++i) {
                if (!term(v.ne[i], c.ne[i]) || !term(v.nb[i], c.nb[i])) return false;
            }
            if (!term(v.view_offs, c.view_offs) || !term(v.root_offs, c.root_offs)) return false;
        }
        return true;
    };
    int64_t mixed_delta;
    if (__builtin_mul_overflow(dW, dT, &mixed_delta) ||
            !add(st.shape_map.nkv, dW) || !add(st.shape_map.nout, dN) ||
            !add(st.shape_map.rs_len, dL) || !add(st.shape_map.rs_head, dH) ||
            !add(st.shape_map.ntok, dT) || !add(st.shape_map.nkv_ntok, mixed_delta)) {
        refuse_envelope("patch_overflow", nullptr, "dW/dN/dT/dH/dNrs overflow",
                        (long long) dW, (long long) dT);
        return false;
    }

    std::vector<uint32_t> keys;
    keys.reserve(patch.size());
    for (const auto & kv : patch) {
        keys.push_back(kv.first);
    }
    std::sort(keys.begin(), keys.end());

    // validate every change before touching the live graph
    bool any_change = false;
    for (uint32_t k : keys) {
        if (k >= record_t.size()) {
            if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: bind refusal site=skeleton@4936\n", "union_rebind_apply");
            ++st.c_refused_skeleton;
            ++st.c_fallbacks;
            return false;
        }
        const union_patch_value & v = patch[k];
        ggml_tensor * x = record_t[k];
        if (record_fixed[k]) {
            bool changed = v.view_offs != (int64_t) x->view_offs;
            for (int i = 0; i < 4; ++i) {
                changed = changed || v.ne[i] != x->ne[i] || v.nb[i] != x->nb[i];
            }
            if (changed) {
                if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: bind refusal site=skeleton@4948\n", "union_rebind_apply");
                ++st.c_refused_skeleton;
                ++st.c_fallbacks;
                return false;
            }
            continue;
        }
        // the map is absolute: an entry equal to the live tensor is not a
        // change. When no entry differs the graph is already at the target
        // state (e.g. a repeat bind) and the caller must not count a bind.
        if (v.view_offs != (int64_t) x->view_offs) {
            any_change = true;
        }
        for (int i = 0; i < 4; ++i) {
            if (v.ne[i] != x->ne[i] || v.nb[i] != (int64_t) x->nb[i]) {
                any_change = true;
            }
        }
        for (int i = 0; i < 4; ++i) {
            if (v.ne[i] < 0 || v.nb[i] < 0) {
                refuse_envelope("patch_negative", x, "patched ne/nb < 0",
                                (long long) v.ne[i], (long long) v.nb[i]);
                return false;
            }
        }
        if (v.view_offs < 0 || v.root_offs < 0) {
            refuse_envelope("patch_negative_offs", x, "patched view/root offs < 0",
                            (long long) v.view_offs, (long long) v.root_offs);
            return false;
        }
        ggml_tensor tmp = *x;
        for (int i = 0; i < 4; ++i) {
            tmp.ne[i] = v.ne[i];
            tmp.nb[i] = (size_t) v.nb[i];
        }
        tmp.view_offs = (size_t) v.view_offs;
        if (x->view_src == nullptr) {
            const size_t span = union_tensor_span(&tmp);
            if (span == SIZE_MAX || (!st.shape_map.ntok_valid && span > record_bytes[k])) {
                refuse_envelope("owned_span", x, "span vs record_bytes",
                                (long long) span, (long long) record_bytes[k]);
                return false;
            }
        } else {
            const size_t span = union_tensor_span(&tmp);
            if (span == SIZE_MAX) {
                refuse_envelope("view_span", x, "view span unrepresentable",
                                (long long) span, -1);
                return false;
            }
            size_t root_limit = record_root_bytes[k];
            if (st.shape_map.ntok_valid) {
                const int32_t root_index = (*set.root)[k];
                if (root_index < 0 || (size_t) root_index >= record_t.size()) return false;
                if (!record_fixed[root_index]) {
                    ggml_tensor root_tensor = *record_t[root_index];
                    const auto root_patch = patch.find((uint32_t) root_index);
                    if (root_patch != patch.end()) {
                        for (int i = 0; i < 4; ++i) {
                            if (root_patch->second.ne[i] < 0 || root_patch->second.nb[i] < 0) return false;
                            root_tensor.ne[i] = root_patch->second.ne[i];
                            root_tensor.nb[i] = (size_t) root_patch->second.nb[i];
                        }
                    }
                    root_limit = union_tensor_span(&root_tensor);
                    if (root_limit == SIZE_MAX) return false;
                }
            }
            if (span != 0 && ((size_t) v.root_offs > root_limit ||
                    span > root_limit - (size_t) v.root_offs)) {
                refuse_envelope("view_root", x, "span vs root_limit - root_offs",
                                (long long) span, (long long) root_limit);
                return false;
            }
        }
    }
    // No change is a valid bind, not a refusal. The three parameters are
    // simultaneous: when a step moved only rs_z (a runtime index that no
    // tensor shape carries), the shape map is already at its targets, and the
    // bind must still record the new row and refresh the input metadata below.
    // Nothing is patched, so the graph cannot be harmed; the hash is re-taken
    // over the unchanged shapes and must equal the reference.
    std::vector<ggml_backend_sched_tensor_layout> layouts;
    layouts.reserve(keys.size());
    std::vector<size_t> view_offsets;
    if (st.shape_map.ntok_valid) view_offsets.reserve(keys.size());
    for (uint32_t k : keys) {
        if (record_fixed[k]) {
            continue;
        }
        const union_patch_value & v = patch[k];
        ggml_backend_sched_tensor_layout layout = {};
        layout.tensor = record_t[k];
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            layout.ne[i] = v.ne[i];
            layout.nb[i] = (size_t) v.nb[i];
        }
        if (st.shape_map.ntok_valid) view_offsets.push_back((size_t) v.view_offs);
        layouts.push_back(layout);
    }
    // Synchronize before preflight and commit; neither may mutate on failure.
    ggml_backend_sched_synchronize(sched.get());
    if (!st.shape_map.ntok_valid) {
        if (!ggml_backend_sched_update_split_copies(sched.get(), layouts.data(), layouts.size(), false) ||
                !ggml_backend_sched_update_split_copies(sched.get(), layouts.data(), layouts.size(), true)) {
            refuse_envelope("split_copies", nullptr, "sched split copy layout refused",
                            (long long) layouts.size(), -1);
            return false;
        }
    }

    if (st.shape_map.ntok_valid) {
        if (!ggml_backend_sched_replan_graph(sched.get(), record_t.data(), record_fixed.data(),
                                            record_t.size(), layouts.data(), view_offsets.data(), layouts.size())) {
            st.record_valid = false;
            ++st.c_refused_backend;
            ++st.c_fallbacks;
            return false;
        }
        // The replan just reallocated the arena at the patched shapes, so the
        // arena width is exactly new_nkv. Raising the bound here is not
        // headroom: it follows an allocation that already happened. The
        // legacy lane below patches in place inside the envelope the graph
        // was built with, so its arena width does not move.
        if (new_nkv != 0) {
            st.record_arena_n_kv = new_nkv;
        }
    } else {
        // The legacy lanes use their existing allocation envelopes.
        for (uint32_t k : keys) {
            ggml_tensor * x = record_t[k];
            if (record_fixed[k]) continue;
            const union_patch_value & v = patch[k];
            for (int i = 0; i < 4; ++i) {
                x->ne[i] = v.ne[i];
                x->nb[i] = (size_t) v.nb[i];
            }
            x->view_offs = (size_t) v.view_offs;
        }
    }


    // No reduction-order fixup is applied on this lane, by design.
    //
    // The binder used to mark the softmax and the value product with
    // GGML_TENSOR_FLAG_REDUCE_PAD_TAIL / MASKED_TAIL_GEMM and pin their
    // accumulation tree to a padded width in op_params[2]. That emulated the
    // padded graph's order, and it could not work: the marked node's tree
    // became a function of the build envelope while stock's tree follows its
    // own ne00, so the two trees differ whenever the envelope is not stock's
    // padded width, however constant the envelope is.
    //
    // The in-place patch above is enough on its own. The attention products
    // carry ATTN_SCORES / ATTN_MASK / ATTN_SOFT_MAX from the graph build
    // (llama-graph.cpp, the two ggml_mul_mat_set_attn_* calls), and those route
    // to the ordered CPU kernels, which accumulate sequentially over the
    // surviving terms. A narrowed graph and a graph built at the same width
    // therefore run the same kernels with the same tree, with no marking at
    // all. See ANALYSIS-EQUALITY.md in the workdir for the site table.

    // every view's data pointer follows its (possibly patched) source; the
    // record is in source-before-consumer order
    for (size_t k = 0; k < record_t.size(); ++k) {
        if (record_fixed[k]) {
            continue;
        }
        ggml_tensor * x = record_t[k];
        if (x->view_src != nullptr && x->view_src->data != nullptr) {
            x->data = (char *) x->view_src->data + x->view_offs;
        }
    }
    ggml_backend_sched_refresh_split_copy_views(sched.get());
    const uint32_t old_bound_nout = st.record_bound_nout;

    st.record_bound         = new_nkv;
    st.record_bound_nout    = new_nout;
    st.record_bound_n_rs    = new_n_rs;
    st.record_bound_ntok    = params.ubatch.n_tokens;
    st.record_bound_rs_head = new_rs_head;
    st.record_bound_rs_z    = union_rs_z_now(params.mctx); // runtime index, diagnostic only

    // Two identities are refreshed here, and both must be:
    //
    //   record_topology_key  the live graph's shapes have moved to the bound
    //     values, so the single graph cache key must move with them. It is
    //     refreshed with the bound state so that "same graph" in the gate at
    //     the top of this function means "same graph as the one currently
    //     live". Without this the next bind would compare the record's
    //     build-shape key against a patched graph and refuse every reuse after
    //     the first. The address walk remains the authority for the
    //     tensor-level identity; this key is the topology half of the same
    //     question.
    //
    //   topology_hash        the width-table hash of llama-union-topology.h,
    //     re-taken over the patched graph with the n_kv, n_outputs and rs_z
    //     dimensions excluded. An unchanged value is the proof that this bind
    //     changed shapes only: same node set, same edges, same op types, same
    //     op params, same non-runtime dimensions. It is the pillar's public
    //     observable (llama_get_nkv_runtime_stats().topology_hash and
    //     llama_get_union_diagnostics().topology_hash), so it is NOT replaced
    //     by the key, which is invariant to those dimensions by construction
    //     and would therefore be stable for the wrong reason.
    //
    // The bind just moved every runtime-coupled shape to the three targets, so
    // arm the same n_kv as the graph's runtime width before re-taking the
    // hash. A hash that cannot be taken means the bind cannot prove what it
    // did, so it is refused and the caller rebuilds.
    // The gate above already has the live key, and the key is invariant to the
    // runtime dimensions this bind patches (the lane's exclusion set removes
    // them), so the bound graph hashes to the same value. Reuse it instead of
    // re-walking the graph after every bind.
    st.record_topology_key = live_topology;
    st.live_topology_key   = live_topology;
    st.live_topology_gf    = res->get_gf();
    if (new_nkv != 0) {
        // The bound is the arena the replan above just allocated, not the
        // width the graph was recorded at: on the token lane the two differ
        // and the build width would refuse the width the bind just applied.
        // The arena is the allocation that backs the patched shapes, so this
        // is the physical capacity boundary, not the KV envelope.
        if (!st.width_table.set_runtime_n_kv(new_nkv, st.record_arena_n_kv)) {
            // The patch and the replan are already committed, so the record
            // no longer describes a graph whose width can be stated. Drop it:
            // returning false alone would let the fresh path see
            // delta_pending() == false (record_bound already moved) and keep
            // executing the half-committed graph.
            st.record_valid = false;
            if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: bind refusal site=arm@5214\n", "union_rebind_apply");
            ++st.c_refused_skeleton;
            ++st.c_fallbacks;
            return false;
        }
    }
    {
        uint64_t hash = 0;
        if (!st.width_table.topology_hash(st.record_nodes, hash)) {
            st.topology_hashed = false;
            st.topology_hash   = 0;
            // The patch and the replan are already committed. The record must
            // die with this refusal: the fresh path computes delta_pending()
            // AFTER the bind, and record_bound has already moved, so a
            // still-valid record would make it skip the rebuild and execute
            // the half-committed graph.
            st.record_valid = false;
            if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: bind refusal site=skeleton@5155\n", "union_rebind_apply");
            ++st.c_refused_skeleton;
            ++st.c_fallbacks;
            return false;
        }
        st.topology_hashed = true;
        st.topology_hash   = hash;
    }

    {
        llm_graph_params bound_params = st.shape_map.ntok_valid ? params : res->get_params();
        bound_params.n_outputs       = new_nout;
        bound_params.n_outputs_build = 0;
        res->set_params(bound_params);
    }
    ++st.c_applied;
    ++st.c_patch_binds;
    if (new_nout > old_bound_nout) {
        ++st.c_nout_grow_binds;
    }

    // refresh the cached runtime metadata of the live graph inputs so the next
    // can_reuse() sees the patched graph as current (rs_z/head/replay_len)
    for (auto & input : res->inputs) {
        input->union_bind_runtime();
    }
    // poka-yoke: this one bind must have applied ALL THREE parameters. The
    // shape map was evaluated at new_nkv, new_nout and new_n_rs/new_rs_head
    // together, and the inputs now report what they execute at. A graph whose
    // closure still reports another width means the patch and the metadata
    // disagree; refuse the bind so the caller rebuilds instead of computing
    // on a graph nobody can describe.
    if (st.record_has_out_ids && res->outputs_bound_target() != new_nout) {
        // Same rule as the hash refusal: the patch is committed, so the
        // record can no longer describe the live graph. Without this the
        // fresh path sees no pending delta and keeps the graph.
        st.record_valid = false;
        if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: bind refusal site=skeleton@5187\n", "union_rebind_apply");
        ++st.c_refused_skeleton;
        ++st.c_fallbacks;
        return false;
    }
    // Refresh per-input context pointers and check the full active layout.
    if (st.shape_map.ntok_valid && !res->can_reuse(params)) {
        st.record_valid = false;
        if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: bind refusal site=skeleton@5194\n", "union_rebind_apply");
        ++st.c_refused_skeleton;
        ++st.c_fallbacks;
        return false;
    }
    return true;
}

// Every registered kind below is a GATE, not a lane of its own: each one
// validates the preconditions it owns and then calls union_rebind_apply with
// the SAME three targets (n_kv, n_outputs, n_rs/head). That is what makes the
// three parameters simultaneous: whichever gate fires, one bind applies all
// three deltas, and a gate whose own target did not move cannot hide a delta
// in another target (the delta test lives in apply, not here).
llama_union_rebind_result llama_context::union_rebind_n_kv(llm_graph_result * res, const llm_graph_params & params, const llama_union_tensor_set & set) {

    auto & st = llama_nkv_ctx_state_get(this);

    // NKV_BINDER_STATS: every binder invocation counts as an attempt
    ++st.c_attempts;

    const uint32_t nkv_now = union_nkv_now(params.mctx);
    if (nkv_now == 0) {
        ++st.c_fallbacks;
        return llama_union_rebind_result::declined;
    }
    // Retain the existing FA width guard before attempting a bind.
    if (!union_graph_gpu_fa_accepts_width(res, nkv_now)) {
        ++st.c_refused_align;
        ++st.c_fallbacks;
        return llama_union_rebind_result::declined;
    }
    // envelope: the live graph was built and sched-allocated AT the recorded
    // cap (see process_ubatch envelope build), so any n_kv in (0, cap] has
    // real backing. record_bound tracks the width the live graph's nkv-coupled
    // shapes are CURRENTLY bound at; anything above the cap has no allocated
    // backing -> full rebuild.
    // No record means no envelope: nothing was built, so nothing can be
    // outside one. Counting an envelope refusal here made the audit's
    // refused_envelope=1 name a rule that had not been asked anything; the
    // pre-record attempt is the documented "legitimate pre-record refusal"
    // and declines.
    if (!st.record_valid) {
        ++st.c_fallbacks;
        return llama_union_rebind_result::declined;
    }
    if (nkv_now > st.record_cap) {
        if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: envelope refusal nkv_precheck: nkv_now > record_cap (%u vs %u)\n",
                        "union_rebind_n_kv", nkv_now, st.record_cap);
        ++st.c_refused_envelope;
        ++st.c_fallbacks;
        return llama_union_rebind_result::declined;
    }
    // The one no-op rule, with this kind's own parameter live (n_kv) and the
    // other four at their recorded values: this kind declines only when n_kv
    // has not moved, even if n_outputs or the token width also changed - the
    // dispatch is first-kind-wins and union_rebind_apply patches every changed
    // parameter.
    if (union_bind_is_noop(st, nkv_now, st.record_bound_nout, st.record_bound_ntok,
                           st.record_bound_rs_z, st.record_bound_rs_head)) {
        ++st.c_fallbacks;
        return llama_union_rebind_result::declined;
    }

    return union_rebind_apply(res, params, set, nkv_now, n_outputs,
                              union_n_rs_now(params.mctx), union_rs_head_now(params.mctx))
        ? llama_union_rebind_result::applied : llama_union_rebind_result::refused;
}

llama_union_rebind_result llama_context::union_rebind_n_tokens(llm_graph_result * res, const llm_graph_params & params, const llama_union_tensor_set & set) {
    auto & st = llama_nkv_ctx_state_get(this);
    ++st.c_attempts;
    // The token width is a bindable parameter only when the symbolic token
    // shape map is valid. When it is not, this lane cannot bind a token width
    // at all, so the kind declines for that reason - not because the width
    // happens to equal the recorded one. Stating it here keeps the helper's
    // "own parameter live, others recorded" contract honest.
    if (!st.shape_map.ntok_valid) {
        ++st.c_fallbacks;
        return llama_union_rebind_result::declined;
    }
    // The one no-op rule, with this kind's own parameter live (the token
    // width) and the other four at their recorded values.
    if (union_bind_is_noop(st, st.record_bound, st.record_bound_nout,
                           params.ubatch.n_tokens,
                           st.record_bound_rs_z, st.record_bound_rs_head)) {
        ++st.c_fallbacks;
        return llama_union_rebind_result::declined;
    }
    return union_rebind_apply(res, params, set, union_nkv_now(params.mctx), n_outputs,
                              union_n_rs_now(params.mctx), union_rs_head_now(params.mctx))
        ? llama_union_rebind_result::applied : llama_union_rebind_result::refused;
}

llama_union_rebind_result llama_context::union_rebind_rs_z(llm_graph_result * res, const llm_graph_params & params, const llama_union_tensor_set & set) {
    auto & st = llama_nkv_ctx_state_get(this);

    ++st.c_attempts;

    // The rs_z zero row itself is a runtime index input (s_zero) consumed by
    // the explicit-dependency zero-row op; it is not a shape parameter and -1
    // is a valid no-op, so a row change alone needs no bind. This gate covers
    // the coupled recurrent state shapes: the used cell count (n_rs) and the
    // head-selected copy views. A row-only change must still let the bind
    // through when n_kv or n_outputs moved, which is why the gate does not
    // test the row.
    if (union_recr_ctx(params.mctx) == nullptr) {
        ++st.c_refused_rs_z;
        ++st.c_fallbacks;
        return llama_union_rebind_result::declined;
    }
    // Two refusals, then the one no-op rule. The refusals are this kind's own
    // preconditions and both are kept: a recurrent context that reports no
    // used cell (n_rs == 0) has no state to bind, and rs_z == -1 (no free
    // zero-state cell) is outside the linear rule domain, so binding it would
    // write a stale zero view. Neither is a "nothing moved" answer, so neither
    // belongs in the no-op rule.
    const uint32_t nrs_now  = union_n_rs_now(params.mctx);
    const int32_t  rs_z_now = union_rs_z_now(params.mctx);
    const uint32_t head_now = union_rs_head_now(params.mctx);
    if (nrs_now == 0) {
        ++st.c_refused_nrs;
        ++st.c_fallbacks;
        return llama_union_rebind_result::declined;
    }
    if (rs_z_now < 0) {
        // rs_z == -1 (no free zero-state cell) is outside the linear rule
        // domain; rebuild instead of binding a stale zero view
        ++st.c_refused_rs_z;
        ++st.c_fallbacks;
        return llama_union_rebind_result::declined;
    }
    // The one no-op rule, with BOTH of this kind's parameters live: this is
    // the one kind that owns two (rs_z and the recurrent head), so it declines
    // only when both have stayed put. The other three are at their recorded
    // values, as in the other kinds.
    if (union_bind_is_noop(st, st.record_bound, st.record_bound_nout,
                           st.record_bound_ntok, rs_z_now, head_now)) {
        ++st.c_fallbacks;
        return llama_union_rebind_result::declined;
    }

    return union_rebind_apply(res, params, set, union_nkv_now(params.mctx), n_outputs,
                              nrs_now, head_now)
        ? llama_union_rebind_result::applied : llama_union_rebind_result::refused;
}

llama_union_rebind_result llama_context::union_rebind_n_outputs(llm_graph_result * res, const llm_graph_params & params, const llama_union_tensor_set & set) {
    auto & st = llama_nkv_ctx_state_get(this);

    ++st.c_attempts;

    const uint32_t nout_now = n_outputs;
    if (!st.record_has_out_ids) {
        // this graph has no output-gather closure; the active count is not a
        // graph key. A zero-width closure is still a closure: has_out_ids()
        // distinguishes it from "no closure" (both report envelope 0).
        ++st.c_fallbacks;
        return llama_union_rebind_result::declined;
    }
    // A zero active count is a SHRINK, not a grow: `0 > record_cap_nout` is
    // false, so the zero step never takes this branch and needs no guard of
    // its own. union_rebind_apply() patches the closure to zero rows like any
    // other width.
    //
    // nout_now > record_cap_nout is the grow case. On a record build the
    // envelope is the high-water mark max(active, observed maximum) (see
    // graph_params), so this fires exactly when the request is above every
    // width built so far. The caller rebuilds once at the new request; that
    // rebuild raises the mark, and every later width up to it binds in
    // place. Counted separately from the n_kv envelope.
    if (!st.shape_map.ntok_valid && nout_now > st.record_cap_nout) {
        ++st.c_refused_nout;
        ++st.c_fallbacks;
        return llama_union_rebind_result::declined;
    }
    // The one no-op rule, with this kind's own parameter live (n_outputs) and
    // the other four at their recorded values.
    if (union_bind_is_noop(st, st.record_bound, nout_now, st.record_bound_ntok,
                           st.record_bound_rs_z, st.record_bound_rs_head)) {
        ++st.c_fallbacks;
        return llama_union_rebind_result::declined;
    }

    return union_rebind_apply(res, params, set, union_nkv_now(params.mctx), nout_now,
                              union_n_rs_now(params.mctx), union_rs_head_now(params.mctx))
        ? llama_union_rebind_result::applied : llama_union_rebind_result::refused;
}

std::unique_ptr<llama_context::llama_shape_plan> llama_context::make_shape_plan(uint32_t n_tokens) {
    const size_t max_nodes = graph_max_nodes(n_tokens);
    auto plan = std::make_unique<llama_shape_plan>();
    plan->res = std::make_unique<llm_graph_result>(max_nodes);
    plan->res->set_owner(this);
    if (!moe_cache_configured) {
        LLAMA_LOG_ERROR("%s: no resolved MoE cache configuration; refusing a per-shape scheduler "
                        "with the environment default\n", __func__);
        return nullptr;
    }
    plan->sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                                             max_nodes, cparams.pipeline_parallel, cparams.op_offload));
    if (!plan->sched) {
        LLAMA_LOG_ERROR("%s: failed to create a per-shape scheduler\n", __func__);
        return nullptr;
    }
    if (!llama_apply_moe_cache_config(plan->sched.get(), llama_to_ggml_moe_cache_mode(moe_cache_mode_resolved),
                                      moe_cache_budget_mib)) {
        LLAMA_LOG_ERROR("%s: failed to apply the MoE cache configuration to the per-shape scheduler\n", __func__);
        plan->sched.reset();
        return nullptr;
    }
    ggml_backend_sched_set_eval_callback(plan->sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
    return plan;
}

void llama_context::flush_shape_plans() {
    for (auto & plan : shape_plans) {
        if (plan && plan->sched) {
            ggml_backend_sched_synchronize(plan->sched.get());
        }
        plan.reset();
    }
    shape_plan_cache.flush();
    gf_res_current    = nullptr;
    shape_last_sched  = nullptr;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch &     ubatch,
                                                 llm_graph_type           gtype,
                                                 llama_memory_context_i * mctx,
                                                 ggml_status &            ret) {
    const int64_t diag_start = ggml_time_us();
    const auto * diag_hybrid = dynamic_cast<const llama_memory_hybrid_idx_context *>(mctx);
    const uint32_t diag_replay = diag_hybrid ? diag_hybrid->get_recr()->get_replay_len() : 0;
    int64_t diag_build = 0, diag_alloc = 0;
    if (backend_tp5_relay != nullptr) {
        // A retained input batch owns this context's host graph buffers until
        // completion. Drain before memory updates, graph reset, or set_inputs.
        // Do not account the current, not-yet-submitted token as synchronized.
        ggml_backend_sched_synchronize(sched.get());
        // The previous decode may have run on a per-shape scheduler. Drain it
        // too before any entry can be evicted or rebuilt.
        if (shape_last_sched != nullptr && shape_last_sched != sched.get()) {
            ggml_backend_sched_synchronize(shape_last_sched);
        }
    }

    // The out_ids tensor is built at the high-water envelope (or at the
    // active count when no envelope is active). An active count above
    // n_outputs_max would write past its buffer in set_input; fail closed
    // instead of relying on the caller contract.
    if (n_outputs > cparams.n_outputs_max) {
        LLAMA_LOG_ERROR("%s: n_outputs %u exceeds n_outputs_max %u\n", __func__, n_outputs, cparams.n_outputs_max);
        ret = GGML_STATUS_FAILED;
        return nullptr;
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

    if (prefab) {
        bool hook;
        if (llama_prefab_decouple_enabled()) {
            // decouple path: the gtype filter is part of the request contract;
            // A passes every DEFAULT/DECODER/DECODER_MTP ubatch to B and B's
            // own runtime gate (via the request) declines what it cannot run.
            // A does not read B's mtp_mode flag here to build the filter.
            hook = gtype == LLM_GRAPH_TYPE_DEFAULT || gtype == LLM_GRAPH_TYPE_DECODER ||
                   gtype == LLM_GRAPH_TYPE_DECODER_MTP;
        } else {
            hook = gtype == LLM_GRAPH_TYPE_DEFAULT || gtype == LLM_GRAPH_TYPE_DECODER ||
                   (prefab->mtp_mode && gtype == LLM_GRAPH_TYPE_DECODER_MTP);
        }
        if (hook) {
            llm_graph_result * res_env = nullptr;
            if (prefab_process_ubatch(ubatch, gtype, mctx, res_env, ret)) {
                return res_env;
            }
        }
    }
    if (prefab) {
        prefab->active = nullptr;
    }
    if (llama_prefab_stats_enabled()) {
        prefab_stats.n_ubatch_stock++;
    }

    // arm the runtime-length n_kv lane on the KV context: the batch context
    // is constructed without an lctx, so the width query carries the binder
    // state on the object. Only a live record proves the symbolic shape map
    // can patch the width in place, and the live split graph must be
    // patch-safe (one split with scheduler-validated copies). Otherwise the
    // stock padded width stays.
    //
    // This runs before the token-lane decision below because that decision
    // reads the width the KV context reports through union_nkv_now().
    //
    // nkv_exact_enabled is the GGML_TP5_UNION_BINDER_EXACT selection: with it
    // off the exact target is never armed, so the stock padded width is what
    // the graph executes at. That is the A/B control lane, not a fallback.
    if (llama_kv_cache_context * kv_ctx = union_kv_ctx(mctx)) {
        const llama_nkv_ctx_state * st = union_binder_state_peek();
        kv_ctx->set_nkv_exact_target(st != nullptr && st->nkv_enabled && !st->env_disabled &&
                                     st->nkv_exact_enabled && st->record_valid &&
                                     union_sched_single_split());
    }

    const llama_nkv_ctx_state * token_state = union_binder_state_peek();
    const bool token_requested = token_state && token_state->nkv_enabled && token_state->ntok_enabled;
    const bool token_recurrent = union_recr_ctx(mctx) != nullptr ||
        dynamic_cast<const llama_memory_hybrid_context *>(mctx) != nullptr ||
        dynamic_cast<const llama_memory_hybrid_idx_context *>(mctx) != nullptr;
    const bool plain_kv = dynamic_cast<const llama_kv_cache_context *>(mctx) != nullptr;
    const ggml_type type_k = token_requested && plain_kv ? get_kv_type_k() : GGML_TYPE_COUNT;
    const ggml_type type_v = token_requested && plain_kv ? get_kv_type_v() : GGML_TYPE_COUNT;
    const bool plain_kv_types = (type_k == GGML_TYPE_F16 || type_k == GGML_TYPE_F32) &&
                                (type_v == GGML_TYPE_F16 || type_v == GGML_TYPE_F32);
    const bool token_candidate = token_requested && !token_recurrent && plain_kv && plain_kv_types &&
        union_sched_cpu_only() && model.arch == LLM_ARCH_LLAMA && !cparams.flash_attn && !cparams.embeddings &&
        !cparams.embeddings_nextn && !cparams.mtp_chain &&
        cparams.kv_unified && cparams.n_seq_max == 1 && ubatch.token && !ubatch.embd &&
        !ubatch.equal_seqs() && ubatch.n_seqs_unq == 1 && ubatch.n_seq_tokens == 1 &&
        ubatch.n_seqs == ubatch.n_tokens && ubatch.n_tokens > 0;
    if (token_requested && !token_candidate) {
        auto & st = llama_nkv_ctx_state_get(this);
        ++st.c_ntok_fallbacks;
        if (token_recurrent) ++st.c_ntok_recurrent_hybrid_fallbacks;
    }

    // Pillar 5: per-shape scheduler/graph selection. A hit reuses the entry's
    // llm_graph_result and scheduler; can_reuse() below still decides whether
    // the graph itself is current. Relay contexts default to two slots so the
    // 1-row decode and 2-row MTP verify shapes do not rebuild on every switch;
    // an explicit GGML_TP5_LLAMA_PLAN_CACHE also enables the cache elsewhere.
    llm_graph_result *   res       = gf_res_prev.get();
    ggml_backend_sched_t run_sched = sched.get();
    int                  shape_slot = -1;
    bool                 shape_hit  = false;
    llama_shape_cache::key shape_key{};
    if (shape_plan_enabled()) {
        shape_key.n_tokens     = ubatch.n_tokens;
        shape_key.n_seq_tokens = ubatch.n_seq_tokens;
        shape_key.n_outputs    = n_outputs;
        shape_key.n_seqs       = ubatch.n_seqs;
        shape_key.n_seqs_unq   = ubatch.n_seqs_unq;
        shape_key.equal_seqs   = ubatch.b_equal_seqs;
        const llama_kv_cache_context * shape_kv = union_kv_ctx(mctx);
        shape_key.n_kv_target = shape_kv != nullptr ? shape_kv->get_n_kv_target() : 0;
        shape_key.gtype       = uint32_t(gtype);
        shape_key.alloc_gen   = alloc_gen_;
        const auto got = shape_plan_cache.acquire(shape_key, ++shape_plan_clock);
        shape_slot = got.index;
        shape_hit  = got.hit;
        if (!got.hit) {
            // The victim's scheduler is idle: the previous decode was drained
            // at the top of this function. Free it before installing a new
            // entry in the same slot, and drop any stale back-pointers.
            if (shape_plans[got.index]) {
                if (shape_plans[got.index]->res.get() == gf_res_current) {
                    gf_res_current = nullptr;
                }
                if (shape_plans[got.index]->sched.get() == shape_last_sched) {
                    shape_last_sched = nullptr;
                }
            }
            shape_plans[got.index].reset();
            shape_plans[got.index] = make_shape_plan(ubatch.n_tokens);
            if (!shape_plans[got.index]) {
                // Allocation failure is not a decode failure: keep the legacy
                // single-scheduler path for this epoch.
                shape_plan_cache.invalidate(got.index);
                shape_slot = -1;
            }
        }
        if (shape_slot >= 0) {
            res       = shape_plans[shape_slot]->res.get();
            run_sched = shape_plans[shape_slot]->sched.get();
        }
    }
    auto * gf  = res->get_gf();

    // previous live envelope, captured before any bind/rebuild can clear it:
    // a rebuild that raises record_cap_nout is the single envelope-expanding
    // rebuild and is counted as such. record_valid is false when the live
    // graph is stock (no envelope), so the first envelope build is not a
    // "grow".
    const llama_nkv_ctx_state * st_prev = union_binder_state_peek();
    const bool     prev_record_valid = st_prev != nullptr && st_prev->record_valid;
    const uint32_t prev_cap_nout     = prev_record_valid ? st_prev->record_cap_nout : 0;

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    auto gparams = graph_params(res, ubatch, mctx, gtype);
    gparams.sched = run_sched;

    bool reused = !graph_reuse_disable && res->can_reuse(gparams);

    if (!reused && !graph_reuse_disable && union_binder_state_peek() != nullptr &&
            union_binder_state_peek()->nkv_enabled) {
        // union-graph runtime parameter binding: the previous graph is a
        // fixed-topology envelope build, so a runtime parameter move (n_kv,
        // n_outputs, recurrent rs_z/head) is a pure arithmetic metadata patch
        // against the symbolic shape map recorded at build time. No scratch
        // graph is built. Fail-closed: any mismatch falls back to a rebuild.
        auto & union_st = llama_nkv_ctx_state_get(this);
        reused = union_st.union_binder.rebind_any(this, res, gparams);
        if (!reused && prev_record_valid) {
            ++union_st.c_rebuild_fallbacks;
        }
    }

    // rs_z runtime parameter: union_rsz_bind_index() writes the zero row into
    // the live graph's I32 [1] tensor. A row that names nothing, or that only
    // some of the ubatch's cells read, must never reach compute, so the value
    // is checked here, before ANY reuse decision (the can_reuse() above, the
    // bind above, or a fresh build below):
    //   - no live lane (rsz_rt_active false): nothing is written, nothing to
    //     check; the stock graph keys on rs_z itself.
    //   - rs_z < 0: no free zero-state cell to name -> _cell.
    //   - rs_z >= zero_view_rows (the cell capacity): names no row of any
    //     group -> _envelope.
    //   - the state tensor is not a whole number of mem_size groups -> _ring.
    //   - the ubatch's sequences disagree on the zero row -> _mixed.
    // Either refusal disables the lane for this context (the live graph is
    // rebuilt stock below, which carries no index tensor), so the refusal is
    // counted once, not once per decode.
    if (union_binder_state != nullptr) {
        auto & union_st = *union_binder_state;
        if (union_st.rsz_rt_active) {
            const int32_t rsz_now = union_rs_z_now(mctx);
            // the rules are llama_rsz_rt_check_row() and
            // llama_rsz_rt_check_relay_row() (llama-nout-state.h), pure
            // predicates a model-free test pins. The relay form is used only
            // when the ring is live: with n_rs_seq == 0 the state tensor is
            // one group and the two rules agree on every input.
            const bool relay_ring = backend_tp5_relay != nullptr && union_st.rsz_rt_state_rows > 0;
            const llama_rsz_rt_row_check row = relay_ring
                ? llama_rsz_rt_check_relay_row(rsz_now, union_st.rsz_rt_zero_view_rows,
                        union_st.rsz_rt_state_rows)
                : llama_rsz_rt_check_row(rsz_now, union_st.rsz_rt_zero_view_rows);
            // One graph carries ONE zero row. rs_idx is per sequence and a
            // sequential ubatch split can place two sequences with different
            // rollback depths in one ubatch, so the row can be ambiguous. That
            // is checked here, before any reuse decision and before
            // set_inputs(), because a wrong row would be computed with, not
            // merely reported afterwards.
            bool mixed_depth = false;
            if (row == llama_rsz_rt_row_check::ok) {
                const auto * recr_chk = union_recr_ctx(mctx);
                int32_t first = -2;
                for (uint32_t s = 0; s < ubatch.n_seqs_unq && recr_chk != nullptr; ++s) {
                    const int32_t r = recr_chk->get_rs_zero_row(ubatch.seq_id_unq[s]);
                    if (s == 0) {
                        first = r;
                    } else if (r != first) {
                        mixed_depth = true;
                        break;
                    }
                }
            }
            if (row != llama_rsz_rt_row_check::ok || mixed_depth) {
                size_t & cause = mixed_depth                                  ? union_st.c_rsz_rt_refused_mixed_depth :
                                 row == llama_rsz_rt_row_check::cell          ? union_st.c_rsz_rt_refused_cell :
                                 row == llama_rsz_rt_row_check::envelope      ? union_st.c_rsz_rt_refused_envelope :
                                                                                union_st.c_rsz_rt_refused_ring;
                union_st.rsz_rt_refuse(cause);
                // the live graph keeps its index tensor until the stock
                // rebuild below replaces it; no path writes through it without
                // rsz_rt_active, so clearing the flag is enough
                union_st.rsz_rt_active = false;
                reused = false;
            }
        }
    }

    const int64_t diag_prepared = ggml_time_us();
    if (reused) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // Token-width binds replan before reaching this point. Legacy binds
        // update metadata within their recorded allocation envelopes.

        if (reused) {
            // with pipeline parallelism, the previous graph_compute_async may still be running
            // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
            // that the previous compute is still reading.
            if (cparams.pipeline_parallel) {
                ggml_backend_sched_synchronize(run_sched);
            }

            n_reused++;
            if (llama_prefab_stats_enabled()) {
                prefab_stats.n_stock_reuses++;
            }
        }
    }
    if (!reused) {
        if (shape_slot >= 0 && shape_hit) {
            // can_reuse() rejected a key hit: the entry now holds this shape.
            shape_plan_cache.rekey(shape_slot, shape_key, shape_plan_clock);
        }
        const int64_t diag_build_start = ggml_time_us();
        // the record points into the graph that res->reset() is about to
        // destroy; drop it first so a build_graph() failure below cannot
        // leave dangling tensor pointers behind (the next bind would trust
        // record_valid and patch freed tensors).
        union_record_invalidate();
        // drain any in-flight compute before the rebuild reuses its buffers
        ggml_backend_sched_synchronize(run_sched);
        res->reset();

        ggml_backend_sched_reset(run_sched);
        ggml_backend_sched_set_eval_callback(run_sched, cparams.cb_eval, cparams.cb_eval_user_data);
        // Build at physical KV and recurrent capacities and the output
        // high-water mark. Live widths are bound after allocation.
        bool     env          = false;
        bool     record_build = false;
        uint32_t cap          = 0;
        uint32_t cap_nrs      = 0;
        const llama_kv_cache_context * kv_ctx = nullptr;
        const llama_nkv_ctx_state * st_peek = union_binder_state_peek();
        const bool shape_arch_ok = union_arch_supported(model.arch) &&
            !model.hparams.use_alibi && model.hparams.n_expert == 0 &&
            model.hparams.n_rel_attn_bkts == 0 && sampling.samplers.empty();
        // The record build needs the exact lane: the padded control lane
        // (GGML_TP5_UNION_BINDER_EXACT=0) is the stock path, so it builds and
        // reuses stock graphs and records nothing. Recording there would make
        // the binder patch a graph the control arm is supposed to leave alone.
        if (!graph_reuse_disable && st_peek != nullptr && st_peek->nkv_enabled &&
                st_peek->nkv_exact_enabled &&
                !st_peek->env_disabled && shape_arch_ok &&
                (gtype == LLM_GRAPH_TYPE_DEFAULT || gtype == LLM_GRAPH_TYPE_DECODER) &&
                !cparams.pipeline_parallel && backend_tp5_relay == nullptr &&
                (union_recr_ctx(mctx) == nullptr || union_sched_cpu_only()) &&
                dynamic_cast<const llama_memory_hybrid_idx_context *>(mctx) == nullptr &&
                !union_is_hybrid_iswa(mctx) && !union_has_attention_views(mctx)) {
            record_build = true;
            if (const auto * h = dynamic_cast<const llama_memory_hybrid_context *>(mctx)) {
                kv_ctx = h->get_attn();
            } else if (const auto * k = dynamic_cast<const llama_kv_cache_context *>(mctx)) {
                kv_ctx = k;
            }
            if (kv_ctx != nullptr) {
                const uint32_t nkv_size = cparams.n_ctx_kv != 0 ? cparams.n_ctx_kv : cparams.n_ctx_seq;
                // full legal range: the physical per-stream capacity, never a
                // bucket or headroom
                cap = std::min(nkv_size, kv_ctx->get_capacity());
                env = cap > 0;
            }
            if (const auto * recr = union_recr_ctx(mctx)) {
                cap_nrs = recr->get_size();
            }
        }
        // The token lane builds the graph at the live KV/token widths
        // (strict line) and never reserves KV or token headroom. The legacy
        // LLAMA_EXPERIMENTAL_VARLEN_NTOK envelope of the base line widened the
        // graph to n_ubatch and is retired: it is exactly the headroom the
        // strict gate forbids. token_candidate already carries the
        // single-stream, simple token-batch, CPU-only and non-flash
        // restrictions, so it replaces the old token_envelope predicate.
        const uint32_t live_nkv = token_candidate && kv_ctx ? kv_ctx->get_n_kv_exact() : 0;
        const bool token_runtime = token_candidate && env && record_build && live_nkv != 0 && live_nkv <= cap;
        if (token_candidate && !token_runtime) ++llama_nkv_ctx_state_get(this).c_ntok_fallbacks;
        auto build_params = gparams;
        if (token_runtime) {
            build_params.n_outputs_build = 0;
            build_params.strict_ntok = true;
        }


        {
            // instrumented-build hygiene: a throwing build must not pollute the
            // node counter with the PREVIOUS graph (gf still holds it here)
            gf = nullptr;
            llama_prefab_build_timer timer{prefab_stats, gf};
            // RAII: an exception from build_graph must not leave the build
            // overrides armed on the KV / recurrent contexts
            union_nkv_build_guard env_guard(env ? const_cast<llama_kv_cache_context *>(kv_ctx) : nullptr,
                                            env ? (token_runtime ? live_nkv : cap) : 0);
            union_nrs_build_guard nrs_guard(
                cap_nrs != 0 ? const_cast<llama_memory_recurrent_context *>(union_recr_ctx(mctx)) : nullptr,
                cap_nrs);
            if (st_peek != nullptr) {
                ++llama_nkv_ctx_state_get(this).c_graph_builds;
            }
            ++n_graph_builds;
            gf = model.build_graph(build_params);
        }
        if (llama_prefab_stats_enabled()) {
            prefab_stats.n_stock_builds++;
        }

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        diag_build = ggml_time_us() - diag_build_start;

        if (record_build) {
            // The record snapshot takes the single graph cache key, and the
            // key's lane exclusion is read from the shape map: a graph whose
            // token shape map is valid has token inputs whose ne[0] the bind
            // patches, so those dimensions must stay out of the key (see
            // union_topology_key_excl()). The snapshot has to run first
            // because the analysis below reads the record's tensor lists, so
            // the key is taken again right after the analysis, with the lane
            // now known. Nothing reads the key in between: the record is not
            // used for a bind until this process_ubatch call returns.
            nkv_record_snapshot(gf);
            auto & st = llama_nkv_ctx_state_get(this);
            st.record_valid = true;
            // The width the graph was built at, read from the applied-width
            // accessor: get_n_kv() reports the true used span when the exact
            // lane is armed, and a build-time width is not a runtime target.
            st.record_cap   = env ? cap : (kv_ctx != nullptr ? kv_ctx->get_n_kv_stock() : 0);
            st.record_bound = token_runtime ? live_nkv : st.record_cap;
            // The arena starts at the width the graph was built at: the
            // allocation envelope on the legacy lanes, the exact live width
            // on the token lane. A successful replan raises it; nothing else
            // may, because nothing else reallocates the arena in place.
            st.record_arena_n_kv = st.record_bound;
            st.record_cap_nout      = res->outputs_envelope();
            st.record_has_out_ids   = res->has_out_ids();
            st.record_bound_nout    = st.record_cap_nout;
            st.record_cap_nrs       = cap_nrs;
            st.record_bound_n_rs    = cap_nrs;
            st.record_bound_rs_z    = 0;
            st.record_bound_rs_head = 0;
            st.record_bound_ntok    = build_params.ubatch.n_tokens;
            st.record_bound_replay_len = union_replay_len_now(mctx);
            if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: record build env=%d cap=%u record_cap=%u record_bound=%u "
                            "cap_nout=%u has_out_ids=%d ntok_build=%u live_nkv=%u token_runtime=%d\n",
                            "union_rebind_apply", env ? 1 : 0, cap, st.record_cap, st.record_bound,
                            st.record_cap_nout, st.record_has_out_ids ? 1 : 0,
                            build_params.ubatch.n_tokens, live_nkv, token_runtime ? 1 : 0);

            // Pillar 1: the node list and the width table are captured AFTER
            // ggml_backend_sched_alloc_graph(), in the block next to the
            // topology key. alloc_graph() runs ggml_backend_sched_split_graph(),
            // which rewrites the graph nodes' src[] to point at scheduler-
            // inserted copy tensors. The width-table hash is a CLOSED hash --
            // every edge target must be a member of the hashed node list -- so
            // a list captured before allocation cannot be hashed afterwards
            // (the copies are not in it) and every bind refused with
            // refused_skeleton. The build widths are unchanged by allocation
            // (the patch runs later), so the table is the same table.
            ++st.c_env_builds;
            ++st.c_shape_analyses;

            // rs_z runtime parameter: record what the built graph carries.
            // The named refusal for "asked but not armed" is counted after
            // the shape analysis below.
            const llm_graph_input_rs * rs_inp = nullptr;
            for (const auto & input : res->inputs) {
                if (const auto * rs = dynamic_cast<const llm_graph_input_rs *>(input.get())) {
                    rs_inp = rs;
                    break;
                }
            }
            st.rsz_rt_active     = rs_inp != nullptr && rs_inp->s_zero != nullptr;
            // rsz_rt_bound_rs_z is NOT the source of the diagnostic: the live
            // value is read from record_bound_rs_z when the record is valid
            // (see union_diagnostics()). It is kept as the fallback for the
            // one window where the record has been invalidated (a stock
            // rebuild) but the lane's last binding is still the honest answer.
            st.rsz_rt_bound_rs_z = st.record_bound_rs_z;
            // The zero-row op clears ONE row of the state tensor, and that row
            // is a runtime value bound in set_input(). What the decode-time
            // check must prove is therefore a property of the state tensor and
            // the cell capacity, not of a view:
            //   rs_z < mem_size        (rs_z is a CELL index, never a row)
            //   mem_size divides rows  (the tensor is (1 + n_rs_seq) groups)
            //   bound row < rows       (the row the op names exists)
            // rsz_rt_state_rows is the state tensor's row count and
            // rsz_rt_zero_view_rows its per-group height (mem->size); the
            // offsets/strides A recorded for its view no longer exist because
            // there is no view. See llama_rsz_rt_check_relay_row().
            st.rsz_rt_zero_view_rows = 0;
            st.rsz_rt_state_rows = 0;
            for (const ggml_tensor * x : st.record_t) {
                if (strcmp(x->name, "union_rs_zero_row") == 0) {
                    // the op is an in-place view of the state tensor
                    st.rsz_rt_state_rows       = (uint32_t) x->ne[1];
                    const auto * recr_rec = union_recr_ctx(mctx);
                    st.rsz_rt_zero_view_rows   = recr_rec != nullptr ? recr_rec->get_size() : 0;
                    break;
                }
            }

            const bool shape_ok = union_shape_analyze(build_params, st.record_t, st.record_src,
                    st.record_fixed, st.record_root, st.record_root_offs,
                    token_runtime ? live_nkv : st.record_cap, st.record_cap_nout,
                    0 /* build head */, cap_nrs, token_runtime, st.shape_map);
            // The record's topology key is NOT taken here. It is taken after
            // ggml_backend_sched_alloc_graph() below, because the key folds
            // is_fixed() (llama-topology-hash.h) and allocation flips that flag
            // for every graph-owned tensor: before alloc nothing has a buffer,
            // after alloc everything does. Every later comparison -- the bind
            // gate in union_rebind_apply() and the refresh after each applied
            // bind -- runs on the allocated graph, so a key taken before
            // allocation can never match and would refuse the first bind of
            // every graph. That was the pre-existing refusal: the topology gate
            // failed, c_refused_topology moved (a counter the stats line did not
            // print), the caller rebuilt stock, env_disabled was set, and the
            // mask PASS arm was never observed.

            // rs_z runtime parameter: the record build either carried the
            // zero-row index (rsz_rt_active) or it did not. A lane that asked
            // for the row and did not get it is a named refusal; the graph
            // then runs stock, where a moved rs_z is a rebuild. This runs
            // AFTER the shape analysis because _no_rs_z is the analysis'
            // verdict.
            //
            // The cases graph_params() already counted (gate, no recurrent
            // context, rs_z < 0) are not counted again here: one refusal, one
            // counter. What is left is what only the built graph can show.
            if (st.rsz_rt_enabled && !st.rsz_rt_active) {
                if (rs_inp == nullptr) {
                    st.rsz_rt_refuse(st.c_rsz_rt_refused_arch);
                } else if (rs_inp->s_zero == nullptr) {
                    st.rsz_rt_refuse(st.c_rsz_rt_refused_no_rs_z);
                } else {
                    st.rsz_rt_refuse(st.c_rsz_rt_refused_shape);
                }
            }
            if (!shape_ok) {
                if (token_runtime) ++st.c_ntok_fallbacks;
                // outside the provable support range: disable the envelope and
                // rebuild at the stock width. This is a real graph build.
                ++st.c_shape_fail;
                if (st.shape_map.fused_qkv_refused) {
                    // The one refusal this pass can name: the fused-QKV
                    // projection's Q/K/V views are non-contiguous row slices
                    // of one matmul, so no symbolic map can bind their token
                    // axis. The lane still disables below; this names the cause.
                    ++st.c_fused_qkv_refused;
                    LLAMA_LOG_INFO("%s: token lane refused: fused QKV projection "
                                   "(blk.*.attn_qkv.weight) is a non-contiguous slice "
                                   "of one fused matmul\n", __func__);
                }
                ++st.c_fallbacks;
                ++st.c_rebuild_fallbacks;
                st.env_disabled = true;
                st.nout_suppress_envelope = true;
                union_record_invalidate();
                record_build = false;

                // drain the envelope build's (not yet computed) graph and any
                // previous in-flight compute before the stock rebuild
                ggml_backend_sched_synchronize(run_sched);
                res->reset();
                ggml_backend_sched_reset(run_sched);
                ggml_backend_sched_set_eval_callback(run_sched, cparams.cb_eval, cparams.cb_eval_user_data);
                gparams = graph_params(res, ubatch, mctx, gtype);
                gparams.sched = run_sched;
                ++st.c_graph_builds;
                ++n_graph_builds;
                gf = model.build_graph(gparams);
                if (!gf) {
                    LLAMA_LOG_ERROR("%s: failed to rebuild graph at the stock width\n", __func__);
                    ret = GGML_STATUS_FAILED;
                    return nullptr;
                }
            }
            if (record_build) {
                // the analysis accepted this graph: the envelope is usable.
                // record_cap_nout is max(active, previous high-water), so it
                // can only stay or rise. A rise after a valid record is the
                // single envelope-expanding rebuild; widths up to the new
                // mark bind in place.
                if (st.record_cap_nout > st.nout_high_water) {
                    st.nout_high_water = st.record_cap_nout;
                }
                if (prev_record_valid && st.record_cap_nout > prev_cap_nout) {
                    ++st.c_nout_grow_rebuilds;
                }
            }
        } else if (union_binder_state_peek() != nullptr) {
            // gate is on but this build is not a record build: drop the record
            // (tensor vectors included) instead of leaving dangling pointers
            union_record_invalidate();
            // a stock build carries no selection row; the rs_z lane is off
            // until the next record build proves it again
            auto & st = llama_nkv_ctx_state_get(this);
            st.rsz_rt_active = false;
        }
        // Only a recorded envelope can use the copy metadata. Keep stock
        // graphs and scheduler reserve paths free of tracking allocations.
        if (!ggml_backend_sched_set_split_copy_tracking(run_sched, record_build)) {
            LLAMA_LOG_ERROR("%s: failed to configure split copy tracking\n", __func__);
            union_record_invalidate();
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        const int64_t diag_alloc_start = ggml_time_us();
        if (!ggml_backend_sched_alloc_graph(run_sched, gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            union_record_invalidate();
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
        diag_alloc = ggml_time_us() - diag_alloc_start;

        // The record's node list, width table and topology key, all taken on
        // the ALLOCATED graph -- the same state every later comparison sees.
        // The address record stays the tensor-identity authority; the key is
        // the topology half of the same question, and the width table is the
        // runtime-dimension half.
        if (record_build) {
            auto & st = llama_nkv_ctx_state_get(this);
            // The CANONICAL node set, not the raw graph node list. The width
            // table exists to exclude the runtime dimensions of the tensors
            // the graph reaches through src[]/view_src -- the KQ mask, the
            // out_ids closure, the nkv runtime input, the K/V views -- and
            // those tensors are NOT graph nodes: the raw list has 222 entries
            // while the canonical set has 236. llama_union_topology_hash::
            // hash_nodes() is a closed hash (every edge target must be a
            // member), so hashing the raw list could never succeed: the
            // excluded tensors were not even members. This is the same set
            // llama_topology_key_of() hashes, so the two hashes describe the
            // same graph.
            //
            // The const_cast is the type boundary only: the canonical list is
            // const because the key never writes through it, while the record
            // stores mutable pointers for the patch. The tensors are owned by
            // the graph result and outlive the record; union_record_invalidate()
            // drops the record before the graph is reset.
            std::vector<const ggml_tensor *> canon;
            llama_topology_detail::topology_index canon_index;
            std::vector<size_t> canon_position;
            llama_topology_canonical_nodes(gf, canon, canon_index, canon_position);
            st.record_nodes.clear();
            st.record_nodes.reserve(canon.size());
            for (const ggml_tensor * x : canon) {
                st.record_nodes.push_back(const_cast<ggml_tensor *>(x));
            }
            const uint32_t width = token_runtime ? live_nkv : st.record_cap;
            const uint32_t build_ntok = token_runtime ? build_params.ubatch.n_tokens : 0;
            // The table's exclusion set is the shape map's coefficient table:
            // the same tensors and axes union_rebind_apply() patches. The map
            // indexes the RECORD list, so the resolution is by tensor identity
            // into the canonical list (see union_width_runtime_dims()).
            const std::vector<llama_union_topology_excl> runtime_dims =
                    union_width_runtime_dims(st.shape_map, st.record_t, st.record_nodes);
            const bool table_ok = st.width_table.build(st.record_nodes, runtime_dims, width,
                    st.record_cap_nout, cap_nrs, 0 /* build head */, build_ntok,
                    model.hparams.n_pos_per_embd(), model.hparams.n_embd_v_gqa_max());
            uint64_t hash = 0;
            st.topology_hashed = table_ok && st.width_table.topology_hash(st.record_nodes, hash);
            st.topology_hash   = st.topology_hashed ? hash : 0;
            if (!table_ok) {
                st.width_table.clear();
            }
            st.record_topology_key = llama_topology_key_of(gf, union_topology_key_excl(st.shape_map, st.record_t));
            st.live_topology_key   = st.record_topology_key;
            st.live_topology_gf    = gf;
            if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: record post-alloc nodes=%zu rt_dims=%zu table_ok=%d hashed=%d hash=%016llx\n",
                    "union_rebind_apply", st.record_nodes.size(), runtime_dims.size(),
                    table_ok ? 1 : 0, st.topology_hashed ? 1 : 0,
                    (unsigned long long) st.topology_hash);
        }

        // the fresh graph is still at the build widths. Patch it to the runtime
        // values BEFORE the first compute; a refusal on a graph the analysis
        // accepted disables the envelope and rebuilds stock, so a deterministic
        // bind failure cannot turn into a per-ubatch rebuild loop.
        //
        // The delta predicate decides whether a bind has anything to do. The
        // record snapshot set record_bound* to the BUILD shapes (:5606-5614),
        // so it is meaningful before the bind. A no-op bind would change
        // nothing, and walking the kinds only to have all four decline would
        // count four lane_declines for nothing: on the token lane the graph is
        // already built at the active widths, so a step whose targets all
        // match the record needs no bind at all.
        //
        // The rs_z zero row is a runtime INDEX input, not a shape: it is
        // bound on every set_inputs() (union_rsz_bind_index) and no tensor
        // layout depends on it. It is therefore not part of the predicate; a
        // row-only move must not force a rebuild.
        auto delta_pending = [&]() {
            auto & st = llama_nkv_ctx_state_get(this);
            const bool nout_changed = st.record_has_out_ids && n_outputs != st.record_bound_nout;
            return union_nkv_now(mctx) != st.record_bound || nout_changed ||
                   union_n_rs_now(mctx) != st.record_bound_n_rs ||
                   union_rs_head_now(mctx) != st.record_bound_rs_head ||
                   ubatch.n_tokens != st.record_bound_ntok;
        };
        bool fresh_bound = false;
        {
            auto & st_dbg = llama_nkv_ctx_state_get(this);
            if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: post-build bind decision record_build=%d pending=%d "
                            "nkv_now=%u record_bound=%u nout=%u bound_nout=%u ntok=%u bound_ntok=%u "
                            "record_cap=%u\n",
                            "union_rebind_apply", record_build ? 1 : 0, delta_pending() ? 1 : 0,
                            union_nkv_now(mctx), st_dbg.record_bound, n_outputs,
                            st_dbg.record_bound_nout, ubatch.n_tokens, st_dbg.record_bound_ntok,
                            st_dbg.record_cap);
        }
        if (record_build && delta_pending()) {
            auto & st = llama_nkv_ctx_state_get(this);
            fresh_bound = st.union_binder.rebind_any(this, res, gparams);
            if (union_bind_trace_on()) LLAMA_LOG_ERROR("%s: post-build bind result fresh_bound=%d\n",
                            "union_rebind_apply", fresh_bound ? 1 : 0);
        }
        const uint32_t out_bound = res->outputs_bound();
        if (record_build && !fresh_bound) {
            auto & st = llama_nkv_ctx_state_get(this);
            // The same predicate the bind decision used: with a no-op bind
            // skipped above, false here means the record already matches the
            // live targets.
            const bool changed = delta_pending();
            bool replan_failed = false;
            if (!changed && st.record_valid && st.shape_map.ntok_valid) {
                // transactional replan at the live token width; a refusal
                // leaves the old arena and record intact and falls back below
                replan_failed = !ggml_backend_sched_replan_graph(run_sched, st.record_t.data(),
                        st.record_fixed.data(), st.record_t.size(), nullptr, nullptr, 0);
                // The replan reallocated the arena at the recorded bound
                // shapes, so the arena width is that bound. A failed replan
                // leaves the old arena and must leave this value alone.
                if (!replan_failed && st.record_bound != 0) {
                    st.record_arena_n_kv = st.record_bound;
                }
                // No reduction-order fixup, for the same reason as
                // union_rebind_apply: the in-place patch is the whole
                // mechanism, and the ordered attention kernels make the tree
                // depend only on the surviving terms.
            }
            if (changed || out_bound > n_outputs || !st.record_valid || replan_failed) {
                // The fresh record is dropped here, so its key must go with it:
                // llama_topology_key_of() never returns 0 for a real graph, so
                // a stale key would let the next bind's topology gate compare
                // a live graph against a record that no longer exists.
                // union_record_invalidate() below does this; the assignment
                // here states it at the site that decides the record is dead.
                // The width-table hash goes with it for the same reason: a
                // hash left over from a record that no longer exists would
                // report a topology nobody can vouch for.
                st.record_topology_key = llama_topology_key{};
                st.topology_hash       = 0;
                st.topology_hashed     = false;
                ++st.c_rebuild_fallbacks;
                st.env_disabled = true;
                st.nout_suppress_envelope = true;
                union_record_invalidate();
                ggml_backend_sched_synchronize(run_sched);
                res->reset();
                ggml_backend_sched_reset(run_sched);
                if (!ggml_backend_sched_set_split_copy_tracking(run_sched, false)) {
                    LLAMA_LOG_ERROR("%s: failed to disable split copy tracking\n", __func__);
                    ret = GGML_STATUS_FAILED;
                    return nullptr;
                }
                ggml_backend_sched_set_eval_callback(run_sched, cparams.cb_eval, cparams.cb_eval_user_data);
                gparams = graph_params(res, ubatch, mctx, gtype);
                gparams.sched = run_sched;
                ++st.c_graph_builds;
                ++n_graph_builds;
                gf = model.build_graph(gparams);
                if (!gf) {
                    LLAMA_LOG_ERROR("%s: failed to rebuild graph at the stock width\n", __func__);
                    ret = GGML_STATUS_FAILED;
                    return nullptr;
                }
                if (!ggml_backend_sched_alloc_graph(run_sched, gf)) {
                    LLAMA_LOG_ERROR("%s: failed to allocate rebuilt graph\n", __func__);
                    union_record_invalidate();
                    ret = GGML_STATUS_ALLOC_FAILED;
                    return nullptr;
                }
            }
        }
    }

    // Validate the actual scheduler placement before submitting any compute.
    // The reference may use CPU shape/index operations, never CPU quantized
    // projections: CPU's native activation policy is not this F32 policy.
    if (backend_quantized_matmul_f32 != nullptr) {
        if (backend_tp5_relay != nullptr && ggml_backend_sched_get_n_splits(run_sched) != 1) {
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
                 ggml_backend_sched_get_tensor_backend(run_sched, tensor) != backend_quantized_matmul_f32)) {
                LLAMA_LOG_ERROR("%s: quantized_matmul_f32 refuses unsupported or non-GPU quantized projection %s\n", __func__, tensor->name);
                ret = GGML_STATUS_FAILED;
                return nullptr;
            }
            if (backend_tp5_relay != nullptr &&
                (tensor->flags & GGML_TENSOR_FLAG_COMPUTE) != 0 && tensor->op != GGML_OP_NONE &&
                tensor->op != GGML_OP_VIEW && tensor->op != GGML_OP_RESHAPE &&
                tensor->op != GGML_OP_PERMUTE && tensor->op != GGML_OP_TRANSPOSE &&
                ggml_backend_sched_get_tensor_backend(run_sched, tensor) != backend_tp5_relay) {
                LLAMA_LOG_ERROR("%s: tp5-relay refuses non-Meta compute node %s\n", __func__, tensor->name);
                ret = GGML_STATUS_FAILED;
                return nullptr;
            }
        }
    }
    const int64_t diag_input_start = ggml_time_us();
    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();


        // poka-yoke: the output-gather closure must execute at the active
        // width before compute. The binder patches it on both the reuse and
        // fresh-build paths; a graph still at a different width would make the
        // gather read stale or uninitialized tail indices.
        const uint32_t out_bound = res->outputs_bound();
        if (res->has_out_ids() && out_bound != n_outputs) {
            LLAMA_LOG_ERROR("%s: output gather bound at %u, active n_outputs %u\n",
                            __func__, out_bound, n_outputs);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        // Pillar 1: the width the attention consumers must execute at is a
        // VALUE, re-armed from the KV cache before every compute (reused and
        // fresh-build paths alike). It is written into the graph's runtime
        // n_kv input by llm_graph_input_nkv_runtime::set_input below.
        if (const llama_nkv_ctx_state * wst = union_binder_state_peek()) {
            if (wst->nkv_enabled && wst->width_table.build_n_kv != 0) {
                const llama_kv_cache_context * wkv = union_kv_ctx(mctx);
                const uint32_t width = wkv != nullptr ? wkv->get_n_kv_runtime() : 0;
                if (width != 0) {
                    // Same bound as the bind's arm: the width the live arena
                    // holds. A width above it has no backing and stays
                    // unarmed (runtime_n_kv = 0), which is the fail-closed
                    // direction; the bind's own envelope precheck is what
                    // refuses such a step before it reaches this point.
                    if (!llama_nkv_ctx_state_get(this).width_table.set_runtime_n_kv(
                                width, wst->record_arena_n_kv)) {
                        // The graph is about to execute at a width its arena
                        // does not hold. Refuse the decode instead of
                        // computing on a mask that cannot cover the width.
                        LLAMA_LOG_ERROR("%s: runtime n_kv %u exceeds the live arena %u\n",
                                        __func__, width, wst->record_arena_n_kv);
                        ret = GGML_STATUS_FAILED;
                        return nullptr;
                    }
                }
            }
        }

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        // The batch the inputs were just filled from, for the mask read-back's
        // test surface (llama_union_mask_row_check). One copy on a path that
        // already copies the batch into the graph parameters.
        //
        // The copy is safe because a ubatch that reaches here owns its data:
        // llama_batch_allocr builds it with a shared_ptr<data_t> and the
        // ubatch's raw pointers point into that store (llama-batch.cpp:777,
        // :846-863), so copying the ubatch copies the shared_ptr and the
        // pointees outlive the per-decode memory context. If that invariant
        // ever changes, the read-back must refuse rather than guess: it
        // already returns UNPROVEN for an empty or unusable batch.
        //
        // Nothing per-decode is stored: the memory context is owned by the
        // unique_ptr in decode() and is freed when decode returns, so the
        // read-back resolves the long-lived cache from `memory` at call time.
        ubatch_inputs_last = ubatch;

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    // Pillar 5 plan identity. The relay provider turns this into its K=8 plan
    // slot; channels is left zero because only Meta knows the stage count.
    // rs_class packs the three recurrent-state switches into one coarse class.
    if (backend_tp5_relay != nullptr) {
        ggml_relay_plan_key plan_key = {};
        plan_key.model       = uint64_t(reinterpret_cast<uintptr_t>(&model));
        plan_key.window      = 0; // the provider is private to one service window
        plan_key.n_tokens    = ubatch.n_tokens;
        plan_key.n_outputs   = n_outputs;
        plan_key.rs_class    = int32_t((uint32_t(cparams.nextn_layer_offset) << 8) |
                                       (uint32_t(gparams.rs_row_armed ? 1 : 0) << 4) |
                                       uint32_t(gtype));
        const llama_kv_cache_context * plan_kv = union_kv_ctx(mctx);
        plan_key.n_kv_bucket = plan_kv != nullptr ? plan_kv->get_n_kv_pad() : 0;
        plan_key.alloc_gen   = alloc_gen_;
        plan_key.channels    = 0;
        ggml_backend_meta_set_plan_key(backend_tp5_relay, &plan_key);
    }

    const int64_t diag_compute_start = ggml_time_us();
    gf_res_current   = res;
    shape_last_sched = shape_plan_active ? run_sched : nullptr;
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1,
                                      shape_plan_active ? run_sched : nullptr);
    if (backend_tp5_relay && std::getenv("TP5_DIAG_GRAPH")) {
        LLAMA_LOG_INFO("TP5_DIAG_GRAPH ctx=%p N=%u pos=%d replay=%u reused=%d plan_hit=%d slot=%d pre_us=%lld build_us=%lld alloc_us=%lld inputs_us=%lld compute_us=%lld total_us=%lld\n",
            (void *) this, ubatch.n_tokens, ubatch.pos ? ubatch.pos[0] : -1, diag_replay, int(reused),
            int(shape_hit), shape_slot,
            (long long)(diag_prepared-diag_start),(long long)diag_build,(long long)diag_alloc,
            (long long)(diag_compute_start-diag_input_start),(long long)(ggml_time_us()-diag_compute_start),
            (long long)(ggml_time_us()-diag_start));
    }
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;

    // c_pad_fallbacks is folded nowhere: the pad bucket and its forced-fallback
    // hook were deleted with the padded lane, and the exact lane now treats a
    // layout refusal as a hard failure in
    // llama_kv_cache_context::get_n_kv(). The counter and its exported field
    // stay (include/llama.h) so a reader can see it is 0; nothing increments
    // it, so a step cannot be attributed a fallback that did not happen.

    return res;
}

int llama_context::encode(const llama_batch & batch_inp) {
    last_llama_activation_valid = false;
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
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched_active(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        // chain mode: logits are [2, n_chain] (token_id, prob) pairs, not [n_vocab, n_tokens]
        const size_t logits_bytes =
            cparams.mtp_chain ? ggml_nbytes(t_logits) : (size_t) n_tokens * n_vocab * sizeof(float);
        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, logits_bytes);
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched_active(), t_embd);
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
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched_active(), t_h_nextn);
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
    last_llama_activation_valid = false;
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

    // handle any pending shifts/copies. false with memory_update_failed set
    // is a real failure; false without it is the normal NO_UPDATE no-op.
    memory_update(false);
    if (memory_update_failed) {
        return memory_update_error;
    }

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
                        if (memory_update_failed) {
                            return memory_update_error;
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
    uint32_t n_ubatches_done = 0;

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
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched_active(), t_logits);
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
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched_active(), t_embd);
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
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched_active(), t_h_nextn);
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
            copy_tensor_async_ints(res->t_sampled, sampling.sampled, seq_to_output_row, sched_active());

            copy_tensor_async_floats(res->t_sampled_logits, sampling.logits, stride, sampling.logits_count,
                                     seq_to_output_row, sched_active());
            copy_tensor_async_floats(res->t_sampled_probs, sampling.probs, stride, sampling.probs_count,
                                     seq_to_output_row, sched_active());
            copy_tensor_async_candidates(res->t_candidates, sampling.candidates, stride, sampling.candidates_count,
                                         seq_to_output_row, sched_active());
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev += ubatch.n_tokens;
        ++n_ubatches_done;
    } while (mctx->next());

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;
    bool activation_rows_ordered = true;

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
                activation_rows_ordered = false;
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
    // The final normalization tensor is already a graph output. Only expose
    // it when one microbatch produced every row in the original order.
    last_llama_activation_valid = model.arch == LLM_ARCH_LLAMA && activation_rows_ordered &&
        n_ubatches_done == 1 && n_outputs_all == n_tokens_all && n_outputs_all > 0;

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs, bool preallocate) {
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

            // ROCm: hipHostFree on the device host buffer waits on HSA
            // signals. A concurrent kernel launch on the same device can hold
            // the ROCm runtime mutex, so the free and the launch deadlock
            // (H2e: ggml_cuda_op_scale -> hipLaunchKernel vs output_reserve
            // -> hipHostFree -> BusyWaitSignal::WaitRelaxed). Retire the
            // buffer instead. ~llama_context synchronizes and then destroys
            // the retired buffers after the owner stopped submitting. In
            // relay mode the constructor reserved the full output envelope,
            // so this path is reached only on a configuration change, never
            // on the first decode.
            buf_output_retired.emplace_back(std::move(buf_output));
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
    // The decode contract is n_outputs <= n_outputs_max. The constructor's
    // explicit envelope reserve is the one caller allowed above it: the
    // buffer is larger, the views are still sized by the requested rows, and
    // later decode calls re-derive them below the allocation.
    if (!preallocate) {
        GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);
    }
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

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched_active(), t);
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
                                           size_t *                       sizes,
                                           const size_t *                 min_sizes,
                                           bool                           preserve_buffers) {
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
    union_record_invalidate();
    if (!ggml_backend_sched_set_split_copy_tracking(sched.get(), false)) {
        LLAMA_LOG_ERROR("%s: failed to disable split copy tracking for reserve\n", __func__);
        return nullptr;
    }

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

    ++n_graph_builds;
    auto * gf = model.build_graph(gparams);

    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            if (preserve_buffers) {
                ggml_backend_sched_reserve_size_preserve(sched.get(), gf, sizes);
            } else {
                ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
            }
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (min_sizes ? !ggml_backend_sched_reserve_min(sched.get(), gf, min_sizes)
                         : !ggml_backend_sched_reserve(sched.get(), gf)) {
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    tp5_assign_graph_uid(backend_tp5_relay, gf);
    return gf;
}

llm_graph_params llama_context::graph_params(llm_graph_result *             res,
                                             const llama_ubatch &           ubatch,
                                             const llama_memory_context_i * mctx,
                                             llm_graph_type                 gtype) const {
    // n_outputs is split into the ACTIVE count (this ubatch, a runtime key)
    // and the BUILD envelope (allocation width). The envelope is only used on
    // a single CPU-backend graph, where the binder can patch the output-gather
    // closure in place; everywhere else the graph builds at the active count.
    // Keep output allocation at the observed high-water mark. Recurrent
    // zero-row execution additionally requires the single CPU backend.
    const llama_nkv_ctx_state * nout_st = union_binder_state_peek();
    const bool union_active =
        nout_st != nullptr && nout_st->nkv_enabled && !nout_st->env_disabled &&
        !graph_reuse_disable && sampling.samplers.empty() &&
        union_arch_supported(model.arch) && !model.hparams.use_alibi &&
        model.hparams.n_expert == 0 && model.hparams.n_rel_attn_bkts == 0 &&
        (gtype == LLM_GRAPH_TYPE_DEFAULT || gtype == LLM_GRAPH_TYPE_DECODER) &&
        (union_recr_ctx(mctx) == nullptr ? (union_sched_cpu_only() || union_sched_single_split())
                                         : union_sched_cpu_only()) &&
        !cparams.pipeline_parallel && backend_tp5_relay == nullptr &&
        dynamic_cast<const llama_memory_hybrid_idx_context *>(mctx) == nullptr &&
        !union_is_hybrid_iswa(mctx) && !union_has_attention_views(mctx);
    const bool nout_envelope = union_active && !nout_st->nout_suppress_envelope;
    const uint32_t nout_build = nout_envelope
        ? std::max<uint32_t>(n_outputs, nout_st->nout_high_water)
        : 0;
    if (nout_st != nullptr) {
        nout_st->nout_suppress_envelope = false;
    }

    // rs_z runtime parameter: the lane is armed per build. A context that
    // never passed the arm checks (gate, recurrent context, a free zero cell,
    // the n_rs_seq row-set condition) builds the stock graph, and the refusal
    // is counted HERE, once per build, by name. The record-build block below
    // then only counts the cases this predicate cannot see (no rs input at
    // all, a build that carried no zero-row index, a malformed index tensor).
    //
    // A refusal here is not a "fallback": nothing was bound and nothing was
    // rebuilt, the build simply never carried the lane. The lane total
    // (c_rsz_rt_fallbacks) counts refusals of the runtime path, so these
    // count too; see llama-nout-state.h.
    //
    // These counters also move on the scheduler's reserve builds, which call
    // graph_params() too. That is intended: the counters answer "why is the
    // lane not armed", not "how many decodes fell back", and they stay 0 on
    // any context where the lane arms.
    const llama_memory_recurrent_context * rsz_recr = union_recr_ctx(mctx);
    const uint32_t rsz_n_rs_seq = rsz_recr != nullptr ? rsz_recr->get_n_rs_seq() : 0;
    // The relay path is a second arming form, not a widening of the first:
    // its gate is the relay lane itself (a single scheduler split, so one
    // topology on every rank and no relay channel bound to a graph input),
    // and it carries the ring (n_rs_seq > 0) by clearing the single row the
    // gathers read. A relay context never passes the stock gate (it refuses
    // backend_tp5_relay), so exactly one form applies.
    const bool rsz_relay_ok = llama_rsz_rt_relay_gate(
        backend_tp5_relay != nullptr,
        nout_st != nullptr && nout_st->rsz_rt_enabled,
        !graph_reuse_disable,
        cparams.pipeline_parallel,
        union_sched_single_split());
    const bool rsz_gate_ok = nout_st != nullptr && nout_st->nkv_enabled && nout_st->rsz_rt_enabled &&
        !graph_reuse_disable && !cparams.pipeline_parallel && backend_tp5_relay == nullptr &&
        !union_is_hybrid_iswa(mctx) && !union_has_attention_views(mctx) &&
        dynamic_cast<const llama_memory_hybrid_idx_context *>(mctx) == nullptr;
    if (nout_st != nullptr && nout_st->rsz_rt_enabled && !rsz_gate_ok && !rsz_relay_ok) {
        nout_st->rsz_rt_refuse(nout_st->c_rsz_rt_refused_gate);
    }
    // n_rs_seq > 0 widens the state tensors to mem_size*(1+n_rs_seq) rows and
    // makes s_copy() return idx*size + src0, i.e. a row index is no longer a
    // cell index. The stock lane bakes rs_z into a view offset, so it is
    // refused by name rather than silently allowed; see build_rs().
    // The relay lane carries the ring: there the zero-row op clears the row
    // rs_idx[seq]*mem_size + rs_z, which is the row the gathers read
    // (llama_rsz_rt_check_relay_row()).
    // The arming rules themselves are llama_rsz_rt_relay_gate(),
    // llama_rsz_rt_relay_armable() and llama_rsz_rt_armable()
    // (llama-nout-state.h), pure predicates a model-free test pins. This site
    // supplies their inputs and counts the refusals; it does not restate the
    // rule. rsz_relay_ok is the gate half, rsz_row_armed below is the whole
    // rule for the relay form.
    const bool rsz_rows_ok = rsz_recr == nullptr || rsz_n_rs_seq == 0;
    if (nout_st != nullptr && nout_st->rsz_rt_enabled && rsz_gate_ok && !rsz_relay_ok && !rsz_rows_ok) {
        nout_st->rsz_rt_refuse(nout_st->c_rsz_rt_refused_n_rs_seq);
    }
    // ONE predicate decides whether this build carries the runtime rs row.
    // It is true on B's binder lane (union_active already creates s_zero) and,
    // independently, on the relay lane: a relay context never passes B's gate
    // (union_active requires backend_tp5_relay == nullptr), yet the ticket
    // needs the row there. On relay the row is written per set_inputs() and no
    // shape is bound -- union_rebind_apply still refuses relay contexts, which
    // is exactly right: a row move must not bind anything.
    //
    // The relay half is llama_rsz_rt_relay_armable() itself, not the bare gate.
    // The predicate is stricter than the bare gate in two ways, and neither may
    // change which builds carry the index tensor:
    //
    //   has_recr   a relay context with no recurrent sub-cache still reaches
    //              this function (graph_params() does not know the arch). The
    //              flag is inert there -- build_rs_inp_impl() is only reached
    //              for graphs that have an rs input -- so the pre-existing
    //              arming is preserved by the `rsz_recr == nullptr` term.
    //
    //   rs_z >= 0  the relay lane arms on a build whose rs_z is still -1 (no
    //              free zero cell yet); the bound row is then the -1 no-op of
    //              union_rsz_zero_row() (src/llama-graph.cpp:511-513). A
    //              negative live rs_z is refused by name at decode time
    //              (process_ubatch, c_rsz_rt_refused_cell), not here.
    //
    // Both relaxations reproduce `union_active || rsz_relay_ok` exactly, so
    // this change cannot move the node set of any graph.
    const int32_t rsz_rs_z_now = union_rs_z_now(mctx);
    const bool rsz_relay_armed = rsz_relay_ok &&
        (rsz_recr == nullptr ||   // no rs input to arm: the flag is inert
         llama_rsz_rt_relay_armable(/*relay_ok=*/true, /*has_recr=*/true,
                                    rsz_n_rs_seq, rsz_rs_z_now) ||
         rsz_rs_z_now < 0);       // -1 row is the op's no-op
    const bool rsz_row_armed = union_active || rsz_relay_armed;
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
        /*.n_outputs_build =*/nout_build,
        /*.union_binder    =*/union_active,
        /*.strict_ntok     =*/false,
        /*.rs_row_armed    =*/rsz_row_armed,

        /*.cb          =*/graph_get_cb(),
        /*.res         =*/res,
    };
}

ggml_status llama_context::graph_compute(ggml_cgraph * gf, bool batched, ggml_backend_sched_t sched_override) {
    int               n_threads = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp        = batched ? threadpool_batch : threadpool;

    // The graph uid must be stable before the scheduler hands this graph to
    // the Meta backend. A reused graph keeps its uid; a rebuilt graph is a new
    // object (llm_graph_result::reset) and receives a fresh one here.
    tp5_assign_graph_uid(backend_tp5_relay, gf);

    set_backend_cpu_threadpool(tp);

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    // Maintenance graphs can reuse the reserve arena between submissions.
    // Only the inference result owns inputs through the fences below.
    const bool retain_inputs = backend_tp5_relay != nullptr && gf_res_current &&
            gf == gf_res_current->get_gf() && !batched &&
            cparams.n_batch == 1 && cparams.n_ubatch == 1;
    ggml_backend_sched_t sched_run = sched_override ? sched_override : sched.get();
    const bool retained_api = retain_inputs && (sched_override == nullptr || shape_plan_active);
    auto status = retained_api ?
            ggml_backend_sched_graph_compute_async_retained_inputs(sched_run, gf) :
            ggml_backend_sched_graph_compute_async(sched_run, gf);
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

llama_host_memory_breakdown_data llama_context::host_memory_breakdown() const {
    return memory ? memory->host_memory_breakdown() : llama_host_memory_breakdown_data {};
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

            ++n_graph_builds;
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
    if (!llama_batch_is_complete(&batch, (int32_t) n_batch, false, true)) {
        llama_batch_free(batch);
        // void API, no error channel: fail-stop is the documented behavior here.
        GGML_ABORT("llama_batch_init failed in opt_epoch");
        return;
    }
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

    // The context holds one live model reference. Acquire it before the first
    // model field is read: llama_model_free defers while live contexts exist,
    // so the model cannot be deleted under construction. Any validation return
    // or constructor exception releases the reference through this guard.
    struct model_lifetime_ref {
        llama_model * model;
        bool acquired;
        bool adopted = false;

        explicit model_lifetime_ref(llama_model * model)
            : model(model), acquired(model->acquire_context()) {}

        ~model_lifetime_ref() {
            if (acquired && !adopted && model->release_context()) {
                delete model;
            }
        }
    } model_ref(model);

    if (!model_ref.acquired) {
        LLAMA_LOG_ERROR("%s: model free already requested\n", __func__);
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
        model_ref.adopted = true;
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
    if (ctx == nullptr) {
        return;
    }
    if (ctx->is_retained()) {
        // Already process-owned. Re-evaluating the schedulers or the relay
        // would touch state that no longer has an owner contract.
        LLAMA_LOG_ERROR("%s: context is already retained; duplicate free ignored\n", __func__);
        return;
    }
    llama_model * model = const_cast<llama_model *>(&ctx->get_model());
    bool retain = false;
    if (ctx->get_tp5_relay_backend() != nullptr) {
        ggml_relay_status status = {};
        ggml_backend_t backend = ctx->get_tp5_relay_backend();
        if (!ggml_backend_relay_snapshot(backend, &status) ||
            (status.epoch != 0 && !status.finished)) {
            ggml_backend_relay_abort(backend, GGML_STATUS_ABORTED);
            // Scheduler buffers and memory are also native readers' owners.
            // Do not enter an unbounded synchronize or release them on failure.
            retain = true;
        }
    }
    if (ctx->has_unproven()) {
        retain = true;
    }
    if (retain) {
        // Retain the whole context: every member (memory, arena, balloc,
        // cvec/loras/sampling, graph results) stays alive, and the model's
        // retained-context count keeps llama_model_free from freeing it.
        // The process owns all of it until exit. This is retention, not a
        // native drain.
        if (!ctx->retain_for_process()) {
            LLAMA_LOG_ERROR("%s: context is already retained; duplicate free ignored\n", __func__);
            return;
        }
        LLAMA_LOG_ERROR("%s: retaining whole context: an async/native submission is still unproven\n", __func__);
        return;
    }
    delete ctx;
    // The context released its model reference. When llama_model_free was
    // requested while the context was alive, the last release owns the delete.
    if (model->release_context()) {
        delete model;
    }
}

bool llama_get_ntok_plan_stats(const llama_context * ctx, llama_ntok_plan_stats * stats) {
    return ctx && ctx->ntok_plan_stats(stats);
}

bool llama_get_nkv_runtime_stats(const llama_context * ctx, llama_nkv_runtime_stats * stats) {
    return ctx && ctx->nkv_runtime_stats(stats);
}

// The KQ mask the live graph's runtime n_kv input filled, if any. The two
// functions below are the read-back's test surface: they read the same
// llm_graph_input_nkv_runtime the fill wrote and re-run the same
// llama_kv_cache::check_mask_row the fill runs. There is no second
// implementation of the check, which is the point: a test that corrupts the
// row and sees the check fail proves the production check can fail.
static const llm_graph_input_nkv_runtime * union_live_mask_input(const llama_context * ctx) {
    if (ctx == nullptr) {
        return nullptr;
    }
    const llm_graph_result * res = ctx->get_gf_res_prev();
    if (res == nullptr) {
        return nullptr;
    }
    const llm_graph_input_nkv_runtime * rt = res->get_nkv_runtime_input();
    if (rt == nullptr || rt->mask_n_kv == 0) {
        return nullptr;
    }
    return rt;
}

// The first registered F32 mask, which is the one the fill checked. The lane
// registers exactly one mask (build_attn_inp_kv_impl); a graph with none has
// nothing to read back.
static ggml_tensor * union_live_mask_tensor(const llm_graph_input_nkv_runtime * rt) {
    if (rt == nullptr) {
        return nullptr;
    }
    for (ggml_tensor * mask : rt->kq_masks) {
        if (mask != nullptr && mask->data != nullptr && mask->type == GGML_TYPE_F32) {
            return mask;
        }
    }
    return nullptr;
}

bool llama_union_kq_mask_row(const llama_context * ctx, uint32_t row, float ** data, int64_t * ne0) {
    if (data != nullptr) {
        *data = nullptr;
    }
    if (ne0 != nullptr) {
        *ne0 = 0;
    }
    const llm_graph_input_nkv_runtime * rt = union_live_mask_input(ctx);
    ggml_tensor * mask = union_live_mask_tensor(rt);
    if (mask == nullptr || data == nullptr) {
        return false;
    }
    const uint32_t width = rt->mask_n_kv;
    if (width == 0 || (int64_t) width > mask->ne[0]) {
        return false;
    }
    // The stride is the tensor's own ne[0], not the fill width: the fill pads
    // every column from `width` up to ne[0], so row i starts at ne[0]*i and
    // width*i would land inside a padded row. The row count is ne[1]*ne[3]
    // (rows per stream times streams).
    if ((int64_t) row >= mask->ne[1] * mask->ne[3]) {
        return false;
    }
    // The live host row, the same pointer the read-back compares. The mask is
    // flagged GGML_TENSOR_FLAG_OUTPUT at registration, so ggml-alloc keeps its
    // block alive and mask->data still holds the row the fill wrote.
    *data = (float *) mask->data + (size_t) row * (size_t) mask->ne[0];
    if (ne0 != nullptr) {
        *ne0 = mask->ne[0];
    }
    return true;
}

enum llama_gate_verdict llama_union_mask_row_check(const llama_context * ctx) {
    const llm_graph_input_nkv_runtime * rt = union_live_mask_input(ctx);
    ggml_tensor * mask = union_live_mask_tensor(rt);
    if (mask == nullptr) {
        return LLAMA_GATE_UNPROVEN;
    }
    const llama_ubatch & ubatch = ctx->get_ubatch_inputs_last();
    if (ubatch.n_tokens == 0) {
        return LLAMA_GATE_UNPROVEN;
    }
    const uint32_t width = rt->mask_n_kv;
    if (width == 0 || (int64_t) width > mask->ne[0]) {
        return LLAMA_GATE_UNPROVEN;
    }
    // The last token of the last batch: its row covers the whole step, so the
    // comparison is not vacuous. The row offset is the check's business.
    //
    // ONE row, deliberately. Multi-row coverage is the fill's job: it walks
    // every row of the batch when the batch is at most k_rows_max tokens
    // (llama-graph.cpp's set_input) and reports the totals through
    // mask_row_checked / mask_rows_all. This function is the test surface for
    // "the same check, called again", not a second sweep, so it must not be
    // widened to compensate for anything the fill does not do.
    const uint32_t tok = ubatch.n_tokens - 1;

    // A step whose bind was refused and whose graph was then rebuilt has a
    // patch that may have landed on a graph nobody executes (the pad-tail and
    // shape patches are written inside union_rebind_apply, before the record
    // is updated and before the post-bind can_reuse refusal). The rebuilt
    // graph is a different object, so the hazard is a stale RECORD, not a
    // stale node -- but a read-back taken across that boundary would compare a
    // row against cells the record no longer describes. Refuse: UNPROVEN, not
    // FAIL, because nothing was compared.
    llama_union_diagnostics diag = {};
    if (llama_get_union_diagnostics(ctx, &diag) && diag.rebuild_fallbacks != 0) {
        return LLAMA_GATE_UNPROVEN;
    }

    // The cache is resolved from the context's own memory, which lives as long
    // as the context. The per-decode memory CONTEXT is not consulted: it is
    // freed when decode returns, and this function is called after that.
    llama_kv_cache * kv = union_kv_cache_of(ctx->get_memory());
    if (kv == nullptr) {
        return LLAMA_GATE_UNPROVEN;
    }
    // The same two rules the fill applies: the context's causal flag and the
    // cache's own SWA type, not defaults.
    const bool swa = kv->get_swa_type() != LLAMA_SWA_TYPE_NONE;
    int64_t first_bad = -1;
    // The check's verdict is returned as-is. PASS and FAIL are observations;
    // UNPROVEN means the check did not run. A caller must compare against the
    // named value, because UNPROVEN is not 0.
    // Compare the live row. The mask is flagged GGML_TENSOR_FLAG_OUTPUT when
    // it is registered (register_nkv_runtime_mask), so ggml-alloc keeps its
    // block and mask->data still holds the row the fill wrote when this runs
    // after llama_decode() has returned. Without that flag the allocator
    // would reuse the block after soft_max and this would read whatever took
    // its place.
    const float * row = (const float *) mask->data;
    const enum llama_gate_verdict verdict = kv->check_mask_row(mask, &ubatch, tok,
            row, width, ctx->get_cparams().causal_attn, swa, &first_bad);
    // Trace the witness: how many of the `width` floats in the live row still
    // agree with the row copied at fill time. A disagreement is evidence the
    // block was reused, never the verdict.
    if (union_bind_trace_on()) {
        size_t agree = 0;
        if (rt->mask_row_snapshot.size() >= (size_t) width) {
            for (uint32_t i = 0; i < width; ++i) {
                agree += (row[i] == rt->mask_row_snapshot[i]) ? 1 : 0;
            }
        }
        LLAMA_LOG_ERROR("%s: mask row check verdict=%d tok=%u width=%u first_bad=%lld "
                        "witness_agree=%zu/%u | mask_n_kv=%u mask_filled=%u mask_refused=%u "
                        "mask_row_checked=%u mask_row_ok=%u mask_row_refused=%u mask_rows_all=%u mask_ne0=%lld\n",
                        "llama_union_mask_row_check", (int) verdict, tok, width,
                        (long long) first_bad, agree, width,
                        rt->mask_n_kv, rt->mask_filled, rt->mask_refused,
                        rt->mask_row_checked, rt->mask_row_ok, rt->mask_row_refused,
                        rt->mask_rows_all, (long long) mask->ne[0]);
    }
    return verdict;
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

// DUALWIN-COMPOSE-EVENT (GGML_TP5_DW_COMPOSE_EVENT=1): resolve one output row
// without synchronizing. The caller must already hold this epoch's output-copy
// completion proof (llama_tp5_output_event_wait, or an equivalent full
// synchronize). Only the host-side output reorder and row resolution happen
// here.
float * llama_get_logits_ith_no_sync(llama_context * ctx, int32_t i) {
    if (ctx == nullptr) {
        return nullptr;
    }

    // The recorded output-copy events cover exactly the child streams of the
    // FULL-LOGITS D2H copy (ggml_backend_meta_output_record mirrors
    // ggml_backend_meta_get_tensor_async's stream set for that one tensor) and
    // nothing else. Device-sampled logits resolve through different tensors,
    // whose async copies run on streams no recorded event orders, so returning
    // them under this proof would read data no event covered. Callers that
    // need sampled outputs must keep the stock synchronizing getters.
    return ctx->get_logits_ith(i);
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

// DUALWIN-COMPOSE-EVENT (GGML_TP5_DW_COMPOSE_EVENT=1): output-copy completion as
// device events. The record must be issued AFTER llama_decode returned, so every
// event is ordered behind the graph AND behind the asynchronous logits D2H copy
// that decode enqueued on the same child stream(s); the wait is then a sound
// host barrier for the readable logits row and replaces the scheduler-wide
// synchronize. No relay counter participates in this proof.
size_t llama_tp5_output_copy_record(llama_context * ctx) {
    return ctx != nullptr ? ctx->tp5_output_copy_record() : 0;
}

size_t llama_tp5_output_event_wait(llama_context * ctx) {
    return ctx != nullptr ? ctx->tp5_output_copy_wait() : 0;
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

bool llama_context::union_diagnostics(llama_union_diagnostics * out) const {
    if (out == nullptr || union_binder_state == nullptr) return false;
    const auto & st = *union_binder_state;
    // `topology_hash` here is the width-table hash of llama-union-topology.h
    // (st.topology_hash), NOT st.record_topology_key.hash. The two answer
    // different questions and the public field keeps the one the pillar is
    // observed through:
    //
    //   st.topology_hash          the live graph's topology with the n_kv,
    //                             n_outputs, rs_z and (on the token lane)
    //                             n_tokens dimensions excluded. It is re-taken
    //                             after every applied bind, so an unchanged
    //                             value across a parameter sweep is evidence
    //                             that the bind changed shapes only.
    //                             llama_get_nkv_runtime_stats() reports the
    //                             same value, and test-nkv-runtime-param and
    //                             test-topology-key-sweep compare it across
    //                             cells.
    //   st.record_topology_key    the single graph cache key of
    //                             llama-topology-hash.h. It is the binder's
    //                             reuse gate and it is invariant to those
    //                             dimensions by construction, so exporting it
    //                             here would report a number that cannot move
    //                             when the parameter does - stable for the
    //                             wrong reason. It stays internal; a caller
    //                             that needs it reads it from the binder, not
    //                             from this diagnostic.
    //
    // `exact_lane` is the GGML_TP5_UNION_BINDER_EXACT selection, a
    // configuration value: 1 when the runtime-length lane is active, 0 when
    // the stock padded control lane is. It is NOT an observation of what a
    // step executed at; bound_n_kv is that observation.
    // 26 values for 26 fields, in declaration order (include/llama.h).
    // The rs_z lane's block is appended after exact_lane, so every earlier
    // field keeps its position.
    //
    // rsz_rt_bound_rs_z: the rs_z the live graph is bound at. The authoritative
    // value is record_bound_rs_z, refreshed at the record build and again by
    // every applied union_rebind_apply() ("record_bound_rs_z =
    // union_rs_z_now(...)"). The stored rsz_rt_bound_rs_z is only the
    // record-build snapshot, so reading it here would report the build-time
    // value on every step after a bind -- the diagnostic would name a row the
    // graph no longer clears. Read the record while it is valid and fall back
    // to the snapshot only when it is not (a stock rebuild invalidates it).
    const int32_t rsz_bound_rs_z = st.record_valid ? st.record_bound_rs_z
                                                   : st.rsz_rt_bound_rs_z;
    *out = { st.c_fallbacks, st.c_rebuild_fallbacks, st.c_patch_binds, st.c_pad_fallbacks,
             st.record_bound_ntok, st.record_bound, st.nkv_enabled, st.env_disabled,
             st.record_valid, st.shape_map.ntok_valid, st.topology_hash,
             st.nkv_exact_enabled ? 1 : 0,
             st.rsz_rt_enabled, st.rsz_rt_active, st.c_rsz_rt_fallbacks,
             st.c_rsz_rt_refused_gate, st.c_rsz_rt_refused_arch,
             st.c_rsz_rt_refused_no_rs_z, st.c_rsz_rt_refused_cell,
             st.c_rsz_rt_refused_envelope, st.c_rsz_rt_refused_shape,
             st.c_rsz_rt_refused_n_rs_seq, st.c_rsz_rt_refused_ring,
             st.c_rsz_rt_refused_mixed_depth,
             rsz_bound_rs_z, st.rsz_rt_zero_view_rows };
    return true;
}

bool llama_context::copy_last_llama_activation(int32_t row, float * dst, size_t count) {
    if (!last_llama_activation_valid || dst == nullptr || row < 0 ||
            (uint32_t) row >= n_outputs || count == 0 || count != model.hparams.n_embd_out() ||
            !gf_res_current || !sched || !backend_cpu) return false;
    auto * t = gf_res_current->get_embd();
    if (!t || t->type != GGML_TYPE_F32 || t->ne[0] != (int64_t) count ||
            t->ne[1] != (int64_t) n_outputs || t->ne[2] != 1 || t->ne[3] != 1 ||
            !ggml_is_contiguous(t) || !t->data ||
            !(t->view_src ? t->view_src->buffer : t->buffer) ||
            ggml_backend_sched_get_tensor_backend(sched.get(), t) != backend_cpu) return false;
    if (count > SIZE_MAX / sizeof(float)) return false;
    const size_t row_bytes = count * sizeof(float);
    if ((size_t) row > SIZE_MAX / row_bytes) return false;
    const size_t offset = (size_t) row * row_bytes;
    const size_t tensor_bytes = ggml_nbytes(t);
    if (offset > tensor_bytes || row_bytes > tensor_bytes - offset) return false;
    synchronize();
    ggml_backend_tensor_get(t, dst, offset, row_bytes);
    return true;
}

bool llama_get_union_diagnostics(const llama_context * ctx, llama_union_diagnostics * out) {
    return ctx && ctx->union_diagnostics(out);
}

bool llama_copy_last_llama_activation(llama_context * ctx, int32_t row, float * dst, size_t count) {
    return ctx && ctx->copy_last_llama_activation(row, dst, count);
}

uint64_t llama_graph_build_count(const llama_context * ctx) {
    return ctx ? ctx->graph_build_count() : 0;
}

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

llama_host_memory_breakdown_data llama_get_host_memory_breakdown(const struct llama_context * ctx) {
    return ctx->host_memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}

//
// factor-prefab envelopes (see llama-prefab.h)
//

static bool prefab_env_flag(const char * name) {
    // enabled only by the exact string "1": unset, "0", "true", "01", "1 ", "2" all stay off,
    // so the disabled path is bitwise-identical to baseline and no loose truthy value can
    // activate an experimental lane
    const char * v = getenv(name);
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

void llama_context::prefab_init() {
    // whole-graph MTP lane (LLAMA_PREFAB_MTP): instead of composing factor envelopes, build the
    // stock model graph (e.g. qwen4exp graph_mtp for LLM_GRAPH_TYPE_DECODER_MTP) into a prefab
    // slot and reuse it when the graph inputs allow. The stock builder reproduces every MTP
    // feature exactly (wide hc stream, hyper-connections, IMROPE, token+embd dual input), so
    // this is a strict correctness superset of the factor path for MTP contexts.
    //
    // GGML_TP5_PREFAB_AB=1 (r7, production): single-flag production lane over the same
    // whole-graph slot machinery. The AB gate itself only widens WHICH lane handles an
    // eligible MTP context: static-contract refusals below still throw (fail closed, they
    // never silently reroute), and ubatches that are not exact-fit candidates stay counted
    // (n_fallback_runtime) and run by the stock builder -- exact can_reuse, no pad, no
    // headroom, no bucket.
    const bool in_graph = llama_prefab_in_graph_enabled();
    const bool ab       = llama_prefab_ab_enabled();
    const bool mtp      = prefab_env_flag("LLAMA_PREFAB_MTP") || in_graph || ab;

    if (!mtp && !prefab_env_flag("LLAMA_PREFAB_ENABLE")) {
        return; // default off
    }
    if (!mtp && !prefab_env_flag("LLAMA_PREFAB_EXPERIMENTAL")) {
        throw std::runtime_error("factor-prefab: LLAMA_PREFAB_ENABLE=1 requires LLAMA_PREFAB_EXPERIMENTAL=1");
    }

    const auto & hparams = model.hparams;

    const bool mtp_ok = mtp && cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && model.arch == LLM_ARCH_QWEN4EXP;

    // LLAMA_PREFAB_MTP=1 / GGML_TP5_PREFAB_IN_GRAPH=1 select the MTP lane only: any context
    // that is not an eligible Qwen4Exp MTP context (e.g. the Qwen4Exp DEFAULT target context
    // kept alive by the speculative driver, or a plain trunk ctx created while the flag is on)
    // runs pure stock -- no prefab state, no factorization claim, no throw.
    if (mtp && !mtp_ok) {
        LLAMA_LOG_WARN("%s: MTP prefab lane requested but this context is not an eligible Qwen4Exp MTP context; running stock (no prefab state)\n",
                       __func__);
        return;
    }

    // static contract: anything the factor decomposition does not reproduce exactly is refused here,
    // so an enabled context can never silently run a half-composed graph.
    // MTP mode (mtp_ok) bypasses the factor-specific rejects (arch / ctx type / swa / alibi /
    // m-rope / n_embd_inp / nextn outputs): the stock-built graph handles all of those natively.
    // Flags-off parity: with LLAMA_PREFAB_MTP unset, mtp_ok=false makes every !mtp_ok-gated
    // condition identical to the original unconditional chain.
    std::string why;
    auto * kv = dynamic_cast<llama_kv_cache *>(memory.get());
    // mtp_ok bypasses the hybrid reject: an MTP ctx over a hybrid-Qwen trunk (qwen3next /
    // qwen35 / qwen35moe / qwen4exp) owns a PLAIN llama_kv_cache by construction
    // (llama-model.cpp mtp_on_hybrid_qwen), so the kv==nullptr / n_stream checks below are
    // the real memory gate; the factor-path hybrid objection does not apply to the
    // stock-built MTP graph this lane caches.
    if (!mtp_ok && (llm_arch_is_recurrent(model.arch) || llm_arch_is_hybrid(model.arch))) {
        why = "recurrent/hybrid memory: rs_z/rs_head are baked into stock views; no envelope factors for it";
    } else if (!mtp_ok && model.arch != LLM_ARCH_LLAMA) {
        why = std::string("arch '") + llm_arch_name(model.arch) + "' has no factor decomposition (supported: llama)";
    } else if (!mtp_ok && cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT) {
        why = "non-default context type";
    } else if (kv == nullptr) {
        why = "memory is not a plain llama_kv_cache";
    } else if (kv->get_n_stream() != 1) {
        why = "multi-stream KV cache (kv_unified=false with n_seq_max>1)";
    } else if (!mtp_ok && (hparams.swa_type != LLAMA_SWA_TYPE_NONE || hparams.use_alibi || hparams.n_pos_per_embd() != 1 ||
               hparams.n_embd_inp() != hparams.n_embd)) {
        why = "unsupported hparams (swa / alibi / m-rope / n_embd_inp != n_embd)";
    } else if (cparams.embeddings && cparams.pooling_type != LLAMA_POOLING_TYPE_NONE) {
        why = "pooled embeddings";
    } else if (!mtp_ok && (cparams.embeddings_nextn || cparams.mtp_chain)) {
        why = "nextn/mtp outputs";
    } else if (mtp_ok && (model.tok_embd == nullptr || model.output == nullptr)) {
        why = "draft-only checkpoint requires ctx_other (token embedding or output projection missing)";
    } else if (std::any_of(cparams.embeddings_layer_inp.begin(), cparams.embeddings_layer_inp.end(), [](bool b) { return b; })) {
        why = "layer-input extraction";
    } else if (backend_tp5_relay != nullptr || backend_quantized_matmul_f32 != nullptr) {
        why = "tp5 relay / quantized_matmul_f32 backends";
    } else if (kv->has_kv_stream_targets()) {
        why = "kv-stream runtime";
    }
    if (!why.empty()) {
        throw std::runtime_error("factor-prefab: refusing context (fail closed): " + why);
    }

    // envelope widths: LLAMA_PREFAB_ENVELOPES="1,2,3,4,512" (default
    // 1,2,3,4,n_ubatch); n_ubatch is always the top envelope. The MTP verify
    // batch is 1 accepted token + up to mtp_depth draft tokens (default depth
    // 1 => 2 rows), so 2 and 3 must be real envelope widths: the A+B exact-fit
    // lane requires e == n_tokens and would otherwise decline every 2/3-row
    // verify to the stock builder, while the smallest-fit lane would pad 2-row
    // verify to 4 and compute two dead rows in every linear/MoE/attention op.
    std::vector<uint32_t> env;
    if (const char * s = getenv("LLAMA_PREFAB_ENVELOPES")) {
        std::string str(s);
        size_t p = 0;
        while (p < str.size()) {
            size_t q = str.find(',', p);
            if (q == std::string::npos) {
                q = str.size();
            }
            const long v = atol(str.substr(p, q - p).c_str());
            if (v > 0) {
                env.push_back((uint32_t) v);
            }
            p = q + 1;
        }
    } else {
        env = { 1, 2, 3, 4 };
    }
    const uint32_t n_max = std::min(cparams.n_ctx, cparams.n_ubatch);
    env.push_back(n_max);
    env.erase(std::remove_if(env.begin(), env.end(), [&](uint32_t v) { return v > n_max; }), env.end());
    std::sort(env.begin(), env.end());
    env.erase(std::unique(env.begin(), env.end()), env.end());

    if (llama_prefab_decouple_enabled()) {
        // ---- decouple path (GGML_TP5_PREFAB_DECOUPLE=1): explicit A->B interface ----
        // A (this layer) computes the eligibility facts and hands them to B as
        // values; B initializes its own state. No direct slot-table writes.
        llama_prefab_eligibility elig;
        elig.mtp_mode   = mtp_ok;
        elig.envelopes  = env;
        prefab = std::make_unique<llama_prefab_state>();
        if (!llama_prefab_state_init(*prefab, elig)) {
            prefab.reset();
            throw std::runtime_error("factor-prefab: empty envelope set (decouple)");
        }
        // A owns the backend list, the node-budget policy and the slot result
        // objects; it attaches ready handles to B's slots.
        if (!moe_cache_configured) {
            throw std::runtime_error("factor-prefab: no resolved MoE cache configuration");
        }
        std::string desc;
        for (size_t i = 0; i < env.size(); ++i) {
            ggml_backend_sched_t slot_sched = ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(),
                                                    backend_ptrs.size(), graph_max_nodes(env[i]),
                                                    cparams.pipeline_parallel, cparams.op_offload);
            if (slot_sched == nullptr) {
                throw std::runtime_error("factor-prefab: failed to create a slot scheduler");
            }
            if (!llama_apply_moe_cache_config(slot_sched, llama_to_ggml_moe_cache_mode(moe_cache_mode_resolved),
                                              moe_cache_budget_mib)) {
                ggml_backend_sched_free(slot_sched);
                throw std::runtime_error("factor-prefab: failed to apply the MoE cache configuration to a slot scheduler");
            }
            llama_prefab_attach_slot(*prefab, i, slot_sched, new llm_graph_result(graph_max_nodes(env[i])));
            desc += (i ? "," : "") + std::to_string(env[i]);
        }
        if (mtp_ok) {
            LLAMA_LOG_WARN("%s: %s (decoupled compose): whole-graph MTP mode%s%s, envelopes N={%s}\n", __func__,
                    ab ? "production A+B compose lane (GGML_TP5_PREFAB_AB=1)" : "factor-prefab EXPERIMENTAL enabled",
                    (in_graph || ab) ? ", A x B compose on rebuild" : "",
                    llama_prefab_noclone_enabled() ? ", shared A nodes (noclone)" : "",
                           desc.c_str());
        } else {
            LLAMA_LOG_WARN("%s: factor-prefab EXPERIMENTAL enabled (decoupled compose): envelopes N={%s}, factors EMBD/QKV/ATTN/FFN/OUT\n",
                           __func__, desc.c_str());
        }
        return;
    }

    prefab = std::make_unique<llama_prefab_state>();
    prefab->envelopes = env;
    prefab->slots.resize(env.size());
    prefab->mtp_mode  = mtp_ok;

    std::string desc;
    for (size_t i = 0; i < env.size(); ++i) {
        auto & s = prefab->slots[i];
        s.n_env = env[i];
        s.res.reset(new llm_graph_result(graph_max_nodes(env[i])));
        desc += (i ? "," : "") + std::to_string(env[i]);
    }
    if (mtp_ok) {
        LLAMA_LOG_WARN("%s: %s: whole-graph MTP mode%s%s, envelopes N={%s}\n", __func__,
                ab ? "production A+B compose lane (GGML_TP5_PREFAB_AB=1)" : "factor-prefab EXPERIMENTAL enabled",
                (in_graph || ab) ? ", A x B compose on rebuild" : "",
                llama_prefab_noclone_enabled() ? ", shared A nodes (noclone)" : "",
                       desc.c_str());
    } else {
        LLAMA_LOG_WARN("%s: factor-prefab EXPERIMENTAL enabled: envelopes N={%s}, factors EMBD/QKV/ATTN/FFN/OUT\n",
                       __func__, desc.c_str());
    }
}

ggml_backend_sched_t llama_context::sched_active() const {
    return (prefab && prefab->active) ? prefab->active : sched.get();
}

bool llama_context::prefab_process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx,
                                          llm_graph_result *& out, ggml_status & ret) {
    auto & st = *prefab;
    out = nullptr;

    if (llama_prefab_decouple_enabled()) {
        // ---- decouple path (GGML_TP5_PREFAB_DECOUPLE=1) ----
        // A runs its fail-closed runtime checks (the SAME checks as the legacy
        // path below, on local values), then hands an explicit request to B.
        // A never reads st.slots / st.scratch_cell to decide or to drive the
        // ubatch; the only state it reads is the mode flag it itself set at init
        // (st.mtp_mode) plus the envelope widths it itself chose (st.envelopes).
        auto * kvc = dynamic_cast<llama_kv_cache_context *>(mctx);
        // IMROPE draft heads (qwen4exp) carry n_pos >= 3 per token; their own pos input
        // (build_inp_pos) writes the full multi-section positions, so the MTP mode only
        // needs the 1-D restriction of the factor path, not of the stock head it reuses.
        const bool pos_ok = !ubatch.is_pos_2d() || st.mtp_mode;
        bool ok = kvc != nullptr && ubatch.token != nullptr && pos_ok &&
                  !kvc->has_attention_views() && !mctx->has_kv_stream_targets() &&
                  (loras == nullptr || loras->empty()) && sampling.samplers.empty();
        if (st.mtp_mode) {
            // MTP mode: only the MTP draft graph; draft batches always carry embd
            if (gtype != LLM_GRAPH_TYPE_DECODER_MTP) {
                llama_prefab_stat_tick(prefab_stats.n_fallback_runtime);
                return false;
            }
        } else {
            // factor mode: token-only ubatches, DEFAULT/DECODER graphs only
            ok = ok && ubatch.embd == nullptr && (gtype == LLM_GRAPH_TYPE_DEFAULT || gtype == LLM_GRAPH_TYPE_DECODER);
        }
        if (ok && cvec) {
            for (int il = 0; il < (int) model.hparams.n_layer(); ++il) {
                if (cvec->tensor_for(il) != nullptr) {
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) {
            llama_prefab_stat_tick(prefab_stats.n_fallback_runtime);
            return false;
        }

        // envelope selection: smallest envelope >= n_tokens (A picks, B validates).
        // GGML_TP5_PREFAB_AB=1 (mtp_mode only): require an exact-fit candidate -- a
        // sub-envelope ubatch is rejected here (counted as a runtime fallback) instead
        // of being padded into a larger envelope; factor-mode envelopes keep the
        // smallest-fit selection, so other gates are untouched.
        uint32_t n_env = 0;
        const bool ab_exact = llama_prefab_ab_enabled() && st.mtp_mode;
        for (uint32_t e : st.envelopes) {
            if (ab_exact ? e == ubatch.n_tokens : ubatch.n_tokens <= e) {
                n_env = e;
                break;
            }
        }
        if (n_env == 0) {
            llama_prefab_stat_tick(prefab_stats.n_fallback_runtime);
            return false;
        }

        llama_prefab_request req;
        req.gtype               = gtype;
        req.n_tokens            = ubatch.n_tokens;
        req.n_env               = n_env;
        req.mtp_mode            = st.mtp_mode;
        req.graph_reuse_disable = graph_reuse_disable;
        req.pipeline_parallel   = cparams.pipeline_parallel;
        req.cb_eval             = cparams.cb_eval;
        req.cb_eval_user_data   = cparams.cb_eval_user_data;
        req.nkv_obs             = llama_prefab_nkv_obs_enabled();

        // res is B's own (slot-owned); the main sched is passed so the graph
        // callback's placement hints (offload_kqv, moe f16 guard) behave
        // exactly as in the legacy path, where gparams.sched was also the
        // main sched at compose time
        const auto gparams = graph_params(nullptr, ubatch, mctx, gtype);

        const auto rep = llama_prefab_process(st, prefab_stats, model, gparams, ubatch, mctx, req,
            [this](ggml_cgraph * gf, bool batched, ggml_backend_sched_t slot_sched) {
                return graph_compute(gf, batched, slot_sched);
            });

        if (!rep.handled) {
            // B declined: fall back to the stock builder (decline counted by B)
            return false;
        }
        ret = rep.status;
        if (rep.reused) {
            // map B's reuse to the context's graphs-reused perf counter
            n_reused++;
        }
        if (rep.status == GGML_STATUS_SUCCESS) {
            out = rep.result;
        }
        return true;
    }

    // this branch handles only the MTP draft graph; DEFAULT/DECODER requests (e.g. warmup)
    // fall through to the stock builder -- the trunk graph must never be built here
    if (st.mtp_mode && gtype != LLM_GRAPH_TYPE_DECODER_MTP) {
        llama_prefab_stat_tick(prefab_stats.n_fallback_runtime);
        return false;
    }
    if (st.mtp_mode) {
        // ---- whole-graph MTP mode (LLAMA_PREFAB_MTP) ----
        // Build the stock DECODER_MTP graph (e.g. qwen4exp graph_mtp) into a slot and reuse it
        // only when every graph input allows (llm_graph_result::can_reuse enforces exact n_kv
        // mask width, n_tokens and n_outputs). No envelope padding, no scratch cell, no
        // canonical ubatch: the real ubatch is built into the graph as-is, so fallback to the
        // stock builder is always safe and exact.
        // runtime contract; anything else falls back to the stock builder (counted)
        auto * kvc_m = dynamic_cast<llama_kv_cache_context *>(mctx);
        // IMROPE draft heads (qwen4exp) carry n_pos >= 3 per token; their own pos input
        // writes the full multi-section positions, so the MTP mode does not need the
        // factor path's 1-D restriction.
        const bool pos_ok_m = !ubatch.is_pos_2d() || st.mtp_mode;
        bool ok_m = kvc_m != nullptr && ubatch.token != nullptr && pos_ok_m &&
                    !kvc_m->has_attention_views() && !mctx->has_kv_stream_targets() &&
                    (loras == nullptr || loras->empty()) && sampling.samplers.empty();
        if (ok_m && cvec) {
            for (int il = 0; il < (int) model.hparams.n_layer(); ++il) {
                if (cvec->tensor_for(il) != nullptr) {
                    ok_m = false;
                    break;
                }
            }
        }
        // exact-fit slot only: the stock MTP graph has no padding, so a sub-envelope slot
        // with n_env > n_tokens would compute on padded mask columns (wrong n_kv semantics)
        const int is_m = ok_m ? st.slot_for(ubatch.n_tokens) : -1;
        if (is_m < 0 || st.slots[is_m].n_env != ubatch.n_tokens) {
            llama_prefab_stat_tick(prefab_stats.n_fallback_runtime);
            return false;
        }

        auto & slot = st.slots[is_m];
        const uint32_t n_env = slot.n_env;

        if (!slot.sched) {
            if (!moe_cache_configured) {
                throw std::runtime_error("factor-prefab: no resolved MoE cache configuration");
            }
            slot.sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                                                    graph_max_nodes(n_env), cparams.pipeline_parallel, cparams.op_offload));
            if (!slot.sched) {
                throw std::runtime_error("factor-prefab: failed to create an MTP slot scheduler");
            }
            if (!llama_apply_moe_cache_config(slot.sched.get(),
                                              llama_to_ggml_moe_cache_mode(moe_cache_mode_resolved),
                                              moe_cache_budget_mib)) {
                slot.sched.reset();
                throw std::runtime_error("factor-prefab: failed to apply the MoE cache configuration to an MTP slot scheduler");
            }
        }

        auto gparams = graph_params(slot.res.get(), ubatch, mctx, gtype);
        gparams.sched = slot.sched.get();
        // gparams.n_outputs keeps the member value set by the decode loop (correct for MTP:
        // build_inp_out_ids is sized by it and can_reuse fails closed on mismatch)

        const llama_prefab_topology_key key = llama_prefab_key_for(model, gparams);

        ggml_cgraph * gf = nullptr;
        bool gf_composed = false;
        bool composed_attempted = false;
        if (slot.built && !graph_reuse_disable && slot.key == key && slot.res->can_reuse(gparams)) {
            gf = slot.res->get_gf();
            if (cparams.pipeline_parallel) {
                ggml_backend_sched_synchronize(slot.sched.get());
            }
            llama_prefab_stat_tick(prefab_stats.n_envelope_reuses);
            n_reused++;
        } else {
            if (llama_prefab_noclone_enabled() && st.mtp_mode) {
                // shared-node compose (GGML_TP5_PREFAB_NOCLONE=1): the graph being replaced
                // references the cached A template's tensors in place. The reuse path above
                // never reaches this branch, so the only graph that can reference them is
                // the one being replaced (one A template per envelope width, one slot per
                // width). Drain every dispatched-but-unsynchronized sched BEFORE the old
                // graph is destroyed and A's allocation state is reset, so the constraint
                // is enforced here rather than assumed from synchronous dispatch.
                st.sync_pending();
            }
            slot.built = false;
            slot.res->reset();
            ggml_backend_sched_reset(slot.sched.get());
            ggml_backend_sched_set_eval_callback(slot.sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

            {
            llama_prefab_build_timer timer{prefab_stats, gf};
            // MTP factor compose (GGML_TP5_PREFAB_IN_GRAPH=1, or the production single-flag
            // lane GGML_TP5_PREFAB_AB=1): the A x B envelope instead of the stock whole
            // graph. Exact fit only; a declined ubatch falls back to the stock builder
            // below, so with both gates off (or on a decline) this is the stock path.
            if ((llama_prefab_in_graph_enabled() || llama_prefab_ab_enabled()) && ubatch.n_tokens == n_env) {
                composed_attempted = true;
                try {
                    gf = llama_prefab_mtp_compose(model, gparams, n_env, st, prefab_stats);
                    gf_composed = gf != nullptr;
                } catch (const std::exception & e) {
                    // a declined ubatch, not a swallowed failure: logged and counted
                    LLAMA_LOG_WARN("%s: mtp factor compose declined: %s\n", __func__, e.what());
                    gf = nullptr;
                }
                if (!gf) {
                    llama_prefab_stat_tick(prefab_stats.n_mtp_declines);
                }
            }
            if (!gf && composed_attempted) {
                // the compose may have filled slot.res with a partial graph and inputs before it
                // declined: rebuild the slot from scratch so the stock graph gets a clean result.
                // Only after an attempt -- with the gate off the path is exactly the stock one.
                slot.res->reset();
                ggml_backend_sched_reset(slot.sched.get());
                ggml_backend_sched_set_eval_callback(slot.sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
            }
            if (!gf) {
                gf = model.build_graph(gparams);
                // a stock-built graph: count it as a stock build (it is one) on top of the
                // envelope alloc that tracks slot bookkeeping
                llama_prefab_stat_tick(prefab_stats.n_stock_builds);
            }
            }
            if (!ggml_backend_sched_alloc_graph(slot.sched.get(), gf)) {
                LLAMA_LOG_ERROR("%s: factor-prefab: failed to allocate MTP graph (N=%u)\n", __func__, n_env);
                ret = GGML_STATUS_ALLOC_FAILED;
                return true;
            }
            llama_prefab_stat_tick(prefab_stats.n_envelope_allocs);
            slot.built = true;
            slot.key   = key;
        }

        llama_prefab_stat_tick(prefab_stats.n_ubatch_prefab);
        if (llama_prefab_stats_enabled()) {
            prefab_stats.last_n_env          = n_env;
            prefab_stats.last_n_tokens       = ubatch.n_tokens;
            prefab_stats.last_graph_nodes    = ggml_graph_n_nodes(gf);
            prefab_stats.last_inp_tokens_ne0 = slot.res->get_inp_tokens() ? slot.res->get_inp_tokens()->ne[0] : 0;
        }
        slot.res->set_inputs(&ubatch);

        st.active = slot.sched.get();
        st.mark_pending(slot.sched.get());

        const auto status = graph_compute(gf, ubatch.n_tokens > 1, slot.sched.get());
        if (gf_composed && status == GGML_STATUS_SUCCESS) {
            llama_prefab_stat_tick(prefab_stats.n_mtp_compose_hits);
        }
        if (status != GGML_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: failed to compute MTP envelope graph, compute status: %d\n", __func__, status);
            ret = status;
            return true;
        }

        ret = GGML_STATUS_SUCCESS;
        out = slot.res.get();
        return true;
    }

    // runtime contract (anything else -> stock builder, counted)
    auto * kvc = dynamic_cast<llama_kv_cache_context *>(mctx);
    bool ok = kvc != nullptr && ubatch.token != nullptr && ubatch.embd == nullptr && !ubatch.is_pos_2d() &&
              !kvc->has_attention_views() && !mctx->has_kv_stream_targets() &&
              (loras == nullptr || loras->empty()) && sampling.samplers.empty();
    if (ok && cvec) {
        for (int il = 0; il < (int) model.hparams.n_layer(); ++il) {
            if (cvec->tensor_for(il) != nullptr) {
                ok = false;
                break;
            }
        }
    }
    const int is = ok ? st.slot_for(ubatch.n_tokens) : -1;
    if (is < 0) {
        llama_prefab_stat_tick(prefab_stats.n_fallback_runtime);
        return false;
    }

    auto & slot = st.slots[is];
    const uint32_t n_env = slot.n_env;

    st.scratch_cell = -1;
    if (ubatch.n_tokens < n_env) {
        st.scratch_cell = kvc->prefab_scratch_cell();
        if (st.scratch_cell < 0) {
            llama_prefab_stat_tick(prefab_stats.n_fallback_no_scratch);
            return false;
        }
    }

    if (!slot.sched) {
        if (!moe_cache_configured) {
            throw std::runtime_error("factor-prefab: no resolved MoE cache configuration");
        }
        slot.sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                                                graph_max_nodes(n_env), cparams.pipeline_parallel, cparams.op_offload));
        if (!slot.sched) {
            throw std::runtime_error("factor-prefab: failed to create a factor slot scheduler");
        }
        if (!llama_apply_moe_cache_config(slot.sched.get(),
                                          llama_to_ggml_moe_cache_mode(moe_cache_mode_resolved),
                                          moe_cache_budget_mib)) {
            slot.sched.reset();
            throw std::runtime_error("factor-prefab: failed to apply the MoE cache configuration to a factor slot scheduler");
        }
    }

    // canonical ubatch: envelope shape, real data (envelope inputs read the real ubatch and pad)
    llama_ubatch canon = ubatch;
    canon.n_tokens     = n_env;
    canon.n_seq_tokens = n_env;
    canon.n_seqs       = 1;
    canon.n_seqs_unq   = 1;
    canon.b_equal_seqs = 0;

    auto gparams = graph_params(slot.res.get(), canon, mctx, gtype);
    gparams.n_outputs = n_env;
    gparams.sched     = slot.sched.get();

    const llama_prefab_topology_key key = llama_prefab_key_for(model, gparams);
    // n_kv is a runtime parameter: reuse the envelope graph while the current n_kv fits the
    // padded KV width the graph was composed for; recompose (new ATTN factor key) only when
    // n_kv outgrows the pad window. can_reuse() alone is not sufficient: the inherited
    // llm_graph_input_attn_kv::can_reuse compares the mask width against the exact n_kv.
    const uint32_t n_kv_pad_cur = kvc->get_n_kv_pad();
    const bool n_kv_fits = slot.n_kv_pad != 0 && kvc->n_kv_pad_fits(slot.n_kv_pad);

    ggml_cgraph * gf = nullptr;
    bool gf_reused = false;
    if (slot.built && !graph_reuse_disable && slot.key == key && n_kv_fits && slot.res->can_reuse(gparams)) {
        gf = slot.res->get_gf();
        gf_reused = true;
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(slot.sched.get());
        }
        llama_prefab_stat_tick(prefab_stats.n_envelope_reuses);
        n_reused++;
    } else {
        if (llama_prefab_noclone_enabled()) {
            // Shared-node factor compose (GGML_TP5_PREFAB_NOCLONE=1): the envelope being
            // replaced references the cached factor modules' tensors in place. The reuse
            // path above never reaches this branch, so the only live graph that references
            // them is the one being replaced (one module set per envelope width, one slot
            // per width). Drain every dispatched-but-unsynchronized sched BEFORE the old
            // graph is destroyed and the shared allocation state is released, so the
            // constraint is enforced here rather than assumed from synchronous dispatch.
            st.sync_pending();
        }
        slot.built = false;
        slot.res->reset();
        ggml_backend_sched_reset(slot.sched.get());
        ggml_backend_sched_set_eval_callback(slot.sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        try {
            gf = llama_prefab_compose(model, gparams, n_env, st, prefab_stats);
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("%s: factor-prefab compose failed: %s\n", __func__, e.what());
            gf = nullptr;
        }
        if (!gf) {
            ret = GGML_STATUS_FAILED;
            return true;
        }
        if (!ggml_backend_sched_alloc_graph(slot.sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: factor-prefab: failed to allocate envelope N=%u\n", __func__, n_env);
            ret = GGML_STATUS_ALLOC_FAILED;
            return true;
        }
        llama_prefab_stat_tick(prefab_stats.n_envelope_allocs);
        slot.built    = true;
        slot.key      = key;
        slot.n_kv_pad = n_kv_pad_cur;
    }

    if (llama_prefab_nkv_obs_enabled()) {
        // env: LLAMA_PREFAB_NKV_OBS=1 - one stderr line per prefab ubatch (parsed by
        // scripts/nkv-gate/analyze.py); a single bool check per ubatch when off.
        fprintf(stderr, "prefab nkv: n_kv=%u n_kv_pad=%u decision=%s\n",
                kvc->get_n_kv(), n_kv_pad_cur, gf_reused ? "reuse" : "build");
    }

    llama_prefab_stat_tick(prefab_stats.n_ubatch_prefab);
    if (ubatch.n_tokens < n_env) {
        llama_prefab_stat_tick(prefab_stats.n_sub_envelope);
        llama_prefab_stat_add(prefab_stats.n_pad_rows, n_env - ubatch.n_tokens);
    }
    if (llama_prefab_stats_enabled()) {
        prefab_stats.last_n_env       = n_env;
        prefab_stats.last_n_tokens    = ubatch.n_tokens;
        prefab_stats.last_graph_nodes = ggml_graph_n_nodes(gf);
        prefab_stats.last_inp_tokens_ne0 = slot.res->get_inp_tokens() ? slot.res->get_inp_tokens()->ne[0] : 0;
    }

    // inputs take the REAL ubatch
    slot.res->set_inputs(&ubatch);

    st.active = slot.sched.get();

    const auto status = graph_compute(gf, ubatch.n_tokens > 1, slot.sched.get());
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute envelope graph, compute status: %d\n", __func__, status);
        ret = status;
        return true;
    }

    ret = GGML_STATUS_SUCCESS;
    out = slot.res.get();
    return true;
}

bool llama_prefab_is_enabled(const llama_context * ctx) {
    return ctx != nullptr && ctx->get_prefab() != nullptr;
}

void llama_prefab_get_stats(const llama_context * ctx, llama_prefab_stats * out) {
    *out = ctx->get_prefab_stats();
}

void llama_prefab_reset_stats(llama_context * ctx) {
    ctx->reset_prefab_stats();
}

size_t llama_prefab_n_cached_factors(const llama_context * ctx) {
    const auto * st = ctx->get_prefab();
    return st ? st->cache.modules.size() : 0;
}
