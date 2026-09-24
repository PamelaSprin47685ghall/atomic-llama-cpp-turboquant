
// W2-CPU regression tests for the MTP verification path (common/sampling.cpp).
//
//   1. greedy argmax / tie-break parity: common_sampler_sample_greedy_row vs the
//      reference greedy chain (and temp(0)+dist, this repo's real greedy CPU shape).
//   2. pure-greedy fast path equivalence: common_sampler_sample_and_accept_n vs the
//      verbatim pre-change per-row loop, on the same logits and drafts (full accept
//      including the bonus row, mid-prefix rejection, first-row rejection).
//   3. common_sampler_clone cur_p aliasing (F11): after the original keeps sampling,
//      the clone's candidate view must stay the clone-time snapshot -- this assertion
//      fails (red) against the pre-fix clone.
//   4. common_sampler_is_pure_greedy_params boundary table (audit F12): every
//      perturbation must reject; the canonical greedy config must accept.
//   5. common_sampler_clone_checkpoint: drops cur/cur_p scratch, and still samples
//      identically to a full clone and to a fresh sampler.
//
// Fixture: models/ggml-vocab-*.gguf are vocabulary-only files (no weights) and cannot
// produce logits, so this test synthesizes a tiny LLM_ARCH_LLAMA model in memory --
// same machinery as tests/test-llama-archs.cpp (llama_model_saver + gguf_init_empty +
// llama_model_init_from_user with deterministic filler weights) -- and then decodes a
// small batch for real per-row logits. No model download, no network, reproducible.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "sampling.h"
#include "common.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

static llama_model * g_model = nullptr;
static const llama_vocab  * g_vocab = nullptr;
static int32_t              g_n_vocab = 0;

struct sampler_free { void operator()(common_sampler * s) const { common_sampler_free(s); } };
struct ctx_free     { void operator()(llama_context * c) const { llama_free(c); } };

using sampler_u = std::unique_ptr<common_sampler, sampler_free>;
using ctx_u     = std::unique_ptr<llama_context, ctx_free>;

// ---------------------------------------------------------------------------
// synthesized tiny model (adapted verbatim from tests/test-llama-archs.cpp)
// ---------------------------------------------------------------------------

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static bool silent_model_load_progress(float /*progress*/, void * /*userdata*/) {
    return true;
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 128;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_GEMMA4) {
        n_embd = 128;
        n_head = 2;
        n_ff   = 192;
        n_layer = 5; // need at least 5 for swa_pattern (every 5th is full_attention)
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_MISTRAL4) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    } else if (arch == LLM_ARCH_QWEN4EXP) {
        n_embd  = 128;
        n_head  = 2;
        n_ff    = 192;
        n_layer = 4; // need interval 2 so we have full_attention and linear layers
    }

    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_QWEN4EXP) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, uint32_t(2));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,      uint32_t(4));
        // Interval 2: linear layers 0/2, full attention layers 1/3; enable QSA at 3.
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,    std::vector<uint32_t>({0, 0, 0, 2}));
        ms.add_kv(LLM_KV_SSM_CONV_KERNEL,              uint32_t(4));
        ms.add_kv(LLM_KV_SSM_INNER_SIZE,               uint32_t(128));
        ms.add_kv(LLM_KV_SSM_STATE_SIZE,               uint32_t(32));
        ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,           uint32_t(2));
        ms.add_kv(LLM_KV_SSM_GROUP_COUNT,              uint32_t(2));

        // PLE convolution history belongs to a recurrent layer.
        ms.add_kv(LLM_KV_PLE_LAYERS,                   std::vector<uint32_t>({0}));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,               uint32_t(2));
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,          uint32_t(2));
        ms.add_kv(LLM_KV_PLE_CONV_KERNEL,              uint32_t(4));
        ms.add_kv(LLM_KV_PLE_EOS_TOKEN_ID,             uint32_t(0));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,   n_embd / 2);
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,        std::vector<uint64_t>({1234567ULL, 7654321ULL}));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,             std::vector<uint64_t>({0, 32}));
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,         std::vector<uint64_t>({32, 32}));
    }

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_MISTRAL4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
    } else if (arch == LLM_ARCH_MINIMAX_M3) {
        // partial rotary: n_rot must not exceed the indexer key length (64)
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(5));
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   arch == LLM_ARCH_MINIMAX_M3 ? n_head : uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,   uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "no_vocab");
    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         uint32_t(2)); // sigmoid
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,       1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,        false);
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         uint32_t(3));

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

// canonical pure-greedy sampling params (server greedy: temp <= 0, filters disabled)
static common_params_sampling pure_greedy_params() {
    common_params_sampling p;
    p.no_perf = true;
    // Fixed seed for every sampler this factory builds. The default
    // common_params_sampling::seed is LLAMA_DEFAULT_SEED, and get_rng_seed falls back
    // to std::random_device() in that case (src/llama-sampler.cpp): two samplers
    // initialized without it then draw from independent random streams. Selection is
    // unaffected under pure greedy (argmax), but any group that compares two
    // independently initialized samplers while a stochastic sampler participates --
    // the temp=0.8 fallback group below is exactly that -- would diverge by stream
    // alone. Fixing it at the factory keeps the whole file deterministic.
    p.seed     = 42;
    p.temp     = 0.0f; // <= 0: greedy
    p.top_k    = 0;    // <= 0: disabled
    p.top_p    = 1.0f; // disabled
    p.min_p    = 0.0f; // disabled
    return p;
}

// verbatim copy of the pre-W2-CPU per-row loop of common_sampler_sample_and_accept_n
// (common/sampling.cpp). The fast path must match this reference exactly.
static std::vector<llama_token> sample_and_accept_loop_reference(
        common_sampler * gsmpl, llama_context * ctx, const std::vector<int> & idxs,
        const std::vector<llama_token> & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        if (draft[i] != id) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

// reference 1: the explicit greedy sampler (llama_sampler_greedy_apply)
static llama_token reference_greedy_chain(const std::vector<float> & logits) {
    std::vector<llama_token_data> cur;
    cur.reserve(logits.size());
    for (llama_token id = 0; id < (llama_token) logits.size(); id++) {
        cur.push_back(llama_token_data{ id, logits[id], 0.0f });
    }
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };

    llama_sampler * greedy = llama_sampler_init_greedy();
    llama_sampler_apply(greedy, &cur_p);
    const llama_token id = cur_p.data[cur_p.selected].id;
    llama_sampler_free(greedy);

    return id;
}

// reference 2: this repo's real greedy CPU shape -- temp(<=0) collapse + dist
static llama_token reference_temp_zero_chain(const std::vector<float> & logits) {
    std::vector<llama_token_data> cur;
    cur.reserve(logits.size());
    for (llama_token id = 0; id < (llama_token) logits.size(); id++) {
        cur.push_back(llama_token_data{ id, logits[id], 0.0f });
    }
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };

    llama_sampler * t = llama_sampler_init_temp_ext(0.0f, 0.0f, 1.0f);
    llama_sampler_apply(t, &cur_p);
    llama_sampler_free(t);

    llama_sampler * d = llama_sampler_init_dist(0);
    llama_sampler_apply(d, &cur_p);
    const llama_token id = cur_p.data[cur_p.selected].id;
    llama_sampler_free(d);

    return id;
}

static void test_greedy_argmax_parity() {
    const std::vector<std::vector<float>> cases = {
        { 1.0f, 2.0f, 3.0f, 4.0f },      // max at the end
        { 4.0f, 3.0f, 2.0f, 1.0f },      // max at the front
        { 5.0f, 5.0f, 1.0f, 5.0f },      // tie -> smallest id (0)
        { 2.0f, 2.0f, 2.0f, 2.0f },      // full tie -> 0
        { -3.0f, -3.0f, -1.0f },         // negative tie -> smaller id
        { -INFINITY, 0.0f, 1.0f },       // minus-inf entries
        { NAN, 1.0f, 1.0f },             // NaN: '>' is false everywhere -> index 0
    };

    for (const auto & logits : cases) {
        const llama_token expect   = reference_greedy_chain(logits);
        const llama_token got      = common_sampler_sample_greedy_row(logits.data(), (int32_t) logits.size());

        printf("  greedy parity: case ref=%d got=%d", (int) expect, (int) got);
        if (!std::isnan(logits[0])) {
            const llama_token expect_t0 = reference_temp_zero_chain(logits);
            printf(" temp0=%d", (int) expect_t0);
            GGML_ASSERT(got == expect_t0);
        }
        printf("\n");

        GGML_ASSERT(got == expect);
    }

    printf("greedy argmax / tie-break parity OK\n");
}

static void test_pure_greedy_boundary() {
    const common_params_sampling base = pure_greedy_params();

    GGML_ASSERT(common_sampler_is_pure_greedy_params(base, false, false));
    GGML_ASSERT(!common_sampler_is_pure_greedy_params(base, true, false));  // grmr present
    GGML_ASSERT(!common_sampler_is_pure_greedy_params(base, false, true));  // rbudget present

    struct perturb { const char * name; common_params_sampling p; };
    const std::vector<perturb> cases = {
        { "temp>0",           []{ common_params_sampling p = pure_greedy_params(); p.temp = 0.8f;                 return p; }() },
        { "dynatemp",         []{ common_params_sampling p = pure_greedy_params(); p.dynatemp_range = 0.5f;        return p; }() },
        { "top_k",            []{ common_params_sampling p = pure_greedy_params(); p.top_k = 40;                   return p; }() },
        { "top_p",            []{ common_params_sampling p = pure_greedy_params(); p.top_p = 0.95f;                return p; }() },
        { "min_p",            []{ common_params_sampling p = pure_greedy_params(); p.min_p = 0.05f;                return p; }() },
        { "typical",          []{ common_params_sampling p = pure_greedy_params(); p.typ_p = 0.5f;                 return p; }() },
        { "top_n_sigma",      []{ common_params_sampling p = pure_greedy_params(); p.top_n_sigma = 0.5f;           return p; }() },
        { "xtc",              []{ common_params_sampling p = pure_greedy_params(); p.xtc_probability = 0.5f;       return p; }() },
        { "dry",              []{ common_params_sampling p = pure_greedy_params(); p.dry_multiplier = 0.5f;        return p; }() },
        { "mirostat",         []{ common_params_sampling p = pure_greedy_params(); p.mirostat = 1;                 return p; }() },
        { "penalty_repeat",   []{ common_params_sampling p = pure_greedy_params(); p.penalty_repeat = 1.1f;       return p; }() },
        { "penalty_freq",     []{ common_params_sampling p = pure_greedy_params(); p.penalty_freq = 0.1f;         return p; }() },
        { "penalty_present",  []{ common_params_sampling p = pure_greedy_params(); p.penalty_present = 0.1f;      return p; }() },
        { "adaptive_p",       []{ common_params_sampling p = pure_greedy_params(); p.adaptive_target = 0.5f;      return p; }() },
        { "backend_sampling", []{ common_params_sampling p = pure_greedy_params(); p.backend_sampling = true;     return p; }() },
        { "logit_bias",       []{ common_params_sampling p = pure_greedy_params(); p.logit_bias = { { 1, -1.0f } }; return p; }() },
        { "infill sampler",   []{ common_params_sampling p = pure_greedy_params(); p.samplers = { COMMON_SAMPLER_TYPE_INFILL };     return p; }() },
        { "adaptive type",    []{ common_params_sampling p = pure_greedy_params(); p.samplers = { COMMON_SAMPLER_TYPE_ADAPTIVE_P }; return p; }() },
    };

    for (const auto & c : cases) {
        const bool got = common_sampler_is_pure_greedy_params(c.p, false, false);
        printf("  boundary: %-16s pure_greedy=%d\n", c.name, (int) got);
        GGML_ASSERT(!got);
    }

    // defaults of the penalty triple must stay eligible
    common_params_sampling with_defaults = pure_greedy_params();
    with_defaults.penalty_repeat  = 1.0f;
    with_defaults.penalty_freq    = 0.0f;
    with_defaults.penalty_present = 0.0f;
    GGML_ASSERT(common_sampler_is_pure_greedy_params(with_defaults, false, false));

    printf("pure-greedy boundary table OK\n");
}

static ctx_u make_ctx(llama_model * model, int32_t n_ctx) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = n_ctx;
    cparams.n_batch   = n_ctx;
    cparams.n_seq_max = 1;

    return ctx_u(llama_init_from_model(model, cparams));
}

static std::vector<llama_token> make_tokens(int32_t n) {
    return get_tokens((uint32_t) n, (uint32_t) g_n_vocab, 1234);
}

// decode one N-token batch so rows 0..N-1 all expose real logits.
//
// Do NOT use llama_batch_get_one() here: it hardcodes logits = nullptr
// (src/llama-batch.cpp), so llama_decode produces an output row for the LAST token
// only -- probing rows 0..N-2 then returns nullptr from llama_get_logits_ith and
// trips the GGML_ASSERT(logits != nullptr) guard inside common_sampler::set_logits
// (a product-side guard with no relation to the fast path under test). Build the
// batch explicitly with logits[i] = 1 on every row instead: that is the same
// multi-row logits shape the MTP verify path itself decodes in the server.
// Batch field indexing follows the tests/test-native-determinism.cpp idiom.
static void decode_rows(ctx_u & ctx, int32_t n) {
    const std::vector<llama_token> toks = make_tokens(n);

    llama_batch batch = llama_batch_init(n, /* embd = */ 0, /* n_seq_max = */ 1);
    GGML_ASSERT(batch.token != nullptr && batch.logits != nullptr);

    for (int32_t i = 0; i < n; i++) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }
    batch.n_tokens = n;

    GGML_ASSERT(llama_decode(ctx.get(), batch) == 0);

    llama_batch_free(batch);
    llama_synchronize(ctx.get());
}

static void test_fast_path_equivalence() {
    auto ctx = make_ctx(g_model, 128);

    const int32_t N = 6;
    decode_rows(ctx, N);

    std::vector<int> idxs(N);
    for (int32_t i = 0; i < N; i++) {
        idxs[i] = i;
    }

    // probe (sample-only, no accept): in pure greedy the per-row ids cannot depend on
    // accept order, so these are the expected tokens for every scenario below
    common_params_sampling sp = pure_greedy_params();
    sampler_u probe(common_sampler_init(g_model, sp));
    std::vector<llama_token> row_ids(N);
    for (int32_t i = 0; i < N; i++) {
        row_ids[i] = common_sampler_sample(probe.get(), ctx.get(), idxs[i]);
    }

    // (a) full accept: every draft row matches -> N tokens incl. the bonus row
    {
        const std::vector<llama_token> draft(row_ids.begin(), row_ids.end() - 1);

        common_params_sampling sp = pure_greedy_params();
        sampler_u a(common_sampler_init(g_model, sp));
        sampler_u b(common_sampler_init(g_model, sp));

        const auto r_fast = common_sampler_sample_and_accept_n(a.get(), ctx.get(), idxs, draft, false);
        const auto r_ref  = sample_and_accept_loop_reference(b.get(), ctx.get(), idxs, draft, false);

        GGML_ASSERT(r_fast.size() == (size_t) N);
        GGML_ASSERT(r_fast == r_ref);
        GGML_ASSERT(r_fast == row_ids);
        GGML_ASSERT(common_sampler_get_token_history(a.get()) == common_sampler_get_token_history(b.get()));

        printf("  fast path (full accept + bonus) matches reference loop\n");
    }

    // (b) mid-prefix rejection at position 2
    {
        std::vector<llama_token> draft(row_ids.begin(), row_ids.end() - 1);
        draft[2] = (llama_token) ((row_ids[2] + 1) % (llama_token) g_n_vocab);
        GGML_ASSERT(draft[2] != row_ids[2]);

        common_params_sampling sp = pure_greedy_params();
        sampler_u a(common_sampler_init(g_model, sp));
        sampler_u b(common_sampler_init(g_model, sp));

        const auto r_fast = common_sampler_sample_and_accept_n(a.get(), ctx.get(), idxs, draft, false);
        const auto r_ref  = sample_and_accept_loop_reference(b.get(), ctx.get(), idxs, draft, false);
        const std::vector<llama_token> expect(row_ids.begin(), row_ids.begin() + 3);

        GGML_ASSERT(r_fast == r_ref);
        GGML_ASSERT(r_fast == expect);

        printf("  fast path (rejection at row 2) matches reference loop\n");
    }

    // (c) first-row rejection -> single token
    {
        std::vector<llama_token> draft(row_ids.begin(), row_ids.end() - 1);
        draft[0] = (llama_token) ((row_ids[0] + 1) % (llama_token) g_n_vocab);
        GGML_ASSERT(draft[0] != row_ids[0]);

        common_params_sampling sp = pure_greedy_params();
        sampler_u a(common_sampler_init(g_model, sp));
        sampler_u b(common_sampler_init(g_model, sp));

        const auto r_fast = common_sampler_sample_and_accept_n(a.get(), ctx.get(), idxs, draft, false);
        const auto r_ref  = sample_and_accept_loop_reference(b.get(), ctx.get(), idxs, draft, false);

        GGML_ASSERT(r_fast.size() == 1);
        GGML_ASSERT(r_fast == r_ref);
        GGML_ASSERT(r_fast[0] == row_ids[0]);

        printf("  fast path (rejection at row 0) matches reference loop\n");
    }

    // (d) a non-greedy sampler must keep taking the original loop (fallback path).
    // Both samplers are built from pure_greedy_params(), so they share its fixed seed:
    // with temp>0 the chain's dist sampler draws from the RNG, and the two streams must
    // be identical for this comparison to mean anything (regression: the default seed
    // made them independent random_device streams and this group failed by stream alone).
    {
        common_params_sampling sp = pure_greedy_params();
        sp.temp  = 0.8f;
        sp.top_p = 0.95f;

        sampler_u a(common_sampler_init(g_model, sp));
        sampler_u b(common_sampler_init(g_model, sp));

        std::vector<llama_token> draft(row_ids.begin(), row_ids.end() - 1);

        const auto r_fast = common_sampler_sample_and_accept_n(a.get(), ctx.get(), idxs, draft, false);
        const auto r_ref  = sample_and_accept_loop_reference(b.get(), ctx.get(), idxs, draft, false);

        GGML_ASSERT(r_fast == r_ref);
        GGML_ASSERT(common_sampler_get_token_history(a.get()) == common_sampler_get_token_history(b.get()));

        printf("  non-greedy sampler still matches reference loop (fallback)\n");
    }

    printf("pure-greedy fast path equivalence OK\n");
}

static void test_clone_candidates_independent() {
    auto ctx = make_ctx(g_model, 128);

    const int32_t N = 6;
    decode_rows(ctx, N);

    common_params_sampling sp = pure_greedy_params();
    sampler_u a(common_sampler_init(g_model, sp));

    const llama_token id0 = common_sampler_sample(a.get(), ctx.get(), 0);
    GGML_ASSERT(id0 != LLAMA_TOKEN_NULL);

    auto * cands = common_sampler_get_candidates(a.get(), false);
    GGML_ASSERT(cands != nullptr && cands->size == (size_t) g_n_vocab);
    const std::vector<llama_token_data> snap(cands->data, cands->data + cands->size);

    common_sampler * b = common_sampler_clone(a.get());
    GGML_ASSERT(b != nullptr);

    // the clone's copied candidate array must equal the original at clone time
    auto * vb = common_sampler_get_candidates(b, false);
    GGML_ASSERT(vb != nullptr && vb->size == snap.size());
    for (size_t i = 0; i < snap.size(); i++) {
        GGML_ASSERT(vb->data[i].id    == snap[i].id);
        GGML_ASSERT(vb->data[i].logit == snap[i].logit);
        GGML_ASSERT(vb->data[i].p     == snap[i].p);
    }

    // keep sampling the ORIGINAL. With the pre-fix clone, b's candidate view aliases a's
    // cur buffer, so the view below drifts (this assertion is the F11 regression);
    // with the fix, b's view must remain the clone-time snapshot.
    bool observed_change = false;
    for (int32_t k = 1; k < N; k++) {
        llama_synchronize(ctx.get());
        (void) common_sampler_sample(a.get(), ctx.get(), k);

        auto * va = common_sampler_get_candidates(a.get(), false);
        for (size_t i = 0; i < snap.size(); i++) {
            if (va->data[i].logit != snap[i].logit) {
                observed_change = true;
                break;
            }
        }
    }

    auto * vb2 = common_sampler_get_candidates(b, false);
    GGML_ASSERT(vb2 != nullptr && vb2->size == snap.size());
    for (size_t i = 0; i < snap.size(); i++) {
        GGML_ASSERT(vb2->data[i].id    == snap[i].id);
        GGML_ASSERT(vb2->data[i].logit == snap[i].logit);
    }

    // guard against a vacuous test: the rows must actually have produced different
    // candidate values, otherwise the alias could not have been observed
    GGML_ASSERT(observed_change && "rows produced identical candidates; the alias test is vacuous");

    // the clone still evolves independently and identically on the same row
    llama_synchronize(ctx.get());
    const llama_token u0 = common_sampler_sample(b, ctx.get(), 0);
    GGML_ASSERT(u0 == id0);

    common_sampler_free(b);

    printf("clone cur_p independence OK\n");
}

static void test_checkpoint_clone() {
    auto ctx = make_ctx(g_model, 128);

    const int32_t N = 6;
    decode_rows(ctx, N);

    const std::vector<int> idxs = { 0, 1 }; // idxs.size() must be draft.size() + 1

    // materialize scratch on the original, then take a checkpoint clone
    common_params_sampling sp = pure_greedy_params();
    sampler_u a(common_sampler_init(g_model, sp));
    (void) common_sampler_sample(a.get(), ctx.get(), 0);
    GGML_ASSERT(common_sampler_get_candidates(a.get(), false)->size == (size_t) g_n_vocab);

    common_sampler * c = common_sampler_clone_checkpoint(a.get());
    GGML_ASSERT(c != nullptr);

    // the checkpoint deliberately drops the candidate scratch
    auto * vc0 = common_sampler_get_candidates(c, false);
    GGML_ASSERT(vc0 == nullptr || vc0->size == 0);

    // sampling on the checkpoint clone rebuilds cur/cur_p via set_logits and matches a
    // fresh sampler and a full clone on the same row
    common_params_sampling sp_fresh = pure_greedy_params();
    sampler_u fresh(common_sampler_init(g_model, sp_fresh));
    common_sampler * full = common_sampler_clone(a.get());

    llama_synchronize(ctx.get());
    const llama_token s0 = common_sampler_sample(c, ctx.get(), 0);

    llama_synchronize(ctx.get());
    const llama_token f0 = common_sampler_sample(fresh.get(), ctx.get(), 0);

    llama_synchronize(ctx.get());
    const llama_token l0 = common_sampler_sample(full, ctx.get(), 0);

    GGML_ASSERT(s0 == f0);
    GGML_ASSERT(s0 == l0);

    auto * vc = common_sampler_get_candidates(c, false);
    GGML_ASSERT(vc != nullptr && vc->size == (size_t) g_n_vocab);

    // full accept_n on the checkpoint clone matches the reference loop
    {
        common_params_sampling sp = pure_greedy_params();
        sampler_u ref(common_sampler_init(g_model, sp));
        const std::vector<llama_token> draft = { f0 };

        const auto r_ckpt = common_sampler_sample_and_accept_n(c, ctx.get(), idxs, draft, false);
        const auto r_ref  = sample_and_accept_loop_reference(ref.get(), ctx.get(), idxs, draft, false);

        GGML_ASSERT(r_ckpt == r_ref);
    }

    common_sampler_free(full);
    common_sampler_free(c);

    printf("checkpoint clone OK\n");
}

int main(int argc, char ** argv) {
    GGML_UNUSED(argc);
    GGML_UNUSED(argv);

    ggml_time_init();
    llama_backend_init();

    // synthetic tiny LLM model: vocab-only GGUFs in models/ have no weights and cannot
    // produce logits, so build one in memory (same fixture approach as test-llama-archs)
    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_LLAMA, false);

    llama_model_params mparams = llama_model_default_params();
    mparams.progress_callback = silent_model_load_progress;

    size_t seed = 1234;
    llama_model * model = llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &seed, mparams);
    GGML_ASSERT(model != nullptr);

    g_model   = model;
    g_vocab   = llama_model_get_vocab(model);
    g_n_vocab = llama_vocab_n_tokens(g_vocab);
    GGML_ASSERT(g_n_vocab > 4);

    test_greedy_argmax_parity();
    test_pure_greedy_boundary();
    test_fast_path_equivalence();
    test_clone_candidates_independent();
    test_checkpoint_clone();

    llama_model_free(model);
    llama_backend_free();

    printf("OK\n");

    return 0;
}
