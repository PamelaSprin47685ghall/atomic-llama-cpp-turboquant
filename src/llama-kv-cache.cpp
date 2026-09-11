#include "llama-kv-cache.h"
#include "llama-triattention.h"
#include "llama-kv-transform.h"
#include "llama-turbo-config.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-xkv-tri.h"
#include "llama-xkv-seal.h"
#include "llama-xkv-backend.h"
#include "llama-xkv-runtime.h"
#include "llama-xkv-transaction.h"
#include "llama-xkv-state.h"

#include <unordered_map>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>

static bool ggml_is_power_of_2(int n) {
    return (n & (n - 1)) == 0;
}

// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];

    assert(ggml_is_power_of_2(n));
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);

    std::vector<float> data_f32;

    float * data = (float *) tensor->data;

    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }

    data[0*n + 0] = 1.0 / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];

                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}

// NOTE: the Hadamard rotation helper (ggml_mul_mat_aux) moved to llama-impl.h
// upstream; the copy that used to live here is gone. The fork's
// GGML_HINT_SRC0_IS_HADAMARD hint is applied in the llama-impl.h version.

// InnerQ: cross-TU shared state for CUDA per-channel equalization.
// These are defined in ggml-cuda/turbo-innerq.cu (when CUDA is enabled).
// When CUDA is not available, we provide stub implementations.
#ifndef INNERQ_MAX_CHANNELS
#define INNERQ_MAX_CHANNELS 128
#endif

#ifdef GGML_USE_CUDA
#if defined(_WIN32) && !defined(__MINGW32__)
#  define TURBO_IQ_IMPORT __declspec(dllimport)
#else
#  define TURBO_IQ_IMPORT
#endif
extern TURBO_IQ_IMPORT bool  g_innerq_finalized;
extern TURBO_IQ_IMPORT float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS];
TURBO_IQ_IMPORT bool turbo_innerq_needs_tensor_update(void);
TURBO_IQ_IMPORT void turbo_innerq_mark_tensor_updated(void);
#else
[[maybe_unused]] static bool  g_innerq_finalized = false;
[[maybe_unused]] static float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS] = {};
[[maybe_unused]] static bool turbo_innerq_needs_tensor_update(void) { return false; }
[[maybe_unused]] static void turbo_innerq_mark_tensor_updated(void) {}
#endif

//
// llama_kv_cache
//

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
        const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share,
    const llama_cparams   * cparams) :
    model(model), hparams(hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max),
    kv_unified(unified),
    n_pad(n_pad), n_swa(n_swa), swa_type(swa_type),
    other(static_cast<llama_kv_cache *>(mem_other)),
    v_cells_impl(other ? other->v_cells_impl : std::make_shared<llama_kv_cells_vec>()),
    v_cells(*v_cells_impl),
    xkv_store(other ? other->xkv_store : nullptr),
    xkv_hot_pool(other ? other->xkv_hot_pool : nullptr) {

    if (cparams && other == nullptr) {
        if (cparams->xkv_mode == LLAMA_XKV_MODE_DENSE || cparams->xkv_mode == LLAMA_XKV_MODE_SR) {
            // MTP draft contexts must never allocate hot resources: the draft
            // cache holds only the speculative tail and shares no hot lifecycle.
            // Non-unified caches are rejected: bounded-hot tensor rows and the
            // shared pool assume a single unified stream.
            if (cparams->ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
                throw std::invalid_argument("llama_kv_cache: bounded-hot XKV is not supported in MTP draft contexts");
            }
            if (!unified) {
                throw std::invalid_argument("llama_kv_cache: bounded-hot XKV requires a unified KV cache");
            }
            xkv_dense_sr = true;
            const uint64_t seg = (uint64_t) cparams->xkv_segment_tokens;
            const uint64_t ub  = (uint64_t) std::max<uint32_t>(cparams->n_batch, cparams->n_ubatch);
            // Fixed config forbids hidden fallbacks: chunk_tokens == 0 throws
            // instead of silently substituting 8.
            if (cparams->xkv_chunk_tokens == 0) {
                throw std::invalid_argument("llama_kv_cache: XKV chunk_tokens must be positive");
            }
            const uint64_t chunk = (uint64_t) cparams->xkv_chunk_tokens;
            if (seg == 0) {
                throw std::invalid_argument("llama_kv_cache: XKV segment_tokens must be positive");
            }
            // Explicit checked add (u32 + u32 always fits, stated for audit).
            const uint64_t raw_hot = seg + ub;
            if (raw_hot == 0 || raw_hot > (uint64_t) UINT32_MAX) {
                throw std::invalid_argument("llama_kv_cache: XKV hot capacity derivation overflow");
            }
            if (raw_hot > UINT32_MAX - chunk) {
                throw std::invalid_argument("llama_kv_cache: XKV hot capacity rounding overflow");
            }
            const uint64_t rounded = ((raw_hot + chunk - 1) / chunk) * chunk;
            if (rounded == 0 || rounded > UINT32_MAX) {
                throw std::invalid_argument("llama_kv_cache: rounded hot capacity overflow");
            }
            // Bounded mode needs room to seal safely: the logical cache must
            // fit a full segment plus the batch headroom. Silently clamping
            // hot below segment+batch would leave no sealable window, so the
            // configuration is rejected instead of clamped.
            if (rounded > (uint64_t) kv_size) {
                throw std::invalid_argument("llama_kv_cache: logical KV capacity below XKV segment+batch hot requirement");
            }
            hot_kv_size = (uint32_t) rounded;
            if (hot_kv_size == 0) {
                throw std::invalid_argument("llama_kv_cache: derived hot capacity is invalid");
            }
            xkv_hot_pool = std::make_shared<llama_xkv::xkv_hot_slot_pool>(hot_kv_size);
        }
    } else if (other) {
        xkv_dense_sr = other->xkv_dense_sr;
        hot_kv_size  = other->hot_kv_size;
    }
    if (hot_kv_size == 0) {
        hot_kv_size = kv_size;
    }

    // shared cells view the source cache's K/V tensors, so the cell count
    // follows the source allocation: a fitted target can be smaller than the
    // draft default and oversized views would overflow the source tensors
    if (other) {
        const uint32_t size_other = other->get_size();
        if (kv_size != size_other) {
            LLAMA_LOG_WARN("%s: kv_size = %u overridden to %u to match the shared source cache\n", __func__, kv_size, size_other);
            kv_size = size_other;
        }
    }

    GGML_ASSERT(kv_size % n_pad == 0);

    // Auto-asymmetric: when symmetric turbo K+V is requested and the model has
    // high GQA ratio (few KV heads serving many Q heads), upgrade K to q8_0.
    // Turbo K quantization error gets amplified by the GQA broadcast factor.
    // Qwen2.5: 4 KV heads / 28 Q heads = 7:1 → turbo3 K PPL catastrophic (2887 vs 7.4 baseline)
    // Mistral:  8 KV heads / 32 Q heads = 4:1 → turbo3 K works fine (+4.4% PPL)
    // Threshold: GQA ratio >= 6 triggers auto-asymmetric.
    {
        const bool k_is_turbo = (type_k == GGML_TYPE_TURBO3_0 || type_k == GGML_TYPE_TURBO4_0 || type_k == GGML_TYPE_TURBO2_0);
        if (k_is_turbo) {
            const uint32_t n_head    = hparams.n_head(0);
            const uint32_t n_head_kv = hparams.n_head_kv(0);
            const uint32_t gqa_ratio = (n_head_kv > 0) ? n_head / n_head_kv : 1;

            const char * env = getenv("TURBO_AUTO_ASYMMETRIC");
            const bool disabled = (env && env[0] == '0');

            if (!disabled && gqa_ratio >= 6 && type_k == type_v) {
                LLAMA_LOG_WARN("%s: auto-asymmetric: GQA ratio %u:1 (n_head=%u, n_head_kv=%u) — "
                               "upgrading K from %s to q8_0 to prevent quality degradation. "
                               "Disable with TURBO_AUTO_ASYMMETRIC=0\n",
                               __func__, gqa_ratio, n_head, n_head_kv, ggml_type_name(type_k));
                type_k = GGML_TYPE_Q8_0;
            }
        }
    }

    // #24060/MTP fix: iterate ALL layers (incl. nextn) so an all-nextn draft
    // (gemma4-assistant: n_layer()==0) registers its KV layers; has_kv() still
    // gates per-layer. Upstream loops the full hparams.n_layer member here.
    const uint32_t n_layer    = hparams.n_layer_all;
    const uint32_t n_layer_kv = hparams.n_layer_kv();

    // Resolve once per cache instance, not once per process. In particular,
    // a turbo2-V cache may auto-select mode 7 while a later turbo3-V cache in
    // the same process must remain turbo3 unless the user explicitly asks for
    // an adaptive mode.
    const uint32_t n_layer_adaptive = hparams.n_layer();
    const char * adaptive_env = getenv("TURBO_LAYER_ADAPTIVE");
    const int adaptive_mode = llama_turbo_layer_adaptive_mode(type_v, n_layer_adaptive, adaptive_env);
    if (adaptive_env != nullptr) {
        if (adaptive_mode > 0) {
            LLAMA_LOG_INFO("llama_kv_cache: layer-adaptive mode %d enabled (env)\n", adaptive_mode);
        }
    } else if (adaptive_mode == 7) {
        LLAMA_LOG_INFO("llama_kv_cache: Boundary V auto-enabled for turbo2-V (opt-out: TURBO_LAYER_ADAPTIVE=0)\n");
    }

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                // +3 for turbo rotation matrices (turbo_rotation + turbo_rotation_inv + turbo_innerq_scale_inv)
                /*.mem_size   =*/ size_t((2u*(1 + n_stream)*n_layer_kv + 3)*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    rerot_write_tags.resize(LLAMA_MAX_SEQ);
    rerot_reader_views.resize(LLAMA_MAX_SEQ);

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    const bool is_mla = hparams.is_mla();

    for (uint32_t il = 0; il < n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        if (share && other) {
            const int32_t il_share = share(il);

            if (il_share >= 0) {
                const auto & layer_share = other->layers[other->map_layer_ids[il_share]];

                LLAMA_LOG_WARN("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", __func__, il, il_share,
                        layer_share.k->data, layer_share.v->data);

                map_layer_ids[il] = layers.size();

                layers.push_back(layer_share);
                layers.back().il = il;

                if (xkv_store) {
                    xkv_store->register_layer_alias(il, (uint32_t) il_share);
                }

                continue;
            }
        }

        if (n_embd_head_k_all == 0) {
            n_embd_head_k_all = (int32_t) hparams.n_embd_head_k(il);
        } else if (n_embd_head_k_all > 0 && n_embd_head_k_all != (int32_t) hparams.n_embd_head_k(il)) {
            n_embd_head_k_all = -1;
        }

        if (!is_mla) {
            if (n_embd_head_v_all == 0) {
                n_embd_head_v_all = (int32_t) hparams.n_embd_head_v(il);
            } else if (n_embd_head_v_all > 0 && n_embd_head_v_all != (int32_t) hparams.n_embd_head_v(il)) {
                n_embd_head_v_all = -1;
            }
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        // TurboQuant zero-padding: for models with non-128-aligned head_dim (e.g. DeepSeek
        // head_dim_k=192), pad each head to the next multiple of 128. The padded zeros don't
        // affect dot products since WHT preserves inner products:
        //   <WHT(Q_padded), WHT(K_padded)> = <Q_padded, K_padded> = <Q, K> + <0, 0> = <Q, K>
        const uint32_t n_embd_head_k = hparams.n_embd_head_k(il);


        const bool has_k = true;
        const bool has_v = !is_mla;

        // Layer-adaptive: use higher precision for quality-sensitive layers
        // Config: TURBO_LAYER_ADAPTIVE env var controls the strategy
        //   0 = uniform (explicit opt-out; unset + turbo2 V auto-selects mode 7)
        //   1 = q8_0 K+V for first+last 4 layers
        //   2 = q8_0 K+V for last 8 layers
        //   5 = Boundary V: first2+last2 V=turbo4, rest V=turbo2 (K unchanged)
        //   6 = V-only: last 8 V=turbo4, rest V=turbo2 (K unchanged)
        //   7 = Boundary V (recommended): first2+last2 V=q8_0, rest V=turbo2 (K unchanged)
        ggml_type layer_type_k = type_k;
        ggml_type layer_type_v = type_v;
        {
            const bool is_turbo = (type_k == GGML_TYPE_TURBO3_0 || type_k == GGML_TYPE_TURBO4_0 || type_k == GGML_TYPE_TURBO2_0);
            const bool v_is_turbo = (type_v == GGML_TYPE_TURBO3_0 || type_v == GGML_TYPE_TURBO4_0 || type_v == GGML_TYPE_TURBO2_0);
            if (adaptive_mode == 1 && is_turbo && n_layer_adaptive >= 8) {
                if (il < 4 || il >= n_layer_adaptive - 4) {
                    layer_type_k = GGML_TYPE_Q8_0;
                    layer_type_v = GGML_TYPE_Q8_0;
                }
            } else if (adaptive_mode == 2 && is_turbo && n_layer_adaptive >= 8) {
                if (il >= n_layer_adaptive - 8) {
                    layer_type_k = GGML_TYPE_Q8_0;
                    layer_type_v = GGML_TYPE_Q8_0;
                }
            } else if (adaptive_mode == 5 && v_is_turbo && n_layer_adaptive >= 8) {
                // Boundary V (turbo4 boundaries): first2+last2 V=turbo4, rest V=turbo2 (excluding MTP layers)
                const bool is_boundary = (il < 2 || (il < n_layer_adaptive && il >= n_layer_adaptive - 2));
                layer_type_v = is_boundary ? GGML_TYPE_TURBO4_0 : GGML_TYPE_TURBO2_0;
                if (il == 0) {
                    LLAMA_LOG_INFO("llama_kv_cache: Boundary V mode 5: first2+last2 V=turbo4, rest V=turbo2\n");
                }
            } else if (adaptive_mode == 6 && v_is_turbo && n_layer_adaptive >= 8) {
                // V-only: last 8 V=turbo4, rest V=turbo2
                layer_type_v = (il >= n_layer_adaptive - 8) ? GGML_TYPE_TURBO4_0 : GGML_TYPE_TURBO2_0;
                if (il == 0) {
                    LLAMA_LOG_INFO("llama_kv_cache: V-only LA mode 6: last8 V=turbo4, rest V=turbo2\n");
                }
            } else if (adaptive_mode == 7 && v_is_turbo && n_layer_adaptive >= 8) {
                // Boundary V (recommended): first2+last2 V=q8_0, rest V=turbo2 (excluding MTP layers)
                const bool is_boundary = (il < 2 || (il < n_layer_adaptive && il >= n_layer_adaptive - 2));
                layer_type_v = is_boundary ? GGML_TYPE_Q8_0 : GGML_TYPE_TURBO2_0;
                if (il == 0) {
                    LLAMA_LOG_INFO("llama_kv_cache: Boundary V mode 7: first2+last2 V=q8_0, rest V=turbo2\n");
                }
            }
        }
        // For turbo types, pad K head_dim to next multiple of 128 for full WHT groups
        uint32_t n_embd_k_gqa_eff = n_embd_k_gqa;
        const bool k_is_turbo = (layer_type_k == GGML_TYPE_TURBO3_0 || layer_type_k == GGML_TYPE_TURBO4_0 || layer_type_k == GGML_TYPE_TURBO2_0);
        if (k_is_turbo && n_embd_head_k % 128 != 0) {
            const uint32_t padded_head_k = ((n_embd_head_k + 127) / 128) * 128;
            const uint32_t n_head_kv = n_embd_k_gqa / n_embd_head_k;
            n_embd_k_gqa_eff = n_head_kv * padded_head_k;
            if (il == 0) {
                LLAMA_LOG_INFO("%s: turbo zero-padding K head_dim %u -> %u (cache %u -> %u)\n",
                               __func__, n_embd_head_k, padded_head_k, n_embd_k_gqa, n_embd_k_gqa_eff);
            }
        }

        // For turbo types, pad V head_dim to next multiple of 128 if needed
        const uint32_t n_embd_head_v = hparams.n_embd_head_v(il);
        uint32_t n_embd_v_gqa_eff = n_embd_v_gqa;
        const bool v_is_turbo = (layer_type_v == GGML_TYPE_TURBO3_0 || layer_type_v == GGML_TYPE_TURBO4_0 || layer_type_v == GGML_TYPE_TURBO2_0);
        if (v_is_turbo && !is_mla && n_embd_head_v % 128 != 0) {
            const uint32_t padded_head_v = ((n_embd_head_v + 127) / 128) * 128;
            const uint32_t n_head_kv = n_embd_v_gqa / n_embd_head_v;
            n_embd_v_gqa_eff = n_head_kv * padded_head_v;
            if (il == 0) {
                LLAMA_LOG_INFO("%s: turbo zero-padding V head_dim %u -> %u (cache %u -> %u)\n",
                               __func__, n_embd_head_v, padded_head_v, n_embd_v_gqa, n_embd_v_gqa_eff);
            }
        }

        ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, layer_type_k, n_embd_k_gqa_eff, hot_kv_size, n_stream) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, layer_type_v, n_embd_v_gqa_eff, hot_kv_size, n_stream) : nullptr;

        has_k && ggml_format_name(k, "cache_k_l%d", il);
        has_v && ggml_format_name(v, "cache_v_l%d", il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(has_k ? ggml_view_2d(ctx, k, n_embd_k_gqa_eff, hot_kv_size, k->nb[1], s*k->nb[2]) : nullptr);
            v_stream.push_back(has_v ? ggml_view_2d(ctx, v, n_embd_v_gqa_eff, hot_kv_size, v->nb[1], s*v->nb[2]) : nullptr);
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, k, v, k_stream, v_stream, });

        // TurboQuant: create rotation matrix tensors (once, shared across layers)
        if (turbo_rotation == nullptr &&
            (type_k == GGML_TYPE_TURBO3_0 || type_k == GGML_TYPE_TURBO4_0 || type_k == GGML_TYPE_TURBO2_0)) {
            turbo_rotation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128);
            ggml_format_name(turbo_rotation, "turbo_rotation");  // R^T
            turbo_rotation_inv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128);
            ggml_format_name(turbo_rotation_inv, "turbo_rotation_inv");  // R

            // InnerQ: per-channel scale_inv tensor (128 floats, initialized to all 1.0)
            turbo_innerq_scale_inv = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, INNERQ_MAX_CHANNELS);
            ggml_format_name(turbo_innerq_scale_inv, "turbo_innerq_scale_inv");
        }
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);

        // Fill turbo rotation matrices AFTER buffer clear (clear zeroes everything)
        if (turbo_rotation != nullptr && turbo_rotation->buffer != nullptr && !model.hparams.no_alloc) {
            #include "turbo-rotation-data.h"
            // ggml is column-major; C arrays are row-major. Storing a row-major matrix
            // into ggml implicitly transposes it. ggml_mul_mat(A, x) computes A^T @ x.
            // To get R @ q: store R^T → ggml sees (R^T)^T_col = R → mul_mat gives R @ q. Wait no —
            // store R so ggml col-major reads it as R^T, then mul_mat gives (R^T)^T = R. ✓
            // Store R for Q forward rotation, R^T for V inverse rotation
            // ggml_mul_mat(A,x) computes A@x for row-major stored A (verified by test)
            ggml_backend_tensor_set(turbo_rotation, TURBO_ROTATION_R, 0, 128 * 128 * sizeof(float));
            ggml_backend_tensor_set(turbo_rotation_inv, TURBO_ROTATION_RT, 0, 128 * 128 * sizeof(float));

            // Initialize InnerQ scale_inv to all 1.0 (identity scaling)
            if (turbo_innerq_scale_inv != nullptr && turbo_innerq_scale_inv->buffer != nullptr) {
                float ones[INNERQ_MAX_CHANNELS];
                for (int i = 0; i < INNERQ_MAX_CHANNELS; i++) ones[i] = 1.0f;
                ggml_backend_tensor_set(turbo_innerq_scale_inv, ones, 0, INNERQ_MAX_CHANNELS * sizeof(float));
            }

            LLAMA_LOG_INFO("%s: TurboQuant rotation matrices initialized (128x128)\n", __func__);
        }
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }

    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    // KV-cache sharing (MTP draft): a shared cache inherits head dims and the
    // resolved rotation policy from its parent so draft and target agree.
    if (other) {
        n_embd_head_k_all = other->n_embd_head_k_all;
        n_embd_head_v_all = other->n_embd_head_v_all;

        attn_rot_k = other->attn_rot_k;
        attn_rot_v = other->attn_rot_v;
    } else {
        // TurboQuant: master's #21038 attention rotation is OFF by default on this
        // fork. Enable per-side via LLAMA_ATTN_ROT_K_OVERRIDE=1 and/or
        // LLAMA_ATTN_ROT_V_OVERRIDE=1 if your specific model+KV combo benefits.
        //
        // Why default OFF: empirical PPL+KLD testing on 7 model families
        // (gemma-4 26B-A4B/31B/E2B, Qwen2.5-7B, Qwen3.5-2B, Mistral-Small-24B,
        // phi-4, on q8/turbo4 KV) showed the optimal rotation policy is highly
        // model-and-quant specific:
        //
        //   • gemma-4 31B Q8 q8/turbo4: V-only rotation gives -43% PPL (huge win).
        //   • gemma-4 26B-A4B Q8 q8/turbo4: V-only gives -3.9%.
        //   • gemma-4 E2B Q4_K_L q8/turbo4: V-only HURTS by +6.7%.
        //   • phi-4 Q8 q8/turbo4: V-side rotation crashes (graph hash overflow).
        //   • Qwen2.5/3.5/Mistral: rotation effect is within standard error.
        //
        // No single default is correct everywhere, including within the same
        // architecture family (gemma-4 above shows three distinct optima across
        // three sizes). Per-arch heuristics in code would silently regress users
        // on variants we haven't tested. Default OFF + per-side env knobs lets
        // each user tune for their specific config; documented findings in the
        // README guide the choice.
        //
        // Reported by @erazortt (TheTom/turboquant_plus#88).
        //
        // LLAMA_ATTN_ROT_DISABLE retained as a hard lock-out: =1 forces rotation
        // off on both sides and blocks the per-side overrides below.
        const char * LLAMA_ATTN_ROT_DISABLE = getenv("LLAMA_ATTN_ROT_DISABLE");
        const bool attn_rot_disable = LLAMA_ATTN_ROT_DISABLE ? (atoi(LLAMA_ATTN_ROT_DISABLE) != 0) : false;

        // Default: rotation OFF on both sides (safe across all tested model families).
        // Override per side via env vars below.
        attn_rot_k = false;
        attn_rot_v = false;

        // Per-side overrides. Set LLAMA_ATTN_ROT_K_OVERRIDE=1 / LLAMA_ATTN_ROT_V_OVERRIDE=1
        // to enable rotation. The cache type and head-dim alignment guards below
        // still apply: rotation only takes effect on quantized types with
        // head_dim % 64 == 0 (master's #21038 requirements).
        const char * ROT_K_OV = getenv("LLAMA_ATTN_ROT_K_OVERRIDE");
        if (ROT_K_OV && atoi(ROT_K_OV) != 0 && !attn_rot_disable) {
            attn_rot_k =
                n_embd_head_k_all > 0 &&
                ggml_is_quantized(type_k) &&
                hparams.n_embd_head_k() % 64 == 0;
        }
        const char * ROT_V_OV = getenv("LLAMA_ATTN_ROT_V_OVERRIDE");
        if (ROT_V_OV && atoi(ROT_V_OV) != 0 && !attn_rot_disable) {
            attn_rot_v =
                n_embd_head_v_all > 0 &&
                ggml_is_quantized(type_v) &&
                hparams.n_embd_head_v() % 64 == 0;
        }

        // always create Hadamard rotation tensors for DeepSeek DSA lightning
        // indexers: this is a functional requirement for these models, not
        // optional tuning, so it overrides the fork's default-off policy (still
        // respects the hard LLAMA_ATTN_ROT_DISABLE lock-out).
        if (!attn_rot_disable &&
            (model.arch == LLM_ARCH_DEEPSEEK32 || model.arch == LLM_ARCH_DEEPSEEK4 || model.arch == LLM_ARCH_GLM_DSA) &&
            hparams.n_embd_head_k_full == hparams.indexer_head_size) {
            attn_rot_k = true;
        }
    }

    LLAMA_LOG_INFO("%s: attn_rot_k = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_k, n_embd_head_k_all);
    LLAMA_LOG_INFO("%s: attn_rot_v = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_v, n_embd_head_v_all);

    // pre-compute the haramard matrices and keep them in host memory
    // TODO: in the future, we can make copies in the backend buffers to avoid host -> device transfers
    if (attn_rot_k || attn_rot_v) {
        for (int64_t n = 64; n <= std::max(n_embd_head_k_all, n_embd_head_v_all); n *= 2) {
            attn_rot_hadamard[n] = std::vector<float>(n*n);

            ggml_init_params params = {
                /* .mem_size   = */ 1*ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };

            ggml_context_ptr ctx { ggml_init(params) };

            ggml_tensor * tmp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n, n);
            tmp->data = attn_rot_hadamard[n].data();

            ggml_gen_hadamard(tmp);
        }
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;
}

void llama_kv_cache::clear(bool data) {
    // Legacy void entry: failure-reporting callers must use try_clear (or the
    // C try-clear API, which returns bool). Logs an explicit refusal and
    // returns with the cache unchanged instead of throwing or aborting, so
    // fault injection (or an active reader breaking quiescence) can never
    // kill the service through this path.
    std::string err;
    if (!try_clear(data, &err)) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, err.c_str());
    }
    }

bool llama_kv_cache::try_clear(bool data, std::string * err) {
    // Unilateral cross-atomic clear (no store-gate dependency): preflight
    // everything BEFORE mutating anything, snapshot the pool as insurance,
    // commit pool reset, then store clear, and only then reset cells.
    // False leaves store, pool, cells and epochs bit-identical: the pool is
    // restored from its snapshot when the store commit fails, the store
    // reports failure before mutating (rechecked preflight contract), and
    // cells are touched last. Object identity of the shared pool is
    // preserved by the in-place reset/restore; views observe the same
    // instance throughout. Never throws.
    try {
        if (xkv_hot_pool && xkv_hot_pool->has_active_reservations()) {
            if (err) *err = "llama_kv_cache::clear: active live reservations exist; cache not quiescent";
            return false;
        }
        if (xkv_store) {
            if (!xkv_store->can_clear(err)) {
                return false;
            }
        }
        // Pool snapshot first (copies; throws only on OOM, caught below
        // with nothing mutated yet).
        llama_xkv::xkv_hot_slot_pool::pool_snapshot pool_snap;
        bool have_pool_snap = false;
        if (xkv_hot_pool) {
            pool_snap = xkv_hot_pool->snapshot_state();
            have_pool_snap = true;
        }
        // Commit order: pool reset, then store clear. A pool-reset failure
        // (post-quiescence race) returns false with the store untouched; a
        // store-clear failure restores the pool snapshot first, so cells
        // (still untouched) never describe a diverged store.
        if (xkv_hot_pool) {
            if (!xkv_hot_pool->reset(err)) {
                return false;
            }
        }
        if (xkv_store) {
            if (!xkv_store->clear(err)) {
                if (have_pool_snap) {
                    xkv_hot_pool->restore_state(pool_snap);
        }
                return false;
            }
        }
        if (xkv_runtime) {
            xkv_runtime->clear_decode_cache();
        }
        for (uint32_t s = 0; s < n_stream; ++s) {
            v_cells[s].reset();
            v_heads[s] = 0;
        }
        fp_bump();
        for (auto & tag : rerot_write_tags) {
            tag.reset();
        }
        for (auto & view : rerot_reader_views) {
            view.reset();
        }
        if (data) {
            for (auto & [_, buf] : ctxs_bufs) {
                ggml_backend_buffer_clear(buf.get(), 0);
            }
            // Re-initialize turbo rotation matrices after buffer clear (clear zeroes everything)
            if (turbo_rotation != nullptr && turbo_rotation->buffer != nullptr) {
                #include "turbo-rotation-data.h"
                ggml_backend_tensor_set(turbo_rotation, TURBO_ROTATION_R, 0, 128 * 128 * sizeof(float));
                ggml_backend_tensor_set(turbo_rotation_inv, TURBO_ROTATION_RT, 0, 128 * 128 * sizeof(float));
                // Re-initialize InnerQ scale_inv to all 1.0
                if (turbo_innerq_scale_inv != nullptr && turbo_innerq_scale_inv->buffer != nullptr) {
                    float ones[INNERQ_MAX_CHANNELS];
                    for (int i = 0; i < INNERQ_MAX_CHANNELS; i++) ones[i] = 1.0f;
                    ggml_backend_tensor_set(turbo_innerq_scale_inv, ones, 0, INNERQ_MAX_CHANNELS * sizeof(float));
                }
            }
        }
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return false;
    } catch (...) {
        if (err) *err = "llama_kv_cache::clear: unknown failure";
        return false;
    }
    return true;
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        // Collect exclusively-owned cells in range. Shared cells (seq_count > 1)
        // only drop this sequence's reference below and release nothing: the
        // payload frees at the last reference, never earlier.
        std::vector<uint32_t> doomed;
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) continue;
            if (cells.seq_has(i, seq_id) && cells.seq_count(i) == 1) {
                doomed.push_back(i);
            }
        }

        // Preflight pool + store with zero mutation; a stale binding or seal
        // lock fails closed before any subsystem mutates.
        xkv_removal_release rel;
        {
            std::string err;
            if (!collect_removal_release(seq_to_stream[seq_id], doomed, rel, &err)) {
                LLAMA_LOG_ERROR("%s: removal preflight failed: %s\n", __func__, err.c_str());
                return false;
            }
            if (!execute_removal_release(rel, &err)) {
                LLAMA_LOG_ERROR("%s: removal release failed: %s\n", __func__, err.c_str());
                return false;
            }
        }

        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            std::vector<uint32_t> doomed;
            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) continue;
                if (!cells.is_empty(i)) {
                    doomed.push_back(i);
                }
            }
            // Aligned hot-only triple + full store lists with dedup; shared
            // payload IDs release once. Fail-closed before cells mutate.
            xkv_removal_release rel;
            {
                std::string err;
                if (!collect_removal_release(s, doomed, rel, &err)) {
                    LLAMA_LOG_ERROR("%s: removal preflight failed: %s\n", __func__, err.c_str());
                    return false;
                }
                if (!execute_removal_release(rel, &err)) {
                    LLAMA_LOG_ERROR("%s: removal release failed: %s\n", __func__, err.c_str());
                    return false;
                }
            }

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    fp_bump();

    return true;
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    // Bounded-hot caches are unified by construction, so a cross-stream copy
    // with hot bindings cannot legitimately occur. Reject before any mutation
    // instead of fabricating physical hot rows (logical index != hot row) or
    // splitting factored shared identity across streams.
    if (is_xkv_bounded_hot()) {
        throw std::runtime_error("llama_kv_cache::seq_cp: cross-stream copy is not supported with bounded-hot XKV");
    }

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            llama_kv_cell_ext ext = v_cells[s0].ext_get(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }

            v_cells[s1].ext_set(i, ext);
            v_cells[s1].rerot_set(i, v_cells[s0].rerot_get(i));
            // Distinct physical copied content receives a fresh payload_id/
            // generation (allocated by pos_set above). Unbounded caches have no
            // hot pool, so the logical index needs no physical translation here.
        }
    }

    v_heads[s1] = v_heads[s0];
    fp_bump();

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    // Cells exclusively owned by other sequences are dropped; their payloads
    // release through the aligned helper with shared-ID dedup.
    std::vector<uint32_t> doomed;
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.is_empty(i) && !cells.seq_has(i, seq_id) && cells.seq_count(i) == 1) {
            doomed.push_back(i);
        }
    }
    // Preflight pool + store with zero mutation; fail closed before cells mutate.
    {
        xkv_removal_release rel;
        std::string err;
        if (!collect_removal_release(seq_to_stream[seq_id], doomed, rel, &err)) {
            throw std::runtime_error("llama_kv_cache::seq_keep: removal preflight failed: " + err);
        }
        if (!execute_removal_release(rel, &err)) {
            throw std::runtime_error("llama_kv_cache::seq_keep: removal release failed: " + err);
        }
    }

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }

    fp_bump();
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    // Bounded hot mapping: cells evicted by a negative shift (pos_add drops
    // them when pos goes negative) must release their pool rows + store
    // payloads, or the pool/store keep stale bindings for dead cells.
    // Pre-scan (read-only) so the removal release is preflighted before any
    // cell mutates; execution after the loop goes through the store-held
    // removal gate. A gate false only logs: this void entry cannot propagate,
    // and matches legacy leak behavior instead of corrupting state.
    xkv_removal_release evict_rel;
    bool have_evict = false;
    if (xkv_hot_pool || xkv_store) {
        std::vector<uint32_t> doomed;
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }
            if (cells.seq_has(i, seq_id) && !cells.is_empty(i) &&
                cells.pos_get(i) + shift < 0) {
                doomed.push_back(i);
            }
        }
        if (!doomed.empty()) {
            std::string pre_err;
            if (collect_removal_release(seq_to_stream[seq_id], doomed, evict_rel, &pre_err)) {
                have_evict = true;
            } else {
                LLAMA_LOG_ERROR("%s: eviction preflight refused: %s\n", __func__, pre_err.c_str());
            }
        }
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    if (have_evict) {
        std::string rel_err;
        if (!execute_removal_release(evict_rel, &rel_err)) {
            LLAMA_LOG_ERROR("%s: eviction release failed: %s\n", __func__, rel_err.c_str());
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
    fp_bump();
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    // Division by zero traps and negative divisors produce nonsense
    // positions; neither is supported. Refuse with zero mutation.
    if (d <= 0) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }

    fp_bump();
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_min(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_max(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams.no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    return ret;
}

bool llama_kv_cache::reserve_hot_slots(slot_info_vec_t & sinfos,
                                        const std::vector<llama_ubatch> & ubatches,
                                        std::vector<llama_xkv::xkv_hot_reservation> & out_reservations,
                                        std::string * err) {
    out_reservations.clear();
    if (!is_xkv_bounded_hot() || !xkv_hot_pool) {
        return true;
    }
    if (sinfos.size() != ubatches.size()) {
        if (err) *err = "reserve_hot_slots: sinfos/ubatches size mismatch";
        return false;
    }
    std::vector<llama_xkv::xkv_hot_reservation> staged;
    staged.reserve(ubatches.size());
    // On any failure, staged reservations roll back via RAII and hot_idxs
    // filled so far are cleared, so callers can fail closed without residue.
    auto clear_filled_hot_idxs = [&]() {
        for (auto & sinfo : sinfos) {
            sinfo.hot_idxs.clear();
        }
    };
    for (size_t u = 0; u < ubatches.size(); ++u) {
        const uint32_t needed = ubatches[u].n_tokens;
        auto & sinfo = sinfos[u];
        if (needed == 0) {
            staged.emplace_back();
            continue;
        }
        std::string res_err;
        auto r = xkv_hot_pool->reserve(needed, &res_err);
        if (!r.valid()) {
            if (err) *err = "reserve_hot_slots: failed to reserve " + std::to_string(needed) +
                            " hot slots for ubatch " + std::to_string(u) + ": " + res_err;
            clear_filled_hot_idxs();
            return false;
        }
        // Fill hot_idxs: one physical row per logical token, stream-major to
        // match the sinfo.idxs layout consumed by apply and page writers.
        if (sinfo.n_stream() == 0 || sinfo.size() == 0 ||
            (uint64_t) sinfo.n_stream() * sinfo.size() != needed) {
            if (err) *err = "reserve_hot_slots: sinfo token count does not match ubatch n_tokens";
            clear_filled_hot_idxs();
            return false;
        }
        sinfo.hot_idxs.resize(sinfo.n_stream());
        size_t slot_offset = 0;
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            sinfo.hot_idxs[s].resize(sinfo.size());
            for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
                sinfo.hot_idxs[s][ii] = r[slot_offset++];
            }
        }
        staged.push_back(std::move(r));
    }
    out_reservations = std::move(staged);
    return true;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    // Fail closed before the first batch: bounded hot tensors must never
    // execute legacy attention without a ready XKV path.
    if (is_xkv_bounded_hot() && xkv_runtime && !xkv_runtime->attention_path_ready()) {
        LLAMA_LOG_ERROR("%s: bounded hot active without a ready XKV path (%s)\n",
            __func__, xkv_runtime->readiness_reason());
        return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true, 0);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        std::vector<llama_xkv::xkv_hot_reservation> hot_res;
        std::string res_err;
        if (!reserve_hot_slots(sinfos, ubatches, hot_res, &res_err)) {
            LLAMA_LOG_WARN("%s: %s\n", __func__, res_err.c_str());
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches), std::move(hot_res));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(optimize);

    bool do_shift = get_has_shift();

    return std::make_unique<llama_kv_cache_context>(this, lctx, do_shift, std::move(sc_info));
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t res;

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remember the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch, /*dry_run=*/true);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

// Shifted-cell record for the bounded context-shift transaction.
struct xkv_shift_item {
    uint32_t stream = 0;
    uint32_t idx = 0;
    uint64_t pid = 0;
    uint64_t gen = 0;
    llama_pos delta = 0;
    llama_pos new_pos = 0;
};

bool llama_kv_cache::update_shift_bounded(llama_context * lctx, std::string * err) {
    // Phase 1: collect shifted cells (read-only). Positions were advanced by
    // seq_add/seq_div already (stock protocol); shift[] holds pending deltas.
    std::vector<xkv_shift_item> items;
    std::unordered_map<uint64_t, uint32_t> pid_cell;
    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells.is_empty(i) || cells.get_shift(i) == 0) {
                continue;
            }
            const uint64_t pid = cells.payload_id_get(i);
            if (pid == 0) {
                if (err) *err = "bounded shift: shifted cell holds untracked residue";
                return false;
            }
            // Cross-cell pid sharing would corrupt row/delta mapping and the
            // position map below; same doctrine as overwrite victims.
            // Any duplicate pid across distinct (stream, cell) visits refuses;
            // the loop visits each cell once, so a hit is always sharing.
            if (!pid_cell.emplace(pid, i).second) {
                if (err) *err = "bounded shift: payload " + std::to_string(pid) +
                                " shared across cells";
                return false;
            }
            xkv_shift_item it;
            it.stream = s;
            it.idx = i;
            it.pid = pid;
            it.gen = cells.storage_generation_get(i);
            it.delta = cells.get_shift(i);
            it.new_pos = cells.pos_get(i);
            items.push_back(it);
        }
    }
    if (items.empty()) {
        for (uint32_t s = 0; s < n_stream; ++s) {
            v_cells[s].reset_shift();
        }
        return true;
    }

    // Phase 2: resolve + partition (read-only). Hot rows rotate; factored
    // segments with landmark streams refresh via COW packs; plain factored
    // canonical (pre-RoPE) needs no store mutation.
    std::unordered_map<uint32_t, llama_pos> row_delta; // hot row -> delta (dedup)
    std::unordered_map<uint64_t, std::vector<const xkv_shift_item *>> seg_items;
    for (const auto & it : items) {
        llama_xkv::xkv_location loc;
        if (!xkv_store || !xkv_store->find_location(it.pid, loc)) {
            if (err) *err = "bounded shift: shifted payload " + std::to_string(it.pid) +
                            " not tracked in store";
            return false;
        }
        if (loc.storage_generation != it.gen) {
            if (err) *err = "bounded shift: generation mismatch for payload " + std::to_string(it.pid);
            return false;
        }
        if (loc.state == llama_xkv::xkv_state::seal_candidate && loc.seal_tx_nonce != 0) {
            if (err) *err = "bounded shift: payload " + std::to_string(it.pid) +
                            " locked in an active seal transaction";
            return false;
        }
        if (loc.kind == llama_xkv::xkv_location_kind::hot) {
            uint32_t slot = 0;
            llama_xkv::xkv_hot_slot_info hinfo;
            if (!xkv_hot_pool || !xkv_hot_pool->find_payload(it.pid, hinfo)) {
                if (err) *err = "bounded shift: hot payload " + std::to_string(it.pid) +
                                " has no pool row";
                return false;
            }
            if (hinfo.storage_generation != it.gen) {
                if (err) *err = "bounded shift: pool generation mismatch for payload " +
                                std::to_string(it.pid);
                return false;
            }
            slot = hinfo.slot;
            auto ins = row_delta.emplace(slot, it.delta);
            if (!ins.second && ins.first->second != it.delta) {
                if (err) *err = "bounded shift: conflicting deltas on shared hot row " +
                                std::to_string(slot);
                return false;
            }
        } else if (loc.kind == llama_xkv::xkv_location_kind::factored) {
            if (loc.segment_id == 0) {
                if (err) *err = "bounded shift: factored payload " + std::to_string(it.pid) +
                                " has no segment";
                return false;
            }
            seg_items[loc.segment_id].push_back(&it);
        } else {
            if (err) *err = "bounded shift: payload " + std::to_string(it.pid) +
                            " is flat-stored with no shift byte path";
            return false;
        }
    }

    // Phase 3: segment profiles. Only segments carrying landmark streams
    // need a refresh; everything else factored is position-independent.
    std::vector<uint64_t> rebuild_seg_ids;
    for (const auto & kv : seg_items) {
        auto seg = xkv_store->get_segment(kv.first);
        if (!seg) {
            if (err) *err = "bounded shift: segment " + std::to_string(kv.first) + " not published";
            return false;
        }
        // Detect landmarks by non-empty chunks table, valid descriptor, or
        // profile contract (never host byte count alone: DEVICE_OWNED has
        // empty host bytes).
        bool has_landmarks = (seg->profile == LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS);
        for (const auto & g : seg->groups) {
            if (!g.landmark_chunks.empty() || g.bytes_landmark > 0 ||
                g.landmark.desc.logical_shape.rows > 0) {
                has_landmarks = true;
                break;
            }
        }
        if (has_landmarks) {
            rebuild_seg_ids.push_back(kv.first);
        }
    }

    // Rotation needed only with RoPE and hot rows; landmark publish needed
    // only with landmark segments. lctx/sched stay untouched on pure-refusal
    // paths above, so a null context fails closed here instead of crashing.
    const bool need_rotate = !row_delta.empty() && hparams.rope_type != LLAMA_ROPE_TYPE_NONE;
    if (lctx == nullptr && (!rebuild_seg_ids.empty() || need_rotate)) {
        if (err) *err = "bounded shift: null context with rotation/publish work pending";
        return false;
    }

    // Native bridge contract: the landmark refresh below decodes factor
    // streams on host. Production device-owned profiles must never decode
    // full streams nor fall back to host reference; their native rebuild
    // kernels are wired via the runtime's native landmark rebuild.
    if (!rebuild_seg_ids.empty()) {
        const bool is_cpu = (xkv_store->get_cparams().xkv_factorizer == LLAMA_XKV_FACTORIZER_CPU_REFERENCE);
        if (!is_cpu) {
            // Verify that runtime is present, armed, and ready for native device landmark rebuild
            if (!xkv_runtime || !xkv_runtime->attention_path_ready() || !xkv_runtime->get_coordinator()) {
                if (err) *err = "bounded shift: device-owned factorizer requires an active native landmark builder and coordinator";
                return false;
            }
        }
    }

    // Phase 4: coordinator exclusion (skipped standalone when no coordinator).
    std::unique_ptr<llama_xkv::xkv_maintenance_guard> mguard;
    if (xkv_tx_coord) {
        std::string merr;
        mguard.reset(new llama_xkv::xkv_maintenance_guard(
            xkv_tx_coord.get(), llama_xkv::xkv_maintenance_op::landmark_publish,
            xkv_shift_maint_id_++, &merr));
        if (!mguard->held()) {
            if (err) *err = "bounded shift: maintenance exclusion contention: " + merr;
            return false;
        }
    }

    // Phase 5: Build exact new phase-aware landmarks in ONE prepared batch COW
    // mutation (all affected segments cloned and re-landmarked atomically),
    // staged off-side. No store mutation commits yet!
    // Full pid -> new-position map: shifted items plus a cell scan for
    // unshifted rows of the same segments (read-only).
    std::unordered_map<uint64_t, int64_t> pid_to_pos;
    if (!rebuild_seg_ids.empty()) {
        const llama_cparams & cparams = lctx->get_cparams();
        const llama_cparams & store_params = xkv_store->get_cparams();
        const ggml_type lm_type = store_params.xkv_landmark_type;
        const uint32_t chunk = store_params.xkv_chunk_tokens;
        for (const auto & it : items) {
            pid_to_pos[it.pid] = (int64_t) it.new_pos;
        }
        for (uint32_t s = 0; s < n_stream; ++s) {
            const auto & cells = v_cells[s];
            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (cells.is_empty(i)) {
                    continue;
                }
                const uint64_t pid = cells.payload_id_get(i);
                if (pid != 0 && pid_to_pos.find(pid) == pid_to_pos.end()) {
                    pid_to_pos[pid] = (int64_t) cells.pos_get(i);
                }
            }
        }
    }

    // Rebuild closure for the atomic batch mutation:
    auto rebuild_fn = [this, &pid_to_pos, lctx](
        const llama_xkv::xkv_segment & old_seg,
        const std::vector<uint32_t> & surviving_rows,
        llama_xkv::xkv_segment & candidate,
        std::string * cerr) -> bool {
        const llama_cparams & cparams = lctx->get_cparams();
        const llama_cparams & store_params = xkv_store->get_cparams();
        const ggml_type lm_type = store_params.xkv_landmark_type;
        const uint32_t chunk = store_params.xkv_chunk_tokens;

        // Two distinct paths:
        // 1. DEVICE_OWNED: refuse closed without host decode. Native device
        //    rebuild issuance (placement buft, allocation IDs, candidate bundle
        //    attach) is owned by the store mutation path; this callback cannot
        //    supply it, so it must never fall back to host reference decode.
        // 2. REFERENCE_HOST: use CPU rebuild_candidate_landmarks_with_positions with decoded host factors.
        if (candidate.residency == GGML_XKV_RES_DEVICE_OWNED) {
            if (!candidate.backend_bundle || !candidate.backend_bundle->is_committed()) {
                if (cerr) *cerr = "shift rebuild: device-owned candidate lacks committed backend bundle";
                return false;
            }
            ggml_backend_t exec = candidate.backend_bundle->get_executor();
            if (!exec) {
                if (cerr) *cerr = "shift rebuild: device-owned candidate backend bundle missing executor";
                return false;
            }
            // Device-owned candidate requires active runtime and coordinator for native landmark builder
            if (!xkv_runtime || !xkv_runtime->attention_path_ready() || !xkv_runtime->get_coordinator()) {
                if (cerr) *cerr = "shift rebuild: native device landmark builder unavailable";
                return false;
            }
            // Build exact positions vector corresponding to surviving_rows
            std::vector<int64_t> dev_positions;
            dev_positions.reserve(surviving_rows.size());
            for (uint32_t r : surviving_rows) {
                if (r >= old_seg.row_payload_ids.size()) {
                    if (cerr) *cerr = "shift rebuild: surviving row out of range";
                    return false;
                }
                const uint64_t pid = old_seg.row_payload_ids[r];
                auto it = pid_to_pos.find(pid);
                if (it == pid_to_pos.end()) {
                    if (cerr) *cerr = "shift rebuild: no position for payload " + std::to_string(pid);
                    return false;
                }
                dev_positions.push_back(it->second);
            }
            std::vector<llama_xkv::xkv_group_phase_spec> specs;
            for (const auto & g : candidate.groups) {
                llama_xkv::layer_group lg;
                lg.group_index = g.group_index;
                lg.owning_layers = g.owning_layers;
                lg.layer_feature_offsets_k = g.layer_feature_offsets_k;
                lg.layer_feature_dims_k = g.layer_feature_dims_k;
                lg.layer_feature_offsets_v = g.layer_feature_offsets_v;
                lg.layer_feature_dims_v = g.layer_feature_dims_v;
                lg.total_dim_k = g.total_dim_k;
                lg.total_dim_v = g.total_dim_v;
                llama_xkv::xkv_group_phase_spec spec;
                spec.group_index = g.group_index;
                if (!llama_xkv::build_landmark_phase_layers(
                        *this, cparams, hparams, lg, spec.layers, spec.fingerprint, cerr)) {
                    return false;
                }
                specs.push_back(std::move(spec));
            }
            // DEVICE_OWNED factors must never be decoded on host: the CPU
            // rebuild decodes source A_K/B_K bytes, which are empty for
            // device-resident segments. Native device rebuild issuance needs
            // the placement buft, allocation IDs, and candidate bundle attach
            // owned by the store mutation path, so this caller fails closed
            // instead of falling back to host reference.
            // DEVICE_OWNED: native device landmark rebuild via
            // xkv_native_landmark_rebuild (zero host factor bytes).
            const auto & src_bundle = candidate.backend_bundle;
            ggml_backend_t rebuild_exec = candidate.backend_bundle->get_executor();
            std::shared_ptr<struct ggml_backend> exec_owner = candidate.backend_bundle->executor;
            std::vector<std::shared_ptr<llama_xkv::xkv_backend_allocation>> rebuilt_lm;
            rebuilt_lm.reserve(candidate.groups.size());
            for (size_t gi = 0; gi < candidate.groups.size(); ++gi) {
                auto & cg = candidate.groups[gi];
                const llama_xkv::xkv_group_phase_spec * spec = nullptr;
                for (const auto & s : specs) {
                    if (s.group_index == cg.group_index) { spec = &s; break; }
                }
                if (!spec) {
                    if (cerr) *cerr = "shift rebuild: device-owned group lacks a phase spec";
                    return false;
                }
                const uint64_t fp_ak = cg.a_k.desc.fingerprint();
                const uint64_t fp_bk = cg.b_k ? cg.b_k->desc.fingerprint() : 0;
                std::shared_ptr<const llama_xkv::xkv_backend_allocation> h_ak, h_bk;
                for (const auto & h : src_bundle->handles) {
                    if (!h) continue;
                    if (h->get_desc().role == llama_xkv::factor_role::a_k &&
                        (h->get_descriptor_fingerprint() == fp_ak || h->get_desc().fingerprint() == fp_ak)) {
                        if (!h_ak) h_ak = h;
                    } else if (cg.b_k && h->get_desc().role == llama_xkv::factor_role::b_k &&
                        (h->get_descriptor_fingerprint() == fp_bk || h->get_desc().fingerprint() == fp_bk)) {
                        if (!h_bk) h_bk = h;
                    }
                }
                if (!h_ak || !h_bk) {
                    if (cerr) *cerr = "shift rebuild: device-owned group missing backend A_K/B_K handle";
                    return false;
                }
                llama_xkv::xkv_native_landmark_rebuild_request req;
                req.a_k = h_ak;
                req.b_k = h_bk;
                req.surviving_rows.reserve(surviving_rows.size());
                for (uint32_t r : surviving_rows) req.surviving_rows.push_back((int32_t) r);
                req.storage_positions = dev_positions;
                req.chunk_tokens = chunk;
                req.landmark_type = lm_type;
                req.seed = 777;
                req.phase_tx_fingerprint = spec->fingerprint;
                req.layers.reserve(spec->layers.size());
                for (const auto & pl : spec->layers) {
                    llama_xkv::xkv_native_seal_hot_layer lay;
                    lay.n_heads = pl.n_heads;
                    lay.head_dim_k = pl.head_dim;
                    lay.head_dim_v = pl.head_dim;
                    lay.rotary_dim_k = pl.rotary_dim;
                    if (hparams.rope_type == LLAMA_ROPE_TYPE_NORM) lay.rope_mode_k = GGML_XKV_ROPE_INTERLEAVED;
                    else lay.rope_mode_k = GGML_XKV_ROPE_HALF;
                    const uint32_t fc = pl.rotary_dim / 2;
                    lay.rope_omega_k = pl.omega;
                    lay.rope_mag_k.clear();
                    lay.rope_mag_k.reserve(fc);
                    for (uint32_t f = 0; f < fc; ++f) {
                        const float sq = (f < pl.freq_scale_sq.size()) ? pl.freq_scale_sq[f] : 1.0f;
                        lay.rope_mag_k.push_back(std::sqrt(sq));
                    }
                    req.layers.push_back(std::move(lay));
                }
                req.executor = exec_owner;
                ggml_backend_buffer_type_t buft = h_ak->get_owning_buft();
                if (!buft) buft = ggml_backend_get_default_buffer_type(rebuild_exec);
                req.buft = buft;
                req.id_gen = &xkv_store->allocation_id_generator();
                req.store_reservation = nullptr;
                req.staging_reservation = nullptr;
                llama_xkv::xkv_native_landmark_rebuild_result lm_res;
                if (!llama_xkv::xkv_native_landmark_rebuild(req, lm_res, cerr)) {
                    return false;
                }
                if (!lm_res.landmark_handle) {
                    if (cerr) *cerr = "shift rebuild: native landmark rebuild produced null handle";
                    return false;
                }
                cg.landmark.desc = lm_res.desc;
                cg.landmark.bytes.clear();
                cg.landmark_chunks = lm_res.chunks;
                cg.landmark_table_fingerprint = lm_res.table_fingerprint;
                cg.refresh_descriptor_fingerprint();
                cg.update_byte_counters();
                rebuilt_lm.push_back(std::move(lm_res.landmark_handle));
            }
            if (rebuilt_lm.size() != candidate.groups.size()) {
                if (cerr) *cerr = "shift rebuild: native landmark handle count mismatch";
                return false;
            }
            std::vector<std::shared_ptr<llama_xkv::xkv_backend_allocation>> new_handles;
            new_handles.reserve(src_bundle->handles.size() + rebuilt_lm.size());
            for (const auto & h : src_bundle->handles) {
                if (h && h->get_desc().role == llama_xkv::factor_role::landmark) continue;
                new_handles.push_back(h);
            }
            for (auto & h : rebuilt_lm) new_handles.push_back(std::move(h));
            llama_xkv::xkv_backend_adopt_result ares;
            ares.handles = std::move(new_handles);
            ares.executor = exec_owner;
            auto new_bundle = ares.make_batch_result(rebuild_exec, exec_owner);
            if (!new_bundle || !new_bundle->is_success() || !new_bundle->is_committed()) {
                if (cerr) *cerr = "shift rebuild: native landmark bundle attach failed";
                return false;
            }
            candidate.backend_bundle = std::move(new_bundle);
            candidate.update_byte_counters();
            return true;
        }
        // Build exact positions vector corresponding to surviving_rows
        std::vector<int64_t> positions;
        positions.reserve(surviving_rows.size());
        for (uint32_t r : surviving_rows) {
            if (r >= old_seg.row_payload_ids.size()) {
                if (cerr) *cerr = "shift rebuild: surviving row out of range";
                return false;
            }
            const uint64_t pid = old_seg.row_payload_ids[r];
            auto it = pid_to_pos.find(pid);
            if (it == pid_to_pos.end()) {
                if (cerr) *cerr = "shift rebuild: no position for payload " + std::to_string(pid);
                return false;
            }
            positions.push_back(it->second);
        }
        std::vector<llama_xkv::xkv_group_phase_spec> specs;
        for (const auto & g : candidate.groups) {
            llama_xkv::layer_group lg;
            lg.group_index = g.group_index;
            lg.owning_layers = g.owning_layers;
            lg.layer_feature_offsets_k = g.layer_feature_offsets_k;
            lg.layer_feature_dims_k = g.layer_feature_dims_k;
            lg.layer_feature_offsets_v = g.layer_feature_offsets_v;
            lg.layer_feature_dims_v = g.layer_feature_dims_v;
            lg.total_dim_k = g.total_dim_k;
            lg.total_dim_v = g.total_dim_v;
            llama_xkv::xkv_group_phase_spec spec;
            spec.group_index = g.group_index;
            if (!llama_xkv::build_landmark_phase_layers(
                    *this, cparams, hparams, lg, spec.layers, spec.fingerprint, cerr)) {
                return false;
            }
            specs.push_back(std::move(spec));
        }
        // REFERENCE_HOST only: the CPU rebuild decodes host factor streams.
        // Device-owned candidates are refused in the branch above; reaching
        // here with device residency is a logic error, so refuse closed.
        if (candidate.residency == GGML_XKV_RES_DEVICE_OWNED) {
            if (cerr) *cerr = "shift rebuild: device-owned candidate reached host rebuild path";
            return false;
        }
        return llama_xkv::rebuild_candidate_landmarks_with_positions(
            old_seg, surviving_rows, positions, specs, candidate,
            lm_type, chunk, cerr);
    };

    // Phase 6: Hot rotation staged into isolated off-side buffers/scratch.
    // Stage rotated hot rows off-side; live K is NOT mutated before store commit!
    // Device-local off-side staging: Allocate dedicated device staging tensors
    // and context holding exactly the affected hot rows on the active backend device.
    // Queue ggml_backend_tensor_copy_async for live-row -> staging-row, synchronize once.
    // On failure, reverse copies are queued and synchronized.
    // No full-history D2H, no host vectors per row, no per-row syncs, and bounded by workspace.
    struct device_row_staging {
        ggml_tensor * k_tensor = nullptr;
        ggml_context * staging_ctx = nullptr;
        ggml_backend_buffer_ptr staging_buf;
        ggml_backend_t backend = nullptr;
        ggml_tensor * staging_tensor = nullptr; // 2D [row_elements, n_affected]
        struct row_view_pair {
            ggml_tensor * live_view = nullptr;
            ggml_tensor * staging_view = nullptr;
        };
        std::vector<row_view_pair> view_pairs;

        device_row_staging() = default;
        ~device_row_staging() {
            reset();
        }
        device_row_staging(const device_row_staging &) = delete;
        device_row_staging & operator=(const device_row_staging &) = delete;
        device_row_staging(device_row_staging && o) noexcept
            : k_tensor(o.k_tensor),
              staging_ctx(o.staging_ctx),
              staging_buf(std::move(o.staging_buf)),
              backend(o.backend),
              staging_tensor(o.staging_tensor),
              view_pairs(std::move(o.view_pairs)) {
            o.k_tensor = nullptr;
            o.staging_ctx = nullptr;
            o.backend = nullptr;
            o.staging_tensor = nullptr;
        }
        device_row_staging & operator=(device_row_staging && o) noexcept {
            if (this != &o) {
                reset();
                k_tensor = o.k_tensor;
                staging_ctx = o.staging_ctx;
                staging_buf = std::move(o.staging_buf);
                backend = o.backend;
                staging_tensor = o.staging_tensor;
                view_pairs = std::move(o.view_pairs);
                o.k_tensor = nullptr;
                o.staging_ctx = nullptr;
                o.backend = nullptr;
                o.staging_tensor = nullptr;
            }
            return *this;
        }
        void reset() noexcept {
            // Rollback safety: queued async copies (live->staging preflight
            // or staging->live reverse restore) must complete before the
            // staging buffer is released, on every exit path including
            // early returns and exceptions that skip the explicit rollback.
            if (backend != nullptr) {
                ggml_backend_synchronize(backend);
            }
            staging_buf.reset(); // Buffer released before context
            if (staging_ctx != nullptr) {
                ggml_free(staging_ctx);
                staging_ctx = nullptr;
            }
            k_tensor = nullptr;
            backend = nullptr;
            staging_tensor = nullptr;
            view_pairs.clear();
        }
    };
    std::vector<device_row_staging> staged_layers;
    if (need_rotate) {
        auto * sched = lctx->get_sched();
        if (!sched) {
            if (err) *err = "bounded shift: scheduler is null for rotation";
            return false;
        }
        staged_layers.reserve(layers.size());
        const size_t n_affected = row_delta.size();
        size_t total_staging_bytes = 0;

        // Preflight 1: Check workspace budget for all staged layers
        for (const auto & layer : layers) {
            if (layer.k && layer.k->buffer) {
                const size_t row_b = ggml_row_size(layer.k->type, layer.k->ne[0]);
                if (n_affected > 0 && row_b > std::numeric_limits<size_t>::max() / n_affected) {
                    if (err) *err = "bounded shift: staging bytes overflow";
                    return false;
                }
                const size_t layer_staging_b = n_affected * row_b;
                if (total_staging_bytes > std::numeric_limits<size_t>::max() - layer_staging_b) {
                    if (err) *err = "bounded shift: total staging bytes overflow";
                    return false;
                }
                total_staging_bytes += layer_staging_b;
            }
        }
        llama_xkv::xkv_device_staging_reservation staging_res;
        if (xkv_store != nullptr && total_staging_bytes > 0) {
            std::string st_err;
            size_t deficit = 0;
            staging_res = xkv_store->reserve_device_staging(total_staging_bytes, &st_err, &deficit);
            if (xkv_store->get_arena().get_capacity_bytes() > 0 && !staging_res.valid()) {
                if (err) *err = "bounded shift: " + st_err;
                return false;
            }
        }

        // Preflight 2: Allocate contexts, buffers, and views; copy live -> staging
        std::vector<ggml_backend_t> active_backends;
        for (const auto & layer : layers) {
            if (layer.k && layer.k->buffer) {
                device_row_staging st;
                st.k_tensor = layer.k;

                // Obtain backend for this specific layer tensor from the scheduler
                st.backend = ggml_backend_sched_get_tensor_backend(sched, layer.k);
                if (!st.backend) {
                    if (err) *err = "bounded shift: could not resolve backend for layer tensor";
                    return false;
                }
                ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(layer.k->buffer);
                if (buft && !ggml_backend_dev_supports_buft(ggml_backend_get_device(st.backend), buft)) {
                    if (err) *err = "bounded shift: backend device does not support layer buffer type";
                    return false;
                }
                if (std::find(active_backends.begin(), active_backends.end(), st.backend) == active_backends.end()) {
                    active_backends.push_back(st.backend);
                }

                // Create allocated staging tensor and context on same buffer type
                ggml_init_params st_params = {
                    /* mem_size   = */ (n_affected * 2 + 8) * ggml_tensor_overhead(),
                    /* mem_buffer = */ nullptr,
                    /* no_alloc   = */ true,
                };
                st.staging_ctx = ggml_init(st_params);
                if (!st.staging_ctx) {
                    if (err) *err = "bounded shift: failed to init staging context";
                    return false;
                }
                st.staging_tensor = ggml_new_tensor_2d(st.staging_ctx, layer.k->type, layer.k->ne[0], n_affected);
                if (!st.staging_tensor) {
                    if (err) *err = "bounded shift: failed to create staging tensor";
                    return false;
                }
                st.staging_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(st.staging_ctx, buft));
                if (!st.staging_buf) {
                    if (err) *err = "bounded shift: failed to allocate staging buffer from buft";
                    return false;
                }

                // Create views for each affected row and queue copy
                st.view_pairs.reserve(n_affected);
                size_t idx = 0;
                for (const auto & rd : row_delta) {
                    device_row_staging::row_view_pair vp;
                    vp.live_view = ggml_view_1d(st.staging_ctx, layer.k, layer.k->ne[0],
                        (size_t) rd.first * layer.k->nb[1]);
                    vp.staging_view = ggml_view_1d(st.staging_ctx, st.staging_tensor, layer.k->ne[0],
                        idx * st.staging_tensor->nb[1]);
                    if (!vp.live_view || !vp.staging_view) {
                        if (err) *err = "bounded shift: failed to create row views for staging";
                        return false;
                    }
                    if (ggml_backend_view_init(vp.live_view) != GGML_STATUS_SUCCESS) {
                        if (err) *err = "bounded shift: live view init failed";
                        return false;
                    }
                    if (ggml_backend_view_init(vp.staging_view) != GGML_STATUS_SUCCESS) {
                        if (err) *err = "bounded shift: staging view init failed";
                        return false;
                    }
                    ggml_backend_tensor_copy_async(st.backend, st.backend, vp.live_view, vp.staging_view);
                    st.view_pairs.push_back(vp);
                    ++idx;
                }
                staged_layers.push_back(std::move(st));
            }
        }
        // Single sync per active backend across all queued row copies
        for (ggml_backend_t b : active_backends) {
            ggml_backend_synchronize(b);
        }

        ggml_backend_sched_reset(sched);
        auto * res = lctx->get_gf_res_reserve();
        res->reset();
        auto * gf = build_graph_shift(res, lctx);
        if (gf == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to build graph for bounded K-shift\n", __func__);
            return false;
        }
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate compute graph for bounded K-shift\n", __func__);
            return false;
        }
        res->set_inputs(nullptr);
        if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: failed to compute bounded K-shift\n", __func__);
            // Device-local reverse copy restore of affected hot rows without D2H
            for (auto & st : staged_layers) {
                for (const auto & vp : st.view_pairs) {
                    ggml_backend_tensor_copy_async(st.backend, st.backend, vp.staging_view, vp.live_view);
                }
            }
            for (ggml_backend_t b : active_backends) {
                ggml_backend_synchronize(b);
            }
            return false;
        }
        ggml_backend_sched_synchronize(sched);
    }

    // Phase 7: Atomic COW store commit of all prepared landmark segments
    // in one transaction with no-fail ordering. If store mutation fails,
    // restore only the affected hot rows on device via queued reverse copy!
    if (!rebuild_seg_ids.empty()) {
        llama_xkv::xkv_batch_mutation bm;
        bm.segments_to_pack = rebuild_seg_ids;
        bm.landmark_rebuild = rebuild_fn;
        llama_xkv::xkv_mutation_result mres;
        if (!xkv_store->execute_mutation_transaction(bm, &mres, err)) {
            LLAMA_LOG_ERROR("%s: failed to execute landmark batch COW mutation: %s\n",
                __func__, err ? err->c_str() : "");
            // Device-local reverse copy restore: live K returns to exact unshifted state without D2H
            for (auto & st : staged_layers) {
                for (const auto & vp : st.view_pairs) {
                    ggml_backend_tensor_copy_async(st.backend, st.backend, vp.staging_view, vp.live_view);
                }
            }
            std::vector<ggml_backend_t> synced_backends;
            for (const auto & st : staged_layers) {
                if (st.backend && std::find(synced_backends.begin(), synced_backends.end(), st.backend) == synced_backends.end()) {
                    ggml_backend_synchronize(st.backend);
                    synced_backends.push_back(st.backend);
                }
            }
            return false;
        }
    }
    // Phase 8: Commit metadata only after every data move and every store
    // candidate succeeds. Hot rotation above is staged device-local with reverse-
    // copy rollback; the landmark COW above is one atomic batch mutation that
    // refreshes (never truncates) landmark bytes/chunks and rejects stale or
    // zero source fingerprints. Retired versions stay readable until the last pin
    // releases; shift deltas clear only here so any failure above stays retryable
    // with bit-identical K/V bytes. Pool rows are unchanged (same rows, rotated bytes).
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset_shift();
    }
    fp_bump();
    return true;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    bool updated = false;

    // Bounded context-shift transaction (positions-aware landmark COW +
    // hot-only rotation, atomically under coordinator exclusion). The stock
    // path below must never run on bounded tensors: it sizes the shift
    // graph over logical cell counts.
    if (do_shift && is_xkv_bounded_hot()) {
        assert(sc_info.empty() && "bounded caches never carry cross-stream copies");
        std::string err;
        if (!update_shift_bounded(lctx, &err)) {
            LLAMA_LOG_ERROR("%s: bounded shift transaction refused/failed: %s\n", __func__, err.c_str());
            return false;
        }
        return true;
    }

    auto * sched = lctx->get_sched();

    if (!sc_info.empty()) {
        assert(n_stream > 1 && "stream copy should never happen with a single stream");

        llama_synchronize(lctx);

        const size_t n_copy = sc_info.ssrc.size();

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info.ssrc[i];
            const auto sdst = sc_info.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);

                if (layer.v_stream[ssrc]) {
                    ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
                }
            }
        }
    }

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (gf == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to build graph for K-shift\n", __func__);
                return updated;
            }
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];

            cells.reset_shift();
        }
        fp_bump();
    }

    return updated;
}

void llama_kv_cache::compact() {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }
    auto planned_cells = v_cells;
    if (!compact_planned(planned_cells)) {
        throw std::runtime_error("KV native compaction is unsupported by this backend/layout");
    }
}

bool llama_kv_cache::compact_planned(llama_kv_cells_vec & planned_cells) {
    fp_bump();

    // In XKV dense/SR mode with bounded hot cache, logical cells are packed semantically,
    // while physical hot tensors are not moved or rebound.
    if (is_xkv_bounded_hot()) {
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];
            const auto plan = cells.make_pack_plan();
            if (plan.moves.empty()) {
                continue;
            }
            cells.apply_pack(plan);
            head = plan.retained_count;
            // NOTE: deliberately no store notification here. kv_pack_plan
            // carries LOGICAL cell indices, which the store would misread as
            // hot-pool physical rows (xkv_location.row), corrupting bindings
            // and epochs. Payload IDs travel with the cells (apply_pack
            // preserves pid/gen/pos/ext/shift/seq/rerot); the store locator
            // (pool row or segment row) is untouched, so no epoch moves.
            // Factored row packing happens via the store mutation
            // transaction, never cell compact.
        }
        return true;
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        auto & cells = planned_cells[s];
        auto & head  = v_heads[s];

        const auto plan = cells.make_pack_plan();

        if (plan.moves.empty()) {
            continue;
        }

        // Check if already dense (no compaction needed)
        bool needs_compaction = false;
        for (const auto & move : plan.moves) {
            if (move.dst_begin != move.src_begin) {
                needs_compaction = true;
                break;
            }
        }
        if (!needs_compaction) {
            continue;
        }

        const uint32_t kv_size = cells.size();

        // Build one ordered backend-native move batch per backing buffer. The
        // backend provides memmove semantics, so overlapping downward pack
        // ranges are safe without CPU staging. We preflight every buffer before
        // executing any data movement; only after every buffer succeeds do we
        // commit cell metadata.
        std::map<ggml_backend_buffer_t, std::vector<ggml_backend_tensor_memmove_region>> buffer_moves;

        auto add_row_move = [&](ggml_tensor * tensor, const auto & move) {
            const size_t row_bytes  = ggml_row_size(tensor->type, tensor->ne[0]);
            const size_t row_stride = tensor->nb[1];

            ggml_backend_tensor_memmove_region region = {};
            region.tensor = tensor;
            region.src_offset = (size_t) move.src_begin * row_stride;
            region.dst_offset = (size_t) move.dst_begin * row_stride;

            if (row_stride == row_bytes) {
                region.size = (size_t) move.length * row_bytes;
                region.n_copies = 1;
                region.src_stride = 0;
                region.dst_stride = 0;
            } else {
                // Preserve non-contiguous row layouts exactly rather than
                // assuming nb[1] == row_size(type, ne[0]).
                region.size = row_bytes;
                region.n_copies = move.length;
                region.src_stride = row_stride;
                region.dst_stride = row_stride;
            }

            buffer_moves[tensor->buffer].push_back(region);
        };

        for (const auto & layer : layers) {
            // Move K rows: one row per cell, row width = ne[0] (may be turbo-padded)
            if (layer.k_stream[s]) {
                auto * k = layer.k_stream[s];

                for (const auto & move : plan.moves) {
                    if (move.dst_begin == move.src_begin) {
                        continue;
                    }
                    add_row_move(k, move);
                }
            }

            // Move V rows
            if (layer.v_stream[s]) {
                auto * v = layer.v_stream[s];

                if (!v_trans) {
                    // Non-transposed V: same row layout as K
                    for (const auto & move : plan.moves) {
                        if (move.dst_begin == move.src_begin) {
                            continue;
                        }
                        add_row_move(v, move);
                    }
                } else {
                    // Transposed V: cell dimension is contiguous inside each
                    // embedding row. Represent the entire embedding dimension
                    // as one strided backend region rather than synchronizing
                    // once per row.
                    if (ggml_blck_size(v->type) != 1) {
                        throw std::runtime_error(
                            "TriAttention native compaction does not support transposed quantized V cache");
                    }

                    const size_t   v_size_el = ggml_type_size(v->type);
                    const uint32_t n_embd_v  = (uint32_t) v->ne[0];
                    const size_t   row_stride = (size_t) kv_size * v_size_el;

                    for (const auto & move : plan.moves) {
                        if (move.dst_begin == move.src_begin) {
                            continue;
                        }

                        ggml_backend_tensor_memmove_region region = {};
                        region.tensor = v;
                        region.src_offset = (size_t) move.src_begin * v_size_el;
                        region.dst_offset = (size_t) move.dst_begin * v_size_el;
                        region.size = (size_t) move.length * v_size_el;
                        region.n_copies = n_embd_v;
                        region.src_stride = row_stride;
                        region.dst_stride = row_stride;
                        buffer_moves[v->buffer].push_back(region);
                    }
                }
            }
        }

        for (const auto & [buffer, moves] : buffer_moves) {
            if (!ggml_backend_tensor_memmove_regions_supported(moves.data(), moves.size())) {
                return false;
            }
        }

        for (const auto & [buffer, moves] : buffer_moves) {
            GGML_UNUSED(buffer);
            if (!ggml_backend_tensor_memmove_regions(moves.data(), moves.size())) {
                throw std::runtime_error("TriAttention native compaction failed after successful preflight");
            }
        }

        // Apply metadata changes after all K/V data is moved
        cells.apply_pack(plan);
        head = plan.retained_count;
    }
    v_cells = std::move(planned_cells);
    return true;
}

void llama_kv_cache::init_triattention(
        const char * stats_path,
        double ratio,
        uint32_t recent_window,
        const llama_cparams & cparams) {
    if (!stats_path || stats_path[0] == '\0') {
        throw std::runtime_error("TriAttention stats path cannot be empty");
    }
    if (!std::isfinite(ratio) || ratio <= 0.0 || ratio > 1.0) {
        throw std::runtime_error("TriAttention ratio must be finite and in (0, 1]");
    }

    tri_ratio = ratio;
    tri_recent_window = recent_window;

    triattention_scorer_config cfg;
    cfg.agg = TRIATTENTION_AGG_MEAN;
    cfg.normalize_scores = true;
    cfg.disable_mlr = false;
    cfg.disable_trig = false;

    const uint32_t head_dim = n_embd_head_k_all > 0 ? (uint32_t) n_embd_head_k_all : hparams.n_embd_head_k(0);
    const uint32_t n_kv_heads = hparams.n_head_kv(0);

    if (layers.empty()) {
        throw std::runtime_error("TriAttention requires at least one KV attention layer");
    }

    // Fail closed early if any KV attention layer uses transposed quantized V cache,
    // which cannot be packed natively by TriAttention.
    if (v_trans) {
        for (const auto & layer : layers) {
            for (auto * v : layer.v_stream) {
                if (v && ggml_blck_size(v->type) != 1) {
                    throw std::runtime_error(
                        "TriAttention native compaction does not support transposed quantized V cache");
                }
            }
        }
    }

    // Inspect existing K/V stream buffers; fail closed if any non-null layer stream
    // buffer cannot support native in-place tensor memmove.
    for (const auto & layer : layers) {
        for (auto * k : layer.k_stream) {
            if (k && k->buffer) {
                ggml_backend_tensor_memmove_region r = {};
                r.tensor = k;
                r.n_copies = 1;
                if (!ggml_backend_tensor_memmove_regions_supported(&r, 1)) {
                    throw std::runtime_error("TriAttention native compaction requires backend memmove_tensor");
                }
            }
        }
        for (auto * v : layer.v_stream) {
            if (v && v->buffer) {
                ggml_backend_tensor_memmove_region r = {};
                r.tensor = v;
                r.n_copies = 1;
                if (!ggml_backend_tensor_memmove_regions_supported(&r, 1)) {
                    throw std::runtime_error("TriAttention native compaction requires backend memmove_tensor");
                }
            }
        }
    }

    uint32_t expected_rope_style = 0;
    switch (llama_model_rope_type(&model)) {
        case LLAMA_ROPE_TYPE_NORM:
            expected_rope_style = 1; // adjacent even/odd vector pairs
            break;
        case LLAMA_ROPE_TYPE_NEOX:
        case LLAMA_ROPE_TYPE_MROPE:
        case LLAMA_ROPE_TYPE_IMROPE:
            expected_rope_style = 0; // front/back NeoX vector pairs
            break;
        case LLAMA_ROPE_TYPE_NONE:
        case LLAMA_ROPE_TYPE_VISION:
            throw std::runtime_error("TriAttention does not support this RoPE layout");
    }

    const uint32_t rotary_dim = hparams.n_rot(layers.front().il);
    if (rotary_dim == 0 || rotary_dim > head_dim || (rotary_dim & 1u) != 0) {
        throw std::runtime_error("TriAttention requires a valid even rotary dimension");
    }
    const uint32_t freq_count = rotary_dim / 2;
    const double rope_theta = (double) model.get_rope_freq_base(cparams, layers.front().il);

    auto build_layer_rope = [&](uint32_t il, std::vector<float> & omega, std::vector<float> & scale_sq) {
        if (hparams.n_rot(il) != rotary_dim) {
            throw std::runtime_error("TriAttention does not yet support heterogeneous rotary dimensions across KV layers");
        }

        std::vector<float> factors;
        const float * factor_ptr = nullptr;
        if (ggml_tensor * factor_tensor = model.get_rope_factors(cparams, il)) {
            if (factor_tensor->type != GGML_TYPE_F32 || factor_tensor->ne[0] < (int64_t) freq_count) {
                throw std::runtime_error("TriAttention requires F32 RoPE factors with sufficient frequency entries");
            }
            factors.resize(freq_count);
            ggml_backend_tensor_get(factor_tensor, factors.data(), 0, freq_count * sizeof(float));
            factor_ptr = factors.data();
        }

        omega.resize(freq_count);
        scale_sq.resize(freq_count);
        if (!triattention_build_rope_tables(
                omega.data(), scale_sq.data(), rotary_dim,
                model.get_rope_freq_base(cparams, il),
                model.get_rope_freq_scale(cparams, il),
                (int32_t) cparams.n_ctx_orig_yarn,
                cparams.yarn_ext_factor,
                cparams.yarn_attn_factor,
                cparams.yarn_beta_fast,
                cparams.yarn_beta_slow,
                factor_ptr)) {
            throw std::runtime_error("failed to derive exact RoPE tables for TriAttention");
        }
    };

    std::vector<float> runtime_omega;
    std::vector<float> runtime_scale_sq;
    build_layer_rope(layers.front().il, runtime_omega, runtime_scale_sq);

    // score_combined currently uses one frequency basis for every sampled
    // attention layer. Fail closed if the model's actual KV layers differ.
    for (size_t l = 1; l < layers.size(); ++l) {
        std::vector<float> layer_omega;
        std::vector<float> layer_scale_sq;
        build_layer_rope(layers[l].il, layer_omega, layer_scale_sq);
        for (uint32_t f = 0; f < freq_count; ++f) {
            const float omega_tol = 1e-6f * std::max(1.0f, std::abs(runtime_omega[f]));
            const float scale_tol = 1e-6f * std::max(1.0f, std::abs(runtime_scale_sq[f]));
            if (std::abs(layer_omega[f] - runtime_omega[f]) > omega_tol ||
                std::abs(layer_scale_sq[f] - runtime_scale_sq[f]) > scale_tol) {
                throw std::runtime_error(
                    "TriAttention does not yet support heterogeneous RoPE frequency/scaling tables across KV layers");
            }
        }
    }

    tri_scorer = std::make_unique<triattention_scorer>(
        stats_path, cfg, rope_theta, head_dim, n_kv_heads,
        rotary_dim, runtime_omega.data(), runtime_scale_sq.data(), (int32_t) expected_rope_style);
    if (!tri_scorer || !tri_scorer->valid()) {
        tri_scorer.reset();
        throw std::runtime_error(std::string("failed to initialize TriAttention scorer from: ") + stats_path);
    }
}

llama_memory_kv_reclaim_result llama_kv_cache::reclaim_kv(const llama_memory_kv_reclaim_request & request) {
    llama_memory_kv_reclaim_result result;

    // Physical owner dedup: view caches (other != nullptr) share v_cells_impl
    // with their owner. Only the owner may execute reclaim/compact.
    if (other) {
        return other->reclaim_kv(request);
    }

    if (!tri_scorer || !tri_scorer->valid()) {
        result.supported = false;
        return result;
    }

    result.supported = true;

    if (v_cells.empty()) {
        return result;
    }

    // Unified KV: all sequences share stream 0
    auto & cells = v_cells[0];

    // physical_before must be hot bound before planning, not initial cells.get_used()
    result.physical_before = (xkv_store && xkv_hot_pool) ? xkv_hot_pool->get_bound() : get_kv_used();
    if (result.physical_before == 0) {
        result.physical_after = 0;
        result.physical_freed = 0;
        result.capacity_satisfied = (request.required_free == 0);
        result.floor_reached = true;
        return result;
    }

    // XKV active branch: delegate lossy reclaim to XKV TriAttention adapter across HOT+FACTORED payloads
    if (xkv_store) {
        const auto * cal = tri_scorer->get_calibration();
        const float * omega = tri_scorer->get_omega();
        const float * fsq = tri_scorer->get_freq_scale_sq();
        if (cal && omega && fsq) {
            llama_xkv::xkv_tri_config cfg;
            cfg.ratio = tri_ratio;
            cfg.recent_window = tri_recent_window;
            llama_xkv::xkv_tri_adapter adapter(*cal, omega, fsq, cfg);

            // Build layer slices from attention layers
            const uint32_t eff_head_dim_k = n_embd_head_k_all > 0 ? (uint32_t)n_embd_head_k_all : hparams.n_embd_head_k(0);
            const uint32_t eff_head_dim_v = n_embd_head_v_all > 0 ? (uint32_t)n_embd_head_v_all : hparams.n_embd_head_v(0);
            std::vector<llama_xkv::xkv_layer_slice> slices;
            slices.reserve(layers.size());
            for (size_t l = 0; l < layers.size(); ++l) {
                llama_xkv::xkv_layer_slice sl;
                sl.model_layer = (uint32_t)layers[l].il;
                sl.owning_layer = xkv_store->resolve_owning_layer(sl.model_layer);
                sl.head_dim = eff_head_dim_k;
                sl.rotary_dim = hparams.n_rot(layers[l].il);
                sl.n_kv_heads = (uint32_t)hparams.n_head_kv(layers[l].il);
                sl.feature_dim_k = sl.n_kv_heads * sl.head_dim;
                sl.feature_dim_v = sl.n_kv_heads * eff_head_dim_v;
                sl.rope_style = 0; // MUST be 0 (half/NeoX pairing; IMRoPE pairing is also NeoX half)
                slices.push_back(sl);
            }

            llama_xkv::xkv_tri_pressure_hooks hooks;
            hooks.hot_span_provider = [this, eff_head_dim_k, omega, fsq](uint32_t ml, uint32_t kv_h, const uint32_t * cell_idxs, size_t n, float * dst, size_t dst_cap) -> bool {
                for (size_t l = 0; l < this->layers.size(); ++l) {
                    if ((uint32_t)this->layers[l].il == ml) {
                        ggml_tensor * k_ten = this->layers[l].k;
                        if (!k_ten) return false;

                        // Resolve physical rows and storage positions from store hot locations
                        std::vector<uint32_t> phys_rows(n);
                        std::vector<int32_t> pos_buf(n);
                        for (size_t ci = 0; ci < n; ++ci) {
                            uint32_t cell_i = cell_idxs[ci];
                            uint64_t pid = this->v_cells[0].payload_id_get(cell_i);
                            if (pid == 0) return false; // missing payload: fail closed
                            llama_xkv::xkv_location loc;
                            if (!this->xkv_store->find_location(pid, loc) ||
                                (loc.kind != llama_xkv::xkv_location_kind::hot &&
                                 loc.kind != llama_xkv::xkv_location_kind::flat_quantized)) {
                                return false; // missing location or not hot/flat: fail closed, never fall back to cell_i!
                            }
                            if (loc.row >= (uint32_t)k_ten->ne[1]) {
                                return false; // row out of bounds: fail closed
                            }
                            phys_rows[ci] = loc.row;
                            pos_buf[ci] = (int32_t)this->v_cells[0].pos_get(cell_i);
                        }

                        // Safe bulk readback using xkv_backend_hot_readback_batch: queued async gets + one sync
                        ggml_backend_buffer_type_t buft_k = k_ten->buffer ? ggml_backend_buffer_get_type(k_ten->buffer) : nullptr;
                        ggml_backend_t backend_k = nullptr;
                        if (buft_k) {
                            ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft_k);
                            if (dev) backend_k = ggml_backend_dev_init(dev, nullptr);
                        }

                        const size_t row_bytes = ggml_row_size(k_ten->type, k_ten->ne[0]);

                        // Turbo rows are stored 128-padded (n_embd_k_gqa_eff), so the
                        // per-head stride and decode width must use the padded head dim.
                        // Non-turbo types have no padding: padded == logical.
                        const bool k_is_turbo_head = k_ten->type == GGML_TYPE_TURBO2_0 ||
                                                k_ten->type == GGML_TYPE_TURBO3_0 ||
                                                k_ten->type == GGML_TYPE_TURBO4_0;
                        const uint32_t padded_hd_k = k_is_turbo_head
                            ? ((eff_head_dim_k + 127) / 128) * 128
                            : eff_head_dim_k;
                        const size_t head_offset_pad = ggml_row_size(k_ten->type, (uint64_t)kv_h * padded_hd_k);

                        llama_xkv::xkv_backend_hot_readback_request hreq;
                        hreq.hot_tensor = k_ten;
                        hreq.hot_capacity = (uint32_t)k_ten->ne[1];
                        hreq.row_stride_bytes = row_bytes;
                        hreq.physical_slots = phys_rows.data();
                        hreq.n_slots = (uint32_t)n;

                        llama_xkv::xkv_backend_hot_readback_result hres;
                        std::string rerr;
                        if (!llama_xkv::xkv_backend_hot_readback_batch(backend_k, hreq, hres, &rerr)) {
                            return false;
                        }

                        // Padded Turbo heads decode the full padded width into scratch,
                        // inverse-rotate every 128-block, then keep the logical prefix.
                        // The 128-aligned path below is bit-identical to the old code.
                        std::vector<float> pad_tmp;
                        if (padded_hd_k != eff_head_dim_k) {
                            pad_tmp.resize(padded_hd_k);
                        }
                        for (size_t ci = 0; ci < n; ++ci) {
                            const uint8_t * src_row = hres.bytes.data() + ci * row_bytes;
                            const uint8_t * src_head = src_row + head_offset_pad;
                            float * dst_head = dst + ci * eff_head_dim_k;
                            if (k_ten->type == GGML_TYPE_F32) {
                                std::memcpy(dst_head, src_head, eff_head_dim_k * sizeof(float));
                            } else {
                                const auto * traits = ggml_get_type_traits(k_ten->type);
                                if (!traits || !traits->to_float) {
                                    return false; // unsupported type: fail closed
                                }
                                if (padded_hd_k != eff_head_dim_k) {
                                    traits->to_float(src_head, pad_tmp.data(), padded_hd_k);
                                    for (uint32_t g = 0; g < padded_hd_k; g += 128) {
                                        ggml_turbo_wht_inverse_row(pad_tmp.data() + g, 128);
                                    }
                                    std::memcpy(dst_head, pad_tmp.data(), eff_head_dim_k * sizeof(float));
                                } else {
                                    traits->to_float(src_head, dst_head, eff_head_dim_k);
                                }
                            }
                        }

                        // Inverse attention rotation (WHT) for Turbo types
                        if (k_is_turbo_head && padded_hd_k == eff_head_dim_k) {
                            for (size_t ci = 0; ci < n; ++ci) {
                                float * row_ptr = dst + ci * eff_head_dim_k;
                                for (uint32_t g = 0; g < eff_head_dim_k; g += 128) {
                                    const uint32_t cur_g = std::min<uint32_t>(128, eff_head_dim_k - g);
                                    if (cur_g == 128) {
                                        ggml_turbo_wht_inverse_row(row_ptr + g, 128);
                                    }
                                }
                            }
                        }

                        // Inverse partial RoPE with exact frequencies and scaling tables
                        const uint32_t rotary_dim = this->hparams.n_rot(this->layers[l].il);
                        const uint32_t freq_count = rotary_dim / 2;
                        // Carve post_rope scratch from arena to avoid heap allocation
                        auto lease_pr = this->xkv_store->acquire_workspace_lease(n * eff_head_dim_k * sizeof(float));
                        float * pr_buf = lease_pr ? lease_pr.as<float>() : nullptr;
                        std::vector<float> fallback_pr;
                        if (!pr_buf) {
                            fallback_pr.resize(n * eff_head_dim_k);
                            pr_buf = fallback_pr.data();
                        }
                        std::memcpy(pr_buf, dst, n * eff_head_dim_k * sizeof(float));

                        triattention_invert_rope(
                            dst,
                            pr_buf,
                            pos_buf.data(),
                            omega,
                            fsq,
                            (uint32_t)n,
                            eff_head_dim_k,
                            rotary_dim,
                            freq_count,
                            0 // half/NeoX pairing
                        );

                        return true;
                    }
                }
                return false;
            };
            // Device-owned selected K fetch using BackendResidency native reconstruct
            hooks.selected_k_fetch = [this, eff_head_dim_k, &adapter](
                uint64_t seg_id,
                uint64_t seg_ver,
                const uint32_t * srows,
                uint32_t rn,
                const llama_xkv::xkv_layer_slice & sl,
                uint32_t kv_h,
                float * dst,
                size_t dst_cap,
                std::string * err
            ) -> bool {
                if (rn == 0) return true;
                if (!dst) {
                    if (err) *err = "fetch: null destination pointer";
                    return false;
                }
                uint64_t req_elems = 0;
                if (__builtin_mul_overflow((uint64_t)rn, (uint64_t)eff_head_dim_k, &req_elems) ||
                    req_elems > (uint64_t)std::numeric_limits<size_t>::max() ||
                    dst_cap < (size_t)req_elems) {
                    if (err) *err = "fetch: destination capacity too small";
                    return false;
                }
                auto seg = seg_ver > 0 ? this->xkv_store->get_segment_version(seg_id, seg_ver)
                                       : this->xkv_store->get_segment(seg_id);
                if (!seg) {
                    if (err) *err = "fetch: segment not found";
                    return false;
                }
                const auto * g = seg->find_group_for_layer(sl.owning_layer);
                if (!g) {
                    if (err) *err = "fetch: group not found for layer";
                    return false;
                }

                // Dispatch by segment residency
                if (seg->residency == GGML_XKV_RES_DEVICE_OWNED) {
                    // DEVICE_OWNED must call BackendResidency native bounded reconstruct/fetch
                    if (!seg->backend_bundle) {
                        if (err) *err = "fetch: DEVICE_OWNED segment missing backend bundle";
                        return false;
                    }
                    // Validate exact group and match backend handles by group/layer identity
                    const auto * fg = seg->find_group_for_layer(sl.owning_layer);
                    if (!fg) {
                        if (err) *err = "fetch: exact group not found for layer";
                        return false;
                    }
                    llama_xkv::xkv_backend_tri_fetch_handles b_handles;
                    const uint64_t fp_ak = fg->a_k.desc.fingerprint();
                    const uint64_t fp_bk = fg->b_k ? fg->b_k->desc.fingerprint() : 0;
                    const uint64_t fp_av = fg->a_v.desc.fingerprint();
                    const uint64_t fp_bv = fg->b_v ? fg->b_v->desc.fingerprint() : 0;
                    for (const auto & h : seg->backend_bundle->handles) {
                        if (!h) continue;
                        const uint64_t h_fp = h->get_descriptor_fingerprint();
                        if (h_fp == fp_ak) b_handles.a_k = h;
                        else if (h_fp == fp_bk) b_handles.b_k = h;
                        else if (h_fp == fp_av) b_handles.a_v = h;
                        else if (h_fp == fp_bv) b_handles.b_v = h;
                    }
                    if (!b_handles.a_k || !b_handles.b_k) {
                        if (err) *err = "fetch: DEVICE_OWNED segment missing required a_k or b_k backend handle for group " + std::to_string(fg->group_index);
                        return false;
                    }

                    // Borrow owning backend from buffer without creating/leaking new backend instance
                    ggml_backend_t dev_be = nullptr;
                    ggml_backend_buffer_type_t buft = b_handles.a_k->get_owning_buft();
                    if (buft) {
                        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
                        if (dev) dev_be = ggml_backend_dev_init(dev, nullptr);
                    }
                    return llama_xkv::xkv_backend_tri_fetch_selected_k(
                        dev_be, b_handles, srows, rn, eff_head_dim_k, kv_h,
                        sl.n_kv_heads, sl.feature_offset_k, sl.feature_dim_k,
                        dst, dst_cap, err);
                }

                // REFERENCE_HOST: dispatch host reconstruct stream_factored_k_head
                try {
                    adapter.stream_factored_k_head(*seg, sl, kv_h, srows, rn, dst, dst_cap,
                                                   32);
                    return true;
                } catch (const std::exception & e) {
                    if (err) *err = std::string("fetch failed: ") + e.what();
                    return false;
                }
            };
            hooks.arena = &xkv_store->get_arena();

            const int64_t t0 = ggml_time_us();
            // Generic required_free represents physical hot/KV slots deficit in XKV reclaim_kv
            auto prop = adapter.plan_pressure(cells, request, slices, *xkv_store, false, hooks,
                                              llama_xkv::xkv_tri_pressure_resource::hot_slots);
            const auto decision = adapter.decide(prop);

            if (decision == llama_xkv::xkv_tri_pressure_decision::reclaimed) {
                // Collect hot removals (pid, generation, slot) to release xkv_hot_slot_pool atomically
                std::vector<uint64_t> hot_pids;
                std::vector<uint64_t> hot_gens;
                std::vector<uint32_t> hot_slots;
                for (size_t i = 0; i < prop.released_hot_payloads.size(); ++i) {
                    uint64_t pid = prop.released_hot_payloads[i];
                    uint32_t slot = prop.released_hot_rows[i];
                    llama_xkv::xkv_location loc;
                    if (xkv_store->find_location(pid, loc)) {
                        hot_pids.push_back(pid);
                        hot_gens.push_back(loc.storage_generation);
                        hot_slots.push_back(slot);
                    }
                }

                // Set batch.removal_precommit using make_pool_removal_gate
                prop.store_proposal.batch.removal_precommit = make_pool_removal_gate(hot_pids, hot_gens, hot_slots);

                // Preflight hot pool release atomically BEFORE mutating store
                if (xkv_hot_pool && !hot_pids.empty()) {
                    std::string can_err;
                    if (!xkv_hot_pool->can_release_batch(hot_pids.data(), hot_gens.data(), hot_slots.data(), hot_pids.size(), &can_err)) {
                        result.supported = true;
                        result.changed = false;
                        result.capacity_satisfied = false;
                        result.floor_reached = false;
                        return result;
                    }
                }

                const uint32_t hot_bound_before = xkv_hot_pool ? xkv_hot_pool->get_bound() : 0;
                std::string apply_err;
                if (prop.store_proposal.apply(*xkv_store, nullptr, &apply_err)) {
                    // Apply reference removals to cells
                    for (const auto & rem : prop.plan.ref_removals) {
                        if (cells.seq_has(rem.cell_index, rem.seq_id)) {
                            cells.seq_rm(rem.cell_index, rem.seq_id);
                        }
                    }
                    compact();
                    const uint32_t hot_bound_after = xkv_hot_pool ? xkv_hot_pool->get_bound() : 0;
                    const uint32_t hot_freed = (hot_bound_before > hot_bound_after) ? (hot_bound_before - hot_bound_after) : prop.hot_slots_freed;
                    auto r = adapter.result_from_proposal(prop, request, result.physical_before);
                    r.physical_after = hot_bound_after; // Map actual resource usage, not semantic cells
                    r.physical_freed = hot_freed; // Factored-only removals count 0 hot slots toward physical_freed
                    r.capacity_satisfied = (hot_freed >= request.required_free);
                    r.score_us = (uint64_t)std::max<int64_t>(0, ggml_time_us() - t0);
                    fp_bump();
                    return r;
                }
                // Store apply failure: return explicit failure, NEVER fall through to legacy
                result.supported = true;
                result.changed = false;
                result.capacity_satisfied = false;
                result.floor_reached = false;
                return result;
            } else if (decision == llama_xkv::xkv_tri_pressure_decision::floor_exhausted_victim) {
                auto r = adapter.result_from_proposal(prop, request, result.physical_before);
                r.floor_reached = true;
                return r;
            } else if (decision == llama_xkv::xkv_tri_pressure_decision::bypass_recurrent_only) {
                result.supported = true;
                result.changed = false;
                result.capacity_satisfied = false;
                result.floor_reached = false;
                return result;
            } else {
                // Any other decision (error, retry_stale): fail closed explicitly, never fall through to legacy scorer!
                result.supported = true;
                result.changed = false;
                result.capacity_satisfied = false;
                result.floor_reached = (decision == llama_xkv::xkv_tri_pressure_decision::floor_exhausted_victim);
                return result;
            }
        }
        // XKV active but scorer initialization failed: fail closed
        result.supported = false;
        return result;
    }

    // Legacy branch for XKV OFF / fallback
    auto planned_cells = v_cells;

    // Build K tensor array and layer_map for score_combined()
    const uint32_t n_kv_layers = (uint32_t) layers.size();
    std::vector<ggml_tensor *> k_tensors(n_kv_layers);
    std::vector<int32_t> layer_map(n_kv_layers);
    for (uint32_t l = 0; l < n_kv_layers; l++) {
        k_tensors[l] = layers[l].k;
        layer_map[l] = (int32_t) layers[l].il;
    }

    std::vector<llama_memory_kv_reclaim_seq_hint> hints = request.seq_hints;
    if (hints.empty()) {
        // The low-level decode retry has no server slot hints. Discover all
        // resident sequences, not just seq 0 (also works with split streams).
        for (llama_seq_id seq = 0; (size_t) seq < seq_to_stream.size(); ++seq) {
            const llama_pos pmax = planned_cells[seq_to_stream[seq]].seq_pos_max(seq);
            if (pmax >= 0) {
                llama_memory_kv_reclaim_seq_hint h{};
                h.seq_id = seq;
                h.logical_tokens = (uint32_t) pmax + 1;
                h.tail_guard = tri_recent_window;
                h.eligible = true;
                hints.push_back(h);
            }
        }
    }

    for (const auto & hint : hints) {
        if (!hint.eligible) {
            continue;
        }

        const llama_seq_id seq_id = hint.seq_id;
        if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
            throw std::runtime_error("TriAttention: invalid sequence hint");
        }
        const uint32_t stream = seq_to_stream[seq_id];
        auto & cells = planned_cells[stream];
        for (uint32_t l = 0; l < n_kv_layers; ++l) {
            k_tensors[l] = layers[l].k_stream[stream];
        }
        const llama_pos max_pos = cells.seq_pos_max(seq_id);
        if (max_pos < 0) {
            continue;
        }

        const uint32_t tail_guard = hint.tail_guard > 0 ? hint.tail_guard : tri_recent_window;
        const uint32_t logical_tokens = hint.logical_tokens > 0 ? hint.logical_tokens : (uint32_t) (max_pos + 1);

        // Target retention based on the configured ratio (§B.9, §B.11).
        uint32_t target_retention = tri_rerot_target_retention(logical_tokens, tri_ratio, tail_guard);

        result.target_references += target_retention;

        // Cells with pos >= recent_threshold are hard-protected
        const llama_pos recent_threshold = (max_pos >= (llama_pos) tail_guard) ? (max_pos - (llama_pos) tail_guard + 1) : 0;

        std::vector<uint32_t> cand_indices;
        std::vector<int32_t>  cand_positions;
        uint32_t n_protected = 0;

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells.is_empty(i) || !cells.seq_has(i, seq_id)) {
                continue;
            }

            const llama_pos pos = cells.pos_get(i);
            const auto & rerot_meta = cells.rerot_get(i);
            // A PENDING structural record is all-or-nothing: it may become
            // PUBLIC at the closing token. Reclaiming any prefix cell before
            // that publication would make the logical run denser than its
            // physical K/V and violate atomic visibility.
            const bool rerot_pending =
                rerot_meta.active() &&
                rerot_meta.visibility == llama_rerot_visibility::pending_record;
            const bool semantic_foreign_tag =
                hint.semantic_episode_id != 0 &&
                rerot_meta.active() &&
                (rerot_meta.episode_id != hint.semantic_episode_id ||
                 rerot_meta.visibility != llama_rerot_visibility::public_live);
            bool semantic_reader_tail = false;
            if (hint.semantic_episode_id != 0) {
                for (const llama_seq_id ref : hint.semantic_seq_ids) {
                    if (ref == seq_id || !cells.seq_has(i, ref)) {
                        continue;
                    }
                    const llama_pos ref_max = cells.seq_pos_max(ref);
                    const llama_pos ref_recent = ref_max >= (llama_pos) tail_guard
                        ? ref_max - (llama_pos) tail_guard + 1
                        : 0;
                    if (pos >= ref_recent) {
                        semantic_reader_tail = true;
                        break;
                    }
                }
            }
            if (pos >= recent_threshold || rerot_pending ||
                semantic_foreign_tag || semantic_reader_tail) {
                n_protected++;
            } else {
                cand_indices.push_back(i);
                cand_positions.push_back((int32_t) pos);
            }
        }

        result.hard_keep += n_protected;

        const uint32_t n_candidates = (uint32_t) cand_indices.size();
        const uint32_t total_seq_cells = n_protected + n_candidates;

        if (total_seq_cells <= target_retention || n_candidates == 0) {
            // Already at or below target retention, no candidates to evict
            continue;
        }

        // How many candidates to keep
        const uint32_t candidates_to_keep = (target_retention > n_protected) ? std::min(target_retention - n_protected, n_candidates) : 0;

        // Score candidates
        std::vector<float> scores(n_candidates);
        const int64_t t_score_start = ggml_time_us();
        tri_scorer->score_combined(
            scores.data(),
            k_tensors.data(),
            n_kv_layers,
            layer_map.data(),
            cand_indices.data(),
            cand_positions.data(),
            n_candidates,
            (int64_t) max_pos);
        std::vector<float> pooled_scores(n_candidates);
        triattention_max_pool_scores(
            pooled_scores.data(), scores.data(), cand_positions.data(), n_candidates, 2);
        scores.swap(pooled_scores);
        result.score_us += (uint64_t) std::max<int64_t>(0, ggml_time_us() - t_score_start);

        // Select top candidates to keep (highest score first)
        std::vector<uint32_t> order(n_candidates);
        std::iota(order.begin(), order.end(), 0);

        if (candidates_to_keep < n_candidates) {
            if (candidates_to_keep > 0) {
                std::partial_sort(order.begin(), order.begin() + candidates_to_keep, order.end(),
                    [&scores, &cand_positions](uint32_t a, uint32_t b) {
                        if (scores[a] != scores[b]) {
                            return scores[a] > scores[b];
                        }
                        return cand_positions[a] > cand_positions[b];
                    });
            }

            // Evict candidates from order[candidates_to_keep .. n_candidates - 1]
            for (uint32_t k = candidates_to_keep; k < n_candidates; ++k) {
                const uint32_t cand_idx = order[k];
                const uint32_t cell_i   = cand_indices[cand_idx];

                const auto rerot_meta = cells.rerot_get(cell_i);
                const bool semantic_cell =
                    hint.semantic_episode_id != 0 &&
                    (!rerot_meta.active() ||
                     (rerot_meta.episode_id == hint.semantic_episode_id &&
                      rerot_meta.visibility == llama_rerot_visibility::public_live));
                if (semantic_cell) {
                    // The archive hint represents one semantic base/PUBLIC
                    // cell regardless of archive/exec bookkeeping refs. Remove
                    // only this episode's refs; another outer completion may
                    // legally retain the same physical base-prefix cell.
                    for (const llama_seq_id ref : hint.semantic_seq_ids) {
                        if (!cells.is_empty(cell_i) && cells.seq_has(cell_i, ref)) {
                            cells.seq_rm(cell_i, ref);
                            result.references_removed++;
                        }
                    }
                } else {
                    if (!cells.is_empty(cell_i) && cells.seq_has(cell_i, seq_id)) {
                        cells.seq_rm(cell_i, seq_id);
                        result.references_removed++;
                    }
                }
            }
        }
    }

    // Report the final physical shared set, not the number of intermediate
    // seq_rm() calls that happened to leave a cell referenced. This makes the
    // metric directly describe the union that must remain resident.
    for (const auto & cells_s : planned_cells) {
      for (uint32_t i = 0; i < cells_s.size(); ++i) {
        if (!cells_s.is_empty(i) && cells_s.seq_count(i) > 1) {
            result.shared_keep++;
        }
      }
    }

    // Pack remaining used cells to [0, retained_count)
    const int64_t t_pack_start = ggml_time_us();
    if (!compact_planned(planned_cells)) {
        const uint32_t before = result.physical_before;
        result = {};
        result.physical_before = result.physical_after = before;
        return result;
    }
    result.pack_us += (uint64_t) std::max<int64_t>(0, ggml_time_us() - t_pack_start);

    result.physical_after = get_kv_used();
    result.physical_freed = result.physical_before - result.physical_after;
    result.changed = (result.references_removed > 0);
    result.capacity_satisfied = (result.physical_freed >= request.required_free);
    result.floor_reached = true;
    fp_bump();

    return result;
}

bool llama_kv_cache::positions_are_sparse() const {
    // Positions may have gaps after TriAttention eviction
    return tri_scorer && tri_scorer->valid();
}

llama_kv_cache::slot_info llama_kv_cache::mtp_slot_info(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const uint32_t st    = seq_to_stream[seq_id];
    const auto &   cells = v_cells[st];

    llama_pos pmax = cells.seq_pos_max(seq_id);

    uint32_t idx = 0;

    if (pmax >= 0) {
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.seq_has(i, seq_id)) {
                continue;
            }
            if (cells.pos_get(i) == pmax) {
                idx = i;
                break;
            }
        }
    }

    slot_info res;
    res.s0   = 0;
    res.s1   = 0;
    res.strm = { (llama_seq_id) st };
    res.idxs.resize(1);
    res.idxs[0] = { idx };

    return res;
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const {

    if (debug > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];

            LLAMA_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.is_empty(i)) {
                        ss += '.';
                    } else {
                        assert(cells.seq_count(i) >= 1);

                        if (cells.seq_count(i) == 1) {
                            ss += std::to_string(cells.seq_get(i));
                        } else {
                            ss += 'M';
                        }
                    }
                    if (i%256 == 255) {
                        ss += " *";
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    std::string cur;
                    if (cells.is_empty(i)) {
                        cur = '.';
                    } else {
                        cur = std::to_string(cells.pos_get(i));
                    }
                    const int n = cur.size();
                    for (int j = 0; j < 5 - n; ++j) {
                        cur += ' ';
                    }
                    ss += cur;
                    if (i%256 == 255) {
                        ss += " *";
                    }
                    if (i%64 == 63) {
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (cells.seq_pos_min(s) < 0) {
                    continue;
                }

                LLAMA_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, stream_id, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
            }
        }
    }

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
        /*.hot_idxs =*/ { },
    };

    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            GGML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            GGML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);

        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        if (n_tokens > cells.size()) {
            LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            return { };
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                //const llama_pos    pos    = ubatch.pos[i];
                //const llama_seq_id seq_id = ubatch.seq_id[i][0];

                // can we use this cell? either:
                //  - the cell is empty
                //  - the cell is occupied only by one sequence:
                //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
                //    - mask SWA, using current max pos for that sequence in the cache
                //                always insert in the cell with minimum pos
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1) {
                    const llama_pos pos_cell = cells.pos_get(idx);

                    // (disabled) causal mask
                    // note: it's better to purge any "future" tokens beforehand
                    //if (cells.seq_has(idx, seq_id)) {
                    //    can_use = pos_cell >= pos;
                    //}

                    if (!can_use) {
                        const llama_seq_id seq_id_cell = cells.seq_get(idx);

                        // SWA mask
                        if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                            can_use = true;
                        }
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs[s].clear();
            }

            if (n_tested >= cells.size()) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                return { };
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);

    return res;
}

bool llama_kv_cache::collect_overwrite_victims(const slot_info & sinfo, xkv_overwrite_victims & out,
                                                std::string * err) {
    out.items.clear();
    if (other) {
        return true;
    }
    if (!xkv_hot_pool && !xkv_store) {
        return true;
    }
    std::vector<uint64_t> seen_pids;
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t idx = sinfo.idxs[s][ii];
            if (cells.is_empty(idx)) {
                continue;
            }
            const uint64_t pid = cells.payload_id_get(idx);
            if (pid == 0) {
                continue;
            }
            for (uint64_t seen : seen_pids) {
                if (seen == pid) {
                    if (err) *err = "collect_overwrite_victims: duplicate payload_id " + std::to_string(pid) +
                                    " across victim cells (cross-cell sharing is not overwritable)";
                    out.items.clear();
                    return false;
                }
            }
            seen_pids.push_back(pid);
            xkv_victim_snapshot snap;
            snap.stream = sinfo.strm[s];
            snap.cell   = idx;
            snap.pos    = cells.pos_get(idx);
            snap.ext    = cells.ext_get(idx);
            snap.shift  = cells.get_shift(idx);
            snap.seqs   = cells.seq_snapshot(idx);
            snap.rerot  = cells.rerot_get(idx);
            snap.pid    = pid;
            snap.gen    = cells.storage_generation_get(idx);
            if (snap.seqs.count() != 1) {
                if (err) *err = "collect_overwrite_victims: victim cell is shared across sequences; "
                                "overwriting it would destroy co-refs (fail closed, zero mutation)";
                out.items.clear();
                return false;
            }
            // Presence-gated alignment: untracked residue (e.g. dry-run cells
            // never registered in store/pool) is overwritten freely.
            if (xkv_hot_pool) {
                llama_xkv::xkv_hot_slot_info hinfo;
                if (xkv_hot_pool->find_payload(pid, hinfo)) {
                    if (hinfo.storage_generation != snap.gen) {
                        if (err) *err = "collect_overwrite_victims: hot generation mismatch for payload " +
                                        std::to_string(pid);
                        out.items.clear();
                        return false;
                    }
                    snap.in_pool = true;
                    snap.hot_row = hinfo.slot;
                }
            }
            if (xkv_store) {
                llama_xkv::xkv_location loc;
                if (xkv_store->find_location(pid, loc)) {
                    if (loc.storage_generation != snap.gen) {
                        if (err) *err = "collect_overwrite_victims: store generation mismatch for payload " +
                                        std::to_string(pid);
                        out.items.clear();
                        return false;
                    }
                    if (loc.state == llama_xkv::xkv_state::seal_candidate && loc.seal_tx_nonce != 0) {
                        if (err) *err = "collect_overwrite_victims: payload " + std::to_string(pid) +
                                        " is locked in an active seal transaction";
                        out.items.clear();
                        return false;
                    }
                    snap.in_store = true;
                    snap.store_state = loc.state;
                }
            }
            out.items.push_back(snap);
        }
    }
    // Bounded-SR landmark rebuild BEFORE any mutation: the runtime owns
    // positions-aware summary repair; false fails closed with zero mutation.
    if (!out.items.empty()) {
        xkv_removal_cells hook_cells;
        for (const auto & snap : out.items) {
            hook_cells.emplace_back(snap.stream, snap.cell);
        }
        if (!run_removal_rebuild_hook(hook_cells, err)) {
            out.items.clear();
            return false;
        }
    }
    // Cross-subsystem preflight with zero mutation: no subsystem may mutate
    // if another would fail. Both validators are read-only.
    std::vector<uint64_t> hot_pids;
    std::vector<uint64_t> hot_gens;
    std::vector<uint32_t> hot_slots;
    for (const auto & snap : out.items) {
        if (snap.in_pool) {
            hot_pids.push_back(snap.pid);
            hot_gens.push_back(snap.gen);
            hot_slots.push_back(snap.hot_row);
        }
    }
    if (!hot_pids.empty()) {
        if (!xkv_hot_pool->can_release_batch(hot_pids.data(), hot_gens.data(), hot_slots.data(),
                                              hot_pids.size(), err)) {
            out.items.clear();
            return false;
        }
    }
    return true;
}

bool llama_kv_cache::release_collected_victims(const xkv_overwrite_victims & victims, std::string * err) {
    std::vector<uint64_t> hot_pids;
    std::vector<uint64_t> hot_gens;
    std::vector<uint32_t> hot_slots;
    std::vector<uint64_t> store_pids;
    std::vector<uint64_t> store_gens;
    for (const auto & snap : victims.items) {
        if (snap.in_pool) {
            hot_pids.push_back(snap.pid);
            hot_gens.push_back(snap.gen);
            hot_slots.push_back(snap.hot_row);
        }
        if (snap.in_store) {
            store_pids.push_back(snap.pid);
            store_gens.push_back(snap.gen);
        }
    }
    // Gate protocol (same guarantees as execute_removal_release): the store
    // fires our pool gate under its lock and commits allocation-free after
    // gate-true. Any false leaves store, pool, cells and epochs
    // bit-identical; the caller releases nothing further on false.
    auto gate = make_pool_removal_gate(hot_pids, hot_gens, hot_slots);
    if (xkv_store) {
        if (!xkv_store->remove_payloads(store_pids, store_gens, nullptr, err,
                                         nullptr, nullptr, gate)) {
            return false;
        }
        return true;
    }
    if (!hot_pids.empty()) {
        if (!xkv_hot_pool || !xkv_hot_pool->release_batch(hot_pids.data(), hot_gens.data(),
                                                           hot_slots.data(), hot_pids.size(), err)) {
            return false;
        }
    }
    return true;
}

void llama_kv_cache::verify_bound_hot_rows(const std::vector<xkv_applied_entry> & entries) {
    // Per-row finiteness probe over committed hot storage. Covers F32/F16
    // directly and any other codec via type traits (e.g. Q8_0 boundary
    // layers); turbo rows need canonical decode and are noted once.
    auto row_finite = [](ggml_tensor * t, uint32_t slot, const char * tag, int il, uint32_t cell, uint64_t pid) {
        if (!t || !t->data) return;
        const uint32_t n = (uint32_t) t->ne[0];
        const uint8_t * base = (const uint8_t *) t->data + (size_t) slot * (size_t) t->nb[1];
        std::vector<float> owned;
        const float * row = nullptr;
        if (t->type == GGML_TYPE_F32) {
            row = (const float *) base;
        } else if (t->type == GGML_TYPE_F16) {
            owned.resize(n);
            for (uint32_t d = 0; d < n; ++d) owned[d] = ggml_fp16_to_fp32(((const ggml_fp16_t *) base)[d]);
            row = owned.data();
        } else if (t->type == GGML_TYPE_TURBO2_0 || t->type == GGML_TYPE_TURBO3_0 || t->type == GGML_TYPE_TURBO4_0) {
            static bool noted = false;
            if (!noted) {
                noted = true;
                LLAMA_LOG_WARN("[llama_kv_cache] verify_bound_hot_rows: turbo %s rows unchecked (need canonical decode)\n", tag);
            }
            return;
        } else {
            const ggml_type_traits * traits = ggml_get_type_traits(t->type);
            const int64_t blck = ggml_blck_size(t->type);
            if (!traits || !traits->to_float || blck <= 0 || n % (uint32_t) blck != 0) return;
            owned.resize(n);
            traits->to_float(base, owned.data(), n);
            row = owned.data();
        }
        uint32_t bad = 0, first = 0;
        for (uint32_t d = 0; d < n; ++d) {
            if (!std::isfinite(row[d])) { if (bad == 0) first = d; ++bad; }
        }
        if (bad > 0) {
            LLAMA_LOG_ERROR("[llama_kv_cache] postcompute: non-finite %s at layer %d, hot_slot %u (cell %u): %u/%u elems, first elem %u: %f (pid %llu type %s)\n",
                tag, il, slot, cell, bad, n, first, row[first], (unsigned long long) pid, ggml_type_name(t->type));
        }
    };
    for (const auto & e : entries) {
        if (!e.has_hot) continue;
        for (const auto & layer : layers) {
            // K + V discriminator: Vcur has no norm/rope/rot, so V NaN + K
            // NaN => shared hidden/embedding fault at this layer's input;
            // V finite + K NaN => K-path only (wk, attn_k_norm, RoPE, k_rot).
            row_finite(layer.k, e.hot_slot, "K", layer.il, e.cell, e.pid);
            row_finite(layer.v, e.hot_slot, "V", layer.il, e.cell, e.pid);
        }
    }
}
void llama_kv_cache::set_removal_rebuild_hook(xkv_removal_rebuild_fn fn) {
    removal_rebuild = std::move(fn);
}

bool llama_kv_cache::run_removal_rebuild_hook(const xkv_removal_cells & cells, std::string * err) {
    if (!removal_rebuild || cells.empty()) {
        return true;
    }
    return removal_rebuild(cells, err);
}

void llama_kv_cache::restore_victim_cells(const xkv_overwrite_victims & victims) {
    // Victim pool/store entries are still bound on every path that calls this
    // (release is deferred to postcompute_success), so only cells need repair.
    if (other) {
        return;
    }
    for (const auto & snap : victims.items) {
        auto & cells = v_cells[snap.stream];
        if (!cells.is_empty(snap.cell)) {
            cells.rm(snap.cell);
        }
        cells.pos_set(snap.cell, snap.pos);
        cells.payload_id_set(snap.cell, snap.pid, snap.gen);
        cells.seq_restore(snap.cell, snap.seqs);
        cells.ext_set(snap.cell, snap.ext);
        if (snap.shift != 0) {
            cells.pos_add(snap.cell, snap.shift);
        }
        cells.rerot_set(snap.cell, snap.rerot);
    }
    fp_bump();
}

// Remove new store entries and drop new cells for applied entries
// [base, size). Victim pool/store entries are untouched. Routes through the
// store-held removal gate (empty pool triple: nothing bound yet on this
// path); false leaves everything bit-identical for the caller to propagate.
bool llama_kv_cache::rollback_applied_entries(const std::vector<xkv_applied_entry> & entries, size_t base,
                                               std::string * err) {
    if (other) {
        return true;
    }
    if (base >= entries.size()) {
        return true;
    }
    std::vector<uint64_t> pids;
    std::vector<uint64_t> gens;
    for (size_t k = base; k < entries.size(); ++k) {
        pids.push_back(entries[k].pid);
        gens.push_back(entries[k].gen);
    }
    if (!pids.empty()) {
        auto gate = make_pool_removal_gate({}, {}, {});
        if (xkv_store) {
            if (!xkv_store->remove_payloads(pids, gens, nullptr, err,
                                             nullptr, nullptr, gate)) {
                return false;
            }
        }
    }
    for (size_t k = base; k < entries.size(); ++k) {
        auto & cells = v_cells[entries[k].stream];
        if (!cells.is_empty(entries[k].cell)) {
            cells.rm(entries[k].cell);
        }
    }
    fp_bump();
    return true;
}

bool llama_kv_cache::rollback_applied_batch(const std::vector<xkv_applied_entry> & entries,
                                             const std::vector<xkv_overwrite_victims> & victims,
                                             std::string * err) {
    if (other) {
        return true;
    }
    // Bounded-SR landmark rebuild BEFORE any mutation.
    if (!entries.empty()) {
        xkv_removal_cells hook_cells;
        for (const auto & e : entries) {
            hook_cells.emplace_back(e.stream, e.cell);
        }
        std::string hook_err;
        if (!run_removal_rebuild_hook(hook_cells, &hook_err)) {
            if (err) *err = "rebuild hook: " + hook_err;
            return false;
        }
    }
    if (entries.empty()) {
        return true;
    }
    // Gate protocol: store removal of new payloads with our pool gate for
    // the bound rows. Gate/store false returns false with store, pool, cells
    // and epochs bit-identical (nothing below runs). Only after true do the
    // infallible cell repairs run: drop new cells, restore victims (victim
    // pool/store entries were never released).
    {
        std::vector<uint64_t> pids;
        std::vector<uint64_t> gens;
        std::vector<uint64_t> hot_pids;
        std::vector<uint64_t> hot_gens;
        std::vector<uint32_t> hot_slots;
        for (const auto & e : entries) {
            pids.push_back(e.pid);
            gens.push_back(e.gen);
            if (e.has_hot) {
                hot_pids.push_back(e.pid);
                hot_gens.push_back(e.gen);
                hot_slots.push_back(e.hot_slot);
            }
        }
        auto gate = make_pool_removal_gate(hot_pids, hot_gens, hot_slots);
        if (xkv_store) {
            if (!xkv_store->remove_payloads(pids, gens, nullptr, err,
                                             nullptr, nullptr, gate)) {
                return false;
            }
        } else if (!hot_pids.empty()) {
            if (!xkv_hot_pool || !xkv_hot_pool->release_batch(hot_pids.data(), hot_gens.data(),
                                                              hot_slots.data(), hot_pids.size(), err)) {
                return false;
            }
        }
    }
    for (const auto & e : entries) {
        auto & cells = v_cells[e.stream];
        if (!cells.is_empty(e.cell)) {
            cells.rm(e.cell);
        }
    }
    for (const auto & v : victims) {
        if (!v.empty()) {
            restore_victim_cells(v);
        }
    }
    fp_bump();
    return true;
}

bool llama_kv_cache::collect_removal_release(uint32_t stream, const std::vector<uint32_t> & idxs,
                                              xkv_removal_release & out, std::string * err) {
    out = xkv_removal_release{};
    if (other) {
        return true;
    }
    if (!xkv_hot_pool && !xkv_store) {
        return true;
    }
    if (stream >= v_cells.size()) {
        if (err) *err = "collect_removal_release: stream out of range";
        return false;
    }
    const auto & cells = v_cells[stream];
    for (uint32_t idx : idxs) {
        if (idx >= cells.size() || cells.is_empty(idx)) {
            continue;
        }
        const uint64_t pid = cells.payload_id_get(idx);
        if (pid == 0) {
            continue;
        }
        const uint64_t gen = cells.storage_generation_get(idx);
        // Deduplicate shared payload IDs: first occurrence owns the release.
        bool seen = false;
        for (uint64_t s : out.store_pids) {
            if (s == pid) { seen = true; break; }
        }
        if (!seen) {
            for (uint64_t h : out.hot_pids) {
                if (h == pid) { seen = true; break; }
            }
        }
        if (seen) {
            continue;
        }
        bool in_pool  = false;
        bool in_store = false;
        uint32_t hot_row = 0;
        llama_xkv::xkv_location loc;
        if (xkv_hot_pool) {
            llama_xkv::xkv_hot_slot_info hinfo;
            if (xkv_hot_pool->find_payload(pid, hinfo)) {
                if (hinfo.storage_generation != gen) {
                    if (err) *err = "collect_removal_release: hot generation mismatch for payload " +
                                    std::to_string(pid);
                    out = xkv_removal_release{};
                    return false;
                }
                in_pool = true;
                hot_row = hinfo.slot;
            }
        }
        if (xkv_store) {
            if (xkv_store->find_location(pid, loc)) {
                if (loc.storage_generation != gen) {
                    if (err) *err = "collect_removal_release: store generation mismatch for payload " +
                                    std::to_string(pid);
                    out = xkv_removal_release{};
                    return false;
                }
                if (loc.state == llama_xkv::xkv_state::seal_candidate && loc.seal_tx_nonce != 0) {
                    if (err) *err = "collect_removal_release: payload " + std::to_string(pid) +
                                    " is locked in an active seal transaction";
                    out = xkv_removal_release{};
                    return false;
                }
                in_store = true;
            }
        }
        if (!in_pool && !in_store) {
            continue; // untracked residue: cells-only cleanup
        }
        if (in_pool) {
            out.hot_pids.push_back(pid);
            out.hot_gens.push_back(gen);
            out.hot_slots.push_back(hot_row);
        }
        if (in_store) {
            out.store_pids.push_back(pid);
            out.store_gens.push_back(gen);
        }
    }
    // Bounded-SR landmark rebuild BEFORE any mutation; false fails closed.
    if (!idxs.empty()) {
        xkv_removal_cells hook_cells;
        for (uint32_t idx : idxs) {
            hook_cells.emplace_back(stream, idx);
        }
        if (!run_removal_rebuild_hook(hook_cells, err)) {
            out = xkv_removal_release{};
            return false;
        }
    }
    // Cross-subsystem preflight with zero mutation.
    if (!out.hot_pids.empty() &&
        !xkv_hot_pool->can_release_batch(out.hot_pids.data(), out.hot_gens.data(),
                                          out.hot_slots.data(), out.hot_pids.size(), err)) {
        out = xkv_removal_release{};
        return false;
    }
    return true;
}

bool llama_kv_cache::execute_removal_release(const xkv_removal_release & rel, std::string * err) {
    if (rel.empty()) {
        return true;
    }
    // True removal gate protocol: the store prepares clones, revalidates,
    // preflights epochs and reserves, then fires our pool gate with the store
    // lock held; after gate-true the store commit is allocation-free and
    // cannot fail. Gate-false (or store-false) leaves store, pool, cells and
    // epochs bit-identical: the pool release inside the gate is single-lock
    // atomic, and the caller mutates cells only after true. No compensation,
    // no best-effort leak normalization.
    auto gate = make_pool_removal_gate(rel.hot_pids, rel.hot_gens, rel.hot_slots);
    if (xkv_store) {
        // Landmark rebuild provider is runtime-owned (wired separately);
        // null fails LANDMARKS removals closed inside the store, which is
        // the correct interim: hot/reference removals proceed.
        if (!xkv_store->remove_payloads(rel.store_pids, rel.store_gens, nullptr, err,
                                         nullptr, nullptr, gate)) {
            return false;
        }
        return true;
    }
    // No store (unbounded without tracking): pool-only release.
    if (!rel.hot_pids.empty()) {
        if (!xkv_hot_pool || !xkv_hot_pool->release_batch(rel.hot_pids.data(), rel.hot_gens.data(),
                                                           rel.hot_slots.data(), rel.hot_pids.size(), err)) {
            return false;
        }
    }
    return true;
}

llama_xkv::xkv_removal_precommit_fn llama_kv_cache::make_pool_removal_gate(
    const std::vector<uint64_t> & pids,
    const std::vector<uint64_t> & gens,
    const std::vector<uint32_t> & slots) const {
    // Pin pool identity for the gate lifetime; the triple is copied.
    auto pool = xkv_hot_pool;
    return [pids, gens, slots, pool](const llama_xkv::xkv_removal_precommit_ctx & ctx,
                                     std::string * err) -> bool {
        try {
            if (!pool || pids.empty()) {
                return true; // nothing pool-owned: SHADOW/unbounded/factored-only
            }
            if (pids.size() != gens.size() || pids.size() != slots.size()) {
                if (err) *err = "pool gate: triple size mismatch";
                return false;
            }
            // Every triple pid must belong to the store's removal set.
            for (uint64_t pid : pids) {
                bool found = false;
                for (uint64_t c : ctx.payload_ids) {
                    if (c == pid) { found = true; break; }
                }
                if (!found) {
                    if (err) *err = "pool gate: payload " + std::to_string(pid) +
                                    " not in store removal set";
                    return false;
                }
            }
            // Single-lock atomic validate + release; zero store calls, so
            // this is safe under the store-held gate lock.
            return pool->release_batch(pids.data(), gens.data(), slots.data(), pids.size(), err);
        } catch (const std::exception & e) {
            if (err) *err = e.what();
            return false;
        } catch (...) {
            if (err) *err = "pool gate: unknown failure";
            return false;
        }
    };
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch, bool dry_run,
                                    xkv_overwrite_victims * victims_out) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    // Phase 0 (bounded, non-dry): collect + preflight victims with zero
    // mutation on rejection. Dry runs (prepare) never touch pool/store.
    xkv_overwrite_victims local_victims;
    xkv_overwrite_victims & victims = victims_out ? *victims_out : local_victims;
    victims.items.clear();
    if (!dry_run && (xkv_hot_pool || xkv_store)) {
        std::string err;
        if (!collect_overwrite_victims(sinfo, victims, &err)) {
            throw std::runtime_error("llama_kv_cache::apply_ubatch: overwrite preflight failed: " + err);
        }
    }

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d() || ubatch.token || hparams.ple_n_heads > 0) {
                llama_kv_cell_ext ext;

                if (ubatch.is_pos_2d()) {
                    ext.x = ubatch.pos[i + ubatch.n_tokens*2];
                    ext.y = ubatch.pos[i + ubatch.n_tokens];
                }

                if (ubatch.token) {
                    ext.tok = ubatch.token[i];
                } else if (hparams.ple_n_heads > 0) {
                    // embd batch (multimodal input) has no token ids; pad for PLE
                    ext.tok = hparams.ple_image_token_id != 0
                        ? (llama_token) hparams.ple_image_token_id
                        : (llama_token) hparams.ple_eos_token_id;
                }

                cells.ext_set(idx, ext);
            }

            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }

            llama_kv_rerot_meta write_tag;
            bool has_write_tag = false;
            for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
                const llama_seq_id seq_id = ubatch.seq_id[i][j];
                GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < rerot_write_tags.size());

                const auto & candidate = rerot_write_tags[seq_id];
                if (!candidate.active()) {
                    continue;
                }

                GGML_ASSERT(!has_write_tag || candidate == write_tag);
                write_tag = candidate;
                has_write_tag = true;
            }

            if (has_write_tag) {
                cells.rerot_set(idx, write_tag);
            }
        }
    }

    if (!dry_run && xkv_store) {
        std::vector<llama_xkv::xkv_hot_payload_binding> bindings;
        bindings.reserve(ubatch.n_tokens);
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const auto & cells = v_cells[sinfo.strm[s]];
            for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
                const auto idx = sinfo.idxs[s][ii];
                const uint32_t hot_row = (!sinfo.hot_idxs.empty() && s < sinfo.hot_idxs.size() && ii < sinfo.hot_idxs[s].size())
                                             ? sinfo.hot_idxs[s][ii] : idx;
                llama_xkv::xkv_hot_payload_binding b;
                b.payload_id = cells.payload_id_get(idx);
                b.hot_slot_row = hot_row;
                b.storage_generation = cells.storage_generation_get(idx);
                b.state = llama_xkv::xkv_state::hot_writing;
                bindings.push_back(b);
            }
        }
        std::string err;
        if (!xkv_store->register_hot_payloads(bindings, &err)) {
            // Roll back newly allocated cells and restore overwritten victim
            // cells. Victim pool/store entries were never released (release is
            // deferred to postcompute_success), so cells repair is sufficient
            // for an exact return to the pre-apply state.
            for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
                auto & cells = v_cells[sinfo.strm[s]];
                for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
                    cells.rm(sinfo.idxs[s][ii]);
                }
            }
            restore_victim_cells(victims);
            victims.items.clear();
            throw std::runtime_error("llama_kv_cache::apply_ubatch: register_hot_payloads failed: " + err);
        }
    }


    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        GGML_ASSERT(s < seq_to_stream.size());

        auto & cells = v_cells[seq_to_stream[s]];

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }
    fp_bump();
}

bool llama_kv_cache::get_can_shift() const {
    // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
    if (model.arch == LLM_ARCH_STEP35) {
        return false;
    }
    if (hparams.n_pos_per_embd() > 1) {
        return false;
    }
    return true;
}

uint32_t llama_kv_cache::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];

    return cells.size();
}

uint32_t llama_kv_cache::get_hot_size() const {
    return hot_kv_size;
}

uint32_t llama_kv_cache::get_kv_hot_capacity() const {
    return hot_kv_size * n_stream;
}

bool llama_kv_cache::can_use_legacy_attention() const {
    return !is_xkv_bounded_hot();
}

bool llama_kv_cache::is_xkv_bounded_hot() const {
    return xkv_dense_sr && (xkv_hot_pool != nullptr);
}

std::shared_ptr<llama_xkv::xkv_hot_slot_pool> llama_kv_cache::get_hot_slot_pool() const {
    return xkv_hot_pool;
}

bool llama_kv_cache::apply_hot_release_plan(
    const llama_xkv::xkv_hot_release_plan & plan,
    std::string * err) {
    if (!xkv_store) {
        if (err) *err = "xkv_store is null";
        return false;
    }
    if (!xkv_store->validate_hot_release_plan(plan, err)) {
        return false;
    }
    if (!xkv_hot_pool) {
        return true; // No hot pool to release from
    }
    return xkv_hot_pool->release_batch(
        plan.released_payload_ids.data(),
        plan.released_generations.data(),
        plan.released_physical_rows.data(),
        plan.released_payload_ids.size(),
        err
    );
}

bool llama_kv_cache::precommit_hot_release(
    const llama_xkv::xkv_hot_release_plan & plan,
    std::string * err) {
    return precommit_hot_release_impl(plan, /*probe_store=*/ true, err);
}

bool llama_kv_cache::precommit_hot_release_nostore(
    const llama_xkv::xkv_hot_release_plan & plan,
    std::string * err) {
    return precommit_hot_release_impl(plan, /*probe_store=*/ false, err);
}

bool llama_kv_cache::validate_hot_release(
    const llama_xkv::xkv_hot_release_plan & plan,
    std::string * err) {
    // SHADOW and unbounded caches never release: validation is vacuous.
    if (!is_xkv_bounded_hot() || !xkv_hot_pool) {
        return true;
    }
    const size_t n = plan.released_payload_ids.size();
    if (n == 0 || plan.released_physical_rows.size() != n ||
        plan.released_generations.size() != n) {
        if (err) *err = "validate_hot_release: plan triple size mismatch or empty";
        return false;
    }
    // Read-only: pool binding+generation, store presence+generation, and
    // unlocked pre-seal rows. Zero mutation; locks are taken and released
    // internally, so the caller must not hold the store lock here.
    for (size_t i = 0; i < n; ++i) {
        const uint64_t pid  = plan.released_payload_ids[i];
        const uint64_t gen  = plan.released_generations[i];
        const uint32_t row  = plan.released_physical_rows[i];
        if (pid == 0) {
            if (err) *err = "validate_hot_release: payload_id is 0 at index " + std::to_string(i);
            return false;
        }
        llama_xkv::xkv_hot_slot_info hinfo;
        if (!xkv_hot_pool->find_payload(pid, hinfo) || hinfo.slot != row ||
            hinfo.storage_generation != gen) {
            if (err) *err = "validate_hot_release: pool binding mismatch for payload " +
                            std::to_string(pid);
            return false;
        }
        if (xkv_store) {
            llama_xkv::xkv_location loc;
            if (!xkv_store->find_location(pid, loc) || loc.storage_generation != gen) {
                if (err) *err = "validate_hot_release: store binding mismatch for payload " +
                                std::to_string(pid);
                return false;
            }
            if (loc.state == llama_xkv::xkv_state::seal_candidate && loc.seal_tx_nonce != 0) {
                if (err) *err = "validate_hot_release: payload " + std::to_string(pid) +
                                " is locked in an active seal transaction";
                return false;
            }
        }
    }
    return true;
}

bool llama_kv_cache::commit_hot_release(
    const llama_xkv::xkv_hot_release_plan & plan,
    std::string * err) {
    // SHADOW and unbounded caches never release.
    if (!is_xkv_bounded_hot() || !xkv_hot_pool) {
        return true;
    }
    const size_t n = plan.released_payload_ids.size();
    if (n == 0 || plan.released_physical_rows.size() != n ||
        plan.released_generations.size() != n) {
        if (err) *err = "commit_hot_release: plan triple size mismatch or empty";
        return false;
    }
    // Pool release only with zero store calls: safe as the seal-gate callback
    // under the store lock. Single-lock atomic all-or-nothing; gate-false
    // leaves every subsystem unmutated.
    return xkv_hot_pool->release_batch(
        plan.released_payload_ids.data(),
        plan.released_generations.data(),
        plan.released_physical_rows.data(),
        n,
        err
    );
}

bool llama_kv_cache::precommit_hot_release_impl(
    const llama_xkv::xkv_hot_release_plan & plan,
    bool probe_store,
    std::string * err) {
    // Standalone convenience: full preflight, then gate-style commit.
    // probe_store=false (seal-gate callback under the store lock) skips all
    // store calls and preflights pool bindings only.
    if (!is_xkv_bounded_hot() || !xkv_hot_pool) {
        return true;
    }
    if (probe_store) {
        if (!validate_hot_release(plan, err)) {
            return false;
        }
    } else {
        const size_t n = plan.released_payload_ids.size();
        if (n == 0 || plan.released_physical_rows.size() != n ||
            plan.released_generations.size() != n) {
            if (err) *err = "precommit_hot_release: plan triple size mismatch or empty";
            return false;
        }
        for (size_t i = 0; i < n; ++i) {
            const uint64_t pid = plan.released_payload_ids[i];
            const uint64_t gen = plan.released_generations[i];
            const uint32_t row = plan.released_physical_rows[i];
            llama_xkv::xkv_hot_slot_info hinfo;
            if (pid == 0 || !xkv_hot_pool->find_payload(pid, hinfo) || hinfo.slot != row ||
                hinfo.storage_generation != gen) {
                if (err) *err = "precommit_hot_release: pool binding mismatch for payload " +
                                std::to_string(pid);
                return false;
            }
        }
    }
    return commit_hot_release(plan, err);
}

uint32_t llama_kv_cache::get_owning_layer(uint32_t model_layer) const {
    auto it = map_layer_ids.find(model_layer);
    if (it != map_layer_ids.end() && (size_t) it->second < layers.size()) {
        return layers[it->second].il;
    }
    return model_layer;
}

uint32_t llama_kv_cache::get_kv_capacity() const {
    // In bounded XKV mode, capacity is the physical hot pool capacity.
    // Server admission/preemption depends on physical hot capacity, not logical metadata.
    if (is_xkv_bounded_hot() && xkv_hot_pool) {
        return xkv_hot_pool->get_capacity();
    }
    uint64_t result = 0;
    for (const auto & cells : v_cells) {
        result += cells.size();
    }
    return (uint32_t) std::min<uint64_t>(result, UINT32_MAX);
}

uint32_t llama_kv_cache::get_kv_used() const {
    // In bounded XKV mode, get_kv_used returns actual physical hot pool live count.
    // AGENTS invariant: physical cells, not logical references/metadata.
    if (is_xkv_bounded_hot() && xkv_hot_pool) {
        return xkv_hot_pool->get_live();
    }
    uint64_t result = 0;
    for (const auto & cells : v_cells) {
        result += cells.get_used();
    }
    return (uint32_t) std::min<uint64_t>(result, UINT32_MAX);
}

uint32_t llama_kv_cache::get_kv_seq_used(llama_seq_id seq_id) const {
    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        return 0;
    }

    const auto & cells = v_cells[seq_to_stream[seq_id]];
    return cells.seq_get_used(seq_id);
}

uint32_t llama_kv_cache::get_n_stream() const {
    return n_stream;
}

bool llama_kv_cache::get_has_shift() const {
    bool result = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        result |= v_cells[s].get_has_shift();
    }

    return result;
}

ggml_type llama_kv_cache::type_k() const {
    return layers[0].k->type;
}

ggml_type llama_kv_cache::type_v() const {
    return layers[0].v->type;
}

ggml_type llama_kv_cache::layer_type_k(int32_t il) const {
    return layers.at(map_layer_ids.at(il)).k->type;
}

ggml_type llama_kv_cache::layer_type_v(int32_t il) const {
    return layers.at(map_layer_ids.at(il)).v->type;
}

ggml_type llama_kv_cache_context::layer_type_k(int32_t il) const {
    return kv->layer_type_k(il);
}

ggml_type llama_kv_cache_context::layer_type_v(int32_t il) const {
    return kv->layer_type_v(il);
}


// FlashPrefill legal-fragment planning (CacheFragments)
//

static bool fp_stamp_usable(const llama_flashprefill_cell_stamp & s) {
    return s.stamp != 0 && s.stamp != std::numeric_limits<uint64_t>::max() &&
           s.generation != std::numeric_limits<uint64_t>::max();
}

void llama_kv_cache::flashprefill_enable_tracking() const {
    // One-time lazy opt-in on the flash path only (OFF never reaches here).
    // Idempotent; the enable itself bumps each stream once, and every build
    // captures stamps afterwards, so no pre-enable state can validate.
    fp_active = true;
    for (auto & cells : v_cells) {
        const_cast<llama_kv_cells &>(cells).set_generation_enabled(true);
    }
}

// Text-pattern partial IMRoPE (Qwen35/Ornith text): n_pos>=3 rows whose extra
// coords replicate the sequence position ([p,p,p] + zero fourth) are exactly
// 1D-equivalent for legality. Only genuinely non-text rows gate out.
static bool fp_row_is_text_pattern(const llama_ubatch & ubatch, uint32_t q) {
    if (ubatch.n_pos < 3) {
        return true;
    }
    if (ubatch.n_pos > 4) {
        return false;
    }
    const uint32_t n = ubatch.n_tokens;
    const llama_pos p = ubatch.pos[q];
    if (ubatch.pos[q + n] != p || ubatch.pos[q + 2 * n] != p) {
        return false;
    }
    if (ubatch.n_pos >= 4 && ubatch.pos[q + 3 * n] != 0) {
        return false;
    }
    return true;
}

// A member cell that can never flip a p0==p1 tiebreak against a text-pattern
// row: the mask skips iff ext.y>P || (==P && ext.x>P), which is exactly the
// complement below. 1D-written cells carry ext {0,0} and always qualify.
static bool fp_cell_text_compatible(const llama_kv_cells & cells, uint32_t idx, llama_pos pos) {
    const llama_kv_cell_ext & ext = cells.ext_get(idx);
    return ext.y < pos || (ext.y == pos && ext.x <= pos);
}

bool llama_kv_cache::flashprefill_tracking_enabled() const {
    for (const auto & cells : v_cells) {
        if (!cells.get_generation_enabled()) {
            return false;
        }
    }
    return true;
}

void llama_kv_cache::flashprefill_cell_stamps(std::vector<llama_flashprefill_cell_stamp> & out) const {
    out.clear();
    out.reserve(v_cells.size());
    for (const auto & cells : v_cells) {
        out.push_back({ cells.get_generation_stamp(), cells.get_generation() });
    }
}

uint32_t llama_kv_cache::flashprefill_n_streams() const {
    return (uint32_t) v_cells.size();
}


bool llama_flashprefill_layout::validate(uint32_t scan_width, std::string * error) const {
    const auto fail = [&](const char * msg) -> bool {
        if (error) { *error = msg; }
        return false;
    };
    if (!eligible) {
        if (!queries.empty() || !fragments.empty() || !cell_refs.empty() || !groups.empty() ||
            !group_offsets.empty() || !uses.empty() || !use_offsets.empty() || !exact_rows.empty() ||
            !exact_groups.empty() || !exact_flags.empty() || !exact_offsets.empty()) {
            return fail("flashprefill layout: ineligible layout carries payload");
        }
        return true;
    }
    const uint64_t sizes[] = {
        queries.size(), fragments.size(), cell_refs.size(), groups.size(), group_offsets.size(),
        uses.size(), use_offsets.size(), exact_rows.size(), exact_groups.size(), exact_flags.size(),
        exact_offsets.size(),
    };
    for (uint64_t s : sizes) {
        if (s > uint64_t(std::numeric_limits<int32_t>::max())) {
            return fail("flashprefill layout: size exceeds I32 wire domain");
        }
    }
    const uint32_t nq = (uint32_t) queries.size();
    if (group_offsets.size() != size_t(nq) + 1 || use_offsets.size() != size_t(nq) + 1) {
        return fail("flashprefill layout: per-query offset size mismatch");
    }
    if (group_offsets.front() != 0 || group_offsets.back() != groups.size()) {
        return fail("flashprefill layout: group offsets inconsistent");
    }
    if (use_offsets.front() != 0 || use_offsets.back() != uses.size()) {
        return fail("flashprefill layout: use offsets inconsistent");
    }
    for (uint32_t q = 0; q < nq; ++q) {
        if (group_offsets[q] > group_offsets[q + 1] || use_offsets[q] > use_offsets[q + 1]) {
            return fail("flashprefill layout: offsets not monotonic");
        }
        for (uint32_t g = group_offsets[q]; g < group_offsets[q + 1]; ++g) {
            if (groups[g].query_index != q) {
                return fail("flashprefill layout: group references another query");
            }
            if (groups[g].effective_pos < 0) {
                return fail("flashprefill layout: negative effective position");
            }
        }
    }
    for (const auto & frag : fragments) {
        if (frag.token_count == 0) {
            return fail("flashprefill layout: empty fragment");
        }
        if (frag.logical_end < frag.logical_begin) {
            return fail("flashprefill layout: fragment logical range inverted");
        }
        if (frag.contiguous) {
            if (frag.cell_ref_offset != UINT32_MAX) {
                return fail("flashprefill layout: contiguous fragment carries ref offset");
            }
            if (uint64_t(frag.cell_begin) + uint64_t(frag.token_count) > scan_width) {
                return fail("flashprefill layout: fragment range out of bounds");
            }
        } else {
            if (uint64_t(frag.cell_ref_offset) + uint64_t(frag.token_count) > cell_refs.size()) {
                return fail("flashprefill layout: fragment refs out of bounds");
            }
            for (uint32_t k = 0; k < frag.token_count; ++k) {
                if (cell_refs[frag.cell_ref_offset + k] >= scan_width) {
                    return fail("flashprefill layout: fragment ref out of bounds");
                }
            }
        }
    }
    // (query, fragment) uniqueness over uses.
    {
        std::vector<uint64_t> keys;
        keys.reserve(uses.size());
        for (const auto & use : uses) {
            if (use.fragment >= fragments.size() || use.query >= nq) {
                return fail("flashprefill layout: use references unknown query/fragment");
            }
            if (use.group >= groups.size() || groups[use.group].query_index != use.query) {
                return fail("flashprefill layout: use references another query's group");
            }
            if (use.sub_count == 0) {
                return fail("flashprefill layout: empty use");
            }
            const auto & frag = fragments[use.fragment];
            if (uint64_t(use.sub_off) + uint64_t(use.sub_count) > frag.token_count) {
                return fail("flashprefill layout: use subrange outside fragment");
            }
            if (use.flags & ~llama_flashprefill_use::FLAG_MANDATORY) {
                return fail("flashprefill layout: use flag out of range");
            }
            keys.push_back((uint64_t(use.query) << 32) | use.fragment);
        }
        std::sort(keys.begin(), keys.end());
        for (size_t i = 1; i < keys.size(); ++i) {
            if (keys[i] == keys[i - 1]) {
                return fail("flashprefill layout: duplicate (query, fragment) use");
            }
        }
    }
    // Oracle exact rows: offset exactness, group ownership, flag domain, per-row dedup.
    if (!exact_offsets.empty()) {
        if (exact_rows.size() != exact_groups.size() || exact_rows.size() != exact_flags.size()) {
            return fail("flashprefill layout: exact payload size mismatch");
        }
        if (exact_offsets.size() != size_t(nq) + 1 || exact_offsets.front() != 0 ||
            exact_offsets.back() != exact_rows.size()) {
            return fail("flashprefill layout: exact offsets inconsistent");
        }
        for (uint32_t q = 0; q < nq; ++q) {
            if (exact_offsets[q] > exact_offsets[q + 1]) {
                return fail("flashprefill layout: exact offsets not monotonic");
            }
            std::vector<uint32_t> row;
            row.reserve(exact_offsets[q + 1] - exact_offsets[q]);
            for (uint32_t e = exact_offsets[q]; e < exact_offsets[q + 1]; ++e) {
                if (exact_rows[e] >= scan_width) {
                    return fail("flashprefill layout: exact row out of bounds");
                }
                if (exact_groups[e] >= groups.size() || groups[exact_groups[e]].query_index != q) {
                    return fail("flashprefill layout: exact row references another query's group");
                }
                if (exact_flags[e] & ~llama_flashprefill_use::FLAG_MANDATORY) {
                    return fail("flashprefill layout: exact flag out of range");
                }
                row.push_back(exact_rows[e]);
            }
            std::sort(row.begin(), row.end());
            for (size_t i = 1; i < row.size(); ++i) {
                if (row[i] == row[i - 1]) {
                    return fail("flashprefill layout: duplicate physical token in query row");
                }
            }
        }
    } else if (!exact_rows.empty() || !exact_groups.empty() || !exact_flags.empty()) {
        return fail("flashprefill layout: exact payload without offsets");
    }
    return true;
}

// Coverage of one fragment's logical span for one query row: 0 = no legal
// member, 1 = partial (a strict non-empty subset), 2 = full. Edge tests only,
// O(1): masking is monotone over the causal prefix for every supported SWA
// shape, and member order is position-ascending, so span endpoints decide.
static int fp_ord_coverage(
        bool causal, uint32_t n_swa, llama_swa_type swa_type,
        llama_pos mn, llama_pos mx, llama_pos qpos) {
    if (causal && mn > qpos) {
        return 0;
    }
    const bool causal_full = !causal || mx <= qpos;
    bool swa_none = false;
    bool swa_full = true;
    if (swa_type != LLAMA_SWA_TYPE_NONE && n_swa != 0) {
        // monotone: masked(min) => all masked; !masked(max) => none masked.
        swa_none = llama_hparams::is_masked_swa(n_swa, swa_type, mn, qpos);
        swa_full = !llama_hparams::is_masked_swa(n_swa, swa_type, mx, qpos);
    }
    if (swa_none) {
        return 0;
    }
    if (causal_full && swa_full) {
        return 2;
    }
    return 1;
}

struct fp_ord_member {
    uint32_t idx = 0;
    llama_pos pos = 0;
};

struct fp_ord_frag_work {
    std::bitset<LLAMA_MAX_SEQ> sig;
    std::vector<fp_ord_member> members; // sorted by (pos, idx) before emission
    llama_pos mn = 0;
    llama_pos mx = -1;
};

// Ordinary-path fragment planning over resident owned cells. Partition key is
// (stream, logical BN block, exact membership signature); members are emitted
// in (pos, idx) order so every query's legal set is one contiguous interval.
llama_flashprefill_build_status llama_flashprefill_build_ordinary_plan(
        const llama_kv_cells_vec & v_cells,
        const std::vector<uint32_t> & seq_to_stream,
        uint32_t n_swa,
        llama_swa_type swa_type,
        bool causal,
        bool require_text_cells,
        const llama_ubatch & ubatch,
        const llama_flashprefill_layout_params & params,
        llama_flashprefill_layout & L,
        std::string * error) {
    const auto hard_error = [&](const std::string & msg) -> llama_flashprefill_build_status {
        if (error) { *error = msg; }
        return llama_flashprefill_build_status::HARD_ERROR;
    };

    const uint32_t n_streams = (uint32_t) v_cells.size();
    if (n_streams == 0) {
        return hard_error("flashprefill layout: no KV streams");
    }

    // Queried primary sequences per stream: the ONLY seqs that can appear in
    // any query row (ubatch seq_id[q][0], matching the KQ-mask and qr.seq_id
    // convention). Membership signatures draw exclusively from these, so
    // foreign idle/private KV never splits fragments, never trips
    // text-compat, and never shifts counts or admission. Unique + validated;
    // unrelated seq refs on a cell are ignored for partitioning.
    std::vector<std::vector<llama_seq_id>> queried(n_streams);
    for (uint32_t q = 0; q < ubatch.n_tokens; ++q) {
        if (ubatch.n_seq_id[q] < 1 || ubatch.seq_id[q] == nullptr) {
            return hard_error("flashprefill layout: query row without sequence");
        }
        const llama_seq_id seq = ubatch.seq_id[q][0];
        if (seq < 0 || (size_t) seq >= seq_to_stream.size() || (size_t) seq >= LLAMA_MAX_SEQ) {
            return hard_error("flashprefill layout: query sequence out of range");
        }
        const uint32_t stream = seq_to_stream[seq];
        if (stream >= n_streams) {
            return hard_error("flashprefill layout: query stream out of range");
        }
        auto & vec = queried[stream];
        if (std::find(vec.begin(), vec.end(), seq) == vec.end()) {
            vec.push_back(seq);
        }
    }

    // Partition. Outer vector per stream; inner map BN block -> frags by sig.
    std::vector<std::map<uint32_t, std::vector<fp_ord_frag_work>>> table(n_streams);
    for (uint32_t s = 0; s < n_streams; ++s) {
        const auto & cells = v_cells[s];
        for (uint32_t idx = 0; idx < cells.size(); ++idx) {
            if (cells.is_empty(idx)) {
                continue;
            }
            const llama_pos pos = cells.pos_get(idx);
            if (pos < 0) {
                return hard_error("flashprefill layout: resident cell with negative position");
            }
            std::bitset<LLAMA_MAX_SEQ> sig;
            for (const llama_seq_id seq : queried[s]) {
                if (cells.seq_has(idx, seq)) {
                    sig.set((size_t) seq);
                }
            }
            if (sig.none()) {
                continue; // referenced by no queried sequence: invisible to
                          // every row; skipped BEFORE any text/layout check so
                          // foreign content can neither gate nor split
            }
            if (require_text_cells && !fp_cell_text_compatible(cells, idx, pos)) {
                // Image-pattern resident content: the stock KQ mask applies a
                // 2-D tiebreak this 1D-equivalent legality cannot reproduce.
                if (error) { *error = "flashprefill layout: image-pattern cell keeps the stock path"; }
                return llama_flashprefill_build_status::INELIGIBLE;
            }
            const uint32_t block = uint32_t(pos) / params.block_k;
            auto & bucket = table[s][block];
            fp_ord_frag_work * work = nullptr;
            for (auto & cand : bucket) {
                if (cand.sig == sig) {
                    work = &cand;
                    break;
                }
            }
            if (!work) {
                bucket.push_back(fp_ord_frag_work{});
                work = &bucket.back();
                work->sig = sig;
                work->mn = pos;
                work->mx = pos;
            }
            work->members.push_back({ idx, pos });
            if (pos < work->mn) { work->mn = pos; }
            if (pos > work->mx) { work->mx = pos; }
        }
    }

    // Emit fragments stream-major. Members sorted by (pos, idx); contiguous
    // range form only when idx-consecutive in that order (positions then
    // ascend by construction).
    std::vector<uint32_t> stream_begin(n_streams + 1, 0);
    for (uint32_t s = 0; s < n_streams; ++s) {
        stream_begin[s] = (uint32_t) L.fragments.size();
        for (auto & block_it : table[s]) {
            for (auto & work : block_it.second) {
                auto & members = work.members;
                std::sort(members.begin(), members.end(), [](const fp_ord_member & a, const fp_ord_member & b) {
                    if (a.pos != b.pos) { return a.pos < b.pos; }
                    return a.idx < b.idx;
                });
                if (work.mx == std::numeric_limits<llama_pos>::max()) {
                    return hard_error("flashprefill layout: logical position at INT32_MAX");
                }
                llama_flashprefill_fragment frag;
                frag.domain = llama_flashprefill_fragment_domain::ORDINARY;
                frag.stream = s;
                frag.boundary_partial = false;
                frag.token_count = (uint32_t) members.size();
                frag.logical_block = block_it.first;
                frag.logical_begin = work.mn;
                frag.logical_end = work.mx + 1;
                frag.members = work.sig;
                frag.gated = true;
                bool contiguous = true;
                for (size_t k = 1; k < members.size(); ++k) {
                    if (members[k].idx != members[k - 1].idx + 1) {
                        contiguous = false;
                        break;
                    }
                }
                if (contiguous) {
                    frag.contiguous = true;
                    frag.cell_begin = members.front().idx;
                    frag.cell_ref_offset = UINT32_MAX;
                } else {
                    frag.contiguous = false;
                    frag.cell_begin = 0;
                    frag.cell_ref_offset = (uint32_t) L.cell_refs.size();
                    for (const auto & m : members) {
                        L.cell_refs.push_back(m.idx);
                    }
                }
                L.fragments.push_back(frag);
            }
        }
    }
    stream_begin[n_streams] = (uint32_t) L.fragments.size();

    // Queries.
    for (uint32_t q = 0; q < ubatch.n_tokens; ++q) {
        if (ubatch.n_seq_id[q] < 1 || ubatch.seq_id[q] == nullptr) {
            return hard_error("flashprefill layout: query row without sequence");
        }
        const llama_seq_id seq = ubatch.seq_id[q][0];
        if (seq < 0 || (size_t) seq >= seq_to_stream.size()) {
            return hard_error("flashprefill layout: query sequence out of range");
        }
        const uint32_t stream = seq_to_stream[seq];
        if (stream >= n_streams) {
            return hard_error("flashprefill layout: query stream out of range");
        }
        const llama_pos qpos = ubatch.pos[q];
        if (qpos < 0) {
            return hard_error("flashprefill layout: query with negative position");
        }
        llama_flashprefill_query qr;
        qr.query_index = q;
        qr.seq_id = seq;
        qr.stream = stream;
        qr.query_pos = qpos;
        qr.query_virtual_pos = -1;
        L.queries.push_back(qr);
    }

    // Per-query emission: one group + uses over the query's stream range.
    L.group_offsets.push_back(0);
    L.use_offsets.push_back(0);
    if (params.want_exact_rows) {
        L.exact_offsets.push_back(0);
    }
    uint64_t exact_total = 0;
    for (uint32_t q = 0; q < (uint32_t) L.queries.size(); ++q) {
        const auto & qr = L.queries[q];
        llama_rerot_attn_group group;
        group.query_index = q;
        group.effective_pos = qr.query_pos;
        L.groups.push_back(group);
        const uint32_t gid = (uint32_t) L.groups.size() - 1;
        L.group_offsets.push_back((uint32_t) L.groups.size());

        for (uint32_t f = stream_begin[qr.stream]; f < stream_begin[qr.stream + 1]; ++f) {
            auto & frag = L.fragments[f];
            if (!frag.members.test((size_t) qr.seq_id)) {
                continue;
            }
            const int cov = fp_ord_coverage(
                causal, n_swa, swa_type, frag.logical_begin, frag.logical_end - 1, qr.query_pos);
            if (cov == 0) {
                continue;
            }
            uint32_t sub_off = 0;
            uint32_t sub_count = frag.token_count;
            uint32_t flags = 0;
            if (cov == 1) {
                // Interval scan over member order (position-ascending): the
                // legal set must be one contiguous interval; anything else is
                // an ordering bug and fails closed.
                const auto & cells = v_cells[qr.stream];
                uint32_t first = frag.token_count;
                uint32_t last = frag.token_count;
                bool seen_illegal = false;
                for (uint32_t k = 0; k < frag.token_count; ++k) {
                    const uint32_t idx = frag.contiguous ? frag.cell_begin + k
                                                         : L.cell_refs[frag.cell_ref_offset + k];
                    const llama_pos p = cells.pos_get(idx);
                    const bool legal = (!causal || p <= qr.query_pos) &&
                        (swa_type == LLAMA_SWA_TYPE_NONE || n_swa == 0 ||
                         !llama_hparams::is_masked_swa(n_swa, swa_type, p, qr.query_pos));
                    if (legal) {
                        if (seen_illegal) {
                            return hard_error("flashprefill layout: non-interval legal subset");
                        }
                        if (first == frag.token_count) {
                            first = k;
                        }
                        last = k;
                    } else {
                        seen_illegal = first != frag.token_count;
                    }
                }
                if (first == frag.token_count) {
                    return hard_error("flashprefill layout: partial fragment with empty subset");
                }
                sub_off = first;
                sub_count = last - first + 1;
                flags = llama_flashprefill_use::FLAG_MANDATORY;
                frag.boundary_partial = true;
            }
            llama_flashprefill_use use;
            use.query = q;
            use.fragment = f;
            use.group = gid;
            use.sub_off = sub_off;
            use.sub_count = sub_count;
            use.flags = flags;
            L.uses.push_back(use);

            if (params.want_exact_rows) {
                if (exact_total + sub_count > params.exact_cap) {
                    return hard_error("flashprefill layout: exact row capacity exceeded");
                }
                const auto & cells = v_cells[qr.stream];
                for (uint32_t k = 0; k < sub_count; ++k) {
                    const uint32_t idx = frag.contiguous ? frag.cell_begin + sub_off + k
                                                         : L.cell_refs[frag.cell_ref_offset + sub_off + k];
                    (void) cells;
                    L.exact_rows.push_back(idx);
                    L.exact_groups.push_back(gid);
                    L.exact_flags.push_back(flags);
                }
                exact_total += sub_count;
            }
        }
        L.use_offsets.push_back((uint32_t) L.uses.size());
        if (params.want_exact_rows) {
            L.exact_offsets.push_back((uint32_t) L.exact_rows.size());
        }
    }
    return llama_flashprefill_build_status::OK;
}

struct fp_rerot_view_key {
    uint64_t episode = 0;
    llama_rerot_node_id reader = LLAMA_REROT_NODE_INVALID;
    llama_rerot_run_id query_run = LLAMA_REROT_RUN_INVALID;
    uint64_t frontier = 0;
    uint64_t topology_epoch = 0;
    uint64_t publish_epoch = 0;
    uint64_t layout_epoch = 0;
    llama_rerot_frontier_mode frontier_mode = LLAMA_REROT_FRONTIER_STRONG;
    std::vector<llama_rerot_run_id> ordered_runs;

    bool operator==(const fp_rerot_view_key & o) const {
        return episode == o.episode && reader == o.reader && query_run == o.query_run &&
               frontier == o.frontier && topology_epoch == o.topology_epoch &&
               publish_epoch == o.publish_epoch && layout_epoch == o.layout_epoch &&
               frontier_mode == o.frontier_mode && ordered_runs == o.ordered_runs;
    }
};

struct fp_tagged_member {
    uint32_t idx = 0;
    llama_pos storage = 0;
    uint64_t frontier = 0;
    llama_rerot_visibility vis = llama_rerot_visibility::normal;
    bool gated = true;
    std::bitset<LLAMA_MAX_SEQ> sig;
};

struct fp_base_member {
    uint32_t idx = 0;
    llama_pos storage = 0;
    std::bitset<LLAMA_MAX_SEQ> sig;
};

// RERoT-path fragment planning. One reusable run/span table per distinct
// reader view (O(K) each); per-query work then touches fragments only
// (O(Q*F)). Visibility mirrors llama_rerot_build_query_layout exactly: FULL
// members (position-independent public) are legal for every sharing query,
// gated members (base, private/pending, own-node frontier-equal public) keep
// per-query ownership+causal checks. The old per-query full-KV expansion is
// never built on this path.
llama_flashprefill_build_status llama_flashprefill_build_rerot_plan(
        const llama_kv_cells & cells,
        const std::vector<llama_rerot_reader_state> & views,
        const llama_ubatch & ubatch,
        const llama_flashprefill_layout_params & params,
        llama_flashprefill_layout & L,
        std::string * error) {
    const auto hard_error = [&](const std::string & msg) -> llama_flashprefill_build_status {
        if (error) { *error = msg; }
        return llama_flashprefill_build_status::HARD_ERROR;
    };

    // Live sequences for ownership signatures (unified stream).
    std::vector<llama_seq_id> live;
    for (llama_seq_id seq = 0; seq < LLAMA_MAX_SEQ; ++seq) {
        if (cells.seq_get_used(seq) > 0) {
            live.push_back(seq);
        }
    }
    const auto member_sig = [&](uint32_t idx) {
        std::bitset<LLAMA_MAX_SEQ> sig;
        for (const llama_seq_id seq : live) {
            if (cells.seq_has(idx, seq)) {
                sig.set((size_t) seq);
            }
        }
        return sig;
    };

    struct view_group {
        fp_rerot_view_key key;
        const llama_rerot_reader_state * view = nullptr;
        std::vector<uint32_t> rows; // ubatch rows, in order
    };
    std::vector<view_group> vgroups;
    for (uint32_t q = 0; q < ubatch.n_tokens; ++q) {
        if (ubatch.n_seq_id[q] < 1 || ubatch.seq_id[q] == nullptr) {
            return hard_error("flashprefill layout: query row without sequence");
        }
        const llama_seq_id seq = ubatch.seq_id[q][0];
        if (seq < 0 || (size_t) seq >= views.size() || !views[seq].active()) {
            return hard_error("flashprefill layout: query without active reader view");
        }
        const auto & view = views[seq];
        fp_rerot_view_key key;
        key.episode = view.episode_id;
        key.reader = view.reader;
        key.query_run = view.query_run;
        key.frontier = view.frontier;
        key.topology_epoch = view.topology_epoch;
        key.publish_epoch = view.publish_epoch;
        key.layout_epoch = view.layout_epoch;
        key.frontier_mode = view.frontier_mode;
        key.ordered_runs = view.ordered_runs;
        view_group * group = nullptr;
        for (auto & cand : vgroups) {
            if (cand.key == key) {
                group = &cand;
                break;
            }
        }
        if (!group) {
            vgroups.push_back(view_group{});
            group = &vgroups.back();
            group->key = std::move(key);
            group->view = &view;
        }
        group->rows.push_back(q);
    }

    struct frag_range {
        uint32_t begin = 0;
        uint32_t end = 0;
    };
    std::vector<frag_range> vgroup_frags;
    vgroup_frags.reserve(vgroups.size());
    std::vector<std::map<llama_seq_id, std::unordered_map<llama_pos, std::pair<uint32_t, uint32_t>>>> own_by_vgroup;
    own_by_vgroup.reserve(vgroups.size());

    // Reusable table per distinct view.
    for (const auto & vg : vgroups) {
        const auto & view = *vg.view;
        const uint32_t fbegin = (uint32_t) L.fragments.size();
        // Physical key -> (fragment, fragment-local index), for own-cell
        // lookup without rescanning tables per query.
        std::vector<uint32_t> key_frag(cells.size(), UINT32_MAX);
        std::vector<uint32_t> key_local(cells.size(), UINT32_MAX);

        std::unordered_map<llama_rerot_run_id, uint32_t> rank;
        rank.reserve(view.ordered_runs.size());
        for (uint32_t r = 0; r < view.ordered_runs.size(); ++r) {
            const auto run = view.ordered_runs[r];
            if (run == LLAMA_REROT_RUN_INVALID || !rank.emplace(run, r).second) {
                return hard_error("flashprefill layout: reader view with invalid/duplicate run");
            }
        }

        std::vector<fp_base_member> base;
        std::vector<std::vector<fp_tagged_member>> runs(view.ordered_runs.size());
        for (uint32_t idx = 0; idx < cells.size(); ++idx) {
            if (cells.is_empty(idx)) {
                continue;
            }
            const llama_pos storage = cells.pos_get(idx);
            if (storage < 0) {
                return hard_error("flashprefill layout: resident cell with negative position");
            }
            const auto & meta = cells.rerot_get(idx);
            if (!meta.active()) {
                auto sig = member_sig(idx);
                if (sig.none()) {
                    continue;
                }
                base.push_back({ idx, storage, sig });
                continue;
            }
            if (meta.episode_id != view.episode_id) {
                continue;
            }
            const auto rank_it = rank.find(meta.run_id);
            if (rank_it == rank.end()) {
                continue;
            }
            const bool full = llama_rerot_cell_visible_public_full(meta, view);
            bool maybe = false;
            if (!full) {
                if (meta.visibility == llama_rerot_visibility::public_live) {
                    maybe = meta.frontier == view.frontier && meta.node_id == view.reader;
                } else if (meta.visibility == llama_rerot_visibility::private_control ||
                           meta.visibility == llama_rerot_visibility::pending_record) {
                    maybe = meta.node_id == view.reader;
                }
            }
            if (!full && !maybe) {
                continue;
            }
            runs[rank_it->second].push_back({ idx, storage, meta.frontier, meta.visibility, !full, member_sig(idx) });
        }

        auto by_storage = [](const auto & a, const auto & b) {
            if (a.storage != b.storage) { return a.storage < b.storage; }
            return a.idx < b.idx;
        };
        std::sort(base.begin(), base.end(), by_storage);
        for (auto & run : runs) {
            std::sort(run.begin(), run.end(), [](const fp_tagged_member & a, const fp_tagged_member & b) {
                if (a.storage != b.storage) { return a.storage < b.storage; }
                if (a.frontier != b.frontier) { return a.frontier < b.frontier; }
                return a.idx < b.idx;
            });
        }

        // Dense virtualization: base first, then runs in view order.
        if (base.size() > uint64_t(std::numeric_limits<llama_pos>::max()) + 1u) {
            return hard_error("flashprefill layout: base exceeds llama_pos range");
        }
        std::vector<llama_pos> run_v0(runs.size(), 0);
        int64_t cursor = (int64_t) base.size();
        for (size_t r = 0; r < runs.size(); ++r) {
            if (cursor > std::numeric_limits<llama_pos>::max()) {
                return hard_error("flashprefill layout: virtual positions exceed llama_pos range");
            }
            run_v0[r] = (llama_pos) cursor;
            cursor += (int64_t) runs[r].size();
        }
        if (cursor > int64_t(std::numeric_limits<llama_pos>::max()) + 1) {
            return hard_error("flashprefill layout: virtual positions exceed llama_pos range");
        }

        const auto emit_members = [&](llama_flashprefill_fragment & frag, const std::vector<uint32_t> & idxs) {
            frag.token_count = (uint32_t) idxs.size();
            bool contiguous = true;
            for (size_t k = 1; k < idxs.size(); ++k) {
                if (idxs[k] != idxs[k - 1] + 1) {
                    contiguous = false;
                    break;
                }
            }
            if (contiguous) {
                frag.contiguous = true;
                frag.cell_begin = idxs.front();
                frag.cell_ref_offset = UINT32_MAX;
            } else {
                frag.contiguous = false;
                frag.cell_begin = 0;
                frag.cell_ref_offset = (uint32_t) L.cell_refs.size();
                for (uint32_t idx : idxs) {
                    L.cell_refs.push_back(idx);
                }
            }
        };

        // Base fragments: cut on membership-signature, virtual-BN, and phase-bias changes.
        {
            size_t start = 0;
            const auto cut_before = [&](size_t i) {
                if (base[i].sig != base[i - 1].sig) { return true; }
                if (uint64_t(i) / params.block_k != uint64_t(i - 1) / params.block_k) { return true; }
                return (int64_t(base[i].storage) - int64_t(i)) != (int64_t(base[i - 1].storage) - int64_t(i - 1));
            };
            const auto emit_base = [&](size_t b, size_t e) {
                std::vector<uint32_t> idxs;
                idxs.reserve(e - b);
                for (size_t k = b; k < e; ++k) { idxs.push_back(base[k].idx); }
                llama_flashprefill_fragment frag;
                frag.domain = llama_flashprefill_fragment_domain::REROT_BASE;
                frag.stream = 0;
                frag.boundary_partial = false;
                frag.logical_block = uint32_t(uint64_t(b) / params.block_k);
                frag.logical_begin = (llama_pos) b;
                frag.logical_end = (llama_pos) e;
                frag.members = base[b].sig;
                frag.episode_id = view.episode_id;
                frag.virtual_pos0 = (llama_pos) b;
                // Table-phase constant P = storage - piece-local index. The
                // cut keeps (storage - base_index) fixed over base indices,
                // so P == base[b].storage exactly.
                frag.phase_bias = int64_t(base[b].storage);
                frag.gated = true;
                emit_members(frag, idxs);
                {
                    const uint32_t new_frag = (uint32_t) L.fragments.size();
                    for (size_t k = 0; k < idxs.size(); ++k) {
                        key_frag[idxs[k]] = new_frag;
                        key_local[idxs[k]] = (uint32_t) k;
                    }
                }
                L.fragments.push_back(frag);
            };
            for (size_t i = 1; i <= base.size(); ++i) {
                if (i == base.size() || cut_before(i)) {
                    if (i > start) { emit_base(start, i); }
                    start = i;
                }
            }
        }

        // Run fragments: reusable split once per run, then membership post-split for gated pieces.
        for (size_t r = 0; r < runs.size(); ++r) {
            const auto & run = runs[r];
            if (run.empty()) {
                continue;
            }
            std::vector<llama_rerot_table_member> tm;
            tm.reserve(run.size());
            for (const auto & t : run) {
                tm.push_back({ t.idx, t.storage, t.frontier, t.vis, t.gated });
            }
            std::vector<llama_rerot_table_fragment> splits;
            try {
                splits = llama_rerot_split_table_fragments(tm.data(), tm.size(), run_v0[r], params.block_k);
            } catch (const std::exception & e) {
                return hard_error(std::string("flashprefill layout: run split failed: ") + e.what());
            }
            for (const auto & sp : splits) {
                size_t pstart = sp.begin;
                while (pstart < sp.end) {
                    size_t pend = pstart + 1;
                    if (sp.gated) {
                        while (pend < sp.end && run[pend].sig == run[pstart].sig) { ++pend; }
                    } else {
                        pend = sp.end;
                    }
                    std::vector<uint32_t> idxs;
                    idxs.reserve(pend - pstart);
                    for (size_t k = pstart; k < pend; ++k) { idxs.push_back(run[k].idx); }
                    const llama_pos vp0 = run_v0[r] + (llama_pos) pstart;
                    llama_flashprefill_fragment frag;
                    frag.domain = llama_flashprefill_fragment_domain::REROT_RUN;
                    frag.stream = 0;
                    frag.boundary_partial = false;
                    frag.logical_block = uint32_t(uint64_t(vp0) / params.block_k);
                    frag.logical_begin = vp0;
                    frag.logical_end = vp0 + (llama_pos) idxs.size();
                    if (sp.gated) { frag.members = run[pstart].sig; }
                    frag.episode_id = view.episode_id;
                    frag.run_id = view.ordered_runs[r];
                    frag.visibility = sp.visibility;
                    frag.run_rank = (uint32_t) r;
                    frag.virtual_pos0 = vp0;
                    // Table-phase constant P = split bias + run table origin
                    // + piece start: storage minus piece-local index.
                    frag.phase_bias = sp.phase_bias + int64_t(run_v0[r]) + int64_t(pstart);
                    frag.gated = sp.gated;
                    emit_members(frag, idxs);
                    {
                        const uint32_t new_frag = (uint32_t) L.fragments.size();
                        for (size_t k = 0; k < idxs.size(); ++k) {
                            key_frag[idxs[k]] = new_frag;
                            key_local[idxs[k]] = (uint32_t) k;
                        }
                    }
                    L.fragments.push_back(frag);
                    pstart = pend;
                }
            }
        }

        vgroup_frags.push_back({ fbegin, (uint32_t) L.fragments.size() });

        // Per-(query-seq) data inside this view: ownership is fixed, so the
        // own-cell last-match locations are built once per seq, not per query.
        // Map: query storage -> (fragment, fragment-local index), overwrite
        // wins, replicating the old builder's last-match scan over global
        // order. Per-query virtuals are derived from these in the emission
        // pass below, AFTER that query's own filtering — never from fixed
        // table virtuals.
        std::map<llama_seq_id, std::vector<uint32_t>> rows_by_seq;
        for (uint32_t q : vg.rows) {
            rows_by_seq[ubatch.seq_id[q][0]].push_back(q);
        }
        std::map<llama_seq_id, std::unordered_map<llama_pos, std::pair<uint32_t, uint32_t>>> own_by_seq;
        for (const auto & seq_rows : rows_by_seq) {
            const llama_seq_id qseq = seq_rows.first;
            if (qseq < 0 || (size_t) qseq >= views.size()) {
                return hard_error("flashprefill layout: query sequence out of range");
            }
            const auto rank_qr = rank.find(view.query_run);
            if (rank_qr == rank.end()) {
                return hard_error("flashprefill layout: query run absent from reader view");
            }
            std::unordered_map<llama_pos, std::pair<uint32_t, uint32_t>> own_match;
            const auto & qrun = runs[rank_qr->second];
            for (size_t j = 0; j < qrun.size(); ++j) {
                const uint32_t idx = qrun[j].idx;
                const auto & meta = cells.rerot_get(idx);
                if (meta.node_id == view.reader && cells.seq_has(idx, qseq)) {
                    own_match[qrun[j].storage] = { key_frag[idx], key_local[idx] };
                }
            }
            own_by_seq[qseq] = std::move(own_match);
            for (uint32_t q : seq_rows.second) {
                const llama_pos qpos = ubatch.pos[q];
                if (qpos < 0) {
                    return hard_error("flashprefill layout: query with negative position");
                }
                llama_flashprefill_query qr;
                qr.query_index = q;
                qr.seq_id = qseq;
                qr.stream = 0;
                qr.query_pos = qpos;
                qr.query_virtual_pos = -1; // filled per query in the emission pass
                L.queries.push_back(qr);
            }
        }
        own_by_vgroup.push_back(std::move(own_by_seq));
    }

    // Queries were appended view-major; restore ubatch order for dense
    // per-query ranges (groups/uses stay query-indexed either way).
    std::sort(L.queries.begin(), L.queries.end(), [](const llama_flashprefill_query & a, const llama_flashprefill_query & b) {
        return a.query_index < b.query_index;
    });

    // Per-query emission over the query's own view range.
    L.group_offsets.push_back(0);
    L.use_offsets.push_back(0);
    if (params.want_exact_rows) {
        L.exact_offsets.push_back(0);
    }
    uint64_t exact_total = 0;
    // ubatch row -> vgroup index.
    std::vector<uint32_t> row_vgroup(ubatch.n_tokens, UINT32_MAX);
    for (uint32_t v = 0; v < vgroups.size(); ++v) {
        for (uint32_t q : vgroups[v].rows) { row_vgroup[q] = v; }
    }
    for (uint32_t qi = 0; qi < (uint32_t) L.queries.size(); ++qi) {
        auto & qr = L.queries[qi];
        const uint32_t v = row_vgroup[qr.query_index];
        std::map<llama_pos, uint32_t> eff2group;
        const auto group_for = [&](int64_t eff, std::string & msg) -> int64_t {
            if (eff < 0 || eff > std::numeric_limits<llama_pos>::max()) {
                msg = "flashprefill layout: effective query position out of range";
                return -1;
            }
            const llama_pos e = (llama_pos) eff;
            const auto it = eff2group.find(e);
            if (it != eff2group.end()) { return (int64_t) it->second; }
            llama_rerot_attn_group group;
            group.query_index = qi;
            group.effective_pos = e;
            L.groups.push_back(group);
            const uint32_t gid = (uint32_t) L.groups.size() - 1;
            eff2group[e] = gid;
            return (int64_t) gid;
        };
        // Pass A: legality + legal counts in global fragment order. This is
        // the per-query filtering the old builder performs BEFORE dense
        // virtualization: base ownership, gated causal edges, and future-key
        // exclusion all shape the virtual address space of THIS query.
        const uint32_t use_begin = (uint32_t) L.uses.size();
        int64_t total = 0;
        for (uint32_t f = vgroup_frags[v].begin; f < vgroup_frags[v].end; ++f) {
            auto & frag = L.fragments[f];
            uint32_t sub_off = 0;
            uint32_t sub_count = frag.token_count;
            uint32_t flags = 0;
            if (!frag.gated) {
                // FULL: visible to every sharing query, whole fragment legal.
            } else {
                if (!frag.members.test((size_t) qr.seq_id)) {
                    continue;
                }
                // Storage span: smin is the table-phase constant (first
                // member storage exactly); smax is ground truth off the last
                // member. Gaps inside the span only shrink the legal prefix.
                const int64_t smin = frag.phase_bias;
                const uint32_t last_idx = frag.contiguous
                    ? frag.cell_begin + frag.token_count - 1
                    : L.cell_refs[frag.cell_ref_offset + frag.token_count - 1];
                const int64_t smax = cells.pos_get(last_idx);
                if (smin > qr.query_pos) {
                    continue;
                }
                if (smax > qr.query_pos) {
                    // Prefix scan in storage order: legal prefix then illegal
                    // tail; anything else fails closed.
                    uint32_t k = 0;
                    for (; k < frag.token_count; ++k) {
                        const uint32_t idx = frag.contiguous ? frag.cell_begin + k
                                                             : L.cell_refs[frag.cell_ref_offset + k];
                        if (cells.pos_get(idx) > qr.query_pos) { break; }
                    }
                    for (uint32_t t = k; t < frag.token_count; ++t) {
                        const uint32_t idx = frag.contiguous ? frag.cell_begin + t
                                                             : L.cell_refs[frag.cell_ref_offset + t];
                        if (cells.pos_get(idx) <= qr.query_pos) {
                            return hard_error("flashprefill layout: non-interval gated subset");
                        }
                    }
                    if (k == 0) {
                        return hard_error("flashprefill layout: gated fragment with empty subset");
                    }
                    sub_count = k;
                    flags = llama_flashprefill_use::FLAG_MANDATORY;
                    frag.boundary_partial = true;
                }
            }
            llama_flashprefill_use use;
            use.query = qi;
            use.fragment = f;
            use.group = UINT32_MAX; // assigned in pass B once qvirt is known
            use.sub_off = sub_off;
            use.sub_count = sub_count;
            use.flags = flags;
            L.uses.push_back(use);
            total += sub_count;
        }
        // Own virtual position: prefix count of legal members before the
        // own fragment plus the fragment-local index — or the legal total
        // when the query has no resident own cell (read-only refresh). The
        // uses are fragment-ordered, so one linear prefix scan suffices.
        int64_t qvirt = total;
        {
            const auto & own_by_seq = own_by_vgroup[v];
            const auto seq_it = own_by_seq.find(qr.seq_id);
            if (seq_it != own_by_seq.end()) {
                const auto oit = seq_it->second.find(qr.query_pos);
                if (oit != seq_it->second.end()) {
                    const uint32_t frag_o = oit->second.first;
                    const uint32_t local_o = oit->second.second;
                    int64_t c = 0;
                    bool found = false;
                    for (uint32_t u = use_begin; u < (uint32_t) L.uses.size(); ++u) {
                        const auto & uu = L.uses[u];
                        if (uu.fragment < frag_o) {
                            c += uu.sub_count;
                            continue;
                        }
                        if (uu.fragment == frag_o) {
                            found = true;
                        }
                        break;
                    }
                    if (!found) {
                        return hard_error("flashprefill layout: own cell without covering use");
                    }
                    qvirt = c + local_o;
                }
            }
        }
        if (qvirt > std::numeric_limits<llama_pos>::max()) {
            return hard_error("flashprefill layout: query virtual position out of range");
        }
        qr.query_virtual_pos = (llama_pos) qvirt;
        // Pass B: phase groups + oracle expansion. Effective position of a
        // use is qvirt + P - C, where P is the fragment's table-phase
        // constant (storage minus piece-local index, uniform by split) and C
        // is the legal prefix count before the fragment for THIS query.
        {
            int64_t c = 0;
            for (uint32_t u = use_begin; u < (uint32_t) L.uses.size(); ++u) {
                auto & uu = L.uses[u];
                const auto & frag = L.fragments[uu.fragment];
                std::string gmsg;
                const int64_t gid = group_for(qvirt + frag.phase_bias - c, gmsg);
                if (gid < 0) {
                    return hard_error(gmsg);
                }
                uu.group = (uint32_t) gid;
                if (params.want_exact_rows) {
                    if (exact_total + uu.sub_count > params.exact_cap) {
                        return hard_error("flashprefill layout: exact row capacity exceeded");
                    }
                    for (uint32_t k = 0; k < uu.sub_count; ++k) {
                        const uint32_t idx = frag.contiguous ? frag.cell_begin + uu.sub_off + k
                                                             : L.cell_refs[frag.cell_ref_offset + uu.sub_off + k];
                        L.exact_rows.push_back(idx);
                        L.exact_groups.push_back((uint32_t) gid);
                        L.exact_flags.push_back(uu.flags);
                    }
                    exact_total += uu.sub_count;
                }
                c += uu.sub_count;
            }
        }
        L.group_offsets.push_back((uint32_t) L.groups.size());
        L.use_offsets.push_back((uint32_t) L.uses.size());
        if (params.want_exact_rows) {
            L.exact_offsets.push_back((uint32_t) L.exact_rows.size());
        }
    }
    return llama_flashprefill_build_status::OK;
}

bool llama_kv_cache::flashprefill_build_layout(
        const llama_ubatch & ubatch,
        int32_t role,
        const llama_flashprefill_layout_params & params,
        llama_flashprefill_layout & out,
        std::string * error) const {
    out.clear();
    const auto hard_error = [&](const std::string & msg) -> bool {
        out.clear();
        out.eligible = false;
        out.error = msg;
        if (error) { *error = msg; }
        return false;
    };
    const auto ineligible = [&](int32_t reason, const std::string & msg) -> bool {
        out.clear();
        out.eligible = false;
        out.dense_reason = reason;
        if (error) { *error = msg; }
        return false;
    };

    if (params.block_k == 0) {
        return hard_error("flashprefill layout: block_k must be non-zero");
    }
    if (ubatch.n_tokens == 0 || ubatch.pos == nullptr || ubatch.seq_id == nullptr || ubatch.n_seq_id == nullptr) {
        return hard_error("flashprefill layout: ubatch carries no positions/sequences");
    }
    // Qwen35/Ornith text partial IMRoPE (n_pos==4, [p,p,p,0]) is 1D-equivalent
    // and stays eligible; only genuinely non-text rows/cells gate out below.
    // The RERoT indexed path consumes the scalar slot-0 coordinate exactly
    // like the old builder, so it needs no pattern gate for oracle equality.
    const bool ubatch_2d = ubatch.is_pos_2d();
    if (hparams.use_alibi) {
        return ineligible(LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED,
            "flashprefill layout: ALiBi bias keeps the stock path");
    }
    const bool want_rerot = (role == LLAMA_FLASHPREFILL_ROLE_REROT_TEACHER_FORCED);
    if (role != LLAMA_FLASHPREFILL_ROLE_PREFILL && !want_rerot) {
        return ineligible(LLAMA_FLASHPREFILL_ROUTE_DENSE_ROLE,
            "flashprefill layout: role is not prefill-eligible");
    }
    // Existing mixed-ubatch contract: throws when RERoT and ordinary rows mix.
    const bool rerot_on = rerot_batch_active(ubatch);
    if (want_rerot && !rerot_on) {
        return ineligible(LLAMA_FLASHPREFILL_ROUTE_DENSE_ROLE,
            "flashprefill layout: ordinary ubatch with RERoT-teacher role");
    }
    if (!want_rerot && rerot_on) {
        return ineligible(LLAMA_FLASHPREFILL_ROUTE_DENSE_ROLE,
            "flashprefill layout: RERoT-active ubatch needs the RERoT-teacher role");
    }

    // Lazy CellGeneration opt-in: the flash path only. OFF never reaches here.
    flashprefill_enable_tracking();

    std::string msg;
    llama_flashprefill_build_status st = llama_flashprefill_build_status::HARD_ERROR;
    if (want_rerot) {
        if (n_stream != 1 || v_cells.size() != 1) {
            return ineligible(LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED,
                "flashprefill layout: RERoT fragments require unified KV");
        }
        if (n_swa != 0 || swa_type != LLAMA_SWA_TYPE_NONE) {
            return ineligible(LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED,
                "flashprefill layout: RERoT+SWA has no indexed legality; stock path");
        }
        out.is_rerot = 1;
        out.causal = 1;
        out.swa_window = 0;
        out.swa_type = 0;
        st = llama_flashprefill_build_rerot_plan(v_cells[0], rerot_reader_views, ubatch, params, out, &msg);
    } else {
        if (!params.causal && n_swa != 0 && swa_type != LLAMA_SWA_TYPE_NONE) {
            return ineligible(LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED,
                "flashprefill layout: non-causal SWA keeps the stock path");
        }
        if (ubatch_2d) {
            for (uint32_t q = 0; q < ubatch.n_tokens; ++q) {
                if (!fp_row_is_text_pattern(ubatch, q)) {
                    return ineligible(LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED,
                        "flashprefill layout: non-text 2-D row keeps the stock path");
                }
            }
        }
        out.is_rerot = 0;
        out.causal = params.causal ? 1u : 0u;
        out.swa_window = n_swa;
        out.swa_type = (uint32_t) swa_type;
        st = llama_flashprefill_build_ordinary_plan(v_cells, seq_to_stream, n_swa, swa_type, params.causal, ubatch_2d,
            ubatch, params, out, &msg);
    }
    if (st == llama_flashprefill_build_status::INELIGIBLE) {
        return ineligible(LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED, msg);
    }
    if (st != llama_flashprefill_build_status::OK) {
        out.clear();
        out.eligible = false;
        out.error = msg;
        if (error) { *error = msg; }
        return false;
    }

    uint32_t width = 0;
    for (const auto & cells : v_cells) {
        width = std::max(width, cells.size());
    }
    out.block_k = params.block_k;
    out.cells_epoch = fp_epoch;
    flashprefill_cell_stamps(out.cell_stamps);
    out.n_kv_at_build = width;
    out.requires_cache_writes = true;
    out.eligible = true;

    std::string vmsg;
    if (!out.validate(width, &vmsg)) {
        out.clear();
        out.eligible = false;
        out.error = "flashprefill layout: invalid derived layout: " + vmsg;
        if (error) { *error = out.error; }
        return false;
    }
    return true;
}

bool llama_kv_cache_context::flashprefill_build_layout(
        uint32_t ubatch_index,
        int32_t role,
        const llama_flashprefill_layout_params & params,
        std::string * error) const {
    if (status != LLAMA_MEMORY_STATUS_SUCCESS || kv == nullptr) {
        if (error) { *error = "flashprefill layout: no memory context"; }
        return false;
    }
    if (ubatches.empty() || sinfos.empty() || ubatch_index >= ubatches.size()) {
        if (error) { *error = "flashprefill layout: ubatch index out of range"; }
        return false;
    }
    const llama_ubatch & ub = ubatches[ubatch_index];

    llama_flashprefill_build_key key;
    key.ubatch_index = ubatch_index;
    key.role = role;
    key.block_k = params.block_k;
    key.causal = params.causal ? 1u : 0u;
    key.want_exact_rows = params.want_exact_rows ? 1u : 0u;
    key.n_tokens = ub.n_tokens;
    key.exact_cap_bucket = llama_flashprefill_layout_params::bucket_for(params.exact_cap);

    // Reuse the owned planning when topology matches and owner stamps are fresh.
    if (fp_key_valid && fp_key.ubatch_index == key.ubatch_index && fp_key.role == key.role &&
        fp_key.block_k == key.block_k && fp_key.causal == key.causal &&
        fp_key.want_exact_rows == key.want_exact_rows && fp_key.n_tokens == key.n_tokens &&
        fp_key.exact_cap_bucket == key.exact_cap_bucket && flashprefill_layout_is_fresh()) {
        return fp_layout.eligible;
    }

    std::string msg;
    const bool eligible = kv->flashprefill_build_layout(ub, role, params, fp_layout, &msg);

    // Finalize the topology key post-build (group capacity and path are known now).
    key.group_cap_bucket = llama_flashprefill_layout_params::bucket_for(fp_layout.n_groups());
    key.is_rerot = fp_layout.is_rerot;
    fp_key = key;
    fp_key_valid = true;
    fp_cells_epoch = kv->flashprefill_epoch();
    kv->flashprefill_cell_stamps(fp_cell_stamps);

    if (error) { *error = msg; }
    return eligible;
}

bool llama_kv_cache_context::flashprefill_build_current_layout(
        int32_t role,
        const llama_flashprefill_layout_params & params,
        std::string * error) const {
    if (status != LLAMA_MEMORY_STATUS_SUCCESS || kv == nullptr) {
        if (error) { *error = "flashprefill layout: no memory context"; }
        return false;
    }
    if (ubatches.empty() || sinfos.empty() || i_cur >= ubatches.size()) {
        if (error) { *error = "flashprefill layout: no current ubatch"; }
        return false;
    }
    return flashprefill_build_layout((uint32_t) i_cur, role, params, error);
}

const llama_flashprefill_layout & llama_kv_cache_context::flashprefill_get_layout() const {
    return fp_layout;
}

bool llama_kv_cache_context::flashprefill_layout_is_fresh() const {
    if (!fp_key_valid || kv == nullptr) {
        return false;
    }
    if (fp_cells_epoch != kv->flashprefill_epoch()) {
        return false;
    }
    std::vector<llama_flashprefill_cell_stamp> cur;
    kv->flashprefill_cell_stamps(cur);
    if (cur.size() != fp_cell_stamps.size()) {
        return false;
    }
    for (size_t i = 0; i < cur.size(); ++i) {
        if (cur[i].stamp != fp_cell_stamps[i].stamp ||
            cur[i].generation != fp_cell_stamps[i].generation) {
            return false;
        }
        if (!fp_stamp_usable(cur[i])) {
            return false;
        }
    }
    return true;
}


const llama_flashprefill_build_key & llama_kv_cache_context::flashprefill_layout_key() const {
    return fp_key;
}

std::vector<uint32_t> llama_kv_cache::get_layer_ids() const {
    std::vector<uint32_t> res;
    res.reserve(layers.size());

    for (const auto & layer : layers) {
        res.push_back(layer.il);
    }

    return res;
}

ggml_tensor * llama_kv_cache::get_k_storage(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].k;
}

ggml_tensor * llama_kv_cache::get_v_storage(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].v;
}

int32_t llama_kv_cache::get_attn_rot_k_nrot() const {
    if (!attn_rot_k) {
        return 0;
    }
    const char * LLAMA_ATTN_ROT_K_NROT = getenv("LLAMA_ATTN_ROT_K_NROT");
    int nrot = LLAMA_ATTN_ROT_K_NROT ? atoi(LLAMA_ATTN_ROT_K_NROT) : 64;
    if (nrot == 0) {
        nrot = 64;
        do {
            nrot *= 2;
        } while (n_embd_head_k_all > 0 && n_embd_head_k_all % nrot == 0);
        nrot /= 2;
    }
    return nrot;
}

int32_t llama_kv_cache::get_attn_rot_v_nrot() const {
    if (!attn_rot_v) {
        return 0;
    }
    return 64;
}

std::vector<ggml_context *> llama_kv_cache::get_buffer_contexts() const {
    std::vector<ggml_context *> res;
    for (const auto & [ctx, _] : ctxs_bufs) {
        res.push_back(ctx.get());
    }
    return res;
}

bool llama_kv_cache::validate_seq_id(llama_seq_id seq_id) const {
    return seq_id >= 0 && (size_t) seq_id < seq_to_stream.size();
}

uint32_t llama_kv_cache::get_stream_for_seq(llama_seq_id seq_id) const {
    if (!validate_seq_id(seq_id)) {
        return 0;
    }
    return seq_to_stream[seq_id];
}

bool llama_kv_cache::can_capture_prerope_range(
        llama_seq_id seq_id,
        uint32_t cell_start,
        uint32_t cell_count,
        std::string * reason) const {
    if (cell_count == 0) {
        if (reason) *reason = "cell_count is zero";
        return false;
    }
    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        if (reason) *reason = "invalid seq_id";
        return false;
    }
    if (cell_start > UINT32_MAX - cell_count) {
        if (reason) *reason = "cell range arithmetic overflow";
        return false;
    }

    const auto & stream_cells = v_cells[seq_to_stream[seq_id]];
    if (cell_start + cell_count > stream_cells.size()) {
        if (reason) *reason = "cell range exceeds cache capacity";
        return false;
    }

    // Strict sealing conditions (§5.2):
    // Must be committed, occupied, seq_has(seq_id), and NOT PRIVATE / PENDING.
    for (uint32_t i = 0; i < cell_count; ++i) {
        uint32_t cell_idx = cell_start + i;
        if (stream_cells.is_empty(cell_idx)) {
            if (reason) *reason = "cell range contains empty cell at index " + std::to_string(cell_idx);
            return false;
        }
        if (!stream_cells.seq_has(cell_idx, seq_id)) {
            if (reason) *reason = "cell does not belong to sequence at index " + std::to_string(cell_idx);
            return false;
        }
        // Consult XKV store hot_committed state to reject ordinary tentative tokens
        if (xkv_store) {
            const uint64_t pid = stream_cells.payload_id_get(cell_idx);
            llama_xkv::xkv_location loc;
            if (!xkv_store->find_location(pid, loc) || loc.state != llama_xkv::xkv_state::hot_committed) {
                if (reason) *reason = "cell payload " + std::to_string(pid) + " is not in hot_committed state";
                return false;
            }
        }
        const auto & meta = stream_cells.rerot_get(cell_idx);
        if (meta.active()) {
            if (meta.visibility == llama_rerot_visibility::pending_record) {
                if (reason) *reason = "cell in range has uncommitted PENDING record status at index " + std::to_string(cell_idx);
                return false;
            }
            if (meta.visibility == llama_rerot_visibility::private_control) {
                if (reason) *reason = "cell in range is private_control at index " + std::to_string(cell_idx);
                return false;
            }
        }
    }

    return true;
}

const llama_kv_cells & llama_kv_cache::get_cells(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    return v_cells[seq_to_stream[seq_id]];
}

bool llama_kv_cache::rerot_set_write_tag(
        llama_seq_id seq_id,
        const llama_kv_rerot_meta & tag) {
    if (seq_id < 0 || (size_t) seq_id >= rerot_write_tags.size()) {
        return false;
    }

    if (tag.active()) {
        if (tag.node_id == LLAMA_REROT_NODE_INVALID || tag.run_id == LLAMA_REROT_RUN_INVALID ||
            tag.visibility == llama_rerot_visibility::normal ||
            (tag.visibility == llama_rerot_visibility::pending_record && tag.publish_epoch != 0)) {
            return false;
        }
        rerot_write_tags[seq_id] = tag;
    } else {
        rerot_write_tags[seq_id].reset();
    }

    fp_bump();
    return true;
}

void llama_kv_cache::rerot_clear_write_tag(llama_seq_id seq_id) {
    if (seq_id >= 0 && (size_t) seq_id < rerot_write_tags.size()) {
        rerot_write_tags[seq_id].reset();
        fp_bump();
    }
}

size_t llama_kv_cache::rerot_publish_run(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        uint64_t publish_epoch) {
    size_t count = 0;
    if (publish_epoch == 0 || !rerot_can_publish_run(episode_id, run_id, &count)) {
        return 0;
    }

    std::vector<std::pair<uint32_t, uint32_t>> matches;
    GGML_ASSERT(rerot_find_run_cells(episode_id, run_id, &matches) == count);

    for (const auto & match : matches) {
        const bool published = v_cells[match.first].rerot_publish(
            match.second, episode_id, run_id, publish_epoch);
        GGML_ASSERT(published);
    }

    fp_bump();
    return matches.size();
}

size_t llama_kv_cache::rerot_reclassify_run(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        llama_rerot_visibility expected,
        llama_rerot_visibility replacement,
        uint64_t publish_epoch) {
    size_t count = 0;
    if (!rerot_can_reclassify_run(
            episode_id, run_id, expected, replacement, publish_epoch, &count)) {
        return 0;
    }

    std::vector<std::pair<uint32_t, uint32_t>> matches;
    GGML_ASSERT(rerot_find_run_cells(episode_id, run_id, &matches) == count);

    for (const auto & match : matches) {
        const bool changed = v_cells[match.first].rerot_reclassify(
            match.second, episode_id, run_id, expected, replacement, publish_epoch);
        GGML_ASSERT(changed);
    }
    fp_bump();
    return matches.size();
}

bool llama_kv_cache::rerot_can_add_run_ref(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        llama_seq_id seq_id,
        size_t * count) const {
    if (count) {
        *count = 0;
    }
    if (episode_id == 0 || run_id == LLAMA_REROT_RUN_INVALID ||
        seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        return false;
    }

    const uint32_t dst_stream = seq_to_stream[seq_id];
    size_t matches = 0;
    for (uint32_t stream = 0; stream < v_cells.size(); ++stream) {
        const auto & cells = v_cells[stream];
        for (uint32_t cell = 0; cell < cells.size(); ++cell) {
            if (cells.is_empty(cell)) {
                continue;
            }
            const auto & meta = cells.rerot_get(cell);
            if (meta.episode_id != episode_id || meta.run_id != run_id) {
                continue;
            }
            // A logical run cannot be copied between independent KV streams,
            // and only an atomically published run is eligible for a keeper.
            if (stream != dst_stream || meta.visibility != llama_rerot_visibility::public_live ||
                meta.publish_epoch == 0) {
                return false;
            }
            ++matches;
        }
    }

    if (count) {
        *count = matches;
    }
    return matches > 0;
}

size_t llama_kv_cache::rerot_add_run_ref(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        llama_seq_id seq_id) {
    size_t count = 0;
    if (!rerot_can_add_run_ref(episode_id, run_id, seq_id, &count)) {
        return 0;
    }

    auto & cells = v_cells[seq_to_stream[seq_id]];
    size_t seen = 0;
    for (uint32_t cell = 0; cell < cells.size(); ++cell) {
        if (cells.is_empty(cell)) {
            continue;
        }
        const auto & meta = cells.rerot_get(cell);
        if (meta.episode_id != episode_id || meta.run_id != run_id) {
            continue;
        }
        if (!cells.seq_has(cell, seq_id)) {
            cells.seq_add(cell, seq_id);
        }
        ++seen;
    }

    GGML_ASSERT(seen == count);
    fp_bump();
    return seen;
}

size_t llama_kv_cache::rerot_find_run_cells(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        std::vector<std::pair<uint32_t, uint32_t>> * out) const {
    if (episode_id == 0 || run_id == LLAMA_REROT_RUN_INVALID) {
        return 0;
    }

    size_t count = 0;
    for (uint32_t stream = 0; stream < v_cells.size(); ++stream) {
        std::vector<uint32_t> idxs;
        count += v_cells[stream].rerot_collect_run(episode_id, run_id, idxs);
        if (out != nullptr) {
            for (const uint32_t cell : idxs) {
                out->emplace_back(stream, cell);
            }
        }
    }
    return count;
}

bool llama_kv_cache::rerot_can_freeze_to_archive(
        uint64_t episode_id,
        llama_seq_id exec_seq,
        llama_seq_id archive_seq,
        size_t * count) const {
    if (count != nullptr) {
        *count = 0;
    }
    if (episode_id == 0 || exec_seq < 0 || archive_seq < 0 || exec_seq == archive_seq ||
        (size_t) exec_seq >= seq_to_stream.size() || (size_t) archive_seq >= seq_to_stream.size()) {
        return false;
    }
    if (other != nullptr || seq_to_stream[exec_seq] != seq_to_stream[archive_seq]) {
        return false;
    }

    size_t kept = 0;
    const auto & cells = v_cells[seq_to_stream[exec_seq]];
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.is_empty(i) || !cells.seq_has(i, exec_seq)) {
            continue;
        }
        const auto & meta = cells.rerot_get(i);
        if (meta.active() && meta.episode_id == episode_id &&
            meta.visibility == llama_rerot_visibility::public_live && meta.publish_epoch != 0) {
            ++kept;
        }
    }

    if (count != nullptr) {
        *count = kept;
    }
    return true;
}

size_t llama_kv_cache::rerot_freeze_to_archive(
        uint64_t episode_id,
        llama_seq_id exec_seq,
        llama_seq_id archive_seq) {
    size_t count = 0;
    if (!rerot_can_freeze_to_archive(episode_id, exec_seq, archive_seq, &count)) {
        return 0;
    }

    auto & cells = v_cells[seq_to_stream[exec_seq]];
    const size_t kept = cells.rerot_freeze_to_archive(episode_id, exec_seq, archive_seq);

    GGML_ASSERT(kept == count);
    fp_bump();
    return kept;
}

void llama_kv_cache::shift_turbo_keys(const llama_cparams & cparams) {
    // K-shift is infrequent. Use the checked codec for every backend until a
    // native fused Turbo shift exists. Read/write one layer/stream snapshot,
    // not one synchronous GPU transfer per cell; leave unshifted bytes alone.
    for (const auto & layer : layers) {
        if (!llama_kv_is_turbo(layer.k->type)) {
            continue;
        }
        const uint32_t il = layer.il;
        const uint32_t hd = hparams.n_embd_head_k(il);
        const uint32_t storage_hd = ((hd + 127) / 128) * 128;
        const uint32_t rotary_dim = hparams.n_rot(il);
        const uint32_t n_heads = hparams.n_head_kv(il);
        std::vector<float> factors;
        if (auto * tensor = model.get_rope_factors(cparams, il)) {
            if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] < rotary_dim / 2) {
                throw std::runtime_error("Turbo K-shift: invalid RoPE factors");
            }
            factors.resize(rotary_dim / 2);
            ggml_backend_tensor_get(tensor, factors.data(), 0, factors.size() * sizeof(float));
        }
        std::vector<float> omega(rotary_dim / 2), scale_sq(rotary_dim / 2);
        if (!triattention_build_rope_tables(
                omega.data(), scale_sq.data(), rotary_dim,
                model.get_rope_freq_base(cparams, il), model.get_rope_freq_scale(cparams, il),
                (int32_t) cparams.n_ctx_orig_yarn, cparams.yarn_ext_factor,
                cparams.yarn_attn_factor, cparams.yarn_beta_fast, cparams.yarn_beta_slow,
                factors.empty() ? nullptr : factors.data())) {
            throw std::runtime_error("Turbo K-shift: invalid RoPE parameters");
        }
        const bool neox = hparams.rope_type != LLAMA_ROPE_TYPE_NORM;
        const auto * traits = ggml_get_type_traits(layer.k->type);
        for (uint32_t s = 0; s < n_stream; ++s) {
            const auto & cells = v_cells[s];
            if (!cells.get_has_shift()) {
                continue;
            }
            auto * k = layer.k_stream[s];
            const size_t bytes = ggml_nbytes(k);
            std::vector<uint8_t> snapshot(bytes);
            std::vector<float> row(storage_hd * n_heads);
            ggml_backend_tensor_get(k, snapshot.data(), 0, bytes);
            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (cells.is_empty(i) || cells.get_shift(i) == 0) {
                    continue;
                }
                const llama_pos delta = cells.get_shift(i);
                uint8_t * encoded = snapshot.data() + (size_t) i * k->nb[1];
                llama_kv_decode_key(k->type, encoded, row.data(), (uint32_t) row.size());
                for (uint32_t h = 0; h < n_heads; ++h) {
                    float * key = row.data() + h * storage_hd;
                    llama_kv_hadamard(key, hd, attn_rot_k_nrot);
                    const uint32_t rope_offset = hparams.n_lora_kv > 0 ? hd - rotary_dim : 0;
                    llama_kv_shift_key(key + rope_offset, rotary_dim, neox, omega.data(), delta);
                    llama_kv_hadamard(key, hd, attn_rot_k_nrot);
                }
                // from_float_ref performs the forward Turbo WHT itself.
                traits->from_float_ref(row.data(), encoded, (int64_t) row.size());
            }
            ggml_backend_tensor_set(k, snapshot.data(), 0, bytes);
        }
    }
}

bool llama_kv_cache::rerot_blocks_state_save(llama_seq_id seq_id) const {
    if (seq_id == -1) {
        for (const auto & tag : rerot_write_tags) {
            if (tag.active()) {
                return true;
            }
        }
        for (const auto & view : rerot_reader_views) {
            if (view.active()) {
                return true;
            }
        }
        for (const auto & cells : v_cells) {
            if (cells.rerot_has_active()) {
                return true;
            }
        }
        return false;
    }

    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        return false;
    }
    if ((size_t) seq_id < rerot_write_tags.size() && rerot_write_tags[seq_id].active()) {
        return true;
    }
    if ((size_t) seq_id < rerot_reader_views.size() && rerot_reader_views[seq_id].active()) {
        return true;
    }
    return v_cells[seq_to_stream[seq_id]].rerot_has_active_seq(seq_id);
}

bool llama_kv_cache::rerot_can_reclassify_run(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        llama_rerot_visibility expected,
        llama_rerot_visibility replacement,
        uint64_t publish_epoch,
        size_t * count) const {
    if (count) {
        *count = 0;
    }
    if (episode_id == 0 || run_id == LLAMA_REROT_RUN_INVALID ||
        expected == llama_rerot_visibility::normal || replacement == llama_rerot_visibility::normal ||
        (replacement == llama_rerot_visibility::public_live && publish_epoch == 0) ||
        (replacement != llama_rerot_visibility::public_live && publish_epoch != 0)) {
        return false;
    }

    size_t matches = 0;
    for (const auto & cells : v_cells) {
        for (uint32_t cell = 0; cell < cells.size(); ++cell) {
            if (cells.is_empty(cell)) {
                continue;
            }
            const auto & meta = cells.rerot_get(cell);
            if (meta.episode_id != episode_id || meta.run_id != run_id) {
                continue;
            }
            if (meta.visibility != expected) {
                return false;
            }
            ++matches;
        }
    }

    if (count) {
        *count = matches;
    }
    return matches > 0;
}

bool llama_kv_cache::rerot_can_publish_run(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        size_t * count) const {
    if (count) {
        *count = 0;
    }
    if (episode_id == 0 || run_id == LLAMA_REROT_RUN_INVALID) {
        return false;
    }

    size_t matches = 0;
    for (const auto & cells : v_cells) {
        for (uint32_t cell = 0; cell < cells.size(); ++cell) {
            if (cells.is_empty(cell)) {
                continue;
            }
            const auto & meta = cells.rerot_get(cell);
            if (meta.episode_id != episode_id || meta.run_id != run_id) {
                continue;
            }
            if (meta.visibility != llama_rerot_visibility::pending_record) {
                return false;
            }
            ++matches;
        }
    }

    if (count) {
        *count = matches;
    }
    return matches > 0;
}

bool llama_kv_cache::rerot_set_reader_view(
        llama_seq_id seq_id,
        const llama_rerot_reader_state & view) {
    if (seq_id < 0 || (size_t) seq_id >= rerot_reader_views.size() || !view.active() ||
        view.reader == LLAMA_REROT_NODE_INVALID || view.query_run == LLAMA_REROT_RUN_INVALID ||
        view.ordered_runs.empty()) {
        return false;
    }

    std::unordered_set<llama_rerot_run_id> seen;
    seen.reserve(view.ordered_runs.size());
    bool query_run_seen = false;
    for (const auto run_id : view.ordered_runs) {
        if (run_id == LLAMA_REROT_RUN_INVALID || !seen.insert(run_id).second) {
            return false;
        }
        query_run_seen |= run_id == view.query_run;
    }
    if (!query_run_seen) {
        return false;
    }

    rerot_reader_views[seq_id] = view;
    fp_bump();
    return true;
}

void llama_kv_cache::rerot_clear_reader_view(llama_seq_id seq_id) {
    if (seq_id >= 0 && (size_t) seq_id < rerot_reader_views.size()) {
        rerot_reader_views[seq_id].reset();
        fp_bump();
    }
}

bool llama_kv_cache::rerot_batch_active(const llama_ubatch & ubatch) const {
    bool any = false;
    bool all = true;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (!ubatch.seq_id || !ubatch.seq_id[i] || ubatch.n_seq_id[i] < 1) {
            all = false;
            continue;
        }
        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        const bool active = seq_id >= 0 && (size_t) seq_id < rerot_reader_views.size() &&
                            rerot_reader_views[seq_id].active();
        any |= active;
        all &= active;
    }
    if (any && !all) {
        throw std::runtime_error("RERoT and ordinary query rows cannot share one ubatch");
    }
    return any;
}

llama_rerot_attn_layout llama_kv_cache::rerot_build_attn_layout(
        const llama_ubatch & ubatch,
        uint32_t n_kv) const {
    llama_rerot_attn_layout result;
    if (!rerot_batch_active(ubatch)) {
        return result;
    }
    if (n_stream != 1 || v_cells.size() != 1) {
        throw std::runtime_error("RERoT indexed attention requires unified KV");
    }

    const auto & cells = v_cells[0];
    if (n_kv > cells.size()) {
        throw std::runtime_error("RERoT attention layout exceeds KV cache size");
    }

    result.n_queries = ubatch.n_tokens;
    result.query_offsets.reserve(size_t(result.n_queries) + 1);
    result.query_offsets.push_back(0);

    for (uint32_t query = 0; query < ubatch.n_tokens; ++query) {
        const llama_seq_id seq_id = ubatch.seq_id[query][0];
        const auto & reader = rerot_reader_views.at(seq_id);

        std::vector<llama_rerot_key_record> keys;
        keys.reserve(n_kv);
        for (uint32_t key = 0; key < n_kv; ++key) {
            if (cells.is_empty(key)) {
                continue;
            }
            keys.push_back({
                key,
                cells.pos_get(key),
                cells.seq_has(key, seq_id),
                cells.rerot_get(key),
            });
        }

        auto query_layout = llama_rerot_build_query_layout(reader, ubatch.pos[query], keys);
        const uint32_t group_base = static_cast<uint32_t>(result.groups.size());
        for (auto group : query_layout.groups) {
            group.query_index = query;
            result.groups.push_back(group);
        }
        for (auto entry : query_layout.entries) {
            entry.group_index += group_base;
            result.entries.push_back(entry);
        }
        result.query_offsets.push_back(static_cast<uint32_t>(result.entries.size()));
    }

    std::string error;
    if (!result.validate(n_kv, &error)) {
        throw std::runtime_error("invalid RERoT attention layout: " + error);
    }
    return result;
}

llama_kv_cache::rerot_resolved_view llama_kv_cache::rerot_resolve_view(
        const llama_rerot_reader_view & view) const {
    rerot_resolved_view result;
    result.episode_id = view.episode_id;
    result.reader = view.reader;

    for (const auto & logical_run : view.runs) {
        std::vector<rerot_resolved_cell> run_cells;

        for (uint32_t stream = 0; stream < v_cells.size(); ++stream) {
            const auto & cells = v_cells[stream];
            for (uint32_t cell = 0; cell < cells.size(); ++cell) {
                if (cells.is_empty(cell)) {
                    continue;
                }

                const auto & meta = cells.rerot_get(cell);
                if (meta.episode_id != view.episode_id || meta.run_id != logical_run.run_id ||
                    meta.node_id != logical_run.owner) {
                    continue;
                }

                run_cells.push_back({
                    stream,
                    cell,
                    cells.pos_get(cell),
                    0,
                    meta,
                });
            }
        }

        std::stable_sort(run_cells.begin(), run_cells.end(), [](const auto & lhs, const auto & rhs) {
            if (lhs.storage_pos != rhs.storage_pos) {
                return lhs.storage_pos < rhs.storage_pos;
            }
            if (lhs.meta.frontier != rhs.meta.frontier) {
                return lhs.meta.frontier < rhs.meta.frontier;
            }
            if (lhs.stream != rhs.stream) {
                return lhs.stream < rhs.stream;
            }
            return lhs.cell < rhs.cell;
        });

        for (auto & resolved : run_cells) {
            resolved.virtual_pos = result.query_virtual_pos++;
            result.cells.push_back(std::move(resolved));
        }
    }

    return result;
}

uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const {
    uint32_t result = 0;

    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];

        result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
    }

    return result;
}

uint32_t llama_kv_cache::get_n_kv_pos_contiguous(const slot_info & sinfo, const llama_ubatch & ubatch) const {
    if (sinfo.n_stream() != 1 || ubatch.n_seqs_unq != 1 || ubatch.n_tokens == 0 ||
        ubatch.pos == nullptr || ubatch.n_seq_id == nullptr ||
        ubatch.seq_id == nullptr || ubatch.seq_id[0] == nullptr) {
        return 0;
    }

    const llama_seq_id seq_id = ubatch.seq_id[0][0];
    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        return 0;
    }

    const uint32_t stream = seq_to_stream[seq_id];
    if (sinfo.strm[0] < 0 || (uint32_t) sinfo.strm[0] != stream || stream >= v_cells.size()) {
        return 0;
    }

    const auto & cells = v_cells[stream];
    const llama_pos pos_max = cells.seq_pos_max(seq_id);

    if (pos_max < 0 || pos_max >= (llama_pos) cells.size()) {
        return 0;
    }

    // the banded op aligns Q to the tail of K: the ubatch must be that monotonic tail, else dense bias
    if ((uint32_t) pos_max + 1 < ubatch.n_tokens) {
        return 0;
    }
    const llama_pos pos_start = pos_max + 1 - ubatch.n_tokens;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.pos[i] != pos_start + (llama_pos) i ||
            ubatch.n_seq_id[i] < 1 || ubatch.seq_id[i] == nullptr || ubatch.seq_id[i][0] != seq_id) {
            return 0;
        }
    }

    for (llama_pos pos = 0; pos <= pos_max; ++pos) {
        if (cells.is_empty(pos) || cells.pos_get(pos) != pos || !cells.seq_has(pos, seq_id)) {
            return 0;
        }
    }

    return pos_max + 1;
}

ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const uint64_t kv_size      = get_hot_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    // For turbo-padded caches, n_embd_k_gqa may be larger than hparams value
    const bool k_is_turbo = (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0 || k->type == GGML_TYPE_TURBO2_0);
    if (k_is_turbo) {
        assert(n_embd_k_gqa >= hparams.n_embd_k_gqa(il));
    } else {
        assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));
    }

    // Use padded head_dim for turbo types so the full padded data is returned
    const uint32_t head_k = hparams.n_embd_head_k(il);
    const uint32_t head_k_eff = (k_is_turbo && head_k % 128 != 0)
        ? ((head_k + 127) / 128) * 128 : head_k;

    uint32_t n_kv_eff = n_kv;
    uint32_t s0 = sinfo.s0;
    uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
    if (is_xkv_bounded_hot()) {
        n_kv_eff = std::min(n_kv, (uint32_t) kv_size);
        if (s0 >= n_stream) {
            s0 = 0;
        }
        if (s0 + ns > n_stream) {
            ns = (n_stream > s0) ? (n_stream - s0) : 1;
        }
        if (ns == 0) {
            ns = 1;
        }
    }

    return ggml_view_4d(ctx, k,
            head_k_eff, hparams.n_head_kv(il), n_kv_eff, ns,
            ggml_row_size(k->type, head_k_eff),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*s0);
}

ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const uint64_t kv_size      = get_hot_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE] — for turbo-padded V, cache may be larger
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    // Use padded head_dim for turbo types
    const bool v_is_turbo = (v->type == GGML_TYPE_TURBO3_0 || v->type == GGML_TYPE_TURBO4_0 || v->type == GGML_TYPE_TURBO2_0);
    const uint32_t head_v = hparams.n_embd_head_v(il);
    const uint32_t head_v_eff = (v_is_turbo && head_v % 128 != 0)
        ? ((head_v + 127) / 128) * 128 : head_v;

    uint32_t n_kv_eff = n_kv;
    uint32_t s0 = sinfo.s0;
    uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
    if (is_xkv_bounded_hot()) {
        n_kv_eff = std::min(n_kv, (uint32_t) kv_size);
        if (s0 >= n_stream) {
            s0 = 0;
        }
        if (s0 + ns > n_stream) {
            ns = (n_stream > s0) ? (n_stream - s0) : 1;
        }
        if (ns == 0) {
            ns = 1;
        }
    }

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                head_v_eff, hparams.n_head_kv(il), n_kv_eff, ns,
                ggml_row_size(v->type, head_v_eff),                      // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                    // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),            // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*s0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv_eff, hparams.n_head_kv(il), head_v_eff, ns,
            ggml_row_size(v->type, kv_size*head_v_eff),              // v->nb[1]
            ggml_row_size(v->type, kv_size),                         // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),            // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*s0);
}

ggml_tensor * llama_kv_cache::get_hot_k(ggml_context * ctx, int32_t il, const slot_info & sinfo) const {
    if (!is_xkv_bounded_hot()) {
        LLAMA_LOG_ERROR("%s: hot K view requested on unbounded cache\n", __func__);
        return nullptr;
    }
    // Full physical span: hot tensor rows are indexed sparsely by hot_idxs,
    // so the view covers hot_kv_size rows, never a logical n_kv count.
    return get_k(ctx, il, hot_kv_size, sinfo);
}

ggml_tensor * llama_kv_cache::get_hot_v(ggml_context * ctx, int32_t il, const slot_info & sinfo) const {
    if (!is_xkv_bounded_hot()) {
        LLAMA_LOG_ERROR("%s: hot V view requested on unbounded cache\n", __func__);
        return nullptr;
    }
    return get_v(ctx, il, hot_kv_size, sinfo);
}

ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k = layers[ikv].k;

    int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    // Turbo zero-padding: pad each head to next multiple of 128 before merging dims.
    // k_cur shape here is (n_embd_head, n_head, n_tokens).
    // ggml_pad pads ne[0] with zeros — exactly what we need per-head.
    const bool k_is_turbo = (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0 || k->type == GGML_TYPE_TURBO2_0);
    const bool k_needs_pad = k_is_turbo && (n_embd_head % 128 != 0);
    if (k_needs_pad) {
        const int64_t pad_amount = ((n_embd_head + 127) / 128) * 128 - n_embd_head;
        k_cur = ggml_pad(ctx, k_cur, pad_amount, 0, 0, 0);
        n_embd_head = k_cur->ne[0];  // now 128-aligned
    }

    int64_t n_embd_gqa = n_embd_head * n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        const int64_t kv_size = get_hot_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // Logging hook to inspect k_cur inputs at build time
    LLAMA_LOG_DEBUG("[cpy_k] layer %d: k_cur [%lld, %lld], n_tokens=%lld, n_embd_gqa=%lld, k type=%d\n",
        il, (long long)k_cur->ne[0], (long long)k_cur->ne[1], (long long)n_tokens, (long long)n_embd_gqa, (int)k->type);

    // Logging hook to inspect k_idxs values when available on host
    if (k_idxs && k_idxs->data && ggml_backend_buffer_is_host(k_idxs->buffer)) {
        const int64_t * idxs_ptr = (const int64_t *) k_idxs->data;
        for (int64_t i = 0; i < std::min<int64_t>(10, n_tokens); ++i) {
            LLAMA_LOG_ERROR("[cpy_k] layer %d: token %lld -> k_idx %lld\n", il, (long long)i, (long long)idxs_ptr[i]);
        }
    }

    // store the current K values into the cache
    ggml_tensor * result = ggml_set_rows(ctx, k, k_cur, k_idxs);

    // For turbo: store WHT group size in op_params so the CUDA kernel knows.
    // With zero-padding, all groups are always full 128-element WHT groups.
    if (k_is_turbo) {
        int32_t wht_group = 128;  // always 128 with padding
        memcpy(result->op_params, &wht_group, sizeof(int32_t));
    }

    return result;
}

ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    // Turbo zero-padding: pad V head_dim to next multiple of 128
    const bool v_is_turbo = (v->type == GGML_TYPE_TURBO3_0 || v->type == GGML_TYPE_TURBO4_0 || v->type == GGML_TYPE_TURBO2_0);
    const bool v_needs_pad = v_is_turbo && (n_embd_head % 128 != 0);
    if (v_needs_pad) {
        const int64_t pad_amount = ((n_embd_head + 127) / 128) * 128 - n_embd_head;
        v_cur = ggml_pad(ctx, v_cur, pad_amount, 0, 0, 0);
        n_embd_head = v_cur->ne[0];  // now 128-aligned
    }

    int64_t n_embd_gqa = n_embd_head * n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (n_stream > 1) {
            const int64_t kv_size = get_hot_size();

            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream);
        }

        ggml_tensor * result = ggml_set_rows(ctx, v, v_cur, v_idxs);
        // With zero-padding, all groups are always full 128-element WHT groups
        if (v_is_turbo) {
            int32_t wht_group = 128;  // always 128 with padding
            memcpy(result->op_params, &wht_group, sizeof(int32_t));
        }
        return result;
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_cache::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs;

    if (!v_trans) {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    } else {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max());
    }

    ggml_set_input(v_idxs);

    return v_idxs;
}

ggml_tensor * llama_kv_cache::build_input_k_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_k) {
        // EXPERIMENT (master TODO): force smallest rotation matrix (nrot=64)
        // for K, mirroring V's choice. Master defaults to the largest power-of-2
        // that divides head_dim, but the upstream comment hypothesizes smaller
        // tiles preserve more local structure → less PPL hit on sensitive models
        // (gemma-4 26B-A4B reportedly regresses with the largest tile).
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4141323088
        const char * LLAMA_ATTN_ROT_K_NROT = getenv("LLAMA_ATTN_ROT_K_NROT");
        int nrot = LLAMA_ATTN_ROT_K_NROT ? atoi(LLAMA_ATTN_ROT_K_NROT) : 64;

        // Original master behavior (largest power-of-2): set LLAMA_ATTN_ROT_K_NROT=0
        if (nrot == 0) {
            nrot = 64;
            do {
                nrot *= 2;
            } while (n_embd_head_k_all % nrot == 0);
            nrot /= 2;
        }

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_k_rot");
    }

    return res;
}

ggml_tensor * llama_kv_cache::build_input_v_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_v) {
        int nrot = 64;
        // using smaller rotation matrices for V seems beneficial
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570
        //do {
        //    nrot *= 2;
        //} while (hparams.n_embd_head_v() % nrot == 0);
        //nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_v_rot");
    }

    return res;
}

void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    const bool use_hot = is_xkv_bounded_hot() && !sinfo.hot_idxs.empty();
    const int64_t stride = get_hot_size();

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*stride;
        const auto & cur_idxs = use_hot ? sinfo.hot_idxs[s] : sinfo.idxs[s];

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + cur_idxs[i];
        }
    }
    // Host-side trace for the hot-slot38 NaN hunt: ubatch row -> dst slot,
    // with RoPE position and output flag. RoPE of finite inputs with finite
    // positions is finite, so finite positions here pin the NaN to pre-rope
    // Kcur (matmul/norm); a non-finite position pins it to the RoPE cache.
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const long long pos = ubatch->pos ? (long long)ubatch->pos[i] : -1ll;
        const int out = ubatch->output ? (int)ubatch->output[i] : -1;
        LLAMA_LOG_ERROR("[k_idxs] ubatch_row %u -> dst slot %lld (pos %lld out %d hot %d)\n",
            i, (long long)data[i], pos, out, (int)use_hot);
    }
}

void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    const bool use_hot = is_xkv_bounded_hot() && !sinfo.hot_idxs.empty();
    const int64_t stride = get_hot_size();

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*stride;
            const auto & cur_idxs = use_hot ? sinfo.hot_idxs[s] : sinfo.idxs[s];

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = offs + cur_idxs[i];
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = stride;

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;
            const auto & cur_idxs = use_hot ? sinfo.hot_idxs[s] : sinfo.idxs[s];

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + cur_idxs[i];
                }
            }
        }
    }
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;


    if (is_xkv_bounded_hot()) {
        const uint32_t hot_size = get_hot_size();
        std::fill(data, data + hot_size * n_stream, 0);
        if (xkv_hot_pool) {
            for (uint32_t s = 0; s < n_stream; ++s) {
                const auto & cells = v_cells[s];
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (!cells.is_empty(i) && cells.get_shift(i) != 0) {
                        uint64_t pid = cells.payload_id_get(i);
                        uint32_t slot = 0;
                        if (xkv_hot_pool->find_slot(pid, slot) && slot < hot_size) {
                            data[s * hot_size + slot] = cells.get_shift(i);
                        }
                    }
                }
            }
        }
        return;
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
        }
    }
}

void llama_kv_cache::set_input_k_shift_rows(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_I32);

    int32_t * data = (int32_t *) dst->data;
    int64_t n = 0;
    const int64_t cap = dst->ne[0];

    // Global hot-row indices (stream*hot + slot) for cells carrying a
    // nonzero shift. Bounded caches resolve pool slots; unbounded caches
    // address logical rows directly. Mirrors the build-time count scan, so
    // counts agree (no interleaving mutation between build and set_input).
    if (is_xkv_bounded_hot() && xkv_hot_pool) {
        const uint32_t hot_size = get_hot_size();
        for (uint32_t s = 0; s < n_stream && n < cap; ++s) {
            const auto & cells = v_cells[s];
            for (uint32_t i = 0; i < cells.size() && n < cap; ++i) {
                if (cells.is_empty(i) || cells.get_shift(i) == 0) {
                    continue;
                }
                uint32_t slot = 0;
                if (xkv_hot_pool->find_slot(cells.payload_id_get(i), slot) && slot < hot_size) {
                    data[n++] = (int32_t)(s * hot_size + slot);
                }
            }
        }
    } else {
        for (uint32_t s = 0; s < n_stream && n < cap; ++s) {
            const auto & cells = v_cells[s];
            for (uint32_t i = 0; i < cells.size() && n < cap; ++i) {
                if (!cells.is_empty(i) && cells.get_shift(i) != 0) {
                    data[n++] = (int32_t)(s * cells.size() + i);
                }
            }
        }
    }
}

struct args_set_input_kq_mask {
    const llama_hparams & hparams;
    const llama_ubatch  * ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t>       & seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;
};

template<typename T, bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    const T mask_keep = llama_cast<T>(0.0f);
    const T mask_drop = llama_cast<T>(-INFINITY);

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

                  llama_pos p0 = -1;
            const llama_pos p1 = ubatch->pos[i];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

            const uint64_t idst = n_kv*i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];

                    const uint64_t idst_prev = n_kv*srct;

                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                uint32_t j = jj;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        j = idxs[jj];
                    }
                }

                if (cells.is_empty(j)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(j, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(j);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        goto skip;
                    }

                    // M-RoPE causal mask
                    if (is_2d) {
                        if (p0 == p1) {
                            const auto & p0_ext = cells.ext_get(j);

                            if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                goto skip;
                            }
                        }
                    }
                }

                // apply SWA if any
                if (swa) {
                    if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        goto skip;
                    }
                }

                if (alibi) {
                    data[idst + j] = llama_cast<T>(static_cast<float>(-std::abs(p0 - p1)));
                } else {
                    data[idst + j] = mask_keep;
                }

                continue;
skip:
                data[idst + j] = mask_drop;
            }
        }
    }
}

template<typename T, bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl<T, causal, swa, is_2d, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, is_2d, false>(args, data);
    }
}

template<typename T, bool causal, bool swa>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl<T, causal, swa, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, false>(args, data);
    }
}

template<typename T, bool causal>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl<T, causal, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, false>(args, data);
    }
}

template<typename T>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data, bool causal_attn) {
    if (causal_attn) {
        set_input_kq_mask_impl<T, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, false>(args, data);
    }
}

void llama_kv_cache::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream = dst->ne[3]; // num streams in the current ubatch

    GGML_ASSERT(n_tokens%n_stream == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps = n_tokens/n_stream;

    //const int64_t t_start = ggml_time_us();

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream,
        /*.n_tps            =*/ n_tps,
    };

    if (dst->type == GGML_TYPE_F16) {
        set_input_kq_mask_impl<ggml_fp16_t>(args, (ggml_fp16_t *) dst->data, causal_attn);
    } else {
        set_input_kq_mask_impl<float>(args, (float *) dst->data, causal_attn);
    }

    //const int64_t t_end = ggml_time_us();

    //LLAMA_LOG_ERROR("%s: kq mask time: %0.3f ms\n", __func__, (t_end - t_start)/1000.0);
}

void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams");
    const auto & cells = v_cells[0];

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

void llama_kv_cache::set_input_pos_rel_flat(ggml_tensor * dst, const llama_ubatch * ubatch, uint32_t extent) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    const int64_t n_kv = dst->ne[0];
    GGML_ASSERT(dst->ne[1] == n_tokens);

    // [n_kv, n_tokens] in GLOBAL token order (stream-major, same as the KQ mask)
    for (int64_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        const llama_pos p1 = ubatch->pos[i];

        for (int64_t j = 0; j < n_kv; ++j) {
            // use the ACTUAL absolute position in the KV cell; physical slot order is not monotonic
            int32_t rel = (int32_t) extent; // zero-bias column
            if (!cells.is_empty(j)) {
                const llama_pos d = p1 - cells.pos_get(j);
                if (d >= 0 && d < (llama_pos) extent) {
                    rel = (int32_t) d;
                }
            }

            data[i*n_kv + j] = (int32_t) (i*(extent + 1)) + rel;
        }
    }
}

void llama_kv_cache::set_input_k_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

void llama_kv_cache::set_input_v_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

bool llama_kv_cache::has_cell_ext() const {
    // M-RoPE needs the 2D position, the PLE n-gram hash needs the token id
    return hparams.n_pos_per_embd() > 1 || hparams.ple_n_heads > 0;
}

void llama_kv_cache::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    res.clear();
    res.resize(n_tokens*n, LLAMA_TOKEN_NULL);

    if (n == 0) {
        return;
    }

    std::vector<uint32_t> ord;
    std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idx;

    if (!ubatch.token) {
        ord.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            auto & v = seq_idx[ubatch.seq_id[i][0]];
            ord[i] = v.size();
            v.push_back(i);
        }
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch.seq_id[i][0];

        for (uint32_t j = 0; j < n; ++j) {
            const llama_pos d = (llama_pos) (n - j);

            llama_pos p;
            if (!ubatch.token) {
                const auto & v = seq_idx[seq_id];
                const int64_t k = (int64_t) ord[i] - d;
                p = k >= 0 ? ubatch.pos[v[k]] : ubatch.pos[v[0]] + (llama_pos) k;
            } else {
                p = ubatch.pos[i] - d;
            }

            if (p < 0) {
                continue;
            }

            res[i*n + j] = v_cells[seq_to_stream[seq_id]].seq_pos_tok_le(seq_id, p);
        }
    }
}

void llama_kv_cache::state_read_sinfo(
        llama_io_read_i & io,
           llama_seq_id   seq_id,
  llama_state_seq_flags   flags,
      slot_info_vec_t *   sinfos_out,
const slot_info_vec_t *   sinfos_in) {
    // Minimal port: restore via the existing state_read path.
    // Mirrored layouts (sinfos_in) are best-effort for qwen4exp idx/attn pairing.
    GGML_UNUSED(flags);

    if (other) {
        return;
    }

    if (sinfos_out) {
        sinfos_out->assign(n_stream, slot_info{});
    }

    if (sinfos_in && sinfos_in->size() != n_stream) {
        throw std::runtime_error("failed to restore kv cache: mirrored slot layout has the wrong stream count");
    }

    // Reuse the classic restore; capturing exact slot infos is only required for
    // perfect idx/attn mirroring on state load, not for regular inference.
    state_read(io, seq_id, flags);

    if (sinfos_in) {
        // Ensure the caller-provided layout is at least acknowledged.
        GGML_UNUSED(sinfos_in);
    }
}

size_t llama_kv_cache::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += layer.v ? ggml_nbytes(layer.v) : 0;
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * rot,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale,
                   uint32_t   il) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;

    const auto & n_rot     = hparams.n_rot(il);
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;
    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        // rotate back
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        // rotate fwd
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size*n_stream]

    // I64 [n_shifted]: global hot-row indices (stream*hot + slot) with
    // nonzero shift, built host-side alongside k_shift. Turbo write-back
    // (set_rows) addresses only these rows so unshifted rows stay
    // byte-identical (a dense re-encode would spray requant noise).
    // Null when no Turbo K layer needs it.
    ggml_tensor * k_shift_rows = nullptr;

    // note: assumes k_rot^2 == I
    ggml_tensor * k_rot = nullptr;

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
    }

    if (k_shift_rows) {
        kv_self->set_input_k_shift_rows(k_shift_rows);
    }

    if (k_rot) {
        kv_self->set_input_k_rot(k_rot);
    }
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    GGML_ASSERT(!other);

    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    const int64_t k_shift_size = (int64_t) get_hot_size() * n_stream;
    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, k_shift_size);
    ggml_set_input(inp->k_shift);

    inp->k_rot = build_input_k_rot(ctx);

    const auto & cparams = lctx->get_cparams();

    // Number of hot rows carrying a nonzero shift (exact set_rows count).
    // Counted here at build time from cells+pool; set_input fills the same
    // scan with no interleaving mutation, so the counts agree.
    int64_t n_shifted_rows = 0;
    bool any_turbo_k = false;
    for (const auto & layer : layers) {
        if (layer.k->type == GGML_TYPE_TURBO2_0 || layer.k->type == GGML_TYPE_TURBO3_0 ||
            layer.k->type == GGML_TYPE_TURBO4_0) {
            any_turbo_k = true;
            break;
        }
    }
    if (any_turbo_k) {
        const uint32_t hot_size = get_hot_size();
        for (uint32_t s = 0; s < n_stream; ++s) {
            const auto & cells = v_cells[s];
            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.is_empty(i) && cells.get_shift(i) != 0) {
                    // Bounded caches count only pool-bound rows (mirrors the
                    // setter exactly); unbounded caches count every shifted
                    // cell. Same scan, same order: counts agree at set_input.
                    if (is_xkv_bounded_hot() && xkv_hot_pool) {
                        uint64_t pid = cells.payload_id_get(i);
                        uint32_t slot = 0;
                        if (pid != 0 && xkv_hot_pool->find_slot(pid, slot)) {
                            ++n_shifted_rows;
                        }
                    } else if (!is_xkv_bounded_hot()) {
                        ++n_shifted_rows;
                    }
                }
            }
        }
        if (n_shifted_rows > 0) {
            inp->k_shift_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_shifted_rows);
            ggml_set_input(inp->k_shift_rows);
        }
    }

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;
        const bool is_turbo_k = (layer.k->type == GGML_TYPE_TURBO2_0 || layer.k->type == GGML_TYPE_TURBO3_0 || layer.k->type == GGML_TYPE_TURBO4_0);
        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const auto n_rot         = hparams.n_rot(il);
        const auto n_embd_head_k = hparams.n_embd_head_k(il);
        const auto n_embd_nope   = hparams.n_lora_kv > 0 ? n_embd_head_k - n_rot : 0;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        if (is_turbo_k) {
            // TurboQuant hot rows store WHT-domain quantized blocks, so the
            // n_rot-column rope view (which would split 128-element blocks)
            // and the generic cpy write-back (no Turbo quantize path on
            // device) cannot run. Decode full block-aligned head rows,
            // rotate the IDENTICAL rope section as stock, re-encode in place.
            // Only shifted rows are written back (set_rows by k_shift_rows)
            // so unshifted rows stay byte-identical (no requant noise).
            if (inp->k_shift_rows == nullptr) {
                continue; // no shifted rows addressable: nothing to do
            }
            const int64_t head_w = layer.k->ne[0] / n_head_kv;
            if (n_head_kv <= 0 || layer.k->ne[0] % n_head_kv != 0 || head_w % 128 != 0) {
                LLAMA_LOG_ERROR("%s: turbo K layer %u head width not 128-aligned\n", __func__, il);
                return nullptr;
            }
            ggml_tensor * kfull = ggml_view_3d(ctx, layer.k,
                head_w, n_head_kv, k_shift_size,
                ggml_row_size(layer.k->type, head_w), layer.k->nb[1], 0);
            ggml_tensor * dec = ggml_cast(ctx, kfull, GGML_TYPE_F32);
            ggml_tensor * canon = ggml_turbo_wht(ctx, dec, 1 /*inverse*/, 128, nullptr);
            // Rope full canon rows (first n_rot cols only, identical to the old
            // sec-view section) so the result chains into the forward WHT.
            // NOTE: rope must chain into forward WHT. ggml_rope_ext is out-of-place
            // (dup) and ggml_rope_ext_inplace returns a view alias: discarding the
            // return value orphans the ROPE node from the graph, so forward WHT
            // would re-encode unrotated canon (CPU no-op). Rope full canon rows
            // with n_rot (touches first n_rot cols only, identical to sec view)
            // so the result chains directly into the forward WHT.
            ggml_tensor * roped = build_rope_shift(cparams, ctx, canon, inp->k_shift, inp->k_rot,
                rope_factors, freq_base_l, freq_scale_l, il);
            // Gather the shifted subset for write-back (F32 get_rows is
            // universal); set_rows quantizes on device with the same
            // wht_group contract as the K-write path (which performs forward WHT
            // rotation inside quantization, e.g. copy_to_quant.comp / quantize_row_turbo*_group).
            // Feeding an already-WHT-rotated tensor into set_rows would double-rotate.
            ggml_tensor * flat = ggml_view_2d(ctx, roped, head_w * n_head_kv, k_shift_size,
                roped->nb[2], 0);
            ggml_tensor * sub = ggml_get_rows(ctx, flat, inp->k_shift_rows);
            ggml_tensor * dst = layer.k;
            if (n_stream > 1) {
                dst = ggml_reshape_2d(ctx, dst, head_w * n_head_kv, k_shift_size);
            }
            ggml_tensor * back = ggml_set_rows(ctx, dst, sub, inp->k_shift_rows);
            int32_t wht_group = 128;
            memcpy(back->op_params, &wht_group, sizeof(int32_t));
            ggml_build_forward_expand(gf, back);
            continue;
        }

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_rot, n_head_kv, k_shift_size,
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                ggml_row_size(layer.k->type, n_embd_nope));

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, inp->k_rot, rope_factors, freq_base_l, freq_scale_l, il);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return gf;
}

void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    // v1 rule (§5.5/§25): the serialized cell format carries no per-cell
    // episode/node/run/visibility classification and no write-tag/reader-view
    // control state. Refuse explicitly instead of persisting bytes that would
    // restore as silently wrong ordinary KV. Ordinary untagged state is
    // unaffected: rerot_blocks_state_save() is false when OFF.
    if (rerot_blocks_state_save(seq_id)) {
        throw std::runtime_error(
            "llama_kv_cache::state_write: refusing slot save / prompt-cache persist while a RERoT "
            "episode is active (resident classified cells, write tags, or reader views present). "
            "v1 does not support persisting RERoT episodes; clear the episode first.");
    }

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool add_cell = true;

            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            // check the cell is not SWA-masked
            if (add_cell && seq_id != -1) {
                const bool is_masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));

                add_cell = !is_masked;
            }

            if (add_cell) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }

    // XKV trailer: full + per-seq subset (PARTIAL_ONLY checkpoints excluded
    // inside the helper). OFF writes zero extra bytes.
    xkv_state_write_trailer(io, seq_id, flags);
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    // Retained granted-slot order for the XKV trailer (per-seq rebind).
    slot_info last_sinfo;
    uint32_t last_strm = 0;
    bool have_slots = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id);

        try {
            res = res && state_read_data(io, strm, cell_count, sinfo);
        } catch (...) {
            res = false;
        }

        if (!res) {
            if (seq_id == -1) {
                clear(true);
            } else {
                seq_rm(seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }
        last_sinfo = sinfo;
        last_strm = strm;
        have_slots = true;
    }
    // XKV trailer: full + per-seq subset (PARTIAL_ONLY checkpoints excluded
    // inside). Absent sections = legacy bytes.
    xkv_state_read_trailer(io, seq_id, flags, have_slots ? &last_sinfo : nullptr, last_strm);
    fp_bump();
}

void llama_kv_cache::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (hparams.n_pos_per_embd() > 1) {
                const llama_kv_cell_ext ext = cells.ext_get(i);
                io.write(&ext, sizeof(ext));
            }

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

void llama_kv_cache::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const auto & cells = v_cells[cr.strm];

    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        auto * k = layer.k_stream[cr.strm];

        // Use actual tensor width (may be padded for turbo types: e.g. 576→640)
        const uint32_t n_embd_k_gqa = (uint32_t) k->ne[0];

        // Write key type
        const int32_t k_type_i = (int32_t) k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length and write out
        for (const auto & range : cr.data) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(k, range.first * k_size_row, buf_size);
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Use actual tensor width (may be padded for turbo types)
            const uint32_t n_embd_v_gqa = (uint32_t) v->ne[0];

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length and write out
            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v, range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length and write out
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v, src_offset, buf_size);
                }
            }
        }
    }
}

llama_xkv::xkv_state_fingerprints llama_kv_cache::xkv_state_expected_fps(
        const llama_xkv::xkv_state_config & cfg) const {
    auto fp = llama_xkv::make_config_fingerprints(cfg);
    std::vector<uint8_t> buf;
    auto pu32 = [&](uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            buf.push_back((uint8_t) ((v >> (i * 8)) & 0xFF));
        }
    };
    auto pu64 = [&](uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            buf.push_back((uint8_t) ((v >> (i * 8)) & 0xFF));
        }
    };
    auto pf = [&](float v) {
        uint32_t b = 0;
        std::memcpy(&b, &v, sizeof(b));
        pu32(b);
    };
    // Model load-artifact identity (name, arch, quant type, tensor count, file
    // size, structural dims): fast metadata screening only, NOT content proof.
    // Same-name same-size same-shape files with different weights can match
    // here; exact file-content proof is provenance.model_sha256 (required
    // nonzero for factored restore, enforced in validate_image).
    buf.clear();
    pu32((uint32_t) model.arch);
    for (char ch : model.name) {
        buf.push_back((uint8_t) ch);
    }
    buf.push_back(0);
    pu32((uint32_t) model.ftype());
    pu64((uint64_t) model.n_tensors());
    pu64((uint64_t) model.size());
    pu32(hparams.n_layer());
    pu32(hparams.n_embd);
    pu32(hparams.n_ctx_train);
    pu32(hparams.n_head_kv(0));
    pu32(hparams.n_embd_head_k(0));
    pu32(hparams.n_embd_head_v(0));
    fp.model = llama_xkv::xkv_state_checksum(buf.data(), buf.size());
    // RoPE configuration identity.
    buf.clear();
    pf(hparams.rope_freq_base_train);
    pf(hparams.rope_freq_scale_train);
    pu32((uint32_t) hparams.rope_type);
    pu32((uint32_t) hparams.rope_scaling_type_train);
    pf(hparams.rope_attn_factor);
    pu32(hparams.n_rot(0));
    fp.rope = llama_xkv::xkv_state_checksum(buf.data(), buf.size());
    // Tri calibration identity (presence + ratio + window + exact validated
    // content fingerprint). Two same-shape files with different stats hash
    // differently here and in the provenance digest below.
    buf.clear();
    pu32(tri_scorer != nullptr ? 1u : 0u);
    uint64_t ratio_bits = 0;
    double ratio = tri_ratio;
    std::memcpy(&ratio_bits, &ratio, sizeof(ratio_bits));
    pu64(ratio_bits);
    pu32(tri_recent_window);
    uint64_t cal_content = 0;
    if (tri_scorer != nullptr) {
        cal_content = tri_scorer->calibration_content_fingerprint();
    }
    pu64(cal_content);
    fp.tri_calibration = llama_xkv::xkv_state_checksum(buf.data(), buf.size());
    return fp;
}

// Cache-side provenance digests. model_sha256 is the EXACT source-artifact
// file-content digest (lazy, cached in llama_model; zeros when source paths
// are unavailable, e.g. memory/file-object loads). It is NEVER a metadata
// digest: same-name same-size same-shape GGUFs with different weights hash
// differently because every content byte is hashed. Tri digest is the
// calibration content SHA-256 when a valid scorer exists (zeros = Tri OFF).
// Source digest stays zero cache-side; enforced via fp.source + config equality.
static llama_xkv::xkv_state_provenance xkv_state_bridge_provenance(
        const llama_model & model, const triattention_scorer * scorer) {
    llama_xkv::xkv_state_provenance prov;
    uint8_t digest[32];
    if (model.source_artifact_sha256(digest)) {
        std::memcpy(prov.model_sha256, digest, 32);
    }
    if (scorer != nullptr) {
        if (scorer->calibration_content_sha256(digest)) {
            std::memcpy(prov.tri_calibration_sha256, digest, 32);
        }
    }
    return prov;
}

void llama_kv_cache::xkv_state_write_trailer(llama_io_write_i & io, llama_seq_id seq_id,
                                              llama_state_seq_flags flags) const {
    const bool per_seq = (seq_id != -1);
    // XKV OFF (no store) writes zero extra bytes. PARTIAL_ONLY checkpoints
    // (speculative/RERoT snapshots) stay lightweight stamps/refs: no store.
    if (other || !xkv_store) {
        return;
    }
    if (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) {
        return;
    }
    const auto acc = xkv_store->get_accounting();
    if (acc.factored_bytes == 0) {
        return; // hot-only/empty: legacy bytes already restore everything
    }
    auto fail = [](const std::string & m) {
        throw std::runtime_error("llama_kv_cache::xkv_state_write_trailer: " + m);
    };
    // 1. Enumerate live cells in legacy save order (mirrors state_write cell
    // selection exactly: non-empty, seq match, SWA-mask). Ordinal = position
    // among saved cells of the stream; restore rebinds by ordinal.
    struct enum_cell {
        uint32_t stream;
        uint32_t ordinal;
        uint64_t pid;
        uint64_t gen;
    };
    std::vector<enum_cell> ecells;
    std::vector<std::pair<uint64_t, uint64_t>> ids; // dedup (pid, gen)
    for (uint32_t s = 0; s < (uint32_t) v_cells.size(); ++s) {
        const auto & cells = v_cells[s];
        uint32_t ordinal = 0;
        for (uint32_t i = 0; i < (uint32_t) cells.size(); ++i) {
            if (cells.is_empty(i)) {
                continue;
            }
            if (seq_id != -1) {
                if (!cells.seq_has(i, seq_id)) {
                    continue;
                }
                const bool masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i),
                                                                 cells.seq_pos_max(seq_id));
                if (masked) {
                    continue;
                }
            }
            const uint64_t pid = cells.payload_id_get(i);
            if (pid == 0) {
                continue;
            }
            const uint64_t gen = cells.storage_generation_get(i);
            ecells.push_back({s, ordinal, pid, gen});
            ++ordinal;
            bool seen = false;
            for (const auto & q : ids) {
                if (q.first == pid) {
                    seen = true;
                    break;
                }
            }
            if (!seen) {
                ids.emplace_back(pid, gen);
            }
        }
    }
    // 2. Hot bindings are authoritative for hot rows.
    const auto hot_snap = xkv_store->list_hot_payload_bindings();
    std::map<uint64_t, llama_xkv::xkv_hot_payload_binding> hot_by_id;
    for (const auto & h : hot_snap.bindings) {
        hot_by_id[h.payload_id] = h;
    }
    // 3. Factored/flat live records from store locations.
    std::vector<uint64_t> factored_pids;
    std::vector<llama_xkv::xkv_state_payload> frec;
    std::set<uint64_t> factored_set; // pids with factored locations (for cellmap)
    for (const auto & pr : ids) {
        const uint64_t pid = pr.first;
        const uint64_t gen = pr.second;
        const auto hit = hot_by_id.find(pid);
        if (hit != hot_by_id.end()) {
            if (hit->second.storage_generation != gen) {
                fail("hot generation skew for payload " + std::to_string(pid));
            }
            llama_xkv::xkv_location loc;
            if (xkv_store->find_location(pid, loc) &&
                loc.kind != llama_xkv::xkv_location_kind::hot) {
                fail("hot/store kind skew for payload " + std::to_string(pid));
            }
            continue;
        }
        llama_xkv::xkv_location loc;
        if (!xkv_store->find_location(pid, loc)) {
            continue; // pool-only residue; legacy tensor bytes cover it
        }
        if (loc.storage_generation != gen) {
            fail("store generation skew for payload " + std::to_string(pid));
        }
        if (loc.state == llama_xkv::xkv_state::seal_candidate && loc.seal_tx_nonce != 0) {
            fail("seal transaction active; quiesce maintenance before state save");
        }
        if (loc.kind == llama_xkv::xkv_location_kind::hot) {
            fail("hot-kind location without hot binding for payload " + std::to_string(pid));
        }
        if (loc.kind == llama_xkv::xkv_location_kind::factored) {
            factored_pids.push_back(pid);
            factored_set.insert(pid);
        } else if (loc.kind != llama_xkv::xkv_location_kind::flat_quantized) {
            fail("unknown storage kind for payload " + std::to_string(pid));
        }
        llama_xkv::xkv_state_payload rec;
        rec.payload_id = pid;
        rec.generation = gen;
        rec.live       = 1;
        rec.locator    = loc;
        frec.push_back(rec);
    }
    if (factored_pids.empty()) {
        return; // no enumerable factored state; legacy bytes suffice
    }
    // Factored cell bindings in legacy save order (ordinal per stream).
    // Hot/flat cells stay legacy dense and are never rebound: excluded here.
    std::vector<llama_xkv::xkv_cell_binding> bindings;
    for (const auto & e : ecells) {
        if (factored_set.count(e.pid) == 0) {
            continue;
        }
        llama_xkv::xkv_cell_binding b;
        b.stream_idx = e.stream;
        b.ordinal    = e.ordinal;
        b.payload_id = e.pid;
        b.generation = e.gen;
        bindings.push_back(b);
    }
    // Hot bindings in scope (informational only; restore never installs pool
    // rows from state). Cell-less pool residue is excluded.
    std::vector<llama_xkv::xkv_hot_payload_binding> hot_in_scope;
    for (const auto & h : hot_snap.bindings) {
        bool tracked = false;
        for (const auto & q : ids) {
            if (q.first == h.payload_id) {
                tracked = true;
                break;
            }
        }
        if (tracked) {
            hot_in_scope.push_back(h);
        }
    }
    // 4. Pinned segment snapshots behind the factored payloads.
    const auto views = xkv_store->query_payload_segments(factored_pids);
    std::vector<std::shared_ptr<const llama_xkv::xkv_segment>> raw_segs;
    for (const auto & v : views.views) {
        auto h = v.pin.handle();
        if (!h) {
            fail("segment pin lost during state save");
        }
        bool seen = false;
        for (const auto & s : raw_segs) {
            if (s.get() == h.get()) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            raw_segs.push_back(h);
        }
    }
    if (raw_segs.empty()) {
        fail("factored payloads without pinned segments");
    }

    // Batched encoded-stream readback for DEVICE_OWNED segments.
    // DEVICE_OWNED segments have empty host byte vectors; their code streams live on
    // device handles (BackendResidency). We collect all unique allocations, queue
    // readbacks using xkv_backend_readback_batch with one sync, and populate
    // xkv_state_readback so capture_image serializes exact code streams.
    llama_xkv::xkv_state_readback dev_readback;
    {
        // Check if any segment is DEVICE_OWNED
        bool has_device = false;
        for (const auto & s : raw_segs) {
            if (s->residency == GGML_XKV_RES_DEVICE_OWNED) {
                has_device = true;
                break;
            }
        }
        if (has_device) {
            // Group readbacks by actual executor owner from backend_bundle:
            // Every device segment must have a valid executor, and all must agree or be grouped.
            // No fallback temporary backend creation!
            struct stream_key_map {
                uint64_t seg_id;
                uint64_t seg_ver;
                uint32_t group_idx;
                uint32_t role;
                uint64_t alloc_id;
            };
            std::map<ggml_backend_t, std::map<uint64_t, std::shared_ptr<const llama_xkv::xkv_backend_allocation>>> exec_to_allocs;
            std::vector<std::pair<ggml_backend_t, stream_key_map>> exec_key_maps;

            for (const auto & s : raw_segs) {
                if (s->residency != GGML_XKV_RES_DEVICE_OWNED || !s->backend_bundle) continue;
                ggml_backend_t s_exec = s->backend_bundle->get_executor();
                if (!s_exec) {
                    fail("DEVICE_OWNED segment missing real executor in backend_bundle");
                }
                const auto & bundle = *s->backend_bundle;
                for (const auto & g : s->groups) {
                    auto match_handle = [&](uint64_t fp, llama_xkv::factor_role role) {
                        for (const auto & h : bundle.handles) {
                            if (h && h->get_descriptor_fingerprint() == fp && h->get_desc().role == role) {
                                uint64_t aid = h->get_allocation_id();
                                exec_to_allocs[s_exec][aid] = h;
                                exec_key_maps.push_back({s_exec, {s->segment_id, s->segment_version, g.group_index, (uint32_t)role, aid}});
                                break;
                            }
                        }
                    };
                    match_handle(g.a_k.desc.fingerprint(), llama_xkv::factor_role::a_k);
                    if (g.b_k) match_handle(g.b_k->desc.fingerprint(), llama_xkv::factor_role::b_k);
                    match_handle(g.a_v.desc.fingerprint(), llama_xkv::factor_role::a_v);
                    if (g.b_v) match_handle(g.b_v->desc.fingerprint(), llama_xkv::factor_role::b_v);
                    if (!g.landmark.bytes.empty() || g.landmark.desc.logical_shape.rows > 0) {
                        match_handle(g.landmark.desc.fingerprint(), llama_xkv::factor_role::landmark);
                    }
                }
            }

            std::map<std::pair<ggml_backend_t, uint64_t>, std::vector<uint8_t>> downloaded_bytes;
            for (const auto & kv_exec : exec_to_allocs) {
                ggml_backend_t cur_exec = kv_exec.first;
                const auto & cur_allocs = kv_exec.second;
                if (cur_allocs.empty()) continue;

                std::vector<uint64_t> alloc_ids_order;
                std::vector<std::shared_ptr<const llama_xkv::xkv_backend_allocation>> alloc_vec;
                for (const auto & kv_a : cur_allocs) {
                    alloc_ids_order.push_back(kv_a.first);
                    alloc_vec.push_back(kv_a.second);
                }

                llama_xkv::xkv_backend_batch_config rb_cfg;
                rb_cfg.residency = GGML_XKV_RES_DEVICE_OWNED;
                llama_xkv::xkv_backend_readback_result rb_res;
                std::string rb_err;
                if (!llama_xkv::xkv_backend_readback_batch(cur_exec, alloc_vec, rb_cfg, rb_res, &rb_err)) {
                    fail("DEVICE_OWNED batched stream readback failed: " + rb_err);
                }
                if (rb_res.stream_bytes.size() != alloc_vec.size()) {
                    fail("DEVICE_OWNED readback stream count mismatch");
                }

                for (size_t i = 0; i < alloc_ids_order.size(); ++i) {
                    downloaded_bytes[{cur_exec, alloc_ids_order[i]}] = std::move(rb_res.stream_bytes[i]);
                }
            }

            for (const auto & ekm : exec_key_maps) {
                const auto & km = ekm.second;
                llama_xkv::xkv_state_readback_entry rbe;
                rbe.segment_id = km.seg_id;
                rbe.segment_version = km.seg_ver;
                rbe.group_index = km.group_idx;
                rbe.role = km.role;
                rbe.bytes = downloaded_bytes[{ekm.first, km.alloc_id}];
                dev_readback.entries.push_back(std::move(rbe));
            }
        }
    }

    // Per-sequence export granularity (RAM demotion isolation):
    // When seq_id != -1, serialize ONLY the rows referenced by the target
    // sequence (shared prefix + private seq rows). Unrelated private rows from
    // other sequences MUST NOT be copied into the envelope. We construct
    // state-local compact segments that pack A_K/A_V rows, slice/recompute
    // exact landmark chunks, and remap the factored_pids / frec locators.
    std::vector<std::shared_ptr<const llama_xkv::xkv_segment>> segs;
    if (per_seq) {
        std::set<uint64_t> seq_pids(factored_pids.begin(), factored_pids.end());
        for (const auto & orig_seg : raw_segs) {
            // Find which rows in this segment belong to the target sequence
            std::vector<uint32_t> kept_rows;
            for (uint32_t r = 0; r < orig_seg->n_rows; ++r) {
                if (orig_seg->live_rows[r] && seq_pids.count(orig_seg->row_payload_ids[r]) != 0) {
                    kept_rows.push_back(r);
                }
            }
            if (kept_rows.empty()) {
                continue;
            }
            if (kept_rows.size() == orig_seg->n_rows) {
                // All rows referenced; keep original segment as-is
                segs.push_back(orig_seg);
                continue;
            }

            // Build state-local compact segment
            const uint32_t compact_n_rows = (uint32_t) kept_rows.size();
            auto cseg = std::make_shared<llama_xkv::xkv_segment>();
            cseg->segment_id = orig_seg->segment_id;
            cseg->segment_version = orig_seg->segment_version;
            cseg->profile = orig_seg->profile;
            cseg->source = orig_seg->source;
            cseg->profile_fingerprint = orig_seg->profile_fingerprint;
            cseg->source_fingerprint = orig_seg->source_fingerprint;
            cseg->layer_group_map_fingerprint = orig_seg->layer_group_map_fingerprint;
            cseg->n_rows = compact_n_rows;
            cseg->n_live_rows = compact_n_rows;
            cseg->live_rows.assign(compact_n_rows, true);

            // Remap row payload IDs and record new row index for locator updates
            std::map<uint64_t, uint32_t> pid_to_compact_row;
            for (uint32_t dst_r = 0; dst_r < compact_n_rows; ++dst_r) {
                uint32_t src_r = kept_rows[dst_r];
                uint64_t pid = orig_seg->row_payload_ids[src_r];
                cseg->row_payload_ids.push_back(pid);
                pid_to_compact_row[pid] = dst_r;
            }

            // Update frec locators to point to compact row indices
            for (auto & rec : frec) {
                if (rec.locator.segment_id == orig_seg->segment_id &&
                    rec.locator.segment_version == orig_seg->segment_version) {
                    auto it = pid_to_compact_row.find(rec.payload_id);
                    if (it != pid_to_compact_row.end()) {
                        rec.locator.row = it->second;
                    }
                }
            }

            // Pack groups: pack A_K / A_V rows, preserve shared B handles,
            // slice landmark chunks
            for (const auto & og : orig_seg->groups) {
                llama_xkv::xkv_factor_group_payload cg;
                cg.group_index = og.group_index;
                cg.owning_layers = og.owning_layers;
                cg.rank_k = og.rank_k;
                cg.rank_v = og.rank_v;
                cg.layer_feature_offsets_k = og.layer_feature_offsets_k;
                cg.layer_feature_dims_k = og.layer_feature_dims_k;
                cg.layer_feature_offsets_v = og.layer_feature_offsets_v;
                cg.layer_feature_dims_v = og.layer_feature_dims_v;
                cg.total_dim_k = og.total_dim_k;
                cg.total_dim_v = og.total_dim_v;
                cg.config_fingerprint = og.config_fingerprint;
                cg.baseline_original_row_bytes = og.baseline_original_row_bytes;
                uint64_t base_bytes = 0;
                if (og.baseline_original_row_bytes > 0 && (uint64_t) compact_n_rows > UINT64_MAX / (uint64_t) og.baseline_original_row_bytes) {
                    fail("per-seq export: baseline_original_bytes multiplication overflow");
                }
                base_bytes = (uint64_t) og.baseline_original_row_bytes * (uint64_t) compact_n_rows;
                cg.baseline_original_bytes = base_bytes;

                // Shared B handles are preserved verbatim
                cg.b_k = og.b_k;
                cg.b_v = og.b_v;

                // Find source stream bytes: for DEVICE_OWNED segments, og.a_k/a_v/landmark.bytes
                // are empty, so we must copy row-wise from the readback bytes in dev_readback!
                auto get_source_bytes = [&](llama_xkv::factor_role role, const llama_xkv::encoded_matrix & em) -> const std::vector<uint8_t> * {
                    if (!em.bytes.empty()) {
                        return &em.bytes;
                    }
                    for (const auto & rbe : dev_readback.entries) {
                        if (rbe.segment_id == orig_seg->segment_id &&
                            rbe.segment_version == orig_seg->segment_version &&
                            rbe.group_index == og.group_index &&
                            rbe.role == (uint32_t)role) {
                            return &rbe.bytes;
                        }
                    }
                    return &em.bytes;
                };

                const auto * src_ak_bytes = get_source_bytes(llama_xkv::factor_role::a_k, og.a_k);
                const auto * src_av_bytes = get_source_bytes(llama_xkv::factor_role::a_v, og.a_v);

                // Pack A_K rows
                const size_t stride_ak = og.a_k.desc.row_stride_bytes;
                cg.a_k.desc = og.a_k.desc;
                cg.a_k.desc.logical_shape.rows = compact_n_rows;
                cg.a_k.desc.padded_shape.rows = compact_n_rows;
                if (stride_ak != 0 && (uint64_t) compact_n_rows > (uint64_t) std::numeric_limits<size_t>::max() / (uint64_t) stride_ak) {
                    fail("A_K compact bytes multiplication overflow during per-seq compaction");
                }
                cg.a_k.bytes.resize(compact_n_rows * stride_ak);
                for (uint32_t dst_r = 0; dst_r < compact_n_rows; ++dst_r) {
                    uint32_t src_r = kept_rows[dst_r];
                    if ((src_r + 1) * stride_ak > src_ak_bytes->size()) {
                        fail("A_K row source bytes out of bounds during per-seq compaction");
                    }
                    std::memcpy(cg.a_k.bytes.data() + dst_r * stride_ak,
                                src_ak_bytes->data() + src_r * stride_ak,
                                stride_ak);
                }

                // Pack A_V rows
                const size_t stride_av = og.a_v.desc.row_stride_bytes;
                cg.a_v.desc = og.a_v.desc;
                cg.a_v.desc.logical_shape.rows = compact_n_rows;
                cg.a_v.desc.padded_shape.rows = compact_n_rows;
                if (stride_av != 0 && (uint64_t) compact_n_rows > (uint64_t) std::numeric_limits<size_t>::max() / (uint64_t) stride_av) {
                    fail("A_V compact bytes multiplication overflow during per-seq compaction");
                }
                cg.a_v.bytes.resize(compact_n_rows * stride_av);
                for (uint32_t dst_r = 0; dst_r < compact_n_rows; ++dst_r) {
                    uint32_t src_r = kept_rows[dst_r];
                    if ((src_r + 1) * stride_av > src_av_bytes->size()) {
                        fail("A_V row source bytes out of bounds during per-seq compaction");
                    }
                    std::memcpy(cg.a_v.bytes.data() + dst_r * stride_av,
                                src_av_bytes->data() + src_r * stride_av,
                                stride_av);
                }

                // Slice/rebuild landmark chunks if present
                const auto * src_lm_bytes = get_source_bytes(llama_xkv::factor_role::landmark, og.landmark);
                const bool has_lm = !src_lm_bytes->empty() || og.landmark.desc.logical_shape.rows != 0;
                if (has_lm) {
                    // Check which chunks fall entirely within the kept row range
                    const size_t stride_lm = og.landmark.desc.row_stride_bytes;
                    std::vector<uint32_t> kept_chunk_indices;
                    std::vector<llama_xkv::xkv_landmark_chunk> new_chunks;

                    uint32_t compact_cursor = 0;
                    for (size_t ci = 0; ci < og.landmark_chunks.size(); ++ci) {
                        const auto & och = og.landmark_chunks[ci];
                        // Intact requires exact source identity in order: the
                        // chunk interval must lie inside the source segment
                        // (overflow-safe), carry a nonzero source fingerprint
                        // (chunk provenance), a sane error bound, and every
                        // source row must appear consecutively in kept_rows at
                        // the current compact cursor. Export compaction never
                        // shifts storage positions or phase, so exact identity
                        // plus provenance preserves validity; anything else
                        // takes the bounded rebuild path below, never a copy.
                        if (och.row_count == 0 ||
                            och.row_count > orig_seg->n_rows ||
                            och.row_begin > orig_seg->n_rows - och.row_count ||
                            och.source_fingerprint == 0 ||
                            !std::isfinite(och.error_bound) || och.error_bound < 0.0f) {
                            break; // corrupt/non-provenanced chunk: force rebuild path
                        }
                        bool chunk_intact = true;
                        for (uint32_t k = 0; k < och.row_count; ++k) {
                            const uint64_t dst = (uint64_t) compact_cursor + k;
                            if (dst >= kept_rows.size() ||
                                kept_rows[(size_t) dst] != och.row_begin + k) {
                                chunk_intact = false;
                                break;
                            }
                        }
                        if (chunk_intact) {
                            kept_chunk_indices.push_back((uint32_t) ci);
                            llama_xkv::xkv_landmark_chunk nch;
                            nch.row_begin = compact_cursor;
                            nch.row_count = och.row_count;
                            nch.error_bound = och.error_bound;
                            nch.source_fingerprint = och.source_fingerprint;
                            new_chunks.push_back(nch);
                            compact_cursor += och.row_count;
                        }
                    }

                    if (compact_cursor == compact_n_rows && !new_chunks.empty()) {
                        // All compact rows covered by intact original chunks: copy summary rows
                        const uint32_t n_lm_rows = (uint32_t) new_chunks.size();
                        cg.landmark.desc = og.landmark.desc;
                        cg.landmark.desc.logical_shape.rows = n_lm_rows;
                        cg.landmark.desc.padded_shape.rows = n_lm_rows;
                        if (stride_lm != 0 && (uint64_t) n_lm_rows > (uint64_t) std::numeric_limits<size_t>::max() / (uint64_t) stride_lm) {
                            fail("Landmark compact bytes multiplication overflow during per-seq compaction");
                        }
                        cg.landmark.bytes.resize(n_lm_rows * stride_lm);
                        for (uint32_t dci = 0; dci < n_lm_rows; ++dci) {
                            uint32_t sci = kept_chunk_indices[dci];
                            if ((sci + 1) * stride_lm > src_lm_bytes->size()) {
                                fail("Landmark row source bytes out of bounds during per-seq compaction");
                            }
                            std::memcpy(cg.landmark.bytes.data() + dci * stride_lm,
                                        src_lm_bytes->data() + sci * stride_lm,
                                        stride_lm);
                        }
                        cg.landmark_chunks = std::move(new_chunks);
                        cg.landmark_table_fingerprint =
                            compute_landmark_table_fingerprint(cg.landmark_chunks);
                    } else {
                        // Partial chunks across sequence boundaries: intact copy
                        // is unsafe (summaries are position-sensitive), so a
                        // semantic rebuild is required. The CPU hook decodes
                        // host factor streams and is reference-host only: for
                        // device-resident segments it is skipped entirely (no
                        // host decode, no silent fallback) and the export
                        // fails closed below.
                        bool rebuilt = false;
                        const bool host_factors = (orig_seg->residency != GGML_XKV_RES_DEVICE_OWNED);
                        if (xkv_runtime && host_factors) {
                            auto rcb = xkv_runtime->bound_removal_callback();
                            if (rcb) {
                                auto dummy_seg = std::make_shared<llama_xkv::xkv_segment>();
                                if (rcb(*orig_seg, kept_rows, *dummy_seg, nullptr)) {
                                    const auto * fg = dummy_seg->find_group(cg.group_index);
                                    if (fg) {
                                        cg.landmark = fg->landmark;
                                        cg.landmark_chunks = fg->landmark_chunks;
                                        cg.landmark_table_fingerprint = fg->landmark_table_fingerprint;
                                        rebuilt = true;
                                    }
                                }
                            }
                        }
                        if (!rebuilt) {
                            fail(orig_seg->residency == GGML_XKV_RES_DEVICE_OWNED
                                ? "per-seq export: device-owned landmark chunks cut across sequence boundaries; native rebuild unavailable in export, refusing partial leak"
                                : "per-seq export: landmark chunks cut across sequence boundaries without rebuild hook; refusing partial leak");
                        }
                    }
                }

                cg.refresh_descriptor_fingerprint();
                cg.update_byte_counters();
                cseg->groups.push_back(std::move(cg));
            }

            cseg->update_byte_counters();
            segs.push_back(cseg);
        }
    } else {
        segs = raw_segs;
    }

    // 5. Config / fingerprints / stamp / exact count caps.
    const auto & cp = xkv_store->get_cparams();
    auto cfg = llama_xkv::make_bridge_config(cp);
    if (cfg.store_mib == 0) {
        fail("admission xkv_store_mib unresolved");
    }
    const auto fps   = xkv_state_expected_fps(cfg);
    const auto stamp = xkv_store->current_stamp();
    size_t total_groups = 0;
    uint32_t max_groups = 1, max_own = 1, max_rows = 1;
    for (const auto & s : segs) {
        total_groups += s->groups.size();
        if ((uint32_t) s->groups.size() > max_groups) {
            max_groups = (uint32_t) s->groups.size();
        }
        if (s->n_rows > max_rows) {
            max_rows = s->n_rows;
        }
        for (const auto & g : s->groups) {
            if ((uint32_t) g.owning_layers.size() > max_own) {
                max_own = (uint32_t) g.owning_layers.size();
            }
        }
    }
    if (segs.size() > (size_t) std::numeric_limits<uint32_t>::max() ||
        total_groups > (size_t) std::numeric_limits<uint32_t>::max() - 8 ||
        hot_in_scope.size() + frec.size() > (size_t) std::numeric_limits<uint32_t>::max() - 8) {
        fail("state census exceeds 32-bit count caps");
    }
    const uint64_t store_budget = (uint64_t) cfg.store_mib << 20;
    const auto limits = llama_xkv::xkv_state_limits::from_store_budgets(
        store_budget, acc.hot_bytes, (uint32_t) segs.size(), max_groups,
        (uint32_t) (hot_in_scope.size() + frec.size()), (uint32_t) (5 * total_groups + 8), max_own,
        max_rows);
    const llama_xkv::xkv_state_provenance prov =
        xkv_state_bridge_provenance(model, tri_scorer.get());
    llama_xkv::xkv_state_image img;
    std::string err;
    if (!llama_xkv::capture_image(segs, hot_in_scope, frec, acc, cp.xkv_workspace_mib,
                                  cp.xkv_decode_cache_mib, cfg, fps, prov, stamp, limits, img,
                                  dev_readback.entries.empty() ? nullptr : &dev_readback,
                                  &err)) {
        fail("capture: " + err);
    }
    std::vector<uint8_t> env;
    if (!llama_xkv::encode_image(img, env, limits, &err)) {
        fail("encode: " + err);
    }
    if (!llama_xkv::write_state_section(io, env, &err)) {
        fail("section write: " + err);
    }
    // Second section: factored cell bindings in legacy save order.
    if (!llama_xkv::write_cellmap_section(io, bindings, &err)) {
        fail("cellmap write: " + err);
    }
}

static bool xkv_restore_segments_identical(const std::shared_ptr<const llama_xkv::xkv_segment> & live,
                                           const std::shared_ptr<const llama_xkv::xkv_segment> & img,
                                           const std::map<uint64_t, const std::vector<uint8_t> *> & handle_to_rb_bytes) {
    if (!live || !img) {
        return false;
    }
    if (live->segment_id != img->segment_id || live->segment_version != img->segment_version) {
        return false;
    }
    if (live->n_rows != img->n_rows || live->n_live_rows != img->n_live_rows) {
        return false;
    }
    if (live->row_payload_ids != img->row_payload_ids) {
        return false;
    }
    if (live->live_rows.size() != img->live_rows.size()) {
        return false;
    }
    for (size_t i = 0; i < live->live_rows.size(); ++i) {
        if ((bool) live->live_rows[i] != (bool) img->live_rows[i]) {
            return false;
        }
    }
    if (live->groups.size() != img->groups.size()) {
        return false;
    }
    const bool is_device = (live->residency == GGML_XKV_RES_DEVICE_OWNED && live->backend_bundle != nullptr);
    for (size_t gi = 0; gi < live->groups.size(); ++gi) {
        const auto & a = live->groups[gi];
        const auto & b = img->groups[gi];
        if (a.group_index != b.group_index || a.owning_layers != b.owning_layers) {
            return false;
        }
        if (a.rank_k != b.rank_k || a.rank_v != b.rank_v) {
            return false;
        }
        if (a.layer_feature_offsets_k != b.layer_feature_offsets_k ||
            a.layer_feature_dims_k != b.layer_feature_dims_k ||
            a.layer_feature_offsets_v != b.layer_feature_offsets_v ||
            a.layer_feature_dims_v != b.layer_feature_dims_v) {
            return false;
        }
        if (a.total_dim_k != b.total_dim_k || a.total_dim_v != b.total_dim_v) {
            return false;
        }
        if (!(a.a_k.desc == b.a_k.desc)) {
            return false;
        }
        if (!a.b_k || !b.b_k || !(a.b_k->desc == b.b_k->desc)) {
            return false;
        }
        if (!(a.a_v.desc == b.a_v.desc)) {
            return false;
        }
        if (!a.b_v || !b.b_v || !(a.b_v->desc == b.b_v->desc)) {
            return false;
        }
        const bool a_has = !a.landmark.bytes.empty() || a.landmark.desc.logical_shape.rows != 0;
        const bool b_has = !b.landmark.bytes.empty() || b.landmark.desc.logical_shape.rows != 0;
        if (a_has != b_has) {
            return false;
        }
        if (a_has && !(a.landmark.desc == b.landmark.desc)) {
            return false;
        }
        if (!is_device) {
            // Host flows: compare code stream bytes directly
            if (a.a_k.bytes != b.a_k.bytes || a.b_k->bytes != b.b_k->bytes ||
                a.a_v.bytes != b.a_v.bytes || a.b_v->bytes != b.b_v->bytes ||
                (a_has && a.landmark.bytes != b.landmark.bytes)) {
                return false;
            }
        } else {
            // DEVICE_OWNED: live segment has empty host bytes, so compare exact content identity.
            // Every backend allocation must match descriptor AND stored upload-time checksum.
            // If checksum is not bound, compare against the batched readback stream map!
            auto find_and_verify_handle = [&](uint64_t fp, llama_xkv::factor_role role,
                                              const std::vector<uint8_t> & expected_bytes) -> bool {
                for (const auto & h : live->backend_bundle->handles) {
                    if (h && h->get_descriptor_fingerprint() == fp && h->get_desc().role == role) {
                        if (h->is_checksum_bound()) {
                            uint64_t expected_cs = llama_xkv::xkv_backend_checksum(expected_bytes.data(), expected_bytes.size());
                            return h->get_checksum() == expected_cs;
                        } else {
                            // Compare against prepared batched readback result (zero per-handle D2H loops)
                            auto it_rb = handle_to_rb_bytes.find(h->get_allocation_id());
                            if (it_rb != handle_to_rb_bytes.end() && it_rb->second != nullptr) {
                                return *(it_rb->second) == expected_bytes;
                            }
                            return false; // missing required readback verification
                        }
                    }
                }
                return false;
            };
            if (!find_and_verify_handle(b.a_k.desc.fingerprint(), llama_xkv::factor_role::a_k, b.a_k.bytes)) return false;
            if (!find_and_verify_handle(b.b_k->desc.fingerprint(), llama_xkv::factor_role::b_k, b.b_k->bytes)) return false;
            if (!find_and_verify_handle(b.a_v.desc.fingerprint(), llama_xkv::factor_role::a_v, b.a_v.bytes)) return false;
            if (!find_and_verify_handle(b.b_v->desc.fingerprint(), llama_xkv::factor_role::b_v, b.b_v->bytes)) return false;
            if (a_has) {
                if (!find_and_verify_handle(b.landmark.desc.fingerprint(), llama_xkv::factor_role::landmark, b.landmark.bytes)) return false;
            }
        }
        if (a.landmark_chunks != b.landmark_chunks) {
            return false;
        }
        if (a.landmark_table_fingerprint != b.landmark_table_fingerprint) {
            return false;
        }
    }
    return true;
}

void llama_kv_cache::xkv_state_read_trailer(llama_io_read_i & io, llama_seq_id seq_id,
                                            llama_state_seq_flags flags,
                                            const slot_info * granted, uint32_t granted_strm) {
    if (other) {
        return;
    }
    if (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) {
        return; // lightweight checkpoints carry no trailer
    }
    const bool per_seq = seq_id != -1;
    auto refuse = [&](const std::string & m) {
        if (per_seq) {
            seq_rm(seq_id, -1, -1);
        } else {
            clear(true);
        }
        throw std::runtime_error("llama_kv_cache::xkv_state_read_trailer: " + m);
    };
    if (!xkv_store) {
        return;
    }
    const auto & cp = xkv_store->get_cparams();
    const uint64_t store_budget = (uint64_t) cp.xkv_store_mib << 20;
    const auto acc = xkv_store->get_accounting();
    // Exact byte caps from live budgets; wide anti-malformed count caps.
    const auto limits = llama_xkv::xkv_state_limits::from_store_budgets(
        store_budget, acc.hot_bytes, 1 << 16, 256, 1 << 24, 1 << 20, 4096, 1 << 24);
    std::vector<uint8_t> env;
    std::string err;
    const auto r = llama_xkv::read_state_section(io, limits.max_envelope_bytes, env, &err);
    if (r == llama_xkv::xkv_section_read_result::absent) {
        return; // legacy file: dense-only restore stands
    }
    if (r == llama_xkv::xkv_section_read_result::corrupt) {
        refuse("corrupt trailer: " + err);
    }
    const auto cfg = llama_xkv::make_bridge_config(cp);
    const auto fps = xkv_state_expected_fps(cfg);
    const llama_xkv::xkv_state_provenance prov =
        xkv_state_bridge_provenance(model, tri_scorer.get());
    llama_xkv::xkv_state_image img;
    if (!llama_xkv::decode_image(env.data(), env.size(), img, fps, prov, limits, &err)) {
        refuse("decode/validate: " + err);
    }
    // Exact effective-config compatibility: a Turbo2-saved bundle under a
    // Turbo4-requested live config (same profile) refuses here by codec type,
    // as do mode/budget/SR/refine/factorizer/balance/rank drifts.
    if (!llama_xkv::check_config_compatible(img.config, cfg, &err)) {
        refuse("config compatible: " + err);
    }
    // Restored-cell census bounds the cellmap (full: non-empty per stream;
    // per-seq: granted slots). The cellmap is required with the envelope.
    std::vector<uint32_t> restored_counts;
    uint64_t cell_total = 0;
    if (per_seq) {
        if (granted == nullptr || granted->idxs.empty()) {
            refuse("per-seq trailer without granted slots");
        }
        cell_total = (uint64_t) granted->idxs[0].size();
    } else {
        for (uint32_t s = 0; s < (uint32_t) v_cells.size(); ++s) {
            const auto & cells = v_cells[s];
            uint32_t n = 0;
            for (uint32_t i = 0; i < (uint32_t) cells.size(); ++i) {
                if (!cells.is_empty(i)) {
                    ++n;
                }
            }
            restored_counts.push_back(n);
            cell_total += n;
        }
    }
    if (cell_total > llama_xkv::XKV_STATE_CELLMAP_MAX_ENTRIES) {
        refuse("restored cell census exceeds cellmap hard maximum");
    }
    std::vector<llama_xkv::xkv_cell_binding> bindings;
    const auto rc = llama_xkv::read_cellmap_section(io, cell_total, bindings, &err);
    if (rc != llama_xkv::xkv_section_read_result::ok) {
        refuse(rc == llama_xkv::xkv_section_read_result::absent
                   ? "envelope present but cellmap absent (cannot rejoin cells)"
                   : "corrupt cellmap: " + err);
    }
    // Off-side binding validation against restored cells + decoded image.
    std::map<uint64_t, llama_xkv::xkv_location> plan_locs;
    for (const auto & pl : img.payloads) {
        plan_locs[pl.payload_id] = pl.locator;
    }
    for (const auto & b : bindings) {
        if (b.stream_idx >= (uint32_t) v_cells.size()) {
            refuse("cellmap stream out of range");
        }
        uint64_t bound = per_seq ? cell_total : restored_counts[b.stream_idx];
        if (per_seq && b.stream_idx != granted_strm) {
            refuse("cellmap stream disagrees with per-seq granted stream");
        }
        if (b.ordinal >= bound) {
            refuse("cellmap ordinal out of restored range");
        }
        const auto it = plan_locs.find(b.payload_id);
        if (it == plan_locs.end() || it->second.kind != llama_xkv::xkv_location_kind::factored ||
            it->second.storage_generation != b.generation) {
            refuse("cellmap entry without matching factored payload record");
        }
    }
    llama_xkv::xkv_state_import_plan plan;
    if (!llama_xkv::build_import_plan(img, limits, plan, &err)) {
        refuse("import plan: " + err);
    }
    // PID preflight before any mutation: future allocate_payload_id() calls
    // must exceed every restored id (cross-process counter resets otherwise
    // collide). Burns counter gaps only; harmless on later refusal paths.
    {
        uint64_t max_pid = 0;
        for (const auto & pl : plan.bundle.locations) {
            if (pl.first > max_pid) {
                max_pid = pl.first;
            }
        }
        if (!llama_kv_cells::reserve_payload_ids_through(max_pid)) {
            refuse("pid reservation failed (counter saturated)");
        }
    }
    // Backend device-batch hook (BackendResidency): translate plan allocations
    // to backend streams and verify pre-upload. Host/CPU flows skip (no provider
    // bound). Nothing here decodes full streams or falls back to host reference.
    // STAGED on provider details; tracked in the yield.
    // Dedup-narrow vs live store: reuse byte-identical segments, import the rest.
    std::vector<std::shared_ptr<const llama_xkv::xkv_segment>> need;
    // For any candidate live device segment, collect handles with unbound checksums
    // and batch-readback once per executor (zero per-handle D2H loops).
    std::map<uint64_t, const std::vector<uint8_t> *> handle_to_rb_bytes;
    std::map<uint64_t, std::vector<uint8_t>> rb_storage; // holds downloaded bytes
    {
        std::map<ggml_backend_t, std::vector<std::shared_ptr<const llama_xkv::xkv_backend_allocation>>> exec_to_allocs;
        std::map<uint64_t, std::shared_ptr<const llama_xkv::xkv_backend_allocation>> unique_unbound;
        for (const auto & s : plan.bundle.segments) {
            auto live = xkv_store->get_segment_version(s->segment_id, s->segment_version);
            if (live && live->residency == GGML_XKV_RES_DEVICE_OWNED && live->backend_bundle) {
                ggml_backend_t exec = live->backend_bundle->get_executor();
                if (exec) {
                    for (const auto & h : live->backend_bundle->handles) {
                        if (h && !h->is_checksum_bound() && unique_unbound.emplace(h->get_allocation_id(), h).second) {
                            exec_to_allocs[exec].push_back(h);
                        }
                    }
                }
            }
        }
        for (const auto & kv_ex : exec_to_allocs) {
            llama_xkv::xkv_backend_batch_config rb_cfg;
            rb_cfg.residency = GGML_XKV_RES_DEVICE_OWNED;
            llama_xkv::xkv_backend_readback_result rb_res;
            std::string rb_err;
            if (llama_xkv::xkv_backend_readback_batch(kv_ex.first, kv_ex.second, rb_cfg, rb_res, &rb_err)) {
                for (size_t i = 0; i < kv_ex.second.size() && i < rb_res.stream_bytes.size(); ++i) {
                    uint64_t aid = kv_ex.second[i]->get_allocation_id();
                    rb_storage[aid] = std::move(rb_res.stream_bytes[i]);
                    handle_to_rb_bytes[aid] = &rb_storage[aid];
                }
            }
        }
    }

    for (const auto & s : plan.bundle.segments) {
        auto live = xkv_store->get_segment_version(s->segment_id, s->segment_version);
        if (live) {
            if (!xkv_restore_segments_identical(live, s, handle_to_rb_bytes)) {
                refuse("conflicting live segment for restored bundle");
            }
            continue; // shared handle reused, never duplicated
        }
        need.push_back(s);
    }
    if (!need.empty()) {
        // New segments require the atomic store import. The narrowed set always
        // equals the full plan set here: any already-present shared segment
        // would have been reused above, and a partially overlapping store
        // cannot merge, so import (which itself demands zero factored state)
        // is the only consistent path. Fenced per TransactionCore contract.
        if (!xkv_tx_coord) {
            refuse("no transaction coordinator for fenced import");
        }
        std::string ferr;
        llama_xkv::xkv_quiesce_options qo;
        qo.wait = true;
        if (xkv_tx_coord->wait_for_readers_drained(qo, &ferr) != llama_xkv::xkv_tx_status::ok) {
            refuse("reader drain refused import: " + ferr);
        }
        // NOTE to TransactionCore: no restore op exists yet; seal is the closest
        // existing exclusion fence. Please add xkv_maintenance_op::restore.
        llama_xkv::xkv_maintenance_guard mguard(
            xkv_tx_coord.get(), llama_xkv::xkv_maintenance_op::seal,
            plan.bundle.next_seal_tx_nonce, &ferr);
        if (!mguard.held()) {
            refuse("maintenance guard refused import: " + ferr);
        }
        // Adapt to the store contract: factored locations only (hot/flat never
        // install pool/dense rows from state), zero nonces, nonzero generations.
        llama_xkv::llama_xkv_cache_store::xkv_snapshot_import_bundle sb;
        sb.segments = need;
        uint64_t sealed = 0;
        for (const auto & pl : plan.bundle.locations) {
            if (pl.second.kind != llama_xkv::xkv_location_kind::factored) {
                continue;
            }
            if (pl.second.seal_tx_nonce != 0 || pl.second.storage_generation == 0) {
                refuse("importable location violates zero-nonce/nonzero-generation contract");
            }
            sb.locations.push_back(pl);
            ++sealed;
        }
        sb.stamp              = plan.bundle.stamp;
        sb.sealed_count       = sealed;
        sb.next_segment_id    = plan.bundle.next_segment_id;
        sb.next_seal_tx_nonce = plan.bundle.next_seal_tx_nonce;
        sb.next_alloc_id      = plan.bundle.next_alloc_id;
        // Pre-commit dedup fit against the validated bundle cap (the store
        // enforces its live cap internally as well).
        {
            std::map<const llama_xkv::encoded_matrix *, size_t> seen;
            uint64_t dedup = 0;
            for (const auto & s : sb.segments) {
                for (const auto & g : s->groups) {
                    if (seen.emplace(g.b_k.get(), g.b_k->bytes.size()).second) {
                        dedup += g.b_k->bytes.size();
                    }
                    if (seen.emplace(g.b_v.get(), g.b_v->bytes.size()).second) {
                        dedup += g.b_v->bytes.size();
                    }
                    dedup += g.a_k.bytes.size() + g.a_v.bytes.size() + g.landmark.bytes.size();
                }
            }
            if (dedup > plan.bundle.store_cap_bytes) {
                refuse("import footprint exceeds validated bundle cap");
            }
        }

        // DEVICE_OWNED restoration path: upload state code streams via
        // xkv_backend_import_code_streams under StoreReservation, attach
        // immutable backend_bundle + real executor owner, and release host bytes.
        // CPU reference flows skip this and keep host bytes.
        llama_xkv::xkv_capacity_reservation store_cap_handle;
        const bool needs_device = llama_xkv_profile_is_device_owned(
            (llama_xkv_storage_profile) plan.bundle.config.profile,
            (llama_xkv_factorizer) plan.bundle.config.factorizer);
        if (needs_device) {
            ggml_tensor * k0 = get_k_storage(0);
            ggml_backend_buffer_type_t buft0 = (k0 && k0->buffer) ? ggml_backend_buffer_get_type(k0->buffer) : nullptr;
            ggml_backend_dev_t dev0 = buft0 ? ggml_backend_buft_get_device(buft0) : nullptr;
            if (dev0) {
                std::shared_ptr<struct ggml_backend> exec_owner = llama_xkv::xkv_backend_make_owner(ggml_backend_dev_init(dev0, nullptr));
                ggml_backend_t exec_b = exec_owner.get();
                if (!exec_b) {
                    refuse("cannot initialize backend device for code-stream import");
                }

                // Build backend import streams from image allocations preserving dedup_key
                std::vector<llama_xkv::xkv_backend_import_stream> b_streams;
                b_streams.reserve(img.allocations.size());
                for (const auto & a : img.allocations) {
                    llama_xkv::xkv_backend_import_stream bs;
                    bs.desc = a.desc;
                    bs.data = a.bytes.data();
                    bs.size = a.bytes.size();
                    bs.expected_checksum = a.checksum;
                    bs.desc_bytes = a.desc_bytes.data();
                    bs.desc_size = a.desc_bytes.size();
                    bs.dedup_key = (uint64_t) a.alloc_id;
                    b_streams.push_back(bs);
                }

                llama_xkv::xkv_backend_batch_config b_cfg;
                b_cfg.residency = GGML_XKV_RES_DEVICE_OWNED;
                b_cfg.enforce_residency = true;
                b_cfg.backend_owner = exec_owner;

                // Calculate exact newly-staged dedup actual bytes (backend alignment + segment metadata)
                size_t exact_staged_bytes = 0;
                for (const auto & bs : b_streams) {
                    size_t raw_sz = bs.desc.logical_shape.rows > 0 ? llama_xkv::encoded_matrix_bytes(bs.desc) : bs.size;
                    size_t align = buft0 ? ggml_backend_buft_get_alignment(buft0) : 128;
                    size_t stream_aligned = (raw_sz + align - 1) & ~(align - 1);
                    exact_staged_bytes += stream_aligned;
                }
                for (const auto & seg : sb.segments) {
                    exact_staged_bytes += sizeof(*seg);
                }

                size_t deficit = 0;
                std::string cap_err;
                store_cap_handle = xkv_store->reserve_capacity(
                    exact_staged_bytes > 0 ? exact_staged_bytes : (size_t) plan.bundle.store_cap_bytes, &cap_err, &deficit);
                if (!store_cap_handle.valid()) {
                    refuse("DEVICE_OWNED store capacity reservation failed: " + cap_err);
                }
                llama_xkv::xkv_backend_store_reservation s_res = store_cap_handle.backend_reservation();

                // Advance/seed the store-owned persistent ID generator from the image's
                // validated high-water next_alloc_id so new imported allocations are >= high-water
                // and cannot collide with previously issued IDs.
                auto & id_gen = xkv_store->allocation_id_generator();
                if (plan.bundle.next_alloc_id > id_gen.current_id()) {
                    id_gen.reset(plan.bundle.next_alloc_id);
                }

                llama_xkv::xkv_backend_import_result b_res;
                std::string b_err;
                if (!llama_xkv::xkv_backend_import_code_streams(
                        exec_b, buft0, b_streams, b_cfg,
                        id_gen,
                        b_res, &b_err, &s_res)) {
                    refuse("DEVICE_OWNED xkv_backend_import_code_streams failed: " + b_err);
                }

                // Verify whole bundle
                if (!llama_xkv::xkv_backend_bundle_verify(b_res.handles, GGML_XKV_RES_DEVICE_OWNED, false, &b_err)) {
                    refuse("DEVICE_OWNED imported handles bundle verification failed: " + b_err);
                }

                // Verify batch_res generation:
                auto batch_res = b_res.make_batch_result(exec_b, exec_owner);
                if (!batch_res || !batch_res->is_success() || !batch_res->is_committed()) {
                    refuse("DEVICE_OWNED make_batch_result failed or uncommitted");
                }

                // Attach immutable backend_bundle subsets matched by allocation/fingerprint to each segment,
                // and clear ALL host factor bytes (A and B) post-upload so zero host mirrors remain.
                // Mapping uses wire stream index b_res.handles[wire_stream_index] verified by descriptor+checksum,
                // never ambiguous descriptor-only match and never comparing new runtime ID to serialized ID!
                for (auto & seg : sb.segments) {
                    auto mut_seg = std::const_pointer_cast<llama_xkv::xkv_segment>(seg);
                    mut_seg->residency = GGML_XKV_RES_DEVICE_OWNED;

                    // Construct segment-specific bundle subset
                    llama_xkv::xkv_backend_import_result seg_res;
                    for (const auto & iseg : img.segments) {
                        if (iseg.segment_id == mut_seg->segment_id && iseg.segment_version == mut_seg->segment_version) {
                            for (const auto & ig : iseg.groups) {
                                auto add_wire_handle = [&](uint32_t wire_stream_idx) {
                                    if (wire_stream_idx >= b_res.handles.size() || wire_stream_idx >= img.allocations.size()) {
                                        refuse("DEVICE_OWNED wire stream index out of range");
                                    }
                                    const auto & h = b_res.handles[wire_stream_idx];
                                    const auto & a = img.allocations[wire_stream_idx];
                                    if (!h || !(h->get_desc() == a.desc)) {
                                        refuse("DEVICE_OWNED imported handle descriptor mismatch with wire allocation");
                                    }
                                    seg_res.handles.push_back(h);
                                };
                                add_wire_handle(ig.stream_ak);
                                add_wire_handle(ig.stream_bk);
                                add_wire_handle(ig.stream_av);
                                add_wire_handle(ig.stream_bv);
                                if (ig.has_landmark) {
                                    add_wire_handle(ig.stream_landmark);
                                }
                            }
                            break;
                        }
                    }
                    auto seg_batch_res = seg_res.make_batch_result(exec_b, exec_owner);
                    if (!seg_batch_res || !seg_batch_res->is_success()) {
                        refuse("DEVICE_OWNED segment make_batch_result failed");
                    }
                    mut_seg->backend_bundle = seg_batch_res;
                    // Release host bytes truthfully post-upload (both A and B)
                }

                // Deterministic host memory release: swap with empty std::vector<uint8_t>()
                // guarantees zero allocated capacity and releases accounting truthfully.
                // A and landmark streams are per-segment; shared B handles are cleared
                // exactly once after all segment mappings have succeeded!
                std::set<const llama_xkv::encoded_matrix *> cleared_shared_b;
                for (auto & seg : sb.segments) {
                    auto mut_seg = std::const_pointer_cast<llama_xkv::xkv_segment>(seg);
                    for (auto & g : mut_seg->groups) {
                        std::vector<uint8_t>().swap(g.a_k.bytes);
                        std::vector<uint8_t>().swap(g.a_v.bytes);
                        std::vector<uint8_t>().swap(g.landmark.bytes);
                        if (g.b_k && cleared_shared_b.insert(g.b_k.get()).second) {
                            auto mut_bk = std::const_pointer_cast<llama_xkv::encoded_matrix>(g.b_k);
                            std::vector<uint8_t>().swap(mut_bk->bytes);
                        }
                        if (g.b_v && cleared_shared_b.insert(g.b_v.get()).second) {
                            auto mut_bv = std::const_pointer_cast<llama_xkv::encoded_matrix>(g.b_v);
                            std::vector<uint8_t>().swap(mut_bv->bytes);
                        }
                    }
                }
            } else {
                refuse("DEVICE_OWNED profile requires a valid device/buft (fail-closed, no silent fallback)");
            }
        }

        llama_xkv::xkv_capacity_reservation * cap_res_ptr = store_cap_handle.valid() ? &store_cap_handle : nullptr;
        if (!xkv_store->import_snapshot_segments(sb, &err, cap_res_ptr)) {
            refuse("store import refused: " + err);
        }
        xkv_tx_coord->sync_from_store(xkv_store->current_stamp());
    } else {
        // Shared steady state: every plan segment already present byte-identical
        // and no import was attempted. Verify all factored locations against the
        // live store before rebind (exact match; racing maintenance refuses).
        for (const auto & pl : plan.bundle.locations) {
            if (pl.second.kind != llama_xkv::xkv_location_kind::factored) {
                continue;
            }
            llama_xkv::xkv_location live;
            if (!xkv_store->find_location(pl.first, live) || !(live == pl.second)) {
                refuse("live store diverged from restored bindings");
            }
        }
    }
    // Rebind factored cells with zero store mutation on this path: legacy
    // ordinal for whole-cache, granted order for per-seq. Hot/flat stay dense.
    try {
        for (const auto & b : bindings) {
            const uint32_t target = per_seq ? granted->idxs[0][b.ordinal] : b.ordinal;
            auto & cells = v_cells[b.stream_idx];
            if (cells.payload_id_get(target) != 0) {
                refuse("restored cell already bound (unexpected pre-bound state)");
            }
            cells.payload_id_set(target, b.payload_id, b.generation);
        }
    } catch (const std::bad_alloc &) {
        refuse("cell rebind allocation failure");
    }
    // Verify every rebound cell against live store locations.
    for (const auto & b : bindings) {
        const uint32_t target = per_seq ? granted->idxs[0][b.ordinal] : b.ordinal;
        if (v_cells[b.stream_idx].payload_id_get(target) != b.payload_id) {
            refuse("rebind did not stick (logic error)");
        }
        llama_xkv::xkv_location loc;
        if (!xkv_store->find_location(b.payload_id, loc) ||
            loc.kind != llama_xkv::xkv_location_kind::factored ||
            loc.storage_generation != b.generation) {
            refuse("rebound cell without live factored location");
        }
    }

    // Invalidate decoded tile cache upon successful state restore so restored
    // streams cannot hit stale pre-restore decoded tiles (fail-closed, §13).
    if (xkv_runtime) {
        xkv_runtime->clear_decode_cache();
    }
}

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        // Restored bytes are ordinary KV with no RERoT classification: drop
        // any write tag/reader view so apply_ubatch() below cannot mis-tag
        // them. (Whole-cache restore goes through clear(), which already
        // resets cells, tags, and views.)
        rerot_clear_write_tag(dest_seq_id);
        rerot_clear_reader_view(dest_seq_id);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));

                ubatch.pos[i + ubatch.n_tokens]   = ext.y;
                ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        sinfo = find_slot(ubatch, false);
        if (sinfo.empty()) {
            LLAMA_LOG_ERROR("%s: failed to find %d available cells in kv cache\n", __func__,  cell_count);
            return false;
        }

        // TODO: we cannot yet restore llama_kv_cell_ext as the apply_ubatch() does not support it yet
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        apply_ubatch(sinfo, ubatch);

        LLAMA_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, cell_count, dest_seq_id);

        // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(sinfo.idxs[0].size() == cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const uint32_t idx = sinfo.idxs[0][i];
            GGML_ASSERT(cells.pos_get(idx) == ubatch.pos[i]);
            GGML_ASSERT(cells.seq_has(idx, dest_seq_id));
        }
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));
                cells.ext_set(i, ext);
            }

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }

            if (xkv_store) {
                xkv_store->register_hot_payload(cells.payload_id_get(i), i, cells.storage_generation_get(i));
            }
        }

        // Create contiguous slot_info for whole cache restore
        sinfo.s0 = strm;
        sinfo.s1 = strm;
        sinfo.resize(1);
        sinfo.strm[0] = strm;
        sinfo.idxs[0].resize(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            sinfo.idxs[0][i] = i;
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo) {
    auto & cells = v_cells[strm];

    uint32_t v_trans;
    uint32_t n_layer;

    io.read(&v_trans, sizeof(v_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        auto * k = layer.k_stream[strm];

        // Use actual tensor width (may be padded for turbo types)
        const uint32_t n_embd_k_gqa = (uint32_t) k->ne[0];

        // Read type of key
        int32_t k_type_i_ref;
        io.read(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            if (sinfo.is_contiguous()) {
                // Fast path: contiguous cells, single memcpy
                io.read_tensor(k, sinfo.head() * k_size_row, cell_count * k_size_row);
            } else {
                // Slow path: scatter to non-contiguous positions
                for (uint32_t i = 0; i < cell_count; ++i) {
                    const size_t dst_offset = sinfo.idxs[0][i] * k_size_row;
                    io.read_tensor(k, dst_offset, k_size_row);
                }
            }
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Use actual tensor width (may be padded for turbo types)
            const uint32_t n_embd_v_gqa = (uint32_t) v->ne[0];

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells, single memcpy
                    io.read_tensor(v, sinfo.head() * v_size_row, cell_count * v_size_row);
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        const size_t dst_offset = sinfo.idxs[0][i] * v_size_row;
                        io.read_tensor(v, dst_offset, v_size_row);
                    }
                }
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells
                    const uint32_t h = sinfo.head();
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        const size_t dst_offset = (h + j * cells.size()) * v_size_el;
                        io.read_tensor(v, dst_offset, cell_count * v_size_el);
                    }
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        for (uint32_t i = 0; i < cell_count; ++i) {
                            const size_t dst_offset = (sinfo.idxs[0][i] + j * cells.size()) * v_size_el;
                            io.read_tensor(v, dst_offset, v_size_el);
                        }
                    }
                }
            }
        }
    }

    return true;
}

//
// llama_kv_cache_context
//

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) : status(status) {}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    // Reserve graph. The widest a single forward can ever look is one sequence's context: prefill
    // runs exactly one sequence and a decode token only ever attends within its own, and the two
    // never run together. So no live graph reaches past n_ctx_train, while the pool (which is deep
    // for the sake of concurrency) is storage, not scratch. Sizing this by the pool made the
    // scheduler allocate working buffers for it (measured: 4.1 GiB per device at a 262144-token
    // pool against 9.3 GiB at 524288, all of it compute and none of it needed by any forward).
    n_kv = (int32_t) std::min<uint32_t>(kv->get_size(), kv->get_hparams().n_ctx_train);

    const uint32_t n_stream = kv->get_n_stream();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_context * lctx,
        bool do_shift,
        stream_copy_info sc_info) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), sc_info(std::move(sc_info)) {
    if (!do_shift && this->sc_info.empty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_kv_cache::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches,
        std::vector<llama_xkv::xkv_hot_reservation> hot_reservations) :
    status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)),
    hot_reservations(std::move(hot_reservations)), ubatches(std::move(ubatches)) {
    // Pre-compute n_kv so read-only paths (e.g. MTP draft, which calls process_ubatch with
    // apply_mctx=false) get a valid mask/K/V shape based on current cache occupancy.
    // For paths that DO call apply(), n_kv is re-derived there after apply_ubatch().
    if (!this->sinfos.empty()) {
        n_kv = (int32_t) kv->get_n_kv(this->sinfos[0]);
    } else {
        n_kv = 0;
    }
}

llama_kv_cache_context::~llama_kv_cache_context() = default;

bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    rerot_layout_ready = false;
    rerot_layout = {};

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, sc_info);

        return true;
    }

    postcompute_finalized = false;
    const auto & sinfo = sinfos[i_cur];
    const bool bounded = kv->is_xkv_bounded_hot();
    const bool has_res = i_cur < hot_reservations.size() && hot_reservations[i_cur].valid();

    // apply_ubatch binds new cells + store entries and defers victim release
    // until postcompute_success. It repairs cells on store-register failure
    // and throws only after restoring the exact pre-apply state.
    llama_kv_cache::xkv_overwrite_victims victims;
    try {
        kv->apply_ubatch(sinfo, ubatches[i_cur], false, bounded ? &victims : nullptr);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: apply_ubatch failed: %s\n", __func__, e.what());
        return false;
    }

    // Record applied entries for postcompute commit/rollback.
    const size_t entry_base = applied_entries.size();
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = kv->get_cells(sinfo.strm[s]);
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t cell_idx = sinfo.idxs[s][ii];
            llama_kv_cache::xkv_applied_entry e;
            e.stream = sinfo.strm[s];
            e.cell   = cell_idx;
            e.pid    = cells.payload_id_get(cell_idx);
            e.gen    = cells.storage_generation_get(cell_idx);
            e.has_hot = bounded && !sinfo.hot_idxs.empty() && s < sinfo.hot_idxs.size() &&
                        ii < sinfo.hot_idxs[s].size();
            e.hot_slot = e.has_hot ? sinfo.hot_idxs[s][ii] : 0;
            if (e.has_hot) {
                LLAMA_LOG_ERROR("[llama_kv_cache] apply_ubatch: bound cell %u to hot_slot %u (pid %llu, gen %llu)\n",
                    cell_idx, e.hot_slot, (unsigned long long)e.pid, (unsigned long long)e.gen);
            }
            applied_entries.push_back(e);
        }
    }

    // For bounded hot cache, commit the reservation binding payload IDs and generations.
    // Victim release stays deferred: it runs only at postcompute_success.
    if (has_res) {
        std::vector<uint64_t> pids;
        std::vector<uint64_t> gens;
        pids.reserve(applied_entries.size() - entry_base);
        gens.reserve(applied_entries.size() - entry_base);
        for (size_t k = entry_base; k < applied_entries.size(); ++k) {
            pids.push_back(applied_entries[k].pid);
            gens.push_back(applied_entries[k].gen);
        }
        std::string err;
        if (!hot_reservations[i_cur].commit(pids, gens, &err)) {
            LLAMA_LOG_ERROR("%s: failed to commit hot reservation: %s\n", __func__, err.c_str());
            // Gated rollback: the store-held gate removes new entries with an
            // empty pool triple (nothing bound yet). On false the store/cells
            // stay exactly as applied (bit-identical to the pre-commit
            // attempt): victims and the uncommitted reservation are left
            // alone and the failure propagates.
            std::string rb_err;
            if (!kv->rollback_applied_entries(applied_entries, entry_base, &rb_err)) {
                LLAMA_LOG_ERROR("%s: apply rollback refused: %s\n", __func__, rb_err.c_str());
                return false;
            }
            applied_entries.resize(entry_base);
            kv->restore_victim_cells(victims);
            hot_reservations[i_cur].rollback();
            return false;
        }
    }
    pending_victims.push_back(std::move(victims));

    n_kv = kv->get_n_kv(sinfos[i_cur]);
    rerot_layout_ready = false;
    rerot_layout = {};

    // InnerQ: check if CUDA calibration finalized and tensor needs update
    if (kv->get_turbo_innerq_scale_inv() != nullptr && turbo_innerq_needs_tensor_update()) {
        ggml_tensor * t = kv->get_turbo_innerq_scale_inv();
        if (t->buffer != nullptr) {
            ggml_backend_tensor_set(t, g_innerq_scale_inv_host, 0, INNERQ_MAX_CHANNELS * sizeof(float));
            turbo_innerq_mark_tensor_updated();
            LLAMA_LOG_INFO("%s: InnerQ scale_inv tensor updated\n", __func__);
        }
    }

    return true;
}

bool llama_kv_cache_context::postcompute_success() {
    // Exactly-once: repeat calls return the recorded outcome.
    if (postcompute_finalized) {
        return postcompute_ok;
    }
    if (!kv) {
        postcompute_finalized = true;
        postcompute_ok = true;
        return true;
    }
    // Atomic commit: applied hot_writing rows become hot_committed FIRST.
    // Deferred victims release only after a successful commit, so a commit
    // failure returns false with victims still bound and bookkeeping retained
    // for the failure path (no partial release, no silent continuation).
    // A failed commit does NOT finalize: postcompute_failure (or a retried
    // success) may still run. Only spent states finalize.
    auto store = kv->get_xkv_store();
    if (store && !applied_entries.empty()) {
        std::vector<uint64_t> pids;
        pids.reserve(applied_entries.size());
        for (const auto & e : applied_entries) {
            pids.push_back(e.pid);
        }
        if (!store->commit_hot_payloads(pids)) {
            LLAMA_LOG_ERROR("%s: commit_hot_payloads failed; victims retained, entries retained\n", __func__);
            postcompute_ok = false;
            return false;
        }
        kv->verify_bound_hot_rows(applied_entries);
    }
    for (const auto & victims : pending_victims) {
        if (!victims.empty()) {
            std::string err;
            if (!kv->release_collected_victims(victims, &err)) {
                // Committed above; victim bookkeeping is spent. Report the
                // failure instead of continuing silently.
                LLAMA_LOG_ERROR("%s: failed to release overwrite victims: %s\n", __func__, err.c_str());
                applied_entries.clear();
                pending_victims.clear();
                postcompute_finalized = true;
                postcompute_ok = false;
                return false;
            }
        }
    }
    applied_entries.clear();
    pending_victims.clear();
    postcompute_finalized = true;
    postcompute_ok = true;
    return true;
    }

bool llama_kv_cache_context::postcompute_failure() {
    // Exactly-once: repeat calls return the recorded outcome.
    if (postcompute_finalized) {
        return postcompute_ok;
    }
    if (!kv) {
    postcompute_finalized = true;
        postcompute_ok = true;
        return true;
    }
    // One coordinated rollback transaction: store removal of new payloads,
    // then pool release of new rows, then new-cell drops, then victim-cell
    // restores. Victim pool/store entries were never released, so no
    // re-binding is needed. Uncommitted reservations (including
    // never-applied later ubatches) roll back below; committed ones had
    // their rows freed by the transaction.
    // A failed transaction does NOT finalize: records are retained and the
    // rollback (or a commit retry) may run again. Only spent states
    // finalize; repeats after true return the recorded outcome.
    std::string err;
    const bool tx_ok = kv->rollback_applied_batch(applied_entries, pending_victims, &err);
    if (!tx_ok) {
        LLAMA_LOG_ERROR("%s: rollback transaction incomplete: %s\n", __func__, err.c_str());
        postcompute_ok = false;
        return false;
    }
    postcompute_finalized = true;
        postcompute_ok = true;
    applied_entries.clear();
    pending_victims.clear();
    for (auto & res : hot_reservations) {
        res.rollback();
    }
        return true;
    }

llama_memory_status llama_kv_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    return n_kv;
}

uint32_t llama_kv_cache_context::get_n_kv_pos_contiguous() const {
    // Full-cache and update contexts do not carry a concrete ubatch/slot pair.
    if (ubatches.empty() || sinfos.empty() || i_cur >= ubatches.size() || i_cur >= sinfos.size()) {
        // reserve context: report the whole cache as position-contiguous so the worst-case graph
        // is the banded path; reserving the dense fallback is unallocatable at large n_ctx
        if (kv != nullptr && lctx == nullptr && kv->get_n_stream() == 1) {
            return n_kv;
        }
        return 0;
    }

    const uint32_t result = kv->get_n_kv_pos_contiguous(sinfos[i_cur], ubatches[i_cur]);
    return result <= (uint32_t) n_kv ? result : 0;
}

ggml_type llama_kv_cache_context::type_k() const {
    return kv->type_k();
}

ggml_type llama_kv_cache_context::type_v() const {
    return kv->type_v();
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_xkv_hot_k(ggml_context * ctx, int32_t il) const {
    return kv->get_hot_k(ctx, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_xkv_hot_v(ggml_context * ctx, int32_t il) const {
    return kv->get_hot_v(ctx, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_turbo_rotation() const {
    return kv->get_turbo_rotation();
}

ggml_tensor * llama_kv_cache_context::get_turbo_rotation_inv() const {
    return kv->get_turbo_rotation_inv();
}

ggml_tensor * llama_kv_cache_context::get_turbo_rot_forward() const {
    return kv->get_turbo_rotation();
}

ggml_tensor * llama_kv_cache_context::get_turbo_rot_inverse() const {
    return kv->get_turbo_rotation_inv();
}

ggml_tensor * llama_kv_cache_context::get_turbo_innerq_scale_inv() const {
    return kv->get_turbo_innerq_scale_inv();
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_context::build_input_v_rot(ggml_context * ctx) const {
    return kv->build_input_v_rot(ctx);
}

bool llama_kv_cache_context::rerot_active() const {
    return !ubatches.empty() && i_cur < ubatches.size() && kv->rerot_batch_active(ubatches[i_cur]);
}

const llama_rerot_attn_layout & llama_kv_cache_context::get_rerot_attn_layout() const {
    if (!rerot_layout_ready) {
        rerot_layout = rerot_active()
            ? kv->rerot_build_attn_layout(ubatches[i_cur], n_kv)
            : llama_rerot_attn_layout{};
        rerot_layout_ready = true;
    }
    return rerot_layout;
}

ggml_tensor * llama_kv_cache_context::build_input_rerot_q_indices(ggml_context * ctx) const {
    const auto & layout = get_rerot_attn_layout();
    GGML_ASSERT(!layout.empty() && !layout.groups.empty());
    auto * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, layout.groups.size());
    ggml_set_input(result);
    return result;
}

ggml_tensor * llama_kv_cache_context::build_input_rerot_q_pos(ggml_context * ctx, uint32_t n_pos) const {
    const auto & layout = get_rerot_attn_layout();
    GGML_ASSERT(!layout.empty() && !layout.groups.empty());
    GGML_ASSERT(n_pos == 1 || n_pos == 4);
    auto * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, layout.groups.size() * n_pos);
    ggml_set_input(result);
    return result;
}

ggml_tensor * llama_kv_cache_context::build_input_rerot_entries(ggml_context * ctx) const {
    const auto & layout = get_rerot_attn_layout();
    GGML_ASSERT(!layout.empty() && !layout.entries.empty());
    auto * result = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, layout.entries.size());
    ggml_set_input(result);
    return result;
}

ggml_tensor * llama_kv_cache_context::build_input_rerot_offsets(ggml_context * ctx) const {
    const auto & layout = get_rerot_attn_layout();
    GGML_ASSERT(!layout.empty());
    auto * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, layout.query_offsets.size());
    ggml_set_input(result);
    return result;
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_rerot_q_indices(ggml_tensor * dst) const {
    const auto & layout = get_rerot_attn_layout();
    GGML_ASSERT(dst && dst->type == GGML_TYPE_I32 && (size_t) dst->ne[0] == layout.groups.size());
    std::vector<int32_t> data(layout.groups.size());
    for (size_t i = 0; i < layout.groups.size(); ++i) {
        data[i] = static_cast<int32_t>(layout.groups[i].query_index);
    }
    ggml_backend_tensor_set(dst, data.data(), 0, data.size() * sizeof(data[0]));
}

void llama_kv_cache_context::set_input_rerot_q_pos(ggml_tensor * dst, uint32_t n_pos) const {
    const auto & layout = get_rerot_attn_layout();
    GGML_ASSERT(dst && dst->type == GGML_TYPE_I32 && (n_pos == 1 || n_pos == 4));
    GGML_ASSERT((size_t) dst->ne[0] == layout.groups.size() * n_pos);
    std::vector<int32_t> data(layout.groups.size() * n_pos, 0);
    for (size_t i = 0; i < layout.groups.size(); ++i) {
        const auto pos = layout.groups[i].effective_pos;
        data[i] = pos;
        if (n_pos == 4) {
            data[layout.groups.size() + i] = pos;
            data[2 * layout.groups.size() + i] = pos;
            data[3 * layout.groups.size() + i] = 0;
        }
    }
    ggml_backend_tensor_set(dst, data.data(), 0, data.size() * sizeof(data[0]));
}

void llama_kv_cache_context::set_input_rerot_entries(ggml_tensor * dst) const {
    const auto & layout = get_rerot_attn_layout();
    GGML_ASSERT(dst && dst->type == GGML_TYPE_I32 && dst->ne[0] == 2 &&
        (size_t) dst->ne[1] == layout.entries.size());
    std::vector<int32_t> data(layout.entries.size() * 2);
    for (size_t i = 0; i < layout.entries.size(); ++i) {
        data[2 * i + 0] = static_cast<int32_t>(layout.entries[i].key_index);
        data[2 * i + 1] = static_cast<int32_t>(layout.entries[i].group_index);
    }
    ggml_backend_tensor_set(dst, data.data(), 0, data.size() * sizeof(data[0]));
}

void llama_kv_cache_context::set_input_rerot_offsets(ggml_tensor * dst) const {
    const auto & layout = get_rerot_attn_layout();
    GGML_ASSERT(dst && dst->type == GGML_TYPE_I32 &&
        (size_t) dst->ne[0] == layout.query_offsets.size());
    std::vector<int32_t> data(layout.query_offsets.size());
    for (size_t i = 0; i < layout.query_offsets.size(); ++i) {
        data[i] = static_cast<int32_t>(layout.query_offsets[i]);
    }
    ggml_backend_tensor_set(dst, data.data(), 0, data.size() * sizeof(data[0]));
}

void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_context::set_input_pos_rel_flat(ggml_tensor * dst, const llama_ubatch * ubatch, uint32_t extent) const {
    kv->set_input_pos_rel_flat(dst, ubatch, extent);
}

void llama_kv_cache_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

void llama_kv_cache_context::set_input_v_rot(ggml_tensor * dst) const {
    kv->set_input_v_rot(dst);
}

void llama_kv_cache_context::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    kv->get_prev_tokens(ubatch, n, res);
}

std::shared_ptr<llama_xkv::llama_xkv_cache_store> llama_kv_cache::get_xkv_store() const {
    return xkv_store;
}

void llama_kv_cache::init_xkv_store(const llama_cparams & cparams) {
    if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
        return;
    }
    // Custom hot-cache rotation (attn_rot_k/v) is supported on every XKV path:
    // the CPU reference inverts it at gather, and the native builder runs the
    // chain in the rotated domain (see xkv_build_attention_native). No gate here.
    if (cparams.xkv_mode != LLAMA_XKV_MODE_OFF) {
        if (!xkv_store) {
            if (other && other->xkv_store) {
                xkv_store = other->xkv_store;
            } else {
                xkv_store = std::make_shared<llama_xkv::llama_xkv_cache_store>(cparams);
            }
        }
        // Register layer aliases deterministically from map_layer_ids
        if (xkv_store) {
            for (const auto & kv_pair : map_layer_ids) {
                const uint32_t model_layer = (uint32_t) kv_pair.first;
                const size_t   idx         = (size_t) kv_pair.second;
                if (idx < layers.size()) {
                    const uint32_t owning_layer = layers[idx].il;
                    xkv_store->register_layer_alias(model_layer, owning_layer);
                }
            }
        }
        // Cache-owned runtime coordinator + single transaction coordinator.
        // One runtime + one coordinator per shared-store family: inherit
        // both from `other` when it shares this store, else create fresh.
        // A fresh per-cache pair could race maintenance on one store.
        // Non-owner/MTP caches never receive a runtime (MTP returns above),
        // so their layer gate and maintenance stay false.
        if (other && other->xkv_store == xkv_store && other->xkv_runtime && other->xkv_tx_coord) {
            xkv_runtime = other->xkv_runtime;
            xkv_tx_coord = other->xkv_tx_coord;
        } else {
            if (!xkv_tx_coord) {
                xkv_tx_coord = std::make_shared<llama_xkv::xkv_transaction_coordinator>();
            }
            if (!xkv_runtime) {
                xkv_runtime = llama_xkv::llama_xkv_runtime::create(cparams);
            }
        }
        if (xkv_runtime) {
            xkv_runtime->set_coordinator(xkv_tx_coord.get());
            set_removal_rebuild_hook(xkv_runtime->bind_removal_hook(*this));
        }
        // Bind the coordinator to the authoritative store stamp at creation;
        // maintain() re-syncs before taking the exclusion (never a stale
        // zero-stamp gate). Full provider/delegate sync lives with the
        // transaction owner.
        if (xkv_tx_coord && xkv_store) {
            xkv_tx_coord->sync_from_store(xkv_store->current_stamp());
        }
    }
}

llama_kv_cache::~llama_kv_cache() = default;

llama_xkv::llama_xkv_runtime * llama_kv_cache::get_xkv_runtime() {
    return xkv_runtime.get();
}

const llama_xkv::llama_xkv_runtime * llama_kv_cache::get_xkv_runtime() const {
    return xkv_runtime.get();
}

std::shared_ptr<struct ggml_backend> llama_kv_cache::get_xkv_executor(ggml_backend_buffer_type_t buft) const {
    if (other) {
        return other->get_xkv_executor(buft);
    }
    if (!xkv_store || !buft) {
        return nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (!dev) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(xkv_executor_mutex);
    const auto found = xkv_executors.find(dev);
    if (found != xkv_executors.end()) {
        return found->second;
    }

    ggml_backend_t raw = ggml_backend_dev_init(dev, nullptr);
    if (!raw) {
        return nullptr;
    }
    std::shared_ptr<struct ggml_backend> owner(raw, ggml_backend_free);
    try {
        xkv_executors.emplace(dev, owner);
    } catch (...) {
        return nullptr;
    }
    return owner;
}

bool llama_kv_cache::get_admission_snapshot(struct llama_memory_admission_snapshot * out) const {
    if (out == nullptr) {
        return false;
    }
    if (xkv_runtime) {
        std::string err;
        if (!xkv_runtime->fill_admission_snapshot(*this, *out, &err)) {
            return false;
        }
        return out->logical_capacity != 0;
    }
    // No runtime (OFF/MTP/rejected): exact legacy mapping via base default.
    return llama_memory_i::get_admission_snapshot(out);
}

llama_memory_maintenance_status llama_kv_cache::maintain_safe_boundary() {
    if (!xkv_runtime) {
        return LLAMA_MEMORY_MAINTENANCE_NO_ACTION; // XKV OFF/MTP: always a no-op
    }
    std::string err;
    // Explicit server maintenance request is treated as pressure (forced).
    // Exactly one segment seals per call; the server repeats until the hot
    // deficit clears. No segment-count cap.
    const llama_xkv::xkv_maintenance_outcome oc = xkv_runtime->maintain(*this, 0, true, &err);
    switch (oc) {
        case llama_xkv::xkv_maintain_sealed:
            return LLAMA_MEMORY_MAINTENANCE_PROGRESS;
        case llama_xkv::xkv_maintain_no_action:
        case llama_xkv::xkv_maintain_evaluated:
            return LLAMA_MEMORY_MAINTENANCE_NO_ACTION;
        case llama_xkv::xkv_maintain_retry_stale:
            return LLAMA_MEMORY_MAINTENANCE_RETRY_STALE;
        case llama_xkv::xkv_maintain_error:
        default:
            LLAMA_LOG_ERROR("%s: xkv maintenance failed: %s\n", __func__, err.c_str());
            return LLAMA_MEMORY_MAINTENANCE_ERROR;
    }
}

bool llama_kv_cache::get_xkv_runtime_snapshot(struct llama_memory_xkv_runtime_snapshot * out) const {
    if (out == nullptr) {
        return false;
    }
    *out = {};
    if (!xkv_runtime) {
        return false;
    }
    std::string err;
    return xkv_runtime->fill_runtime_snapshot(*this, *out, &err);
}

bool llama_kv_cache_context::xkv_layer_enabled(uint32_t il) const {
    if (!kv) {
        return false;
    }
    auto * rt = kv->get_xkv_runtime();
    if (!rt) {
        return false;
    }
    return rt->xkv_layer_enabled(kv->get_owning_layer(il));
}

llama_kv_cache * llama_kv_cache_context::get_kv() const {
    return kv;
}

size_t llama_kv_cache_context::get_ubatch_index() const {
    return i_cur;
}

size_t llama_kv_cache_context::get_ubatch_count() const {
    return ubatches.size();
}

size_t llama_kv_cache_context::get_sinfo_count() const {
    return sinfos.size();
}

const llama_ubatch & llama_kv_cache_context::get_ubatch_at(size_t i) const {
    GGML_ASSERT(i < ubatches.size());
    return ubatches[i];
}

const llama_kv_cache::slot_info & llama_kv_cache_context::get_sinfo_at(size_t i) const {
    GGML_ASSERT(i < sinfos.size());
    return sinfos[i];
}
