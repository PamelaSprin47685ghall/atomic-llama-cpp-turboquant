#include "llama-kv-cache.h"
#include "llama-triattention.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
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
    const  layer_share_cb & share) :
    model(model), hparams(hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type),
    other(static_cast<llama_kv_cache *>(mem_other)),
    v_cells_impl(other ? other->v_cells_impl : std::make_shared<llama_kv_cells_vec>()),
    v_cells(*v_cells_impl) {

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
        //   0 = uniform (default)
        //   1 = q8_0 K+V for first+last 4 layers
        //   2 = q8_0 K+V for last 8 layers
        //   5 = Boundary V: first2+last2 V=turbo4, rest V=turbo2 (K unchanged)
        //   6 = V-only: last 8 V=turbo4, rest V=turbo2 (K unchanged)
        //   7 = Boundary V (recommended): first2+last2 V=q8_0, rest V=turbo2 (K unchanged)
        ggml_type layer_type_k = type_k;
        ggml_type layer_type_v = type_v;
        {
            static const int adaptive_mode = [&]() {
                const char * env = getenv("TURBO_LAYER_ADAPTIVE");
                if (env) {
                    int mode = atoi(env);
                    if (mode > 0) {
                        LLAMA_LOG_INFO("llama_kv_cache: layer-adaptive mode %d enabled (env)\n", mode);
                    }
                    return mode;
                }
                // Auto-enable Boundary V (mode 7) when V is turbo2
                if (type_v == GGML_TYPE_TURBO2_0 && hparams.n_layer() >= 8) {
                    LLAMA_LOG_INFO("llama_kv_cache: Boundary V auto-enabled for turbo2-V (opt-out: TURBO_LAYER_ADAPTIVE=0)\n");
                    return 7;
                }
                return 0;
            }();
            const bool is_turbo = (type_k == GGML_TYPE_TURBO3_0 || type_k == GGML_TYPE_TURBO4_0 || type_k == GGML_TYPE_TURBO2_0);
            const bool v_is_turbo = (type_v == GGML_TYPE_TURBO3_0 || type_v == GGML_TYPE_TURBO4_0 || type_v == GGML_TYPE_TURBO2_0);
            const uint32_t n_layer = hparams.n_layer();
            if (adaptive_mode == 1 && is_turbo && n_layer >= 8) {
                if (il < 4 || il >= n_layer - 4) {
                    layer_type_k = GGML_TYPE_Q8_0;
                    layer_type_v = GGML_TYPE_Q8_0;
                }
            } else if (adaptive_mode == 2 && is_turbo && n_layer >= 8) {
                if (il >= n_layer - 8) {
                    layer_type_k = GGML_TYPE_Q8_0;
                    layer_type_v = GGML_TYPE_Q8_0;
                }
            } else if (adaptive_mode == 5 && v_is_turbo && n_layer >= 8) {
                // Boundary V (turbo4 boundaries): first2+last2 V=turbo4, rest V=turbo2 (excluding MTP layers)
                const bool is_boundary = (il < 2 || (il < n_layer && il >= n_layer - 2));
                layer_type_v = is_boundary ? GGML_TYPE_TURBO4_0 : GGML_TYPE_TURBO2_0;
                if (il == 0) {
                    LLAMA_LOG_INFO("llama_kv_cache: Boundary V mode 5: first2+last2 V=turbo4, rest V=turbo2\n");
                }
            } else if (adaptive_mode == 6 && v_is_turbo && n_layer >= 8) {
                // V-only: last 8 V=turbo4, rest V=turbo2
                layer_type_v = (il >= n_layer - 8) ? GGML_TYPE_TURBO4_0 : GGML_TYPE_TURBO2_0;
                if (il == 0) {
                    LLAMA_LOG_INFO("llama_kv_cache: V-only LA mode 6: last8 V=turbo4, rest V=turbo2\n");
                }
            } else if (adaptive_mode == 7 && v_is_turbo && n_layer >= 8) {
                // Boundary V (recommended): first2+last2 V=q8_0, rest V=turbo2 (excluding MTP layers)
                const bool is_boundary = (il < 2 || (il < n_layer && il >= n_layer - 2));
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

        ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, layer_type_k, n_embd_k_gqa_eff, kv_size, n_stream) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, layer_type_v, n_embd_v_gqa_eff, kv_size, n_stream) : nullptr;

        has_k && ggml_format_name(k, "cache_k_l%d", il);
        has_v && ggml_format_name(v, "cache_v_l%d", il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(has_k ? ggml_view_2d(ctx, k, n_embd_k_gqa_eff, kv_size, k->nb[1], s*k->nb[2]) : nullptr);
            v_stream.push_back(has_v ? ggml_view_2d(ctx, v, n_embd_v_gqa_eff, kv_size, v->nb[1], s*v->nb[2]) : nullptr);
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
    fp_bump();
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

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
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    fp_bump();

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

    return true;
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    fp_bump();

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
        }
    }

    v_heads[s1] = v_heads[s0];

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    fp_bump();

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

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
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    fp_bump();

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

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    fp_bump();

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
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

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

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

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches));
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
        apply_ubatch(sinfo_new, ubatch);
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

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    fp_bump();

    bool updated = false;

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
    }

    return updated;
}

void llama_kv_cache::compact() {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    fp_bump();

    for (uint32_t s = 0; s < n_stream; ++s) {
        auto & cells = v_cells[s];
        auto & head  = v_heads[s];

        const auto plan = cells.make_pack_plan();

        if (plan.moves.empty()) {
            head = 0;
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
            head = plan.retained_count;
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
                throw std::runtime_error(
                    std::string("TriAttention native compaction is unsupported by KV backend/layout: ") +
                    (buffer ? ggml_backend_buffer_name(buffer) : "<null>"));
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

    fp_bump();

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

    result.physical_before = cells.get_used();
    if (result.physical_before == 0) {
        result.physical_after = 0;
        result.physical_freed = 0;
        result.capacity_satisfied = (request.required_free == 0);
        result.floor_reached = true;
        return result;
    }

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
        llama_memory_kv_reclaim_seq_hint h;
        h.seq_id = 0;
        h.logical_tokens = (uint32_t) (cells.seq_pos_max(0) >= 0 ? cells.seq_pos_max(0) + 1 : 0);
        h.tail_guard = tri_recent_window;
        h.eligible = true;
        hints.push_back(h);
    }

    for (const auto & hint : hints) {
        if (!hint.eligible) {
            continue;
        }

        const llama_seq_id seq_id = hint.seq_id;
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
                rerot_meta.active() &&
                (hint.semantic_episode_id == 0 ||
                 rerot_meta.episode_id != hint.semantic_episode_id ||
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

                if (cells.is_empty(cell_i)) {
                    continue;
                }

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
                    bool removed_any = false;
                    for (const llama_seq_id ref : hint.semantic_seq_ids) {
                        if (!cells.is_empty(cell_i) && cells.seq_has(cell_i, ref)) {
                            cells.seq_rm(cell_i, ref);
                            result.references_removed++;
                            removed_any = true;
                        }
                    }
                    if (!removed_any && !cells.is_empty(cell_i) && cells.seq_has(cell_i, seq_id)) {
                        cells.seq_rm(cell_i, seq_id);
                        result.references_removed++;
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
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.is_empty(i) && cells.seq_count(i) > 1) {
            result.shared_keep++;
        }
    }

    // Pack remaining used cells to [0, retained_count)
    const int64_t t_pack_start = ggml_time_us();
    compact();
    result.pack_us += (uint64_t) std::max<int64_t>(0, ggml_time_us() - t_pack_start);

    result.physical_after = cells.get_used();
    result.physical_freed = result.physical_before - result.physical_after;
    result.changed = (result.physical_freed > 0);
    result.capacity_satisfied = (result.physical_freed >= request.required_free);
    result.floor_reached = true;

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

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    fp_bump();

    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

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

            if (ubatch.is_pos_2d()) {
                llama_kv_cell_ext ext {
                    /*.x =*/ ubatch.pos[i + ubatch.n_tokens*2],
                    /*.y =*/ ubatch.pos[i + ubatch.n_tokens],
                };
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

uint32_t llama_kv_cache::get_kv_capacity() const {
    uint64_t result = 0;
    for (const auto & cells : v_cells) {
        result += cells.size();
    }
    return (uint32_t) std::min<uint64_t>(result, UINT32_MAX);
}

uint32_t llama_kv_cache::get_kv_used() const {
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
    // TurboQuant uses kernel-level WHT rotation -- position shift is a no-op
    if (!layers.empty() && (layers[0].k->type == GGML_TYPE_TURBO2_0 || layers[0].k->type == GGML_TYPE_TURBO3_0 || layers[0].k->type == GGML_TYPE_TURBO4_0)) { return false; }
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

    fp_bump();

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

    return true;
}

void llama_kv_cache::rerot_clear_write_tag(llama_seq_id seq_id) {
    fp_bump();
    if (seq_id >= 0 && (size_t) seq_id < rerot_write_tags.size()) {
        rerot_write_tags[seq_id].reset();
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

    fp_bump();

    std::vector<std::pair<uint32_t, uint32_t>> matches;
    GGML_ASSERT(rerot_find_run_cells(episode_id, run_id, &matches) == count);

    for (const auto & match : matches) {
        const bool published = v_cells[match.first].rerot_publish(
            match.second, episode_id, run_id, publish_epoch);
        GGML_ASSERT(published);
    }

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

    fp_bump();

    std::vector<std::pair<uint32_t, uint32_t>> matches;
    GGML_ASSERT(rerot_find_run_cells(episode_id, run_id, &matches) == count);

    for (const auto & match : matches) {
        const bool changed = v_cells[match.first].rerot_reclassify(
            match.second, episode_id, run_id, expected, replacement, publish_epoch);
        GGML_ASSERT(changed);
    }
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

    fp_bump();

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

    fp_bump();

    auto & cells = v_cells[seq_to_stream[exec_seq]];
    const size_t kept = cells.rerot_freeze_to_archive(episode_id, exec_seq, archive_seq);

    GGML_ASSERT(kept == count);
    return kept;
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
    fp_bump();
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
    return true;
}

void llama_kv_cache::rerot_clear_reader_view(llama_seq_id seq_id) {
    fp_bump();
    if (seq_id >= 0 && (size_t) seq_id < rerot_reader_views.size()) {
        rerot_reader_views[seq_id].reset();
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

    const uint64_t kv_size      = get_size();
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

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k,
            head_k_eff, hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, head_k_eff),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE] — for turbo-padded V, cache may be larger
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    // Use padded head_dim for turbo types
    const bool v_is_turbo = (v->type == GGML_TYPE_TURBO3_0 || v->type == GGML_TYPE_TURBO4_0 || v->type == GGML_TYPE_TURBO2_0);
    const uint32_t head_v = hparams.n_embd_head_v(il);
    const uint32_t head_v_eff = (v_is_turbo && head_v % 128 != 0)
        ? ((head_v + 127) / 128) * 128 : head_v;

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                head_v_eff, hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(v->type, head_v_eff),                      // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                    // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),            // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), head_v_eff, ns,
            ggml_row_size(v->type, kv_size*head_v_eff),              // v->nb[1]
            ggml_row_size(v->type, kv_size),                         // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),            // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
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
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
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
            const int64_t kv_size = get_size();

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

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }
}

void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = get_size();

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfo.idxs[s][i];
                }
            }
        }
    }
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
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

    // note: assumes k_rot^2 == I
    ggml_tensor * k_rot = nullptr;

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
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

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
    ggml_set_input(inp->k_shift);

    inp->k_rot = build_input_k_rot(ctx);

    const auto & cparams = lctx->get_cparams();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;
        const bool is_turbo_k = (layer.k->type == GGML_TYPE_TURBO2_0 || layer.k->type == GGML_TYPE_TURBO3_0 || layer.k->type == GGML_TYPE_TURBO4_0);
        if (is_turbo_k) { continue; }

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const auto n_rot         = hparams.n_rot(il);
        const auto n_embd_head_k = hparams.n_embd_head_k(il);
        const auto n_embd_nope   = hparams.n_lora_kv > 0 ? n_embd_head_k - n_rot : 0;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_rot, n_head_kv, get_size()*n_stream,
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
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    fp_bump();

    GGML_UNUSED(flags);

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

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
    }
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
    n_kv = kv->get_size();

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
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
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

    fp_key_valid = false;
    fp_layout.clear();
    fp_cell_stamps.clear();

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, sc_info);

        return true;
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    n_kv = kv->get_n_kv(sinfos[i_cur]);
    rerot_layout_ready = false;
    rerot_layout = {};

    fp_key_valid = false;
    fp_layout.clear();
    fp_cell_stamps.clear();

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
