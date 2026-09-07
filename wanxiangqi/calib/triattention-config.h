#pragma once

#include "llama-hparams.h"
#include "llama.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace wxq {

struct triattention_model_config {
    int32_t n_layer = 0;
    int32_t n_head = 0;
    int32_t n_head_kv = 0;
    int32_t head_dim = 0;
    double rope_theta = 0.0;
    uint32_t rope_style = 0;
    uint32_t rotary_dim = 0;
    uint32_t freq_count = 0;
    std::vector<int> full_attn_layers;
};

// Use the geometry resolved by the model loader, not guessed GGUF prefixes.
// In particular, Nanbeige has 44 logical layers and d_k=128, although its
// physical block count is 22 and n_embd/n_head is only 64.
inline triattention_model_config get_triattention_model_config(
        const llama_hparams & hp, llama_rope_type rope_type) {
    triattention_model_config mc;
    mc.n_layer = (int32_t) hp.n_layer();
    for (uint32_t il = 0; il < hp.n_layer(); ++il) {
        if (!hp.is_recr(il) && !hp.is_swa(il) && hp.n_head_kv(il) > 0) {
            mc.full_attn_layers.push_back((int) il);
        }
    }
    if (mc.full_attn_layers.empty()) {
        throw std::runtime_error("TriAttention calibration requires a full-attention layer");
    }

    const uint32_t first = (uint32_t) mc.full_attn_layers.front();
    mc.n_head = (int32_t) hp.n_head(first);
    mc.n_head_kv = (int32_t) hp.n_head_kv(first);
    mc.head_dim = (int32_t) hp.n_embd_head_k(first);
    mc.rotary_dim = hp.n_rot(first);
    mc.rope_theta = hp.rope_freq_base_train;
    if (mc.n_head <= 0 || mc.n_head_kv <= 0 || mc.n_head % mc.n_head_kv != 0 ||
        mc.head_dim <= 0 || mc.rotary_dim == 0 || (mc.rotary_dim & 1u) != 0 ||
        mc.rotary_dim > (uint32_t) mc.head_dim ||
        !std::isfinite(mc.rope_theta) || mc.rope_theta <= 1.0) {
        throw std::runtime_error("invalid attention/RoPE geometry for TriAttention calibration");
    }
    mc.freq_count = mc.rotary_dim / 2;
    for (int il : mc.full_attn_layers) {
        if (hp.n_head(il) != (uint32_t) mc.n_head ||
            hp.n_head_kv(il) != (uint32_t) mc.n_head_kv ||
            hp.n_embd_head_k(il) != (uint32_t) mc.head_dim ||
            hp.n_rot(il) != mc.rotary_dim) {
            throw std::runtime_error("TriAttention calibration cannot represent heterogeneous attention geometry");
        }
    }

    // IMRoPE interleaves position sections, not the vector rotation pairs.
    switch (rope_type) {
        case LLAMA_ROPE_TYPE_NORM:   mc.rope_style = 1; break;
        case LLAMA_ROPE_TYPE_NEOX:
        case LLAMA_ROPE_TYPE_MROPE:
        case LLAMA_ROPE_TYPE_IMROPE: mc.rope_style = 0; break;
        default:
            throw std::runtime_error("TriAttention calibration does not support this RoPE layout");
    }
    return mc;
}

} // namespace wxq
