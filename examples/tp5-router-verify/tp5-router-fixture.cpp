#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#if defined(TP5_FIXTURE_CONTRACT)
#  include "router-fixture-contract.h"
#endif

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Writes the smallest real Qwen4Exp checkpoint the production router-scratch
// path accepts: 512 experts / top-10 routing (the hard-wired gate in
// src/models/qwen4exp.cpp) with every hidden width shrunk to the minimum that
// ggml still accepts (n_embd = 8, two trunk blocks + one NextN block).
//
// Only public headers are used (llama.h, gguf.h, ggml.h, ggml-backend.h,
// ggml-alloc.h), so the file also builds against a pristine stock tree where
// src/ is not on the include path.
//
// The file is a normal GGUF, so it is loaded through the stock llama_model
// loader: the router projection a real decode runs is [512, n_tokens] exactly
// as in production, the MoE weights are real (random) F32 tensors, and nothing
// is stubbed. The only thing that differs from the production checkpoint is
// the hidden dimension.
//
// All data is generated from a fixed seed, so the file is reproducible.
//
// Usage:
//   llama-tp5-router-fixture -o m.gguf [--seed 1234] [--write-attn polys|none]
//
//   --write-attn none   (default) writes the recurrent blocks' attn_qkv and
//                       attn_gate (the gated delta net's qkv/z projections) as
//                       zeros. The trunk decode graph does not execute them for
//                       this layer count (2 trunk layers, both linear layers, so
//                       the MTP block is not part of the trunk graph); they are
//                       bound only so the loader finds every tensor. The NextN
//                       block's full-attention tensors are always written with
//                       real values.
//   --write-attn polys  writes real values into those recurrent tensors too.

namespace {

constexpr uint32_t N_VOCAB       = 128;
constexpr uint32_t N_EMBD        = 8;   // hidden width, shrunk from 2560
constexpr uint32_t N_HEAD        = 1;
constexpr uint32_t N_HEAD_KV     = 1;
constexpr uint32_t N_LAYER       = 2;   // trunk blocks, both recurrent (see below)
constexpr uint32_t N_LAYER_NEXTN = 1;   // trailing NextN/MTP block
constexpr uint32_t N_LAYER_ALL   = N_LAYER + N_LAYER_NEXTN;
constexpr uint32_t HC            = 4;   // production hyper-connection count
constexpr uint32_t HC_LR         = 2;   // production value is 320; any >0 works
constexpr uint32_t N_EXPERT      = 512; // production values: the real gate
constexpr uint32_t N_EXPERT_USED = 10;
constexpr uint32_t N_FF_EXP      = 2;
constexpr uint32_t N_FF_SHEXP    = 2;
constexpr uint32_t IDX_DIM       = 2;   // indexer head size (indexer.key_length)
constexpr uint32_t IDX_N_HEAD    = 1;   // indexer head count
constexpr uint32_t SSM_CONV      = 2;
constexpr uint32_t SSM_STATE     = 2;
constexpr uint32_t SSM_INNER     = 4;   // must equal 2 * ssm_dt_rank
constexpr uint32_t SSM_RANK      = 2;
// Every trunk block is a gated-delta-net layer. With full_attention_interval
// absent, hparams marks layer il recurrent unless (il + 1) % interval == 0, so
// an interval of 8 leaves both of the two trunk layers recurrent and the whole
// full-attention/indexer path out of the graph.
constexpr uint32_t FULL_ATTN_INTERVAL = 8;

constexpr uint32_t HC_DIM = HC * N_EMBD;

struct tensor_spec {
    std::string name;
    std::vector<int64_t> ne;
};

struct fixture_spec {
    std::vector<tensor_spec> tensors;
};

void add(fixture_spec & fs, const std::string & name, std::vector<int64_t> ne) {
    fs.tensors.push_back({ name, std::move(ne) });
}

fixture_spec build_spec() {
    fixture_spec fs;

    add(fs, "token_embd.weight", { N_EMBD, N_VOCAB });
    add(fs, "output_hc_norm.weight", { N_EMBD, HC });
    add(fs, "output_hc_down.weight", { HC_DIM, HC_LR });
    add(fs, "output_hc_up.weight", { HC_LR, HC_DIM });
    add(fs, "output.weight", { N_EMBD, N_VOCAB });

    // src/models/qwen4exp.cpp:319-325, verbatim:
    //   head_k_dim = head_v_dim = ssm_d_state, n_k_heads = ssm_n_group,
    //   n_v_heads = ssm_dt_rank, key_dim = head_k_dim*n_k_heads,
    //   value_dim = head_v_dim*n_v_heads, conv_dim = 2*key_dim + value_dim
    const int64_t head_k_dim = SSM_STATE;
    const int64_t head_v_dim = SSM_STATE;
    const int64_t n_k_heads  = N_HEAD_KV;
    const int64_t n_v_heads  = SSM_RANK;   // ssm_dt_rank, the time-step rank
    const int64_t key_dim    = head_k_dim * n_k_heads;
    const int64_t value_dim  = head_v_dim * n_v_heads;
    const int64_t conv_dim   = 2 * key_dim + value_dim;

    // full-attention head width used by the NextN block (hparams.n_embd_head_k):
    // n_embd / n_head, one head per kv head here
    const int64_t n_embd_head_k = N_EMBD / N_HEAD;

    for (uint32_t il = 0; il < N_LAYER_ALL; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const bool nextn = il >= N_LAYER;

        // Token mixer tensors, chosen by is_recr(il) exactly as
        // qwen4exp.cpp:337-360 does. Trunk blocks are recurrent and take the
        // gated-delta-net branch (attn_qkv, attn_gate, ssm_*); the NextN block
        // is past the trunk, so is_recr/is_ple are false there and it takes
        // the full-attention branch (qkv / attn_output / q/k norms /
        // indexer.*), as the comment at :312-313 says.
        if (!nextn) {
            const std::string names[] = {
                p + "attn_qkv.weight",   p + "attn_gate.weight",
                p + "ssm_conv1d.weight", p + "ssm_dt.bias",
                p + "ssm_a",             p + "ssm_beta.weight",
                p + "ssm_alpha.weight",  p + "ssm_norm.weight",
                p + "ssm_out.weight",
            };
            const std::vector<int64_t> shapes[] = {
                { N_EMBD, 2 * key_dim + value_dim },   // attn_qkv
                { N_EMBD, value_dim },                 // attn_gate
                { SSM_CONV, conv_dim },                // ssm_conv1d
                { n_v_heads },                         // ssm_dt
                { n_v_heads },                         // ssm_a
                { N_EMBD, n_v_heads },                 // ssm_beta
                { N_EMBD, n_v_heads },                 // ssm_alpha
                { head_v_dim },                        // ssm_norm
                { value_dim, N_EMBD },                 // ssm_out
            };
            for (int i = 0; i < 9; ++i) {
                add(fs, names[i], shapes[i]);
            }
        } else {
            // full attention (create_tensor_qkv with q+k+v, so no split q/k/v):
            //   attn_qkv [n_embd, 2*n_embd_head_k*n_head + 2*n_embd_k_gqa]
            //            (wq holds [q|gate] interleaved per head, then k, then v)
            //   attn_output [n_embd_head_k*n_head, n_embd]
            //   attn_q_norm / attn_k_norm [n_embd_head_k]
            //   indexer.q_proj [n_embd, n_idx_head*idx_dim], indexer.k_proj
            //   [n_embd, idx_dim], indexer.q_norm / indexer.k_norm [idx_dim]
            const int64_t q_dim = n_embd_head_k * N_HEAD * 2;   // q + gate
            const int64_t k_dim = n_embd_head_k * N_HEAD;       // n_embd_k_gqa, 1 kv head
            const int64_t v_dim = n_embd_head_k * N_HEAD;       // n_embd_v_gqa
            //   attn_output [n_embd_head_k*n_head, n_embd], the q half of q_dim
            add(fs, p + "attn_qkv.weight",    { N_EMBD, q_dim + k_dim + v_dim });
            add(fs, p + "attn_output.weight", { n_embd_head_k * N_HEAD, N_EMBD });
            add(fs, p + "attn_q_norm.weight", { n_embd_head_k });
            add(fs, p + "attn_k_norm.weight", { n_embd_head_k });
            add(fs, p + "indexer.q_proj.weight", { N_EMBD, IDX_N_HEAD * IDX_DIM });
            add(fs, p + "indexer.k_proj.weight", { N_EMBD, IDX_DIM });
            add(fs, p + "indexer.q_norm.weight", { IDX_DIM });
            add(fs, p + "indexer.k_norm.weight", { IDX_DIM });
        }

        add(fs, p + "hc_attn_norm.weight",   { N_EMBD, HC });
        add(fs, p + "hc_attn_down.weight",   { HC_DIM, HC_LR });
        add(fs, p + "hc_attn_up.weight",     { HC_LR, HC_DIM });
        add(fs, p + "hc_attn_inject.weight", { HC_DIM, HC });
        add(fs, p + "hc_ffn_norm.weight",    { N_EMBD, HC });
        add(fs, p + "hc_ffn_down.weight",    { HC_DIM, HC_LR });
        add(fs, p + "hc_ffn_up.weight",      { HC_LR, HC_DIM });
        add(fs, p + "hc_ffn_inject.weight",  { HC_DIM, HC });

        // MoE: the production router and expert layout, shrunk expert width
        add(fs, p + "ffn_gate_inp.weight",      { N_EMBD, N_EXPERT });
        add(fs, p + "ffn_gate_exps.weight",     { N_EMBD, N_FF_EXP, N_EXPERT });
        add(fs, p + "ffn_up_exps.weight",       { N_EMBD, N_FF_EXP, N_EXPERT });
        add(fs, p + "ffn_down_exps.weight",     { N_FF_EXP, N_EMBD, N_EXPERT });
        add(fs, p + "ffn_gate_inp_shexp.weight",{ N_EMBD });
        add(fs, p + "ffn_gate_shexp.weight",    { N_EMBD, N_FF_SHEXP });
        add(fs, p + "ffn_up_shexp.weight",      { N_EMBD, N_FF_SHEXP });
        add(fs, p + "ffn_down_shexp.weight",    { N_FF_SHEXP, N_EMBD });

        if (nextn) {
            add(fs, p + "nextn.enorm.weight",   { N_EMBD });
            add(fs, p + "nextn.hnorm.weight",   { HC_DIM });
            add(fs, p + "nextn.eh_proj.weight", { 2 * N_EMBD, N_EMBD });
        }
    }
    return fs;
}

void set_kv_common(gguf_context * ctx) {
    gguf_set_val_str(ctx, "general.architecture", "qwen4exp");
    gguf_set_val_str(ctx, "general.name", "qwen4exp-router-scratch-fixture");
    gguf_set_val_str(ctx, "tokenizer.ggml.model", "no_vocab");
    gguf_set_val_u32(ctx, "qwen4exp.vocab_size", N_VOCAB); // no_vocab reads the token count here
    gguf_set_val_u32(ctx, "qwen4exp.context_length", 64);
    gguf_set_val_u32(ctx, "qwen4exp.embedding_length", N_EMBD);
    gguf_set_val_u32(ctx, "qwen4exp.block_count", N_LAYER_ALL);
    gguf_set_val_u32(ctx, "qwen4exp.feed_forward_length", N_FF_SHEXP);
    gguf_set_val_u32(ctx, "qwen4exp.attention.head_count", N_HEAD);
    gguf_set_val_u32(ctx, "qwen4exp.attention.head_count_kv", N_HEAD_KV);
    // full-attention head widths (n_head_kv and n_head are both 1 here, so the
    // kv width equals the full width); qwen4exp declares no SWA layers
    gguf_set_val_u32(ctx, "qwen4exp.attention.key_length",   N_EMBD / N_HEAD);
    gguf_set_val_u32(ctx, "qwen4exp.attention.value_length", N_EMBD / N_HEAD);
    gguf_set_val_f32(ctx, "qwen4exp.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(ctx, "qwen4exp.nextn_predict_layers", N_LAYER_NEXTN);

    gguf_set_val_u32(ctx, "qwen4exp.full_attention_interval", FULL_ATTN_INTERVAL);
    // per-layer linear-attention flags, one entry per block including NextN.
    // qwen4exp.cpp:235-243: when this key is present it is used as-is, so the
    // NextN block must be marked 0 (full attention) -- the interval fallback
    // would have said 0 for it too, since i >= n_layer(). Writing all-ones made
    // blk.2 recurrent and the loader asked for the GDN attn_qkv [8, 8].
    std::vector<uint32_t> recurrent(N_LAYER_ALL, 0);
    for (uint32_t il = 0; il < N_LAYER; ++il) {
        recurrent[il] = 1;
    }
    gguf_set_arr_data(ctx, "qwen4exp.attention.recurrent_layers", GGUF_TYPE_UINT32, recurrent.data(), recurrent.size());
    // no attention.indexer.types: only glm-dsa reads that key (optional there) and
    // qwen4exp never does, so the array would be dead metadata of an unknown length

    // the loader reads exactly 4 sections (mrope [t, h, w, e]); the sum must be
    // the rotary width, which here is the whole head because there is one head
    std::vector<uint32_t> rope_sect = { 2, 2, 2, 2 };
    gguf_set_arr_data(ctx, "qwen4exp.rope.dimension_sections", GGUF_TYPE_UINT32, rope_sect.data(), rope_sect.size());

    gguf_set_val_u32(ctx, "qwen4exp.ssm.conv_kernel", SSM_CONV);
    gguf_set_val_u32(ctx, "qwen4exp.ssm.inner_size", SSM_INNER);
    gguf_set_val_u32(ctx, "qwen4exp.ssm.state_size", SSM_STATE);
    gguf_set_val_u32(ctx, "qwen4exp.ssm.time_step_rank", SSM_RANK);
    gguf_set_val_u32(ctx, "qwen4exp.ssm.group_count", N_HEAD_KV);

    gguf_set_val_u32(ctx, "qwen4exp.hyper_connection.count", HC);
    gguf_set_val_u32(ctx, "qwen4exp.hyper_connection.low_rank", HC_LR);

    gguf_set_val_u32(ctx, "qwen4exp.attention.indexer.head_count", IDX_N_HEAD);
    gguf_set_val_u32(ctx, "qwen4exp.attention.indexer.key_length", IDX_DIM);
    gguf_set_val_u32(ctx, "qwen4exp.attention.indexer.top_k", 1);

    std::vector<uint32_t> compress(N_LAYER_ALL, 0);
    gguf_set_arr_data(ctx, "qwen4exp.attention.compress_ratios", GGUF_TYPE_UINT32, compress.data(), compress.size());

    gguf_set_val_u32(ctx, "qwen4exp.expert_feed_forward_length", N_FF_EXP);
    gguf_set_val_u32(ctx, "qwen4exp.expert_shared_feed_forward_length", N_FF_SHEXP);
    gguf_set_val_u32(ctx, "qwen4exp.expert_count", N_EXPERT);
    gguf_set_val_u32(ctx, "qwen4exp.expert_used_count", N_EXPERT_USED);
}

} // namespace

int main(int argc, char ** argv) {
    std::string out_path;
    uint64_t seed = 20261031ull;
    bool     write_attn = false;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        const bool has_next = i + 1 < argc;
        if (arg == "-o" && has_next) {
            out_path = argv[++i];
        } else if (arg == "--seed" && has_next) {
            seed = (uint64_t) strtoull(argv[++i], nullptr, 10);
        } else if (arg == "--write-attn" && has_next) {
            const std::string v = argv[++i];
            if (v == "none") {
                write_attn = false;
            } else if (v == "polys") {
                write_attn = true;
            } else {
                fprintf(stderr, "%s: bad --write-attn '%s' (none|polys)\n", argv[0], v.c_str());
                return 1;
            }
        } else {
            fprintf(stderr, "usage: %s -o model.gguf [--seed N] [--write-attn none|polys]\n", argv[0]);
            return 1;
        }
    }
    if (out_path.empty()) {
        fprintf(stderr, "usage: %s -o model.gguf [--seed N] [--write-attn none|polys]\n", argv[0]);
        return 1;
    }

    const fixture_spec fs = build_spec();
    uint64_t total_bytes = 0;
    for (const tensor_spec & t : fs.tensors) {
        int64_t n = 1;
        for (int64_t v : t.ne) {
            n *= v;
        }
        total_bytes += (uint64_t) n * sizeof(float);
    }

    // the context owns the tensor memory, so every tensor's bytes stay valid
    // from creation until the last fwrite
    ggml_init_params ip = {};
    ip.mem_size   = ggml_tensor_overhead() * (fs.tensors.size() + 16) + total_bytes + 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = false;
    ggml_context * ctx_data = ggml_init(ip);
    if (!ctx_data) {
        fprintf(stderr, "%s: ggml_init failed\n", argv[0]);
        return 1;
    }

    gguf_context * ctx = gguf_init_empty();
    set_kv_common(ctx);

    std::vector<ggml_tensor *> created;
    created.reserve(fs.tensors.size());

    // "none": the attention block tensors keep their production shape (n_embd
    // rows) but are written as zeros and never contribute, because every trunk
    // block in this model is a linear layer and the attention path is not built.
    for (const tensor_spec & t : fs.tensors) {
        const std::vector<int64_t> & ne = t.ne;
        ggml_tensor * cur = ggml_new_tensor(ctx_data, GGML_TYPE_F32, (int) ne.size(), ne.data());
        if (!cur) {
            fprintf(stderr, "%s: failed to create tensor '%s'\n", argv[0], t.name.c_str());
            return 1;
        }
        ggml_set_name(cur, t.name.c_str());
        gguf_add_tensor(ctx, cur);
        created.push_back(cur);
    }

    // random-but-reproducible data for every tensor; written straight into the
    // tensor the context owns, which gguf keeps serving to the writer.
    // gguf_add_tensor copies the tensor descriptor, so data and ->data are
    // filled on the context tensor, not on the gguf copy.
    uint64_t state = seed ? seed : 0x9e3779b97f4a7c15ull;
    for (size_t i = 0; i < created.size(); ++i) {
        ggml_tensor * cur = created[i];
        const int64_t ne = ggml_nelements(cur);
        float * buf = (float *) cur->data;
        if (buf == nullptr) {
            fprintf(stderr, "%s: tensor '%s' has no storage\n", argv[0], cur->name);
            return 1;
        }

        const std::string name(cur->name);
        // only the recurrent blocks' qkv path is inert; never zero the NextN
        // block's full-attention tensors (blk.2), which the loader binds for a
        // full-attention layer
        const bool nextn_block = name.rfind("blk." + std::to_string(N_LAYER) + ".", 0) == 0;
        bool zero = false;
        if (!write_attn && !nextn_block &&
                (name.find(".attn_qkv.weight") != std::string::npos ||
                 name.find(".attn_gate.weight") != std::string::npos)) {
            zero = true;
        }
        bool gamma = false;
        if (name == "output_hc_norm.weight") {
            gamma = true;
        } else if (name.find(".hc_attn_norm.weight") != std::string::npos ||
                   name.find(".hc_ffn_norm.weight")  != std::string::npos ||
                   name.find(".ssm_norm.weight") != std::string::npos ||
                   name.find(".nextn.enorm.weight")  != std::string::npos ||
                   name.find(".nextn.hnorm.weight")  != std::string::npos) {
            gamma = true;
        }

        for (int64_t j = 0; j < ne; ++j) {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            const uint64_t r = state * 0x2545f4914f6cdd1dull;
            if (zero) {
                buf[j] = 0.0f;
            } else if (gamma) {
                // 1 + small offset, the convention the converter folds to
                buf[j] = 1.0f + ((float) ((r >> 40) & 0xffffff) / (float) 0xffffff - 0.5f) * 0.1f;
            } else {
                // mean ~0, std ~0.02: router logits land in a non-degenerate range
                const float u = (float) ((r >> 40) & 0xffffff) / (float) 0xffffff;
                buf[j] = (u - 0.5f) * 0.08f;
            }
        }
        // the gated delta net needs an invertible decay and finite gates
        if (name.find(".ssm_a") != std::string::npos) {
            for (int64_t j = 0; j < ne; ++j) {
                buf[j] = 1.0f; // exp(-1) < 1: a stable per-channel decay
            }
        }
        if (name.find(".ssm_beta.weight") != std::string::npos ||
            name.find(".ssm_alpha.weight") != std::string::npos) {
            for (int64_t j = 0; j < ne; ++j) {
                buf[j] = buf[j] * 0.5f + 0.5f; // sigmoid-friendly, finite
            }
        }

        // gguf keeps a pointer to the storage, so every tensor must own its
        // bytes for the whole write; refuse a truncated model instead
        if (ggml_nbytes(cur) != (size_t) ne * sizeof(float)) {
            fprintf(stderr, "%s: tensor '%s' is not a dense F32 block\n", argv[0], cur->name);
            return 1;
        }
        gguf_set_tensor_data(ctx, cur->name, cur->data);
    }

    // meta data + tensor data -> file
    {
        // gguf_get_data_offset() is only meaningful for a context read from a
        // file; a freshly built context reports 0, so the tensor data start is
        // derived from the meta block size the way the writer lays it out:
        // meta block, padded up to the file alignment
        const size_t size_meta = gguf_get_meta_size(ctx);
        const size_t alignment = gguf_get_alignment(ctx);
        GGML_ASSERT(alignment > 0 && (alignment & (alignment - 1)) == 0 && alignment <= 64);
        const size_t data_off = GGML_PAD(size_meta, alignment);
        std::vector<uint8_t> meta(size_meta);
        gguf_get_meta_data(ctx, meta.data());

        // gguf writes every tensor at the aligned offset gguf_add_tensor
        // recorded; check that the stream below starts each one exactly there,
        // so the file cannot come out shifted or truncated. Nothing is open yet,
        // so a failure here only releases the two gguf/ggml contexts.
        size_t expect_off = data_off;
        size_t total_data = 0;
        for (const ggml_tensor * cur : created) {
            const int64_t id = gguf_find_tensor(ctx, cur->name);
            if (id < 0 || data_off + gguf_get_tensor_offset(ctx, id) != expect_off ||
                    gguf_get_tensor_size(ctx, id) != ggml_nbytes(cur)) {
                fprintf(stderr, "%s: layout mismatch for tensor '%s'\n", argv[0], cur->name);
                gguf_free(ctx);
                ggml_free(ctx_data);
                return 1;
            }
            expect_off += GGML_PAD(ggml_nbytes(cur), alignment);
            total_data += ggml_nbytes(cur);
        }
        if (total_data == 0 || total_data != total_bytes) {
            fprintf(stderr, "%s: data size mismatch (%zu written vs %llu planned)\n",
                    argv[0], total_data, (unsigned long long) total_bytes);
            gguf_free(ctx);
            ggml_free(ctx_data);
            return 1;
        }

        FILE * f = fopen(out_path.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "%s: cannot open '%s': %s\n", argv[0], out_path.c_str(), strerror(errno));
            gguf_free(ctx);
            ggml_free(ctx_data);
            return 1;
        }
        static const uint8_t zeros[64] = {0};
        bool ok = fwrite(meta.data(), 1, meta.size(), f) == meta.size();
        // gguf pads the end of the meta block up to the tensor data offset
        const size_t meta_pad = data_off - size_meta;
        ok = ok && (meta_pad == 0 || fwrite(zeros, 1, meta_pad, f) == meta_pad);
        for (size_t i = 0; ok && i < created.size(); ++i) {
            const ggml_tensor * cur = created[i];
            if (i > 0) {
                // gguf aligns every tensor start
                const size_t rem = ggml_nbytes(created[i - 1]) % alignment;
                const size_t pad = rem == 0 ? 0 : alignment - rem;
                ok = pad == 0 || fwrite(zeros, 1, pad, f) == pad;
            }
            if (cur->data == nullptr || ggml_nbytes(cur) == 0) {
                fprintf(stderr, "%s: tensor '%s' would be written empty\n", argv[0], cur->name);
                ok = false;
                break;
            }
            ok = ok && fwrite(cur->data, 1, ggml_nbytes(cur), f) == ggml_nbytes(cur);
        }
        ok = ok && fclose(f) == 0;
        if (!ok) {
            fprintf(stderr, "%s: write of '%s' failed\n", argv[0], out_path.c_str());
            gguf_free(ctx);
            ggml_free(ctx_data);
            return 1;
        }
    }
    gguf_free(ctx);
    ggml_free(ctx_data);

    // ---- self-check: the file must load through the real llama model loader
    //      and expose the production router width ----
    int rc = 1;

#if defined(TP5_FIXTURE_CONTRACT)
    // in-tree builds: run the real Qwen4Exp loader's contract over the file
    // first, so a wrong key, array length or tensor shape is named here instead
    // of surfacing later as a loader exception in the verifier
    {
        const fixture_contract_diff cdiff = fixture_check_contract(out_path);
        printf("FIXTURE contract problems=%zu meta_checked=%d tensors_checked=%d\n",
               cdiff.problems.size(), cdiff.n_meta_checked, cdiff.n_tensors_checked);
        for (const std::string & p : cdiff.problems) {
            printf("FIXTURE contract_problem %s\n", p.c_str());
        }
        if (!cdiff.problems.empty()) {
            return 1;
        }
    }
#endif

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(out_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "%s: self-check: the fixture does not load\n", argv[0]);
        llama_backend_free();
        return 1;
    }

    char meta_buf[32] = {0};
    int n_expert_used = -1;
    int n_expert      = -1;
    if (llama_model_meta_val_str(model, "qwen4exp.expert_used_count", meta_buf, sizeof(meta_buf)) >= 0) {
        n_expert_used = atoi(meta_buf);
    }
    if (llama_model_meta_val_str(model, "qwen4exp.expert_count", meta_buf, sizeof(meta_buf)) >= 0) {
        n_expert = atoi(meta_buf);
    }

    // one Qwen4Exp layer's MoE with the router nodes build_layer_ffn emits:
    // gate_inp [n_embd, 512] x rows -> softmax -> top-k -> routed experts.
    // The weights are the fixture's own bytes read back from the file and the
    // graph runs through the production gallocr + CPU backend.
    const int64_t n_used = 10;

    std::vector<uint8_t> w_gate_inp((size_t) N_EMBD * N_EXPERT * sizeof(float));
    std::vector<uint8_t> w_gate_exps((size_t) N_EMBD * N_FF_EXP * N_EXPERT * sizeof(float));
    int64_t data_off = 0;
    {
        struct gguf_init_params ip = {};
        ip.no_alloc = true;
        ip.ctx      = nullptr;
        gguf_context * raw = gguf_init_from_file(out_path.c_str(), ip);
        FILE * f = raw ? fopen(out_path.c_str(), "rb") : nullptr;
        if (!raw || !f) {
            fprintf(stderr, "%s: self-check: cannot re-read the fixture\n", argv[0]);
            if (f) fclose(f);
            if (raw) gguf_free(raw);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        data_off = (int64_t) gguf_get_data_offset(raw);
        bool ok = true;
        for (int which = 0; which < 2 && ok; ++which) {
            const char * name = which == 0 ? "blk.0.ffn_gate_inp.weight" : "blk.0.ffn_gate_exps.weight";
            std::vector<uint8_t> & dst = which == 0 ? w_gate_inp : w_gate_exps;
            const int64_t id = gguf_find_tensor(raw, name);
            if (id < 0) {
                fprintf(stderr, "%s: self-check: tensor '%s' missing from the fixture\n", argv[0], name);
                ok = false;
                break;
            }
            if (fseek(f, (long) (data_off + (int64_t) gguf_get_tensor_offset(raw, id)), SEEK_SET) != 0 ||
                    fread(dst.data(), 1, dst.size(), f) != dst.size()) {
                fprintf(stderr, "%s: self-check: cannot read tensor '%s'\n", argv[0], name);
                ok = false;
            }
        }
        fclose(f);
        gguf_free(raw);
        if (!ok) {
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
    }

    // public registry initialisation; examples do not include ggml-cpu.h
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "%s: self-check: no CPU backend\n", argv[0]);
        ggml_free(ctx_data);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    ggml_init_params gp = {};
    gp.mem_size   = 64ull * 1024 * 1024;
    gp.mem_buffer = nullptr;
    gp.no_alloc   = true;
    ggml_context * ctx_g = ggml_init(gp);

    // production MoE input layout: build_moe_ffn reshapes the layer input to
    // [n_embd, 1, n_tokens] (llama-graph.cpp:2203) and hands it to MUL_MAT_ID,
    // whose ids are [n_expert_used, n_tokens]; id b was 2-D here, so
    // b->ne[2] == 1 could never match ids->ne[1] == 3
    ggml_tensor * x      = ggml_new_tensor_3d(ctx_g, GGML_TYPE_F32, N_EMBD, 1, 3);
    ggml_tensor * w_gate = ggml_new_tensor_2d(ctx_g, GGML_TYPE_F32, N_EMBD, N_EXPERT);
    ggml_tensor * w_exps = ggml_new_tensor_3d(ctx_g, GGML_TYPE_F32, N_EMBD, N_FF_EXP, N_EXPERT);
    ggml_set_name(x, "moe_input");
    ggml_set_input(x);
    ggml_set_input(w_gate);
    ggml_set_input(w_exps);

    // router projection over the same input: b is 3-D, so build_lora_mm reshapes
    // it the way the model does before the multiplication
    ggml_tensor * x_mm = ggml_reshape_2d(ctx_g, x, N_EMBD, 3);
    ggml_tensor * router_logits = ggml_mul_mat(ctx_g, w_gate, x_mm);    // [512, 3]
    ggml_set_name(router_logits, "router_logits");
    ggml_tensor * probs   = ggml_soft_max(ctx_g, router_logits);        // [512, 3]
    ggml_tensor * weights = ggml_top_k(ctx_g, probs, n_used);           // [10, 3]
    ggml_set_name(weights, "ffn_moe_weights");
    ggml_tensor * sorted  = ggml_argsort(ctx_g, probs, GGML_SORT_ORDER_DESC);
    // the selected expert ids of each row, as the fused route produces them
    ggml_tensor * ids     = ggml_view_2d(ctx_g, sorted, n_used, 3,
            ggml_row_size(sorted->type, N_EXPERT), 0);
    ggml_set_name(ids, "ffn_moe_argsort_ids");
    ggml_tensor * out     = ggml_mul_mat_id(ctx_g, w_exps, x, ids);      // [n_ff, 10, 3]

    ggml_cgraph * gf = ggml_new_graph_custom(ctx_g, 256, false);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    const bool reserved  = ggml_gallocr_reserve(galloc, gf);
    std::vector<ggml_tensor *> leaves = { x, w_gate, w_exps };
    size_t leaves_bytes = 0;
    for (const ggml_tensor * leaf : leaves) {
        leaves_bytes += GGML_PAD(ggml_nbytes(leaf), 64);
    }
    ggml_backend_buffer_t buf = reserved
            ? ggml_backend_alloc_buffer(backend, ggml_gallocr_get_buffer_size(galloc, 0) + leaves_bytes)
            : nullptr;
    bool allocated = false;
    if (buf != nullptr) {
        ggml_tallocr talloc = ggml_tallocr_new(buf);
        for (ggml_tensor * leaf : leaves) {
            if (ggml_tallocr_alloc(&talloc, leaf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "%s: self-check: cannot allocate input '%s'\n", argv[0], leaf->name);
            }
        }
        allocated = ggml_gallocr_alloc_graph(galloc, gf);
    }

    bool computed = false;
    double out_sum2 = 0.0;
    if (allocated) {
        // x is [n_embd, 1, 3]: one embd vector per row, same byte count
        std::vector<float> hx((size_t) ggml_nelements(x));
        for (size_t i = 0; i < hx.size(); ++i) {
            hx[i] = 0.1f * (float) ((i * 37) % 11) - 0.5f;
        }
        ggml_backend_tensor_set(x, hx.data(), 0, hx.size() * sizeof(float));
        ggml_backend_tensor_set(w_gate, w_gate_inp.data(), 0, w_gate_inp.size());
        ggml_backend_tensor_set(w_exps, w_gate_exps.data(), 0, w_gate_exps.size());
        computed = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        if (computed) {
            std::vector<float> hout((size_t) ggml_nelements(out));
            ggml_backend_tensor_get(out, hout.data(), 0, hout.size() * sizeof(float));
            for (float v : hout) {
                out_sum2 += (double) v * (double) v;
            }
        }
    }

    // the router shape the retention predicate matches: logits [512, rows],
    // softmax over them, and the top-k weights of the same shape
    bool has_router = false;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (ggml_graph_node(gf, i) == router_logits) {
            has_router = true;
        }
    }
    has_router = has_router && weights->ne[0] == n_used && weights->ne[1] == 3 &&
            probs->ne[0] == (int64_t) N_EXPERT && probs->ne[1] == 3;

    printf("FIXTURE selfcheck model_load=%d router_shape=%d reserve=%d alloc=%d compute=%d out_sum2=%.6e data_offset=%lld\n",
           model != nullptr, has_router ? 1 : 0, reserved ? 1 : 0, allocated ? 1 : 0,
           computed ? 1 : 0, out_sum2, (long long) data_off);

    if (buf != nullptr) {
        ggml_backend_buffer_free(buf);
    }
    ggml_gallocr_free(galloc);
    ggml_free(ctx_g);
    ggml_backend_free(backend);

    printf("FIXTURE path=%s bytes=%zu seed=%llu n_embd=%u n_layer=%u+%u hc=%u hc_lr=%u "
           "n_expert=%d n_expert_used=%d n_ff_exp=%u n_vocab=%u write_attn=%s\n",
           out_path.c_str(), total_bytes, (unsigned long long) seed,
           N_EMBD, N_LAYER, N_LAYER_NEXTN, HC, HC_LR,
           n_expert, n_expert_used, N_FF_EXP, N_VOCAB, write_attn ? "polys" : "none");
    rc = (has_router && n_expert == (int) N_EXPERT && n_expert_used == (int) N_EXPERT_USED &&
          reserved && allocated && computed) ? 0 : 1;

    llama_model_free(model);
    llama_backend_free();
    return rc;
}
