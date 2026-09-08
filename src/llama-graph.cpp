#include "llama-graph.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-flashprefill-pack.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-msa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

// FlashPrefill V2 wire schema + reference math (WireReference owner;
// GraphIntegration consumes the I32 metadata/plan pack helpers only).
#include "ggml-flashprefill.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <tuple>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

// dedup helpers

static ggml_tensor * build_attn_inp_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    // flash attention requires an f16 mask
    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool can_reuse_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

// impl

void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    }

    if (ubatch->embd) {
        GGML_ASSERT(n_embd == embd->ne[0]);

        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   &&   embd->ne[1] == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_embd_h::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens = ubatch->n_tokens;

    if (ubatch->token) {
        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    } else {
        // note: mtmd embedding input goes through here
        GGML_ASSERT(ubatch->embd);
        GGML_ASSERT(n_embd == embd->ne[0]);

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(h));
    }

    // TODO: extend llama_ubatch to differentiate between token embeddings and hidden states
    //       for now, we assume that the hidden state is always provided as an embedding
    //       ref: https://github.com/ggml-org/llama.cpp/pull/23643
    if (ubatch->embd) {
        GGML_ASSERT(n_embd == h->ne[0]);

        ggml_backend_tensor_set(h, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(h));
    }
}

bool llm_graph_input_embd_h::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   && embd->ne[1]   == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (h      && h->ne[1]      == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_pos::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && pos) {
        const int64_t n_tokens = ubatch->n_tokens;

        if (ubatch->token && n_pos_per_embd == 4) {
            // in case we're using M-RoPE with text tokens, convert the 1D positions to 4D
            // the 3 first dims are the same, and 4th dim is all 0
            std::vector<llama_pos> pos_data(n_tokens*n_pos_per_embd);
            // copy the first dimension
            for (int i = 0; i < n_tokens; ++i) {
                pos_data[               i] = ubatch->pos[i];
                pos_data[    n_tokens + i] = ubatch->pos[i];
                pos_data[2 * n_tokens + i] = ubatch->pos[i];
                pos_data[3 * n_tokens + i] = 0; // 4th dim is 0
            }
            ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size()*ggml_element_size(pos));
        } else {
            ggml_backend_tensor_set(pos, ubatch->pos, 0, n_tokens*n_pos_per_embd*ggml_element_size(pos));
        }
    }
}

bool llm_graph_input_pos::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= pos->ne[0] == params.ubatch.n_tokens*n_pos_per_embd;

    return res;
}

void llm_graph_input_attn_temp::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && attn_scale) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(f_attn_temp_scale != 0.0f);
        GGML_ASSERT(n_attn_temp_floor_scale != 0);

        std::vector<float> attn_scale_data(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const float pos = ubatch->pos[i];
            attn_scale_data[i] = std::log(
                std::floor((pos + f_attn_temp_offset) / n_attn_temp_floor_scale) + 1.0
            ) * f_attn_temp_scale + 1.0;
        }

        ggml_backend_tensor_set(attn_scale, attn_scale_data.data(), 0, n_tokens*ggml_element_size(attn_scale));
    }
}

void llm_graph_input_pos_bucket::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(pos_bucket->buffer));
        GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

        int32_t * data = (int32_t *) pos_bucket->data;

        for (int j = 0; j < n_tokens; ++j) {
            for (int i = 0; i < n_tokens; ++i) {
                data[j*n_tokens + i] = llama_relative_position_bucket(ubatch->pos[i], ubatch->pos[j], hparams.n_rel_attn_bkts, true);
            }
        }
    }
}

void llm_graph_input_pos_bucket_kv::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        mctx->set_input_pos_bucket(pos_bucket, ubatch);
    }
}

void llm_graph_input_out_ids::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(out_ids);

    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(out_ids->buffer));
    int32_t * data = (int32_t *) out_ids->data;

    if (n_outputs == n_tokens) {
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = i;
        }

        return;
    }

    GGML_ASSERT(ubatch->output);

    int n_outputs = 0;

    for (int i = 0; i < n_tokens; ++i) {
        if (ubatch->output[i]) {
            data[n_outputs++] = i;
        }
    }
}

bool llm_graph_input_out_ids::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= n_outputs == params.n_outputs;

    return res;
}

void llm_graph_input_mean::set_input(const llama_ubatch * ubatch) {
    if (cparams.embeddings   &&
       (cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK )) {

        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

        GGML_ASSERT(mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(mean->buffer));

        float * data = (float *) mean->data;
        memset(mean->data, 0, n_tokens*n_seqs_unq*ggml_element_size(mean));

        std::vector<uint64_t> sums(n_seqs_unq, 0);
        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                sums[seq_idx] += ubatch->n_seq_tokens;
            }
        }

        std::vector<float> div(n_seqs_unq, 0.0f);
        for (int s = 0; s < n_seqs_unq; ++s) {
            const uint64_t sum = sums[s];
            if (sum > 0) {
                div[s] = 1.0f/float(sum);
            }
        }

        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                for (int j = 0; j < n_seq_tokens; ++j) {
                    data[seq_idx*n_tokens + i + j] = div[seq_idx];
                }
            }
        }
    }
}

void llm_graph_input_cls::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens     = ubatch->n_tokens;
    const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

    if (cparams.embeddings && (
        cparams.pooling_type == LLAMA_POOLING_TYPE_CLS  ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_LAST
    )) {
        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_seqs_unq*ggml_element_size(cls));

        std::vector<int> target_pos(n_seqs_unq, -1);
        std::vector<int> target_row(n_seqs_unq, -1);

        const bool last = (
             cparams.pooling_type == LLAMA_POOLING_TYPE_LAST ||
            (cparams.pooling_type == LLAMA_POOLING_TYPE_RANK && (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL)) // qwen3 reranking & embedding models use last token
        );

        for (int i = 0; i < n_tokens; ++i) {
            const llama_pos pos = ubatch->pos[i];

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                if (
                    (target_pos[seq_idx] == -1) ||
                    ( last && pos >= target_pos[seq_idx]) ||
                    (!last && pos <  target_pos[seq_idx])
                ) {
                    target_pos[seq_idx] = pos;
                    target_row[seq_idx] = i;
                }
            }
        }

        for (int s = 0; s < n_seqs_unq; ++s) {
            if (target_row[s] >= 0) {
                data[s] = target_row[s];
            }
        }
    }
}

void llm_graph_input_rs::set_input(const llama_ubatch * ubatch) {
    set_input_recurrent(mctx, ubatch);
}

void llm_graph_input_rs::set_input_recurrent(
        const llama_memory_recurrent_context * current,
        const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);
    mctx = current;

    const int64_t n_rs = mctx->get_n_rs();

    // Read rollback-aware brain indices before s_copy() consumes and resets
    // each sequence's rollback selector.
    if (brain_copy && brain_copy->buffer != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(brain_copy->buffer));
        int32_t * data = (int32_t *) brain_copy->data;
        for (uint32_t i = 0; i < mctx->get_ubatch().n_seqs; ++i) {
            data[i] = mctx->brain_copy((int32_t) i);
            GGML_ASSERT(data[i] >= 0);
        }
    }

    const auto public_groups = mctx->public_brain_groups();
    GGML_ASSERT(public_groups.size() == rbb_groups.size());
    size_t group_index = 0;
    for (const auto & [brain_row, rows] : public_groups) {
        const auto & group = rbb_groups[group_index++];
        GGML_ASSERT(group.brain_row == brain_row);
        GGML_ASSERT(group.public_rows != nullptr);
        GGML_ASSERT((size_t) group.public_rows->ne[0] == rows.size());
        // The fused all-writer Parallel Delta node encodes this membership in
        // its graph shape and does not consume the index tensor.
        if (group.public_rows->buffer != nullptr) {
            GGML_ASSERT(ggml_backend_buffer_is_host(group.public_rows->buffer));
            ggml_backend_tensor_set(
                group.public_rows, rows.data(), 0, rows.size() * sizeof(rows[0]));
        }

        std::vector<int32_t> default_shared_rows;
        default_shared_rows.reserve(rows.size());
        for (const int32_t row : rows) {
            if (!mctx->is_child_row(row)) {
                default_shared_rows.push_back(row);
            }
        }
        GGML_ASSERT((group.default_shared_rows != nullptr) == !default_shared_rows.empty());
        if (group.default_shared_rows != nullptr) {
            GGML_ASSERT((size_t) group.default_shared_rows->ne[0] == default_shared_rows.size());
            if (group.default_shared_rows->buffer != nullptr) {
                GGML_ASSERT(ggml_backend_buffer_is_host(group.default_shared_rows->buffer));
                ggml_backend_tensor_set(
                    group.default_shared_rows,
                    default_shared_rows.data(),
                    0,
                    default_shared_rows.size() * sizeof(default_shared_rows[0]));
            }
        }
    }

    if (s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_copy->buffer));
        int32_t * data = (int32_t *) s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->s_copy(i);
        }
    }
}

bool llm_graph_input_rs::can_reuse(const llm_graph_params & params) {
    const auto * current = static_cast<const llama_memory_recurrent_context *>(params.mctx);
    return can_reuse_recurrent(current, params.ubatch);
}

bool llm_graph_input_rs::can_reuse_recurrent(
        const llama_memory_recurrent_context * current,
        const llama_ubatch & ubatch) {
    mctx = current;

    bool res = true;

    res &= s_copy->ne[0] == mctx->get_n_rs();
    res &= s_copy_main->ne[0]  == ubatch.n_seqs;
    res &= s_copy_extra->ne[0] == mctx->get_n_rs() - ubatch.n_seqs;
    res &= (brain_copy != nullptr) == mctx->is_grouped();
    if (brain_copy) {
        res &= brain_copy->ne[0] == ubatch.n_seqs;
    }

    const auto public_groups = mctx->public_brain_groups();
    res &= public_groups.size() == rbb_groups.size();
    if (res) {
        size_t group_index = 0;
        for (const auto & [brain_row, rows] : public_groups) {
            const auto & group = rbb_groups[group_index++];
            res &= group.brain_row == brain_row;
            res &= group.public_rows != nullptr;
            res &= (size_t) group.public_rows->ne[0] == rows.size();

            size_t n_default_shared = 0;
            for (const int32_t row : rows) {
                n_default_shared += !mctx->is_child_row(row);
            }
            res &= (group.default_shared_rows != nullptr) == (n_default_shared != 0);
            if (group.default_shared_rows != nullptr) {
                res &= (size_t) group.default_shared_rows->ne[0] == n_default_shared;
            }
        }
    }

    res &= head == mctx->get_head();
    res &= rs_z == mctx->get_rs_z();

    return res;
}

void llm_graph_input_cross_embd::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (cross_embd && !cross->v_embd.empty()) {
        assert(cross_embd->type == GGML_TYPE_F32);

        ggml_backend_tensor_set(cross_embd, cross->v_embd.data(), 0, ggml_nbytes(cross_embd));
    }
}

template <typename T>
static void print_mask(const T * data, int64_t n_tokens, int64_t n_kv, int64_t n_swa, llama_swa_type swa_type) {
    LLAMA_LOG_DEBUG("%s: === Attention mask ===\n", __func__);
    const char * swa_type_str = "unknown";

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:      swa_type_str = "LLAMA_SWA_TYPE_NONE"; break;
        case LLAMA_SWA_TYPE_STANDARD:  swa_type_str = "LLAMA_SWA_TYPE_STANDARD"; break;
        case LLAMA_SWA_TYPE_CHUNKED:   swa_type_str = "LLAMA_SWA_TYPE_CHUNKED"; break;
        case LLAMA_SWA_TYPE_SYMMETRIC: swa_type_str = "LLAMA_SWA_TYPE_SYMMETRIC"; break;
    };

    LLAMA_LOG_DEBUG("%s: n_swa : %d, n_kv: %d, swa_type: %s\n", __func__, (int)n_swa, (int)n_kv, swa_type_str);
    LLAMA_LOG_DEBUG("%s: '0' = can attend, '∞' = masked\n", __func__);
    LLAMA_LOG_DEBUG("%s: Rows = query tokens, Columns = key/value tokens\n\n", __func__);

    LLAMA_LOG_DEBUG("    ");
    for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
        LLAMA_LOG_DEBUG("%2d", j);
    }
    LLAMA_LOG_DEBUG("\n");

    for (int i = 0; i < std::min((int64_t)20, n_tokens); ++i) {
        LLAMA_LOG_DEBUG(" %2d ", i);
        for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
            float val = llama_cast<float>(data[i * n_kv + j]);
            if (val == -INFINITY) {
                LLAMA_LOG_DEBUG(" ∞");
            } else {
                LLAMA_LOG_DEBUG(" 0");
            }
        }
        LLAMA_LOG_DEBUG("\n");
    }
}

void llm_graph_input_attn_no_cache::set_input(const llama_ubatch * ubatch) {
    const int64_t n_kv     = ubatch->n_tokens;
    const int64_t n_tokens = ubatch->n_tokens;

    const auto fill_mask = [&](auto * data, int64_t ne, int n_swa, llama_swa_type swa_type) {
        using T = std::remove_reference_t<decltype(*data)>;
        std::fill(data, data + ne, llama_cast<T>(-INFINITY));

        for (int i1 = 0; i1 < n_tokens; ++i1) {
            const llama_seq_id s1 = ubatch->seq_id[i1][0];
            const llama_pos    p1 = ubatch->pos[i1];

            const uint64_t idst = i1*n_kv;

            for (int i0 = 0; i0 < n_tokens; ++i0) {
                const llama_seq_id s0 = ubatch->seq_id[i0][0];
                const llama_pos p0    = ubatch->pos[i0];

                // mask different sequences
                if (s0 != s1) {
                    continue;
                }

                // mask future tokens
                if (cparams.causal_attn && p0 > p1) {
                    continue;
                }

                // apply SWA if any
                if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                    continue;
                }

                data[idst + i0] = llama_cast<T>(hparams.use_alibi ? -std::abs(p0 - p1) : 0.0f);
            }
        }

        if (debug) {
            print_mask(data, n_tokens, n_kv, n_swa, swa_type);
        }
    };

    GGML_ASSERT(self_kq_mask);
    GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer));
    if (self_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE);
    } else {
        fill_mask((float       *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE);
    }

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        GGML_ASSERT(self_kq_mask_swa);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));
        if (self_kq_mask_swa->type == GGML_TYPE_F16) {
            fill_mask((ggml_fp16_t *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type);
        } else {
            fill_mask((float       *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type);
        }
    }
}

// ---------------------------------------------------------------------------
// llm_graph_input_attn_rerot (RERoT DDVR graph input, §§9-12, 21)
// ---------------------------------------------------------------------------

uint32_t llm_graph_input_attn_rerot::capacity_bucket(uint32_t n) {
    if (n == 0) {
        return 0;
    }
    return ((n + SPAN_BUCKET - 1) / SPAN_BUCKET) * SPAN_BUCKET;
}

bool llm_graph_input_attn_rerot::graph_reuse_disabled() {
    const char * env = getenv("LLAMA_REROT_DISABLE_GRAPH_REUSE");
    return env != nullptr && env[0] != '\0' && !(env[0] == '0' && env[1] == '\0');
}

bool llm_graph_input_attn_rerot::is_supported_arch(const llama_hparams & hparams) {
    // v1 is text-only. Multimodal embedding batches fall back to the ordinary
    // serial path model-side (ubatch.embd); only VISION RoPE is a hard error.
    return hparams.rope_type != LLAMA_ROPE_TYPE_VISION;
}

llm_rerot_kernel_variant llm_graph_input_attn_rerot::select_variant(ggml_backend_sched_t sched) {
    if (sched == nullptr) {
        return llm_rerot_kernel_variant::REROT_KERNEL_CPU;
    }
    const int n = ggml_backend_sched_get_n_backends(sched);
    bool seen_vulkan = false;
    for (int i = 0; i < n; ++i) {
        ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
        if (b == nullptr) {
            continue;
        }
        const char * name = ggml_backend_name(b);
        if (name == nullptr) {
            continue;
        }
        const std::string s(name);
        if (s.find("CUDA") != std::string::npos ||
            s.find("Metal") != std::string::npos ||
            s.find("RPC") != std::string::npos) {
            return llm_rerot_kernel_variant::REROT_KERNEL_UNSUPPORTED;
        }
        if (s.find("Vulkan") != std::string::npos || s.find("vulkan") != std::string::npos) {
            seen_vulkan = true;
        }
    }
    return seen_vulkan
        ? llm_rerot_kernel_variant::REROT_KERNEL_VULKAN_FUSED
        : llm_rerot_kernel_variant::REROT_KERNEL_CPU;
}

void llm_graph_input_attn_rerot::require_supported(
        ggml_backend_sched_t sched,
        const llama_hparams & hparams,
        const char * where) {
    if (!is_supported_arch(hparams)) {
        throw std::runtime_error(
            std::string("RERoT DDVR: unsupported RoPE architecture in ") + where +
            " (multimodal/VISION is outside the text-only v1 scope;"
            " refusing silent stock attention)");
    }
    if (select_variant(sched) == llm_rerot_kernel_variant::REROT_KERNEL_UNSUPPORTED) {
        std::string names;
        if (sched != nullptr) {
            const int n = ggml_backend_sched_get_n_backends(sched);
            for (int i = 0; i < n; ++i) {
                ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
                if (b == nullptr) {
                    continue;
                }
                const char * name = ggml_backend_name(b);
                if (name == nullptr) {
                    continue;
                }
                if (!names.empty()) {
                    names += ",";
                }
                names += name;
            }
        }
        throw std::runtime_error(
            "RERoT DDVR: no RERoT kernel for backend(s) [" + names + "] in " +
            where + " (CUDA/Metal/RPC today); refusing silent stock attention");
    }
}

llm_rerot_span_reuse_key llm_graph_input_attn_rerot::make_key(
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t n_entries,
        llm_rerot_kernel_variant variant,
        bool rerot_on,
        llama_rerot_frontier_mode mode) {
    llm_rerot_span_reuse_key k;
    k.n_tokens   = n_tokens;
    k.group_cap  = capacity_bucket(n_groups);
    k.entry_cap  = capacity_bucket(n_entries);
    k.variant    = variant;
    k.rerot_on   = rerot_on;
    k.frontier_mode = mode;
    return k;
}

void llm_graph_input_attn_rerot::build_span_tensors(
        ggml_context * ctx0,
        ggml_tensor *& q_indices,
        ggml_tensor *& q_pos,
        ggml_tensor *& entries,
        ggml_tensor *& offsets,
        const llama_ubatch & ubatch,
        const llama_hparams & hparams,
        const llama_cparams & cparams,
        const llama_kv_cache_context * attn,
        ggml_backend_sched_t sched) {
    GGML_ASSERT(attn && attn->rerot_active());
    const auto & layout = attn->get_rerot_attn_layout();
    if (layout.empty() || layout.groups.empty() || layout.entries.empty()) {
        throw std::runtime_error(
            "RERoT DDVR: active batch produced an empty span layout;"
            " refusing silent stock attention");
    }
    const uint32_t n_pos = hparams.n_pos_per_embd();
    if (n_pos != 1 && n_pos != 4) {
        throw std::runtime_error("RERoT DDVR: unsupported position width (text-only v1 supports 1 or 4)");
    }
    if (layout.query_offsets.size() != (size_t) ubatch.n_tokens + 1) {
        throw std::runtime_error("RERoT DDVR: query offsets do not cover the ubatch rows");
    }
    // Backend support itself is enforced at op selection (require_supported);
    // the variant is recorded here so a backend change forces a rebuild.
    key = make_key(
        ubatch.n_tokens,
        (uint32_t) layout.groups.size(),
        (uint32_t) layout.entries.size(),
        select_variant(sched), true, cparams.rerot_frontier);
    key_valid = true;

    q_indices = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, key.group_cap);
    ggml_set_input(q_indices);

    q_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) key.group_cap * n_pos);
    ggml_set_input(q_pos);

    entries = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 2, key.entry_cap);
    ggml_set_input(entries);

    offsets = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) ubatch.n_tokens + 1);
    ggml_set_input(offsets);
}

bool llm_graph_input_attn_rerot::spans_can_reuse(
        const llama_ubatch & ubatch,
        const llama_hparams & hparams,
        const llama_cparams & cparams,
        const llama_kv_cache_context * attn,
        ggml_backend_sched_t sched,
        const ggml_tensor * q_indices,
        const ggml_tensor * q_pos,
        const ggml_tensor * entries,
                    const ggml_tensor * offsets) const {
    const bool has = attn && attn->rerot_active();
    const bool had = q_indices != nullptr;
    if (!had && !has) {
        return true;
    }
    if (had != has) {
        return false;
    }
    if (!key_valid) {
        return false;
    }
    // Explicit first-CPU-prototype hatch: disable reuse while RERoT is active.
    // The capacity-bucketed key design above stays authoritative regardless.
    if (graph_reuse_disabled()) {
        return false;
    }
    const auto & layout = attn->get_rerot_attn_layout();
    if (layout.empty() || layout.groups.empty() || layout.entries.empty()) {
        return false;
    }
    const auto cur = make_key(
        ubatch.n_tokens,
        (uint32_t) layout.groups.size(),
        (uint32_t) layout.entries.size(),
        select_variant(sched), true, cparams.rerot_frontier);
    if (cur != key) {
        return false;
    }
    // Sanity: topology must equal the recorded capacity buckets. Span data
    // churn (virtual_pos0 / storage_pos0 / counts within capacity /
    // visibility) intentionally does not appear here.
    const uint32_t n_pos = hparams.n_pos_per_embd();
    if (q_indices->ne[0] != (int64_t) key.group_cap) {
        return false;
    }
    if (q_pos == nullptr || q_pos->ne[0] != (int64_t) key.group_cap * n_pos) {
        return false;
    }
    if (entries == nullptr || entries->ne[0] != 2 || entries->ne[1] != (int64_t) key.entry_cap) {
        return false;
    }
    if (offsets == nullptr || offsets->ne[0] != (int64_t) key.n_tokens + 1) {
        return false;
    }
    return true;
}

void llm_graph_input_attn_rerot::fill_spans(
        ggml_tensor * q_indices,
        ggml_tensor * q_pos,
        ggml_tensor * entries,
        ggml_tensor * offsets,
        const llama_kv_cache_context * attn,
        uint32_t n_pos) {
    GGML_ASSERT(q_indices && q_pos && entries && offsets && attn);
    GGML_ASSERT(n_pos == 1 || n_pos == 4);
    const auto & layout = attn->get_rerot_attn_layout();
    if (layout.empty() || layout.groups.empty() || layout.entries.empty()) {
        throw std::runtime_error("RERoT DDVR: cannot fill span tensors from an empty layout");
    }
    const size_t n_groups  = layout.groups.size();
    const size_t n_entries = layout.entries.size();
    const size_t n_offsets = layout.query_offsets.size();
    const int64_t group_cap = q_indices->ne[0];
    if ((int64_t) n_groups > group_cap) {
        throw std::runtime_error(
            "RERoT DDVR: span layout exceeds tensor capacity (stale reuse key);"
            " rebuild required, refusing to truncate");
    }
    if (q_pos->ne[0] != group_cap * (int64_t) n_pos) {
        throw std::runtime_error("RERoT DDVR: q_pos capacity does not match q_indices capacity");
    }
    if (entries->ne[0] != 2 || entries->ne[1] < (int64_t) n_entries) {
        throw std::runtime_error(
            "RERoT DDVR: entry layout exceeds tensor capacity;"
            " rebuild required, refusing to truncate");
    }
    if (offsets->ne[0] != (int64_t) n_offsets) {
        throw std::runtime_error("RERoT DDVR: query offsets do not match tensor shape");
    }

    // Per-instance staging snapshot: no statics, no aliasing of layout
    // internals, so a later frontier cannot overwrite span tables still
    // referenced by an in-flight graph. Callers order fill_spans after the
    // previous frontier's graph has completed (same discipline as k_idxs).
    st_q_indices.assign((size_t) group_cap, 0);
    for (size_t i = 0; i < n_groups; ++i) {
        st_q_indices[i] = (int32_t) layout.groups[i].query_index;
    }

    // q_pos uses the tensor's own group-capacity stride (coordinate k lives at
    // k * group_cap), matching how ggml_rope reads multi-pos. IMRoPE text
    // rule: (p, p, p, 0) with the delta on the first three coordinates only.
    st_q_pos.assign((size_t) group_cap * n_pos, 0);
    for (size_t i = 0; i < n_groups; ++i) {
        const int32_t p = (int32_t) layout.groups[i].effective_pos;
        st_q_pos[i] = p;
        if (n_pos == 4) {
            st_q_pos[(size_t) group_cap + i]     = p;
            st_q_pos[(size_t) group_cap * 2 + i] = p;
            st_q_pos[(size_t) group_cap * 3 + i] = 0;
        }
    }

    st_entries.assign((size_t) entries->ne[1] * 2, 0);
    for (size_t i = 0; i < n_entries; ++i) {
        st_entries[2 * i + 0] = (int32_t) layout.entries[i].key_index;
        st_entries[2 * i + 1] = (int32_t) layout.entries[i].group_index;
    }

    st_offsets.assign(n_offsets, 0);
    for (size_t i = 0; i < n_offsets; ++i) {
        st_offsets[i] = (int32_t) layout.query_offsets[i];
    }

    ggml_backend_tensor_set(q_indices, st_q_indices.data(), 0, ggml_nbytes(q_indices));
    ggml_backend_tensor_set(q_pos,     st_q_pos.data(),     0, ggml_nbytes(q_pos));
    ggml_backend_tensor_set(entries,   st_entries.data(),   0, ggml_nbytes(entries));
    ggml_backend_tensor_set(offsets,   st_offsets.data(),   0, ggml_nbytes(offsets));

    ++epoch;
}

bool llm_graph_input_attn_kv::rerot_spans_can_reuse(
        const llama_ubatch & ubatch,
        ggml_backend_sched_t sched,
        const llama_cparams & cparams,
        const llama_kv_cache_context * attn) const {
    return rerot_spans.spans_can_reuse(
        ubatch, hparams, cparams, attn, sched,
        self_rerot_q_indices, self_rerot_q_pos, self_rerot_entries, self_rerot_offsets);
}

void llm_graph_input_attn_kv::rerot_spans_fill(
        const llama_ubatch * ubatch,
        const llama_kv_cache_context * attn) {
    GGML_UNUSED(ubatch);
    rerot_spans.fill_spans(
        self_rerot_q_indices, self_rerot_q_pos, self_rerot_entries, self_rerot_offsets,
        attn, hparams.n_pos_per_embd());
}

void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->set_input_v_idxs(self_v_idxs, ubatch);

    // the mask is left unallocated when the graph only stores K/V without attending
    // (e.g. DFlash's KV-injection pass)
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot && self_v_rot->buffer) {
        mctx->set_input_v_rot(self_v_rot);
    }

    if (self_rerot_q_indices) {
        // RERoT DDVR span data refresh (input tensor data, not topology).
        // Ordinary k/v/mask fills above are untouched; OFF batches never enter.
        rerot_spans_fill(ubatch, mctx);
    }

    if (fp) {
        // FlashPrefill wire metadata refresh (counts/epochs/physical maps as
        // tensor data, never topology). Inactive (null) graphs skip entirely.
        fp->set_input(ubatch);
    }
}

bool llm_graph_input_attn_kv::rerot_semantic() const {
    return mctx != nullptr && mctx->rerot_active();
}

bool llm_graph_input_attn_kv::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    // RERoT DDVR: span metadata/offsets/visibility are input tensor data, not
    // topology. The reuse key covers only (n_token rows, span capacity bucket,
    // kernel variant, rerot on/off, frontier mode); virtual_pos0 /
    // storage_pos0 / count (within capacity) / visibility churn refreshes data
    // without forcing a rebuild. OFF pairs (!had && !has) reuse as before.
    res &= rerot_spans_can_reuse(params.ubatch, params.sched, params.cparams, mctx);

    // FlashPrefill: topology key (policy/mode/role/buckets/dtype/variant)
    // decides reuse; the companion refreshes its row snapshot from the
    // current params (never the stale build-time copy) before set_input.
    if (fp) {
        fp->mctx = mctx;
        res &= fp->can_reuse(params);
    } else {
        // No companion reuses only against params that also build none.
        // wants_companion runs the shared owner-building probe (not a
        // get-only peek), so a fixed-capacity dense->sparse transition
        // actually rebuilds instead of reusing dense forever; steady
        // deny-states keep reusing without rebuild churn.
        res &= !llm_graph_input_attn_flashprefill::wants_companion(params, mctx);
    }

    return res;
}

void llm_graph_input_attn_k::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);

    mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    return res;
}

llm_graph_input_attn_kv_msa::llm_graph_input_attn_kv_msa(
        const llama_hparams & hparams,
        const llama_cparams & cparams,
        const llama_kv_cache_msa_context * mctx) :
    llm_graph_input_attn_kv(hparams, cparams, mctx->get_base()),
    mctx_msa(mctx) {
}

void llm_graph_input_attn_kv_msa::set_input(const llama_ubatch * ubatch) {
    llm_graph_input_attn_kv::set_input(ubatch);

    if (self_k_idxs_idx) {
        mctx_msa->get_idx()->set_input_k_idxs(self_k_idxs_idx, ubatch);
    }
}

bool llm_graph_input_attn_kv_msa::can_reuse(const llm_graph_params & params) {
    mctx_msa = static_cast<const llama_kv_cache_msa_context *>(params.mctx);

    // the parent class operates on the base cache context
    this->mctx = mctx_msa->get_base();

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    if (self_k_idxs_idx) {
        res &= self_k_idxs_idx->ne[0] == params.ubatch.n_tokens;
    }

    res &= can_reuse_kq_mask(self_kq_mask, this->mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k_dsa::set_input(const llama_ubatch * ubatch) {
    mctx->get_mla()->set_input_k_idxs(self_k_idxs_mla, ubatch);

    mctx->get_mla()->set_input_kq_mask(self_kq_mask_mla, ubatch, cparams.causal_attn);

    mctx->get_lid()->set_input_k_idxs(self_k_idxs_lid, ubatch);

    mctx->get_lid()->set_input_kq_mask(self_kq_mask_lid, ubatch, cparams.causal_attn);

    mctx->get_lid()->set_input_k_rot(self_k_rot_lid);
}

bool llm_graph_input_attn_k_dsa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_dsa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs_mla->ne[0] == params.ubatch.n_tokens;
    res &= self_k_idxs_lid->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask_mla, mctx->get_mla(), params.ubatch, params.cparams);
    res &= can_reuse_kq_mask(self_kq_mask_lid, mctx->get_lid(), params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_kv_iswa::set_input(const llama_ubatch * ubatch) {
    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
        if (self_v_idxs) {
            mctx->get_base()->set_input_v_idxs(self_v_idxs, ubatch);
        }
    }

    // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
        if (self_v_idxs_swa) {
            mctx->get_swa()->set_input_v_idxs(self_v_idxs_swa, ubatch);
        }
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot && self_v_rot->buffer) {
        mctx->get_base()->set_input_v_rot(self_v_rot);
    }

    if (self_k_rot_swa && self_k_rot_swa->buffer) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }

    if (self_v_rot_swa && self_v_rot_swa->buffer) {
        mctx->get_swa()->set_input_v_rot(self_v_rot_swa);
    }
}

bool llm_graph_input_attn_kv_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }

    return res;
}

void llm_graph_input_attn_k_iswa::set_input(const llama_ubatch * ubatch) {
    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
    }

    // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_k_rot_swa && self_k_rot_swa->buffer) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }
}

bool llm_graph_input_attn_k_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }

    return res;
}

static void dsv4_set_i64(ggml_tensor * dst, const std::vector<int64_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_i32(ggml_tensor * dst, const std::vector<int32_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_kq_mask(
        ggml_tensor * dst,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);
    GGML_ASSERT(dst->ne[0] == plan.n_kv);
    GGML_ASSERT(dst->ne[1] == (int64_t) n_tokens/n_stream);
    GGML_ASSERT(dst->ne[2] == 1);
    GGML_ASSERT(dst->ne[3] == n_stream);
    GGML_ASSERT((int64_t) plan.n_visible.size() == (int64_t) n_tokens);
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    if (dst->type == GGML_TYPE_F32) {
        float * data = (float *) dst->data;

        for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
            const int32_t n_visible = plan.n_visible[i];

            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                data[i*dst->ne[0] + j] = j < n_visible ? 0.0f : -INFINITY;
            }
        }
    } else if (dst->type == GGML_TYPE_F16) {
        ggml_fp16_t * data = (ggml_fp16_t *) dst->data;
        const ggml_fp16_t fp16_ninf = llama_cast<ggml_fp16_t>(-INFINITY);
        const ggml_fp16_t fp16_zero = llama_cast<ggml_fp16_t>(0.0f);

        for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
            const int32_t n_visible = plan.n_visible[i];

            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                data[i*dst->ne[0] + j] = j < n_visible ? fp16_zero : fp16_ninf;
            }
        }
    }
}

static ggml_tensor * dsv4_build_raw_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_dsv4_raw_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);

    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool dsv4_can_reuse_raw_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_dsv4_raw_context * mctx,
        const llama_ubatch & ubatch,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

static std::string dsv4_plan_positions(const std::vector<int32_t> & values) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            ss << ", ";
        }
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}

static bool dsv4_compress_debug() {
    static const bool debug = []() {
        const char * env = getenv("LLAMA_DSV4_COMPRESS_DEBUG");
        return env && atoi(env) > 0;
    }();

    return debug;
}

static void dsv4_set_comp_inputs(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        bool debug,
        uint32_t n_tokens,
        int64_t n_stream) {
    dsv4_set_i32(inp.state_pos, plan.state_pos);
    dsv4_set_i32(inp.state_persist_src_idxs, plan.state_persist_src_idxs);
    dsv4_set_i32(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs);
    dsv4_set_i32(inp.state_restore_src_idxs, plan.state_restore_src_idxs);
    dsv4_set_i32(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs);
    dsv4_set_i32(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs);
    dsv4_set_i32(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs);
    dsv4_set_i32(inp.state_read_idxs, plan.state_read_idxs);
    dsv4_set_i64(inp.state_write_idxs, plan.state_write_idxs);
    dsv4_set_i32(inp.state_write_pos, plan.state_write_pos);
    dsv4_set_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    if (debug || dsv4_compress_debug()) {
        LLAMA_LOG_INFO("%s: %s n_tokens=%u, n_stream=%d, state_persist_dst=%s, state_write_pos=%s\n",
                __func__, name, n_tokens, (int) n_stream,
                dsv4_plan_positions(plan.state_persist_dst_idxs).c_str(),
                dsv4_plan_positions(plan.state_write_pos).c_str());
    }
}

static bool dsv4_can_reuse_tensor_1d(ggml_tensor * t, int64_t ne0) {
    return (t == nullptr && ne0 == 0) || (t != nullptr && t->ne[0] == ne0);
}

static bool dsv4_can_reuse_kq_mask(
        ggml_tensor * t,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (plan.n_kv == 0) {
        return t == nullptr;
    }

    GGML_ASSERT(n_stream > 0);

    return t != nullptr &&
           t->ne[0] == plan.n_kv &&
           t->ne[1] == (int64_t) n_tokens/n_stream &&
           t->ne[2] == 1 &&
           t->ne[3] == n_stream;
}

static bool dsv4_can_reuse_comp_input(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    bool res = true;
    res &= dsv4_can_reuse_tensor_1d(inp.state_pos, plan.state_pos.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_src_idxs, plan.state_persist_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_restore_src_idxs, plan.state_restore_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_read_idxs, plan.state_read_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_idxs, plan.state_write_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_pos, plan.state_write_pos.size());
    res &= dsv4_can_reuse_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    return res;
}

static ggml_tensor * dsv4_build_input_1d(
        ggml_context * ctx,
        ggml_type type,
        int64_t ne0,
        const std::string & name) {
    if (ne0 == 0) {
        return nullptr;
    }

    ggml_tensor * res = ggml_new_tensor_1d(ctx, type, ne0);
    ggml_set_input(res);
    ggml_set_name(res, name.c_str());

    return res;
}

static void dsv4_build_comp_inputs(
        ggml_context * ctx,
        llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        const llama_cparams & cparams,
        int64_t n_stream) {
    inp.state_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_pos.size(), std::string("dsv4_") + name + "_state_pos");
    inp.state_persist_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_src_idxs.size(), std::string("dsv4_") + name + "_state_persist_src_idxs");
    inp.state_persist_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_dst_idxs.size(), std::string("dsv4_") + name + "_state_persist_dst_idxs");
    inp.state_restore_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_restore_src_idxs.size(), std::string("dsv4_") + name + "_state_restore_src_idxs");
    inp.state_restore_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_restore_dst_idxs.size(), std::string("dsv4_") + name + "_state_restore_dst_idxs");
    inp.state_snapshot_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_snapshot_src_idxs.size(), std::string("dsv4_") + name + "_state_snapshot_src_idxs");
    inp.state_snapshot_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_snapshot_dst_idxs.size(), std::string("dsv4_") + name + "_state_snapshot_dst_idxs");
    inp.state_read_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_read_idxs.size(), std::string("dsv4_") + name + "_state_read_idxs");
    inp.state_write_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I64, plan.state_write_idxs.size(), std::string("dsv4_") + name + "_state_write_idxs");
    inp.state_write_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_write_pos.size(), std::string("dsv4_") + name + "_state_write_pos");

    if (plan.n_kv > 0) {
        const int64_t n_tokens = (int64_t) plan.n_visible.size();

        GGML_ASSERT(n_stream > 0);
        GGML_ASSERT(n_tokens%n_stream == 0);

        inp.kq_mask = ggml_new_tensor_4d(ctx, (strcmp(name, "lid") != 0 && cparams.flash_attn) || (strcmp(name, "lid") == 0 && cparams.fused_lid) ? GGML_TYPE_F16 : GGML_TYPE_F32, plan.n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp.kq_mask);
        ggml_set_name(inp.kq_mask, (std::string("dsv4_") + name + "_kq_mask").c_str());
    }
}

void llm_graph_input_dsv4_raw::set_input(const llama_ubatch * ubatch) {
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->set_input_k_idxs(self_k_idxs);
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    if (self_k_rot) {
        mctx->set_input_k_rot(self_k_rot);
    }
}

void llm_graph_input_dsv4::set_input(const llama_ubatch * ubatch) {
    const auto & plan_csa = mctx->get_csa_plan(*ubatch);
    const auto & plan_hca = mctx->get_hca_plan(*ubatch);
    const auto & plan_lid = mctx->get_lid_plan(*ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    inp_raw->mctx = mctx->get_raw();
    inp_raw->set_input(ubatch);

    dsv4_set_comp_inputs(inp_csa, plan_csa, "csa", debug > 0, ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_hca, plan_hca, "hca", debug > 0, ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_lid, plan_lid, "lid", debug > 0, ubatch->n_tokens, n_stream);

    if (inp_csa.k_rot && inp_csa.k_rot->buffer) {
        mctx->get_csa()->set_input_k_rot(inp_csa.k_rot);
    }

    if (inp_hca.k_rot && inp_hca.k_rot->buffer) {
        mctx->get_hca()->set_input_k_rot(inp_hca.k_rot);
    }

    if (inp_lid.k_rot && inp_lid.k_rot->buffer) {
        mctx->get_lid()->set_input_k_rot(inp_lid.k_rot);
    }
}

bool llm_graph_input_dsv4::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx);

    this->mctx = mctx;
    inp_raw->mctx = mctx->get_raw();

    bool res = true;

    const auto & plan_csa = mctx->get_csa_plan(params.ubatch);
    const auto & plan_hca = mctx->get_hca_plan(params.ubatch);
    const auto & plan_lid = mctx->get_lid_plan(params.ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    const auto * raw_ctx = mctx->get_raw();
    inp_raw->mctx = raw_ctx;

    if (inp_raw->self_k_idxs && inp_raw->self_k_idxs->buffer) {
        res &= inp_raw->self_k_idxs->ne[0] == raw_ctx->get_n_write();
    }
    if (inp_raw->self_kq_mask && inp_raw->self_kq_mask->buffer) {
        res &= dsv4_can_reuse_raw_kq_mask(inp_raw->self_kq_mask, raw_ctx, params.ubatch, n_stream);
    }

    res &= dsv4_can_reuse_comp_input(inp_csa, plan_csa, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_hca, plan_hca, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_lid, plan_lid, params.ubatch.n_tokens, n_stream);

    return res;
}

void llm_graph_input_attn_cross::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(cross_kq_mask);

    const int64_t n_enc    = cross_kq_mask->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(cross_kq_mask->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    const auto fill_mask = [&](auto * data) {
        using T = std::remove_reference_t<decltype(*data)>;
        for (int i = 0; i < n_tokens; ++i) {
            GGML_ASSERT(!cross->seq_ids_enc.empty() && "llama_encode must be called first");
            for (int j = 0; j < n_enc; ++j) {
                float f = -INFINITY;

                for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                    const llama_seq_id seq_id = ubatch->seq_id[i][s];

                    if (cross->seq_ids_enc[j].find(seq_id) != cross->seq_ids_enc[j].end()) {
                        f = 0.0f;
                    }
                }

                data[i*n_enc + j] = llama_cast<T>(f);
            }
        }
    };

    if (cross_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) cross_kq_mask->data);
    } else {
        fill_mask((float *) cross_kq_mask->data);
    }
}

void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
    mctx->get_attn()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);

    // The dedicated RERoT attention op consumes the resolved visibility/span
    // tensors instead of the ordinary membership mask. ggml therefore prunes
    // self_kq_mask from that graph and leaves its input tensor unallocated.
    // Keep the strict ordinary-path assertion in set_input_kq_mask(), but do
    // not write an intentionally unused input.
    if (inp_attn->self_kq_mask->buffer) {
        mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);
    } else {
        GGML_ASSERT(inp_attn->rerot_active());
    }

    if (inp_attn->self_k_rot) {
        mctx->get_attn()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot) {
        mctx->get_attn()->set_input_v_rot(inp_attn->self_v_rot);
    }

    if (inp_attn->rerot_active()) {
        // RERoT DDVR span data for hybrid full-attention layers. Recurrent
        // state above stays lane-local; only the shared-attention span tables
        // refresh here. OFF batches never enter.
        inp_attn->rerot_spans_fill(ubatch, mctx->get_attn());
    }

    if (inp_attn->fp) {
        // FlashPrefill wire metadata refresh (same data-not-topology
        // discipline as the span tables above).
        inp_attn->fp->set_input(ubatch);
    }

    inp_rs->set_input_recurrent(mctx->get_recr(), ubatch);
}

bool llm_graph_input_mem_hybrid::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    // RERoT DDVR span reuse: same capacity-bucketed key as the plain KV path
    // (n_token rows, span capacity bucket, kernel variant, on/off, frontier
    // mode). Span data churn alone must not force a rebuild.
    res &= inp_attn->rerot_spans_can_reuse(
        params.ubatch, params.sched, params.cparams, mctx->get_attn());

    // FlashPrefill companion (same had-none stability rule as the plain KV
    // path above: owner-building probe, no rebuild loops, no forced
    // rebuilds as a substitute for per-submit data refresh).
    if (inp_attn->fp) {
        inp_attn->fp->mctx = mctx->get_attn();
        res &= inp_attn->fp->can_reuse(params);
    } else {
        res &= !llm_graph_input_attn_flashprefill::wants_companion(params, mctx->get_attn());
    }

    res &= inp_rs->can_reuse_recurrent(mctx->get_recr(), params.ubatch);

    return res;
}

// TODO: Hybrid input classes are a bit redundant.
// Instead of creating a hybrid input, the graph can simply create 2 separate inputs.
// Refactoring is required in the future.
void llm_graph_input_mem_hybrid_k::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);

    mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);

    inp_rs->set_input_recurrent(mctx->get_recr(), ubatch);
}

bool llm_graph_input_mem_hybrid_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->can_reuse_recurrent(mctx->get_recr(), params.ubatch);

    return res;
}

void llm_graph_input_mem_hybrid_iswa::set_input(const llama_ubatch * ubatch) {
    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        attn_ctx->get_base()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
        attn_ctx->get_base()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);
    }

    if (inp_attn->self_kq_mask && inp_attn->self_kq_mask->buffer) {
        attn_ctx->get_base()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        attn_ctx->get_swa()->set_input_k_idxs(inp_attn->self_k_idxs_swa, ubatch);
        attn_ctx->get_swa()->set_input_v_idxs(inp_attn->self_v_idxs_swa, ubatch);
    }

    if (inp_attn->self_kq_mask_swa && inp_attn->self_kq_mask_swa->buffer) {
        attn_ctx->get_swa()->set_input_kq_mask(inp_attn->self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (inp_attn->self_k_rot) {
        attn_ctx->get_base()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot) {
        attn_ctx->get_base()->set_input_v_rot(inp_attn->self_v_rot);
    }

    if (inp_attn->self_k_rot_swa) {
        attn_ctx->get_swa()->set_input_k_rot(inp_attn->self_k_rot_swa);
    }

    if (inp_attn->self_v_rot_swa) {
        attn_ctx->get_swa()->set_input_v_rot(inp_attn->self_v_rot_swa);
    }

    inp_rs->set_input_recurrent(mctx->get_recr(), ubatch);
}

bool llm_graph_input_mem_hybrid_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, attn_ctx->get_base(), params.ubatch, params.cparams);

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        res &= inp_attn->self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask_swa, attn_ctx->get_swa(), params.ubatch, params.cparams);

    res &= inp_rs->can_reuse_recurrent(mctx->get_recr(), params.ubatch);

    return res;
}

void llm_graph_input_sampling::set_input(const llama_ubatch * ubatch) {
    // set the inputs only for the active samplers in the current ubatch
    std::unordered_set<llama_seq_id> active_samplers;
    for (uint32_t i = 0; i < ubatch->n_tokens; i++) {
        if (ubatch->output[i]) {
            llama_seq_id seq_id = ubatch->seq_id[i][0];
            active_samplers.insert(seq_id);
        }
    }

    for (auto seq_id : active_samplers) {
        if (samplers.find(seq_id) == samplers.end()) {
            continue;
        }

        auto & sampler = samplers[seq_id];

        if (sampler->iface->backend_set_input) {
            sampler->iface->backend_set_input(sampler);
        }
    }
}

bool llm_graph_input_sampling::can_reuse(const llm_graph_params & params) {
    if (samplers.size() != params.samplers.size()) {
        return false;
    }

    for (const auto & [seq_id, sampler] : params.samplers) {
        if (samplers[seq_id] != sampler) {
            return false;
        }
    }

    return true;
}

//
// llm_graph_result
//

llm_graph_result::llm_graph_result(int64_t max_nodes) : max_nodes(max_nodes) {
    reset();

    const char * LLAMA_GRAPH_RESULT_DEBUG = getenv("LLAMA_GRAPH_RESULT_DEBUG");
    debug = LLAMA_GRAPH_RESULT_DEBUG ? atoi(LLAMA_GRAPH_RESULT_DEBUG) : 0;
}

int64_t llm_graph_result::get_max_nodes() const {
    return max_nodes;
}

void llm_graph_result::reset() {
    t_inp_tokens  = nullptr;
    t_inp_embd    = nullptr;
    t_logits      = nullptr;
    t_embd        = nullptr;
    t_embd_pooled = nullptr;
    t_h_nextn     = nullptr;

    t_layer_inp.resize(LLAMA_MAX_LAYERS + 1);
    std::fill(t_layer_inp.begin(), t_layer_inp.end(), nullptr);

    t_attn_q_pre_rope.resize(LLAMA_MAX_LAYERS);
    std::fill(t_attn_q_pre_rope.begin(), t_attn_q_pre_rope.end(), nullptr);

    t_sampled.clear();
    t_sampled_probs.clear();
    t_sampled_logits.clear();
    t_candidates.clear();

    params = {};

    inputs.clear();
    fused_nodes.clear();
    flashprefill_plans.clear();
    flashprefill_plan_ils.clear();
    flashprefill_pool_peaks.clear();
    flashprefill_summary = llm_graph_flashprefill_summary{};

    buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false);
}

// Eligible FULL-attention layer count (defined with the FlashPrefill
// section below; declared here for the post-build gate in set_outputs).
static int32_t llm_fp_count_full_layers(const llama_hparams & hparams);

void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {
    for (auto & input : inputs) {
        input->set_input(ubatch);
    }
    // FlashPrefill submit refresh: per-submit route/visible/candidate
    // counts refresh from the latest pack even on reuse (no rebuild), so
    // the summary always describes the ubatch actually submitted. Layer and
    // plan accumulators (sparse/dense layers/rows, plans, byte sums) are
    // never clobbered here — only the pack-level route data.
    for (auto & input : inputs) {
        llm_graph_input_attn_flashprefill * fp =
            input ? input->get_fp_input() : nullptr;
        if (fp == nullptr || !fp->active() || fp->is_reserve()) {
            continue;
        }
        flashprefill_summary.visible_tokens       = fp->built_summary.visible_tokens;
        flashprefill_summary.expected_sparse_rows = fp->built_summary.expected_sparse_rows;
        flashprefill_summary.expected_forced_rows = fp->built_summary.expected_forced_rows;
        flashprefill_summary.ubatch_tokens        = fp->built_summary.ubatch_tokens;
        flashprefill_summary.n_rows    = fp->built_summary.n_rows;
        flashprefill_summary.n_uses    = fp->built_summary.n_uses;
        flashprefill_summary.n_cells   = fp->built_summary.n_cells;
        flashprefill_summary.n_groups  = fp->built_summary.n_groups;
        break; // single companion per graph
    }
}

void llm_graph_result::set_outputs(const llm_graph_params & params) {
    if (t_logits != nullptr) {
        ggml_set_output(t_logits);
    }
    if (t_embd != nullptr) {
        ggml_set_output(t_embd);
    }
    if (t_embd_pooled != nullptr) {
        ggml_set_output(t_embd_pooled);
    }
    if (t_h_nextn != nullptr) {
        ggml_set_output(t_h_nextn);
    }
    {
        const auto & embeddings_layer_inp = params.cparams.embeddings_layer_inp;
        for (size_t il = 0; il < embeddings_layer_inp.size(); ++il) {
            if (embeddings_layer_inp[il]) {
                GGML_ASSERT(t_layer_inp[il] != nullptr && "layer input tensor is null");
                ggml_set_output(t_layer_inp[il]);
            }
        }
    }
    {
        const auto & attention_q_pre_rope = params.cparams.attention_q_pre_rope;
        for (size_t il = 0; il < attention_q_pre_rope.size(); ++il) {
            if (attention_q_pre_rope[il]) {
                GGML_ASSERT(t_attn_q_pre_rope[il] != nullptr && "pre-RoPE Q tensor is null");
                ggml_set_output(t_attn_q_pre_rope[il]);
            }
        }
    }
    for (auto & [seq_id, t] : t_sampled) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_sampled_probs) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_sampled_logits) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_candidates) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }

    // FlashPrefill post-build consumption gate (GraphIntegration): an
    // active companion with zero recorded plans means no layer consumed it.
    // All-designed (designed_dense_layers covers every eligible full layer,
    // no capability dense, no sparse) records FULL_PREFIX authoritatively in
    // both modes. Anything else throws unsupported in REQUIRED here, before
    // execution — unwired architectures or unhandled capability fallbacks
    // must never surface as a Metrics-inferred NO_PLAN. AUTO records
    // UNSUPPORTED for the unconsumed case. Metrics must trust
    // dense_reason/designed_dense_layers/expected_* and never re-route from
    // physical KV size. Reserve companions never participate.
    {
        bool any_active = false;
        for (auto & input : inputs) {
            llm_graph_input_attn_flashprefill * fp =
                input ? input->get_fp_input() : nullptr;
            if (fp != nullptr && fp->active() && !fp->is_reserve()) {
                any_active = true;
                break;
            }
        }
        if (any_active && flashprefill_plans.empty()) {
            const int32_t n_full = llm_fp_count_full_layers(params.hparams);
            const bool all_designed =
                n_full > 0 &&
                flashprefill_summary.designed_dense_layers >= n_full &&
                flashprefill_summary.dense_layers == 0 &&
                flashprefill_summary.sparse_layers == 0;
            if (all_designed) {
                flashprefill_summary.dense_reason = LLM_FP_DENSE_FULL_PREFIX;
            } else if (params.cparams.flashprefill.mode == LLAMA_FLASHPREFILL_MODE_REQUIRED) {
                throw std::runtime_error(
                    "flashprefill: wanted companion but no handled layers/plans (required, unsupported)");
            } else {
                flashprefill_summary.dense_reason = LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED;
            }
        }
    }
}

bool llm_graph_result::can_reuse(const llm_graph_params & params) {
    if (!this->params.allow_reuse(params)) {
        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: cannot reuse graph due to incompatible graph parameters\n", __func__);
        }

        return false;
    }

    if (debug > 1) {
        LLAMA_LOG_DEBUG("%s: checking compatibility of %d inputs:\n", __func__, (int) inputs.size());
    }

    bool res = true;

    for (auto & input : inputs) {
        const bool cur = input->can_reuse(params);

        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: can_reuse = %d\n", "placeholder", cur);
        }

        res = res && cur;
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: can reuse graph = %d\n", __func__, res);
    }

    return res;
}

llm_graph_input_i * llm_graph_result::add_input(llm_graph_input_ptr input) {
    inputs.emplace_back(std::move(input));
    return inputs.back().get();
}

void llm_graph_result::add_fused_node(llm_graph_fused_node result) {
    fused_nodes.push_back(result);
}

void llm_graph_result::set_params(const llm_graph_params & params) {
    this->params = params;
}

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    arch             (params.arch),
    hparams          (params.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    n_embd           (hparams.n_embd),
    n_layer          (hparams.n_layer()),
    n_layer_nextn    (hparams.n_layer_nextn),
    n_rot            (hparams.n_rot()),
    n_ctx            (cparams.n_ctx),
    n_head           (hparams.n_head()),
    n_head_kv        (hparams.n_head_kv()),
    n_embd_head_k    (hparams.n_embd_head_k()),
    n_embd_k_gqa     (hparams.n_embd_k_gqa()),
    n_embd_head_v    (hparams.n_embd_head_v()),
    n_embd_v_gqa     (hparams.n_embd_v_gqa()),
    n_expert         (hparams.n_expert),
    n_expert_used    (cparams.warmup ? hparams.n_expert : hparams.n_expert_used),
    freq_base        (cparams.rope_freq_base),
    freq_scale       (cparams.rope_freq_scale),
    ext_factor       (cparams.yarn_ext_factor),
    attn_factor      (cparams.yarn_attn_factor),
    beta_fast        (cparams.yarn_beta_fast),
    beta_slow        (cparams.yarn_beta_slow),
    norm_eps         (hparams.f_norm_eps),
    norm_rms_eps     (hparams.f_norm_rms_eps),
    n_tokens         (ubatch.n_tokens),
    n_outputs        (params.n_outputs),
    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    pooling_type     (cparams.pooling_type),
    rope_type        (hparams.rope_type),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    cvec             (params.cvec),
    loras            (params.loras),
    mctx             (params.mctx),
    cross            (params.cross),
    fp_reserve_sizing(params.flashprefill_reserve_sizing),
    samplers         (params.samplers),
    cb_func          (params.cb),
    res              (params.res),
    ctx0             (res->get_ctx()),
    gf               (res->get_gf()) {
        res->set_params(params);
    }

void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il);
    }
}



ggml_tensor * llm_graph_context::build_cvec(
         ggml_tensor * cur,
                 int   il) const {
    return cvec->apply_to(ctx0, cur, il);
}

ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s,
        enum ggml_prec   prec) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    if (prec != GGML_PREC_DEFAULT) {
        // Set precision on the base MUL_MAT before an optional scale/LoRA attachment changes the root op.
        ggml_mul_mat_set_prec(res, prec);
    }

    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);
    }

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_lora_mm_id(
          ggml_tensor * w,   // ggml_tensor * as
          ggml_tensor * cur, // ggml_tensor * b
          ggml_tensor * ids,
          ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);

    if (w_s) {
        const int64_t n_expert = w_s->ne[0];
        const int64_t n_tokens = cur->ne[2];
        ggml_tensor * s = ggml_reshape_3d(ctx0, w_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx0, s, ids);
        res = ggml_mul(ctx0, res, s);
    }
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    switch (type) {
        case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
            } break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
}


llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il) const {
    const int64_t n_embd_q  = n_embd_head * n_head;
    const int64_t n_embd_kv = n_embd_head * n_head_kv;

    ggml_tensor * Qcur, * Kcur, * Vcur;

    if (layer.wqkv) {
        // fused QKV path
        ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
        cb(qkv, "wqkv", il);
        if (layer.wqkv_b) {
            qkv = ggml_add(ctx0, qkv, layer.wqkv_b);
            cb(qkv, "wqkv_b", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            qkv = ggml_clamp(ctx0, qkv, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(qkv, "wqkv_clamped", il);
        }
        Qcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head,    n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1], 0);
        Kcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],
            ggml_row_size(qkv->type, n_embd_q));
        Vcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],
            ggml_row_size(qkv->type, n_embd_q + n_embd_kv));
    } else {
        // separate Q/K/V path
        Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
        cb(Qcur, "Qcur", il);
        if (layer.wq_b) {
            Qcur = ggml_add(ctx0, Qcur, layer.wq_b);
            cb(Qcur, "Qcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Qcur = ggml_clamp(ctx0, Qcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Qcur, "Qcur_clamped", il);
        }
        Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        cb(Kcur, "Kcur", il);
        if (layer.wk_b) {
            Kcur = ggml_add(ctx0, Kcur, layer.wk_b);
            cb(Kcur, "Kcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Kcur = ggml_clamp(ctx0, Kcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Kcur, "Kcur_clamped", il);
        }
        Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        cb(Vcur, "Vcur", il);
        if (layer.wv_b) {
            Vcur = ggml_add(ctx0, Vcur, layer.wv_b);
            cb(Vcur, "Vcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Vcur = ggml_clamp(ctx0, Vcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Vcur, "Vcur_clamped", il);
        }
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    }

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    return { Qcur, Kcur, Vcur };
}


ggml_tensor * llm_graph_context::build_ffn(
         ggml_tensor * cur,
         ggml_tensor * up,
         ggml_tensor * up_b,
         ggml_tensor * up_s,
         ggml_tensor * gate,
         ggml_tensor * gate_b,
         ggml_tensor * gate_s,
         ggml_tensor * down,
         ggml_tensor * down_b,
         ggml_tensor * down_s,
         ggml_tensor * act_scales,
     llm_ffn_op_type   type_op,
   llm_ffn_gate_type   type_gate,
                 int   il) const {
    // NVFP4 support is currently restricted to
    // 1) LORA absence (*_s would be applied after LORA residual, which is incorrect)
    // 2) bias absense (*_s would be applied after bias addition, which is incorrect)
    // TODO: disambiguate LLM-architectural scales (which use *_s) from NVFP4 scale_2 (which also uses *_s currently)
    auto has_lora = [this](ggml_tensor * w) {
        if (!w) {
            return false;
        }
        for (const auto & lora : *loras) {
            if (lora.first->get_weight(w) != nullptr) {
                return true;
            }
        }
        return false;
    };

    GGML_ASSERT(!up_s   || !up_b   || !up   || up->type   != GGML_TYPE_NVFP4);
    GGML_ASSERT(!gate_s || !gate_b || !gate || gate->type != GGML_TYPE_NVFP4);
    GGML_ASSERT(!down_s || !down_b || !down || down->type != GGML_TYPE_NVFP4);
    GGML_ASSERT(!up_s   || !up   || up->type   != GGML_TYPE_NVFP4 || !has_lora(up));
    GGML_ASSERT(!gate_s || !gate || gate->type != GGML_TYPE_NVFP4 || !has_lora(gate));
    GGML_ASSERT(!down_s || !down || down->type != GGML_TYPE_NVFP4 || !has_lora(down));

    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx0, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx0, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                if (il >= 0) {
                    const float limit = hparams.swiglu_clamp_shexp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        tmp = ggml_clamp(ctx0, tmp, -limit, limit);
                        cb(tmp, "ffn_up_clamped", il);

                        if (arch == LLM_ARCH_DEEPSEEK4 || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0)) {
                            cur = ggml_clamp(ctx0, cur, -INFINITY, limit);
                            cb(cur, "ffn_gate_clamped", il);
                            cur = ggml_swiglu_split(ctx0, cur, tmp);
                        } else {
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, tmp);
                        }
                        cb(cur, "ffn_swiglu_limited", il);
                        type_gate = LLM_FFN_SEQ;
                        break;
                    }
                }

                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_SITU:
            {
                // Kimi K3 SiTU-GLU: [beta*tanh(gate/beta)*sigmoid(gate)] * [linear_beta*tanh(up/linear_beta)]
                const float beta  = hparams.situ_beta;
                const float lbeta = hparams.situ_linear_beta;
                GGML_ASSERT(beta > 0.0f && lbeta > 0.0f);

                ggml_tensor * gate_act = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/beta)), beta);
                gate_act = ggml_mul(ctx0, gate_act, ggml_sigmoid(ctx0, cur));
                cb(gate_act, "ffn_situ", il);

                if (gate && type_gate == LLM_FFN_PAR) {
                    ggml_tensor * up_cap = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, tmp, 1.0f/lbeta)), lbeta);
                    cur = ggml_mul(ctx0, gate_act, up_cap);
                    cb(cur, "ffn_situ_glu", il);
                    type_gate = LLM_FFN_SEQ;
                } else {
                    cur = gate_act;
                }
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_reglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx0, cur);
                cb(cur, "ffn_swiglu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            if (gate && type_gate == LLM_FFN_PAR) {
                // same alpha/limit constants as gpt-oss
                const float alpha = 1.702f;
                const float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, tmp, alpha, limit);
                cb(cur, "ffn_swiglu_oai", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                GGML_ABORT("LLM_FFN_SWIGLU_OAI_MOE requires a parallel gate");
            } break;
        case LLM_FFN_GEGLU:
            {
                cur = ggml_geglu(ctx0, cur);
                cb(cur, "ffn_geglu", il);
            } break;
        case LLM_FFN_REGLU:
            {
                cur = ggml_reglu(ctx0, cur);
                cb(cur, "ffn_reglu", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = build_lora_mm(down, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx0, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
         llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {
    return build_moe_ffn(
        cur,
        gate_inp,  /* gate_inp_b  */ nullptr,
        up_exps,   /* up_exps_b   */ nullptr,
        gate_exps, /* gate_exps_b */ nullptr,
        down_exps, /* down_exps_b */ nullptr,
        exp_probs_b,
        n_expert,
        n_expert_used,
        type_op,
        norm_w,
        w_scale,
        gating_op,
        il,
        probs_in,
        gate_up_exps,
        /* gate_up_exps_b */ nullptr,
        up_exps_s,
        gate_exps_s,
        down_exps_s,
        selected_experts_in
    );
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * gate_up_exps_b,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in) const {
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    ggml_tensor * logits = nullptr;

    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
            ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
        }
        cb(logits, "ffn_moe_logits", il);
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
            {
                probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, logits)); // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }

    // select experts
    ggml_tensor * selected_experts = selected_experts_in;
    if (selected_experts == nullptr) {
        selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
        cb(selected_experts->src[0], "ffn_moe_argsort", il);
    }
    cb(selected_experts, "ffn_moe_topk", il);

    if (arch == LLM_ARCH_GROVEMOE && n_expert != hparams.n_expert) {
        // TODO: Use scalar div instead when/if implemented
        ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(hparams.n_group_experts)), GGML_TYPE_I32);
        probs = ggml_reshape_3d(ctx0, probs, 1, hparams.n_expert, n_tokens);
    } else {
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    }

    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);


    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        // Avoid division by zero, clamp to smallest number representable by F16
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (w_scale != 0.0f && w_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    //call early so that topk-moe can be used
    ggml_build_forward_expand(gf, weights);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    if (weight_before_ffn) {
        // repeat cur to [n_embd, n_expert_used, n_tokens]
        ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
        cur = ggml_mul(ctx0, repeated, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    ggml_tensor * up = nullptr;
    ggml_tensor * experts = nullptr;

    if (gate_up_exps) {
        // merged gate_up path: one mul_mat_id, then split into gate and up views
        ggml_tensor * gate_up = build_lora_mm_id(gate_up_exps, cur, selected_experts, up_exps_s); // [n_ff*2, n_expert_used, n_tokens]
        cb(gate_up, "ffn_moe_gate_up", il);

        if (up_exps_s) {
            cb(gate_up, "ffn_moe_gate_up_scaled", il);
        }

        if (gate_up_exps_b) {
            gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, selected_experts);
            cb(gate_up, "ffn_moe_gate_up_biased", il);
        }

        const int64_t n_ff = gate_up->ne[0] / 2;
        cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        cb(cur, "ffn_moe_gate", il);
        up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
        cb(up, "ffn_moe_up", il);
    } else {
        // separate gate and up path
        up = build_lora_mm_id(up_exps, cur, selected_experts, up_exps_s); // [n_ff, n_expert_used, n_tokens]
        cb(up, "ffn_moe_up", il);

        if (up_exps_s) {
            cb(up, "ffn_moe_up_scaled", il);
        }

        if (up_exps_b) {
            up = ggml_add_id(ctx0, up, up_exps_b, selected_experts);
            cb(up, "ffn_moe_up_biased", il);
        }

        if (gate_exps) {
            cur = build_lora_mm_id(gate_exps, cur, selected_experts, gate_exps_s); // [n_ff, n_expert_used, n_tokens]
            cb(cur, "ffn_moe_gate", il);
        } else {
            cur = up;
        }

        if (gate_exps_s) {
            cb(cur, "ffn_moe_gate_scaled", il);
        }

        if (gate_exps_b) {
            cur = ggml_add_id(ctx0, cur, gate_exps_b, selected_experts);
            cb(cur, "ffn_moe_gate_biased", il);
        }
    }

    const bool has_gate = gate_exps || gate_up_exps;

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate_exps) {
                if (il >= 0) {
                    const float limit = hparams.swiglu_clamp_exp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        up = ggml_clamp(ctx0, up, -limit, limit);
                        cb(up, "ffn_moe_up_clamped", il);

                        if (arch == LLM_ARCH_DEEPSEEK4 || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0)) {
                            cur = ggml_clamp(ctx0, cur, -INFINITY, limit);
                            cb(cur, "ffn_moe_gate_clamped", il);
                            cur = ggml_swiglu_split(ctx0, cur, up);
                        } else {
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_moe_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_moe_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, up);
                        }
                        cb(cur, "ffn_moe_swiglu_limited", il);
                        break;
                    }
                }
            }

            if (has_gate) {
                cur = ggml_swiglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_moe_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (has_gate) {
                cur = ggml_geglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_moe_gelu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            {
                // TODO: move to hparams?
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                cb(cur, "ffn_moe_swiglu_oai", il);
            } break;
        case LLM_FFN_SITU:
            {
                // Kimi K3 SiTU-GLU: [beta*tanh(gate/beta)*sigmoid(gate)] * [linear_beta*tanh(up/linear_beta)]
                const float beta  = hparams.situ_beta;
                const float lbeta = hparams.situ_linear_beta;
                GGML_ASSERT(beta > 0.0f && lbeta > 0.0f);
                GGML_ASSERT(has_gate && "SiTU without gate branch not implemented");

                ggml_tensor * gate_act = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/beta)), beta);
                gate_act = ggml_mul(ctx0, gate_act, ggml_sigmoid(ctx0, cur));
                cb(gate_act, "ffn_moe_situ", il);

                ggml_tensor * up_cap = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, up, 1.0f/lbeta)), lbeta);
                cur = ggml_mul(ctx0, gate_act, up_cap);
                cb(cur, "ffn_moe_situ_glu", il);
            } break;
        case LLM_FFN_RELU:
            if (has_gate) {
                cur = ggml_reglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_reglu", il);
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_moe_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            if (has_gate) {
                // TODO: add support for gated squared relu
                GGML_ABORT("fatal error: gated squared relu not implemented");
            } else {
                cur = ggml_relu(ctx0, cur);
                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_moe_relu_sqr", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    experts = build_lora_mm_id(down_exps, cur, selected_experts, down_exps_s); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);

    if (down_exps_s) {
        cb(experts, "ffn_moe_down_scaled", il);
    }

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }

    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_weighted", il);
    }

    ggml_build_forward_expand(gf, experts);

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

    assert(n_expert_used > 0);

    // order the views before the adds
    for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);

        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    // aggregate experts
    // note: here we explicitly use hparams.n_expert_used instead of n_expert_used
    //       to avoid potentially a large number of add nodes during warmup
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14753
    ggml_tensor * moe_out = cur_experts[0];

    for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);

        ggml_build_forward_expand(gf, moe_out);
    }

    if (hparams.n_expert_used == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx0, moe_out);
    }

    cb(moe_out, "ffn_moe_out", il);

    return moe_out;
}

// input embeddings with optional lora
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd) const {
    const int64_t n_embd_inp = hparams.n_embd_inp();
    const int64_t n_embd     = hparams.n_embd;

    assert(n_embd_inp >= n_embd);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
    cb(inp->tokens, "inp_tokens", -1);
    ggml_set_input(inp->tokens);
    res->t_inp_tokens = inp->tokens;

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
    cb(inp->embd, "inp_embd", -1);
    ggml_set_input(inp->embd);

    // select one of the 2 inputs, based on the batch contents
    // ref: https://github.com/ggml-org/llama.cpp/pull/18550
    std::array<ggml_tensor *, 2> inps;

    // token embeddings path (ubatch.token != nullptr)
    {
        auto & cur = inps[0];

        cur = ggml_get_rows(ctx0, tok_embd, inp->tokens);

        // apply lora for embedding tokens if needed
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b, // non-transposed lora_b
                        ggml_get_rows(ctx0, lw->a, inp->tokens)
                        ), scale);

            cur = ggml_add(ctx0, cur, inpL_delta);
        }

        if (n_embd_inp != n_embd) {
            cur = ggml_pad(ctx0, cur, hparams.n_embd_inp() - n_embd, 0, 0, 0);
        }
    }

    // vector embeddings path (ubatch.embd != nullptr)
    {
        auto & cur = inps[1];

        cur = inp->embd;
    }

    assert(ggml_are_same_shape (inps[0], inps[1]));
    assert(ggml_are_same_stride(inps[0], inps[1]));

    ggml_tensor * cur = ggml_build_forward_select(gf, inps.data(), inps.size(), ubatch.token ? 0 : 1);

    if (n_embd_inp != n_embd) {
        cur = ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0);
    }

    res->t_inp_embd = cur;

    // For Granite architecture
    // NOTE: For deepstack models, only apply scale to token inputs (ie text-only input).
    //  Raw embeddings are assumed to be multimodal inputs that should not be scaled.
    if (hparams.f_embedding_scale != 0.0f && (ubatch.token || hparams.n_deepstack_layers == 0)) {
        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }

    cb(cur, "embd", -1);

    res->add_input(std::move(inp));

    // make sure the produced embeddings are immediately materialized in the ggml graph
    // ref: https://github.com/ggml-org/llama.cpp/pull/18599
    ggml_build_forward_expand(gf, cur);

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos() const {
    auto inp = std::make_unique<llm_graph_input_pos>(hparams.n_pos_per_embd());

    auto & cur = inp->pos;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_attn_scale() const {
    auto inp = std::make_unique<llm_graph_input_attn_temp>(hparams.n_attn_temp_floor_scale, hparams.f_attn_temp_scale, hparams.f_attn_temp_offset);

    auto & cur = inp->attn_scale;

    // this need to be 1x1xN for broadcasting
    cur = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
    ggml_set_input(cur);
    ggml_set_name(cur, "attn_scale");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_out_ids() const {
    // note: when all tokens are output, we could skip this optimization to spare the ggml_get_rows() calls,
    //       but this would make the graph topology depend on the number of output tokens, which can interfere with
    //       features that require constant topology such as pipeline parallelism
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14275#issuecomment-2987424471
    //if (n_outputs < n_tokens) {
    //    return nullptr;
    //}

    auto inp = std::make_unique<llm_graph_input_out_ids>(hparams, cparams, n_outputs);

    auto & cur = inp->out_ids;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_mean() const {
    auto inp = std::make_unique<llm_graph_input_mean>(cparams);

    auto & cur = inp->mean;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cls() const {
    auto inp = std::make_unique<llm_graph_input_cls>(cparams, arch);

    auto & cur = inp->cls;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cross_embd() const {
    auto inp = std::make_unique<llm_graph_input_cross_embd>(cross);

    auto & cur = inp->cross_embd;

    // if we have the output embeddings from the encoder, use them directly
    // TODO: needs more work to be correct, for now just use the tensor shape
    //if (cross->t_embd) {
    //    cur = ggml_view_tensor(ctx0, cross->t_embd);

    //    return cur;
    //}

    const auto n_embd = !cross->v_embd.empty() ? cross->n_embd : hparams.n_embd_inp();
    const auto n_enc  = !cross->v_embd.empty() ? cross->n_enc  : hparams.n_ctx_train;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_enc);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_enc() const {
    auto inp = std::make_unique<llm_graph_input_pos_bucket>(hparams);

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_dec() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_pos_bucket_kv>(hparams, mctx_cur);

    const auto n_kv = mctx_cur->get_n_kv();

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const {
    ggml_tensor * pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1]);
    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);

    pos_bias = ggml_reshape_3d(ctx0, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1]);
    pos_bias = ggml_permute   (ctx0, pos_bias, 2, 0, 1, 3);
    pos_bias = ggml_cont      (ctx0, pos_bias);

    cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
               float   kq_scale,
                 int   il) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    // split the batch into streams if needed
    const auto n_stream = k->ne[3];

    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    // TurboQuant note: graph-side Q rotation (pre-rotate-queries) is implemented below
    // in the flash-attn path. The VEC kernel bug (wrong Q/K stride in
    // vec_dot_fattn_vec_KQ_turbo3_0) was fixed in fattn-common.cuh to match f16 pattern.

    ggml_tensor * cur;

    const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
    if (use_flash_attn) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});

        ggml_flash_attn_ext_add_sinks(cur, sinks);
        ggml_flash_attn_ext_set_prec (cur, GGML_PREC_F32);

        // TurboQuant: inverse WHT on FA output when V values are WHT-rotated.
        // For MLA, V is a view of K with different ne[0] (e.g. V=512, K=576).
        // Group size must come from K (which determines the WHT rotation), not V.
        if (v->type == GGML_TYPE_TURBO3_0 || v->type == GGML_TYPE_TURBO4_0 || v->type == GGML_TYPE_TURBO2_0) {
            const bool k_is_turbo = (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0 || k->type == GGML_TYPE_TURBO2_0);
            const ggml_tensor * group_src = k_is_turbo ? k : v;
            const int turbo_group = (group_src->ne[0] % 128 == 0) ? 128 : 64;
            if (cur->ne[0] % turbo_group == 0) {
                if (!ggml_is_contiguous(cur)) { cur = ggml_cont(ctx0, cur); }
                ggml_tensor * innerq_scale = mctx ? mctx->get_turbo_innerq_scale_inv() : nullptr;
                cur = ggml_turbo_wht(ctx0, cur, 1, turbo_group, innerq_scale);  // 1 = inverse
            }
        }

        if (v_mla) {
#if 0
            // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
            // However, the code is optimized for dimensions 0 and 1 being large, so this is inefficient.
            cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
#else
            // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
            // The permutations are noops and only change how the tensor data is interpreted.
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
            cb(cur, "fattn_mla", il);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "kq", il);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        if (arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplier
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping));
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled", il);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_1", il);
            kq = ggml_tanh (ctx0, kq);
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_2", il);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
            cb(kq, "kq_plus_kq_b", il);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il);

        if (!v_trans) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cb(kqv, "kqv", il);

        // TurboQuant: inverse WHT on attention output (non-FA path)
        if (v->type == GGML_TYPE_TURBO3_0 || v->type == GGML_TYPE_TURBO4_0 || v->type == GGML_TYPE_TURBO2_0) {
            const bool k_is_turbo = (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0 || k->type == GGML_TYPE_TURBO2_0);
            const ggml_tensor * group_src = k_is_turbo ? k : v;
            const int turbo_group = (group_src->ne[0] % 128 == 0) ? 128 : 64;
            if (kqv->ne[0] % turbo_group == 0) {
                if (!ggml_is_contiguous(kqv)) { kqv = ggml_cont(ctx0, kqv); }
                ggml_tensor * innerq_scale = mctx ? mctx->get_turbo_innerq_scale_inv() : nullptr;
                kqv = ggml_turbo_wht(ctx0, kqv, 1, turbo_group, innerq_scale);
            }
        }

        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
            cb(kqv, "kqv_mla", il);
        }

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        // recombine streams
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }

    // TurboQuant: graph-side inverse WHT on attention output (undoes V rotation)

    ggml_build_forward_expand(gf, cur);

    return cur;
}

llm_graph_input_attn_no_cache * llm_graph_context::build_attn_inp_no_cache() const {
    auto inp = std::make_unique<llm_graph_input_attn_no_cache>(hparams, cparams);

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    // note: there is no KV cache, so the number of KV values is equal to the number of tokens in the batch
    inp->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
    ggml_set_input(inp->self_kq_mask);

    inp->self_kq_mask_cnv = inp->self_kq_mask;

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
        ggml_set_input(inp->self_kq_mask_swa);

        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    } else {
        inp->self_kq_mask_swa     = nullptr;
        inp->self_kq_mask_swa_cnv = nullptr;
    }

    return (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_no_cache * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    GGML_UNUSED(n_tokens);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const bool is_swa = hparams.is_swa(il);

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // [TAG_NO_CACHE_PAD]
    // TODO: if ubatch.equal_seqs() == true, we can split the three tensors below into ubatch.n_seqs_unq streams
    //       but it might not be worth it: https://github.com/ggml-org/llama.cpp/pull/15636
    //assert(!ubatch.equal_seqs() || (k_cur->ne[3] == 1 && k_cur->ne[3] == ubatch.n_seqs_unq));

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur,
          ggml_backend_sched_t sched,
                     bool fp_reserve_sizing,
          llm_graph_result * res) {

    auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    inp->self_k_rot = mctx_cur->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->build_input_v_rot(ctx0);

    // FlashPrefill sparse-path companion FIRST (built once here, shared by
    // every full-attention layer). Inactive (null) unless the policy is
    // enabled with routable per-ubatch rows, or for reserve sizing
    // (worst-case caps, zero counts, never runs). OFF costs nothing.
    inp->fp = llm_graph_input_attn_flashprefill::build_if_wanted(
        ctx0, ubatch, hparams, cparams, mctx_cur, sched, fp_reserve_sizing, res);

    // Legacy RERoT DDVR spans LAZILY: only when an actual dense RERoT layer
    // needs them. Skipped solely for committed sparse-rerot (rerot layout +
    // rerot cache state + full_attn_layers==0), where the hook routes sparse
    // or heals via the lazy ensure in build_rerot_q_groups. Every other
    // combination (ordinary layouts even under rerot state, fp-null
    // fallbacks, full-N prefixes) keeps the correct legacy path, so
    // span-hooked architectures outside the sparse hook keep DDVR intact.
    // Reserve may over-cover spans (safe direction, single charge).
    const bool fp_covers_all_rerot = (inp->fp != nullptr) && !inp->fp->is_reserve() &&
        cparams.flashprefill.full_attn_layers == 0 &&
        inp->fp->reuse_key().is_rerot && mctx_cur->rerot_active();
    if (mctx_cur->rerot_active() && !fp_covers_all_rerot) {
        // RERoT DDVR span descriptors: capacity-bucketed topology. The reuse
        // key (n_token rows, span capacity bucket, kernel variant, on/off,
        // frontier mode) is recorded inside; per-frontier PAC-DFS churn lands
        // in tensor data via fill_spans, not in topology.
        inp->rerot_spans.build_span_tensors(
            ctx0,
            inp->self_rerot_q_indices, inp->self_rerot_q_pos,
            inp->self_rerot_entries, inp->self_rerot_offsets,
            ubatch, hparams, cparams, mctx_cur, sched);
    }

    return inp;
}

llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur, sched, fp_reserve_sizing, res);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rerot_q_groups(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * q_raw,
        ggml_tensor * freq_factors,
        int sections[GGML_MROPE_SECTIONS],
        int il) const {
    GGML_ASSERT(inp != nullptr);
    if (inp->get_rerot_q_indices() == nullptr) {
        // Lazy legacy spans for an actual dense RERoT fallback: the input
        // builder skipped them (fp active, full_attn_layers==0) but an AUTO
        // fallback needs them now. The sparse branch never calls here, so
        // O(QK) entry lists materialize only on the real fallback path;
        // prebuilt inputs (full prefix, fp-null fallbacks) skip this as a
        // cheap null-check no-op. Null mctx falls through to the loud
        // asserts below (never silent stock attention).
        const auto * attn = inp->mctx;
        if (attn != nullptr) {
            inp->rerot_spans.build_span_tensors(
                ctx0,
                inp->self_rerot_q_indices, inp->self_rerot_q_pos,
                inp->self_rerot_entries, inp->self_rerot_offsets,
                ubatch, hparams, cparams, attn, sched);
        }
    }
    GGML_ASSERT(inp && inp->rerot_active());
    GGML_ASSERT(q_raw && q_raw->ne[2] == n_tokens);

    // DDVR Q pre-phasing gate: per-group effective RoPE (query_virtual +
    // storage_base - virtual_base; IMRoPE text delta on the first three
    // coords, fourth pinned 0 by fill_spans) is applied here, BEFORE Turbo
    // WHT (see build_attn_rerot). K storage is never moved. Unsupported
    // backends throw below, never silent stock attention.
    llm_graph_input_attn_rerot::require_supported(sched, hparams, "build_rerot_q_groups");

    if (!ggml_is_contiguous(q_raw)) {
        q_raw = ggml_cont(ctx0, q_raw);
    }

    const int64_t head_dim = q_raw->ne[0];
    const int64_t heads = q_raw->ne[1];
    const int64_t groups = inp->get_rerot_q_indices()->ne[0];

    ggml_tensor * q_flat = ggml_reshape_2d(ctx0, q_raw, head_dim * heads, n_tokens);
    ggml_tensor * q_grouped = ggml_get_rows(ctx0, q_flat, inp->get_rerot_q_indices());
    q_grouped = ggml_reshape_3d(ctx0, q_grouped, head_dim, heads, groups);
    cb(q_grouped, "rerot_q_grouped_raw", il);

    const int mode = static_cast<int>(rope_type);
    if (mode == GGML_ROPE_TYPE_MROPE || mode == GGML_ROPE_TYPE_IMROPE || mode == GGML_ROPE_TYPE_VISION) {
        GGML_ASSERT(sections != nullptr);
        q_grouped = ggml_rope_multi(
            ctx0, q_grouped, inp->get_rerot_q_pos(), freq_factors,
            n_rot, sections, mode, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    } else {
        q_grouped = ggml_rope_ext(
            ctx0, q_grouped, inp->get_rerot_q_pos(), freq_factors,
            n_rot, mode, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    }
    cb(q_grouped, "rerot_q_grouped", il);
    return q_grouped;
}

ggml_tensor * llm_graph_context::build_attn_rerot(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_groups,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    GGML_ASSERT(inp && inp->rerot_active());
    GGML_ASSERT(q_groups && k_cur && v_cur);

    // Capability gate: CUDA/Metal/RPC have no RERoT kernel today and throw
    // here instead of silently running stock attention. Downstream
    // GGML_OP_FLASH_ATTN_EXT_REROT keeps one global online softmax per query
    // entry range over the pre-filtered span descriptors.
    llm_graph_input_attn_rerot::require_supported(sched, hparams, "build_attn_rerot");

    if (inp->self_k_rot) {
        q_groups = llama_mul_mat_hadamard(ctx0, q_groups, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }
    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    ggml_build_forward_expand(gf, q_groups);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;
    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, inp->get_k_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, inp->get_v_idxs(), il));

    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);
    ggml_tensor * q = q_groups;

    // TurboQuant caches hold WHT-domain K/V. Transform each RERoT query group
    // once, after reader-relative RoPE, so all keys in that group reuse it.
    if (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0 || k->type == GGML_TYPE_TURBO2_0) {
        if (q->ne[0] % 128 != 0) {
            const int64_t pad = ((q->ne[0] + 127) / 128) * 128 - q->ne[0];
            q = ggml_pad(ctx0, q, pad, 0, 0, 0);
        }
        if (!ggml_is_contiguous(q)) {
            q = ggml_cont(ctx0, q);
        }
        q = ggml_turbo_wht(ctx0, q, 0, 0, mctx_cur->get_turbo_innerq_scale_inv());
    }

    const bool v_trans = v->nb[1] > v->nb[2];
    q = ggml_permute(ctx0, q, 0, 2, 1, 3); // [D, group, head, 1]
    k = ggml_permute(ctx0, k, 0, 2, 1, 3); // [D, key,   head, 1]
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);
    if (v_trans) {
        v = ggml_transpose(ctx0, v);
    }

    ggml_tensor * cur = ggml_flash_attn_ext_rerot(
        ctx0, q, k, v,
        inp->get_rerot_entries(), inp->get_rerot_offsets(), sinks,
        kq_scale, hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
    ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
    cb(cur, "rerot_indexed_attn", il);

    if (v->type == GGML_TYPE_TURBO3_0 || v->type == GGML_TYPE_TURBO4_0 || v->type == GGML_TYPE_TURBO2_0) {
        const bool k_is_turbo = k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0 || k->type == GGML_TYPE_TURBO2_0;
        const ggml_tensor * group_src = k_is_turbo ? k : v;
        const int turbo_group = (group_src->ne[0] % 128 == 0) ? 128 : 64;
        if (cur->ne[0] % turbo_group == 0) {
            if (!ggml_is_contiguous(cur)) {
                cur = ggml_cont(ctx0, cur);
            }
            cur = ggml_turbo_wht(ctx0, cur, 1, turbo_group, mctx_cur->get_turbo_innerq_scale_inv());
        }
    }

    const int64_t padded_v_head = v->ne[0];
    const int64_t orig_v_head = hparams.n_embd_head_v(il);
    const int64_t n_queries = inp->get_rerot_offsets()->ne[0] - 1;
    if (padded_v_head != orig_v_head) {
        const int64_t n_head_v = hparams.n_head(il);
        cur = ggml_reshape_3d(ctx0, cur, padded_v_head, n_head_v, n_queries);
        cur = ggml_view_3d(ctx0, cur, orig_v_head, n_head_v, n_queries,
                           cur->nb[1], cur->nb[2], 0);
        cur = ggml_cont(ctx0, cur);
    }
    cur = ggml_reshape_2d(ctx0, cur, orig_v_head * hparams.n_head(il), n_queries);

    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }
    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }
    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    if (!cparams.offload_kqv) {
        ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
    }
    ggml_build_forward_expand(gf, cur);
    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla, // TODO: remove
            float     kq_scale,
            int       il) const {
    GGML_ASSERT(v_mla == nullptr);

    // FlashPrefill generic ordinary entry (guide-required): every common-KV
    // caller reaches the sparse pool/select/attn ops here, not only hooked
    // architectures, so required mode can never go silent through an
    // unhooked path. Non-RERoT semantic views only, with the model-roped Q
    // consumed directly (q_raw=null; never re-roped). RERoT batches keep
    // their dedicated raw-Q hooks. Dense old body runs only on nullptr;
    // both converge on the common projection tail below (GLM4/JAIS2 intact).
    ggml_tensor * cur = nullptr;
    if (inp != nullptr && !inp->rerot_semantic()) {
        cur = try_build_attn_flashprefill(inp,
                nullptr, q_cur, k_cur, v_cur, nullptr, kq_b, sinks, kq_scale, il);
    }
    if (cur == nullptr) {

    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    // TurboQuant pre-rotate-queries: O(d log d) WHT rotation via custom op
    // Q shape: (n_embd_head, n_head, n_tokens)
    // For zero-padded models (head_dim not 128-aligned), pad Q to match padded K dim first.
    if (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0 || k->type == GGML_TYPE_TURBO2_0) {
        // Pad Q per-head to next multiple of 128 if needed
        if (q->ne[0] % 128 != 0) {
            const int64_t pad = ((q->ne[0] + 127) / 128) * 128 - q->ne[0];
            q = ggml_pad(ctx0, q, pad, 0, 0, 0);
        }
        if (!ggml_is_contiguous(q)) { q = ggml_cont(ctx0, q); }
        ggml_tensor * innerq_scale = mctx_cur->get_turbo_innerq_scale_inv();
        q = ggml_turbo_wht(ctx0, q, 0, 0, innerq_scale);  // 0 = forward, 0 = auto group size from q->ne[0]
    }

    cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    // TurboQuant: if V was padded, the output has padded dimensions.
    // Extract original V head_dim after inverse WHT (applied inside build_attn_mha).
    // NOTE: gate on v->type (not k->type) for asymmetric configs where K=q8_0 but V=turbo
    if (v->type == GGML_TYPE_TURBO3_0 || v->type == GGML_TYPE_TURBO4_0 || v->type == GGML_TYPE_TURBO2_0) {
        const int64_t orig_v_head = hparams.n_embd_head_v(il);
        // cur is 2D: (n_embd_head * n_head, n_tokens) after build_attn_mha
        const int64_t padded_v_head = v->ne[0];
        if (padded_v_head != orig_v_head) {
            // Reshape to 4D, extract original head_dim, reshape back to 2D
            // Fix #78 (bingh0): cur shape post-MHA is (n_embd_head * n_head, n_tokens),
            // not (n_embd_head * n_head_kv, n_tokens). Reshape needs n_head
            // (Q-head count) so GQA models with n_head != n_head_kv (e.g.
            // Qwen2.5-0.5B head_dim=64 padded → 128) don't fail the element
            // count check in ggml_reshape_3d.
            const int64_t n_head_v = hparams.n_head(il);
            const int64_t n_tokens_cur = cur->ne[1];
            cur = ggml_reshape_3d(ctx0, cur, padded_v_head, n_head_v, n_tokens_cur);
            // ggml_view_3d to extract first orig_v_head elements per head
            cur = ggml_view_3d(ctx0, cur, orig_v_head, n_head_v, n_tokens_cur,
                               cur->nb[1], cur->nb[2], 0);
            cur = ggml_cont(ctx0, cur);
            cur = ggml_reshape_2d(ctx0, cur, orig_v_head * n_head_v, n_tokens_cur);
        }
    }

    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    } // end dense old body (skipped when the sparse route above produced cur)

    // Common projection tail (both routes, preserved EXACTLY): GLM4/JAIS2
    // F32 special projection, then wo_b. try_build returns finished pregate
    // (inverse/trim already applied), so no duplicate tail work here.
    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_k> build_attn_inp_k_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    return inp;
}

llm_graph_input_attn_k * llm_graph_context::build_attn_inp_k() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    // TurboQuant: pre-rotate Q for K-only (MLA) attention
    // For zero-padded models, pad Q to match padded K dim first.
    if (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0 || k->type == GGML_TYPE_TURBO2_0) {
        // Pad Q per-head to next multiple of 128 if needed
        if (q->ne[0] % 128 != 0) {
            const int64_t pad = ((q->ne[0] + 127) / 128) * 128 - q->ne[0];
            q = ggml_pad(ctx0, q, pad, 0, 0, 0);
        }
        if (!ggml_is_contiguous(q)) { q = ggml_cont(ctx0, q); }
        ggml_tensor * innerq_scale = mctx_cur->get_turbo_innerq_scale_inv();
        q = ggml_turbo_wht(ctx0, q, 0, 0, innerq_scale);  // 0 = forward, 0 = auto group size
    }

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    // TurboQuant: if V was padded (MLA: V is view of K, may have padded dim),
    // extract original V head_dim after inverse WHT.
    if (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0 || k->type == GGML_TYPE_TURBO2_0) {
        const int64_t orig_v_head = v_cur->ne[0];  // original V head_dim from model
        const int64_t padded_v_head = v->ne[0];     // padded V head_dim in cache
        if (padded_v_head != orig_v_head) {
            // cur is 2D: (padded_v_head * n_head, n_tokens) after build_attn_mha
            // Fix #78 (bingh0): cur shape post-MHA is (n_embd_head * n_head, n_tokens),
            // not (n_embd_head * n_head_kv, n_tokens). Reshape needs n_head
            // (Q-head count) so GQA models with n_head != n_head_kv (e.g.
            // Qwen2.5-0.5B head_dim=64 padded → 128) don't fail the element
            // count check in ggml_reshape_3d.
            const int64_t n_head_v = hparams.n_head(il);
            const int64_t n_tokens_cur = cur->ne[1];
            cur = ggml_reshape_3d(ctx0, cur, padded_v_head, n_head_v, n_tokens_cur);
            cur = ggml_view_3d(ctx0, cur, orig_v_head, n_head_v, n_tokens_cur,
                               cur->nb[1], cur->nb[2], 0);
            cur = ggml_cont(ctx0, cur);
            cur = ggml_reshape_2d(ctx0, cur, orig_v_head * n_head_v, n_tokens_cur);
        }
    }

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_dsa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
        ggml_tensor * top_k,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx->get_mla();

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs_mla();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask_mla();

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask_top_k, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;
    auto * v_rot = is_swa ? inp->self_v_rot_swa : inp->self_v_rot;

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }
    if (v_rot) {
        if (v_cur) {
            v_cur = llama_mul_mat_hadamard(ctx0, v_cur, v_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    const auto * mctx_iswa = inp->mctx;

    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    if (v_cur) {
        const auto & v_idxs = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    // TurboQuant: pre-rotate Q for ISWA attention (pad to 128-aligned if needed)
    if (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0 || k->type == GGML_TYPE_TURBO2_0) {
        if (q->ne[0] % 128 != 0) {
            const int64_t pad = ((q->ne[0] + 127) / 128) * 128 - q->ne[0];
            q = ggml_pad(ctx0, q, pad, 0, 0, 0);
        }
        if (!ggml_is_contiguous(q)) { q = ggml_cont(ctx0, q); }
        ggml_tensor * innerq_scale = mctx_cur->get_turbo_innerq_scale_inv();
        q = ggml_turbo_wht(ctx0, q, 0, 0, innerq_scale);
    }

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    // TurboQuant: if V was padded, extract original V head_dim after inverse WHT
    // NOTE: gate on v->type (not k->type) for asymmetric configs where K=q8_0 but V=turbo
    if (v->type == GGML_TYPE_TURBO3_0 || v->type == GGML_TYPE_TURBO4_0 || v->type == GGML_TYPE_TURBO2_0) {
        const int64_t orig_v_head = hparams.n_embd_head_v(il);
        const int64_t padded_v_head = v->ne[0];
        if (padded_v_head != orig_v_head) {
            // Fix #78 (bingh0): cur shape post-MHA is (n_embd_head * n_head, n_tokens),
            // not (n_embd_head * n_head_kv, n_tokens). Reshape needs n_head
            // (Q-head count) so GQA models with n_head != n_head_kv (e.g.
            // Qwen2.5-0.5B head_dim=64 padded → 128) don't fail the element
            // count check in ggml_reshape_3d.
            const int64_t n_head_v = hparams.n_head(il);
            const int64_t n_tokens_cur = cur->ne[1];
            cur = ggml_reshape_3d(ctx0, cur, padded_v_head, n_head_v, n_tokens_cur);
            cur = ggml_view_3d(ctx0, cur, orig_v_head, n_head_v, n_tokens_cur,
                               cur->nb[1], cur->nb[2], 0);
            cur = ggml_cont(ctx0, cur);
            cur = ggml_reshape_2d(ctx0, cur, orig_v_head * n_head_v, n_tokens_cur);
        }
    }

    if (v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    GGML_UNUSED(v_cur);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    const auto * mctx_iswa = inp->mctx;
    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // MLA-style attention: the cached K is used as V
    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = k;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (k_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, k_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_cross * llm_graph_context::build_attn_inp_cross() const {
    auto inp = std::make_unique<llm_graph_input_attn_cross>(cross);

    const int32_t n_enc = !cross->v_embd.empty() ? cross->n_enc : hparams.n_ctx_train;

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    inp->cross_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_enc, n_tokens, 1, 1);
    ggml_set_input(inp->cross_kq_mask);

    inp->cross_kq_mask_cnv = inp->cross_kq_mask;

    return (llm_graph_input_attn_cross *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_cross * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto & kq_mask = inp->get_kq_mask_cross();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_k_dsa * llm_graph_context::build_attn_inp_k_dsa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_k_dsa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs_mla = mctx_cur->get_mla()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_mla = build_attn_inp_kq_mask(ctx0, mctx_cur->get_mla(), ubatch, cparams);
        inp->self_kq_mask_mla_cnv = inp->self_kq_mask_mla;
    }

    {
        inp->self_k_idxs_lid = mctx_cur->get_lid()->build_input_k_idxs(ctx0, ubatch);

        // ensure that mask type matches fused lightning indexer use (requires f16 mask)
        auto cparams_copy = cparams;
        cparams_copy.flash_attn = cparams.fused_lid;

        inp->self_kq_mask_lid = build_attn_inp_kq_mask(ctx0, mctx_cur->get_lid(), ubatch, cparams_copy);
        inp->self_kq_mask_lid_cnv = inp->self_kq_mask_lid;

        inp->self_k_rot_lid = mctx_cur->get_lid()->build_input_k_rot(ctx0);
    }

    return (llm_graph_input_attn_k_dsa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_kv_msa * llm_graph_context::build_attn_inp_kv_msa(bool msa_enabled) const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_msa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_msa>(hparams, cparams, mctx_cur);

    const auto * mctx_base = mctx_cur->get_base();
    const auto * mctx_idx  = mctx_cur->get_idx();

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_base->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_base->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_base, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    inp->self_k_rot = mctx_base->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_base->build_input_v_rot(ctx0);

    if (msa_enabled) {
        inp->self_k_idxs_idx = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    }

    return (llm_graph_input_attn_kv_msa *) res->add_input(std::move(inp));
}

// TODO: maybe separate the inner implementation into a separate function
//       like with the non-sliding window equivalent
//       once sliding-window hybrid caches are a thing.
llm_graph_input_attn_kv_iswa * llm_graph_context::build_attn_inp_kv_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->get_base()->build_input_v_rot(ctx0);

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);
    inp->self_v_rot_swa = mctx_cur->get_swa()->build_input_v_rot(ctx0);

    return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_k_iswa * llm_graph_context::build_attn_inp_k_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_k_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);

    return (llm_graph_input_attn_k_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_dsv4 * llm_graph_context::build_inp_dsv4() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(mctx);
    const auto * raw_ctx  = mctx_cur->get_raw();

    auto inp_raw = std::make_unique<llm_graph_input_dsv4_raw>(cparams, raw_ctx);

    const int64_t n_stream = mctx_cur->get_csa_plan(ubatch).n_stream;

    GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "DSV4 expects SWA raw cache");

    inp_raw->self_k_idxs = raw_ctx->build_input_k_idxs(ctx0, ubatch);
    inp_raw->self_kq_mask = dsv4_build_raw_kq_mask(ctx0, raw_ctx, ubatch, cparams, n_stream);
    inp_raw->self_kq_mask_cnv = inp_raw->self_kq_mask;

    inp_raw->self_k_rot = raw_ctx->build_input_k_rot(ctx0);
    auto inp = std::make_unique<llm_graph_input_dsv4>(cparams, std::move(inp_raw), mctx_cur);

    dsv4_build_comp_inputs(ctx0, inp->inp_csa, mctx_cur->get_csa_plan(ubatch), "csa", cparams, n_stream);
    dsv4_build_comp_inputs(ctx0, inp->inp_hca, mctx_cur->get_hca_plan(ubatch), "hca", cparams, n_stream);
    dsv4_build_comp_inputs(ctx0, inp->inp_lid, mctx_cur->get_lid_plan(ubatch), "lid", cparams, n_stream);
    inp->inp_csa.k_rot = mctx_cur->get_csa()->build_input_k_rot(ctx0);
    inp->inp_hca.k_rot = mctx_cur->get_hca()->build_input_k_rot(ctx0);
    inp->inp_lid.k_rot = mctx_cur->get_lid()->build_input_k_rot(ctx0);

    return (llm_graph_input_dsv4 *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        ggml_tensor * s,
        ggml_tensor * state_copy_main,
        ggml_tensor * state_copy_extra,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
           uint32_t   rs_size,
            int32_t   rs_zero,
        const llm_graph_get_rows_fn & get_state_rows) const {

    GGML_UNUSED(rs_size);
    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    if (s->type == GGML_TYPE_F32) {
        ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
        ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));
    }

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
    // {state_size, rs_size} -> {state_size, n_seqs}
    ggml_tensor * output_states = get_state_rows(ctx0, states, state_copy_main);
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    ggml_tensor * states_extra = ggml_get_rows(ctx0, states, state_copy_extra);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_2d(ctx0, s, state_size, (n_rs - n_seqs), s->nb[1], (rs_head + n_seqs)*s->nb[1])));

    return output_states;
}

static std::unique_ptr<llm_graph_input_rs> build_rs_inp_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_memory_recurrent_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);

    const int64_t n_rs   = mctx_cur->get_n_rs();
    const int64_t n_seqs = ubatch.n_seqs;

    inp->s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
    ggml_set_input(inp->s_copy);

    inp->s_copy_main  = ggml_view_1d(ctx0, inp->s_copy, n_seqs, 0);
    inp->s_copy_extra = ggml_view_1d(ctx0, inp->s_copy, n_rs - n_seqs, n_seqs * inp->s_copy->nb[0]);

    if (mctx_cur->is_grouped()) {
        inp->brain_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_seqs);
        ggml_set_input(inp->brain_copy);

        for (const auto & [brain_row, rows] : mctx_cur->public_brain_groups()) {
            llm_graph_input_rs::rbb_group_input group;
            group.brain_row = brain_row;
            group.public_rows =
                ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) rows.size());
            ggml_set_input(group.public_rows);

            size_t n_default_shared = 0;
            for (const int32_t row : rows) {
                n_default_shared += !mctx_cur->is_child_row(row);
            }
            if (n_default_shared != 0) {
                group.default_shared_rows =
                    ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) n_default_shared);
                ggml_set_input(group.default_shared_rows);
            }
            inp->rbb_groups.push_back(group);
        }
    }

    inp->head = mctx_cur->get_head();
    inp->rs_z = mctx_cur->get_rs_z();

    return inp;
}

llm_graph_input_rs * llm_graph_context::build_rs_inp() const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    auto inp = build_rs_inp_impl(ctx0, ubatch, mctx_cur);

    return (llm_graph_input_rs *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs,
        const llm_graph_get_rows_fn & get_state_rows) const {
    const auto * kv_state = inp->mctx;

    return build_rs(s, inp->s_copy_main, inp->s_copy_extra, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(),
                    get_state_rows);
}

ggml_tensor * llm_graph_context::build_rs_shared(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs) const {
    GGML_ASSERT(inp->brain_copy != nullptr);
    GGML_ASSERT(inp->mctx->is_grouped());
    GGML_ASSERT((int64_t) n_seqs == inp->brain_copy->ne[0]);

    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);

    ggml_tensor * output_states = ggml_get_rows(ctx0, states, inp->brain_copy);
    ggml_build_forward_expand(gf, output_states);
    return output_states;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_load(
    llm_graph_input_rs * inp,
    const llama_ubatch & ubatch,
                   int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;

    const int64_t n_seqs  = ubatch.n_seqs;

    ggml_tensor * token_shift_all = mctx_cur->get_r_l(il);

    ggml_tensor * token_shift = build_rs(
            inp, token_shift_all,
            hparams.n_embd_r(), n_seqs);

    token_shift = ggml_reshape_3d(ctx0, token_shift, hparams.n_embd, token_shift_count, n_seqs);

    return token_shift;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_store(
         ggml_tensor * token_shift,
  const llama_ubatch & ubatch,
                 int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;
    const auto n_embd = hparams.n_embd;

    const int64_t n_seqs = ubatch.n_seqs;

    const auto kv_head = mctx_cur->get_head();

    return ggml_cpy(
        ctx0,
        ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
        ggml_view_1d(ctx0, mctx_cur->get_r_l(il), hparams.n_embd_r()*n_seqs, hparams.n_embd_r()*kv_head*ggml_element_size(mctx_cur->get_r_l(il)))
    );
}

llm_graph_input_mem_hybrid * llm_graph_context::build_inp_mem_hybrid() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn(), sched, fp_reserve_sizing, res);

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_k * llm_graph_context::build_inp_mem_hybrid_k() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_k>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_k *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_iswa * llm_graph_context::build_inp_mem_hybrid_iswa() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);

    auto inp_rs = build_rs_inp_impl(ctx0, ubatch, mctx_cur->get_recr());

    // build iswa attention input
    const auto * attn_ctx = mctx_cur->get_attn();

    auto inp_attn = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, attn_ctx);

    {
        inp_attn->self_k_idxs = attn_ctx->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs = attn_ctx->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask = build_attn_inp_kq_mask(ctx0, attn_ctx->get_base(), ubatch, cparams);
        inp_attn->self_kq_mask_cnv = inp_attn->self_kq_mask;
    }

    {
        inp_attn->self_k_idxs_swa = attn_ctx->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs_swa = attn_ctx->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, attn_ctx->get_swa(), ubatch, cparams);
        inp_attn->self_kq_mask_swa_cnv = inp_attn->self_kq_mask_swa;
    }

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_iswa>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_iswa *) res->add_input(std::move(inp));
}

void llm_graph_context::build_dense_out(
    ggml_tensor * dense_2,
    ggml_tensor * dense_2_b,
    ggml_tensor * dense_3) const {
    if (!cparams.embeddings || !(dense_2 || dense_2_b || dense_3)) {
        return;
    }
    ggml_tensor * cur = res->t_embd_pooled != nullptr ? res->t_embd_pooled : res->t_embd;
    GGML_ASSERT(cur != nullptr && "missing t_embd_pooled/t_embd");

    if (dense_2) {
        cur = ggml_mul_mat(ctx0, dense_2, cur);
    }
    if (dense_2_b) {
        cur = ggml_add(ctx0, cur, dense_2_b);
    }
    if (dense_3) {
        cur = ggml_mul_mat(ctx0, dense_3, cur);
    }
    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;
    ggml_build_forward_expand(gf, cur);
}


void llm_graph_context::build_pooling(
        ggml_tensor * cls,
        ggml_tensor * cls_b,
        ggml_tensor * cls_out,
        ggml_tensor * cls_out_b,
        ggml_tensor * cls_norm) const {
    if (!cparams.embeddings) {
        return;
    }

    ggml_tensor * inp = res->t_embd;

    //// find result_norm tensor for input
    //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
    //    inp = ggml_graph_node(gf, i);
    //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
    //        break;
    //    }

    //    inp = nullptr;
    //}

    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

    ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        case LLAMA_POOLING_TYPE_MEAN:
            {
                ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_RANK:
            {
                if (arch == LLM_ARCH_MODERN_BERT) {
                    // modern bert gte reranker builds mean first then applies prediction head and classifier
                    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py#L1404-1411
                    ggml_tensor * inp_mean = build_inp_mean();
                    cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
                } else {
                    ggml_tensor * inp_cls = build_inp_cls();
                    cur = ggml_get_rows(ctx0, inp, inp_cls);
                }

                // classification head
                // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                if (cls) {
                    cur = ggml_mul_mat(ctx0, cls, cur);
                    if (cls_b) {
                        cur = ggml_add(ctx0, cur, cls_b);
                    }
                    if (arch == LLM_ARCH_MODERN_BERT) {
                        cur = ggml_gelu(ctx0, cur);
                    } else {
                        cur = ggml_tanh(ctx0, cur);
                    }
                    if (cls_norm) {
                        // head norm
                        cur = build_norm(cur, cls_norm, NULL, LLM_NORM, -1);
                    }
                }

                // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                // Single layer classification head (direct projection)
                // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
                if (cls_out) {
                    cur = ggml_mul_mat(ctx0, cls_out, cur);
                    if (cls_out_b) {
                        cur = ggml_add(ctx0, cur, cls_out_b);
                    }
                }

                // softmax for qwen3 reranker
                if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL) {
                    cur = ggml_soft_max(ctx0, cur);
                }
            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            }
    }

    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;

    ggml_build_forward_expand(gf, cur);
}

void llm_graph_context::build_sampling() const {
    if (samplers.empty() || !res->t_logits) {
        return;
    }

    std::array<ggml_tensor *, 2> outs;
    outs[0] = res->t_logits;

    auto inp_sampling = std::make_unique<llm_graph_input_sampling>(samplers);
    res->add_input(std::move(inp_sampling));

    std::map<llama_seq_id, int32_t> seq_to_logit_row;
    int32_t logit_row_idx = 0;

    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (ubatch.output[i]) {
            llama_seq_id seq_id = ubatch.seq_id[i][0];
            seq_to_logit_row[seq_id] = logit_row_idx;
            logit_row_idx++;
        }
    }

    // res->t_logits will contain logits for all tokens that want the logits calculated (logits=1 or output=1)
    GGML_ASSERT(res->t_logits != nullptr && "missing t_logits tensor");

    // add a dummy row of logits
    // this trick makes the graph static, regardless of which samplers are activated
    // this is important in order to minimize graph reallocations
    ggml_tensor * logits_t = ggml_pad(ctx0, res->t_logits, 0, 1, 0, 0);

    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = seq_to_logit_row.find(seq_id);

        // inactive samplers always work on the first row
        const auto row_idx = it != seq_to_logit_row.end() ? it->second : 0;
        const int i_out    = it != seq_to_logit_row.end() ? 1          : 0;

        ggml_tensor * logits_seq = ggml_view_1d(ctx0, logits_t, logits_t->ne[0], row_idx * logits_t->nb[1]);
        ggml_format_name(logits_seq, "logits_seq_%d", seq_id);

        struct llama_sampler_data data = {
            /*.logits      =*/ logits_seq,
            /*.probs       =*/ nullptr,
            /*.sampled     =*/ nullptr,
            /*.candidates  =*/ nullptr,
        };

        assert(sampler->iface->backend_apply);
        sampler->iface->backend_apply(sampler, ctx0, gf, &data);

        if (data.sampled != nullptr) {
            res->t_sampled[seq_id] = data.sampled;
            outs[1] = data.sampled;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.probs != nullptr) {
            res->t_sampled_probs[seq_id] = data.probs;
            outs[1] = data.probs;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.logits != nullptr) {
            res->t_sampled_logits[seq_id] = data.logits;
            outs[1] = data.logits;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.candidates != nullptr) {
            res->t_candidates[seq_id] = data.candidates;
            outs[1] = data.candidates;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }
    }

    // TODO: Call llama_sampler_accept_ggml after all samplers have been applied.
    /*
    for (const auto & [seq_id, sampler] : samplers) {
        if (auto it = res->t_sampled.find(seq_id); it != res->t_sampled.end()) {
            ggml_tensor * selected_token = it->second;
            if (selected_token != nullptr) {
                llama_sampler_accept_ggml(sampler, ctx0, gf, selected_token);
            }
        }
    }
    */
}

int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;

    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = std::abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }

    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);

    return relative_bucket;
}

// ============================================================================
// FlashPrefill V2 sparse attention (GraphIntegration)
//
// Ordinary attention KV + RERoT full-attention paths consume the
// CacheFragments compact layout (uses/use_offsets, O(Q*F)) directly: no old
// Q*K expansion anywhere on the new path, exact_rows never touched
// (oracle-only), exact_all routes through identical metadata with the select
// exact_all flag. Per-ubatch roles come from cparams.flashprefill_rows
// (ContextIntegration/StatePolicy, shared ownership); source mapping from
// llama_ubatch::source_row (BatchIdentity, never guessed).
//
// Spread of responsibilities (no duplicate writers):
// - PolicyCore owns route/pack/scratch math (called, never redefined).
// - CacheFragments owns fragments/groups/uses/epochs (built, never mutated).
// - WireReference owns the I32 schema (packed via ggml helpers, offsets
//   never hand-rolled).
// - OpRegistration owns pool/select/attn constructors (called with real
//   src[] edges; cpy_k/cpy_v outputs ride pool src[3]/src[4] ordering edges).
// - VulkanDispatch/Shaders/CpuKernels own kernel numerics and the backend
//   matrix (queried here by backend-name scan only).
// - FittingIntegration owns common/fit (same scratch helper inputs; graph
//   tensors ride the existing graph reserve, never double-charged).
//
// Data-vs-topology split: capacity buckets, policy/mode, role, dtype tags
// and kernel variant are topology (reuse key). Epochs, counts within caps,
// stamps, physical maps and per-call tail coordinates are submit data
// (refreshed every set_input from the current rows + fresh layout).
// Corruption (bad ranges, dup keys, stale mapping, cross-reader mismatch)
// throws in EVERY mode; only designed policy-dense reasons stay dense.
//
// Supported envelope (shape-gated, never arch-whitelisted): ordinary GQA
// full-attention prefill with F32 Q, single position per token, no additive
// KQ bias, no ALiBi, unified or single-stream KV, backbone cache dtypes the
// kernels dequant (F32/F16/quants/Turbo). RERoT teacher-forced adds phased
// groups via the layout fragment-group mapping. Outside the envelope the
// layer keeps its pre-existing dense path: recurrent, SWA, MTP,
// decode/verify/frontier/embedding roles, mixed ubatches, unknown live
// boundaries, short context, dense-tail rows (exact through the new op),
// unsupported backends and over-admission shapes are all explicit dense
// reasons (summary dense_rows/dense_layers); REQUIRED turns the
// unsupported/capacity class into throws. Any model file opts in with the
// 3-line try_build hook (qwen35/qwen35moe wired); unwired archs stay dense.

namespace {

// Slice bucketing (matches the layout CAP_BUCKET discipline).
inline uint32_t llm_fp_bucket(uint32_t n) {
    return llama_flashprefill_layout_params::bucket_for(n);
}

// Sound tile fan-out bound: the distinct packed-Q tiles touched by one
// query's kv-head group (gqa consecutive packed rows). Tight for sane
// block_q (<=2 tiles), exact for degenerate ones (up to gqa at block_q=1).
// Used by reserve/can_reuse/submit capacity estimates; the packer itself
// computes exact ranges with no bound.
inline uint32_t llm_fp_tile_fanout(uint32_t gqa, uint32_t block_q) {
    if (gqa == 0 || block_q == 0) {
        return 0;
    }
    const uint64_t f = (uint64_t) gqa / (uint64_t) block_q + 2u;
    return f > (uint64_t) gqa ? gqa : (uint32_t) f;
}

// Single layout-params construction (callers must stay identical so the
// owner planning cache hits across probe and build).
inline llama_flashprefill_layout_params llm_fp_layout_params_for(
        const llama_flashprefill_config & cfg, const llama_cparams & cparams) {
    llama_flashprefill_layout_params lp;
    lp.block_k = cfg.block_k;
    lp.causal = cparams.causal_attn;
    lp.want_exact_rows = false; // production: compact uses only, no E expansion
    return lp;
}

inline bool llm_fp_is_enabled(const llama_cparams & cparams) {
    return llama_flashprefill_is_enabled(&cparams.flashprefill);
}

inline const std::vector<llama_flashprefill_row> * llm_fp_rows_of(const llama_cparams & cparams) {
    const auto & p = cparams.flashprefill_rows;
    return p ? p.get() : nullptr;
}

// Uniform eligible role (whole ubatch), or UNKNOWN when mixed/ineligible.
// Only PREFILL (ordinary, incl. Tri sparse-position) and
// REROT_TEACHER_FORCED ever route sparse; decode/MTP/verify/replay/
// frontier/embedding/rerank/multimodal/unknown stay dense, never reordered.
int32_t llm_fp_uniform_role(const std::vector<llama_flashprefill_row> & rows) {
    if (rows.empty()) {
        return LLAMA_FLASHPREFILL_ROLE_UNKNOWN;
    }
    const int32_t r0 = rows[0].role;
    if (r0 != LLAMA_FLASHPREFILL_ROLE_PREFILL &&
        r0 != LLAMA_FLASHPREFILL_ROLE_REROT_TEACHER_FORCED) {
        return LLAMA_FLASHPREFILL_ROLE_UNKNOWN;
    }
    for (const auto & r : rows) {
        if (r.role != r0) {
            return LLAMA_FLASHPREFILL_ROLE_UNKNOWN;
        }
    }
    return r0;
}

// StatePolicy reserve snapshot: rows present, no source map
// (ubatch_reserve never carries one), all PREFILL/seq0/unknown-pos/known=false.
// Live unknown-boundary rows (source map present) also route dense but size
// nothing; only the snapshot (or the explicit reserve flag) sizes caps.
bool llm_fp_is_snapshot_rows(const std::vector<llama_flashprefill_row> & rows, const llama_ubatch & ubatch) {
    if (rows.size() != (size_t) ubatch.n_tokens || ubatch.n_tokens == 0) {
        return false;
    }
    if (ubatch.source_row != nullptr) {
        return false;
    }
    for (const auto & r : rows) {
        if (r.role        != LLAMA_FLASHPREFILL_ROLE_PREFILL ||
            r.seq_id      != 0 ||
            r.logical_pos != LLAMA_FLASHPREFILL_POS_UNKNOWN ||
            r.prefill_known) {
            return false;
        }
    }
    return true;
}

// Backend scan mirroring the RERoT variant discipline. CUDA/Metal/RPC have
// no sparse kernels (unsupported); CPU reference + Vulkan fused supported.
// Returns 0 = unsupported, 1 = CPU reference, 2 = Vulkan fused, 3 = CUDA native.
int llm_fp_backend_variant(ggml_backend_sched_t sched, bool * supported_out) {
    bool supported = true;
    bool vulkan = false;
    bool cuda = false;
    if (sched != nullptr) {
        const int n = ggml_backend_sched_get_n_backends(sched);
        for (int i = 0; i < n; ++i) {
            ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
            if (b == nullptr) {
                continue;
            }
            const char * name = ggml_backend_name(b);
            if (name == nullptr) {
                continue;
            }
            const std::string s(name);
            if (s.find("CUDA")  != std::string::npos ||
                s.find("cuda")  != std::string::npos) {
                cuda = true;
            } else if (s.find("Metal") != std::string::npos ||
                s.find("RPC")   != std::string::npos) {
                supported = false;
            }
            if (s.find("Vulkan") != std::string::npos ||
                s.find("vulkan") != std::string::npos) {
                vulkan = true;
            }
        }
    }
    if (supported_out != nullptr) {
        *supported_out = supported;
    }
    if (!supported) {
        return 0;
    }
    if (cuda) {
        return 3;
    }
    return vulkan ? 2 : 1;
}

} // namespace

bool llm_graph_context::flashprefill_is_reserve_snapshot() const {
    const auto * rows = llm_fp_rows_of(cparams);
    if (rows == nullptr) {
        return false;
    }
    return llm_fp_is_snapshot_rows(*rows, ubatch);
}

bool llm_graph_context::flashprefill_backend_supported() const {
    bool supported = true;
    llm_fp_backend_variant(sched, &supported);
    return supported;
}

// Eligible FULL-attention layer predicate (single definition for counting
// and gating): recurrent, SWA and structurally incompatible layers never
// count — only actual full-normal layers do (contract-general; first target
// Qwen has neither SWA nor malformed layers).
static bool llm_fp_layer_eligible(const llama_hparams & hparams, int il) {
    if (hparams.is_recr(il) || hparams.is_swa(il)) {
        return false;
    }
    const uint32_t hj  = hparams.n_head(il);
    const uint32_t hkv = hparams.n_head_kv(il);
    return hj != 0 && hkv != 0 && hj % hkv == 0;
}

static int32_t llm_fp_count_full_layers(const llama_hparams & hparams) {
    int32_t n = 0;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (llm_fp_layer_eligible(hparams, (int) il)) {
            ++n;
        }
    }
    return n;
}

int llm_graph_context::flashprefill_eligible_full_index(int il) const {
    // full_attn_layers counts eligible FULL-attention layers, not model
    // index, so hybrid intervals keep working when layers shift.
    int idx = 0;
    for (int j = 0; j < il; ++j) {
        if (llm_fp_layer_eligible(hparams, j)) {
            ++idx;
        }
    }
    return idx;
}

void llm_graph_result::record_flashprefill_plan(
        ggml_tensor * plan, ggml_tensor * pool, ggml_backend_t backend,
        int il, int32_t sparse_rows, int32_t dense_rows) {
    if (plan == nullptr) {
        return; // fail-closed upstream: the vector never holds nulls
    }
    // Main-directed: pin the plan for the completion-boundary metrics read
    // (single end-sync, never per-layer). OFF/dense graphs record nothing.
    ggml_set_output(plan);
    flashprefill_plans.push_back(plan);
    flashprefill_plan_ils.push_back(il);
    flashprefill_summary.sparse_layers += 1;
    flashprefill_summary.sparse_rows   += sparse_rows;
    flashprefill_summary.dense_rows    += dense_rows;
    // Pinned total: every plan stays live (sum), pools are transient
    // (max per backend). Checked 64-bit; tensor capacities, never estimates.
    flashprefill_summary.scratch_bytes += (uint64_t) ggml_nbytes(plan);
    flashprefill_summary.plans_bytes   += (uint64_t) ggml_nbytes(plan);
    if (pool != nullptr) {
        const uint64_t pb = (uint64_t) ggml_nbytes(pool);
        bool found = false;
        for (auto & e : flashprefill_pool_peaks) {
            if (e.first == backend) {
                found = true;
                if (pb > e.second) {
                    flashprefill_summary.scratch_bytes += pb - e.second;
                    flashprefill_summary.pool_bytes    += pb - e.second;
                    e.second = pb;
                }
                break;
            }
        }
        if (!found) {
            flashprefill_pool_peaks.emplace_back(backend, pb);
            flashprefill_summary.scratch_bytes += pb;
            flashprefill_summary.pool_bytes    += pb;
        }
    }
}

// Worst-case caps, pure in (config, model, probe inputs). Same shape as the
// FittingIntegration accounting (identical scratch helper inputs); graph
// tensors ride the existing graph reserve, never double-charged. All math
// checked 64-bit; false + reason on overflow (caller throws fail-closed).
struct llm_fp_reserve_caps {
    uint32_t f = 0;      // fragments (actual bound, pre-bucket)
    uint32_t u = 0;      // wire uses incl. kv-head fan-out + tile straddle
    uint32_t c = 0;      // cells
    uint32_t r = 0;      // rows
    uint32_t t = 0;      // packed-Q tiles
    uint32_t g = 0;      // Q groups (ordinary: one per query)
    uint32_t maxsel = 0; // per-(tile,head) sound bound: F * queries-per-tile
    uint32_t q_rsv = 0;  // reserve query bound used throughout
};

static bool llm_fp_reserve_caps_for(
        const llama_flashprefill_config & cfg,
        uint32_t n_ctx_cells,
        uint32_t q_rsv,
        uint32_t hq,
        uint32_t hkv,
        uint32_t gqa,
        llm_fp_reserve_caps * out,
        std::string * error) {
    if (out == nullptr) {
        return false;
    }
    *out = llm_fp_reserve_caps{};
    if (cfg.block_q == 0 || cfg.block_k == 0 || hq == 0 || hkv == 0 || gqa == 0 || q_rsv == 0) {
        if (error != nullptr) {
            *error = "flashprefill reserve: zero block/head/query bound";
        }
        return false;
    }
    llama_flashprefill_fragment_budget budget;
    if (!llama_flashprefill_admission_budget_for(
                n_ctx_cells, cfg.block_k, LLAMA_FLASHPREFILL_ADMISSION_STREAMS_UNIFIED,
                q_rsv, &budget, error)) {
        return false;
    }
    const uint64_t F = budget.n_fragments;
    // Wire uses fan out per kv head and tile straddle (sound bound shared
    // with the can_reuse/submit estimates).
    const uint64_t U = F * (uint64_t) q_rsv * (uint64_t) hkv * (uint64_t) llm_fp_tile_fanout(gqa, cfg.block_q);
    const uint64_t C = F * (uint64_t) cfg.block_k;
    const uint64_t R = (uint64_t) q_rsv * (uint64_t) hq;
    const uint64_t T = ((uint64_t) q_rsv * (uint64_t) gqa + (uint64_t) cfg.block_q - 1u) / (uint64_t) cfg.block_q;
    // Distinct queries sharing one packed-Q tile: packed rows of a query run
    // gqa-consecutive, so at most block_q/gqa + 2 queries meet in a tile.
    const uint64_t q_per_tile = (uint64_t) cfg.block_q / (uint64_t) gqa + 2u;
    const uint64_t maxsel = F * (q_per_tile > (uint64_t) q_rsv ? (uint64_t) q_rsv : q_per_tile);
    for (uint64_t v : {U, C, R, T, maxsel}) {
        if (v > (uint64_t) INT32_MAX) {
            if (error != nullptr) {
                *error = "flashprefill reserve exceeds I32 wire domain";
            }
            return false;
        }
    }
    if (T == 0) {
        if (error != nullptr) {
            *error = "flashprefill reserve: zero tiles";
        }
        return false;
    }
    out->f      = (uint32_t) F;
    out->u      = (uint32_t) U;
    out->c      = (uint32_t) C;
    out->r      = (uint32_t) R;
    out->t      = (uint32_t) T;
    out->g      = q_rsv;
    out->maxsel = (uint32_t) maxsel;
    out->q_rsv  = q_rsv;
    return true;
}

// Reserve probe inputs from live params (no layout, no rows needed).
static bool llm_fp_reserve_probe_inputs(
        const llama_cparams & cparams,
        const llama_hparams & hparams,
        const llama_ubatch  & ubatch,
        uint32_t * n_ctx_cells_out,
        uint32_t * q_rsv_out) {
    uint32_t n_ctx_cells = cparams.n_ctx_kv != 0 ? cparams.n_ctx_kv : cparams.n_ctx;
    uint32_t q_rsv = cparams.n_ubatch != 0 ? cparams.n_ubatch : ubatch.n_tokens;
    if (n_ctx_cells == 0 || q_rsv == 0) {
        return false;
    }
    GGML_UNUSED(hparams);
    *n_ctx_cells_out = n_ctx_cells;
    *q_rsv_out       = q_rsv;
    return true;
}

// ---- checked wire pack (plan generation) ----
//
// Packs the CacheFragments compact layout (uses/use_offsets, O(Q*F)) plus
// per-ubatch roles into GGML wire rows/uses/cells. Never expands Q*K, never
// reads exact_rows, never reads KV bytes, never reorders real rows/Q/output
// IDs: only descriptors are sorted. Tile ids are ubatch-relative
// (0-based over this ubatch's packed rows) in every tail scope; only the
// dense-tail DECISION uses scope-absolute coordinates. Runs once per graph
// build (and per submit refill); the expensive GGML full validator never
// runs here (oracle/tests only) — checked construction below rejects
// corruption fail-closed instead.
// Selection privacy: a packed tile must never mix distinct seq/reader
// visibility domains, even under one role. Tile-energy aggregates over the
// packed rows of a tile, so cross-domain sharing would let one domain's
// (possibly huge) values change another domain's selected set — a pollution
// violation despite per-row attention legality. Tiles are therefore
// partitioned by frozen (seq, reader, prompt interval) domain: tile ids are
// assigned per domain (densely compacted, no padding waste) and verified
// disjoint. Pool means stay shared (fragment K/V means are Q-independent);
// phase groups stay per query. Q rows, outputs and recurrent state never
// move (descriptors only). Single-domain batches (all ordinary server
// prefills) number identically to plain packed order: zero regression.
struct llm_fp_domain_key {
    int32_t seq_id;
    uint32_t reader_id;
    int32_t pbegin;
    int32_t pend;
    uint32_t uniq; // query index when seq unknown (each its own domain), else 0

    bool operator<(const llm_fp_domain_key & o) const {
        if (seq_id    != o.seq_id)    return seq_id    < o.seq_id;
        if (reader_id != o.reader_id) return reader_id < o.reader_id;
        if (pbegin    != o.pbegin)    return pbegin    < o.pbegin;
        if (pend      != o.pend)      return pend      < o.pend;
        return uniq < o.uniq;
    }
};

// Domain-aware tile layout (exact): domains in first-appearance order,
// packed rows per domain in ubatch order, tiles numbered densely across
// domains (no padding waste). Shared by the packer and the reuse tile-count
// check so numbering can never diverge. Returns false on degenerate config.
static bool llm_fp_domain_layout(
        const std::vector<llama_flashprefill_row> & rows,
        uint32_t block_q,
        uint32_t gqa,
        std::vector<int64_t> * q_tile_base_out,
        std::vector<int64_t> * q_local_out,
        std::vector<int> * q_domain_out,
        uint64_t * n_tiles_out) {
    if (block_q == 0 || gqa == 0) {
        return false;
    }
    std::map<llm_fp_domain_key, int> domain_ord;
    std::vector<int> q_domain(rows.size(), -1);
    for (size_t q = 0; q < rows.size(); ++q) {
        const auto & r = rows[q];
        llm_fp_domain_key k{r.seq_id, r.reader_id, r.prefill_begin, r.prefill_end,
            r.seq_id == LLAMA_FLASHPREFILL_SEQ_UNKNOWN ? (uint32_t) q : 0u};
        auto it = domain_ord.find(k);
        int ord;
        if (it == domain_ord.end()) {
            ord = (int) domain_ord.size();
            domain_ord[k] = ord;
        } else {
            ord = it->second;
        }
        q_domain[(size_t) q] = ord;
    }
    std::vector<uint64_t> dom_seen(domain_ord.size(), 0);
    std::vector<int64_t> q_local(rows.size(), 0);
    for (size_t q = 0; q < rows.size(); ++q) {
        const int d = q_domain[q];
        q_local[q] = (int64_t) (dom_seen[(size_t) d] * (uint64_t) gqa);
        dom_seen[(size_t) d] += 1;
    }
    std::vector<uint64_t> dom_packed(domain_ord.size(), 0);
    for (size_t d = 0; d < domain_ord.size(); ++d) {
        dom_packed[d] = dom_seen[d] * (uint64_t) gqa;
    }
    std::vector<int64_t> q_tile_base(rows.size(), 0);
    uint64_t base = 0;
    std::vector<uint64_t> dom_base(domain_ord.size(), 0);
    for (size_t d = 0; d < domain_ord.size(); ++d) {
        dom_base[d] = base;
        base += (dom_packed[d] + (uint64_t) block_q - 1u) / (uint64_t) block_q;
    }
    for (size_t q = 0; q < rows.size(); ++q) {
        q_tile_base[q] = (int64_t) dom_base[(size_t) q_domain[q]];
    }
    if (q_tile_base_out != nullptr) {
        *q_tile_base_out = q_tile_base;
    }
    if (q_local_out != nullptr) {
        *q_local_out = q_local;
    }
    if (q_domain_out != nullptr) {
        *q_domain_out = q_domain;
    }
    if (n_tiles_out != nullptr) {
        *n_tiles_out = base;
    }
    return true;
}

llm_fp_pack_rc llm_fp_pack_live(
        const llama_flashprefill_layout & layout,
        const std::vector<llama_flashprefill_row> & rows,
        const llama_flashprefill_config & cfg,
        uint32_t hq,
        uint32_t hkv,
        uint32_t gqa,
        uint32_t n_tokens,
        llm_fp_pack * out,
        std::string * error,
        bool dry_run) {
    if (out == nullptr) {
        return LL_FP_PACK_CORRUPT;
    }
    *out = llm_fp_pack{};
    auto fail = [&](const char * msg) {
        if (error != nullptr) {
            *error = msg;
        }
        return LL_FP_PACK_CORRUPT;
    };
    const uint32_t Q = layout.n_queries();
    const uint32_t F = layout.n_fragments();
    const uint32_t G = layout.n_groups();
    if (Q != n_tokens || Q == 0) {
        return fail("flashprefill pack: layout queries != ubatch rows");
    }
    if (rows.size() != (size_t) n_tokens) {
        return fail("flashprefill pack: row snapshot != ubatch rows");
    }
    if (hq == 0 || hkv == 0 || gqa == 0 || hq % hkv != 0 || hq / hkv != gqa) {
        return fail("flashprefill pack: GQA mapping inconsistent");
    }
    if (cfg.block_q == 0) {
        return fail("flashprefill pack: zero block_q");
    }
    // Dense query identity: the builder emits one query per ubatch row in
    // order; wire source_query is that index (output query id, retained).
    for (uint32_t q = 0; q < Q; ++q) {
        if (layout.queries[q].query_index != q) {
            return fail("flashprefill pack: non-dense query identity");
        }
        const auto & row = rows[q];
        if (row.prefill_begin >= row.prefill_end) {
            return fail("flashprefill pack: invalid prefill interval");
        }
        if (row.logical_pos != LLAMA_FLASHPREFILL_POS_UNKNOWN &&
            (row.logical_pos < row.prefill_begin || row.logical_pos >= row.prefill_end)) {
            return fail("flashprefill pack: logical pos outside interval");
        }
        if (layout.use_offsets.size() != (size_t) Q + 1) {
            return fail("flashprefill pack: use offsets != Q+1");
        }
    }
    if (layout.group_offsets.size() != (G == 0 ? (size_t) 0 : (size_t) Q + 1)) {
        // Ordinary layouts carry no groups (identity: one group per query);
        // RERoT layouts carry per-query group ranges. Anything else is corrupt.
        if (!(G == 0 && layout.group_offsets.empty())) {
            return fail("flashprefill pack: group offsets != Q+1");
        }
    }
    // Ordinary plans may carry one identity group per query for the owner
    // oracle. Their Q is already roped by the model: do not allocate/upload
    // unused gather/position tensors or rotate Q again. Only RERoT consumes
    // real phase groups in the graph.
    const bool use_groups = layout.is_rerot && G != 0;
    if (!layout.is_rerot && G != 0) {
        if (G != Q) {
            return fail("flashprefill pack: ordinary groups are not identity");
        }
        for (uint32_t q = 0; q < Q; ++q) {
            if (layout.groups[q].query_index != q ||
                layout.group_offsets[q] != q || layout.group_offsets[q + 1] != q + 1) {
                return fail("flashprefill pack: ordinary group mapping is not identity");
            }
        }
    }
    if (use_groups) {
        for (uint32_t g = 0; g < G; ++g) {
            if (layout.groups[g].query_index >= Q) {
                return fail("flashprefill pack: group query out of range");
            }
        }
    }
    // Per-query resident-visible-legal counts from the compact uses (never
    // physical capacity, never Tri-deleted history).
    std::vector<uint64_t> vcount(Q, 0);
    for (uint32_t q = 0; q < Q; ++q) {
        const uint32_t b = layout.use_offsets[q];
        const uint32_t e = layout.use_offsets[q + 1];
        if (b > e || (uint64_t) e > (uint64_t) layout.uses.size()) {
            return fail("flashprefill pack: use range out of bounds");
        }
        uint32_t prev_frag = UINT32_MAX;
        for (uint32_t i = b; i < e; ++i) {
            const auto & u = layout.uses[i];
            if (u.query != q) {
                return fail("flashprefill pack: use query mismatch");
            }
            if (!layout.is_rerot && G != 0 && u.group != q) {
                return fail("flashprefill pack: ordinary use group is not identity");
            }
            if (u.fragment >= F) {
                return fail("flashprefill pack: use fragment out of range");
            }
            if (u.sub_count == 0) {
                return fail("flashprefill pack: empty use");
            }
            // Frozen planning guarantee: fragment ids strictly ascending
            // within each query (at most one use per pair, in order).
            if (i > b && u.fragment <= prev_frag) {
                return fail("flashprefill pack: unordered/duplicate (query,fragment)");
            }
            prev_frag = u.fragment;
            const auto & fr = layout.fragments[u.fragment];
            if (fr.token_count == 0) {
                return fail("flashprefill pack: empty fragment");
            }
            // Subrange inside the fragment member range, member order
            // (positions/storage ascending): legal subsets are prefixes, so
            // the wire offset is base + relative.
            if (!fr.contiguous && fr.cell_ref_offset == UINT32_MAX) {
                return fail("flashprefill pack: fragment addressing invalid");
            }
            const uint64_t rel_end = (uint64_t) u.sub_off + (uint64_t) u.sub_count;
            if (rel_end > (uint64_t) fr.token_count) {
                return fail("flashprefill pack: use subrange outside fragment");
            }
            if (!fr.contiguous &&
                (uint64_t) fr.cell_ref_offset + (uint64_t) fr.token_count > (uint64_t) layout.cell_refs.size()) {
                return fail("flashprefill pack: cell refs out of bounds");
            }
            if (u.flags & ~llama_flashprefill_use::FLAG_MANDATORY) {
                return fail("flashprefill pack: use flag out of range");
            }
            vcount[q] += (uint64_t) u.sub_count;
            if (vcount[q] > (uint64_t) INT32_MAX) {
                if (error != nullptr) {
                    *error = "flashprefill pack: per-query tokens exceed I32";
                }
                return LL_FP_PACK_OVERFLOW;
            }
        }
        if (vcount[q] == 0) {
            // The builder must cover V(r) (non-empty: at least the query's
            // own KV). Empty coverage is a builder-contract breach, never a
            // silent exact-over-nothing.
            return fail("flashprefill pack: empty query coverage");
        }
        out->visible_tokens += vcount[q];
    }
    // Packed-Q geometry (exact; BM may not divide G). Call-scope totals are
    // validated (overflow-safe); tile NUMBERING is domain-partitioned (see
    // llm_fp_domain_layout): tiles never mix seq/reader visibility domains,
    // numbered densely per domain in first-appearance order. The tail
    // DECISION alone uses scope-absolute coordinates (call range or frozen
    // logical interval). Single-domain batches number identically to plain
    // packed order.
    uint32_t total_packed = 0;
    {
        uint32_t tmp_total = 0, tmp_tiles = 0;
        if (llama_flashprefill::packed_layout_checked(
                    n_tokens, gqa, cfg.block_q, &tmp_total, &tmp_tiles) != LLAMA_FLASHPREFILL_OK) {
            if (error != nullptr) {
                *error = "flashprefill pack: packed layout overflow";
            }
            return LL_FP_PACK_OVERFLOW;
        }
        total_packed = tmp_total;
        (void) tmp_tiles;
    }
    std::vector<int64_t> q_tile_base;
    std::vector<int64_t> q_local;
    std::vector<int> q_domain;
    uint64_t n_tiles_dom = 0;
    if (!llm_fp_domain_layout(rows, cfg.block_q, gqa, &q_tile_base, &q_local, &q_domain, &n_tiles_dom)) {
        return fail("flashprefill pack: domain layout failed");
    }
    if (n_tiles_dom == 0 || n_tiles_dom > (uint64_t) INT32_MAX) {
        if (error != nullptr) {
            *error = "flashprefill pack: tile count out of I32 range";
        }
        return LL_FP_PACK_OVERFLOW;
    }
    out->n_tiles = (int64_t) n_tiles_dom;
    const bool logical_scope = (cfg.tail_scope == LLAMA_FLASHPREFILL_TAIL_LOGICAL_PROMPT);
    const bool exact_all_cfg = cfg.exact_all;
    // Row records (unsorted): one per (query, head); tail/short evaluated
    // per packed row. UNKNOWN logical positions force exact (position-free)
    // rather than guessing a tail.
    std::vector<llm_fp_rowrec> rows_u;
    if (!dry_run) {
        rows_u.reserve((size_t) n_tokens * (size_t) hq);
    }
    std::vector<char> forced_flat;
    forced_flat.reserve((size_t) n_tokens * (size_t) hq);
    int32_t sparse_rows = 0;
    int32_t forced_rows = 0;
    for (uint32_t q = 0; q < Q; ++q) {
        const auto & row = rows[q];
        for (uint32_t h = 0; h < hq; ++h) {
            const uint32_t s = h % gqa;
            const uint32_t kh = h / gqa;
            const uint64_t pr_call = (uint64_t) q * (uint64_t) gqa + (uint64_t) s;
            // Domain-partitioned tile (never shared across visibility
            // domains); pr_call stays call-scope-absolute for routing only.
            const int32_t tile = (int32_t) (q_tile_base[(size_t) q] +
                (q_local[(size_t) q] + (int64_t) s) / (int64_t) cfg.block_q);
            bool forced = false;
            if (!exact_all_cfg) {
                if (logical_scope) {
                    if (row.logical_pos == LLAMA_FLASHPREFILL_POS_UNKNOWN) {
                        forced = true;
                    } else {
                        const int64_t off = (int64_t) row.logical_pos - (int64_t) row.prefill_begin;
                        const uint64_t pr_pre = (uint64_t) off * (uint64_t) gqa + (uint64_t) s;
                        const uint64_t total_pre =
                            (uint64_t) ((int64_t) row.prefill_end - (int64_t) row.prefill_begin) * (uint64_t) gqa;
                        if (pr_pre > (uint64_t) UINT32_MAX || total_pre > (uint64_t) UINT32_MAX ||
                            total_pre == 0 || pr_pre >= total_pre) {
                            if (error != nullptr) {
                                *error = "flashprefill pack: logical packed range overflow/empty";
                            }
                            return LL_FP_PACK_OVERFLOW;
                        }
                        const auto r = llama_flashprefill::route_row_logical(
                            &cfg, &row, (uint32_t) vcount[q], true,
                            (uint32_t) pr_pre, (uint32_t) total_pre);
                        if (r == LLAMA_FLASHPREFILL_ROUTE_DENSE_TAIL ||
                            r == LLAMA_FLASHPREFILL_ROUTE_DENSE_SHORT_CONTEXT) {
                            forced = true;
                        } else if (r != LLAMA_FLASHPREFILL_ROUTE_SPARSE &&
                                   r != LLAMA_FLASHPREFILL_ROUTE_EXACT_ALL) {
                            return fail("flashprefill pack: unexpected per-row route");
                        }
                    }
                } else {
                    const auto r = llama_flashprefill::route_row_call(
                        &cfg, &row, (uint32_t) vcount[q], true,
                        (uint32_t) pr_call, total_packed);
                    if (r == LLAMA_FLASHPREFILL_ROUTE_DENSE_TAIL ||
                        r == LLAMA_FLASHPREFILL_ROUTE_DENSE_SHORT_CONTEXT) {
                        forced = true;
                    } else if (r != LLAMA_FLASHPREFILL_ROUTE_SPARSE &&
                               r != LLAMA_FLASHPREFILL_ROUTE_EXACT_ALL) {
                        return fail("flashprefill pack: unexpected per-row route");
                    }
                }
            }
            forced_flat.push_back(forced ? 1 : 0);
            if (forced) {
                ++forced_rows;
            } else {
                ++sparse_rows;
            }
            if (dry_run) {
                continue;
            }
            llm_fp_rowrec rec;
            rec.domain  = (int32_t) q_domain[(size_t) q];
            rec.src_q   = (int32_t) q;
            rec.kv_head = (int32_t) kh;
            rec.q_head  = (int32_t) h;
            rec.tile    = tile;
            rec.log_pos = row.logical_pos;
            rec.pbegin  = row.prefill_begin;
            rec.pend    = row.prefill_end;
            rec.forced  = forced;
            rows_u.push_back(rec);
        }
    }
    out->sparse_rows = sparse_rows;
    out->forced_rows = forced_rows;
    if (dry_run) {
        // Eligibility probe: route/tail outcome only (counts + visible +
        // tiles already recorded), no wire emission.
        return LL_FP_PACK_OK;
    }
    // Per-(query,tile,kv_head) forced triples: any forced row forces every
    // use of its triple to MANDATORY (wire: triples touching a DENSE_FORCE
    // row are all-mandatory).
    std::map<std::tuple<int32_t, int32_t, int32_t>, bool> triple_forced;
    for (const auto & r : rows_u) {
        if (r.forced) {
            triple_forced[{r.src_q, r.tile, r.kv_head}] = true;
        }
    }
    // Cell table: fragments concatenated in frag_id order.
    std::vector<int32_t> cells;
    std::vector<uint32_t> frag_base(F, 0);
    {
        uint64_t base = 0;
        for (uint32_t f = 0; f < F; ++f) {
            const auto & fr = layout.fragments[f];
            if (base + fr.token_count > (uint64_t) INT32_MAX) {
                if (error != nullptr) {
                    *error = "flashprefill pack: cells exceed I32";
                }
                return LL_FP_PACK_OVERFLOW;
            }
            frag_base[f] = (uint32_t) base;
            if (fr.contiguous) {
                for (uint32_t i = 0; i < fr.token_count; ++i) {
                    const uint64_t cell = (uint64_t) fr.cell_begin + i;
                    if (cell > (uint64_t) INT32_MAX) {
                        return fail("flashprefill pack: cell out of I32 range");
                    }
                    cells.push_back((int32_t) cell);
                }
            } else {
                for (uint32_t i = 0; i < fr.token_count; ++i) {
                    const uint32_t cell = layout.cell_refs[(size_t) fr.cell_ref_offset + i];
                    if ((uint64_t) cell > (uint64_t) INT32_MAX) {
                        return fail("flashprefill pack: cell ref out of I32 range");
                    }
                    cells.push_back((int32_t) cell);
                }
            }
            base += fr.token_count;
        }
    }
    // Use fan-out: one wire use per (use, distinct tile, kv head). Heads of
    // one kv group share a use only when their packed rows share the tile
    // (triple key); the sort below groups them for the selector.
    std::vector<llm_fp_userec> uses_u;
    uses_u.reserve(layout.uses.size() > 0 ? layout.uses.size() : 1);
    for (uint32_t q = 0; q < Q; ++q) {
        const uint32_t b = layout.use_offsets[q];
        const uint32_t e = layout.use_offsets[q + 1];
        for (uint32_t kh = 0; kh < hkv; ++kh) {
            // Exact tile range for this (query, kv-head) pair: its gqa
            // packed rows are consecutive, so touched tiles run contiguously
            // first..last in the same domain numbering as the row records
            // above (no array bound, no fail-closed truncation).
            // KV heads are a separate plan axis. Every KV head uses the
            // same query-local subhead range [0, gqa), as the row records
            // above do via h % gqa. Adding kh*gqa shifts uses away from
            // their rows (and can create tiles past n_tiles).
            const int64_t plo = q_tile_base[(size_t) q] +
                q_local[(size_t) q] / (int64_t) cfg.block_q;
            const int64_t last_s = (int64_t) gqa - 1;
            const int64_t phi = q_tile_base[(size_t) q] +
                (q_local[(size_t) q] + last_s) / (int64_t) cfg.block_q;
            if (plo > phi || plo > (int64_t) INT32_MAX || phi > (int64_t) INT32_MAX) {
                return fail("flashprefill pack: tile range out of I32 range");
            }
            for (int64_t t = plo; t <= phi; ++t) {
                const int32_t tile = (int32_t) t;
            for (uint32_t i = b; i < e; ++i) {
                const auto & u = layout.uses[i];
                [[maybe_unused]] const auto & fr = layout.fragments[u.fragment];
                const uint64_t wsub = (uint64_t) frag_base[u.fragment] + (uint64_t) u.sub_off;
                if (wsub > (uint64_t) INT32_MAX ||
                    (uint64_t) u.sub_count > (uint64_t) INT32_MAX) {
                    return fail("flashprefill pack: wire subrange out of I32 range");
                }
                if ((int64_t) u.group >= (use_groups ? (int64_t) G : (int64_t) Q)) {
                    return fail("flashprefill pack: use group out of range");
                }
                llm_fp_userec w;
                w.domain    = (int32_t) q_domain[(size_t) q];
                w.frag      = (int32_t) u.fragment;
                w.tile      = tile;
                w.kv_head   = (int32_t) kh;
                w.q_group   = use_groups ? (int32_t) u.group : (int32_t) q;
                w.sub_off   = (int64_t) wsub;
                w.sub_count = (int64_t) u.sub_count;
                w.flags     = (int32_t) u.flags;
                w.src_q     = (int32_t) q;
                auto it = triple_forced.find({w.src_q, w.tile, w.kv_head});
                if (it != triple_forced.end() && it->second) {
                    w.flags |= (int32_t) llama_flashprefill_use::FLAG_MANDATORY;
                }
                uses_u.push_back(w);
            }
            }
        }
    }
    if (uses_u.size() > (size_t) INT32_MAX || rows_u.size() > (size_t) INT32_MAX ||
        cells.size() > (size_t) INT32_MAX) {
        if (error != nullptr) {
            *error = "flashprefill pack: rows/uses/cells exceed I32";
        }
        return LL_FP_PACK_OVERFLOW;
    }
    // Canonical pack order (frozen with the selector): domain first, then
    // (tile, kv_head), secondary (source_query, fragment_id / q_head).
    // Tiles are disjoint across domains by construction (verified below),
    // so the selector's (tile,kv_head) binary search stays valid.
    // Descriptors only; Q rows, outputs and recurrent state never move.
    std::sort(uses_u.begin(), uses_u.end(), [](const llm_fp_userec & a, const llm_fp_userec & b) {
        if (a.domain  != b.domain)  return a.domain  < b.domain;
        if (a.tile    != b.tile)    return a.tile    < b.tile;
        if (a.kv_head != b.kv_head) return a.kv_head < b.kv_head;
        if (a.src_q   != b.src_q)   return a.src_q   < b.src_q;
        return a.frag < b.frag;
    });
    std::sort(rows_u.begin(), rows_u.end(), [](const llm_fp_rowrec & a, const llm_fp_rowrec & b) {
        if (a.domain  != b.domain)  return a.domain  < b.domain;
        if (a.tile    != b.tile)    return a.tile    < b.tile;
        if (a.kv_head != b.kv_head) return a.kv_head < b.kv_head;
        if (a.src_q   != b.src_q)   return a.src_q   < b.src_q;
        return a.q_head < b.q_head;
    });
    // Selection-privacy verification: no tile is ever shared across
    // visibility domains (pollution isolation). Construction guarantees it;
    // this sweep rejects any violation fail-closed, every mode.
    {
        std::map<int32_t, int32_t> tile_domain;
        for (const auto & r : rows_u) {
            auto it = tile_domain.find(r.tile);
            if (it == tile_domain.end()) {
                tile_domain[r.tile] = r.domain;
            } else if (it->second != r.domain) {
                return fail("flashprefill pack: tile shared across domains");
            }
        }
        for (const auto & u : uses_u) {
            auto it = tile_domain.find(u.tile);
            if (it == tile_domain.end()) {
                tile_domain[u.tile] = u.domain;
            } else if (it->second != u.domain) {
                return fail("flashprefill pack: use tile shared across domains");
            }
        }
    }
    // Per-(tile,head) use counts: plan max_sel_pair is the worst actual pair.
    int64_t max_sel = 0;
    {
        size_t i = 0;
        while (i < uses_u.size()) {
            size_t j = i + 1;
            while (j < uses_u.size() &&
                   uses_u[j].tile == uses_u[i].tile &&
                   uses_u[j].kv_head == uses_u[i].kv_head) {
                ++j;
            }
            const int64_t cnt = (int64_t) (j - i);
            if (cnt > max_sel) {
                max_sel = cnt;
            }
            i = j;
        }
        if (max_sel > (int64_t) INT32_MAX) {
            if (error != nullptr) {
                *error = "flashprefill pack: pair count exceeds I32";
            }
            return LL_FP_PACK_OVERFLOW;
        }
    }
    // Group RoPE positions (rerot only; ordinary uses identity + inp_pos).
    std::vector<int32_t> qpos;
    if (use_groups) {
        qpos.assign(G, 0);
        for (uint32_t g = 0; g < G; ++g) {
            const int64_t p = (int64_t) layout.groups[g].effective_pos;
            if (p < (int64_t) INT32_MIN || p > (int64_t) INT32_MAX) {
                return fail("flashprefill pack: group pos out of I32 range");
            }
            qpos[g] = (int32_t) p;
        }
    }
    out->rows.swap(rows_u);
    out->uses.swap(uses_u);
    out->cells.swap(cells);
    out->frag_base.swap(frag_base);
    out->max_sel = max_sel;
    out->n_groups = use_groups ? (int32_t) G : (int32_t) Q;
    out->qpos.swap(qpos);
    out->sparse_rows = sparse_rows;
    out->forced_rows = forced_rows;
    return LL_FP_PACK_OK;
}

// ---- build prescreen (no layout build): shapes/roles/config/global gates ----

enum class llm_fp_pre {
    WANT,        // proceed to layout build
    DENSE,       // designed-dense (silent, every mode)
    UNSUPPORTED, // backend only: caller maps by mode (throw REQUIRED, dense AUTO)
};

static llm_fp_pre llm_fp_prescreen(
        const llama_ubatch & ubatch,
        const llama_hparams & hparams,
        const llama_cparams & cparams,
        ggml_backend_sched_t sched,
        const std::vector<llama_flashprefill_row> * rows,
        int32_t * role_out,
        bool * backend_out,
        int32_t * reason_out = nullptr) {
    auto note = [&](int32_t reason) {
        if (reason_out != nullptr) {
            *reason_out = reason;
        }
    };
    if (role_out != nullptr) {
        *role_out = LLAMA_FLASHPREFILL_ROLE_UNKNOWN;
    }
    if (backend_out != nullptr) {
        *backend_out = false;
    }
    if (rows == nullptr) {
        note(LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF);
        return llm_fp_pre::DENSE;
    }
    if (rows->size() != (size_t) ubatch.n_tokens || ubatch.n_tokens == 0) {
        throw std::runtime_error("flashprefill: row snapshot size != ubatch rows (corrupt source map)");
    }
    const uint32_t hq  = hparams.n_head();
    const uint32_t hkv = hparams.n_head_kv();
    if (hq == 0 || hkv == 0 || hq % hkv != 0) {
        note(LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED);
        return llm_fp_pre::DENSE;
    }
    if (hparams.n_embd_head_k() == 0 || hparams.n_embd_head_v() == 0) {
        note(LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED);
        return llm_fp_pre::DENSE;
    }
    const int32_t role = llm_fp_uniform_role(*rows);
    if (role == LLAMA_FLASHPREFILL_ROLE_UNKNOWN) {
        // Mixed or non-eligible roles (decode/MTP/verify/frontier/
        // embedding/...): conservative dense, never reordered.
        note(LLAMA_FLASHPREFILL_ROUTE_DENSE_ROLE);
        return llm_fp_pre::DENSE;
    }
    // Bounds are required to pack honest wire rows in every tail scope;
    // unknown live boundaries route dense, never guessed.
    for (const auto & r : *rows) {
        if (!r.prefill_known) {
            note(LLAMA_FLASHPREFILL_ROUTE_DENSE_UNKNOWN_BOUNDARY);
            return llm_fp_pre::DENSE;
        }
        if (r.prefill_begin >= r.prefill_end) {
            throw std::runtime_error("flashprefill: invalid prefill interval (corrupt rows)");
        }
        if (r.logical_pos != LLAMA_FLASHPREFILL_POS_UNKNOWN &&
            (r.logical_pos < r.prefill_begin || r.logical_pos >= r.prefill_end)) {
            throw std::runtime_error("flashprefill: logical pos outside interval (corrupt rows)");
        }
    }
    bool supported = true;
    llm_fp_backend_variant(sched, &supported);
    if (backend_out != nullptr) {
        *backend_out = supported;
    }
    if (!supported) {
        return llm_fp_pre::UNSUPPORTED;
    }
    // Designed-dense global gates (silent, every mode): MTP contexts,
    // embedding/pooling paths, ALiBi-style max bias, multi-pos batches.
    // Recurrent layers and the MTP draft graph never reach try_build.
    if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
        note(LLAMA_FLASHPREFILL_ROUTE_DENSE_ROLE);
        return llm_fp_pre::DENSE;
    }
    if (cparams.embeddings || cparams.pooling_type != LLAMA_POOLING_TYPE_NONE || ubatch.embd != nullptr) {
        note(LLAMA_FLASHPREFILL_ROUTE_DENSE_ROLE);
        return llm_fp_pre::DENSE;
    }
    if (hparams.f_max_alibi_bias != 0.0f) {
        note(LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED);
        return llm_fp_pre::DENSE;
    }
    // Single-pos and IMRoPE 4-pos batches (text (p,p,p,0) discipline); wider
    // position layouts keep the pre-existing dense path.
    if (hparams.n_pos_per_embd() != 1 && hparams.n_pos_per_embd() != 4) {
        note(LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED);
        return llm_fp_pre::DENSE;
    }
    if (role_out != nullptr) {
        *role_out = role;
    }
    return llm_fp_pre::WANT;
}

static bool llm_fp_is_required(const llama_flashprefill_config & cfg) {
    return cfg.mode == LLAMA_FLASHPREFILL_MODE_REQUIRED;
}

// ---- companion factory ----

// Whole-ubatch dense decision with no companion: records the bounded reason
// plus full-layer dense counts into the result summary (so metrics never
// mislabels a plan-less graph), allocates nothing, syncs nothing.
static std::unique_ptr<llm_graph_input_attn_flashprefill> llm_fp_deny(
        llm_graph_result * res, const llama_hparams & hparams,
        const llama_ubatch & ubatch, int32_t reason) {
    if (res != nullptr) {
        // Authoritative per-ubatch route data even with zero plans, so
        // Metrics never re-derives from physical KV size: short sequences
        // surrounded by idle KV report their own small shape + SHORT reason,
        // not the global cache width. Visible/candidates stay 0 (no layout
        // built); the reason explains why.
        res->flashprefill_summary.dense_reason = reason;
        res->flashprefill_summary.dense_layers = llm_fp_count_full_layers(hparams);
        res->flashprefill_summary.ubatch_tokens = (int32_t) ubatch.n_tokens;
        if (reason == LLM_FP_DENSE_FULL_PREFIX) {
            // Actual handled rows for the FULL_PREFIX bucket (generic layer
            // loop over eligible full layers), so the dense total grows even
            // with no companion and zero plans. Never physical-KV derived.
            int64_t dr = 0;
            for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
                if (llm_fp_layer_eligible(hparams, (int) il)) {
                    dr += (int64_t) ubatch.n_tokens * (int64_t) hparams.n_head(il);
                }
            }
            res->flashprefill_summary.designed_dense_rows = dr;
        }
    }
    return nullptr;
}

std::unique_ptr<llm_graph_input_attn_flashprefill> llm_graph_input_attn_flashprefill::build_if_wanted(
        ggml_context * ctx0,
        const llama_ubatch & ubatch,
        const llama_hparams & hparams,
        const llama_cparams & cparams,
        const llama_kv_cache_context * mctx_cur,
        ggml_backend_sched_t sched,
        bool reserve_sizing,
        llm_graph_result * res) {
    const auto t0 = std::chrono::steady_clock::now();
    const llama_flashprefill_config & cfg = cparams.flashprefill;
    if (!llm_fp_is_enabled(cparams) || mctx_cur == nullptr || ubatch.n_tokens == 0) {
        return llm_fp_deny(res, hparams, ubatch, LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF);
    }
    const auto * rows = llm_fp_rows_of(cparams);
    const bool snapshot = (rows != nullptr) && llm_fp_is_snapshot_rows(*rows, ubatch);
    const bool want_reserve = reserve_sizing || snapshot;
    const uint32_t hq  = hparams.n_head();
    const uint32_t hkv = hparams.n_head_kv();
    if (hq == 0 || hkv == 0 || hq % hkv != 0) {
        return llm_fp_deny(res, hparams, ubatch, LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED);
    }
    const uint32_t gqa = hq / hkv;
    // First full-attention layer fixes the uniform head dims for the key;
    // per-layer enforcement in try_build keeps non-uniform models dense.
    int il_first = -1;
    for (int il = 0; (uint32_t) il < hparams.n_layer(); ++il) {
        if (!hparams.is_recr(il) && !hparams.is_swa(il)) {
            il_first = il;
            break;
        }
    }
    if (il_first < 0) {
        return llm_fp_deny(res, hparams, ubatch, LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF); // no full-attention layer
    }
    // Reserve probe inputs (pure; also reused for the live-vs-reserve fit).
    uint32_t n_ctx_cells = 0, q_rsv = 0;
    if (!llm_fp_reserve_probe_inputs(cparams, hparams, ubatch, &n_ctx_cells, &q_rsv)) {
        return llm_fp_deny(res, hparams, ubatch, LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF);
    }
    llm_fp_reserve_caps rsv;
    std::string rsv_err;
    if (!llm_fp_reserve_caps_for(cfg, n_ctx_cells, q_rsv, hq, hkv, gqa, &rsv, &rsv_err)) {
        throw std::runtime_error(std::string("flashprefill reserve: ") + rsv_err);
    }
    // Metadata dimensions are shared; storage types are per layer. In
    // particular, Turbo boundary layers may store Q8_0 V while interior
    // layers store Turbo2. That is not a dtype drift or a corrupt cache.
    ggml_tensor * k_probe = mctx_cur->get_k(ctx0, il_first);
    ggml_tensor * v_probe = mctx_cur->get_v(ctx0, il_first);
    const int32_t dk = (int32_t) k_probe->ne[0];
    const int32_t dv = (int32_t) v_probe->ne[0];
    const int32_t k_type = (int32_t) k_probe->type;
    const int32_t v_type = (int32_t) v_probe->type;
    if (dk <= 0 || dv <= 0) {
        // Degenerate cache dims are corruption, not policy: always loud,
        // so the can_reuse probe can never disagree with the build.
        throw std::runtime_error("flashprefill: degenerate KV head dims (corrupt cache)");
    }
    const int variant = llm_fp_backend_variant(sched, nullptr);
    auto out = std::make_unique<llm_graph_input_attn_flashprefill>(hparams, cparams, mctx_cur);
    out->cur_rows = cparams.flashprefill_rows;
    out->key.layer_kv_types.assign(hparams.n_layer(), {-1, -1});
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (llm_fp_layer_eligible(hparams, (int) il)) {
            out->key.layer_kv_types[il] = {
                (int32_t) mctx_cur->layer_type_k((int32_t) il),
                (int32_t) mctx_cur->layer_type_v((int32_t) il)};
        }
    }
    if (want_reserve) {
        // Reserve sizing: worst-case caps, zero counts, never runs. No
        // layout build (the synthetic ubatch was never applied); unknown
        // boundaries are sized, never guessed known. Tensor and key share
        // identical bucketed caps (never raw vs bucketed mixes).
        const uint32_t rf = llm_fp_bucket(rsv.f);
        const uint32_t rr = llm_fp_bucket(rsv.r);
        const uint32_t ru = llm_fp_bucket(rsv.u);
        const uint32_t rc = llm_fp_bucket(rsv.c);
        int64_t meta_words = 0;
        if (ggml_flashprefill_metadata_words(rf, rr, ru, rc, &meta_words) != GGML_FLASHPREFILL_OK) {
            throw std::runtime_error("flashprefill reserve: metadata words overflow");
        }
        out->meta = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, meta_words);
        ggml_set_input(out->meta);
        out->st_meta.assign((size_t) meta_words, 0);
        if (ggml_flashprefill_metadata_init(out->st_meta.data(), meta_words,
                    (int64_t) rf, (int64_t) rr, (int64_t) ru, (int64_t) rc,
                    dk, dv, (int32_t) hkv, (int32_t) q_rsv, (int32_t) hq) != GGML_FLASHPREFILL_OK) {
            throw std::runtime_error("flashprefill reserve: metadata init failed");
        }
        if (ggml_flashprefill_metadata_set_counts(out->st_meta.data(), meta_words, 0, 0, 0, 0) != GGML_FLASHPREFILL_OK) {
            throw std::runtime_error("flashprefill reserve: zero counts failed");
        }
        out->key.mode = cfg.mode;
        out->key.tail_scope = cfg.tail_scope;
        out->key.block_q = cfg.block_q;
        out->key.block_k = cfg.block_k;
        out->key.sink_blocks = cfg.sink_blocks;
        out->key.window_blocks = cfg.window_blocks;
        out->key.dense_tail_tiles = cfg.dense_tail_tiles;
        out->key.min_kv = cfg.min_kv;
        out->key.full_attn_layers = cfg.full_attn_layers;
        out->key.mean_correction = cfg.mean_correction;
        out->key.exact_all = cfg.exact_all;
        out->key.role = LLAMA_FLASHPREFILL_ROLE_PREFILL;
        out->key.is_rerot = false;
        out->key.reserve_sizing = true;
        out->key.n_tokens = q_rsv;
        out->key.f_cap = rf;
        out->key.r_cap = rr;
        out->key.u_cap = ru;
        out->key.c_cap = rc;
        out->key.n_tiles = rsv.t;
        out->key.max_sel_pair = llm_fp_bucket(rsv.maxsel);
        out->key.u_layout_cap = llm_fp_bucket(rsv.u);
        out->key.n_groups = q_rsv;
        out->key.dk = dk;
        out->key.dv = dv;
        out->key.n_kv_heads = (int32_t) hkv;
        out->key.n_q_heads = (int32_t) hq;
        out->key.k_type = k_type;
        out->key.v_type = v_type;
        out->key.n_pos = (int32_t) hparams.n_pos_per_embd();
        out->key.backend_variant = variant;
        out->key_valid = true;
        out->built_summary.n_tiles = (int32_t) rsv.t;
        out->built_summary.f_cap = (int32_t) out->key.f_cap;
        out->built_summary.r_cap = (int32_t) out->key.r_cap;
        out->built_summary.u_cap = (int32_t) out->key.u_cap;
        out->built_summary.c_cap = (int32_t) out->key.c_cap;
        out->built_summary.max_sel_pair = (int32_t) rsv.maxsel;
        out->built_summary.n_groups = 0; // reserve never runs: counts stay zero, caps above
        out->built_summary.ubatch_tokens = (int32_t) ubatch.n_tokens;
        out->built_summary.scratch_bytes = (uint64_t) ggml_nbytes(out->meta); // shared meta only
        const auto t1 = std::chrono::steady_clock::now();
        out->stat_layout_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        return out;
    }
    // Live path.
    int32_t role = LLAMA_FLASHPREFILL_ROLE_UNKNOWN;
    bool backend_ok = false;
    int32_t pre_reason = LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF;
    const llm_fp_pre pre = llm_fp_prescreen(ubatch, hparams, cparams, sched, rows, &role, &backend_ok, &pre_reason);
    if (pre == llm_fp_pre::DENSE) {
        return llm_fp_deny(res, hparams, ubatch, pre_reason);
    }
    if (pre == llm_fp_pre::UNSUPPORTED) {
        if (llm_fp_is_required(cfg)) {
            throw std::runtime_error("flashprefill: eligible prefill has no sparse backend (required)");
        }
        return llm_fp_deny(res, hparams, ubatch, LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED);
    }
    // NOTE: no min_kv gate here on global padded get_n_kv — idle KV is not
    // visible coverage. The honest visible gate runs post-pack on
    // layout-derived incidences below, mirroring the can_reuse probe.
    // Full-prefix early deny (orphan-input guard): full_attn_layers covers
    // every eligible full layer, so no layer could consume a companion.
    // Deny with FULL_PREFIX instead of building metadata no op references
    // (the scheduler would never allocate it). Designed dense in every
    // mode, including REQUIRED.
    {
        const int32_t n_full = llm_fp_count_full_layers(hparams);
        if (n_full > 0 && (uint64_t) n_full <= (uint64_t) cfg.full_attn_layers) {
            return llm_fp_deny(res, hparams, ubatch, LLM_FP_DENSE_FULL_PREFIX);
        }
    }
    const llama_flashprefill_layout_params lp = llm_fp_layout_params_for(cfg, cparams);
    std::string build_err;
    if (!mctx_cur->flashprefill_build_current_layout(role, lp, &build_err)) {
        const std::string why = mctx_cur->flashprefill_get_layout().error;
        if (!why.empty()) {
            throw std::runtime_error(std::string("flashprefill layout: ") + why);
        }
        if (llm_fp_is_required(cfg)) {
            throw std::runtime_error("flashprefill: eligible prefill has no sparse layout (required)");
        }
        return llm_fp_deny(res, hparams, ubatch, LLM_FP_DENSE_NO_LAYOUT);
    }
    const llama_flashprefill_layout & layout = mctx_cur->flashprefill_get_layout();
    if (!layout.eligible) {
        if (!layout.error.empty()) {
            throw std::runtime_error(std::string("flashprefill layout: ") + layout.error);
        }
        if (llm_fp_is_required(cfg)) {
            throw std::runtime_error("flashprefill: eligible prefill has no sparse layout (required)");
        }
        return llm_fp_deny(res, hparams, ubatch, LLM_FP_DENSE_NO_LAYOUT);
    }
    {
        std::string verr;
        if (!layout.validate(layout.n_kv_at_build, &verr)) {
            throw std::runtime_error(std::string("flashprefill layout invalid: ") + verr);
        }
    }
    // Admission bound (resource invariant): the pool must stay summary-scale,
    // never a full F32 KV mirror (Fworst == K is forbidden). Admitted shapes
    // are bounded by the fragment budget at ACTUAL resident K with the same
    // seam/run allowances as reserve (4 membership/phase/visibility seams
    // per BN block + 65 run heads + base); at BN=128 the worst admitted pool
    // is ~4/128 of the resident KV token count times the F32/quant width
    // ratio. Beyond bound: dense AUTO (high estimated cost), explicit
    // resource error REQUIRED. All legal refs retained via dense fallback;
    // descriptors are never truncated. Reserve uses the identical bound at
    // n_ctx probe width, so live admission implies reserve fit.
    {
        uint32_t n_streams = 1;
        for (const auto & fr : layout.fragments) {
            if (fr.stream + 1 > n_streams) {
                n_streams = fr.stream + 1;
            }
        }
        llama_flashprefill_fragment_budget admit;
        std::string admit_err;
        if (!llama_flashprefill_admission_budget_for(layout.n_kv_at_build, cfg.block_k,
                    n_streams, ubatch.n_tokens, &admit, &admit_err)) {
            if (llm_fp_is_required(cfg)) {
                throw std::runtime_error(std::string("flashprefill admission overflow (required): ") + admit_err);
            }
            return llm_fp_deny(res, hparams, ubatch, LLAMA_FLASHPREFILL_ROUTE_DENSE_CAPACITY);
        }
        if (layout.n_fragments() > admit.n_fragments) {
            if (llm_fp_is_required(cfg)) {
                throw std::runtime_error("flashprefill: fragment count beyond admitted bound (required, resource)");
            }
            return llm_fp_deny(res, hparams, ubatch, LLM_FP_DENSE_HIGH_COST); // highly fragmented: dense keeps every ref
        }
    }
    if (layout.block_k != cfg.block_k || layout.n_queries() != ubatch.n_tokens) {
        throw std::runtime_error("flashprefill: layout/policy shape mismatch (stale layout)");
    }
    llm_fp_pack pack;
    std::string pack_err;
    const llm_fp_pack_rc prc = llm_fp_pack_live(layout, *rows, cfg, hq, hkv, gqa, ubatch.n_tokens, &pack, &pack_err);
    if (prc == LL_FP_PACK_CORRUPT) {
        throw std::runtime_error(std::string("flashprefill pack: ") + pack_err);
    }
    if (prc == LL_FP_PACK_OVERFLOW) {
        if (llm_fp_is_required(cfg)) {
            throw std::runtime_error(std::string("flashprefill pack overflow (required): ") + pack_err);
        }
        return llm_fp_deny(res, hparams, ubatch, LLAMA_FLASHPREFILL_ROUTE_DENSE_CAPACITY);
    }
    // The layout scans the physical cache capacity, whereas attention views
    // end at the padded resident high-water mark. They need not be equal
    // (e.g. 512 allocated cells, a 256-wide view, and 128 live cells).
    // Every referenced cell must nevertheless fit the actual tensor view.
    const uint32_t view_n_kv = mctx_cur->get_n_kv();
    for (const auto cell : pack.cells) {
        if (cell < 0 || (uint64_t) cell >= view_n_kv) {
            throw std::runtime_error("flashprefill: packed cell outside KV tensor view");
        }
    }
    // Visible gate on layout-derived incidences (pack.visible_tokens), never
    // the global padded get_n_kv: short sequences surrounded by idle KV
    // report their own small shape. Mirrors the can_reuse probe exactly.
    {
        const uint32_t vis32 = pack.visible_tokens > (uint64_t) UINT32_MAX
            ? UINT32_MAX : (uint32_t) pack.visible_tokens;
        const auto rr = llama_flashprefill::route_for_role(&cfg, role, vis32, backend_ok);
        if (rr != LLAMA_FLASHPREFILL_ROUTE_SPARSE && rr != LLAMA_FLASHPREFILL_ROUTE_EXACT_ALL) {
            return llm_fp_deny(res, hparams, ubatch, rr);
        }
    }
    // All-forced (zero sparse rows, no exact_all): every row is exact-tail,
    // so the new path would do dense work at sparse overhead. Stay dense
    // like the probe; exact_all keeps the new path (rows are EXACT_ALL).
    if (pack.sparse_rows == 0 && !cfg.exact_all) {
        return llm_fp_deny(res, hparams, ubatch, LLAMA_FLASHPREFILL_ROUTE_DENSE_TAIL);
    }
    // Bucketed caps; live actuals must fit the reserve measurement or the
    // submit could not have been reserved (dense AUTO, throw REQUIRED).
    const uint32_t f_cap = llm_fp_bucket((uint32_t) layout.n_fragments());
    const uint32_t r_cap = llm_fp_bucket((uint32_t) pack.rows.size());
    const uint32_t u_cap = llm_fp_bucket((uint32_t) pack.uses.size());
    const uint32_t c_cap = llm_fp_bucket((uint32_t) pack.cells.size());
    const uint32_t maxsel_cap = llm_fp_bucket((uint32_t) pack.max_sel);
    const uint32_t g_cap = llm_fp_bucket((uint32_t) pack.n_groups);
    const uint32_t u_layout_cap = llm_fp_bucket(layout.n_uses());
    if (f_cap > llm_fp_bucket(rsv.f) || r_cap > llm_fp_bucket(rsv.r) ||
        u_cap > llm_fp_bucket(rsv.u) || c_cap > llm_fp_bucket(rsv.c)) {
        if (llm_fp_is_required(cfg)) {
            throw std::runtime_error("flashprefill: live caps exceed reserve (required, DENSE_CAPACITY)");
        }
        return llm_fp_deny(res, hparams, ubatch, LLAMA_FLASHPREFILL_ROUTE_DENSE_CAPACITY);
    }
    int64_t meta_words = 0;
    if (ggml_flashprefill_metadata_words(f_cap, r_cap, u_cap, c_cap, &meta_words) != GGML_FLASHPREFILL_OK) {
        if (llm_fp_is_required(cfg)) {
            throw std::runtime_error("flashprefill: metadata words overflow (required)");
        }
        return llm_fp_deny(res, hparams, ubatch, LLAMA_FLASHPREFILL_ROUTE_DENSE_CAPACITY);
    }
    const bool has_groups = !pack.qpos.empty();
    const uint32_t n_tiles = (uint32_t) pack.n_tiles;
    // Group positions use the native DDVR pairing: q_pos is [g_cap, n_pos]
    // flattened with coordinate k at k*g_cap (IMRoPE text rule (p,p,p,0) for
    // n_pos==4); the Q gather covers the full capacity with zero tails.
    const int64_t n_pos = (int64_t) hparams.n_pos_per_embd();
    if (has_groups && n_pos != 1 && n_pos != 4) {
        throw std::runtime_error("flashprefill: group positions need n_pos 1 or 4");
    }
    out->meta = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, meta_words);
    ggml_set_input(out->meta);
    if (has_groups) {
        out->q_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) g_cap * n_pos);
        ggml_set_input(out->q_pos);
        out->q_gather = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) g_cap);
        ggml_set_input(out->q_gather);
    }
    out->st_meta.assign((size_t) meta_words, 0);
    if (has_groups) {
        out->st_qpos.assign((size_t) g_cap * (size_t) n_pos, 0);
        out->st_gather.assign((size_t) g_cap, 0);
    }
    int32_t * md = out->st_meta.data();
    if (ggml_flashprefill_metadata_init(md, meta_words,
                (int64_t) f_cap, (int64_t) r_cap, (int64_t) u_cap, (int64_t) c_cap,
                dk, dv, (int32_t) hkv, pack.n_groups, (int32_t) hq) != GGML_FLASHPREFILL_OK) {
        throw std::runtime_error("flashprefill: metadata init failed (corrupt caps)");
    }
    if (ggml_flashprefill_metadata_set_counts(md, meta_words,
                (int64_t) layout.n_fragments(), (int64_t) pack.rows.size(),
                (int64_t) pack.uses.size(), (int64_t) pack.cells.size()) != GGML_FLASHPREFILL_OK) {
        throw std::runtime_error("flashprefill: metadata counts failed (corrupt pack)");
    }
    for (uint32_t f = 0; f < layout.n_fragments(); ++f) {
        const auto & fr = layout.fragments[f];
        if (fr.logical_block > (uint32_t) INT32_MAX) {
            throw std::runtime_error("flashprefill: logical block out of I32 range");
        }
        // Wire fragments are uniform membership/phase: the global boundary
        // flag stays 0 even when the owner diagnostic (boundary_partial) is
        // set for some query. Partial visibility rides per-use subrange +
        // MANDATORY, from which selectors derive tile-level cand_full and
        // force partial columns. A global flag would wrongly force exact in
        // every tile, including later fully-legal ones (upstream call
        // selector breaks on large ubatches). The owner diagnostic is left
        // untouched (never erased); only the wire encoding stays uniform.
        const int32_t flags = 0;
        if (ggml_flashprefill_metadata_set_frag(md, meta_words, (int64_t) f,
                    (int64_t) pack.frag_base[f],
                    (int64_t) fr.token_count, (int32_t) fr.logical_block,
                    (int32_t) llama_flashprefill_layout::wire_domain(fr.domain), flags) != GGML_FLASHPREFILL_OK) {
            throw std::runtime_error("flashprefill: metadata frag failed (corrupt pack)");
        }
    }
    for (size_t i = 0; i < pack.rows.size(); ++i) {
        const auto & r = pack.rows[i];
        const int32_t flags = r.forced ? GGML_FLASHPREFILL_ROW_FLAG_DENSE_FORCE : 0;
        if (ggml_flashprefill_metadata_set_row(md, meta_words, (int64_t) i,
                    r.src_q, r.kv_head, r.log_pos, r.tile, r.pbegin, r.pend, flags, r.q_head) != GGML_FLASHPREFILL_OK) {
            throw std::runtime_error("flashprefill: metadata row failed (corrupt pack)");
        }
    }
    for (size_t i = 0; i < pack.uses.size(); ++i) {
        const auto & u = pack.uses[i];
        int32_t flags = u.flags;
        if (flags != GGML_FLASHPREFILL_USE_FLAG_MANDATORY && flags != 0) {
            throw std::runtime_error("flashprefill: metadata use flag corrupt");
        }
        if (ggml_flashprefill_metadata_set_use(md, meta_words, (int64_t) i,
                    u.frag, u.tile, u.kv_head, u.q_group, u.sub_off, u.sub_count, flags, u.src_q) != GGML_FLASHPREFILL_OK) {
            throw std::runtime_error("flashprefill: metadata use failed (corrupt pack)");
        }
    }
    for (size_t i = 0; i < pack.cells.size(); ++i) {
        if (ggml_flashprefill_metadata_set_cell(md, meta_words, (int64_t) i, pack.cells[i]) != GGML_FLASHPREFILL_OK) {
            throw std::runtime_error("flashprefill: metadata cell failed (corrupt pack)");
        }
    }
    if (has_groups) {
        // Native DDVR pairing: coordinate k lives at k*g_cap (IMRoPE text
        // rule (p,p,p,0) for n_pos==4); gather tails stay 0 (query 0,
        // deterministic, unreferenced by actual use groups).
        const size_t gc = (size_t) g_cap;
        for (int32_t g = 0; g < pack.n_groups; ++g) {
            const int32_t p = pack.qpos[(size_t) g];
            out->st_qpos[(size_t) g] = p;
            if (n_pos == 4) {
                out->st_qpos[gc + (size_t) g]         = p;
                out->st_qpos[gc * 2 + (size_t) g]     = p;
                out->st_qpos[gc * 3 + (size_t) g]     = 0;
            }
            out->st_gather[(size_t) g] = (int32_t) layout.groups[(size_t) g].query_index;
        }
    }
    out->key.mode = cfg.mode;
    out->key.tail_scope = cfg.tail_scope;
    out->key.block_q = cfg.block_q;
    out->key.block_k = cfg.block_k;
    out->key.sink_blocks = cfg.sink_blocks;
    out->key.window_blocks = cfg.window_blocks;
    out->key.dense_tail_tiles = cfg.dense_tail_tiles;
    out->key.min_kv = cfg.min_kv;
    out->key.full_attn_layers = cfg.full_attn_layers;
    out->key.mean_correction = cfg.mean_correction;
    out->key.exact_all = cfg.exact_all;
    out->key.role = role;
    out->key.is_rerot = layout.is_rerot != 0;
    out->key.reserve_sizing = false;
    out->key.n_tokens = ubatch.n_tokens;
    out->key.f_cap = f_cap;
    out->key.r_cap = r_cap;
    out->key.u_cap = u_cap;
    out->key.c_cap = c_cap;
    out->key.n_tiles = n_tiles;
    out->key.max_sel_pair = maxsel_cap;
    out->key.u_layout_cap = u_layout_cap;
    out->key.n_groups = (uint32_t) pack.n_groups;
    out->key.dk = dk;
    out->key.dv = dv;
    out->key.n_kv_heads = (int32_t) hkv;
    out->key.n_q_heads = (int32_t) hq;
    out->key.k_type = k_type;
    out->key.v_type = v_type;
    out->key.n_pos = (int32_t) hparams.n_pos_per_embd();
    out->key.backend_variant = variant;
    out->key_valid = true;
    out->built_cells_epoch = layout.cells_epoch;
    out->built_n_kv = view_n_kv;
    out->last_sparse_rows = pack.sparse_rows;
    out->last_forced_rows = pack.forced_rows;
    out->built_summary.n_tiles = (int32_t) n_tiles;
    out->built_summary.n_fragments = (int32_t) layout.n_fragments();
    out->built_summary.sparse_rows = pack.sparse_rows;
    out->built_summary.dense_rows = pack.forced_rows;
    out->built_summary.ubatch_tokens = (int32_t) ubatch.n_tokens;
    out->built_summary.visible_tokens = (int64_t) pack.visible_tokens;
    out->built_summary.expected_sparse_rows = pack.sparse_rows;
    out->built_summary.expected_forced_rows = pack.forced_rows;
    out->built_summary.n_rows = (int32_t) pack.rows.size();
    out->built_summary.n_uses = (int32_t) pack.uses.size();
    out->built_summary.n_cells = (int32_t) pack.cells.size();
    out->built_summary.n_groups = pack.n_groups;
    out->built_summary.max_sel_pair = (int32_t) pack.max_sel;
    out->built_summary.f_cap = (int32_t) f_cap;
    out->built_summary.r_cap = (int32_t) r_cap;
    out->built_summary.u_cap = (int32_t) u_cap;
    out->built_summary.c_cap = (int32_t) c_cap;
    // Shared-input bytes only (real tensor capacities; per-layer pinned
    // plans and live pools accumulate via record, never multiplied here).
    out->built_summary.scratch_bytes =
        (uint64_t) ggml_nbytes(out->meta) +
        (out->q_pos   != nullptr ? (uint64_t) ggml_nbytes(out->q_pos)   : 0u) +
        (out->q_gather != nullptr ? (uint64_t) ggml_nbytes(out->q_gather) : 0u);
    const auto t1 = std::chrono::steady_clock::now();
    out->stat_layout_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    return out;
}

bool llm_graph_input_attn_flashprefill::wants_companion(
        const llm_graph_params & params,
        const llama_kv_cache_context * attn) {
    // Had-none stability probe: answers whether these params would carry a
    // companion, so a missing companion reuses instead of rebuild-looping —
    // and so a fixed-capacity dense->sparse transition actually happens.
    // Eligibility can NOT depend on a layout cache populated only by graph
    // rebuilds (that deadlocks: dense reuses forever because no layout was
    // ever built). So when the cheap gates pass, this probe triggers the
    // shared owner build (build_current_layout; the later rebuild hits the
    // same keyed planning cache, no duplicate planning work), then applies
    // the same pure policy checks as the build path, with layout-derived
    // visible counts (never the global padded get_n_kv). Returns true only
    // for an actual new sparse route; short/tail/full-N/caps stay dense.
    // Hard errors return true to force a rebuild, where the build throws
    // loudly instead of hiding inside a dense reuse.
    const llama_cparams & cparams = params.cparams;
    const llama_ubatch & ubatch = params.ubatch;
    const llama_hparams & hparams = params.hparams;
    const llama_flashprefill_config & cfg = cparams.flashprefill;
    const bool required = llm_fp_is_required(cfg);
    if (!llm_fp_is_enabled(cparams) || attn == nullptr || ubatch.n_tokens == 0) {
        return false;
    }
    const auto * rows = llm_fp_rows_of(cparams);
    int32_t role = LLAMA_FLASHPREFILL_ROLE_UNKNOWN;
    bool backend_ok = false;
    if (llm_fp_prescreen(ubatch, hparams, cparams, params.sched, rows, &role, &backend_ok) != llm_fp_pre::WANT) {
        return false;
    }
    if (!backend_ok) {
        return required; // build would throw-required; force the rebuild so it does
    }
    const uint32_t hq  = hparams.n_head();
    const uint32_t hkv = hparams.n_head_kv();
    if (hq == 0 || hkv == 0 || hq % hkv != 0) {
        return false;
    }
    const uint32_t gqa = hq / hkv;
    {
        // Mirror the build's layer gates exactly (or dense reuses loop
        // against a denying build): no full layers at all, or the full
        // prefix covers everything — both deny with no companion.
        bool any_full = false;
        for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
            if (!hparams.is_recr(il) && !hparams.is_swa(il)) {
                any_full = true;
                break;
            }
        }
        if (!any_full) {
            return false;
        }
        const int32_t n_full = llm_fp_count_full_layers(hparams);
        if (n_full > 0 && (uint64_t) n_full <= (uint64_t) cfg.full_attn_layers) {
            return false; // full prefix covers everything: deny, no companion
        }
    }
    const llama_flashprefill_layout_params lp = llm_fp_layout_params_for(cfg, cparams);
    std::string build_err;
    if (!attn->flashprefill_build_current_layout(role, lp, &build_err)) {
        if (!attn->flashprefill_get_layout().error.empty()) {
            return true; // hard error: rebuild surfaces it loudly
        }
        return required; // ineligible layout: build throws-required, else stay dense
    }
    const llama_flashprefill_layout & layout = attn->flashprefill_get_layout();
    if (!layout.eligible) {
        if (!layout.error.empty()) {
            return true;
        }
        return required;
    }
    {
        std::string verr;
        if (!layout.validate(layout.n_kv_at_build, &verr)) {
            return true; // corrupt: rebuild throws loudly
        }
    }
    if (layout.block_k != cfg.block_k || layout.n_queries() != ubatch.n_tokens) {
        return true; // stale shape: rebuild throws loudly
    }
    // Admission bound (same call as build).
    {
        uint32_t n_streams = 1;
        for (const auto & fr : layout.fragments) {
            if (fr.stream + 1 > n_streams) {
                n_streams = fr.stream + 1;
            }
        }
        llama_flashprefill_fragment_budget admit;
        std::string admit_err;
        if (!llama_flashprefill_admission_budget_for(layout.n_kv_at_build, cfg.block_k,
                    n_streams, ubatch.n_tokens, &admit, &admit_err)) {
            return required;
        }
        if (layout.n_fragments() > admit.n_fragments) {
            return required;
        }
    }
    // Tail shape: all-forced (zero sparse rows) stays dense like the build
    // deny; anything else proceeds. Full dry pack (no wire emission).
    {
        llm_fp_pack probe;
        std::string probe_err;
        const llm_fp_pack_rc prc = llm_fp_pack_live(
            layout, *rows, cfg, hq, hkv, gqa, ubatch.n_tokens, &probe, &probe_err, true);
        if (prc == LL_FP_PACK_CORRUPT) {
            return true; // corrupt: rebuild throws loudly
        }
        if (prc == LL_FP_PACK_OVERFLOW) {
            return required;
        }
        // Visible gate on the dry pack's layout-derived incidences (the exact
        // value the build will see) — never the global padded get_n_kv, so
        // short sequences surrounded by idle KV report honestly.
        {
            const uint32_t vis32 = probe.visible_tokens > (uint64_t) UINT32_MAX
                ? UINT32_MAX : (uint32_t) probe.visible_tokens;
            const auto rr = llama_flashprefill::route_for_role(&cfg, role, vis32, backend_ok);
            if (rr != LLAMA_FLASHPREFILL_ROUTE_SPARSE && rr != LLAMA_FLASHPREFILL_ROUTE_EXACT_ALL) {
                return false;
            }
        }
        if (probe.sparse_rows == 0 && !cfg.exact_all) {
            return false;
        }
        // Capacity estimates vs reserve (same bounds as build): exceed keeps
        // the dense graph in AUTO, forces the rebuild-throw in REQUIRED.
        uint32_t n_ctx_cells = 0, q_rsv = 0;
        if (!llm_fp_reserve_probe_inputs(cparams, hparams, ubatch, &n_ctx_cells, &q_rsv)) {
            return false;
        }
        llm_fp_reserve_caps rsv;
        std::string rsv_err;
        if (!llm_fp_reserve_caps_for(cfg, n_ctx_cells, q_rsv, hq, hkv, gqa, &rsv, &rsv_err)) {
            return required;
        }
        // Same bucket discipline as build (monotonic: estimate-fits implies
        // actual-fits, so no rebuild loop; estimate-miss stays dense, safe).
        const uint64_t w_est = (uint64_t) layout.uses.size() * (uint64_t) hkv *
            (uint64_t) llm_fp_tile_fanout(gqa, cfg.block_k);
        const uint64_t c_est = (uint64_t) layout.n_fragments() * (uint64_t) cfg.block_k;
        const uint64_t r_exact = (uint64_t) ubatch.n_tokens * (uint64_t) hq;
        bool over = llm_fp_bucket(layout.n_fragments()) > llm_fp_bucket(rsv.f);
        over = over || r_exact > (uint64_t) UINT32_MAX ||
            llm_fp_bucket((uint32_t) r_exact) > llm_fp_bucket(rsv.r);
        over = over || w_est > (uint64_t) UINT32_MAX ||
            llm_fp_bucket((uint32_t) w_est) > llm_fp_bucket(rsv.u);
        over = over || c_est > (uint64_t) UINT32_MAX ||
            llm_fp_bucket((uint32_t) c_est) > llm_fp_bucket(rsv.c);
        if (over) {
            return required;
        }
    }
    return true;
}

bool llm_graph_input_attn_flashprefill::submit_guards_ok(const llama_ubatch & ubatch, std::string * error) const {
    // Cheap submit-time guards (freshness, key topology, n_kv, counts<=caps).
    // Full layout validation + checked packing ran once at plan generation;
    // the expensive GGML validator never runs per submit (oracle/tests only).
    auto fail = [&](const char * msg) {
        if (error != nullptr) {
            *error = msg;
        }
        return false;
    };
    if (!active() || mctx == nullptr) {
        return fail("flashprefill submit: inactive input");
    }
    if (ubatch.n_tokens != key.n_tokens && !key.reserve_sizing) {
        return fail("flashprefill submit: token count changed");
    }
    const llama_flashprefill_layout & layout = mctx->flashprefill_get_layout();
    if (!layout.eligible || !mctx->flashprefill_layout_is_fresh()) {
        return fail("flashprefill submit: stale layout");
    }
    if (mctx->get_n_kv() != built_n_kv) {
        return fail("flashprefill submit: n_kv drifted");
    }
    if (layout.n_queries() != ubatch.n_tokens) {
        return fail("flashprefill submit: query count changed");
    }
    if (layout.n_fragments() > key.f_cap) {
        return fail("flashprefill submit: fragments exceed cap");
    }
    // u_cap is allocated from the EXACT packed use count. A worst-case
    // fan-out estimate may exceed it for a valid fresh graph (GQA=6 with
    // BM=128 is one example). set_input repacks and checks every exact
    // count against its cap before any upload; do not reject using an
    // upper bound as though it were the actual number of writes.
    return true;
}

void llm_graph_input_attn_flashprefill::set_input(const llama_ubatch * ubatch) {
    if (!active() || key.reserve_sizing || ubatch == nullptr) {
        return; // reserve graphs never run; inactive graphs skip entirely
    }
    if (!used_by_graph) {
        // No sparse layer consumed the metadata (all-designed dense or AUTO
        // fallback): the scheduler never allocated it, so uploading would
        // hit a null buffer. Skip silently; the summary reason (FULL_PREFIX
        // / UNSUPPORTED) already describes the route.
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::string guard_err;
    if (!submit_guards_ok(*ubatch, &guard_err)) {
        throw std::runtime_error(std::string("flashprefill submit: ") + guard_err);
    }
    const llama_flashprefill_layout & layout = mctx->flashprefill_get_layout();
    const auto & rows_ptr = (cur_rows != nullptr) ? cur_rows : cparams.flashprefill_rows;
    if (rows_ptr == nullptr || rows_ptr->size() != (size_t) ubatch->n_tokens) {
        throw std::runtime_error("flashprefill submit: row snapshot unavailable (stale reuse)");
    }
    const uint32_t hq  = (uint32_t) key.n_q_heads;
    const uint32_t hkv = (uint32_t) key.n_kv_heads;
    const uint32_t gqa = hq / hkv;
    llm_fp_pack pack;
    std::string pack_err;
    const llm_fp_pack_rc prc = llm_fp_pack_live(
        layout, *rows_ptr, cparams.flashprefill, hq, hkv, gqa, ubatch->n_tokens, &pack, &pack_err);
    if (prc != LL_FP_PACK_OK) {
        // Submit cannot fall back (topology fixed): any pack failure is a
        // fail-closed throw, every mode. Reuse should have rebuilt first.
        throw std::runtime_error(std::string("flashprefill submit pack: ") + pack_err);
    }
    // Exact submit guards against the build-time caps/buckets.
    if ((uint32_t) pack.rows.size()  > key.r_cap ||
        (uint32_t) pack.uses.size()  > key.u_cap ||
        (uint32_t) pack.cells.size() > key.c_cap ||
        (uint32_t) pack.max_sel > key.max_sel_pair ||
        pack.n_tiles != (int64_t) key.n_tiles ||
        (uint32_t) pack.n_groups > ((q_pos != nullptr) ? (uint32_t) q_pos->ne[0] : (uint32_t) key.n_groups)) {
        throw std::runtime_error("flashprefill submit: repack exceeds build caps (stale reuse)");
    }
    std::fill(st_meta.begin(), st_meta.end(), 0);
    int32_t * md = st_meta.data();
    const int64_t meta_words = (int64_t) st_meta.size();
    if (ggml_flashprefill_metadata_init(md, meta_words,
                (int64_t) key.f_cap, (int64_t) key.r_cap, (int64_t) key.u_cap, (int64_t) key.c_cap,
                key.dk, key.dv, key.n_kv_heads, pack.n_groups, key.n_q_heads) != GGML_FLASHPREFILL_OK) {
        throw std::runtime_error("flashprefill submit: metadata init failed");
    }
    if (ggml_flashprefill_metadata_set_counts(md, meta_words,
                (int64_t) layout.n_fragments(), (int64_t) pack.rows.size(),
                (int64_t) pack.uses.size(), (int64_t) pack.cells.size()) != GGML_FLASHPREFILL_OK) {
        throw std::runtime_error("flashprefill submit: metadata counts failed");
    }
    for (uint32_t f = 0; f < layout.n_fragments(); ++f) {
        const auto & fr = layout.fragments[f];
        // Uniform wire fragments (see build site): global boundary flag 0;
        // partial visibility rides per-use MANDATORY. Owner flag untouched.
        const int32_t flags = 0;
        if (ggml_flashprefill_metadata_set_frag(md, meta_words, (int64_t) f,
                    (int64_t) pack.frag_base[f], (int64_t) fr.token_count,
                    (int32_t) fr.logical_block,
                    (int32_t) llama_flashprefill_layout::wire_domain(fr.domain), flags) != GGML_FLASHPREFILL_OK) {
            throw std::runtime_error("flashprefill submit: metadata frag failed");
        }
    }
    for (size_t i = 0; i < pack.rows.size(); ++i) {
        const auto & r = pack.rows[i];
        const int32_t flags = r.forced ? GGML_FLASHPREFILL_ROW_FLAG_DENSE_FORCE : 0;
        if (ggml_flashprefill_metadata_set_row(md, meta_words, (int64_t) i,
                    r.src_q, r.kv_head, r.log_pos, r.tile, r.pbegin, r.pend, flags, r.q_head) != GGML_FLASHPREFILL_OK) {
            throw std::runtime_error("flashprefill submit: metadata row failed");
        }
    }
    for (size_t i = 0; i < pack.uses.size(); ++i) {
        const auto & u = pack.uses[i];
        if (ggml_flashprefill_metadata_set_use(md, meta_words, (int64_t) i,
                    u.frag, u.tile, u.kv_head, u.q_group, u.sub_off, u.sub_count, u.flags, u.src_q) != GGML_FLASHPREFILL_OK) {
            throw std::runtime_error("flashprefill submit: metadata use failed");
        }
    }
    for (size_t i = 0; i < pack.cells.size(); ++i) {
        if (pack.cells[i] < 0 || (uint64_t) pack.cells[i] >= built_n_kv) {
            throw std::runtime_error("flashprefill submit: packed cell outside KV tensor view");
        }
        if (ggml_flashprefill_metadata_set_cell(md, meta_words, (int64_t) i, pack.cells[i]) != GGML_FLASHPREFILL_OK) {
            throw std::runtime_error("flashprefill submit: metadata cell failed");
        }
    }
    ggml_backend_tensor_set(meta, st_meta.data(), 0, ggml_nbytes(meta));
    if (q_pos != nullptr && q_gather != nullptr && !pack.qpos.empty()) {
        // Native strided pairing (coordinate k at k*g_cap, IMRoPE text rule
        // for n_pos==4); tails stay zeroed (deterministic, unreferenced).
        if ((uint32_t) pack.n_groups > (uint32_t) q_pos->ne[0] / (uint32_t) key.n_pos) {
            throw std::runtime_error("flashprefill submit: groups exceed qpos capacity");
        }
        std::fill(st_qpos.begin(), st_qpos.end(), 0);
        std::fill(st_gather.begin(), st_gather.end(), 0);
        const size_t gc = (size_t) q_pos->ne[0] / (size_t) key.n_pos;
        for (int32_t g = 0; g < pack.n_groups; ++g) {
            const int32_t p = pack.qpos[(size_t) g];
            st_qpos[(size_t) g] = p;
            if (key.n_pos == 4) {
                st_qpos[gc + (size_t) g]     = p;
                st_qpos[gc * 2 + (size_t) g] = p;
                st_qpos[gc * 3 + (size_t) g] = 0;
            }
            st_gather[(size_t) g] = (int32_t) layout.groups[(size_t) g].query_index;
        }
        ggml_backend_tensor_set(q_pos, st_qpos.data(), 0, ggml_nbytes(q_pos));
        ggml_backend_tensor_set(q_gather, st_gather.data(), 0, ggml_nbytes(q_gather));
    }
    last_sparse_rows = pack.sparse_rows;
    last_forced_rows = pack.forced_rows;
    built_summary.sparse_rows = pack.sparse_rows;
    built_summary.dense_rows = pack.forced_rows;
    built_summary.visible_tokens = (int64_t) pack.visible_tokens;
    built_summary.expected_sparse_rows = pack.sparse_rows;
    built_summary.expected_forced_rows = pack.forced_rows;
    built_summary.n_rows = (int32_t) pack.rows.size();
    built_summary.n_uses = (int32_t) pack.uses.size();
    built_summary.n_cells = (int32_t) pack.cells.size();
    built_summary.n_groups = pack.n_groups;
    built_summary.max_sel_pair = (int32_t) pack.max_sel;
    const auto t1 = std::chrono::steady_clock::now();
    stat_layout_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

bool llm_graph_input_attn_flashprefill::can_reuse(const llm_graph_params & params) {
    // Refresh the owned row snapshot from the CURRENT params first (Main):
    // tail coordinates/epochs/counts are submit data, never topology, and
    // must never pin the prior ubatch through the stale build-time copy.
    cur_rows = params.cparams.flashprefill_rows;
    if (!key_valid) {
        return false;
    }
    if (params.flashprefill_reserve_sizing != key.reserve_sizing) {
        return false; // reserve graphs never alias live graphs
    }
    if (mctx == nullptr || params.hparams.n_layer() != key.layer_kv_types.size()) {
        return false;
    }
    if (!key.reserve_sizing && mctx->get_n_kv() != built_n_kv) {
        return false; // tensor shapes changed; rebuild rather than fail at submit
    }
    for (uint32_t il = 0; il < params.hparams.n_layer(); ++il) {
        if (llm_fp_layer_eligible(params.hparams, (int) il) &&
            !key.matches_layer_types((int32_t) il,
                (int32_t) mctx->layer_type_k((int32_t) il),
                (int32_t) mctx->layer_type_v((int32_t) il))) {
            return false;
        }
    }
    const llama_ubatch & ub = params.ubatch;
    const uint32_t expect_n = key.reserve_sizing
        ? key.n_tokens
        : ub.n_tokens;
    if (key.reserve_sizing) {
        uint32_t n_ctx_cells = 0, q_rsv = 0;
        if (!llm_fp_reserve_probe_inputs(params.cparams, params.hparams, ub, &n_ctx_cells, &q_rsv)) {
            return false;
        }
        if (q_rsv != expect_n) {
            return false;
        }
    } else if (ub.n_tokens != expect_n) {
        return false;
    }
    {
        // Policy topology vs the build-time key (allow_reuse compared the
        // full configs first; this keeps the input self-consistent).
        const auto & fc = params.cparams.flashprefill;
        if (fc.mode != key.mode || fc.tail_scope != key.tail_scope || fc.block_q != key.block_q ||
            fc.block_k != key.block_k || fc.sink_blocks != key.sink_blocks ||
            fc.window_blocks != key.window_blocks || fc.dense_tail_tiles != key.dense_tail_tiles ||
            fc.min_kv != key.min_kv || fc.full_attn_layers != key.full_attn_layers ||
            fc.mean_correction != key.mean_correction || fc.exact_all != key.exact_all) {
            return false;
        }
    }
    if (key.reserve_sizing) {
        // Reserve graphs never run and never build a layout: caps are pure
        // in (config, ctx cells, reserve queries, dims). Reuse on those
        // alone; the synthetic snapshot is covered by allow_reuse already.
        if (params.cparams.n_ctx_kv != cparams.n_ctx_kv ||
            params.cparams.n_ctx    != cparams.n_ctx    ||
            params.cparams.n_ubatch != cparams.n_ubatch) {
            return false;
        }
        if ((int64_t) params.hparams.n_head()    != (int64_t) key.n_q_heads  ||
            (int64_t) params.hparams.n_head_kv() != (int64_t) key.n_kv_heads ||
            (int64_t) params.hparams.n_embd_head_k() != (int64_t) key.dk     ||
            (int64_t) params.hparams.n_embd_head_v() != (int64_t) key.dv ||
            (int64_t) params.hparams.n_pos_per_embd() != (int64_t) key.n_pos) {
            return false;
        }
        bool backend_ok = false;
        if (llm_fp_backend_variant(params.sched, &backend_ok) != key.backend_variant || !backend_ok) {
            return false;
        }
        return true;
    }
    // Rows topology: presence, size, role, known flag. Coordinates and
    // intervals are submit data (refreshed above), never topology. Tile
    // numbering is domain-partitioned, so recompute the exact tile count
    // from the current rows (cheap, no layout): a different domain
    // partition must rebuild even inside the same buckets.
    if (cur_rows == nullptr || cur_rows->size() != (size_t) ub.n_tokens) {
        return false;
    }
    {
        if (key.n_kv_heads <= 0 || key.n_q_heads <= 0 || key.block_q == 0) {
            return false;
        }
        const uint32_t gqa_now = (uint32_t) key.n_q_heads / (uint32_t) key.n_kv_heads;
        uint64_t tiles_now = 0;
        if (gqa_now == 0 ||
            !llm_fp_domain_layout(*cur_rows, (uint32_t) key.block_q, gqa_now, nullptr, nullptr, nullptr, &tiles_now)) {
            return false;
        }
        if (tiles_now != (uint64_t) key.n_tiles) {
            return false;
        }
    }
    if (llm_fp_uniform_role(*cur_rows) != key.role) {
        return false;
    }
    // Layout topology (get-only): fresh, eligible, same buckets/role/shape.
    if (mctx == nullptr) {
        return false;
    }
    const llama_flashprefill_layout & layout = mctx->flashprefill_get_layout();
    if (!layout.eligible || !mctx->flashprefill_layout_is_fresh()) {
        return false;
    }
    const auto & lkey = mctx->flashprefill_layout_key();
    if (lkey.role != key.role || lkey.block_k != params.cparams.flashprefill.block_k ||
        lkey.n_tokens != ub.n_tokens || lkey.want_exact_rows != 0) {
        return false;
    }
    if ((layout.is_rerot != 0) != key.is_rerot) {
        return false;
    }
    if (llm_fp_bucket(layout.n_fragments()) != key.f_cap) {
        return false;
    }
    {
        // Wire-use estimate (uses x kv-head fan-out x sound tile straddle,
        // checked 64-bit): exceeding the cap forces a rebuild.
        const uint32_t gqa_now =
            (key.n_kv_heads > 0 && key.n_q_heads > 0) ? (uint32_t) key.n_q_heads / (uint32_t) key.n_kv_heads : 0u;
        const uint64_t west = (uint64_t) layout.uses.size() *
            (uint64_t) (key.n_kv_heads > 0 ? key.n_kv_heads : 1) *
            (uint64_t) llm_fp_tile_fanout(gqa_now, params.cparams.flashprefill.block_q);
        if (west > (uint64_t) key.u_cap) {
            return false;
        }
    }
    {
        uint64_t cells_est = (uint64_t) layout.n_fragments() * (uint64_t) cparams.flashprefill.block_k;
        if (cells_est > (uint64_t) key.c_cap) {
            return false;
        }
    }
    if ((uint64_t) ub.n_tokens * (uint64_t) key.n_q_heads > (uint64_t) key.r_cap) {
        return false;
    }
    if (llm_fp_bucket((uint32_t) layout.uses.size()) != key.u_layout_cap) {
        return false;
    }
    {
        // Capacity-static Q: gather/pos tensors are g_cap-shaped with zero
        // tails, so the actual group count is submit data (refilled), not
        // topology. Presence must match the build (tensor set) and the count
        // must fit the cap; kernels bound by the header actual.
        const bool has_groups_now = layout.is_rerot && layout.n_groups() != 0;
        if (has_groups_now != (q_pos != nullptr)) {
            return false;
        }
        const uint32_t g_now = has_groups_now ? layout.n_groups() : (uint32_t) ub.n_tokens;
        if (g_now > llm_fp_bucket(key.n_groups)) {
            return false;
        }
    }
    // Head/dim mapping + backend variant (topology).
    if ((int64_t) params.hparams.n_head()    != (int64_t) key.n_q_heads  ||
        (int64_t) params.hparams.n_head_kv() != (int64_t) key.n_kv_heads ||
        (int64_t) params.hparams.n_embd_head_k() != (int64_t) key.dk     ||
        (int64_t) params.hparams.n_embd_head_v() != (int64_t) key.dv ||
        (int64_t) params.hparams.n_pos_per_embd() != (int64_t) key.n_pos) {
        return false;
    }
    bool backend_ok = false;
    if (llm_fp_backend_variant(params.sched, &backend_ok) != key.backend_variant || !backend_ok) {
        return false;
    }
    return true;
}

// ---- sparse attention dispatch (per full-attention layer) ----

ggml_tensor * llm_graph_context::try_build_attn_flashprefill(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * q_raw_or_null,
        ggml_tensor * q_roped_or_null,
        ggml_tensor * k_roped,
        ggml_tensor * v,
        int *         sections_or_null,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        float         kq_scale,
        int           il) const {
    llm_graph_input_attn_flashprefill * fp = (inp != nullptr) ? inp->get_fp() : nullptr;
    if (fp == nullptr || !fp->active() || k_roped == nullptr || v == nullptr) {
        return nullptr; // OFF / shape-gated / policy-dense: no footprint, no counts
    }
    const llm_graph_fp_key & key = fp->reuse_key();
    const llama_flashprefill_config & cfg = cparams.flashprefill;
    const bool required = llm_fp_is_required(cfg);
    const bool reserve = fp->is_reserve();
    // Dense counters only for live graphs (reserve graphs are discarded).
    // Designed reasons (SWA / full-attention prefix) count separately and
    // authoritatively; capability fallbacks count under dense_layers (and
    // throw in required mode via need_throw instead of arriving silent).
    // A rerot fallback with absent legacy spans is safe: build_rerot_q_groups
    // lazily ensures them on the real fallback path (never on sparse).
    auto use_dense = [&](bool count, bool designed = false) -> ggml_tensor * {
        if (count && !reserve) {
            if (designed) {
                res->flashprefill_summary.designed_dense_layers += 1;
                res->flashprefill_summary.designed_dense_rows +=
                    (int64_t) ubatch.n_tokens * (int64_t) hparams.n_head(il);
            } else {
                res->flashprefill_summary.dense_layers += 1;
            }
        }
        return nullptr;
    };
    auto need_throw = [&](const char * msg) -> ggml_tensor * {
        if (required) {
            throw std::runtime_error(msg);
        }
        return use_dense(true);
    };
    // Designed-dense per-layer gates (silent, every mode). Recurrent layers
    // never arrive here (separate linear helper); the check is a backstop.
    if (hparams.is_recr(il)) {
        return use_dense(false);
    }
    if (hparams.is_swa(il)) {
        return use_dense(true, true);
    }
    // Widened compare (never narrow the user count): a huge prefix keeps
    // every eligible layer dense; eligible_index is small and non-negative.
    if ((uint32_t) flashprefill_eligible_full_index(il) < cfg.full_attn_layers) {
        return use_dense(true, true);
    }
    if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
        return use_dense(false); // bypassed upstream; backstop
    }
    if (cparams.embeddings || cparams.pooling_type != LLAMA_POOLING_TYPE_NONE || ubatch.embd != nullptr) {
        return use_dense(false);
    }
    if (kq_b != nullptr) {
        // Special KQ bias (ALiBi-style addends and friends): the sparse
        // kernels take scale/softcap only, never additive bias.
        return need_throw("flashprefill: special KQ bias keeps dense (required)");
    }
    const uint32_t Hq  = hparams.n_head(il);
    const uint32_t Hkv = hparams.n_head_kv(il);
    if (Hq == 0 || Hkv == 0 || Hq % Hkv != 0) {
        return need_throw("flashprefill: non-GQA layer (required)");
    }
    if ((int64_t) Hq != (int64_t) key.n_q_heads || (int64_t) Hkv != (int64_t) key.n_kv_heads) {
        // Non-uniform head mapping across layers: shapes genuinely differ,
        // so dense is correct (the shared input was keyed for il_first).
        return need_throw("flashprefill: head mapping drifted (required)");
    }
    const bool is_rerot = key.is_rerot;
    if (is_rerot && q_raw_or_null == nullptr) {
        throw std::runtime_error("flashprefill: rerot route missing raw Q (hook bug)");
    }
    if (!is_rerot && q_roped_or_null == nullptr) {
        throw std::runtime_error("flashprefill: ordinary route missing roped Q (hook bug)");
    }
    // Reserve mirrors live shapes: no sparse nodes where live stays dense.
    if (!flashprefill_backend_supported()) {
        return need_throw("flashprefill: eligible prefill has no sparse backend (required)");
    }
    const auto * mctx_cur = inp->mctx;
    if (mctx_cur == nullptr) {
        throw std::runtime_error("flashprefill: null KV context (hook bug)");
    }
    // Q form checks (ordinary: model-roped, single group per query). Head
    // dim is enforced here so a mismatch routes dense/throws instead of
    // tripping the constructor asserts below.
    const int64_t Dk_exp = (int64_t) hparams.n_embd_head_k(il);
    if (!is_rerot) {
        if (q_roped_or_null->type != GGML_TYPE_F32) {
            return need_throw("flashprefill: non-F32 Q (required)");
        }
        if (q_roped_or_null->ne[0] != Dk_exp ||
            q_roped_or_null->ne[1] != (int64_t) Hq || q_roped_or_null->ne[2] != n_tokens) {
            return need_throw("flashprefill: Q shape mismatch (required)");
        }
    } else {
        if (reserve) {
            throw std::runtime_error("flashprefill: reserve never takes the rerot route (hook bug)");
        }
        if (q_raw_or_null->type != GGML_TYPE_F32) {
            return need_throw("flashprefill: non-F32 Q (required)");
        }
        if (q_raw_or_null->ne[0] != Dk_exp ||
            q_raw_or_null->ne[1] != (int64_t) Hq || q_raw_or_null->ne[2] != n_tokens) {
            return need_throw("flashprefill: Q shape mismatch (required)");
        }
        if (sections_or_null == nullptr) {
            return need_throw("flashprefill: rerot route needs rope sections (required)");
        }
    }
    // Fresh layout for live builds (cheap guards only; full validation ran
    // once at plan generation). Group counts are submit data (Q tensors are
    // capacity-static); only eligibility/freshness/shape bind the build.
    if (!reserve) {
        const llama_flashprefill_layout & layout = mctx_cur->flashprefill_get_layout();
        if (!layout.eligible || !mctx_cur->flashprefill_layout_is_fresh()) {
            throw std::runtime_error("flashprefill: stale layout at attention build");
        }
        if (layout.n_queries() != (uint32_t) n_tokens) {
            throw std::runtime_error("flashprefill: layout/ubatch drift at attention build");
        }
        if (mctx_cur->get_n_kv() != fp->built_n_kv) {
            throw std::runtime_error("flashprefill: n_kv drift at attention build");
        }
    }
    // K/V rotations mirror build_attn (kv): k_rot on Q(resolved below) and
    // K, v_rot on V pre-copy, inverse v_rot post-attention.
    ggml_tensor * q_base = is_rerot ? q_raw_or_null : q_roped_or_null;
    ggml_tensor * k_cur  = k_roped;
    ggml_tensor * v_cur  = v;
    if (inp->self_k_rot != nullptr) {
        // Ordinary Q is already roped; rerot Q is roped per group below, so
        // the InnerQ-style rotation applies after grouping in that branch.
        if (!is_rerot) {
            q_base = llama_mul_mat_hadamard(ctx0, q_base, inp->self_k_rot);
        }
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }
    if (inp->self_v_rot != nullptr) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }
    // Grouped Q [D, H, G]: ordinary identity (already roped, never
    // re-roped) or rerot gather + single phased RoPE (never double).
    ggml_tensor * q_grouped = nullptr;
    if (!is_rerot) {
        q_grouped = q_base;
    } else {
        // Capacity-static Q (native DDVR pairing): gather + RoPE cover the
        // full group capacity; unused rows are 0/pos-0 (deterministic,
        // unreferenced by actual use groups). Kernels bound by the header
        // actual (meta.n_groups <= Q.ne[1]).
        if (fp->q_gather == nullptr || fp->q_pos == nullptr) {
            throw std::runtime_error("flashprefill: rerot route missing group tensors (hook bug)");
        }
        const int64_t head_dim = q_base->ne[0];
        const int64_t g_cap = fp->q_gather->ne[0];
        ggml_tensor * q_flat = ggml_reshape_2d(ctx0, q_base, head_dim * (int64_t) Hq, n_tokens);
        ggml_tensor * qg = ggml_get_rows(ctx0, q_flat, fp->q_gather);
        qg = ggml_reshape_3d(ctx0, qg, head_dim, (int64_t) Hq, g_cap);
        cb(qg, "flashprefill_q_grouped_raw", il);
        const int mode = static_cast<int>(rope_type);
        // Full-capacity pos tensor (native pairing, no slicing): coordinate
        // k at k*g_cap, IMRoPE text (p,p,p,0) for n_pos==4.
        if (mode == GGML_ROPE_TYPE_MROPE || mode == GGML_ROPE_TYPE_IMROPE || mode == GGML_ROPE_TYPE_VISION) {
            qg = ggml_rope_multi(ctx0, qg, fp->q_pos, nullptr,
                    n_rot, sections_or_null, mode, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
        } else {
            qg = ggml_rope_ext(ctx0, qg, fp->q_pos, nullptr,
                    n_rot, mode, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
        }
        cb(qg, "flashprefill_q_grouped", il);
        if (inp->self_k_rot != nullptr) {
            qg = llama_mul_mat_hadamard(ctx0, qg, inp->self_k_rot);
        }
        q_grouped = qg;
    }
    // Store to KV cache; the pool ordering edges (k_dep/v_dep) carry the
    // dependency as real graph edges, never source-order assumption.
    // Nothing is expanded yet: co-expansion happens after the capability
    // probe below, so an AUTO-dense fallback leaves no stray nodes behind
    // (the dense path expands its own copies).
    ggml_tensor * k_dep = mctx_cur->cpy_k(ctx0, k_cur, inp->get_k_idxs(), il);
    ggml_tensor * v_dep = mctx_cur->cpy_v(ctx0, v_cur, inp->get_v_idxs(), il);
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v_cache = mctx_cur->get_v(ctx0, il);
    // FA-convention layout mirror (build_attn_mha): the wire ops consume
    // permuted K/V ([D, n_kv, Hkv, streams]), never raw cache views.
    // Transposed-V caches stay dense (ctor ne-shape discipline; future
    // kernel stride work), as do multi-stream caches (ctors require
    // ne[3]==1; hybrid unified is 1).
    if (v_cache->nb[1] > v_cache->nb[2]) {
        return need_throw("flashprefill: transposed V cache keeps dense (required)");
    }
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v_cache = ggml_permute(ctx0, v_cache, 0, 2, 1, 3);
    if (k->ne[3] != 1 || v_cache->ne[3] != 1) {
        return need_throw("flashprefill: multi-stream cache keeps dense (required)");
    }
    // Compare with this layer's build-time type, not the first layer's.
    // Real same-layer drift still fails closed in every mode.
    if (!key.matches_layer_types(il, (int32_t) k->type, (int32_t) v_cache->type)) {
        throw std::runtime_error("flashprefill: KV dtype drifted (corrupt cache)");
    }
    if ((int64_t) k->ne[0] != (int64_t) key.dk || (int64_t) v_cache->ne[0] != (int64_t) key.dv) {
        return need_throw("flashprefill: KV head dims drifted (required)");
    }
    // TurboQuant Q forward WHT mirrors the dense path exactly (RoPE -> WHT
    // -> dot): pad per-head D to the 128 multiple, then forward transform
    // with the inverse InnerQ scale. Non-turbo caches skip untouched.
    ggml_tensor * q_wht = q_grouped;
    const bool k_turbo =
        k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0 || k->type == GGML_TYPE_TURBO2_0;
    if (k_turbo) {
        if (q_wht->ne[0] % 128 != 0) {
            const int64_t pad = ((q_wht->ne[0] + 127) / 128) * 128 - q_wht->ne[0];
            q_wht = ggml_pad(ctx0, q_wht, pad, 0, 0, 0);
        }
        if (!ggml_is_contiguous(q_wht)) {
            q_wht = ggml_cont(ctx0, q_wht);
        }
        q_wht = ggml_turbo_wht(ctx0, q_wht, 0, 0, mctx_cur->get_turbo_innerq_scale_inv());
    }
    // Agreed wire Q: F32 [Dk, n_groups, Hq, 1]; O: [Dv, Hq, Nq, 1].
    ggml_tensor * q4d = ggml_permute(ctx0, q_wht, 0, 2, 1, 3);
    const int32_t dk = (int32_t) k->ne[0];
    const int32_t dv = (int32_t) v_cache->ne[0];
    const float softcap = hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f;
    ggml_tensor * pool = ggml_flash_prefill_pool(ctx0, k, v_cache, fp->meta, k_dep, v_dep,
            dk, dv, (int32_t) Hkv, (int64_t) key.f_cap);
    cb(pool, "flashprefill_pool", il);
    ggml_tensor * plan = ggml_flash_prefill_select(ctx0, q4d, pool, fp->meta,
            (int64_t) key.n_tiles, (int64_t) Hkv, (int64_t) key.max_sel_pair,
            kq_scale, cfg.alpha, softcap, cfg.exact_all ? 1 : 0, cfg.mean_correction);
    cb(plan, "flashprefill_plan", il);
    ggml_tensor * cur = ggml_flash_prefill_attn(ctx0, q4d, k, v_cache, pool, plan, fp->meta, sinks,
            n_tokens, (int64_t) Hq, dv, kq_scale, softcap, cfg.mean_correction);
    cb(cur, "flashprefill_attn", il);
    // Backend op capability (frozen): real supports_op probes for all three
    // ops on the intended side — never name-only. The intended side follows
    // actual placement, not the offload preference alone (offload_kqv stays
    // true by default even on CPU-only contexts): a GPU backend existing
    // plus offload means GPU-resident K/V; otherwise CPU. Requiring the
    // wrong side would falsely reject CPU references or scheduler-fallback
    // across devices with full-KV transfers, so REQUIRED throws and AUTO
    // keeps dense. Capable tensors are explicitly pinned (no mere
    // any-GPU-supported + scheduler fallback). A null scheduler (unit
    // tests) skips the probe as CPU-oracle.
    ggml_backend_t fp_sel = nullptr;
    if (sched != nullptr) {
        bool cpu_ok = false, gpu_ok = false, gpu_present = false;
        ggml_backend_t cpu_sel = nullptr, gpu_sel = nullptr;
        const int nb = ggml_backend_sched_get_n_backends(sched);
        for (int bi = 0; bi < nb; ++bi) {
            ggml_backend_t b = ggml_backend_sched_get_backend(sched, bi);
            if (b == nullptr) {
                continue;
            }
            const char * nm = ggml_backend_name(b);
            const bool is_cpu = (nm != nullptr && std::string(nm).find("CPU") != std::string::npos);
            gpu_present = gpu_present || !is_cpu;
            const bool all3 = ggml_backend_supports_op(b, pool) &&
                              ggml_backend_supports_op(b, plan) &&
                              ggml_backend_supports_op(b, cur);
            if (!all3) {
                continue;
            }
            if (is_cpu) {
                cpu_ok = true;
                if (cpu_sel == nullptr) {
                    cpu_sel = b;
                }
            } else {
                gpu_ok = true;
                if (gpu_sel == nullptr) {
                    gpu_sel = b;
                }
            }
        }
        const bool want_gpu = gpu_present && cparams.offload_kqv;
        ggml_backend_t sel = want_gpu ? gpu_sel : cpu_sel;
        const bool ok = want_gpu ? gpu_ok : cpu_ok;
        if (!ok || sel == nullptr) {
            return need_throw("flashprefill: no capable backend for pool/select/attn (required)");
        }
        fp_sel = sel;
        ggml_backend_sched_set_tensor_backend(sched, pool, sel);
        ggml_backend_sched_set_tensor_backend(sched, plan, sel);
        ggml_backend_sched_set_tensor_backend(sched, cur, sel);
    }
    // The metadata is now referenced by live sparse nodes: mark consumed so
    // submit uploads it. Layers that never reach here (dense fallbacks)
    // leave it unmarked and submit skips the orphan upload.
    fp->mark_used_by_graph();
    // Co-expansion (split discipline): producers, copies and sparse nodes
    // ride one graph region; k_dep/v_dep edges order cpy before pool.
    ggml_build_forward_expand(gf, q_grouped);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_dep);
    ggml_build_forward_expand(gf, v_dep);
    ggml_build_forward_expand(gf, pool);
    ggml_build_forward_expand(gf, plan);
    // TurboQuant V inverse WHT + asymmetric trim mirror the dense paths:
    // undo the V rotation on the attention output, then extract the original
    // V head dim (padded storage vs logical dim).
    if (v_cache->type == GGML_TYPE_TURBO3_0 || v_cache->type == GGML_TYPE_TURBO4_0 ||
        v_cache->type == GGML_TYPE_TURBO2_0) {
        const bool k_is_turbo =
            k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0 || k->type == GGML_TYPE_TURBO2_0;
        const ggml_tensor * group_src = k_is_turbo ? k : v_cache;
        const int turbo_group = (group_src->ne[0] % 128 == 0) ? 128 : 64;
        if (cur->ne[0] % turbo_group == 0) {
            if (!ggml_is_contiguous(cur)) {
                cur = ggml_cont(ctx0, cur);
            }
            cur = ggml_turbo_wht(ctx0, cur, 1, turbo_group, mctx_cur->get_turbo_innerq_scale_inv());
        }
    }
    {
        const int64_t padded_v_head = v_cache->ne[0];
        const int64_t orig_v_head = hparams.n_embd_head_v(il);
        if (padded_v_head != orig_v_head) {
            cur = ggml_reshape_3d(ctx0, cur, padded_v_head, (int64_t) Hq, n_tokens);
            cur = ggml_view_3d(ctx0, cur, orig_v_head, (int64_t) Hq, n_tokens,
                    cur->nb[1], cur->nb[2], 0);
            cur = ggml_cont(ctx0, cur);
            cur = ggml_reshape_2d(ctx0, cur, orig_v_head * (int64_t) Hq, n_tokens);
        } else {
            cur = ggml_reshape_2d(ctx0, cur, (int64_t) dv * (int64_t) Hq, n_tokens);
        }
    }
    if (inp->self_v_rot != nullptr) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }
    if (reserve) {
        // Reserve measurement: identical route including transforms (same
        // tensor lifetime/live ranges as runtime ON), zero metadata counts,
        // never executed. Plan output-marked like live; no metrics, no
        // summary pollution, no extra dense route from the caller.
        ggml_set_output(plan);
        return cur;
    }
    // First sparse layer seeds the shared-input bytes once (no per-layer
    // multiplier on shared meta) plus caps/counts; every layer then records
    // its pinned plan + live pool. Metrics reads plan headers once at the
    // completion boundary; the scratch total needs no sync.
    if (res->get_flashprefill_plans().empty()) {
        res->flashprefill_summary.scratch_bytes = fp->built_summary.scratch_bytes;
        res->flashprefill_summary.meta_bytes    = fp->built_summary.scratch_bytes;
        res->flashprefill_summary.ubatch_tokens = fp->built_summary.ubatch_tokens;
        res->flashprefill_summary.visible_tokens = fp->built_summary.visible_tokens;
        res->flashprefill_summary.expected_sparse_rows = fp->built_summary.expected_sparse_rows;
        res->flashprefill_summary.expected_forced_rows = fp->built_summary.expected_forced_rows;
        res->flashprefill_summary.n_tiles = (int32_t) key.n_tiles;
        res->flashprefill_summary.n_fragments = fp->built_summary.n_fragments;
        res->flashprefill_summary.n_rows = fp->built_summary.n_rows;
        res->flashprefill_summary.n_uses = fp->built_summary.n_uses;
        res->flashprefill_summary.n_cells = fp->built_summary.n_cells;
        res->flashprefill_summary.n_groups = fp->built_summary.n_groups;
        res->flashprefill_summary.max_sel_pair = (int32_t) key.max_sel_pair;
        res->flashprefill_summary.f_cap = (int32_t) key.f_cap;
        res->flashprefill_summary.r_cap = (int32_t) key.r_cap;
        res->flashprefill_summary.u_cap = (int32_t) key.u_cap;
        res->flashprefill_summary.c_cap = (int32_t) key.c_cap;
        res->flashprefill_summary.layout_us = fp->stat_layout_us;
    }
    res->record_flashprefill_plan(plan, pool, fp_sel, il, fp->last_sparse_rows, fp->last_forced_rows);
    return cur;
}
