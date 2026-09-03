// Trie-driven TriAttention calibration collector.
//
// Feeds a request-token prefix trie to the model and collects pre-RoPE Q
// statistics (complex mean, mean magnitude per frequency band) for all
// full-attention layers. Writes a .triattention calibration file.
//
// Design:
// - Trie DFS: each branch node decoded exactly once (same as trie-calib.cpp)
// - Selected pre-RoPE Q tensors are graph outputs, preserving normal graph execution.
// - Vulkan queues every layer's D2H copy asynchronously; one synchronization makes
//   the complete batch visible to CPU accumulation.
// - No evaluation callback and no per-layer graph cuts or synchronization.
//
// The calibration file stores:
//   q_mean_real[f] = Re(E[q_f])  — complex mean of pre-RoPE Q per freq band
//   q_mean_imag[f] = Im(E[q_f])
//   q_abs_mean[f]  = E[||q_f||]  — mean magnitude per freq band
//   r_f[f]         = ||E[q_f]|| / E[||q_f||]  — concentration ratio

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-ext.h"
#include "llama-triattention.h"

#include "trie.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================================
// Model config extraction
// ============================================================================

struct model_config {
    int32_t  n_layer       = 0;    // trunk layers (not MTP)
    int32_t  n_head        = 0;    // attention heads
    int32_t  n_head_kv     = 0;    // KV heads
    int32_t  head_dim      = 0;
    double   rope_theta    = 0.0;
    uint32_t rope_style    = 0;    // 0=half/NeoX pairing, 1=even-odd pairing
    uint32_t rotary_dim    = 0;    // partial RoPE dimension
    uint32_t freq_count    = 0;    // rotary_dim / 2
    std::vector<int> full_attn_layers;  // layer indices with standard attention
};

static model_config get_model_config(llama_model * model) {
    model_config mc;
    mc.n_layer    = llama_model_n_layer(model);
    mc.n_head     = llama_model_n_head(model);
    mc.n_head_kv  = llama_model_n_head_kv(model);

    // head_dim: use architecture-specific key_length if available, else compute
    {
        char buf[256];
        // Try common architecture prefixes
        const char * prefixes[] = {"qwen35moe", "qwen3", "llama", nullptr};
        bool found = false;
        for (int i = 0; prefixes[i] && !found; ++i) {
            std::string key = std::string(prefixes[i]) + ".attention.key_length";
            if (llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf))) {
                mc.head_dim = (int32_t)std::strtol(buf, nullptr, 10);
                found = true;
            }
        }
        if (!found) {
            mc.head_dim = mc.n_head > 0 ? llama_model_n_embd(model) / mc.n_head : 0;
        }
    }

    // rope_theta: try architecture-specific rope.freq_base
    {
        char buf[256];
        const char * prefixes[] = {"qwen35moe", "qwen3", "llama", nullptr};
        bool found = false;
        for (int i = 0; prefixes[i] && !found; ++i) {
            std::string key = std::string(prefixes[i]) + ".rope.freq_base";
            if (llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf))) {
                mc.rope_theta = std::strtod(buf, nullptr);
                found = true;
            }
        }
        if (!found) {
            // Fallback to generic rope_theta
            if (llama_model_meta_val_str(model, "rope_theta", buf, sizeof(buf))) {
                mc.rope_theta = std::strtod(buf, nullptr);
            }
        }
    }

    // Pairing follows ggml's vector rotation layout, not M-RoPE section order.
    // IMROPE interleaves position sections but still uses NeoX/front-back pairs.
    switch (llama_model_rope_type(model)) {
        case LLAMA_ROPE_TYPE_NORM:
            mc.rope_style = 1;
            break;
        case LLAMA_ROPE_TYPE_NEOX:
        case LLAMA_ROPE_TYPE_MROPE:
        case LLAMA_ROPE_TYPE_IMROPE:
            mc.rope_style = 0;
            break;
        case LLAMA_ROPE_TYPE_NONE:
        case LLAMA_ROPE_TYPE_VISION:
            throw std::runtime_error("TriAttention calibration does not support this RoPE layout");
    }

    // rotary_dim: use architecture-specific rope.dimension_count
    {
        char buf[256];
        const char * prefixes[] = {"qwen35moe", "qwen3", "llama", nullptr};
        bool found = false;
        for (int i = 0; prefixes[i] && !found; ++i) {
            std::string key = std::string(prefixes[i]) + ".rope.dimension_count";
            if (llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf))) {
                mc.rotary_dim = (uint32_t)std::strtoul(buf, nullptr, 10);
                found = true;
            }
        }
        if (!found) {
            mc.rotary_dim = (uint32_t)mc.head_dim;  // full RoPE
        }
    }

    mc.freq_count = mc.rotary_dim / 2;

    // Determine full-attention layers using full_attention_interval
    {
        char buf[256];
        const char * prefixes[] = {"qwen35moe", "qwen3", nullptr};
        int interval = 0;
        bool found = false;
        for (int i = 0; prefixes[i] && !found; ++i) {
            std::string key = std::string(prefixes[i]) + ".full_attention_interval";
            if (llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf))) {
                interval = (int)std::strtol(buf, nullptr, 10);
                found = true;
            }
        }
        if (found && interval > 0) {
            // Full attention at layers where (i+1) % interval == 0
            for (int i = 0; i < mc.n_layer; ++i) {
                if ((i + 1) % interval == 0) {
                    mc.full_attn_layers.push_back(i);
                }
            }
        } else {
            // Fallback: all layers are full attention
            for (int i = 0; i < mc.n_layer; ++i) {
                mc.full_attn_layers.push_back(i);
            }
        }
    }

    return mc;
}

// ============================================================================
// Q statistics collector
// ============================================================================

struct q_head_stats {
    // Per-frequency accumulators (double for precision over long runs)
    std::vector<double> q_sum_real;    // sum of Re(q_f)
    std::vector<double> q_sum_imag;    // sum of Im(q_f)
    std::vector<double> q_sum_abs;     // sum of ||q_f||
    uint64_t            n_samples = 0;

    void init(uint32_t fc) {
        q_sum_real.assign(fc, 0.0);
        q_sum_imag.assign(fc, 0.0);
        q_sum_abs.assign(fc, 0.0);
        n_samples = 0;
    }
};

struct q_collector {
    model_config mc;
    std::set<int> sampled_layers;
    std::map<std::pair<int, int>, q_head_stats> stats;

    uint32_t collect_stride = 1;
    uint64_t ubatch_idx = 0;

    void init(const model_config & cfg, const std::set<int> & layers) {
        mc = cfg;
        sampled_layers = layers;
        for (int il : sampled_layers) {
            for (int h = 0; h < mc.n_head; ++h) {
                stats[{il, h}].init(mc.freq_count);
            }
        }
    }

    void collect(llama_context * ctx, size_t n_tokens) {
        ++ubatch_idx;
        if (collect_stride > 1 && ubatch_idx % collect_stride != 0) {
            return;
        }

        // Decode queued one async D2H copy per sampled layer. Synchronize all
        // copies once, then consume the host buffers without layer-by-layer GPU stalls.
        llama_synchronize(ctx);

        const uint32_t hd = mc.head_dim;
        const uint32_t fc = mc.freq_count;
        const uint32_t nh = mc.n_head;

        for (int il : sampled_layers) {
            const float * q_data = llama_get_attention_q_pre_rope(ctx, (uint32_t) il);
            for (size_t tok = 0; tok < n_tokens; ++tok) {
                for (uint32_t h = 0; h < nh; ++h) {
                    const float * q_vec = q_data + (tok * nh + h) * hd;
                    auto & hs = stats[{il, (int) h}];

                    for (uint32_t f = 0; f < fc; ++f) {
                        float re;
                        float im;
                        if (mc.rope_style == 0) {
                            re = q_vec[f];
                            im = q_vec[f + fc];
                        } else {
                            re = q_vec[2 * f];
                            im = q_vec[2 * f + 1];
                        }

                        hs.q_sum_real[f] += (double) re;
                        hs.q_sum_imag[f] += (double) im;
                        hs.q_sum_abs[f]  += (double) std::sqrt(re * re + im * im);
                    }
                    ++hs.n_samples;
                }
            }
        }
    }
};

// ============================================================================
// Trie DFS traversal (same pattern as trie-calib.cpp)
// ============================================================================

using wxq::trie;

struct dfs_driver {
    llama_context * ctx = nullptr;
    llama_batch     batch{};
    int32_t         n_batch = 0;
    q_collector *   coll = nullptr;
    uint64_t        n_decoded = 0;
    uint64_t        max_nodes = 0;
    bool            stop = false;
    uint64_t        peak_snapshot_bytes = 0;
    uint64_t        live_snapshot_bytes = 0;
    uint64_t        n_total = 0;
    int64_t         t_start_us = 0;
    int64_t         t_last_report_us = 0;
    int64_t         report_period_us = 5000000;
    uint32_t        max_ctx = 0;
    uint64_t        n_ctx_resets = 0;
    llama_pos       cur_pos = 0;

    void reset_window() {
        llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
        cur_pos = 0;
        ++n_ctx_resets;
    }

    bool decode_span(const std::vector<llama_token> & toks) {
        for (size_t off = 0; off < toks.size(); off += (size_t) n_batch) {
            if (max_nodes != 0 && n_decoded >= max_nodes) {
                stop = true;
                return true;
            }
            const size_t n = std::min((size_t) n_batch, toks.size() - off);
            if (max_ctx != 0 && cur_pos + (llama_pos) n > (llama_pos) max_ctx) {
                reset_window();
            }
            common_batch_clear(batch);
            for (size_t i = 0; i < n; ++i) {
                common_batch_add(batch, toks[off + i], cur_pos + (llama_pos) i, { 0 }, i + 1 == n);
            }

            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("%s: llama_decode failed at pos %d\n", __func__, (int) cur_pos);
                return false;
            }

            coll->collect(ctx, n);

            const int64_t t1 = ggml_time_us();
            cur_pos   += (llama_pos) n;
            n_decoded += n;

            if (t1 - t_last_report_us >= report_period_us) {
                const double el   = (double) (t1 - t_start_us) / 1e6;
                const double rate = (double) n_decoded / std::max(el, 1e-6);
                const double frac = n_total ? (double) n_decoded / (double) n_total : 0.0;
                const double eta  = (rate > 0.0 && n_total > n_decoded)
                                  ? (double) (n_total - n_decoded) / rate : 0.0;
                uint64_t total_samples = 0;
                for (const auto & [k, v] : coll->stats) total_samples += v.n_samples;
                LOG_INF("%s: pos %7d | %7" PRIu64 "/%" PRIu64 " (%4.1f%%) | %7.1f tok/s | eta %5.0fs | Q samples %" PRIu64 "\n",
                        __func__, (int) cur_pos, n_decoded, n_total, 100.0*frac, rate, eta, total_samples);
                fflush(stderr);
                t_last_report_us = t1;
            }
        }
        return true;
    }

    bool visit(const trie & t, uint64_t node) {
        if (stop) return true;

        std::vector<llama_token> span;
        uint64_t cur = node;
        while (true) {
            span.push_back(t.nodes[cur - 1].token);
            const auto & kids = t.children[cur];
            if (kids.size() == 1) {
                cur = kids[0];
                continue;
            }
            break;
        }

        if (!decode_span(span)) return false;

        const auto & kids = t.children[cur];
        if (kids.empty() || stop) return true;

        // Branch: snapshot state, replay for each sibling
        const size_t snap_size = llama_state_seq_get_size(ctx, 0);
        std::vector<uint8_t> snap(snap_size);
        if (llama_state_seq_get_data(ctx, snap.data(), snap.size(), 0) != snap_size) {
            LOG_ERR("%s: state_seq_get_data short read at pos %d\n", __func__, (int) cur_pos);
            return false;
        }
        const llama_pos snap_pos = cur_pos;
        live_snapshot_bytes += snap_size;
        peak_snapshot_bytes = std::max(peak_snapshot_bytes, live_snapshot_bytes);

        bool ok = true;
        for (size_t i = 0; i < kids.size() && ok; ++i) {
            if (i > 0) {
                if (llama_state_seq_set_data(ctx, snap.data(), snap.size(), 0) != snap_size) {
                    LOG_ERR("%s: state_seq_set_data short write at pos %d\n", __func__, (int) cur_pos);
                    ok = false;
                    break;
                }
                cur_pos = snap_pos;
            }
            ok = visit(t, kids[i]);
        }

        live_snapshot_bytes -= snap_size;
        return ok;
    }
};

// ============================================================================
// Calibration file writer
// ============================================================================

static bool write_triattention(
    const std::string & path,
    const model_config & mc,
    const q_collector & coll,
    const std::string & model_name)
{
    const uint32_t fc = mc.freq_count;
    const uint32_t n_sampled = (uint32_t)coll.stats.size();

    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        LOG_ERR("%s: cannot open %s for writing\n", __func__, path.c_str());
        return false;
    }

    // Header
    uint32_t magic = TRIATTENTION_MAGIC;
    uint32_t version = TRIATTENTION_VERSION;
    fwrite(&magic, sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(uint32_t), 1, f);
    fwrite(&mc.head_dim, sizeof(uint32_t), 1, f);
    fwrite(&mc.n_layer, sizeof(uint32_t), 1, f);
    fwrite(&mc.n_head, sizeof(uint32_t), 1, f);
    fwrite(&mc.n_head_kv, sizeof(uint32_t), 1, f);
    fwrite(&mc.rope_theta, sizeof(double), 1, f);
    fwrite(&mc.rope_style, sizeof(uint32_t), 1, f);
    fwrite(&n_sampled, sizeof(uint32_t), 1, f);
    fwrite(&fc, sizeof(uint32_t), 1, f);
    fwrite(&mc.rotary_dim, sizeof(uint32_t), 1, f);  // v2 field

    // Model name, including exactly one trailing NUL.
    uint32_t name_len = (uint32_t) model_name.size() + 1;
    fwrite(&name_len, sizeof(uint32_t), 1, f);
    fwrite(model_name.c_str(), 1, name_len, f);

    // Per sampled head
    for (const auto & [key, hs] : coll.stats) {
        const uint32_t layer_idx = (uint32_t)key.first;
        const uint32_t head_idx  = (uint32_t)key.second;
        fwrite(&layer_idx, sizeof(uint32_t), 1, f);
        fwrite(&head_idx, sizeof(uint32_t), 1, f);

        if (hs.n_samples == 0) {
            // No samples — write zeros
            std::vector<float> zeros(fc, 0.0f);
            fwrite(zeros.data(), sizeof(float), fc, f);  // q_mean_real
            fwrite(zeros.data(), sizeof(float), fc, f);  // q_mean_imag
            fwrite(zeros.data(), sizeof(float), fc, f);  // q_abs_mean
            fwrite(zeros.data(), sizeof(float), fc, f);  // r_f
            continue;
        }

        const double inv_n = 1.0 / (double)hs.n_samples;

        // q_mean_real[f] = Re(E[q_f])
        std::vector<float> q_mean_real(fc), q_mean_imag(fc), q_abs_mean(fc), r_f(fc);
        for (uint32_t f_ix = 0; f_ix < fc; ++f_ix) {
            q_mean_real[f_ix] = (float)(hs.q_sum_real[f_ix] * inv_n);
            q_mean_imag[f_ix] = (float)(hs.q_sum_imag[f_ix] * inv_n);
            q_abs_mean[f_ix]  = (float)(hs.q_sum_abs[f_ix]  * inv_n);

            float mean_abs = std::sqrt(q_mean_real[f_ix] * q_mean_real[f_ix] +
                                        q_mean_imag[f_ix] * q_mean_imag[f_ix]);
            r_f[f_ix] = (q_abs_mean[f_ix] > 1e-10f) ? mean_abs / q_abs_mean[f_ix] : 0.0f;
        }

        fwrite(q_mean_real.data(), sizeof(float), fc, f);
        fwrite(q_mean_imag.data(), sizeof(float), fc, f);
        fwrite(q_abs_mean.data(),  sizeof(float), fc, f);
        fwrite(r_f.data(),         sizeof(float), fc, f);
    }

    fclose(f);
    LOG_INF("%s: wrote %s (%u sampled heads, %u freq_count, rotary_dim=%u)\n",
            __func__, path.c_str(), n_sampled, fc, mc.rotary_dim);
    return true;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    common_params params;

    std::string trie_dir = "wanxiangqi/calibration";
    std::string out_path = "ornith-1.5-35b.triattention";
    std::string model_name = "Ornith-1.5-35B-A3B";
    uint32_t    collect_stride = 1;
    uint32_t    max_ctx = 0;
    uint64_t    max_nodes = 0;
    std::string sampled_layers_str;  // comma-separated layer indices, empty = all full-attn

    for (int i = 1; i < argc; ++i) {
        const char * val = (i + 1 < argc) ? argv[i + 1] : nullptr;
        if (val == nullptr) continue;
        bool taken = true;
        if (std::strcmp(argv[i], "--trie") == 0) {
            trie_dir = val;
        } else if (std::strcmp(argv[i], "--out") == 0) {
            out_path = val;
        } else if (std::strcmp(argv[i], "--model-name") == 0) {
            model_name = val;
        } else if (std::strcmp(argv[i], "--collect-stride") == 0) {
            collect_stride = (uint32_t)std::strtoul(val, nullptr, 10);
        } else if (std::strcmp(argv[i], "--max-ctx") == 0) {
            max_ctx = (uint32_t)std::strtoul(val, nullptr, 10);
        } else if (std::strcmp(argv[i], "--max-nodes") == 0) {
            max_nodes = std::strtoull(val, nullptr, 10);
        } else if (std::strcmp(argv[i], "--sampled-layers") == 0) {
            sampled_layers_str = val;
        } else {
            taken = false;
        }
        if (taken) {
            argv[i] = argv[i + 1] = const_cast<char *>("--ignore");
            ++i;
        }
    }

    // Strip consumed options
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--ignore") != 0) {
            args.push_back(argv[i]);
        }
    }

    common_init();

    if (!common_params_parse((int)args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    // Disable speculative decoding (same reason as trie-calib.cpp)
    params.speculative.types.clear();
    params.speculative.draft.n_max = 0;
    params.warmup = false;

    // Load model first (without context) to get config
    common_init_result_ptr llama_init_model = common_init_from_params(params, true);
    llama_model * model = llama_init_model->model();
    if (model == nullptr) {
        LOG_ERR("%s: failed to load model\n", __func__);
        return 1;
    }

    // Get model config
    model_config mc = get_model_config(model);

    LOG_INF("%s: model config:\n", __func__);
    LOG_INF("%s:   n_layer=%d, n_head=%d, n_head_kv=%d, head_dim=%d\n",
            __func__, mc.n_layer, mc.n_head, mc.n_head_kv, mc.head_dim);
    LOG_INF("%s:   rope_theta=%.1f, rope_style=%u, rotary_dim=%u, freq_count=%u\n",
            __func__, mc.rope_theta, mc.rope_style, mc.rotary_dim, mc.freq_count);
    LOG_INF("%s:   full_attn_layers: ", __func__);
    for (int il : mc.full_attn_layers) LOG_INF("%d ", il);
    LOG_INF("(%zu layers)\n", mc.full_attn_layers.size());

    // Determine which layers to sample
    std::set<int> sampled_layers;
    if (!sampled_layers_str.empty()) {
        const std::string & s = sampled_layers_str;
        size_t pos = 0;
        while (pos < s.size()) {
            int val = std::strtol(s.c_str() + pos, nullptr, 10);
            sampled_layers.insert(val);
            size_t comma = s.find(',', pos);
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    } else {
        for (int il : mc.full_attn_layers) {
            sampled_layers.insert(il);
        }
    }

    LOG_INF("%s: sampling %zu layers × %d heads = %zu (layer,head) pairs\n",
            __func__, sampled_layers.size(), mc.n_head,
            sampled_layers.size() * mc.n_head);
    LOG_INF("%s: collect_stride=%u (collect Q every %u ubatches)\n",
            __func__, collect_stride, collect_stride);

    q_collector coll;
    coll.init(mc, sampled_layers);
    coll.collect_stride = collect_stride;

    auto cparams = common_context_params_to_llama(params);
    if (params.n_ctx_kv_auto || params.n_ctx_kv > 0) {
        cparams.n_seq_recurrent = 1;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        LOG_ERR("%s: failed to create context\n", __func__);
        return 1;
    }

    for (int il : sampled_layers) {
        llama_set_attention_q_pre_rope(ctx, (uint32_t) il, true);
    }

    // Load trie
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    trie t;
    try {
        t = wxq::load_trie(trie_dir, n_vocab);
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        return 1;
    }

    // Trie stats
    uint64_t max_depth = 0;
    size_t n_branch = 0;
    {
        std::vector<uint64_t> depth(t.n_nodes() + 1, 0);
        for (uint64_t i = 0; i < t.n_nodes(); ++i) {
            depth[i + 1] = depth[t.nodes[i].parent] + 1;
            max_depth = std::max(max_depth, depth[i + 1]);
        }
        for (const auto & kids : t.children) {
            if (kids.size() > 1) ++n_branch;
        }
    }

    LOG_INF("%s: trie: %s\n", __func__, trie_dir.c_str());
    LOG_INF("%s:   nodes %" PRIu64 ", leaves %zu, branch points %zu, max depth %" PRIu64 "\n",
            __func__, t.n_nodes(), t.leaves.size(), n_branch, max_depth);
    LOG_INF("%s:   leaf positions %" PRIu64 " -> DFS decodes %" PRIu64 " (%.1fx saving)\n",
            __func__, t.n_leaf_positions, t.n_nodes(),
            (double)t.n_leaf_positions / (double)std::max<uint64_t>(t.n_nodes(), 1));

    // DFS traversal
    const uint32_t n_ctx   = llama_n_ctx(ctx);
    const uint32_t n_batch = llama_n_batch(ctx);

    uint64_t depth_needed = max_nodes ? std::min<uint64_t>(max_depth, max_nodes) : max_depth;
    if (max_ctx != 0) depth_needed = std::min<uint64_t>(depth_needed, max_ctx);
    if (depth_needed > n_ctx) {
        LOG_ERR("%s: context %u < depth needed %" PRIu64 "; pass -c %" PRIu64 " or --max-ctx %u\n",
                __func__, n_ctx, depth_needed, depth_needed, n_ctx);
        return 1;
    }

    LOG_INF("%s: n_ctx=%u, n_batch=%u, n_ubatch=%u\n",
            __func__, n_ctx, n_batch, llama_n_ubatch(ctx));

    dfs_driver drv;
    drv.ctx        = ctx;
    drv.n_batch    = (int32_t)n_batch;
    drv.coll       = &coll;
    drv.max_nodes  = max_nodes;
    drv.max_ctx    = max_ctx;
    drv.n_total    = max_nodes ? std::min<uint64_t>(max_nodes, t.n_nodes()) : t.n_nodes();
    drv.t_start_us = ggml_time_us();
    drv.t_last_report_us = drv.t_start_us;
    drv.batch      = llama_batch_init((int32_t)n_batch, 0, 1);

    bool ok = true;
    try {
        for (uint64_t root_child : t.children[0]) {
            llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
            drv.cur_pos = 0;
            ok = drv.visit(t, root_child);
            if (!ok) break;
        }
    } catch (const std::exception & e) {
        LOG_ERR("%s: traversal aborted: %s\n", __func__, e.what());
        ok = false;
    }

    llama_batch_free(drv.batch);

    LOG_INF("%s: decoded %" PRIu64 " positions, %" PRIu64 " context resets, peak snapshot %.2f GiB\n",
            __func__, drv.n_decoded, drv.n_ctx_resets,
            (double)drv.peak_snapshot_bytes / (1024.0 * 1024.0 * 1024.0));

    // Print collection stats
    uint64_t total_samples = 0;
    for (const auto & [k, v] : coll.stats) {
        total_samples += v.n_samples;
    }
    LOG_INF("%s: total Q samples: %" PRIu64 "\n", __func__, total_samples);
    for (const auto & [k, v] : coll.stats) {
        if (v.n_samples > 0) {
            LOG_INF("%s:   layer %d head %2d: %" PRIu64 " samples\n",
                    __func__, k.first, k.second, v.n_samples);
        }
    }

    // Write calibration file
    if (!write_triattention(out_path, mc, coll, model_name)) {
        LOG_ERR("%s: failed to write calibration file\n", __func__);
        return 1;
    }

    LOG_INF("%s: %s (%s)\n", __func__, out_path.c_str(),
            ok ? "complete" : "PARTIAL — traversal did not finish");

    return ok ? 0 : 1;
}
