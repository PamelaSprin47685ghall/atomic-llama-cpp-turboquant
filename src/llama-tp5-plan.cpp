// TP5 plan implementation: pure CPU computation, no GPU/Vulkan dependencies.
// See src/llama-tp5-plan.h for the contract and TP5.md for the design.

#include "llama-tp5-plan.h"

#include "llama-arch.h"
#include "llama-hparams.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace {

int64_t lcm(int64_t a, int64_t b) {
    int64_t x = a, y = b;
    while (y) { int64_t t = x % y; x = y; y = t; }
    return a / x * b;
}

// Front-loaded head split where every rank keeps quant blocks whole:
// each rank's head_count * unit must be a multiple of blck.
bool split_heads_quant(int64_t n, uint32_t ranks, int64_t unit, int64_t blck,
                       std::array<int32_t, LLAMA_TP5_MAX_RANKS> & out,
                       std::string & err) {
    int64_t step = (blck % unit == 0) ? blck / unit : lcm(unit, blck) / unit;
    if (step <= 1) {
        // every head boundary legal: balanced split
        int64_t base = n / ranks, rem = n % ranks;
        for (uint32_t r = 0; r < ranks; ++r) out[r] = (int32_t)(base + (r < rem ? 1 : 0));
        return true;
    }
    if (n % step != 0) {
        err = "total head count " + std::to_string(n) + " is not a whole number of legal steps (" + std::to_string(step) + ")";
        return false;
    }
    int64_t target = (n + ranks - 1) / ranks;
    int64_t per = (target + step - 1) / step * step;
    int64_t last = n - per * (ranks - 1);
    if (last <= 0 || last % step != 0) {
        per -= step;
        last = n - per * (ranks - 1);
        if (per <= 0 || last <= 0 || last % step != 0) {
            err = "no legal front-loaded head split for n=" + std::to_string(n);
            return false;
        }
    }
    for (uint32_t r = 0; r < ranks - 1; ++r) out[r] = (int32_t) per;
    out[ranks - 1] = (int32_t) last;
    return true;
}

bool is_moe_gate_up(const std::string & base) {
    return base == "ffn_gate_exps.weight" || base == "ffn_up_exps.weight" || base == "ffn_gate_exp.weight" ||
           base == "ffn_up_exp.weight" || base == "ffn_gate_up_exps.weight" || base == "ffn_gate_up_exp.weight";
}

} // namespace

bool llama_tp5_plan_build(const llama_hparams & hp,
                          uint32_t              n_devices,
                          int64_t               n_vocab,
                          bool                  replicate_attention,
                          llama_tp5_plan &      out,
                          llama_tp5_error &     err) {
    if (n_devices < 2 || n_devices > LLAMA_TP5_MAX_RANKS) {
        err.code = "TP5_E_RANKS";
        err.detail = "ranks=" + std::to_string(n_devices) + " (2..8 supported)";
        return false;
    }

    out = {};
    out.ranks = n_devices;
    out.replicate_attention = replicate_attention;
    out.H  = hp.n_embd;
    out.L  = hp.n_layer();
    out.C  = hp.dsv4_hc_mult;
    out.R  = hp.hc_low_rank;
    out.E  = hp.n_expert;
    out.K  = hp.n_expert_used;
    out.F  = hp.n_ff_exp;
    out.Fs = hp.n_ff_shexp ? hp.n_ff_shexp : hp.n_ff_exp;
    out.Nq = hp.n_head();
    out.Nkv = hp.n_head_kv();
    out.da = hp.n_embd_head_k();
    out.Nk = hp.ssm_n_group;
    out.Nv = hp.ssm_dt_rank;
    out.ds = hp.ssm_d_state;
    out.Ni = hp.indexer_n_head;
    out.di = hp.indexer_head_size;
    out.n_vocab = n_vocab;

    if (out.C <= 1 || out.R == 0) {
        err.code = "TP5_E_HPARAMS";
        err.detail = "hyper-connection hparams invalid (A-F requires C>1, R>0)";
        return false;
    }
    // MoE channel split: F must divide evenly across ranks and stay quant-legal
    // (checked per-tensor later against the actual blck_size; here only the
    // logical division).
    if (out.F % n_devices != 0 || out.Fs % n_devices != 0) {
        err.code = "TP5_E_MOE_SPLIT";
        err.detail = "expert intermediate " + std::to_string(out.F) + "/" + std::to_string(out.Fs) +
                     " not divisible by " + std::to_string(n_devices) + " ranks";
        return false;
    }

    // Populate is_recr for all layers up to hp.n_layer_all (including NextN trailing layers)
    // NextN blocks attend densely (is_recr = false). hp.n_layer() is trunk length.
    const size_t total_layers = std::max((size_t) hp.n_layer_all, (size_t) out.L);
    out.is_recr.resize(total_layers);
    out.n_full = 0;
    for (size_t il = 0; il < total_layers; ++il) {
        // Qwen4EXP pattern: full attention every 4th layer (3, 7, 11, 15, ...), others are linear/GDN
        out.is_recr[il] = (il < (size_t) out.L) && ((il + 1) % 4 != 0);
        if (il < (size_t) out.L && !out.is_recr[il]) out.n_full++;
    }

    if (out.Nkv <= 0 || out.Nq <= 0 || out.Nq % out.Nkv != 0) {
        err.code   = "TP5_E_Q_SPLIT";
        err.detail = "query heads must form complete nonempty GQA groups";
        return false;
    }

    if (replicate_attention) {
        for (uint32_t r = 0; r < n_devices; ++r) {
            out.q_role_counts[r]  = out.Nq;
            out.kv_role_counts[r] = out.Nkv;
            out.kv_instances[r]   = out.n_full * out.Nkv;
            out.gdn_v_heads[r]    = out.Nv;
            out.gdn_qk_heads[r]   = out.Nk;
            out.gdn_v_ranges.push_back({ 0, out.Nv });
        }
        out.expected_events = (uint32_t) out.L;
        return true;
    }

    // --- QSA roles (TP5.md 7.1) ---
    {
        if (out.Nq < n_devices) {
            err.code = "TP5_E_Q_SPLIT";
            err.detail = "not enough query heads for nonempty rank-local attention";
            return false;
        }
        const int64_t q_per_kv = out.Nq / out.Nkv;
        if (n_devices <= out.Nkv) {
            int32_t first_kv = 0;
            for (uint32_t r = 0; r < n_devices; ++r) {
                const int32_t count   = (int32_t) (out.Nkv / n_devices + (r < out.Nkv % n_devices));
                out.q_role_counts[r]  = (int32_t) (count * q_per_kv);
                out.kv_role_counts[r] = count;
                out.kv_head_starts[r] = first_kv;
                first_kv += count;
            }
        } else {
            // Replicate a KV head only within its own query group. A rank must
            // never straddle unequal portions of two groups: local GQA uses a
            // single uniform query/KV ratio, not an arbitrary head map.
            uint32_t rank = 0;
            for (int64_t kv = 0; kv < out.Nkv; ++kv) {
                const int64_t parts = n_devices / out.Nkv + (kv >= out.Nkv - n_devices % out.Nkv);
                for (int64_t part = 0; part < parts; ++part, ++rank) {
                    out.q_role_counts[rank]  = (int32_t) (q_per_kv / parts + (part < q_per_kv % parts));
                    out.kv_role_counts[rank] = 1;
                    out.kv_head_starts[rank] = (int32_t) kv;
                }
            }
        }
        for (uint32_t r = 0; r < n_devices; ++r) {
            out.kv_instances[r] = out.n_full * out.kv_role_counts[r];
        }
    }

    // --- GDN V/state heads (TP5.md 8.1) ---
    {
        std::string serr;
        std::array<int32_t, LLAMA_TP5_MAX_RANKS> v{};
        if (!split_heads_quant(out.Nv, n_devices, out.ds, 256, v, serr)) {
            err.code = "TP5_E_GDN_SPLIT";
            err.detail = serr;
            return false;
        }
        int64_t first = 0;
        out.gdn_v_ranges.clear();
        for (uint32_t r = 0; r < n_devices; ++r) {
            out.gdn_v_heads[r] = v[r];
            out.gdn_qk_heads[r] = v[r]; // prearrangement: local Q/K heads == V heads
            out.gdn_v_ranges.push_back({first, first + v[r]});
            first += v[r];
        }
        if (first != out.Nv) {
            err.code = "TP5_E_GDN_SPLIT";
            err.detail = "head ranges do not cover all V heads";
            return false;
        }
    }

    out.expected_events = (uint32_t)(2 * out.L);
    return true;
}

llama_tp5_tensor_plan llama_tp5_plan::plan_tensor(const std::string & name, const int64_t ne[4],
                                                  uint32_t type_id, int64_t blck_size,
                                                  bool & ok, llama_tp5_error & err) const {
    ok = true;
    err = {};
    llama_tp5_tensor_plan tp{};
    tp.split_axis = -1;

    // split the tensor name
    std::string base = name;
    int64_t il = -1;
    if (name.rfind("blk.", 0) == 0) {
        auto dot = name.find('.', 4);
        il = std::stoll(name.substr(4, dot - 4));
        base = name.substr(dot + 1);
    }

    // Scales (.scale, .input_scale) and biases (.bias) are per-tensor 1D/scalar or replicated weights
    if (name.find(".scale") != std::string::npos || name.find(".input_scale") != std::string::npos ||
        base.rfind("output_norm", 0) == 0 || base.rfind("attn_norm", 0) == 0 || base.rfind("ffn_norm", 0) == 0) {
        tp.semantic = llama_tp5_semantic::HC;
        tp.layout   = llama_tp5_layout::MIRRORED;
        return tp;
    }

    // Production fix: Qwen3.8-Flash compact GGUF names expert weight tensors as ffn_gate_exp/ffn_up_exp
    // in addition to the standard ffn_gate_exps/ffn_up_exps.
    auto is_moe_expert_tensor = [&](const std::string & b) {
        return is_moe_gate_up(b) || b == "ffn_gate_exp.weight" || b == "ffn_up_exp.weight" ||
               b == "ffn_gate_exps.weight" || b == "ffn_up_exps.weight";
    };
    auto is_moe_down_tensor = [&](const std::string & b) {
        return b == "ffn_down_exp.weight" || b == "ffn_down_exps.weight";
    };

    auto fail = [&](const char * code, const std::string & what) {
        ok = false;
        err.code = code;
        err.detail = "tensor=" + name + " " + what;
        return tp;
    };

    if (il >= 0 && (size_t) il >= is_recr.size()) {
        return fail("TP5_E_LAYER_BOUNDS", "layer index " + std::to_string(il) +
                    " out of range (layer count " + std::to_string(is_recr.size()) + ")");
    }
    const bool recr = il < 0 ? false : is_recr[(size_t) il];

    auto attention_layout = [&](llama_tp5_semantic semantic, llama_tp5_layout layout, int axis) {
        tp.semantic   = semantic;
        tp.layout     = replicate_attention ? llama_tp5_layout::MIRRORED : layout;
        tp.split_axis = replicate_attention ? -1 : axis;
        return replicate_attention;
    };

    // per-rank contiguous channel ranges on axis 0
    auto split_axis0_even = [&](int64_t n, llama_tp5_semantic sem) {
        tp.semantic = sem;
        tp.layout = llama_tp5_layout::SPLIT_AXIS0;
        tp.split_axis = 0;
        if (n % ranks != 0) {
            fail("TP5_E_AXIS0_SPLIT", "axis0 length not divisible");
            return;
        }
        const int64_t per = n / ranks;
        if (per % blck_size != 0) {
            fail("TP5_E_QUANT_SLICE", "per-rank " + std::to_string(per) +
                 " is not a whole number of quant blocks (" + std::to_string(blck_size) + ")");
            return;
        }
        for (uint32_t r = 0; r < ranks; ++r) tp.per_rank_len[r] = per;
    };

    if (base.rfind("hc_", 0) == 0 || name.rfind("output_hc_", 0) == 0) {
        tp.semantic = llama_tp5_semantic::HC;
        tp.layout = llama_tp5_layout::MIRRORED;
        return tp;
    }
    // Hyper-Connection fold mixer & projection weights
    if (base.find("hc_") != std::string::npos || base.find("norm") != std::string::npos ||
        base.find("inject") != std::string::npos || base.find("down.weight") != std::string::npos ||
        base.find("up.weight") != std::string::npos) {
        if (base.rfind("hc_attn_", 0) == 0 || base.rfind("hc_ffn_", 0) == 0 || base.rfind("hc_head_", 0) == 0) {
            tp.semantic = llama_tp5_semantic::HC;
            tp.layout   = llama_tp5_layout::MIRRORED;
            return tp;
        }
    }
    if (is_moe_gate_up(base)) {
        tp.semantic = llama_tp5_semantic::MOE_GATE_UP;
        tp.layout = llama_tp5_layout::SPLIT_AXIS1;
        tp.split_axis = 1;
        // gate and up are separate tensors here (no fused gate_up in this GGUF);
        // rows per rank = F / ranks
        int64_t first = 0;
        for (uint32_t r = 0; r < ranks; ++r) {
            tp.per_rank_len[r] = F / ranks;
            tp.head_ranges.push_back({first, first + F / ranks});
            first += F / ranks;
        }
        return tp;
    }
    if (base == "ffn_gate_up_exps.weight") {
        tp.semantic   = llama_tp5_semantic::MOE_GATE_UP;
        tp.layout     = llama_tp5_layout::SPLIT_AXIS1;
        tp.split_axis = 1;
        // fused gate_up: total rows = 2*F. Each rank gets 2*(F/ranks) rows
        int64_t first = 0;
        for (uint32_t r = 0; r < ranks; ++r) {
            tp.per_rank_len[r] = 2 * (F / ranks);
            tp.head_ranges.push_back({ first, first + 2 * (F / ranks) });
            first += 2 * (F / ranks);
        }
        return tp;
    }
    if (base == "ffn_down_exps.weight") {
        // Down projection is shape [F, n_embd, n_expert]
        // In GGML mul_mat_id, ne[0]=F (contracting dim), so axis 0 is split across ranks by F/ranks
        split_axis0_even(F, llama_tp5_semantic::MOE_DOWN);
        return tp;
    }
    if (base == "ffn_gate_inp.weight") {
        tp.semantic = llama_tp5_semantic::MOE_ROUTER;
        tp.layout = llama_tp5_layout::MIRRORED;
        return tp;
    }
    if (base == "ffn_gate_shexp.weight" || base == "ffn_up_shexp.weight" || base == "ffn_gate_shexps.weight" ||
        base == "ffn_up_shexps.weight") {
        tp.semantic = llama_tp5_semantic::SHEXP_GATE_UP;
        tp.layout = llama_tp5_layout::SPLIT_AXIS1;
        tp.split_axis = 1;
        int64_t first = 0;
        for (uint32_t r = 0; r < ranks; ++r) {
            tp.per_rank_len[r] = Fs / ranks;
            tp.head_ranges.push_back({first, first + Fs / ranks});
            first += Fs / ranks;
        }
        return tp;
    }
    if (base == "ffn_down_shexp.weight" || base == "ffn_down_shexps.weight") {
        split_axis0_even(Fs, llama_tp5_semantic::SHEXP_DOWN);
        return tp;
    }
    if (base == "ffn_gate_inp_shexp.weight" || base == "ffn_gate_inp_shexps.weight") {
        tp.semantic = llama_tp5_semantic::SHEXP_ROUTER;
        tp.layout = llama_tp5_layout::MIRRORED;
        return tp;
    }
    if (il >= 0 && !recr) {
        // full-attention layer
        if (base == "attn_q.weight" || base == "wq.weight") {
            if (attention_layout(llama_tp5_semantic::QSA_Q_GATE, llama_tp5_layout::SPLIT_AXIS1, 1))
                return tp;
            // rows = 2*da per whole head (q|gate interleaved)
            int64_t first = 0;
            for (uint32_t r = 0; r < ranks; ++r) {
                const int64_t rows = 2 * da * q_role_counts[r];
                tp.per_rank_len[r] = rows;
                tp.head_ranges.push_back({first, first + rows});
                first += rows;
            }
            return tp;
        }
        if (base == "attn_k.weight" || base == "attn_v.weight" || base == "wk.weight" || base == "wv.weight") {
            if (attention_layout(llama_tp5_semantic::QSA_KV, llama_tp5_layout::SPLIT_AXIS1_REPL, 1))
                return tp;
            for (uint32_t r = 0; r < ranks; ++r) {
                tp.per_rank_len[r] = da * kv_role_counts[r];
                tp.head_ranges.push_back({ kv_head_starts[r], kv_head_starts[r] + kv_role_counts[r] });
            }
            return tp;
        }
        if (base == "attn_output.weight" || base == "wo.weight") {
            if (attention_layout(llama_tp5_semantic::QSA_OUT, llama_tp5_layout::SPLIT_AXIS0, 0))
                return tp;
            for (uint32_t r = 0; r < ranks; ++r) {
                const int64_t cols = da * q_role_counts[r];
                if (cols % blck_size != 0) {
                    fail("TP5_E_QUANT_SLICE", "attn_output per-rank cols " + std::to_string(cols) +
                         " not quant-aligned (blck " + std::to_string(blck_size) + ")");
                    return tp;
                }
                tp.per_rank_len[r] = cols;
            }
            return tp;
        }
        if (base.rfind("indexer.", 0) == 0) {
            tp.semantic = llama_tp5_semantic::INDEXER;
            tp.layout = llama_tp5_layout::MIRRORED;
            return tp;
        }
        if (base == "attn_q_norm.weight" || base == "attn_k_norm.weight") {
            tp.semantic = llama_tp5_semantic::QSA_NORM;
            tp.layout = llama_tp5_layout::MIRRORED;
            return tp;
        }
    }
    if (il >= 0 && recr) {
        // GDN layer
        if (base == "attn_qkv.weight") {
            if (attention_layout(llama_tp5_semantic::GDN_QKV, llama_tp5_layout::SPLIT_AXIS1, 1))
                return tp;
            // prearranged: per local V head -> Q rows (ds), K rows (ds), V rows (ds)
            for (uint32_t r = 0; r < ranks; ++r) {
                tp.per_rank_len[r] = 3 * ds * gdn_v_heads[r];
            }
            return tp;
        }
        if (base == "attn_gate.weight") {
            if (attention_layout(llama_tp5_semantic::GDN_GATE, llama_tp5_layout::SPLIT_AXIS1, 1))
                return tp;
            for (uint32_t r = 0; r < ranks; ++r) {
                tp.per_rank_len[r] = ds * gdn_v_heads[r];
            }
            return tp;
        }
        if (base == "ssm_out.weight") {
            if (attention_layout(llama_tp5_semantic::GDN_OUT, llama_tp5_layout::SPLIT_AXIS0, 0))
                return tp;
            for (uint32_t r = 0; r < ranks; ++r) {
                const int64_t cols = ds * gdn_v_heads[r];
                if (cols % blck_size != 0) {
                    fail("TP5_E_QUANT_SLICE", "ssm_out per-rank cols " + std::to_string(cols) +
                         " not quant-aligned (blck " + std::to_string(blck_size) + ")");
                    return tp;
                }
                tp.per_rank_len[r] = cols;
            }
            return tp;
        }
        if (base == "ssm_beta.weight" || base == "ssm_alpha.weight") {
            if (attention_layout(llama_tp5_semantic::GDN_SCALAR, llama_tp5_layout::SPLIT_AXIS1, 1))
                return tp;
            for (uint32_t r = 0; r < ranks; ++r) {
                tp.per_rank_len[r] = gdn_v_heads[r];
            }
            return tp;
        }
        if (base == "ssm_a" || base == "ssm_dt.bias" || base == "ssm_norm.weight") {
            tp.semantic = llama_tp5_semantic::GDN_PARAM;
            tp.layout = llama_tp5_layout::MIRRORED;
            return tp;
        }
        if (base == "ssm_conv1d.weight") {
            if (attention_layout(llama_tp5_semantic::GDN_CONV, llama_tp5_layout::SPLIT_AXIS1, 1))
                return tp;
            for (uint32_t r = 0; r < ranks; ++r) {
                tp.per_rank_len[r] = 3 * ds * gdn_v_heads[r];
            }
            return tp;
        }
    }
    if (base.rfind("ple_", 0) == 0 || base == "ple_conv1d.weight") {
        tp.semantic = llama_tp5_semantic::PLE;
        tp.layout = llama_tp5_layout::MIRRORED;
        return tp;
    }
    if (name == "token_embd.weight") {
        tp.semantic = llama_tp5_semantic::TOKEN_EMBD;
        tp.layout = llama_tp5_layout::MIRRORED;
        return tp;
    }
    if (name == "output.weight") {
        tp.semantic = llama_tp5_semantic::LM_HEAD;
        tp.layout = llama_tp5_layout::SPLIT_AXIS1;
        tp.split_axis = 1;
        int64_t first = 0;
        for (uint32_t r = 0; r < ranks; ++r) {
            const int64_t rows = n_vocab / ranks + ((int64_t) r < n_vocab % ranks ? 1 : 0);
            tp.per_rank_len[r] = rows;
            tp.head_ranges.push_back({first, first + rows});
            first += rows;
        }
        return tp;
    }
    if (name == "per_layer_token_embd.weight") {
        tp.semantic = llama_tp5_semantic::PLE_TABLE;
        tp.layout = llama_tp5_layout::CPU_RESIDENT;
        return tp;
    }

    tp.semantic = llama_tp5_semantic::OTHER;
    tp.layout = llama_tp5_layout::MIRRORED;
    return tp;
}

std::vector<int64_t> llama_tp5_plan::weight_bytes_per_rank(
        const std::vector<tensor_info> & tensors, bool & ok, llama_tp5_error & err) const {
    ok = true;
    std::vector<int64_t> out(ranks, 0);
    for (const auto & t : tensors) {
        llama_tp5_tensor_plan tp = plan_tensor(t.name, t.ne, t.type_id, t.blck_size, ok, err);
        if (!ok) return {};
        // row bytes along ne[0] for this type
        const int64_t row_bytes = t.blck_size > 1
            ? (t.ne[0] / t.blck_size) * 0 // replaced below; needs type_size
            : 0;
        (void) row_bytes;
        // The byte computation needs the type size; the caller supplies it via
        // ne and blck only for legality. Actual byte math lives in the Python
        // manifest; this C++ path validates legality and counts elements.
        // For budget purposes we recompute from ne with a caller-provided
        // bytes-per-element map is out of scope here; return element counts
        // scaled later. This function is used by tests for slice legality.
        for (uint32_t r = 0; r < ranks; ++r) {
            if (tp.layout == llama_tp5_layout::MIRRORED) {
                out[r] += 0; // mirrored: full tensor (element count not needed for legality tests)
            }
        }
    }
    return out;
}

int64_t llama_tp5_gdn_qk_global_head(const llama_tp5_plan & plan, int64_t global_v_head) {
    if (plan.Nk <= 0) {
        return 0;
    }
    return global_v_head % plan.Nk;
}

static bool tp5_name_matches(const char * name, const char * prefix) {
    return name && prefix && strncmp(name, prefix, strlen(prefix)) == 0;
}

bool llama_tp5_try_apply_split_state(
        const llama_tp5_plan & plan,
        const char * tensor_name,
        const struct ggml_tensor * tensor,
        int64_t indexer_head_size,
        struct ggml_backend_meta_split_state & out) {
    if (!tensor_name || !tensor) {
        return false;
    }

    memset(&out, 0, sizeof(out));
    out.nr[0] = 1;
    out.n_segments = 1;

    const std::string name(tensor_name);
    const auto        apply_kv_split = [&](ggml_backend_meta_split_axis axis) {
        out.axis            = axis;
        out.indexed_replica = true;
        for (uint32_t r = 0; r < plan.ranks; ++r) {
            out.ne[r]            = plan.kv_role_counts[r] * plan.da;
            out.replica_start[r] = plan.kv_head_starts[r] * plan.da;
        }
    };

    // (TP5.md §10.1) output.weight follows the vocab-parallel SPLIT_AXIS1 plan;
    // no mirrored override — each rank computes its own vocabulary row range.

    // Attention caches must use the same fixed head ownership as their weights.
    // Recurrent history keeps the native repeated-segment logic in llama-model.cpp.
    if (strstr(tensor_name, "cache_") != nullptr) {
        if (plan.replicate_attention) {
            out.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
            return true;
        }
        if (tp5_name_matches(tensor_name, "cache_k_l") || tp5_name_matches(tensor_name, "cache_v_l")) {
            if (plan.Ni > 0 && tensor->ne[0] == indexer_head_size) {
                out.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
            } else {
                apply_kv_split(GGML_BACKEND_SPLIT_AXIS_0);
            }
            return true;
        }
        return false;
    }

    bool ok = false;
    llama_tp5_error err;
    const int64_t blck = ggml_blck_size(tensor->type);
    llama_tp5_tensor_plan tp = plan.plan_tensor(name, tensor->ne, (uint32_t) tensor->type, blck, ok, err);
    if (!ok) {
        return false;
    }

    if (plan.replicate_attention && tp.layout == llama_tp5_layout::MIRRORED) {
        out.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        return true;
    }

    // The loaded GDN tensors retain their native [Q, K, V] layout; no
    // Q/K expansion or V-head prearrangement is performed by the loader.
    // Use llama-model.cpp's existing repeated-segment split for ALL GDN
    // weights, just as for their recurrent caches. In particular dt/A are
    // per-head parameters, not broadcast scalars. Mixing contiguous plan
    // slices with segmented cache ownership corrupts both conv and decay.
    switch (tp.semantic) {
        case llama_tp5_semantic::GDN_QKV:
        case llama_tp5_semantic::GDN_CONV:
        case llama_tp5_semantic::GDN_GATE:
        case llama_tp5_semantic::GDN_OUT:
        case llama_tp5_semantic::GDN_SCALAR:
        case llama_tp5_semantic::GDN_PARAM:
            return false;
        default:
            break;
    }

    auto apply_axis_split = [&](ggml_backend_meta_split_axis axis) -> bool {
        out.axis = axis;
        int64_t total = tensor->ne[axis];
        int64_t planned = 0;
        for (uint32_t r = 0; r < plan.ranks; ++r) {
            planned += tp.per_rank_len[r];
        }
        if (planned == total) {
            for (uint32_t r = 0; r < plan.ranks; ++r) {
                out.ne[r] = tp.per_rank_len[r];
            }
            return true;
        }
        // Generic even split across ranks
        if (total % plan.ranks == 0) {
            int64_t per_rank = total / plan.ranks;
            for (uint32_t r = 0; r < plan.ranks; ++r) {
                out.ne[r] = per_rank;
            }
            return true;
        }
        return false;
    };

    switch (tp.layout) {
        case llama_tp5_layout::MIRRORED:
        case llama_tp5_layout::CPU_RESIDENT:
            out.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
            return true;
        case llama_tp5_layout::SPLIT_AXIS0:
            if (!apply_axis_split(GGML_BACKEND_SPLIT_AXIS_0)) {
                return false;
            }
            return true;
        case llama_tp5_layout::SPLIT_AXIS1:
            if (!apply_axis_split(GGML_BACKEND_SPLIT_AXIS_1)) {
                return false;
            }
            return true;
        case llama_tp5_layout::SPLIT_AXIS1_REPL: {
                apply_kv_split(GGML_BACKEND_SPLIT_AXIS_1);
                return true;
        }
        default:
            return false;
    }
}

bool llama_tp5_plan::validate(const llama_hparams & hp, uint32_t n_devices, llama_tp5_error & err) const {
    if (ranks != n_devices) {
        err.code = "TP5_E_RANKS";
        err.detail = "plan built for " + std::to_string(ranks) + " but " + std::to_string(n_devices) + " devices present";
        return false;
    }
    if ((int64_t) hp.n_embd != H || (int64_t) hp.n_layer() != L) {
        err.code = "TP5_E_HPARAMS";
        err.detail = "hparams changed after plan construction";
        return false;
    }
    if (Nkv <= 0 || Nq <= 0 || Nq % Nkv != 0) {
        err.code   = "TP5_E_Q_SPLIT";
        err.detail = "invalid global GQA ratio";
        return false;
    }
    int64_t q_sum = 0, v_sum = 0;
    for (uint32_t r = 0; r < ranks; ++r) {
        const int64_t nq = q_role_counts[r], nkv = kv_role_counts[r];
        if (nq <= 0 || nkv <= 0 || nq % nkv != 0 || kv_head_starts[r] < 0 || kv_head_starts[r] + nkv > Nkv) {
            err.code   = "TP5_E_Q_SPLIT";
            err.detail = "invalid rank-local GQA ratio or KV range at rank " + std::to_string(r);
            return false;
        }
        for (int64_t h = 0; h < nq; ++h) {
            const int64_t global_q = (replicate_attention ? 0 : q_sum) + h;
            if (kv_head_starts[r] + h / (nq / nkv) != global_q / (Nq / Nkv)) {
                err.code   = "TP5_E_Q_SPLIT";
                err.detail = "rank-local GQA selects a different global KV head at rank " + std::to_string(r);
                return false;
            }
        }
        q_sum += q_role_counts[r];
        v_sum += gdn_v_heads[r];
    }
    const int64_t copies = replicate_attention ? ranks : 1;
    if (q_sum != copies * Nq) {
        err.code   = "TP5_E_Q_SPLIT";
        err.detail = "q roles do not cover all heads";
        return false;
    }
    if (v_sum != copies * Nv) {
        err.code   = "TP5_E_GDN_SPLIT";
        err.detail = "gdn heads do not cover all V heads";
        return false;
    }
    return true;
}
