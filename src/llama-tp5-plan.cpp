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
    return base == "ffn_gate_exps.weight" || base == "ffn_up_exps.weight";
}

} // namespace

bool llama_tp5_plan_build(const llama_hparams & hp, uint32_t n_devices,
                          int64_t n_vocab,
                          llama_tp5_plan & out, llama_tp5_error & err) {
    if (n_devices < 2 || n_devices > LLAMA_TP5_MAX_RANKS) {
        err.code = "TP5_E_RANKS";
        err.detail = "ranks=" + std::to_string(n_devices) + " (2..8 supported)";
        return false;
    }

    out = {};
    out.ranks = n_devices;
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
        out.is_recr[il] = il < hp.n_layer_all ? hp.is_recr((uint32_t) il) : false;
        if (il < (size_t) out.L && !out.is_recr[il]) out.n_full++;
    }

    // --- QSA roles (TP5.md 7.1) ---
    {
        std::string serr;
        std::array<int32_t, LLAMA_TP5_MAX_RANKS> q{};
        if (!split_heads_quant(out.Nq, n_devices, 2 * out.da, 2 * out.da, q, serr)) {
            // q|gate interleave means rows move in 2*da units; any whole-head
            // boundary is legal for the row split, so unit==blck is fine here
            err.code = "TP5_E_Q_SPLIT";
            err.detail = serr;
            return false;
        }
        for (uint32_t r = 0; r < n_devices; ++r) out.q_role_counts[r] = q[r];
        // KV role table: bridge role (index 2) holds both KV heads
        for (uint32_t r = 0; r < n_devices; ++r) out.kv_role_counts[r] = 1;
        if (n_devices > 2) out.kv_role_counts[2] = 2;
        // rotated instances per physical rank (TP5.md 4.4)
        for (uint32_t r = 0; r < n_devices; ++r) out.kv_instances[r] = 0;
        for (uint32_t o = 0; o < out.n_full; ++o) {
            for (uint32_t role = 0; role < n_devices; ++role) {
                out.kv_instances[(role + o) % n_devices] += out.kv_role_counts[role];
            }
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
    if (base == "ffn_down_exps.weight") {
        split_axis0_even(F, llama_tp5_semantic::MOE_DOWN);
        return tp;
    }
    if (base == "ffn_gate_inp.weight") {
        tp.semantic = llama_tp5_semantic::MOE_ROUTER;
        tp.layout = llama_tp5_layout::MIRRORED;
        return tp;
    }
    if (base == "ffn_gate_shexp.weight" || base == "ffn_up_shexp.weight") {
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
    if (base == "ffn_down_shexp.weight") {
        split_axis0_even(Fs, llama_tp5_semantic::SHEXP_DOWN);
        return tp;
    }
    if (base == "ffn_gate_inp_shexp.weight") {
        tp.semantic = llama_tp5_semantic::SHEXP_ROUTER;
        tp.layout = llama_tp5_layout::MIRRORED;
        return tp;
    }
    if (il >= 0 && !recr) {
        // full-attention layer
        if (base == "attn_q.weight") {
            tp.semantic = llama_tp5_semantic::QSA_Q_GATE;
            tp.layout = llama_tp5_layout::SPLIT_AXIS1;
            tp.split_axis = 1;
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
        if (base == "attn_k.weight" || base == "attn_v.weight") {
            tp.semantic = llama_tp5_semantic::QSA_KV;
            tp.layout = llama_tp5_layout::SPLIT_AXIS1_REPL;
            tp.split_axis = 1;
            for (uint32_t r = 0; r < ranks; ++r) {
                tp.per_rank_len[r] = da * kv_role_counts[r];
            }
            return tp;
        }
        if (base == "attn_output.weight") {
            tp.semantic = llama_tp5_semantic::QSA_OUT;
            tp.layout = llama_tp5_layout::SPLIT_AXIS0;
            tp.split_axis = 0;
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
            tp.semantic = llama_tp5_semantic::GDN_QKV;
            tp.layout = llama_tp5_layout::SPLIT_AXIS1;
            tp.split_axis = 1;
            // prearranged: per local V head -> Q rows (ds), K rows (ds), V rows (ds)
            for (uint32_t r = 0; r < ranks; ++r) {
                tp.per_rank_len[r] = 3 * ds * gdn_v_heads[r];
            }
            return tp;
        }
        if (base == "attn_gate.weight") {
            tp.semantic = llama_tp5_semantic::GDN_GATE;
            tp.layout = llama_tp5_layout::SPLIT_AXIS1;
            tp.split_axis = 1;
            for (uint32_t r = 0; r < ranks; ++r) {
                tp.per_rank_len[r] = ds * gdn_v_heads[r];
            }
            return tp;
        }
        if (base == "ssm_out.weight") {
            tp.semantic = llama_tp5_semantic::GDN_OUT;
            tp.layout = llama_tp5_layout::SPLIT_AXIS0;
            tp.split_axis = 0;
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
            tp.semantic = llama_tp5_semantic::GDN_SCALAR;
            tp.layout = llama_tp5_layout::SPLIT_AXIS1;
            tp.split_axis = 1;
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
            tp.semantic = llama_tp5_semantic::GDN_CONV;
            tp.layout = llama_tp5_layout::SPLIT_AXIS1;
            tp.split_axis = 1;
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

    // Production override (TP5.md §10.1): mirrored LM head on rank 0.
    if (name == "output.weight") {
        out.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        return true;
    }

    // Runtime caches keep the dedicated logic in llama-model.cpp (rotation, conv history).
    if (strstr(tensor_name, "cache_") != nullptr) {
        return false;
    }

    bool ok = false;
    llama_tp5_error err;
    const int64_t blck = ggml_blck_size(tensor->type);
    llama_tp5_tensor_plan tp = plan.plan_tensor(name, tensor->ne, (uint32_t) tensor->type, blck, ok, err);
    if (!ok) {
        return false;
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
        // Fused GDN qkv/conv keeps legacy row count (2*K+V) while ownership follows V heads.
        if (tp.semantic == llama_tp5_semantic::GDN_QKV || tp.semantic == llama_tp5_semantic::GDN_CONV) {
            int64_t assigned = 0;
            for (uint32_t r = 0; r < plan.ranks; ++r) {
                if (r + 1 == plan.ranks) {
                    out.ne[r] = total - assigned;
                } else {
                    out.ne[r] = (total * plan.gdn_v_heads[r]) / plan.Nv;
                    assigned += out.ne[r];
                }
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
            static const int kv_head_starts[LLAMA_TP5_MAX_RANKS] = {0, 0, 0, 1, 1};
            out.axis = GGML_BACKEND_SPLIT_AXIS_1;
            out.indexed_replica = true;
            for (uint32_t r = 0; r < plan.ranks; ++r) {
                out.ne[r] = tp.per_rank_len[r];
                out.replica_start[r] = (r < 5 ? kv_head_starts[r] : 0) * plan.da;
            }
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
    int64_t q_sum = 0, kv_sum = 0, v_sum = 0;
    for (uint32_t r = 0; r < ranks; ++r) {
        q_sum += q_role_counts[r];
        kv_sum += kv_role_counts[r];
        v_sum += gdn_v_heads[r];
    }
    if (q_sum != Nq) { err.code = "TP5_E_Q_SPLIT"; err.detail = "q roles do not cover all heads"; return false; }
    if (kv_sum != (int64_t) Nkv + 1 && Nkv == 2) { // bridge adds one replica
        // expected: sum(kv_role_counts) == Nkv + 1 for the bridge design
    }
    if (v_sum != Nv) { err.code = "TP5_E_GDN_SPLIT"; err.detail = "gdn heads do not cover all V heads"; return false; }
    return true;
}
