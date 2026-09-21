// TP5 plan implementation: pure CPU computation, no GPU/Vulkan dependencies.
// See src/llama-tp5-plan.h for the contract and TP5.md for the design.

#include "llama-tp5-plan.h"

#include "llama-arch.h"
#include "llama-hparams.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

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

        // --- Balanced QSA head map (opt-in GGML_TP5_QSA_HEADMAP=1) ---
        // Q counts [5,5,5,5,4] with starts [0,5,10,15,20] over KV counts
        // [1,1,2,1,1] with starts [0,0,0,1,1]: every rank computes a nonzero
        // attention slice. The rank-local Q/KV integer ratio is NOT the true
        // global GQA assignment (rank 2 holds Q heads from two KV groups), so
        // FLASH_ATTN_EXT consumers must use the explicit local q->kv map
        // (rank 2: [0,0,1,1,1]; all other ranks: all zeros).
        if (llama_tp5_qsa_headmap_enabled()) {
            static const int32_t Q_COUNT[5] = {5, 5, 5, 5, 4};
            static const int32_t Q_START[5] = {0, 5, 10, 15, 20};
            static const int32_t KV_COUNT[5] = {1, 1, 2, 1, 1};
            static const int32_t KV_START[5] = {0, 0, 0, 1, 1};

            const bool geometry_ok = n_devices == 5 && out.Nq == 24 && out.Nkv == 2 && out.da > 0 &&
                                     out.da % 256 == 0 && out.n_full > 0;
            if (!geometry_ok) {
                err.code   = "TP5_E_QSA_HEADMAP";
                err.detail = "balanced QSA head map requires 5 ranks, Nq=24, Nkv=2 and a head dim "
                             "multiple of 256 (got ranks=" + std::to_string(n_devices) +
                             ", Nq=" + std::to_string(out.Nq) + ", Nkv=" + std::to_string(out.Nkv) +
                             ", da=" + std::to_string(out.da) + ")";
                return false;
            }
            for (uint32_t r = 0; r < n_devices; ++r) {
                out.q_role_counts[r]  = Q_COUNT[r];
                out.kv_role_counts[r] = KV_COUNT[r];
                out.kv_head_starts[r] = KV_START[r];
                out.kv_instances[r]   = out.n_full * KV_COUNT[r];
            }
            out.local_qkv_map.assign(n_devices, {});
            for (int64_t h = 0; h < out.Nq; ++h) {
                const int64_t gkv = h / (out.Nq / out.Nkv);
                for (uint32_t r = 0; r < n_devices; ++r) {
                    for (int32_t l = 0; l < Q_COUNT[r]; ++l) {
                        if (Q_START[r] + l == h) {
                            out.local_qkv_map[r][l] = (int32_t) (gkv - KV_START[r]);
                        }
                    }
                }
            }
            out.qsa_headmap = true;
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

    // --- Balanced GDN head map (opt-in GGML_TP5_GDN_HEADMAP=1) ---
    // Main-chosen paired-head assignment (see llama-tp5-plan.h). Adjacent V
    // head pairs move together so 256-element ssm_out quant blocks stay whole;
    // each rank keeps the unique QK heads its V heads reference. Native
    // repeated-segment layout is kept when the flag is off (all arrays zero,
    // gdn_headmap_enabled = false).
    if (getenv("GGML_TP5_GDN_HEADMAP") && atoi(getenv("GGML_TP5_GDN_HEADMAP")) != 0) {
        // Per-rank global V head order and unique QK head lists from Main's
        // assignment (verified against /tmp/tp5-balanced-head-map-oracle.json:
        // 48 unique V heads covered exactly once, 22 QK copies, all adjacent pairs).
        static const int32_t V_ORDER[5][10] = {
            { 0,  1, 16, 17, 32, 33,  2,  3, 18, 19},
            {34, 35,  4,  5, 20, 21, 36, 37,  6,  7},
            {22, 23, 38, 39,  8,  9, 24, 25, 40, 41},
            {10, 11, 26, 27, 42, 43, 12, 13, 28, 29},
            {44, 45, 14, 15, 30, 31, 46, 47,  0,  0},
        };
        static const int32_t V_COUNT[5] = {10, 10, 10, 10, 8};
        static const int32_t QK_UNIQUE[5][6] = {
            { 0,  1,  2,  3,  0, 0},
            { 2,  3,  4,  5,  6, 7},
            { 6,  7,  8,  9,  0, 0},
            {10, 11, 12, 13,  0, 0},
            {12, 13, 14, 15,  0, 0},
        };
        static const int32_t QK_COUNT[5] = {4, 6, 4, 4, 4};

        const bool geometry_ok = out.Nv == 48 && out.Nk == 16 && n_devices == 5 && out.ds > 0 &&
                                 (2 * out.ds) % 256 == 0; // V head PAIRS move together: 2*ds = one quant block
        if (!geometry_ok) {
            err.code = "TP5_E_GDN_HEADMAP";
            err.detail = "balanced GDN head map requires Nv=48, Nk=16, 5 ranks and "
                         "ssm_d_state pairs not quant-aligned, 2*ds % 256 != 0 (got Nv=" + std::to_string(out.Nv) +
                         ", Nk=" + std::to_string(out.Nk) + ", ranks=" + std::to_string(n_devices) +
                         ", ds=" + std::to_string(out.ds) + ")";
            return false;
        }
        // Verify coverage: every global V head exactly once, every V pair adjacent,
        // every referenced QK head present in the rank's unique list.
        {
            std::vector<int> v_seen(out.Nv, 0);
            for (uint32_t r = 0; r < n_devices; ++r) {
                for (int32_t i = 0; i < V_COUNT[r]; ++i) {
                    const int32_t h = V_ORDER[r][i];
                    if (h < 0 || h >= out.Nv) { err.code = "TP5_E_GDN_HEADMAP"; err.detail = "V head out of range"; return false; }
                    v_seen[h]++;
                    // pair invariant: every head's partner (h^1) must sit at
                    // an adjacent slot so 256-element ssm_out quant blocks
                    // (2*ds) stay inside single spans
                    {
                        const int32_t partner = h ^ 1;
                        const bool adj = (i > 0 && V_ORDER[r][i - 1] == partner) ||
                                        (i + 1 < V_COUNT[r] && V_ORDER[r][i + 1] == partner);
                        if (!adj) {
                            err.code = "TP5_E_GDN_HEADMAP"; err.detail = "V heads must move in adjacent pairs"; return false;
                        }
                    }
                    const int32_t qk = h % out.Nk;
                    bool found = false;
                    for (int32_t q = 0; q < QK_COUNT[r]; ++q) {
                        found = found || QK_UNIQUE[r][q] == qk;
                    }
                    if (!found) { err.code = "TP5_E_GDN_HEADMAP"; err.detail = "rank V head references QK head outside its unique list"; return false; }
                }
            }
            for (int64_t h = 0; h < out.Nv; ++h) {
                if (v_seen[h] != 1) { err.code = "TP5_E_GDN_HEADMAP"; err.detail = "V heads must be covered exactly once"; return false; }
            }
        }
        for (uint32_t r = 0; r < n_devices; ++r) {
            out.gdn_v_global_count[r]  = V_COUNT[r];
            out.gdn_qk_unique_count[r] = QK_COUNT[r];
            for (int32_t i = 0; i < V_COUNT[r]; ++i)  out.gdn_v_global[r][i]  = V_ORDER[r][i];
            for (int32_t i = 0; i < QK_COUNT[r]; ++i) out.gdn_qk_unique[r][i] = QK_UNIQUE[r][i];
        }
        out.gdn_headmap_enabled = true;
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
            if (gdn_headmap_enabled && base != "ssm_norm.weight") {
                if (attention_layout(llama_tp5_semantic::GDN_SCALAR, llama_tp5_layout::SPLIT_AXIS0, 0))
                    return tp;
                for (uint32_t r = 0; r < ranks; ++r) tp.per_rank_len[r] = gdn_v_global_count[r];
                return tp;
            }
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

int32_t llama_tp5_gdn_local_qk(const llama_tp5_plan & plan, uint32_t rank, int32_t v_idx) {
    if (!plan.gdn_headmap_enabled || rank >= plan.ranks ||
            v_idx < 0 || v_idx >= plan.gdn_v_global_count[rank]) {
        return -1;
    }
    const int32_t global_v  = plan.gdn_v_global[rank][v_idx];
    const int32_t global_qk = (int32_t) llama_tp5_gdn_qk_global_head(plan, global_v);
    for (int32_t q = 0; q < plan.gdn_qk_unique_count[rank]; ++q) {
        if (plan.gdn_qk_unique[rank][q] == global_qk) {
            return q;
        }
    }
    return -1; // unreachable: plan build validated QK coverage
}

bool llama_tp5_gdn_headmap_stamp(const llama_tp5_plan & plan, uint32_t rank, struct ggml_tensor * gdn_node) {
    if (!plan.gdn_headmap_enabled || rank >= plan.ranks || gdn_node == nullptr ||
        gdn_node->op != GGML_OP_GATED_DELTA_NET || !gdn_node->src[0] || !gdn_node->src[1] ||
        !gdn_node->src[2] || !gdn_node->src[5]) {
        return false;
    }
    const int32_t count = plan.gdn_v_global_count[rank];
    if (count < 1 || count > GGML_TP5_HEADMAP_MAX_ENTRIES ||
        gdn_node->src[0]->ne[1] != plan.gdn_qk_unique_count[rank] ||
        gdn_node->src[1]->ne[1] != plan.gdn_qk_unique_count[rank] ||
        gdn_node->src[2]->ne[1] != count || gdn_node->src[5]->ne[2] != count) {
        return false;
    }
    uint8_t local_qk[GGML_TP5_HEADMAP_MAX_ENTRIES] = {};
    for (int32_t i = 0; i < count; ++i) {
        const int32_t q = llama_tp5_gdn_local_qk(plan, rank, i);
        if (q < 0 || q >= 8) {
            return false;
        }
        local_qk[i] = (uint8_t) q;
    }
    ggml_tp5_headmap_set(gdn_node, local_qk, count);
    return true;
}

bool llama_tp5_qsa_headmap_enabled(void) {
    const char * env = getenv("GGML_TP5_QSA_HEADMAP");
    return env && atoi(env) != 0;
}

int64_t llama_tp5_qsa_global_kv_head(const llama_tp5_plan & plan, int64_t global_q_head) {
    if (plan.Nkv <= 0) {
        return 0;
    }
    return global_q_head / (plan.Nq / plan.Nkv);
}

bool llama_tp5_qsa_headmap_stamp(const llama_tp5_plan & plan, uint32_t rank, struct ggml_tensor * fa_node) {
    if (!plan.qsa_headmap || fa_node == nullptr ||
        (fa_node->op != GGML_OP_FLASH_ATTN_EXT && fa_node->op != GGML_OP_FLASH_ATTN_EXT_REROT)) {
        return false;
    }
    if (rank >= plan.ranks || rank >= plan.local_qkv_map.size()) {
        return false;
    }
    const int32_t count = plan.q_role_counts[rank];
    if (count <= 0 || count > GGML_TP5_HEADMAP_MAX_ENTRIES) {
        return false;
    }
    // Only stamp nodes whose rank-local geometry matches the plan's Q/KV
    // role counts; anything else is an ordinary node and stays native.
    if (fa_node->src[0] == nullptr || fa_node->src[1] == nullptr || fa_node->src[2] == nullptr ||
            fa_node->src[0]->ne[2] != count ||
            fa_node->src[1]->ne[2] != plan.kv_role_counts[rank] ||
            fa_node->src[2]->ne[2] != plan.kv_role_counts[rank]) {
        return false;
    }
    // The map overlays the unused m0/m1 ALiBi push constants: fail closed on
    // a mapped node that actually uses ALiBi instead of silently corrupting it.
    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) fa_node->op_params + 1, sizeof(float));
    if (max_bias != 0.0f) {
        return false;
    }
    uint8_t map[GGML_TP5_HEADMAP_MAX_ENTRIES];
    for (int32_t h = 0; h < count; ++h) {
        const int32_t kv = plan.local_qkv_map[rank][h];
        if (kv < 0 || kv >= 8) {
            return false;
        }
        map[h] = (uint8_t) kv;
    }
    ggml_tp5_headmap_set(fa_node, map, count);
    return true;
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
        if (plan.gdn_headmap_enabled && tp5_name_matches(tensor_name, "cache_ple_r_l")) {
            out.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
            return true;
        }
        if (plan.gdn_headmap_enabled &&
                (tp5_name_matches(tensor_name, "cache_r_l") || tp5_name_matches(tensor_name, "cache_s_l") ||
                 tp5_name_matches(tensor_name, "cache_d_l"))) {
            // Recurrent caches under the balanced GDN head map: the flat
            // storage root is axis 0 with per-head (or per-head-pair for the
            // 256-element state blocks) mapped spans following the same V
            // order as the weights; derived 4D state views carry the map to
            // axis 2 via the meta reshape propagation.
            // Conv history has (kernel-1) consecutive values per channel;
            // recurrent state has ds*ds consecutive values per V head.
            const bool is_conv = tp5_name_matches(tensor_name, "cache_r_l");
            out.axis = GGML_BACKEND_SPLIT_AXIS_0;
            out.mapped_span = true;
            out.n_segments = 1;
            out.nr[0] = 1;
            const int64_t channels = plan.ds * (2 * plan.Nk + plan.Nv);
            GGML_ASSERT(is_conv ? tensor->ne[0] % channels == 0 :
                                  tensor->ne[0] == plan.ds * plan.ds * plan.Nv);
            const int64_t ds = is_conv ? plan.ds * (tensor->ne[0] / channels) : plan.ds * plan.ds;
            GGML_ASSERT(ds > 0);
            int32_t flat = 0;
            for (uint32_t r = 0; r < plan.ranks; ++r) {
                int32_t n_spans = 0;
                if (is_conv) {
                    // [Q unique | K unique | V ordered] like the QKV rows
                    const int32_t nq = plan.gdn_qk_unique_count[r];
                    for (int pass = 0; pass < 2; ++pass) {
                        const int64_t off = pass == 0 ? 0 : plan.Nk * ds;
                        int32_t i = 0;
                        while (i < nq) {
                            int32_t run = 1;
                            while (i + run < nq &&
                                   plan.gdn_qk_unique[r][i + run] == plan.gdn_qk_unique[r][i + run - 1] + 1) {
                                ++run;
                            }
                            out.span_start[flat] = off + (int64_t) plan.gdn_qk_unique[r][i] * ds;
                            out.span_len[flat]   = run * ds;
                            ++flat; ++n_spans;
                            i += run;
                        }
                    }
                    // V ordered at 2*Nk*ds
                    {
                        int32_t i = 0;
                        const int32_t nv = plan.gdn_v_global_count[r];
                        while (i < nv) {
                            int32_t run = 1;
                            while (i + run < nv &&
                                   plan.gdn_v_global[r][i + run] == plan.gdn_v_global[r][i + run - 1] + 1) {
                                ++run;
                            }
                            out.span_start[flat] = 2 * plan.Nk * ds + (int64_t) plan.gdn_v_global[r][i] * ds;
                            out.span_len[flat]   = run * ds;
                            ++flat; ++n_spans;
                            i += run;
                        }
                    }
                    out.ne[r] = 2 * ds * nq + ds * plan.gdn_v_global_count[r];
                } else {
                    // One span per contiguous V-head run, in state elements.
                    const int32_t nv = plan.gdn_v_global_count[r];
                    int32_t i = 0;
                    while (i < nv) {
                        int32_t run = 1;
                        while (i + run < nv &&
                               plan.gdn_v_global[r][i + run] == plan.gdn_v_global[r][i + run - 1] + 1) {
                            ++run;
                        }
                        out.span_start[flat] = (int64_t) plan.gdn_v_global[r][i] * ds;
                        out.span_len[flat]   = run * ds;
                        ++flat; ++n_spans;
                        i += run;
                    }
                    out.ne[r] = ds * nv;
                }
                out.span_count[r] = n_spans;
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
    // Balanced GDN head map (opt-in GGML_TP5_GDN_HEADMAP=1): GDN weights and
    // recurrent caches are expressed as bounded mapped spans over the
    // original element ranges instead of the native repeated-segment split.
    // QKV/conv rows are [Q unique | K unique | V ordered] per rank: whole
    // quant-row copies, no requantization; V/state/output columns keep the
    // same global V identities (48 covered exactly once). With the flag off,
    // the native repeated-segment fallback below is used unchanged.
    if (plan.gdn_headmap_enabled) {
        const int64_t ds = plan.ds;
        // per-rank unique QK head spans: each QK head h contributes a Q span
        // [h*ds,(h+1)*ds) and a K span [Nk*ds + h*ds, Nk*ds + (h+1)*ds) —
        // adjacent Q/K pairs coalesce into one span each where contiguous.
        // Local storage layout is [ALL unique Q][ALL unique K][V ordered]:
        // emit the full Q pass first, then the full K pass (coalescing
        // contiguous runs inside each pass), then the V spans.
        auto push_qk_spans = [&](uint32_t r, int32_t & n_spans, int32_t & flat, int64_t k_off) {
            const int32_t nq = plan.gdn_qk_unique_count[r];
            for (int pass = 0; pass < 2; ++pass) {
                const int64_t off = pass == 0 ? 0 : k_off;
                int32_t i = 0;
                while (i < nq) {
                    int32_t run = 1;
                    while (i + run < nq &&
                           plan.gdn_qk_unique[r][i + run] == plan.gdn_qk_unique[r][i + run - 1] + 1) {
                        ++run;
                    }
                    const int32_t h = plan.gdn_qk_unique[r][i];
                    out.span_start[flat] = off + (int64_t) h * ds;
                    out.span_len[flat]   = run * ds;
                    ++flat; ++n_spans;
                    i += run;
                }
            }
        };
        auto push_v_spans = [&](uint32_t r, int32_t & n_spans, int32_t & flat, int64_t v_off) {
            const int32_t nv = plan.gdn_v_global_count[r];
            int32_t i = 0;
            while (i < nv) {
                int32_t run = 1;
                while (i + run < nv &&
                       plan.gdn_v_global[r][i + run] == plan.gdn_v_global[r][i + run - 1] + 1) {
                    ++run;
                }
                out.span_start[flat] = v_off + (int64_t) plan.gdn_v_global[r][i] * ds;
                out.span_len[flat]   = run * ds;
                ++flat; ++n_spans;
                i += run;
            }
        };

        switch (tp.semantic) {
            case llama_tp5_semantic::GDN_QKV:
            case llama_tp5_semantic::GDN_CONV: {
                // rows: [Q unique | K unique | V ordered] (axis 1)
                out.axis = GGML_BACKEND_SPLIT_AXIS_1;
                out.mapped_span = true;
                out.n_segments = 1;
                out.nr[0] = 1;
                int32_t flat = 0;
                for (uint32_t r = 0; r < plan.ranks; ++r) {
                    int32_t n_spans = 0;
                    push_qk_spans(r, n_spans, flat, plan.Nk * ds);
                    push_v_spans(r, n_spans, flat, 2 * plan.Nk * ds);
                    out.span_count[r] = n_spans;
                    // local rows: Q unique + K unique + V ordered
                    out.ne[r] = 2 * ds * plan.gdn_qk_unique_count[r] +
                                ds * plan.gdn_v_global_count[r];
                }
                return true;
            }
            case llama_tp5_semantic::GDN_GATE: {
                // gate rows: V ordered (axis 1)
                out.axis = GGML_BACKEND_SPLIT_AXIS_1;
                out.mapped_span = true;
                out.n_segments = 1;
                out.nr[0] = 1;
                int32_t flat = 0;
                for (uint32_t r = 0; r < plan.ranks; ++r) {
                    int32_t n_spans = 0;
                    push_v_spans(r, n_spans, flat, 0);
                    out.span_count[r] = n_spans;
                    out.ne[r] = ds * plan.gdn_v_global_count[r];
                }
                return true;
            }
            case llama_tp5_semantic::GDN_OUT: {
                // ssm_out columns: V ordered (axis 0); whole 256-element
                // original head-pair blocks stay inside single spans.
                out.axis = GGML_BACKEND_SPLIT_AXIS_0;
                out.mapped_span = true;
                out.n_segments = 1;
                out.nr[0] = 1;
                int32_t flat = 0;
                for (uint32_t r = 0; r < plan.ranks; ++r) {
                    int32_t n_spans = 0;
                    push_v_spans(r, n_spans, flat, 0);
                    out.span_count[r] = n_spans;
                    out.ne[r] = ds * plan.gdn_v_global_count[r];
                }
                return true;
            }
            case llama_tp5_semantic::GDN_SCALAR: {
                // Projection rows and per-head dt/A vectors share V identity.
                out.axis = ggml_backend_meta_split_axis(tp.split_axis);
                out.mapped_span = true;
                out.n_segments = 1;
                out.nr[0] = 1;
                int32_t flat = 0;
                for (uint32_t r = 0; r < plan.ranks; ++r) {
                    const int32_t nv = plan.gdn_v_global_count[r];
                    int32_t n_spans = 0;
                    int32_t i = 0;
                    while (i < nv) {
                        int32_t run = 1;
                        while (i + run < nv &&
                               plan.gdn_v_global[r][i + run] == plan.gdn_v_global[r][i + run - 1] + 1) {
                            ++run;
                        }
                        out.span_start[flat] = plan.gdn_v_global[r][i];
                        out.span_len[flat]   = run;
                        ++flat; ++n_spans;
                        i += run;
                    }
                    out.span_count[r] = n_spans;
                    out.ne[r] = nv;
                }
                return true;
            }
            default:
                break;
        }
    }

    // Native fallback (flag off): the loaded GDN tensors retain their native
    // [Q, K, V] layout; llama-model.cpp's existing repeated-segment split is
    // used for ALL GDN weights, just as for their recurrent caches. In
    // particular dt/A are per-head parameters, not broadcast scalars. Mixing
    // contiguous plan slices with segmented cache ownership corrupts both
    // conv and decay.
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
        // With the QSA head map the rank-local Q/KV ratio may be nonuniform
        // (rank 2: Q5 over KV2), so the integer-ratio check only applies to
        // the native layout. The mapped assignment itself is verified below.
        if (nq <= 0 || nkv <= 0 || kv_head_starts[r] < 0 || kv_head_starts[r] + nkv > Nkv ||
                (!qsa_headmap && nq % nkv != 0)) {
            err.code   = "TP5_E_Q_SPLIT";
            err.detail = "invalid rank-local GQA ratio or KV range at rank " + std::to_string(r);
            return false;
        }
        if (!qsa_headmap) {
            for (int64_t h = 0; h < nq; ++h) {
                const int64_t global_q = (replicate_attention ? 0 : q_sum) + h;
                if (kv_head_starts[r] + h / (nq / nkv) != global_q / (Nq / Nkv)) {
                    err.code   = "TP5_E_Q_SPLIT";
                    err.detail = "rank-local GQA selects a different global KV head at rank " + std::to_string(r);
                    return false;
                }
            }
        }
        q_sum += nq;
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
    if (qsa_headmap) {
        if (local_qkv_map.size() != ranks) {
            err.code   = "TP5_E_QSA_HEADMAP";
            err.detail = "local q->kv map missing for the mapped layout";
            return false;
        }
        for (uint32_t r = 0; r < ranks; ++r) {
            for (int32_t h = 0; h < q_role_counts[r]; ++h) {
                const int32_t kv = local_qkv_map[r][h];
                if (kv < 0 || kv >= kv_role_counts[r]) {
                    err.code   = "TP5_E_QSA_HEADMAP";
                    err.detail = "local q->kv map out of range at rank " + std::to_string(r);
                    return false;
                }
            }
        }
    }
    return true;
}
