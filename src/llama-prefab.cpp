#include "llama-prefab.h"

#include "llama-prefab-env.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-kv-cache.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <cstdlib>
#include <cstdio>

//
// topology key / cache / state
//

bool llama_prefab_topology_key::operator==(const llama_prefab_topology_key & o) const {
    return model == o.model && memory == o.memory && arch == o.arch &&
           n_embd == o.n_embd && n_layer == o.n_layer && n_head == o.n_head && n_head_kv == o.n_head_kv &&
           kv_size == o.kv_size && type_k == o.type_k && type_v == o.type_v &&
           flash_attn == o.flash_attn && v_trans == o.v_trans && causal == o.causal && warmup == o.warmup &&
           rope_freq_base == o.rope_freq_base && rope_freq_scale == o.rope_freq_scale;
}

bool llama_prefab_factor_module::owns(const ggml_tensor * t) const {
    ggml_context * ctx = res->get_ctx();
    const char * b = (const char *) ggml_get_mem_buffer(ctx);
    const char * e = b + ggml_get_mem_size(ctx);
    const char * p = (const char *) t;
    return p >= b && p < e;
}

bool llama_prefab_factor_cache::validate_or_reset(const llama_prefab_topology_key & k) {
    if (has_key && key == k) {
        return false;
    }
    const bool had = has_key && !modules.empty();
    clear();
    key     = k;
    has_key = true;
    return had;
}

void llama_prefab_factor_cache::clear() {
    modules.clear();
    has_key = false;
    key = {};
}

int llama_prefab_state::slot_for(uint32_t n_tokens) const {
    for (size_t i = 0; i < slots.size(); ++i) {
        if (n_tokens <= slots[i].n_env) {
            return (int) i;
        }
    }
    return -1;
}

void llama_prefab_state::sync_pending() {
    // every slot whose graph was executed since the last synchronize may still be running: the
    // output copies read from its buffers, and resetting the slot would free them under the
    // backend. Drain all of them, not just the last one that became active.
    for (ggml_backend_sched_t s : pending) {
        if (s != nullptr) {
            ggml_backend_sched_synchronize(s);
        }
    }
    pending.clear();
}

void llama_prefab_state::mark_pending(ggml_backend_sched_t s) {
    // the envelope graph was just dispatched on s: its outputs are read asynchronously and its
    // buffers must not be reset/freed before a synchronize drains it
    if (s == nullptr) {
        return;
    }
    for (ggml_backend_sched_t p : pending) {
        if (p == s) {
            return;
        }
    }
    pending.push_back(s);
}

void llama_prefab_state::reset_slots() {
    // nothing may be reset while a slot's graph is still in flight
    sync_pending();
    for (auto & s : slots) {
        if (s.res) {
            s.res->reset();
        }
        if (s.sched) {
            ggml_backend_sched_reset(s.sched.get());
        }
        s.built = false;
    }
    if (llama_prefab_mtp_lifetime_enabled()) {
        mtp_a.clear();
    }
    active = nullptr;
}

//
// envelope inputs
//

bool llm_graph_input_is_prefab(const llm_graph_input_i * inp) {
    return dynamic_cast<const llm_graph_input_prefab_tokens  *>(inp) != nullptr ||
           dynamic_cast<const llm_graph_input_prefab_pos     *>(inp) != nullptr ||
           dynamic_cast<const llm_graph_input_prefab_out_ids *>(inp) != nullptr ||
           dynamic_cast<const llm_graph_input_prefab_attn_kv *>(inp) != nullptr;
}

void llm_graph_input_prefab_tokens::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(ubatch->token != nullptr);
    const uint32_t n = ubatch->n_tokens;
    GGML_ASSERT(n >= 1 && n <= n_env);
    std::vector<int32_t> data(n_env);
    for (uint32_t i = 0; i < n_env; ++i) {
        data[i] = ubatch->token[i < n ? i : n - 1];
    }
    ggml_backend_tensor_set(tokens, data.data(), 0, n_env*sizeof(int32_t));
}

bool llm_graph_input_prefab_tokens::can_reuse(const llm_graph_params & params) {
    return tokens->ne[0] == (int64_t) params.ubatch.n_tokens && params.ubatch.n_tokens == n_env;
}

void llm_graph_input_prefab_pos::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(ubatch->pos != nullptr);
    const uint32_t n = ubatch->n_tokens;
    GGML_ASSERT(n >= 1 && n <= n_env);
    std::vector<int32_t> data(n_env);
    for (uint32_t i = 0; i < n_env; ++i) {
        data[i] = ubatch->pos[i < n ? i : n - 1];
    }
    ggml_backend_tensor_set(pos, data.data(), 0, n_env*sizeof(int32_t));
}

bool llm_graph_input_prefab_pos::can_reuse(const llm_graph_params & params) {
    return pos->ne[0] == (int64_t) params.ubatch.n_tokens && params.ubatch.n_tokens == n_env;
}

void llm_graph_input_prefab_out_ids::set_input(const llama_ubatch * ubatch) {
    const uint32_t n = ubatch->n_tokens;
    GGML_ASSERT(n >= 1 && n <= n_env);
    std::vector<int32_t> data;
    data.reserve(n_env);
    for (uint32_t i = 0; i < n; ++i) {
        if (ubatch->output == nullptr || ubatch->output[i]) {
            data.push_back((int32_t) i);
        }
    }
    // pad rows: duplicate the last real output (or the last real row if there is no output)
    const int32_t pad = data.empty() ? (int32_t) (n - 1) : data.back();
    while (data.size() < n_env) {
        data.push_back(pad);
    }
    ggml_backend_tensor_set(out_ids, data.data(), 0, n_env*sizeof(int32_t));
}

bool llm_graph_input_prefab_out_ids::can_reuse(const llm_graph_params & params) {
    return out_ids->ne[0] == (int64_t) params.n_outputs && params.n_outputs == n_env;
}

void llm_graph_input_prefab_attn_kv::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(readers.empty());
    GGML_ASSERT(st != nullptr);
    const uint32_t n = ubatch->n_tokens;
    GGML_ASSERT(n >= 1 && n <= n_env);
    GGML_ASSERT(n == n_env || st->scratch_cell >= 0);

    mctx->prefab_set_input_idxs(self_k_idxs, self_v_idxs, ubatch, n_env, st->scratch_cell);

    if (self_kq_mask && self_kq_mask->buffer) {
        // the mask is [n_kv_pad, n_env, 1, 1]; the writer masks pad columns
        // [live n_kv, n_kv_pad) with -inf so the extra K/V view columns never contribute
        mctx->prefab_set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn, n_env);
    }
    if (self_k_rot && self_k_rot->buffer) {
        mctx->set_input_k_rot(self_k_rot);
    }
    if (self_v_rot && self_v_rot->buffer) {
        mctx->set_input_v_rot(self_v_rot);
    }
}

bool llm_graph_input_prefab_attn_kv::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    if (!readers.empty() || mctx->has_attention_views()) {
        return false;
    }

    // re-bind the (possibly new) context, like the stock can_reuse
    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;

    // n_kv is a runtime parameter: the mask/K/V-view width is the padded width, and the
    // graph is reused as long as the live n_kv fits it (the mask writer masks the pad
    // columns). ne[1]/ne[2]/ne[3] checks as in the stock can_reuse_kq_mask, with n_env rows.
    res &= self_kq_mask->ne[0] == (int64_t) n_kv_pad;
    res &= self_kq_mask->ne[1] == (int64_t) n_env;
    res &= self_kq_mask->ne[2] == 1;
    res &= self_kq_mask->ne[3] == 1;
    res &= mctx->n_kv_pad_fits(n_kv_pad);

    return res;
}

//
// factor modules
//

static const llm_graph_cb & prefab_cb() {
    // names only: the stock callback pins "norm"/"l_last" on the *main* sched, which never runs envelopes
    // (the llama builder never emits those names, so this is placement-neutral for the supported archs)
    static const llm_graph_cb cb = [](const llama_ubatch &, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }
    };
    return cb;
}

static constexpr int64_t PREFAB_FACTOR_MAX_NODES = 2048;

struct prefab_module_builder {
    std::unique_ptr<llama_prefab_factor_module> m;
    llm_graph_params                            p;
    std::unique_ptr<llm_graph_context>          g;

    prefab_module_builder(const llm_graph_params & params, const llama_prefab_factor_key & key) {
        m = std::make_unique<llama_prefab_factor_module>();
        m->key = key;
        m->res = std::make_unique<llm_graph_result>(PREFAB_FACTOR_MAX_NODES);
        p = params;
        p.res = m->res.get();
        p.cb  = prefab_cb();
        g = std::make_unique<llm_graph_context>(p);
    }

    ggml_tensor * input(const char * name, ggml_type type, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1) {
        ggml_tensor * t = ggml_new_tensor_4d(g->ctx0, type, ne0, ne1, ne2, ne3);
        ggml_format_name(t, "pf_in_%s", name);
        m->inputs[name] = t;
        return t;
    }

    // placeholder with the exact layout (ne + nb) of a producer tensor of another factor
    ggml_tensor * input_like(const char * name, const ggml_tensor * ref) {
        ggml_tensor * t = ggml_new_tensor(g->ctx0, ref->type, GGML_MAX_DIMS, ref->ne);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            t->nb[i] = ref->nb[i];
        }
        ggml_format_name(t, "pf_in_%s", name);
        m->inputs[name] = t;
        return t;
    }

    void bind(const char * name, ggml_tensor * t) {
        if (t == nullptr) {
            return;
        }
        m->inputs[name] = t;
    }

    void output(const char * name, ggml_tensor * t) {
        GGML_ASSERT(t != nullptr);
        ggml_build_forward_expand(g->gf, t);
        m->outputs[name] = t;
    }

    std::unique_ptr<llama_prefab_factor_module> finish() {
        g.reset();
        return std::move(m);
    }
};

static std::unique_ptr<llama_prefab_factor_module> prefab_compile_embd(
        const llama_model & model, const llm_graph_params & params, uint32_t n_env) {
    prefab_module_builder b(params, { LLAMA_PREFAB_FACTOR_EMBD, -1, n_env, 0 });
    auto & g = *b.g;

    ggml_tensor * tokens = b.input("tokens", GGML_TYPE_I32, n_env);
    ggml_tensor * cur = ggml_get_rows(g.ctx0, model.tok_embd, tokens);
    if (g.hparams.f_embedding_scale != 0.0f) {
        cur = ggml_scale(g.ctx0, cur, g.hparams.f_embedding_scale);
    }
    b.output("inpL", cur);
    return b.finish();
}

static std::unique_ptr<llama_prefab_factor_module> prefab_compile_qkv(
        const llama_model & model, const llm_graph_params & params, int il, uint32_t n_env) {
    prefab_module_builder b(params, { LLAMA_PREFAB_FACTOR_QKV, il, n_env, 0 });
    auto & g = *b.g;

    const int64_t n_embd_head = g.hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == g.hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == g.n_rot);

    ggml_tensor * inpL    = b.input("inpL",    GGML_TYPE_F32, g.n_embd, n_env);
    ggml_tensor * inp_pos = b.input("inp_pos", GGML_TYPE_I32, n_env);

    // == llama_model_llama::graph<false> (src/models/llama.cpp), per-layer attention prologue
    ggml_tensor * cur = g.build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
    g.cb(cur, "attn_norm", il);

    ggml_tensor * rope_factors = model.get_rope_factors(g.cparams, il);

    auto [Qcur, Kcur, Vcur] = g.build_qkv(model.layers[il], cur, n_embd_head, g.n_head, g.n_head_kv, il);

    Qcur = ggml_rope_ext(g.ctx0, Qcur, inp_pos, rope_factors,
            g.n_rot, g.rope_type, g.n_ctx_orig, g.freq_base, g.freq_scale,
            g.ext_factor, g.attn_factor, g.beta_fast, g.beta_slow);
    Kcur = ggml_rope_ext(g.ctx0, Kcur, inp_pos, rope_factors,
            g.n_rot, g.rope_type, g.n_ctx_orig, g.freq_base, g.freq_scale,
            g.ext_factor, g.attn_factor, g.beta_fast, g.beta_slow);

    g.cb(Qcur, "Qcur", il);
    g.cb(Kcur, "Kcur", il);
    g.cb(Vcur, "Vcur", il);

    if (g.hparams.use_kq_norm) {
        Qcur = ggml_rms_norm(g.ctx0, Qcur, g.hparams.f_norm_rms_eps);
        Kcur = ggml_rms_norm(g.ctx0, Kcur, g.hparams.f_norm_rms_eps);
        g.cb(Qcur, "Qcur_normed", il);
        g.cb(Kcur, "Kcur_normed", il);
    }

    // same expansion order as build_attn (q, v, k)
    b.output("q", Qcur);
    b.output("v", Vcur);
    b.output("k", Kcur);
    return b.finish();
}

static std::unique_ptr<llama_prefab_factor_module> prefab_compile_attn(
        const llama_model & model, const llm_graph_params & params, int il, uint32_t n_env, uint32_t n_kv_pad,
        const llama_prefab_factor_module & qkv) {
    prefab_module_builder b(params, { LLAMA_PREFAB_FACTOR_ATTN, il, n_env, n_kv_pad });
    auto & g = *b.g;

    const auto * kvc = static_cast<const llama_kv_cache_context *>(g.mctx);
    // n_kv is a runtime parameter: the factor is compiled for the padded width, which must
    // cover the current n_kv (the envelope asserts the same before composing)
    GGML_ASSERT(kvc->n_kv_pad_fits(n_kv_pad));

    const int64_t n_embd_head = g.hparams.n_embd_head_v();
    const float kq_scale = g.hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : g.hparams.f_attention_scale;

    ggml_tensor * q = b.input_like("q", qkv.outputs.at("q"));
    ggml_tensor * k = b.input_like("k", qkv.outputs.at("k"));
    ggml_tensor * v = b.input_like("v", qkv.outputs.at("v"));

    // formal KV inputs: same constructors as the envelope inputs -> same shapes/types
    llm_graph_input_attn_kv inp(g.hparams, g.cparams, kvc);
    inp.n_kv_pad = n_kv_pad; // build_attn sizes the K/V views by the padded width
    inp.self_k_idxs = kvc->build_input_k_idxs(g.ctx0, g.ubatch);
    inp.self_v_idxs = kvc->build_input_v_idxs(g.ctx0, g.ubatch);
    inp.self_kq_mask = ggml_new_tensor_4d(g.ctx0, g.cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, n_kv_pad, n_env, 1, 1);
    inp.self_kq_mask_cnv = inp.self_kq_mask;
    inp.self_k_rot = kvc->build_input_k_rot(g.ctx0);
    inp.self_v_rot = kvc->build_input_v_rot(g.ctx0);

    b.bind("k_idxs",  inp.self_k_idxs);
    b.bind("v_idxs",  inp.self_v_idxs);
    b.bind("kq_mask", inp.self_kq_mask);
    b.bind("k_rot",   inp.self_k_rot);
    b.bind("v_rot",   inp.self_v_rot);

    ggml_tensor * cur = g.build_attn(&inp,
            model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
            q, k, v, nullptr, nullptr, nullptr, kq_scale, il);
    g.cb(cur, "attn_out", il);

    b.output("out", cur);
    return b.finish();
}

static std::unique_ptr<llama_prefab_factor_module> prefab_compile_ffn(
        const llama_model & model, const llm_graph_params & params, int il, uint32_t n_env) {
    prefab_module_builder b(params, { LLAMA_PREFAB_FACTOR_FFN, il, n_env, 0 });
    auto & g = *b.g;

    ggml_tensor * cur   = b.input("cur",   GGML_TYPE_F32, g.n_embd, n_env);
    ggml_tensor * inpSA = b.input("inpSA", GGML_TYPE_F32, g.n_embd, n_env);

    ggml_tensor * ffn_inp = ggml_add(g.ctx0, cur, inpSA);
    g.cb(ffn_inp, "ffn_inp", il);

    if (model.layers[il].ffn_gate_inp == nullptr) {
        cur = g.build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        g.cb(cur, "ffn_norm", il);

        cur = g.build_ffn(cur,
                model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, model.layers[il].ffn_gate_s,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, model.layers[il].ffn_down_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        g.cb(cur, "ffn_out", il);
    } else {
        cur = g.build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        g.cb(cur, "ffn_norm", il);

        cur = g.build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                nullptr,
                g.n_expert, g.n_expert_used,
                LLM_FFN_SILU, true,
                g.hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                il,
                nullptr, nullptr,
                model.layers[il].ffn_up_exps_s,
                model.layers[il].ffn_gate_exps_s,
                model.layers[il].ffn_down_exps_s);
        g.cb(cur, "ffn_moe_out", il);
    }
    cur = ggml_add(g.ctx0, cur, ffn_inp);
    g.cb(cur, "ffn_out", il);

    cur = g.build_cvec(cur, il);
    g.cb(cur, "l_out", il);

    b.output("out", cur);
    return b.finish();
}

static std::unique_ptr<llama_prefab_factor_module> prefab_compile_out(
        const llama_model & model, const llm_graph_params & params, uint32_t n_env) {
    prefab_module_builder b(params, { LLAMA_PREFAB_FACTOR_OUT, -1, n_env, 0 });
    auto & g = *b.g;

    ggml_tensor * cur = b.input("inpL", GGML_TYPE_F32, g.n_embd, n_env);

    cur = g.build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    g.cb(cur, "result_norm", -1);
    b.output("embd", cur);

    cur = g.build_lora_mm(model.output, cur, model.output_s);
    g.cb(cur, "result_output", -1);
    b.output("logits", cur);
    return b.finish();
}

#include <set>

//
// Shared-node compose (GGML_TP5_PREFAB_NOCLONE, exact "1"): the factor's DAG is bound into
// the envelope graph IN PLACE — no tensor is re-instantiated per splice (n_factor_clones
// stays 0). This generalizes the MTP shared-node pattern (llama-prefab-mtp.cpp:284-344)
// to a chain of cached factors:
//
//   - rewiring, not substitution: a compiled module's nodes read its own formal placeholders.
//     On the first in-place splice the (node, src-slot) wire points referencing each
//     placeholder are recorded on the module (llama_prefab_factor_module::wiring); every
//     splice then rewrites exactly those points to the current actuals, unconditionally —
//     never by comparing against the previous actual. This is what makes rebinding correct
//     when an actual changes between builds (per-build KV mask/views, the envelope's
//     out_ids gather, an ATTN factor recompiled for a new pad width): after the first
//     splice the srcs hold the previous actuals, which a placeholder comparison could no
//     longer find, and a stale src would silently wire the wrong tensor.
//   - the placeholders never enter the envelope graph: after the rewire nothing references
//     them, so the envelope's scheduler never plans them; the envelope's own input tensors
//     (created by the caller exactly as on the clone path) are what the rewired nodes read.
//   - stale allocations: after a rebuild the module-owned tensors carry data/buffer
//     pointers from the previous plan. The release pass below clears exactly those (test
//     by context range, like the MTP release; weights and upstream factor tensors are not
//     ours and are not at risk — their own module releases them), so the envelope's
//     scheduler plans them afresh as graph tensors; leaving them set would make the
//     planner treat them as external and hand out overlapping allocations.
//   - node order: the module's nodes are expanded in the module's emission order — the
//     same order the clone path expands its clones in — so the composed graph's node
//     sequence and every descriptor byte match the clone path.
//   - safety: at most one live envelope graph references a module (one slot per width;
//     the rebuild sites drain in-flight work and destroy the old graph first — the
//     sync_pending gates at the rebuild branches cover the factor mode as well), and the
//     modules die with the cache (topology change, release_compute) only after the graphs
//     referencing them are gone.
//

static void prefab_factor_release_owned(const llama_prefab_factor_module & mod) {
    std::set<const ggml_tensor *> visited;
    std::function<void(ggml_tensor *)> visit = [&](ggml_tensor * t) {
        // ownership test on the pointer value only; a stale src pointer (previous
        // envelope) is never dereferenced — non-module tensors stop the walk
        if (t == nullptr || visited.count(t) != 0 || !mod.owns(t)) {
            return;
        }
        visited.insert(t);
        t->data = nullptr;
        t->buffer = nullptr;
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            visit(t->src[i]);
        }
        if (t->view_src != nullptr) {
            visit(t->view_src);
        }
    };
    ggml_cgraph * gf = mod.res->get_gf();
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        visit(ggml_graph_node(gf, i));
    }
    for (const auto & kv : mod.inputs) {
        visit(kv.second);
    }
}

static std::map<std::string, ggml_tensor *> prefab_splice_shared(
        llm_graph_context & dst,
        llama_prefab_factor_module & mod,
        const std::map<std::string, ggml_tensor *> & subs,
        llama_prefab_stats & stats) {
    // 1. validate exactly like the clone kernel: fail closed on an unbound input or a
    // layout mismatch — the rewired nodes would otherwise read malformed bindings.
    for (const auto & [name, ph] : mod.inputs) {
        auto it = subs.find(name);
        if (it == subs.end() || it->second == nullptr) {
            throw std::runtime_error("factor-prefab: unbound factor input '" + name + "'");
        }
        ggml_tensor * t = it->second;
        bool same = t->type == ph->type;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            same = same && t->ne[i] == ph->ne[i] && t->nb[i] == ph->nb[i];
        }
        if (!same) {
            throw std::runtime_error("factor-prefab: layout mismatch for factor input '" + name + "'");
        }
    }
    for (const auto & [name, t] : subs) {
        if (t != nullptr && mod.inputs.find(name) == mod.inputs.end()) {
            throw std::runtime_error("factor-prefab: substitution for unknown factor input '" + name + "'");
        }
    }

    // 2. record the wire points once; the module's DAG is fixed after compile, so the
    // record stays valid for the module's lifetime. The walk never dereferences a
    // non-module tensor (weights cannot reference the module's placeholders).
    if (mod.wiring.empty()) {
        std::set<const ggml_tensor *> seen;
        std::function<void(ggml_tensor *)> record = [&](ggml_tensor * t) {
            if (t == nullptr || !mod.owns(t) || seen.count(t) != 0) {
                return;
            }
            seen.insert(t);
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                ggml_tensor * s = t->src[i];
                if (s == nullptr) {
                    continue;
                }
                for (const auto & [name, ph] : mod.inputs) {
                    if (s == ph) {
                        mod.wiring[ph].push_back({ t, i });
                        break;
                    }
                }
                record(s);
            }
            if (t->view_src != nullptr) {
                for (const auto & [name, ph] : mod.inputs) {
                    if (t->view_src == ph) {
                        mod.wiring[ph].push_back({ t, -1 }); // -1 marks the view_src slot
                        break;
                    }
                }
                record(t->view_src);
            }
        };
        ggml_cgraph * gf = mod.res->get_gf();
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            record(ggml_graph_node(gf, i));
        }
    }

    // 3. rewire the recorded points to the current actuals — unconditional, so changed
    // actuals rebind correctly on every rebuild.
    for (const auto & [name, actual] : subs) {
        const auto ph_it = mod.inputs.find(name);
        if (ph_it == mod.inputs.end()) {
            continue;
        }
        const auto wires = mod.wiring.find(ph_it->second);
        if (wires == mod.wiring.end()) {
            continue;
        }
        for (const auto & [node, slot] : wires->second) {
            if (slot < 0) {
                node->view_src = actual;
            } else {
                node->src[slot] = actual;
            }
        }
    }

    // 4. release the module's stale plan pointers so this envelope's scheduler re-plans
    // the shared tensors as graph tensors (fresh allocations cannot overlap them).
    prefab_factor_release_owned(mod);

    // 5. bind in place: the module's nodes enter the envelope graph in the module's
    // emission order — the same order the clone path expands its clones in.
    ggml_cgraph * gf = mod.res->get_gf();
    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_build_forward_expand(dst.gf, ggml_graph_node(gf, i));
    }

    llama_prefab_stat_tick(stats.n_factor_splices);
    return mod.outputs;   // the resident tensors: the next factor reads them in place
}

// splice: bind the factor's DAG into the envelope context (GGML_TP5_PREFAB_NOCLONE=1) or
// clone it per envelope build (default, byte-identical to the pre-noclone splice)

static std::map<std::string, ggml_tensor *> prefab_splice_clone(
        llm_graph_context & dst,
        const llama_prefab_factor_module & mod,
        const std::map<std::string, ggml_tensor *> & subs,
        llama_prefab_stats & stats);

static std::map<std::string, ggml_tensor *> prefab_splice(
        llm_graph_context & dst,
        llama_prefab_factor_module & mod,
        const std::map<std::string, ggml_tensor *> & subs,
        llama_prefab_stats & stats) {
    if (llama_prefab_noclone_enabled()) {
        return prefab_splice_shared(dst, mod, subs, stats);
    }
    return prefab_splice_clone(dst, mod, subs, stats);
}

// clone splice (the r7 path, byte-identical with the gate off): the factor's DAG is
// re-instantiated node-by-node into the envelope context; weights, KV cache storage and
// adapter tensors are shared by pointer, never cloned.
static std::map<std::string, ggml_tensor *> prefab_splice_clone(
        llm_graph_context & dst,
        const llama_prefab_factor_module & mod,
        const std::map<std::string, ggml_tensor *> & subs,
        llama_prefab_stats & stats) {
    std::unordered_map<const ggml_tensor *, ggml_tensor *> map;

    for (const auto & [name, ph] : mod.inputs) {
        auto it = subs.find(name);
        if (it == subs.end() || it->second == nullptr) {
            throw std::runtime_error("factor-prefab: unbound factor input '" + name + "'");
        }
        ggml_tensor * t = it->second;
        bool same = t->type == ph->type;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            same = same && t->ne[i] == ph->ne[i] && t->nb[i] == ph->nb[i];
        }
        if (!same) {
            throw std::runtime_error("factor-prefab: layout mismatch for factor input '" + name + "'");
        }
        map[ph] = t;
    }
    for (const auto & [name, t] : subs) {
        if (t != nullptr && mod.inputs.find(name) == mod.inputs.end()) {
            throw std::runtime_error("factor-prefab: substitution for unknown factor input '" + name + "'");
        }
    }

    std::function<ggml_tensor * (ggml_tensor *)> clone = [&](ggml_tensor * src) -> ggml_tensor * {
        if (src == nullptr) {
            return nullptr;
        }
        auto it = map.find(src);
        if (it != map.end()) {
            return it->second;
        }
        if (!mod.owns(src)) {
            // weights, KV cache storage, adapter tensors: shared, never cloned
            map[src] = src;
            return src;
        }
        if (src->op == GGML_OP_NONE && src->view_src == nullptr) {
            throw std::runtime_error(std::string("factor-prefab: factor leaf '") + src->name + "' is not a formal input");
        }

        ggml_tensor * t = ggml_new_tensor(dst.ctx0, src->type, GGML_MAX_DIMS, src->ne);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            t->nb[i] = src->nb[i];
        }
        t->op    = src->op;
        t->flags = src->flags;
        memcpy(t->op_params, src->op_params, sizeof(t->op_params));
        ggml_set_name(t, src->name);

        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            t->src[i] = clone(src->src[i]);
        }
        if (src->view_src != nullptr) {
            t->view_src  = clone(src->view_src);
            t->view_offs = src->view_offs;
            if (t->view_src->data != nullptr) {
                t->data = (char *) t->view_src->data + t->view_offs;
            }
        }

        map[src] = t;
        llama_prefab_stat_tick(stats.n_factor_clones);
        return t;
    };

    const int n_nodes = ggml_graph_n_nodes(mod.res->get_gf());
    for (int i = 0; i < n_nodes; ++i) {
        ggml_build_forward_expand(dst.gf, clone(ggml_graph_node(mod.res->get_gf(), i)));
    }

    std::map<std::string, ggml_tensor *> outs;
    for (const auto & [name, t] : mod.outputs) {
        outs[name] = clone(t);
    }
    llama_prefab_stat_tick(stats.n_factor_splices);
    return outs;
}

//
// envelope composition
//

static llama_prefab_topology_key prefab_make_key(const llama_model & model, const llm_graph_params & params) {
    const auto * kvc = static_cast<const llama_kv_cache_context *>(params.mctx);
    const llama_kv_cache * kv = kvc->get_kv_cache();

    llama_prefab_topology_key k;
    k.model      = &model;
    k.memory     = kv;
    k.arch       = (int32_t) model.arch;
    k.n_embd     = params.hparams.n_embd;
    k.n_layer    = params.hparams.n_layer();
    k.n_head     = params.hparams.n_head();
    k.n_head_kv  = params.hparams.n_head_kv();
    k.kv_size    = kv->get_size();
    // a filtered cache (an MTP context holds only its nextn layer(s)) has no entry for
    // global layer 0; take the storage type of a layer this cache actually maps
    const int32_t il_kv = kv->first_layer_id();
    k.type_k     = (int32_t) kv->get_k_storage(il_kv)->type;
    k.type_v     = kv->get_v_storage(il_kv) ? (int32_t) kv->get_v_storage(il_kv)->type : -1;
    k.flash_attn = params.cparams.flash_attn;
    k.v_trans    = kv->get_v_transposed();
    k.causal     = params.cparams.causal_attn;
    k.warmup     = params.cparams.warmup;
    k.rope_freq_base  = params.cparams.rope_freq_base;
    k.rope_freq_scale = params.cparams.rope_freq_scale;
    return k;
}

llama_prefab_topology_key llama_prefab_key_for(const llama_model & model, const llm_graph_params & params) {
    return prefab_make_key(model, params);
}

ggml_cgraph * llama_prefab_compose(
        const llama_model & model,
        const llm_graph_params & params,
        uint32_t n_env,
        llama_prefab_state & st,
        llama_prefab_stats & stats) {
    GGML_ASSERT(params.ubatch.n_tokens == n_env);
    GGML_ASSERT(params.n_outputs == n_env);

    const auto * kvc = static_cast<const llama_kv_cache_context *>(params.mctx);
    // n_kv is a RUNTIME parameter, not a recompile key: the ATTN factor and the envelope
    // graph are compiled for a padded KV width (multiple of 1024, clamped to the cache
    // size) that covers the current n_kv. While n_kv stays within the pad window, the
    // cached factor and the envelope graph are reused as-is: the K/V views and kq_mask
    // cover [0, n_kv_pad) and the pad columns are masked out by the prefab mask writer.
    // When n_kv outgrows the window, the ATTN factor is recompiled for the new padded
    // width (a new factor key); QKV/FFN/EMBD/OUT factors are unaffected (cache hits).
    const uint32_t n_kv     = kvc->get_n_kv();
    const uint32_t n_kv_pad = kvc->get_n_kv_pad();
    GGML_ASSERT(n_kv_pad >= n_kv);

    if (st.cache.validate_or_reset(prefab_make_key(model, params))) {
        llama_prefab_stat_tick(stats.n_cache_resets);
    }

    llm_graph_params p = params;
    p.cb = prefab_cb();
    llm_graph_context g(p);

    auto factor = [&](const llama_prefab_factor_key & key, auto && compile) -> llama_prefab_factor_module & {
        auto it = st.cache.modules.find(key);
        if (it != st.cache.modules.end()) {
            llama_prefab_stat_tick(stats.n_factor_cache_hits);
            return *it->second;
        }
        auto mod = compile();
        llama_prefab_stat_tick(stats.n_factor_compiles);
        auto & ref = *mod;
        st.cache.modules[key] = std::move(mod);
        return ref;
    };

    // --- envelope inputs (fixed width n_env, padded at set_input time from the real ubatch)
    auto inp_tok = std::make_unique<llm_graph_input_prefab_tokens>(n_env);
    inp_tok->tokens = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, n_env);
    ggml_set_name(inp_tok->tokens, "inp_tokens");
    ggml_set_input(inp_tok->tokens);
    g.res->t_inp_tokens = inp_tok->tokens;
    ggml_tensor * tokens = inp_tok->tokens;
    g.res->add_input(std::move(inp_tok));

    auto inp_pos_u = std::make_unique<llm_graph_input_prefab_pos>(n_env);
    inp_pos_u->pos = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, n_env);
    ggml_set_name(inp_pos_u->pos, "inp_pos");
    ggml_set_input(inp_pos_u->pos);
    ggml_tensor * inp_pos = inp_pos_u->pos;
    g.res->add_input(std::move(inp_pos_u));

    auto inp_attn_u = std::make_unique<llm_graph_input_prefab_attn_kv>(g.hparams, g.cparams, kvc, n_env, n_kv_pad, &st);
    inp_attn_u->self_k_idxs = kvc->build_input_k_idxs(g.ctx0, g.ubatch);
    inp_attn_u->self_v_idxs = kvc->build_input_v_idxs(g.ctx0, g.ubatch);
    inp_attn_u->self_kq_mask = ggml_new_tensor_4d(g.ctx0, g.cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, n_kv_pad, n_env, 1, 1);
    ggml_set_input(inp_attn_u->self_kq_mask);
    ggml_set_name(inp_attn_u->self_kq_mask, "attn_inp_kq_mask");
    inp_attn_u->self_kq_mask_cnv = inp_attn_u->self_kq_mask;
    inp_attn_u->self_k_rot = kvc->build_input_k_rot(g.ctx0);
    inp_attn_u->self_v_rot = kvc->build_input_v_rot(g.ctx0);
    auto * inp_attn = (llm_graph_input_prefab_attn_kv *) g.res->add_input(std::move(inp_attn_u));

    auto inp_out_u = std::make_unique<llm_graph_input_prefab_out_ids>(n_env);
    inp_out_u->out_ids = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, n_env);
    ggml_set_name(inp_out_u->out_ids, "inp_out_ids");
    ggml_set_input(inp_out_u->out_ids);
    ggml_tensor * out_ids = inp_out_u->out_ids;
    g.res->add_input(std::move(inp_out_u));

    // --- A: embedding factor
    auto & f_embd = factor({ LLAMA_PREFAB_FACTOR_EMBD, -1, n_env, 0 }, [&]() { return prefab_compile_embd(model, p, n_env); });
    ggml_tensor * inpL = prefab_splice(g, f_embd, { { "tokens", tokens } }, stats).at("inpL");
    g.res->t_inp_embd = inpL;

    // --- B: per-layer factors QKV(il) + ATTN(il, n_kv) + FFN(il)
    for (int il = 0; il < g.n_layer; ++il) {
        g.res->t_layer_inp[il] = inpL;

        auto & f_qkv = factor({ LLAMA_PREFAB_FACTOR_QKV, il, n_env, 0 }, [&]() { return prefab_compile_qkv(model, p, il, n_env); });
        auto qkv = prefab_splice(g, f_qkv, { { "inpL", inpL }, { "inp_pos", inp_pos } }, stats);

        auto & f_attn = factor({ LLAMA_PREFAB_FACTOR_ATTN, il, n_env, n_kv_pad },
                [&]() { return prefab_compile_attn(model, p, il, n_env, n_kv_pad, f_qkv); });
        std::map<std::string, ggml_tensor *> attn_subs = {
            { "q", qkv.at("q") }, { "k", qkv.at("k") }, { "v", qkv.at("v") },
            { "k_idxs",  inp_attn->self_k_idxs },
            { "v_idxs",  inp_attn->self_v_idxs },
            { "kq_mask", inp_attn->self_kq_mask },
        };
        if (inp_attn->self_k_rot) attn_subs["k_rot"] = inp_attn->self_k_rot;
        if (inp_attn->self_v_rot) attn_subs["v_rot"] = inp_attn->self_v_rot;
        ggml_tensor * cur = prefab_splice(g, f_attn, attn_subs, stats).at("out");

        ggml_tensor * inpSA = inpL;
        if (il == g.n_layer - 1) {
            // envelope out_ids: real outputs first, pad rows duplicate the last one
            cur   = ggml_get_rows(g.ctx0,   cur, out_ids);
            inpSA = ggml_get_rows(g.ctx0, inpSA, out_ids);
        }

        auto & f_ffn = factor({ LLAMA_PREFAB_FACTOR_FFN, il, n_env, 0 }, [&]() { return prefab_compile_ffn(model, p, il, n_env); });
        inpL = prefab_splice(g, f_ffn, { { "cur", cur }, { "inpSA", inpSA } }, stats).at("out");
    }

    // --- output factor
    auto & f_out = factor({ LLAMA_PREFAB_FACTOR_OUT, -1, n_env, 0 }, [&]() { return prefab_compile_out(model, p, n_env); });
    auto outs = prefab_splice(g, f_out, { { "inpL", inpL } }, stats);

    g.res->t_embd   = outs.at("embd");
    g.res->t_logits = outs.at("logits");

    ggml_build_forward_expand(g.gf, g.res->t_logits);
    g.res->set_outputs(p);

    llama_prefab_stat_tick(stats.n_envelope_builds);

    return g.res->get_gf();
}

//
// C API
//
// (implemented in llama-context.cpp, which owns the state)

//
// decouple interface (GGML_TP5_PREFAB_DECOUPLE=1)
//
// The eligibility layer (llama-context.cpp) calls these entry points with
// explicit request values; this layer never reaches back into llama_context.
// Gate off: these functions are never called (llama-context.cpp keeps the
// legacy direct-coupling path), so behavior is byte-identical to baseline.
//

#include "../ggml/src/ggml-backend-moe-cache.h"

bool llama_prefab_decouple_enabled() {
    // enabled only by the exact string "1": unset, "0", "true", "01", "1 ", "2" all stay off
    static const bool on = []() {
        const char * s = getenv("GGML_TP5_PREFAB_DECOUPLE");
        return s != nullptr && s[0] == '1' && s[1] == '\0';
    }();
    return on;
}

bool llama_prefab_state_init(llama_prefab_state & st, const llama_prefab_eligibility & elig) {
    if (elig.envelopes.empty()) {
        return false;
    }
    st = llama_prefab_state{};
    st.envelopes = elig.envelopes;
    st.slots.resize(elig.envelopes.size());
    for (size_t i = 0; i < st.slots.size(); ++i) {
        st.slots[i].n_env = elig.envelopes[i];
    }
    st.mtp_mode  = elig.mtp_mode;
    return true;
}

// slot scheduler + graph-result attachment, decouple path: the eligibility
// layer owns the backend list (llama_context::backend_ptrs/buft), the node
// budget policy (llama_context::graph_max_nodes) and the slot result objects;
// it creates them and hands ready handles to the compose layer. The moe-cache
// probe is a pure query over (model, cparams, backends) with no llama_context
// state, so the eligibility layer applies the same moe-cache settings it
// applies to its own scheds. Backend ownership and device probing stay
// entirely in the eligibility layer; the compose layer only sees ready
// sched/result handles.

void llama_prefab_attach_slot(llama_prefab_state & st, size_t slot_index,
                              ggml_backend_sched_t sched, llm_graph_result * res) {
    GGML_ASSERT(slot_index < st.slots.size());
    GGML_ASSERT(sched != nullptr && res != nullptr);
    auto & slot = st.slots[slot_index];
    slot.sched.reset(sched);
    slot.res.reset(res);
    slot.built = false;
}

llama_prefab_reply llama_prefab_process(
        llama_prefab_state & st,
        llama_prefab_stats & stats,
        const llama_model & model,
        const llm_graph_params & params,
        const llama_ubatch & ubatch,
        llama_memory_context_i * mctx,
        const llama_prefab_request & req,
        const std::function<ggml_status(ggml_cgraph * gf, bool batched, ggml_backend_sched_t sched)> & compute) {
    llama_prefab_reply rep;

    // slot selection by explicit request values (no st.slot_for reach-in from A)
    int is = -1;
    for (size_t i = 0; i < st.slots.size(); ++i) {
        if (req.n_tokens <= st.slots[i].n_env) {
            is = (int) i;
            break;
        }
    }
    if (is < 0) {
        llama_prefab_stat_tick(stats.n_decouple_declines);
        return rep;
    }
    auto & slot = st.slots[is];
    const uint32_t n_env = slot.n_env;

    if (req.mtp_mode) {
        // ---- whole-graph MTP mode: exact-fit slot only ----
        if (n_env != req.n_tokens) {
            llama_prefab_stat_tick(stats.n_decouple_declines);
            return rep;
        }
    } else {
        // ---- factor mode: scratch cell for sub-envelope padding ----
        st.scratch_cell = -1;
        if (req.n_tokens < n_env) {
            const auto * kvc = dynamic_cast<const llama_kv_cache_context *>(mctx);
            if (kvc == nullptr) {
                llama_prefab_stat_tick(stats.n_decouple_declines);
                return rep;
            }
            st.scratch_cell = kvc->prefab_scratch_cell();
            if (st.scratch_cell < 0) {
                llama_prefab_stat_tick(stats.n_fallback_no_scratch);
                llama_prefab_stat_tick(stats.n_decouple_declines);
                return rep;
            }
        }
    }

    // canonical ubatch: envelope shape, real data (envelope inputs read the real ubatch and pad)
    llama_ubatch canon = ubatch;
    if (!req.mtp_mode) {
        canon.n_tokens     = n_env;
        canon.n_seq_tokens = n_env;
        canon.n_seqs       = 1;
        canon.n_seqs_unq   = 1;
        canon.b_equal_seqs = 0;
    }

    llm_graph_result * slot_res = slot.res.get();
    if (slot_res == nullptr || slot.sched == nullptr) {
        // slot not fully attached by the eligibility layer: decline, never crash
        llama_prefab_stat_tick(stats.n_decouple_declines);
        return rep;
    }

    auto gparams = llm_graph_params{
        /*.arch        =*/model.arch,
        /*.hparams     =*/model.hparams,
        /*.cparams     =*/params.cparams,
        /*.ubatch      =*/canon,
        /*.gtype       =*/req.gtype,
        /*.sched       =*/params.sched,
        /*.backend_cpu =*/params.backend_cpu,
        /*.cvec        =*/params.cvec,
        /*.loras       =*/params.loras,
        /*.mctx        =*/mctx,
        /*.cross       =*/params.cross,
        /*.samplers    =*/params.samplers,
        /*.n_outputs   =*/req.mtp_mode ? params.n_outputs : n_env,
        /*.cb          =*/params.cb,
        /*.res         =*/slot_res,
    };

    const llama_prefab_topology_key key = llama_prefab_key_for(model, gparams);

    ggml_cgraph * gf = nullptr;
    bool gf_reused = false;
    bool gf_composed = false;
    bool composed_attempted = false;
    bool n_kv_fits = true;
    if (!req.mtp_mode) {
        // n_kv is a runtime parameter: reuse the envelope graph while the current
        // n_kv fits the padded KV width the graph was composed for
        const auto * kvc = dynamic_cast<const llama_kv_cache_context *>(mctx);
        n_kv_fits = kvc != nullptr && slot.n_kv_pad != 0 && kvc->n_kv_pad_fits(slot.n_kv_pad);
    }

    if (slot.built && !req.graph_reuse_disable && slot.key == key && n_kv_fits && slot_res->can_reuse(gparams)) {
        gf = slot_res->get_gf();
        gf_reused = true;
        if (req.pipeline_parallel) {
            ggml_backend_sched_synchronize(slot.sched.get());
        }
        llama_prefab_stat_tick(stats.n_envelope_reuses);
        rep.reused = true;
    } else {
        if (llama_prefab_noclone_enabled()) {
            // Shared-node compose (GGML_TP5_PREFAB_NOCLONE=1): the graph being replaced
            // references the cached templates' tensors in place -- the MTP A template and,
            // since the factor-lane extension, the cached factor modules as well. The reuse
            // path above never reaches this branch, so the only live graph that references
            // them is the one being replaced (one template set per envelope width, one slot
            // per width). Drain every dispatched-but-unsynchronized sched BEFORE the old
            // graph is destroyed and the shared allocation state is released, so the
            // constraint is enforced here rather than assumed from synchronous dispatch.
            st.sync_pending();
        }
        slot.built = false;
        slot_res->reset();
        ggml_backend_sched_reset(slot.sched.get());
        ggml_backend_sched_set_eval_callback(slot.sched.get(), req.cb_eval, req.cb_eval_user_data);

        {
        llama_prefab_build_timer timer{stats, gf};
        if (req.mtp_mode) {
            // MTP factor compose (GGML_TP5_PREFAB_IN_GRAPH=1): the A x B envelope instead of
            // the stock whole graph. Exact fit only; a declined ubatch falls back to the stock
            // builder below, so gate off (or a declined ubatch) is the stock path.
            if ((llama_prefab_in_graph_enabled() || llama_prefab_ab_enabled()) && req.n_tokens == n_env) {
                composed_attempted = true;
                try {
                    gf = llama_prefab_mtp_compose(model, gparams, n_env, st, stats);
                    gf_composed = gf != nullptr;
                } catch (const std::exception & e) {
                    // a declined ubatch, not a swallowed failure: logged and counted
                    LLAMA_LOG_WARN("mtp factor compose declined: %s\n", e.what());
                    gf = nullptr;
                }
                if (!gf) {
                    llama_prefab_stat_tick(stats.n_mtp_declines);
                }
            }
            if (!gf && composed_attempted) {
                // the compose may have filled slot_res with a partial graph and inputs before it
                // declined: rebuild the slot from scratch so the stock graph gets a clean result.
                // Only after an attempt -- with the gate off the path is exactly the stock one.
                slot_res->reset();
                ggml_backend_sched_reset(slot.sched.get());
                ggml_backend_sched_set_eval_callback(slot.sched.get(), req.cb_eval, req.cb_eval_user_data);
            }
            if (!gf) {
                gf = model.build_graph(gparams);
                llama_prefab_stat_tick(stats.n_stock_builds);
            }
            if (!gf) {
                rep.handled = true;
                rep.status  = GGML_STATUS_FAILED;
                rep.result  = nullptr;
                return rep;
            }
        } else {
            try {
                gf = llama_prefab_compose(model, gparams, n_env, st, stats);
            } catch (const std::exception & e) {
                LLAMA_LOG_ERROR("factor-prefab compose failed: %s\n", e.what());
                gf = nullptr;
            }
            if (!gf) {
                rep.handled = true;
                rep.status  = GGML_STATUS_FAILED;
                return rep;
            }
        }
        }
        if (!ggml_backend_sched_alloc_graph(slot.sched.get(), gf)) {
            rep.handled = true;
            rep.status  = GGML_STATUS_ALLOC_FAILED;
            return rep;
        }
        llama_prefab_stat_tick(stats.n_envelope_allocs);
        slot.built    = true;
        slot.key      = key;
        if (!req.mtp_mode) {
            const auto * kvc = dynamic_cast<const llama_kv_cache_context *>(mctx);
            slot.n_kv_pad = kvc ? kvc->get_n_kv_pad() : 0;
        } else {
            slot.n_kv_pad = 0;
        }
    }

    if (req.nkv_obs && !req.mtp_mode) {
        const auto * kvc = dynamic_cast<const llama_kv_cache_context *>(mctx);
        if (kvc) {
            fprintf(stderr, "prefab nkv: n_kv=%u n_kv_pad=%u decision=%s\n",
                    kvc->get_n_kv(), slot.n_kv_pad, gf_reused ? "reuse" : "build");
        }
    }

    llama_prefab_stat_tick(stats.n_ubatch_prefab);
    llama_prefab_stat_tick(stats.n_decouple_handled);
    if (req.n_tokens < n_env) {
        llama_prefab_stat_tick(stats.n_sub_envelope);
        llama_prefab_stat_add(stats.n_pad_rows, n_env - req.n_tokens);
    }
    if (llama_prefab_stats_enabled()) {
        stats.last_n_env       = n_env;
        stats.last_n_tokens    = req.n_tokens;
        stats.last_graph_nodes = ggml_graph_n_nodes(gf);
        stats.last_inp_tokens_ne0 = slot_res->get_inp_tokens() ? slot_res->get_inp_tokens()->ne[0] : 0;
    }

    // inputs take the REAL ubatch
    slot_res->set_inputs(&ubatch);

    st.active = slot.sched.get();
    st.mark_pending(slot.sched.get());

    const auto status = compute(gf, req.n_tokens > 1, slot.sched.get());
    if (status != GGML_STATUS_SUCCESS) {
        rep.handled = true;
        rep.status  = status;
        return rep;
    }
    if (gf_composed) {
        llama_prefab_stat_tick(stats.n_mtp_compose_hits);
    }

    rep.handled = true;
    rep.status  = GGML_STATUS_SUCCESS;
    rep.result  = slot_res;
    return rep;
}
