#include "llama-prefab.h"
#include "llama-prefab-splice.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-kv-cache.h"

#include "models/models.h"

#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>

//
// MTP factor compose (GGML_TP5_PREFAB_IN_GRAPH=1)
//
// The DECODER_MTP head of qwen4exp is cut in two where the block input has been produced
// (graph_mtp::build_mtp_prologue records it in llm_graph_result::t_prefab_cut):
//
//   A (head prologue)   token embedding, hnorm/enorm, eh_proj, hc_attn mix
//   B (block+epilogue)  QKV, IMRoPE, dense attention, gate, wo, hc combine, MoE FFN,
//                       hc combine, h_nextn export, out_ids gather, head mixer, LM head
//
// A is emitted once per (model, n_env) into its own ggml context and cached in
// llama_prefab_state::mtp_a. Default (clone splice): every envelope graph re-instantiates A
// node-by-node into its own context (llama-prefab-splice.h); weights stay shared -- they live
// outside A's context -- and the envelope's own input tensors are substituted for A's formal
// inputs. Shared-node compose (GGML_TP5_PREFAB_NOCLONE=1): the envelope graph references A's
// tensors in place -- the envelope's input writer feeds A's formal input tensors and B is
// emitted directly on A's resident cut tensors -- so nothing is re-instantiated per compose.
// In both cases B is emitted into the envelope context, because it binds the KV cache context
// of the running llama_context and the envelope's own attention/mask inputs. The prologue is
// therefore built once per envelope width and never recomputed per graph.
//
// Both halves come from graph_mtp, i.e. exactly the code of the stock graph, so the composed
// graph has the same nodes, op_params and tensor shapes as the stock one. The stock ctor runs
// the same two halves in order, so gate off is byte-identical to the pre-split builder.
//
// Scope of the claim: the compose removes the prologue recompute only. It does not change the
// KV layout -- the inherited nkv padding of the decouple lane (LLAMA_PREFAB_NKV_PAD, slot
// n_kv_pad) is not used here, and B binds the stock KV input exactly as the stock head does.
//

bool llama_prefab_in_graph_enabled() {
    // enabled only by the exact string "1": unset, "0", "true", "01", "1 ", "2" all stay off
    static const bool on = []() {
        const char * s = getenv("GGML_TP5_PREFAB_IN_GRAPH");
        return s != nullptr && s[0] == '1' && s[1] == '\0';
    }();
    return on;
}

// r8 shared-node A x B compose (see llama-prefab.h): enabled only by the exact string "1";
// unset, "0", "true", "01", "1 ", "2" all keep the clone splice (byte-identical to baseline).
bool llama_prefab_noclone_enabled() {
    static const bool on = []() {
        const char * s = getenv("GGML_TP5_PREFAB_NOCLONE");
        return s != nullptr && s[0] == '1' && s[1] == '\0';
    }();
    return on;
}

bool llama_prefab_mtp_lifetime_enabled() {
    static const bool on = []() {
        const char * s = getenv("GGML_TP5_PREFAB_MTP_LIFETIME");
        return s != nullptr && s[0] == '1' && s[1] == '\0';
    }();
    return on;
}

bool llama_prefab_stats_enabled() {
    static const bool on = []() {
        const char * s = getenv("GGML_TP5_PREFAB_STATS");
        return s != nullptr && s[0] == '1' && s[1] == '\0';
    }();
    return on;
}

llama_prefab_mtp_a_module * llama_prefab_mtp_a_cache::get(const llama_model & model, uint32_t n_env) {
    auto it = modules.find({ &model, n_env });
    return it == modules.end() ? nullptr : it->second.get();
}

void llama_prefab_mtp_a_cache::clear() {
    modules.clear();
}

//
// A: the cached head prologue
//
// A is captured by running graph_mtp::build_mtp_prologue into its own graph: it returns the block
// input and records it (with the HC residual stream and the scatter weights) in t_prefab_cut /
// t_prefab_cut_res / t_prefab_cut_inject, and A's graph is built by expanding exactly those three.
// Its formal inputs are the head's own input tensors (tokens / h); the KV/pos/out_ids inputs
// belong to B.
//

static std::unique_ptr<llama_prefab_mtp_a_module> prefab_mtp_compile_a(
        const llama_model & model, const llm_graph_params & params, uint32_t n_env) {
    auto mod = std::make_unique<llama_prefab_mtp_a_module>();
    mod->res = std::make_unique<llm_graph_result>(PREFAB_MTP_A_MAX_NODES);

    llm_graph_params p = params;
    p.ubatch.n_tokens     = n_env;
    p.ubatch.n_seq_tokens = n_env;
    p.ubatch.n_seqs       = 1;
    p.ubatch.n_seqs_unq   = 1;
    p.ubatch.b_equal_seqs = 0;
    p.n_outputs = n_env;
    p.res       = mod->res.get();

    // only the prologue runs here: the no-build ctor binds the members without running either
    // head half, and the prologue binds no KV/pos/out_ids input
    llama_model_qwen4exp::graph_mtp g(model, p, llama_model_qwen4exp::graph_mtp::no_build_t{});
    g.build_mtp_prologue();

    ggml_tensor * block_in = mod->res->t_prefab_cut;
    ggml_tensor * res_hc   = mod->res->t_prefab_cut_res;
    ggml_tensor * inject   = mod->res->t_prefab_cut_inject;
    if (block_in == nullptr || res_hc == nullptr || inject == nullptr) {
        throw std::runtime_error("mtp factor compose: head prologue did not reach the A/B cut");
    }

    // A's formal inputs: the head's own input tensors (op == NONE, never allocated on this
    // graph). They are only reachable through the graph inputs the prologue registered. The
    // prologue reads tokens (token embedding) and h (hidden stream); embd is registered by the
    // stock head input but the prologue does not consume it, so it is not an A input.
    mod->inputs["tokens"] = nullptr;
    mod->inputs["h"]      = nullptr;

    for (const auto & inp : mod->res->inputs) {
        if (auto * e = dynamic_cast<llm_graph_input_embd_h *>(inp.get())) {
            mod->inputs["tokens"] = e->tokens;
            mod->inputs["h"]      = e->h;
        }
    }
    for (const auto & [name, t] : mod->inputs) {
        if (t == nullptr) {
            throw std::runtime_error("mtp factor compose: missing head input '" + name + "'");
        }
    }
    mod->outputs["block_in"] = block_in;
    mod->outputs["res_hc"]   = res_hc;
    mod->outputs["inject"]   = inject;

    // A's graph is exactly the prologue: expand the three cut tensors in emission order. Their
    // ancestors are the whole prologue (the prologue emits nothing else) and the weights are
    // shared, not expanded.
    ggml_cgraph * gf = mod->res->get_gf();
    ggml_build_forward_expand(gf, block_in);
    ggml_build_forward_expand(gf, res_hc);
    ggml_build_forward_expand(gf, inject);

    const int n_keep = ggml_graph_n_nodes(gf);
    if (n_keep == 0) {
        throw std::runtime_error("mtp factor compose: empty head prologue");
    }
    mod->n_a = n_keep;

    return mod;
}

//
// splice: clone A's DAG into the envelope context, substituting the formal inputs
//

static std::map<std::string, ggml_tensor *> prefab_mtp_splice_a(
        llm_graph_context & dst,
        const llama_prefab_mtp_a_module & mod,
        const std::map<std::string, ggml_tensor *> & subs,
        llama_prefab_stats & stats) {
    // the shared ggml-only splice kernel (llama-prefab-splice.h): A is a DAG in its own
    // context, its formal inputs are substituted by this graph's inputs, weights are shared.
    // Every failure is a throw, i.e. a decline that falls back to the stock builder.
    llama_prefab_dag dag;
    dag.ctx     = mod.res->get_ctx();
    dag.gf      = mod.res->get_gf();
    dag.n_nodes = mod.n_a;
    dag.inputs  = mod.inputs;
    dag.outputs = mod.outputs;

    return llama_prefab_dag_splice(dst.ctx0, dst.gf, dag, subs, &stats.n_mtp_a_clones);
}

//
// envelope input of the composed MTP graph
//

// tokens / h: the head's dual input (token ids + the trunk's wide hidden state).
// Exact fit only: the compose path never runs with n_tokens < n_env, so this writes the whole
// ubatch and pads nothing.
class llm_graph_input_prefab_mtp_inp : public llm_graph_input_i {
public:
    llm_graph_input_prefab_mtp_inp(int64_t n_embd, uint32_t n_env) : n_embd(n_embd), n_env(n_env) {}
    ~llm_graph_input_prefab_mtp_inp() override = default;

    void set_input(const llama_ubatch * ubatch) override {
        // the same contract the stock llm_graph_input_embd_h relies on; a graph that does not
        // match it is never built (can_reuse) nor run (compose eligibility check)
        GGML_ASSERT(ubatch->token != nullptr);
        GGML_ASSERT(ubatch->embd  != nullptr);
        GGML_ASSERT(ubatch->n_tokens == n_env);

        // stock llm_graph_input_embd_h semantics: h takes the provided embeddings
        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_env*sizeof(int32_t));
        ggml_backend_tensor_set(h,      ubatch->embd,  0, (size_t) n_env*n_embd*sizeof(float));
    }

    bool can_reuse(const llm_graph_params & params) override {
        return tokens->ne[0] == (int64_t) params.ubatch.n_tokens && params.ubatch.n_tokens == n_env;
    }

    ggml_tensor * tokens = nullptr; // I32 [n_env]
    ggml_tensor * h      = nullptr; // F32 [n_embd, n_env]

    const int64_t  n_embd;
    const uint32_t n_env;
};

//
// compose
//
ggml_cgraph * llama_prefab_mtp_compose(
        const llama_model & model,
        const llm_graph_params & params,
        uint32_t n_env,
        llama_prefab_state & st,
        llama_prefab_stats & stats) {
    // eligibility is checked, never asserted: a declined ubatch must not abort the process.
    // Exact fit only (no envelope padding, no KV headroom); n_outputs may be smaller than the
    // envelope (the stock head sizes out_ids by it), never larger.
    if (model.arch != LLM_ARCH_QWEN4EXP) {
        throw std::runtime_error("mtp factor compose: arch is not QWEN4EXP");
    }
    if (params.gtype != LLM_GRAPH_TYPE_DECODER_MTP) {
        throw std::runtime_error("mtp factor compose: graph type is not DECODER_MTP");
    }
    if (params.ubatch.n_tokens != n_env) {
        throw std::runtime_error("mtp factor compose: ubatch does not fill the envelope");
    }
    if (params.res == nullptr) {
        throw std::runtime_error("mtp factor compose: no result object");
    }
    if (params.ubatch.token == nullptr || params.ubatch.embd == nullptr) {
        throw std::runtime_error("mtp factor compose: ubatch lacks token or embedding input");
    }
    if (params.n_outputs <= 0 || params.n_outputs > (int64_t) n_env) {
        // the stock head sizes out_ids by n_outputs and the driver reads exactly that many rows
        // back; more outputs than the envelope has rows cannot be produced
        throw std::runtime_error("mtp factor compose: n_outputs does not fit the envelope");
    }

    // n_kv is the live count: B binds the stock KV input, whose can_reuse compares the mask width
    // against the exact n_kv, so the graph is rebuilt whenever n_kv changes. The MTP compose never
    // widens the KV window.

    // params is used as-is: the stock head sizes out_ids by params.n_outputs, exactly as on the
    // stock path, so the composed graph returns the same rows
    llm_graph_params p = params;
    llm_graph_context g(p);

    const int64_t hc_dim = (int64_t) g.hparams.n_embd_out();

    // --- A: cached head prologue, compiled once per (model, n_env). Fetched before the input
    // tensors because the shared-node path binds this graph's inputs to A's own tensors.
    llama_prefab_mtp_a_module * a = st.mtp_a.get(model, n_env);
    if (a != nullptr) {
        llama_prefab_stat_tick(stats.n_mtp_a_reuses);
        llama_prefab_stat_tick(stats.template_hits);
    } else {
        auto mod = prefab_mtp_compile_a(model, params, n_env);
        llama_prefab_stat_tick(stats.n_mtp_a_compiles);
        llama_prefab_stat_add(stats.n_nodes_built, mod->n_a);
        a = mod.get();
        st.mtp_a.modules[{ &model, n_env }] = std::move(mod);
    }

    // --- inputs, same shapes/types as the stock head's own inputs
    auto inp_u = std::make_unique<llm_graph_input_prefab_mtp_inp>(hc_dim, n_env);
    ggml_tensor * tokens = nullptr;
    ggml_tensor * h      = nullptr;
    std::map<std::string, ggml_tensor *> a_outs;
    if (llama_prefab_noclone_enabled()) {
        // shared-node A x B compose: A's tensors are referenced in place, never
        // re-instantiated. The envelope's input writer feeds A's formal input tensors and
        // B is emitted directly on A's resident cut tensors; when B's outputs are expanded,
        // A's nodes enter this graph through the normal src walk (the weights were already
        // shared that way). Before the graph is built, the data/buffer pointers of every
        // tensor A owns are cleared so THIS graph's scheduler allocation plans them afresh:
        // a pointer left over from a previous allocation is indistinguishable from an
        // external tensor and would be left out of a re-planned layout, letting fresh
        // allocations overlap A's memory. A fresh template has null pointers, so this is a
        // no-op on the first compose.
        ggml_cgraph * ga = a->res->get_gf();
        // ownership range as integers: relational comparison of pointers into unrelated
        // objects is unspecified in C++, so compare uintptr_t values instead.
        const uintptr_t buf_beg = reinterpret_cast<uintptr_t>(ggml_get_mem_buffer(a->res->get_ctx()));
        const uintptr_t buf_end = buf_beg + ggml_get_mem_size(a->res->get_ctx());
        auto release_owned = [&](ggml_tensor * t) {
            if (t == nullptr) {
                return;
            }
            if (reinterpret_cast<uintptr_t>(t) < buf_beg || reinterpret_cast<uintptr_t>(t) >= buf_end) {
                return; // weights and other external tensors: not owned by A, never touched
            }
            t->data = nullptr;
            t->buffer = nullptr;
        };
        // walk the template graph's nodes and transitively their sources and view sources:
        // this reaches every tensor of the template graph, including leaves (the formal
        // input placeholders and any leaf a future prologue might add), so no A-owned
        // tensor can retain a stale allocation pointer into a re-planned pool. Tensors
        // outside A's context (weights) fail the ownership test and are never touched.
        std::set<const ggml_tensor *> visited;
        std::function<void(ggml_tensor *)> visit = [&](ggml_tensor * t) {
            if (t == nullptr || visited.count(t) != 0) {
                return;
            }
            visited.insert(t);
            release_owned(t);
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                visit(t->src[i]);
            }
            if (t->view_src != nullptr) {
                visit(t->view_src);
            }
        };
        for (int i = 0; i < ggml_graph_n_nodes(ga); ++i) {
            visit(ggml_graph_node(ga, i));
        }
        for (const auto & kv : a->inputs) {
            visit(kv.second); // formal inputs, incl. any the prologue registered but the
                              // graph does not reach (defensive; data stays null there)
        }
        tokens = a->inputs.at("tokens");
        h      = a->inputs.at("h");
        GGML_ASSERT(tokens->type == GGML_TYPE_I32 && tokens->ne[0] == (int64_t) n_env);
        GGML_ASSERT(h->type == GGML_TYPE_F32 && h->ne[0] == hc_dim && h->ne[1] == (int64_t) n_env);
        // A's formal inputs already carry the input flag and their build-time names (the
        // prologue's own embedding input registered them), so they are bound as-is.
        inp_u->tokens = tokens;
        inp_u->h      = h;
        a_outs        = a->outputs; // the shared cut tensors: B reads A in place
    } else {
        // clone path (unchanged): fresh envelope input tensors, A re-instantiated per compose
        inp_u->tokens = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, n_env);
        ggml_set_name(inp_u->tokens, "inp_tokens");
        ggml_set_input(inp_u->tokens);
        inp_u->h = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, hc_dim, n_env);
        ggml_set_input(inp_u->h);
        ggml_set_name(inp_u->h, "mtp_h_input");
        tokens = inp_u->tokens;
        h      = inp_u->h;
        a_outs = prefab_mtp_splice_a(g, *a, { { "tokens", tokens }, { "h", h } }, stats);
    }
    g.res->t_inp_tokens = tokens;
    g.res->add_input(std::move(inp_u));

    // --- B: the trunk-style block + epilogue, emitted by the stock head code on top of the
    // A cut tensors (shared in place with GGML_TP5_PREFAB_NOCLONE=1, re-instantiated by the
    // splice otherwise). It binds this graph's pos / out_ids / KV inputs; out_ids is sized by
    // params.n_outputs, exactly as on the stock path.
    llama_model_qwen4exp::graph_mtp gm(model, p, llama_model_qwen4exp::graph_mtp::no_build_t{});
    // --- B's KV input: the padded KV window of the whole-graph slot (n_kv is a runtime
    // parameter; the prefab mask writer -INF-masks the pad columns, so attention never
    // reads beyond the live width and the composed graph is reused while n_kv fits the
    // window the slot was composed for -- the reuse judgement in llama-prefab-hotpath.h)
    const auto * kvc = static_cast<const llama_kv_cache_context *>(params.mctx);
    const uint32_t n_kv_pad = kvc->get_n_kv_pad();
    GGML_ASSERT(llama_prefab_nkv_fits(n_kv_pad, kvc->get_n_kv()));
    auto inp_attn_u = std::make_unique<llm_graph_input_prefab_attn_kv>(
            g.hparams, g.cparams, kvc, n_env, n_kv_pad, &st);
    // the input object owns the per-ubatch tensors the stock attention build consumes:
    // cache cell indices (fed to cpy_k/cpy_v), the padded KV mask, and the rotation tables.
    // Same constructors as the stock head (build_attn_inp_kv_impl) and the factor envelope
    // (llama-prefab.cpp), so shapes/types match and set_input can write them each ubatch.
    inp_attn_u->self_k_idxs  = kvc->build_input_k_idxs(g.ctx0, g.ubatch);
    inp_attn_u->self_v_idxs  = kvc->build_input_v_idxs(g.ctx0, g.ubatch);
    inp_attn_u->self_kq_mask = ggml_new_tensor_4d(g.ctx0,
            g.cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, n_kv_pad, n_env, 1, 1);
    ggml_set_input(inp_attn_u->self_kq_mask);
    ggml_set_name(inp_attn_u->self_kq_mask, "attn_inp_kq_mask");
    inp_attn_u->self_kq_mask_cnv = inp_attn_u->self_kq_mask;
    inp_attn_u->self_k_rot = kvc->build_input_k_rot(g.ctx0);
    inp_attn_u->self_v_rot = kvc->build_input_v_rot(g.ctx0);
    // register BEFORE the block build: the slot result owns the input object
    // (res->inputs), so set_input/can_reuse run per ubatch and the raw pointer
    // stays valid past this scope; an unregistered input would die with this
    // function and its mask/idxs tensors would never be written per request.
    auto * inp_attn = (llm_graph_input_attn_kv *) g.res->add_input(std::move(inp_attn_u));
    gm.build_mtp_block(a_outs.at("block_in"), a_outs.at("res_hc"), a_outs.at("inject"), inp_attn);
    if (gm.res->t_h_nextn == nullptr || gm.res->t_logits == nullptr) {
        throw std::runtime_error("mtp factor compose: head block did not emit its outputs");
    }
    llama_prefab_stat_tick(stats.n_mtp_b_builds);

    g.res->t_h_nextn = gm.res->t_h_nextn;
    g.res->t_logits  = gm.res->t_logits;

    // the stock builder does this in llama_model::build_graph; without it the scheduler may
    // reuse t_h_nextn / t_logits as scratch and clobber them before they are read back
    g.res->set_outputs(p);

    return g.res->get_gf();
}
