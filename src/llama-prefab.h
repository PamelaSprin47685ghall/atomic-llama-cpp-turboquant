#pragma once

// factor-prefab: compose decoder graphs from independently compiled, cached factor modules
// into fixed-width microbatch envelopes (default N = 1 / 4 / n_ubatch).
//
// Default OFF. Enabled only with LLAMA_PREFAB_ENABLE=1 *and* LLAMA_PREFAB_EXPERIMENTAL=1.
//
//   envelope   : a graph compiled for n_env rows; a ubatch with n_tokens <= n_env runs on it with
//                runtime padding (pad rows write a scratch KV cell, mask rows replicate the last
//                real row, out_ids select the real outputs first).
//   factor     : an isolated sub-graph (EMBD / QKV(il) / ATTN(il) / FFN(il) / OUT) compiled once into
//                its own ggml context with formal placeholder inputs, cached per (kind, il, n_env[, n_kv])
//                and spliced into the envelope graph: default by clone + input substitution; with
//                GGML_TP5_PREFAB_NOCLONE=1 (exact "1") the module's DAG is referenced in place --
//                shared nodes/views, no per-splice clone (see llama-prefab.cpp / llama-prefab-mtp.cpp).
//   ownership  : the factor cache lives in llama_context (no global state); it is validated against a
//                topology key (model, memory, shapes, cache types, FA, warmup, ...) and reset on mismatch.

#include "llama.h"
#include "llama-graph.h"
#include "llama-prefab-iface.h"
#include "llama-prefab-hotpath.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

struct llama_model;
struct llama_kv_cache;
class  llama_kv_cache_context;

#ifdef __cplusplus
extern "C" {
#endif

// per-context counters: see llama-prefab-hotpath.h (ggml-free seam: the POD
// and the padded-KV reuse judgement of the whole-graph slot lane live there).

LLAMA_API bool llama_prefab_is_enabled(const struct llama_context * ctx);
LLAMA_API void llama_prefab_get_stats  (const struct llama_context * ctx, struct llama_prefab_stats * out);
LLAMA_API void llama_prefab_reset_stats(struct llama_context * ctx);
// number of cached factor modules currently owned by the context
LLAMA_API size_t llama_prefab_n_cached_factors(const struct llama_context * ctx);

#ifdef __cplusplus
}
#endif
bool llama_prefab_mtp_lifetime_enabled();
bool llama_prefab_stats_enabled();

struct llama_prefab_build_timer {
    llama_prefab_stats & stats;
    ggml_cgraph *& graph;
    bool enabled = llama_prefab_stats_enabled();
    std::chrono::steady_clock::time_point start = enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ~llama_prefab_build_timer() {
        if (!enabled) return;
        stats.build_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count();
        if (graph) stats.n_nodes_built += ggml_graph_n_nodes(graph);
    }
};

enum llama_prefab_factor_kind : int32_t {
    LLAMA_PREFAB_FACTOR_EMBD = 0,
    LLAMA_PREFAB_FACTOR_QKV  = 1,
    LLAMA_PREFAB_FACTOR_ATTN = 2,
    LLAMA_PREFAB_FACTOR_FFN  = 3,
    LLAMA_PREFAB_FACTOR_OUT  = 4,
};

struct llama_prefab_factor_key {
    int32_t  kind;
    int32_t  il;
    uint32_t n_env;
    uint32_t n_kv; // only for ATTN: the padded KV width the factor is compiled for (multiple of 1024, >= n_kv)

    bool operator<(const llama_prefab_factor_key & o) const {
        if (kind  != o.kind)  return kind  < o.kind;
        if (il    != o.il)    return il    < o.il;
        if (n_env != o.n_env) return n_env < o.n_env;
        return n_kv < o.n_kv;
    }
};

// everything a cached factor module bakes in, besides (kind, il, n_env, n_kv).
// note: n_kv is deliberately NOT part of this key - the ATTN factor is compiled for a padded
// KV width and stays valid across n_kv changes within the pad window (see llama_prefab_compose)
struct llama_prefab_topology_key {
    const void * model  = nullptr;
    const void * memory = nullptr;
    int32_t  arch       = -1;
    uint32_t n_embd     = 0;
    uint32_t n_layer    = 0;
    uint32_t n_head     = 0;
    uint32_t n_head_kv  = 0;
    uint32_t kv_size    = 0;
    int32_t  type_k     = -1;
    int32_t  type_v     = -1;
    bool     flash_attn = false;
    bool     v_trans    = false;
    bool     causal     = true;
    bool     warmup     = false;
    float    rope_freq_base  = 0.0f;
    float    rope_freq_scale = 0.0f;

    bool operator==(const llama_prefab_topology_key & o) const;
    bool operator!=(const llama_prefab_topology_key & o) const { return !(*this == o); }
};

// an isolated, compiled sub-graph with formal (placeholder) inputs and named outputs
struct llama_prefab_factor_module {
    llama_prefab_factor_key key = {};

    std::unique_ptr<llm_graph_result> res; // owns the ggml context + cgraph of the factor

    std::map<std::string, ggml_tensor *> inputs;  // formal placeholders (op == NONE, never allocated)
    std::map<std::string, ggml_tensor *> outputs; // block_in / res_hc / inject

    bool owns(const ggml_tensor * t) const;

    // Shared-node compose (GGML_TP5_PREFAB_NOCLONE, exact "1"): for every formal input, the
    // (node, src-slot) wire points of this module's DAG that reference its placeholder,
    // recorded on the first in-place splice. Every later in-place splice rewrites exactly
    // these points to the current actuals — unconditionally, never by comparing against the
    // previous actual — so variable bindings (the per-build KV mask/views, the envelope's
    // out_ids gather, an ATTN factor recompiled for a new pad width) rebind correctly and
    // no stale src survives a rebuild. Placeholders are module-owned and never scheduled:
    // after the rewire nothing in the envelope graph references them.
    std::map<const ggml_tensor *, std::vector<std::pair<ggml_tensor *, int>>> wiring;

};

// owned by llama_context; dies with it (and is cleared on release_compute / topology change)
struct llama_prefab_factor_cache {
    bool has_key = false;
    llama_prefab_topology_key key;

    std::map<llama_prefab_factor_key, std::unique_ptr<llama_prefab_factor_module>> modules;

    // returns true if the cache was reset
    bool validate_or_reset(const llama_prefab_topology_key & k);
    void clear();
};

struct llama_prefab_slot {
    uint32_t n_env = 0;

    std::unique_ptr<llm_graph_result> res;
    ggml_backend_sched_ptr            sched;

    bool     built = false;
    // padded KV width the envelope graph was composed for (n_kv is a runtime parameter:
    // the graph is reused while n_kv fits this width, recomposed when it outgrows it)
    uint32_t n_kv_pad = 0;
    llama_prefab_topology_key key;
};

// one cached A part (head prologue) of the MTP factor compose. A is captured by running
// graph_mtp::build_mtp_prologue() into its own graph and keeping the node prefix that produces
// llm_graph_result::t_prefab_cut / t_prefab_cut_res / t_prefab_cut_inject; its formal inputs are
// the head input tensors.
struct llama_prefab_mtp_a_module {
    std::unique_ptr<llm_graph_result> res; // owns the ggml context + cgraph of A

    std::map<std::string, ggml_tensor *> inputs;  // formal placeholders (op == NONE)
    std::map<std::string, ggml_tensor *> outputs; // block_in / res_hc / inject

    int n_a = 0; // number of prologue nodes in res's graph
};

// per-context cache of the A parts; validated against the model pointer and n_env
struct llama_prefab_mtp_a_cache {
    std::map<std::pair<const void *, uint32_t>, std::unique_ptr<llama_prefab_mtp_a_module>> modules;

    llama_prefab_mtp_a_module * get(const llama_model & model, uint32_t n_env);
    void clear();
};

struct llama_prefab_state {
    std::vector<uint32_t>          envelopes; // ascending, last == max
    std::vector<llama_prefab_slot> slots;     // one per envelope
    llama_prefab_factor_cache      cache;

    ggml_backend_sched_t active = nullptr; // sched of the last executed envelope graph
    std::vector<ggml_backend_sched_t> pending; // envelope scheds with work not yet synchronized

    // pad bookkeeping for the current ubatch (consumed by the envelope inputs at set_input time)
    int64_t scratch_cell = -1;

    // whole-graph MTP mode (LLAMA_PREFAB_MTP): slots hold stock-built DECODER_MTP graphs
    // (e.g. qwen4exp graph_mtp) instead of composed factor envelopes; see llama-context.cpp
    bool mtp_mode = false;

    // MTP factor compose (GGML_TP5_PREFAB_IN_GRAPH=1): cached A parts (head prologue) of the
    // qwen4exp DECODER_MTP head, keyed by (model, n_env). Empty unless the gate is on.
    llama_prefab_mtp_a_cache mtp_a;

    int slot_for(uint32_t n_tokens) const; // -1 if none
    void sync_pending(); // wait for every envelope sched that has unsynchronized work
    void mark_pending(ggml_backend_sched_t s); // record a sched whose graph was just dispatched
    void reset_slots(); // drop envelope graphs (keeps factor cache), after sync_pending()
};

llama_prefab_topology_key llama_prefab_key_for(const llama_model & model, const llm_graph_params & params);

// Compose the envelope graph for params (params.ubatch.n_tokens == n_env, params.res == slot result).
// Returns nullptr on failure.
ggml_cgraph * llama_prefab_compose(
        const llama_model & model,
        const llm_graph_params & params,
        uint32_t n_env,
        llama_prefab_state & st,
        llama_prefab_stats & stats);

// ---- decouple interface (GGML_TP5_PREFAB_DECOUPLE=1) ----
// See llama-prefab-iface.h (gate fn declared there). These entry points are
// the gate-on path: the eligibility layer (llama-context.cpp) calls them with
// explicit request values instead of reading/writing llama_prefab_state
// internals.

// initialize B's state from explicit eligibility facts (envelope widths,
// mtp_mode). Returns false if the envelope set is empty (caller must not
// enable prefab).
bool llama_prefab_state_init(llama_prefab_state & st, const llama_prefab_eligibility & elig);

// process one ubatch through the compose layer. All facts arrive as values in
// `req`; the reply says whether B handled it. `compute` is B's only way to run
// the graph (B never calls llama_context methods directly).
llama_prefab_reply llama_prefab_process(
        llama_prefab_state & st,
        llama_prefab_stats & stats,
        const llama_model & model,
        const llm_graph_params & params,
        const llama_ubatch & ubatch,
        llama_memory_context_i * mctx,
        const llama_prefab_request & req,
        const std::function<ggml_status(ggml_cgraph * gf, bool batched, ggml_backend_sched_t sched)> & compute);

// attach a ready-made scheduler and graph-result to a slot (decouple path):
// the eligibility layer owns the backend list, the node budget policy
// (graph_max_nodes) and the slot result objects; the compose layer only
// receives ready handles. Takes ownership of both.
void llama_prefab_attach_slot(llama_prefab_state & st, size_t slot_index,
                              ggml_backend_sched_t sched, llm_graph_result * res);

// Input classes of an envelope graph. They read the *real* ubatch (n_tokens <= n_env) and pad.
// (llm_graph_input_i is not modified: only the existing virtuals are overridden.)
class llm_graph_input_prefab_tokens : public llm_graph_input_i {
public:
    explicit llm_graph_input_prefab_tokens(uint32_t n_env) : n_env(n_env) {}
    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * tokens = nullptr; // I32 [n_env]
    const uint32_t n_env;
};

class llm_graph_input_prefab_pos : public llm_graph_input_i {
public:
    explicit llm_graph_input_prefab_pos(uint32_t n_env) : n_env(n_env) {}
    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * pos = nullptr; // I32 [n_env]
    const uint32_t n_env;
};

class llm_graph_input_prefab_out_ids : public llm_graph_input_i {
public:
    explicit llm_graph_input_prefab_out_ids(uint32_t n_env) : n_env(n_env) {}
    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * out_ids = nullptr; // I32 [n_env]: real outputs first, then padding
    const uint32_t n_env;
};

class llm_graph_input_prefab_attn_kv : public llm_graph_input_attn_kv {
public:
    llm_graph_input_prefab_attn_kv(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_context * mctx,
            uint32_t n_env,
            uint32_t n_kv_pad,
            const llama_prefab_state * st) :
        llm_graph_input_attn_kv(hparams, cparams, mctx), n_env(n_env), st(st) {
        this->n_kv_pad = n_kv_pad;
    }

    void set_input(const llama_ubatch * ubatch) override;
    // n_kv is a runtime parameter: reuse the graph while the live n_kv fits the padded
    // KV width (the mask/K/V-view width). The inherited can_reuse compares the mask
    // width against the exact n_kv and would force a rebuild on every 256-boundary.
    bool can_reuse(const llm_graph_params & params) override;

    const uint32_t n_env;
    const llama_prefab_state * st;
};

// free-function probe (no new virtuals on llm_graph_input_i): is this input an envelope input?
bool llm_graph_input_is_prefab(const llm_graph_input_i * inp);

// ---- MTP factor compose (GGML_TP5_PREFAB_IN_GRAPH=1) ----
//
//   A: the head prologue -- token embedding, hnorm/enorm normalisation, eh_proj, the HC mix
//      that produces the block input. Depends only on the head inputs and the head weights,
//      so it is built once per (model, n_env). Default: cloned into every envelope. With
//      GGML_TP5_PREFAB_NOCLONE=1 the composed graph references A's tensors in place instead
//      (shared nodes/views, no per-compose clone).
//   B: the trunk-style block -- QKV, IMRoPE, dense attention, output gate, wo, HC combine,
//      MoE FFN, HC combine -- plus the head epilogue (h_nextn export, head mixer, LM head).
//      Built per envelope because it binds the KV cache context of the running context.
// Both parts are emitted by the same code that builds the stock graph (qwen4exp
// graph_mtp::build_mtp_prologue / build_mtp_block), so the composed graph is node-for-node the
// stock graph: A is built once per envelope width, B runs in every envelope on top of it.
//
// The gate is exact "1" (llama_prefab_in_graph_enabled). Off: these functions are never
// called and the MTP slot keeps the whole-graph cache path (byte-identical to baseline).

// true only for the exact string "1" of GGML_TP5_PREFAB_IN_GRAPH
bool llama_prefab_in_graph_enabled();

// node budget of a cached A module (the head prologue is a few dozen nodes)
static constexpr int64_t PREFAB_MTP_A_MAX_NODES = 512;

// r8 shared-node A x B compose: true only for the exact string "1" of GGML_TP5_PREFAB_NOCLONE.
// Set together with the compose lane (GGML_TP5_PREFAB_IN_GRAPH=1 or GGML_TP5_PREFAB_AB=1) it
// makes every A x B compose reference the cached A template's tensors in place: the envelope's
// input writer feeds A's formal input tensors and B is emitted directly on A's resident cut
// tensors, so no A-owned tensor is ever re-instantiated (n_mtp_a_clones stays 0). Off: the
// clone splice runs exactly as before.
bool llama_prefab_noclone_enabled();
// Compose the DECODER_MTP envelope graph for params (params.ubatch.n_tokens == n_env,
// params.res == slot result). Uses/refills st.mtp_a. Returns nullptr on failure.

ggml_cgraph * llama_prefab_mtp_compose(
        const llama_model & model,
        const llm_graph_params & params,
        uint32_t n_env,
        llama_prefab_state & st,
        llama_prefab_stats & stats);
