#pragma once

//
// llama-nout-state.h
//
// ABI-stable side tables for the combined runtime-engine patches
// (env: LLAMA_GRAPH_UNION_NKV_RUNTIME, LLAMA_GRAPH_UNION_OUTPUTS_RUNTIME).
//
// The original patches added data members to llama_context,
// llama_cparams, llm_graph_params, llm_graph_input_out_ids and
// llama_kv_cache_context. To keep every public/internal header layout
// byte-identical to baseline (no offset/sizeof change, no aggregate-init
// reorder), all of that per-object state instead lives in .cpp-local side
// tables keyed by object identity:
//
//   - llama_context *            -> nkv binder record + counters + nout flags
//   - llama_kv_cache_context *   -> build-scoped n_kv envelope override
//                                   (side table local to llama-kv-cache.cpp)
//   - llm_graph_input_out_ids *  -> bind_outputs envelope map
//                                   (side table local to llama-graph.cpp)
//
// This internal header exposes only free-function accessors; it adds no
// struct/class definitions and is included by exactly one .cpp besides the
// owning TU, so it cannot change any layout.
//

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

struct ggml_tensor;
struct llama_context;
struct llm_graph_params;

// per-llama_context state for the union-graph runtime n_kv binder
// (env: LLAMA_GRAPH_UNION_NKV_RUNTIME) and the runtime n_outputs binding
// lane (env: LLAMA_GRAPH_UNION_OUTPUTS_RUNTIME)
struct llama_nkv_ctx_state {
    // ---- nkv binder (LLAMA_GRAPH_UNION_NKV_RUNTIME) ----
    bool                     nkv_enabled    = false;  // was llama_context::graph_union_nkv_runtime
    void *                   nkv_scratch    = nullptr; // llm_graph_result_ptr held as void* (see .cpp)
    bool                     record_valid   = false;   // was nkv_record_valid
    uint32_t                 record_cap     = 0;       // was nkv_record_cap
    uint32_t                 record_bound    = 0;       // was nkv_record_bound
    std::vector<ggml_tensor *> record_t;               // was nkv_record_t
    std::vector<int32_t>      record_src;               // was nkv_record_src
    std::vector<uint8_t>     record_fixed;             // was nkv_record_fixed
    std::vector<size_t>      record_bytes;             // was nkv_record_bytes
    size_t c_attempts = 0, c_applied = 0, c_refused_align = 0, c_refused_skeleton = 0,
           c_refused_envelope = 0, c_refused_relay = 0, c_refused_pipeline = 0,
           c_fallbacks = 0, c_env_builds = 0;          // 9 counters (was nkv_binder_*)

    // ---- nout binding lane (GGML_TP5_NOUT_REUSE, strict value "1") ----
    bool   nout_enabled = false;      // was llama_cparams::graph_union_outputs_runtime
    bool   nout_gate    = false;      // raw env parse ("1"), before the sampler veto
    size_t nout_binds  = 0;          // successful runtime binds (incl. 0-row)
    size_t nout_fallbacks = 0;       // bind failures that fell back to a rebuild
    // one-shot: set by the bind-fail fallback rebuild so the next
    // graph_params() call derives the ACTIVE n_outputs (not the envelope),
    // standing in for the original patch's per-params
    // union_outputs_runtime = false; consumed (cleared) by graph_params()
    bool   nout_suppress_envelope = false;

    // ---- nout floor1 lane (LLAMA_NOUT_FLOOR1_REUSE, strict value "1") ----
    // floor1: graphs build at max(active, 1) rows - a floor of one output row
    // for 0-output steps, the exact active count otherwise. This is NOT an
    // envelope/capacity build: the built width never exceeds the largest
    // active count seen since the last grow-rebuild. The bind lane narrows
    // the executed output closure to the active count in place (only the
    // 0-output step ever narrows: built == max(active, 1) >= active); growth
    // past the built width is refused and falls back to an exact-count
    // rebuild (stock semantics). Mutually exclusive with the envelope lane
    // above: the envelope lane owns the reuse machinery when its gate is on.
    bool   nout1_gate      = false;  // raw env parse
    bool   nout1_enabled   = false;  // effective: gate && envelope lane off && no samplers
    size_t nout1_binds     = 0;      // successful narrowing binds (incl. 0-row)
    size_t nout1_fallbacks = 0;      // bind failures that fell back to a rebuild
    // (suppress: floor1 shares the envelope lane's nout_suppress_envelope
    //  one-shot flag; one flag, one consumption point in graph_params)

    // ---- nout floor1 r8 trace (LLAMA_NOUT_FLOOR1_TRACE, strict value "1") ----
    // decision/build/skip instrumentation for the floor1 allow_reuse path.
    // Parsed in the llama_context constructor; every r8 block below is gated
    // on this exact match, so unset or any other value leaves the context
    // bit-identical to stock (counters never touched, no logs).
    bool     t_enabled       = false;  // exact env parse ("1")
    uint64_t t_dec           = 0;      // reuse decisions observed (decoder graphs)
    uint64_t t_dec_reuse     = 0;      // can_reuse passed: graph reused, build skipped
    uint64_t t_miss_first    = 0;      // miss on a never-built stored result (first step)
    uint64_t t_miss_disabled = 0;      // reuse forced off (LLAMA_GRAPH_REUSE_DISABLE)
    uint64_t t_miss_ubatch   = 0;      // allow_reuse miss: ubatch shape/seqs/token/embd
    uint64_t t_miss_outputs  = 0;      // allow_reuse miss: output-count reuse key
    uint64_t t_miss_samplers = 0;      // allow_reuse miss: sampler state
    uint64_t t_miss_cparams  = 0;      // allow_reuse miss: cparams/arch/gtype/cvec/loras/cross
    uint64_t t_miss_inputs   = 0;      // allow_reuse passed, a graph input refused reuse
    uint64_t t_builds        = 0;      // fresh floor1 decoder builds (non-fallback)
    uint64_t t_builds_fb     = 0;      // bind-failure fallback rebuilds
    uint64_t t_bind_zero     = 0;      // binds that narrowed the gather to 0 rows
    uint64_t t_compute_skip  = 0;      // steps that actually computed on a reused graph
    uint64_t t_graph_gen     = 0;      // graph generations: ++ per successful build
};

// side-table entry for the given context (created on first use)
llama_nkv_ctx_state & llama_nkv_ctx_state_get(llama_context * ctx);

// held scratch llm_graph_result
void llama_nkv_ctx_state_erase(llama_context * ctx);

// restore every non-fixed recorded tensor's src[]/view_src pointers from the
// pre-allocation record (undoes the scheduler's split rewrites) and clear the
// stale allocation state (buffer/data/extra) on every recorded non-fixed
// tensor that is not a view of a fixed one, so the next scheduler plan
// re-derives everything fresh. Shared by the nkv in-place patch and the nout
// bind; no-op without a valid record.
void llama_nkv_record_restore(llama_context * ctx);
void llama_nkv_record_clear_alloc(llama_context * ctx);

// env: LLAMA_GRAPH_UNION_OUTPUTS_RUNTIME - effective flag for the context
// owning the given graph params (derived on the fly; the original patch
// stored this as llm_graph_params::union_outputs_runtime)
bool llama_nout_runtime_enabled(const llm_graph_params & params);
// env: LLAMA_NOUT_FLOOR1_REUSE - effective flag for the context owning the
// given graph params (floor1 lane: graphs build at max(active, 1) rows, the
// exact active count otherwise; fail-closed on an unregistered result)
bool llama_nout1_runtime_enabled(const llm_graph_params & params);
