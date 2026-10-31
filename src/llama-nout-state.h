#pragma once

//
// llama-nout-state.h
//
// Per-context state for the combined runtime-engine patches
// (env gates: GGML_TP5_UNION_BINDER gates the legacy n_outputs lane,
// GGML_TP5_NKV_TRUEWIDTH gates the r7 true-width nkv lane; both exact "1";
// the nkv binding registers only under its own gate - there is no envelope
// fallback path for either lane). r8 adds the coverage lane: GGML_TP5_NKV_COV
// (exact "1") gates the coverage audit + binder_bind/graph_rebuild counters;
// unset or any other value is stock (r7) for everything r8.
//
// State placement, exactly as measured against the stock baseline (no
// blanket "layout unchanged" claim: two classes DO gain members below):
//
//   - llama_cparams, llm_graph_params, llm_graph_input_out_ids and the
//     llama_kv_cache_* contexts carry NO added data members; their
//     per-object state lives in TU-local side tables keyed by object
//     identity (llama-graph.cpp / llama-kv-cache.cpp).
//   - llama_context (llama-context.h) DOES gain one opaque member: a
//     std::unique_ptr<llama_nkv_ctx_state> binder-state slot (see below),
//     holding the nkv binder record + counters + nout flags. Internal
//     layout changes by design; the public llama.h ABI is untouched.
//   - llm_graph_result (llama-graph.h) DOES gain an owner back-pointer
//     member (see llama-graph.h).
//
// This internal header defines one state struct plus free-function
// accessors; it is included by llama-context.h (for the forward declaration
// and the opaque member) and by the owning TUs.
//

#include "ggml.h"
#include "llama-union-binder.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

struct ggml_tensor;
struct llama_context;
struct llm_graph_params;

// ---- r8 coverage lane (GGML_TP5_NKV_COV, exact "1") ----
//
// nkv-coupled coverage categories of a recorded graph. The nkv binder binds
// the whole graph generically (llama-context.cpp, union_rebind_n_kv); these
// categories make the nkv-coupled tensor families explicit so the acceptance
// run can verify each family was actually covered (present in the record /
// patched by binds):
//   mask    - graph inputs whose first dim is the current cache width (KQ
//             masks and the relative-position buckets over the cache)
//   view    - cache k/v read views (views over pre-existing cache storage)
//   rope    - RoPE ops (GGML_OP_ROPE / GGML_OP_ROPE_BACK) and the Hadamard
//             rotation input tensors (attn_inp_k_rot / attn_inp_v_rot)
//   kvwrite - cache write nodes (SET_ROWS / CPY whose src0 is cache storage)
//   fa      - flash-attention nodes
// Classification is structural (op / flags / name / first dim vs the built
// width); it never reads runtime state. ACCOUNTING ONLY: the counts are
// acceptance bookkeeping for which families are present/patched - not per-op
// correctness or coverage evidence. A shape coincidence (an unrelated input
// whose ne[0] equals the built width) may over-count MASK by one.
enum llama_nkv_cov_cat : int {
    LLAMA_NKV_COV_MASK    = 0,
    LLAMA_NKV_COV_VIEW    = 1,
    LLAMA_NKV_COV_ROPE    = 2,
    LLAMA_NKV_COV_KVWRITE = 3,
    LLAMA_NKV_COV_FA      = 4,
    LLAMA_NKV_COV_OTHER   = 5,
    LLAMA_NKV_COV_N       = 6,
};

// classify one recorded graph tensor into its nkv-coupled coverage category;
// pure function of the recorded fields (no ggml context, no runtime state):
//   op              - the tensor's ggml op
//   name            - the tensor's name (may be null; unnamed tensors exist)
//   is_input        - GGML_TENSOR_FLAG_INPUT was set on the record
//   is_fixed        - pre-existing storage (weights, KV cache): never bound
//   src0_fixed      - src[0] resolves to fixed storage (cache write nodes)
//   view_src_fixed  - view_src resolves to fixed storage (cache read views)
//   ne0             - the tensor's first dim
//   nkv_built       - the exact n_kv the graph was built at (0 when the nkv
//                     lane did not record this build)
static inline int llama_nkv_cov_classify(int32_t op, const char * name, bool is_input,
                                         bool is_fixed, bool src0_fixed, bool view_src_fixed,
                                         int64_t ne0, uint32_t nkv_built) {
    if (is_fixed) {
        return LLAMA_NKV_COV_OTHER;    // pre-existing storage: never nkv-coupled
    }
    if (op == GGML_OP_FLASH_ATTN_EXT) {
        return LLAMA_NKV_COV_FA;
    }
    if (op == GGML_OP_VIEW) {
        return view_src_fixed ? LLAMA_NKV_COV_VIEW : LLAMA_NKV_COV_OTHER;
    }
    if (op == GGML_OP_SET_ROWS || op == GGML_OP_CPY) {
        return src0_fixed ? LLAMA_NKV_COV_KVWRITE : LLAMA_NKV_COV_OTHER;
    }
    if (op == GGML_OP_ROPE || op == GGML_OP_ROPE_BACK) {
        return LLAMA_NKV_COV_ROPE;     // the rope family (rope ops)
    }
    if (is_input && name != nullptr &&
            (strncmp(name, "attn_inp_k_rot", 14) == 0 || strncmp(name, "attn_inp_v_rot", 14) == 0)) {
        return LLAMA_NKV_COV_ROPE;     // the rope family (Hadamard rotation inputs)
    }
    if (is_input && nkv_built > 0 && ne0 == (int64_t) nkv_built) {
        return LLAMA_NKV_COV_MASK;     // graph inputs over the current cache width
    }
    return LLAMA_NKV_COV_OTHER;
}

// per-llama_context state for the union-graph runtime parameter binder and
// its n_outputs lane (the n_outputs lane registers only under
// GGML_TP5_UNION_BINDER == "1", the nkv lane only under
// GGML_TP5_NKV_TRUEWIDTH == "1")
struct llama_nkv_ctx_state {
    // ---- nkv binder (GGML_TP5_NKV_TRUEWIDTH, exact "1") ----
    bool                     nkv_enabled    = false;  // was llama_context::graph_union_nkv_runtime
    llama_union_binder       union_binder;            // kind -> (tensor set provider, rebind)
    void *                   nkv_scratch    = nullptr; // llm_graph_result_ptr held as void* (see .cpp)
    bool                     record_valid   = false;   // was nkv_record_valid
    uint32_t                 record_cap     = 0;       // exact n_kv the live graph was BUILT at (no envelope)
    uint32_t                 record_bound   = 0;       // width the live graph's nkv shapes are currently bound at
    std::vector<ggml_tensor *> record_t;               // was nkv_record_t
    std::vector<int32_t>      record_src;               // was nkv_record_src
    std::vector<uint8_t>     record_fixed;             // was nkv_record_fixed
    std::vector<size_t>      record_bytes;             // was nkv_record_bytes
    size_t c_attempts = 0, c_applied = 0, c_refused_width = 0, c_refused_envelope = 0,
           c_refused_skeleton = 0, c_refused_grow = 0, c_refused_relay = 0,
           c_refused_pipeline = 0, c_fallbacks = 0, c_record_builds = 0;  // 10 counters
    // ---- nout binding lane (registered with the binder under GGML_TP5_UNION_BINDER) ----
    bool   nout_enabled = false;      // the n_outputs binding is registered and the lane is active
    size_t nout_binds  = 0;           // successful n_outputs reconciles
    // validity/cap/bound of the n_outputs record. The skeleton/byte vectors
    // are SHARED with the nkv lane (one live graph, one pre-allocation record
    // of the latest build); these fields are the n_outputs lane's own view.
    // No pad-to-max: the record is taken on every fresh decoder build and
    // nout_cap is the BUILT width - the rebind only moves the executed shapes
    // within [0, nout_cap] (shrink or restore); growth beyond the built width
    // is refused and falls back to a full rebuild.
    bool     nout_record_valid = false;
    uint32_t nout_cap          = 0;    // width the live graph was BUILT at
    uint32_t nout_bound        = 0;    // active count the live graph's output shapes currently hold

    // ---- r8 coverage lane (GGML_TP5_NKV_COV, exact "1"; see the classifier
    // above). Every field here is r8-only: with the gate off (unset or any
    // value other than exactly "1") none of them is ever written and the
    // behavior is exactly r7. The counts are acceptance accounting (which
    // families are present/patched), not per-op correctness or coverage
    // evidence. cov_present describes the CURRENT live record (reset per
    // record); c_cov_patched accumulates.----
    bool     cov_enabled    = false;  // the gate is exactly "1" (read once in the ctor)
    size_t   c_binder_bind  = 0;      // committed nkv binds; c_applied also includes the n_outputs lane
    size_t   c_graph_rebuild = 0;     // full rebuilds under the enabled nkv lane (mock mirror: "rebuilds")
    size_t   c_refused_view = 0;      // refused: the graph carries attention-view readers
    uint32_t cov_present[LLAMA_NKV_COV_N]  = {};  // per-category count in the CURRENT live record (reset per record)
    uint32_t c_cov_patched[LLAMA_NKV_COV_N] = {}; // cumulative per-category tensors patched by binds
    uint32_t cov_pending[LLAMA_NKV_COV_N] = {};   // staged nkv patch until scheduler allocation succeeds
    uint32_t cov_nkv_built = 0;       // the exact n_kv the recorded graph was built at (classifier input)

    // out-of-line destructor (llama-nout-state.cpp): releases the held scratch
    // llm_graph_result. Out-of-line so llama-context.h can hold this struct
    // behind a unique_ptr to the (there incomplete) type and still compile.
    ~llama_nkv_ctx_state();
};

// per-context state slot owned by llama_context (llama-context.h declares the
// member as a unique_ptr to this forward-declared type so the state struct
// definition stays out of that header; the member itself is a new
// llama_context data member - its internal layout changes by design).
//
// Lifetime: the slot is created lazily by the llama_context constructor and
// only when the GGML_TP5_UNION_BINDER gate is exactly "1" - with the gate off
// (stock OFF path) no binder state is allocated at all. It is destroyed with
// the context (~llama_context destroys the unique_ptr member; the state's
// out-of-line destructor releases the held scratch graph result). Nothing is
// shared between contexts or threads.
llama_nkv_ctx_state & llama_nkv_ctx_state_get(llama_context * ctx);
