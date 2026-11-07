#pragma once

// llama-prefab-iface.h — explicit composition interface between the prefab
// eligibility layer (A, lives in llama-context.cpp) and the prefab compose
// layer (B, lives in llama-prefab.cpp).
//
// Gate: GGML_TP5_PREFAB_DECOUPLE (exact "1"). When the gate is unset or not
// exactly "1", both layers keep their legacy direct coupling and behavior is
// byte-identical to the baseline (A+B compose + MTP elig cumulative patch).
// When the gate is exactly "1":
//   - A no longer reads B's internals (llama_prefab_state::mtp_mode, slot
//     tables, scratch_cell, active sched pointer) to decide eligibility or to
//     drive the ubatch; it fills an explicit request struct and calls B
//     through the functions below.
//   - B no longer reaches back into llama_context for eligibility facts; all
//     facts it needs arrive as values in the request struct.
//
// This header is the ONLY shared surface between the two layers under the
// gate. It deliberately contains no llama_context / llama_prefab_state
// definitions so neither layer can silently grow a new dependency through it.

#include "ggml-backend.h"

#include "llama-graph.h"

#include <cstdint>
#include <vector>

// exact-"1" env gate for the decouple lane (unset / "0" / "true" / "01" /
// "1 " / "2" all stay off)
bool llama_prefab_decouple_enabled();

// ---- A -> B: explicit eligibility facts (values, not object reach-ins) ----
//
// Filled by the eligibility layer (A) after its fail-closed checks; consumed
// by the compose layer (B). Every field is a plain value or plain pointer
// that B is allowed to use; nothing here grants B access to A's internals.
struct llama_prefab_eligibility {
    // whole-graph MTP mode: slots hold stock-built DECODER_MTP graphs
    // (e.g. qwen4exp graph_mtp) instead of composed factor envelopes
    bool     mtp_mode = false;

    // envelope widths (ascending, last == max); one slot per width
    std::vector<uint32_t> envelopes;
};

// ---- A -> B: per-ubatch request (values, not object reach-ins) ----
//
// Filled by A for one ubatch; B answers through llama_prefab_reply. A never
// inspects B's slot tables, factor cache, scratch bookkeeping or active-sched
// pointer to build this — the fields below are the complete contract.
struct llama_prefab_request {
    // graph type the caller wants (DEFAULT / DECODER / DECODER_MTP)
    llm_graph_type gtype = LLM_GRAPH_TYPE_DEFAULT;

    // real ubatch token count (<= the chosen envelope width)
    uint32_t n_tokens = 0;

    // envelope width selected for this ubatch (must be one of envelopes)
    uint32_t n_env = 0;

    // whole-graph MTP mode request (exact-fit slot semantics)
    bool mtp_mode = false;

    // graph_reuse_disable (env LLAMA_GRAPH_REUSE_DISABLE) — disables slot
    // reuse for both factor and MTP slots equally
    bool graph_reuse_disable = false;

    // pipeline parallel: slot sched must be synchronized before set_inputs
    // on the reuse path
    bool pipeline_parallel = false;

    // eval callback for the slot sched (may be null)
    ggml_backend_sched_eval_callback cb_eval = nullptr;
    void * cb_eval_user_data = nullptr;

    // per-ubatch observation gate (LLAMA_PREFAB_NKV_OBS, exact "1")
    bool nkv_obs = false;
};

// ---- B -> A: explicit reply (A does not read B's state to learn these) ----
struct llama_prefab_reply {
    // false: B declined the ubatch; A must fall back to the stock builder
    // (A increments its own fallback counters — B does not touch A's stats)
    bool handled = false;

    // hard failure inside B after accepting the ubatch (alloc/compute);
    // when set, handled == true and status carries the ggml_status
    ggml_status status = GGML_STATUS_SUCCESS;

    // the slot graph was reused without rebuild (A maps this to its own
    // graphs-reused perf counter; B cannot reach it)
    bool reused = false;

    // the graph result B executed (slot-owned); null iff handled == false
    llm_graph_result * result = nullptr;
};
