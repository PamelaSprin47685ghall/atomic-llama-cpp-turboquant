// GGML_TP5_NOUT_REUSE: runtime output-capacity planner.
//
// Plans the row narrowing of a graph's output-gather closure from the
// build-time output envelope down to the active output count. Pure function
// over ggml structs: it reads tensors and derives a plan; the caller applies
// it (llama_nout_plan_apply) or rebuilds the graph when the plan fails.
//
#include "ggml.h"
// Accepted closure ops (anything else rejects the bind, fail-closed):
//   GET_ROWS (root), ADD, MUL, RMS_NORM, NORM, SCALE, CONT, GLU (split form
//   only, both inputs), MUL_MAT (narrowed tensor as src1 / activation).
// A rejected bind leaves every tensor untouched.
// Layout contract: the narrowed dim is ne[1] (rows) everywhere except the
// out_ids index tensor (ne[0]); nb is recomputed as contiguous, which every
// accepted op's dst is (fresh result tensor, no views inside the closure).

#pragma once

#include <stdint.h>

struct ggml_cgraph;
struct ggml_tensor;

// one planned tensor: target row count plus the build-time envelope shape
struct llama_nout_plan_entry {
    ggml_tensor * t;     // planned tensor
    int64_t       rows;  // target ne[1] after the bind
    int64_t       env_ne[GGML_MAX_DIMS];  // build-time envelope shape
    size_t        env_nb[GGML_MAX_DIMS];  // build-time envelope strides
};

// the whole plan: out_ids plus every closure tensor it reaches
struct llama_nout_plan {
    ggml_tensor *            out_ids;    // the I32 index tensor (ne[0] narrowed)
    uint32_t                 n_active;   // active row count this plan narrows to
    size_t                   n_entries;  // planned tensor count
    llama_nout_plan_entry *  entries;    // caller frees via delete[]
};

// Plan the output-closure narrowing of a graph built at the n_env envelope
// down to n_active rows. Validates the whole closure before returning:
// - gf must contain at least one GET_ROWS node reading out_ids;
// - every closure op must be shape-consistent under the narrowing;
// - n_active must be within [0, n_env]; n_active == 0 is a valid empty bind.
// Returns true and fills *plan (entries owned, freed by llama_nout_plan_apply
// or llama_nout_plan_free). Returns false and leaves every tensor untouched
// when any closure op is not provably safe under the narrowing.
bool llama_nout_plan_bind(struct ggml_cgraph * gf,
                          struct ggml_tensor * out_ids,
                          uint32_t             n_env,
                          uint32_t             n_active,
                          struct llama_nout_plan * plan);

// Apply a successful plan: narrow every planned tensor in place. Idempotent
// per plan; the next bind must first restore the envelope (llama-nout state
// side table in llama-graph.cpp owns the restore/record cycle).
void llama_nout_plan_apply(const struct llama_nout_plan * plan);

// Free a plan's entries (safe on a zeroed plan; does not touch tensors).
void llama_nout_plan_free(struct llama_nout_plan * plan);
