#ifndef TP5_ROUTER_SCRATCH_DIAG_H
#define TP5_ROUTER_SCRATCH_DIAG_H

// Diagnostic-only guard for the TP5 MoE router scratch candidate
// (GGML_TP5_MOE_ROUTER_SCRATCH).
//
// SCOPE — read this before using any line below. This observer is deliberately
// weak and must not be promoted into evidence on its own:
//
//   * CPU only, non-relay only. `tp5_relay` rejects a non-null `cb_eval` outright
//     (src/llama-context.cpp: "no sampler, callback or alternate compute
//     policy"), so this wiring cannot be used on the TP5 relay path at all.
//   * Installed observer changes scheduler timing. A non-null `cb_eval` disables
//     the single-split `defer_input_flush` fast path
//     (ggml-backend.cpp: `defer_input_flush = ... && sched->callback_eval == nullptr`)
//     and the callback branch synchronizes the split backend once per batched
//     range (`ggml_backend_synchronize(split_backend)`). Any measurement taken
//     with this observer enabled is a different schedule from production.
//   * Visible set is only the executed split's nodes. Cross-backend input
//     copies and their dependency nodes live in `sched->graph`, never in
//     `split->graph`, so the callback cannot see them. A retained extent is
//     therefore compared against executed nodes only: DISJOINT here means
//     "disjoint from the nodes the scheduler submitted", NOT "disjoint from
//     every allocation in the plan".
//   * Kernel-internal dispatch (the fused topk_moe write path) is invisible.
//
// Because of the above, the authoritative overlap evidence is the production
// allocator audit (`GGML_TP5_ROUTER_SCRATCH_AUDIT=1`, `ggml-alloc.c`), which runs
// inside reserve over the complete node + leaf plan, needs no callback, and does
// not touch scheduler timing. This header only exists to answer a narrower
// question: is the retention flag still on an executed node, and does it avoid
// the executed nodes of the same step.
//
// Wiring (one instance per llama context, CPU non-relay contexts only):
//
//   tp5_router_scratch_diag diag;
//   tp5_router_scratch_diag_init(diag);
//   cparams.cb_eval           = tp5_router_scratch_diag_eval;
//   cparams.cb_eval_user_data = &diag;
//   ...
//   llama_decode(ctx, batch);
//   tp5_router_scratch_diag_end_step(diag, rc);
//   ...
//   tp5_router_scratch_diag_summary(diag);
//   tp5_router_scratch_diag_close(diag);
//
// Enable with GGML_TP5_ROUTER_SCRATCH_DIAG=1 (exact); anything else is a no-op.
// Output goes to $GGML_TP5_ROUTER_SCRATCH_DIAG_FILE when set, else stderr.
//
// Lines:
//   RETAIN     a NO_REUSE node was submitted by the scheduler (flag survived
//              build, reservation and plan replay). Named ffn_moe_logits-<il>,
//              so trunk vs MTP attribution is by name, not by count.
//   DISJOINT   that node's extent does not overlap the executed nodes of the
//              same step (weaker than the allocator audit, see above).
//   OVERLAP    it does; this is a real conflict for the fused consumer.
//   UNRESOLVED no extent available; never counted as disjoint, never as fire.
//   STEP       per-step totals.
//   summary    per-run totals.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct tp5_router_scratch_diag_row {
    const ggml_tensor * t;
    int64_t             ne0;
    int64_t             ne1;
    const void *        addr;
    size_t              nbytes;
    bool                overlaps;
};

struct tp5_router_scratch_diag {
    bool   enabled     = false;
    bool   file_opened = false;
    FILE * out         = stderr;

    int steps          = 0;   // decode steps observed
    int retains        = 0;   // ask-pass observations of a NO_REUSE node
    int retain_steps   = 0;   // steps with at least one such observation
    int fired_steps    = 0;   // steps where every observation was resolved and clean
    int disjoint       = 0;   // observations with no overlap against executed nodes
    int overlaps       = 0;   // observations that overlap an executed node
    int unresolved     = 0;   // observations without a resolved extent
    int retain_n1      = 0;   // observations with ne[1] == 1 (must stay 0)

    std::vector<tp5_router_scratch_diag_row> observed;      // current step
    std::vector<const ggml_tensor *>         seen;          // current step, executed nodes
    std::vector<const ggml_tensor *>         overlap_hit;   // current step
};

inline void tp5_router_scratch_diag_print(tp5_router_scratch_diag & d, const char * fmt, ...) {
    if (!d.enabled || d.out == nullptr) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vfprintf(d.out, fmt, ap);
    va_end(ap);
    fflush(d.out);
}

inline void tp5_router_scratch_diag_init(tp5_router_scratch_diag & d) {
    const char * on = getenv("GGML_TP5_ROUTER_SCRATCH_DIAG");
    d.enabled = on != nullptr && strcmp(on, "1") == 0;
    if (!d.enabled) {
        return;
    }
    const char * path = getenv("GGML_TP5_ROUTER_SCRATCH_DIAG_FILE");
    if (path != nullptr && path[0] != '\0') {
        FILE * f = fopen(path, "ab");
        if (f != nullptr) {
            d.out         = f;
            d.file_opened = true;
        }
    }
    const char * retain_env = getenv("GGML_TP5_MOE_ROUTER_SCRATCH");
    tp5_router_scratch_diag_print(d,
            "ROUTER_SCRATCH_DIAG begin retain_env=%s scope=cpu_non_relay_executed_nodes_only\n",
            retain_env ? retain_env : "(unset)");
}

inline void tp5_router_scratch_diag_close(tp5_router_scratch_diag & d) {
    if (d.file_opened && d.out != nullptr) {
        fclose(d.out);
        d.out         = nullptr;
        d.file_opened = false;
    }
}

inline size_t tp5_router_scratch_diag_extent(const ggml_tensor * t) {
    if (t == nullptr || t->data == nullptr) {
        return 0;
    }
    if (t->buffer != nullptr) {
        return ggml_backend_buffer_get_alloc_size(t->buffer, t);
    }
    return ggml_nbytes(t);
}

inline bool tp5_router_scratch_diag_ranges_overlap(const void * a, size_t na, const void * b, size_t nb) {
    if (a == nullptr || b == nullptr || na == 0 || nb == 0) {
        return false;
    }
    const uintptr_t alo = (uintptr_t) a;
    const uintptr_t blo = (uintptr_t) b;
    return alo < blo + nb && blo < alo + na;
}

// ask=true: observe the submitted node. Always returns false, so the scheduler
// batches the whole split into one graph_view and no ask=false (which could
// abort compute) is ever reached. Note this is already the non-production path
// described at the top of the file.
inline bool tp5_router_scratch_diag_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * d = static_cast<tp5_router_scratch_diag *>(user_data);
    if (d == nullptr || !d->enabled || t == nullptr) {
        return false;
    }
    if (ask) {
        d->seen.push_back(t);
        if (t->flags & GGML_TENSOR_FLAG_NO_REUSE) {
            tp5_router_scratch_diag_row row;
            row.t        = t;
            row.ne0      = t->ne[0];
            row.ne1      = t->ne[1];
            row.addr     = t->data;
            row.nbytes   = tp5_router_scratch_diag_extent(t);
            row.overlaps = false;
            d->observed.push_back(row);
            d->retains++;
            if (t->ne[1] == 1) {
                d->retain_n1++;
            }
            if (d->steps == 0) {
                tp5_router_scratch_diag_print(*d,
                        "ROUTER_SCRATCH_DIAG RETAIN tensor='%s' ne=[%lld,%lld] addr=%p nbytes=%zu\n",
                        t->name, (long long) t->ne[0], (long long) t->ne[1], t->data, row.nbytes);
            }
        }
    }
    return false;
}

// Per-step audit; call right after llama_decode returns.
inline void tp5_router_scratch_diag_end_step(tp5_router_scratch_diag & d, int decode_rc = 0) {
    if (!d.enabled) {
        return;
    }
    d.steps++;

    bool fired = !d.observed.empty();
    bool audited = fired;

    for (tp5_router_scratch_diag_row & row : d.observed) {
        const size_t kept_n = row.nbytes;
        if (row.addr == nullptr || kept_n == 0) {
            d.unresolved++;
            audited = false;
            tp5_router_scratch_diag_print(d, "ROUTER_SCRATCH_DIAG UNRESOLVED step=%d tensor='%s'\n",
                    d.steps, row.t->name);
            continue;
        }
        bool clash = false;
        for (const ggml_tensor * other : d.seen) {
            if (other == row.t) {
                continue;
            }
            const size_t n = tp5_router_scratch_diag_extent(other);
            if (!tp5_router_scratch_diag_ranges_overlap(row.addr, kept_n, other->data, n)) {
                continue;
            }
            clash = true;
            audited = false;
            d.overlap_hit.push_back(other);
            tp5_router_scratch_diag_print(d,
                    "ROUTER_SCRATCH_DIAG OVERLAP step=%d tensor='%s' ne=[%lld,%lld] other='%s' [%s] delta=%lld\n",
                    d.steps, row.t->name, (long long) row.ne0, (long long) row.ne1,
                    other->name, ggml_op_desc(other),
                    (long long) ((intptr_t) other->data - (intptr_t) row.addr));
            break;
        }
        if (clash) {
            row.overlaps = true;
            d.overlaps++;
        } else {
            d.disjoint++;
        }
    }

    if (fired) {
        d.retain_steps++;
    }
    if (fired && audited) {
        d.fired_steps++;
    }

    tp5_router_scratch_diag_print(d,
            "ROUTER_SCRATCH_DIAG STEP step=%d rc=%d retained=%zu observed=%zu overlaps_step=%zu fired=%d\n",
            d.steps, decode_rc, d.observed.size(), d.seen.size(), d.overlap_hit.size(),
            (fired && audited) ? 1 : 0);

    d.observed.clear();
    d.seen.clear();
    d.overlap_hit.clear();
}

inline void tp5_router_scratch_diag_summary(tp5_router_scratch_diag & d) {
    tp5_router_scratch_diag_print(d,
            "ROUTER_SCRATCH_DIAG summary steps=%d retains=%d retain_steps=%d fired_steps=%d "
            "disjoint=%d overlaps=%d unresolved=%d n1_shape=%d scope=executed_nodes_only\n",
            d.steps, d.retains, d.retain_steps, d.fired_steps, d.disjoint, d.overlaps,
            d.unresolved, d.retain_n1);
}

#endif // TP5_ROUTER_SCRATCH_DIAG_H
