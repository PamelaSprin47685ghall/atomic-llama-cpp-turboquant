#include "llama-nout-bind.h"

//
// llama-nout-bind.cpp
//
// GGML_TP5_NOUT_REUSE: runtime output-capacity lane.
//
// The planner walks the output-gather closure of every GET_ROWS reading
// out_ids and validates that every consumer narrows its own shape the same
// way. Membership is derived, never assumed: a tensor joins the plan because
// it is a gather dst, or because its producing node consumes a planned tensor
// and its op narrows rows. A consumer whose operands cannot be resolved at
// plan time is retried until the closure stops growing, then rejects the bind.
//
// Every accepted op is one whose dst shape is provably consistent under the
// narrowing:
//
//   GET_ROWS  dst = [a->ne[0], index count]; the kernel asserts
//             ggml_nrows(dst) == ggml_nelements(src1) and reads only the
//             first n_active ids (set_input wrote exactly those)
//   ADD/MUL   dst = src0's shape (ggml_can_repeat(src1, src0)); both operands
//             narrow together; a single-row operand is a broadcast weight and
//             keeps its shape
//   RMS_NORM/NORM   row-wise over dim 0, dst = src shape
//   SCALE     elementwise with a scalar param, dst = src shape
//   CONT      contiguous copy, every dim shape-checked
//   MUL_MAT   dst ne[1] = src1->ne[1] (ggml_mul_mat); the weight (src0) is
//             never narrowed
//   GLU       split form only (ggml_swiglu_split): src0/src1/dst carry
//             identical shapes and narrow together; the packed one-input form
//             (src1 == nullptr) rejects
//
// Anything else in the closure (VIEW, RESHAPE, PERMUTE, TRANSPOSE, PAD,
// softmax, MUL_MAT_ID, UNARY, sampler nodes) rejects the bind: the caller
// rebuilds the graph at the exact active count, which is stock behavior.
//
// Zero active rows are accepted: the closure narrows to zero rows. ggml skips
// empty tensors (ggml_compute_forward returns early on ggml_is_empty), so a
// 0-row closure executes zero rows for real - no padding.
//

#include <cstring>
#include <vector>

namespace {

using nout_entry = llama_nout_plan_entry;

struct nout_planner {
    std::vector<nout_entry> entries;

    // dense index by tensor pointer for the operand checks
    std::vector<ggml_tensor *> tset;

    long find(const ggml_tensor * t) const {
        for (size_t i = 0; i < tset.size(); ++i) {
            if (tset[i] == t) {
                return (long) i;
            }
        }
        return -1;
    }

    size_t add(ggml_tensor * t) {
        const size_t k = entries.size();
        nout_entry e;
        e.t    = t;
        e.rows = -1;
        memcpy(e.env_ne, t->ne, sizeof(e.env_ne));
        memcpy(e.env_nb, t->nb, sizeof(e.env_nb));
        tset.push_back(t);
        entries.push_back(e);
        return k;
    }

    // require `t` to carry exactly `rows` rows; false on conflict
    bool require(ggml_tensor * t, int64_t rows) {
        long k = find(t);
        if (k < 0) {
            k = (long) add(t);
        }
        if (entries[k].rows < 0) {
            entries[k].rows = rows;
            return true;
        }
        return entries[k].rows == rows;
    }

    enum operand_res { OP_OK, OP_DEFER, OP_REJECT };

    // ADD/MUL operand classification:
    //   OP_OK     - single-row broadcast weight (keeps its shape), or a tensor
    //               already in the plan at the same row count
    //   OP_DEFER  - a row-bearing tensor not yet in the plan (it may still
    //               join through its own producer; retry the consumer later)
    //   OP_REJECT - a row-bearing tensor that is neither broadcast, in the
    //               plan, nor at the envelope width
    operand_res operand(ggml_tensor * t, int64_t rows, uint32_t n_env, bool broadcast) const {
        if (!ggml_is_contiguous(t)) {
            return OP_REJECT;
        }
        const long k = find(t);
        if (k >= 0) {
            return entries[k].rows == rows ? OP_OK : OP_REJECT;
        }
        if (broadcast && t->ne[1] == 1) {
            return OP_OK;
        }
        if (t->ne[1] != (int64_t) n_env) {
            return OP_REJECT;
        }
        return OP_DEFER;
    }
};

// recompute the contiguous nb[] of a narrowed tensor: rows (dim 1) stay the
// outermost varying dim for every accepted op (all narrowed tensors are
// verified contiguous), so nb[0] keeps its value and nb[1..3] follow ne[]
void nout_narrow_nb(ggml_tensor * t) {
    const size_t blck = ggml_blck_size(t->type);
    t->nb[1] = t->nb[0] * (size_t) (t->ne[0] / blck);
    for (int i = 2; i < GGML_MAX_DIMS; ++i) {
        t->nb[i] = t->nb[i - 1] * (size_t) t->ne[i - 1];
    }
}

} // namespace

bool llama_nout_plan_bind(struct ggml_cgraph * gf,
                                     struct ggml_tensor * out_ids,
                                     uint32_t             n_env,
                                     uint32_t             n_active,
                                     struct llama_nout_plan * plan) {
    if (plan) {
        plan->out_ids   = nullptr;
        plan->n_active  = 0;
        plan->n_entries = 0;
        plan->entries   = nullptr;
    }
    if (gf == nullptr || out_ids == nullptr || plan == nullptr) {
        return false;
    }
    if (out_ids->type != GGML_TYPE_I32 || out_ids->ne[1] != 1 || out_ids->ne[2] != 1 || out_ids->ne[3] != 1) {
        return false;
    }
    if (out_ids->nb[0] != sizeof(int32_t) || !ggml_is_contiguous(out_ids)) {
        return false;
    }
    if (n_active > n_env) {
        return false;
    }

    nout_planner p;

    // roots: every gather over out_ids. The gather dst is built over the
    // envelope; the index reads only the first n_active ids. The gather src
    // (pre-gather activations) is upstream and stays at its full token count.
    bool have_root = false;
    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->op != GGML_OP_GET_ROWS || node->src[1] != out_ids) {
            continue;
        }
        if (node->src[0] == nullptr || !ggml_is_contiguous(node->src[0]) || node->src[0]->ne[2] != 1) {
            return false;
        }
        if (node->ne[1] != (int64_t) n_env || node->ne[2] != 1 || node->ne[3] != 1) {
            return false;
        }
        if (!p.require(node, (int64_t) n_active)) {
            return false;
        }
        have_root = true;
    }
    if (!have_root) {
        return false;
    }
    // the index tensor itself: the narrowed dim is ne[0]; every non-gather
    // consumer of it rejects (checked in the consumer walk below)
    if (!p.require(out_ids, (int64_t) n_active)) {
        return false;
    }

    // consumer walk to a fixpoint: every op reading a planned tensor must
    // narrow its own dst the same way; a consumer whose operands are not all
    // resolvable yet is deferred and retried after the closure grows; at the
    // fixpoint any consumer still unresolved rejects the bind
    std::vector<size_t>     work;   // planned indices pending consumer scan
    std::vector<uint8_t>    done;   // per entry: consumer scan finished
    work.push_back(p.find(out_ids)); // the index tensor's consumers must reject
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_GET_ROWS && node->src[1] == out_ids) {
            work.push_back(p.find(node)); // walk every gather dst's closure
        }
    }
    done.assign(p.entries.size(), 0);

    bool progress = true;
    while (progress) {
        progress = false;
        done.resize(p.entries.size(), 0); // entries may have grown last round
        std::vector<size_t> next;
        for (const size_t k : work) {
            if (done[k]) {
                continue;
            }
            done[k] = 1;
            bool entry_deferred = false;
            ggml_tensor * src = p.entries[k].t;
            const int64_t rows = p.entries[k].rows;

            for (int i = 0; i < n_nodes; ++i) {
                ggml_tensor * node = ggml_graph_node(gf, i);

                bool consumes = false;
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    if (node->src[s] == src) {
                        consumes = true;
                        break;
                    }
                }
                if (!consumes) {
                    continue;
                }

                if (node->op == GGML_OP_GET_ROWS && node->src[1] == out_ids) {
                    continue; // the output gather root itself
                }

                // dst shapes were built over the envelope; the active count
                // only exists after the apply phase, so validate against the
                // envelope width here
                bool ok = true;
                bool defer = false;
                switch (node->op) {
                    case GGML_OP_ADD:
                    case GGML_OP_MUL:
                        {
                            // dst = src0's shape (ggml_add/ggml_mul require
                            // ggml_can_repeat(src1, src0)); a single-row operand
                            // is a broadcast weight and keeps its shape
                            ggml_tensor * a = node->src[0];
                            ggml_tensor * b = node->src[1];
                            if (a == nullptr || b == nullptr || !ggml_is_contiguous(node)) {
                                ok = false;
                                break;
                            }
                            if (node->ne[1] != (int64_t) n_env || node->ne[0] != a->ne[0]) {
                                ok = false;
                                break;
                            }
                            const nout_planner::operand_res ra = p.operand(a, rows, n_env, false);
                            const nout_planner::operand_res rb = p.operand(b, rows, n_env, true);
                            if (ra == nout_planner::OP_REJECT || rb == nout_planner::OP_REJECT) {
                                ok = false;
                            } else if (ra == nout_planner::OP_DEFER || rb == nout_planner::OP_DEFER) {
                                defer = true;
                            } else if (!p.require(node, rows)) {
                                ok = false;
                            }
                        }
                        break;
                    case GGML_OP_RMS_NORM:
                    case GGML_OP_NORM:
                    case GGML_OP_SCALE:
                        // row-wise over dim 0 (RMS_NORM/NORM) or elementwise
                        // with a scalar param (SCALE): dst is a fresh
                        // duplicate of src
                        if (node->src[0] != src || !ggml_is_contiguous(node)) {
                            ok = false;
                            break;
                        }
                        if (node->ne[0] != src->ne[0] || node->ne[1] != (int64_t) n_env ||
                            node->ne[2] != src->ne[2] || node->ne[3] != src->ne[3]) {
                            ok = false;
                            break;
                        }
                        if (!p.require(node, rows)) {
                            ok = false;
                        }
                        break;
                    case GGML_OP_CONT:
                        // contiguous copy: every dim must match the narrowed src
                        if (node->src[0] != src || !ggml_is_contiguous(node)) {
                            ok = false;
                            break;
                        }
                        if (node->ne[0] != src->ne[0] || node->ne[1] != (int64_t) n_env ||
                            node->ne[2] != src->ne[2] || node->ne[3] != src->ne[3]) {
                            ok = false;
                            break;
                        }
                        if (!p.require(node, rows)) {
                            ok = false;
                        }
                        break;
                    case GGML_OP_MUL_MAT:
                        {
                            // dst rows derive from src1 (ggml_mul_mat: dst
                            // ne[1] = src1->ne[1]); the weight (src0) is never
                            // narrowed
                            ggml_tensor * a = node->src[0];
                            ggml_tensor * b = node->src[1];
                            if (b != src || a == nullptr) {
                                ok = false;
                                break;
                            }
                            if (node->ne[1] != (int64_t) n_env ||
                                node->ne[2] != b->ne[2] || node->ne[3] != b->ne[3]) {
                                ok = false;
                                break;
                            }
                            if (!p.require(node, rows)) {
                                ok = false;
                            }
                        }
                        break;
                    case GGML_OP_GLU:
                        {
                            // split form only (ggml_swiglu_split): src0/src1/
                            // dst carry identical shapes and narrow together;
                            // the packed one-input form (src1 == nullptr) and
                            // packed-input unpacked-dst shapes reject
                            ggml_tensor * a = node->src[0];
                            ggml_tensor * b = node->src[1];
                            if (a == nullptr || b == nullptr) {
                                ok = false;
                                break;
                            }
                            if (!ggml_are_same_shape(a, b) || a->type != b->type) {
                                ok = false;
                                break;
                            }
                            if (!ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(node)) {
                                ok = false;
                                break;
                            }
                            if (node->ne[0] != a->ne[0] || node->ne[1] != (int64_t) n_env) {
                                ok = false;
                                break;
                            }
                            const nout_planner::operand_res ra = p.operand(a, rows, n_env, false);
                            const nout_planner::operand_res rb = p.operand(b, rows, n_env, false);
                            if (ra == nout_planner::OP_REJECT || rb == nout_planner::OP_REJECT) {
                                ok = false;
                            } else if (ra == nout_planner::OP_DEFER || rb == nout_planner::OP_DEFER) {
                                defer = true;
                            } else if (!p.require(node, rows)) {
                                ok = false;
                            }
                        }
                        break;
                    default:
                        // anything else in the closure (VIEW, RESHAPE,
                        // PERMUTE, TRANSPOSE, PAD, softmax, MUL_MAT_ID,
                        // UNARY, sampler nodes, ...) fails closed
                        ok = false;
                        break;
                }

                if (!ok) {
                    return false;
                }
                if (defer) {
                    entry_deferred = true;
                    continue;
                }
                progress = true; // the node's dst joined the plan
                // the closure is transitive: the joined dst's own consumers
                // must be scanned as well; done[] skips already-scanned
                // entries and require() is idempotent for re-visits
                const long dst_idx = p.find(node);
                // tensors during this walk); keep the scan index in range
                if (dst_idx >= 0) {
                    if ((size_t) dst_idx >= done.size()) {
                        done.resize(p.entries.size(), 0);
                    }
                    if (!done[(size_t) dst_idx]) {
                        next.push_back((size_t) dst_idx);
                    }
                }
            }
            if (entry_deferred) {
                // operands not all resolvable yet: rescan this entry after
                // the closure grows; clear done so the retry actually runs
                done[k] = 0;
                next.push_back(k);
            }
        }
        if (!progress && !next.empty()) {
            // deferred consumers whose operands can no longer join: reject
            return false;
        }
        work.swap(next);
    }

    plan->out_ids   = out_ids;
    plan->n_active  = n_active;
    plan->n_entries = p.entries.size();
    plan->entries   = new llama_nout_plan_entry[p.entries.size()];
    for (size_t i = 0; i < p.entries.size(); ++i) {
        plan->entries[i].t     = p.entries[i].t;
        plan->entries[i].rows  = p.entries[i].rows;
        memcpy(plan->entries[i].env_ne, p.entries[i].env_ne, sizeof(plan->entries[i].env_ne));
        memcpy(plan->entries[i].env_nb, p.entries[i].env_nb, sizeof(plan->entries[i].env_nb));
    }
    return true;
}

void llama_nout_plan_apply(const struct llama_nout_plan * plan) {
    if (plan == nullptr) {
        return;
    }
    for (size_t i = 0; i < plan->n_entries; ++i) {
        const llama_nout_plan_entry * e = &plan->entries[i];
        if (e->rows < 0) {
            continue;
        }
        if (e->t == plan->out_ids) {
            // the index tensor: the narrowed dim is ne[0]; nb[0] (element
            // stride) and the buffer (allocated for the envelope) are
            // unchanged, the gather kernel reads only the first n_active ids
            e->t->ne[0] = (int64_t) plan->n_active;
            nout_narrow_nb(e->t);
            continue;
        }
        e->t->ne[1] = e->rows;
        nout_narrow_nb(e->t);
    }
}

void llama_nout_plan_free(struct llama_nout_plan * plan) {
    delete[] plan->entries;
    plan->entries   = nullptr;
    plan->n_entries = 0;
}
