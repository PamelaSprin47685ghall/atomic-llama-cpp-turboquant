// llama-flashprefill-metrics.cpp — bounded FlashPrefill V2 metrics aggregator.
//
// Owner: MetricsIntegration. See the header for the full contract.

#include "llama-flashprefill-metrics.h"

#include "llama-context.h"
#include "llama.h"

#include <cstdint>
#include <limits>

namespace llama_flashprefill_metrics {

namespace {

uint64_t sat_add(uint64_t a, uint64_t b) {
    return a > (std::numeric_limits<uint64_t>::max)() - b
        ? (std::numeric_limits<uint64_t>::max)()
        : a + b;
}

// Same frozen eligibility as the server sidecar
// (server_flashprefill_routing::role_is_sparse_eligible): only PREFILL and
// REROT_TEACHER_FORCED are ever sparse-eligible. Compares frozen values
// directly; no second convention.
bool row_role_eligible(int32_t role) {
    return role == LLAMA_FLASHPREFILL_ROLE_PREFILL ||
           role == LLAMA_FLASHPREFILL_ROLE_REROT_TEACHER_FORCED;
}

bool row_role_mtp(int32_t role) {
    return role == LLAMA_FLASHPREFILL_ROLE_MTP_VERIFY ||
           role == LLAMA_FLASHPREFILL_ROLE_MTP_DRAFT;
}

} // namespace

const char * dense_bucket_name(uint32_t bucket) {
    switch (bucket) {
        case DENSE_BUCKET_DECODE:      return "decode";
        case DENSE_BUCKET_MTP_VERIFY:  return "mtp_verify";
        case DENSE_BUCKET_ROLE_OTHER:  return "role_other";
        case DENSE_BUCKET_SHORT:       return "short_context";
        case DENSE_BUCKET_TAIL:        return "dense_tail";
        case DENSE_BUCKET_UNKNOWN:     return "unknown_boundary";
        case DENSE_BUCKET_UNSUPPORTED: return "unsupported";
        case DENSE_BUCKET_HIGHCOST:    return "high_cost";
        case DENSE_BUCKET_NO_PLAN:     return "no_plan";
        case DENSE_BUCKET_FULL_PREFIX: return "full_attention_layer";
        default:                       return "?";
    }
}

const char * pool_reason_name(uint32_t reason) {
    switch (reason) {
        case POOL_REASON_SLICE:     return "slice";
        case POOL_REASON_EXACT_ALL: return "exact_all";
        default:                    return "?";
    }
}

const char * plan_reason_name(uint32_t reason) {
    switch (reason) {
        case PLAN_REASON_BYPASSED:        return "bypassed";
        case PLAN_REASON_EMPTY_SNAPSHOT:  return "empty_snapshot";
        case PLAN_REASON_NO_PLAN:         return "no_plan";
        default:                          return "?";
    }
}

int32_t parse_plan_header_stats(
        const int32_t hdr24[24],
        uint64_t expected_ne_words,
        struct ggml_flashprefill_plan_stats * out) {
    if (hdr24 == nullptr || out == nullptr) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    *out = ggml_flashprefill_plan_stats();
    if (hdr24[0] != GGML_FLASHPREFILL_PLAN_MAGIC) {
        return GGML_FLASHPREFILL_ERR_BAD_MAGIC;
    }
    if (hdr24[1] != GGML_FLASHPREFILL_VERSION) {
        return GGML_FLASHPREFILL_ERR_BAD_VERSION;
    }
    if (hdr24[2] != GGML_FLASHPREFILL_PLAN_HEADER_WORDS) {
        return GGML_FLASHPREFILL_ERR_BAD_LAYOUT;
    }
    // Canonical offset arithmetic from (n_tiles, n_kv_heads, max_sel_pair),
    // all checked 64-bit; any mismatch (or absurd size) fails closed.
    // Stored words are int32: negativity is checked before any narrowing.
    if (hdr24[3] < 0 || hdr24[4] < 0 || hdr24[5] < 0 ||
        hdr24[6] < 0 || hdr24[7] < 0 || hdr24[8] < 0 || hdr24[9] < 0) {
        return GGML_FLASHPREFILL_ERR_BAD_LAYOUT;
    }
    const uint64_t ut = (uint64_t) hdr24[3];
    const uint64_t uh = (uint64_t) hdr24[4];
    const uint64_t um = (uint64_t) hdr24[5];
    if (ut > (uint64_t) (1u << 24) || uh > (uint64_t) (1u << 16) || um > (uint64_t) (1u << 24)) {
        return GGML_FLASHPREFILL_ERR_BAD_RANGE;
    }
    if (uh != 0 && um > UINT64_MAX / uh) {
        return GGML_FLASHPREFILL_ERR_OVERFLOW;
    }
    const uint64_t hm = uh * um;
    if (ut != 0 && hm > UINT64_MAX / ut) {
        return GGML_FLASHPREFILL_ERR_OVERFLOW;
    }
    const uint64_t table = ut * hm; // T*H*max entries per table
    uint64_t total = 24u;
    if (table > (UINT64_MAX - total) / 2u) {
        return GGML_FLASHPREFILL_ERR_OVERFLOW;
    }
    total += 2u * table;
    if (uh != 0 && ut > UINT64_MAX / uh) {
        return GGML_FLASHPREFILL_ERR_OVERFLOW;
    }
    const uint64_t th = ut * uh;
    if (th > (UINT64_MAX - total) / 2u) {
        return GGML_FLASHPREFILL_ERR_OVERFLOW;
    }
    total += 2u * th;
    if ((uint64_t) hdr24[6] != 24u || (uint64_t) hdr24[7] != 24u + table ||
        (uint64_t) hdr24[8] != 24u + 2u * table || (uint64_t) hdr24[9] != total) {
        return GGML_FLASHPREFILL_ERR_BAD_LAYOUT;
    }
    if (total > expected_ne_words) {
        // The snapshot's tensor cannot hold its own claimed layout: never
        // interpret counters past the actual storage.
        return GGML_FLASHPREFILL_ERR_BAD_RANGE;
    }
    // Nonnegative counters (negative packed counts or token sums are never
    // real measurements). The error word itself rides through untouched.
    if (hdr24[11] < 0 || hdr24[12] < 0 || hdr24[13] < 0 || hdr24[14] < 0) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    const uint64_t vis_lo = (uint32_t) hdr24[16];
    const uint64_t vis_hi = (uint32_t) hdr24[17];
    const uint64_t ext_lo = (uint32_t) hdr24[18];
    const uint64_t ext_hi = (uint32_t) hdr24[19];
    const uint64_t visible = (vis_hi << 32) | vis_lo;
    const uint64_t exact   = (ext_hi << 32) | ext_lo;
    if (visible > (uint64_t) INT64_MAX || exact > (uint64_t) INT64_MAX) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    out->error           = hdr24[10];
    out->sparse_rows     = hdr24[11];
    out->dense_rows      = hdr24[12];
    out->selected_total  = hdr24[13];
    out->corrected_total = hdr24[14];
    out->req_exact_all   = hdr24[15];
    out->visible_tokens  = (int64_t) visible;
    out->exact_tokens    = (int64_t) exact;
    return GGML_FLASHPREFILL_OK;
}

uint32_t classify_dense_bucket(int32_t route, int32_t role) {
    switch (route) {
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_ROLE:
            if (role == LLAMA_FLASHPREFILL_ROLE_DECODE) {
                return DENSE_BUCKET_DECODE;
            }
            if (row_role_mtp(role)) {
                return DENSE_BUCKET_MTP_VERIFY;
            }
            if (role >= LLAMA_FLASHPREFILL_ROLE_UNKNOWN &&
                role <= LLAMA_FLASHPREFILL_ROLE_MULTIMODAL) {
                return DENSE_BUCKET_ROLE_OTHER;
            }
            return DENSE_BUCKET_COUNT; // out-of-range role: never trusted dense
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_SHORT_CONTEXT:
            return DENSE_BUCKET_SHORT;
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_TAIL:
            return DENSE_BUCKET_TAIL;
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_UNKNOWN_BOUNDARY:
            return DENSE_BUCKET_UNKNOWN;
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED:
            return DENSE_BUCKET_UNSUPPORTED;
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_CAPACITY:
            return DENSE_BUCKET_HIGHCOST;
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF:
            // OFF slices never reach the classifier with rows (OFF returns an
            // empty delta up front); OFF is not a per-row dense reason here.
            return DENSE_BUCKET_COUNT;
        default:
            // SPARSE / EXACT_ALL / out-of-range: not dense.
            return DENSE_BUCKET_COUNT;
    }
}

bool slice_is_empty(const slice_delta & d) {
    if (d.eligible_rows != 0 || d.sparse_rows != 0 || d.dense_packed != 0 ||
        d.selected_blocks != 0 || d.corrected_blocks != 0 ||
        d.visible_tokens != 0 || d.exact_tokens != 0 ||
        d.scratch_live_bytes != 0 || d.scratch_peak_bytes != 0 ||
        d.has_layout_us || d.has_gpu_us) {
        return false;
    }
    for (uint32_t i = 0; i < DENSE_BUCKET_COUNT; ++i) {
        if (d.dense_rows[i] != 0) {
            return false;
        }
    }
    for (uint32_t i = 0; i < POOL_REASON_COUNT; ++i) {
        if (d.pool_rebuild[i] != 0) {
            return false;
        }
    }
    for (uint32_t i = 0; i < PLAN_REASON_COUNT; ++i) {
        if (d.plan_invalidations[i] != 0) {
            return false;
        }
    }
    return true;
}

bool accum::empty() const {
    if (eligible_rows != 0 || sparse_rows != 0 || dense_packed != 0 ||
        selected_blocks != 0 || corrected_blocks != 0 ||
        visible_tokens != 0 || exact_tokens != 0 ||
        scratch_live_bytes != 0 || scratch_peak_bytes != 0 ||
        layout_us_total != 0 || layout_slices_measured != 0 ||
        gpu_pool_us_total != 0 || gpu_select_us_total != 0 ||
        gpu_attn_us_total != 0 || gpu_slices_measured != 0 || has_policy) {
        return false;
    }
    for (uint32_t i = 0; i < DENSE_BUCKET_COUNT; ++i) {
        if (dense_rows[i] != 0) {
            return false;
        }
    }
    for (uint32_t i = 0; i < POOL_REASON_COUNT; ++i) {
        if (pool_rebuild[i] != 0) {
            return false;
        }
    }
    for (uint32_t i = 0; i < PLAN_REASON_COUNT; ++i) {
        if (plan_invalidations[i] != 0) {
            return false;
        }
    }
    return true;
}

void delta_merge(slice_delta & a, const slice_delta & d) {
    a.eligible_rows     = sat_add(a.eligible_rows, d.eligible_rows);
    a.sparse_rows       = sat_add(a.sparse_rows, d.sparse_rows);
    a.dense_packed      = sat_add(a.dense_packed, d.dense_packed);
    a.selected_blocks   = sat_add(a.selected_blocks, d.selected_blocks);
    a.corrected_blocks  = sat_add(a.corrected_blocks, d.corrected_blocks);
    a.visible_tokens    = sat_add(a.visible_tokens, d.visible_tokens);
    a.exact_tokens      = sat_add(a.exact_tokens, d.exact_tokens);
    for (uint32_t i = 0; i < DENSE_BUCKET_COUNT; ++i) {
        a.dense_rows[i] = sat_add(a.dense_rows[i], d.dense_rows[i]);
    }
    for (uint32_t i = 0; i < POOL_REASON_COUNT; ++i) {
        a.pool_rebuild[i] = sat_add(a.pool_rebuild[i], d.pool_rebuild[i]);
    }
    for (uint32_t i = 0; i < PLAN_REASON_COUNT; ++i) {
        a.plan_invalidations[i] = sat_add(a.plan_invalidations[i], d.plan_invalidations[i]);
    }
    a.scratch_live_bytes = d.scratch_live_bytes;
    if (d.scratch_peak_bytes > a.scratch_peak_bytes) {
        a.scratch_peak_bytes = d.scratch_peak_bytes;
    }
    if (d.has_layout_us) {
        a.layout_us = sat_add(a.layout_us, d.layout_us);
        a.has_layout_us = true;
    }
    if (d.has_gpu_us) {
        a.gpu_pool_us   = sat_add(a.gpu_pool_us, d.gpu_pool_us);
        a.gpu_select_us = sat_add(a.gpu_select_us, d.gpu_select_us);
        a.gpu_attn_us   = sat_add(a.gpu_attn_us, d.gpu_attn_us);
        a.has_gpu_us = true;
    }
}

void accum_merge(accum & a, const slice_delta & d) {
    a.eligible_rows     = sat_add(a.eligible_rows, d.eligible_rows);
    a.sparse_rows       = sat_add(a.sparse_rows, d.sparse_rows);
    a.dense_packed      = sat_add(a.dense_packed, d.dense_packed);
    a.selected_blocks   = sat_add(a.selected_blocks, d.selected_blocks);
    a.corrected_blocks  = sat_add(a.corrected_blocks, d.corrected_blocks);
    a.visible_tokens    = sat_add(a.visible_tokens, d.visible_tokens);
    a.exact_tokens      = sat_add(a.exact_tokens, d.exact_tokens);
    for (uint32_t i = 0; i < DENSE_BUCKET_COUNT; ++i) {
        a.dense_rows[i] = sat_add(a.dense_rows[i], d.dense_rows[i]);
    }
    for (uint32_t i = 0; i < POOL_REASON_COUNT; ++i) {
        a.pool_rebuild[i] = sat_add(a.pool_rebuild[i], d.pool_rebuild[i]);
    }
    for (uint32_t i = 0; i < PLAN_REASON_COUNT; ++i) {
        a.plan_invalidations[i] = sat_add(a.plan_invalidations[i], d.plan_invalidations[i]);
    }
    // Gauges, not counters: live overwrites (buffers are reused across
    // layers/slices), peak takes the max, layout time sums only measured
    // slices (unmeasured slices contribute nothing, never fake zero).
    a.scratch_live_bytes = d.scratch_live_bytes;
    if (d.scratch_peak_bytes > a.scratch_peak_bytes) {
        a.scratch_peak_bytes = d.scratch_peak_bytes;
    }
    if (d.has_layout_us) {
        a.layout_us_total = sat_add(a.layout_us_total, d.layout_us);
        a.layout_slices_measured += 1;
    }
    if (d.has_gpu_us) {
        a.gpu_pool_us_total   = sat_add(a.gpu_pool_us_total, d.gpu_pool_us);
        a.gpu_select_us_total = sat_add(a.gpu_select_us_total, d.gpu_select_us);
        a.gpu_attn_us_total   = sat_add(a.gpu_attn_us_total, d.gpu_attn_us);
        a.gpu_slices_measured += 1;
    }
}

slice_delta accum_delta_since(const accum & acc, const accum & mark) {
    slice_delta d;
    d.eligible_rows     = acc.eligible_rows     - mark.eligible_rows;
    d.sparse_rows       = acc.sparse_rows       - mark.sparse_rows;
    d.dense_packed      = acc.dense_packed      - mark.dense_packed;
    d.selected_blocks   = acc.selected_blocks   - mark.selected_blocks;
    d.corrected_blocks  = acc.corrected_blocks  - mark.corrected_blocks;
    d.visible_tokens    = acc.visible_tokens    - mark.visible_tokens;
    d.exact_tokens      = acc.exact_tokens      - mark.exact_tokens;
    for (uint32_t i = 0; i < DENSE_BUCKET_COUNT; ++i) {
        d.dense_rows[i] = acc.dense_rows[i] - mark.dense_rows[i];
    }
    for (uint32_t i = 0; i < POOL_REASON_COUNT; ++i) {
        d.pool_rebuild[i] = acc.pool_rebuild[i] - mark.pool_rebuild[i];
    }
    for (uint32_t i = 0; i < PLAN_REASON_COUNT; ++i) {
        d.plan_invalidations[i] = acc.plan_invalidations[i] - mark.plan_invalidations[i];
    }
    // Gauges ride as snapshots, not deltas: the consumer takes the current
    // values. Layout/GPU time deltas need the slice counts to stay
    // meaningful, so they are carried alongside (all plain differences).
    d.scratch_live_bytes = acc.scratch_live_bytes;
    d.scratch_peak_bytes = acc.scratch_peak_bytes;
    d.layout_us          = acc.layout_us_total - mark.layout_us_total;
    d.has_layout_us      = (acc.layout_slices_measured != mark.layout_slices_measured);
    d.gpu_pool_us        = acc.gpu_pool_us_total - mark.gpu_pool_us_total;
    d.gpu_select_us      = acc.gpu_select_us_total - mark.gpu_select_us_total;
    d.gpu_attn_us        = acc.gpu_attn_us_total - mark.gpu_attn_us_total;
    d.has_gpu_us         = (acc.gpu_slices_measured != mark.gpu_slices_measured);
    return d;
}

namespace {

// Packed subhead range for one row's tail check (frozen row-major packing:
// packed index of (row i, subhead s) is i*gqa+s call-scope, or
// (pos-begin)*gqa+s logical-scope). Returns false when the position must
// not be guessed (UNKNOWN verdict by the caller, never a fabricated index).
bool tail_range_for_row(
        const struct llama_flashprefill_row & row,
        uint32_t i,
        uint32_t gqa,
        uint64_t total_call_packed_64,
        bool use_logical_tail,
        uint64_t & range_base_64,
        uint64_t & range_total_64) {
    range_base_64 = 0;
    range_total_64 = 0;
    if (use_logical_tail) {
        if (row.logical_pos == LLAMA_FLASHPREFILL_POS_UNKNOWN ||
            row.logical_pos < row.prefill_begin || row.logical_pos >= row.prefill_end) {
            return false;
        }
        const uint64_t tok_off = (uint64_t) (row.logical_pos - row.prefill_begin);
        const uint64_t span    = (uint64_t) (row.prefill_end - row.prefill_begin);
        range_base_64  = tok_off * (uint64_t) gqa;
        range_total_64 = span * (uint64_t) gqa;
        return range_total_64 <= (uint64_t) UINT32_MAX && range_base_64 <= (uint64_t) UINT32_MAX;
    }
    range_base_64  = (uint64_t) i * (uint64_t) gqa;
    range_total_64 = total_call_packed_64;
    return true;
}

} // namespace

int32_t build_slice(
        slice_delta & out,
        const struct llama_flashprefill_config * cfg,
        const struct llama_flashprefill_row * rows,
        uint32_t n_rows,
        int32_t graph_dense_reason,
        uint32_t gqa,
        bool bypassed,
        const struct ggml_flashprefill_plan_stats * plans,
        uint32_t n_plans,
        bool plans_present,
        uint64_t scratch_live_bytes,
        uint64_t scratch_peak_bytes,
        uint64_t layout_us,
        bool has_layout_us,
        bool plans_deferred) {
    out = slice_delta();

    // OFF (or null/invalid config): empty delta, rows never touched, no
    // timers armed, no reads queued. Invalid configs fail closed upstream at
    // context construction; here they just count nothing.
    if (cfg == nullptr || llama_flashprefill_validate_config(cfg) != LLAMA_FLASHPREFILL_OK) {
        return 0;
    }
    if (!llama_flashprefill_is_enabled(cfg)) {
        return 0;
    }
    if (gqa == 0) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }

    // Bypassed calls (MTP contexts, embedding/rerank pooling) keep the
    // pre-existing attention path with no per-row arrays: one slice-level
    // invalidation, no row attribution.
    if (bypassed) {
        out.plan_invalidations[PLAN_REASON_BYPASSED] = 1;
        return 0;
    }

    if (n_rows == 0 || rows == nullptr) {
        // Enabled but no owned ubatch rows: legitimate dense (map-less ubatch,
        // legacy path). One slice-level invalidation, nothing else.
        if (n_rows != 0 && rows == nullptr) {
            return GGML_FLASHPREFILL_ERR_BAD_ARG;
        }
        out.plan_invalidations[PLAN_REASON_EMPTY_SNAPSHOT] = 1;
        return 0;
    }

    // Plan-header error gate (CpuKernels contract): inspect EVERY layer's
    // error word before trusting ANY counter or the ATTN output. SELECT
    // writes plan[10] (single writer after barrier); ATTN may set it
    // atomically. The caller guarantees full-graph completion before this
    // call, so these stats already cover both writers. First error wins;
    // the delta stays empty and the slice must fail (no successful output,
    // no sampling/state metrics downstream).
    int32_t first_error = 0;
    bool any_exact_all = false;
    if (plans_present) {
        if (plans == nullptr || n_plans == 0) {
            return GGML_FLASHPREFILL_ERR_BAD_ARG;
        }
        for (uint32_t p = 0; p < n_plans; ++p) {
            if (plans[p].error != 0) {
                if (first_error == 0) {
                    first_error = plans[p].error;
                }
            }
            if (plans[p].req_exact_all != 0) {
                any_exact_all = true;
            }
            // Corrupt stats fields fail closed: negative packed counts or
            // negative 64-bit token sums are never real measurements.
            if (plans[p].sparse_rows < 0 || plans[p].dense_rows < 0 ||
                plans[p].selected_total < 0 || plans[p].corrected_total < 0 ||
                plans[p].visible_tokens < 0 || plans[p].exact_tokens < 0) {
                if (first_error == 0) {
                    first_error = GGML_FLASHPREFILL_ERR_BAD_ARG;
                }
            }
        }
        if (first_error != 0) {
            out = slice_delta();
            return first_error;
        }
        // Plan-confirmed GPU work only (WireReference units): packed rows for
        // sparse/dense, use counts for selected/corrected, use-record sums
        // (64-bit, subhead fan-out NOT multiplied) for visible/exact.
        for (uint32_t p = 0; p < n_plans; ++p) {
            out.sparse_rows       = sat_add(out.sparse_rows, (uint64_t) plans[p].sparse_rows);
            out.dense_packed      = sat_add(out.dense_packed, (uint64_t) plans[p].dense_rows);
            out.selected_blocks   = sat_add(out.selected_blocks, (uint64_t) plans[p].selected_total);
            // Executed corrections only: under the mean_correction=false
            // ablation the proxy lists still classify (header14 stays valid
            // for coverage validators) but ATTN deliberately skips them, so
            // the metric reports 0 instead of claiming unapplied work.
            if (cfg->mean_correction) {
                out.corrected_blocks = sat_add(out.corrected_blocks, (uint64_t) plans[p].corrected_total);
            }
            out.visible_tokens    = sat_add(out.visible_tokens, (uint64_t) plans[p].visible_tokens);
            out.exact_tokens      = sat_add(out.exact_tokens, (uint64_t) plans[p].exact_tokens);
        }
    } else {
        if (plans != nullptr || n_plans != 0) {
            return GGML_FLASHPREFILL_ERR_BAD_ARG;
        }
        // No plan stats here by design: plan-less ubatches are classified
        // from the authoritative graph verdict below (or skipped for the
        // fold when deferred). Residual rows file the NO_PLAN bucket only;
        // no invalidation fires without plans (nothing to invalidate).
    }

    // Row-side classification. Exact row data (role, known-boundary) applies
    // on every path; every dense verdict beyond that comes from the
    // authoritative graph outcome — never from global KV fullness or guessed
    // backend support.
    const bool use_logical_tail = (cfg->tail_scope == LLAMA_FLASHPREFILL_TAIL_LOGICAL_PROMPT);

    // Call-scope packed size (row-major packing: packed index of (row i,
    // subhead s) is i*gqa+s, matching packed_layout_checked's total). 64-bit
    // with an explicit overflow guard; overflow fails the slice closed.
    const uint64_t total_call_packed_64 = (uint64_t) n_rows * (uint64_t) gqa;
    if (total_call_packed_64 > (uint64_t) UINT32_MAX) {
        out = slice_delta();
        return GGML_FLASHPREFILL_ERR_OVERFLOW;
    }

    bool any_new_path_packed = false;

    // Whole-ubatch verdict buckets for plan-less ubatches (frozen routes file
    // literally; HIGH_COST is the graph-internal bound; anything else is an
    // unclassifiable residual filed as NO_PLAN).
    const bool verdict_short     = !plans_present && !plans_deferred &&
        graph_dense_reason == LLAMA_FLASHPREFILL_ROUTE_DENSE_SHORT_CONTEXT;
    const bool verdict_unknown   = !plans_present && !plans_deferred &&
        graph_dense_reason == LLAMA_FLASHPREFILL_ROUTE_DENSE_UNKNOWN_BOUNDARY;
    const bool verdict_unsup     = !plans_present && !plans_deferred &&
        graph_dense_reason == LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED;
    const bool verdict_capacity  = !plans_present && !plans_deferred &&
        (graph_dense_reason == LLAMA_FLASHPREFILL_ROUTE_DENSE_CAPACITY ||
         graph_dense_reason == LLM_FP_DENSE_HIGH_COST);
    const bool verdict_tail_all  = !plans_present && !plans_deferred &&
        graph_dense_reason == LLAMA_FLASHPREFILL_ROUTE_DENSE_TAIL;
    const bool verdict_designed  = !plans_present && !plans_deferred &&
        (graph_dense_reason == LLM_FP_DENSE_SPARSE_ACTIVE ||
         graph_dense_reason == LLM_FP_DENSE_FULL_PREFIX ||
         graph_dense_reason == LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF);

    for (uint32_t i = 0; i < n_rows; ++i) {
        const struct llama_flashprefill_row & row = rows[i];
        if (llama_flashprefill_validate_row(&row) != LLAMA_FLASHPREFILL_OK) {
            // A corrupt row inside an otherwise-validated snapshot fails the
            // slice closed (never silent fallback to dense for metrics: the
            // execution path already validated these rows at attach; a
            // mismatch here means memory corruption or a use-after-clear).
            out = slice_delta();
            return GGML_FLASHPREFILL_ERR_BAD_ARG;
        }

        const bool eligible_role = row_role_eligible(row.role);
        if (eligible_role && row.prefill_known) {
            out.eligible_rows = sat_add(out.eligible_rows, 1);
        }

        // Non-eligible roles: every packed subhead shares the role reason.
        // Exact row data — applies on every path, including designed-dense.
        if (!eligible_role) {
            const uint32_t b = classify_dense_bucket(LLAMA_FLASHPREFILL_ROUTE_DENSE_ROLE, row.role);
            if (b < DENSE_BUCKET_COUNT) {
                out.dense_rows[b] = sat_add(out.dense_rows[b], (uint64_t) gqa);
            }
            continue;
        }

        // Eligible-but-unknown rows: exact row data (server parity) on every
        // path, including designed-dense.
        if (!row.prefill_known) {
            out.dense_rows[DENSE_BUCKET_UNKNOWN] =
                sat_add(out.dense_rows[DENSE_BUCKET_UNKNOWN], (uint64_t) gqa);
            continue;
        }

        // Eligible + known rows.
        if (plans_present) {
            // GPU-confirmed path: TAIL by frozen position, the rest is
            // new-path work covered by the plan totals above.
            uint64_t range_base_64 = 0, range_total_64 = 0;
            if (!tail_range_for_row(row, i, gqa, total_call_packed_64, use_logical_tail,
                        range_base_64, range_total_64)) {
                out.dense_rows[DENSE_BUCKET_UNKNOWN] =
                    sat_add(out.dense_rows[DENSE_BUCKET_UNKNOWN], (uint64_t) gqa);
                continue;
            }
            const uint32_t range_total = (uint32_t) range_total_64;
            for (uint32_t s = 0; s < gqa; ++s) {
                const uint32_t packed = (uint32_t) (range_base_64 + (uint64_t) s);
                if (llama_flashprefill::packed_row_in_dense_tail(
                            packed, range_total, cfg->block_q, cfg->dense_tail_tiles)) {
                    out.dense_rows[DENSE_BUCKET_TAIL] =
                        sat_add(out.dense_rows[DENSE_BUCKET_TAIL], 1);
                } else {
                    any_new_path_packed = true;
                }
            }
            continue;
        }
        if (plans_deferred) {
            // This ubatch queued plan reads: TAIL by frozen position now
            // (exact rows); the rest resolves from parsed headers at fold.
            uint64_t range_base_64 = 0, range_total_64 = 0;
            if (!tail_range_for_row(row, i, gqa, total_call_packed_64, use_logical_tail,
                        range_base_64, range_total_64)) {
                out.dense_rows[DENSE_BUCKET_UNKNOWN] =
                    sat_add(out.dense_rows[DENSE_BUCKET_UNKNOWN], (uint64_t) gqa);
                continue;
            }
            const uint32_t range_total = (uint32_t) range_total_64;
            for (uint32_t s = 0; s < gqa; ++s) {
                const uint32_t packed = (uint32_t) (range_base_64 + (uint64_t) s);
                if (llama_flashprefill::packed_row_in_dense_tail(
                            packed, range_total, cfg->block_q, cfg->dense_tail_tiles)) {
                    out.dense_rows[DENSE_BUCKET_TAIL] =
                        sat_add(out.dense_rows[DENSE_BUCKET_TAIL], 1);
                }
            }
            continue;
        }
        // Plan-less ubatch: the authoritative verdict decides. Designed-dense
        // (sparse-active with zero plans, e.g. full-attention layer policy)
        // files nothing per-row; residuals file NO_PLAN.
        if (verdict_designed) {
            continue;
        }
        if (verdict_short) {
            // Whole-ubatch SHORT dominates tail (frozen precedence): every
            // packed row files here without per-row slicing.
            out.dense_rows[DENSE_BUCKET_SHORT] = sat_add(out.dense_rows[DENSE_BUCKET_SHORT], (uint64_t) gqa);
            continue;
        }
        if (verdict_unknown) {
            out.dense_rows[DENSE_BUCKET_UNKNOWN] =
                sat_add(out.dense_rows[DENSE_BUCKET_UNKNOWN], (uint64_t) gqa);
            continue;
        }
        if (verdict_tail_all) {
            out.dense_rows[DENSE_BUCKET_TAIL] =
                sat_add(out.dense_rows[DENSE_BUCKET_TAIL], (uint64_t) gqa);
            continue;
        }
        if (verdict_unsup || verdict_capacity) {
            // Precedence-faithful tail refinement first (positions exact),
            // then the verdict bucket for the rest.
            uint64_t range_base_64 = 0, range_total_64 = 0;
            if (!tail_range_for_row(row, i, gqa, total_call_packed_64, use_logical_tail,
                        range_base_64, range_total_64)) {
                out.dense_rows[DENSE_BUCKET_UNKNOWN] =
                    sat_add(out.dense_rows[DENSE_BUCKET_UNKNOWN], (uint64_t) gqa);
                continue;
            }
            const uint32_t vbucket = verdict_unsup ? DENSE_BUCKET_UNSUPPORTED : DENSE_BUCKET_HIGHCOST;
            const uint32_t range_total = (uint32_t) range_total_64;
            for (uint32_t s = 0; s < gqa; ++s) {
                const uint32_t packed = (uint32_t) (range_base_64 + (uint64_t) s);
                if (llama_flashprefill::packed_row_in_dense_tail(
                            packed, range_total, cfg->block_q, cfg->dense_tail_tiles)) {
                    out.dense_rows[DENSE_BUCKET_TAIL] =
                        sat_add(out.dense_rows[DENSE_BUCKET_TAIL], 1);
                } else {
                    out.dense_rows[vbucket] = sat_add(out.dense_rows[vbucket], 1);
                }
            }
            continue;
        }
        // Residual: role-deny over eligible rows, NO_LAYOUT, or unrecognized
        // verdicts — dense without a plan and without a classified reason.
        // Counted in the bucket for observability only: no plan existed, so
        // there is nothing to invalidate, and no invalidation counter fires
        // for verdict-driven dense (SHORT/TAIL/designed-dense included).
        out.dense_rows[DENSE_BUCKET_NO_PLAN] = sat_add(out.dense_rows[DENSE_BUCKET_NO_PLAN], (uint64_t) gqa);
    }

    // Pool accounting (first-version pools rebuild per sparse slice,
    // PREFILL.md section 11): exactly one rebuild per slice that ran new-path
    // packed rows through plans. No pool without plans (nothing built).
    if (plans_present && any_new_path_packed) {
        out.pool_rebuild[any_exact_all ? POOL_REASON_EXACT_ALL : POOL_REASON_SLICE] = 1;
        out.scratch_live_bytes = scratch_live_bytes;
        out.scratch_peak_bytes = scratch_peak_bytes >= scratch_live_bytes
            ? scratch_peak_bytes : scratch_live_bytes;
    }

    if (has_layout_us) {
        out.layout_us = layout_us;
        out.has_layout_us = true;
    }

    return 0;
}

} // namespace llama_flashprefill_metrics

// The public C snapshot (llama.h) carries fixed bounds; the internal ledger
// must match them exactly. Any bucket/reason extension updates both sides.
static_assert(llama_flashprefill_metrics::DENSE_BUCKET_COUNT == 10, "dense buckets vs llama.h");
static_assert(llama_flashprefill_metrics::POOL_REASON_COUNT == 2, "pool reasons vs llama.h");
static_assert(llama_flashprefill_metrics::PLAN_REASON_COUNT == 3, "plan reasons vs llama.h");

// Public drain (declared in llama.h by ConfigIntegration; defined here so no
// src/llama-context.{h,cpp} edit is needed for the server handoff). Uses only
// the public consume() ledger: transactional drain-once semantics live in
// llama_context and are reused verbatim, never reimplemented.
int32_t llama_flashprefill_metrics_drain(struct llama_context * ctx, llama_flashprefill_metrics_slice * out) {
    if (ctx == nullptr || out == nullptr) {
        return -1;
    }
    llama_flashprefill_metrics_slice zero = {};
    *out = zero;
    llama_flashprefill_metrics::slice_delta d;
    if (!ctx->flashprefill_metrics_consume(d)) {
        return 1;
    }
    out->eligible_rows      = d.eligible_rows;
    out->sparse_rows        = d.sparse_rows;
    out->dense_packed       = d.dense_packed;
    for (uint32_t i = 0; i < 10; ++i) {
        out->dense_rows[i] = d.dense_rows[i];
    }
    out->selected_blocks    = d.selected_blocks;
    out->corrected_blocks   = d.corrected_blocks;
    out->visible_tokens     = d.visible_tokens;
    out->exact_tokens       = d.exact_tokens;
    for (uint32_t i = 0; i < 2; ++i) {
        out->pool_rebuild[i] = d.pool_rebuild[i];
    }
    for (uint32_t i = 0; i < 3; ++i) {
        out->plan_invalidations[i] = d.plan_invalidations[i];
    }
    out->scratch_live_bytes = d.scratch_live_bytes;
    out->scratch_peak_bytes = d.scratch_peak_bytes;
    out->layout_us          = d.layout_us;
    out->has_layout_us      = d.has_layout_us ? (uint8_t) 1 : (uint8_t) 0;
    out->gpu_pool_us        = d.gpu_pool_us;
    out->gpu_select_us      = d.gpu_select_us;
    out->gpu_attn_us        = d.gpu_attn_us;
    out->has_gpu_us         = d.has_gpu_us ? (uint8_t) 1 : (uint8_t) 0;
    return 0;
}
