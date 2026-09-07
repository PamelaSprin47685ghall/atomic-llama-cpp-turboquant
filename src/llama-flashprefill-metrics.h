// llama-flashprefill-metrics.h — bounded FlashPrefill V2 metrics aggregator.
//
// Owner: MetricsIntegration. NEW module (no other worker owns this file).
//
// What this is: a pure, allocation-free aggregator that turns per-slice
// observations into transactional deltas. Every counter is backed by a real
// measurement taken on the slice it counts:
//
//   - eligible/sparse/dense/selected/corrected/visible/exact come from the
//     owned per-ubatch row snapshot (PolicyCore roles/routes, reused — never
//     redefined) plus the GGML plan headers (words 11..14 and 16..19) read
//     AFTER the existing graph completion boundary (VulkanDispatch contract:
//     ggml_backend_vk_graph_compute status only, no per-tensor status, no
//     extra/per-layer waits).
//   - Error propagation (CpuKernels contract): SELECT carries the plan-header
//     error word (plan[10]); ATTN may set it atomically. The end-of-call fold
//     parses EVERY queued layer header (parse_plan_header_stats, header-only)
//     and fails the whole call on the first nonzero error BEFORE any counter
//     is trusted or the ATTN output is consumed downstream. POOL/ATTN have no
//     other error channel and GGML_ABORT on corruption, so there is no
//     success to count on those paths either.
//   - Transactional: merge() applies a slice delta with saturating addition
//     exactly once per SUCCESSFUL slice. Retry / partial-failure / cancel /
//     narrowed-batch paths never call merge(), so failed rows are never
//     double counted (PREFILL.md section 10.4: stats valid only after the
//     corresponding graph succeeds).
//   - OFF: build_slice() returns an empty delta without touching the rows
//     pointer, allocating nothing, arming no timers,queueing no reads.
//
// Units (do not mix denominators):
//   - eligible_rows: SOURCE rows presented with a sparse-eligible role and a
//     known boundary (same definition as the server's fp_eligible_rows).
//   - sparse_rows / dense_packed: PACKED (source_query, q_head) execution
//     rows. sparse_rows is plan-confirmed GPU work (WireReference: a packed
//     row is sparse iff >= 1 matched use is proxy-listed or missing from
//     both plan lists). dense_packed is the plan's dense packed count.
//   - dense_by_bucket[]: PACKED rows that executed dense. Role and
//     known-boundary buckets come from exact row data; every other verdict
//     is the graph's own per-ubatch outcome (SHORT/UNSUPPORTED/CAPACITY and
//     friends file literally; TAIL refines by frozen position where the
//     verdict leaves room). Nothing is inferred from global KV fullness or
//     guessed backend support. Verdict 0 (designed-dense, e.g.
//     full-attention layer policy) files no per-row bucket; unclassifiable
//     residuals file NO_PLAN (dense without plan or classified reason).
//   - selected_blocks: plan exact USE counts (header 13).
//   - corrected_blocks: EXECUTED proxy corrections (header 14) — 0 under the
//     mean_correction=false ablation, where proxy lists still classify but
//     ATTN deliberately skips them. Sparse/exact/visible still report the
//     skipped work; only the correction claim is gated.
//   - visible/exact_tokens: plan use-record token sums (header 16..19, 64-bit;
//     each use-record counted once, subhead fan-out NOT multiplied).
//   - Ratios must keep one unit: sparse/(sparse+dense_packed), or
//     exact_tokens/visible_tokens. Never sparse/source-rows over capacity.
//
// Timings (never fake zeros as elapsed internally): host layout time comes
// from chrono around the ubatch slice build (layout_us + has_layout_us).
// GPU pool/select/attention times come from the VulkanDispatch per-backend
// timestamp proc, sampled once per call after the single end-sync and
// differenced against call-start baselines (no extra waits); has_gpu_us
// marks sampled slices. Internally unmeasured stays absent; the serializer
// always emits the series (zeros until reported) per the matrix contract.
//
// Scratch: live/peak bytes CONSUMED, never estimated and never maxima.
// Per executed span, live/peak take the build's own total from the graph
// summary (shared metadata once + pinned plans + max pool — real tensor
// capacities via ggml_nbytes, checked u64). The backend split workspace
// adds the exact reported bytes (0 while the split path is inactive) from
// the VulkanDispatch per-backend proc, sampled post-sync with start
// baselines. Live overwrites per span (current build residency) and adds
// split-current once per call; peak maxes, so repeated ubatches on one
// build never double count. Merged only on success. Zero while no sparse
// tensors were built is truthful (nothing allocated).
//
// Async lifetime / per-slice merge interface (submit is not completion:
// process_ubatch returns after an ASYNC submit, and the decode-end
// synchronize is commented out, so nothing GPU-side is readable until the
// boundary below):
//   1. Per ubatch, after a successful submit, the host stages the row-side
//      delta via build_slice() (host-owned rows: eligible/dense buckets/
//      layout time; no wait) with plans_deferred set iff this ubatch queued
//      plan reads. Rationale (shader contract): pool NaN-poison surfaces as
//      SELECT plan status and ATTN may set the plan error word atomically,
//      so ALL layers' plan error words must be inspected after FULL-graph
//      completion before the ATTN output (or any downstream success metric)
//      is trusted.
//   2. Per ubatch with plan nodes (GraphIntegration supplier,
//      output-marked for lifetime), the host queues one
//      ggml_backend_tensor_get_async() per layer plan header (24 x I32 =
//      96 contiguous bytes, next to the existing logits/output reads) into
//      owned per-call snapshots — no wait at queue time, never a per-layer
//      wait, no whole-plan readback. Ubutches without plans stage rows only.
//   3. After ALL ubatches, iff any reads were queued, the host synchronizes
//      ONCE for the call (this single end-sync is shared with the would-be
//      getter wait; OFF and dense-only calls skip it entirely), parses every
//      header with parse_plan_header_stats, and
//      on any error fails the call (-3 + hygiene) with nothing committed.
//      Clean folds merge plan totals + pool + actual scratch, sample GPU
//      phase times (VulkanDispatch hook, still no extra waits), and commit
//      the staged deltas exactly once.
//   4. The server drains the host accum once per successful decode slice
//      (consume: delta = accum - watermark) and merges into its own accum
//      with the same once-per-slice discipline as fp_counters.
//
// Prometheus (serializer lives in tools/server/server-context.cpp
// get_metrics; result struct in tools/server/server-task.h/.cpp — both
// MetricsIntegration-owned): every series uses the existing `llamacpp:`
// prefix. Labels are bounded (dense reason <= 10 values, pool reason 2,
// plan reason 3, fingerprint single-series info). No sequence ids, prompt
// text, or unbounded reader ids in labels. Series are always emitted (zeros
// while OFF at no counting cost); only the fingerprint info is gated.

#pragma once

#include "llama-flashprefill.h"
#include "ggml-flashprefill.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

namespace llama_flashprefill_metrics {

// ---------------------------------------------------------------------------
// Bounded reason domains (fixed cardinality; do not extend without updating
// the serializer label tables and this comment).
// ---------------------------------------------------------------------------

// Dense-bucket index into slice_delta::dense_rows / accum::dense_rows.
enum dense_bucket : uint32_t {
    DENSE_BUCKET_DECODE = 0,        // DENSE_ROLE with a DECODE row role
    DENSE_BUCKET_MTP_VERIFY = 1,    // DENSE_ROLE with MTP_VERIFY/MTP_DRAFT (draft rides the verify path)
    DENSE_BUCKET_ROLE_OTHER = 2,    // DENSE_ROLE with any other non-eligible role
                                    // (frontier/replay/multimodal/unknown/embedding/rerank)
    DENSE_BUCKET_SHORT = 3,         // DENSE_SHORT_CONTEXT (resident-visible < min_kv, real count)
    DENSE_BUCKET_TAIL = 4,          // DENSE_TAIL (all packed subheads in the dense tail tiles)
    DENSE_BUCKET_UNKNOWN = 5,       // DENSE_UNKNOWN_BOUNDARY (eligible role, no known interval/pos)
    DENSE_BUCKET_UNSUPPORTED = 6,   // DENSE_UNSUPPORTED (backend capability refused;
                                    // observed only when the caller passes backend_supported=false)
    DENSE_BUCKET_HIGHCOST = 7,      // DENSE_CAPACITY (plan capacity gate refused;
                                    // observed only via the capacity-gate producer)
    DENSE_BUCKET_NO_PLAN = 8,       // slice presented sparse candidates but no plan nodes
                                    // existed, so the graph ran dense (observed fallback)
    DENSE_BUCKET_FULL_PREFIX = 9,   // designed-dense full-prefix/SWA layers: actual
                                    // query-head-layer incidences from the graph
                                    // summary (authoritative, zero-plan included)
    DENSE_BUCKET_COUNT = 10,
};

// Pool-rebuild reason index into slice_delta::pool_rebuild / accum::pool_rebuild.
// First-version pools are rebuilt per sparse slice (PREFILL.md section 11);
// both reasons are observed in the hook, never assumed.
enum pool_reason : uint32_t {
    POOL_REASON_SLICE = 0,          // successful sparse slice rebuilt its pool means
    POOL_REASON_EXACT_ALL = 1,      // same, under the exact_all debug gate (plan echo)
    POOL_REASON_COUNT = 2,
};

// Plan-invalidation reason index into slice_delta::plan_invalidations /
// accum::plan_invalidations. All observed in the hook.
enum plan_reason : uint32_t {
    PLAN_REASON_BYPASSED = 0,       // call bypassed (MTP context / embedding-rerank pooling)
    PLAN_REASON_EMPTY_SNAPSHOT = 1, // enabled but no owned ubatch rows (route dense)
    PLAN_REASON_NO_PLAN = 2,        // reserved: every plan-less outcome now
                                    // carries an authoritative verdict, so no
                                    // event fires this (frozen ABI slot)
    PLAN_REASON_COUNT = 3,
};

const char * dense_bucket_name(uint32_t bucket); // Prometheus label value; "?" if out of range
const char * pool_reason_name(uint32_t reason);  // same
const char * plan_reason_name(uint32_t reason);  // same

// Checked header-only plan-stats parser over a 24-word host snapshot
// (frozen v1 layout, shared ggml-flashprefill.h constants only — no wire
// list read, no full-tensor readback). Validates magic/version/header-words,
// canonical offset arithmetic (exact/proxy/counts/total from
// n_tiles/n_kv_heads/max_sel_pair, checked 64-bit), total_words against the
// caller-supplied expected tensor length, and nonnegativity of every counter.
// The plan error word rides through in out->error (never a parse failure by
// itself — the caller fails the slice on nonzero error before trusting any
// counter or the ATTN output). Returns GGML_FLASHPREFILL_OK with *out filled,
// or a GGML_FLASHPREFILL_ERR_* code with *out cleared (fail closed; the
// caller must fail the slice, never silently dense).
int32_t parse_plan_header_stats(
        const int32_t hdr24[24],
        uint64_t expected_ne_words,
        struct ggml_flashprefill_plan_stats * out);

// Map a (frozen route, row role) pair to a dense bucket. Returns
// DENSE_BUCKET_COUNT when the route is not dense (SPARSE/EXACT_ALL) or the
// inputs are out of range — the caller must not count those as dense.
uint32_t classify_dense_bucket(int32_t route, int32_t role);

// ---------------------------------------------------------------------------
// Per-slice delta (stack-owned, memcpy-able) and cumulative accumulator.
// Zero-initialized == empty. Counters use saturating addition on merge.
// ---------------------------------------------------------------------------

struct slice_delta {
    uint64_t eligible_rows = 0;                 // source rows, eligible role + known boundary
    uint64_t sparse_rows = 0;                  // packed rows, plan-confirmed sparse GPU work
    uint64_t dense_packed = 0;                 // packed rows, plan dense (packed-unit denominator)
    uint64_t dense_rows[DENSE_BUCKET_COUNT] = {}; // packed rows by presentation reason
    uint64_t selected_blocks = 0;              // plan exact uses (header 13)
    uint64_t corrected_blocks = 0;             // plan proxy uses (header 14)
    uint64_t visible_tokens = 0;               // plan use-record sum (header 16/17, 64-bit)
    uint64_t exact_tokens = 0;                 // plan exact-use sum (header 18/19, 64-bit)
    uint64_t pool_rebuild[POOL_REASON_COUNT] = {};
    uint64_t plan_invalidations[PLAN_REASON_COUNT] = {};
    uint64_t scratch_live_bytes = 0;           // actual slice sizing (0 == none built)
    uint64_t scratch_peak_bytes = 0;           // max live observed this slice
    uint64_t layout_us = 0;                    // measured host layout time
    bool     has_layout_us = false;            // false == unmeasured (omit, never fake zero)
    uint64_t gpu_pool_us = 0;                  // GPU pool-dispatch time (post-completion sampled)
    uint64_t gpu_select_us = 0;                // GPU select-dispatch time
    uint64_t gpu_attn_us = 0;                  // GPU attention-dispatch time
    bool     has_gpu_us = false;               // false == sampler reported nothing this slice
};

struct accum {
    uint64_t eligible_rows = 0;
    uint64_t sparse_rows = 0;
    uint64_t dense_packed = 0;
    uint64_t dense_rows[DENSE_BUCKET_COUNT] = {};
    uint64_t selected_blocks = 0;
    uint64_t corrected_blocks = 0;
    uint64_t visible_tokens = 0;
    uint64_t exact_tokens = 0;
    uint64_t pool_rebuild[POOL_REASON_COUNT] = {};
    uint64_t plan_invalidations[PLAN_REASON_COUNT] = {};
    uint64_t scratch_live_bytes = 0;           // last successful slice (gauge: overwrite)
    uint64_t scratch_peak_bytes = 0;           // max over slices (gauge: max)
    uint64_t layout_us_total = 0;              // sum over measured slices (counter)
    uint64_t layout_slices_measured = 0;       // slices contributing to layout_us_total
    uint64_t gpu_pool_us_total = 0;            // sum over sampled calls (counter)
    uint64_t gpu_select_us_total = 0;
    uint64_t gpu_attn_us_total = 0;
    uint64_t gpu_slices_measured = 0;          // calls contributing GPU times
    uint64_t policy_fingerprint = 0;           // set once from the live context
    bool     has_policy = false;

    bool empty() const;
};

// Optional GPU phase sampler (VulkanDispatch producer). Wire name (backend-
// reg proc table, resolved — never linked — by the context hook):
//   ggml_status ggml_backend_vk_flashprefill_times(
//       ggml_backend_t backend, uint64_t * pool_ns, uint64_t * select_ns,
//       uint64_t * attn_ns)
// Per-backend cumulative nanoseconds since that backend's init (attn
// includes merge), advanced post-fence per graph; nullable out-args; FAILED
// iff not a Vulkan backend (else SUCCESS, zeros when a class is absent).
// The hook samples every scheduler backend once per decode call after the
// single end-sync and attributes sample-minus-last per backend (unknown
// backends stamp without contributing; backward steps contribute zero).
// Zero extra waits; absence is a clean skip (no link risk).
using gpu_phase_sampler_fn =
    ggml_status (*)(ggml_backend_t backend, uint64_t * pool_ns, uint64_t * select_ns, uint64_t * attn_ns);

bool slice_is_empty(const slice_delta & d);

// Same saturating/gauge rules as accum_merge, but delta-to-delta: used for
// per-call staging (stage each successful ubatch, commit once on call
// success, discard on call failure). Fingerprint fields do not exist on
// deltas and are handled at commit time.
void delta_merge(slice_delta & acc, const slice_delta & d);

// Saturating merge. Call EXACTLY ONCE per committed call (the host stages
// ubatch deltas with delta_merge and commits once on call success).
// scratch_live overwrites (reused buffers), scratch_peak takes the max,
// layout time sums only when the slice measured it. Fingerprint is set-once
// (policy is immutable for the context lifetime; mismatched fingerprints are
// a caller bug and keep the first).
void accum_merge(accum & a, const slice_delta & d);

// Consume-ledger helper for the context->server handoff: delta = acc - mark;
// mark = acc. Exact until u64 saturation (documented; saturation sticks and
// the delta stays exact at the stick point).
slice_delta accum_delta_since(const accum & acc, const accum & mark);

// ---------------------------------------------------------------------------
// Slice builder (pure except for reading its inputs; no allocation, no sync,
// no waits — the caller guarantees GPU completion before passing plans).
//
// The builder NEVER infers routing: every dense verdict comes from the
// authoritative per-ubatch graph outcome (graph_dense_reason, recorded for
// every ubatch including zero-plan ones) or from exact row data (role,
// known-boundary, frozen positions). In particular it never reads global KV
// fullness to infer SHORT and never guesses backend support: SHORT,
// UNSUPPORTED, CAPACITY and friends arrive as the graph's own verdict.
// Required-mode enforcement lives graph-side (it throws); metrics only
// counts, never fails for routing.
//
// Returns 0 and fills `out` (possibly empty for legitimate dense: OFF,
// bypassed, empty snapshot, designed-dense rows). Returns the FIRST nonzero
// plan error (a GGML_FLASHPREFILL_* code, or GGML_FLASHPREFILL_ERR_BAD_ARG
// for a corrupt stats field) with `out` cleared: the caller must fail the
// slice with no successful output and merge nothing.
//
// Contract on inputs:
//   cfg: frozen policy (may be OFF; null == OFF, empty delta, no row touch).
//   rows/n_rows: owned ubatch snapshot (may be null iff n_rows == 0).
//   graph_dense_reason: authoritative ubatch verdict from the graph summary
//     (LLAMA_FLASHPREFILL_ROUTE_* for whole-ubatch dense, LLM_FP_DENSE_* for
//     graph-internal bounds, 0 == sparse-active or designed-dense). Consulted
//     only when this ubatch built no plans; ignored on the plans path.
//   gqa: query heads per KV head for this slice (>= 1; packed-row fan-out
//     from frozen model dims).
//   bypassed: MTP-context / embedding-rerank-pool bypass (no per-row arrays).
//   plans/n_plans + plans_present: per-layer stats (parsed headers). Direct
//     callers pass parsed stats; the context hook always passes
//     plans_present=false and lets the end-of-call fold supply plan totals
//     from its queued snapshots instead.
//   scratch_live/peak_bytes: actual slice sizing (direct callers; the hook
//     leaves these zero and lets the fold consume the graph byte totals).
//   layout_us/has_layout_us: measured host layout time for this slice.
//   plans_deferred: reads queued for this ubatch (parsed at call end).
//     Provisional rows skip the verdict here for the fold to resolve.
// Per-row rules (packed units, gqa fan-out): non-eligible roles file role
// buckets and eligible-but-unknown rows file UNKNOWN from exact row data on
// every path. Eligible-known rows on the plans path file TAIL by frozen
// position (the rest is GPU-confirmed). On a plan-less ubatch they file the
// authoritative verdict (SHORT/UNKNOWN/UNSUPPORTED/CAPACITY/HIGHCOST/TAIL
// literally; TAIL-checked first where positions decide); designed-dense
// verdicts (0 sparse-active, 102 full-prefix) file nothing, and
// unclassifiable residuals file NO_PLAN rows without raising any
// invalidation (no plan existed to invalidate).
// ---------------------------------------------------------------------------

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
        bool plans_deferred);

} // namespace llama_flashprefill_metrics
