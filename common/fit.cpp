#include "fit.h"

#include "log.h"

#include "../src/llama-ext.h"
#include "../src/llama-flashprefill.h"
#include "../src/llama-flashprefill-layout.h"
#include "../src/llama-model.h"
#include "../src/llama-xkv-factor.h"
#include "../src/llama-xkv-codec.h"

#include <array>
#include <cassert>
#include <cmath>
#include <climits>
#include <cstdio>
#include <dirent.h>
#include <stdexcept>
#include <cinttypes>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

// this enum is only used in llama_params_fit_impl but needs to be defined outside of it to fix a Windows compilation issue
// enum to identify part of a layer for distributing its tensors:
enum common_layer_fraction_t {
    LAYER_FRACTION_NONE = 0, // nothing
    LAYER_FRACTION_ATTN = 1, // attention
    LAYER_FRACTION_UP   = 2, // attention + up
    LAYER_FRACTION_GATE = 3, // attention + up + gate
    LAYER_FRACTION_MOE  = 4, // everything but sparse MoE weights
};

class common_params_fit_exception : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// --- FlashPrefill V2 fit budgeting (FittingIntegration owner) ------------------------
//
// Resource contract (PREFILL.md §11; coordinated with PolicyCore, VulkanDispatch,
// GraphIntegration, CacheFragments, ConfigIntegration, ContextIntegration):
// - Single sizing helper: llama_flashprefill::scratch_bytes_checked (PolicyCore).
//   Fit and runtime call the same function; the formula is never duplicated here.
//   Per-layer peak, reused across layers: single-layer F/Hkv/Dk/Dv, do NOT multiply
//   by n_layer. Worst-case pool precision is F32 (mean_bytes = 4) until the graph
//   proves F16 (PolicyCore).
// - Pool/plan/outputs are graph tensors (VulkanDispatch, confirmed: SELECT scoring
//   is on-chip plus graph tensors, zero extra workspace). Split M/L/O scratch lives
//   in the backend-private prealloc_split_k vk_buffer pool (code proof beside
//   common_fp_native_split_bytes): lazily grown at dispatch, invisible to scheduler
//   sizes and probes. The graph reserve sizes worst-case sparse caps from probe
//   inputs via the same shared helper when llm_graph_params.flashprefill_reserve_sizing
//   is set (GraphIntegration; reserve graphs never alias live via the reuse key). The
//   context factory scopes the flag via RAII in graph_reserve (StatePolicy, landed):
//   true for synthetic builds incl. fit probes, false for live decode/training. The
//   probe's compute thus holds every graph-owned byte and no manual reserve applies
//   to those (no double charge); coverage proof is reserve tensors present in probe
//   compute, never the helper log line alone. Meanwhile the
//   split-pool peak is added here exactly once per Vulkan device
//   with the dispatch formula (single source until the agreed size hook lands). The
//   fitter publishes no live rows (source-map channel stays null; unknown live
//   boundary stays unknown) — reserve-snapshot synthetic rows are GraphIntegration's
//   sanctioned channel, not fit data.
// - Fragment bound via CacheFragments' checked fragment_budget_for() (never
//   ceil(K/BN), never clamped) with the frozen admission allowances
//   (GraphIntegration): streams=1, split=4, run=65, q_cap token rows, for both
//   paths; Fcap capped at K inside the helper (structural token invariant). These
//   allowances are a DEVELOPMENT unbenchmarked guard — not a measured cutoff —
//   pinned to the layout/schema version (bump version to change; fingerprint covers
//   the version, hence the fixed policy). Single definition in
//   llama-flashprefill-layout.h (admission_budget_for, landed): graph+fitter call
//   the same function, no private copies here. Beyond-bound rows fall back to dense
//   (AUTO) or resource-error (REQUIRED) at runtime — never truncated. Helper-guard
//   breach reports unavailable (fail closed, never saturation, never silent fallback).
// - OFF (or invalid) config -> no extra probes, no capacity change, no logs.
// - Recurrent/MTP decode rows are never sparse-eligible: common_fit_recurrent_cache
//   budgets zero FlashPrefill bytes by design (see its note below).

static bool common_fp_is_enabled(const llama_context_params * cparams) {
    return cparams != nullptr && llama_flashprefill_is_enabled(&cparams->flashprefill);
}

struct common_fp_dims {
    uint32_t hkv = 0;
    uint32_t dk  = 0;
    uint32_t dv  = 0;
    uint32_t gqa = 0;
    uint32_t hq  = 0; // upper bound on Q heads (gqa*hqkv; exact for uniform models)
};

// Backend doubt resolved by code inspection, not assertion
// (ggml/src/ggml-vulkan/ggml-vulkan.cpp): prealloc_split_k is a backend-private
// vk_buffer on ggml_backend_vk_context, grown lazily at dispatch via
// ggml_vk_preallocate_buffers (indexed-FA site sizes
//   split_k_size = S>1 ? n_flat*(HSV+2)*S*4 : 0, n_flat = n_queries*n_head_q).
// Raw VkDeviceMemory is invisible to ggml_backend_sched_get_buffer_size, hence to
// probes — same class as the Tri memmove_scratch +8MiB precedent. The pool is shared
// across layers/ops (grown to max), so the fitter adds the worst-case peak exactly
// once per Vulkan device here. Final contract (VulkanDispatch): rows_bound is the
// structural n_output*Hq cap (ATTN dst ne[2]*ne[1], zero U slack); n_ubatch*Hq_max
// covers it definitionally and is adopted as final — rb_alloc dropped. S<=4 hard;
// S=1 default allocates zero (kept as worst case here).
static bool common_fp_is_vulkan_dev(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char * name = reg ? ggml_backend_reg_name(reg) : nullptr;
    return name != nullptr && std::string(name).find("Vulkan") != std::string::npos;
}

static bool common_fp_native_split_bytes(
        const llama_flashprefill_config * cfg,
        const common_fp_dims * dims,
        uint32_t n_ubatch,
        uint32_t n_ctx_kv,
        ggml_backend_dev_t dev,
        uint64_t & out_bytes) {
    out_bytes = 0;
    if (cfg == nullptr || dims == nullptr) {
        return true; // unknown dims (e.g. recurrent-only): no attention, exact zero
    }
    if (!common_fp_is_vulkan_dev(dev)) {
        return true; // prealloc pools are Vulkan-backend-private
    }
    if (dims->hq == 0 || dims->hq > 4096 || n_ubatch > (1u << 24)) {
        return false;
    }
    // U/F from the shared admission budget (streams=1 unified, q_cap token rows).
    llama_flashprefill_fragment_budget budget;
    if (!llama_flashprefill_admission_budget_for(
            n_ctx_kv, cfg->block_k, LLAMA_FLASHPREFILL_ADMISSION_STREAMS_UNIFIED, n_ubatch, &budget, nullptr)) {
        return false;
    }
    // S_max=4 hard in dispatch (S=1 default allocates zero); worst case takes 4.
    // n_flat bound confirmed sound (VulkanDispatch): n_queries are TOKEN rows
    // (PackGQA/BM packing is SELECT-internal), unique on (source_query,q_head) with
    // tiles partitioning the query set, so rows <= n_output*Hq <= n_ubatch*Hq_max
    // (final contract: rows_bound is the structural n_output*Hq cap, rb_alloc dropped).
    const uint32_t s = budget.n_fragments == 0 ? 1u : std::min(4u, budget.n_fragments);
    if (s <= 1) {
        return true;
    }
    // dims.dv is Turbo-padded, matching the HSV width; terms fit u64 by the guards.
    out_bytes = (uint64_t) n_ubatch * (uint64_t) dims->hq
        * ((uint64_t) dims->dv + 2ull) * (uint64_t) s * 4ull;
    return true;
}

// Log-only budget number: never added to probe sums (graph counts the caps).
static bool common_fp_budget_bytes(
        const llama_flashprefill_config * cfg,
        const common_fp_dims * dims,
        uint32_t n_ubatch,
        uint32_t n_ctx_kv,
        uint64_t & out_bytes,
        uint32_t & out_fragments) {
    out_bytes = 0;
    if (cfg == nullptr || dims == nullptr || !llama_flashprefill_is_enabled(cfg)) {
        return false;
    }
    if (cfg->block_k == 0 || cfg->block_q == 0) {
        return false;
    }
    uint32_t n_packed = 0;
    uint32_t n_tiles  = 0;
    if (llama_flashprefill::packed_layout_checked(n_ubatch, dims->gqa, cfg->block_q, &n_packed, &n_tiles)
            != LLAMA_FLASHPREFILL_OK) {
        return false;
    }
    if (n_ctx_kv == 0 || n_ubatch == 0) {
        return false;
    }
    // F via the CacheFragments checked budget (never ceil(K/BN), never clamped) with
    // the frozen admission allowances (GraphIntegration): streams=1, split=4, run=65,
    // q_cap token rows (n_ubatch). I32 breach or a helper-guard breach reports
    // unavailable (fail closed).
    out_fragments = 0;
    llama_flashprefill_fragment_budget budget;
    if (!llama_flashprefill_admission_budget_for(
            n_ctx_kv, cfg->block_k, LLAMA_FLASHPREFILL_ADMISSION_STREAMS_UNIFIED, n_ubatch, &budget, nullptr)) {
        return false;
    }
    // Shared helper already caps Fcap at K (structural token invariant: pool sized on
    // actual admitted F); enforce the scratch helper's fragment guard here (fail closed).
    if (budget.n_fragments == 0 || budget.n_fragments > (1u << 24)) {
        return false;
    }
    out_fragments = budget.n_fragments;
    llama_flashprefill::scratch_inputs in{};
    in.n_fragments   = budget.n_fragments;
    in.n_kv_heads    = dims->hkv;
    in.d_k           = dims->dk;
    in.d_v           = dims->dv;
    in.n_packed_rows = n_packed;
    in.n_splits      = 1; // default path (S=1, no scratch); bounded-S<=4 split triples reuse the observed prealloc_split_k pool (VulkanDispatch)
    in.mean_bytes    = 4; // F32 pool worst case (PolicyCore)
    return llama_flashprefill::scratch_bytes_checked(cfg, &in, &out_bytes) == LLAMA_FLASHPREFILL_OK;
}

static bool common_fp_dims_from_model(const llama_model * model, const llama_context_params * cparams, common_fp_dims & out) {
    if (model == nullptr || cparams == nullptr) {
        return false;
    }
    // Max-over-KV-layers getters (ConfigIntegration, include/llama.h).
    const int32_t hkv = llama_model_n_head_kv_max(model);
    const int32_t dk  = llama_model_n_embd_head_k(model);
    const int32_t dv  = llama_model_n_embd_head_v(model);
    const int32_t gqa = llama_model_n_gqa_max(model);
    if (hkv <= 0 || dk <= 0 || dv <= 0 || gqa <= 0) {
        return false; // recurrent-only or unknown layout: no sparse pool to predict
    }
    out.hkv = (uint32_t) hkv;
    out.dk  = (uint32_t) dk;
    out.dv  = (uint32_t) dv;
    out.gqa = (uint32_t) gqa;
    // Upper bound on Q heads per layer (exact for uniform models; the two maxima may
    // come from different hybrid layers, which only over-states worst-case rows).
    const uint64_t hq64 = (uint64_t) (uint32_t) hkv * (uint64_t) (uint32_t) gqa;
    if (hq64 == 0 || hq64 > (uint64_t) UINT32_MAX) {
        return false;
    }
    out.hq = (uint32_t) hq64;
    // TurboQuant KV caches zero-pad heads to multiples of 128 (src/llama-kv-cache.cpp).
    // Pad conservatively (MLA has no separate V cache, so this over-counts <= 127
    // elements there; negligible and on the safe side for a worst-case bound).
    auto is_turbo = [](ggml_type t) {
        return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
    };
    if (is_turbo(cparams->type_k) && out.dk % 128 != 0) {
        out.dk = ((out.dk + 127) / 128) * 128;
    }
    if (is_turbo(cparams->type_v) && out.dv % 128 != 0) {
        out.dv = ((out.dv + 127) / 128) * 128;
    }
    return true;
}

// Resolve worst-case layer dims for the budget log via one metadata-only model load.
// Called only when FlashPrefill is enabled (OFF adds zero work here).
static bool common_fp_load_dims(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        common_fp_dims & out) {
    if (!common_fp_is_enabled(cparams)) {
        return false;
    }
    llama_model_params meta_mparams = *mparams;
    meta_mparams.no_alloc  = true;
    meta_mparams.load_mode = LLAMA_LOAD_MODE_NONE;
    llama_model * meta = llama_model_load_from_file(path_model, meta_mparams);
    if (meta == nullptr) {
        return false;
    }
    const bool ok = common_fp_dims_from_model(meta, cparams, out);
    llama_model_free(meta);
    return ok;
}

// Enabled-only fit summary: fitted capacity plus the single helper-sized worst-case
// eligible scratch number with the real shapes used. OFF emits nothing.
static void common_fp_log_fit_result(
        const char * what,
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        uint32_t n_ctx_kv) {
    if (!common_fp_is_enabled(cparams)) {
        return;
    }
    constexpr uint64_t KiB = 1024ull;
    common_fp_dims dims;
    if (!common_fp_load_dims(path_model, mparams, cparams, dims)) {
        LOG_INF("%s: FlashPrefill enabled, fitted %s K=%u tokens; budget bytes unavailable (dims unknown)\n",
            __func__, what, n_ctx_kv);
        return;
    }
    // Same single source the graph reserve sizes with (common_fp_budget_bytes).
    uint64_t bytes = 0;
    uint32_t frags = 0;
    if (!common_fp_budget_bytes(&cparams->flashprefill, &dims, cparams->n_ubatch, n_ctx_kv, bytes, frags)) {
        LOG_INF("%s: FlashPrefill enabled, fitted %s K=%u tokens; budget bytes unavailable (shape exceeds checked guards)\n",
            __func__, what, n_ctx_kv);
        return;
    }
    LOG_INF("%s: FlashPrefill enabled, fitted %s K=%u tokens (Hkv=%u Dk=%u Dv=%u GQA=%u ubatch=%u BN=%u F=%u); worst-case eligible scratch peak=%llu KiB\n",
        __func__, what, n_ctx_kv, dims.hkv, dims.dk, dims.dv, dims.gqa, cparams->n_ubatch, cparams->flashprefill.block_k, frags,
        (unsigned long long) (bytes / KiB));
}



static bool xkv_checked_mul_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (a == 0 || b == 0) {
        out = 0;
        return true;
    }
    if (a > UINT64_MAX / b) {
        return false;
    }
    out = a * b;
    return true;
}

static bool xkv_checked_add_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (UINT64_MAX - a < b) {
        return false;
    }
    out = a + b;
    return true;
}

static bool xkv_checked_sum_fit_bytes(uint64_t model, uint64_t context, uint64_t compute,
        uint64_t reserved, uint64_t headroom, uint64_t & out) {
    uint64_t sum = 0;
    return xkv_checked_add_u64(model, context, sum) &&
           xkv_checked_add_u64(sum, compute, sum) &&
           xkv_checked_add_u64(sum, reserved, sum) &&
           xkv_checked_add_u64(sum, headroom, out);
}

bool common_xkv_fit_reserve_bytes(const llama_context_params * cparams, uint64_t scratch_bytes,
        common_xkv_fit_reserve * out, uint64_t dedup_scratch_bytes) {
    if (out == nullptr) {
        return false;
    }
    *out = {};
    if (cparams == nullptr || !llama_xkv_is_enabled(cparams->xkv_mode)) {
        return true;
    }
    constexpr uint64_t MiB = 1024ull * 1024ull;
    uint64_t workspace = 0;
    uint64_t decode_cache = 0;
    // Any overflow fails closed: the caller treats it as "does not fit".
    if (!xkv_checked_mul_u64((uint64_t) cparams->xkv_workspace_mib, MiB, workspace) ||
        !xkv_checked_mul_u64((uint64_t) cparams->xkv_decode_cache_mib, MiB, decode_cache)) {
        *out = {};
        out->total_bytes = UINT64_MAX;
        return false;
    }
    if (workspace == 0) {
        *out = {};
        out->total_bytes = UINT64_MAX;
        return false; // transient budget is required when XKV is enabled
    }
    // Sub-budget discipline: the decode sub-budget plus the exact seal
    // scratch must fit inside the one workspace allocation (or an explicit
    // shared-eviction protocol, which does not exist: fail closed).
    uint64_t concurrent = 0;
    if (!xkv_checked_add_u64(decode_cache, scratch_bytes, concurrent) || concurrent > workspace) {
        *out = {};
        out->total_bytes = UINT64_MAX;
        return false;
    }
    out->workspace_bytes       = workspace;
    out->decode_cache_bytes    = decode_cache;
    out->factor_scratch_bytes  = scratch_bytes;
    // Store-owned host dedup vectors live outside the workspace arena:
    // carried for host-charge accounting, never device-partitioned and
    // never double-counted against the workspace sub-budget above.
    out->dedup_scratch_bytes   = dedup_scratch_bytes;
    out->total_bytes           = workspace; // once: never the sum
    return true;
}

uint32_t common_xkv_store_mib_for_bytes(uint64_t dense_ctx_bytes, double min_saving) {
    constexpr uint64_t MiB = 1024ull * 1024ull;
    if (dense_ctx_bytes == 0) {
        return 0;
    }
    // Invalid min_saving must fail at caller (return 0), not silently change
    // policy in production helper.
    if (!std::isfinite(min_saving) || min_saving < 0.0 || min_saving >= 1.0) {
        return 0;
    }
    // Compute conservative ceil bytes with long double (never underbudget
    // exactly at MiB boundaries due to float truncation):
    const long double ld_dense = (long double) dense_ctx_bytes;
    const long double ld_factor = (long double) 1.0 - (long double) min_saving;
    const long double ld_exact_target = ld_dense * ld_factor;
    if (!std::isfinite(ld_exact_target) || ld_exact_target <= 0.0L || ld_exact_target > (long double) UINT64_MAX) {
        return 0;
    }
    // std::ceil ensures we never underbudget mathematical target by even 1 byte
    const long double ld_ceil_bytes = std::ceil(ld_exact_target);
    if (ld_ceil_bytes > (long double) UINT64_MAX) {
        return 0;
    }
    uint64_t est_bytes = (uint64_t) ld_ceil_bytes;
    if (est_bytes < MiB) {
        est_bytes = MiB; // minimum 1 MiB nonzero budget
    }
    // Checked ceil MiB division
    uint64_t mib_num = 0;
    if (!xkv_checked_add_u64(est_bytes, MiB - 1, mib_num)) {
        return 0;
    }
    const uint64_t mib = mib_num / MiB;
    if (mib == 0 || mib > UINT32_MAX) {
        return 0;
    }
    return (uint32_t) mib;
}

uint32_t common_xkv_store_mib_with_overlap(
        uint64_t dense_ctx_bytes,
        double min_saving,
        uint32_t seg_tokens,
        uint32_t k_tokens) {
    const uint32_t base = common_xkv_store_mib_for_bytes(dense_ctx_bytes, min_saving);
    if (base == 0 || seg_tokens == 0 || k_tokens == 0) {
        return base;
    }
    // One segment's dense share, same factored ratio, capped at the base:
    // an in-flight COW replacement holds old + new versions of one segment.
    uint64_t seg_dense = 0;
    if (!xkv_checked_mul_u64(dense_ctx_bytes, seg_tokens, seg_dense)) {
        return 0;
    }
    seg_dense = seg_dense / k_tokens;
    if (seg_dense > dense_ctx_bytes) {
        seg_dense = dense_ctx_bytes;
    }
    const uint32_t seg_mib = common_xkv_store_mib_for_bytes(seg_dense, min_saving);
    uint64_t total_mib = base;
    if (seg_mib > 0) {
        uint64_t sum = 0;
        if (!xkv_checked_add_u64(base, seg_mib > base ? base : seg_mib, sum) || sum > UINT32_MAX) {
            return 0;
        }
        total_mib = sum;
    }
    return (uint32_t) total_mib;
}

bool common_xkv_partition_budget(uint64_t total, const uint64_t * weights, uint64_t * shares, size_t n) {
    if (n == 0) {
        return true;
    }
    if (weights == nullptr || shares == nullptr) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        shares[i] = 0;
    }
    if (total == 0) {
        return true;
    }
    uint64_t wsum = 0;
    size_t imax = 0;
    for (size_t i = 0; i < n; ++i) {
        uint64_t next = 0;
        if (!xkv_checked_add_u64(wsum, weights[i], next)) {
            return false; // weight sums never overflow in practice; fail closed
        }
        wsum = next;
        if (weights[i] > weights[imax]) {
            imax = i;
        }
    }
    if (wsum == 0) {
        // No placement info: split equally, remainder to shares[0].
        const uint64_t q = total / n;
        uint64_t acc = 0;
        for (size_t i = 0; i < n; ++i) {
            shares[i] = q;
            acc += q;
        }
        shares[0] += total - acc;
        return true;
    }
    // Exact-sum partition: floor shares plus the whole remainder (which is
    // < n) to the largest weight, so the shares sum to exactly total.
    __uint128_t acc = 0;
    for (size_t i = 0; i < n; ++i) {
        if (i == imax) {
            continue;
        }
        shares[i] = (uint64_t) ((__uint128_t) total * weights[i] / wsum);
        acc += shares[i];
    }
    shares[imax] = total - (uint64_t) acc;
    return true;
}

bool common_xkv_scratch_for_group(
        uint32_t seg_tokens,
        uint32_t chunk_tokens,
        uint64_t feat_k,
        uint64_t feat_v,
        uint32_t rank_k,
        uint32_t rank_v,
        ggml_type landmark_type,
        uint64_t * out_bytes,
        ggml_type type_a_k,
        ggml_type type_b_k,
        ggml_type type_a_v,
        ggml_type type_b_v) {
    if (out_bytes == nullptr) {
        return false;
    }
    *out_bytes = 0;
    if (seg_tokens == 0 || feat_k == 0 || feat_v == 0 || rank_k == 0 || rank_v == 0) {
        return false;
    }
    if (landmark_type != GGML_TYPE_F32 && landmark_type != GGML_TYPE_F16 &&
        landmark_type != GGML_TYPE_Q8_0 && landmark_type != GGML_TYPE_TURBO4_0) {
        return false;
    }
    const int64_t lm_blck = ggml_blck_size(landmark_type);
    if (lm_blck <= 0 || (int64_t) feat_k % lm_blck != 0) {
        return false;
    }
    const uint32_t chunk = chunk_tokens == 0 ? 1 : chunk_tokens;
    std::string err;
    // Exact rSVD factorize peaks per stream (outputs + live scratch), plus
    // both FP factor outputs (K and V stay live into shadow evaluation).
    uint64_t ws_k = 0;
    uint64_t ws_v = 0;
    uint64_t out_k = 0;
    uint64_t out_v = 0;
    if (!llama_xkv::estimate_factorize_matrix_workspace_bytes(seg_tokens, feat_k, rank_k, 16, 2, &ws_k, &err) ||
        !llama_xkv::estimate_factorize_matrix_workspace_bytes(seg_tokens, feat_v, rank_v, 16, 2, &ws_v, &err) ||
        !llama_xkv::estimate_factor_output_bytes(seg_tokens, feat_k, rank_k, &out_k, &err) ||
        !llama_xkv::estimate_factor_output_bytes(seg_tokens, feat_v, rank_v, &out_v, &err)) {
        return false;
    }
    // Simultaneous K+V peak (K output live during V factorization), same
    // terms as estimate_factorize_kv_workspace_bytes.
    uint64_t term2 = 0;
    uint64_t ws_fact = ws_k;
    if (!xkv_checked_add_u64(out_k, ws_v, term2)) {
        return false;
    }
    ws_fact = std::max(ws_fact, term2);
    // Exact candidate encoded bytes via the canonical codec descriptors
    // (same recipe the seal path preflights). Turbo rank-128 padding is
    // included: encoded can EXCEED FP at tiny rank, never inferred from FP
    // logical size.
    auto xkv_is_turbo = [](ggml_type t) {
        return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
    };
    const uint32_t grp_k = xkv_is_turbo(type_a_k) ? 128 : 0;
    const uint32_t grp_v = xkv_is_turbo(type_a_v) ? 128 : 0;
    uint64_t enc_ak = 0;
    uint64_t enc_bk = 0;
    uint64_t enc_av = 0;
    uint64_t enc_bv = 0;
    uint64_t pad_ra = 0;
    uint64_t pad_rb = 0;
    size_t dec_tmp_ak = 0;
    size_t dec_tmp_bk = 0;
    size_t dec_tmp_av = 0;
    size_t dec_tmp_bv = 0;
    try {
        using namespace llama_xkv;
        const codec_desc d_ak = make_codec_desc(factor_role::a_k, type_a_k,
            orientation::token_major, matrix_shape{(uint64_t) seg_tokens, (uint64_t) rank_k}, grp_k, 42);
        const codec_desc d_bk = make_codec_desc(factor_role::b_k, type_b_k,
            orientation::feature_major_transposed, matrix_shape{feat_k, (uint64_t) rank_k}, grp_k, 42);
        const codec_desc d_av = make_codec_desc(factor_role::a_v, type_a_v,
            orientation::token_major, matrix_shape{(uint64_t) seg_tokens, (uint64_t) rank_v}, grp_v, 43);
        const codec_desc d_bv = make_codec_desc(factor_role::b_v, type_b_v,
            orientation::feature_major_transposed, matrix_shape{feat_v, (uint64_t) rank_v}, grp_v, 43);
        enc_ak = encoded_matrix_bytes(d_ak);
        enc_bk = encoded_matrix_bytes(d_bk);
        enc_av = encoded_matrix_bytes(d_av);
        enc_bv = encoded_matrix_bytes(d_bv);
            pad_ra = std::max(d_ak.padded_shape.cols, d_av.padded_shape.cols);
            pad_rb = std::max(d_bk.padded_shape.cols, d_bv.padded_shape.cols);
            if (!decode_rows_scratch_bytes(d_ak, dec_tmp_ak) ||
                !decode_rows_scratch_bytes(d_bk, dec_tmp_bk) ||
                !decode_rows_scratch_bytes(d_av, dec_tmp_av) ||
                !decode_rows_scratch_bytes(d_bv, dec_tmp_bv)) {
                return false;
            }
        } catch (...) {
            return false;
        }
    uint64_t cand_encoded = 0;
    if (!xkv_checked_add_u64(enc_ak, enc_bk, cand_encoded) ||
        !xkv_checked_add_u64(cand_encoded, enc_av, cand_encoded) ||
        !xkv_checked_add_u64(cand_encoded, enc_bv, cand_encoded)) {
        return false;
    }
    // Quantized-shadow peak mirrors estimate_quantized_shadow_workspace_bytes:
    // live FP factors + exact encoded streams + decode tile scratch (tile
    // buffers sized by the padded ranks above, never the logical ranks).
    uint64_t fp_live = 0;
    uint64_t tile_a = 0;
    uint64_t tile_b = 0;
    uint64_t tile_idx = 0;
    uint64_t ws_shadow = 0;
    const uint64_t max_dec_tmp = (uint64_t) std::max(std::max(dec_tmp_ak, dec_tmp_bk),
        std::max(dec_tmp_av, dec_tmp_bv));
    uint64_t dec_tmp_budget = 0;
    if (!xkv_checked_add_u64(out_k, out_v, fp_live) ||
        !xkv_checked_mul_u64(64, sizeof(float), tile_a) ||
        !xkv_checked_mul_u64(tile_a, pad_ra, tile_a) ||
        !xkv_checked_mul_u64(64, sizeof(float), tile_b) ||
        !xkv_checked_mul_u64(tile_b, pad_rb, tile_b) ||
        !xkv_checked_mul_u64(128, sizeof(uint64_t), tile_idx) ||
        !xkv_checked_add_u64(max_dec_tmp, 1024, dec_tmp_budget) ||
        !xkv_checked_add_u64(fp_live, cand_encoded, ws_shadow) ||
        !xkv_checked_add_u64(ws_shadow, tile_a, ws_shadow) ||
        !xkv_checked_add_u64(ws_shadow, tile_b, ws_shadow) ||
        !xkv_checked_add_u64(ws_shadow, tile_idx, ws_shadow) ||
        !xkv_checked_add_u64(ws_shadow, dec_tmp_budget, ws_shadow)) {
        return false;
    }
    // Group peak is the worse of factorize vs shadow (arena lease grows).
    const uint64_t ws_group_peak = std::max(ws_fact, ws_shadow);
    // Canonical pre-RoPE capture peak (F32 K+V rows for the segment).
    uint64_t feat_sum = 0;
    uint64_t cap_elems = 0;
    uint64_t capture = 0;
    if (!xkv_checked_add_u64(feat_k, feat_v, feat_sum) ||
        !xkv_checked_mul_u64(seg_tokens, feat_sum, cap_elems) ||
        !xkv_checked_mul_u64(cap_elems, sizeof(float), capture)) {
        return false;
    }
    // Landmark stream bound: one pooled vector per fragment. Landmarks
    // summarize reconstructed K only (never K+V) over feat_k, via the
    // canonical descriptor so Turbo landmark padding matches the seal path.
    const uint64_t n_frag = ((uint64_t) seg_tokens + chunk - 1) / chunk;
    if (feat_k > (uint64_t) INT64_MAX || n_frag > (uint64_t) INT64_MAX) {
        return false;
    }
    uint64_t landmark = 0;
    try {
        using namespace llama_xkv;
        const codec_desc d_lm = make_codec_desc(factor_role::landmark, landmark_type,
            orientation::token_major, matrix_shape{n_frag, feat_k}, 0, 777);
        landmark = encoded_matrix_bytes(d_lm);
    } catch (...) {
        return false;
    }
    // Telemetry and staging metadata scratch:
    // Status tensors, singular value vectors S, landmark error bounds eb,
    // source fingerprints srcfp, and staging context overhead.
    uint64_t n_frag_eb = 0;
    uint64_t n_frag_srcfp = 0;
    uint64_t telem_staging = 0;
    if (!xkv_checked_mul_u64(n_frag, sizeof(float), n_frag_eb) ||
        !xkv_checked_mul_u64(n_frag, sizeof(uint64_t), n_frag_srcfp) ||
        !xkv_checked_add_u64(n_frag_eb, n_frag_srcfp, telem_staging) ||
        !xkv_checked_add_u64(telem_staging, 4096, telem_staging)) {
        return false;
    }
    // Backend stream allocator alignment padding (e.g. 256-byte alignment per stream
    // across the 5 destination streams: A_K, B_K, A_V, B_V, and landmark).
    constexpr uint64_t stream_alloc_align = 5ull * 256ull;

    // Conservative simultaneous sum: peaks may not fully coincide, but
    // under-reserving seal-time workspace is the unsafe direction.
    uint64_t total = 0;
    if (!xkv_checked_add_u64(ws_group_peak, capture, total) ||
        !xkv_checked_add_u64(total, landmark, total) ||
        !xkv_checked_add_u64(total, telem_staging, total) ||
        !xkv_checked_add_u64(total, stream_alloc_align, *out_bytes)) {
        return false;
    }
    return true;
}

bool common_xkv_scratch_bytes(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        uint64_t * out_bytes,
        uint64_t * out_dedup_scratch_bytes) {
    if (out_bytes == nullptr) {
        return false;
    }
    *out_bytes = 0;
    if (out_dedup_scratch_bytes != nullptr) {
        *out_dedup_scratch_bytes = 0;
    }
    if (path_model == nullptr || mparams == nullptr || cparams == nullptr) {
        return false;
    }
    if (!llama_xkv_is_enabled(cparams->xkv_mode)) {
        return true; // OFF: no scratch
    }
    const uint32_t group = cparams->xkv_group_size;
    if (group == 0 || cparams->xkv_segment_tokens == 0 ||
        cparams->xkv_rank_k == 0 || cparams->xkv_rank_v == 0) {
        return false;
    }
    // Metadata-only model (never XKV runtime) for hparams geometry.
    llama_model_params meta_mparams = *mparams;
    meta_mparams.no_alloc = true;
    meta_mparams.load_mode = LLAMA_LOAD_MODE_NONE;
    llama_model * meta_model = llama_model_load_from_file(path_model, meta_mparams);
    if (meta_model == nullptr) {
        return false;
    }
    uint64_t dk_max = 0;
    uint64_t dv_max = 0;
    bool any_trunk = false;
    uint64_t n_attn_layers = 0;
    const auto & hp = meta_model->hparams;
    const uint32_t n_layer = hp.n_layer();
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (hp.is_recr(il) || hp.is_swa(il)) {
            continue;
        }
        // Dedup owning layers when model specifies layer KV sharing
        // (e.g. Gemma3n / Gemma4 assistant where layers beyond n_layer_kv_from_start
        // share/reuse earlier KV storage; other architectures have independent layers).
        if (hp.n_layer_kv_from_start >= 0 && il >= (uint32_t) hp.n_layer_kv_from_start) {
            continue; // Reused layer: storage is already accounted under owning layer
        }
        if (!hp.has_kv(il)) {
            continue;
        }
        const uint64_t dk = hp.n_embd_k_gqa(il);
        const uint64_t dv = hp.n_embd_v_gqa(il);
        if (dk == 0 || dv == 0) {
            continue; // no KV (projection-only layer)
        }
        any_trunk = true;
        n_attn_layers++;
        dk_max = std::max(dk_max, dk);
        dv_max = std::max(dv_max, dv);
    }
    llama_model_free(meta_model);
    if (!any_trunk) {
        // Hybrid/MoE model with no full-attention trunk layers (or all
        // recurrent/SWA/sharing layers): XKV factor scratch is zero. Never
        // abort auto-fit; degrade to zero reserve with an explicit warning.
        LOG_WRN("%s: no full-attention trunk layers found in model; degrading XKV scratch reserve to 0\n", __func__);
        *out_bytes = 0;
        if (out_dedup_scratch_bytes != nullptr) {
            *out_dedup_scratch_bytes = 0;
        }
        return true;
    }
    // Largest alias-dedup group bound: every member is at most the per-layer
    // max, so group_size * max bounds any spliced group the store builds.
    uint64_t feat_k = 0;
    uint64_t feat_v = 0;
    if (!xkv_checked_mul_u64(group, dk_max, feat_k) ||
        !xkv_checked_mul_u64(group, dv_max, feat_v)) {
        return false;
    }
    // Store-owned host dedup-scratch upper bound from the fit topology
    // (mirrors ensure_accounting_scratch_locked): 8 B per B-matrix pointer
    // in each of the two unique vectors plus 8 B per backend handle, over
    // published + one retired + one candidate segment.
    if (out_dedup_scratch_bytes != nullptr) {
        // When called before KV capacity fitting (e.g. initial recurrent probe),
        // n_ctx_kv and n_ctx may both be 0: fall back to 512 tokens minimum.
        const uint64_t k_raw = std::max(cparams->n_ctx_kv, cparams->n_ctx);
        const uint64_t k_tokens = k_raw > 0 ? k_raw : 512;
        const uint64_t seg = cparams->xkv_segment_tokens;
        uint64_t n_seg = 0;
        uint64_t n_groups = 0;
        uint64_t sum_b = 0;
        uint64_t sum_backend = 0;
        uint64_t dedup = 0;
        if (seg == 0 || group == 0 ||
            !xkv_checked_add_u64((k_tokens + seg - 1) / seg, 2, n_seg) ||
            (n_groups = (n_attn_layers + group - 1) / group) == 0 ||
            !xkv_checked_mul_u64(2, n_groups, sum_b) ||
            !xkv_checked_mul_u64(sum_b, n_seg, sum_b) ||
            !xkv_checked_mul_u64(8, n_groups, sum_backend) ||
            !xkv_checked_mul_u64(sum_backend, n_seg, sum_backend) ||
            !xkv_checked_add_u64(sum_b, sum_b, dedup) ||
            !xkv_checked_add_u64(dedup, sum_backend, dedup) ||
            !xkv_checked_mul_u64(dedup, 8, dedup)) {
            return false;
        }
        // Floor at the store's initial reservation (64 + 64 ptrs + 128 ids).
        if (dedup < 2048) {
            dedup = 2048;
        }
        *out_dedup_scratch_bytes = dedup;
    }
    return common_xkv_scratch_for_group(
        cparams->xkv_segment_tokens, cparams->xkv_chunk_tokens,
        feat_k, feat_v, cparams->xkv_rank_k, cparams->xkv_rank_v,
        cparams->xkv_landmark_type, out_bytes,
        cparams->xkv_factor_a_k, cparams->xkv_factor_b_k,
        cparams->xkv_factor_a_v, cparams->xkv_factor_b_v);
}

uint64_t common_xkv_store_budget_bytes(const llama_context_params * cparams) {
    if (cparams == nullptr || !llama_xkv_is_enabled(cparams->xkv_mode)) {
        return 0;
    }
    // SHADOW is evaluate-only with no segment publication for every profile.
    if (cparams->xkv_mode == LLAMA_XKV_MODE_SHADOW) {
        return 0;
    }
    if (cparams->xkv_store_mib == 0) {
        return 0; // auto-derive placeholder, resolved after auto-fit
    }
    uint64_t bytes = 0;
    if (!xkv_checked_mul_u64((uint64_t) cparams->xkv_store_mib, 1024ull * 1024ull, bytes)) {
        return UINT64_MAX;
    }
    return bytes;
}

uint64_t common_xkv_store_device_bytes(const llama_context_params * cparams, uint64_t store_bytes) {
    if (cparams == nullptr) {
        return 0;
    }
    // Same effective residency predicate the runtime enforces
    // (llama_xkv_profile_is_device_owned): profile alone never decides.
    if (!llama_xkv_profile_is_device_owned(cparams->xkv_storage_profile, cparams->xkv_factorizer)) {
        return 0;
    }
    return store_bytes;
}

uint32_t common_xkv_derive_store_mib(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        uint32_t n_ctx_kv_fit,
        ggml_log_level log_level) {
    if (path_model == nullptr || mparams == nullptr || cparams == nullptr || n_ctx_kv_fit == 0) {
        return 0;
    }
    if (!llama_xkv_is_enabled(cparams->xkv_mode)) {
        return 0;
    }
    (void) log_level;
    // Genuine dense KV equivalent: exact same-row original K/V bytes only
    // across all owning attention layers (excluding non-KV context/compute
    // memory). Derived directly from model hparams and effective K/V tensor
    // row sizes × n_ctx_kv_fit.
    llama_model_params meta_mparams = *mparams;
    meta_mparams.no_alloc = true;
    meta_mparams.load_mode = LLAMA_LOAD_MODE_NONE;
    llama_model * meta_model = llama_model_load_from_file(path_model, meta_mparams);
    if (meta_model == nullptr) {
        return 0;
    }
    const auto & hp = meta_model->hparams;
    const uint32_t n_layer = hp.n_layer();
    const ggml_type type_k = cparams->type_k;
    const ggml_type type_v = cparams->type_v;
    const bool k_is_turbo = (type_k == GGML_TYPE_TURBO3_0 || type_k == GGML_TYPE_TURBO4_0 || type_k == GGML_TYPE_TURBO2_0);
    const bool v_is_turbo = (type_v == GGML_TYPE_TURBO3_0 || type_v == GGML_TYPE_TURBO4_0 || type_v == GGML_TYPE_TURBO2_0);

    uint64_t single_token_kv_bytes = 0;
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (hp.is_recr(il) || hp.is_swa(il)) {
            continue;
        }
        // Dedup owning layers when model specifies layer KV sharing
        // (e.g. Gemma3n / Gemma4 assistant where layers beyond n_layer_kv_from_start
        // share/reuse earlier KV storage; other architectures have independent layers).
        if (hp.n_layer_kv_from_start >= 0 && il >= (uint32_t) hp.n_layer_kv_from_start) {
            continue; // Reused layer: storage is already accounted under owning layer
        }
        if (!hp.has_kv(il)) {
            continue;
        }
        const uint64_t head_k = (uint64_t) hp.n_embd_head_k(il);
        const uint64_t head_v = (uint64_t) hp.n_embd_head_v(il);
        const uint64_t n_head_kv = (uint64_t) hp.n_head_kv(il);
        if (head_k == 0 || head_v == 0 || n_head_kv == 0) {
            continue;
        }
        // Checked rounding for Turbo 128-alignment (never uint32 overflow):
        uint64_t head_k_eff = head_k;
        if (k_is_turbo && head_k % 128 != 0) {
            uint64_t sum_pad = 0;
            if (!xkv_checked_add_u64(head_k, 127, sum_pad)) {
                llama_model_free(meta_model);
                return 0;
            }
            head_k_eff = (sum_pad / 128) * 128;
        }
        uint64_t head_v_eff = head_v;
        if (v_is_turbo && head_v % 128 != 0) {
            uint64_t sum_pad = 0;
            if (!xkv_checked_add_u64(head_v, 127, sum_pad)) {
                llama_model_free(meta_model);
                return 0;
            }
            head_v_eff = (sum_pad / 128) * 128;
        }
        // Checked multiplication and INT64_MAX validation before ggml_row_size:
        uint64_t ne_k_u64 = 0;
        uint64_t ne_v_u64 = 0;
        if (!xkv_checked_mul_u64(n_head_kv, head_k_eff, ne_k_u64) ||
            !xkv_checked_mul_u64(n_head_kv, head_v_eff, ne_v_u64) ||
            ne_k_u64 > (uint64_t) INT64_MAX || ne_v_u64 > (uint64_t) INT64_MAX) {
            llama_model_free(meta_model);
            return 0;
        }
        const int64_t blck_k = ggml_blck_size(type_k);
        const int64_t blck_v = ggml_blck_size(type_v);
        if (blck_k <= 0 || blck_v <= 0 || (int64_t) ne_k_u64 % blck_k != 0 || (int64_t) ne_v_u64 % blck_v != 0) {
            // Reject unsupported type or non-divisible block geometry without asserts
            llama_model_free(meta_model);
            return 0;
        }
        const size_t row_k = ggml_row_size(type_k, (int64_t) ne_k_u64);
        const size_t row_v = ggml_row_size(type_v, (int64_t) ne_v_u64);
        if (row_k == 0 || row_v == 0) {
            llama_model_free(meta_model);
            return 0;
        }
        uint64_t layer_row = 0;
        if (!xkv_checked_add_u64((uint64_t) row_k, (uint64_t) row_v, layer_row) ||
            !xkv_checked_add_u64(single_token_kv_bytes, layer_row, single_token_kv_bytes)) {
            llama_model_free(meta_model);
            return 0;
        }
    }
    llama_model_free(meta_model);
    if (single_token_kv_bytes == 0) {
        return 0;
    }
    uint64_t dense_ctx = 0;
    if (!xkv_checked_mul_u64(single_token_kv_bytes, (uint64_t) n_ctx_kv_fit, dense_ctx)) {
        return 0;
    }
    // Worst-case one-segment seal/COW overlap included, so runtime never
    // refuses maintenance at pressure for lack of budgeted overlap bytes.
    // Backend alignment is covered by the MiB round-up.
    return common_xkv_store_mib_with_overlap(
        dense_ctx, cparams->xkv_min_saving, cparams->xkv_segment_tokens, n_ctx_kv_fit);
}


// A probe differs from the previous one only in its context parameters, but it used to
// reload the model every time -- metadata and tensor map for every shard, seconds on a
// 79 GB model, repeated for every step of the fit's search (and the RERoT fit repeats the
// whole search per slot count). Probe models are loaded with no_alloc, so they hold no
// device memory at all; keeping the last one alive makes a probe cost only its context.
static llama_model * common_fit_probe_model(const char * path_model, const llama_model_params & mparams) {
    static std::string   key_cached;
    static llama_model * model_cached = nullptr;

    std::string key = std::string(path_model) + "|" + std::to_string((int) mparams.no_alloc) + "|" +
                      std::to_string(mparams.n_gpu_layers) + "|" + std::to_string((int) mparams.split_mode) + "|" +
                      std::to_string(mparams.main_gpu) + "|" + std::to_string((int) mparams.load_mode);
    for (int i = 0; i < llama_max_devices(); i++) {
        key += "|";
        key += mparams.devices != nullptr && mparams.devices[i] != nullptr ? ggml_backend_dev_name(mparams.devices[i]) : "-";
        key += ":";
        key += std::to_string(mparams.tensor_split[i]);
    }

    if (model_cached != nullptr && key == key_cached) {
        return model_cached;
    }
    if (model_cached != nullptr) {
        llama_model_free(model_cached);
        model_cached = nullptr;
    }
    model_cached = llama_model_load_from_file(path_model, mparams);
    key_cached   = key;
    return model_cached;
}

// GTT (system memory) used by a device, from the kernel. The dry probe allocates and writes
// like a real load, so if the weights or the pool do not fit in VRAM the driver starts
// migrating pages to system memory; that is the failure mode that ends in a system-wide OOM
// (measured: 45 GiB of shmem, then the OOM killer). The fit therefore checks this after every
// probe and aborts at once instead of measuring how far past the edge it went.
// Shmem of the host, in kB. This is the only signal that told the truth about a load that does
// not fit: the CPU-offloaded tensors live in GTT by design and the driver pools its freed device
// buffers, so GTT climbs for healthy runs too - but pages that end up in system memory because
// the card is full are shmem, and 34 GiB of them took the machine down once already.
static uint64_t common_fit_shmem_kb() {
    FILE * f = fopen("/proc/meminfo", "r");
    if (f == nullptr) {
        return 0;
    }
    char line[256];
    uint64_t shmem_kb = 0;
    while (fgets(line, sizeof(line), f) != nullptr) {
        if (sscanf(line, "Shmem: %" SCNu64 " kB", &shmem_kb) == 1) {
            break;
        }
    }
    fclose(f);
    return shmem_kb;
}

static bool common_fit_device_gtt(ggml_backend_dev_t dev, uint64_t & gtt_out) {
    if (dev == nullptr) {
        return false;
    }
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(dev, &props);
    if (props.device_id == nullptr || props.device_id[0] == '\0') {
        return false;
    }
    const std::string bdf = props.device_id;

    DIR * dir = opendir("/sys/class/drm");
    if (dir == nullptr) {
        return false;
    }
    while (dirent * ent = readdir(dir)) {
        const std::string name = ent->d_name;
        if (name.rfind("card", 0) != 0 || name.find('-') != std::string::npos) {
            continue;
        }
        const std::string base = "/sys/class/drm/" + name + "/device";
        char target[PATH_MAX];
        const ssize_t n = readlink(base.c_str(), target, sizeof(target) - 1);
        if (n <= 0) {
            continue;
        }
        target[n] = '\0';
        const std::string path = target;
        const size_t pos = path.rfind(bdf);
        if (pos == std::string::npos) {
            continue;
        }
        const size_t end = pos + bdf.size();
        if (end < path.size() && path[end] != '/') {
            continue;
        }
        FILE * f = fopen((base + "/mem_info_gtt_used").c_str(), "r");
        if (f == nullptr) {
            continue;
        }
        const bool ok = fscanf(f, "%" SCNu64, &gtt_out) == 1;
        fclose(f);
        closedir(dir);
        return ok;
    }
    closedir(dir);
    return false;
}

// Free/total VRAM of a backend device from the kernel's own accounting.
//
// The backend's ggml_backend_dev_memory() is not VRAM free space on RADV: it reports the
// driver's own budget, which on an otherwise empty 16 GB card can be a few GB and has
// nothing to do with the physical capacity. Every fit decision compares a planned total
// against it, so the search either rejects everything ("does not fit") or accepts values
// the driver then has to migrate away. On Linux/amdgpu the kernel exposes exact per-card
// numbers; the DRM card is matched to the backend device through its PCI slot name
// (device_id, e.g. "0000:03:00.0"). Falls back to the API value when unavailable.
static bool common_fit_device_vram(ggml_backend_dev_t dev, uint64_t & free_out, uint64_t & total_out) {
    if (dev == nullptr) {
        return false;
    }
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(dev, &props);
    if (props.device_id == nullptr || props.device_id[0] == '\0') {
        return false;
    }
    const std::string bdf = props.device_id;

    DIR * dir = opendir("/sys/class/drm");
    if (dir == nullptr) {
        return false;
    }

    bool ok = false;
    while (dirent * ent = readdir(dir)) {
        const std::string name = ent->d_name;
        if (name.rfind("card", 0) != 0 || name.find('-') != std::string::npos) {
            continue;   // connectors show up as cardN-XXX
        }
        const std::string base = "/sys/class/drm/" + name + "/device";
        char target[PATH_MAX];
        const ssize_t n = readlink(base.c_str(), target, sizeof(target) - 1);
        if (n <= 0) {
            continue;
        }
        target[n] = '\0';
        const std::string path = target;
        const size_t pos = path.rfind(bdf);
        if (pos == std::string::npos) {
            continue;
        }
        const size_t end = pos + bdf.size();
        if (end < path.size() && path[end] != '/') {
            continue;
        }

        uint64_t used = 0;
        uint64_t total = 0;
        FILE * f = fopen((base + "/mem_info_vram_used").c_str(), "r");
        if (f == nullptr) {
            continue;
        }
        const bool got_used = fscanf(f, "%" SCNu64, &used) == 1;
        fclose(f);
        if (!got_used) {
            continue;
        }
        f = fopen((base + "/mem_info_vram_total").c_str(), "r");
        if (f == nullptr) {
            continue;
        }
        const bool got_total = fscanf(f, "%" SCNu64, &total) == 1;
        fclose(f);
        if (!got_total || total == 0) {
            continue;
        }

        free_out  = total > used ? total - used : 0;
        total_out = total;
        ok = true;
        break;
    }
    closedir(dir);
    return ok;
}

static std::vector<llama_device_memory_data> common_get_device_memory_data_impl(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level,
        bool no_alloc = true) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        // Forward all ERROR and WARN messages unconditionally so context creation
        // failures are visible regardless of probe min_level.
        const ggml_log_level level_eff = (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN || level >= ud->min_level)
            ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    llama_model_params mparams_copy = *mparams;
    mparams_copy.no_alloc  = no_alloc;
    // Dry probe (no_alloc == false): every buffer is allocated exactly as a real load allocates
    // it - same offload, same load mode, same host side - and the weight transfer is skipped.
    // That is the whole difference to a real load, so its memory picture is the real one.
    mparams_copy.dry_run = !no_alloc;
    if (no_alloc) {
        // metadata-only probe: no buffers at all, nothing to transfer either
        mparams_copy.load_mode = LLAMA_LOAD_MODE_NONE;
    }

    // The dry (no_alloc = false) probe owns its model: caching it would keep ~52 GiB of device
    // buffers alive, and it is the only probe that allocates at all.
    llama_model * model = mparams_copy.no_alloc ? common_fit_probe_model(path_model, mparams_copy)
                                                : llama_model_load_from_file(path_model, mparams_copy);
    if (model == nullptr) {
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw std::runtime_error("failed to load model");
    }

    llama_context * ctx = llama_init_from_model(model, *cparams);
    if (ctx == nullptr) {
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw std::runtime_error("failed to create llama_context from model");
    }

    const size_t nd = llama_model_n_devices(model);
    std::vector<llama_device_memory_data> ret(nd + 1);

    // When the probe allocates the model buffers (no_alloc == false, and load mode NONE means
    // they are reserved but never filled), decode once and record what the graph/compute
    // buffers really take: the reserve the context reports is an upper bound that over-books
    // by ~2x, and a fit that books it rejects models whose weights nearly fill the card.
    // free_before - free_after is that real cost, device by device.
    std::vector<size_t> compute_measured(nd, 0);
    std::vector<size_t> free_after_decode(nd, 0);
    if (!no_alloc) {
        std::vector<size_t> free_before(nd, 0);
        // Back the whole memory before taking the baseline: a lazily allocating driver (RADV)
        // reserves the cache but only backs it when written, so a probe that skips this measures
        // a pool that is not resident - and then accepts a capacity the real load cannot hold,
        // which shows up as KV-sized allocations pushed out of VRAM into system memory (GTT).
        // With this, what a dry probe measures is what a real load gets, and a candidate that
        // cannot hold its own pool fails here instead of in production.
        llama_memory_materialize(ctx);
        for (size_t i = 0; i < nd; i++) {
            size_t total_tmp = 0;
            ggml_backend_dev_memory(llama_model_get_device(model, i), &free_before[i], &total_tmp);
        }
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.n_tokens     = 1;
        batch.token[0]     = 0;
        batch.pos[0]       = 0;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;
        llama_decode(ctx, batch);   // a failure is fine: the allocations happen before it
        llama_batch_free(batch);
        for (size_t i = 0; i < nd; i++) {
            size_t total_tmp = 0;
            ggml_backend_dev_memory(llama_model_get_device(model, i), &free_after_decode[i], &total_tmp);
        }
        // Second decode, and then measure across both: the first one faults in whatever the driver
        // had only reserved - Vulkan allocates lazily and the KV pool is the largest of those - so
        // its own delta charges the pool to the graph as well. The caller subtracts the declared
        // context (the pool) from what is measured here, so what is left after both decodes is the
        // steady cost of running, counted once.
        {
            llama_batch batch2 = llama_batch_init(1, 0, 1);
            batch2.n_tokens     = 1;
            batch2.token[0]     = 0;
            batch2.pos[0]       = 1;
            batch2.n_seq_id[0]  = 1;
            batch2.seq_id[0][0] = 0;
            batch2.logits[0]    = 1;
            llama_decode(ctx, batch2);
            llama_batch_free(batch2);
        }
        for (size_t i = 0; i < nd; i++) {
            size_t total_tmp = 0;
            size_t free_final = 0;
            ggml_backend_dev_memory(llama_model_get_device(model, i), &free_final, &total_tmp);
            free_after_decode[i] = free_final;
            compute_measured[i] = free_before[i] > free_final ? free_before[i] - free_final : 0;
        }
    }

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    for (const auto & [buft, mb] : memory_breakdown) {
        if (ggml_backend_buft_is_host(buft)) {
            ret.back().mb.model   += mb.model;
            ret.back().mb.context += mb.context;
            ret.back().mb.compute += mb.compute;
            continue;
        }

        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            continue;
        }
        for (size_t i = 0; i < nd; i++) {
            if (dev == llama_model_get_device(model, i)) {
                ret[i].mb.model   += mb.model;
                ret[i].mb.context += mb.context;
                ret[i].mb.compute += mb.compute;
                break;
            }
        }
    }

    // The KV store is allocated as one block, so the breakdown reports all of it against a single
    // buffer type and a single device, while every device only ever holds the share of the layers
    // it runs. Charging one card for the whole pool (measured: 13.5 GiB for a 524288-token pool on
    // a card that holds a fifth of the layers) rejects configurations the machine runs fine and
    // hides the card that really is short. Spread it by the number of layers each device holds.
    {
        uint64_t ctx_total  = 0;
        int      ctx_dev    = -1;   // -1: none, >=0: exactly one, -2: several
        for (size_t i = 0; i < ret.size(); i++) {
            if (ret[i].mb.context > 0) {
                ctx_total += ret[i].mb.context;
                ctx_dev = ctx_dev == -1 ? (int) i : -2;
            }
        }
        const int64_t n_layer = llama_model_n_layer(model);
        if (ctx_total > 0 && ctx_dev >= 0 && nd > 1 && n_layer > 0) {
            std::vector<uint64_t> layers_per_dev(nd, 0);
            uint64_t layers_seen = 0;
            for (int64_t il = 0; il < n_layer; ++il) {
                ggml_backend_dev_t d = model->dev_layer((int) il);
                for (size_t i = 0; i < nd; ++i) {
                    if (d == llama_model_get_device(model, i)) {
                        layers_per_dev[i]++;
                        layers_seen++;
                        break;
                    }
                }
            }
            if (layers_seen > 0) {
                for (size_t i = 0; i < ret.size(); i++) {
                    ret[i].mb.context = 0;
                }
                uint64_t assigned = 0;
                for (size_t i = 0; i < nd; ++i) {
                    ret[i].mb.context = ctx_total * layers_per_dev[i] / layers_seen;
                    assigned += ret[i].mb.context;
                }
                // whatever integer division dropped stays with the device that owned the pool
                ret[ctx_dev].mb.context += ctx_total - assigned;
            }
            LOG_WRN("%s: KV pool %llu MiB reported on device %d; layers per device:", __func__, ctx_total >> 20, ctx_dev);
            for (size_t i = 0; i < nd; ++i) {
                LOG_WRN(" [%zu]=%llu", i, layers_per_dev[i]);
            }
            LOG_WRN(" seen=%llu\n", layers_seen);
        }
    }

    {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error("no CPU backend found");
        }
        size_t free;
        size_t total;
        ggml_backend_dev_memory(cpu_dev, &free, &total);
        ret.back().free  = free;
        ret.back().total = total;
    }
    for (size_t i = 0; i < nd; i++) {
        ggml_backend_dev_t dev = llama_model_get_device(model, i);

        size_t free;
        size_t total;
        ggml_backend_dev_memory(dev, &free, &total);

        // prefer the kernel's accounting: the backend budget is not VRAM free space on RADV
        {
            uint64_t vram_free  = 0;
            uint64_t vram_total = 0;
            if (common_fit_device_vram(dev, vram_free, vram_total)) {
                free  = (size_t) vram_free;
                total = (size_t) vram_total;
            }
        }

        // Some non-GPU accelerator backends, such as BLAS, report 0/0 and rely on
        // the host-memory fallback. For GPU-like backends, keep 0/0 so --fit does
        // not assign anything to a device with an unknown memory budget.
        if (free == 0 && total == 0) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                LOG_WRN("%s: device %s did not report memory; --fit will not use it\n",
                        __func__, ggml_backend_dev_name(dev));
            } else {
                free  = ret.back().free;
                total = ret.back().total;
            }
        }
        if (!no_alloc && i < nd) {
            // report the capacity as if the model were not resident, so the caller's
            // model + context + compute <= free test stays valid, and replace the planned
            // compute with the measured one
            free = free_after_decode[i] + ret[i].mb.model;
            if (compute_measured[i] > 0) {
                // The measured delta is what the context costs on top of the weights, and the
                // declared context (the pool) is already part of it: book only the remainder, so
                // the fit's decision does not charge the pool twice.
                ret[i].mb.compute = compute_measured[i] > ret[i].mb.context
                    ? compute_measured[i] - ret[i].mb.context : 0;
            }
        }
        ret[i].free  = free;
        ret[i].total = total;
    }

    devs.clear();
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devs.push_back(llama_model_get_device(model, i));
    }

    hp_ngl         = llama_model_n_layer(model) + llama_model_n_layer_nextn(model);
    hp_n_ctx_train = llama_model_n_ctx_train(model);
    hp_n_expert    = llama_model_n_expert(model);


    common_memory_breakdown_print(ctx);

    llama_free(ctx);
    if (!mparams_copy.no_alloc) {
        llama_model_free(model);
    }
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);

    return ret;
}

common_device_memory_data_vec common_get_device_memory_data(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level,
        bool no_alloc) {
    std::vector<llama_device_memory_data> impl = common_get_device_memory_data_impl(
            path_model, mparams, cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level, no_alloc);

    common_device_memory_data_vec ret(impl.size());
    for (size_t i = 0; i < impl.size(); i++) {
        ret[i].total   = impl[i].total;
        ret[i].free    = impl[i].free;
        ret[i].model   = impl[i].mb.model;
        ret[i].context = impl[i].mb.context;
        ret[i].compute = impl[i].mb.compute;
    }

    return ret;
}

static void common_params_fit_impl(
        const char * path_model, struct llama_model_params * mparams, struct llama_context_params * cparams,
        float * tensor_split, struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins_s, uint32_t n_ctx_min, enum ggml_log_level log_level) {
    if (mparams->split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        throw common_params_fit_exception("llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort");
    }
    constexpr int64_t MiB = 1024*1024;
    typedef std::vector<llama_device_memory_data> dmds_t;
    const llama_model_params default_mparams = llama_model_default_params();

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    // step 1: get data for default parameters and check whether any changes are necessary in the first place

    LOG_TRC("%s: getting device memory data for initial parameters:\n", __func__);
    const dmds_t dmds_full = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
    const size_t nd = devs.size(); // number of devices

    std::vector<int64_t> margins; // this function uses int64_t rather than size_t for memory sizes to more conveniently handle deficits
    margins.reserve(nd);
    if (nd == 0) {
        margins.push_back(margins_s[0]);
    } else {
        for (size_t id = 0; id < nd; id++) {
            margins.push_back(margins_s[id]);
        }
    }

    std::vector<std::string> dev_names;
    {
        dev_names.reserve(nd);
        size_t max_length = 0;
        for (const auto & dev : devs) {
            std::string name = ggml_backend_dev_name(dev);
            name += " (";
            name += ggml_backend_dev_description(dev);
            name += ")";
            dev_names.push_back(name);
            max_length = std::max(max_length, name.length());
        }
        for (std::string & dn : dev_names) {
            dn.insert(dn.end(), max_length - dn.length(), ' ');
        }
    }

    int64_t sum_free            = 0;
    int64_t sum_projected_free  = 0;
    int64_t sum_projected_used  = 0;
    int64_t sum_projected_model = 0;
    std::vector<int64_t> projected_free_per_device;
    projected_free_per_device.reserve(nd);

    if (nd == 0) {
        sum_projected_used = dmds_full.back().mb.total();
        sum_free           = dmds_full.back().total;
        sum_projected_free = sum_free - sum_projected_used;
        LOG_TRC("%s: projected to use %" PRId64 " MiB of host memory vs. %" PRId64 " MiB of total host memory\n",
            __func__, sum_projected_used/MiB, sum_free/MiB);
        if (sum_projected_free >= margins[0]) {
            LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of system memory, no changes needed\n",
                __func__, sum_projected_free/MiB, margins[0]/MiB);
            return;
        }
    } else {
        if (nd > 1) {
            LOG_TRC("%s: projected memory use with initial parameters [MiB]:\n", __func__);
        }
        for (size_t id = 0; id < nd; id++) {
            const llama_device_memory_data & dmd = dmds_full[id];

            const int64_t projected_used = dmd.mb.total();
            const int64_t projected_free = dmd.free - projected_used;
            projected_free_per_device.push_back(projected_free);

            sum_free            += dmd.free;
            sum_projected_used  += projected_used;
            sum_projected_free  += projected_free;
            sum_projected_model += dmd.mb.model;

            if (nd > 1) {
                LOG_TRC("%s:   - %s: %6" PRId64 " total, %6" PRId64 " used, %6" PRId64 " free vs. target of %6" PRId64 "\n",
                    __func__, dev_names[id].c_str(), dmd.total/MiB, projected_used/MiB, projected_free/MiB, margins[id]/MiB);
            }
        }
        assert(sum_free >= 0 && sum_projected_used >= 0);
        LOG_TRC("%s: projected to use %" PRId64 " MiB of device memory vs. %" PRId64 " MiB of free device memory\n",
            __func__, sum_projected_used/MiB, sum_free/MiB);
        if (nd == 1) {
            if (projected_free_per_device[0] >= margins[0]) {
                LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of free device memory, no changes needed\n",
                    __func__, projected_free_per_device[0]/MiB, margins[0]/MiB);
                return;
            }
        } else {
            bool changes_needed = false;
            for (size_t id = 0; id < nd; id++) {
                if (projected_free_per_device[id] < margins[id]) {
                    changes_needed = true;
                    break;
                }
            }
            if (!changes_needed) {
                LOG_TRC("%s: targets for free memory can be met on all devices, no changes needed\n", __func__);
                return;
            }
        }
    }

    // step 2: try reducing memory use by reducing the context size

    {
        int64_t global_surplus = sum_projected_free;
        if (nd == 0) {
            global_surplus -= margins[0];
        } else {
            for (size_t id = 0; id < nd; id++) {
                global_surplus -= margins[id];
            }
        }
        if (global_surplus < 0) {
            if (nd <= 1) {
                LOG_TRC("%s: cannot meet free memory target of %" PRId64 " MiB, need to reduce device memory by %" PRId64 " MiB\n",
                    __func__, margins[0]/MiB, -global_surplus/MiB);
            } else {
                LOG_TRC(
                    "%s: cannot meet free memory targets on all devices, need to use %" PRId64 " MiB less in total\n",
                    __func__, -global_surplus/MiB);
            }
            if (cparams->n_ctx == 0) {
                if (hp_nct > n_ctx_min) {
                    int64_t sum_used_target = sum_free;
                    if (nd == 0) {
                        sum_used_target -= margins[0];
                    } else {
                        for (size_t id = 0; id < nd; id++) {
                            sum_used_target -= margins[id];
                        }
                    }
                    if (nd > 1) {
                        // for multiple devices we need to be more conservative in terms of how much context we think can fit:
                        //   - for dense models only whole layers can be assigned to devices
                        //   - for MoE models only whole tensors can be assigned to devices, which we estimate to be <= 1/3 of a layer
                        //   - on average we expect a waste of 0.5 layers/tensors per device
                        //   - use slightly more than the expected average for nd devices to be safe
                        const int64_t model_per_layer = sum_projected_model / std::min(uint32_t(mparams->n_gpu_layers), hp_ngl);
                        sum_used_target -= (nd + 1) * model_per_layer / (hp_nex == 0 ? 2 : 6);
                    }

                    int64_t sum_projected_used_min_ctx = 0;
                    cparams->n_ctx = n_ctx_min;
                    const dmds_t dmds_min_ctx = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
                    if (nd == 0) {
                        sum_projected_used_min_ctx = dmds_min_ctx.back().mb.total();
                    } else {
                        for (size_t id = 0; id < nd; id++) {
                            sum_projected_used_min_ctx += dmds_min_ctx[id].mb.total();
                        }
                    }
                    if (sum_used_target > sum_projected_used_min_ctx) {
                        // linear interpolation between minimum and maximum context size:
                        cparams->n_ctx += (hp_nct - n_ctx_min) * (sum_used_target - sum_projected_used_min_ctx)
                            / (sum_projected_used - sum_projected_used_min_ctx);
                        cparams->n_ctx = std::max(cparams->n_ctx - cparams->n_ctx % 256, n_ctx_min); // round down context for CUDA backend

                        const int64_t bytes_per_ctx = (sum_projected_used - sum_projected_used_min_ctx) / (hp_nct - n_ctx_min);
                        const int64_t memory_reduction = (hp_nct - cparams->n_ctx) * bytes_per_ctx;
                        LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                            __func__, hp_nct, cparams->n_ctx, memory_reduction/MiB);
                        if (nd <= 1) {
                            LOG_TRC("%s: entire model can be fit by reducing context\n", __func__);
                            return;
                        }
                        LOG_TRC("%s: entire model should be fit across devices by reducing context\n", __func__);
                    } else {
                        const int64_t memory_reduction = sum_projected_used - sum_projected_used_min_ctx;
                        LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                            __func__, hp_nct, cparams->n_ctx, memory_reduction/MiB);
                    }
                } else {
                    // Context was not reduced (either full context requested or
                    // model training context <= n_ctx_min).  Resolve n_ctx from
                    // the model so downstream code (e.g. recurrent target) sees
                    // a non-zero per-sequence context instead of 0.
                    cparams->n_ctx = hp_nct;
                    if (n_ctx_min == UINT32_MAX) {
                        LOG_TRC("%s: user has requested full context size of %" PRIu32 " -> no change\n", __func__, hp_nct);
                    } else {
                        LOG_TRC("%s: default model context size is %" PRIu32 " which is <= the min. context size of %" PRIu32 " -> no change\n",
                            __func__, hp_nct, n_ctx_min);
                    }
                }
            } else {
                LOG_TRC("%s: context size set by user to %" PRIu32 " -> no change\n", __func__, cparams->n_ctx);
            }
        }
    }
    if (nd == 0) {
        throw common_params_fit_exception("was unable to fit model into system memory by reducing context, abort");
    }

    if (mparams->n_gpu_layers != default_mparams.n_gpu_layers) {
        LOG_INF("%s: n_gpu_layers already set by user to %d; skipping parameter fitting\n", __func__, mparams->n_gpu_layers);
        return;
    }
    if (nd > 1) {
        if (!tensor_split) {
            throw common_params_fit_exception("did not provide a buffer to write the tensor_split to, abort");
        }
        if (mparams->tensor_split) {
            for (size_t id = 0; id < nd; id++) {
                if (mparams->tensor_split[id] != 0.0f) {
                    throw common_params_fit_exception("model_params::tensor_split already set by user, abort");
                }
            }
        }
        if (mparams->split_mode == LLAMA_SPLIT_MODE_ROW) {
            throw common_params_fit_exception("changing weight allocation for LLAMA_SPLIT_MODE_ROW not implemented, abort");
        }
    }
    if (!tensor_buft_overrides) {
        throw common_params_fit_exception("did not provide buffer to set tensor_buft_overrides, abort");
    }
    if (mparams->tensor_buft_overrides && (mparams->tensor_buft_overrides->pattern || mparams->tensor_buft_overrides->buft)) {
        throw common_params_fit_exception("model_params::tensor_buft_overrides already set by user, abort");
    }

    // step 3: iteratively fill the back to front with "dense" layers
    //   - for a dense model simply fill full layers, giving each device a contiguous slice of the model
    //   - for a MoE model, same as dense model but with all MoE tensors in system memory

    // utility function that returns a static C string matching the tensors for a specific layer index and layer fraction:
    auto get_overflow_pattern = [&](const size_t il, const common_layer_fraction_t lf) -> const char * {
        constexpr size_t n_strings = 1000;
        if (il >= n_strings) {
            throw std::runtime_error("at most " + std::to_string(n_strings) + " model layers are supported");
        }
        switch (lf) {
            case LAYER_FRACTION_ATTN: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|up|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_UP: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_GATE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_down.*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_MOE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(up|down|gate_up|gate)_(ch|)exps";
                }
                return patterns[il].c_str();
            }
            default:
                GGML_ABORT("fatal error");
        }
    };

    struct ngl_t {
        uint32_t n_layer = 0; // number of total layers
        uint32_t n_part  = 0; // number of partial layers, <= n_layer

        // for the first partial layer varying parts can overflow, all further layers use LAYER_FRACTION_MOE:
        common_layer_fraction_t overflow_type = LAYER_FRACTION_MOE;

        uint32_t n_full() const {
            assert(n_layer >= n_part);
            return n_layer - n_part;
        }
    };

    const size_t ntbo = llama_max_tensor_buft_overrides();

    // utility function to set n_gpu_layers and tensor_split
    auto set_ngl_tensor_split_tbo = [&](
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts,
            llama_model_params & mparams) {
        mparams.n_gpu_layers = 0;
        for (size_t id = 0; id < nd; id++) {
            mparams.n_gpu_layers += ngl_per_device[id].n_layer;
            if (nd > 1) {
                tensor_split[id] = ngl_per_device[id].n_layer;
            }
        }
        assert(uint32_t(mparams.n_gpu_layers) <= hp_ngl + 1);
        uint32_t il0 = hp_ngl + 1 - mparams.n_gpu_layers; // start index for tensor buft overrides

        mparams.tensor_split = tensor_split;

        size_t itbo = 0;
        for (size_t id = 0; id < nd; id++) {
            il0 += ngl_per_device[id].n_full();
            for (uint32_t il = il0; il < il0 + ngl_per_device[id].n_part; il++) {
                if (itbo + 1 >= ntbo) {
                    tensor_buft_overrides[itbo].pattern = nullptr;
                    tensor_buft_overrides[itbo].buft    = nullptr;
                    itbo++;
                    mparams.tensor_buft_overrides = tensor_buft_overrides;
                    throw common_params_fit_exception("llama_max_tensor_buft_overrides() == "
                        + std::to_string(ntbo) + " is insufficient for model");
                }
                tensor_buft_overrides[itbo].pattern = get_overflow_pattern(il, il == il0 ? ngl_per_device[id].overflow_type : LAYER_FRACTION_MOE);
                tensor_buft_overrides[itbo].buft = il == il0 ? overflow_bufts[id] : ggml_backend_cpu_buffer_type();
                itbo++;
            }
            il0 += ngl_per_device[id].n_part;
        }
        tensor_buft_overrides[itbo].pattern = nullptr;
        tensor_buft_overrides[itbo].buft    = nullptr;
        itbo++;
        mparams.tensor_buft_overrides = tensor_buft_overrides;
    };

    // utility function that returns the memory use per device for given numbers of layers per device
    auto get_memory_for_layers = [&](
            const char * func_name,
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts) -> std::vector<int64_t> {
        llama_model_params mparams_copy = *mparams;
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, mparams_copy);

        const dmds_t dmd_nl = common_get_device_memory_data_impl(
            path_model, &mparams_copy, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);

        LOG_TRC("%s: memory for test allocation by device:\n", func_name);
        for (size_t id = 0; id < nd; id++) {
            const ngl_t & n = ngl_per_device[id];
            LOG_TRC(
                "%s: id=%zu, n_layer=%2" PRIu32 ", n_part=%2" PRIu32 ", overflow_type=%d, mem=%6" PRId64 " MiB\n",
                func_name, id, n.n_layer, n.n_part, int(n.overflow_type), dmd_nl[id].mb.total()/MiB);
        }

        std::vector<int64_t> ret;
        ret.reserve(nd);
        for (size_t id = 0; id < nd; id++) {
            ret.push_back(dmd_nl[id].mb.total());
        }
        return ret;
    };

    int64_t global_surplus_cpu_moe = 0;
    if (hp_nex > 0) {
        const static std::string pattern_moe_all = "blk\\.\\d+\\.ffn_(up|down|gate_up|gate)_(ch|)exps"; // matches all MoE tensors
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
        tensor_buft_overrides[0] = {pattern_moe_all.c_str(), cpu_buft};
        tensor_buft_overrides[1] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;

        LOG_TRC("%s: getting device memory data with all MoE tensors moved to system memory:\n", __func__);
        const dmds_t dmds_cpu_moe = common_get_device_memory_data_impl(
            path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);

        for (size_t id = 0; id < nd; id++) {
            global_surplus_cpu_moe += dmds_cpu_moe[id].free;
            global_surplus_cpu_moe -= int64_t(dmds_cpu_moe[id].mb.total()) + margins[id];
        }

        if (global_surplus_cpu_moe > 0) {
            LOG_TRC("%s: with only dense weights in device memory there is a total surplus of %" PRId64 " MiB\n",
                __func__, global_surplus_cpu_moe/MiB);
        } else {
            LOG_TRC("%s: with only dense weights in device memory there is still a total deficit of %" PRId64 " MiB\n",
                __func__, -global_surplus_cpu_moe/MiB);
        }

        // reset
        tensor_buft_overrides[0] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;
    }

    std::vector<int64_t> targets; // maximum acceptable memory use per device
    targets.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        targets.push_back(dmds_full[id].free - margins[id]);
        LOG_TRC("%s: id=%zu, target=%" PRId64 " MiB\n", __func__, id, targets[id]/MiB);
    }

    std::vector<ggml_backend_buffer_type_t> overflow_bufts; // which bufts the first partial layer of a device overflows to:
    overflow_bufts.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        overflow_bufts.push_back(ggml_backend_cpu_buffer_type());
    }

    std::vector<ngl_t> ngl_per_device(nd);
    std::vector<int64_t> mem = get_memory_for_layers(__func__, ngl_per_device, overflow_bufts);

    // optimize the number of layers per device using the method of false position:
    //   - ngl_per_device has 0 layers for each device, lower bound
    //   - try a "high" configuration where a device is given all unassigned layers
    //   - interpolate the memory use / layer between low and high linearly to get a guess where it meets our target
    //   - check memory use of our guess, replace either the low or high bound
    //   - once we only have a difference of a single layer, stop and return the lower bound that just barely still fits
    //   - the last device has the output layer, which cannot be a partial layer
    if (hp_nex == 0) {
        LOG_TRC("%s: filling dense layers back-to-front:\n", __func__);
    } else {
        LOG_TRC("%s: filling dense-only layers back-to-front:\n", __func__);
    }
    for (int id = nd - 1; id >= 0; id--) {
        uint32_t n_unassigned = hp_ngl + 1;
        for (size_t jd = id + 1; jd < nd; ++jd) {
            assert(n_unassigned >= ngl_per_device[jd].n_layer);
            n_unassigned -= ngl_per_device[jd].n_layer;
        }

        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        ngl_per_device_high[id].n_layer = n_unassigned;
        if (hp_nex > 0) {
            ngl_per_device_high[id].n_part = size_t(id) < nd - 1 ? ngl_per_device_high[id].n_layer : ngl_per_device_high[id].n_layer - 1;
        }
        if (ngl_per_device_high[id].n_layer > 0) {
            std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);
            if (mem_high[id] > targets[id]) {
                assert(ngl_per_device_high[id].n_layer > ngl_per_device[id].n_layer);
                uint32_t delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                LOG_TRC("%s: start filling device %" PRIu32 ", delta=%" PRIu32 "\n", __func__, id, delta);
                while (delta > 1) {
                    uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                    step_size = std::max(step_size, uint32_t(1));
                    step_size = std::min(step_size, delta - 1);

                    std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                    ngl_per_device_test[id].n_layer += step_size;
                    if (hp_nex) {
                        ngl_per_device_test[id].n_part += size_t(id) == nd - 1 && ngl_per_device_test[id].n_part == 0 ?
                            step_size - 1 : step_size; // the first layer is the output layer which must always be full
                    }
                    const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                    if (mem_test[id] <= targets[id]) {
                        ngl_per_device = ngl_per_device_test;
                        mem            = mem_test;
                        LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
                    } else {
                        ngl_per_device_high = ngl_per_device_test;
                        mem_high            = mem_test;
                        LOG_TRC("%s: set ngl_per_device_high[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device_high[id].n_layer);
                    }
                    delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                }
            } else {
                assert(ngl_per_device_high[id].n_layer == n_unassigned);
                ngl_per_device = ngl_per_device_high;
                mem            = mem_high;
                LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers, %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, mem[id]/MiB, projected_margin/MiB);
    }
    if (hp_nex == 0 || global_surplus_cpu_moe <= 0) {
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
        return;
    }

    // step 4: for a MoE model where all dense tensors fit,
    //     convert the dense-only layers in the back to full layers in the front until all devices are full
    // essentially the same procedure as for the dense-only layers except front-to-back
    // also, try fitting at least part of one more layer to reduce waste for "small" GPUs with e.g. 24 GiB VRAM

    size_t id_dense_start = nd;
    for (int id = nd - 1; id >= 0; id--) {
        if (ngl_per_device[id].n_layer > 0) {
            id_dense_start = id;
            continue;
        }
        break;
    }
    assert(id_dense_start < nd);

    LOG_TRC("%s: converting dense-only layers to full layers and filling them front-to-back with overflow to next device/system memory:\n", __func__);
    for (size_t id = 0; id <= id_dense_start && id_dense_start < nd; id++) {
        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        for (size_t jd = id_dense_start; jd < nd; jd++) {
            const uint32_t n_layer_move = jd < nd - 1 ? ngl_per_device_high[jd].n_layer : ngl_per_device_high[jd].n_layer - 1;
            ngl_per_device_high[id].n_layer += n_layer_move;
            ngl_per_device_high[jd].n_layer -= n_layer_move;
            ngl_per_device_high[jd].n_part = 0;
        }
        size_t id_dense_start_high = nd - 1;
        std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);

        if (mem_high[id] > targets[id]) {
            assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
            uint32_t delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            while (delta > 1) {
                uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                step_size = std::max(step_size, uint32_t(1));
                step_size = std::min(step_size, delta - 1);

                std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                size_t id_dense_start_test = id_dense_start;
                uint32_t n_converted_test = 0;
                for (;id_dense_start_test < nd; id_dense_start_test++) {
                    const uint32_t n_convert_jd = std::min(step_size - n_converted_test, ngl_per_device_test[id_dense_start_test].n_part);
                    ngl_per_device_test[id_dense_start_test].n_layer -= n_convert_jd;
                    ngl_per_device_test[id_dense_start_test].n_part -= n_convert_jd;
                    ngl_per_device_test[id].n_layer += n_convert_jd;
                    n_converted_test += n_convert_jd;

                    if (ngl_per_device_test[id_dense_start_test].n_part > 0) {
                        break;
                    }
                }
                const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                if (mem_test[id] <= targets[id]) {
                    ngl_per_device = ngl_per_device_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                } else {
                    ngl_per_device_high = ngl_per_device_test;
                    mem_high            = mem_test;
                    id_dense_start_high = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device_high[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start_high=%zu\n",
                        __func__, id, ngl_per_device_high[id].n_layer, ngl_per_device_high[id].n_part, id_dense_start_high);
                }
                assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
                delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            }
        } else {
            ngl_per_device = ngl_per_device_high;
            mem            = mem_high;
            id_dense_start = id_dense_start_high;
            LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
        }

        // try to fit at least part of one more layer
        if (ngl_per_device[id_dense_start].n_layer > (id < nd - 1 ? 0 : 1)) {
            std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
            size_t id_dense_start_test = id_dense_start;
            ngl_per_device_test[id_dense_start_test].n_layer--;
            ngl_per_device_test[id_dense_start_test].n_part--;
            ngl_per_device_test[id].n_layer++;
            ngl_per_device_test[id].n_part++;
            if (ngl_per_device_test[id_dense_start_test].n_part == 0) {
                id_dense_start_test++;
            }
            ngl_per_device_test[id].overflow_type = LAYER_FRACTION_UP;
            std::vector<ggml_backend_buffer_type_t> overflow_bufts_test = overflow_bufts;
            if (id < nd - 1) {
                overflow_bufts_test[id] = ggml_backend_dev_buffer_type(devs[id + 1]);
            }
            LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_UP\n", __func__);
            std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
            if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                ngl_per_device = ngl_per_device_test;
                overflow_bufts = overflow_bufts_test;
                mem            = mem_test;
                id_dense_start = id_dense_start_test;
                LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", UP), id_dense_start=%zu\n",
                    __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);

                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_GATE;
                LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_GATE\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", GATE), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            } else {
                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_ATTN;
                LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_ATTN\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", ATTN), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    // print info for devices that were not changed during the conversion from dense only to full layers:
    for (size_t id = id_dense_start + 1; id < nd; id++) {
        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
}

enum common_params_fit_status common_fit_params(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams,
        float * tensor_split,
        llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins,
        uint32_t n_ctx_min,
        ggml_log_level log_level) {
    const int64_t t0_us = llama_time_us();
    common_params_fit_status status = COMMON_PARAMS_FIT_STATUS_SUCCESS;
    try {
        common_params_fit_impl(path_model, mparams, cparams, tensor_split, tensor_buft_overrides, margins, n_ctx_min, log_level);
        LOG_TRC("%s: successfully fit params to free device memory\n", __func__);
    } catch (const common_params_fit_exception & e) {
        LOG_WRN("%s: failed to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_FAILURE;
    } catch (const std::runtime_error & e) {
        LOG_ERR("%s: encountered an error while trying to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_ERROR;
    }
    const int64_t t1_us = llama_time_us();
    LOG_TRC("%s: fitting params to free memory took %.2f seconds\n", __func__, (t1_us - t0_us) * 1e-6);
    return status;
}

void common_memory_breakdown_print(const struct llama_context * ctx) {
    //const auto & devices = ctx->get_model().devices;
    const auto * model = llama_get_model(ctx);

    std::vector<ggml_backend_dev_t> devices;
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devices.push_back(llama_model_get_device(model, i));
    }

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    std::vector<std::array<std::string, 9>> table_data;
    table_data.reserve(devices.size());
    const std::string template_header = "%s: | %s | %s   %s    %s   %s   %s   %s    %s |\n";
    const std::string template_gpu    = "%s: | %s | %s = %s + (%s = %s + %s + %s) + %s |\n";
    const std::string template_other  = "%s: | %s | %s   %s    %s = %s + %s + %s    %s |\n";

    table_data.push_back({template_header, "memory breakdown [MiB]", "total", "free", "self", "model", "context", "compute", "unaccounted"});

    constexpr size_t MiB = 1024 * 1024;
    const std::vector<std::string> desc_prefixes_strip = {"NVIDIA ", "GeForce ", "Tesla ", "AMD ", "Radeon ", "Instinct "};

    // track seen buffer types to avoid double counting:
    std::set<ggml_backend_buffer_type_t> seen_buffer_types;

    // accumulative memory breakdown for each device and for host:
    std::vector<llama_memory_breakdown_data> mb_dev(devices.size());
    llama_memory_breakdown_data              mb_host;

    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (ggml_backend_buft_is_host(buft)) {
            mb_host.model   += mb.model;
            mb_host.context += mb.context;
            mb_host.compute += mb.compute;
            seen_buffer_types.insert(buft);
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev) {
            int i_dev = -1;
            for (size_t i = 0; i < devices.size(); i++) {
                if (devices[i] == dev) {
                    i_dev = i;
                    break;
                }
            }
            if (i_dev != -1) {
                mb_dev[i_dev].model   += mb.model;
                mb_dev[i_dev].context += mb.context;
                mb_dev[i_dev].compute += mb.compute;
                seen_buffer_types.insert(buft);
                continue;
            }
        }
    }

    // print memory breakdown for each device:
    for (size_t i = 0; i < devices.size(); i++) {
        ggml_backend_dev_t dev = devices[i];
        llama_memory_breakdown_data mb = mb_dev[i];

        const std::string name = ggml_backend_dev_name(dev);
        std::string desc = ggml_backend_dev_description(dev);
        for (const std::string & prefix : desc_prefixes_strip) {
            if (desc.length() >= prefix.length() && desc.substr(0, prefix.length()) == prefix) {
                desc = desc.substr(prefix.length());
            }
        }

        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);

        const size_t self = mb.model + mb.context + mb.compute;
        const int64_t unaccounted = static_cast<int64_t>(total) - static_cast<int64_t>(free) - static_cast<int64_t>(self);

        table_data.push_back({
            template_gpu,
            "  - " + name + " (" + desc + ")",
            std::to_string(total / MiB),
            std::to_string(free / MiB),
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            std::to_string(unaccounted / static_cast<int64_t>(MiB))});
    }

    // print memory breakdown for host:
    {
        const size_t self = mb_host.model + mb_host.context + mb_host.compute;
        table_data.push_back({
            template_other,
            "  - Host",
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb_host.model / MiB),
            std::to_string(mb_host.context / MiB),
            std::to_string(mb_host.compute / MiB),
            ""}); // unaccounted
    }

    // print memory breakdown for all remaining buffer types:
    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (seen_buffer_types.count(buft) == 1) {
            continue;
        }
        const std::string name = ggml_backend_buft_name(buft);
        const size_t self = mb.model + mb.context + mb.compute;
        table_data.push_back({
            template_other,
            "  - " + name,
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            ""}); // unaccounted
        seen_buffer_types.insert(buft);
    }

    for (size_t j = 1; j < table_data[0].size(); j++) {
        size_t max_len = 0;
        for (const auto & td : table_data) {
            max_len = std::max(max_len, td[j].length());
        }
        for (auto & td : table_data) {
            td[j].insert(j == 1 ? td[j].length() : 0, max_len - td[j].length(), ' ');
        }
    }
    for (const auto & td : table_data) {
        LOG_TRC(td[0].c_str(),
            __func__, td[1].c_str(), td[2].c_str(), td[3].c_str(), td[4].c_str(), td[5].c_str(),
            td[6].c_str(), td[7].c_str(), td[8].c_str());
    }
}


// Working margin per device for the automatic KV sizing: what must stay free for the
// driver's own allocations and transients. Measured on the 5x RX 6800 box - below the
// driver's eviction watermark it migrates pages to system memory and throughput collapses:
// free ~2.0-2.6 GiB -> 21-22 tok/s, free ~0.25 GiB -> 1.85 tok/s (9 GiB migrated per card).
// Measured on the 5x RX 6800 box: at a 393216-token pool the loosest card keeps ~430 MiB free
// while the tightest is over by ~264 MiB - and the driver migrates whole buffers, so that
// shortfall costs two ~800 MiB KV layers pushed into system memory (GTT 1569 MiB on that card,
// 14 MiB on the others). The margin has to cover the migration granularity, not just the
// arithmetic difference: 2 GiB leaves every card room for another layer-sized buffer.
static constexpr uint64_t FIT_KV_MARGIN = 2ull << 30; // 2 GiB

common_params_fit_status common_fit_kv_cache(
                         const char * path_model,
           const llama_model_params * mparams,
               llama_context_params * cparams,
        const std::vector<std::pair<ggml_backend_dev_t, size_t>> & reserve,
        const llama_context_params * extra_cparams,
                     ggml_log_level   log_level) {
    if (!cparams->kv_unified) {
        LOG_WRN("%s: automatic KV sizing requires unified KV cache\n", __func__);
        return COMMON_PARAMS_FIT_STATUS_FAILURE;
    }
    if (!cparams->offload_kqv) {
        LOG_WRN("%s: automatic KV sizing requires device-backed KV cache\n", __func__);
        return COMMON_PARAMS_FIT_STATUS_FAILURE;
    }

    constexpr uint32_t n_align = 256;
    constexpr uint32_t n_max = UINT32_MAX - (UINT32_MAX % n_align);

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0;
    uint32_t hp_n_ctx_train = 0;
    uint32_t hp_n_expert = 0;

    // FlashPrefill native-split dims, resolved once (metadata-only load, enabled-only).
    // OFF: no load, fp_nat_have=false, probes add zero.
    common_fp_dims fp_nat_dims;
    const bool fp_nat_have = common_fp_load_dims(path_model, mparams, cparams, fp_nat_dims);
    // XKV reserve is computed exactly once per fit invocation (not per
    // probe). workspace_mib is the TOTAL transient budget (once); the exact
    // seal scratch must fit the remainder after the decode sub-budget.
    // OFF yields zeros and changes nothing.
    uint64_t xkv_scratch_once = 0;
    uint64_t xkv_dedup_once = 0;
    if (llama_xkv_is_enabled(cparams->xkv_mode)) {
        if (!common_xkv_scratch_bytes(path_model, mparams, cparams, &xkv_scratch_once, &xkv_dedup_once)) {
            LOG_WRN("%s: XKV scratch estimation failed; degrading scratch reserve to 0\n", __func__);
            xkv_scratch_once = 0;
            xkv_dedup_once = 0;
        }
    }
    common_xkv_fit_reserve xkv_res = {};
    if (!common_xkv_fit_reserve_bytes(cparams, xkv_scratch_once, &xkv_res, xkv_dedup_once)) {
        LOG_WRN("%s: XKV reserve accounting failed (overflow or scratch exceeds workspace-decode), aborting auto-fit\n", __func__);
        return COMMON_PARAMS_FIT_STATUS_FAILURE;
    }
    if (xkv_res.total_bytes != 0) {
        LOG_INF("%s: XKV auto-fit reserve: workspace=%llu (once) decode_cache=%llu (inside) factor_scratch=%llu (inside) dedup_scratch=%llu (host)\n",
            __func__,
            (unsigned long long) xkv_res.workspace_bytes,
            (unsigned long long) xkv_res.decode_cache_bytes,
            (unsigned long long) xkv_res.factor_scratch_bytes,
            (unsigned long long) xkv_res.dedup_scratch_bytes);
    }

    // Persistent factor store budget (explicit configuration only) is
    // accounted separately from transient seal-time peaks above: it caps
    // persistent device bytes, not scratch. Derived budgets resolve after
    // auto-fit and are capped at the dense equivalent, so they add nothing.
    const uint64_t xkv_store_once_bytes = common_xkv_store_budget_bytes(cparams);
    if (xkv_store_once_bytes == UINT64_MAX) {
        LOG_WRN("%s: XKV store budget accounting overflow, aborting auto-fit\n", __func__);
        return COMMON_PARAMS_FIT_STATUS_FAILURE;
    }
    if (xkv_store_once_bytes != 0) {
        LOG_INF("%s: XKV persistent store reserve: %llu bytes (once per device, separate from transient peaks)\n",
            __func__, (unsigned long long) xkv_store_once_bytes);
    }

    // Device-owned store share from the effective residency predicate: the
    // reference profile and every cpu-reference TQ configuration stay
    // host-resident (charged once below under unlimited-host fit); only TQ
    // profiles on device factorizers are device-owned and partitioned here.
    const uint64_t xkv_store_dev_total = common_xkv_store_device_bytes(cparams, xkv_store_once_bytes);
    {
        uint64_t host_charge = 0;
        // Host bucket: store-owned dedup vectors plus the host share of the
        // persistent store. Device xkv_global below carries only workspace +
        // device store (no double count; dedup never lives in the arena).
        uint64_t host_store = 0;
        if (xkv_checked_add_u64(xkv_res.dedup_scratch_bytes,
                xkv_store_once_bytes - xkv_store_dev_total, host_store) &&
            xkv_checked_add_u64(xkv_res.total_bytes, host_store, host_charge)) {
            LOG_INF("%s: XKV host-charged (once, unlimited-host fit): %llu bytes\n",
                __func__, (unsigned long long) host_charge);
        }
    }

    // Budget base for every projection: the free memory is read once, before any context is
    // built. The driver keeps its pooled device allocations after a probe's context is
    // freed, so a live reading shrinks probe by probe and later ones reject a configuration
    // that the runtime fits (measured: used 14.7 GiB against a shrunken free 12.3 GiB).
    std::vector<int64_t> free_base;

    auto get_data = [&](uint32_t n_ctx_kv, common_device_memory_data_vec & data, uint32_t n_seq_max_override = 0) {
        llama_context_params test = *cparams;
        test.n_ctx_kv = n_ctx_kv;
        // n_seq_max override: the server's placeholder slot count
        // (llama_max_parallel_sequences) makes every probe unusable for a hybrid model,
        // because the recurrent state alone then costs more than the device has. The
        // solve below probes one and two slots at the real capacity instead.
        if (n_seq_max_override > 0) {
            test.n_seq_max = n_seq_max_override;
        }
        try {
            // no_alloc probes only: allocating the model buffers per probe (52 GiB of device
            // memory plus the host side, once per step of the search) exhausted RAM and took
            // the whole machine down (system-wide OOM, then a panic reboot). The decision
            // below compensates by not booking the compute reserve, which is what made the
            // allocated probe look necessary in the first place.
            data = common_get_device_memory_data(path_model, mparams, &test, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level);
        } catch (const std::exception & e) {
            LOG_WRN("%s: KV size %u probe threw exception: %s\n", __func__, n_ctx_kv, e.what());
            return false;
        }

        if (devs.empty()) {
            return false;
        }


        {
            uint64_t ctx_sum = 0;
            for (size_t i = 0; i < data.size(); ++i) {
                ctx_sum += data[i].context;
            }
            LOG_INF("%s: probe n_ctx_kv=%u -> context %llu MiB over %zu entries\n",
                    __func__, n_ctx_kv, (unsigned long long) (ctx_sum >> 20), data.size());
            for (size_t i = 0; i < data.size(); ++i) {
                LOG_INF("%s:   entry %zu: context %llu MiB, model %llu MiB, compute %llu MiB\n",
                        __func__, i, (unsigned long long) (data[i].context >> 20),
                        (unsigned long long) (data[i].model >> 20),
                        (unsigned long long) (data[i].compute >> 20));
            }
        }

        common_device_memory_data_vec extra_data;
        std::vector<ggml_backend_dev_t> extra_devs;
        if (extra_cparams != nullptr) {
            llama_context_params extra = *extra_cparams;
            // MTP/draft contexts use the same n_ctx_kv as the target — the n_ctx_kv >= n_ctx
            // check requires it. However, the MTP context's KV cache only covers the MTP head
            // layers (via layer filter), so its actual memory is much smaller than the target's.
            extra.n_ctx_kv = n_ctx_kv;
            uint32_t extra_ngl = 0;
            uint32_t extra_n_ctx_train = 0;
            uint32_t extra_n_expert = 0;
            try {
                extra_data = common_get_device_memory_data(
                    path_model, mparams, &extra, extra_devs,
                    extra_ngl, extra_n_ctx_train, extra_n_expert, log_level);
            } catch (const std::exception & e) {
                LOG_WRN("%s: extra context KV size %u probe threw exception: %s\n", __func__, n_ctx_kv, e.what());
                return false;
            }
        }

        uint64_t xkv_global = 0;
        std::vector<uint64_t> xkv_wts(devs.size(), 0);
        std::vector<uint64_t> xkv_shares(devs.size(), 0);
        for (size_t i = 0; i < devs.size(); ++i) {
            xkv_wts[i] = (uint64_t) data[i].context;
        }
        if (!xkv_checked_add_u64(xkv_res.total_bytes, xkv_store_dev_total, xkv_global) ||
            !common_xkv_partition_budget(xkv_global, xkv_wts.data(), xkv_shares.data(), xkv_shares.size())) {
            return false;
        }

        for (size_t i = 0; i < devs.size(); ++i) {
            size_t reserved = 0;
            for (const auto & [dev, bytes] : reserve) {
                if (dev == devs[i]) {
                    reserved += bytes;
                }
            }

            // FlashPrefill: graph caps arrive via data[i].compute (no manual reserve,
            // no double charge); the backend-private split pool does not, so its
            // worst-case peak is added here exactly once per device (shared formula).
            if (fp_nat_have) {
                uint64_t nat_bytes = 0;
                if (!common_fp_native_split_bytes(&cparams->flashprefill, &fp_nat_dims, cparams->n_ubatch, n_ctx_kv, devs[i], nat_bytes)) {
                    LOG_TRC("%s: KV size %u rejected: native split shape unsizable\n", __func__, n_ctx_kv);
                    return false;
                }
                reserved += (size_t) std::min<uint64_t>(nat_bytes, (uint64_t) SIZE_MAX);
            }

            // Vulkan TriAttention compaction uses one reusable 8 MiB
            // device-local memmove scratch buffer per Vulkan device. Keep
            // this in sync with GGML_VK_MEMMOVE_SCRATCH_SIZE so --total-kv
            // auto never consumes the memory that native pack needs later.
            if (cparams->triattention) {
                ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(devs[i]);
                const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
                if (reg_name && std::string(reg_name).find("Vulkan") != std::string::npos) {
                    // Keep in sync with GGML_VK_MEMMOVE_SCRATCH_SIZE
                    reserved += 8ull * 1024ull * 1024ull;
                } else if (reg_name && std::string(reg_name).find("CUDA") != std::string::npos) {
                    // Keep in sync with GGML_CUDA_MEMMOVE_SCRATCH_SIZE
                    reserved += 8ull * 1024ull * 1024ull;
                }
            }

            // RERoT span-table/DDVR phase inputs, frontier query rows,
            // parked-recurrent metadata descriptors and episode host metadata
            // (KV/recurrent payloads are sized by the surrounding fit paths,
            // never double-counted here). Keep in sync with
            // common_rerot_scratch_reserve_bytes() in common/common.h.
            if (cparams->rerot) {
                reserved += 8ull * 1024ull * 1024ull;
            }

            for (size_t j = 0; j < extra_devs.size(); ++j) {
                if (extra_devs[j] == devs[i]) {
                    reserved += extra_data[j].context;
                }
            }

            constexpr uint64_t MiB = 1024ull * 1024ull;
            const uint64_t runtime_headroom = std::min<uint64_t>(
                256ull * MiB, std::max<uint64_t>(32ull * MiB, data[i].total / 100ull));
            // XKV reserve is accounted exactly once per device evaluation
            // (not once per reserve entry): configured workspace, decode
            // cache, and factor scratch headroom for seal/factorize peaks.
            // Global XKV budgets are partitioned by target-layer placement
            // (per-device context bytes) so device shares sum to exactly the
            // total instead of over-reserving N× on N devices.
            // Partitioned global is workspace (once, scratch inside) plus the
            // device-owned store share; scratch is never added outside.
            uint64_t reserved_all = 0;
            if (!xkv_checked_add_u64(reserved, xkv_shares[i], reserved_all)) {
                LOG_WRN("%s: KV size %u does not fit: XKV share accounting overflow\n",
                    __func__, n_ctx_kv);
                return false;
            }
            uint64_t used = 0;
            if (!xkv_checked_sum_fit_bytes(data[i].model, data[i].context, data[i].compute,
                    reserved_all, runtime_headroom, used)) {
                return false;
            }
            uint64_t used_decision = 0;
            // The graph/compute term is measured here, not reserved: the probe allocates and
            // decodes, so data[i].compute is what the configuration really takes. Leaving it out
            // let a candidate through that pushed 30 GiB into system memory before the guard
            // killed the process. runtime_headroom stays out: the free number already contains
            // the driver's footprint, and it stays visible in the logged total.
            // The model term is zero on purpose: the free reading is taken with the probe's model
            // resident (measured: 7.5 GiB free at probe time on a card whose weights are 10 GiB),
            // so adding it again is double counting and rejects configurations the box runs.
            if (!xkv_checked_sum_fit_bytes(0, data[i].context, data[i].compute,
                    reserved_all, 0, used_decision)) {
                return false;
            }
            const int64_t free_i = i < free_base.size() ? free_base[i] : data[i].free;
            // Transient headroom, measured: a live run of exactly this configuration settles with
            // ~200 MiB free per card, so a decision that demands the last byte rejects a plan the
            // machine actually runs. This is the only allowance - it is not a percentage and not a
            // guess, and anything larger hides a real overcommit.
            constexpr uint64_t transient_slack = 512ull * MiB;
            if (free_i <= 0 || used_decision > (uint64_t) free_i + transient_slack) {
                LOG_WRN("%s: KV size %u does not fit on [%llu] %s: model=%llu context=%llu compute=%llu reserved=%llu headroom=%llu used=%llu free=%lld\n",
                    __func__, n_ctx_kv, (unsigned long long) i, ggml_backend_dev_name(devs[i]),
                    (unsigned long long)data[i].model, (unsigned long long)data[i].context,
                    (unsigned long long)data[i].compute, (unsigned long long)reserved_all,
                    (unsigned long long)runtime_headroom, (unsigned long long)used,
                    (long long) (i < free_base.size() ? free_base[i] : data[i].free));
                return false;
            }
        }

        return true;
    };

    common_device_memory_data_vec data;
    llama_context_params baseline = *cparams;
    baseline.n_ctx_kv = 0;
    // one slot as well: with the caller's placeholder (llama_max_parallel_sequences()) the
    // probe really allocates the recurrent state for that many sequences (~9 GiB here) and
    // the free memory it reports is what is left *after* that - the budget base has to be
    // read from a clean device.
    baseline.n_seq_max = 1;
    try {
        // Dry run of the production loader as the budget measurement: buffers are reserved but
        // never filled (load mode NONE), one decode allocates the graph/compute buffers, and the
        // probe reports the real free memory plus the measured cost, releasing both again. This
        // is the only probe that allocates; the search below stays on no_alloc probes.
        data = common_get_device_memory_data(path_model, mparams, &baseline, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level, /*no_alloc=*/false);
    } catch (const std::exception & e) {
        LOG_WRN("%s: failed to inspect device memory: %s\n", __func__, e.what());
        return COMMON_PARAMS_FIT_STATUS_ERROR;
    }

    if (devs.empty()) {
        LOG_WRN("%s: no device memory available for automatic KV sizing\n", __func__);
        return COMMON_PARAMS_FIT_STATUS_FAILURE;
    }

    free_base.assign(devs.size(), 0);
    // Fallback only: the pre-probe reading above is the reference. This fills in devices whose
    // kernel reading failed, and never overwrites a value that was measured on an idle card.
    // The budget base must be read with nothing allocated. The dry probe above allocates and
    // frees the whole model, but the driver keeps those pages in its own pool instead of
    // returning them to the kernel, so a base read from inside the probe is optimistic by
    // exactly that amount (measured: the real load then ran 1-2 GiB short per device and the
    // driver migrated that much to system memory). The kernel numbers at process start are
    // the ones the real load will see.
    for (size_t i = 0; i < devs.size(); ++i) {
        uint64_t vram_free  = 0;
        uint64_t vram_total = 0;
        if (common_fit_device_vram(devs[i], vram_free, vram_total) &&
            (free_base[i] == 0 || vram_free > (uint64_t) free_base[i])) {
            free_base[i] = (int64_t) vram_free;
        }
    }

    // No device-side GTT check here. GTT is not the signal: the CPU-offloaded tensors live in it
    // by design (~5 GiB per card for per_layer_token_embd) and the driver keeps its freed device
    // buffers in its own pool rather than returning them, so a run that is perfectly healthy
    // shows tens of GiB of GTT (measured: 19.7 GiB of GTT with Shmem flat at 11 MiB). What only
    // ever happens when the system is really out of memory is Shmem and MemAvailable moving - the
    // 34 GiB of Shmem that took the machine down earlier.
    const auto gtt_spilled = [&]() {
        // The device is only half the picture: the load path can also fill host memory with
        // spilled pages. Measured on this box: 34 GiB of Shmem and an exhausted swap, after which
        // every further GPU allocation failed and the machine went down. A clean box sits at a few
        // MiB of Shmem with most of RAM available, so either of these means the candidate is
        // costing the system memory and must not be probed further.
        FILE * f = fopen("/proc/meminfo", "r");
        if (f != nullptr) {
            char line[256];
            uint64_t shmem_kb = 0, avail_kb = 0;
            while (fgets(line, sizeof(line), f) != nullptr) {
                if (sscanf(line, "Shmem: %" SCNu64 " kB", &shmem_kb) == 1) {
                    continue;
                }
                sscanf(line, "MemAvailable: %" SCNu64 " kB", &avail_kb);
            }
            fclose(f);
            if (shmem_kb > (2ull << 20) || (avail_kb > 0 && avail_kb < (8ull << 20))) {
                LOG_WRN("%s: host pressure (Shmem %llu MiB, MemAvailable %llu MiB) - aborting\n",
                        __func__, (unsigned long long) (shmem_kb >> 10), (unsigned long long) (avail_kb >> 10));
                return true;
            }
        }
        return false;
    };
    if (gtt_spilled()) {
        return COMMON_PARAMS_FIT_STATUS_FAILURE;
    }

    // Resolve the per-sequence context from the model's training context when the
    // user did not specify -c.  This mirrors llama_context's own resolution and
    // makes the value visible to common_dynamic_recurrent_target after we return.
    if (cparams->n_ctx == 0) {
        cparams->n_ctx = hp_n_ctx_train;
    }
    const uint32_t n_ctx_seq = cparams->n_ctx;
    if (n_ctx_seq > n_max) {
        LOG_WRN("%s: per-sequence context of %u tokens exceeds the maximum aligned KV capacity\n", __func__, n_ctx_seq);
        return COMMON_PARAMS_FIT_STATUS_FAILURE;
    }

    // When TriAttention is enabled, the physical KV only needs to cover the
    // configured residency floor plus operational overhead (recent window + ubatch),
    // not the full per-sequence context. This allows physical KV << logical context.
    uint32_t n_min;
    if (cparams->triattention) {
        const uint32_t tri_floor = (uint32_t) std::ceil((double) n_ctx_seq * cparams->triattention_ratio);
        // operational_floor = recent_window(128) + n_ubatch, aligned
        const uint32_t op_floor = cparams->n_ubatch + 128;
        n_min = std::max<uint32_t>(n_align, (uint32_t) (((uint64_t) std::max(tri_floor, op_floor) + n_align - 1) / n_align * n_align));
        LOG_INF("%s: TriAttention enabled, minimum physical KV = %u (tri_floor=%u, op_floor=%u, n_ctx_seq=%u)\n",
                __func__, n_min, tri_floor, op_floor, n_ctx_seq);
    } else {
        n_min = std::max<uint32_t>(n_align, (uint32_t) (((uint64_t) n_ctx_seq + n_align - 1) / n_align * n_align));
    }

    // np=1 for the first probe: the caller's slot count is a placeholder in the auto case
    // (llama_max_parallel_sequences()), whose recurrent state alone exceeds the device on a
    // hybrid model, so without this override every fit rejects the requested context before
    // the solve below can even estimate the real slot ceiling.
    // The per-sequence context is a hard requirement (it is the context every slot must be able
    // to reach, i.e. n_ctx_seq), so it is never traded away here. The number of slots is what the
    // solve below spends the remaining memory on, with n_ctx_seq/2 per slot as the average floor.

    // Budget base, read now, before anything is loaded: the cards are idle here. A reading taken
    // after a probe picks up the driver's pooled leftovers, and every later candidate is then
    // rejected against a shrunken number (measured: 12.3 GiB free against a real 17.1 GiB).
    free_base.assign(devs.size(), 0);
    for (size_t i = 0; i < devs.size(); ++i) {
        uint64_t vram_free  = 0;
        uint64_t vram_total = 0;
        if (common_fit_device_vram(devs[i], vram_free, vram_total)) {
            free_base[i] = (int64_t) vram_free;
        }
    }

    if (!get_data(n_min, data, /*n_seq_max_override=*/1)) {
        LOG_WRN("%s: requested per-sequence context of %u tokens does not fit in device memory\n", __func__, n_min);
        return COMMON_PARAMS_FIT_STATUS_FAILURE;
    }

    const uint32_t n_probe = (uint32_t) std::min<uint64_t>(n_max, std::max<uint64_t>((uint64_t) n_min * 2, (uint64_t) n_min + 4096));
    common_device_memory_data_vec data_probe;
    if (n_probe > n_min) {
        // The slot count has to match the pool: a pool of two per-sequence contexts declared as a
        // single sequence is not a configuration anyone runs, and the hybrid KV sizes its
        // per-sequence bookkeeping from it (measured: each card reported 12 to 19 GiB of context
        // for that inconsistent probe, against 0.6 GiB for the consistent one).
        if (get_data(n_probe, data_probe, /*n_seq_max_override=*/(n_probe + n_min - 1) / n_min)) {
            bool device_context_grows = false;
            for (size_t i = 0; i < devs.size(); ++i) {
                device_context_grows |= data_probe[i].context > data[i].context;
            }
            if (!device_context_grows) {
                LOG_WRN("%s: no device-backed KV memory growth detected; automatic KV sizing is not applicable\n", __func__);
                return COMMON_PARAMS_FIT_STATUS_FAILURE;
            }
        }
    }

    // ---- slots and pool ------------------------------------------------------------------
    // np=1 has been probed above. Estimate the slot ceiling from the probed slopes, probe at
    // that ceiling, and then interpolate (false position) between the last fitting and the
    // first failing np until the bracket closes - every step stays inside the bracket.
    //   k_i    = (context(2*n_min) - context(n_min)) / n_min   KV bytes per token
    //   slot_i = free(1 slot) - free(2 slots)                  per-slot state
    // The pool must keep every slot at `target` tokens on average:
    //   pool(np) = n_min + (room_i - (np-1)*slot_i)/k_i >= np*target
    auto slack_of = [&devs, &free_base](const common_device_memory_data_vec & d) {
        int64_t s = INT64_MAX;
        for (size_t i = 0; i < devs.size() && i < d.size(); ++i) {
            // free is read while the probe's model is resident, so the model is already counted in
            // it; what a candidate still has to fit is its own KV plus its graph/compute buffers.
            const int64_t free_i = i < free_base.size() ? free_base[i] : d[i].free;
            const int64_t si = free_i - (int64_t) (d[i].context + d[i].compute);
            s = std::min(s, si);
        }
        return s;
    };

    if (n_probe > n_min && !data_probe.empty()) {
        common_device_memory_data_vec data_np2;
        if (get_data(n_min, data_np2, /*n_seq_max_override=*/2)) {
            const uint32_t target = std::max<uint32_t>(n_align, n_ctx_seq / 2);
            // Everything below is computed from the probes: k_i (KV bytes per token) from the
            // two capacities, slot_i (per-slot state) from the two slot counts, room_i from
            // the free memory the kernel reports for the one-slot probe. From those, for any
            // slot count the maximum KV capacity is a function
            //
            //     max_kv(np) = min_i ( room_i - (np - 1) * slot_i ) / k_i + n_min
            //
            // and the question is the largest np whose slots all still keep `target` tokens:
            //     max_kv(np) >= np * target
            const auto kv_of = [&](size_t i) -> uint64_t {
                const uint64_t bytes = data_probe[i].context > data[i].context ?
                    (uint64_t) (data_probe[i].context - data[i].context) : 0;
                return bytes / (n_probe - n_min);
            };
            const auto slot_of = [&](size_t i) -> uint64_t {
                return (uint64_t) std::max<int64_t>(data[i].free - data_np2[i].free, 0);
            };
            const auto room_of = [&](size_t i) -> uint64_t {
                const int64_t free_i = i < free_base.size() ? free_base[i] : data[i].free;
                return (uint64_t) free_i > FIT_KV_MARGIN ? (uint64_t) free_i - FIT_KV_MARGIN : 0;
            };
            const auto max_kv_of = [&](uint32_t np) -> uint64_t {
                uint64_t best = 0;
                bool any = false;
                for (size_t i = 0; i < devs.size(); ++i) {
                    const uint64_t k_i = kv_of(i);
                    if (k_i == 0) {
                        continue;
                    }
                    const uint64_t used_slots = (uint64_t) (np > 0 ? np - 1 : 0) * slot_of(i);
                    const uint64_t room = room_of(i) > used_slots ? room_of(i) - used_slots : 0;
                    const uint64_t cap  = (uint64_t) n_min + room / k_i;
                    if (!any || cap < best) {
                        best = cap;
                        any  = true;
                    }
                }
                return any ? best : 0;
            };

            uint64_t np_est = 0;
            size_t   bind   = devs.size();
            for (size_t i = 0; i < devs.size(); ++i) {
                const uint64_t k_i = kv_of(i);
                if (k_i == 0 || data[i].free <= 0) {
                    continue;
                }
                const uint64_t np_i = (room_of(i) + (uint64_t) n_min * k_i + slot_of(i)) /
                                      ((uint64_t) target * k_i + slot_of(i));
                if (bind == devs.size() || np_i < np_est) {
                    np_est = np_i;
                    bind   = i;
                }
            }
            if (bind < devs.size() && np_est >= 1) {
                uint32_t np_ok  = 1;
                int64_t  s_ok   = slack_of(data);
                uint32_t np_prev = np_ok;   // the measured point before np_ok, for the zero crossing
                int64_t  s_prev  = s_ok;
                uint32_t np_bad = 0;
                int64_t  s_bad  = 0;
                uint32_t np_try = (uint32_t) std::min<uint64_t>(np_est, LLAMA_MAX_SEQ);
                int      probes = 0;
                int      spilled = 0;   // system-memory spills tolerated while bracketing
                for (int step = 0; step < 8 && np_try > np_ok; ++step) {
                    const uint64_t pool_try = std::min<uint64_t>((uint64_t) np_try * target, n_max);
                    common_device_memory_data_vec d_np;
                    // Gate before probing. On this box a probe that does not fit is paid for in
                    // system memory: measured 5 GiB of growing Shmem plus 19 GiB of GTT against a
                    // card with 657 MiB free, which is the road to the OOM that took the machine
                    // down. So a candidate whose predicted KV (from the measured slope) plus the
                    // last measured graph cost does not fit is failed here without allocating it,
                    // and the interpolation still gets a bound to work with.
                    bool    gated  = false;
                    int64_t s_gate = INT64_MAX;
                    for (size_t i = 0; i < devs.size() && i < data.size(); ++i) {
                        // Slope from the two capacities that were measured: bytes of KV per token
                        // on this device. Zero means the probing did not see this device grow, and
                        // then only its fixed costs are counted.
                        const uint64_t slope = (n_probe > n_min && i < data_probe.size() &&
                                                data_probe[i].context > data[i].context)
                                             ? (data_probe[i].context - data[i].context) / (n_probe - n_min) : 0;
                        const uint64_t pred_ctx = slope * (pool_try > n_min ? pool_try - n_min : 0);
                        const uint64_t need     = pred_ctx + data[i].context + data[i].compute;
                        const int64_t  free_i   = i < free_base.size() ? free_base[i] : (int64_t) data[i].free;
                        s_gate = std::min(s_gate, free_i - (int64_t) need);
                    }
                    if (s_gate < -(int64_t) (512ull << 20)) {
                        gated = true;
                        LOG_WRN("%s: %u slots x %llu tokens refused before probing (predicted slack %lld MiB)\n",
                                __func__, np_try, (unsigned long long) pool_try, (long long) (s_gate >> 20));
                    }
                    // The verification probe is dry, not metadata-only: it allocates exactly as the
                    // real load will (same buffers, same offload) and skips only the transfer, so
                    // the free memory it reports is the one the real load gets. Extrapolating the
                    // pool's cost from a single slope is what kept missing by 1-2 GiB per device
                    // (measured increment 842 MiB predicted vs 2062 MiB real on Vulkan2).
                    bool fits = false;
                    if (!gated) {
                        try {
                            llama_context_params test = *cparams;
                            test.n_ctx_kv  = (uint32_t) pool_try;
                            test.n_seq_max = np_try;
                            d_np = common_get_device_memory_data(path_model, mparams, &test, devs,
                                                                 hp_ngl, hp_n_ctx_train, hp_n_expert, log_level, /*no_alloc=*/false);
                            fits = !d_np.empty() && slack_of(d_np) >= 0;
                        } catch (const std::exception & e) {
                            LOG_WRN("%s: dry probe (%u slots, %llu tokens) failed: %s\n",
                                    __func__, np_try, (unsigned long long) pool_try, e.what());
                        }
                    }
                    const int64_t s_try = gated ? s_gate : slack_of(d_np);
                    probes++;
                    if (gtt_spilled()) {
                        // Past the edge: take it as the failing bound and interpolate down. Only
                        // one spill is tolerated - a second means the bracket is not converging
                        // and every further attempt would cost the machine more system memory.
                        np_bad = np_try;
                        s_bad  = s_try;
                        if (spilled++ > 0) {
                            LOG_WRN("%s: second spill at %u slots - aborting\n", __func__, np_try);
                            return COMMON_PARAMS_FIT_STATUS_FAILURE;
                        }
                        LOG_WRN("%s: %u slots spilled into system memory; interpolating down\n",
                                __func__, np_try);
                    } else if (fits) {
                        np_prev = np_ok;
                        s_prev  = s_ok;
                        np_ok = np_try;
                        s_ok  = s_try;
                    } else {
                        np_bad = np_try;
                        s_bad  = s_try;
                    }
                    uint64_t np_next;
                    if (np_bad > np_ok && s_ok > 0 && s_ok > s_bad) {
                        // interpolate where the probed slack crosses zero, inside the bracket
                        np_next = (uint64_t) np_ok + (uint64_t) ((__int128) s_ok * (np_bad - np_ok) / (s_ok - s_bad));
                    } else if (np_bad > np_ok) {
                        np_next = ((uint64_t) np_ok + np_bad) / 2;
                    } else {
                        // both probed points fit, so extrapolate where the measured slack line
                        // hits zero: the dry probes are cheap and this converges to the ceiling
                        // instead of stepping one slot at a time.
                        if (np_ok > np_prev && s_prev > s_ok) {
                            np_next = (uint64_t) np_prev + (uint64_t) ((__int128) s_prev * (np_ok - np_prev) / (s_prev - s_ok));
                        } else {
                            np_next = (uint64_t) np_try + 1;
                        }
                    }
                    if (np_next <= np_ok || np_next > LLAMA_MAX_SEQ) {
                        break;
                    }
                    np_try = (uint32_t) np_next;
                }
                cparams->n_seq_max = np_ok;
                cparams->n_ctx_kv  = (uint32_t) std::max<uint64_t>((uint64_t) np_ok * target, n_min);
                // Feedback before committing. Everything above is the cost model's opinion, and
                // that opinion has been wrong about this fork's memory repeatedly: a configuration
                // it accepted pushed 19 GiB into system memory on the first device. Load the chosen
                // configuration once (dry, like every probe) and let the devices answer. A spill
                // steps the slot count down, and the loop is bounded - a machine that pays for
                // repeated mistakes is a machine that goes down.
                for (int verify = 0; verify < 3; ++verify) {
                    common_device_memory_data_vec d_chk;
                    const uint64_t shmem_before = common_fit_shmem_kb();
                    const bool     loaded       = get_data(cparams->n_ctx_kv, d_chk, cparams->n_seq_max);
                    const uint64_t shmem_growth = common_fit_shmem_kb() - shmem_before;
                    if (loaded && !gtt_spilled() && shmem_growth < (256ull << 10)) {   // < 256 MiB
                        break;
                    }
                    LOG_WRN("%s: verification of %u slots x %u tokens grew host Shmem by %llu MiB\n",
                            __func__, cparams->n_seq_max, cparams->n_ctx_kv,
                            (unsigned long long) (shmem_growth >> 10));
                    if (cparams->n_seq_max <= 1) {
                        LOG_WRN("%s: the chosen configuration spills with a single slot\n", __func__);
                        return COMMON_PARAMS_FIT_STATUS_FAILURE;
                    }
                    cparams->n_seq_max -= 1;
                    cparams->n_ctx_kv = (uint32_t) std::max<uint64_t>((uint64_t) cparams->n_seq_max * target, n_min);
                    LOG_WRN("%s: verification spilled; stepping down to %u slots x %u tokens\n",
                            __func__, cparams->n_seq_max, cparams->n_ctx_kv);
                }
                LOG_INF("%s: automatic unified KV capacity = %u tokens for %u slots x %u avg "
                        "(estimate %llu, %d probes, binding device %s)\n",
                        __func__, cparams->n_ctx_kv, cparams->n_seq_max, target,
                        (unsigned long long) np_est, probes,
                        ggml_backend_dev_name(devs[bind]));
                return COMMON_PARAMS_FIT_STATUS_SUCCESS;
            }
        }
    }

    // ---- fallback: pool-only secant over the probed slack ---------------------------------
    // The capacity is smooth and nearly linear in n_ctx_kv, so two probed points predict the
    // zero crossing and every further probe only refines it. Steps stay inside the bracket;
    // bisection is the fallback when the interpolation stalls.
    uint32_t x_ok  = n_min;   // fits: probed above
    int64_t  s_ok  = slack_of(data);
    uint32_t x_bad = 0;       // largest capacity known not to fit (0 = none yet)
    int64_t  s_bad = 0;
    uint32_t x_try = (uint32_t) std::min<uint64_t>(n_max, std::max<uint64_t>((uint64_t) n_min * 2, (uint64_t) n_min + 4096));
    x_try = (uint32_t) (x_try / n_align * n_align);

    int n_probes = 0;
    for (int step = 0; step < 6 && x_try > x_ok; ++step) {
        common_device_memory_data_vec d_try;
        const bool fits = get_data(x_try, d_try);
        const int64_t s_try = slack_of(d_try);
        n_probes++;

        if (fits) {
            x_ok = x_try;
            s_ok = s_try;
            if (x_ok >= n_max) {
                break;
            }
        } else {
            x_bad = x_try;
            s_bad = s_try;
        }

        uint64_t x_next;
        if (x_bad > x_ok && s_ok > 0 && s_ok > s_bad) {
            // secant: where the line through the two probes crosses zero slack
            x_next = (uint64_t) x_ok + (uint64_t) ((__int128) s_ok * (x_bad - x_ok) / (s_ok - s_bad));
        } else if (x_bad > x_ok) {
            x_next = ((uint64_t) x_ok + x_bad) / 2;
        } else {
            x_next = std::min<uint64_t>(n_max, (uint64_t) x_ok * 2);   // no upper bound known yet
        }
        x_next = x_next / n_align * n_align;
        if (x_bad > x_ok && x_next >= x_bad) {
            x_next = ((uint64_t) x_ok + x_bad) / 2 / n_align * n_align;   // secant left the bracket
        }
        if (x_next <= x_ok) {
            x_next = (uint64_t) x_ok + n_align;
        }
        if (x_next > n_max || x_next <= x_ok) {
            break;
        }
        x_try = (uint32_t) x_next;
    }

    cparams->n_ctx_kv = x_ok;
    LOG_INF("%s: automatic unified KV capacity = %u tokens (%d secant probes)\n", __func__, cparams->n_ctx_kv, n_probes);
    // ---- slots and pool ----------------------------------------------------------------
    // Both slopes come from probes, and the solve is an interpolation between probed points
    // rather than an extrapolation:
    //   k_i    = (context(2*n_min) - context(n_min)) / n_min      KV bytes per token
    //   slot_i = free(1 slot) - free(2 slots)                     per-slot state
    // The pool must give every slot at least `target` tokens on average:
    //   pool(np) = n_min + (room_i - (np-1)*slot_i) / k_i  >=  np * target
    //   np       <= (room_i + n_min*k_i + slot_i) / (target*k_i + slot_i)
    // The first np is then verified with a probe, and any correction interpolates between
    // the last fitting np and the first failing one, so no probe is ever extrapolated.
    if (n_probe > n_min) {
        common_device_memory_data_vec data_np2;
        if (get_data(n_min, data_np2, /*n_seq_max_override=*/2)) {
            const uint32_t target = std::max<uint32_t>(n_align, n_ctx_seq / 2);
            uint64_t np_hi   = UINT64_MAX;
            size_t   bind    = devs.size();
            for (size_t i = 0; i < devs.size(); ++i) {
                const uint64_t kv_bytes = data_probe[i].context > data[i].context ?
                    (uint64_t) (data_probe[i].context - data[i].context) : 0;
                const uint64_t k_i = kv_bytes / (n_probe - n_min);
                if (k_i == 0 || data[i].free <= 0) {
                    continue;
                }
                const uint64_t slot_i = (uint64_t) std::max<int64_t>(data[i].free - data_np2[i].free, 0);
                const uint64_t room_i = (uint64_t) data[i].free > FIT_KV_MARGIN ? (uint64_t) data[i].free - FIT_KV_MARGIN : 0;
                const uint64_t np_i = (room_i + (uint64_t) n_min * k_i + slot_i) /
                                      ((uint64_t) target * k_i + slot_i);
                if (np_i < np_hi) {
                    np_hi = np_i;
                    bind  = i;
                }
            }
            if (bind < devs.size() && np_hi >= 1) {
                uint32_t np_ok  = 1;                                  // one slot always fits (probed)
                uint32_t np_try = (uint32_t) std::min<uint64_t>(np_hi, LLAMA_MAX_SEQ);
                uint32_t np_bad = 0;
                int probes = 0;
                for (int step = 0; step < 4 && np_try > np_ok; ++step) {
                    const uint64_t pool_of = (uint64_t) np_try * target;
                    common_device_memory_data_vec d_np;
                    const bool fits = get_data((uint32_t) std::min<uint64_t>(pool_of, n_max), d_np, np_try);
                    probes++;
                    if (fits) {
                        np_ok = np_try;
                    } else {
                        np_bad = np_try;
                    }
                    uint64_t np_next;
                    if (np_bad > np_ok) {
                        np_next = ((uint64_t) np_ok + np_bad) / 2;      // interpolate inside the bracket
                    } else {
                        np_next = (uint64_t) np_try + 1;
                    }
                    if (np_next <= np_ok || np_next > LLAMA_MAX_SEQ) {
                        break;
                    }
                    np_try = (uint32_t) np_next;
                }
                cparams->n_seq_max = np_ok;
                cparams->n_ctx_kv  = (uint32_t) std::max<uint64_t>((uint64_t) np_ok * target, n_min);
                LOG_INF("%s: automatic unified KV capacity = %u tokens for %u slots x %u avg "
                        "(%d slot probes, binding device %s)\n",
                        __func__, cparams->n_ctx_kv, cparams->n_seq_max, target, probes,
                        bind < devs.size() ? ggml_backend_dev_name(devs[bind]) : "?");
                common_fp_log_fit_result("unified KV", path_model, mparams, cparams, cparams->n_ctx_kv);
                return COMMON_PARAMS_FIT_STATUS_SUCCESS;
            }
        }
    }

    common_fp_log_fit_result("unified KV", path_model, mparams, cparams, cparams->n_ctx_kv);
    return COMMON_PARAMS_FIT_STATUS_SUCCESS;
}

common_params_fit_status common_fit_recurrent_cache(
                         const char * path_model,
           const llama_model_params * mparams,
               llama_context_params * cparams,
        const std::vector<std::pair<ggml_backend_dev_t, size_t>> & reserve,
        const llama_context_params * extra_cparams,
                     ggml_log_level   log_level) {
    const uint32_t n_max = std::max(1u, cparams->n_seq_max);
    const uint32_t n_target = std::max(1u, std::min(cparams->n_seq_recurrent == 0 ? 1u : cparams->n_seq_recurrent, n_max));

    // FlashPrefill: recurrent-state slots serve recurrent/MTP decode rows, which are
    // never sparse-eligible, so this fit budgets zero FlashPrefill bytes by design
    // (no reserve, no prediction). Prefill scratch is accounted in
    // common_fit_kv_cache / common_fit_rerot_capacities only.

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0;
    uint32_t hp_n_ctx_train = 0;
    uint32_t hp_n_expert = 0;

    // XKV reserve is computed exactly once per fit invocation (not per
    // probe): configured workspace, decode cache, and factor scratch.
    uint64_t xkv_scratch_once = 0;
    uint64_t xkv_dedup_once = 0;
    if (llama_xkv_is_enabled(cparams->xkv_mode)) {
        if (!common_xkv_scratch_bytes(path_model, mparams, cparams, &xkv_scratch_once, &xkv_dedup_once)) {
            LOG_WRN("%s: XKV scratch estimation failed; degrading scratch reserve to 0\n", __func__);
            xkv_scratch_once = 0;
            xkv_dedup_once = 0;
        }
    }
    common_xkv_fit_reserve xkv_res = {};
    // Dedup vectors are host-resident (carried in the reserve); device
    // probes below carry workspace + device store only, no double count.
    if (!common_xkv_fit_reserve_bytes(cparams, xkv_scratch_once, &xkv_res, xkv_dedup_once)) {
        LOG_WRN("%s: XKV reserve accounting failed, aborting auto-fit\n", __func__);
        return COMMON_PARAMS_FIT_STATUS_FAILURE;
    }
    const uint64_t xkv_store_once_bytes = common_xkv_store_budget_bytes(cparams);
    if (xkv_store_once_bytes == UINT64_MAX) {
        LOG_WRN("%s: XKV store budget accounting overflow, aborting auto-fit\n", __func__);
        return COMMON_PARAMS_FIT_STATUS_FAILURE;
    }
    const uint64_t xkv_store_dev_total = common_xkv_store_device_bytes(cparams, xkv_store_once_bytes);

    auto fits = [&](uint32_t n_seq_recurrent) {
        llama_context_params test = *cparams;
        test.n_seq_recurrent = n_seq_recurrent;

        common_device_memory_data_vec data;
        try {
            data = common_get_device_memory_data(
                path_model, mparams, &test, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level);
        } catch (const std::exception & e) {
            LOG_WRN("%s: recurrent capacity %u probe threw exception: %s\n", __func__, n_seq_recurrent, e.what());
            return false;
        }

        if (devs.empty()) {
            return true;
        }

        common_device_memory_data_vec extra_data;
        std::vector<ggml_backend_dev_t> extra_devs;
        if (extra_cparams != nullptr) {
            llama_context_params extra = *extra_cparams;
            extra.n_ctx_kv = cparams->n_ctx_kv;
            extra.n_seq_recurrent = n_seq_recurrent;
            uint32_t extra_ngl = 0;
            uint32_t extra_n_ctx_train = 0;
            uint32_t extra_n_expert = 0;
            try {
                extra_data = common_get_device_memory_data(
                    path_model, mparams, &extra, extra_devs,
                    extra_ngl, extra_n_ctx_train, extra_n_expert, log_level);
            } catch (const std::exception & e) {
                LOG_WRN("%s: extra context recurrent probe threw exception: %s\n", __func__, e.what());
                return false;
            }
        }

        uint64_t xkv_global = 0;
        std::vector<uint64_t> xkv_wts(devs.size(), 0);
        std::vector<uint64_t> xkv_shares(devs.size(), 0);
        for (size_t i = 0; i < devs.size(); ++i) {
            xkv_wts[i] = (uint64_t) data[i].context;
        }
        if (!xkv_checked_add_u64(xkv_res.total_bytes, xkv_store_dev_total, xkv_global) ||
            !common_xkv_partition_budget(xkv_global, xkv_wts.data(), xkv_shares.data(), xkv_shares.size())) {
            LOG_WRN("%s: recurrent capacity %u does not fit: XKV partition accounting overflow\n",
                __func__, n_seq_recurrent);
            return false;
        }

        for (size_t i = 0; i < devs.size(); ++i) {
            uint64_t reserved = 0;
            for (const auto & [dev, bytes] : reserve) {
                if (dev == devs[i]) {
                    reserved += bytes;
                }
            }
            if (cparams->triattention) {
                ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(devs[i]);
                const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
                if (reg_name && std::string(reg_name).find("Vulkan") != std::string::npos) {
                    // Keep in sync with GGML_VK_MEMMOVE_SCRATCH_SIZE
                    reserved += 8ull * 1024ull * 1024ull;
                } else if (reg_name && std::string(reg_name).find("CUDA") != std::string::npos) {
                    // Keep in sync with GGML_CUDA_MEMMOVE_SCRATCH_SIZE
                    reserved += 8ull * 1024ull * 1024ull;
                }
            }
            // RERoT metadata scratch (span tables / frontier rows / episode
            // host metadata; payloads sized elsewhere, never double-counted).
            // Keep in sync with common_rerot_scratch_reserve_bytes().
            if (cparams->rerot) {
                reserved += 8ull * 1024ull * 1024ull;
            }
            for (size_t j = 0; j < extra_devs.size(); ++j) {
                if (extra_devs[j] == devs[i]) {
                    reserved += extra_data[j].context;
                }
            }

            constexpr uint64_t MiB = 1024ull * 1024ull;
            const uint64_t runtime_headroom = std::min<uint64_t>(
                256ull * MiB, std::max<uint64_t>(32ull * MiB, data[i].total / 100ull));
            uint64_t reserved_all = 0;
            if (!xkv_checked_add_u64(reserved, xkv_shares[i], reserved_all)) {
                return false;
            }

            uint64_t used = 0;
            if (!xkv_checked_sum_fit_bytes(data[i].model, data[i].context, data[i].compute,
                    reserved_all, runtime_headroom, used)) {
                return false;
            }
            // Same decision rule as the KV fit: the compute term is a reserve upper bound and the
            // measured free already contains the driver's footprint, so neither is booked here.
            // Booking them rejected the pool the KV fit had just accepted (one recurrent slot
            // reported as not fitting: used 15.2 GiB against 14.5 GiB free).
            uint64_t used_decision = 0;
            if (!xkv_checked_sum_fit_bytes(data[i].model, data[i].context, 0,
                    reserved_all, 0, used_decision)) {
                return false;
            }
            if (data[i].free <= 0 || used_decision > (uint64_t) data[i].free) {
                LOG_WRN("%s: recurrent capacity %u does not fit on device %zu (%s): used=%llu (model=%llu ctx=%llu comp=%llu res=%llu headroom=%llu) > free=%lld\n",
                    __func__, n_seq_recurrent, i, ggml_backend_dev_name(devs[i]),
                    (unsigned long long) used, (unsigned long long) data[i].model,
                    (unsigned long long) data[i].context, (unsigned long long) data[i].compute,
                    (unsigned long long) reserved_all, (unsigned long long) runtime_headroom,
                    (long long) data[i].free);
                return false;
            }
        }

        return true;
    };

    if (!fits(1)) {
        LOG_WRN("%s: one recurrent state slot does not fit in device memory\n", __func__);
        return COMMON_PARAMS_FIT_STATUS_FAILURE;
    }

    if (devs.empty()) {
        cparams->n_seq_recurrent = n_target;
        LOG_INF("%s: no device-backed recurrent memory detected; keeping %u physical slots (logical sequences = %u)\n",
                __func__, cparams->n_seq_recurrent, cparams->n_seq_max);
        return COMMON_PARAMS_FIT_STATUS_SUCCESS;
    }

    if (n_target == 1 || fits(n_target)) {
        cparams->n_seq_recurrent = n_target;
        LOG_INF("%s: recurrent state capacity = %u physical slots (target = %u, logical sequences = %u)\n",
                __func__, cparams->n_seq_recurrent, n_target, cparams->n_seq_max);
        return COMMON_PARAMS_FIT_STATUS_SUCCESS;
    }

    uint32_t lo = 1;
    uint32_t hi = n_target;
    while (lo + 1 < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (fits(mid)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    cparams->n_seq_recurrent = lo;
    LOG_INF("%s: recurrent state capacity = %u physical slots (target = %u, logical sequences = %u)\n",
            __func__, cparams->n_seq_recurrent, n_target, cparams->n_seq_max);
    return COMMON_PARAMS_FIT_STATUS_SUCCESS;
}

common_rerot_fit_result common_fit_rerot_capacities(
                         const char * path_model,
           const llama_model_params * mparams,
               llama_context_params * cparams,
        const std::vector<std::pair<ggml_backend_dev_t, size_t>> & reserve,
        const llama_context_params * extra_cparams,
                     ggml_log_level   log_level) {
    common_rerot_fit_result best;
    best.status = COMMON_PARAMS_FIT_STATUS_FAILURE;

    // Read model context training size C if not set
    uint32_t c_context = cparams->n_ctx;
    if (c_context == 0) {
        llama_model_params meta_mparams = *mparams;
        meta_mparams.no_alloc = true;
        meta_mparams.load_mode = LLAMA_LOAD_MODE_NONE;
        llama_model * meta_model = llama_model_load_from_file(path_model, meta_mparams);
        if (meta_model) {
            int32_t n_train = llama_model_n_ctx_train(meta_model);
            if (n_train > 0) {
                c_context = (uint32_t) n_train;
            }
            llama_model_free(meta_model);
        }
        if (c_context == 0) {
            c_context = 4096;
        }
    }

    const double rho = cparams->triattention ? cparams->triattention_ratio : 1.0;
    const uint32_t n_align = 256;
    uint32_t k_min = 0;
    if (cparams->triattention) {
        uint32_t needed = std::max<uint32_t>(
            static_cast<uint32_t>(std::ceil(rho * double(c_context))),
            cparams->n_ubatch + 128);
        k_min = (needed + n_align - 1) / n_align * n_align;
    } else {
        k_min = (c_context + n_align - 1) / n_align * n_align;
    }
    best.k_min = k_min;

    const double e_k = std::max(1.0, (rho * double(c_context)) / 2.0);

    // XKV reserve is computed exactly once per fit invocation (not per
    // probe): configured workspace, decode cache, and factor scratch.
    uint64_t xkv_scratch_rerot = 0;
    uint64_t xkv_dedup_rerot = 0;
    if (llama_xkv_is_enabled(cparams->xkv_mode)) {
        if (!common_xkv_scratch_bytes(path_model, mparams, cparams, &xkv_scratch_rerot, &xkv_dedup_rerot)) {
            LOG_WRN("%s: XKV scratch estimation failed; degrading scratch reserve to 0\n", __func__);
            xkv_scratch_rerot = 0;
            xkv_dedup_rerot = 0;
        }
    }
    common_xkv_fit_reserve xkv_res_rerot = {};
    if (!common_xkv_fit_reserve_bytes(cparams, xkv_scratch_rerot, &xkv_res_rerot, xkv_dedup_rerot)) {
        LOG_WRN("%s: XKV reserve accounting failed, aborting auto-fit\n", __func__);
        return best;
    }
    // Persistent store budget (explicit only), separate from the one
    // workspace allocation above.
    const uint64_t xkv_store_rerot_bytes = common_xkv_store_budget_bytes(cparams);
    if (xkv_store_rerot_bytes == UINT64_MAX) {
        LOG_WRN("%s: XKV store budget accounting overflow, aborting auto-fit\n", __func__);
        return best;
    }
    const uint64_t xkv_store_rerot_dev = common_xkv_store_device_bytes(cparams, xkv_store_rerot_bytes);

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0;
    uint32_t hp_n_ctx_train = 0;
    uint32_t hp_n_expert = 0;

    // FlashPrefill native-split dims, resolved once (metadata-only load, enabled-only).
    common_fp_dims fp_nat_dims;
    const bool fp_nat_have = common_fp_load_dims(path_model, mparams, cparams, fp_nat_dims);

    auto test_fit = [&](uint32_t b, uint32_t p, uint32_t k_val, int64_t & min_margin_out) -> bool {
        llama_context_params test = *cparams;
        test.n_ctx_kv = k_val;
        test.n_person_max = b;
        test.n_pen_max = p;
        test.n_seq_recurrent = b;
        test.n_seq_max = LLAMA_MAX_SEQ;
        test.n_outputs_max = std::max(1u, p * (1u + test.n_rs_seq));

        common_device_memory_data_vec data;
        try {
            data = common_get_device_memory_data(path_model, mparams, &test, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level);
        } catch (const std::exception & e) {
            LOG_TRC("%s: probe failed for B=%u P=%u K=%u: %s\n", __func__, b, p, k_val, e.what());
            return false;
        }

        if (devs.empty()) {
            return false;
        }

        common_device_memory_data_vec extra_data;
        std::vector<ggml_backend_dev_t> extra_devs;
        if (extra_cparams != nullptr) {
            llama_context_params extra = *extra_cparams;
            extra.n_ctx_kv = k_val;
            // The speculative context owns only live execution sequences. It
            // never stores the RERoT document's parked logical sequence IDs,
            // brains, or pens. Giving it LLAMA_MAX_SEQ forces output_reserve()
            // to allocate one vocabulary row for thousands of nonexistent
            // draft readers and incorrectly collapses the fitted pen arena.
            extra.n_seq_max = std::max(1u, p);
            extra.n_outputs_max = std::max(1u, p);
            extra.n_person_max = 0;
            extra.n_pen_max = 0;
            extra.n_seq_recurrent = std::max(1u, p);
            uint32_t extra_ngl = 0;
            uint32_t extra_n_ctx_train = 0;
            uint32_t extra_n_expert = 0;
            try {
                extra_data = common_get_device_memory_data(
                    path_model, mparams, &extra, extra_devs,
                    extra_ngl, extra_n_ctx_train, extra_n_expert, log_level);
            } catch (const std::exception & e) {
                return false;
            }
        }

        uint64_t xkv_global = 0;
        std::vector<uint64_t> xkv_wts(devs.size(), 0);
        std::vector<uint64_t> xkv_shares(devs.size(), 0);
        for (size_t i = 0; i < devs.size(); ++i) {
            xkv_wts[i] = (uint64_t) data[i].context;
        }
        if (!xkv_checked_add_u64(xkv_res_rerot.total_bytes, xkv_store_rerot_dev, xkv_global) ||
            !common_xkv_partition_budget(xkv_global, xkv_wts.data(), xkv_shares.data(), xkv_shares.size())) {
            return false;
        }

        int64_t min_margin = INT64_MAX;
        for (size_t i = 0; i < devs.size(); ++i) {
            uint64_t reserved = 0;
            for (const auto & [dev, bytes] : reserve) {
                if (dev == devs[i]) {
                    reserved += bytes;
                }
            }
            // FlashPrefill: graph caps arrive via data[i].compute (no manual reserve,
            // no double charge); the backend-private split pool does not, so its
            // worst-case peak is added here exactly once per device (shared formula,
            // frozen admission allowances).
            if (fp_nat_have) {
                uint64_t nat_bytes = 0;
                if (!common_fp_native_split_bytes(&cparams->flashprefill, &fp_nat_dims, cparams->n_ubatch, k_val, devs[i], nat_bytes)) {
                    LOG_TRC("%s: probe rejected for B=%u P=%u K=%u: native split shape unsizable\n", __func__, b, p, k_val);
                    return false;
                }
                reserved += nat_bytes;
            }
            if (cparams->triattention) {
                ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(devs[i]);
                const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
                if (reg_name && std::string(reg_name).find("Vulkan") != std::string::npos) {
                    // Keep in sync with GGML_VK_MEMMOVE_SCRATCH_SIZE
                    reserved += 8ull * 1024ull * 1024ull;
                } else if (reg_name && std::string(reg_name).find("CUDA") != std::string::npos) {
                    // Keep in sync with GGML_CUDA_MEMMOVE_SCRATCH_SIZE
                    reserved += 8ull * 1024ull * 1024ull;
                }
            }

            // RERoT span-table/DDVR phase inputs, frontier query rows,
            // parked-recurrent metadata descriptors and episode host metadata
            // (KV/recurrent payloads are sized by the surrounding fit paths,
            // never double-counted here). Keep in sync with
            // common_rerot_scratch_reserve_bytes() in common/common.h.
            // This joint B/P/K path always runs with RERoT enabled, so the
            // reserve is unconditional here (unlike the KV/recurrent fits
            // which gate on cparams->rerot).
            reserved += 8ull * 1024ull * 1024ull;
            for (size_t j = 0; j < extra_devs.size(); ++j) {
                if (extra_devs[j] == devs[i]) {
                    reserved += extra_data[j].context;
                }
            }

            constexpr uint64_t MiB = 1024ull * 1024ull;
            const uint64_t runtime_headroom = std::min<uint64_t>(
                256ull * MiB, std::max<uint64_t>(32ull * MiB, data[i].total / 100ull));
            uint64_t reserved_all = 0;
            if (!xkv_checked_add_u64(reserved, xkv_shares[i], reserved_all)) {
                return false;
            }
            uint64_t used = 0;
            if (!xkv_checked_sum_fit_bytes(data[i].model, data[i].context, data[i].compute,
                    reserved_all, runtime_headroom, used)) {
                LOG_WRN("%s: KV size %u does not fit: memory total overflow\n", __func__, k_val);
                return false;
            }
            uint64_t used_decision = 0;
            if (!xkv_checked_sum_fit_bytes(data[i].model, data[i].context, 0,
                    reserved_all, 0, used_decision)) {
                return false;
            }
            if (data[i].free <= 0 || used_decision > (uint64_t) data[i].free) {
                return false;
            }
            const int64_t diff = (int64_t) data[i].free - (int64_t) used_decision;
            if (diff < min_margin) {
                min_margin = diff;
            }
        }
        min_margin_out = min_margin;
        return true;
    };

    auto search_max_k = [&](uint32_t b, uint32_t p, uint32_t & max_k_out, int64_t & max_k_margin_out) -> bool {
        int64_t test_margin = 0;
        if (!test_fit(b, p, k_min, test_margin)) {
            return false;
        }

        uint32_t low = k_min;
        uint32_t high = k_min * 16;
        if (high < 65536) high = 65536;

        while (high > low && test_fit(b, p, high, test_margin)) {
            low = high;
            if (high > UINT32_MAX / 2) {
                break;
            }
            high *= 2;
        }

        while (high - low > n_align) {
            uint32_t mid = low + (high - low) / 2;
            mid = (mid / n_align) * n_align;
            if (mid <= low) {
                break;
            }
            if (test_fit(b, p, mid, test_margin)) {
                low = mid;
            } else {
                high = mid;
            }
        }
        max_k_out = low;
        test_fit(b, p, max_k_out, max_k_margin_out);
        return true;
    };

    // Negative feedback probing (§B.10 / user feedback model):
    // Fixed 6 pens per person: P = 6 * B. Pens are dynamically allocated first-come, first-served
    // at runtime up to the total pen capacity P.
    // Expected KV capacity per person is E_K = rho * C / 2 (ctx / 2).
    // No hardcoded candidate arrays:
    // 1. Initial baseline probe at (B=1, P=6) establishes maximum possible physical KV capacity K1.
    // 2. K1 / E_K directly establishes the physical upper bound B_max without arbitrary constants.
    // 3. Negative feedback bisection on B:
    //    - If (B, 6B, K_min) overflows VRAM -> overcommit negative feedback pulls high bound down.
    //    - If it fits, observed K yields actual support capacity B_supported = floor(K / E_K).
    //      - If B <= B_supported -> capacity is sustainable! Record best and probe higher.
    //      - If B > B_supported  -> KV deficit negative feedback pulls high bound down to min(B-1, B_supported).
    const uint32_t pens_per_person = 6;

    // Step 1: Baseline probe at B=1, P=6
    uint32_t k1_found = 0;
    int64_t  m1_margin = 0;
    if (search_max_k(1, pens_per_person, k1_found, m1_margin) && k1_found >= k_min) {
        best.status = COMMON_PARAMS_FIT_STATUS_SUCCESS;
        best.b_people = 1;
        best.p_pens = pens_per_person;
        best.k_tokens = k1_found;
        best.min_device_margin = m1_margin;

        // Theoretical physical ceiling: more people take more brain/hand memory, so K(B) <= K1.
        // Therefore B cannot physically exceed floor(K1 / E_K), nor LLAMA_MAX_SEQ / pens_per_person.
        const uint32_t max_seq_people = (uint32_t) (LLAMA_MAX_SEQ / pens_per_person);
        const uint32_t b_ceiling = std::max(1u, (uint32_t) (double(k1_found) / e_k));
        const uint32_t b_upper = std::min(max_seq_people, b_ceiling);

        uint32_t low = 2;
        uint32_t high = b_upper;

        while (low <= high) {
            const uint32_t mid = low + (high - low + 1) / 2;
            const uint32_t p_mid = mid * pens_per_person;

            uint32_t k_mid = 0;
            int64_t  margin_mid = 0;
            if (!search_max_k(mid, p_mid, k_mid, margin_mid) || k_mid < k_min) {
                // VRAM overflow: overcommit negative feedback pulls high down
                high = mid - 1;
            } else {
                const uint32_t b_supported = std::max(1u, (uint32_t) std::round(double(k_mid) / e_k));
                if (mid <= b_supported) {
                    // Sustainable candidate: record and probe higher
                    best.b_people = mid;
                    best.p_pens = p_mid;
                    best.k_tokens = k_mid;
                    best.min_device_margin = margin_mid;
                    low = mid + 1;
                } else {
                    // KV deficit: observed KV capacity only supports b_supported people.
                    // Negative feedback pulls high down directly to min(mid - 1, b_supported).
                    high = std::min(mid - 1, b_supported);
                }
            }
        }

        LOG_INF("RERoT auto-fit selected (feedback probe): B=%u people, P=%u pens, K=%u tokens (K_min=%u, min_margin=%lld B)\n",
            best.b_people, best.p_pens, best.k_tokens, best.k_min, (long long) best.min_device_margin);
        common_fp_log_fit_result("RERoT", path_model, mparams, cparams, best.k_tokens);
        return best;
    }

    // Step 2: Safety fallback for ultra-constrained devices where even (B=1, P=6) does not fit:
    for (uint32_t p_fallback = pens_per_person - 1; p_fallback >= 1; --p_fallback) {
        uint32_t k_found = 0;
        int64_t margin_found = 0;
        if (!search_max_k(1, p_fallback, k_found, margin_found)) {
            continue;
        }
        if (k_found < k_min) {
            continue;
        }
        best.status = COMMON_PARAMS_FIT_STATUS_SUCCESS;
        best.b_people = 1;
        best.p_pens = p_fallback;
        best.k_tokens = k_found;
        best.min_device_margin = margin_found;
        LOG_INF("RERoT auto-fit fallback selected: B=1 person, P=%u pens, K=%u tokens (K_min=%u, min_margin=%lld B)\n",
            best.p_pens, best.k_tokens, best.k_min, (long long) best.min_device_margin);
        common_fp_log_fit_result("RERoT", path_model, mparams, cparams, best.k_tokens);
        return best;
    }

    return best;
}

void common_fit_print(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams) {
    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    auto dmd = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);
    GGML_ASSERT(dmd.size() == devs.size() + 1);

    for (size_t id = 0; id < devs.size(); id++) {
        printf("%s ",  ggml_backend_dev_name(devs[id]));
        printf("%zu ", dmd[id].mb.model/1024/1024);
        printf("%zu ", dmd[id].mb.context/1024/1024);
        printf("%zu ", dmd[id].mb.compute/1024/1024);
        printf("\n");
    }

    printf("Host ");
    printf("%zu ", dmd.back().mb.model/1024/1024);
    printf("%zu ", dmd.back().mb.context/1024/1024);
    printf("%zu ", dmd.back().mb.compute/1024/1024);
    printf("\n");
}
