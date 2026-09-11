#pragma once

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <string>
#include <unordered_set>
#include <list>
#include <map>
#include <ostream>
#include <sstream>
#include <iomanip>
#include <vector>

// TODO: prevent including the whole server-common.h as we only use server_tokens
#include "server-common.h"

using json = nlohmann::ordered_json;

enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_EMBEDDING,
    SERVER_TASK_TYPE_RERANK,
    SERVER_TASK_TYPE_INFILL,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_CONTROL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_GET_LORA,
    SERVER_TASK_TYPE_SET_LORA,
};

// TODO: change this to more generic "response_format" to replace the "format_response_*" in server-common
enum task_response_type {
    TASK_RESPONSE_TYPE_NONE, // llama.cpp native format
    TASK_RESPONSE_TYPE_OAI_CHAT,
    TASK_RESPONSE_TYPE_OAI_CMPL,
    TASK_RESPONSE_TYPE_OAI_RESP,
    TASK_RESPONSE_TYPE_OAI_ASR, // transcriptions API
    TASK_RESPONSE_TYPE_OAI_EMBD,
    TASK_RESPONSE_TYPE_ANTHROPIC,
};

enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};

struct task_params {
    bool stream          = false;
    bool include_usage   = false;
    bool cache_prompt    = true; // remember the prompt to avoid reprocessing all prompt
    bool return_tokens   = false;
    bool return_progress = false;

    int32_t sse_ping_interval = 30; // seconds between SSE comment pings while the stream stays silent, -1 disables

    int32_t n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t n_discard =  0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t n_predict = -1; // new tokens to predict
    int32_t n_indent  =  0; // minimum line indentation for the generated text in number of whitespace characters
    int32_t n_cmpl    =  1; // number of completions to generate from this prompt

    int32_t n_cache_reuse = 0; // min chunk size to attempt reusing from the cache via KV shifting (0 = disabled)

    int64_t t_max_prompt_ms  = -1; // TODO: implement
    int64_t t_max_predict_ms = -1; // if positive, limit the generation phase to this time limit

    std::map<int, float> lora; // mapping adapter ID -> scale

    std::vector<std::string> antiprompt;
    std::vector<std::string> response_fields;

    bool timings_per_token   = false;
    bool post_sampling_probs = false;

    struct common_params_sampling sampling;
    struct common_params_speculative speculative;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    std::string        reasoning_effort;

    // realtime control (SERVER_TASK_TYPE_CONTROL)
    std::string        control_action;
    std::string        control_cmpl_id;

    // per-request parameters for chat parsing
    common_chat_parser_params chat_parser_params;

    // message spans for checkpointing
    common_chat_msg_spans message_spans;

    // Original chat request used to re-render ordinary vs DAG tool prefixes
    // (AGENTS.md §02.3). Empty for non-chat completions.
    json rerot_chat_messages = json::array();
    json rerot_chat_tools = json::array();
    int64_t rerot_chat_now_ms = 0;

    // Embeddings
    int32_t embd_normalize = 2; // (-1=none, 0=max absolute int16, 1=taxicab, 2=Euclidean/L2, >2=p-norm)

    // RERoT outer compatibility (§§18,26,A.13-A.20). Per-request overrides.
    // Defaults come from the server-global common_params via
    // apply_rerot_defaults() after schema eval (server-schema.cpp ignores
    // unknown keys, so the post-step in server-common.cpp applies the raw
    // request JSON). RERoT OFF: all false/strong, no allocation.
    bool rerot_enabled = false;
    llama_rerot_frontier_mode rerot_frontier = LLAMA_REROT_FRONTIER_STRONG;
    bool rerot_trace = false; // explicit opt-in for rerot.trace.* SSE events; default off

    // Fill rerot_* from the server-global base params. No-op when base OFF.
    void apply_rerot_defaults(const common_params & base);
    // Per-request static gate. Dynamic Tri/speculation/state compatibility is
    // enforced by the active runtime; see common_rerot_validate_stage0.
    // Embedding/rerank requests never enter
    // RERoT (A.19) and pass with effective() == false, not an error.
    // OFF requests always pass with no allocation.
    bool rerot_validate_request(server_task_type task_type, bool has_media, std::string & error) const;
    // Effective switch for this request (A.19): EMBEDDING/RERANK never enter
    // RERoT even when globally enabled. Multimodal prompts stay effective:
    // fork happens prelude-then-fork and DDVR never remaps visual positions.
    bool rerot_effective(server_task_type task_type) const;

    json format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const;
    json to_json(bool only_metrics = false) const;
};

// RERoT episode key for ABA-safe runtime callbacks (§A.15). Every callback
// carries (task id, episode id, generation); a callback whose key does not
// match the current task state is stale (e.g. cancel/retry raced an in-flight
// frontier commit) and must be dropped without touching KV.
struct server_rerot_episode_key {
    int task_id = -1;
    uint64_t episode_id = 0;
    uint64_t generation = 0;

    bool valid() const {
        return task_id >= 0 && episode_id != 0;
    }
    bool matches(int task, uint64_t episode, uint64_t gen) const {
        return task_id == task && episode_id == episode && generation == gen;
    }
};

// RERoT metrics suite (§A.26), additive to the existing Tri metrics.
// Counters are cumulative for the process lifetime. All zero when RERoT OFF
// (no runtime allocation, no metric emission: to_json omits the rerot block).
struct server_rerot_metrics {
    uint64_t episode_total = 0;
    uint64_t episode_active = 0;
    uint64_t nodes_created = 0;
    uint64_t nodes_started = 0;
    uint64_t nodes_retired = 0;
    uint64_t nodes_queued = 0;
    uint64_t queue_max = 0;
    uint64_t forks_total = 0;
    uint64_t max_depth = 0;
    uint64_t max_live_lanes = 0;
    uint64_t public_tokens = 0;
    uint64_t private_tokens = 0;
    uint64_t pending_tokens = 0;
    uint64_t completed_episodes = 0;
    uint64_t completed_model_tokens = 0;
    uint64_t parallel_model_tokens = 0;
    double completed_episode_seconds = 0.0;
    double parallel_seconds = 0.0;
    uint64_t frontiers = 0;
    uint64_t topology_barriers = 0;
    uint64_t refresh_total = 0;
    uint64_t mtp_invalidations = 0;
    uint64_t context_shifts = 0;
    uint64_t hard_aborts = 0;
    uint64_t final_fences = 0;
    uint64_t span_count = 0;
    uint64_t parked_total = 0;
    uint64_t archive_total = 0;
    double ddvr_seconds = 0.0;

    // Multi-capacity and grouped scheduler metrics (§B.14)
    uint64_t people_capacity = 0;
    uint64_t people_resident = 0;
    uint64_t people_runnable = 0;
    uint64_t people_waiting = 0;

    uint64_t pens_capacity = 0;
    uint64_t pens_allocated = 0;
    uint64_t pens_running = 0;
    uint64_t pens_suspended = 0;
    uint64_t pen_queue_depth = 0;
    uint64_t pens_per_person_max_observed = 0;
    double   pen_utilization = 0.0;

    uint64_t batch_people = 0;
    uint64_t batch_pens = 0;
    uint64_t frontier_rows = 0;

    uint64_t brain_bytes = 0;
    uint64_t hand_bytes = 0;
    uint64_t grouped_scratch_bytes = 0;

    bool empty() const;
    json to_json() const;
    void accumulate(const server_rerot_metrics & delta);
};

// XKV metrics suite (§16), additive to the Tri/RERoT metrics.
// All zero/empty when XKV OFF: to_json() output is omitted by the caller
// and to_prometheus() emits nothing, so OFF responses keep the exact
// pre-XKV schema. Store/runtime counters stay zero until the XKV runtime
// fills them; admission fields are populated from the live snapshot.
struct server_xkv_metrics {
    bool        has_admission = false;
    uint32_t    safe_next_ubatch = 0;
    std::string limit_reason;
    uint32_t    logical_capacity = 0;
    uint32_t    logical_used = 0;
    uint32_t    hot_capacity = 0;
    uint32_t    hot_used = 0;
    uint32_t    hot_free = 0;
    uint32_t    hot_reserved = 0;
    uint64_t    factor_live_bytes = 0;
    uint64_t    factor_reserved_bytes = 0;
    uint64_t    factor_budget_bytes = 0;
    uint64_t    factor_free_bytes = 0;
    uint32_t    factor_safe_tokens = UINT32_MAX; // planner bound (MAX = unconstrained)
    uint64_t    workspace_live_bytes = 0;
    uint64_t    workspace_peak_bytes = 0;
    uint64_t    workspace_budget_bytes = 0;
    uint64_t    workspace_free_bytes = 0;
    uint32_t    workspace_safe_tokens = UINT32_MAX; // planner bound (MAX = unconstrained)
    uint32_t    recurrent_capacity = 0;
    uint32_t    recurrent_used = 0;
    std::string requested_profile;
    std::string effective_profile;
    // Requested/effective XKV mode (off|shadow|dense|sr). Storage profile
    // stays separate: mode tells R2-dense apart from R4-SR on the same family.
    std::string requested_mode;
    std::string effective_mode;
    std::string source; // observed source, never assumed
    // Codec identity (strings): requested from params, effective from runtime.
    std::string requested_a_k, effective_a_k;
    std::string requested_b_k, effective_b_k;
    std::string requested_a_v, effective_a_v;
    std::string requested_b_v, effective_b_v;
    std::string requested_landmark, effective_landmark;
    std::string requested_factorizer, effective_factorizer;
    std::string requested_balance, effective_balance;
    // Factorization seed (exact hex/string; never float gauge >2^53).
    std::string requested_seed, effective_seed;
    // Canonical landmark codec table seeds (exact hex/string). Factor seeds
    // derive per-group (seed + group*1000); landmark tables use the canonical
    // codec seed (777). Requested is the configured default; effective is the
    // observed runtime value. Separate from factor seeds; never conflated.
    std::string requested_landmark_seed, effective_landmark_seed;
    // Tri identity:
    uint64_t    tri_calibration_fingerprint = 0;
    std::string tri_calibration_sha256;
    std::string tri_ratio_str;
    uint32_t    tri_recent_window = 0;
    bool        tri_scorer_valid = false;
    // Tri-state compression goal: true iff evaluated
    bool        compression_goal_evaluated = false;
    // Evaluator model SHA-256 (exact lowercase hex from llama_model):
    std::string model_sha256;
    uint64_t    codec_fingerprint = 0;
    uint64_t    backend_fingerprint = 0;
    uint64_t    source_fingerprint = 0;
    uint64_t    profile_fingerprint = 0;
    // Full §16 effective config fingerprint. The runtime stamps segments with
    // config_fingerprint() (see profile_fingerprint); exposed separately so the
    // evaluator need not guess which fingerprint carries the config.
    uint64_t    config_fingerprint = 0;
    uint32_t    rank_k = 0;
    uint32_t    rank_v = 0;
    uint64_t    factor_streams = 0;
    uint64_t    actual_bytes = 0;
    uint64_t    nominal_bytes = 0;
    uint64_t    hot_bytes = 0;
    uint64_t    flat_bytes = 0;
    uint64_t    factor_ak_bytes = 0;
    uint64_t    factor_bk_bytes = 0;
    uint64_t    factor_av_bytes = 0;
    uint64_t    factor_bv_bytes = 0;
    uint64_t    factor_payload_bytes = 0;
    uint64_t    factor_metadata_bytes = 0;
    uint64_t    factor_padding_bytes = 0;
    uint64_t    landmark_payload_bytes = 0;
    uint64_t    landmark_metadata_bytes = 0;
    uint64_t    landmark_exception_bytes = 0;
    uint64_t    index_bytes = 0;
    uint64_t    codec_shared_bytes = 0;
    uint64_t    decode_tile_cache_bytes = 0;
    uint64_t    capture_bytes = 0;
    uint64_t    candidate_bytes = 0;
    // Transient device staging reservation (live bytes). No separate staging
    // ring exists in decoded-hot mode: transient staging lives inside the
    // workspace arena / candidate scratch, so this is 0 when none is held.
    // Exposed explicitly so "staging" is observed, never inferred.
    uint64_t    staging_bytes = 0;
    uint64_t    snapshot_pinned_bytes = 0;
    uint64_t    allocator_live_bytes = 0;
    uint64_t    allocator_reserved_bytes = 0;
    uint64_t    device_peak_bytes = 0;
    uint64_t    host_peak_bytes = 0;
    uint64_t    unique_payloads = 0;
    uint64_t    aliased_payloads = 0;
    uint64_t    baseline_same_rows_bytes = 0;
    uint64_t    covered_compressed_bytes = 0; // compressed bytes covering the baseline
    double      factored_baseline_byte_coverage = 0.0;
    double      factor_quant_ratio = 0.0;
    double      net_extra_compression_ratio = 0.0;
    double      net_extra_compression_ratio_reserved = 0.0;
    // Peak-basis net extra compression ratio (§11.1 "running peak" pair member):
    // same-row live baseline / peak allocated bytes
    // (device_peak + host_peak high-water). Gated by ratios_evaluated like the
    // live/reserved pair; never mixed with them.
    double      net_extra_compression_ratio_peak = 0.0;
    // Explicit evaluated status (not_evaluated renders as JSON null /
    // Prometheus NaN, never as 0.0). Transport-only:
    bool        ratios_evaluated = false;
    bool        seal_timers_evaluated = false;
    bool        quant_timers_evaluated = false;
    bool        graph_timings_evaluated = false;
    bool        pack_timer_evaluated = false;
    bool        sr_counters_evaluated = false;
    bool        spec_counters_evaluated = false;
    // Transport-only: XKV was enabled and a runtime was present, even if no
    // snapshot was armed. False preserves exact OFF all-zero output. Excluded
    // from empty() so unobserved snapshots stay silent on Prometheus.
    bool        observed = false;
    double      seal_seconds = 0.0;
    double      factor_quant_seconds = 0.0;
    double      landmark_quant_seconds = 0.0;
    double      select_seconds = 0.0;
    double      refine_seconds = 0.0;
    double      reconstruct_seconds = 0.0;
    double      read_seconds = 0.0;
    double      pack_seconds = 0.0;
    uint64_t    segments_sealed = 0;
    std::map<std::string, uint64_t> segments_skipped_by_reason;
    uint64_t    sr_selected_rows = 0;
    uint64_t    sr_fragments = 0;
    uint64_t    effective_chunk_size = 0;
    uint64_t    landmark_refine_rows = 0;
    uint64_t    landmark_refine_cap_hits = 0;
    uint64_t    spec_stale_total = 0;
    uint64_t    transaction_abort_total = 0;
    uint64_t    synchronize_total = 0;
    bool        compression_goal_met = false;
    uint64_t    throttle_total = 0;
    std::string throttle_reason;

    bool empty() const {
        return !has_admission
            && requested_profile.empty() && effective_profile.empty()
            && requested_mode.empty() && effective_mode.empty()
            && source.empty()
            && codec_fingerprint == 0 && backend_fingerprint == 0
            && source_fingerprint == 0 && profile_fingerprint == 0
            && config_fingerprint == 0
            && rank_k == 0 && rank_v == 0 && factor_streams == 0
            && actual_bytes == 0 && nominal_bytes == 0
            && hot_bytes == 0 && flat_bytes == 0
            && factor_ak_bytes == 0 && factor_bk_bytes == 0
            && factor_av_bytes == 0 && factor_bv_bytes == 0
            && factor_payload_bytes == 0 && factor_metadata_bytes == 0 && factor_padding_bytes == 0
            && landmark_payload_bytes == 0 && landmark_metadata_bytes == 0 && landmark_exception_bytes == 0
            && index_bytes == 0 && codec_shared_bytes == 0
            && decode_tile_cache_bytes == 0 && capture_bytes == 0 && candidate_bytes == 0
            && staging_bytes == 0
            && snapshot_pinned_bytes == 0
            && allocator_live_bytes == 0 && allocator_reserved_bytes == 0
            && device_peak_bytes == 0 && host_peak_bytes == 0
            && unique_payloads == 0 && aliased_payloads == 0
            && baseline_same_rows_bytes == 0 && factored_baseline_byte_coverage == 0.0
            && covered_compressed_bytes == 0
            && factor_quant_ratio == 0.0 && net_extra_compression_ratio == 0.0
            && net_extra_compression_ratio_reserved == 0.0
            && net_extra_compression_ratio_peak == 0.0
            && seal_seconds == 0.0 && factor_quant_seconds == 0.0 && landmark_quant_seconds == 0.0
            && select_seconds == 0.0 && refine_seconds == 0.0 && reconstruct_seconds == 0.0
            && read_seconds == 0.0 && pack_seconds == 0.0
            && segments_sealed == 0 && segments_skipped_by_reason.empty()
            && sr_selected_rows == 0 && sr_fragments == 0 && effective_chunk_size == 0
            && landmark_refine_rows == 0 && landmark_refine_cap_hits == 0
            && spec_stale_total == 0 && transaction_abort_total == 0
            && synchronize_total == 0 && !compression_goal_met
            && throttle_total == 0 && throttle_reason.empty()
            && safe_next_ubatch == 0 && limit_reason.empty()
            && logical_capacity == 0 && logical_used == 0
            && hot_capacity == 0 && hot_used == 0 && hot_free == 0 && hot_reserved == 0
            && factor_live_bytes == 0 && factor_reserved_bytes == 0
            && factor_budget_bytes == 0 && factor_free_bytes == 0
            && factor_safe_tokens == UINT32_MAX
            && workspace_live_bytes == 0 && workspace_peak_bytes == 0
            && workspace_budget_bytes == 0 && workspace_free_bytes == 0
            && workspace_safe_tokens == UINT32_MAX
            && recurrent_capacity == 0 && recurrent_used == 0;
    }

    json to_json() const {
        json adm = json::object();
        if (has_admission) {
            adm = json {
                { "safe_next_ubatch",  safe_next_ubatch },
                { "limit_reason",      limit_reason },
                { "logical_capacity",  logical_capacity },
                { "logical_used",      logical_used },
                { "hot_capacity",      hot_capacity },
                { "hot_used",          hot_used },
                { "hot_free",          hot_free },
                { "hot_reserved",      hot_reserved },
                { "factor_live_bytes",     factor_live_bytes },
                { "factor_reserved_bytes", factor_reserved_bytes },
                { "factor_budget_bytes",   factor_budget_bytes },
                { "factor_free_bytes",     factor_free_bytes },
                { "factor_safe_tokens",    factor_safe_tokens },
                { "workspace_live_bytes",   workspace_live_bytes },
                { "workspace_peak_bytes",   workspace_peak_bytes },
                { "workspace_budget_bytes", workspace_budget_bytes },
                { "workspace_free_bytes",   workspace_free_bytes },
                { "workspace_safe_tokens",  workspace_safe_tokens },
                { "recurrent_capacity", recurrent_capacity },
                { "recurrent_used",     recurrent_used },
            };
        }
        json skipped = json::object();
        for (const auto & kv : segments_skipped_by_reason) {
            skipped[kv.first] = kv.second;
        }
        return json {
            { "xkv_admission",               adm },
            { "xkv_requested_profile",       requested_profile },
            { "xkv_effective_profile",       effective_profile },
            { "xkv_requested_mode",          requested_mode },
            { "xkv_effective_mode",          effective_mode },
            { "xkv_requested_a_k",           requested_a_k },
            { "xkv_effective_a_k",           effective_a_k },
            { "xkv_requested_b_k",           requested_b_k },
            { "xkv_effective_b_k",           effective_b_k },
            { "xkv_requested_a_v",           requested_a_v },
            { "xkv_effective_a_v",           effective_a_v },
            { "xkv_requested_b_v",           requested_b_v },
            { "xkv_effective_b_v",           effective_b_v },
            { "xkv_requested_landmark",      requested_landmark },
            { "xkv_effective_landmark",      effective_landmark },
            { "xkv_requested_factorizer",    requested_factorizer },
            { "xkv_effective_factorizer",    effective_factorizer },
            { "xkv_requested_balance",       requested_balance },
            { "xkv_effective_balance",       effective_balance },
            { "xkv_requested_seed",          requested_seed },
            { "xkv_effective_seed",          effective_seed },
            { "xkv_requested_landmark_seed", requested_landmark_seed },
            { "xkv_effective_landmark_seed", effective_landmark_seed },
            { "xkv_model_sha256",            model_sha256 },
            { "tri_calibration_fingerprint", tri_calibration_fingerprint },
            { "tri_calibration_sha256",      tri_calibration_sha256 },
            { "tri_ratio",                   tri_ratio_str },
            { "tri_recent_window",           tri_recent_window },
            { "tri_scorer_valid",            tri_scorer_valid },
            { "xkv_compression_goal_evaluated", compression_goal_evaluated },
            { "xkv_compression_goal_met",    !observed || compression_goal_evaluated ? json(compression_goal_met) : json(nullptr) },
            { "xkv_source",                  source },
            { "xkv_codec_fingerprint",       codec_fingerprint },
            { "xkv_backend_fingerprint",     backend_fingerprint },
            { "xkv_source_fingerprint",      source_fingerprint },
            { "xkv_profile_fingerprint",     profile_fingerprint },
            { "xkv_config_fingerprint",      config_fingerprint },
            { "xkv_rank_k",                  rank_k },
            { "xkv_rank_v",                  rank_v },
            { "xkv_factor_streams",          factor_streams },
            { "xkv_actual_bytes",            actual_bytes },
            { "xkv_nominal_bytes",           nominal_bytes },
            { "xkv_hot_bytes",               hot_bytes },
            { "xkv_flat_bytes",              flat_bytes },
            { "xkv_factor_ak_bytes",         factor_ak_bytes },
            { "xkv_factor_bk_bytes",         factor_bk_bytes },
            { "xkv_factor_av_bytes",         factor_av_bytes },
            { "xkv_factor_bv_bytes",         factor_bv_bytes },
            { "xkv_factor_payload_bytes",    factor_payload_bytes },
            { "xkv_factor_metadata_bytes",   factor_metadata_bytes },
            { "xkv_factor_padding_bytes",    factor_padding_bytes },
            { "xkv_landmark_payload_bytes",  landmark_payload_bytes },
            { "xkv_landmark_metadata_bytes", landmark_metadata_bytes },
            { "xkv_landmark_exception_bytes", landmark_exception_bytes },
            { "xkv_index_bytes",             index_bytes },
            { "xkv_codec_shared_bytes",      codec_shared_bytes },
            { "xkv_decode_tile_cache_bytes", decode_tile_cache_bytes },
            { "xkv_capture_bytes",           capture_bytes },
            { "xkv_candidate_bytes",         candidate_bytes },
            { "xkv_staging_bytes",           staging_bytes },
            { "xkv_snapshot_pinned_bytes",   snapshot_pinned_bytes },
            { "xkv_allocator_live_bytes",    allocator_live_bytes },
            { "xkv_allocator_reserved_bytes", allocator_reserved_bytes },
            { "xkv_device_peak_bytes",       device_peak_bytes },
            { "xkv_host_peak_bytes",         host_peak_bytes },
            { "xkv_unique_payloads",         unique_payloads },
            { "xkv_aliased_payloads",        aliased_payloads },
            { "xkv_baseline_same_rows_bytes", baseline_same_rows_bytes },
            { "xkv_covered_compressed_bytes", covered_compressed_bytes },
            // Unevaluated doubles render as JSON null (not_evaluated), never
            // as 0.0 — but only once XKV was observed. Unobserved (OFF)
            // snapshots keep exact legacy zeros. select/refine/reconstruct/
            // read/pack have no live hooks yet and are always unevaluated.
            { "xkv_factored_baseline_byte_coverage", !observed || ratios_evaluated ? json(factored_baseline_byte_coverage) : json(nullptr) },
            { "xkv_factor_quant_ratio",      !observed || ratios_evaluated ? json(factor_quant_ratio) : json(nullptr) },
            { "xkv_net_extra_compression_ratio", !observed || ratios_evaluated ? json(net_extra_compression_ratio) : json(nullptr) },
            { "xkv_net_extra_compression_ratio_reserved", !observed || ratios_evaluated ? json(net_extra_compression_ratio_reserved) : json(nullptr) },
            { "xkv_net_extra_compression_ratio_peak", !observed || ratios_evaluated ? json(net_extra_compression_ratio_peak) : json(nullptr) },
            { "xkv_ratios_evaluated",        observed && ratios_evaluated },
            { "xkv_seal_timers_evaluated",   observed && seal_timers_evaluated },
            { "xkv_quant_timers_evaluated",  observed && quant_timers_evaluated },
            { "xkv_seal_seconds",            !observed || seal_timers_evaluated ? json(seal_seconds) : json(nullptr) },
            { "xkv_factor_quant_seconds",    !observed || quant_timers_evaluated ? json(factor_quant_seconds) : json(nullptr) },
            { "xkv_landmark_quant_seconds",  !observed || quant_timers_evaluated ? json(landmark_quant_seconds) : json(nullptr) },
            { "xkv_select_seconds",          !observed ? json(select_seconds) : json(nullptr) },
            { "xkv_refine_seconds",          !observed ? json(refine_seconds) : json(nullptr) },
            { "xkv_reconstruct_seconds",     !observed ? json(reconstruct_seconds) : json(nullptr) },
            { "xkv_read_seconds",            !observed ? json(read_seconds) : json(nullptr) },
            { "xkv_pack_seconds",            !observed ? json(pack_seconds) : json(nullptr) },
            { "xkv_segments_sealed",         segments_sealed },
            { "xkv_segments_skipped_by_reason", skipped },
            // Observed-but-unevaluated counters render as JSON null
            // (not_evaluated), never 0-as-measured; unobserved (OFF)
            // snapshots keep exact legacy zeros. Mirrors the Prometheus
            // NaN branches below.
            { "xkv_sr_selected_rows",        !observed || sr_counters_evaluated ? json(sr_selected_rows) : json(nullptr) },
            { "xkv_sr_fragments",            !observed || sr_counters_evaluated ? json(sr_fragments) : json(nullptr) },
            { "xkv_effective_chunk_size",    effective_chunk_size },
            { "xkv_landmark_refine_rows",    !observed || sr_counters_evaluated ? json(landmark_refine_rows) : json(nullptr) },
            { "xkv_landmark_refine_cap_hits", !observed || sr_counters_evaluated ? json(landmark_refine_cap_hits) : json(nullptr) },
            { "xkv_spec_stale_total",        !observed || spec_counters_evaluated ? json(spec_stale_total) : json(nullptr) },
            { "xkv_transaction_abort_total", !observed || spec_counters_evaluated ? json(transaction_abort_total) : json(nullptr) },
            { "xkv_synchronize_total",       synchronize_total },
            { "xkv_throttle_total",          throttle_total },
            { "xkv_throttle_reason",         throttle_reason },
            { "xkv_graph_timings_evaluated", graph_timings_evaluated },
            { "xkv_pack_timer_evaluated",    pack_timer_evaluated },
            { "xkv_sr_counters_evaluated",   sr_counters_evaluated },
            { "xkv_spec_counters_evaluated", spec_counters_evaluated },
        };
    }

    void to_prometheus(std::ostream & os) const {
        if (empty()) {
            return;
        }
        const auto emit = [&os](const char * type, const char * name, const char * help, const auto & value) {
            os << "# HELP llamacpp:" << name << " " << help << "\n"
               << "# TYPE llamacpp:" << name << " " << type << "\n"
               << "llamacpp:" << name << " " << value << "\n";
        };
        const auto label_escape = [](const std::string & s) {
            std::string out;
            for (char c : s) {
                if (c == '\\' || c == '"') { out += '\\'; }
                out += c;
            }
            return out;
        };
        if (has_admission) {
            emit("gauge", "xkv_safe_next_ubatch", "Tokens safe to admit next.", safe_next_ubatch);
            emit("gauge", "xkv_logical_capacity", "Logical KV cells.", logical_capacity);
            emit("gauge", "xkv_logical_used", "Resident logical cells.", logical_used);
            emit("gauge", "xkv_hot_capacity", "Physical hot slots across streams.", hot_capacity);
            emit("gauge", "xkv_hot_used", "Committed hot slots.", hot_used);
            emit("gauge", "xkv_hot_free", "Immediately writable hot slots.", hot_free);
            emit("gauge", "xkv_hot_reserved", "Hot slots held by in-flight reservations.", hot_reserved);
            emit("gauge", "xkv_factor_live_bytes", "Live factor store payload bytes.", factor_live_bytes);
            emit("gauge", "xkv_factor_reserved_bytes", "Allocated factor store bytes.", factor_reserved_bytes);
            emit("gauge", "xkv_factor_budget_bytes", "Configured factor store budget.", factor_budget_bytes);
            emit("gauge", "xkv_factor_free_bytes", "Factor store budget headroom.", factor_free_bytes);
            emit("gauge", "xkv_factor_safe_tokens", "Factor store planner token bound.", factor_safe_tokens);
            emit("gauge", "xkv_workspace_live_bytes", "Currently leased workspace bytes.", workspace_live_bytes);
            emit("gauge", "xkv_workspace_peak_bytes", "Workspace high-water mark.", workspace_peak_bytes);
            emit("gauge", "xkv_workspace_budget_bytes", "Configured workspace budget.", workspace_budget_bytes);
            emit("gauge", "xkv_workspace_free_bytes", "Workspace budget headroom.", workspace_free_bytes);
            emit("gauge", "xkv_workspace_safe_tokens", "Workspace planner token bound.", workspace_safe_tokens);
            emit("gauge", "xkv_recurrent_capacity", "Recurrent slots.", recurrent_capacity);
            emit("gauge", "xkv_recurrent_used", "Resident recurrent slots.", recurrent_used);
            os << "# HELP llamacpp:xkv_limit_reason Admission limiting reason.\n"
               << "# TYPE llamacpp:xkv_limit_reason gauge\n"
               << "llamacpp:xkv_limit_reason{reason=\"" << label_escape(limit_reason) << "\"} 1\n";
        }
        if (!requested_profile.empty() || !effective_profile.empty()) {
            os << "# HELP llamacpp:xkv_profile_info Configured XKV profiles.\n"
               << "# TYPE llamacpp:xkv_profile_info gauge\n"
               << "llamacpp:xkv_profile_info{requested=\"" << label_escape(requested_profile)
               << "\",effective=\"" << label_escape(effective_profile) << "\"} 1\n";
        }
        if (!requested_mode.empty() || !effective_mode.empty()) {
            os << "# HELP llamacpp:xkv_mode_info Configured XKV modes (R2 dense vs R4 SR).\n"
               << "# TYPE llamacpp:xkv_mode_info gauge\n"
               << "llamacpp:xkv_mode_info{requested_mode=\"" << label_escape(requested_mode)
               << "\",effective_mode=\"" << label_escape(effective_mode) << "\"} 1\n";
            // Evaluator gate alias: xkv_mode{mode="..."}
            const std::string effective_or_req = !effective_mode.empty() ? effective_mode : requested_mode;
            os << "# HELP llamacpp:xkv_mode Effective XKV operational mode.\n"
               << "# TYPE llamacpp:xkv_mode gauge\n"
               << "llamacpp:xkv_mode{mode=\"" << label_escape(effective_or_req) << "\"} 1\n";
        }
        if (!effective_a_k.empty() || !requested_a_k.empty()) {
            os << "# HELP llamacpp:xkv_codec_profile_info Effective factor and landmark codec types.\n"
               << "# TYPE llamacpp:xkv_codec_profile_info gauge\n"
               << "llamacpp:xkv_codec_profile_info{"
               << "a_k=\"" << label_escape(!effective_a_k.empty() ? effective_a_k : requested_a_k) << "\","
               << "b_k=\"" << label_escape(!effective_b_k.empty() ? effective_b_k : requested_b_k) << "\","
               << "a_v=\"" << label_escape(!effective_a_v.empty() ? effective_a_v : requested_a_v) << "\","
               << "b_v=\"" << label_escape(!effective_b_v.empty() ? effective_b_v : requested_b_v) << "\","
               << "landmark=\"" << label_escape(!effective_landmark.empty() ? effective_landmark : requested_landmark) << "\","
               << "factorizer=\"" << label_escape(!effective_factorizer.empty() ? effective_factorizer : requested_factorizer) << "\","
               << "balance=\"" << label_escape(!effective_balance.empty() ? effective_balance : requested_balance) << "\""
               << "} 1\n";
        }
        if (!effective_seed.empty() || !requested_seed.empty()) {
            const std::string eff_seed = !effective_seed.empty() ? effective_seed : requested_seed;
            os << "# HELP llamacpp:xkv_seed_info Factorization seed hex.\n"
               << "# TYPE llamacpp:xkv_seed_info gauge\n"
               << "llamacpp:xkv_seed_info{seed=\"" << label_escape(eff_seed) << "\"} 1\n";
        }
        if (!effective_landmark_seed.empty() || !requested_landmark_seed.empty()) {
            const std::string eff_lm = !effective_landmark_seed.empty() ? effective_landmark_seed : requested_landmark_seed;
            os << "# HELP llamacpp:xkv_landmark_seed_info Landmark codec table seed hex.\n"
               << "# TYPE llamacpp:xkv_landmark_seed_info gauge\n"
               << "llamacpp:xkv_landmark_seed_info{seed=\"" << label_escape(eff_lm) << "\"} 1\n";
        }
        if (!model_sha256.empty()) {
            os << "# HELP llamacpp:model_artifact_info Model artifact content digest (SHA-256).\n"
               << "# TYPE llamacpp:model_artifact_info gauge\n"
               << "llamacpp:model_artifact_info{sha256=\"" << label_escape(model_sha256) << "\"} 1\n";
        }
        if (tri_scorer_valid || tri_calibration_fingerprint != 0 || !tri_ratio_str.empty()) {
            std::ostringstream ss_cal;
            ss_cal << std::hex << std::nouppercase << std::setw(16) << std::setfill('0') << tri_calibration_fingerprint;
            os << "# HELP llamacpp:tri_calibration_info Tri calibration content fingerprint and presence.\n"
               << "# TYPE llamacpp:tri_calibration_info gauge\n"
               << "llamacpp:tri_calibration_info{fingerprint=\"" << ss_cal.str() << "\",valid=\""
               << (tri_scorer_valid ? "true" : "false") << "\"";
            if (!tri_calibration_sha256.empty()) {
                os << ",sha256=\"" << label_escape(tri_calibration_sha256) << "\"";
            }
            os << "} 1\n";
            os << "# HELP llamacpp:tri_config_info Tri configured retention ratio and recent window.\n"
               << "# TYPE llamacpp:tri_config_info gauge\n"
               << "llamacpp:tri_config_info{ratio=\"" << label_escape(tri_ratio_str.empty() ? "3/32" : tri_ratio_str)
               << "\",recent_window=\"" << tri_recent_window << "\"} 1\n";
        }
        if (!source.empty()) {
            os << "# HELP llamacpp:xkv_source_info Observed XKV source.\n"
               << "# TYPE llamacpp:xkv_source_info gauge\n"
               << "llamacpp:xkv_source_info{source=\"" << label_escape(source) << "\"} 1\n";
        }
        // 64-bit fingerprints lose precision as Prometheus numbers (>2^53):
        // emit exact lowercase hex in info series (JSON retains uint64).
        const auto emit_fp = [&os, &label_escape](const char * name, const char * help, uint64_t fp) {
            std::ostringstream ss;
            ss << std::hex << std::nouppercase << std::setw(16) << std::setfill('0') << fp;
            os << "# HELP llamacpp:" << name << " " << help << "\n"
               << "# TYPE llamacpp:" << name << " gauge\n"
               << "llamacpp:" << name << "{fingerprint=\"" << ss.str() << "\"} 1\n";
            (void) label_escape;
        };
        emit_fp("xkv_codec_info", "Factor codec fingerprint.", codec_fingerprint);
        emit_fp("xkv_backend_info", "Backend fingerprint.", backend_fingerprint);
        // NOTE: the source fingerprint series is xkv_source_fingerprint_info
        // (not xkv_source_info, which already carries the {source} label).
        // Emitting both HELP/TYPE blocks under one name is a duplicate.
        emit_fp("xkv_source_fingerprint_info", "Source fingerprint.", source_fingerprint);
        emit_fp("xkv_profile_fingerprint_info", "Profile fingerprint.", profile_fingerprint);
        emit_fp("xkv_config_fingerprint_info", "Effective config fingerprint.", config_fingerprint);
        emit("gauge", "xkv_rank_k", "Factor rank K.", rank_k);
        emit("gauge", "xkv_rank_v", "Factor rank V.", rank_v);
        emit("gauge", "xkv_factor_streams", "Encoded factor streams.", factor_streams);
        emit("gauge", "xkv_actual_bytes", "Actual XKV bytes.", actual_bytes);
        emit("gauge", "xkv_nominal_bytes", "Nominal XKV bytes.", nominal_bytes);
        emit("gauge", "xkv_hot_bytes", "Hot KV cache bytes.", hot_bytes);
        emit("gauge", "xkv_flat_bytes", "Flat quantized bytes.", flat_bytes);
        emit("gauge", "xkv_factor_ak_bytes", "Factor A_K bytes.", factor_ak_bytes);
        emit("gauge", "xkv_factor_bk_bytes", "Factor B_K bytes.", factor_bk_bytes);
        emit("gauge", "xkv_factor_av_bytes", "Factor A_V bytes.", factor_av_bytes);
        emit("gauge", "xkv_factor_bv_bytes", "Factor B_V bytes.", factor_bv_bytes);
        emit("gauge", "xkv_factor_payload_bytes", "Factor payload bytes.", factor_payload_bytes);
        emit("gauge", "xkv_factor_metadata_bytes", "Factor metadata bytes.", factor_metadata_bytes);
        emit("gauge", "xkv_factor_padding_bytes", "Factor padding bytes.", factor_padding_bytes);
        emit("gauge", "xkv_landmark_payload_bytes", "Landmark payload bytes.", landmark_payload_bytes);
        emit("gauge", "xkv_landmark_metadata_bytes", "Landmark metadata bytes.", landmark_metadata_bytes);
        emit("gauge", "xkv_landmark_exception_bytes", "Landmark exception bytes.", landmark_exception_bytes);
        emit("gauge", "xkv_index_bytes", "Index bytes.", index_bytes);
        emit("gauge", "xkv_codec_shared_bytes", "Shared codec table bytes.", codec_shared_bytes);
        emit("gauge", "xkv_decode_tile_cache_bytes", "Decode tile cache bytes.", decode_tile_cache_bytes);
        emit("gauge", "xkv_capture_bytes", "Source capture bytes.", capture_bytes);
        emit("gauge", "xkv_candidate_bytes", "Candidate encoding scratch bytes.", candidate_bytes);
        emit("gauge", "xkv_staging_bytes", "Transient device staging bytes.", staging_bytes);
        emit("gauge", "xkv_snapshot_pinned_bytes", "Snapshot pinned bytes.", snapshot_pinned_bytes);
        emit("gauge", "xkv_allocator_live_bytes", "Allocator live bytes.", allocator_live_bytes);
        emit("gauge", "xkv_allocator_reserved_bytes", "Allocator reserved bytes.", allocator_reserved_bytes);
        emit("gauge", "xkv_device_peak_bytes", "Device peak bytes.", device_peak_bytes);
        emit("gauge", "xkv_host_peak_bytes", "Host peak bytes.", host_peak_bytes);
        emit("gauge", "xkv_unique_payloads", "Unique payloads.", unique_payloads);
        emit("gauge", "xkv_aliased_payloads", "Aliased payloads.", aliased_payloads);
        emit("gauge", "xkv_baseline_same_rows_bytes", "Baseline same-rows bytes.", baseline_same_rows_bytes);
        emit("gauge", "xkv_covered_compressed_bytes", "Compressed bytes covering the baseline.", covered_compressed_bytes);
        // Unevaluated doubles render as NaN (not_evaluated), never 0.0 — but
        // only once XKV was observed; unobserved snapshots keep legacy output.
        // select/refine/reconstruct/read/pack have no live hooks (always NaN
        // once observed). NaN is valid Prometheus exposition; Go/Python
        // parsers accept it case-insensitively.
        const auto emit_opt = [&](const char * type, const char * name, const char * help,
                                  double value, bool evaluated) {
            if (!observed || evaluated) {
                emit(type, name, help, value);
            } else {
                os << "# HELP llamacpp:" << name << " " << help << "\n"
                   << "# TYPE llamacpp:" << name << " " << type << "\n"
                   << "llamacpp:" << name << " NaN\n";
            }
        };
        emit_opt("gauge", "xkv_factored_baseline_byte_coverage", "Factored baseline byte coverage.", factored_baseline_byte_coverage, ratios_evaluated);
        emit_opt("gauge", "xkv_factor_quant_ratio", "Factor quantization ratio.", factor_quant_ratio, ratios_evaluated);
        emit_opt("gauge", "xkv_net_extra_compression_ratio", "Net extra compression ratio.", net_extra_compression_ratio, ratios_evaluated);
        emit_opt("gauge", "xkv_net_extra_compression_ratio_reserved", "Net extra compression ratio (reserved basis).", net_extra_compression_ratio_reserved, ratios_evaluated);
        emit_opt("gauge", "xkv_net_extra_compression_ratio_peak", "Net extra compression ratio (peak basis).", net_extra_compression_ratio_peak, ratios_evaluated);
        emit_opt("counter", "xkv_seal_seconds", "Wall time sealing segments.", seal_seconds, seal_timers_evaluated);
        emit_opt("counter", "xkv_factor_quant_seconds", "Wall time quantizing factors.", factor_quant_seconds, quant_timers_evaluated);
        emit_opt("counter", "xkv_landmark_quant_seconds", "Wall time quantizing landmarks.", landmark_quant_seconds, quant_timers_evaluated);
        emit_opt("counter", "xkv_select_seconds", "Wall time selecting rows.", select_seconds, graph_timings_evaluated);
        emit_opt("counter", "xkv_refine_seconds", "Wall time refining landmarks.", refine_seconds, graph_timings_evaluated);
        emit_opt("counter", "xkv_reconstruct_seconds", "Wall time reconstructing rows.", reconstruct_seconds, graph_timings_evaluated);
        emit_opt("counter", "xkv_read_seconds", "Wall time reading segments.", read_seconds, graph_timings_evaluated);
        emit_opt("counter", "xkv_pack_seconds", "Wall time packing segments.", pack_seconds, pack_timer_evaluated);
        os << "# HELP llamacpp:xkv_ratios_evaluated Ratios comparable (not not_evaluated).\n"
           << "# TYPE llamacpp:xkv_ratios_evaluated gauge\n"
           << "llamacpp:xkv_ratios_evaluated " << (observed && ratios_evaluated ? 1 : 0) << "\n";
        os << "# HELP llamacpp:xkv_timers_evaluated Seal/quant timers accumulated live.\n"
           << "# TYPE llamacpp:xkv_timers_evaluated gauge\n"
           << "llamacpp:xkv_timers_evaluated " << (observed && (seal_timers_evaluated || quant_timers_evaluated) ? 1 : 0) << "\n";
        os << "# HELP llamacpp:xkv_graph_timings_evaluated Graph execution timings accumulated live.\n"
           << "# TYPE llamacpp:xkv_graph_timings_evaluated gauge\n"
           << "llamacpp:xkv_graph_timings_evaluated " << (observed && graph_timings_evaluated ? 1 : 0) << "\n";
        os << "# HELP llamacpp:xkv_pack_timer_evaluated Segment pack timings accumulated live.\n"
           << "# TYPE llamacpp:xkv_pack_timer_evaluated gauge\n"
           << "llamacpp:xkv_pack_timer_evaluated " << (observed && pack_timer_evaluated ? 1 : 0) << "\n";
        os << "# HELP llamacpp:xkv_sr_counters_evaluated SR fragments and row counters evaluated.\n"
           << "# TYPE llamacpp:xkv_sr_counters_evaluated gauge\n"
           << "llamacpp:xkv_sr_counters_evaluated " << (observed && sr_counters_evaluated ? 1 : 0) << "\n";
        os << "# HELP llamacpp:xkv_spec_counters_evaluated Speculative draft counters evaluated.\n"
           << "# TYPE llamacpp:xkv_spec_counters_evaluated gauge\n"
           << "llamacpp:xkv_spec_counters_evaluated " << (observed && spec_counters_evaluated ? 1 : 0) << "\n";
        emit("counter", "xkv_segments_sealed", "Segments sealed.", segments_sealed);
        os << "# HELP llamacpp:xkv_segments_skipped_total Segments skipped by reason.\n"
           << "# TYPE llamacpp:xkv_segments_skipped_total counter\n";
        for (const auto & kv : segments_skipped_by_reason) {
            os << "llamacpp:xkv_segments_skipped_total{reason=\"" << label_escape(kv.first) << "\"} " << kv.second << "\n";
        }
        if (!observed || sr_counters_evaluated) {
            emit("gauge", "xkv_sr_selected_rows", "SR selected rows.", sr_selected_rows);
            emit("gauge", "xkv_sr_fragments", "SR fragments.", sr_fragments);
            emit("gauge", "xkv_landmark_refine_rows", "Landmark refine rows.", landmark_refine_rows);
            emit("gauge", "xkv_landmark_refine_cap_hits", "Landmark refine cap hits.", landmark_refine_cap_hits);
        } else {
            os << "# HELP llamacpp:xkv_sr_selected_rows SR selected rows.\n# TYPE llamacpp:xkv_sr_selected_rows gauge\nllamacpp:xkv_sr_selected_rows NaN\n";
            os << "# HELP llamacpp:xkv_sr_fragments SR fragments.\n# TYPE llamacpp:xkv_sr_fragments gauge\nllamacpp:xkv_sr_fragments NaN\n";
            os << "# HELP llamacpp:xkv_landmark_refine_rows Landmark refine rows.\n# TYPE llamacpp:xkv_landmark_refine_rows gauge\nllamacpp:xkv_landmark_refine_rows NaN\n";
            os << "# HELP llamacpp:xkv_landmark_refine_cap_hits Landmark refine cap hits.\n# TYPE llamacpp:xkv_landmark_refine_cap_hits gauge\nllamacpp:xkv_landmark_refine_cap_hits NaN\n";
        }
        if (!observed || spec_counters_evaluated) {
            emit("counter", "xkv_spec_stale_total", "Stale speculative drafts.", spec_stale_total);
            emit("counter", "xkv_transaction_abort_total", "Transaction aborts.", transaction_abort_total);
        } else {
            os << "# HELP llamacpp:xkv_spec_stale_total Stale speculative drafts.\n# TYPE llamacpp:xkv_spec_stale_total counter\nllamacpp:xkv_spec_stale_total NaN\n";
            os << "# HELP llamacpp:xkv_transaction_abort_total Transaction aborts.\n# TYPE llamacpp:xkv_transaction_abort_total counter\nllamacpp:xkv_transaction_abort_total NaN\n";
        }
        emit("gauge", "xkv_effective_chunk_size", "Effective chunk size.", effective_chunk_size);
        emit("counter", "xkv_synchronize_total", "Backend synchronizations.", synchronize_total);
        if (!observed || compression_goal_evaluated) {
            emit("gauge", "xkv_compression_goal_met", "Compression goal met.", compression_goal_met ? 1 : 0);
        } else {
            os << "# HELP llamacpp:xkv_compression_goal_met Compression goal met.\n"
               << "# TYPE llamacpp:xkv_compression_goal_met gauge\n"
               << "llamacpp:xkv_compression_goal_met NaN\n";
        }
        os << "# HELP llamacpp:xkv_compression_goal_evaluated Compression goal evaluated.\n"
           << "# TYPE llamacpp:xkv_compression_goal_evaluated gauge\n"
           << "llamacpp:xkv_compression_goal_evaluated " << (observed && compression_goal_evaluated ? 1 : 0) << "\n";
        emit("counter", "xkv_throttle_total", "Admission throttles.", throttle_total);
        if (!throttle_reason.empty()) {
            os << "# HELP llamacpp:xkv_throttle_reason Current admission throttle reason.\n"
               << "# TYPE llamacpp:xkv_throttle_reason gauge\n"
               << "llamacpp:xkv_throttle_reason{reason=\"" << label_escape(throttle_reason) << "\"} 1\n";
        }
    }
};

// struct for tracking the state of a task (e.g., for streaming)
struct task_result_state {
    // tracking diffs for partial tool calls
    std::vector<common_chat_msg_diff> diffs;
    common_chat_parser_params chat_parser_params;
    common_chat_msg chat_msg;
    std::string generated_text; // append new chunks of generated text here
    std::string rerot_reasoning_prefix;
    bool rerot_serial_started = false;
    std::vector<std::string> generated_tool_call_ids;
    std::unordered_set<size_t> sent_tool_call_names;

    // The first payload chunk owns the assistant role / message-start event.
    // A begin marker only flushes HTTP headers and does not consume this flag.
    bool stream_started = false;

    // for OpenAI Responses and Anthropic streaming API:
    // track output item / content block state across chunks
    bool thinking_block_started = false;
    bool text_block_started = false;

    // for OpenAI Responses streaming API
    bool oai_resp_created = false;
    const std::string oai_resp_id;
    const std::string oai_resp_reasoning_id;
    const std::string oai_resp_message_id;
    std::string oai_resp_fc_id; // function call ID for current args delta

    task_result_state(const common_chat_parser_params & chat_parser_params);

    // parse partial tool calls and update the internal state
    common_chat_msg update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls = false);

    common_chat_msg update_rerot_msg(
        const std::string & reasoning,
        const std::string & content,
        std::vector<common_chat_msg_diff> & diffs);

    common_chat_msg update_rerot_content(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs);
};

struct server_task {
    int id = -1; // to be filled by server_queue

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // used when there are multiple prompts (batch request)

    // used by SERVER_TASK_TYPE_CANCEL
    int id_target = -1;
    int id_slot   = -1;
    std::string cache_key;

    // used by parallel sampling (multiple completions from same prompt)
    int id_parent  = -1;
    // temporary store of child tasks for scheduling
    // note: accessing to elements is invalid after the task is moved to server_slot
    std::vector<server_task> child_tasks;

    // used by SERVER_TASK_TYPE_INFERENCE
    task_params   params;
    server_tokens tokens;
    bool is_retry = false;

    // only used by CLI, this allow tokenizing CLI inputs on server side
    // we need this because mtmd_context and vocab are not accessible outside of server_context
    bool                    cli = false;
    std::string             cli_prompt;
    std::vector<raw_buffer> cli_files;

    server_task_type type;

    // used by SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE, SERVER_TASK_TYPE_SLOT_ERASE
    struct slot_action {
        // All task types move this member, including ordinary completions.
        int id_slot = -1;
        std::string filename;
        std::string filepath;
    };
    slot_action slot_action;

    // used by SERVER_TASK_TYPE_METRICS
    bool metrics_reset_bucket = false;

    // used by SERVER_TASK_TYPE_SET_LORA
    std::map<int, float> set_lora; // mapping adapter ID -> scale

    // RERoT episode bookkeeping (A.13-A.15, §18). episode_id is the visibility
    // domain: 0 = no episode assigned (RERoT OFF or not yet adopted). Each
    // outer completion (this task plus each child from n_cmpl) owns an
    // independent episode; different completions never share an episode id,
    // while outer shared prompt cells stay shared as the untagged prefix.
    // generation guards runtime callbacks against ABA across cancel/retry:
    // stale callbacks carry an older generation and must be dropped.
    uint64_t rerot_episode_id = 0;
    uint64_t rerot_generation = 0;
    // Exact last real user text for the private final-acquire instruction.
    // This is never published as a RERoT run; the original prompt remains the
    // causal source. Empty for non-chat/multimedia-only requests.
    std::string rerot_original_user_text;

    server_rerot_episode_key rerot_key() const;
    bool rerot_key_matches(int task, uint64_t episode, uint64_t gen) const;
    void rerot_bump_generation();
    // Assign independent episode ids to this task and every child (A.13.1).
    // Outer shared prompt cells stay shared; visibility diverges only after
    // the fork point via distinct episode ids. No-op when params RERoT is not
    // effective for this task type. Consumes ids from next_episode_id (never 0).
    void assign_rerot_episodes(uint64_t & next_episode_id, uint64_t generation = 1);
    // Response ownership (§18.3): the HTTP/SSE stream belongs to the outer
    // completion task id, never to an internal lane. Internal lanes must map
    // their output through this id and must never emit SSE/HTTP themselves.
    int rerot_response_owner() const;
    // Effective switch for this task instance (params + type gate, A.19).
    bool rerot_effective() const;

    server_task() = default;

    server_task(server_task_type type) : type(type) {}

    int32_t n_tokens() const {
        return tokens.size();
    }

    bool need_embd() const {
        switch (type) {
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                return true;
            default:
                return false;
        }
    }

    bool need_logits() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
                return true;
            default:
                return false;
        }
    }

    bool need_sampling() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
                return true;
            default:
                return false;
        }
    }

    // utility function
    static std::unordered_set<int> get_list_id(const std::vector<server_task> & tasks) {
        std::unordered_set<int> ids(tasks.size());
        for (size_t i = 0; i < tasks.size(); i++) {
            ids.insert(tasks[i].id);
            for (auto & child : tasks[i].child_tasks) {
                ids.insert(child.id);
            }
        }
        return ids;
    }

    void add_child(int id_parent, int id_child) {
        server_task copy;

        copy.id        = id_child;
        copy.id_parent = id_parent;
        copy.params    = params;
        copy.type      = type;
        copy.tokens    = tokens.clone();
        copy.id_slot   = -1; // child tasks cannot specify slot
        copy.cache_key.clear();
        // Outer n_cmpl isolation (A.13.1): the child starts with no episode.
        // assign_rerot_episodes() gives parent and each child independent
        // episode ids afterwards; episode 0 must never be shared.
        copy.rerot_episode_id = 0;
        copy.rerot_generation = rerot_generation;
        copy.rerot_original_user_text = rerot_original_user_text;

        // use different sampling seed for each child
        // note: https://github.com/ggml-org/llama.cpp/pull/18700#discussion_r2675115723
        if (copy.params.sampling.seed != LLAMA_DEFAULT_SEED) {
            copy.params.sampling.seed += (uint32_t)child_tasks.size() + 1;
        }

        child_tasks.push_back(std::move(copy));
    }

    // the task will be moved into queue, then onto slots
    // however, the state must be kept by caller (e.g., HTTP thread)
    task_result_state create_state() const {
        return task_result_state(params.chat_parser_params);
    }

    bool is_parent() const {
        return child_tasks.size() > 0;
    }

    bool is_child() const {
        return id_parent != -1;
    }
};

struct result_timings {
    int32_t cache_n = -1;

    int32_t prompt_n = -1;
    double prompt_ms = 0.0;
    double prompt_per_token_ms = 0.0;
    double prompt_per_second = 0.0;

    int32_t predicted_n = -1;
    double predicted_ms = 0.0;
    double predicted_per_token_ms = 0.0;
    double predicted_per_second = 0.0;

    // Optional speculative metrics - only included when > 0
    int32_t draft_n = 0;
    int32_t draft_n_accepted = 0;

    json to_json() const;
};

struct result_prompt_progress {
    int32_t total = 0;
    int32_t cache = 0;
    int32_t processed = 0;
    int64_t time_ms = 0;

    json to_json() const;
};

struct server_task_result {
    int id           = -1;
    int id_slot      = -1;

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // to be used for batched tasks

    virtual bool is_error() {
        // only used by server_task_result_error
        return false;
    }
    virtual bool is_stop() {
        // only used by server_task_result_cmpl_*
        return true;
    }
    virtual void update(task_result_state &) {
        // only used by server_task_result_cmpl_*
    }
    virtual json to_json() = 0;
    virtual ~server_task_result() = default;
    virtual server_task_result * clone() const {
        GGML_ABORT("not implemented for this task type");
    }
};

// using shared_ptr for polymorphism of server_task_result
using server_task_result_ptr = std::unique_ptr<server_task_result>;

struct completion_token_output {
    llama_token tok;
    float prob;
    std::string text_to_send;
    struct prob_info {
        llama_token tok;
        std::string txt;
        float prob;
    };
    std::vector<prob_info> probs;

    json to_json(bool post_sampling_probs) const;

    static json probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs);

    static float logarithm(float x);

    static std::vector<unsigned char> str_to_bytes(const std::string & str);

};

struct server_task_result_cmpl_final : server_task_result {
    std::string content;
    std::string rerot_reasoning;
    bool rerot_explicit_channels = false;
    llama_tokens tokens;

    bool stream;
    bool include_usage;
    result_timings timings;
    std::string prompt;

    bool truncated;
    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;
    int32_t n_tokens_cached;
    bool has_new_line;
    std::string stopping_word;
    stop_type stop = STOP_TYPE_NONE;

    uint64_t rerot_probe_tokens = 0;
    uint64_t rerot_frame_tokens = 0;
    uint64_t rerot_source_end_tokens = 0;

    bool post_sampling_probs;
    std::vector<completion_token_output> probs_output;
    std::vector<std::string>  response_fields;

    task_params generation_params;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    common_chat_msg    oaicompat_msg; // to be populated by update()

    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // for OpenAI Responses API
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;

    virtual bool is_stop() override {
        return true; // in stream mode, final responses are considered stop
    }

    virtual json to_json() override;

    virtual void update(task_result_state & state) override {
        is_updated = true;
        if (rerot_explicit_channels) {
            oaicompat_msg = stream
                ? state.update_rerot_content(
                    content, false, oaicompat_msg_diffs)
                : state.update_rerot_msg(
                    rerot_reasoning, content, oaicompat_msg_diffs);
        } else {
            oaicompat_msg =
                state.update_chat_msg(content, false, oaicompat_msg_diffs);
        }

        oai_resp_id = state.oai_resp_id;
        oai_resp_reasoning_id = state.oai_resp_reasoning_id;
        oai_resp_message_id = state.oai_resp_message_id;
    }

    json to_json_non_oaicompat();

    json usage_json_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_chat_stream();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_resp_stream();

    json to_json_oaicompat_asr();

    json to_json_anthropic();

    json to_json_anthropic_stream();
};

struct server_task_result_cmpl_partial : server_task_result {
    std::string  content;
    llama_tokens tokens;

    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;

    bool post_sampling_probs;
    bool is_progress = false;
    bool is_begin = false; // whether to send 200 status to HTTP client (begin of SSE stream)
                           // ref: https://github.com/ggml-org/llama.cpp/pull/23884
    completion_token_output prob_output;
    result_timings timings;
    result_prompt_progress progress;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;
    bool first_chunk = false;

    // RERoT reasoning is already structured. Final-answer bytes still pass
    // through the request chat parser so tool calls retain their API shape.
    bool is_rerot_reasoning = false;
    bool is_rerot_content   = false;

    // Streaming state copied from task_result_state for this chunk
    bool thinking_block_started = false;
    bool text_block_started     = false;

    // for OpenAI Responses API
    bool oai_resp_created = false;
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;
    std::string oai_resp_fc_id;

    // for Anthropic API: track if any reasoning content has been generated
    bool anthropic_has_reasoning = false;

    virtual bool is_stop() override {
        return false; // in stream mode, partial responses are not considered stop
    }

    virtual void update(task_result_state & state) override;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_asr();

    json to_json_anthropic();
};

struct server_task_result_embd : server_task_result {
    std::vector<std::vector<float>> embedding;

    int32_t n_tokens;

    // response formatting
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();
};

struct server_task_result_rerank : server_task_result {
    float score = -1e6;

    int32_t n_tokens;

    virtual json to_json() override;
};

struct server_task_result_error : server_task_result {
    error_type err_type = ERROR_TYPE_SERVER;
    std::string err_msg;

    // for ERROR_TYPE_EXCEED_CONTEXT_SIZE
    int32_t n_prompt_tokens = 0;
    int32_t n_ctx           = 0;

    virtual bool is_error() override {
        return true;
    }

    virtual json to_json() override;
};

struct server_task_result_metrics : server_task_result {
    // XKV planner policy (§16), inline for unit tests. Fill-first: sealing
    // maintenance runs only when hot slots cannot cover the upcoming batch
    // (or the snapshot is unknown); SHADOW evaluate-only always runs.
    // No fixed attempt constant: one maintain seals ~one segment, so the
    // bound derives from the deficit: ceil(deficit/segment_tokens) plus one
    // per eligible token domain. Unknown snapshots fall back to the domain
    // count (or hot_capacity/segment when known).
    static uint64_t xkv_maintain_max_attempts(bool xkv_enabled, bool have_snapshot,
                                               uint32_t hot_free, uint32_t hot_capacity,
                                               uint32_t required_kv, uint32_t seg_tokens) {
        if (!xkv_enabled) {
            return 0;
        }
        constexpr uint64_t kDomains = 4; // logical/hot/factor/workspace
        if (!have_snapshot) {
            if (hot_capacity > 0 && seg_tokens > 0) {
                return hot_capacity / seg_tokens + kDomains;
            }
            return kDomains;
        }
        if (required_kv <= hot_free) {
            return 0;
        }
        const uint64_t deficit = (uint64_t) required_kv - hot_free;
        if (seg_tokens == 0) {
            return kDomains;
        }
        return (deficit + seg_tokens - 1) / seg_tokens + kDomains;
    }
    static bool xkv_should_maintain(bool xkv_enabled, bool is_shadow, bool have_snapshot,
                                      uint32_t hot_free, uint32_t required_kv,
                                      uint64_t attempts, uint64_t max_attempts) {
        if (!xkv_enabled || attempts >= max_attempts) {
            return false;
        }
        if (is_shadow) {
            return true;
        }
        if (!have_snapshot) {
            return true;
        }
        return hot_free < required_kv;
    }

    int n_idle_slots;
    int n_processing_slots;
    int n_tasks_deferred;
    int64_t t_start;

    // TODO: somehow reuse server_metrics in the future, instead of duplicating the fields
    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    // TriAttention runtime observability. Counters are cumulative for the
    // process lifetime; cell/reference values describe the most recent
    // successful reclaim operation.
    uint64_t tri_drain_total              = 0;
    uint64_t tri_maintenance_total        = 0;
    uint64_t tri_floor_exhausted_total    = 0;
    uint64_t tri_atomic_fallback_kv_total = 0;
    uint64_t tri_atomic_fallback_recurrent_total = 0;
    uint64_t tri_cells_freed_total        = 0;
    uint64_t tri_score_us_total           = 0;
    uint64_t tri_pack_us_total            = 0;

    uint64_t tri_cells_before       = 0;
    uint64_t tri_cells_after        = 0;
    uint64_t tri_cells_freed        = 0;
    uint64_t tri_references_removed = 0;
    uint64_t tri_target_references  = 0;
    uint64_t tri_hard_keep          = 0;
    uint64_t tri_shared_keep        = 0;

    // RERoT metrics (§A.26), additive to Tri metrics. Empty (all zero) when
    // RERoT OFF: to_json omits every rerot_* key, so OFF responses keep the
    // exact pre-RERoT schema with no allocation for the rerot block.
    server_rerot_metrics rerot;

    // FlashPrefill V2 GPU-determined totals (MetricsIntegration owner).
    // Copied from server_metrics (context drain merged post_decode,
    // successful-only) by the SERVER_TASK_TYPE_METRICS handler. All zero
    // while OFF (serializer emits no flashprefill series then). dense_rows
    // index order: 0 decode, 1 mtp_verify, 2 role_other, 3 short_context,
    // 4 dense_tail, 5 unknown_boundary, 6 unsupported, 7 high_cost, 8 no_plan,
    // 9 full_attention_layer (graph summary actuals);
    // pool_rebuild: 0 slice, 1 exact_all; plan_invalidations: 0 bypassed,
    // 1 empty_snapshot, 2 no_plan. Server-side eligible/dense-reason totals
    // ride the existing fp_eligible_rows/fp_dense_by_reason fields below
    // (ServerRouting owner) and are not duplicated here.
    uint64_t fp_sparse_rows = 0;
    uint64_t fp_dense_packed = 0;
    uint64_t fp_dense_rows[10] = {};
    uint64_t fp_selected_blocks = 0;
    uint64_t fp_corrected_blocks = 0;
    uint64_t fp_visible_tokens = 0;
    uint64_t fp_exact_tokens = 0;
    uint64_t fp_pool_rebuild[2] = {};
    uint64_t fp_plan_invalidations[3] = {};
    uint64_t fp_scratch_live_bytes = 0;
    uint64_t fp_scratch_peak_bytes = 0;
    uint64_t fp_layout_us_total = 0;
    uint64_t fp_layout_slices_measured = 0;
    uint64_t fp_gpu_pool_us_total = 0;
    uint64_t fp_gpu_select_us_total = 0;
    uint64_t fp_gpu_attn_us_total = 0;
    uint64_t fp_gpu_slices_measured = 0;

    // FlashPrefill server-side attribution (ServerRouting owner; copied here
    // for the serializer, same as the Tri fields above).
    uint64_t fp_eligible_rows = 0;
    uint64_t fp_dense_by_reason[7] = {};
    uint64_t fp_policy_fingerprint = 0;
    bool     fp_has_policy = false;

    // XKV metrics (§16), additive to Tri/RERoT metrics. Empty when XKV OFF:
    // to_json omits every xkv_* key, so OFF responses keep the exact
    // pre-XKV schema.
    server_xkv_metrics xkv;

    // while we can also use std::vector<server_slot> this requires copying the slot object which can be quite messy
    // therefore, we use json to temporarily store the slot.to_json() result
    json slots_data = json::array();

    virtual json to_json() override;
};

// --- RERoT outer-compat free helpers (defined in server-task.cpp unless noted) ---
//
// Ownership: episode scheduling, KV visibility, and the frozen llama_rerot_*
// C-API (ContextGlue, include/llama.h) are consumed by
// tools/server/server-rerot.*. The helpers below are the outer-compat surface
// only: request plumbing (server-common.cpp), grammar/tool isolation, LoRA
// inheritance, multimodal/embedding gates, response ownership, streaming trace
// gating, cancellation fan-out, and metrics merging. Every helper early-outs
// with no allocation when RERoT is not effective.

// Apply raw request JSON rerot keys ("rerot", "rerot_frontier",
// "rerot_trace") onto task.params after schema eval, starting from base
// globals. Unknown/absent keys keep schema/base values. Defined in
// server-common.cpp (request plumbing lives with the oaicompat parsers).
void server_rerot_apply_request_json(server_task & task, const json & data, const common_params & base);

// Validate the task's effective RERoT request (Stage-0 gates + A.19).
// Defined in server-common.cpp.
bool server_rerot_validate_task(const server_task & task, std::string & error);

// Grammar isolation (§26/A.16.1): the planner <ol> constraint must never
// pollute the user grammar. Save the user grammar before planner injection,
// restore the stock tool/JSON path after the final fence.
inline common_grammar server_rerot_take_user_grammar(task_params & params) {
    // A.16.1: snapshot the user/tool grammar before planner <ol> injection so
    // the planner constraint can never pollute it. Cheap move, no parse.
    common_grammar saved = std::move(params.sampling.grammar);
    params.sampling.grammar = common_grammar{};
    return saved;
}

inline void server_rerot_restore_user_grammar(task_params & params, const common_grammar & saved) {
    // Restore the stock user/tool path after the final fence (§26). The planner
    // grammar is discarded; the serial tail decodes with the original sampler.
    params.sampling.grammar = saved;
}

// Tool execution is disabled during concurrent reasoning; only the final
// acquire fence + serial tail restores the stock tool path (§26).
bool server_rerot_tool_calls_allowed(bool serial_tail_done);

// LoRA inheritance (A.17): all lanes share the root adapters; batching honors
// the existing adapter constraints. Returns false + error on mismatch.
bool server_rerot_check_lora_inheritance(
    const std::vector<common_adapter_lora_info> & root_loras,
    const std::vector<common_adapter_lora_info> & lane_loras,
    std::string & error);
bool server_rerot_can_batch_with(const task_params & a, const task_params & b);

// Multimodal prelude-then-fork (A.18): fork is allowed only after the shared
// multimodal prefill; DDVR never remaps visual positions. Returns false for
// visual-remap attempts (unsupported-error at the call site).
bool server_rerot_visual_remap_allowed();
bool server_rerot_fork_ready(bool has_media, bool prelude_done);

// Streaming (A.14): committed PUBLIC Lane text is emitted as complete
// reasoning lines in actual completion order. PENDING/PRIVATE bytes never
// enter client deltas; rerot.trace.* remains separate and explicitly opt-in.
bool server_rerot_trace_allowed(const task_params & params);
json server_rerot_trace_event(
    const std::string & kind,
    uint64_t episode_id,
    uint64_t node_id,
    uint64_t frontier,
    const json & data = json::object());

// Cancellation/retry fan-out (§A.15): collect every outer task id in the
// episode rooted at root (root id + child ids). The runtime clears queued
// children, parked lineage, pending XML, and episode KV refs atomically;
// shared outer prompt refs survive when other requests still use them.
std::vector<int> server_rerot_cancel_targets(const server_task & root);

struct server_task_result_slot_save_load : server_task_result {
    std::string filename;
    bool is_save; // true = save, false = load

    size_t n_tokens;
    size_t n_bytes;
    double t_ms;

    virtual json to_json() override;
};

struct server_task_result_slot_erase : server_task_result {
    size_t n_erased;

    virtual json to_json() override;
};

struct server_task_result_control : server_task_result {
    bool        success = false;
    std::string message; // optional detail when success is false

    virtual json to_json() override {
        json out = json { { "success", success } };
        if (!message.empty()) {
            out["message"] = message;
        }
        return out;
    }
};

struct server_task_result_get_lora : server_task_result {
    struct lora {
        common_adapter_lora_info info;
        std::string  alora_invocation_string;
        llama_tokens alora_invocation_tokens;
    };
    std::vector<lora> loras;

    virtual json to_json() override;
};

struct server_task_result_apply_lora : server_task_result {
    virtual json to_json() override;
};

struct server_prompt {
    server_tokens tokens;

    std::list<common_prompt_checkpoint> checkpoints;

    void clear() {
        tokens.clear();
        checkpoints.clear();
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            checkpoints,
        };
    }
};

struct server_prompt_data {
    std::vector<uint8_t> main;
    std::vector<uint8_t> drft;
    std::vector<uint8_t> spec;

    size_t size() const {
        return main.size() + drft.size() + spec.size();
    }
};

struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    size_t size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt.size();
        }

        return res;
    }
};

struct server_prompt_cache {
    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens) {
        this->limit_size   = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
        this->limit_tokens = limit_tokens;
    }

    std::list<server_prompt_cache_state> states;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    size_t size() const;

    size_t n_tokens() const;

    server_prompt_cache_state * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft, size_t state_size_spec);

    bool load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot,
              std::vector<uint8_t> * state_spec, bool * loaded_state);

    void update();
};

// used exclusively by router mode
struct server_task_result_router : server_task_result {
    json data;
    virtual json to_json() override { return data; }
    virtual server_task_result * clone() const override {
        return new server_task_result_router(*this);
    }
};
