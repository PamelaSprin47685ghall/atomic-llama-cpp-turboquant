#pragma once

#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"

#include "llama-flashprefill.h"

#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>

// FlashPrefill V2 server routing: pure, server-independent portion.
// Ownership: ServerRouting. Frozen policy types are PolicyCore's
// include/llama-flashprefill.h; the versioned execution copy/hook is
// ContextIntegration's (llama_decode_with_flashprefill + policy fingerprint
// getter); config declarations are ConfigIntegration's (llama.h). The GGML
// wire schema (WireReference) is not needed on the server path.
// These helpers classify rows and freeze boundaries without touching KV,
// graph, or GPU state, so tests/test-flashprefill-state.cpp can use them
// standalone. Later state-policy work extends (not replaces) this section.

// Forward declaration of the public drain snapshot (fully defined in
// llama.h by ConfigIntegration; complete type needed only in translation
// units that merge or emit, which include llama.h). Global scope: the merge
// declaration inside the namespace below refers to this same global entity.
struct llama_flashprefill_metrics_slice;

namespace server_flashprefill_routing {

// Frozen logical prefill suffix for one slot, persisted across ubatches.
// Set once after the prefix-cache final n_past decision; cleared on
// slot reset/release. known==false forbids guessing (dense with
// DENSE_UNKNOWN_BOUNDARY). Only true prompt rows are eligible; decode,
// MTP, frontier, embedding/rerank, and multimodal rows are dense.
struct fp_slot_boundary {
    bool known = false;
    int32_t begin = 0; // inclusive prompt-token index
    int32_t end = 0;   // exclusive prompt-token index
};

// Freeze [n_past_final, n_task_tokens) after all prefix-cache adjustments
// (common prefix, alora clamp, chunk-reuse shifts, checkpoint restore, and
// the TAG_PROMPT_LOGITS guarantee-evaluate-one decrement). Returns unknown
// when there is no suffix to compute (empty task or fully-cached edge the
// caller already forced to re-evaluate). Pure; no allocation.
inline fp_slot_boundary freeze_prefill_boundary(int32_t n_past_final, int32_t n_task_tokens) {
    fp_slot_boundary b;
    if (n_task_tokens <= 0 || n_past_final < 0 || n_past_final >= n_task_tokens) {
        return b;
    }
    b.known = true;
    b.begin = n_past_final;
    b.end = n_task_tokens;
    return b;
}

// Exact known teacher-forced RERoT injection interval. The full injection is
// tokenized up front (rerot_set_injection), so [0, injection_size) is known
// before the first frontier commits; cursor advances per successful frontier.
// Returns unknown for empty injections (nothing eligible).
inline fp_slot_boundary freeze_rerot_injection_interval(int32_t injection_size) {
    fp_slot_boundary b;
    if (injection_size <= 0) {
        return b;
    }
    b.known = true;
    b.begin = 0;
    b.end = injection_size;
    return b;
}

// Minimal inputs to classify one source row. Deliberately uses only
// bools + an explicit is_prompt_row phase flag: the caller passes the real
// slot/task state (inference_mode + slot.state), never n_tokens>1 guessing.
struct fp_row_inputs {
    bool is_embedding = false;   // task type EMBEDDING
    bool is_rerank = false;      // task type RERANK
    bool has_mtmd = false;       // multimodal prompt (visual chunk present)
    bool rerot_internal = false; // physical slot borrowed as a RERoT Lane
    bool rerot_forced = false;   // this row is a known teacher-forced injection token
    bool is_mtp_verify = false;  // target-verification batch (any speculative draft present)
    bool is_spec_replay = false; // checkpoint-restore replay batch
    bool is_prompt_row = false;  // true only for ordinary logical-prefill suffix rows
};

// Classify one row into a frozen execution role. Only PREFILL and
// REROT_TEACHER_FORCED are ever sparse-eligible; every other role is a
// conservative dense route. Precedence: embedding > rerank > multimodal >
// speculative-replay > mtp-verify > rerot-frontier/teacher-forced > prefill >
// decode. RERoT frontier generation rows are never prefill, even when a
// frontier carries several rows. Pure; no allocation.
inline int32_t classify_row_role(const fp_row_inputs & in) {
    if (in.is_embedding) {
        return LLAMA_FLASHPREFILL_ROLE_EMBEDDING;
    }
    if (in.is_rerank) {
        return LLAMA_FLASHPREFILL_ROLE_RERANK;
    }
    if (in.has_mtmd) {
        return LLAMA_FLASHPREFILL_ROLE_MULTIMODAL;
    }
    if (in.is_spec_replay) {
        return LLAMA_FLASHPREFILL_ROLE_SPECULATIVE_REPLAY;
    }
    if (in.is_mtp_verify) {
        return LLAMA_FLASHPREFILL_ROLE_MTP_VERIFY;
    }
    if (in.rerot_internal) {
        if (in.rerot_forced) {
            return LLAMA_FLASHPREFILL_ROLE_REROT_TEACHER_FORCED;
        }
        return LLAMA_FLASHPREFILL_ROLE_REROT_FRONTIER;
    }
    if (in.is_prompt_row) {
        return LLAMA_FLASHPREFILL_ROLE_PREFILL;
    }
    return LLAMA_FLASHPREFILL_ROLE_DECODE;
}

// True for the two sparse-eligible roles. All other roles take the
// pre-existing attention path (DENSE_ROLE on the server side; the context
// may refine short/tail/unsupported/capacity reasons later).
inline bool role_is_sparse_eligible(int32_t role) {
    return role == LLAMA_FLASHPREFILL_ROLE_PREFILL ||
           role == LLAMA_FLASHPREFILL_ROLE_REROT_TEACHER_FORCED;
}

// Build one frozen row descriptor. version/struct_size tagged, reserved
// zeroed, prefill interval attached only when the caller supplies a known
// boundary; otherwise prefill_known=false with sentinel interval values.
// Pure; no allocation; validates via llama_flashprefill_validate_row.
inline llama_flashprefill_row make_row(
        int32_t role, int32_t seq_id, uint32_t reader_id,
        int32_t logical_pos, const fp_slot_boundary & b) {
    llama_flashprefill_row row = {};
    row.version = LLAMA_FLASHPREFILL_ROW_VERSION;
    row.struct_size = (uint32_t) sizeof(llama_flashprefill_row);
    row.role = role;
    row.seq_id = seq_id;
    row.reader_id = reader_id;
    row.logical_pos = logical_pos;
    if (b.known) {
        row.prefill_begin = b.begin;
        row.prefill_end = b.end;
        row.prefill_known = true;
    } else {
        row.prefill_begin = LLAMA_FLASHPREFILL_SEQ_UNKNOWN;
        row.prefill_end = LLAMA_FLASHPREFILL_SEQ_UNKNOWN;
        row.prefill_known = false;
    }
    return row;
}

// Bounded dense-reason labels for metrics. Only DENSE_* routes are valid
// labels (7 values); sequence/reader ids and prompt text are never labels.
// Backed by the frozen route names, so label strings cannot drift.
inline int32_t fp_dense_reason_count() {
    return 7; // DENSE_OFF..DENSE_CAPACITY
}

inline int32_t fp_dense_reason_index(int32_t route) {
    if (route < LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF ||
        route > LLAMA_FLASHPREFILL_ROUTE_DENSE_CAPACITY) {
        return -1;
    }
    return route - LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF;
}

inline int32_t fp_dense_index_to_route(int idx) {
    if (idx < 0 || idx >= 7) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_ROLE;
    }
    return LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF + idx;
}

// Store-isolation decisions (pure; server-context.cpp enforces them against
// the live RAM prompt cache). Compared values are StatePolicy core state keys
// (llama_flashprefill_state_cache_key(): policy fingerprint + context serial
// + adapter generation); this helper is the pure u64 equality-with-unknown
// decision shape over those keys. Slot files carry no sidecar: their single
// authority is the core envelope validated inside the state bytes on load.
enum fp_store_decision {
    FP_STORE_USABLE = 0,
    FP_STORE_REJECT_MISMATCH = 1,
    FP_STORE_REJECT_UNKNOWN = 2,
};

// Usable iff both sides are stamped and equal. Either side unstamped (no
// policy yet, or a store entry that predates versioning) is UNKNOWN, never
// silently usable: the caller maps UNKNOWN to usable only for the narrow
// legacy case below. Pure; no allocation.
inline fp_store_decision fp_isolate_store(
        uint64_t stored_fp, bool stored_known,
        uint64_t current_fp, bool current_known) {
    if (!stored_known || !current_known) {
        return FP_STORE_REJECT_UNKNOWN;
    }
    return stored_fp == current_fp ? FP_STORE_USABLE : FP_STORE_REJECT_MISMATCH;
}

// Legacy slot files predate sidecars: usable only while no policy is enabled
// (exact old behavior preserved); once any policy is enabled they are
// policy-unknown and must be re-prefilled, never trusted. Pure.
inline bool fp_legacy_file_usable(bool policy_enabled) {
    return !policy_enabled;
}

// Successful-only counters. Incremented exclusively from the post_decode
// success path (one slice's rows counted exactly once after its decode
// returns 0). Retry (decode returns false), fatal errors (throw +
// abort_all_slots), and cancel/release paths never touch these, so
// canceled/failed/retried rows are never double counted. GPU-determined
// sparse/selected/corrected counts are deliberately absent here: they arrive
// via the pending aggregate hook below, never faked on the server.
struct fp_counters {
    uint64_t eligible_rows = 0; // PREFILL / TEACHER_FORCED with known boundary
    uint64_t dense_by_reason[7] = {}; // indexed by fp_dense_reason_index()

    void add_eligible(uint64_t n) {
        eligible_rows += n;
    }

    void add_dense(int32_t route, uint64_t n = 1) {
        const int32_t idx = fp_dense_reason_index(route);
        if (idx >= 0) {
            dense_by_reason[idx] += n;
        }
    }

    void clear() {
        eligible_rows = 0;
        for (int i = 0; i < 7; ++i) {
            dense_by_reason[i] = 0;
        }
    }

    bool empty() const {
        if (eligible_rows != 0) {
            return false;
        }
        for (int i = 0; i < 7; ++i) {
            if (dense_by_reason[i] != 0) {
                return false;
            }
        }
        return true;
    }
};

// GPU-determined snapshot (MetricsIntegration owner; ServerRouting's
// fp_counters / fp_eligible_rows / fp_dense_by_reason above are untouched).
// Producer: the context drain (llama_flashprefill_metrics_drain, declared in
// llama.h by ConfigIntegration, implemented in
// src/llama-flashprefill-metrics.cpp over the transactional per-ubatch
// ledger). Deltas cover successful slices only (retry/failure/cancel paths
// never merge context-side), and are merged here with the same
// once-per-slice discipline as fp_counters (post_decode success path only).
// The drain's eligible_rows field is intentionally NOT merged here:
// server-side fp_eligible_rows (fp_count_slice_success over batch.fp_rows)
// stays the single canonical eligible total; merging both would double
// count the same source rows.
//
// Units: sparse/dense_packed are PACKED (source_query, q_head) execution
// rows (plan-confirmed GPU work); dense_rows[10] are packed rows by granular
// reason (index order fixed: 0 decode, 1 mtp_verify,
// 2 role_other, 3 short_context, 4 dense_tail, 5 unknown_boundary,
// 6 unsupported, 7 high_cost, 8 no_plan, 9 full_attention_layer — the last
// from graph summary actuals); selected/corrected are plan use
// counts; visible/exact are 64-bit use-record token sums (subhead fan-out
// NOT multiplied); scratch live/peak are actual slice-sized bytes;
// layout_us_total sums host-measured layout slices only (GPU pool/select/
// attention breakdown intentionally omitted: per-layer timestamp waits are
// forbidden, see src/llama-flashprefill-metrics.h).
struct fp_gpu_deltas {
    uint64_t sparse_rows = 0;
    uint64_t dense_packed = 0;
    uint64_t dense_rows[10] = {};
    uint64_t selected_blocks = 0;
    uint64_t corrected_blocks = 0;
    uint64_t visible_tokens = 0;
    uint64_t exact_tokens = 0;
    uint64_t pool_rebuild[2] = {};       // 0 slice, 1 exact_all
    uint64_t plan_invalidations[3] = {}; // 0 bypassed, 1 empty_snapshot, 2 no_plan
    uint64_t scratch_live_bytes = 0;     // gauge: last successful slice
    uint64_t scratch_peak_bytes = 0;     // gauge: max over slices
    uint64_t layout_us_total = 0;        // counter: measured slices only
    uint64_t layout_slices_measured = 0;
    uint64_t gpu_pool_us_total = 0;      // counter: sampled calls only (VulkanDispatch hook)
    uint64_t gpu_select_us_total = 0;
    uint64_t gpu_attn_us_total = 0;
    uint64_t gpu_slices_measured = 0;

    bool empty() const {
        if (sparse_rows != 0 || dense_packed != 0 ||
            selected_blocks != 0 || corrected_blocks != 0 ||
            visible_tokens != 0 || exact_tokens != 0 ||
            scratch_live_bytes != 0 || scratch_peak_bytes != 0 ||
            layout_us_total != 0 || layout_slices_measured != 0 ||
            gpu_pool_us_total != 0 || gpu_select_us_total != 0 ||
            gpu_attn_us_total != 0 || gpu_slices_measured != 0) {
            return false;
        }
        for (int i = 0; i < 10; ++i) {
            if (dense_rows[i] != 0) {
                return false;
            }
        }
        for (int i = 0; i < 2; ++i) {
            if (pool_rebuild[i] != 0) {
                return false;
            }
        }
        for (int i = 0; i < 3; ++i) {
            if (plan_invalidations[i] != 0) {
                return false;
            }
        }
        return true;
    }
};

// Saturating once-per-successful-slice merge of one drain delta (defined in
// server-context.cpp). Gauges: scratch_live overwrites, scratch_peak maxes,
// layout time sums measured slices only. The snapshot type is the global
// ::llama_flashprefill_metrics_slice (llama.h); the unqualified elaborated
// specifier here resolves to it via ordinary lookup.
void fp_gpu_merge_slice(fp_gpu_deltas & acc, const struct llama_flashprefill_metrics_slice & d);

// Bounded Prometheus label values for the granular dense buckets (mirror of
// llama_flashprefill_metrics::dense_bucket_name; kept here so the server
// serializer needs no src-internal include).
inline const char * fp_gpu_dense_name(int idx) {
    switch (idx) {
        case 0:  return "decode";
        case 1:  return "mtp_verify";
        case 2:  return "role_other";
        case 3:  return "short_context";
        case 4:  return "dense_tail";
        case 5:  return "unknown_boundary";
        case 6:  return "unsupported";
        case 7:  return "high_cost";
        case 8:  return "no_plan";
        case 9:  return "full_attention_layer";
        default: return "?";
    }
}

inline const char * fp_gpu_pool_name(int idx) {
    return idx == 1 ? "exact_all" : (idx == 0 ? "slice" : "?");
}

inline const char * fp_gpu_plan_name(int idx) {
    switch (idx) {
        case 0:  return "bypassed";
        case 1:  return "empty_snapshot";
        case 2:  return "no_plan";
        default: return "?";
    }
}

} // namespace server_flashprefill_routing

struct server_context_impl; // private implementation

struct server_context_meta {
    std::string build_info;
    std::string model_name;
    std::set<std::string> model_aliases;
    std::set<std::string> model_tags;
    std::string model_path;
    bool has_mtmd;
    bool has_inp_image;
    bool has_inp_audio;
    bool has_inp_video;
    json json_ui_settings;
    int slot_n_ctx;
    enum llama_pooling_type pooling_type;

    // chat params
    server_chat_params & chat_params;
    std::map<std::string, bool> chat_template_caps;

    // tokens
    std::string bos_token_str;
    std::string eos_token_str;
    llama_token fim_pre_token;
    llama_token fim_sub_token;
    llama_token fim_mid_token;
    llama_token fim_pad_token;
    llama_token fim_rep_token;
    llama_token fim_sep_token;

    // sampling
    std::vector<llama_logit_bias> logit_bias_eog;

    // model meta
    enum llama_vocab_type model_vocab_type;
    int32_t model_vocab_n_tokens;
    int32_t model_n_ctx_train;
    int32_t model_n_embd_inp;
    uint64_t model_n_params;
    uint64_t model_size;
    std::string model_ftype;
};

enum server_state {
    SERVER_STATE_DOWNLOADING,
    SERVER_STATE_LOADING,
    SERVER_STATE_READY,
    SERVER_STATE_SLEEPING,
};

static std::string server_state_to_str(server_state state) {
    switch (state) {
        case SERVER_STATE_DOWNLOADING: return "downloading";
        case SERVER_STATE_LOADING:     return "loading";
        case SERVER_STATE_READY:       return "ready";
        case SERVER_STATE_SLEEPING:    return "sleeping";
        default: GGML_ASSERT(false && "invalid server_state");
    }
}

static server_state server_state_from_str(const std::string & str) {
    if (str == "downloading") return SERVER_STATE_DOWNLOADING;
    if (str == "loading")     return SERVER_STATE_LOADING;
    if (str == "ready")       return SERVER_STATE_READY;
    if (str == "sleeping")    return SERVER_STATE_SLEEPING;
    GGML_ASSERT(false && "invalid server_state string");
}

using server_state_callback_t = std::function<void(server_state, json /* payload */)>;

class server_token_hack {
public:
    struct config {
        int         max_injections = 1;
        std::string progress_text  = "Now, reasoning progress is at ";
    };

    server_token_hack(llama_context * ctx, config cfg);
    ~server_token_hack();

    void reset();
    llama_tokens after_block(const llama_tokens & confirmed_block);

private:
    struct impl;

    config cfg;
    std::unique_ptr<impl> pimpl;
    llama_tokens progress_tokens;
    int64_t n_confirmed = 0;
    int n_injected = 0;
};

using server_token_hack_factory_t = std::function<std::unique_ptr<server_token_hack>(llama_context * ctx)>;

struct server_context {
    std::unique_ptr<server_context_impl> impl;

    server_context();
    ~server_context();

    // load the model and initialize llama_context
    // returns true on success
    bool load_model(common_params & params);

    // this function will block main thread until termination
    void start_loop();

    // terminate main loop (will unblock start_loop)
    void terminate();

    // get the underlaying llama_context, can return nullptr if sleeping
    // not thread-safe, should only be used from the main thread
    llama_context * get_llama_context() const;

    // get a new response reader, used by CLI application
    server_response_reader get_response_reader();

    // get server metadata (read-only), can only be called after load_model()
    // not thread-safe, should only be used from the main thread
    server_context_meta get_meta() const;

    // note: must be set before load_model() is called
    void set_state_callback(server_state_callback_t callback);

    // Install one stateful append-only generation hook per slot.
    // Must be set before load_model() is called.
    void set_token_hack_factory(server_token_hack_factory_t factory);
};


// forward declarations
struct server_res_generator;

struct server_routes {
    server_routes(const common_params & params, server_context & ctx_server);

    void init_routes();

    // note: this is not thread-safe and can only when ctx_http.is_ready is false
    void update_meta(const server_context & ctx_server) {
        this->meta = std::make_unique<server_context_meta>(ctx_server.get_meta());
    }

    // handlers using lambda function, so that they can capture `this` without `std::bind`
    // they won't be called until ctx_http.is_ready is set to true
    server_http_context::handler_t get_health;
    server_http_context::handler_t get_metrics;
    server_http_context::handler_t get_slots;
    server_http_context::handler_t post_slots;
    server_http_context::handler_t get_props;
    server_http_context::handler_t post_props;
    server_http_context::handler_t post_infill;
    server_http_context::handler_t post_completions;
    server_http_context::handler_t post_completions_oai;
    server_http_context::handler_t post_chat_completions;
    server_http_context::handler_t post_chat_completions_tok;
    server_http_context::handler_t post_control;
    server_http_context::handler_t post_responses_oai;
    server_http_context::handler_t post_responses_tok_oai;
    server_http_context::handler_t post_transcriptions_oai;
    server_http_context::handler_t post_anthropic_messages;
    server_http_context::handler_t post_anthropic_count_tokens;
    server_http_context::handler_t post_apply_template;
    server_http_context::handler_t get_models;
    server_http_context::handler_t post_tokenize;
    server_http_context::handler_t post_detokenize;
    server_http_context::handler_t post_embeddings;
    server_http_context::handler_t post_embeddings_oai;
    server_http_context::handler_t post_rerank;
    server_http_context::handler_t get_lora_adapters;
    server_http_context::handler_t post_lora_adapters;

    // to be used in router mode
    json get_model_info() const;

private:
    std::unique_ptr<server_res_generator> handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type);
    std::unique_ptr<server_res_generator> handle_slots_save(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_restore(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_erase(const server_http_req &, int id_slot);
    std::unique_ptr<server_res_generator> handle_embeddings_impl(const server_http_req & req, task_response_type res_type);
    std::unique_ptr<server_res_generator> handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const server_http_req & req, task_response_type res_type);

    // using unique_ptr to allow late initialization of const
    std::unique_ptr<const server_context_meta> meta;

    const common_params & params;
    const server_context_impl & ctx_server;

    server_queue & queue_tasks;
    server_response & queue_results;
    std::unique_ptr<server_res_generator> create_response(bool bypass_sleep = false);
};
