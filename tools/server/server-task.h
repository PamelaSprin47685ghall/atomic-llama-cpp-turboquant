#pragma once

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <string>
#include <unordered_set>
#include <list>
#include <map>
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

    // realtime control (SERVER_TASK_TYPE_CONTROL)
    std::string        control_action;
    std::string        control_cmpl_id;

    // per-request parameters for chat parsing
    common_chat_parser_params chat_parser_params;

    // message spans for checkpointing
    common_chat_msg_spans message_spans;

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

    bool empty() const;
    json to_json() const;
    void accumulate(const server_rerot_metrics & delta);
};

// struct for tracking the state of a task (e.g., for streaming)
struct task_result_state {
    // tracking diffs for partial tool calls
    std::vector<common_chat_msg_diff> diffs;
    common_chat_parser_params chat_parser_params;
    common_chat_msg chat_msg;
    std::string generated_text; // append new chunks of generated text here
    std::vector<std::string> generated_tool_call_ids;
    std::unordered_set<size_t> sent_tool_call_names;

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
        int id_slot;
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
        oaicompat_msg = state.update_chat_msg(content, false, oaicompat_msg_diffs);

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
common_grammar server_rerot_take_user_grammar(task_params & params);
void server_rerot_restore_user_grammar(task_params & params, const common_grammar & saved);
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

// Streaming (A.14): internal lane text never enters content deltas; the only
// optional lane visibility is rerot.trace.* events under the explicit flag.
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
