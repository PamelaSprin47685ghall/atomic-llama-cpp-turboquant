#include "server-context.h"
#include "../../wanxiangqi/server/integration.h"
#include "repetition-guard.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-rerot.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-schema.h"
#include "server-stream.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include "../../src/llama-ext.h" // staging API: llama_get_ctx_other (used by mmproj draft mirroring)

#include <algorithm>
#include <cstddef>
#include <cinttypes>
#include <exception>
#include <memory>
#include <filesystem>
#include <utility>
#include <unordered_map>
#include <fstream>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

constexpr int HTTP_POLLING_SECONDS = 1;

struct server_token_hack::impl {
    llama_context * ctx;
    wanxiangqi_repetition_guard repetition_guard;

    explicit impl(llama_context * ctx)
        : ctx(ctx) {
    }
};

server_token_hack::server_token_hack(llama_context * ctx, config cfg)
    : cfg(std::move(cfg)), pimpl(std::make_unique<impl>(ctx)) {
    progress_tokens = common_tokenize(ctx, this->cfg.progress_text, false, false);
    reset();
}

server_token_hack::~server_token_hack() = default;

void server_token_hack::reset() {
    n_confirmed = 0;
    n_injected = 0;
    pimpl->repetition_guard.reset();
}

llama_tokens server_token_hack::after_block(const llama_tokens & confirmed_block) {
    n_confirmed += (int64_t) confirmed_block.size();

    if (confirmed_block.empty() || progress_tokens.empty() || n_injected >= cfg.max_injections) {
        return {};
    }

    std::string confirmed_text;
    for (llama_token token : confirmed_block) {
        confirmed_text += common_token_to_piece(pimpl->ctx, token, true);
    }

    const auto anomaly = pimpl->repetition_guard.observe(confirmed_text);
    if (anomaly == wanxiangqi_repetition_state::normal) {
        return {};
    }

    ++n_injected;
    pimpl->repetition_guard.reset();

    SRV_INF("token hack anomaly injection: anomaly=%s block=%zu confirmed=%" PRId64 " injection=%d/%d text='%s'\n",
            wanxiangqi_repetition_state_name(anomaly), confirmed_block.size(), n_confirmed,
            n_injected, cfg.max_injections, cfg.progress_text.c_str());

    return progress_tokens;
}

static uint32_t server_n_outputs_max(const common_params & params) {
    const uint32_t n_batch = params.n_batch;
    const uint32_t n_seq_max = std::max(1, params.rerot_enabled ? LLAMA_MAX_SEQ : params.n_parallel);

    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return std::max(n_batch, n_seq_max);
    }

    const uint32_t n_outputs_per_seq = 1 + common_speculative_n_max(&params.speculative);
    const uint64_t n_outputs = (uint64_t) n_seq_max * n_outputs_per_seq;

    // llama_context::output_reserve() always keeps at least one output row per sequence.
    return std::max<uint32_t>(n_seq_max, std::min<uint64_t>(n_batch, n_outputs));
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

enum server_inference_mode {
    SERVER_INFERENCE_MODE_DECODE,
    SERVER_INFERENCE_MODE_PREFILL,
};

enum class server_rerot_injection_kind : uint8_t {
    none = 0,
    planner,
    planner_open,
    heading,
    refresh,
    worker,
    worker_close,
    finalizer,
};

static constexpr size_t SERVER_REROT_PRIVATE_BATCH = 32;

static bool server_rerot_private_microbatch(server_rerot_injection_kind injection) {
    return injection == server_rerot_injection_kind::planner ||
           injection == server_rerot_injection_kind::refresh ||
           injection == server_rerot_injection_kind::worker ||
           injection == server_rerot_injection_kind::finalizer;
}

struct server_slot; // forward declaration

struct server_batch {
    llama_batch batch;
    bool batch_rendered = false;

    struct token {
        int32_t id_slot;
        llama_token token;
        llama_pos pos;
        bool output;
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    // in embd mode, we temporarily swap out the tokens arr and restore it on clear()
    bool has_embd = false;
    llama_token * tokens_ptr = nullptr;
    std::vector<float> embd;

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    server_batch() {
        batch.pos = nullptr; // sentinel: uninitialized batch
    }

    ~server_batch() {
        if (batch.pos != nullptr) {
            clear();
            llama_batch_free(batch);
        }
    }

    void init(int32_t n_tokens_alloc, int32_t n_embd) {
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        batch = llama_batch_init(n_tokens_alloc, 0, 1);
        tokens_ptr = batch.token;
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output) {
        GGML_ASSERT(!has_embd); // cannot mix tokens + embd in same batch
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, pos, output });
        return true;
    }

    bool add(int32_t id_slot, const std::vector<float> & embd_in, llama_pos pos, bool output) {
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, LLAMA_TOKEN_NULL, pos, output });
        has_embd = true;
        embd.insert(embd.end(), embd_in.begin(), embd_in.end());
        return true;
    }

    void clear() {
        tokens.clear();
        embd.clear();
        common_batch_clear(batch);
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
        batch_rendered    = false;
        has_embd          = false;
        if (batch.token == nullptr) {
            batch.token = tokens_ptr;
            batch.embd  = nullptr;
        }
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    void render() {
        GGML_ASSERT(!batch_rendered);
        GGML_ASSERT(batch.pos != nullptr);
        common_batch_clear(batch);
        for (int32_t i = 0; i < size(); i++) {
            const auto & t = tokens[i];
            common_batch_add(batch, t.token, t.pos, { t.id_slot }, t.output);
        }
        if (has_embd) {
            batch.token = nullptr; // will be restored on clear()
            batch.embd  = embd.data();
        }
        batch_rendered = true;
    }

    llama_batch get_view(int32_t off, int32_t n_tokens) const {
        GGML_ASSERT(batch.pos != nullptr);
        GGML_ASSERT(batch_rendered);
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        auto * token = batch.token ? batch.token + off          : nullptr;
        auto * embd  = batch.embd  ? batch.embd  + off * n_embd : nullptr;

        llama_batch view = {
            n_tokens,
            token,
            embd,
            batch.pos      + off,
            batch.n_seq_id + off,
            batch.seq_id   + off,
            batch.logits   + off,
        };

        return view;
    }
};

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    common_memory mem;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    // speculative decoding
    common_speculative * spec;

    llama_tokens spec_draft;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;
    bool spec_is_replay = false;

    // RERoT thin glue (§18.2): a physical slot is never a logical Lane id.
    // These fields only tag which episode/node (if any) currently borrows
    // this slot as its execution handle; the tree/queue/frontier truth lives
    // in server_rerot_runtime. OFF: all remain at their sentinels.
    bool rerot_internal = false;
    uint64_t rerot_episode_id = 0;
    uint32_t rerot_node_id = UINT32_MAX;
    llama_seq_id rerot_exec_seq = -1;
    // MTP draft binding stamp (§A.6) captured when this slot started drafting
    // under a RERoT view; checked for staleness before accept.
    struct llama_rerot_view_stamp rerot_draft_stamp = {0, 0, 0};
    bool rerot_has_draft_stamp = false;

    // Each sampled/public Lane contributes one token per frontier. Contiguous
    // PRIVATE control spans may use multiple causal rows in the same decode;
    // they are invisible to peers and commit only after the last row succeeds.
    server_rerot_injection_kind rerot_injection = server_rerot_injection_kind::none;
    llama_tokens rerot_injection_tokens;
    size_t rerot_injection_cursor = 0;
    std::optional<server_rerot_token_plan> rerot_inflight_plan;
    std::string rerot_inflight_bytes;
    std::vector<server_rerot_token_plan> rerot_inflight_extra_plans;
    std::vector<std::string> rerot_inflight_extra_bytes;
    bool rerot_inflight_forced = false;
    bool rerot_serial_tail = false;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_keep      = 0;
    int32_t n_decoded   = 0;
    int32_t n_decoded_start = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;

    int32_t n_prompt_tokens_original          = 0;
    int32_t n_prompt_tokens_cache             = 0;
    int32_t n_prompt_tokens_cache_response    = 0;
    int32_t n_prompt_tokens_processed         = 0;
    int32_t n_prompt_tokens_processed_accum   = 0;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;
    bool triattention_compressed = false;

    server_prompt prompt;
    server_tokens response_prompt;

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
        std::vector<uint8_t> state_spec;
        common_speculative_get_state(spec, id, state_spec);

        const size_t cur_size = cur_size_tgt + cur_size_dft + state_spec.size();

        SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft, state_spec.size());
        if (cur == nullptr) {
            return false;
        }

        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }
        if (!state_spec.empty()) {
            std::memcpy(cur->data.spec.data(), state_spec.data(), state_spec.size());
        }

        return true;
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        std::vector<uint8_t> state_spec;
        bool loaded_state = false;
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id, &state_spec, &loaded_state);
        if (res && loaded_state) {
            common_speculative_set_state(spec, id, state_spec);
        }
        if (res) {
            triattention_compressed = loaded_state &&
                llama_memory_seq_get_kv_used(llama_get_memory(ctx_tgt), id) < prompt.tokens.size();
        }
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear() {
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        mem.seq_rm(id, -1, -1);
        common_speculative_set_state(spec, id, {});

        prompt.clear();
        triattention_compressed = false;
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    std::unique_ptr<server_token_hack> token_hack;
    llama_token token_append_anchor = LLAMA_TOKEN_NULL;
    llama_tokens token_append_chunk;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    std::vector<float> inp_embd;

    // stats
    size_t n_sent_text = 0; // number of sent text character

    // TODO @ngxson : move all metrics to a sub-struct for clarity
    int64_t t_start_process_prompt;
    int64_t t_start_generation;
    int64_t t_print_last = 0;
    int32_t n_decoded_last = 0;

    double t_prompt_processing = 0.0; // ms
    double t_prompt_processing_accum = 0.0;
    double t_token_generation = 0.0;  // ms

    std::function<void(int /* id_slot */)> callback_on_release;

    // Speculative decoding stats
    int32_t n_draft_total = 0;      // Total draft tokens generated
    int32_t n_draft_accepted = 0;   // Draft tokens actually accepted
    int32_t n_draft_verif_steps = 0; // Total draft token verification steps by the target model
    std::vector<int32_t> n_accepted_per_pos; // Accepted tokens per draft position

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        spec_is_replay = false;

        n_decoded_start                  = 0;
        n_prompt_tokens_original         = 0;
        n_prompt_tokens_cache            = 0;
        n_prompt_tokens_cache_response   = 0;
        n_prompt_tokens_processed_accum  = 0;
        response_prompt.clear();

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;
        t_prompt_processing_accum = 0.0;

        if (spec) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        token_append_anchor = LLAMA_TOKEN_NULL;
        token_append_chunk.clear();
        if (token_hack) {
            token_hack->reset();
        }
        json_schema = json();

        // clear speculative decoding stats
        n_draft_total = 0;
        n_draft_accepted = 0;
        n_draft_verif_steps = 0;
        n_accepted_per_pos.clear();

        task_prev = std::move(task);
        task.reset();

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();

        // RERoT: physical slot released, logical Lane mapping (if any) is
        // owned by the runtime (detach/erase); just drop the local tags.
        // OFF: sentinels already, no behavior change.
        rerot_internal = false;
        rerot_episode_id = 0;
        rerot_node_id = UINT32_MAX;
        rerot_exec_seq = -1;
        rerot_draft_stamp = {0, 0, 0};
        rerot_has_draft_stamp = false;
        rerot_injection = server_rerot_injection_kind::none;
        rerot_injection_tokens.clear();
        rerot_injection_cursor = 0;
        rerot_inflight_plan.reset();
        rerot_inflight_bytes.clear();
        rerot_inflight_extra_plans.clear();
        rerot_inflight_extra_bytes.clear();
        rerot_inflight_forced = false;
        rerot_serial_tail = false;
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd() || (spec && common_speculative_need_embd(spec));
    }

    bool need_embd_nextn() const {
        GGML_ASSERT(task);
        return spec && common_speculative_need_embd_nextn(spec);
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    // (MTP supports splitting — uses task->need_embd() not need_embd())
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx_tgt) && llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type
            && rerot_internal == other_slot.rerot_internal
            && inp_embd.size() == other_slot.inp_embd.size()
            && are_lora_equal(lora, other_slot.lora);
    }

    bool has_budget(const common_params & global_params) {
        GGML_ASSERT(task);

        if (task->params.n_predict == -1 && global_params.n_predict == -1) {
            return true; // limitless
        }

        n_remaining = -1;

        if (task->params.n_predict != -1) {
            n_remaining = task->params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec && token_append_chunk.empty();
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        bool handled_append_chunk = false;
        if (!token_append_chunk.empty()) {
            GGML_ASSERT(spec_draft.empty());
            GGML_ASSERT(token_append_anchor != LLAMA_TOKEN_NULL);
            GGML_ASSERT((int32_t) token_append_chunk.size() + 1 <= batch.n_tokens_alloc - batch.size());
            handled_append_chunk = true;

            auto pos0 = prompt.tokens.pos_next();
            const int32_t batch0 = batch.size();

            add_ok &= batch.add(id, token_append_anchor, pos0++, false);
            for (size_t i = 0; i < token_append_chunk.size(); ++i) {
                add_ok &= batch.add(id, token_append_chunk[i], pos0++, i + 1 == token_append_chunk.size());
            }

            i_batch = batch0 + (int32_t) token_append_chunk.size();

            prompt.tokens.push_back(token_append_anchor);
            prompt.tokens.insert(token_append_chunk);
            sampled = token_append_chunk.back();
            token_append_anchor = LLAMA_TOKEN_NULL;
            token_append_chunk.clear();
        } else if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.size();

            if (!inp_embd.empty()) {
                add_ok &= batch.add(id, inp_embd, prompt.tokens.pos_next(), true);
            } else {
                add_ok &= batch.add(id, sampled, prompt.tokens.pos_next(), true);
            }

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(id, sampled, pos0++, true);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->id, token, pos0++, true);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        if (!handled_append_chunk) {
            prompt.tokens.push_back(sampled);
            prompt.tokens.insert(spec_draft);
        }
    }

    void release(bool schedule_deferred = true) {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used        =  ggml_time_us();
            t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear();
            }

            reset();

            if (schedule_deferred) {
                callback_on_release(id);
            }
        }
    }

    int32_t n_prompt_tokens_processed_total() const {
        return n_prompt_tokens_processed_accum + n_prompt_tokens_processed;
    }

    result_timings get_timings() const {
        result_timings timings;
        timings.cache_n = n_prompt_tokens_cache_response;

        const int32_t n_prompt_total = n_prompt_tokens_processed_total();
        timings.prompt_n  = n_prompt_total;
        timings.prompt_ms = t_prompt_processing;
        if (n_prompt_total > 0 && t_prompt_processing > 0.0) {
            timings.prompt_per_token_ms = t_prompt_processing / n_prompt_total;
            timings.prompt_per_second   = 1e3 / t_prompt_processing * n_prompt_total;
        }

        timings.predicted_n            = n_decoded;
        timings.predicted_ms           = t_token_generation;
        timings.predicted_per_token_ms = t_token_generation / n_decoded;
        timings.predicted_per_second   = 1e3 / t_token_generation * n_decoded;

        // Add speculative metrics
        if (n_draft_total > 0) {
            timings.draft_n          = n_draft_total;
            timings.draft_n_accepted = n_draft_accepted;
        }

        return timings;
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (n_decoded < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = 1e3 / (t_token_generation)   * (n_decoded);
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (n_decoded - n_decoded_last);

        t_print_last = t_now;
        n_decoded_last = n_decoded;

        SLT_INF(*this, "n_decoded = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", n_decoded, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;
        const double f_progress = (float) prompt.n_tokens() / task->n_tokens();

        if (t_prompt_processing < 3000.0) {
            return;
        }

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                n_prompt_tokens_processed, f_progress, t_prompt_processing / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const int32_t n_prompt_total = n_prompt_tokens_processed_total();
        const double t_prompt        = n_prompt_total > 0 ? t_prompt_processing / n_prompt_total : 0.0;
        const double n_prompt_second = t_prompt_processing > 0.0 ? 1e3 / t_prompt_processing * n_prompt_total : 0.0;

        const double t_gen        =       t_token_generation / n_decoded;
        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_processing, n_prompt_total, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_token_generation, n_decoded, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_processing + t_token_generation, n_prompt_total + n_decoded);

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
            SLT_TRC(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = n_prompt_tokens_processed;
            res["n_prompt_tokens_cache"]     = n_prompt_tokens_cache;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = {
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining},
                    {"n_decoded",      n_decoded},
                }
            };

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.n_decoded   = n_decoded;
        other.n_remaining = n_remaining;
        other.i_batch     = i_batch;

        other.t_start_process_prompt          = t_start_process_prompt;
        other.t_prompt_processing             = t_prompt_processing;
        other.t_prompt_processing_accum       = t_prompt_processing_accum;
        other.n_prompt_tokens_cache           = n_prompt_tokens_cache;
        other.n_prompt_tokens_cache_response  = n_prompt_tokens_cache_response;
        other.n_prompt_tokens_processed       = n_prompt_tokens_processed;
        other.n_prompt_tokens_processed_accum = n_prompt_tokens_processed_accum;
        other.truncated                       = truncated;

        std::vector<uint8_t> state_spec;
        common_speculative_get_state(spec, id, state_spec);
        common_speculative_set_state(spec, other.id, state_spec);

        other.prompt = prompt.clone();
        other.init_sampler();
    }

    // returns 0 on success
    // caller need to update prompt.tokens after a successful call to keep track of the processing progress
    int process_mtmd_chunk(size_t idx, size_t & n_tokens_out) {
        GGML_ASSERT(mctx);
        const auto & input_tokens = task->tokens;
        const auto & chunk = input_tokens.find_chunk(idx);
        int32_t res = 0;

        auto try_decode = [&]() -> int32_t {
            if (mbatch) {
                float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
                if (embd) {
                    void * cb_data = spec;
                    static auto cb = [](llama_batch batch, void * user_data) {
                        common_speculative * spec = static_cast<common_speculative *>(user_data);
                        if (!common_speculative_process(spec, batch)) {
                            return 1;
                        }
                        return 0;
                    };

                    llama_pos new_n_past; // unused for now
                    res = mtmd_helper_decode_image_chunk(
                        mctx,
                        ctx_tgt,
                        chunk.get(),
                        embd,
                        prompt.tokens.pos_next(),
                        id,
                        llama_n_batch(ctx_tgt),
                        &new_n_past,
                        cb,
                        cb_data
                    );
                    if (res != 0) {
                        SLT_ERR(*this, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                        return -1;
                    }
                    n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                    return 0; // success
                }
            }
            return 1; // (non-error) need to create & encode batch
        };

        // if the batch is already exist, try searching & encode
        res = try_decode();
        if (res == 0) {
            return 0;
        }
        if (res < 0) {
            // fatal error
            return res;
        }

        // otherwise, the batch is either uninitialized or is used up
        // we need to create & encode a new batch
        mbatch.reset(mtmd_batch_init(mctx));
        res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
        GGML_ASSERT(res == 0); // we should never have an empty batch

        // try batching as much as possible
        int n_added = 1;
        size_t idx_cur = idx;
        while (res == 0) {
            auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
            if (next_chunk == nullptr) {
                break;
            }
            res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
            n_added += (res == 0 ? 1 : 0);
            idx_cur = next_idx;
            SLT_DBG(*this, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
            // if res != 0, batch is full or chunk is not compatible -> this loop breaks
        }

        // TODO @ngxson : move this log line to debug when it become more stable
        SLT_TRC(*this, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

        res = mtmd_batch_encode(mbatch.get());
        if (res != 0) {
            SLT_ERR(*this, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
            return -1;
        }

        return try_decode();
    }
};



//
// server_metrics
//

struct server_metrics {
    int64_t t_start = 0;

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

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const server_slot & slot) {
        const double t_prompt_attempt = std::max(0.0, slot.t_prompt_processing - slot.t_prompt_processing_accum);
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed       += slot.n_prompt_tokens_processed;
        t_prompt_processing             += t_prompt_attempt;
        t_prompt_processing_total       += t_prompt_attempt;

        n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
    }

    void on_prediction(const server_slot & slot) {
        n_tokens_predicted_total   += slot.n_decoded;
        n_tokens_predicted         += slot.n_decoded;
        t_tokens_generation        += slot.t_token_generation;
        t_tokens_generation_total  += slot.t_token_generation;
    }

    void on_decoded(const std::vector<server_slot> & slots) {
        n_decode_total++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
            n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};


//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    void dump_request_tokens(
            const std::vector<llama_token> & tokens,
            server_token_dump_kind kind) const {
        request_token_dump.append(tokens, kind);
    }

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};
    server_token_hack_factory_t token_hack_factory;

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    server_batch batch;

    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;

    common_speculative_init_result_ptr spec_init;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    struct server_retry_state {
        int32_t n_decoded = 0;
        int32_t n_prompt_tokens_original = 0;
        int32_t n_prompt_tokens_cache_response = 0;
        int32_t n_prompt_tokens_processed = 0;
        server_tokens response_prompt;
        size_t last_nl_pos = 0;
        std::string generated_text;
        llama_tokens generated_tokens;
        std::vector<completion_token_output> generated_token_probs;
        size_t n_sent_text = 0;
        bool has_new_line = false;
        bool truncated = false;
        int64_t t_start_generation = 0;
        double t_prompt_processing = 0.0;
    };

    // slots / clients
    std::vector<server_slot> slots;
    std::unordered_map<std::string, int> cache_key_slots;
    std::unordered_map<int, server_retry_state> retry_states;

    int trace = 0;
    int slots_debug = 0;
    int n_empty_consecutive = 0;

    server_inference_mode inference_mode = SERVER_INFERENCE_MODE_DECODE;
    int32_t id_slot_prefill = -1;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    // RERoT thin glue (Stage 7 + compat core, §§18-21,A.11): the logical
    // episode/tree/queue/frontier truth lives in server_rerot_runtime (already
    // implemented in server-rerot.*). This context only holds the runtime and
    // calls it from four gated hook points below. OFF: rerot stays null and
    // every hook returns immediately, so scheduling stays byte-identical.
    std::unique_ptr<server_rerot_runtime> rerot;
    uint64_t rerot_episode_id = 0;
    uint64_t rerot_task_episode_id_next = 1;
    uint64_t rerot_task_generation_next = 1;

    struct rerot_transport_state {
        server_task response_task;
        std::vector<common_adapter_lora_info> lora;
        std::unordered_map<llama_rerot_node_id, server_tokens> lineage_tokens;
        std::unordered_map<llama_rerot_node_id, std::vector<uint8_t>> parent_recurrent_snapshots;
        std::unordered_map<llama_rerot_run_id, std::string> run_bytes;
        server_rerot_line_mux line_mux;
        std::string streamed_reasoning;
        uint64_t streamed_lines = 0;
        uint64_t sampled_tokens = 0;
        uint32_t root_seed = LLAMA_DEFAULT_SEED;
        int32_t prompt_tokens_processed = 0;
        int32_t prompt_tokens_processed_accum = 0;
        int32_t prompt_tokens_cache_response = 0;
        double prompt_processing_ms = 0.0;
        bool segmented_active = false;
        bool triattention_compressed = false;
        uint64_t triattention_synced_public_tokens = 0;
        uint64_t model_tokens = 0;
        uint64_t parallel_model_tokens = 0;
        int64_t started_us = 0;
        int64_t parallel_started_us = 0;
        uint64_t parallel_elapsed_us = 0;
        bool parallel_finished = false;

        explicit rerot_transport_state(server_task && task)
            : response_task(std::move(task)) {
        }
    };

    std::unordered_map<uint64_t, std::unique_ptr<rerot_transport_state>> rerot_transport;

    bool rerot_active() const {
        return rerot != nullptr && rerot_episode_id != 0 && rerot->episode(rerot_episode_id) != nullptr;
    }

    const server_rerot_episode * rerot_active_episode() const {
        return rerot_active() ? rerot->episode(rerot_episode_id) : nullptr;
    }

    size_t rerot_private_batch_size(const server_slot & slot) const {
        if (!ctx_tgt ||
            !server_rerot_private_microbatch(slot.rerot_injection) ||
            slot.rerot_injection_cursor >= slot.rerot_injection_tokens.size()) {
            return 1;
        }
        const size_t fair_batch_capacity = std::max<size_t>(
            1, llama_n_batch(ctx_tgt) / std::max<size_t>(1, slots.size()));
        return std::min({
            slot.rerot_injection_tokens.size() - slot.rerot_injection_cursor,
            SERVER_REROT_PRIVATE_BATCH,
            fair_batch_capacity,
        });
    }

    void rerot_emit_reasoning_lines(
            uint64_t episode_id,
            rerot_transport_state & transport,
            server_rerot_stream_lines && ready) {
        if (!ready.ok) {
            if (rerot && episode_id != 0) {
                rerot->hard_abort(
                    episode_id,
                    "rerot_stream_error: " + ready.error);
            }
            return;
        }
        if (!transport.response_task.params.stream) {
            return;
        }

        for (auto & line : ready.lines) {
            auto res = std::make_unique<server_task_result_cmpl_partial>();
            res->id = transport.response_task.id;
            res->index = transport.response_task.index;
            res->content = line;
            res->n_decoded = static_cast<int32_t>(std::min<uint64_t>(
                transport.sampled_tokens,
                std::numeric_limits<int32_t>::max()));
            res->n_prompt_tokens = transport.response_task.n_tokens();
            res->n_prompt_tokens_cache = transport.prompt_tokens_cache_response;
            res->post_sampling_probs = transport.response_task.params.post_sampling_probs;
            res->verbose = transport.response_task.params.verbose;
            res->res_type = transport.response_task.params.res_type;
            res->oaicompat_model = transport.response_task.params.oaicompat_model;
            res->oaicompat_cmpl_id = transport.response_task.params.oaicompat_cmpl_id;
            queue_results.send(std::move(res));

            transport.streamed_reasoning += line;
            ++transport.streamed_lines;
        }
    }

    bool rerot_stream_commit(
            uint64_t episode_id,
            llama_rerot_node_id node_id,
            llama_rerot_run_id run_id,
            std::string_view bytes) {
        auto transport_it = rerot_transport.find(episode_id);
        auto * episode = rerot ? rerot->episode(episode_id) : nullptr;
        if (transport_it == rerot_transport.end() || !episode) {
            return false;
        }
        auto & transport = *transport_it->second;
        if (!transport.response_task.params.stream) {
            return true;
        }

        auto ready = transport.line_mux.append(
            node_id, run_id, bytes, episode->document);
        const bool ok = ready.ok;
        rerot_emit_reasoning_lines(episode_id, transport, std::move(ready));
        return ok;
    }

    bool rerot_stream_visibility_changed(
            uint64_t episode_id,
            llama_rerot_node_id node_id) {
        auto transport_it = rerot_transport.find(episode_id);
        auto * episode = rerot ? rerot->episode(episode_id) : nullptr;
        if (transport_it == rerot_transport.end() || !episode) {
            return false;
        }
        auto & transport = *transport_it->second;
        if (!transport.response_task.params.stream) {
            return true;
        }

        auto ready = transport.line_mux.drain(node_id, episode->document);
        const bool ok = ready.ok;
        rerot_emit_reasoning_lines(episode_id, transport, std::move(ready));
        return ok;
    }

    bool rerot_stream_finish_lane(
            uint64_t episode_id,
            llama_rerot_node_id node_id) {
        auto transport_it = rerot_transport.find(episode_id);
        auto * episode = rerot ? rerot->episode(episode_id) : nullptr;
        if (transport_it == rerot_transport.end() || !episode) {
            return false;
        }
        auto & transport = *transport_it->second;
        if (!transport.response_task.params.stream) {
            return true;
        }

        auto ready = transport.line_mux.finish(node_id, episode->document);
        const bool ok = ready.ok;
        rerot_emit_reasoning_lines(episode_id, transport, std::move(ready));
        return ok;
    }

    void rerot_note_parallel_start(uint64_t episode_id) {
        auto transport_it = rerot_transport.find(episode_id);
        if (transport_it == rerot_transport.end() ||
            transport_it->second->parallel_started_us != 0 ||
            transport_it->second->parallel_finished) {
            return;
        }
        transport_it->second->parallel_started_us = ggml_time_us();
    }

    void rerot_note_parallel_end(uint64_t episode_id, int64_t now_us) {
        auto transport_it = rerot_transport.find(episode_id);
        if (transport_it == rerot_transport.end() ||
            transport_it->second->parallel_started_us == 0 ||
            transport_it->second->parallel_finished) {
            return;
        }
        transport_it->second->parallel_elapsed_us =
            static_cast<uint64_t>(std::max<int64_t>(
                0, now_us - transport_it->second->parallel_started_us));
        transport_it->second->parallel_finished = true;
    }

    void rerot_record_completed_episode(uint64_t episode_id) {
        auto transport_it = rerot_transport.find(episode_id);
        if (transport_it == rerot_transport.end()) {
            return;
        }

        const int64_t now_us = ggml_time_us();
        rerot_note_parallel_end(episode_id, now_us);
        const auto & transport = *transport_it->second;
        const double episode_seconds = static_cast<double>(
            std::max<int64_t>(0, now_us - transport.started_us)) / 1.0e6;
        const double parallel_seconds =
            static_cast<double>(transport.parallel_elapsed_us) / 1.0e6;

        ++rerot_metrics.completed_episodes;
        rerot_metrics.completed_model_tokens += transport.model_tokens;
        rerot_metrics.parallel_model_tokens += transport.parallel_model_tokens;
        rerot_metrics.completed_episode_seconds += episode_seconds;
        rerot_metrics.parallel_seconds += parallel_seconds;

        const double aggregate_tps = episode_seconds > 0.0
            ? static_cast<double>(transport.model_tokens) / episode_seconds
            : 0.0;
        const double parallel_tps = parallel_seconds > 0.0
            ? static_cast<double>(transport.parallel_model_tokens) / parallel_seconds
            : 0.0;
        SRV_INF("RERoT throughput: episode=%" PRIu64
                " model_tokens=%" PRIu64 " wall_s=%.6f aggregate_tps=%.3f"
                " parallel_tokens=%" PRIu64 " parallel_s=%.6f parallel_tps=%.3f\n",
            episode_id,
            transport.model_tokens,
            episode_seconds,
            aggregate_tps,
            transport.parallel_model_tokens,
            parallel_seconds,
            parallel_tps);
    }

    server_task rerot_clone_task(const server_task & source) const {
        server_task copy(source.type);
        copy.id = source.id;
        copy.index = source.index;
        copy.id_slot = -1;
        copy.id_parent = source.id_parent;
        copy.params = source.params;
        copy.tokens = source.tokens.clone();
        copy.is_retry = false;
        copy.cache_key.clear();
        copy.rerot_episode_id = source.rerot_episode_id;
        copy.rerot_generation = source.rerot_generation;
        copy.rerot_original_user_text = source.rerot_original_user_text;
        return copy;
    }

    static common_grammar rerot_planner_remainder_grammar() {
        // The structural opener "<ol>\n<li>" is injected before model
        // sampling. Constrain exactly the remainder: one non-empty text item,
        // then zero or more flat sibling items, then the single closing tag.
        // Excluding '<' from item text prevents </think> or nested markup from
        // entering an otherwise still-PENDING planner record.
        return {
            COMMON_GRAMMAR_TYPE_USER,
            "root ::= item-text \"</li>\" ws (\"<li>\" item-text \"</li>\" ws)* \"</ol>\"\n"
            "item-text ::= [^<]+\n"
            "ws ::= [ \\t\\r\\n]*\n",
        };
    }

    static void rerot_install_planner_grammar(task_params & params) {
        (void) server_rerot_take_user_grammar(params);
        params.sampling.grammar = rerot_planner_remainder_grammar();
        params.sampling.grammar_lazy = false;
        params.sampling.grammar_triggers.clear();
        params.sampling.preserved_tokens.clear();
    }

    static void rerot_remove_planner_grammar(task_params & params) {
        params.sampling.grammar = common_grammar{};
        params.sampling.grammar_lazy = false;
        params.sampling.grammar_triggers.clear();
        params.sampling.preserved_tokens.clear();
    }

    std::string rerot_planner_control_prompt(std::string_view assigned_task = {}) const {
        std::string prompt;
        if (assigned_task.empty()) {
            prompt = std::string(server_rerot_planner_prompt());
        } else {
            prompt = "我当前在推进【";
            prompt.append(assigned_task);
            prompt += "】。我直接用单个 li 确认该目标并用 /ol 闭合，随后展开具体推导；只有这项任务确实包含多个需要并行的独立子环节时，才拆成多个 li。\n";
        }
        return prompt;
    }

    std::string rerot_worker_control_prompt(std::string_view assigned_task) const {
        std::string prompt = "我当前负责推进子任务【";
        prompt.append(assigned_task);
        prompt +=
            "】。我看到上面父节点已经完成了结构规划并给出了 ol 列表。"
            "当前已经是具体子任务，我直接在此深入推导并得出结论；"
            "除非确实很有必要开启并行的子子任务，我才仿照父节点使用 ol 格式拆分子子任务。否则我直接写出完整推理过程，并在完成本任务后输出 &lt;/think&gt; 结束。\n";
        return prompt;
    }

    static std::string rerot_finalizer_control_prompt(std::string_view original_user_text) {
        GGML_UNUSED(original_user_text);
        // Pure <think></think> layer boundary: closing thinking transitions
        // directly to the final assistant response body.
        return "</think>\n\n";
    }

    bool rerot_set_injection(
            server_slot & slot,
            server_rerot_injection_kind kind,
            std::string_view text) {
        slot.rerot_injection_tokens = common_tokenize(ctx_tgt, std::string(text), false, true);
        slot.rerot_injection_cursor = 0;
        slot.rerot_injection = kind;
        if (slot.rerot_injection_tokens.empty()) {
            if (rerot && slot.rerot_episode_id != 0) {
                rerot->hard_abort(slot.rerot_episode_id, "rerot_protocol_error: empty forced control sequence");
            }
            return false;
        }
        return true;
    }

    bool rerot_start_root(server_slot & slot) {
        if (!rerot || !ctx_tgt || !slot.task ||
            !slot.task->params.rerot_effective(slot.task->type) ||
            rerot_active()) {
            return false;
        }

        server_task root_lane_task = rerot_clone_task(*slot.task);
        rerot_install_planner_grammar(root_lane_task.params);
        if (root_lane_task.params.sampling.seed != LLAMA_DEFAULT_SEED &&
            root_lane_task.index > 0) {
            uint64_t mixed = root_lane_task.params.sampling.seed ^
                (uint64_t(root_lane_task.index) * 0x9e3779b97f4a7c15ULL);
            mixed ^= mixed >> 33;
            mixed *= 0xff51afd7ed558ccdULL;
            mixed ^= mixed >> 33;
            root_lane_task.params.sampling.seed = static_cast<uint32_t>(mixed);
            if (root_lane_task.params.sampling.seed == LLAMA_DEFAULT_SEED) {
                --root_lane_task.params.sampling.seed;
            }
        }
        common_sampler_ptr root_lane_sampler;
        try {
            root_lane_sampler.reset(common_sampler_init(
                model_tgt,
                root_lane_task.params.sampling,
                (int32_t) llama_n_ctx(ctx_tgt)));
        } catch (const std::exception &) {
            return false;
        }

        const uint64_t episode_id = rerot->adopt_root(
            slot.task->id,
            slot.task->rerot_response_owner(),
            slot.id,
            slot.id,
            slot.prompt.tokens.pos_next(),
            slot.task->rerot_episode_id);
        if (episode_id == 0) {
            return false;
        }

        llama_rerot_episode_params episode_params = {
            slot.task->params.rerot_frontier,
            1,
            1,
            1,
        };
        if (!llama_rerot_episode_begin(ctx_tgt, episode_id, &episode_params)) {
            rerot->erase_episode(episode_id);
            return false;
        }

        // Bound total work at episode scope, never per child/depth and never
        // by reserving a synthetic final-answer quota. The logical token cap
        // is the request's normal completion budget bounded by context; queue
        // pressure follows the internal sequence-id arena used by lineages.
        const uint64_t context_limit = std::max<uint64_t>(1, llama_n_ctx(ctx_tgt));
        const uint64_t context_budget = slot.task->params.n_predict > 0
            ? std::min<uint64_t>(context_limit, slot.task->params.n_predict)
            : context_limit;
        const uint64_t internal_seq_budget =
            LLAMA_MAX_SEQ > slots.size() ? LLAMA_MAX_SEQ - slots.size() : 1;
        rerot->set_hard_limits(episode_id, {
            context_budget,
            std::max<uint64_t>(64, internal_seq_budget * 16),
            internal_seq_budget,
            context_budget,
        });

        auto transport = std::make_unique<rerot_transport_state>(
            rerot_clone_task(*slot.task));
        transport->lora = slot.lora;
        transport->lineage_tokens.emplace(0, slot.prompt.tokens.clone());
        transport->root_seed = common_sampler_get_seed(root_lane_sampler.get());
        transport->prompt_tokens_processed = slot.n_prompt_tokens_processed;
        transport->prompt_tokens_processed_accum = slot.n_prompt_tokens_processed_accum;
        transport->prompt_tokens_cache_response = slot.n_prompt_tokens_cache_response;
        transport->prompt_processing_ms = slot.t_prompt_processing;
        transport->triattention_compressed = slot.triattention_compressed;
        transport->started_us = ggml_time_us();
        rerot_transport.emplace(episode_id, std::move(transport));

        slot.task = std::make_unique<const server_task>(std::move(root_lane_task));
        slot.smpl = std::move(root_lane_sampler);
        llama_set_sampler(ctx_tgt, slot.id, nullptr);

        ++rerot_metrics.episode_total;
        ++rerot_metrics.episode_active;
        ++rerot_metrics.nodes_created;
        ++rerot_metrics.nodes_started;
        rerot_metrics.max_live_lanes = std::max<uint64_t>(
            rerot_metrics.max_live_lanes, 1);

        rerot_episode_id = episode_id;
        slot.rerot_internal = true;
        slot.rerot_episode_id = episode_id;
        slot.rerot_node_id = 0;
        slot.rerot_exec_seq = slot.id;
        slot.rerot_serial_tail = false;
        slot.rerot_inflight_plan.reset();
        slot.rerot_inflight_bytes.clear();
        slot.rerot_inflight_extra_plans.clear();
        slot.rerot_inflight_extra_bytes.clear();
        slot.rerot_inflight_forced = false;
        slot.state = SLOT_STATE_GENERATING;
        slot.i_batch = -1;
        slot.spec_draft.clear();
        slot.spec_i_batch.clear();
        if (spec) {
            common_speculative_set_state(spec.get(), slot.id, {});
        }

        if (!rerot_set_injection(
                slot,
                server_rerot_injection_kind::planner,
                rerot_planner_control_prompt())) {
            rerot_propagate_hard_abort();
            return false;
        }

        SRV_INF("RERoT episode start: request=%d episode=%" PRIu64 " root_slot=%d frontier=%s\n",
            slot.task->id, episode_id, slot.id,
            llama_rerot_frontier_mode_name(slot.task->params.rerot_frontier));
        return true;
    }

    bool rerot_plan_next_token(server_slot & slot) {
        if (!slot.rerot_internal || !rerot || slot.rerot_episode_id == 0 ||
            slot.rerot_inflight_plan.has_value() ||
            !slot.rerot_inflight_extra_plans.empty()) {
            return false;
        }

        auto * lane = rerot->node(slot.rerot_episode_id, slot.rerot_node_id);
        if (!lane || lane->exec_seq != slot.id || lane->physical_slot != slot.id) {
            rerot->hard_abort(slot.rerot_episode_id, "rerot_state_error: slot/Lane mapping drift");
            return false;
        }

        const bool forced =
            slot.rerot_injection != server_rerot_injection_kind::none &&
            slot.rerot_injection_cursor < slot.rerot_injection_tokens.size();
        const llama_token token = forced
            ? slot.rerot_injection_tokens[slot.rerot_injection_cursor]
            : slot.sampled;
        const std::string bytes = common_token_to_piece(ctx_tgt, token, true);

        std::optional<server_rerot_token_plan> plan;
        if (forced && server_rerot_private_microbatch(slot.rerot_injection)) {
            auto plans = rerot->plan_private_span(
                slot.rerot_episode_id,
                slot.rerot_node_id,
                lane->storage_pos_next,
                rerot_private_batch_size(slot));
            if (plans.has_value() && !plans->empty()) {
                plan = std::move(plans->front());
                for (size_t i = 1; i < plans->size(); ++i) {
                    slot.rerot_inflight_extra_plans.push_back(std::move((*plans)[i]));
                    slot.rerot_inflight_extra_bytes.push_back(common_token_to_piece(
                        ctx_tgt,
                        slot.rerot_injection_tokens[slot.rerot_injection_cursor + i],
                        true));
                }
            }
        } else if (forced &&
                   (slot.rerot_injection == server_rerot_injection_kind::planner_open ||
                    slot.rerot_injection == server_rerot_injection_kind::worker_close)) {
            plan = rerot->plan_generated_token(
                slot.rerot_episode_id, slot.rerot_node_id, lane->storage_pos_next, bytes);
        } else if (forced && slot.rerot_injection == server_rerot_injection_kind::heading) {
            plan = rerot->plan_heading_token(
                slot.rerot_episode_id, slot.rerot_node_id, lane->storage_pos_next);
        } else if (slot.rerot_serial_tail) {
            plan = rerot->plan_serial_token(
                slot.rerot_episode_id, slot.rerot_node_id, lane->storage_pos_next);
        } else {
            plan = rerot->plan_generated_token(
                slot.rerot_episode_id, slot.rerot_node_id, lane->storage_pos_next, bytes);
        }

        if (!plan.has_value()) {
            SRV_ERR("RERoT plan failed: episode=%" PRIu64
                    " node=%u serial=%d forced=%d pos=%d token=%d\n",
                slot.rerot_episode_id, slot.rerot_node_id,
                slot.rerot_serial_tail ? 1 : 0, forced ? 1 : 0,
                lane->storage_pos_next, token);
            rerot->hard_abort(slot.rerot_episode_id, "rerot_protocol_error: failed to plan Lane token");
            return false;
        }

        slot.rerot_inflight_plan = std::move(plan);
        slot.rerot_inflight_bytes = bytes;
        slot.rerot_inflight_forced = forced;
        return true;
    }

    bool rerot_append_planned_token(server_slot & slot) {
        if (!slot.rerot_internal || !slot.rerot_inflight_plan.has_value() || !rerot) {
            return false;
        }
        auto transport_it = rerot_transport.find(slot.rerot_episode_id);
        const bool segmented =
            transport_it != rerot_transport.end() &&
            transport_it->second->segmented_active;
        size_t span_count = 0;
        const int64_t ddvr_started_us = segmented ? ggml_time_us() : 0;
        const bool installed = segmented
            ? rerot->install_token_plan(
                slot.rerot_episode_id,
                slot.rerot_node_id,
                *slot.rerot_inflight_plan,
                &span_count)
            : rerot->install_token_plan_write_only(
                slot.rerot_episode_id,
                slot.rerot_node_id,
                *slot.rerot_inflight_plan);
        if (segmented && installed) {
            rerot_metrics.span_count += span_count;
            rerot_metrics.ddvr_seconds +=
                static_cast<double>(ggml_time_us() - ddvr_started_us) / 1.0e6;
        }
        if (!installed) {
            SRV_ERR("RERoT install failed in server: episode=%" PRIu64
                    " node=%u serial=%d run=%u pos=%d\n",
                slot.rerot_episode_id, slot.rerot_node_id,
                slot.rerot_serial_tail ? 1 : 0,
                slot.rerot_inflight_plan->run_id,
                slot.rerot_inflight_plan->storage_pos);
            return false;
        }

        const size_t n_plans = 1 + slot.rerot_inflight_extra_plans.size();
        for (size_t i = 0; i < n_plans; ++i) {
            const auto & plan = i == 0
                ? *slot.rerot_inflight_plan
                : slot.rerot_inflight_extra_plans[i - 1];
            const llama_token token = slot.rerot_inflight_forced
                ? slot.rerot_injection_tokens.at(slot.rerot_injection_cursor)
                : slot.sampled;
            const bool forced_last = slot.rerot_inflight_forced &&
                slot.rerot_injection_cursor + 1 == slot.rerot_injection_tokens.size();
            const bool needs_logits = !slot.rerot_inflight_forced ||
                (forced_last &&
                 slot.rerot_injection != server_rerot_injection_kind::heading &&
                 slot.rerot_injection != server_rerot_injection_kind::planner &&
                 slot.rerot_injection != server_rerot_injection_kind::worker_close);
            const int32_t i_batch = batch.size();
            if (!batch.add(slot.id, token, plan.storage_pos, needs_logits)) {
                rerot->hard_abort(
                    slot.rerot_episode_id,
                    "rerot_resource_exhausted: private microbatch capacity");
                return false;
            }

            slot.i_batch = i_batch;
            slot.prompt.tokens.push_back(token);
            if (slot.rerot_inflight_forced) {
                ++slot.rerot_injection_cursor;
            }
        }
        return true;
    }

    uint32_t rerot_lane_seed(
            const server_rerot_episode & episode,
            llama_rerot_node_id node_id,
            uint32_t root_seed) const {
        uint64_t hash = root_seed;
        for (const uint32_t component : episode.document.tree_path(node_id)) {
            hash ^= uint64_t(component) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        }
        hash ^= hash >> 33;
        hash *= 0xff51afd7ed558ccdULL;
        hash ^= hash >> 33;
        return static_cast<uint32_t>(hash);
    }

    void rerot_make_slot_idle(server_slot & slot) {
        slot.prompt.clear();
        slot.state = SLOT_STATE_IDLE;
        slot.reset();
    }

    bool rerot_prepare_child_slot(
            uint64_t episode_id,
            llama_rerot_node_id node_id,
            server_slot & slot) {
        auto transport_it = rerot_transport.find(episode_id);
        auto * episode = rerot ? rerot->episode(episode_id) : nullptr;
        auto * lane = rerot ? rerot->node(episode_id, node_id) : nullptr;
        if (transport_it == rerot_transport.end() || !episode || !lane ||
            lane->physical_slot != slot.id || lane->exec_seq != slot.id) {
            return false;
        }

        auto lineage_it = transport_it->second->lineage_tokens.find(node_id);
        if (lineage_it == transport_it->second->lineage_tokens.end()) {
            rerot->hard_abort(episode_id, "rerot_state_error: missing child lineage token tape");
            return false;
        }

        server_task task = rerot_clone_task(transport_it->second->response_task);
        rerot_remove_planner_grammar(task.params);
        task.params.sampling.seed = rerot_lane_seed(
            *episode, node_id, transport_it->second->root_seed);
        task.rerot_episode_id = episode_id;

        try {
            slot.smpl.reset(common_sampler_init(
                model_tgt, task.params.sampling, (int32_t) llama_n_ctx(ctx_tgt)));
        } catch (const std::exception & e) {
            rerot->hard_abort(episode_id, std::string("rerot_sampler_error: ") + e.what());
            return false;
        }

        slot.task = std::make_unique<const server_task>(std::move(task));
        llama_set_sampler(ctx_tgt, slot.id, nullptr);
        slot.lora = transport_it->second->lora;
        slot.prompt.clear();
        slot.prompt.tokens = lineage_it->second.clone();
        slot.response_prompt.clear();
        slot.generated_text.clear();
        slot.generated_tokens.clear();
        slot.generated_token_probs.clear();
        slot.n_sent_text = 0;
        slot.n_decoded = 0;
        slot.n_decoded_start = 0;
        slot.n_remaining = -1;
        slot.n_prompt_tokens_original = transport_it->second->response_task.n_tokens();
        slot.n_prompt_tokens_processed = 0;
        slot.n_prompt_tokens_processed_accum = 0;
        slot.t_start_process_prompt = transport_it->second->started_us;
        slot.t_start_generation = transport_it->second->started_us;
        slot.has_next_token = true;
        slot.has_new_line = false;
        slot.truncated = false;
        slot.stop = STOP_TYPE_NONE;
        slot.stopping_word.clear();
        slot.state = SLOT_STATE_GENERATING;
        slot.i_batch = -1;
        slot.sampled = LLAMA_TOKEN_NULL;
        slot.spec_draft.clear();
        slot.spec_i_batch.clear();
        if (spec) {
            common_speculative_set_state(spec.get(), slot.id, {});
        }
        llama_set_sampler(ctx_tgt, slot.id, nullptr);

        slot.rerot_internal = true;
        slot.rerot_episode_id = episode_id;
        slot.rerot_node_id = node_id;
        slot.rerot_exec_seq = slot.id;
        slot.rerot_serial_tail = false;
        slot.rerot_inflight_plan.reset();
        slot.rerot_inflight_bytes.clear();
        slot.rerot_inflight_extra_plans.clear();
        slot.rerot_inflight_extra_bytes.clear();
        slot.rerot_inflight_forced = false;
        slot.triattention_compressed = transport_it->second->triattention_compressed;

        const auto * child_doc = episode->document.node(node_id);
        if (child_doc && ctx_tgt) {
            auto snap_it = transport_it->second->parent_recurrent_snapshots.find(child_doc->parent);
            if (snap_it != transport_it->second->parent_recurrent_snapshots.end() &&
                !snap_it->second.empty()) {
                llama_state_seq_set_data_ext(
                    ctx_tgt,
                    snap_it->second.data(),
                    snap_it->second.size(),
                    slot.id,
                    LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
        }

        const std::string heading = rerot->heading_text(episode_id, node_id);
        if (!rerot_set_injection(slot, server_rerot_injection_kind::heading, heading)) {
            return false;
        }

        ++rerot_metrics.nodes_started;
        rerot_metrics.queue_max = std::max<uint64_t>(
            rerot_metrics.queue_max, episode->ready_queue.size());
        const uint64_t live_lanes =
            episode->running.size() + episode->starting.size();
        rerot_metrics.max_live_lanes = std::max<uint64_t>(
            rerot_metrics.max_live_lanes, live_lanes);
        if (live_lanes > 1) {
            rerot_note_parallel_start(episode_id);
        }

        SRV_INF("RERoT Lane admitted: episode=%" PRIu64 " node=%u slot=%d queued=%zu\n",
            episode_id, node_id, slot.id, episode->ready_queue.size());
        return true;
    }

    bool rerot_admit_ready(uint64_t episode_id) {
        auto * episode = rerot ? rerot->episode(episode_id) : nullptr;
        if (!episode || episode->hard_aborted) {
            return false;
        }

        while (!episode->ready_queue.empty()) {
            llama_memory_kv_usage recurrent = {};
            if (llama_memory_get_recurrent_usage(llama_get_memory(ctx_tgt), &recurrent)) {
                // Already-running Lanes are included in recurrent.used.
                // STARTING Lanes still share parked tails and each may need
                // one new physical COW cell on its first joint frontier.
                const uint64_t required_after_cow =
                    recurrent.used + episode->starting.size() + 1;
                if (required_after_cow > recurrent.capacity) {
                    // Physical recurrent capacity temporarily saturated;
                    // queued children remain safely in ready_queue waiting for
                    // running slots to complete. Never abort!
                    break;
                }
            }

            server_slot * free_slot = nullptr;
            for (auto & candidate : slots) {
                if (!candidate.is_processing()) {
                    free_slot = &candidate;
                    break;
                }
            }
            if (!free_slot) {
                break;
            }

            // Idle slots may retain ordinary prompt-cache refs. Admission owns
            // the execution seq, so clear those refs before recurrent/base
            // prefix state is installed.
            free_slot->prompt_clear();

            llama_rerot_node_id admitted = LLAMA_REROT_NODE_INVALID;
            if (!rerot->admit_next_child(
                    episode_id, free_slot->id, free_slot->id, &admitted) ||
                admitted == LLAMA_REROT_NODE_INVALID ||
                !rerot_prepare_child_slot(episode_id, admitted, *free_slot)) {
                rerot->hard_abort(episode_id, "rerot_state_error: child admission failed");
                return false;
            }
        }
        return true;
    }

    std::string rerot_render_public_document(
            uint64_t episode_id,
            llama_rerot_node_id reader) const {
        const auto * episode = rerot ? rerot->episode(episode_id) : nullptr;
        const auto transport_it = rerot_transport.find(episode_id);
        if (!episode || transport_it == rerot_transport.end()) {
            return {};
        }

        const auto view = episode->document.build_view(reader);
        std::string result;
        for (const auto & view_run : view.runs) {
            const auto * run = episode->document.run(view_run.run_id);
            if (!run || run->visibility != llama_rerot_visibility::public_live) {
                continue;
            }
            const auto text_it = transport_it->second->run_bytes.find(run->id);
            if (text_it != transport_it->second->run_bytes.end()) {
                result += text_it->second;
            }
        }
        return result;
    }

    void rerot_erase_episode(uint64_t episode_id) {
        if (ctx_tgt) {
            llama_rerot_episode_end(ctx_tgt, episode_id);
        }
        if (rerot) {
            rerot->erase_episode(episode_id);
        }
        rerot_transport.erase(episode_id);
        if (rerot_metrics.episode_active > 0) {
            --rerot_metrics.episode_active;
        }
        if (rerot_episode_id == episode_id) {
            rerot_episode_id = 0;
        }
    }

    void rerot_start_next_waiting_root() {
        if (!rerot || rerot_active()) {
            return;
        }
        for (auto & waiting : slots) {
            if (waiting.state != SLOT_STATE_DONE_PROMPT || !waiting.task ||
                !waiting.task->params.rerot_effective(waiting.task->type)) {
                continue;
            }
            if (rerot_start_root(waiting)) {
                return;
            }
            send_error(waiting, "failed to start queued RERoT episode", ERROR_TYPE_SERVER);
            waiting.prompt.clear();
            waiting.release();
        }
    }

    bool rerot_commit_inflight(server_slot & slot) {
        if (!rerot || !slot.rerot_inflight_plan.has_value()) {
            return false;
        }

        const server_rerot_token_plan plan = *slot.rerot_inflight_plan;
        const uint64_t episode_id = slot.rerot_episode_id;
        const llama_rerot_node_id node_id = slot.rerot_node_id;
        auto transport_it = rerot_transport.find(episode_id);
        if (transport_it == rerot_transport.end()) {
            rerot->hard_abort(episode_id, "rerot_state_error: missing transport state at commit");
            return false;
        }
        if (slot.rerot_inflight_extra_plans.size() !=
            slot.rerot_inflight_extra_bytes.size()) {
            rerot->hard_abort(episode_id, "rerot_state_error: private microbatch plan drift");
            return false;
        }

        const auto commit_one = [&](const server_rerot_token_plan & current_plan,
                                    const std::string & bytes) {
            if (!rerot->commit_token(episode_id, node_id, current_plan)) {
                return false;
            }
            transport_it->second->run_bytes[current_plan.run_id] += bytes;
            if (!slot.rerot_serial_tail &&
                current_plan.visibility != llama_rerot_visibility::private_control &&
                !rerot_stream_commit(
                    episode_id, node_id, current_plan.run_id, bytes)) {
                return false;
            }
            ++transport_it->second->model_tokens;
            if (transport_it->second->parallel_started_us != 0 &&
                !transport_it->second->parallel_finished) {
                ++transport_it->second->parallel_model_tokens;
            }
            switch (current_plan.visibility) {
                case llama_rerot_visibility::public_live:     ++rerot_metrics.public_tokens;  break;
                case llama_rerot_visibility::private_control: ++rerot_metrics.private_tokens; break;
                case llama_rerot_visibility::pending_record:  ++rerot_metrics.pending_tokens; break;
                case llama_rerot_visibility::normal: break;
            }
            return true;
        };
        if (!commit_one(plan, slot.rerot_inflight_bytes)) {
            return false;
        }
        for (size_t i = 0; i < slot.rerot_inflight_extra_plans.size(); ++i) {
            if (!commit_one(
                    slot.rerot_inflight_extra_plans[i],
                    slot.rerot_inflight_extra_bytes[i])) {
                return false;
            }
        }

        const bool forced = slot.rerot_inflight_forced;
        const auto injection = slot.rerot_injection;
        const bool forced_complete = forced &&
            slot.rerot_injection_cursor == slot.rerot_injection_tokens.size();
        const bool terminal_plan =
            plan.parser_step.record_closed &&
            plan.parser_step.items.size() == 1;
        const bool lane_finished =
            plan.marker_step.marker_closed ||
            (plan.parser_step.record_closed && !terminal_plan);
        if (lane_finished && !rerot_stream_finish_lane(episode_id, node_id)) {
            return false;
        }

        slot.rerot_inflight_plan.reset();
        slot.rerot_inflight_bytes.clear();
        slot.rerot_inflight_extra_plans.clear();
        slot.rerot_inflight_extra_bytes.clear();
        slot.rerot_inflight_forced = false;

        if (terminal_plan) {
            const auto * document_node = rerot->episode(episode_id)
                ? rerot->episode(episode_id)->document.node(node_id)
                : nullptr;
            const std::string_view assigned_task =
                document_node && !document_node->title.empty()
                    ? std::string_view(document_node->title)
                    : std::string_view(plan.parser_step.items.front());

            // The planner grammar is complete at </ol> and must not constrain
            // the terminal worker. Rebuild only this Lane's sampler with the
            // same deterministic seed and no user/tool grammar; the original
            // request sampler is restored separately after the final fence.
            server_task worker_task = rerot_clone_task(*slot.task);
            rerot_remove_planner_grammar(worker_task.params);
            common_sampler_ptr worker_sampler;
            try {
                worker_sampler.reset(common_sampler_init(
                    model_tgt,
                    worker_task.params.sampling,
                    (int32_t) llama_n_ctx(ctx_tgt)));
            } catch (const std::exception & e) {
                rerot->hard_abort(
                    episode_id,
                    std::string("rerot_sampler_error: failed to leave planner grammar: ") + e.what());
                return false;
            }
            slot.task = std::make_unique<const server_task>(std::move(worker_task));
            slot.smpl = std::move(worker_sampler);
            llama_set_sampler(ctx_tgt, slot.id, nullptr);

            if (!rerot_set_injection(
                    slot,
                    server_rerot_injection_kind::worker,
                    rerot_worker_control_prompt(assigned_task))) {
                rerot->hard_abort(episode_id, "rerot_protocol_error: failed to arm terminal Lane work");
                return false;
            }
            return true;
        }

        if (!forced_complete) {
            return true;
        }

        slot.rerot_injection_tokens.clear();
        slot.rerot_injection_cursor = 0;
        slot.rerot_injection = server_rerot_injection_kind::none;

        if (injection == server_rerot_injection_kind::heading) {
            const auto * episode = rerot->episode(episode_id);
            const auto * document_node = episode
                ? episode->document.node(node_id)
                : nullptr;
            if (!document_node ||
                !rerot->publish_heading(episode_id, node_id, plan.run_id) ||
                !rerot->complete_admission(episode_id, node_id)) {
                rerot->hard_abort(episode_id, "rerot_protocol_error: child heading publication failed");
                return false;
            }
            // Child and grandchild nodes default to direct worker reasoning.
            // They do not automatically start with <ol> or planner grammar;
            // if the LLM itself generates a valid <ol></ol>, the runtime
            // parser detects it and continues tree branching.
            if (!rerot_stream_visibility_changed(episode_id, node_id) ||
                !rerot_set_injection(
                    slot,
                    server_rerot_injection_kind::worker,
                    rerot_worker_control_prompt(document_node->title))) {
                rerot->hard_abort(episode_id, "rerot_protocol_error: child heading/worker admission failed");
                return false;
            }
        } else if (injection == server_rerot_injection_kind::planner) {
            // Only root planner opens <ol>\n<li> automatically to begin the parallel ring.
            if (!rerot_set_injection(
                    slot,
                    server_rerot_injection_kind::planner_open,
                    "<ol>\n<li>")) {
                rerot->hard_abort(episode_id, "rerot_protocol_error: failed to arm planner list");
                return false;
            }
        }
        return true;
    }

    bool rerot_begin_serial_tail(
            uint64_t episode_id,
            llama_rerot_node_id final_node) {
        auto * lane = rerot ? rerot->node(episode_id, final_node) : nullptr;
        if (!lane || lane->physical_slot < 0 ||
            lane->physical_slot >= static_cast<int>(slots.size())) {
            return false;
        }
        auto & slot = slots[lane->physical_slot];
        auto transport_it = rerot_transport.find(episode_id);
        if (transport_it == rerot_transport.end() || !slot.task) {
            return false;
        }

        server_task final_task =
            rerot_clone_task(transport_it->second->response_task);
        final_task.params.sampling.seed = slot.task->params.sampling.seed;
        common_sampler_ptr final_sampler;
        try {
            final_sampler.reset(common_sampler_init(
                model_tgt,
                final_task.params.sampling,
                (int32_t) llama_n_ctx(ctx_tgt)));
        } catch (const std::exception &) {
            rerot->hard_abort(episode_id, "rerot_state_error: failed to restore final sampler grammar");
            return false;
        }

        std::vector<uint32_t> fence_runs;
        if (!rerot->refresh_final_fence(episode_id, final_node, &fence_runs) ||
            !llama_rerot_refresh_barrier(ctx_tgt, episode_id) ||
            !rerot->complete_serial_tail(episode_id, final_node)) {
            rerot->hard_abort(episode_id, "rerot_state_error: final acquire fence failed");
            return false;
        }
        rerot_note_parallel_end(episode_id, ggml_time_us());

        std::string logical_reasoning =
            rerot_render_public_document(episode_id, final_node);
        if (slot.task && slot.task->params.rerot_trace) {
            SRV_INF("rerot.trace.document: episode=%" PRIu64 " node=%u bytes=%zu text=%s\n",
                episode_id, final_node, logical_reasoning.size(), logical_reasoning.c_str());
        }

        const bool chronological_stream = final_task.params.stream;
        if (chronological_stream && !transport_it->second->line_mux.empty()) {
            rerot->hard_abort(
                episode_id,
                "rerot_stream_error: final fence reached with unfinished Lane lines");
            return false;
        }
        const size_t reasoning_bytes_sent =
            chronological_stream ? transport_it->second->streamed_reasoning.size() : 0;
        std::string response_reasoning = chronological_stream
            ? transport_it->second->streamed_reasoning
            : std::move(logical_reasoning);
        response_reasoning += "</think>\n\n";

        slot.task = std::make_unique<const server_task>(std::move(final_task));
        slot.smpl = std::move(final_sampler);
        llama_set_sampler(ctx_tgt, slot.id, nullptr);
        slot.generated_text = std::move(response_reasoning);
        slot.generated_tokens.clear();
        slot.generated_token_probs.clear();
        slot.n_sent_text = reasoning_bytes_sent;
        slot.n_prompt_tokens_original = transport_it->second->response_task.n_tokens();
        slot.n_prompt_tokens_processed = transport_it->second->prompt_tokens_processed;
        slot.n_prompt_tokens_processed_accum =
            transport_it->second->prompt_tokens_processed_accum;
        slot.n_prompt_tokens_cache_response =
            transport_it->second->prompt_tokens_cache_response;
        slot.t_prompt_processing = transport_it->second->prompt_processing_ms;
        slot.n_decoded = 0;
        slot.n_decoded_start = 0;
        slot.has_next_token = true;
        slot.stop = STOP_TYPE_NONE;
        slot.stopping_word.clear();
        slot.rerot_serial_tail = true;
        if (!rerot_set_injection(
                slot,
                server_rerot_injection_kind::finalizer,
                rerot_finalizer_control_prompt(
                    transport_it->second->response_task.rerot_original_user_text))) {
            rerot->hard_abort(episode_id, "rerot_protocol_error: failed to inject finalizer turn");
            return false;
        }

        ++rerot_metrics.final_fences;
        ++rerot_metrics.refresh_total;
        SRV_INF("RERoT final fence: episode=%" PRIu64
                " node=%u slot=%d public_bytes=%zu streamed_lines=%" PRIu64 "\n",
            episode_id,
            final_node,
            slot.id,
            slot.generated_text.size(),
            transport_it->second->streamed_lines);
        return true;
    }

    bool rerot_handle_finished_frontier(uint64_t episode_id) {
        auto result = rerot->finish_frontier(episode_id);
        ++rerot_metrics.frontiers;
        if (result.topology_barrier) {
            ++rerot_metrics.topology_barriers;
            ++rerot_metrics.refresh_total;
        }
        if (result.hard_aborted) {
            return false;
        }

        auto transport_it = rerot_transport.find(episode_id);
        auto * episode = rerot->episode(episode_id);
        if (transport_it == rerot_transport.end() || !episode) {
            rerot->hard_abort(episode_id, "rerot_state_error: missing episode after frontier");
            return false;
        }

        rerot_metrics.nodes_retired += result.retired.size();
        rerot_metrics.archive_total += result.retired.size();
        for (const auto parent_id : result.forked) {
            auto * parent = rerot->node(episode_id, parent_id);
            const auto * parent_doc = episode->document.node(parent_id);
            if (!parent || !parent_doc || parent->physical_slot < 0 ||
                parent->physical_slot >= static_cast<int>(slots.size())) {
                rerot->hard_abort(episode_id, "rerot_state_error: fork parent lost physical slot");
                return false;
            }

            auto & parent_slot = slots[parent->physical_slot];
            for (const auto child_id : parent_doc->children) {
                transport_it->second->lineage_tokens.insert_or_assign(
                    child_id, parent_slot.prompt.tokens.clone());
            }

            if (ctx_tgt) {
                const size_t snap_sz = llama_state_seq_get_size_ext(
                    ctx_tgt, parent_slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                if (snap_sz > 0) {
                    std::vector<uint8_t> snap(snap_sz);
                    if (llama_state_seq_get_data_ext(
                            ctx_tgt, snap.data(), snap_sz, parent_slot.id,
                            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) > 0) {
                        transport_it->second->parent_recurrent_snapshots.insert_or_assign(
                            parent_id, std::move(snap));
                    }
                }
            }

            ++rerot_metrics.forks_total;
            ++rerot_metrics.archive_total;
            rerot_metrics.parked_total += parent_doc->children.size();
            rerot_metrics.nodes_created += parent_doc->children.size();
            rerot_metrics.nodes_queued += parent_doc->children.size();
            rerot_metrics.queue_max = std::max<uint64_t>(
                rerot_metrics.queue_max, episode->ready_queue.size());
            for (const auto child_id : parent_doc->children) {
                rerot_metrics.max_depth = std::max<uint64_t>(
                    rerot_metrics.max_depth,
                    episode->document.tree_path(child_id).size());
            }

            const int released_slot = parent->physical_slot;
            if (!rerot->freeze_fork_parent(episode_id, parent_id)) {
                return false;
            }

            // Scale thinking budget by number of sub-lanes, bounded by context window
            const uint64_t context_limit = std::max<uint64_t>(1, llama_n_ctx(ctx_tgt));
            const uint64_t base_budget = transport_it->second->response_task.params.n_predict > 0
                ? std::min<uint64_t>(context_limit, (uint64_t) transport_it->second->response_task.params.n_predict)
                : context_limit;
            const uint64_t sublanes = std::max<uint64_t>(1, episode->document.node_count());
            const uint64_t scaled_budget = std::min<uint64_t>(context_limit, base_budget * sublanes);
            const uint64_t internal_seq_budget =
                LLAMA_MAX_SEQ > slots.size() ? LLAMA_MAX_SEQ - slots.size() : 1;
            rerot->set_hard_limits(episode_id, {
                scaled_budget,
                std::max<uint64_t>(64, internal_seq_budget * 16),
                internal_seq_budget,
                scaled_budget,
            });

            transport_it->second->segmented_active = true;
            transport_it->second->triattention_compressed |=
                parent_slot.triattention_compressed;
            rerot_make_slot_idle(slots[released_slot]);

            SRV_INF("RERoT fork: episode=%" PRIu64 " parent=%u children=%zu released_slot=%d\n",
                episode_id, parent_id, parent_doc->children.size(), released_slot);
        }

        for (const int released_slot : result.released_slots) {
            if (released_slot >= 0 && released_slot < static_cast<int>(slots.size())) {
                rerot_make_slot_idle(slots[released_slot]);
            }
        }

        if (result.topology_barrier) {
            if (!llama_rerot_refresh_barrier(ctx_tgt, episode_id)) {
                rerot->hard_abort(episode_id, "rerot_state_error: topology barrier failed");
                return false;
            }
            // Existing next-token logits were produced while the structural
            // record was still PENDING. A private one-token control refresh
            // re-evaluates each otherwise-idle Lane on the newly published
            // PAC-DFS view before sampling. Lanes already entering a forced
            // heading/planner/worker sequence refresh on that sequence.
            for (auto & lane_slot : slots) {
                if (!lane_slot.rerot_internal ||
                    lane_slot.rerot_episode_id != episode_id ||
                    lane_slot.rerot_serial_tail ||
                    lane_slot.rerot_injection != server_rerot_injection_kind::none) {
                    continue;
                }
                const auto * lane_node = episode->document.node(lane_slot.rerot_node_id);
                if (!lane_node ||
                    lane_node->state == llama_rerot_node_state::forked ||
                    lane_node->state == llama_rerot_node_state::retired) {
                    continue;
                }
                if (!rerot_set_injection(
                        lane_slot,
                        server_rerot_injection_kind::refresh,
                        "\n")) {
                    return false;
                }
            }
        }

        if (!rerot_admit_ready(episode_id)) {
            return false;
        }

        if (result.natural_final() &&
            !rerot_begin_serial_tail(episode_id, result.final_node)) {
            return false;
        }

        return true;
    }

    bool rerot_sample_slot(server_slot & slot, int tok_idx) {
        if (!slot.smpl || tok_idx < 0) {
            SRV_ERR("RERoT sample unavailable: episode=%" PRIu64
                    " node=%u sampler=%d tok_idx=%d i_batch=%d\n",
                slot.rerot_episode_id, slot.rerot_node_id,
                slot.smpl ? 1 : 0, tok_idx, slot.i_batch);
            if (rerot && slot.rerot_episode_id != 0) {
                rerot->hard_abort(
                    slot.rerot_episode_id,
                    string_format(
                        "rerot_state_error: invalid sampler/logit row (sampler=%d tok_idx=%d i_batch=%d)",
                        slot.smpl ? 1 : 0, tok_idx, slot.i_batch));
            }
            return false;
        }

        llama_token id;
        {
            scoped_timer timer(t_sampl, n_sampl);
            id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
        }
        if (slot.task && slot.task->params.rerot_trace) {
            SRV_INF("rerot.trace.sample: episode=%" PRIu64 " node=%u serial=%d tok_idx=%d token=%d piece=%s\n",
                slot.rerot_episode_id, slot.rerot_node_id, slot.rerot_serial_tail ? 1 : 0,
                tok_idx, id, common_token_to_piece(slot.ctx_tgt, id, true).c_str());
        }
        common_sampler_accept(slot.smpl.get(), id, true);
        slot.i_batch = -1;
        if (const auto transport_it = rerot_transport.find(slot.rerot_episode_id);
            transport_it != rerot_transport.end()) {
            ++transport_it->second->sampled_tokens;
        }

        const int64_t now = ggml_time_us();
        if (slot.t_start_generation == 0) {
            slot.t_start_generation = now;
        }
        ++slot.n_decoded;
        slot.t_token_generation = std::max<int64_t>(1, now - slot.t_start_generation) / 1e3;

        if (!slot.rerot_serial_tail) {
            if (llama_vocab_is_eog(vocab, id)) {
                const auto * episode = rerot->episode(slot.rerot_episode_id);
                const auto * document_node = episode
                    ? episode->document.node(slot.rerot_node_id)
                    : nullptr;
                if (document_node &&
                    (document_node->state == llama_rerot_node_state::terminal_running ||
                     document_node->state == llama_rerot_node_state::planning) &&
                    slot.rerot_injection == server_rerot_injection_kind::none &&
                    rerot_set_injection(
                        slot,
                        server_rerot_injection_kind::worker_close,
                        "</think>")) {
                    SRV_INF("RERoT Lane EOG normalized to record close: episode=%" PRIu64
                            " node=%u slot=%d\n",
                        slot.rerot_episode_id, slot.rerot_node_id, slot.id);
                    return true;
                }
                rerot->hard_abort(
                    slot.rerot_episode_id,
                    "rerot_protocol_error: Lane reached EOG outside terminal record");
                return false;
            }
            slot.sampled = id;
            return true;
        }

        slot.sampled = id;
        completion_token_output output;
        output.tok = id;
        output.text_to_send = common_token_to_piece(
            slot.ctx_tgt,
            id,
            params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(id) !=
                    slot.task->params.sampling.preserved_tokens.end());
        output.prob = 1.0f;
        if (slot.task->params.sampling.n_probs > 0) {
            populate_token_probs(
                slot, output, slot.task->params.post_sampling_probs,
                params_base.special, tok_idx);
        }

        if (process_token(output, slot)) {
            slot.print_timings_tg();
            return true;
        }

        const uint64_t episode_id = slot.rerot_episode_id;
        if (const auto transport_it = rerot_transport.find(episode_id);
            transport_it != rerot_transport.end()) {
            slot.n_decoded = static_cast<int32_t>(std::min<uint64_t>(
                transport_it->second->sampled_tokens,
                std::numeric_limits<int32_t>::max()));
        }
        SRV_INF("RERoT serial tail complete: episode=%" PRIu64 " node=%u token=%d stop=%d\n",
            episode_id, slot.rerot_node_id, id, slot.stop);
        rerot_record_completed_episode(episode_id);
        slot.print_timings();
        send_final_response(slot);
        metrics.on_prediction(slot);
        rerot_erase_episode(episode_id);
        slot.prompt.clear();
        slot.release();
        rerot_start_next_waiting_root();
        return true;
    }

    // Frontier scheduling entry (§§19-20): one frontier is one atomic budget
    // unit. Delegates to runtime::begin_frontier; on hard-limit breach the
    // runtime takes whole-episode HARD_ABORT (no survivor/answer/tool call)
    // and this hook propagates it to slots. Returns false when aborted.
    bool rerot_frontier_enter() {
        if (!rerot_active()) {
            return true;
        }
        const auto * episode = rerot_active_episode();
        if (episode && episode->serial_tail) {
            return true;
        }
        if (rerot->begin_frontier(rerot_episode_id)) {
            return true;
        }
        rerot_propagate_hard_abort();
        return false;
    }

    // Topology-barrier refresh trigger (§17): after a frontier commits with a
    // topology change, re-observe stable shared memory via a read-only
    // context refresh (no public token write; recurrent checkpoint/restore
    // handled inside llama_rerot_refresh_barrier when needed).
    void rerot_on_topology_barrier(bool barrier) {
        if (!rerot_active() || !barrier || ctx_tgt == nullptr) {
            return;
        }
        if (!llama_rerot_refresh_barrier(ctx_tgt, rerot_episode_id)) {
            rerot_propagate_hard_abort();
        }
    }

    // Final-fence serial transition (§§21-22): rebuilds the survivor view on
    // stable shared memory (acquire fence) then retires the parallel
    // scheduler so ordinary tool-call / body decode continues from the fence
    // survivor on the root response stream. Failures propagate as hard abort.
    void rerot_on_final_fence(llama_rerot_node_id final_node) {
        if (!rerot_active() || final_node == LLAMA_REROT_NODE_INVALID || ctx_tgt == nullptr) {
            return;
        }
        std::vector<uint32_t> fence_runs;
        if (!rerot->refresh_final_fence(rerot_episode_id, final_node, &fence_runs)) {
            rerot_propagate_hard_abort();
            return;
        }
        if (!rerot->complete_serial_tail(rerot_episode_id, final_node)) {
            rerot_propagate_hard_abort();
        }
    }

    // Hard-abort propagation (§20): one external error for the episode, then
    // release every internal Lane and every parked/archive reference. Lane
    // clones share the root task id, so emitting once per slot would violate
    // the single-termination response contract.
    void rerot_propagate_hard_abort() {
        if (!rerot || rerot_episode_id == 0 || !rerot->episode(rerot_episode_id)) {
            return;
        }
        const uint64_t episode_id = rerot_episode_id;
        ++rerot_metrics.hard_aborts;
        const server_rerot_episode * ep = rerot->episode(episode_id);
        const std::string reason = (ep && !ep->abort_reason.empty())
            ? ep->abort_reason : "rerot_resource_exhausted";
        rerot->hard_abort(episode_id, reason);

        const server_task * response_task = nullptr;
        const auto transport_it = rerot_transport.find(episode_id);
        if (transport_it != rerot_transport.end() && transport_it->second) {
            response_task = &transport_it->second->response_task;
        }

        server_slot * response_slot = nullptr;
        for (auto & slot : slots) {
            if (slot.rerot_internal && slot.rerot_episode_id == episode_id && slot.is_processing()) {
                if (!response_slot) {
                    response_slot = &slot;
                }
            }
        }
        if (response_slot) {
            send_error(*response_slot, "RERoT episode hard-aborted: " + reason, ERROR_TYPE_SERVER);
        } else if (response_task) {
            // Fork publication may have already released the last physical
            // Lane before child admission discovers recurrent exhaustion.
            // The transport owns the original response task until episode
            // teardown, so it remains the authoritative HTTP/SSE endpoint.
            send_error(*response_task, "RERoT episode hard-aborted: " + reason, ERROR_TYPE_SERVER);
        }

        for (auto & slot : slots) {
            if (slot.rerot_internal && slot.rerot_episode_id == episode_id && slot.is_processing()) {
                slot.prompt.clear();
                slot.release();
            }
        }
        rerot_erase_episode(episode_id);
        rerot_start_next_waiting_root();
    }

    // Episode-aware slot test (§A.11): a physical slot currently borrowed by
    // the active episode must never be preempted/demoted/shifted alone.
    bool rerot_owns_slot(int id_slot) const {
        if (!rerot_active()) {
            return false;
        }
        return rerot->episode_for_slot(id_slot) != nullptr;
    }

    wanxiangqi_server_token_dump request_token_dump;

    server_metrics metrics;
    server_rerot_metrics rerot_metrics;

    uint64_t tri_drain_count = 0;
    uint64_t tri_maintenance_count = 0;
    uint64_t tri_floor_exhausted_count = 0;
    uint64_t tri_atomic_fallback_kv_count = 0;
    uint64_t tri_atomic_fallback_recurrent_count = 0;

    uint64_t tri_cells_freed_total = 0;
    uint64_t tri_score_us_total = 0;
    uint64_t tri_pack_us_total = 0;

    uint64_t tri_cells_before = 0;
    uint64_t tri_cells_after = 0;
    uint64_t tri_cells_freed = 0;
    uint64_t tri_references_removed = 0;
    uint64_t tri_target_references = 0;
    uint64_t tri_hard_keep = 0;
    uint64_t tri_shared_keep = 0;
    uint32_t kv_batch_limit = UINT32_MAX;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;
    float  slot_cache_key_similarity = 0.0f;
    size_t slot_cache_key_min_prefix = 0;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    int64_t t_last_load_progress_ms = 0;

    void destroy() {
        // RERoT: end any active context episode gate before tearing down the
        // context, then drop the runtime (which releases parked/archive
        // lineage). OFF: both branches no-op.
        if (rerot_active() && ctx_tgt != nullptr) {
            llama_rerot_episode_end(ctx_tgt, rerot_episode_id);
        }
        rerot.reset();
        rerot_episode_id = 0;

        spec.reset();
        spec_init.reset();

        ctx_dft   = nullptr;
        model_dft = nullptr;

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        load_progress_data load_progress_text  (this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec  (this, "spec_model");

        const bool is_resume = sleeping;

        params_base = params;
        params_base.n_ctx_kv_reserve.clear();
        params_base.n_parallel_pp = 1;
        params_base.n_outputs_max = server_n_outputs_max(params_base);
        inference_mode = SERVER_INFERENCE_MODE_DECODE;
        id_slot_prefill = -1;

        if (!request_token_dump.init(params_base, is_resume)) {
            return false;
        }

        const bool has_mmproj = !params.mmproj.path.empty();
        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                        params_base.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
        const bool has_spec = has_draft || spec_mtp;

        if (has_draft && (params_base.n_ctx_kv_auto || params_base.n_ctx_kv > 0)) {
            SRV_ERR("%s", "dynamic unified KV sizing is not supported with an external draft model yet\n");
            return false;
        }

        if (callback_state) {
            std::vector<std::string> stages = {"text_model"};
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }


        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        std::string & mmproj_path = params_base.mmproj.path;
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
            mparams.media_marker     = get_media_marker();
            // progress callback
            mparams.progress_callback           = load_progress_callback;
            mparams.progress_callback_user_data = &load_progress_mmproj;
        }

        // optionally get the memory usage of mmproj
        if (has_mmproj && (params_base.fit_params || params_base.n_ctx_kv_auto)) {
            int64_t t_start = ggml_time_us();
            auto mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            int64_t t_elapsed = ggml_time_us() - t_start;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                                params_base.n_ctx_kv_reserve.emplace_back(dev, size);
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // optionally reserve VRAM for the draft / MTP context before fitting the target model
        if (params_base.fit_params) {
            if (has_spec) {
                // MTP draft context lives on the target model, only context+compute are new
                bool measure_model_bytes = has_draft;

                common_params params_dft = common_base_params_to_speculative(params_base);

                auto mparams_dft = common_model_params_to_llama(params_dft);
                auto cparams_dft = common_context_params_to_llama(params_dft);
                if (spec_mtp) {
                    cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                    if (params_base.n_ctx_kv_auto || params_base.n_ctx_kv > 0) {
                        // Dynamic KV/recurrent planning bootstraps from one physical
                        // recurrent state. The final MTP capacity follows the target
                        // context after unified KV sizing instead of reserving 4 here.
                        cparams_dft.n_seq_recurrent = 1;
                    }
                }
                cparams_dft.n_rs_seq = 0;

                std::vector<ggml_backend_dev_t> devs;
                uint32_t hp_ngl = 0;
                uint32_t hp_nct = 0;
                uint32_t hp_nex = 0;
                try {
                    auto dmd = common_get_device_memory_data(
                        params_dft.model.path.c_str(), &mparams_dft, &cparams_dft,
                        devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);

                    GGML_ASSERT(!params_base.fit_params_target.empty());
                    size_t total = 0;

                    std::vector<ggml_backend_dev_t> tgt_devices = params.devices;

                    if (tgt_devices.empty()) {
                        for(size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                           tgt_devices.push_back(ggml_backend_dev_get(i));
                        }
                    }

                    for (size_t j = 0; j < devs.size(); ++j) {
                        const size_t bytes = (measure_model_bytes ? dmd[j].model : 0) + dmd[j].context + dmd[j].compute;
                        total += bytes;
                        for (size_t i = 0; i < tgt_devices.size(); i++) {
                            if (tgt_devices[i] == devs[j]) {
                                SRV_DBG("[spec] adding %.2f MiB to fit_params_target for device %s\n",
                                        bytes / (1024.0 * 1024.0), ggml_backend_dev_name(devs[j]));
                                params_base.fit_params_target[i] += bytes;
                                break;
                            }
                        }
                    }
                    SRV_TRC("[spec] estimated memory usage of %s is %.2f MiB\n",
                            has_draft ? "draft model" : "MTP context",
                            total / (1024.0 * 1024.0));
                } catch (const std::exception & e) {
                    SRV_WRN("[spec] failed to measure %s memory: %s\n",
                            has_draft ? "draft model" : "MTP context", e.what());
                }
            }
        }

        // attach a progress callback
        {
            params_base.load_progress_callback = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        if (ctx_tgt == nullptr) {
            SRV_ERR("failed to create_context with model '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (has_spec) {
            // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
            load_progress_callback(0.0f, &load_progress_spec);
            load_progress_spec.t_last_load_progress_ms = 0;  // reset so internal cbs aren't delayed

            {
                common_params params_dft = common_base_params_to_speculative(params_base);

                // progress callback
                params_dft.load_progress_callback           = load_progress_callback;
                params_dft.load_progress_callback_user_data = &load_progress_spec;

                spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
                model_dft = spec_init->model();
                ctx_dft   = spec_init->context();

                if (has_draft && model_dft == nullptr) {
                    SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                    return false;
                }

                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create MTP context\n");
                    return false;
                }

                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft;
            }

            load_progress_callback(1.0f, &load_progress_spec);
        }

        if (has_mmproj) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {{"stage", "mmproj_model"}});
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;
        slot_cache_key_similarity = params_base.slot_cache_key_similarity;
        slot_cache_key_min_prefix = std::max<int32_t>(0, params_base.slot_cache_key_min_prefix);

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        int n_ctx_slot = llama_n_ctx_seq(ctx_tgt);
        if (n_ctx_slot > n_ctx_train) {
            SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - using rope scaling to extend\n", n_ctx_slot, n_ctx_train);
            // Do not cap: caller has configured rope scaling (--rope-scale / --rope-scaling yarn) to handle extended context.
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        const uint32_t n_seq_recr = llama_n_seq_recurrent(ctx_tgt);
        if (n_seq_recr > 0 && params_base.n_parallel > (int) n_seq_recr) {
            SRV_INF("capping n_parallel (%d -> %u) to match fitted recurrent state capacity\n",
                    params_base.n_parallel, n_seq_recr);
            params_base.n_parallel = (int) n_seq_recr;
        }

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n",
                params_base.n_parallel, n_ctx_slot, params_base.kv_unified ? "true" : "false");

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
        }

        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
        } else {
            spec_init.reset();
            ctx_dft   = nullptr;
            model_dft = nullptr;
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft;
            slot.mem.init(ctx_tgt, ctx_dft);
            if (token_hack_factory) {
                slot.token_hack = token_hack_factory(ctx_tgt);
            }
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot;

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.reset();
        }

        // RERoT runtime holder (OFF: stays null, every hook below no-ops).
        // Logical seq arena is LLAMA_MAX_SEQ; physical recurrent capacity
        // stays at n_parallel (see common.cpp wiring). Internal parked/
        // archive ids allocate from [n_parallel, LLAMA_MAX_SEQ).
        if (params_base.rerot_enabled) {
            llama_memory_t mem = ctx_tgt != nullptr ? llama_get_memory(ctx_tgt) : nullptr;
            rerot = std::make_unique<server_rerot_runtime>(
                mem, params_base.rerot_frontier,
                (uint32_t) std::max(0, params_base.n_parallel), (uint32_t) LLAMA_MAX_SEQ);
            rerot_episode_id = 0;
            SRV_INF("RERoT runtime armed (frontier=%s)\n",
                llama_rerot_frontier_mode_name(params_base.rerot_frontier));
        } else {
            rerot.reset();
            rerot_episode_id = 0;
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            const int32_t n_embd  = llama_model_n_embd_inp(model_tgt);
            batch.init(std::max(n_batch, params_base.n_parallel), n_embd);
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            const size_t cache_token_limit = params_base.kv_unified && params_base.n_ctx_kv > 0 ? 0 : n_ctx;
            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, cache_token_limit);
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task) {
            process_single_task(std::move(task));
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified && params_base.n_ctx_kv > 0) {
                    SRV_TRC("%s", "idle slots will remain in unified KV and move to prompt cache on memory pressure\n");
                } else if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* allow_video           */ mctx ? mtmd_helper_support_video(mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning");
                bool enabled = it != params_base.default_template_kwargs.end();
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cache_key(const std::string & cache_key) {
        if (cache_key.empty()) {
            return nullptr;
        }

        auto it = cache_key_slots.find(cache_key);
        if (it == cache_key_slots.end()) {
            return nullptr;
        }

        server_slot * slot = get_slot_by_id(it->second);
        if (slot == nullptr) {
            cache_key_slots.erase(it);
            return nullptr;
        }

        if (slot->prompt.tokens.empty()) {
            SLT_INF(*slot, "ignoring cache_key slot with empty prompt, key = %s\n", cache_key.c_str());
            cache_key_slots.erase(it);
            return nullptr;
        }

        if (slot->is_processing()) {
            SLT_INF(*slot, "ignoring busy cache_key slot, key = %s\n", cache_key.c_str());
            return nullptr;
        }

        return slot;
    }

    bool cache_key_slot_has_enough_similarity(const server_slot & slot, const server_task & task) const {
        if (slot.prompt.tokens.empty() || task.tokens.empty()) {
            SLT_INF(slot, "ignoring cache_key slot with empty prompt or task, key = %s\n", task.cache_key.c_str());
            return false;
        }

        const size_t n_common = slot.prompt.tokens.get_common_prefix(task.tokens);
        const float  sim_cur  = float(n_common) / task.tokens.size();
        const bool enough_prefix = n_common >= slot_cache_key_min_prefix;
        const bool enough_similarity = slot_cache_key_similarity <= 0.0f || sim_cur >= slot_cache_key_similarity;
        if (enough_prefix && enough_similarity) {
            SLT_INF(slot, "selected slot by cache_key, sim = %.3f (>= %.3f thold), common = %zu (>= %zu), key = %s\n",
                    sim_cur, slot_cache_key_similarity, n_common, slot_cache_key_min_prefix, task.cache_key.c_str());
            return true;
        }

        SLT_INF(slot, "ignoring cache_key slot, sim = %.3f (< %.3f thold) or common = %zu (< %zu), key = %s\n",
                sim_cur, slot_cache_key_similarity, n_common, slot_cache_key_min_prefix, task.cache_key.c_str());
        return false;
    }

    void clear_cache_keys_for_slot(int id_slot) {
        for (auto it = cache_key_slots.begin(); it != cache_key_slots.end(); ) {
            if (it->second == id_slot) {
                it = cache_key_slots.erase(it);
            } else {
                ++it;
            }
        }
    }

    void bind_cache_key_to_slot(const std::string & cache_key, int id_slot) {
        clear_cache_keys_for_slot(id_slot);

        if (!cache_key.empty()) {
            cache_key_slots[cache_key] = id_slot;
        }
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task, bool allow_prompt_similarity = true) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
            }
        }

        // find the slot that has at least n% prompt similarity
        if (allow_prompt_similarity && ret == nullptr && slot_prompt_similarity != 0.0f) {
            float f_sim_best = 0;

            for (server_slot & slot : slots) {
                if (task.id_slot != -1 && slot.id != task.id_slot) {
                    continue;
                }

                // skip the slot if it is not available
                if (slot.is_processing()) {
                    SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing());
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const size_t lcp_len = tokens.get_common_prefix(task.tokens);
                const float f_sim_cur = float(lcp_len) / task.tokens.size();

                SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

                // select the current slot if the criteria match
                if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                    f_sim_best = f_sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                if (task.id_slot == -1) {
                    SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                            f_sim_best, slot_prompt_similarity, f_keep);
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);

                update_cache = true;
            }
        }

        if (ret) {
            update_cache = update_cache || task.is_retry;
            update_cache = update_cache && prompt_cache;

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                ret->prompt_save(*prompt_cache);

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear();
                }

                prompt_cache->update();

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    bool try_clear_idle_slots() {
        if (!params_base.kv_unified) {
            return false;
        }

        server_slot * victim = nullptr;
        for (auto & slot : slots) {
            if (slot.is_processing() || slot.prompt.n_tokens() == 0) {
                continue;
            }
            // RERoT episode-aware idle demotion (§A.11.2): the episode is the
            // minimum reliable unit. Never demote a single Lane slot alone;
            // idle slots never belong to an active episode (lanes are
            // processing, queued children hold no physical slot), so this
            // only skips on a race. OFF: rerot_owns_slot false.
            if (rerot_owns_slot(slot.id)) {
                continue;
            }
            if (victim == nullptr || slot.t_last_used < victim->t_last_used) {
                victim = &slot;
            }
        }

        if (victim == nullptr) {
            return false;
        }

        SLT_DBG(*victim, "demoting idle KV, n_tokens = %zu\n", victim->prompt.tokens.size());
        if (prompt_cache && params_base.cache_idle_slots) {
            victim->prompt_save(*prompt_cache);
            prompt_cache->update();
        }
        victim->prompt_clear();
        clear_cache_keys_for_slot(victim->id);
        return true;
    }

    server_slot * find_preemption_victim(int32_t id_slot_protected, bool recurrent_pressure) {
        server_slot * victim = nullptr;
        uint64_t victim_kv_used = 0;
        size_t n_preemptible = 0;
        llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);
        llama_memory_t mem_dft = ctx_dft ? llama_get_memory(ctx_dft) : nullptr;

        for (auto & slot : slots) {
            if (slot.id == id_slot_protected || slot.state != SLOT_STATE_GENERATING || !slot.task) {
                continue;
            }
            // RERoT episode-aware active preemption (§A.11.1/A.27): never
            // leave an orphan Lane. A borrowed Lane slot is not independently
            // preemptible; the whole episode is the resource unit (freeze /
            // cancel all writers, release all exec/parked/pending handles
            // together via rerot_propagate_hard_abort or a whole-episode
            // retry). Skipping here forces the pressure planner to treat the
            // episode as one atomic victim. OFF: false.
            if (rerot_owns_slot(slot.id)) {
                continue;
            }
            ++n_preemptible;

            const bool has_recurrent_tgt = llama_memory_seq_get_recurrent_used(mem_tgt, slot.id) != 0;
            const bool has_recurrent_dft = mem_dft && mem_dft != mem_tgt &&
                llama_memory_seq_get_recurrent_used(mem_dft, slot.id) != 0;
            if (recurrent_pressure && !has_recurrent_tgt && !has_recurrent_dft) {
                continue;
            }

            uint64_t kv_used = llama_memory_seq_get_kv_used(mem_tgt, slot.id);
            if (mem_dft && mem_dft != mem_tgt) {
                kv_used += llama_memory_seq_get_kv_used(mem_dft, slot.id);
            }
            if (victim == nullptr || kv_used > victim_kv_used) {
                victim = &slot;
                victim_kv_used = kv_used;
            }
        }

        if (id_slot_protected < 0 && n_preemptible <= 1) {
            return nullptr;
        }

        return victim;
    }

    server_task make_retry_task(const server_slot & slot) const {
        GGML_ASSERT(slot.task);
        GGML_ASSERT(slot.state == SLOT_STATE_GENERATING);

        server_task retry(slot.task->type);
        retry.id        = slot.task->id;
        retry.index     = slot.task->index;
        retry.id_target = slot.task->id_target;
        retry.id_slot   = -1;
        retry.cache_key = slot.task->cache_key;
        retry.id_parent = -1;
        retry.params    = slot.task->params;
        if (retry.params.n_keep < 0) {
            retry.params.n_keep = slot.n_prompt_tokens_original;
        }
        retry.tokens    = slot.prompt.tokens.clone();
        retry.is_retry  = true;

        if (!slot.token_append_chunk.empty()) {
            GGML_ASSERT(slot.token_append_anchor != LLAMA_TOKEN_NULL);
            retry.tokens.push_back(slot.token_append_anchor);
            retry.tokens.insert(slot.token_append_chunk);
        } else if (slot.sampled != LLAMA_TOKEN_NULL) {
            retry.tokens.push_back(slot.sampled);
        }

        return retry;
    }

    bool preempt_slot(server_slot & slot) {
        if (!slot.task || slot.state != SLOT_STATE_GENERATING) {
            return false;
        }

        // RERoT: refuse single-Lane preemption (§A.11.1). Whole-episode
        // handling lives in rerot_propagate_hard_abort / future whole-episode
        // retry; preempting one Lane would orphan siblings and leak
        // parked/archive lineage. OFF: false.
        if (rerot_owns_slot(slot.id)) {
            SLT_WRN(slot, "%s", "refusing single-Lane preemption while a RERoT episode is active "
                "(whole-episode unit; use rerot_propagate_hard_abort)\n");
            return false;
        }

        const int id_task = slot.task->id;
        server_task retry = make_retry_task(slot);

        if (prompt_cache) {
            slot.prompt_save(*prompt_cache);
            prompt_cache->update();
        }

        server_retry_state state;
        state.n_decoded                      = slot.n_decoded;
        state.n_prompt_tokens_original       = slot.n_prompt_tokens_original;
        state.n_prompt_tokens_cache_response = slot.n_prompt_tokens_cache_response;
        state.n_prompt_tokens_processed      = slot.n_prompt_tokens_processed_total();
        state.response_prompt                = slot.response_prompt.empty() ? slot.task->tokens.clone() : std::move(slot.response_prompt);
        state.last_nl_pos              = slot.last_nl_pos;
        state.generated_text           = std::move(slot.generated_text);
        state.generated_tokens         = std::move(slot.generated_tokens);
        state.generated_token_probs    = std::move(slot.generated_token_probs);
        state.n_sent_text              = slot.n_sent_text;
        state.has_new_line             = slot.has_new_line;
        state.truncated                = slot.truncated;
        state.t_start_generation       = slot.t_start_generation;
        state.t_prompt_processing      = slot.t_prompt_processing;
        retry_states[id_task]          = std::move(state);

        SLT_WRN(slot, "preempting task %d, committed_kv = %d, retry_tokens = %d, n_decoded = %d\n",
                id_task, slot.prompt.n_tokens(), retry.n_tokens(), slot.n_decoded);

        slot.prompt_clear();
        clear_cache_keys_for_slot(slot.id);
        queue_tasks.defer(std::move(retry));
        slot.release(false);
        return true;
    }

    uint32_t estimate_next_kv_cells(int32_t & id_slot_protected) {
        id_slot_protected = -1;

        if (inference_mode == SERVER_INFERENCE_MODE_DECODE) {
            uint64_t required = 0;
            server_slot * slot_batched = nullptr;
            const int n_draft_cfg = common_speculative_n_max(&params_base.speculative);

            for (auto & slot : slots) {
                if (slot.state != SLOT_STATE_GENERATING) {
                    continue;
                }
                if (slot_batched == nullptr) {
                    slot_batched = &slot;
                } else if (!slot_batched->can_batch_with(slot)) {
                    continue;
                }

                uint32_t n = 1;
                if (slot.rerot_internal &&
                    server_rerot_private_microbatch(slot.rerot_injection) &&
                    slot.rerot_injection_cursor < slot.rerot_injection_tokens.size()) {
                    n = static_cast<uint32_t>(rerot_private_batch_size(slot));
                } else if (!slot.token_append_chunk.empty()) {
                    n += slot.token_append_chunk.size();
                } else if (slot.can_speculate() && n_draft_cfg > 0) {
                    n += std::min(n_draft_cfg, std::max(0, slot.get_n_draft_max()));
                }
                required += n;
            }

            return (uint32_t) std::min<uint64_t>(required, UINT32_MAX);
        }

        server_slot * slot = id_slot_prefill >= 0 ? get_slot_by_id(id_slot_prefill) : nullptr;
        if (slot == nullptr || !slot->task || !slot->is_processing()) {
            return 0;
        }

        id_slot_protected = slot->id;

        size_t n_past = slot->prompt.n_tokens();
        if (slot->state == SLOT_STATE_STARTED) {
            n_past = (slot->task->params.cache_prompt || slot->task->is_retry) ? slot->prompt.tokens.get_common_prefix(slot->task->tokens) : 0;
        }

        const size_t n_task_tokens = slot->task->n_tokens();
        const size_t n_remaining = n_task_tokens > n_past ? n_task_tokens - n_past : 0;
        return (uint32_t) std::min<size_t>(n_remaining, llama_n_batch(ctx_tgt));
    }

    uint32_t estimate_next_recurrent_cells(llama_memory_t mem, bool draft_only) {
        if (inference_mode == SERVER_INFERENCE_MODE_DECODE) {
            uint32_t required = 0;
            server_slot * slot_batched = nullptr;
            for (auto & slot : slots) {
                if (slot.state != SLOT_STATE_GENERATING || (draft_only && !slot.can_speculate())) {
                    continue;
                }
                if (slot_batched == nullptr) {
                    slot_batched = &slot;
                } else if (!slot_batched->can_batch_with(slot)) {
                    continue;
                }
                required += llama_memory_seq_get_recurrent_used(mem, slot.id) == 0 ? 1 : 0;
            }
            return required;
        }

        if (draft_only) {
            return 0;
        }

        server_slot * slot = id_slot_prefill >= 0 ? get_slot_by_id(id_slot_prefill) : nullptr;
        if (slot == nullptr || !slot->task || !slot->is_processing()) {
            return 0;
        }
        return llama_memory_seq_get_recurrent_used(mem, slot->id) == 0 ? 1 : 0;
    }

    bool ensure_next_kv_capacity() {
        kv_batch_limit = UINT32_MAX;

        if (!params_base.kv_unified || params_base.n_ctx_kv == 0) {
            return true;
        }

        // One pressure-planning call can iterate several times while idle
        // slots are cleared or victims are preempted. Count each fallback
        // reason once per planner invocation rather than once per loop pass.
        bool counted_kv_fallback = false;
        bool counted_recurrent_fallback = false;

        while (true) {
            llama_memory_kv_usage usage_tgt = {};
            llama_memory_kv_usage usage_dft = {};
            llama_memory_kv_usage usage_recurrent_tgt = {};
            llama_memory_kv_usage usage_recurrent_dft = {};
            llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);
            llama_memory_t mem_dft = ctx_dft ? llama_get_memory(ctx_dft) : nullptr;
            const bool has_tgt_usage = llama_memory_get_kv_usage(mem_tgt, &usage_tgt);
            const bool has_dft_usage = mem_dft && mem_dft != mem_tgt && llama_memory_get_kv_usage(mem_dft, &usage_dft);
            const bool has_recurrent_tgt = llama_memory_get_recurrent_usage(mem_tgt, &usage_recurrent_tgt);
            const bool has_recurrent_dft = mem_dft && mem_dft != mem_tgt &&
                llama_memory_get_recurrent_usage(mem_dft, &usage_recurrent_dft);
            if (!has_tgt_usage && !has_dft_usage && !has_recurrent_tgt && !has_recurrent_dft) {
                return true;
            }

            int32_t id_slot_protected = -1;
            const uint32_t required_kv = estimate_next_kv_cells(id_slot_protected);
            const uint32_t required_recurrent_tgt = has_recurrent_tgt ? estimate_next_recurrent_cells(mem_tgt, false) : 0;
            const uint32_t required_recurrent_dft = has_recurrent_dft ? estimate_next_recurrent_cells(mem_dft, true) : 0;
            const uint64_t available_tgt = has_tgt_usage ? usage_tgt.capacity - std::min(usage_tgt.capacity, usage_tgt.used) : UINT64_MAX;
            const uint64_t available_dft = has_dft_usage ? usage_dft.capacity - std::min(usage_dft.capacity, usage_dft.used) : UINT64_MAX;
            const uint64_t available_recurrent_tgt = has_recurrent_tgt ? usage_recurrent_tgt.capacity - std::min(usage_recurrent_tgt.capacity, usage_recurrent_tgt.used) : UINT64_MAX;
            const uint64_t available_recurrent_dft = has_recurrent_dft ? usage_recurrent_dft.capacity - std::min(usage_recurrent_dft.capacity, usage_recurrent_dft.used) : UINT64_MAX;

            const bool kv_pressure = required_kv > available_tgt || required_kv > available_dft;
            const bool recurrent_pressure = required_recurrent_tgt > available_recurrent_tgt ||
                                            required_recurrent_dft > available_recurrent_dft;
            const uint32_t kv_deficit_tgt = has_tgt_usage && required_kv > available_tgt
                ? (uint32_t) std::min<uint64_t>(UINT32_MAX, (uint64_t) required_kv - available_tgt)
                : 0;
            const uint32_t kv_deficit_dft = has_dft_usage && required_kv > available_dft
                ? (uint32_t) std::min<uint64_t>(UINT32_MAX, (uint64_t) required_kv - available_dft)
                : 0;

            bool tri_maintenance_due = false;
            if (params_base.triattention_enabled && !kv_pressure) {
                const auto * maintenance_episode = rerot_active_episode();
                const auto maintenance_transport_it = maintenance_episode
                    ? rerot_transport.find(maintenance_episode->id)
                    : rerot_transport.end();
                if (maintenance_episode &&
                    maintenance_transport_it != rerot_transport.end() &&
                    maintenance_transport_it->second->triattention_compressed) {
                    const uint64_t synced =
                        maintenance_transport_it->second->triattention_synced_public_tokens;
                    const uint64_t current =
                        maintenance_episode->generated_public_tokens;
                    if (current > synced) {
                        const uint64_t base = static_cast<uint64_t>(
                            std::max<llama_pos>(0, maintenance_episode->base_prefix_end));
                        const uint64_t old_target = std::max<uint64_t>(
                            128,
                            static_cast<uint64_t>(std::ceil(
                                double(base + synced) * params_base.triattention_ratio)));
                        const uint64_t new_target = std::max<uint64_t>(
                            128,
                            static_cast<uint64_t>(std::ceil(
                                double(base + current) * params_base.triattention_ratio)));
                        const uint64_t target_growth = new_target - old_target;
                        tri_maintenance_due =
                            current - synced > target_growth + 128;
                    }
                }
                for (const auto & slot : slots) {
                    if (!slot.is_processing() || !slot.triattention_compressed ||
                        slot.rerot_internal) {
                        continue;
                    }

                    const uint32_t logical_tokens = (uint32_t) slot.prompt.n_tokens();
                    const uint32_t target = std::max<uint32_t>(128, (uint32_t) std::ceil((double) logical_tokens * params_base.triattention_ratio));
                    const uint32_t resident = llama_memory_seq_get_kv_used(mem_tgt, slot.id);
                    if (resident > target + 128) {
                        tri_maintenance_due = true;
                        break;
                    }
                }
            }

            if (!kv_pressure && !recurrent_pressure && !tri_maintenance_due) {
                return true;
            }

            // TriAttention only reclaims KV. Recurrent-only pressure must go directly
            // to the existing recurrent fallback path.
            if (params_base.triattention_enabled && (kv_pressure || tri_maintenance_due)) {
                // Initial pressure drains every resident sequence. Maintenance only
                // touches sequences that have already entered compressed mode.
                std::vector<llama_memory_kv_reclaim_seq_hint> seq_hints;
                std::vector<llama_seq_id> rerot_semantic_seq_ids;
                const auto * rerot_episode = rerot_active_episode();
                if (rerot_episode && !rerot->sync_public_archive(
                        rerot_episode->id, &rerot_semantic_seq_ids)) {
                    rerot->hard_abort(
                        rerot_episode->id,
                        "rerot_state_error: failed to synchronize PUBLIC archive before TriAttention reclaim");
                    rerot_propagate_hard_abort();
                    return false;
                }
                for (auto & slot : slots) {
                    if (slot.prompt.n_tokens() == 0 ||
                        (rerot_episode && slot.rerot_internal &&
                         slot.rerot_episode_id == rerot_episode->id) ||
                        (!kv_pressure && !slot.triattention_compressed)) {
                        continue;
                    }
                    llama_memory_kv_reclaim_seq_hint hint;
                    hint.seq_id = slot.id;
                    hint.logical_tokens = (uint32_t) slot.prompt.n_tokens();
                    hint.tail_guard = 128; // recent window
                    hint.eligible = true;
                    seq_hints.push_back(hint);
                }
                const auto * active_episode = rerot_active_episode();
                const auto transport_it = active_episode
                    ? rerot_transport.find(active_episode->id)
                    : rerot_transport.end();
                if (active_episode && active_episode->archive_seq >= 0 &&
                    transport_it != rerot_transport.end() &&
                    (kv_pressure || transport_it->second->triattention_compressed) &&
                    llama_memory_seq_get_kv_used(mem_tgt, active_episode->archive_seq) > 0) {
                    const uint64_t logical_u64 =
                        uint64_t(std::max<llama_pos>(0, active_episode->base_prefix_end)) +
                        active_episode->generated_public_tokens;
                    llama_memory_kv_reclaim_seq_hint hint;
                    hint.seq_id = active_episode->archive_seq;
                    hint.logical_tokens = static_cast<uint32_t>(
                        std::min<uint64_t>(UINT32_MAX, logical_u64));
                    hint.tail_guard = 128;
                    hint.eligible = true;
                    hint.semantic_episode_id = active_episode->id;
                    hint.semantic_seq_ids = rerot_semantic_seq_ids;
                    seq_hints.push_back(std::move(hint));
                }

                llama_memory_kv_reclaim_request req;
                // required_free is a deficit, not the whole incoming batch.
                // drain_to_floor still makes the initial pressure event perform
                // the configured-ratio drain regardless of the deficit size.
                req.required_free = kv_deficit_tgt;
                req.drain_to_floor = true;
                req.seq_hints = std::move(seq_hints);

                // Context-level reclaim synchronizes pending compute before any
                // physical K/V movement and invalidates graph reservations after.
                auto result_tgt = llama_context_reclaim_kv(ctx_tgt, &req);
                if (result_tgt.supported && result_tgt.changed) {
                    if (kv_pressure) {
                        tri_drain_count++;
                    } else {
                        tri_maintenance_count++;
                    }
                    tri_cells_before       = result_tgt.physical_before;
                    tri_cells_after        = result_tgt.physical_after;
                    tri_cells_freed        = result_tgt.physical_freed;
                    tri_references_removed = result_tgt.references_removed;
                    tri_target_references  = result_tgt.target_references;
                    tri_hard_keep          = result_tgt.hard_keep;
                    tri_shared_keep        = result_tgt.shared_keep;
                    tri_cells_freed_total += result_tgt.physical_freed;
                    tri_score_us_total    += result_tgt.score_us;
                    tri_pack_us_total     += result_tgt.pack_us;

                    if (rerot_episode) {
                        const auto sync_transport_it =
                            rerot_transport.find(rerot_episode->id);
                        if (sync_transport_it != rerot_transport.end()) {
                            sync_transport_it->second->triattention_synced_public_tokens =
                                rerot_episode->generated_public_tokens;
                        }
                    }

                    SRV_INF("TriAttention %s: before=%u after=%u freed=%u refs_removed=%u deficit=%u target_refs=%u hard_keep=%u shared_keep=%u score_ms=%.3f pack_ms=%.3f floor_reached=%s\n",
                            kv_pressure ? "drain" : "maintenance",
                            result_tgt.physical_before, result_tgt.physical_after, result_tgt.physical_freed,
                            result_tgt.references_removed, kv_deficit_tgt,
                            result_tgt.target_references, result_tgt.hard_keep,
                            result_tgt.shared_keep,
                            result_tgt.score_us / 1000.0, result_tgt.pack_us / 1000.0,
                            result_tgt.floor_reached ? "true" : "false");

                    if (kv_pressure) {
                        for (auto & slot : slots) {
                            if (slot.prompt.n_tokens() > 0 &&
                                llama_memory_seq_get_kv_used(mem_tgt, slot.id) > 0) {
                                slot.triattention_compressed = true;
                            }
                        }
                        const auto * active_episode = rerot_active_episode();
                        const auto transport_it = active_episode
                            ? rerot_transport.find(active_episode->id)
                            : rerot_transport.end();
                        if (active_episode && transport_it != rerot_transport.end()) {
                            for (const auto & slot : slots) {
                                if (slot.rerot_internal &&
                                    slot.rerot_episode_id == active_episode->id &&
                                    slot.triattention_compressed) {
                                    transport_it->second->triattention_compressed = true;
                                    break;
                                }
                            }
                        }
                        if (active_episode && active_episode->archive_seq >= 0 &&
                            transport_it != rerot_transport.end() &&
                            llama_memory_seq_get_kv_used(mem_tgt, active_episode->archive_seq) > 0) {
                            transport_it->second->triattention_compressed = true;
                        }
                    }

                    // Also reclaim on draft if separate
                    if (has_dft_usage && mem_dft != mem_tgt) {
                        llama_memory_kv_reclaim_request req_dft = req;
                        req_dft.required_free = kv_deficit_dft;
                        auto result_dft = llama_context_reclaim_kv(ctx_dft, &req_dft);
                        if (result_dft.supported && result_dft.changed) {
                            tri_cells_freed_total += result_dft.physical_freed;
                            tri_score_us_total    += result_dft.score_us;
                            tri_pack_us_total     += result_dft.pack_us;
                        }
                    }
                    // Re-check capacity by continuing the while loop
                    continue;
                }

                if (kv_pressure && result_tgt.supported) {
                    if ((!result_tgt.changed || (result_tgt.floor_reached && !result_tgt.capacity_satisfied)) &&
                        !counted_kv_fallback) {
                        tri_floor_exhausted_count++;
                        tri_atomic_fallback_kv_count++;
                        counted_kv_fallback = true;
                    }
                    SRV_INF("TriAttention KV floor exhausted: used=%u deficit=%u target_refs=%u hard_keep=%u shared_keep=%u; entering atomic fallback\n",
                            result_tgt.physical_after, kv_deficit_tgt, result_tgt.target_references,
                            result_tgt.hard_keep, result_tgt.shared_keep);
                } else if (kv_pressure && !result_tgt.supported) {
                    SRV_ERR("%s\n", "TriAttention enabled but target KV memory does not support reclaim; entering atomic fallback");
                }
            }

            if (!kv_pressure && !recurrent_pressure) {
                return true;
            }

            if (params_base.triattention_enabled && recurrent_pressure && !counted_recurrent_fallback) {
                tri_atomic_fallback_recurrent_count++;
                counted_recurrent_fallback = true;
                SRV_INF("%s\n", "recurrent pressure: entering atomic fallback without TriAttention KV reclaim");
            }

            if (try_clear_idle_slots()) {
                continue;
            }

            server_slot * victim = find_preemption_victim(id_slot_protected, recurrent_pressure);
            if (victim == nullptr) {
                SRV_DBG("dynamic memory needs kv=%u recurrent_tgt=%u recurrent_dft=%u, target available=%" PRIu64 ", draft available=%" PRIu64 ", recurrent_tgt available=%" PRIu64 ", recurrent_dft available=%" PRIu64 "\n",
                        required_kv, required_recurrent_tgt, required_recurrent_dft,
                        available_tgt, available_dft, available_recurrent_tgt, available_recurrent_dft);

                // The current prefill slot is protected from preemption. Process only
                // the cells that fit, then re-enter the pressure planner before adding
                // more tokens. This avoids exhausting the cache inside decode's
                // smaller-batch retry loop.
                if (kv_pressure && id_slot_protected >= 0) {
                    const uint64_t available = std::min(available_tgt, available_dft);
                    kv_batch_limit = (uint32_t) std::min<uint64_t>(available, llama_n_batch(ctx_tgt));
                    if (kv_batch_limit > 0) {
                        SRV_INF("KV floor exhausted during prefill; limiting next batch to %u cells\n", kv_batch_limit);
                    }
                }
                return true;
            }

            if (!preempt_slot(*victim)) {
                return false;
            }
        }
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(
                        model_tgt, task.params.sampling, (int32_t) llama_n_ctx(ctx_tgt)));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool backend_sampling = true;

            backend_sampling &= task.params.sampling.backend_sampling;

            // TODO: speculative decoding requires multiple samples per batch - not supported yet
            backend_sampling &= !(slot.can_speculate());

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            backend_sampling &= !need_pre_sample_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());
        } else {
            slot.smpl.reset();
        }

        slot.task = std::make_unique<const server_task>(std::move(task));

        auto retry_it = retry_states.find(slot.task->id);
        if (retry_it != retry_states.end()) {
            auto & retry = retry_it->second;
            slot.n_decoded_start                  = retry.n_decoded;
            slot.n_prompt_tokens_original         = retry.n_prompt_tokens_original;
            slot.n_prompt_tokens_cache_response   = retry.n_prompt_tokens_cache_response;
            slot.n_prompt_tokens_processed_accum  = retry.n_prompt_tokens_processed;
            slot.response_prompt                  = std::move(retry.response_prompt);
            slot.last_nl_pos             = retry.last_nl_pos;
            slot.generated_text          = std::move(retry.generated_text);
            slot.generated_tokens        = std::move(retry.generated_tokens);
            slot.generated_token_probs   = std::move(retry.generated_token_probs);
            slot.n_sent_text             = retry.n_sent_text;
            slot.has_new_line            = retry.has_new_line;
            slot.truncated               = retry.truncated;
            slot.t_start_generation      = retry.t_start_generation;
            slot.t_prompt_processing_accum = retry.t_prompt_processing;
            retry_states.erase(retry_it);
        } else {
            slot.n_decoded_start                 = 0;
            slot.n_prompt_tokens_original        = slot.task->n_tokens();
            slot.n_prompt_tokens_cache_response  = 0;
            slot.n_prompt_tokens_processed_accum = 0;
            slot.response_prompt.clear();
            slot.t_start_generation              = 0;
            slot.t_prompt_processing_accum       = 0.0;
        }

        slot.n_decoded   = slot.n_decoded_start;
        slot.n_remaining = -1;

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        bind_cache_key_to_slot(slot.task->cache_key, slot.id);

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_decoded = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), slot.n_decoded, slot.n_ctx);
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params_base)) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n", slot.n_decoded, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_decoded = %d, n_indent = %d\n", slot.n_decoded, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && (ggml_time_us() - slot.t_start_generation > 1000.0f*slot.task->params.t_max_predict_ms)) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n", slot.n_decoded, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n", slot.n_decoded, slot.n_remaining, result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx, n_probs_request);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    // Gate slot save/restore/erase on slot content (does it hold media),
    // not model capability: a multimodal model may hold a pure-text slot.
    bool check_slot_no_media(const server_slot & slot, const int id_task) {
        if (slot.prompt.tokens.has_media()) {
            send_error(id_task,
                "This operation is not supported while the slot holds image/audio tokens (a pure-text prefix is supported)",
                ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }
        return true;
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.n_prompt_tokens_cache;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = (ggml_time_us() - slot.t_start_process_prompt) / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.n_prompt_tokens_original;
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache_response;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->timings = slot.get_timings();
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->timings         = slot.get_timings();
        res->prompt          = (slot.response_prompt.empty() ? slot.task->tokens : slot.response_prompt).detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.n_prompt_tokens_original;
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache_response;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        const int id_task = slot.task->id;

        // evict checkpoints within min-step of a previous checkpoint, unless they were
        // created by the current task
        int64_t last = -1;
        for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
            if (it->id_task != id_task && last >= 0 && it->n_tokens <= last + params_base.checkpoint_min_step) {
                SLT_TRC(slot, "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        it->pos_min, it->pos_max, it->n_tokens, (float) it->size() / 1024 / 1024);

                it = slot.prompt.checkpoints.erase(it);
                continue;
            }

            last = it->n_tokens;
            ++it;
        }

        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        auto & cur = slot.prompt.checkpoints.emplace_back();

        cur.id_task = id_task;

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        cur.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        cur.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        // stash the draft's speculative state with the checkpoint
        common_speculative_get_state(spec.get(), slot.id, cur.data_spec);

        SLT_TRC(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);
    }

    void process_single_task(server_task && task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }
                    if (task.params.rerot_effective(task.type) &&
                        task.rerot_episode_id == 0) {
                        uint64_t generation = rerot_task_generation_next++;
                        if (generation == 0) {
                            generation = rerot_task_generation_next++;
                        }
                        task.assign_rerot_episodes(
                            rerot_task_episode_id_next, generation);
                    } else if (task.params.rerot_effective(task.type) &&
                               task.is_retry) {
                        task.rerot_bump_generation();
                    }

                    const int id_slot = task.id_slot;
                    const int id_task = task.id;

                    server_slot * slot = nullptr;
                    if (id_slot != -1) {
                        slot = get_slot_by_id(id_slot);
                    } else if (!task.cache_key.empty()) {
                        server_slot * slot_cache_key = get_slot_by_cache_key(task.cache_key);
                        if (slot_cache_key != nullptr && cache_key_slot_has_enough_similarity(*slot_cache_key, task)) {
                            slot = slot_cache_key;
                        }
                    }
                    if (slot == nullptr) {
                        slot = get_available_slot(task, task.cache_key.empty());
                    }

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots && params_base.n_ctx_kv == 0) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                if (slot.prompt_save(*prompt_cache)) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                if (params_base.kv_unified && params_base.n_ctx_kv == 0) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    slot.prompt_clear();
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    retry_states.erase(task.id_target);

                    // A request owns the whole RERoT episode, not one cloned
                    // Lane slot. Cancel every writer and release archive/
                    // parked state atomically; releasing only the first clone
                    // would orphan siblings under the same response task id.
                    uint64_t cancelled_episode = 0;
                    for (const auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target &&
                            slot.rerot_internal) {
                            cancelled_episode = slot.rerot_episode_id;
                            break;
                        }
                    }
                    if (cancelled_episode != 0 && rerot &&
                        rerot->episode(cancelled_episode)) {
                        rerot->hard_abort(cancelled_episode, "rerot_cancelled");
                        for (auto & slot : slots) {
                            if (slot.rerot_internal &&
                                slot.rerot_episode_id == cancelled_episode) {
                                slot.prompt.clear();
                                slot.release();
                            }
                        }
                        rerot_erase_episode(cancelled_episode);
                        rerot_start_next_waiting_root();
                        break;
                    }

                    // Ordinary requests and queued (not-yet-started) RERoT
                    // roots retain the stock one-task/one-slot cancellation.
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %s on unknown completion id=%s, no live slot\n",
                                task.params.control_action.c_str(), task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    json slots_data = json::array();

                    int n_idle_slots       = 0;
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        json slot_data = slot.to_json(slots_debug == 0);

                        if (slot.is_processing()) {
                            n_processing_slots++;
                        } else {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot_data);
                    }
                    SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->slots_data          = std::move(slots_data);
                    res->n_idle_slots        = n_idle_slots;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->t_start             = metrics.t_start;

                    res->n_prompt_tokens_processed_total = metrics.n_prompt_tokens_processed_total;
                    res->t_prompt_processing_total       = metrics.t_prompt_processing_total;
                    res->n_tokens_predicted_total        = metrics.n_tokens_predicted_total;
                    res->t_tokens_generation_total       = metrics.t_tokens_generation_total;

                    res->n_tokens_max = metrics.n_tokens_max;

                    res->n_prompt_tokens_processed = metrics.n_prompt_tokens_processed;
                    res->t_prompt_processing       = metrics.t_prompt_processing;
                    res->n_tokens_predicted        = metrics.n_tokens_predicted;
                    res->t_tokens_generation       = metrics.t_tokens_generation;

                    res->n_decode_total          = metrics.n_decode_total;
                    res->n_busy_slots_total      = metrics.n_busy_slots_total;

                    res->tri_drain_total              = tri_drain_count;
                    res->tri_maintenance_total        = tri_maintenance_count;
                    res->tri_floor_exhausted_total    = tri_floor_exhausted_count;
                    res->tri_atomic_fallback_kv_total = tri_atomic_fallback_kv_count;
                    res->tri_atomic_fallback_recurrent_total = tri_atomic_fallback_recurrent_count;
                    res->tri_cells_freed_total        = tri_cells_freed_total;
                    res->tri_score_us_total           = tri_score_us_total;
                    res->tri_pack_us_total            = tri_pack_us_total;

                    res->tri_cells_before       = tri_cells_before;
                    res->tri_cells_after        = tri_cells_after;
                    res->tri_cells_freed        = tri_cells_freed;
                    res->tri_references_removed = tri_references_removed;
                    res->tri_target_references  = tri_target_references;
                    res->tri_hard_keep          = tri_hard_keep;
                    res->tri_shared_keep        = tri_shared_keep;
                    res->rerot                  = rerot_metrics;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (!check_slot_no_media(*slot, task.id)) {
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // RERoT v1 refusal (§§25,A.8): no per-slot persist of an
                    // active episode Lane (tree/run/epoch/parked/archive/MTP
                    // lineage has no versioned format yet). OFF: false.
                    if (rerot_owns_slot(id_slot)) {
                        send_error(task, "RERoT episode active: manual slot save is unsupported in v1 "
                            "(no versioned RERoT state with REROT_STATE_MAGIC/VERSION + fingerprint)", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    const llama_tokens tokens = slot->prompt.tokens.get_text_tokens();
                    const size_t token_count = tokens.size();
                    const size_t nwrite = llama_state_seq_save_file(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), token_count);

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = token_count;
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // RERoT v1 refusal (§§25,A.8): no manual restore into an
                    // active episode Lane; future restores must match
                    // REROT_STATE_MAGIC/VERSION + model/rope/Tri fingerprint.
                    // OFF: false.
                    if (rerot_owns_slot(id_slot)) {
                        send_error(task, "RERoT episode active: manual slot restore is unsupported in v1", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    llama_tokens tokens;
                    tokens.resize(slot->n_ctx);
                    size_t token_count = 0;
                    size_t nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), tokens.size(), &token_count);
                    if (nread == 0) {
                        slot->prompt.clear(); // KV may already been invalidated?
                        send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    tokens.resize(token_count);
                    slot->prompt.clear();
                    slot->prompt.tokens.insert(tokens);
                    slot->triattention_compressed =
                        llama_memory_seq_get_kv_used(llama_get_memory(ctx_tgt), slot->id) < token_count;

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = token_count;
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    // Gate on slot content, consistent with save/restore.
                    if (!check_slot_no_media(*slot, task.id)) {
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear();

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }
    }

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                if (slot.rerot_internal && rerot &&
                    rerot->episode(slot.rerot_episode_id)) {
                    rerot->hard_abort(
                        slot.rerot_episode_id,
                        std::string("rerot_backend_error: ") + e.what());
                    rerot_propagate_hard_abort();
                } else {
                    send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                    slot.release();
                }
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                if (slot->rerot_internal && rerot &&
                    rerot->episode(slot->rerot_episode_id)) {
                    rerot->hard_abort(
                        slot->rerot_episode_id,
                        std::string("rerot_backend_error: ") + e.what());
                    rerot_propagate_hard_abort();
                } else {
                    send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                    slot->release();
                }
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        retry_states.clear();

        // RERoT hard-abort propagation (§20): a fatal scheduler error while
        // an episode is active must not leave parked/archive lineage or a
        // context episode gate behind. OFF: rerot_active false, no change.
        if (rerot_active()) {
            rerot_propagate_hard_abort();
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    bool is_prefill_slot(const server_slot & slot) const {
        return slot.state == SLOT_STATE_STARTED || slot.state == SLOT_STATE_PROCESSING_PROMPT;
    }

    server_inference_mode select_inference_mode() {
        if (id_slot_prefill >= 0 && id_slot_prefill < (int32_t) slots.size()) {
            if (is_prefill_slot(slots[id_slot_prefill])) {
                return SERVER_INFERENCE_MODE_PREFILL;
            }
        }

        id_slot_prefill = -1;
        for (auto & slot : slots) {
            if (is_prefill_slot(slot)) {
                id_slot_prefill = slot.id;
                return SERVER_INFERENCE_MODE_PREFILL;
            }
        }

        return SERVER_INFERENCE_MODE_DECODE;
    }

    void update_inference_mode() {
        const auto new_mode = select_inference_mode();
        if (new_mode == inference_mode) {
            return;
        }

        llama_synchronize(ctx_tgt);
        if (ctx_dft) {
            llama_synchronize(ctx_dft);
        }

        inference_mode = new_mode;
        SRV_DBG("inference mode = %s, prefill slot = %d\n",
                inference_mode == SERVER_INFERENCE_MODE_PREFILL ? "prefill" : "decode", id_slot_prefill);
    }

    void update_slots() {
#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");
                return; // skip further processing

            } else {
                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        update_inference_mode();

        // RERoT frontier entry (§§19.3,20): one frontier is one atomic budget
        // unit; when hard limits are crossed the whole episode HARD_ABORTs
        // here instead of running a partial Lane subset. OFF: no-op true.
        if (!rerot_frontier_enter()) {
            return;
        }

        if (!ensure_next_kv_capacity()) {
            return;
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            pre_decode();
            batch.render();
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));
        }

        GGML_ASSERT(batch.slot_batched || batch.size() == 0);

        if (batch.slot_batched) {
            auto & slot_batched      = batch.slot_batched;
            auto & alora_scale       = batch.alora_scale;
            auto & alora_disabled_id = batch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        llama_batch batch_view;
        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(ctx_tgt);
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                batch_view = batch.get_view(off, n_tokens);
                bool ok = decode(n_batch, off, batch_view);
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(n_tokens, off, batch_view);
            } catch (const std::exception & e) {
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }
        }

        // RERoT barrier/fence poll (Stage 7, §§17,21): all gates inside are
        // episode-active checks, so OFF costs one null + one zero compare.
        // The full per-token frontier loop (plan/commit/finish_frontier) will
        // drive these flags; this poll only ensures a pending barrier still
        // refreshes read-only logits and a finalized single survivor still
        // takes the acquire fence + serial-tail transition even if the loop
        // lands in a later slice.
        if (rerot_active()) {
            if (const server_rerot_episode * ep = rerot_active_episode()) {
                if (ep->topology_barrier_pending && !ep->hard_aborted && !ep->serial_tail) {
                    rerot_on_topology_barrier(true);
                }
                if (ep->finalizing && !ep->hard_aborted && !ep->serial_tail &&
                    ep->ready_queue.empty() && ep->starting.empty() && ep->running.size() == 1) {
                    rerot_on_final_fence(*ep->running.begin());
                }
                if (ep->hard_aborted) {
                    rerot_propagate_hard_abort();
                }
            }
        }
    }

    void pre_decode() {
        // apply context-shift if needed
        // TODO: simplify and improve
        iterate(slots, [&](server_slot & slot) {
            if (inference_mode != SERVER_INFERENCE_MODE_DECODE) {
                return;
            }

            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // RERoT v1 refusal (§§25,A.9): no linear seq_rm/seq_add shift
                // on an active episode Lane. Shared history needs episode-aware
                // semantic truncation (oldest PUBLIC region dropped for all
                // readers + run-table repack + layout/publish bump + MTP
                // invalidation, recurrent state never rewritten). OFF: false.
                if (rerot_owns_slot(slot.id)) {
                    rerot->hard_abort(
                        slot.rerot_episode_id,
                        "rerot_resource_exhausted: active episode reached context-shift boundary");
                    rerot_propagate_hard_abort();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                slot.mem.seq_rm (slot.id, n_keep            , n_keep + n_discard);
                slot.mem.seq_add(slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        });

        // start populating the batch for this iteration
        batch.clear();

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // determine which slots are generating and drafting
        iterate(slots, [&](server_slot & slot) {
            if (inference_mode != SERVER_INFERENCE_MODE_DECODE) {
                return;
            }

            if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;

                if (!slot.token_append_chunk.empty()) {
                    return;
                }

                // RERoT MTP compat (§A.6): drafts bind to the reader view
                // stamp (topology/publish/layout). While a Lane borrows this
                // slot, a stale draft (peer PUBLIC commit / topology or
                // layout change since draft time) must be invalidated with
                // checkpoint restore before any new draft. v1 prototype also
                // forces n_draft_max=0 for episode Lanes (frontier-local
                // speculative only); the stale check below stays so the
                // epoch-aware path is already wired when drafting re-enables.
                // OFF: rerot_owns_slot false, no behavior change.
                if (rerot_owns_slot(slot.id)) {
                    if (slot.rerot_has_draft_stamp && ctx_tgt != nullptr &&
                        llama_rerot_mtp_is_stale(ctx_tgt, slot.id, &slot.rerot_draft_stamp)) {
                        // Stale: restore checkpoint, drop uncommitted draft
                        // KV/recurrent, re-draft from the latest view next
                        // frontier (here: just drop; next frontier re-drafts).
                        slot.spec_ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        if (slot.ctx_dft) {
                            slot.spec_ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }
                        common_speculative_set_state(spec.get(), slot.id, slot.spec_ckpt.data_spec);
                        slot.mem.seq_rm(slot.id, slot.spec_ckpt.pos_max + 1, -1);
                        slot.spec_draft.clear();
                        slot.spec_i_batch.clear();
                        slot.rerot_has_draft_stamp = false;
                    }
                    // v1: no cross-frontier speculative window for RERoT lanes.
                    return;
                }

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.get_n_draft_max();

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }
                        slot.spec_ckpt.data_spec.clear();
                        common_speculative_get_state(spec.get(), slot.id, slot.spec_ckpt.data_spec);

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .n_past   = */ slot.prompt.n_tokens(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        });

        // generate the actual drafts (if any)
        if (inference_mode == SERVER_INFERENCE_MODE_DECODE) {
            common_speculative_draft(spec.get());
        }

        // make checkpoints if needed
        iterate(drafting, [&](server_slot & slot) {
            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.n_draft_total += draft.size();

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }

                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, ckpt.pos_max + 1, -1)) {
                    GGML_ABORT("failed to remove sequence %d\n", slot.id);
                }
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
        });

        // Plan every Lane first, then install every reader view. This
        // two-phase ordering is required for strong-frontier visibility: the
        // first reader's PAC-DFS order must already contain peer runs created
        // later in the same frontier.
        bool rerot_plan_ok = true;
        iterate(generating, [&](server_slot & slot) {
            if (slot.rerot_internal && !rerot_plan_next_token(slot)) {
                rerot_plan_ok = false;
            }
        });
        if (!rerot_plan_ok) {
            rerot_propagate_hard_abort();
            return;
        }

        iterate(generating, [&](server_slot & slot) {
            if (slot.rerot_internal) {
                if (!rerot_append_planned_token(slot)) {
                    rerot_plan_ok = false;
                }
            } else {
                slot.handle_last_sampled_token(batch);
            }
        });
        if (!rerot_plan_ok) {
            rerot_propagate_hard_abort();
            return;
        }

        // process in chunks of params.n_batch, capped when the pressure planner
        // can only fit a partial prefill batch at the TriAttention floor.
        int32_t n_batch  = (int32_t) std::min<uint32_t>(llama_n_batch(ctx_tgt), kv_batch_limit);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            iterate(slots, [&](server_slot & slot) {
                if (inference_mode != SERVER_INFERENCE_MODE_PREFILL || slot.id != id_slot_prefill) {
                    return;
                }

                if (!add_ok || batch.size() >= n_batch) {
                    return; // batch is full, skip remaining slots
                }

                if (!slot.is_processing()) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    return;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.t_start_process_prompt = ggml_time_us();
                        if (slot.n_decoded_start == 0) {
                            slot.t_start_generation = 0;
                        }

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            if (slot.task->params.cache_prompt || slot.task->is_retry) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                // RERoT compat (§A.10.3): ordinary chunk-reuse
                                // shifting is only legal on serial prefixes /
                                // frozen ordinary coordinates, never on active
                                // DDVR PUBLIC runs. llama_memory_can_shift
                                // already returns false while an episode is
                                // active; this explicit lane check keeps the
                                // refusal obvious at the call site. OFF: false.
                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd &&
                                    !rerot_owns_slot(slot.id);

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            slot.mem.seq_rm (slot.id, head_p, head_c);
                                            slot.mem.seq_add(slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - 4, 0);
                                    const int np1 = std::min<int>(n_past + 6, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                const bool can_reuse_sparse_frontier =
                                    slot.triattention_compressed &&
                                    has_new_tokens &&
                                    n_past == slot.prompt.n_tokens();

                                if (pos_min >= pos_min_thold && !can_reuse_sparse_frontier) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
                                            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                            if (cur.pos_max > pos_next) {
                                                return false;
                                            }
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        it->load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        it->load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        // restore the draft's speculative state
                                        common_speculative_set_state(spec.get(), slot.id, it->data_spec);

                                        pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                        n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                        SLT_TRC(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) it->size() / 1024 / 1024);
                                    }

                                    if (do_reset) {
                                        SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next) {
                                        SLT_TRC(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.n_prompt_tokens_cache = n_past;
                        if (slot.n_decoded_start == 0) {
                            slot.n_prompt_tokens_cache_response = n_past;
                        }
                        slot.n_prompt_tokens_processed = 0;

                        slot.prompt.tokens.keep_first(n_past);

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream && slot.n_decoded_start == 0) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            return;
                        }
                    }

                    const int64_t t_now = ggml_time_us();
                    slot.t_prompt_processing = slot.t_prompt_processing_accum + (t_now - slot.t_start_process_prompt) / 1e3;
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    slot.mem.seq_rm(slot.id, p0, -1);

                    // A cache mismatch can discard every sparse gap. Once the
                    // retained prefix is dense again, this is a new fill-first
                    // sequence rather than a sticky compressed continuation.
                    if (slot.triattention_compressed &&
                        llama_memory_seq_get_kv_used(llama_get_memory(ctx_tgt), slot.id) >= (uint32_t) slot.prompt.n_tokens()) {
                        slot.triattention_compressed = false;
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    bool has_mtmd = false;

                    // check if we should process the image
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        // process the image
                        size_t n_tokens_out = 0;
                        int32_t res = slot.process_mtmd_chunk(cur_token_idx, n_tokens_out);
                        if (res != 0) {
                            SLT_ERR(slot, "failed to process image, res = %d\n", res);
                            send_error(slot, "failed to process image", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        if (ctx_dft && llama_get_ctx_other(ctx_dft) != ctx_tgt) {
                            // TODO: in the future, figure out how to infuse target embeddings to the images
                            //       for now, we skip this for simplicity
                            //       maybe we simply need to call `common_speculative_process()` on the mtmd batches in the `process_chunk` above?
                            //       [TAG_MTMD_DRAFT_PROCESSING]
                            res = input_tokens.process_chunk(ctx_dft, mctx, slot.prompt.n_tokens(), slot.prompt.tokens.pos_next(), slot.id, n_tokens_out);
                            if (res != 0) {
                                GGML_ABORT("failed to process multi-modal data on draft context\n");
                            }
                        }

                        slot.n_prompt_tokens_processed += n_tokens_out;

                        // add the image chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            slot.prompt.tokens.push_back(chunk.get()); // copy
                        }

                        has_mtmd = true;
                    }

                    const auto & spans = slot.task->params.message_spans;
                    const auto last_user_pos = spans.last_user_message_pos();

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        add_ok &= batch.add(slot.id,
                            cur_tok,
                            slot.prompt.tokens.pos_next(),
                            slot.need_embd());
                        slot.prompt.tokens.push_back(cur_tok);

                        slot.n_prompt_tokens_processed++;

                        // break at the last user message, or at user messages at least min step past the last checkpoint
                        if (do_checkpoint && spans.is_user_start(slot.prompt.n_tokens())) {
                            const auto pos = slot.prompt.n_tokens();
                            const auto & checkpoints = slot.prompt.checkpoints;

                            if (pos == last_user_pos || checkpoints.empty() || pos > checkpoints.back().n_tokens + params_base.checkpoint_min_step) {
                                break;
                            }
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.size() - n_tokens_prev;

                    const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    const bool is_user_start = spans.is_user_start(n_tokens_start);
                    const bool is_last_user_message = n_tokens_start == last_user_pos;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.size() > 0);

                        // extract the logits only for the last token
                        batch.set_output(batch.size() - 1, true);

                        slot.n_decoded = slot.n_decoded_start;
                        slot.i_batch   = batch.size() - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                        // message or we are near the end of the prompt
                        if (!is_user_start && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together, unless it's the last user message
                    do_checkpoint = do_checkpoint && (
                            slot.prompt.checkpoints.empty() ||
                            is_last_user_message || near_prompt_end ||
                            n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }
    }

    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    bool decode(int32_t & n_batch, int32_t off, llama_batch & batch_view) {
        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        if (batch.size() == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true; // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
        // this case is not currently used by any models, but may need to be supported in the future
        if (spec && batch.has_embd) {
            if (llama_model_n_embd_inp(model_dft) != llama_model_n_embd_inp(model_tgt)) {
                SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
                throw std::runtime_error("unsupported batch.has_embd + spec case");
            }
        }

        const int ret = llama_decode(ctx_tgt, batch_view);

        metrics.on_decoded(slots);

        if (ret != 0) {
            if (rerot_active() && batch.slot_batched && batch.slot_batched->rerot_internal) {
                const std::string reason = ret == 1
                    ? "rerot_resource_exhausted: memory cannot commit the complete frontier"
                    : "rerot_backend_error: complete frontier decode failed";
                rerot->hard_abort(rerot_episode_id, reason);
                rerot_propagate_hard_abort();
                throw std::runtime_error(reason);
            }
            if (ret == 1 && n_batch == 1 && params_base.kv_unified && params_base.n_ctx_kv > 0 && try_clear_idle_slots()) {
                return false;
            }

            {
                std::string err;

                if (n_batch == 1 && ret == 1) {
                    // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                    //       need to remove the tokens from the current batch too
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (!err.empty()) {
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.prompt_clear();
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache
            if (!try_clear_idle_slots()) {
                n_batch /= 2;
            }

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

            return false; // retry with the updated n_batch
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        if (!common_speculative_process(spec.get(), batch_view, /*defer_mtp=*/ true)) {
            SRV_ERR("%s", "failed to process speculative batch\n");

            // TODO: handle error
            throw std::runtime_error("failed to process speculative batch");
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                    slot.copy_state_to(*child);
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }

        return true;
    }

    void post_decode(int32_t n_batch_tokens, int32_t off, llama_batch & batch_view) {
        // for checking if a given batch index is inside batch_view
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate(slots, [&](server_slot & slot) {
            for (auto & i : slot.spec_i_batch) {
                if (!is_inside_view(i)) {
                    throw std::runtime_error(string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off, off + n_batch_tokens));
                }
            }
        });

        // RERoT owns its token lifecycle: first commit every physical KV
        // write in the frontier, then mutate topology/retire/admit, and only
        // then sample the next token from the jointly produced logits.
        std::set<uint64_t> rerot_committed_episodes;
        bool rerot_ok = true;
        iterate(slots, [&](server_slot & slot) {
            if (!is_inside_view(slot.i_batch)) {
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task &&
                slot.task->params.rerot_effective(slot.task->type)) {
                // The context-level memory registry currently executes one
                // RERoT visibility domain at a time. Keep additional outer
                // n_cmpl/concurrent roots parked at their completed prelude;
                // completion/abort starts the next root without re-prefill.
                if (rerot_active()) {
                    return;
                }
                if (!rerot_start_root(slot)) {
                    send_error(slot, "failed to start RERoT episode", ERROR_TYPE_SERVER);
                    slot.release();
                    rerot_ok = false;
                }
                return;
            }

            if (slot.state == SLOT_STATE_GENERATING && slot.rerot_internal &&
                slot.rerot_inflight_plan.has_value()) {
                const uint64_t episode_id = slot.rerot_episode_id;
                if (!rerot_commit_inflight(slot)) {
                    rerot_ok = false;
                    return;
                }
                rerot_committed_episodes.insert(episode_id);
            }
        });

        if (rerot_ok) {
            for (const uint64_t episode_id : rerot_committed_episodes) {
                if (!rerot_handle_finished_frontier(episode_id)) {
                    rerot_ok = false;
                    break;
                }
            }
        }

        if (!rerot_ok) {
            rerot_propagate_hard_abort();
            return;
        } else {
            iterate(slots, [&](server_slot & slot) {
                if (slot.rerot_internal && slot.task && slot.task->params.rerot_trace) {
                    SRV_INF("rerot.trace.post_sample_check: episode=%" PRIu64
                            " node=%u state=%d i_batch=%d off=%d n=%d inside=%d injection=%d serial=%d\n",
                        slot.rerot_episode_id, slot.rerot_node_id, slot.state,
                        slot.i_batch, off, n_batch_tokens,
                        is_inside_view(slot.i_batch) ? 1 : 0,
                        static_cast<int>(slot.rerot_injection),
                        slot.rerot_serial_tail ? 1 : 0);
                }
                if (!slot.rerot_internal || slot.state != SLOT_STATE_GENERATING ||
                    !is_inside_view(slot.i_batch)) {
                    return;
                }

                if (slot.rerot_injection != server_rerot_injection_kind::none) {
                    slot.i_batch = -1;
                    return;
                }

                const auto * episode = rerot ? rerot->episode(slot.rerot_episode_id) : nullptr;
                const auto * logical = episode
                    ? episode->document.node(slot.rerot_node_id)
                    : nullptr;
                if (!episode || !logical || episode->hard_aborted ||
                    logical->state == llama_rerot_node_state::forked ||
                    logical->state == llama_rerot_node_state::retired) {
                    if (slot.task && slot.task->params.rerot_trace) {
                        SRV_INF("rerot.trace.post_sample_skip: episode=%" PRIu64
                                " node=%u episode=%d hard_abort=%d logical_state=%d\n",
                            slot.rerot_episode_id, slot.rerot_node_id,
                            episode ? 1 : 0,
                            episode && episode->hard_aborted ? 1 : 0,
                            logical ? static_cast<int>(logical->state) : -1);
                    }
                    return;
                }

                const int tok_idx = slot.i_batch - off;
                if (!rerot_sample_slot(slot, tok_idx)) {
                    rerot_ok = false;
                }
            });
            if (!rerot_ok) {
                rerot_propagate_hard_abort();
                return;
            }
        }

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        auto accept_hack_block = [&](server_slot & slot, const llama_tokens & confirmed_block) {
            if (!slot.token_hack || confirmed_block.empty()) {
                return true;
            }

            llama_tokens appended = slot.token_hack->after_block(confirmed_block);
            if (appended.empty()) {
                return true;
            }

            slot.spec_draft.clear();
            slot.spec_i_batch.clear();
            slot.token_append_anchor = confirmed_block.back();
            slot.token_append_chunk = std::move(appended);

            for (llama_token token : slot.token_append_chunk) {
                common_sampler_accept(slot.smpl.get(), token, true);

                completion_token_output forced;
                forced.tok          = token;
                forced.text_to_send = common_token_to_piece(slot.ctx_tgt, token, accept_special_token(slot, token));
                forced.prob         = 1.0f;

                slot.n_decoded += 1;
                if (!process_token(forced, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();
                    return false;
                }
            }

            slot.sampled = slot.token_append_anchor;
            return true;
        };

        iterate(slots, [&](server_slot & slot) {
            if (slot.rerot_internal ||
                (slot.state == SLOT_STATE_DONE_PROMPT && slot.task &&
                 slot.task->params.rerot_effective(slot.task->type))) {
                return;
            }

            // optionally send prompt processing progress
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->params.stream && slot.task->params.return_progress && slot.n_decoded_start == 0) {
                    send_partial_response(slot, {}, true);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // prompt evaluated for next-token prediction
                slot.state = SLOT_STATE_GENERATING;

                if (slot.can_speculate()) {
                    common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                }
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return; // sample using speculative decoding
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
            }

            slot.i_batch = -1;

            common_sampler_accept(slot.smpl.get(), id, true);

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.n_decoded += 1;

            if (slot.t_start_generation == 0) {
                slot.t_start_generation = t_now;
            }
            if (slot.n_decoded == slot.n_decoded_start + 1) {
                slot.t_print_last = t_now;
                slot.n_decoded_last = slot.n_decoded_start;
                if (slot.n_decoded_start == 0) {
                    slot.t_prompt_processing = slot.t_prompt_processing_accum + (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                }
                metrics.on_prompt_eval(slot);
            }

            slot.t_token_generation = std::max<int64_t>(1, t_now - slot.t_start_generation) / 1e3;

            completion_token_output result;
            result.tok          = id;
            result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
            result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

            if (slot.task->params.sampling.n_probs > 0) {
                populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
            }

            if (!process_token(result, slot)) {
                // release slot because of stop condition
                slot.print_timings();
                send_final_response(slot);
                metrics.on_prediction(slot);
                slot.release();

                return;
            }

            if (!accept_hack_block(slot, llama_tokens { id })) {
                return;
            }

            slot.print_timings_tg();
        });

        // speculative decoding - main model sample and accept
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() || slot.spec_draft.empty()) {
                return;
            }

            // RERoT MTP epoch invalidation (§A.6): a draft bound to an old
            // view stamp must not verify against new shared memory. Drop it,
            // restore the checkpoint, and re-draft next frontier. OFF: false.
            if (rerot_owns_slot(slot.id) && slot.rerot_has_draft_stamp && ctx_tgt != nullptr &&
                llama_rerot_mtp_is_stale(ctx_tgt, slot.id, &slot.rerot_draft_stamp)) {
                slot.spec_ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                if (slot.ctx_dft) {
                    slot.spec_ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
                common_speculative_set_state(spec.get(), slot.id, slot.spec_ckpt.data_spec);
                slot.mem.seq_rm(slot.id, slot.spec_ckpt.pos_max + 1, -1);
                slot.prompt.tokens.keep_first(slot.spec_ckpt.n_tokens);
                slot.spec_draft.clear();
                slot.spec_i_batch.clear();
                slot.rerot_has_draft_stamp = false;
                return;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            GGML_ASSERT(n_draft > 0);

            // verify and try to accept the draft
            {
                // save the sampler sampler state in case we need to restore it
                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                auto accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                slot.spec_i_batch.clear();

                GGML_ASSERT(accepted.size() >= 1);

                const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));

                // check for partial draft acceptance
                if (n_rollback > 0) {
                    if (use_ckpt_tgt) {
                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                        }

                        // partial acceptance is not supported by the context -> truncate the draft and restore the state
                        slot.spec_is_replay = true;
                        slot.spec_draft = std::move(accepted);

                        const auto & ckpt = slot.spec_ckpt;

                        SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                        ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                        if (slot.ctx_dft) {
                            ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }
                        common_speculative_set_state(spec.get(), slot.id, ckpt.data_spec);

                        slot.mem.seq_rm(slot.id, ckpt.pos_max + 1, -1);

                        slot.prompt.tokens.keep_first(ckpt.n_tokens);
                        slot.smpl = std::move(smpl_save);

                        return;
                    }
                }

                if (trace > 0) {
                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                }

                common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);

                slot.spec_draft = std::move(accepted);
            }

            const int64_t t_now = ggml_time_us();

            const auto ids = std::move(slot.spec_draft);

            size_t n_accepted = ids.size() - 1;
            if (slot.spec_is_replay && n_accepted > 0) {
                n_accepted--;
            }
            slot.spec_is_replay = false;

            slot.t_token_generation = std::max<int64_t>(1, t_now - slot.t_start_generation) / 1e3;

            // update how many tokens out of those tested were accepted
            slot.n_draft_accepted += n_accepted;
            slot.n_draft_verif_steps += 1;

            if (slot.n_accepted_per_pos.empty()) {
                slot.n_accepted_per_pos.resize(common_speculative_n_max(&params_base.speculative), 0);
            }
            for (size_t i = 0; i < n_accepted && i < slot.n_accepted_per_pos.size(); ++i) {
                slot.n_accepted_per_pos[i]++;
            }

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

            slot.sampled = ids.back(); // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

            slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // TODO: set result.probs

                slot.n_decoded += 1;

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();

                    return;
                }
            }

            if (!accept_hack_block(slot, llama_tokens(ids.begin(), ids.end()))) {
                return;
            }

            slot.print_timings_tg();

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) n_accepted, (int) n_draft, slot.prompt.n_tokens());
        });

        if (!common_speculative_commit(spec.get())) {
            throw std::runtime_error("failed to commit deferred speculative state");
        }
    }

    int get_slot_n_ctx() {
        return slots.back().n_ctx;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* has_inp_video          */ impl->chat_params.allow_video,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* slot_n_ctx             */ impl->get_slot_n_ctx(),
        /* pooling_type           */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
        /* model_ftype            */ ftype_name,
    };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_token_hack_factory(server_token_hack_factory_t factory) {
    GGML_ASSERT(impl->ctx_tgt == nullptr && "token hack factory must be set before load_model");
    impl->token_hack_factory = std::move(factory);
}

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
    impl->queue_tasks.on_sleeping_state([this](bool sleeping) {
        if (sleeping) {
            impl->callback_state(SERVER_STATE_SLEEPING, {});
        }
        // for sleeping == false, event is emitted by load_model()
    });
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        // message delimiters for checkpointing
        auto delimiters = common_chat_msg_delimiters_parse(json_value(data, "message_delimiters", json::array()));
        delimiters.tokenize(ctx_server.vocab);

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->slot_n_ctx,
                    meta->logit_bias_eog,
                    data);
            server_rerot_apply_request_json(task, data, params);

            std::string rerot_error;
            if (!server_rerot_validate_task(task, rerot_error)) {
                throw std::invalid_argument(rerot_error);
            }

            task.params.message_spans = task.tokens.find_message_spans(delimiters);

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;
            task.cache_key = json_value(data, "cache_key", json_value(data, "session_id", std::string()));

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            ctx_server.dump_request_tokens(
                task.tokens.get_text_tokens(),
                type == SERVER_TASK_TYPE_INFILL
                    ? server_token_dump_kind::infill
                    : server_token_dump_kind::completion);

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) res_task->n_prompt_tokens_processed_total}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) res_task->t_prompt_processing_total / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) res_task->n_tokens_predicted_total}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) res_task->t_tokens_generation_total / 1.e3}
            }, {
                    {"name",  "n_decode_total"},
                    {"help",  "Total number of llama_decode() calls"},
                    {"value",  res_task->n_decode_total}
            }, {
                    {"name",  "n_tokens_max"},
                    {"help",  "Largest observed n_tokens."},
                    {"value",  res_task->n_tokens_max}
            }, {
                    {"name",  "tri_drain_total"},
                    {"help",  "TriAttention initial pressure drains completed."},
                    {"value",  res_task->tri_drain_total}
            }, {
                    {"name",  "tri_maintenance_total"},
                    {"help",  "TriAttention sticky maintenance reclaims completed."},
                    {"value",  res_task->tri_maintenance_total}
            }, {
                    {"name",  "tri_floor_exhausted_total"},
                    {"help",  "KV pressure events that reached the TriAttention residency floor."},
                    {"value",  res_task->tri_floor_exhausted_total}
            }, {
                    {"name",  "tri_cells_freed_total"},
                    {"help",  "Physical KV cells freed by TriAttention over the process lifetime."},
                    {"value",  res_task->tri_cells_freed_total}
            }, {
                    {"name",  "tri_score_seconds"},
                    {"help",  "Wall time spent in TriAttention importance scoring."},
                    {"value",  res_task->tri_score_us_total / 1.e6}
            }, {
                    {"name",  "tri_pack_seconds"},
                    {"help",  "Wall time spent compacting TriAttention K/V data."},
                    {"value",  res_task->tri_pack_us_total / 1.e6}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  res_task->n_prompt_tokens_processed ? 1.e3 / res_task->t_prompt_processing * res_task->n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  res_task->n_tokens_predicted ? 1.e3 / res_task->t_tokens_generation * res_task->n_tokens_predicted : 0.}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of requests processing."},
                    {"value",  (uint64_t) res_task->n_processing_slots}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of requests deferred."},
                    {"value",  (uint64_t) res_task->n_tasks_deferred}
            },{
                    {"name",  "n_busy_slots_per_decode"},
                    {"help",  "Average number of busy slots per llama_decode() call"},
                    {"value",  (float) res_task->n_busy_slots_total / std::max((float) res_task->n_decode_total, 1.f)}
            },{
                    {"name",  "tri_cells_before"},
                    {"help",  "Physical KV cells before the most recent successful TriAttention reclaim."},
                    {"value",  res_task->tri_cells_before}
            },{
                    {"name",  "tri_cells_after"},
                    {"help",  "Physical KV cells after the most recent successful TriAttention reclaim."},
                    {"value",  res_task->tri_cells_after}
            },{
                    {"name",  "tri_cells_freed"},
                    {"help",  "Physical KV cells freed by the most recent successful TriAttention reclaim."},
                    {"value",  res_task->tri_cells_freed}
            },{
                    {"name",  "tri_references_removed"},
                    {"help",  "Per-sequence KV references removed by the most recent successful TriAttention reclaim."},
                    {"value",  res_task->tri_references_removed}
            },{
                    {"name",  "tri_target_references"},
                    {"help",  "Reference-level residency targets in the most recent successful TriAttention reclaim."},
                    {"value",  res_task->tri_target_references}
            },{
                    {"name",  "tri_hard_keep"},
                    {"help",  "Hard-protected KV references in the most recent successful TriAttention reclaim."},
                    {"value",  res_task->tri_hard_keep}
            },{
                    {"name",  "tri_shared_keep"},
                    {"help",  "Physical KV cells retained by multiple sequences after the most recent TriAttention reclaim."},
                    {"value",  res_task->tri_shared_keep}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        if (!res_task->rerot.empty()) {
            const auto emit_rerot = [&prometheus](
                    const char * type,
                    const char * name,
                    const char * help,
                    const auto & value) {
                prometheus << "# HELP llamacpp:" << name << " " << help << "\n"
                           << "# TYPE llamacpp:" << name << " " << type << "\n"
                           << "llamacpp:" << name << " " << value << "\n";
            };
            const auto & r = res_task->rerot;
            emit_rerot("counter", "rerot_episode_total", "RERoT episodes started.", r.episode_total);
            emit_rerot("gauge", "rerot_episode_active", "RERoT episodes currently active.", r.episode_active);
            emit_rerot("counter", "rerot_nodes_created", "RERoT logical nodes created.", r.nodes_created);
            emit_rerot("counter", "rerot_nodes_started", "RERoT logical nodes admitted to execution.", r.nodes_started);
            emit_rerot("counter", "rerot_nodes_retired", "RERoT logical nodes retired.", r.nodes_retired);
            emit_rerot("counter", "rerot_nodes_queued", "RERoT logical nodes enqueued.", r.nodes_queued);
            emit_rerot("gauge", "rerot_queue_max", "Largest observed RERoT ready queue.", r.queue_max);
            emit_rerot("counter", "rerot_forks_total", "RERoT fork publications.", r.forks_total);
            emit_rerot("gauge", "rerot_max_depth", "Largest observed RERoT tree depth.", r.max_depth);
            emit_rerot("gauge", "rerot_max_live_lanes", "Largest observed concurrent RERoT Lane count.", r.max_live_lanes);
            emit_rerot("counter", "rerot_public_tokens", "RERoT PUBLIC tokens committed.", r.public_tokens);
            emit_rerot("counter", "rerot_private_tokens", "RERoT PRIVATE control tokens committed.", r.private_tokens);
            emit_rerot("counter", "rerot_pending_tokens", "RERoT PENDING record tokens committed.", r.pending_tokens);
            emit_rerot("counter", "rerot_completed_episode_total", "RERoT episodes that produced a final response.", r.completed_episodes);
            emit_rerot("counter", "rerot_completed_model_tokens", "Model-forward tokens committed by completed RERoT episodes; includes PUBLIC, PRIVATE, and PENDING tokens.", r.completed_model_tokens);
            emit_rerot("counter", "rerot_parallel_model_tokens", "Model-forward tokens committed between first multi-Lane admission and the final acquire fence.", r.parallel_model_tokens);
            emit_rerot("counter", "rerot_completed_episode_seconds", "RERoT episode wall time after prompt prefill; divide completed model tokens by this value for aggregate throughput.", r.completed_episode_seconds);
            emit_rerot("counter", "rerot_parallel_seconds", "Multi-Lane phase wall time; divide parallel model tokens by this value for parallel aggregate throughput.", r.parallel_seconds);
            emit_rerot("counter", "rerot_frontiers", "RERoT frontiers committed.", r.frontiers);
            emit_rerot("counter", "rerot_topology_barriers", "RERoT topology barriers completed.", r.topology_barriers);
            emit_rerot("counter", "rerot_refresh_total", "RERoT view refreshes completed.", r.refresh_total);
            emit_rerot("counter", "rerot_mtp_invalidations", "RERoT-stale speculative drafts invalidated.", r.mtp_invalidations);
            emit_rerot("counter", "rerot_context_shifts", "RERoT shared-log context shifts completed.", r.context_shifts);
            emit_rerot("counter", "rerot_hard_aborts", "RERoT episodes atomically aborted.", r.hard_aborts);
            emit_rerot("counter", "rerot_final_fences", "RERoT final acquire fences completed.", r.final_fences);
            emit_rerot("counter", "rerot_span_count", "DDVR spans rendered for RERoT views.", r.span_count);
            emit_rerot("counter", "rerot_parked_total", "RERoT recurrent lineages parked.", r.parked_total);
            emit_rerot("counter", "rerot_archive_total", "RERoT public runs archived.", r.archive_total);
            emit_rerot("counter", "rerot_ddvr_seconds", "Wall time spent building RERoT DDVR views.", r.ddvr_seconds);
        }

        // Labeled fallback counter is emitted separately because the compact
        // metric-definition helper above intentionally models unlabeled series.
        prometheus << "# HELP llamacpp:tri_atomic_fallback_total Atomic fallback entries after dynamic-memory pressure, by reason.\n"
                   << "# TYPE llamacpp:tri_atomic_fallback_total counter\n"
                   << "llamacpp:tri_atomic_fallback_total{reason=\"kv\"} "
                   << res_task->tri_atomic_fallback_kv_total << "\n"
                   << "llamacpp:tri_atomic_fallback_total{reason=\"recurrent\"} "
                   << res_task->tri_atomic_fallback_recurrent_total << "\n";

        res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->t_start);
        res->content_type = "text/plain; version=0.0.4";
        res->status = 200;
        res->data = prometheus.str();
        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto * res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->slots_data);
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        task_params tparams;
        tparams.sampling = params.sampling;
        json default_generation_settings_for_props = json {
            { "params", tparams.to_json(true) },
            { "n_ctx",  meta->slot_n_ctx },
        };

        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        std::string tmpl_tools   = common_chat_templates_source(meta->chat_params.tmpls.get(), "tool_use");

        json props = {
            { "default_generation_settings", default_generation_settings_for_props },
            { "total_slots",                 params.n_parallel },
            { "model_alias",                 meta->model_name },
            { "model_ftype",                 meta->model_ftype },
            { "model_path",                  meta->model_path },
            { "modalities",                  json {
                {"vision", meta->has_inp_image},
                {"video",  meta->has_inp_video},
                {"audio",  meta->has_inp_audio},
            } },
            { "media_marker",                get_media_marker() },
            { "endpoint_slots",              params.endpoint_slots },
            { "endpoint_props",              params.endpoint_props },
            { "endpoint_metrics",            params.endpoint_metrics },
            { "ui",                          params.ui },
            { "ui_settings",                 meta->json_ui_settings },
            { "chat_template",               tmpl_default },
            { "chat_template_caps",          meta->chat_template_caps },
            { "bos_token",                   meta->bos_token_str },
            { "eos_token",                   meta->eos_token_str },
            { "build_info",                  meta->build_info },
            { "is_sleeping",                 queue_tasks.is_sleeping() },
            { "cors_proxy_enabled",          params.ui_mcp_proxy },
        };
        if (params.use_jinja) {
            if (!tmpl_tools.empty()) {
                props["chat_template_tool_use"] = tmpl_tools;
            }
        }
        res->ok(props);
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        json models = {
            {"models", {
                {
                    {"name",  meta->model_name},
                    {"model", meta->model_name},
                    {"modified_at", ""},
                    {"size", ""},
                    {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                    {"type", "model"},
                    {"description", ""},
                    {"tags", {""}},
                    {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})},
                    {"parameters", ""},
                    {"details", {
                        {"parent_model", ""},
                        {"format", "gguf"},
                        {"family", ""},
                        {"families", {""}},
                        {"parameter_size", ""},
                        {"quantization_level", ""}
                    }}
                }
            }},
            {"object", "list"},
            {"data", {
                get_model_info(),
            }}
        };

        res->ok(models);
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i]);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                ctx_server.dump_request_tokens(tmp.get_text_tokens(), server_token_dump_kind::rerank);
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

json server_routes::get_model_info() const {
    return json {
        {"id",       meta->model_name},
        {"aliases",  meta->model_aliases},
        {"tags",     meta->model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta->model_vocab_type},
            {"n_vocab",     meta->model_vocab_n_tokens},
            {"n_ctx",       meta->slot_n_ctx},
            {"n_ctx_train", meta->model_n_ctx_train},
            {"n_embd",      meta->model_n_embd_inp},
            {"n_params",    meta->model_n_params},
            {"size",        meta->model_size},
            {"ftype",       meta->model_ftype},
        }},
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            ctx_server.dump_request_tokens(
                tokenized_prompts[i].get_text_tokens(),
                server_token_dump_kind::embedding);
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(mctx, prompt.get<std::string>(), files, true).size();
    } else {
        n_tokens = tokenize_mixed(vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}
