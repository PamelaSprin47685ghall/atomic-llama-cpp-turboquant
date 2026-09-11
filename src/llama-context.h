#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-flashprefill-metrics.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <array>
#include <deque>
#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    // Lightweight constructor for control-plane and unit tests (skips sched_reserve) (§B.13 Phase 2)
    llama_context(
            const llama_model & model,
            const llama_cparams & cparams,
            bool /* test_only */);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;
    ggml_backend_sched_compute_pool_t get_compute_pool() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_ctx_kv()  const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    // Synchronous KV reclaim — calls synchronize() then memory->reclaim_kv()
    // Returns the reclaim result. If memory doesn't support reclaim, result.supported=false.
    llama_memory_kv_reclaim_result memory_reclaim_kv(const llama_memory_kv_reclaim_request & request);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);
    float * get_attention_q_pre_rope(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_attention_q_pre_rope(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    // Returns false (refusing with zero mutation) when the change is effective
    // but cannot be applied safely: FlashPrefill enabled atop an active RERoT
    // episode, where clearing stale KV would orphan episode lineage. Identical
    // sets are a no-op success (true). True also covers every OFF / inactive /
    // successfully-applied case.
    bool set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    // FlashPrefill V2 explicit execution path (ContextIntegration owner).
    // ABI-safe: `exec` is a borrowed versioned view (PolicyCore: struct
    // llama_flashprefill_exec); rows are validated and copied into owned
    // context storage before any async graph work, so no dangling arrays.
    // NULL exec == ordinary dense (same as decode). Mismatched version,
    // struct size, or n_rows != batch.n_tokens is rejected with -1 before
    // any memory/graph state mutates. Ordinary decode() always routes
    // unknown/dense. Declared publicly by ConfigIntegration in llama.h.
    int decode_with_flashprefill(
            const llama_batch & batch_inp,
            const struct llama_flashprefill_exec * exec);

    // Immutable policy fingerprint for later state/cache isolation and
    // metrics (StatePolicy consumer). Deterministic over the frozen policy
    // fields plus model identity (see .cpp for exact sources); 0 on error.
    // Never changes over the context lifetime.
    uint64_t flashprefill_policy_fingerprint() const;

    // Conservative per-context identity token. Assigned (process-unique) only
    // when the policy is enabled; 0 when OFF (OFF contexts use the legacy
    // stateless path and carry no isolation identity). State contract:
    // within-context RAM/checkpoint round-trips observe the same serial
    // (allowed with a matching policy fingerprint and adapter generation);
    // cross-context durable restore observes a different serial and is
    // rejected (model.desc is not a weight hash, so no verifiable durable
    // identity exists today). Never changes after construction.
    uint64_t flashprefill_context_serial() const;

    // Effective adapter generation for FlashPrefill state identity
    // (StatePolicy owner). Starts at 1; bumped on every effective LoRA
    // set/remove/scale or cvec mutation (no-op calls that change nothing do
    // not bump). Saturates at UINT64_MAX, which is always-invalid (consumers
    // must treat MAX as never-reusable, like CellGeneration). Bumped on the
    // rare adapter-mutation path only — zero decode-path cost.
    uint64_t flashprefill_adapter_generation() const;

    // Persistent state/cache key getter (StatePolicy-owned, ServerRouting-
    // called; single source of truth for RAM-cache stamps and slot sidecars).
    // Mixes the policy fingerprint, context serial, and adapter generation
    // (see llama_flashprefill_state::state_cache_key). Returns 0 when the
    // policy is OFF or the fingerprint is unavailable (no isolation identity
    // needed: OFF uses the legacy stateless path). Never changes for a fixed
    // adapter set; changes on every effective adapter mutation (stale stores
    // are rejected, never silently reused).
    uint64_t flashprefill_state_cache_key() const;

    // Immutable policy accessor (references cparams storage; valid for the
    // context lifetime; default OFF).
    const struct llama_flashprefill_config & get_flashprefill_config() const;

    // Owned call-level row count from the last validated exec (0 when no
    // exec is attached, the policy is OFF, or the call was bypassed).
    uint32_t flashprefill_call_rows() const;

    // Owned per-ubatch row snapshot for the most recently prepared ubatch,
    // sliced via the BatchIdentity source-row map (never guessed).
    // Returns nullptr with *n_rows_out == 0 when unavailable (OFF, bypassed,
    // legacy decode, or no source map for the ubatch) — the caller must route
    // dense. Valid until the next decode/graph_reserve call on this context;
    // never freed by the caller. GraphIntegration hook for the next slice.
    const struct llama_flashprefill_row * flashprefill_ubatch_rows(uint32_t * n_rows_out) const;

    // FlashPrefill metrics snapshot (MetricsIntegration owner; append-only
    // block, disjoint from StatePolicy state/adapter/builder lines).
    // Cumulative COMMITTED-call totals (see llama-flashprefill-metrics.h for
    // units and merge discipline; per-ubatch deltas stage during the call
    // and publish only when the whole decode_impl succeeds). OFF contexts
    // return an empty accum (no allocation, no timers, no reads ever
    // queued). Cheap copy-out; single-threaded decode ordering assumed
    // (same as the decode path).
    llama_flashprefill_metrics::accum flashprefill_metrics_snapshot() const;

    // Drain-once handoff for the server post_decode path: delta = snapshot
    // minus the last drained watermark, then the watermark advances.
    // Returns false with `out` cleared when OFF or when nothing new committed
    // since the last drain (server merges nothing then). Failed/retried
    // calls never commit, so they are never drained (transactional).
    bool flashprefill_metrics_consume(llama_flashprefill_metrics::slice_delta & out);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.embeddings_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    void extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    // async-copy enabled normalized pre-RoPE Q tensors into host buffers
    void extract_attention_q_pre_rope(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    // FlashPrefill internals (all no-ops / empty when the policy is OFF).
    int decode_impl(const llama_batch & batch_inp);
    void flashprefill_clear_call();
    bool flashprefill_attach_call(
            const llama_batch & batch_inp,
            const struct llama_flashprefill_exec * exec);
    // Slices fp_rows_call into fp_rows_ubatch via the BatchIdentity
    // source-row map and publishes the graph-visible snapshot. Returns true
    // on success (including legitimate dense: inactive/bypassed/empty calls
    // and synthetic map-less ubatches with no live rows). Returns false on a
    // corrupt map for an enabled live call (map missing while rows are live,
    // ubatch wider than the call, or an out-of-range source row): the caller
    // must fail the execution explicitly (decode error code) with no half
    // snapshot left behind — never silent fallback to dense.
    bool flashprefill_build_ubatch(const llama_ubatch & ubatch);
    // True when this call must stay dense regardless of row roles: MTP
    // contexts (draft/verify keep the pre-existing attention path) and
    // embedding/rerank pooling (non-generation path).
    bool flashprefill_bypassed() const;
    // MetricsIntegration: build + stage one successful-ubatch delta for this
    // ubatch (owned rows + authoritative graph verdict; plans dock at the
    // end-of-call fold when reads were queued). graph_dense_reason is the
    // graph summary verdict for this ubatch (recorded even with zero plans).
    // plans_deferred must equal whether this ubatch queued plan reads.
    // Returns 0 on success (delta staged, possibly empty); nonzero stats
    // error means the caller must fail the slice with no successful output
    // and stage nothing. The staged deltas commit into the ledger only when
    // the whole decode_impl succeeds. OFF and ordinary dense calls without
    // an exec are cheap no-ops. Never infers routing from global KV fullness
    // or guessed backend support; required enforcement lives graph-side.
    int flashprefill_note_slice_success(
            const llama_ubatch & ubatch, uint64_t layout_us, bool has_layout_us, bool plans_deferred,
            int32_t graph_dense_reason);
    // MetricsIntegration: queue one 96B async plan-header read per plan of
    // this ubatch's completed-submit graph into the owned per-call snapshot
    // (no wait). Returns 1 when reads were queued, 0 when the graph built no
    // plans (dense ubatch), <0 on malformed plan tensors (fail the slice
    // closed — never queue a short/OOB read).
    int flashprefill_queue_plan_reads(const llm_graph_result & res, bool & out_queued);
    // MetricsIntegration: after the single end-of-call synchronize, parse ALL
    // queued headers (header-only parser, no full-tensor reads) and fold
    // plan-confirmed totals + pool + actual scratch into `out`. Returns 0, or
    // the first plan/parse error with `out` cleared (fail the call closed).
    int flashprefill_fold_plan_reads(llama_flashprefill_metrics::slice_delta & out);
    // MetricsIntegration: snapshot per-backend GPU phase counters at call
    // start, before the first submit (enabled-only, host reads, no waits).
    // Baselines the end-of-call difference so the first FP call counts as
    // measured instead of being dropped; never assumes counters begin at 0
    // or share the context lifetime.
    void flashprefill_snapshot_gpu_phases();
    // MetricsIntegration: sample the VulkanDispatch phase counters
    // post-sync (proc address resolved on demand, null-safe) and stage this
    // call's GPU dispatch times into pending. No extra waits, no-ops without
    // queued plans or without the producer hook.
    void flashprefill_sample_gpu_phases();
    // MetricsIntegration: clear per-call staging (pending + plan log) for a
    // failed call. Committed ledger and server watermark are untouched.
    void flashprefill_metrics_clear_call_state();

    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    // FlashPrefill state envelope (StatePolicy owner). Writes the framed
    // identity header (small length-prefixed marker + 64-byte body) when the
    // policy is enabled (no-op when OFF, preserving
    // legacy bytes). Reads + fail-closed-validates it before any KV bytes
    // are consumed or context/memory state mutates; throws
    // std::runtime_error naming the domain + re-prefill path on
    // missing/unknown/mismatching headers (converted to a 0 return by the
    // existing state wrappers). scope is kScopeFull/kScopeSeq respectively.
    void flashprefill_state_write_envelope(llama_io_write_i & io, uint32_t scope);
    void flashprefill_state_read_envelope(llama_io_read_i & io, uint32_t expected_scope);

    // Bumps fp_adapter_gen once (saturates at MAX, never wraps). Called only
    // on effective adapter mutations; no-op calls that change nothing, and
    // failed cvec applies, do not bump.
    void flashprefill_bump_adapter_generation();

    // Gate for effective adapter changes: false iff the policy is enabled and
    // a RERoT episode is active, where swapping the effective adapters has no
    // safe re-resolution path for episode-bound derived state — the caller
    // must refuse BEFORE applying, with zero mutation, leaving the coherent
    // graph untouched. True in every other case (OFF or no active episode);
    // identical (no-op) changes never reach the gate. Normal (non-episode)
    // switching keeps baseline KV semantics: this gate never deletes KV.
    bool flashprefill_adapter_change_allowed() const;

    // Effective-adapter-change coordinator: bumps the adapter generation so
    // pre-switch saved blobs and RAM entries reject on load (they pin the old
    // generation). Never deletes KV: resident history keeps baseline semantics
    // and the server owns per-slot LoRA/prompt compatibility; graph/scheduler
    // re-reservation happens in the setters (sched_need_reserve) and derived
    // means rebuild per graph under the new generation. OFF behaves
    // identically (bump only).
    void flashprefill_on_adapter_change();

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.embeddings_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;

    // host buffers for normalized attention Q before RoPE, per layer
    // populated when cparams.attention_q_pre_rope[il] is true
    std::vector<buffer_view<float>> attention_q_pre_rope;

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    bool sched_need_reserve = true;

    // FlashPrefill V2 owned execution snapshot (ContextIntegration owner).
    // fp_rows_call: validated owned copy of the call's source rows, taken
    //   before any async graph work (never a borrowed pointer, never null
    //   when active). Empty unless an exec passed validation with the policy
    //   enabled and the call bypassed neither MTP nor embedding/rerank.
    // fp_rows_ubatch: this ubatch's owned slice of fp_rows_call, keyed by the
    //   BatchIdentity source-row map after internal splits/retries. Empty
    //   means "route dense" (never a guess).
    // fp_exec_active: an owned snapshot is attached to the in-flight call.
    // fp_serial: per-context identity, unique within the process (nonzero
    //   only when the policy is enabled, 0 when OFF). NEVER cross-process
    //   identity on its own: the counter restarts at 1 in every process.
    // fp_nonce0/1: the fixed 128-bit process nonce at construction time
    //   (nonzero only when the policy is enabled, 0 when OFF). Envelope
    //   compare is exact on (nonce, serial) jointly, so cross-restart reuse
    //   is probabilistically impossible even when serial/policy/adapter
    //   coincide. Probabilistic anti-collision, not a secret.
    // fp_adapter_gen: effective adapter generation for state identity (see
    //   flashprefill_adapter_generation()). Starts at 1; MAX is sticky and
    //   always-invalid. Bumped only on effective adapter mutations.
    // OFF cost: empty members + null snapshot + zero identity (no buffers,
    // maps, randomness, or synchronization on any OFF path).
    std::vector<struct llama_flashprefill_row> fp_rows_call;
    std::vector<struct llama_flashprefill_row> fp_rows_ubatch;
    bool     fp_exec_active = false;
    uint64_t fp_serial      = 0;
    uint64_t fp_nonce0      = 0;
    uint64_t fp_nonce1      = 0;
    uint64_t fp_adapter_gen = 1;

    // FlashPrefill reserve-sizing scope (StatePolicy writer, GraphIntegration
    // reader). True only while graph_reserve() builds the synthetic sizing
    // graph; false for every live decode graph. Read by graph_params() into
    // llm_graph_params.flashprefill_reserve_sizing (reuse-keyed, so reserve
    // graphs never alias live graphs). Toggled only through the RAII guard in
    // graph_reserve(), which restores the prior value on all exits including
    // early returns and exceptions. OFF cost: one bool store per reserve.
    bool fp_reserve_sizing_active = false;

    // Phase of the graph computed last: prompt processing and token generation never run at the
    // same time. On a change, the graph for the new phase is reserved, which releases the buffers
    // of the finished one (see graph_reserve).
    int      last_graph_phase = -1;   // -1 nothing yet, 0 token generation, 1 prompt processing
    uint32_t res_n_tokens_pp  = 0;
    uint32_t res_n_seqs_pp    = 0;
    uint32_t res_n_outputs_pp = 0;
    uint32_t res_n_seqs_tg    = 0;

    // FlashPrefill metrics ledger (MetricsIntegration owner; append-only).
    // fp_metrics_pending: per-call staging; each successful ubatch merges
    //   here, and decode_impl commits it into fp_metrics_accum exactly once
    //   on return 0, or discards it on every failure return. A narrowed-batch
    //   retry therefore recounts only re-executed work — never double.
    // fp_metrics_accum: cumulative committed-call totals (server drains this).
    // fp_metrics_mark: server drain watermark for flashprefill_metrics_consume.
    // fp_plan_headers: lazily built deque with one stable 24-word slot per
    //   queued plan (deque element references never invalidate on push, so
    //   queued async copies survive later pushes/ubatches until a scheduler
    //   sync completes them; freed only post-sync on success, or post-sync
    //   in clear_call_state on failure — never freed with copies
    //   outstanding). Null until the first actual plan read is queued: OFF
    //   performs no heap allocation here (a bare deque can allocate its map
    //   even when empty; the unique_ptr keeps OFF at exactly nullptr).
    // fp_plan_counts: plan count per queued ubatch, in loop order.
    // fp_plan_expected: expected full-tensor word count per queued plan
    //   (header total_words is validated against it; no whole-plan readback).
    // fp_plan_summaries: per-queued-ubatch CPU build summary (no sync).
    // fp_plan_reads_pending: true from the first async queue until a scheduler
    //   sync completes the reads (success end-sync or failure-clear sync).
    //   Tracks outstanding copies — not container fullness — so a later clear
    //   never re-syncs already-completed reads. Headers are retained for
    //   parsing after the completing sync; freed separately.
    // fp_gpu_last_by_backend: last per-backend phase sample; baselined at
    //   call start (snapshot, enabled-only) so the first call counts.
    // fp_met_call_pos_min: reusable call-wide hygiene range (per-sequence
    //   minima for fail-closed cleanup). Sized once while enabled and reused
    //   across calls; stays empty while OFF (immutable config: OFF allocates
    //   nothing here, ever).
    // OFF cost: zeroed structs + empty containers (no buffers, no timers, no sync).
    bool fp_plan_reads_pending = false;
    std::vector<llama_pos> fp_met_call_pos_min;
    llama_flashprefill_metrics::slice_delta fp_metrics_pending;
    llama_flashprefill_metrics::accum fp_metrics_accum;
    llama_flashprefill_metrics::accum fp_metrics_mark;
    // Owned per-call plan-header snapshot log. Each queued plan owns one
    // stable 24-word slot: std::deque never invalidates element references
    // on push, so a queued async GPU copy stays valid across later pushes
    // and later ubatches until a scheduler sync completes it. Slots are
    // freed only after completion: on the success path the end-of-call sync
    // covers them before parsing; on every failure path clear_call_state
    // syncs first (direct scheduler sync, never the stats wrapper).
    std::unique_ptr<std::deque<std::array<int32_t, 24>>> fp_plan_headers;
    std::vector<uint32_t> fp_plan_counts;
    std::vector<uint64_t> fp_plan_expected;
    std::vector<llm_graph_flashprefill_summary> fp_plan_summaries;
    // Last per-backend GPU phase sample (VulkanDispatch producer). Keyed by
    // live backend pointer; entries for dead backends are pruned at each
    // finalize. Unknown backends stamp without contributing (never attribute
    // pre-existing cumulative time). OFF cost: empty map.
    struct fp_gpu_backend_last {
        uint64_t pool_ns = 0;
        uint64_t select_ns = 0;
        uint64_t attn_ns = 0;
        bool     stamped = false;
        uint64_t split_cur_b = 0;
        uint64_t split_peak_b = 0;
    };
    std::map<ggml_backend_t, fp_gpu_backend_last> fp_gpu_last_by_backend;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // NOTE: these must be declared *after* `backends`.
    //
    // Members are destroyed in reverse declaration order, and
    // ggml_backend_sched_free() synchronizes every backend the scheduler was
    // built over. Declaring the scheduler first destroys the backends first and
    // then lets the scheduler touch them - a use-after-free that surfaces as a
    // segfault inside the graphics driver (resetCommandPool on an
    // already-destroyed command pool) on every context teardown.
    //
    // `sched` is built over `compute_pool`, so it must also stay declared after
    // it in order to be destroyed first.
    ggml_backend_sched_compute_pool_ptr compute_pool;
    ggml_backend_sched_ptr sched;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
