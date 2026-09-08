#pragma once

#include "llama-arch.h"
#include "llama-batch.h"
#include "llama-hparams.h"
#include "llama-adapter.h"

// FlashPrefill V2 contracts (GraphIntegration consumes, never redefines):
// - PolicyCore: public policy/roles/routes + internal route/pack/scratch helpers.
// - CacheFragments: legal-fragment layout, compact uses, build key, budgets.
#include "llama-flashprefill.h"
#include "llama-flashprefill-layout.h"

class llm_graph_result; // metrics recording target (defined below)
class llm_graph_input_attn_flashprefill; // companion input (defined below)

#include <cstdint>
#include <vector>
#include <memory>
#include <set>
#include <functional>
#include <map>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

struct llama_cparams;
struct llama_layer;

struct llama_memory_context_i;

class llama_kv_cache_context;
class llama_kv_cache_dsa_context;
class llama_kv_cache_msa_context;
class llama_kv_cache_dsv4_raw_context;
class llama_kv_cache_dsv4_context;
class llama_kv_cache_iswa_context;
class llama_memory_recurrent_context;
class llama_memory_hybrid_context;
class llama_memory_hybrid_iswa_context;

namespace llama_xkv { class xkv_graph_snapshot; }

// certain models (typically multi-modal) can produce different types of graphs
enum llm_graph_type {
    LLM_GRAPH_TYPE_DEFAULT,
    LLM_GRAPH_TYPE_ENCODER,
    LLM_GRAPH_TYPE_DECODER,
    LLM_GRAPH_TYPE_DECODER_MTP,
};

enum llm_fused_op {
    LLM_FUSED_OP_FLASH_ATTN,
    LLM_FUSED_OP_GDN_AR,
    LLM_FUSED_OP_GDN_CH,
    LLM_FUSED_OP_LIGHTNING_INDEXER,
    LLM_FUSED_OP_DSV4_HC_PRE,
    LLM_FUSED_OP_DSV4_HC_COMB,
    LLM_FUSED_OP_DSV4_HC_POST,
};

enum llm_ffn_op_type : int {
    LLM_FFN_NONE = 0,           // sentinel: unset; archs must assign before use
    LLM_FFN_SILU,
    LLM_FFN_GELU,
    LLM_FFN_RELU,
    LLM_FFN_RELU_SQR,
    LLM_FFN_SWIGLU,
    LLM_FFN_GEGLU,
    LLM_FFN_REGLU,
    LLM_FFN_SWIGLU_OAI_MOE,
    LLM_FFN_SITU,               // kimi k3: soft-capped SiLU gate + soft-capped up branch
};

enum llm_ffn_gate_type {
    LLM_FFN_SEQ,
    LLM_FFN_PAR, // ffn_gate is parallel to ffn_up
};

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
    LLM_NORM_GROUP,
};

// TODO: tmp - need something better to pass the data from the encoder to the decoder
struct llama_cross {
    // the output embeddings from the encoder as a ggml tensor
    // TODO: this needs more work to be correct, for now copy the embeddings data to host memory
    //       ref: https://github.com/ggml-org/llama.cpp/pull/11213#discussion_r1969892524
    //ggml_tensor * t_embd = nullptr;

    int64_t n_embd = 0;
    int64_t n_enc  = 0;

    // embeddings data copied to host memory (tmp)
    std::vector<float> v_embd;

    // needed to construct the cross-attention mask in the decoder
    std::vector<std::set<llama_seq_id>> seq_ids_enc;
};

struct llm_graph_params;

//
// llm_graph_input
//

class llm_graph_input_i {
public:
    llm_graph_input_i() {
        const char * LLAMA_GRAPH_INPUT_DEBUG = getenv("LLAMA_GRAPH_INPUT_DEBUG");
        debug = LLAMA_GRAPH_INPUT_DEBUG ? atoi(LLAMA_GRAPH_INPUT_DEBUG) : 0;
    }

    virtual ~llm_graph_input_i() = default;

    // FlashPrefill companion discovery (internal only, default null).
    virtual llm_graph_input_attn_flashprefill * get_fp_input() { return nullptr; }

    virtual void set_input(const llama_ubatch * ubatch) = 0;

    // return true if the resulting input tensors using the provided graph parameters would be
    //   the same as the previous input tensors that we have currently stored in the object
    virtual bool can_reuse(const llm_graph_params & params) {
        // returning false here by default will prevent from reusing the graph if the check
        //   for the input type has not been implemented yet
        GGML_UNUSED(params);
        return false;
    }
protected:
    // env: LLAMA_GRAPH_INPUT_DEBUG
    int debug = 0;
};

using llm_graph_input_ptr = std::unique_ptr<llm_graph_input_i>;

class llm_graph_input_embd : public llm_graph_input_i {
public:
    llm_graph_input_embd(int64_t n_embd) : n_embd(n_embd) {}
    virtual ~llm_graph_input_embd() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * tokens = nullptr; // I32 [n_batch]
    ggml_tensor * embd   = nullptr; // F32 [n_embd, n_batch]

    const int64_t n_embd = 0;
};

// similar to llm_graph_input_embd but with an additional hidden state input
class llm_graph_input_embd_h : public llm_graph_input_i {
public:
    llm_graph_input_embd_h(int64_t n_embd) : n_embd(n_embd) {}
    virtual ~llm_graph_input_embd_h() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * tokens = nullptr; // I32 [n_batch]
    ggml_tensor * embd   = nullptr; // F32 [n_embd, n_batch]
    ggml_tensor * h      = nullptr; // F32 [n_embd, n_batch]

    const int64_t n_embd = 0;
};

class llm_graph_input_pos : public llm_graph_input_i {
public:
    llm_graph_input_pos(uint32_t n_pos_per_embd) : n_pos_per_embd(n_pos_per_embd) {}
    virtual ~llm_graph_input_pos() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * pos = nullptr; // I32 [n_batch]

    const uint32_t n_pos_per_embd = 1;
};

// temperature tuning, used by llama4
class llm_graph_input_attn_temp : public llm_graph_input_i {
public:
    llm_graph_input_attn_temp(uint32_t n_attn_temp_floor_scale, float f_attn_temp_scale, float f_attn_temp_offset)
        : n_attn_temp_floor_scale(n_attn_temp_floor_scale), f_attn_temp_scale(f_attn_temp_scale), f_attn_temp_offset(f_attn_temp_offset) {}
    virtual ~llm_graph_input_attn_temp() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * attn_scale = nullptr; // F32 [n_batch]

    const uint32_t n_attn_temp_floor_scale;
    const float    f_attn_temp_scale;
    const float    f_attn_temp_offset;
};

class llm_graph_input_pos_bucket : public llm_graph_input_i {
public:
    llm_graph_input_pos_bucket(const llama_hparams & hparams) : hparams(hparams) {}
    virtual ~llm_graph_input_pos_bucket() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * pos_bucket = nullptr; // I32 [n_batch, n_batch]

    const llama_hparams hparams;
};

class llm_graph_input_pos_bucket_kv : public llm_graph_input_i {
public:
    llm_graph_input_pos_bucket_kv(
            const llama_hparams & hparams,
            const llama_kv_cache_context * mctx) : hparams(hparams), mctx(mctx) {}
    virtual ~llm_graph_input_pos_bucket_kv() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * pos_bucket = nullptr; // I32 [n_kv, n_batch]

    const llama_hparams hparams;

    const llama_kv_cache_context * mctx;
};

class llm_graph_input_out_ids : public llm_graph_input_i {
public:
    llm_graph_input_out_ids(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            uint32_t n_outputs) : hparams(hparams), cparams(cparams), n_outputs(n_outputs) {}
    virtual ~llm_graph_input_out_ids() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * out_ids; // I32 [n_outputs]

    const llama_hparams hparams;
    const llama_cparams cparams;

    const uint32_t n_outputs;
};

class llm_graph_input_mean : public llm_graph_input_i {
public:
    llm_graph_input_mean(const llama_cparams & cparams) : cparams(cparams) {}
    virtual ~llm_graph_input_mean() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * mean; // F32 [n_batch, n_batch]

    const llama_cparams cparams;
};

class llm_graph_input_cls : public llm_graph_input_i {
public:
    llm_graph_input_cls(const llama_cparams & cparams, const llm_arch arch) : cparams(cparams), arch(arch) {}
    virtual ~llm_graph_input_cls() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * cls; // I32 [n_batch]

    const llama_cparams cparams;
    const llm_arch arch;
};

class llm_graph_input_rs : public llm_graph_input_i {
public:
    llm_graph_input_rs(const llama_memory_recurrent_context * mctx) : mctx(mctx) {}
    virtual ~llm_graph_input_rs() = default;

    void set_input(const llama_ubatch * ubatch) override;
    void set_input_recurrent(
        const llama_memory_recurrent_context * current,
        const llama_ubatch * ubatch);

    bool can_reuse(const llm_graph_params & params) override;
    bool can_reuse_recurrent(
        const llama_memory_recurrent_context * current,
        const llama_ubatch & ubatch);

    ggml_tensor * s_copy;      // I32 [n_rs]
    ggml_tensor * brain_copy;  // I32 [n_seqs], grouped shared-S rows

    struct rbb_group_input {
        int32_t brain_row = -1;
        ggml_tensor * public_rows = nullptr; // I32 [n_public_rows]
    };
    std::vector<rbb_group_input> rbb_groups;

    // views of s_copy, computed once per graph
    // and shared across layers which use build_rs
    ggml_tensor * s_copy_main;   // I32 [n_seqs]
    ggml_tensor * s_copy_extra;  // I32 [n_rs - n_seqs]

    const llama_memory_recurrent_context * mctx;

    // used in view offsets, need to match for valid graph reuse
    uint32_t head;
    int32_t rs_z;
};

// Managed XKV graph input: owns bounded-hot XKV op handles through compute
// (stable custom-op userdata addresses via shared ownership), disables graph
// reuse while bounded XKV is active, and exposes post-compute validation so
// stale/codec/workspace callback status becomes retry/hard failure instead of
// a silently accepted zeroed tensor. Owned by llm_graph_result.
class llm_graph_input_xkv : public llm_graph_input_i {
public:
    explicit llm_graph_input_xkv(bool bounded_active) : bounded_active(bounded_active) {}
    ~llm_graph_input_xkv() override;
    void set_input(const llama_ubatch * ubatch) override;
    // Bounded XKV reuses the graph only when the capacity key matches and
    // every adopted snapshot proves freshness (store stamps unchanged).
    // OFF/SHADOW never install this input. Until the snapshot refresh API
    // lands (XkvRuntimeSealer), any store mutation refuses reuse by stamp.
    bool can_reuse(const llm_graph_params & params) override;
    // Capacity key for graph reuse. Exact today (bucket size 1): padded
    // snapshot refresh + bucketed builds widen these to buckets without
    // changing the comparison sites. n_kv covers storage-view shapes.
    struct xkv_reuse_key {
        uint32_t n_tokens = 0;
        int xkv_mode = 0;
        int storage_profile = 0;
        bool cpu_branch = true;
        bool operator==(const xkv_reuse_key & o) const {
            // No n_kv: bounded hot views are fixed-capacity pool views, so
            // storage shapes are stable across ubatches by pool invariant
            // (HotSafety); logical cache growth does not reshape them.
            return n_tokens == o.n_tokens && xkv_mode == o.xkv_mode &&
                storage_profile == o.storage_profile && cpu_branch == o.cpu_branch;
        }
        bool operator!=(const xkv_reuse_key & o) const { return !(*this == o); }
    };
    // Adopt one per-KV-head op handle with a status poller returning
    // 0 = ok, 1 = retry (stale stamp), 2 = hard error. validity_fn reports
    // snapshot freshness (e.g. store stamp unchanged); absent means
    // unverifiable and refuses reuse when bounded.
    void adopt(std::shared_ptr<void> handle, std::function<int()> status_fn, std::string label,
        std::function<bool()> validity_fn = {},
        std::function<void()> release_fn = {},
        std::function<bool(std::string &)> acquire_fn = {},
        // Fill hook for graph-created metadata tensors (native path): replays
        // exact precomputed bytes into set_input-marked tensors; throws on
        // capacity overflow (rebuild required, refusing to truncate).
        std::function<void()> fill_fn = {},
        llama_xkv::xkv_graph_snapshot * snapshot = nullptr);
    // Snapshot sidecar for refresh/freshness (non-owning; lifetime held by
    // handle). Passed explicitly to avoid type-erased round-trip casts.
    void note_snapshot(llama_xkv::xkv_graph_snapshot * snap);
    // Coordinator reader lease lifecycle (XkvRuntimeSealer guard): release
    // runs after final validation on EVERY poll path (ok/retry/hard) and in
    // the destructor, so a cached graph/result can never block maintenance
    // forever. Segment pins are untouched (COW-immutable). Acquire runs at
    // set_input start (fresh guard before snapshot refresh); acquire failure
    // surfaces as hard error at the next poll. Non-const: polling mutates
    // lease state.
    void release_reader_leases();
    void set_key(xkv_reuse_key key, const llama_memory_context_i * mctx) {
        key_ = key;
        key_valid_ = true;
        build_mctx_ = mctx;
    }
    // True iff every adopted op computed successfully. err carries the first
    // failure. Must be consulted after graph compute; a zeroed output tensor
    // alone is never success.
    bool postcompute_ok(std::string * err);
    int post_action();
    bool bounded() const { return bounded_active; }
    size_t n_ops() const { return entries.size(); }
private:
    struct entry {
        std::shared_ptr<void> handle;
        std::function<int()> status_fn;
        std::string label;
        std::function<bool()> validity_fn;
        std::function<void()> release_fn;
        std::function<bool(std::string &)> acquire_fn;
        std::string acquire_err;
        bool lease_outstanding = false;
        llama_xkv::xkv_graph_snapshot * snapshot = nullptr;
        std::function<void()> fill_fn;
    };
    bool bounded_active = false;
    xkv_reuse_key key_;
    bool key_valid_ = false;
    const llama_memory_context_i * build_mctx_ = nullptr;
    std::vector<entry> entries;
    // Last ubatch seen in set_input (defensive record; refresh hook point).
    uint32_t last_n_tokens_ = 0;
};

// StateCarryFix instrumented-run stash: s_copy values copied at set_input
// time (llama-graph.cpp), consumed post-fence in llama-context.cpp. Plain
// data (not tensor pointers), so no graph-lifetime coupling across TUs.
int     sc_dbg_scopy_ntokens();
int     sc_dbg_scopy_n();
int32_t sc_dbg_scopy(int i);

class llm_graph_input_cross_embd : public llm_graph_input_i {
public:
    llm_graph_input_cross_embd(
            const llama_cross * cross) : cross(cross) {}
    virtual ~llm_graph_input_cross_embd() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * cross_embd; // F32 [n_embd, n_outputs_enc]

    const llama_cross * cross;
};

class llm_graph_input_attn_no_cache : public llm_graph_input_i {
public:
    llm_graph_input_attn_no_cache(const llama_hparams & hparams, const llama_cparams & cparams) :
        hparams(hparams),
        cparams(cparams) {
    }
    ~llm_graph_input_attn_no_cache() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * get_kq_mask()     const { return self_kq_mask_cnv; }
    ggml_tensor * get_kq_mask_swa() const { return self_kq_mask_swa_cnv; }

    // n_tokens == n_batch
    ggml_tensor * self_kq_mask         = nullptr; // F32/F16 [n_tokens, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv     = nullptr; //         [n_tokens, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa     = nullptr; // F32/F16 [n_tokens, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa_cnv = nullptr; //         [n_tokens, n_batch/n_stream, 1, n_stream]

    const llama_hparams hparams;
    const llama_cparams cparams;
};

// RERoT DDVR graph input (Stage 5 remainder, §§9-12, 21).
//
// Frozen DDVR contract (owned by src/llama-rerot.* + GGML_OP_FLASH_ATTN_EXT_REROT;
// consumed here, never redefined):
// - Q arrives at the DDVR kernel PRE-PHASED per group graph-side:
//     effective = query_virtual + storage_base - virtual_base.
// - Qwen3.5 text IMRoPE is a 4-tuple (p, p, p, 0); the delta applies to the
//   first three coordinates only, the fourth stays 0.
// - RoPE runs BEFORE Turbo WHT; K storage is never moved, not one byte.
// - Kernels consume span descriptors (entries [2, E] + offsets [n_queries+1])
//   with one global online softmax per query range. Strong-frontier visibility
//   is resolved layout-side by entry pre-filtering (strong vs lag1), never by
//   the kernel and never by ordinary seq-membership masks.
// - Span metadata / offsets / visibility are INPUT TENSOR DATA, not graph
//   topology: the reuse key covers only (n_token rows, span capacity bucket,
//   kernel variant, rerot on/off, frontier mode). Changes to virtual_pos0,
//   storage_pos0, counts (within capacity) or visibility refresh tensor data
//   via fill and must not force a rebuild.
// - Pipeline-parallel lifetime: staging buffers are owned per input instance
//   (never statics); fill uploads a snapshot after the previous frontier's
//   graph has completed, following the same discipline as k_idxs/kq_mask.
//   No use-after-overwrite of span tables across frontiers.
// - First-CPU-prototype hatch: LLAMA_REROT_DISABLE_GRAPH_REUSE=1 forces a
//   rebuild whenever RERoT is active. The capacity-bucketed reuse-key design
//   stays authoritative regardless.
// - Unsupported backends (no RERoT kernel today: CUDA / Metal / RPC) take an
//   explicit capability-error path (throw), never silent stock attention.
// RERoT OFF: none of this runs; ordinary graph construction is byte-identical.
//

enum class llm_rerot_kernel_variant : uint8_t {
    REROT_KERNEL_CPU = 0,
    REROT_KERNEL_VULKAN_FUSED = 1,
    REROT_KERNEL_UNSUPPORTED = 2,
};

struct llm_rerot_span_reuse_key {
    uint32_t n_tokens  = 0;
    uint32_t group_cap = 0;
    uint32_t entry_cap = 0;
    llm_rerot_kernel_variant variant = llm_rerot_kernel_variant::REROT_KERNEL_CPU;
    bool rerot_on = false;
    llama_rerot_frontier_mode frontier_mode = LLAMA_REROT_FRONTIER_STRONG;

    bool operator==(const llm_rerot_span_reuse_key & o) const {
        return n_tokens == o.n_tokens && group_cap == o.group_cap && entry_cap == o.entry_cap &&
               variant == o.variant && rerot_on == o.rerot_on && frontier_mode == o.frontier_mode;
    }
    bool operator!=(const llm_rerot_span_reuse_key & o) const {
        return !(*this == o);
    }
};

// Backend-neutral span-descriptor lifecycle for the RERoT DDVR path.
// Converts per-reader PAC-DFS logical views (llama_rerot_set_frontier_views
// state, resolved layout-side into llama_rerot_attn_layout) into input tensor
// data: physical key index / Q-group index entries, per-group effective RoPE
// positions, and per-query entry offsets. Visibility (strong vs lag1) is
// already baked into the entry set by the layout builder.
class llm_graph_input_attn_rerot {
public:
    static constexpr uint32_t SPAN_BUCKET = 256;

    static uint32_t capacity_bucket(uint32_t n);
    static bool graph_reuse_disabled();
    static bool is_supported_arch(const llama_hparams & hparams);
    static llm_rerot_kernel_variant select_variant(ggml_backend_sched_t sched);
    static void require_supported(
            ggml_backend_sched_t sched,
            const llama_hparams & hparams,
            const char * where);
    static llm_rerot_span_reuse_key make_key(
            uint32_t n_tokens,
            uint32_t n_groups,
            uint32_t n_entries,
            llm_rerot_kernel_variant variant,
            bool rerot_on,
            llama_rerot_frontier_mode mode);

    llm_graph_input_attn_rerot() = default;

    // Allocate capacity-bucketed span tensors (topology) and record the reuse
    // key. Data is uploaded later by fill_spans, so virtual_pos0 /
    // storage_pos0 / count / visibility churn within capacity never rebuilds.
    void build_span_tensors(
            ggml_context * ctx0,
            ggml_tensor *& q_indices,
            ggml_tensor *& q_pos,
            ggml_tensor *& entries,
            ggml_tensor *& offsets,
            const llama_ubatch & ubatch,
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_context * attn,
            ggml_backend_sched_t sched);

    bool spans_can_reuse(
            const llama_ubatch & ubatch,
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_context * attn,
            ggml_backend_sched_t sched,
            const ggml_tensor * q_indices,
            const ggml_tensor * q_pos,
            const ggml_tensor * entries,
            const ggml_tensor * offsets) const;

    // Snapshot the current layout into owned staging, then upload. Only the
    // exact prefix carries live descriptors; tails are zeroed so padded Q
    // gathers stay in-bounds and deterministic. q_pos uses the tensor's own
    // group-capacity stride (coordinate k lives at k * group_cap), matching
    // how ggml_rope reads multi-pos; the IMRoPE text rule (p, p, p, 0) is
    // preserved with the fourth coordinate pinned to 0.
    void fill_spans(
            ggml_tensor * q_indices,
            ggml_tensor * q_pos,
            ggml_tensor * entries,
            ggml_tensor * offsets,
            const llama_kv_cache_context * attn,
            uint32_t n_pos);

    const llm_rerot_span_reuse_key & reuse_key() const { return key; }
    bool has_key() const { return key_valid; }
    uint64_t applied_epoch() const { return epoch; }

private:
    llm_rerot_span_reuse_key key;
    bool key_valid = false;
    uint64_t epoch = 0;

    // Pipeline-parallel lifetime: per-instance staging snapshot. Never a
    // static; never an alias of layout internals. Overwrites happen only via
    // fill_spans, ordered after the previous frontier's graph completion.
    std::vector<int32_t> st_q_indices;
    std::vector<int32_t> st_q_pos;
    std::vector<int32_t> st_entries;
    std::vector<int32_t> st_offsets;
};

// FlashPrefill V2 sparse-path graph input (GraphIntegration).
//
// Owns the GGML wire metadata tensor (I32) plus the per-group Q-position
// tensor, with per-instance CPU staging. The pool (F32 means) and plan (I32
// select output) tensors are op outputs, not inputs. No Q*K expansion is
// ever packed here: rows/uses come from the CacheFragments compact layout
// (uses/use_offsets, O(Q*F)), never from exact_rows (oracle-only).
//
// Pack order (frozen with VulkanShaders/selector): uses AND rows sorted by
// (tile_id, kv_head), secondary (source_query, fragment_id). The wire stays
// order-agnostic; the selector binary-searches these ranges and raises
// BAD_LAYOUT on violation. Only descriptors are sorted: Q rows, output IDs
// (wire source_query), recurrent state and sampling rows are never reordered.
//
// Reuse key (topology only): policy/mode/route-shape fields, role, block_k,
// capacity buckets, dtype tags, kernel variant, head/dim mapping. Epochs,
// counts within caps, stamps and physical maps are tensor DATA refreshed by
// set_input, never topology.
struct llm_graph_fp_key {
    // policy topology (frozen v1 fields affecting the sparse topology)
    int32_t  mode            = LLAMA_FLASHPREFILL_MODE_OFF;
    int32_t  tail_scope      = LLAMA_FLASHPREFILL_TAIL_LOGICAL_PROMPT;
    uint32_t block_q         = 0;
    uint32_t block_k         = 0;
    uint32_t sink_blocks     = 0;
    uint32_t window_blocks   = 0;
    uint32_t dense_tail_tiles = 0;
    uint32_t min_kv          = 0;
    uint32_t full_attn_layers = 0;
    bool     mean_correction = true;
    bool     exact_all       = false;
    // route shape
    int32_t  role          = LLAMA_FLASHPREFILL_ROLE_UNKNOWN; // single role of this ubatch (mixed => inactive)
    bool     is_rerot      = false;
    bool     reserve_sizing = false; // worst-case caps, never runs
    // capacities (bucketed)
    uint32_t n_tokens      = 0;
    uint32_t f_cap         = 0;
    uint32_t r_cap         = 0;
    uint32_t u_cap         = 0;
    uint32_t c_cap         = 0;
    uint32_t n_tiles       = 0;
    uint32_t max_sel_pair  = 0; // bucketed actual worst pair (wire: sized from worst pair)
    uint32_t u_layout_cap  = 0; // bucketed layout-uses count (pair-distribution stability)
    uint32_t n_groups      = 0;
    // head/dim/dtype mapping
    int32_t  dk = 0;
    int32_t  dv = 0;
    int32_t  n_kv_heads = 0;
    int32_t  n_q_heads  = 0;
    int32_t  k_type = -1; // ggml_type tag of the K cache tensor at build
    int32_t  v_type = -1; // ggml_type tag of the V cache tensor at build
    int32_t  n_pos  = 1;  // positions per token (model-fixed; q_pos stride)
    int32_t  backend_variant = 0; // 0 = unset, 1 = CPU reference, 2 = Vulkan fused

    // The metadata is shared by layers, but their K/V formats need not be.
    // Indexed by logical layer; (-1,-1) denotes an ineligible layer.
    std::vector<std::pair<int32_t, int32_t>> layer_kv_types;

    bool matches_layer_types(int32_t il, int32_t k, int32_t v) const {
        return il >= 0 && (size_t) il < layer_kv_types.size() &&
            layer_kv_types[il].first == k && layer_kv_types[il].second == v;
    }

    bool operator==(const llm_graph_fp_key & o) const {
        return mode == o.mode && tail_scope == o.tail_scope && block_q == o.block_q &&
               block_k == o.block_k && sink_blocks == o.sink_blocks &&
               window_blocks == o.window_blocks && dense_tail_tiles == o.dense_tail_tiles &&
               min_kv == o.min_kv && full_attn_layers == o.full_attn_layers &&
               mean_correction == o.mean_correction && exact_all == o.exact_all &&
               role == o.role && is_rerot == o.is_rerot && reserve_sizing == o.reserve_sizing &&
               n_tokens == o.n_tokens && f_cap == o.f_cap && r_cap == o.r_cap &&
               u_cap == o.u_cap && c_cap == o.c_cap && n_tiles == o.n_tiles &&
               max_sel_pair == o.max_sel_pair && u_layout_cap == o.u_layout_cap && n_groups == o.n_groups &&
               dk == o.dk && dv == o.dv && n_kv_heads == o.n_kv_heads &&
               n_q_heads == o.n_q_heads && k_type == o.k_type && v_type == o.v_type &&
               n_pos == o.n_pos && backend_variant == o.backend_variant &&
               layer_kv_types == o.layer_kv_types;
    }
    bool operator!=(const llm_graph_fp_key & o) const { return !(*this == o); }
};

// CPU-side per-build summary (no GPU sync, no KV readback). Counters mirror
// the plan-header semantics (per source_query rows, proxy fan-out uncounted).
// Ubatch-level dense reason for the summary (frozen, bounded): the PolicyCore
// route value when the whole-ubatch decision maps to one, else >= 100 (wire
// routes are single-digit). Recorded even when no companion exists, so
// metrics never mislabels a plan-less graph: 0 = sparse-active (or reserve),
// DENSE_OFF = policy off / no rows, DENSE_ROLE = mixed/ineligible roles,
// DENSE_UNSUPPORTED = backend/shape, DENSE_SHORT_CONTEXT = min_kv,
// DENSE_UNKNOWN_BOUNDARY = unknown live bounds, DENSE_CAPACITY = wire-domain
// overflow, HIGH_COST = admission bound (would-be F32 mirror),
// NO_LAYOUT = layout unavailable/ineligible without hard error.
enum llm_fp_dense_reason : int32_t {
    LLM_FP_DENSE_SPARSE_ACTIVE = 0,
    LLM_FP_DENSE_HIGH_COST     = 100, // admission bound exceeded
    LLM_FP_DENSE_NO_LAYOUT     = 101, // layout unavailable/ineligible, no hard error
    LLM_FP_DENSE_FULL_PREFIX   = 102, // every handled layer designed-dense (full prefix/SWA)
};

struct llm_graph_flashprefill_summary {
    int32_t sparse_layers = 0; // layers that built pool/select/attn
    int32_t dense_layers  = 0; // eligible-arch full-attention layers kept dense
    int32_t sparse_rows   = 0; // (query,head) rows on the sparse select path
    int32_t dense_rows    = 0; // DENSE_FORCE (exact tail) rows through the new op
    int32_t n_tiles       = 0;
    int32_t n_fragments   = 0;
    // Metadata/count caps summary of the last sparse build (bucketed caps +
    // actual counts; fit/metrics size from here, never GPU readback).
    int32_t n_rows     = 0;
    int32_t n_uses     = 0;
    int32_t n_cells    = 0;
    int32_t n_groups   = 0;
    int32_t max_sel_pair = 0;
    int32_t f_cap = 0;
    int32_t r_cap = 0;
    int32_t u_cap = 0;
    int32_t c_cap = 0;
    // Real generation cost + bytes (builder/fill only, never whole-graph
    // wall): layout build + wire pack microseconds. scratch_bytes is the
    // true pinned total, maintained incrementally with checked 64-bit math:
    // shared meta/group-input bytes ONCE (no per-layer multiplier) + the SUM
    // of every pinned per-layer plan + the MAX live pool per backend
    // (pools are transient; plans are pinned). Split-K scratch is unused
    // here (n_splits=1); any native split extra is reported separately via
    // Metrics/Vulkan. pool_bytes exposes the live-pool peak component.
    // No estimates, no readback.
    int64_t  layout_us    = 0;
    uint64_t scratch_bytes = 0;
    uint64_t pool_bytes    = 0;
    uint64_t meta_bytes    = 0; // shared meta/group inputs, counted once
    uint64_t plans_bytes   = 0; // sum of pinned per-layer plans
    // Ubatch-level dense reason, recorded even when no companion exists
    // (0 = sparse-active/reserve; route values or LLM_FP_DENSE_* above).
    int32_t  dense_reason = 0;
    // Authoritative per-ubatch route data for Metrics (never re-derived
    // from physical KV size): ubatch token count, resident+visible+legal
    // incidences, and expected sparse/forced candidate rows from the latest
    // pack — present even with zero recorded plans. Layers dense for
    // designed reasons (SWA / full-attention prefix) count separately;
    // required mode throws capability fallbacks instead of counting them.
    int32_t  ubatch_tokens = 0;
    int64_t  visible_tokens = 0;
    int32_t  expected_sparse_rows = 0;
    int32_t  expected_forced_rows = 0;
    int32_t  designed_dense_layers = 0;
    // Sum of actual n_tokens*n_head(il) over designed-dense full-prefix
    // layers (checked 64-bit, non-negative; sign-check safe). Separate from
    // new-plan forced rows (those ride dense_rows via record) to avoid
    // double count. FULL_PREFIX buckets with zero plans read this, never
    // physical KV. Additive with plan totals on partial spans (a layer is
    // either designed-dense or plan-recorded, never both).
    int64_t designed_dense_rows = 0;
};

class llm_graph_input_attn_flashprefill : public llm_graph_input_i {
public:
    llm_graph_input_attn_flashprefill(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_flashprefill() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    // Had-none stability probe: answers whether these params would carry a
    // companion, running the shared owner-building probe (prescreen, owner
    // layout build, visible/admission/tail/caps gates) so fixed-capacity
    // dense->sparse transitions rebuild instead of reusing dense forever,
    // while steady deny-states reuse without rebuild churn. Never guesses
    // from global padded KV.
    static bool wants_companion(
            const llm_graph_params & params,
            const llama_kv_cache_context * attn);

    // Build the companion once per graph (null when the sparse path is
    // inactive): live routing packs the CacheFragments compact layout into
    // wire metadata; reserve sizing allocates worst-case caps with zero
    // counts. Never expands Q*K, never touches exact_rows. Throws fail-closed
    // on corrupt/overflowing construction (any mode); returns null for
    // policy-dense (OFF, roles, bypasses, shape gates).
    // Shape-gated, never arch-whitelisted: any model whose ordinary
    // prefill matches the supported shape (GQA, F32-ready Q, single-pos,
    // no special bias) can opt in by calling try_build_attn_flashprefill;
    // other archs simply never call it and stay dense (their graphs carry
    // one unused metadata tensor only when the policy is enabled).
    static std::unique_ptr<llm_graph_input_attn_flashprefill> build_if_wanted(
            ggml_context * ctx0,
      const llama_ubatch & ubatch,
     const llama_hparams & hparams,
     const llama_cparams & cparams,
     const llama_kv_cache_context * mctx_cur,
           ggml_backend_sched_t sched,
                      bool reserve_sizing,
            llm_graph_result * res);

    bool active() const { return meta != nullptr; }
    // Reserve graphs carry worst-case caps with zero counts and never run.
    bool is_reserve() const { return active() && key.reserve_sizing; }
    // True once a sparse layer actually consumed the metadata (pool/select
    // nodes reference it). Uploads are skipped until set: an active but
    // unconsumed companion is an orphan input the scheduler never
    // allocates, and uploading to it would hit a null buffer. Set at build
    // time only; reuse replays the same graph, so the route never flips.
    bool used_by_graph = false;
    void mark_used_by_graph() { used_by_graph = true; }

    ggml_tensor * get_metadata() const { return meta; }
    ggml_tensor * get_q_pos()    const { return q_pos; }

    const llm_graph_fp_key & reuse_key() const { return key; }
    bool has_key() const { return key_valid; }

    // Cheap submit-time guards (freshness, key topology, n_kv, counts<=caps).
    // Full layout validation + checked packing run once at build (plan
    // generation); the expensive GGML metadata validator never runs per
    // layer/submit (oracle/tests only).
    bool submit_guards_ok(const llama_ubatch & ubatch, std::string * error) const;

    ggml_tensor * meta     = nullptr; // I32 [meta_words] wire metadata
    ggml_tensor * q_pos    = nullptr; // I32 [g_cap * n_pos] per-group Q positions, coord k at k*g_cap (rerot only)
    ggml_tensor * q_gather = nullptr; // I32 [g_cap] group->query gather indices (rerot only)

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_context * mctx;

private:
    llm_graph_fp_key key;
    bool key_valid = false;

    // Pipeline-parallel lifetime: per-instance staging snapshot, never a
    // static, never aliasing layout internals. Overwritten only by the fill
    // path ordered after the previous submit's graph completion (same
    // discipline as k_idxs/kq_mask/rerot spans).
    std::vector<int32_t> st_meta;
    std::vector<int32_t> st_qpos;
    std::vector<int32_t> st_gather; // RERoT group->query gather indices (ordinary: unused)

    // Freshness observed at build (topology-excluded, data-validated).
    uint64_t built_cells_epoch = 0;
    uint32_t built_n_kv = 0;

    // Current per-ubatch row snapshot, refreshed from the reuse params on
    // every can_reuse (never the stale build-time cparams copy): tail
    // coordinates/epochs/counts are submit data, not topology. Shared
    // ownership keeps the rows alive across async execution.
    std::shared_ptr<const std::vector<struct llama_flashprefill_row>> cur_rows;

    // Last submit pack outcome (plan recording + submit guards).
    int32_t last_sparse_rows = 0;
    int32_t last_forced_rows = 0;
    // Build-time caps/counts snapshot for the result summary (no sync).
    llm_graph_flashprefill_summary built_summary;
    // Accumulated layout+pack generation time, builder/fill only.
    int64_t stat_layout_us = 0;

    friend struct llm_graph_context;
    friend class llm_graph_result;
};

class llm_graph_input_attn_kv : public llm_graph_input_i {
public:
    llm_graph_input_attn_kv(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_kv() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    // FlashPrefill sparse-path companion input (null when the sparse path is
    // inactive for this graph). Built once alongside this input, shared by
    // every full-attention layer; per-layer dense/sparse choice stays in the
    // attention builder. Other memory wrappers (MSA/DSA/ISWA/DSV4) keep their
    // pre-existing inputs untouched.
    llm_graph_input_attn_flashprefill * get_fp() const { return fp.get(); }
    llm_graph_input_attn_flashprefill * get_fp_input() override { return fp.get(); }

    ggml_tensor * get_k_idxs() const { return self_k_idxs; }
    ggml_tensor * get_v_idxs() const { return self_v_idxs; }

    ggml_tensor * get_kq_mask() const { return self_kq_mask_cnv; }

    bool rerot_active() const { return self_rerot_q_indices != nullptr; }
    // Cache-state RERoT (semantic): true when the KV context carries RERoT
    // batch state, independent of whether legacy DDVR span tensors were
    // built. Out-of-line (full cache type visible in graph.cpp); model hooks
    // use this, never mctx directly (incomplete type in model TUs). Legacy
    // span-index callers keep using rerot_active() above.
    bool rerot_semantic() const;
    ggml_tensor * get_rerot_q_indices() const { return self_rerot_q_indices; }
    ggml_tensor * get_rerot_q_pos() const { return self_rerot_q_pos; }
    ggml_tensor * get_rerot_entries() const { return self_rerot_entries; }
    ggml_tensor * get_rerot_offsets() const { return self_rerot_offsets; }

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]

    ggml_tensor * self_kq_mask     = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

    ggml_tensor * self_rerot_q_indices = nullptr; // I32 [group_cap] (capacity-bucketed, see llm_graph_input_attn_rerot)
    ggml_tensor * self_rerot_q_pos     = nullptr; // I32 [group_cap*n_pos] (capacity stride, IMRoPE 4-tuple)
    ggml_tensor * self_rerot_entries   = nullptr; // I32 [2, entry_cap] (capacity-bucketed span descriptors)
    ggml_tensor * self_rerot_offsets   = nullptr; // I32 [n_queries + 1] (exact: keyed by n_token rows)

    // RERoT DDVR span lifecycle: capacity-bucketed reuse key + per-instance
    // staging for pipeline-parallel lifetime. Ordinary path never touches it.
    llm_graph_input_attn_rerot rerot_spans;

    // FlashPrefill sparse-path companion (null when inactive). Built once in
    // build_attn_inp_kv_impl, shared by every full-attention layer of this
    // graph. OFF graphs never allocate it (no extra nodes, no extra inputs).
    std::unique_ptr<llm_graph_input_attn_flashprefill> fp;

    bool rerot_spans_can_reuse(
            const llama_ubatch & ubatch,
            ggml_backend_sched_t sched,
            const llama_cparams & cparams,
            const llama_kv_cache_context * attn) const;
    void rerot_spans_fill(const llama_ubatch * ubatch, const llama_kv_cache_context * attn);

    // note: assumes v_rot^2 == I
    ggml_tensor * self_k_rot = nullptr;
    ggml_tensor * self_v_rot = nullptr;

    // note: these have to be copies because in order to be able to reuse a graph, its inputs
    //       need to carry these parameters with them. otherwise, they can point to freed
    //       llm_graph_params from a previous batch, causing stack-use-after-return
    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_context * mctx;
};

// V-less input for the KV cache
// ref: https://github.com/ggml-org/llama.cpp/pull/19067
class llm_graph_input_attn_k : public llm_graph_input_i {
public:
    llm_graph_input_attn_k(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs() const { return self_k_idxs; }

    ggml_tensor * get_kq_mask() const { return self_kq_mask_cnv; }

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]

    ggml_tensor * self_kq_mask     = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_context * mctx;
};

class llm_graph_input_attn_k_dsa : public llm_graph_input_i {
public:
    llm_graph_input_attn_k_dsa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_dsa_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k_dsa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs_mla() const { return self_k_idxs_mla; }
    ggml_tensor * get_k_idxs_lid() const { return self_k_idxs_lid; }

    ggml_tensor * get_kq_mask_mla() const { return self_kq_mask_mla_cnv; }
    ggml_tensor * get_kq_mask_lid() const { return self_kq_mask_lid; }

    ggml_tensor * self_k_idxs_mla = nullptr; // I64 [n_batch]
    ggml_tensor * self_k_idxs_lid = nullptr; // I64 [n_batch]

    ggml_tensor * self_kq_mask_mla     = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_mla_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_lid     = nullptr; // F32     [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_lid_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

    ggml_tensor * self_k_rot_lid = nullptr;

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_dsa_context * mctx;
};

// standard K/V attention input against the base cache, plus destination indices for the indexer key cache
class llm_graph_input_attn_kv_msa : public llm_graph_input_attn_kv {
public:
    llm_graph_input_attn_kv_msa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_msa_context * mctx);
    ~llm_graph_input_attn_kv_msa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs_idx() const { return self_k_idxs_idx; }

    ggml_tensor * self_k_idxs_idx = nullptr; // I64 [n_batch]

    const llama_kv_cache_msa_context * mctx_msa;
};

class llm_graph_input_attn_kv_iswa : public llm_graph_input_i {
public:
    llm_graph_input_attn_kv_iswa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_iswa_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_kv_iswa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs()     const { return self_k_idxs; }
    ggml_tensor * get_v_idxs()     const { return self_v_idxs; }
    ggml_tensor * get_k_idxs_swa() const { return self_k_idxs_swa; }
    ggml_tensor * get_v_idxs_swa() const { return self_v_idxs_swa; }

    ggml_tensor * get_kq_mask()     const { return self_kq_mask_cnv; }
    ggml_tensor * get_kq_mask_swa() const { return self_kq_mask_swa_cnv; }

    ggml_tensor * self_k_idxs     = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs     = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]
    ggml_tensor * self_k_idxs_swa = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs_swa = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]

    ggml_tensor * self_kq_mask         = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv     = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa     = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

    ggml_tensor * self_k_rot = nullptr;
    ggml_tensor * self_v_rot = nullptr;

    ggml_tensor * self_k_rot_swa = nullptr;
    ggml_tensor * self_v_rot_swa = nullptr;

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_iswa_context * mctx;
};

class llm_graph_input_attn_k_iswa : public llm_graph_input_i {
public:
    llm_graph_input_attn_k_iswa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_iswa_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k_iswa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs()     const { return self_k_idxs; }
    ggml_tensor * get_k_idxs_swa() const { return self_k_idxs_swa; }

    ggml_tensor * get_kq_mask()     const { return self_kq_mask_cnv; }
    ggml_tensor * get_kq_mask_swa() const { return self_kq_mask_swa_cnv; }

    ggml_tensor * self_k_idxs     = nullptr; // I64 [n_batch]
    ggml_tensor * self_k_idxs_swa = nullptr; // I64 [n_batch]

    ggml_tensor * self_kq_mask         = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv     = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa     = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

    ggml_tensor * self_k_rot = nullptr;
    ggml_tensor * self_k_rot_swa = nullptr;

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_iswa_context * mctx;
};

// DSV4 raw graph inputs are SWA-only, but their mask may be stream-shaped
// so raw K can be concatenated with DSV4 compressed K in one attention op.
class llm_graph_input_dsv4_raw {
public:
    llm_graph_input_dsv4_raw(
            const llama_cparams & cparams,
            const llama_kv_cache_dsv4_raw_context * mctx) :
        cparams(cparams),
        mctx(mctx) {
    }

    void set_input(const llama_ubatch * ubatch);

    ggml_tensor * get_k_idxs() const { return self_k_idxs; }
    ggml_tensor * get_kq_mask() const { return self_kq_mask_cnv; }

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]

    ggml_tensor * self_kq_mask     = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

    ggml_tensor * self_k_rot = nullptr;

    const llama_cparams cparams;

    const llama_kv_cache_dsv4_raw_context * mctx;
};

class llm_graph_input_dsv4 : public llm_graph_input_i {
public:
    struct comp_input {
        ggml_tensor * state_pos        = nullptr; // I32 [n_state]
        ggml_tensor * state_persist_src_idxs = nullptr; // I32 [n_state_persist]
        ggml_tensor * state_persist_dst_idxs = nullptr; // I32 [n_state_persist]
        ggml_tensor * state_restore_src_idxs = nullptr; // I32 [n_state_restore]
        ggml_tensor * state_restore_dst_idxs = nullptr; // I32 [n_state_restore]
        ggml_tensor * state_snapshot_src_idxs = nullptr; // I32 [n_state_snapshot]
        ggml_tensor * state_snapshot_dst_idxs = nullptr; // I32 [n_state_snapshot]
        ggml_tensor * state_read_idxs  = nullptr; // I32 [ratio*n_state_write]
        ggml_tensor * state_write_idxs = nullptr; // I64 [n_state_write]
        ggml_tensor * state_write_pos  = nullptr; // I32 [n_state_write]

        ggml_tensor * kq_mask    = nullptr; // F32 [n_kv, n_batch/n_stream, 1, n_stream]

        ggml_tensor * k_rot      = nullptr;
    };

    llm_graph_input_dsv4(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_dsv4_raw> inp_raw,
            const llama_kv_cache_dsv4_context * mctx) :
        inp_raw(std::move(inp_raw)),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_dsv4() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    llm_graph_input_dsv4_raw * get_raw() const { return inp_raw.get(); }
    const comp_input & get_csa() const { return inp_csa; }
    const comp_input & get_hca() const { return inp_hca; }
    const comp_input & get_lid() const { return inp_lid; }

    std::unique_ptr<llm_graph_input_dsv4_raw> inp_raw;

    comp_input inp_csa;
    comp_input inp_hca;
    comp_input inp_lid;

    const llama_cparams cparams;

    const llama_kv_cache_dsv4_context * mctx;
};

class llm_graph_input_attn_cross : public llm_graph_input_i {
public:
    llm_graph_input_attn_cross(const llama_cross * cross) : cross(cross) {}
    ~llm_graph_input_attn_cross() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * get_kq_mask_cross() const { return cross_kq_mask_cnv; }

    ggml_tensor * cross_kq_mask     = nullptr; // F32/F16 [n_outputs_enc, n_batch, 1, 1]
    ggml_tensor * cross_kq_mask_cnv = nullptr; // F32/F16 [n_outputs_enc, n_batch, 1, 1]

    const llama_cross * cross = nullptr;
};

class llm_graph_input_mem_hybrid : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_kv> inp_attn,
            std::unique_ptr<llm_graph_input_rs>      inp_rs,
            const llama_memory_hybrid_context *      mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_kv> inp_attn;
    std::unique_ptr<llm_graph_input_rs>      inp_rs;

    llm_graph_input_attn_kv * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs      * get_recr() const { return inp_rs.get(); }
    llm_graph_input_attn_flashprefill * get_fp_input() override {
        return inp_attn ? inp_attn->get_fp() : nullptr;
    }

    const llama_cparams cparams;

    const llama_memory_hybrid_context * mctx;
};

class llm_graph_input_mem_hybrid_k : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid_k(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_k> inp_attn,
            std::unique_ptr<llm_graph_input_rs>      inp_rs,
            const llama_memory_hybrid_context *      mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid_k() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_k> inp_attn;
    std::unique_ptr<llm_graph_input_rs>      inp_rs;

    llm_graph_input_attn_k * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs      * get_recr() const { return inp_rs.get(); }

    const llama_cparams cparams;

    const llama_memory_hybrid_context * mctx;
};

class llm_graph_input_mem_hybrid_iswa : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid_iswa(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_kv_iswa> inp_attn,
            std::unique_ptr<llm_graph_input_rs>          inp_rs,
            const llama_memory_hybrid_iswa_context *     mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid_iswa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_kv_iswa> inp_attn;
    std::unique_ptr<llm_graph_input_rs>          inp_rs;

    llm_graph_input_attn_kv_iswa * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs           * get_recr() const { return inp_rs.get(); }

    const llama_cparams cparams;

    const llama_memory_hybrid_iswa_context * mctx;
};

class llm_graph_input_sampling : public llm_graph_input_i {
public:
    llm_graph_input_sampling(std::map<llama_seq_id, llama_sampler *> samplers) :
        samplers(std::move(samplers)) { }
    virtual ~llm_graph_input_sampling() = default;

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    std::map<llama_seq_id, llama_sampler *> samplers;
};

//
// llm_graph_result
//

// these objects deliver the result from the graph build process back to the llama_context
// note that the input tensors created for the graph are referenced here - the goal is to be able to populate their
//   specific data, by calling the set_inputs() method
// along with the input tensors, the object also provides commonly used outputs tensors, such as logits, embeddings, etc.
//   these are used by the llama_context to extact the relevant data, based on the compute parameters

// callback that allows us to apply custom logic to each tensor (e.g. ggml-alloc, offloading, etc.)
using llm_graph_cb = std::function<void(const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il)>;

class llm_graph_result;

struct llm_graph_params {
    llm_arch arch = LLM_ARCH_UNKNOWN;

    llama_hparams hparams;
    llama_cparams cparams;

    llama_ubatch ubatch; // note: intentionally make a copy

    llm_graph_type gtype;

    ggml_backend_sched_t sched;
    ggml_backend_t backend_cpu;

    const llama_adapter_cvec     * cvec;
    const llama_adapter_loras    * loras;
    const llama_memory_context_i * mctx;
    const llama_cross            * cross;

    std::map<llama_seq_id, llama_sampler *> samplers;

    static bool samplers_equal(
          const std::map<llama_seq_id, llama_sampler *> & lhs,
          const std::map<llama_seq_id, llama_sampler *> & rhs) {
        if (lhs.size() != rhs.size()) {
            return false;
        }
        for (const auto & [seq_id, sampler] : lhs) {
            auto it = rhs.find(seq_id);
            if (it == rhs.end() || it->second != sampler) {
                return false;
            }
        }
        return true;
    }

    uint32_t n_outputs;

    llm_graph_cb cb;

    llm_graph_result * res;

    // FlashPrefill reserve sizing (GraphIntegration). False for every live
    // decode graph. True only for synthetic reserve/probe graphs built to
    // measure worst-case sparse capacities (StatePolicy reserve snapshot or
    // FittingIntegration probes set it). Part of the reuse key: reserve
    // graphs never alias live graphs.
    bool flashprefill_reserve_sizing = false;

    // FlashPrefill per-ubatch row snapshots compare equal (same routing
    // topology) or not. Shared ownership keeps both sides alive; null means
    // "route dense". Compares presence, size, role and known flag only:
    // seq/reader identities, logical positions and interval bounds are
    // submit DATA (refreshed per call into flags/counts), never topology, so
    // advancing chunks or new readers must not force rebuilds. Epochs/counts
    // are likewise data. The companion refreshes its owned snapshot from the
    // current params on every can_reuse before set_input.
    static bool flashprefill_rows_equal(
            const std::shared_ptr<const std::vector<struct llama_flashprefill_row>> & lhs,
            const std::shared_ptr<const std::vector<struct llama_flashprefill_row>> & rhs) {
        if (lhs == rhs) {
            return true;
        }
        if (!lhs || !rhs) {
            return false;
        }
        if (lhs->size() != rhs->size()) {
            return false;
        }
        for (size_t i = 0; i < lhs->size(); ++i) {
            const auto & a = (*lhs)[i];
            const auto & b = (*rhs)[i];
            if (a.role          != b.role ||
                a.prefill_known != b.prefill_known) {
                return false;
            }
        }
        return true;
    }

    static bool flashprefill_config_equal(
            const struct llama_flashprefill_config & a,
            const struct llama_flashprefill_config & b) {
        return a.version           == b.version           &&
               a.struct_size       == b.struct_size       &&
               a.mode              == b.mode              &&
               a.tail_scope        == b.tail_scope        &&
               a.alpha             == b.alpha             &&
               a.block_q           == b.block_q           &&
               a.block_k           == b.block_k           &&
               a.sink_blocks       == b.sink_blocks       &&
               a.window_blocks     == b.window_blocks     &&
               a.dense_tail_tiles  == b.dense_tail_tiles  &&
               a.min_kv            == b.min_kv            &&
               a.full_attn_layers  == b.full_attn_layers  &&
               a.mean_correction   == b.mean_correction   &&
               a.exact_all         == b.exact_all;
    }

    // return true if the "other" params would result in a graph with the same topology as with the current params
    //   having the same topology allows us to reuse the graph in some cases
    bool allow_reuse(const llm_graph_params & other) const {
        // first check the ubatch
        bool can_reuse_ubatch =
            ubatch.equal_seqs() == other.ubatch.equal_seqs() &&
            ubatch.n_tokens     == other.ubatch.n_tokens &&
            ubatch.n_seq_tokens == other.ubatch.n_seq_tokens &&
            ubatch.n_seqs       == other.ubatch.n_seqs &&
            ubatch.n_seqs_unq   == other.ubatch.n_seqs_unq &&
            (
                (!ubatch.token && !other.ubatch.token) ||
                (!ubatch.embd  && !other.ubatch.embd)  ||
                (ubatch.token && other.ubatch.token && ubatch.embd && other.ubatch.embd)
            );

        // when we split the batch using "equal_seqs" we have to verify that the participating sequences are the same
        //   the reason is because the set of attention streams would be different for different sequences
        if (can_reuse_ubatch && ubatch.equal_seqs()) {
            if (!ubatch.data) {
                // if the old ubatch does not own it's data, then we cannot guarantee that it is still alive, and
                //   therefore we cannot perform the sequence id check. normally should never happen
                can_reuse_ubatch = false;
            } else {
                for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                    can_reuse_ubatch &= ubatch.seq_id_unq[s] == other.ubatch.seq_id_unq[s];
                }
            }
        }

        if (!can_reuse_ubatch) {
            return false;
        }

        if (n_outputs != other.n_outputs) {
            return false;
        }

        if (!samplers_equal(samplers, other.samplers)) {
            return false;
        }

        if (samplers.size() > 0) {
            if (!ubatch.data || !other.ubatch.data) {
                return false;
            }

            // check that the outputs are the same for all samplers
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                if (ubatch.output[i]    != other.ubatch.output[i] ||
                    ubatch.seq_id[i][0] != other.ubatch.seq_id[i][0]) {
                    return false;
                }
            }
        }

        // TODO: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448035248
        if (cparams.nextn_layer_offset != other.cparams.nextn_layer_offset) {
            return false;
        }

        // FlashPrefill topology (GraphIntegration): reserve graphs never
        // alias live graphs; policy fields and per-ubatch roles/boundaries
        // select the sparse/dense topology. Epochs, counts within caps and
        // physical maps are tensor data, excluded here, refreshed per submit.
        // OFF isolation: when both sides are OFF there is no sparse topology
        // (no rows, no scratch), so a single mode check preserves the legacy
        // comparison path bit-for-bit with no multi-field validation on
        // every normal OFF decode reuse. Reserve sizing is always false OFF.
        if (cparams.flashprefill.mode == LLAMA_FLASHPREFILL_MODE_OFF &&
            other.cparams.flashprefill.mode == LLAMA_FLASHPREFILL_MODE_OFF) {
            if (flashprefill_reserve_sizing || other.flashprefill_reserve_sizing) {
                return false; // corrupt: reserve must never be set while OFF
            }
        } else {
            if (flashprefill_reserve_sizing != other.flashprefill_reserve_sizing) {
                return false;
            }

            if (!flashprefill_config_equal(cparams.flashprefill, other.cparams.flashprefill)) {
                return false;
            }

            if (!flashprefill_rows_equal(cparams.flashprefill_rows, other.cparams.flashprefill_rows)) {
                return false;
            }
        }

        return
            cparams.embeddings              == other.cparams.embeddings              &&
            cparams.embeddings_nextn        == other.cparams.embeddings_nextn        &&
            cparams.embeddings_nextn_masked == other.cparams.embeddings_nextn_masked &&
            cparams.causal_attn             == other.cparams.causal_attn             &&
            arch  == other.arch  &&
            gtype == other.gtype &&
            cvec  == other.cvec  &&
            loras == other.loras &&
            cross == other.cross;
    }
};

struct llm_graph_fused_node {
    llm_fused_op op;
    ggml_tensor * tensor;
    int il;
};

class llm_graph_result {
public:
    llm_graph_result(int64_t max_nodes);

    virtual ~llm_graph_result() = default;

    ggml_tensor * get_inp_tokens()  const { return t_inp_tokens; }
    ggml_tensor * get_logits()      const { return t_logits; }
    ggml_tensor * get_embd()        const { return t_embd; }
    ggml_tensor * get_embd_pooled() const { return t_embd_pooled; }
    ggml_tensor * get_h_nextn()     const { return t_h_nextn; }

    ggml_tensor * get_layer_inp      (int il) const { return t_layer_inp[il]; }
    ggml_tensor * get_attn_q_pre_rope(int il) const { return t_attn_q_pre_rope[il]; }

    ggml_cgraph  * get_gf()  const { return gf; }
    ggml_context * get_ctx() const { return ctx_compute.get(); }

    int64_t get_max_nodes() const;

    void reset();

    void set_inputs(const llama_ubatch * ubatch);
    void set_outputs(const llm_graph_params & params);

    // try to update the existing graph result using the new graph parameters in order to reuse it
    // this can only be done if we determine that the resulting graph using the new graph parameters
    //   would be identical to the existing graph. in that case, we simply have to update the memory
    //   contexts of the input tensors of the graph and we can reuse it for another computation
    // return true if the graph was updated and can be reused
    bool can_reuse(const llm_graph_params & params);

    llm_graph_input_i * add_input(llm_graph_input_ptr input);

    void add_fused_node(llm_graph_fused_node result);

    const std::vector<llm_graph_fused_node> & get_fused_nodes() const { return fused_nodes; }

    // FlashPrefill plan nodes (GraphIntegration; consumed by
    // ContextIntegration/ServerRouting/MetricsIntegration, never edited by
    // them). One SELECT-output plan tensor per sparse layer, in build order.
    // Stats (incl. the plan-header error word) are read once at the graph
    // completion boundary via ggml_flashprefill_plan_get_stats, never synced
    // per layer. CPU-side build routing counts need no sync at all.
    // Records one sparse layer: pins the plan (completion-boundary read),
    // accumulates layer/row counts, adds this plan's real bytes to the
    // scratch total and max-tracks this pool's bytes per backend. Pool may
    // be null only when the plan carries no pool (never in this slice).
    void record_flashprefill_plan(ggml_tensor * plan, ggml_tensor * pool, ggml_backend_t backend,
            int il, int32_t sparse_rows, int32_t dense_rows);

    const std::vector<ggml_tensor *> & get_flashprefill_plans() const { return flashprefill_plans; }
    const std::vector<int> & get_flashprefill_plan_ils() const { return flashprefill_plan_ils; }
    const llm_graph_flashprefill_summary & get_flashprefill_summary() const { return flashprefill_summary; }

    int xkv_poll_postcompute(std::string * err);
    bool xkv_has_bounded() const;
    void set_params(const llm_graph_params & params);

    // important graph nodes
    ggml_tensor * t_inp_tokens  = nullptr;
    ggml_tensor * t_inp_embd    = nullptr; // [n_embd_inp, n_tokens]
    ggml_tensor * t_logits      = nullptr;
    ggml_tensor * t_embd        = nullptr;
    ggml_tensor * t_embd_pooled = nullptr;
    ggml_tensor * t_h_nextn     = nullptr; // [n_embd, n_outputs] hidden state before final output norm

    std::vector<ggml_tensor *> t_layer_inp;
    std::vector<ggml_tensor *> t_attn_q_pre_rope;

    std::map<llama_seq_id, ggml_tensor *> t_sampled_logits;
    std::map<llama_seq_id, ggml_tensor *> t_candidates;
    std::map<llama_seq_id, ggml_tensor *> t_sampled;
    std::map<llama_seq_id, ggml_tensor *> t_sampled_probs;

    std::vector<llm_graph_input_ptr> inputs;
    std::vector<llm_graph_fused_node> fused_nodes;

    // FlashPrefill per-graph plan state (see record/get accessors above).
    // Cleared by reset(); accumulated during graph construction only.
    std::vector<ggml_tensor *> flashprefill_plans;
    std::vector<int> flashprefill_plan_ils;
    // Per-backend live-pool peaks (backend, bytes) backing summary.pool_bytes.
    std::vector<std::pair<ggml_backend_t, uint64_t>> flashprefill_pool_peaks;
    llm_graph_flashprefill_summary flashprefill_summary;

    ggml_context_ptr ctx_compute;

    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;

    ggml_cgraph * gf;

    int64_t max_nodes;

private:
    // keep a copy of the previous graph parameters
    // we will use this to determine whether the graph can be reused by comparing them with the new parameters
    // note: these are updated after constructing the new graph
    llm_graph_params params;

    // env: LLAMA_GRAPH_RESULT_DEBUG
    int debug = 0;
};

using llm_graph_result_ptr = std::unique_ptr<llm_graph_result>;

//
// llm_graph_context
//

// used in build_rs to properly order writes and avoid unnecessary copies
using llm_graph_get_rows_fn = std::function<ggml_tensor * (ggml_context *, ggml_tensor * states, ggml_tensor * ids)>;

struct llm_graph_qkv {
    ggml_tensor * q; // [n_embd_head, n_head,    n_tokens]
    ggml_tensor * k; // [n_embd_head, n_head_kv, n_tokens]
    ggml_tensor * v; // [n_embd_head, n_head_kv, n_tokens]
};

struct llm_graph_context {
    const llm_arch arch;

    const llama_hparams & hparams;
    const llama_cparams & cparams;
    const llama_ubatch  & ubatch;

    const int64_t n_embd;
    const int64_t n_layer;
    const int64_t n_layer_nextn;
    const int64_t n_rot;
    const int64_t n_ctx;       // user-specified context size (can be different from n_ctx_train)
    const int64_t n_head;
    const int64_t n_head_kv;
    const int64_t n_embd_head_k;
    const int64_t n_embd_k_gqa;
    const int64_t n_embd_head_v;
    const int64_t n_embd_v_gqa;
    const int64_t n_expert;
    const int64_t n_expert_used;

    const float freq_base;
    const float freq_scale;
    const float ext_factor;
    const float attn_factor;
    const float beta_fast;
    const float beta_slow;
    const float norm_eps;
    const float norm_rms_eps;

    const int64_t n_tokens;
    const int64_t n_outputs;
    const int32_t n_ctx_orig; // yarn

    const enum llama_pooling_type pooling_type;
    const enum llama_rope_type    rope_type;

    ggml_backend_sched_t sched;

    ggml_backend_t backend_cpu; // TODO: needed by build_attn_mha, figure out a way to remove?

    const llama_adapter_cvec     * cvec;
    const llama_adapter_loras    * loras;
    const llama_memory_context_i * mctx;
    const llama_cross            * cross;

    // FlashPrefill reserve sizing (GraphIntegration): mirrors
    // llm_graph_params.flashprefill_reserve_sizing for input construction.
    const bool fp_reserve_sizing;

    std::map<llama_seq_id, llama_sampler *> samplers;

    const llm_graph_cb & cb_func;

    llm_graph_result * res;

    ggml_context * ctx0 = nullptr;
    ggml_cgraph  * gf   = nullptr;

    llm_graph_context(const llm_graph_params & params);
    virtual ~llm_graph_context() = default;

    void cb(ggml_tensor * cur, const char * name, int il) const;

    //
    // common
    //

    ggml_tensor * build_cvec(
             ggml_tensor * cur,
                     int   il) const;

    // do mat_mul, while optionally apply lora and per-tensor scale
    ggml_tensor * build_lora_mm(
              ggml_tensor * w,
              ggml_tensor * cur,
              ggml_tensor * w_s = nullptr,
            enum ggml_prec   prec = GGML_PREC_DEFAULT) const;

    // do mat_mul_id, while optionally apply lora and per-expert scale
    ggml_tensor * build_lora_mm_id(
              ggml_tensor * w,   // ggml_tensor * as
              ggml_tensor * cur, // ggml_tensor * b
              ggml_tensor * ids,
              ggml_tensor * w_s = nullptr) const;

    ggml_tensor * build_norm(
             ggml_tensor * cur,
             ggml_tensor * mw,
             ggml_tensor * mb,
           llm_norm_type   type,
                     int   il) const;


    // compute Q, K, V projections with optional bias and reshape
    // supports both fused wqkv and separate wq/wk/wv paths
    llm_graph_qkv build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il) const;

    ggml_tensor * build_ffn(
             ggml_tensor * cur,
             ggml_tensor * up,
             ggml_tensor * up_b,
             ggml_tensor * up_s,
             ggml_tensor * gate,
             ggml_tensor * gate_b,
             ggml_tensor * gate_s,
             ggml_tensor * down,
             ggml_tensor * down_b,
             ggml_tensor * down_s,
             ggml_tensor * act_scales,
         llm_ffn_op_type   type_op,
       llm_ffn_gate_type   type_gate,
                     int   il) const;

    // build MoE FFN without bias tensors
    ggml_tensor * build_moe_ffn(
             ggml_tensor * cur,
             ggml_tensor * gate_inp,
             ggml_tensor * up_exps,
             ggml_tensor * gate_exps,
             ggml_tensor * down_exps,
             ggml_tensor * exp_probs_b,
                 int64_t   n_expert,
                 int64_t   n_expert_used,
         llm_ffn_op_type   type_op,
                    bool   norm_w,
                   float   w_scale,
            llama_expert_gating_func_type gating_op,
                     int   il,
             ggml_tensor * probs_in = nullptr,
             ggml_tensor * gate_up_exps = nullptr,
             ggml_tensor * up_exps_s = nullptr,
             ggml_tensor * gate_exps_s = nullptr,
             ggml_tensor * down_exps_s = nullptr,
             ggml_tensor * selected_experts_in = nullptr) const;

    ggml_tensor * build_moe_ffn(
             ggml_tensor * cur,
             ggml_tensor * gate_inp,
             ggml_tensor * gate_inp_b,
             ggml_tensor * up_exps,
             ggml_tensor * up_exps_b,
             ggml_tensor * gate_exps,
             ggml_tensor * gate_exps_b,
             ggml_tensor * down_exps,
             ggml_tensor * down_exps_b,
             ggml_tensor * exp_probs_b,
                 int64_t   n_expert,
                 int64_t   n_expert_used,
         llm_ffn_op_type   type_op,
                    bool   norm_w,
                   float   w_scale,
            llama_expert_gating_func_type gating_op,
                     int   il,
             ggml_tensor * probs_in = nullptr,
             ggml_tensor * gate_up_exps = nullptr,
             ggml_tensor * gate_up_exps_b = nullptr,
             ggml_tensor * up_exps_s = nullptr,
             ggml_tensor * gate_exps_s = nullptr,
             ggml_tensor * down_exps_s = nullptr,
             ggml_tensor * selected_experts_in = nullptr) const;

    //
    // inputs
    //

    ggml_tensor * build_inp_embd(ggml_tensor * tok_embd) const;
    ggml_tensor * build_inp_pos() const;
    ggml_tensor * build_inp_attn_scale() const;
    ggml_tensor * build_inp_out_ids() const;
    ggml_tensor * build_inp_mean() const;
    ggml_tensor * build_inp_cls() const;

    ggml_tensor * build_inp_cross_embd() const;
    ggml_tensor * build_inp_pos_bucket_enc() const;
    ggml_tensor * build_inp_pos_bucket_dec() const;
    ggml_tensor * build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const;

    //
    // attention
    //

    ggml_tensor * build_attn_mha(
            ggml_tensor * q,       // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k,       // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v,       // [n_embd_head_v, n_head_v, n_tokens] (v_trans = false)
            ggml_tensor * kq_b,
            ggml_tensor * kq_mask,
            ggml_tensor * sinks,   // [n_head_q]
            ggml_tensor * v_mla,   // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_no_cache * build_attn_inp_no_cache() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_no_cache * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_kv * build_attn_inp_kv() const;

    ggml_tensor * build_rerot_q_groups(
            llm_graph_input_attn_kv * inp,
            ggml_tensor * q_raw,
            ggml_tensor * freq_factors,
            int sections[GGML_MROPE_SECTIONS],
            int il) const;

    ggml_tensor * build_attn_rerot(
            llm_graph_input_attn_kv * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_groups,
            ggml_tensor * k_cur,
            ggml_tensor * v_cur,
            ggml_tensor * sinks,
                  float   kq_scale,
                    int   il) const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_kv * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v] // TODO: remove
                  float   kq_scale,
                    int   il) const;

    // FlashPrefill V2 sparse attention (GraphIntegration).
    //
    // Returns the pre-gate attention output for layer il through the new
    // pool/select/attn ops, or nullptr when this layer must keep the
    // pre-existing dense path. Null is also returned for reserve graphs
    // (reserve carries worst-case sparse CAPACITIES in the fp input, while
    // routing stays dense there; reserve measurement never executes).
    //
    // Contract (explicit Q forms, never re-RoPEd here): ordinary route
    // consumes q_roped (model-rope-applied Q, single group per query) plus
    // the same WHT; RERoT route consumes q_raw (pre-RoPE) through the
    // existing rawQ->phase grouping once, BEFORE Turbo WHT. k_roped is
    // storage-domain K (RoPE applied, never moved, never re-phased); v is
    // raw V. Order preserved: RoPE->WHT->dot, InnerQ scale, V inverse-WHT,
    // asymmetric V trim, k/v_rot; LoRA/gating stay in the caller-shared
    // tail. One global softmax per (query,head) row. The model hook passes
    // explicitly raw (rerot branch) or explicitly roped (ordinary branch) Q;
    // a missing required form throws fail-closed. Generic build_attn (Q
    // already roped) is untouched.
    //
    // Error discipline (frozen): corrupt layout, invalid metadata, stale
    // physical mapping and cross-reader corruption THROW in every mode
    // (never a correctness fallback). Capacity overflow beyond the wire
    // domain and unsupported backends/unavailable layouts throw in REQUIRED
    // and route dense in AUTO. Designed policy-dense reasons (roles, short
    // context, dense tail, full-attention prefix, SWA/recurrent/MTP paths,
    // special bias) stay dense in every mode. All throws happen
    // before graph execution.
    ggml_tensor * try_build_attn_flashprefill(
            llm_graph_input_attn_kv * inp,
            ggml_tensor * q_raw_or_null,
            ggml_tensor * q_roped_or_null,
            ggml_tensor * k_roped,
            ggml_tensor * v,
            int *         sections_or_null,
            ggml_tensor * kq_b,
            ggml_tensor * sinks,
                  float   kq_scale,
                    int   il) const;

    // Counts eligible FULL-attention layers below il (recurrent, SWA and
    // structurally incompatible layers never count). full_attn_layers keeps
    // the FIRST N eligible layers dense by this index, not by model index;
    // hybrid models supported.
    int flashprefill_eligible_full_index(int il) const;

    // True for the StatePolicy reserve snapshot (rows present, source map
    // absent, all PREFILL/seq0/unknown-pos/known=false): route dense, size
    // worst-case sparse caps natively. Live unknown-boundary rows (source
    // map present) also route dense but size nothing. Never guesses known.
    bool flashprefill_is_reserve_snapshot() const;

    // CPU + Vulkan run the sparse kernels; every other backend (CUDA, Metal,
    // RPC, ...) is unsupported (dense in AUTO, throw in REQUIRED). Mirrors
    // the RERoT backend scan; the native kernel matrix itself is owned by
    // VulkanDispatch/CpuKernels.
    bool flashprefill_backend_supported() const;

    llm_graph_input_attn_k  * build_attn_inp_k() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_k * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_k_dsa * build_attn_inp_k_dsa() const;

    llm_graph_input_attn_kv_msa * build_attn_inp_kv_msa(bool msa_enabled) const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_k_dsa * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
            ggml_tensor * top_k, // [n_indexer_top_k, n_tokens]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_kv_iswa * build_attn_inp_kv_iswa() const;

    llm_graph_input_dsv4 * build_inp_dsv4() const;

    // note: if k_cur or v_cur are not provided, they will not be stored in the memory
    ggml_tensor * build_attn(
            llm_graph_input_attn_kv_iswa * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens] optional
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens] optional
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_k_iswa * build_attn_inp_k_iswa() const;

    // note: if k_cur is not provided, it will not be stored in the memory
    // note: the K cache is used as V (MLA-style attention)
    ggml_tensor * build_attn(
            llm_graph_input_attn_k_iswa * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens] optional
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens] optional
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_cross * build_attn_inp_cross() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_cross * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    //
    // recurrent
    //

    // TODO: move this implementation to llama_memory_recurrent.
    //       this is analogous to llama_kv_cache::cpy_k / cpy_v
    //       when moving, avoid passing `ggml_cgraph` - only pass `ggml_context`. would likely need to split the
    //         implementation in 2 separate methods. the goal is to avoid calling `ggml_build_forward_expand` in
    //         `llama_memory_recurrent`
    ggml_tensor * build_rs(
            ggml_tensor * s,
            ggml_tensor * state_copy_main,
            ggml_tensor * state_copy_extra,
                int32_t   state_size,
                int32_t   n_seqs,
               uint32_t   n_rs,
               uint32_t   rs_head,
               uint32_t   rs_size,
                int32_t   rs_zero,
            const llm_graph_get_rows_fn & get_state_rows = ggml_get_rows) const;

    llm_graph_input_rs * build_rs_inp() const;

    ggml_tensor * build_rs(
            llm_graph_input_rs * inp,
            ggml_tensor * s,
                int32_t   state_size,
                int32_t   n_seqs,
            const llm_graph_get_rows_fn & get_state_rows = ggml_get_rows) const;

    ggml_tensor * build_rs_shared(
            llm_graph_input_rs * inp,
            ggml_tensor * s,
                int32_t   state_size,
                int32_t   n_seqs) const;

    ggml_tensor * build_rwkv_token_shift_load(
        llm_graph_input_rs * inp,
        const llama_ubatch & ubatch,
                       int   il) const;

    ggml_tensor * build_rwkv_token_shift_store(
             ggml_tensor * token_shift,
      const llama_ubatch & ubatch,
                     int   il) const;
    //
    // hybrid
    //

    llm_graph_input_mem_hybrid * build_inp_mem_hybrid() const;
    llm_graph_input_mem_hybrid_k * build_inp_mem_hybrid_k() const;

    llm_graph_input_mem_hybrid_iswa * build_inp_mem_hybrid_iswa() const;

    //
    // pooling
    //

    void build_pooling(
            ggml_tensor * cls,
            ggml_tensor * cls_b,
            ggml_tensor * cls_out,
            ggml_tensor * cls_out_b,
            ggml_tensor * cls_norm) const;

    //
    // sampling (backend sampling)
    //

    void build_sampling() const;

    //
    // dense (out)
    //

    void build_dense_out(
            ggml_tensor * dense_2,
            ggml_tensor * dense_2_b,
            ggml_tensor * dense_3) const;
};

// TODO: better name
int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional);
