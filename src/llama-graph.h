#pragma once

#include "llama-arch.h"
#include "llama-batch.h"
#include "llama-hparams.h"
#include "llama-adapter.h"
#include "llama-attention-view.h"

#include <cstdint>
#include <vector>
#include <memory>
#include <set>
#include <functional>
#include <map>
#include <utility>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

struct llama_cparams;
struct llama_layer;

struct llama_memory_context_i;
// Internal marker in the unused final op-parameter slot of LLAMA reshapes.
static constexpr int32_t LLM_TOKEN_AXIS_MARKER = 0x74534b00;

// Internal marker in the same slot for the Q/K/V views of a fused `wqkv`
// projection. Each is a non-contiguous row slice of one fused matmul, so the
// token lane's shape pass cannot bind it and refuses the graph. The tag makes
// that refusal carry its own reason instead of the shared shape-failure
// counter. It is written only when the token lane is armed (`strict_ntok`).
static constexpr int32_t LLM_FUSED_QKV_SLICE_MARKER = 0x74535100;


class llama_kv_cache_context;
class llama_kv_cache_dsa_context;
class llama_kv_cache_msa_context;
class llama_kv_cache_dsv4_raw_context;
class llama_kv_cache_dsv4_context;

class llama_kv_cache_iswa_context;
class llama_memory_recurrent_context;
class llama_memory_hybrid_context;
class llama_memory_hybrid_iswa_context;

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

    virtual void set_input(const llama_ubatch * ubatch) = 0;

    // return true if the resulting input tensors using the provided graph parameters would be
    //   the same as the previous input tensors that we have currently stored in the object
    virtual bool can_reuse(const llm_graph_params & params) {
        // returning false here by default will prevent from reusing the graph if the check
        //   for the input type has not been implemented yet
        GGML_UNUSED(params);
        return false;
    }

    // union-graph runtime binder: refresh this input's cached runtime
    // metadata after an in-place graph patch so the next can_reuse() sees the
    // patched graph as current. Default: nothing cached.
    virtual void union_bind_runtime() {}

    // union-graph runtime binder: true when this input carries per-frontier
    // reader metadata that an in-place tensor patch does not refresh.
    virtual bool union_has_readers() const { return false; }

    // union-graph runtime binder: the n_outputs width this input's closure
    // executes at, or 0 when this input has no output-gather closure. The
    // binder reads it after a bind to prove the n_outputs delta was applied.
    virtual uint32_t union_bound_n_outputs() const { return 0; }

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
            uint32_t n_outputs) : hparams(hparams), cparams(cparams), n_outputs(n_outputs), bound_outputs(n_outputs) {}
    virtual ~llm_graph_input_out_ids() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    // union-graph runtime binder: refresh the cached bound width from the
    // tensor after an in-place patch, so the next can_reuse() compares the
    // active count against what the graph actually executes
    void union_bind_runtime() override;

    // union-graph runtime binder: the n_outputs target this input's closure
    // must execute at, or 0 when this graph has no output-gather closure.
    // llama_context reads it after a bind to prove that one bind applied the
    // n_outputs delta together with the n_kv and rs_z deltas.
    uint32_t union_bound_n_outputs() const override;

    ggml_tensor * out_ids; // I32 [n_outputs]

    const llama_hparams hparams;
    const llama_cparams cparams;

    // build-time allocation envelope (out_ids capacity)
    const uint32_t n_outputs;

    // width the executed out_ids/gather closure is bound at; equals n_outputs
    // until the first successful runtime bind
    uint32_t bound_outputs;
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

    bool can_reuse(const llm_graph_params & params) override;
    void union_bind_runtime() override;

    ggml_tensor * s_copy;  // I32 [n_rs]

    // views of s_copy, computed once per graph
    // and shared across layers which use build_rs
    ggml_tensor * s_copy_main;   // I32 [n_seqs]
    ggml_tensor * s_copy_extra;  // I32 [n_rs - n_seqs]

    // union-graph runtime binder: I32 [1] zero-row index. When present, the
    // zero row is selected at execution time by an in-place CPU op whose
    // output is the source of both state gathers (explicit dependency), so a
    // changed rs_z never changes the graph topology and -1 is a no-op.
    ggml_tensor * s_zero = nullptr;

    // set by set_input() when the zero row could not be bound for the last
    // ubatch (its sequences disagree on the rollback depth, so one row cannot
    // serve them all). can_reuse() then refuses this graph, so the next build
    // is stock and each sequence's row is resolved by the stock view.
    bool rs_zero_row_refused = false;

    const llama_memory_recurrent_context * mctx;

    // used in view offsets, need to match for valid graph reuse
    uint32_t head;
    int32_t rs_z;

    // build-time skeleton width; unlike s_copy_main->ne[0] this is not patched
    // by the runtime binder, so a changed n_seqs forces a rebuild
    uint32_t n_seqs = 0;

    // DRC phase 2: pending-replay length baked into this graph's topology (an extra
    // ggml_gated_delta_net(K=1) reconstruction per replay step) -- must match for valid reuse,
    // same as head/rs_z above.
    uint32_t replay_len = 0;
};

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

class llm_graph_input_attn_kv : public llm_graph_input_i {
public:
    struct reader_span_input {
        ggml_tensor * indices;
        ggml_tensor * position;
        llama_attention_span_resolved values;
    };
    struct reader_input {
        ggml_tensor * indices;
        std::vector<int32_t> values;
        std::vector<reader_span_input> spans;
    };
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
    bool union_has_readers() const override { return !readers.empty(); }

    void set_reader_input();
    std::vector<reader_input> readers;

    ggml_tensor * get_k_idxs() const { return self_k_idxs; }
    ggml_tensor * get_v_idxs() const { return self_v_idxs; }

    ggml_tensor * get_kq_mask() const { return self_kq_mask_cnv; }

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]

    ggml_tensor * self_kq_mask     = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

    // The runtime n_kv input fills this mask. Then set_input() must not fill
    // it again at the tensor's own width: a second fill would overwrite the
    // runtime width with the full width. register_nkv_runtime_mask sets this
    // flag only when a runtime input exists.
    bool kq_mask_runtime_owned = false;

    // note: assumes v_rot^2 == I
    ggml_tensor * self_k_rot = nullptr;
    ggml_tensor * self_v_rot = nullptr;

    // factor-prefab: padded KV width the graph is compiled for (0 = stock behavior:
    // K/V views sized by the exact n_kv). Set only by llm_graph_input_prefab_attn_kv;
    // build_attn reads it to size the K/V views. n_kv is a runtime parameter for such
    // graphs: the mask covers [0, n_kv_pad) and the pad columns are masked out.
    uint32_t n_kv_pad = 0;
    // note: these have to be copies because in order to be able to reuse a graph, its inputs
    //       need to carry these parameters with them. otherwise, they can point to freed
    //       llm_graph_params from a previous batch, causing stack-use-after-return
    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_context * mctx;
};

// Runtime n_kv width of a graph whose attention tensors are bound in place by
// the union binder (GGML_TP5_UNION_BINDER=1).
//
// The tensor is an I32 [1] graph input: its VALUE is the width the attention
// executes at, and it is rewritten before every compute from the KV cache's
// true used span. It is deliberately not a shape: the tensor is [1] at every
// width, so adding it cannot change the graph's topology and a width change
// never rebuilds the graph.
//
// It IS consumed, by the host: set_input() writes the width, reads it back
// from the tensor, and fills every registered KQ mask at that width
// (llama_kv_cache::set_input_kq_mask(..., n_kv)). No ggml kernel reads it.
// The K/V views are not in this list: their width is the binder's in-place
// patch of the view tensors themselves, which is a separate mechanism.
class llm_graph_input_nkv_runtime : public llm_graph_input_i {
public:
    llm_graph_input_nkv_runtime(
            const llama_kv_cache_context * mctx,
            bool causal_attn) : mctx(mctx), causal_attn(causal_attn) {}
    ~llm_graph_input_nkv_runtime();

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * n_kv = nullptr; // I32 [1]: the width to execute at

    // Host storage for n_kv. This tensor has NO graph consumer -- the width is
    // read by host code (the mask fill) -- so the scheduler never allocates it
    // and ggml_backend_tensor_set() aborts on "tensor buffer not set". The
    // input owns the buffer for exactly as long as the graph result owns the
    // input.
    struct ggml_backend_buffer * n_kv_buf = nullptr;

    // KQ masks whose fill takes its width from n_kv. Registered by
    // build_attn_inp_kv_impl after the masks exist.
    std::vector<ggml_tensor *> kq_masks;

    // The row the fill produced, copied out at fill time. This copy is a
    // witness, not the read-back's data source: the mask is flagged
    // GGML_TENSOR_FLAG_OUTPUT at registration, so ggml-alloc keeps its block
    // and mask->data still holds the row the fill wrote when the read-back
    // runs after llama_decode() returns. The snapshot only lets the trace say
    // whether the live row still matches the bytes the fill produced.
    std::vector<float> mask_row_snapshot;
    uint32_t           mask_row_snapshot_tok = 0;

    // What the last set_input() actually did. This is the observation a test
    // reads: `mask_n_kv` is the width the mask fill consumed, read back from
    // the input tensor rather than taken from the width the caller asked
    // for; `mask_filled` counts the registered masks that were filled at
    // that width. A fill that refused (non-host buffer, unwritten width, a
    // width above the mask's own ne[0]) leaves mask_filled at 0 and
    // mask_refused at 1, so a graph whose mask is stale cannot report a
    // successful fill.
    //
    // `mask_row_checked` counts the ROWS read back against the KV cells
    // (llama_kv_cache::check_mask_row) and `mask_row_ok` the rows that agreed
    // at every column below the width. These two are the independent
    // observation: mask_n_kv says what width the fill was told, mask_row_ok
    // says the mask the graph will execute matches the cells that will be
    // attended. A stale mask is checked > 0 with ok < checked.
    //
    // `mask_row_refused` counts rows the check declined to model (a layout it
    // cannot derive). It is separate from a disagreement on purpose: a layout
    // change that makes every check refuse must not look like a wrong mask.
    //
    // `mask_rows_all` is 1 only when every row the batch owns was actually
    // examined: the rows the loop compared (agreed + disagreed + refused,
    // summed over every mask it checked) equal n_tokens * mask_filled. It is
    // measured from those counters after the loop, not inferred from the
    // row-cap decision, so it reads 0 when the batch was wider than the cap,
    // when a filled mask was not checked at all (F16, no host data), and when
    // nothing was filled. 0 means "not every row was looked at"; it never
    // means "there was nothing to look at".
    uint32_t mask_n_kv         = 0;
    uint32_t mask_filled       = 0;
    uint32_t mask_refused      = 0;
    uint32_t mask_row_checked  = 0;
    uint32_t mask_row_ok       = 0;
    uint32_t mask_row_refused  = 0;
    uint32_t mask_rows_all     = 0;

    const llama_kv_cache_context * mctx;
    const bool causal_attn;
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
    void union_bind_runtime() override;
    bool union_has_readers() const override;

    std::unique_ptr<llm_graph_input_attn_kv> inp_attn;
    std::unique_ptr<llm_graph_input_rs>      inp_rs;

    llm_graph_input_attn_kv * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs      * get_recr() const { return inp_rs.get(); }

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
    void union_bind_runtime() override;
    bool union_has_readers() const override;

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
    void union_bind_runtime() override;
    bool union_has_readers() const override;

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

struct llm_graph_fused_node {
    llm_fused_op op;
    ggml_tensor * tensor;
    int il;
};


// callback that allows us to apply custom logic to each tensor (e.g. ggml-alloc, offloading, etc.)
using llm_graph_cb = std::function<void(const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il)>;
class llama_context;

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

    // active output count of this ubatch
    uint32_t n_outputs;

    // allocation envelope for the output-gather closure: 0 = build at
    // n_outputs, otherwise build the graph at this width and bind the active
    // count at execution time. The union binder sets it to max(active,
    // high-water) so the envelope covers only requested widths. This is a
    // derived build key, never a reuse key.
    uint32_t n_outputs_build = 0;

    // Enables the host runtime zero-row input and recurrent envelope build.
    bool union_binder = false;
    bool strict_ntok = false;

    // Carry the I32 zero-row index (union_rs_zero_index) in this build. True
    // on B's binder lane and, independently, on the tp5-relay lane: relay
    // never passes the binder gate, yet the rs row must be a runtime value
    // there too. It is a BUILD key (the tensor's presence changes the node
    // set) and never a reuse key: the row VALUE is bound per set_inputs().
    bool rs_row_armed = false;

    llm_graph_cb cb;

    llm_graph_result * res;

    // return true if the "other" params would result in a graph with the same topology as with the current params
    //   having the same topology allows us to reuse the graph in some cases
    // ignore_runtime skips the runtime-bound keys (n_outputs) so the binder can
    // validate every non-runtime key before patching them in place
    bool allow_reuse(const llm_graph_params & other, bool ignore_runtime = false) const {
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

        if (!ignore_runtime && n_outputs != other.n_outputs) {
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

        return
            cparams.embeddings              == other.cparams.embeddings              &&
            cparams.embeddings_nextn        == other.cparams.embeddings_nextn        &&
            cparams.embeddings_nextn_masked == other.cparams.embeddings_nextn_masked &&
            cparams.mtp_chain               == other.cparams.mtp_chain               &&
            cparams.causal_attn             == other.cparams.causal_attn             &&
            arch  == other.arch  &&
            gtype == other.gtype &&
            cvec  == other.cvec  &&
            loras == other.loras &&
            cross == other.cross;
    }
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

    ggml_tensor * get_layer_inp(int il) const { return t_layer_inp[il]; }

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


    // build-time (envelope) output count this graph was built for
    uint32_t outputs_envelope() const;

    // true when the graph has an output-gather closure (an out_ids input),
    // independent of its width: a zero-width closure is still present and
    // must not be confused with "no closure" (both report envelope 0).
    bool has_out_ids() const;
    // width the output-gather closure currently executes at (0 when the graph
    // has no out_ids input). Equals outputs_envelope() until the binder runs.
    uint32_t outputs_bound() const;

    // n_outputs target of the output-gather closure after a bind, read from
    // the owning input; 0 when the graph has no closure. The binder uses it
    // to prove that the n_outputs delta was applied by the same bind that
    // applied n_kv and rs_z.
    uint32_t outputs_bound_target() const;

    void set_params(const llm_graph_params & params);

    // the params the graph was last built with (encapsulation accessor: keeps
    // the params member private; used by llama_context's union-graph runtime
    // n_kv binder to verify non-nkv graph keys are unchanged before patching;
    // adds no data member of its own - note llm_graph_result DOES gain the
    // owner member below, by design)
    const llm_graph_params & get_params() const;

    // the llama_context that owns this result object (set by the context when
    // the result is (re)created). Ownership invariant only: the binder state
    // is a context member, so no global map or owner lookup is needed.
    const llama_context * get_owner() const { return owner; }
    void set_owner(const llama_context * ctx) { owner = ctx; }

    // important graph nodes
    ggml_tensor * t_inp_tokens  = nullptr;
    ggml_tensor * t_inp_embd    = nullptr; // [n_embd_inp, n_tokens]
    ggml_tensor * t_logits      = nullptr;
    ggml_tensor * t_embd        = nullptr;
    ggml_tensor * t_embd_pooled = nullptr;
    ggml_tensor * t_h_nextn     = nullptr; // [n_embd, n_outputs] hidden state before final output norm
    // factor-prefab MTP compose: the head prologue (A) outputs, i.e. the block input, the HC
    // residual stream the block (B) folds its output into, and the scatter weights that combine
    // uses. Set by graph_mtp::build_mtp_prologue(); consumed only by the compose layer
    // (GGML_TP5_PREFAB_IN_GRAPH=1) to cut A from B, null otherwise.
    ggml_tensor * t_prefab_cut        = nullptr;
    ggml_tensor * t_prefab_cut_res    = nullptr;
    ggml_tensor * t_prefab_cut_inject = nullptr;

    std::vector<ggml_tensor *> t_layer_inp;

    std::map<llama_seq_id, ggml_tensor *> t_sampled_logits;
    std::map<llama_seq_id, ggml_tensor *> t_candidates;
    std::map<llama_seq_id, ggml_tensor *> t_sampled;
    std::vector<llm_graph_fused_node> fused_nodes;
    std::map<llama_seq_id, ggml_tensor *> t_sampled_probs;

    std::vector<llm_graph_input_ptr> inputs;

    // the graph's runtime n_kv input, or nullptr when this build has none
    // (the union binder is off). Used by build_attn_inp_kv_impl to register
    // the KQ masks whose fill takes its width from that value.
    llm_graph_input_nkv_runtime * get_nkv_runtime_input() const;

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

    // the owning llama_context (see get_owner()/set_owner(); nullptr until the
    // owning context claims the result). Pointer member: llama_context is only
    // forward-declared here.
    const llama_context * owner = nullptr;

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

    // union-graph runtime binder active for this build (see llm_graph_params)
    const bool union_binder;
    const bool strict_ntok;
    // carry the I32 zero-row index (see llm_graph_params::rs_row_armed)
    const bool rs_row_armed;

    const enum llama_pooling_type pooling_type;
    const enum llama_rope_type    rope_type;

    ggml_backend_sched_t sched;

    ggml_backend_t backend_cpu; // TODO: needed by build_attn_mha, figure out a way to remove?

    const llama_adapter_cvec     * cvec;
    const llama_adapter_loras    * loras;
    const llama_memory_context_i * mctx;
    const llama_cross            * cross;

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
    // runtime n_kv width input (union binder builds only); nullptr otherwise
    ggml_tensor * build_inp_nkv_runtime() const;
    // Attach a KQ mask to the runtime n_kv input, and take the mask's fill
    // away from owner->set_input(). rt may be null: the function then reads
    // the input from res. owner may be null: the mask is then registered
    // without an owner. No-op when there is no runtime input (binder off).
    void register_nkv_runtime_mask(llm_graph_input_nkv_runtime * rt, llm_graph_input_attn_kv * owner, ggml_tensor * kq_mask) const;
    // Same, for a mask that has no owner input.
    void register_nkv_runtime_mask(ggml_tensor * kq_mask) const;
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
                int64_t   n_kv_max,
                  float   kq_scale,
                    int   il) const;

    ggml_tensor * build_attn_reader(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * q_rope,
        ggml_tensor * q_native,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const;

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
    //
    ggml_tensor * build_rs(
            ggml_tensor * s,
            ggml_tensor * state_copy_main,
            ggml_tensor * state_zero_index,
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
