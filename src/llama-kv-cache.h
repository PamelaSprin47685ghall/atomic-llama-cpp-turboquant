#pragma once

#include "llama-batch.h"
#include "llama-attention-view.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

// Auto-asymmetric turbo-K upgrade decision (see llama-kv-cache.cpp for the
// full rationale: high-GQA-ratio models amplify turbo K's quantization
// error, so symmetric turbo K+V gets K upgraded to q8_0). Exposed so
// llama-context.cpp's block KV streaming page-geometry pre-scan can size
// its bootstrap allocation off the same effective K type the llama_kv_cache
// constructor will actually use, instead of the raw requested type - the
// two must never diverge or the streaming runtime's bootstrap pool ends up
// sized for the wrong page geometry.
ggml_type llama_kv_cache_resolve_stream_type_k(
        const llama_model & model, const llama_hparams & hparams,
        ggml_type type_k, ggml_type type_v);

// Layer-adaptive per-layer KV precision override (TURBO_LAYER_ADAPTIVE env
// var - see llama-kv-cache.cpp for the mode legend). Exposed, like the
// resolver above, so llama-context.cpp's block KV streaming pre-scan can
// detect ahead of time whether a model will actually get non-uniform
// per-layer KV types: the streamed page pool has one page size and one
// buffer type for the whole arena, so mixed q8_0/turbo2/turbo4 layers would
// otherwise pack differently-shaped rows into pages sized for a layer that
// isn't theirs.
int llama_kv_cache_turbo_layer_adaptive_mode(ggml_type type_v, uint32_t n_layer);

ggml_type llama_kv_cache_turbo_layer_adaptive_type_k(
        int mode, ggml_type type_k, ggml_type type_v, uint32_t il, uint32_t n_layer);
ggml_type llama_kv_cache_turbo_layer_adaptive_type_v(
        int mode, ggml_type type_k, ggml_type type_v, uint32_t il, uint32_t n_layer);

//
// llama_kv_cache
//

class llama_kv_cache : public llama_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // Heap payload owned by this slot. The surrounding container counts
        // sizeof(slot_info) itself; this counts only the two vectors and the
        // per-stream index blocks they allocate.
        size_t host_bytes() const {
            size_t n = strm.capacity()*sizeof(llama_seq_id) +
                       idxs.capacity()*sizeof(idx_vec_t);
            for (const auto & idx : idxs) {
                n += idx.capacity()*sizeof(uint32_t);
            }
            return n;
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    // one reusable snapshot of the cache state that prepare_into() restores
    // when the batch cannot be placed. the pool hands these out per ubatch
    struct snapshot_slot {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // journal snapshot of the old cells

        size_t trial_idx_off = 0; // offset into the flat per-stream trial index block

        size_t host_bytes() const {
            size_t n = sizeof(*this) +
                       sinfo.host_bytes() +
                       v_heads_old.capacity()*sizeof(uint32_t) +
                       v_cells.capacity()*sizeof(llama_kv_cells);
            for (const auto & c : v_cells) {
                n += c.host_bytes();
            }
            return n;
        }
    };

    // decode-time buffers reused by one batch context. the context returns
    // them when it dies, so the next decode finds the capacity still there.
    // while a pooled batch context is alive it owns the moved-out
    // sinfos/ubatches, so a host query sees only the cache-side containers
    // and is a lower bound.
    struct decode_scratch {
        slot_info_vec_t sinfos;
        std::vector<llama_ubatch> ubatches;

        // prepare_into() scratch: the recovery stack and the pool guard
        std::vector<snapshot_slot *>                    states;
        std::vector<snapshot_slot *>                    pooled;
        std::vector<std::unique_ptr<snapshot_slot>>     fallback;
        std::vector<size_t>                             trial_idx; // flat per-stream trial indices

        bool in_use    = false;
        bool from_pool = false;

        llama_host_memory_breakdown_data host_memory_breakdown() const {
            llama_host_memory_breakdown_data res;
            res.scratch += sizeof(*this) +
                           sinfos.capacity()*sizeof(slot_info) +
                           ubatches.capacity()*sizeof(llama_ubatch) +
                           states.capacity()*sizeof(snapshot_slot *) +
                           pooled.capacity()*sizeof(snapshot_slot *) +
                           fallback.capacity()*sizeof(std::unique_ptr<snapshot_slot>);
            for (const auto & slot : sinfos) {
                res.scratch += slot.host_bytes();
            }
            res.trial_index += trial_idx.capacity()*sizeof(size_t);
            for (const auto & slot : fallback) {
                res.snapshots += slot->host_bytes();
            }
            return res;
        }
    };

    // fixed-slot pool of decode scratches. Two slots cover a nested decode;
    // an exhausted pool falls back to the single scratch or to local vectors.
    // The pool synchronizes the scratch buffers only: the cache cells and
    // heads still belong to one decode thread at a time.
    class decode_scratch_pool {
    public:
        decode_scratch_pool() = default;

        explicit decode_scratch_pool(uint32_t capacity) {
            reserve(capacity);
        }

        // Startup allocation: call before the first decode.
        void reserve(uint32_t capacity) {
            std::lock_guard<std::mutex> lock(mutex);

            storage.reserve(storage.size() + capacity);
            free.reserve(free.size() + capacity);

            for (uint32_t i = 0; i < capacity; ++i) {
                auto scratch = std::make_unique<decode_scratch>();
                scratch->sinfos.reserve(8);
                scratch->ubatches.reserve(8);
                scratch->states.reserve(8);
                scratch->pooled.reserve(8);
                scratch->fallback.reserve(8);
                free.push_back(scratch.get());
                storage.push_back(std::move(scratch));
            }
        }

        decode_scratch_pool(const decode_scratch_pool &)             = delete;
        decode_scratch_pool & operator=(const decode_scratch_pool &) = delete;

        decode_scratch * acquire() {
            std::lock_guard<std::mutex> lock(mutex);

            if (free.empty()) {
                ++fallbacks_;
                return nullptr;
            }

            decode_scratch * scratch = free.back();
            free.pop_back();
            scratch->in_use    = true;
            scratch->from_pool = true;
            ++in_use_;

            return scratch;
        }

        void release(decode_scratch * scratch) {
            if (scratch == nullptr) {
                return;
            }

            std::lock_guard<std::mutex> lock(mutex);

            // sinfos keeps its capacity; prepare_into() resizes it.
            scratch->ubatches.clear();
            scratch->states.clear();
            scratch->pooled.clear();
            scratch->fallback.clear();
            scratch->in_use    = false;
            scratch->from_pool = false;

            --in_use_;
            free.push_back(scratch);
        }

        uint32_t capacity() const {
            std::lock_guard<std::mutex> lock(mutex);
            return static_cast<uint32_t>(storage.size());
        }

        uint32_t in_use() const {
            std::lock_guard<std::mutex> lock(mutex);
            return in_use_;
        }

        uint64_t fallbacks() const {
            std::lock_guard<std::mutex> lock(mutex);
            return fallbacks_;
        }

        llama_host_memory_breakdown_data host_memory_breakdown() const {
            llama_host_memory_breakdown_data res;
            res.scratch += storage.capacity()*sizeof(std::unique_ptr<decode_scratch>) +
                           free.capacity()*sizeof(decode_scratch *);
            for (const auto & scratch_ptr : storage) {
                const auto scratch = scratch_ptr->host_memory_breakdown();
                res.scratch     += scratch.scratch;
                res.trial_index += scratch.trial_index;
                res.snapshots   += scratch.snapshots;
            }
            return res;
        }

    private:
        // reserve(), acquire() and release() take this mutex. The host
        // breakdown does not: it reads storage/free under the caller's
        // quiescent contract. This mutex protects the free list and storage
        // metadata, not the scratch contents.
        mutable std::mutex mutex;
        std::vector<std::unique_ptr<decode_scratch>> storage;
        std::vector<decode_scratch *> free;
        uint32_t in_use_   = 0;
        uint64_t fallbacks_ = 0;
    };

    // TODO: refactor the memory instances to not depend on `llama_model`
    //       instead pass all necessary info (e.g. hparams, dev layers, arch, etc.) directly
    //       likely through `struct llama_memory_params`
    llama_kv_cache(
            const llama_model & model,
          const llama_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share,
                 const char *   name_tag = "",
                          size_t kv_stream_stage_bytes = 0,
                          void * kv_stream_phase_arena = nullptr,
                          size_t kv_stream_maximum_pool_bytes = 0);

    ~llama_kv_cache() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    bool clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    bool seq_rm_recurrent(llama_seq_id) override { return true; }
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;
    // Host payload of the journal and the prepare/decode scratch. The caller
    // must keep the cache quiescent: no decode/update, no memory mutation,
    // and no live memory context. See llama-ext.h.
    llama_host_memory_breakdown_data host_memory_breakdown() const override;

    bool has_kv_stream_targets() const override;
    std::vector<llama_kv_stream_target> get_kv_stream_targets() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_stream() const;
    std::vector<uint32_t> get_layer_ids() const;
    ggml_tensor * get_k_storage(int32_t il) const;
    ggml_tensor * get_v_storage(int32_t il) const;
    // a global layer id this cache maps (0 when empty); filtered caches (MTP contexts
    // hold only their nextn layer) have no entry for global layer 0
    int32_t first_layer_id() const;
    bool get_v_transposed() const;

    bool kv_stream_adapt(uint32_t active_tokens, uint32_t query_tokens);
    bool kv_stream_resize_pool(
        size_t pool_bytes, uint32_t active_tokens, uint32_t ring_slots);

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    const llama_kv_cells & get_cells(llama_seq_id seq_id) const;

    bool prepare_attention_views(const llama_attention_view * views, size_t count,
                                std::vector<llama_attention_view_owned> & out) const;
    bool validate_attention_batch(const llama_batch & batch) const;
    bool resolve_attention_view(llama_seq_id seq, llama_pos pos,
                                llama_attention_view_resolved & out) const;
    std::vector<llama_attention_view_owned> attention_views;

    // state_read, plus the cells the restored tokens were placed in.
    // a cache that mirrors another one cell for cell (the qwen4exp indexer) cannot search for
    // its own cells here: a second independent search only happens to agree with the first.
    //   sinfos_out: if set, resized to n_stream and filled with the layout used; a stream that
    //               carried no cells leaves an empty entry
    //   sinfos_in : if set, the layout to use instead of searching for one. it must have one
    //               entry per stream and the entry must match the cell count in the blob,
    //               otherwise the read fails as it would on any other corrupt input
    void state_read_sinfo(
            llama_io_read_i & io,
               llama_seq_id   seq_id,
      llama_state_seq_flags   flags,
          slot_info_vec_t *   sinfos_out,
     const slot_info_vec_t *   sinfos_in);

    //
    // graph_build API
    //

    // the PADDED width: the stock 256-cell bucket, rounded up to n_pad and
    // clamped to the physical cell count of the participating streams. This
    // is the width a stock graph executes at. While any batch context of this
    // cache has the exact lane armed, reaching this function with
    // n_pad < 256 aborts: the exact lane is supposed to have answered first,
    // and the test-only forced hook is the one exception.
    uint32_t get_n_kv(const slot_info & sinfo) const;

    // exact-width lane (GGML_TP5_UNION_BINDER_EXACT=1, the default under the
    // binder gate): the true occupied span, with no n_pad rounding and no
    // 256-cell floor. An empty cache yields one masked cell; a transposed V
    // rounds up to its block size. Returns 0 when the span cannot be addressed
    // at one shared width (active streaming, unprovable V blocks, or a width
    // that does not fit every participating stream); an unarmed caller then
    // keeps the applied width, and an armed one aborts. The caller must apply
    // the scheduler and backend width guards before binding. This is the width
    // the union binder binds, and it is a different computation from
    // get_n_kv() above.
    uint32_t get_n_kv_exact(const slot_info & sinfo) const;
    // active cell count when position p lives in physical cell p; 0 for any non-contiguous layout
    uint32_t get_n_kv_pos_contiguous(const slot_info & sinfo, const llama_ubatch & ubatch) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // TurboQuant: get rotation matrices (stored as row-major C arrays)
    // turbo_rotation = R (forward rotation, for Q pre-rotate-queries)
    // turbo_rotation_inv = R^T = R^{-1} (inverse rotation, for V output un-rotation)
    ggml_tensor * get_turbo_rotation() const { return turbo_rotation; }
    ggml_tensor * get_turbo_rotation_inv() const { return turbo_rotation_inv; }

    // TurboQuant InnerQ: per-channel scale_inv for Q/V equalization
    ggml_tensor * get_turbo_innerq_scale_inv() const { return turbo_innerq_scale_inv; }

    // store k_cur and v_cur in the cache based on the provided head location
    // token_axis_marker: the token lane asks the KV write to tag the value
    // tensors it creates with LLM_TOKEN_AXIS_MARKER | 1, so the symbolic shape
    // analysis derives their token axis from the construction site instead of
    // guessing it from a 1-wide build shape. Default false: every other lane
    // is byte-identical to before.
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo,
                        bool token_axis_marker = false) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo,
                        bool token_axis_marker = false) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    // the decode scratch of this cache, or nullptr when another batch context
    // still holds it
    decode_scratch * scratch_acquire();

    // take the vectors back into the scratch and mark it free again
    void scratch_return(slot_info_vec_t & sinfos, std::vector<llama_ubatch> & ubatches, decode_scratch * scratch);

    llama_memory_apply_status update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);
    void shift_backup_save();
    void shift_backup_restore();
    void shift_backup_clear();

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // same search, but fills out in place so that the vectors it already holds
    // keep their capacity. out.idxs is empty on failure
    bool find_slot(const llama_ubatch & ubatch, bool cont, slot_info & out) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    // restore one trial placement: the cells of every stream in sinfo return to
    // v_cells_old[s], and the head of stream sinfo.strm[s] returns to
    // v_heads_old[sinfo.strm[s]]. v_cells_old is indexed by the sinfo stream
    // order, not by stream id.
    static void restore_cells(
            llama_kv_cells_vec & v_cells,
            std::vector<uint32_t> & v_heads,
            const slot_info & sinfo,
            const std::vector<uint32_t> & v_heads_old,
            const std::vector<llama_kv_cells> & v_cells_old) {
        GGML_ASSERT(v_cells_old.size() >= sinfo.n_stream());

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const uint32_t stream = static_cast<uint32_t>(sinfo.strm[s]);

            GGML_ASSERT(stream < v_cells.size());
            GGML_ASSERT(stream < v_heads.size());
            GGML_ASSERT(stream < v_heads_old.size());

            auto & cells = v_cells[stream];
            auto & head  = v_heads[stream];

            cells.set(sinfo.idxs[s], v_cells_old[s]);
            head = v_heads_old[stream];
        }
    }

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    // Fill the KQ mask over the first `n_kv` columns instead of dst->ne[0].
    // Used by the union-binder runtime n_kv wire; the stock two-argument
    // overload forwards here with n_kv == 0 (meaning dst->ne[0]).
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn, uint32_t n_kv) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    // Cross-check ONE filled KQ-mask row against the KV cells themselves, for
    // the mask-observation lane. `mask_data` is the mask tensor's base pointer
    // (the fill's own buffer, row-major) and `tok` is the ubatch token whose
    // row is checked: this function computes the row offset itself, from the
    // fill's layout (row i = s*n_tps + ii at idst = width*i), so the caller
    // cannot hand it the wrong row. `dst` is the mask tensor, `width` the
    // width the fill consumed.
    //
    // This is an INDEPENDENT read of the executed width: the width comes from
    // the mask the graph will execute, the visibility comes from the cache's
    // own cell table, and the two are written by different code paths
    // (llama-graph.cpp's runtime n_kv input vs. llama_kv_cache::apply_ubatch).
    // Every column below `width` must hold exactly 0.0f (visible) or
    // -INFINITY (dropped), and a column must be visible exactly when its cell
    // carries the token's sequence and no causal/SWA rule drops it.
    // `causal_attn` and `swa` are the two rules the fill itself applies; the
    // caller passes the values it filled with.
    //
    // Returns one of three verdicts, so a refusal can never be mistaken for a
    // disagreement or for a pass:
    //    LLAMA_GATE_PASS      every column agrees;
    //    LLAMA_GATE_FAIL      a column disagrees (the mask is wrong) --
    //                         `first_bad` names it;
    //    LLAMA_GATE_UNPROVEN  the layout cannot be modelled (a token count
    //                         that does not divide into the streams, a row
    //                         whose stream disagrees with its token's
    //                         sequence, an unusable tensor). Nothing was
    //                         compared.
    // The caller counts UNPROVEN separately (mask_row_refused): a layout change
    // that makes every check refuse must not look like a wrong mask.
    // `first_bad` receives the first disagreeing column, or -1.
    //
    // Only F32 is compared. The row is read as `const float *`, so an F16 mask
    // would be reinterpreted rather than converted; F16 (flash attention) and
    // every other type are UNPROVEN, not compared.
    enum llama_gate_verdict check_mask_row(
            const ggml_tensor * dst,
            const llama_ubatch * ubatch,
            uint32_t tok,
            const float * mask_data,
            uint32_t width,
            bool causal_attn,
            bool swa,
            int64_t * first_bad) const;

    // The SWA rule this cache's fill applies. The fill's args struct carries
    // the same value; this accessor exists so the read-back and the context's
    // own view can apply that rule instead of assuming there is none.
    llama_swa_type get_swa_type() const { return swa_type; }

    // inkling: fill dst I32 [n_kv, n_tokens] with flat rel-bias gather indices,
    // idx(i, j) = i*(extent + 1) + rel; empty/out-of-band cells map to the zero-bias column `extent`
    void set_input_pos_rel_flat(ggml_tensor * dst, const llama_ubatch * ubatch, uint32_t extent) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

    // factor-prefab envelope support (LLAMA_PREFAB_ENABLE=1, default off).
    // A ubatch with n_tokens < n_env runs on a graph compiled for n_env rows. The pad rows
    // [n_tokens, n_env) must not touch any live cell: they write K/V into one scratch cell that is
    // empty (hence masked -inf for every real row) and their mask rows replicate the last real row
    // (finite softmax, no NaN can leak through the non-FA KQ path).
    // returns -1 when no such cell exists (caller falls back to the stock builder)
    int64_t prefab_scratch_cell(const slot_info & sinfo) const;
    void prefab_set_input_idxs(ggml_tensor * k_dst, ggml_tensor * v_dst, const llama_ubatch * ubatch,
                               const slot_info & sinfo, uint32_t n_env, int64_t scratch) const;
    void prefab_set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn, uint32_t n_env,
                                  const slot_info & sinfo) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;

    // CPU test seam: constructs a real cache and calls the production prepare
    // paths to cover the SWA purge timing. It adds no runtime behavior.
    friend struct llama_kv_cache_prepare_test_access;

    // the pre-pool implementation of prepare(), kept as the flag-off path
    slot_info_vec_t prepare_reference(const std::vector<llama_ubatch> & ubatches);

    // pooled prepare: writes the slot infos into res and reuses the capacity
    // that res already holds. when scratch is set, its states/pooled/fallback
    // vectors are reused too. returns false on failure
    bool prepare_into(const std::vector<llama_ubatch> & ubatches, slot_info_vec_t & res, decode_scratch * scratch);

    // Prepare undo journal for the SWA purge cells. begin() enables it on all
    // streams and snapshots the heads; end() detaches it; replay() restores
    // the full write domain and throws if that restore fails.
    bool prepare_journal_begin();
    void prepare_journal_end(bool journal_cells);
    void prepare_journal_replay(bool journal_cells);

    // Shared KQ-mask fill. `n_rows` is the number of token rows per stream to
    // write. The stock layout passes dst->ne[1] and asserts it equals the
    // batch rows; a prefab envelope mask has n_env rows and passes only its
    // real ubatch->n_tokens rows, leaving the pad rows to the caller. Both
    // write at the tensor's own row stride (dst->ne[0]).
    void fill_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn,
                            uint32_t n_kv_runtime, int64_t n_rows) const;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    // env: LLAMA_ATTN_ROT_DISABLE
    bool attn_rot_k = false;
    bool attn_rot_v = false;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    struct kv_stream_runtime_owner {
        using feedback_fn_t = bool (*)(
            void *, uint64_t *, uint64_t *, double *, uint32_t *,
            uint32_t *, uint32_t *, uint32_t *);
        using span_feedback_fn_t = bool (*)(void *, double);
        using reconfigure_fn_t = bool (*)(void *, uint32_t, uint32_t);
        using repartition_fn_t = bool (*)(void *, uint32_t);
        using decode_layout_fn_t = bool (*)(void *, uint32_t);
        using mark_dirty_rows_fn_t = bool (*)(void *, const int64_t *, size_t);
        using resize_pool_fn_t = bool (*)(void *, size_t, uint32_t, uint32_t);

        void * runtime = nullptr;
        void (*free_fn)(void *) = nullptr;
        feedback_fn_t feedback_fn = nullptr;
        span_feedback_fn_t span_feedback_fn = nullptr;
        reconfigure_fn_t reconfigure_fn = nullptr;
        repartition_fn_t repartition_fn = nullptr;
        decode_layout_fn_t decode_layout_fn = nullptr;
        mark_dirty_rows_fn_t mark_dirty_rows_fn = nullptr;
        resize_pool_fn_t resize_pool_fn = nullptr;
        uint32_t layer_count = 0;
        uint32_t minimum_ring_slots = 0;
        uint32_t decode_layout_pages = 0;
        uint32_t starved_evaluations = 0;
        uint32_t overprovisioned_evaluations = 0;
        uint32_t evaluations_since_repartition = UINT32_MAX;
        uint64_t previous_deadline_samples = 0;
        uint64_t previous_deadline_misses = 0;
        int64_t previous_adapt_us = 0;
        uint32_t previous_query_tokens = UINT32_MAX;

        ~kv_stream_runtime_owner() {
            if (runtime != nullptr) {
                free_fn(runtime);
            }
        }
    };

    // Declared before ctxs_bufs so the custom buffers release their runtime
    // references before this owner releases the initial reference.
    kv_stream_runtime_owner kv_stream_runtime;
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    // Per-stream undo journals for the prepare trials. Only SWA-active caches
    // allocate entries; the normal path records into pre-reserved storage.
    std::vector<cell_journal> cell_journals;
    std::vector<uint32_t>     heads_before;

    // fixed-slot pool of prepare snapshots. the pool allocates every slot once
    // and hands the same slots out again on later decodes
    class snapshot_pool {
    public:
        // nullptr when the pool has no free slot
        snapshot_slot * acquire() {
            std::lock_guard<std::mutex> lock(mutex);

            if (free.empty()) {
                ++fallback;
                return nullptr;
            }

            snapshot_slot * slot = free.back();
            free.pop_back();

            ++acquired;

            return slot;
        }

        void release(snapshot_slot * slot) {
            std::lock_guard<std::mutex> lock(mutex);

            free.push_back(slot);

            ++released;
        }

        // add slots until the pool holds n of them. a later call never shrinks it
        void reserve(size_t n) {
            std::lock_guard<std::mutex> lock(mutex);

            if (n <= storage.size()) {
                return;
            }

            // reserve both vectors before publishing a slot, so neither push
            // below can fail and leave the slot outside free. this also keeps
            // every future release() from allocating: free never holds more
            // entries than storage.
            storage.reserve(n);
            free.reserve(n);

            while (storage.size() < n) {
                storage.push_back(std::make_unique<snapshot_slot>());
                free.push_back(storage.back().get());
            }
        }

        llama_host_memory_breakdown_data host_memory_breakdown() const {
            llama_host_memory_breakdown_data res;
            res.snapshots += storage.capacity()*sizeof(std::unique_ptr<snapshot_slot>) +
                             free.capacity()*sizeof(snapshot_slot *);
            for (const auto & slot : storage) {
                res.snapshots += slot->host_bytes();
            }
            return res;
        }

    private:
        // reserve(), acquire() and release() take this mutex. The host
        // breakdown does not: it reads storage/free under the caller's
        // quiescent contract. This mutex protects the free list and storage
        // metadata, not the slot contents.
        mutable std::mutex mutex;

        std::vector<std::unique_ptr<snapshot_slot>> storage;
        std::vector<snapshot_slot *> free;

        uint64_t acquired = 0;
        uint64_t released = 0;
        uint64_t fallback = 0;
    };

    // reusable snapshots for prepare_into()
    snapshot_pool snapshots;

    // decode-time buffers reused by the current batch context
    decode_scratch decode_scratch_;
    std::mutex decode_scratch_mutex;

    // TP5_DECODE_SCRATCH_POOL=1: a fixed pool of decode scratches,
    // reserved at cache construction so the first decode does not allocate
    decode_scratch_pool scratch_pool;

    // TODO: temporary until we refactor to be able to share the same cells between 2 kv caches [TAG_KV_CACHE_SHARE_CELLS]
    llama_kv_cache * other;

    // K-shift failure/backup state. shift_failed is set only when a K-shift
    // compute may have partially written K/V; clear(true) resets it.
    std::vector<llama_kv_cells> shift_cells_backup;
    std::vector<uint32_t>       shift_heads_backup;
    bool                        shift_failed = false;

    std::shared_ptr<llama_kv_cells_vec> v_cells_impl;

    llama_kv_cells_vec & v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // TurboQuant rotation matrices (128x128, row-major stored)
    ggml_tensor * turbo_rotation = nullptr;      // R (forward rotation)
    ggml_tensor * turbo_rotation_inv = nullptr;   // R^T = R^{-1} (inverse rotation)

    // TurboQuant InnerQ: per-channel scale_inv for Q/V equalization (128 floats)
    ggml_tensor * turbo_innerq_scale_inv = nullptr;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    // sinfo_in, when set, replaces the find_slot call: the cells are given by the caller
    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1, const slot_info * sinfo_in = nullptr);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;
    using decode_scratch   = llama_kv_cache::decode_scratch;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    // returns the decode scratch, when the context holds one
    ~llama_kv_cache_context() override;

    // The context owns the decode scratch, the split ubatches and the cache
    // pointer. A copy would create a second owner and return the same scratch
    // twice; a move would leave the source still holding it. Keep ownership in
    // the unique_ptr that creates the context.
    llama_kv_cache_context(const llama_kv_cache_context &)             = delete;
    llama_kv_cache_context & operator=(const llama_kv_cache_context &) = delete;
    llama_kv_cache_context(llama_kv_cache_context &&)                  = delete;
    llama_kv_cache_context & operator=(llama_kv_cache_context &&)      = delete;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;
    llama_memory_apply_status get_update_status() const override { return update_status; }

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    bool has_kv_stream_targets() const override;
    std::vector<llama_kv_stream_active_target> get_kv_stream_active_targets() const override;

    //
    // llama_kv_cache_context specific API
    //

    // The width this batch context executes at, in three cases, in this order:
    //   1. the build-scoped envelope override (nkv_envelope_begin): the
    //      allocation width, during model.build_graph() only;
    //   2. the exact lane armed (nkv_exact_target): the true used span from
    //      get_n_kv_exact(); if that returns 0 the call aborts. There is no
    //      padded fallback: the bucket floor, GGML_TP5_UNION_PAD_BUCKET and
    //      the test-only forced hook are all deleted;
    //   3. the exact lane not armed (stock context, gate off, envelope
    //      disabled, multi-split): the width the cache applied in apply(),
    //      unchanged from stock, with no abort.
    uint32_t get_n_kv() const;
    // Width computed by apply(): the applied span, independent of the
    // build-scoped override.
    uint32_t get_n_kv_padded() const { return n_kv; }
    // Applied width, independent of the build override.
    // Zero means that this context has not supplied a width.
    uint32_t get_n_kv_stock() const { return n_kv; }
    // the true used span: 0 when unavailable, when a build override is
    // armed, or for reserve/init_full contexts
    uint32_t get_n_kv_exact() const;
    // width the graph is expected to execute at: exactly what get_n_kv()
    // reports. can_reuse_kq_mask() must use this so a bound graph compares
    // against the same width it was built/bound at.
    uint32_t get_n_kv_target() const;
    // Width the attention consumers must execute at, as a VALUE: the true
    // used span for this ubatch, or the applied width when the span cannot be
    // proven. Unlike get_n_kv() it never reports the build-scoped envelope
    // override, so it is the value written into the graph's runtime n_kv
    // input (llm_graph_input_nkv_runtime). 0 means "no width".
    uint32_t get_n_kv_runtime() const;
    uint32_t get_n_kv_pos_contiguous() const;
    // The SWA rule the fill applies, so the read-back can apply the same one
    // instead of assuming there is none. Forwards to the cache's own accessor;
    // the fill's args struct carries the same value.
    llama_swa_type get_swa_type() const {
        return kv != nullptr ? kv->get_swa_type() : LLAMA_SWA_TYPE_NONE;
    }

    // Forwarded to llama_kv_cache::check_mask_row, which documents the three
    // verdicts and the layout it models. This class adds the cache pointer and
    // nothing else.
    enum llama_gate_verdict check_mask_row(
            const ggml_tensor * dst,
            const llama_ubatch * ubatch,
            uint32_t tok,
            const float * mask_data,
            uint32_t width,
            bool causal_attn,
            bool swa,
            int64_t * first_bad) const;

    // union-graph runtime-length n_kv lane: set by the owning llama_context
    // before graph reuse/build, from the binder's own predicate (gate on, no
    // env_disabled, the GGML_TP5_UNION_BINDER_EXACT selection, a live record,
    // one patch-safe split). The batch context is constructed without an
    // lctx, so the target flag travels on the object instead of being read
    // from the owning context. While true, get_n_kv() reports the true used
    // span from get_n_kv_exact() and aborts if the layout refuses it. While
    // false, get_n_kv() keeps the width the cache applied. The owner arms it
    // only for a single split. Each bind also validates scheduler copies
    // before changing the live graph; refusal rebuilds.
    void set_nkv_exact_target(bool enable) {
        nkv_exact_target = enable;
    }
    bool has_attention_views() const;
    llama_attention_view_resolved attention_view(llama_seq_id seq, llama_pos pos) const;

    // physical per-stream capacity of the underlying cache (upper bound for any
    // n_kv the cache can back); used by the union-graph runtime nkv binder to
    // clamp its envelope build to real, allocated cells
    uint32_t get_capacity() const {
        return kv ? kv->get_size() : 0u;
    }

    // number of independent KV streams (1 for a unified cache). The binder
    // refuses an in-place patch when this is > 1: the K/V view offset encodes
    // the build-time stream base, which the symbolic shape map does not model.
    uint32_t get_n_stream() const {
        return kv ? kv->get_n_stream() : 0u;
    }
    // union-graph runtime nkv binder: build-scoped n_kv override. While armed,
    // get_n_kv() reports the envelope width instead of the applied cache n_kv,
    // so a graph built in that scope is sized/allocated for the full envelope.
    // Must not span set_inputs()/compute; only graph construction.
    // The value is a per-object member (no process-global side table): two
    // llama_context instances on different threads never share it.
    void nkv_envelope_begin(uint32_t cap) const;

    void nkv_envelope_end() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // TurboQuant rotation accessors
    ggml_tensor * get_turbo_rotation() const;
    ggml_tensor * get_turbo_rotation_inv() const;

    // Override virtual methods from llama_memory_context_i
    ggml_tensor * get_turbo_rot_forward() const override;
    ggml_tensor * get_turbo_rot_inverse() const override;

    // TurboQuant InnerQ: per-channel scale_inv for Q/V equalization
    ggml_tensor * get_turbo_innerq_scale_inv() const override;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il,
                        bool token_axis_marker = false) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il,
                        bool token_axis_marker = false) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    // Fill the KQ mask at an explicit width. `n_kv` is the width the graph
    // must execute at, taken from the graph's runtime n_kv input rather than
    // from the tensor shape. 0 means "no runtime width" and fills the whole
    // tensor, exactly the stock behaviour. A nonzero width above the
    // tensor's ne[0] is a contract violation and aborts: filling ne[0]
    // instead would hand the graph a wider mask than it executes at.
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn, uint32_t n_kv) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_pos_rel_flat(ggml_tensor * dst, const llama_ubatch * ubatch, uint32_t extent) const; // inkling

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;
    // factor-prefab ATTN factor support: the graph is compiled for a padded KV width
    // (multiple of 1024, >= n_kv, capped at the cache size) instead of the exact n_kv,
    // so the factor/envelope graph stays valid while n_kv grows within the pad window
    // (n_kv itself is 256-aligned). Returns the current padded width; 0 when the
    // context has no concrete ubatch (reserve/update contexts).
    uint32_t get_n_kv_pad() const;
    // true when the current n_kv fits the given padded width (graph reuse condition)
    bool n_kv_pad_fits(uint32_t n_kv_pad) const;
    // K/V views sized by the padded width (prefab graphs only; stock path uses get_k/get_v)
    ggml_tensor * get_k_pad(ggml_context * ctx, int32_t il, uint32_t n_kv_pad) const;
    ggml_tensor * get_v_pad(ggml_context * ctx, int32_t il, uint32_t n_kv_pad) const;

    // factor-prefab envelope helpers (see llama_kv_cache::prefab_*)
    const llama_kv_cache * get_kv_cache() const { return kv; }
    int64_t prefab_scratch_cell() const;
    void prefab_set_input_idxs(ggml_tensor * k_dst, ggml_tensor * v_dst, const llama_ubatch * ubatch, uint32_t n_env, int64_t scratch) const;
    void prefab_set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn, uint32_t n_env) const;

    // attach the decode scratch that this context returns when it dies
    void set_decode_scratch(decode_scratch * scratch);

private:
    llama_memory_status status;

    llama_kv_cache * kv = nullptr;
    llama_context * lctx = nullptr;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    llama_memory_apply_status update_status = LLAMA_MEMORY_APPLY_APPLIED;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    // set by set_decode_scratch(): the scratch that sinfos and ubatches came
    // from, and the one that takes them back on destruction
    decode_scratch * scratch_ = nullptr;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv = 0;

    // build-scoped envelope width for the union-graph runtime nkv binder;
    // 0 = not armed. mutable: armed/cleared through const graph-build paths.
    mutable int32_t nkv_build_override = 0;

    // runtime-length n_kv lane: the owning llama_context arms this before
    // graph reuse/build when the union binder can patch the width in place.
    // The batch context has no lctx, so the flag lives on the object. It is
    // the only condition under which get_n_kv() asks the exact query at all;
    // an unarmed context keeps the width the cache applied.
    //
    // The arming condition (llama-context.cpp, process_ubatch) is the binder
    // gate, the GGML_TP5_UNION_BINDER_EXACT selection, a live record and a
    // single-split scheduler. The stock padded width is what a context without
    // a live record reports, and what a context with nkv_exact_enabled off
    // reports as the A/B control lane. A cache whose layout cannot address the
    // true used span while the exact target is armed is a hard failure in
    // get_n_kv(), not a fallback: the pad bucket, its two knobs and the
    // test-only forced hook are deleted, so there is no padded escape.
    bool nkv_exact_target = false;
};
