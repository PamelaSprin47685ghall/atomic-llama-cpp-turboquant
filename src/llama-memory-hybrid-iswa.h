#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory.h"
#include "llama-memory-recurrent.h"

#include <memory>
#include <vector>

//
// llama_memory_hybrid_iswa
//

// utilizes instances of llama_memory_recurrent and llama_kv_cache_iswa to
//   support models where each layer may be either attention-based (with SWA support) or recurrent

class llama_memory_hybrid_iswa : public llama_memory_i {
public:
    llama_memory_hybrid_iswa(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   swa_full,
                 uint32_t   kv_size,
                 uint32_t   n_ubatch,
                 uint32_t   n_pad,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn = nullptr,
    const layer_filter_cb & filter_recr = nullptr);

    ~llama_memory_hybrid_iswa() = default;

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

    uint32_t get_kv_capacity() const override;
    uint32_t get_kv_used()     const override;
    uint32_t get_kv_seq_used(llama_seq_id seq_id) const override;

    uint32_t get_recurrent_capacity() const override;
    uint32_t get_recurrent_used()     const override;
    uint32_t get_recurrent_seq_used(llama_seq_id seq_id) const override;

    uint32_t get_brain_capacity() const override { return mem_recr ? mem_recr->get_brain_capacity() : 0; }
    uint32_t get_hand_capacity()  const override { return mem_recr ? mem_recr->get_hand_capacity()  : 0; }
    uint32_t get_brain_used()     const override { return mem_recr ? mem_recr->get_brain_used()     : 0; }
    uint32_t get_hand_used()      const override { return mem_recr ? mem_recr->get_hand_used()      : 0; }

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    bool seq_rm_attention(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;
    void seq_cp_attention(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    bool seq_rm_recurrent(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;
    void seq_cp_recurrent(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;

    bool rerot_set_write_tag(llama_seq_id seq_id, const llama_kv_rerot_meta & tag) override;
    void rerot_clear_write_tag(llama_seq_id seq_id) override;
    bool rerot_can_publish_run(uint64_t episode_id, llama_rerot_run_id run_id, size_t * count) const override;
    bool rerot_can_reclassify_run(uint64_t episode_id, llama_rerot_run_id run_id,
        llama_rerot_visibility expected, llama_rerot_visibility replacement,
        uint64_t publish_epoch, size_t * count) const override;
    size_t rerot_publish_run(uint64_t episode_id, llama_rerot_run_id run_id, uint64_t publish_epoch) override;
    size_t rerot_reclassify_run(uint64_t episode_id, llama_rerot_run_id run_id,
        llama_rerot_visibility expected, llama_rerot_visibility replacement, uint64_t publish_epoch) override;
    bool rerot_can_add_run_ref(uint64_t episode_id, llama_rerot_run_id run_id,
        llama_seq_id seq_id, size_t * count) const override;
    size_t rerot_add_run_ref(uint64_t episode_id, llama_rerot_run_id run_id,
        llama_seq_id seq_id) override;
    bool rerot_set_reader_view(llama_seq_id seq_id, const llama_rerot_reader_state & view) override;
    void rerot_clear_reader_view(llama_seq_id seq_id) override;

    bool rerot_capture_hand_seed(llama_seq_id source_seq, std::vector<uint8_t> & seed_out) override {
        return mem_recr ? mem_recr->rerot_capture_hand_seed(source_seq, seed_out) : false;
    }
    bool rerot_apply_hand_seed(llama_seq_id dest_seq, const std::vector<uint8_t> & seed_in) override {
        return mem_recr ? mem_recr->rerot_apply_hand_seed(dest_seq, seed_in) : false;
    }

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    llama_memory_kv_reclaim_result reclaim_kv(const llama_memory_kv_reclaim_request & request) override;

    bool positions_are_sparse() const override;

    //
    // llama_memory_hybrid_iswa specific API
    //

    llama_kv_cache_iswa * get_mem_attn() const;
    llama_memory_recurrent * get_mem_recr() const;

private:
    const llama_hparams & hparams;

    const std::unique_ptr<llama_kv_cache_iswa> mem_attn;
    const std::unique_ptr<llama_memory_recurrent> mem_recr;
};

class llama_memory_hybrid_iswa_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // init failure
    explicit llama_memory_hybrid_iswa_context(llama_memory_status status);

    // init full
    explicit llama_memory_hybrid_iswa_context(llama_memory_hybrid_iswa * mem);

    // init update
    explicit llama_memory_hybrid_iswa_context(
        llama_memory_hybrid_iswa * mem,
                   llama_context * lctx,
                            bool   optimize);

    // init success
    llama_memory_hybrid_iswa_context(
           llama_memory_hybrid_iswa * mem,
                    slot_info_vec_t   sinfos_base,
                    slot_info_vec_t   sinfos_swa,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_iswa_context() = default;

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_hybrid_iswa_context
    //

    const llama_kv_cache_iswa_context * get_attn() const;
    const llama_memory_recurrent_context * get_recr() const;

private:
    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const llama_memory_context_ptr ctx_attn;
    const llama_memory_context_ptr ctx_recr;

    const llama_memory_status status;
};
