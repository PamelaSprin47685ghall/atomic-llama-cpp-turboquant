#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <map>
#include <set>
#include <vector>

//
// llama_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
        const layer_filter_cb & filter);

    ~llama_memory_recurrent() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

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

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    bool get_can_shift() const override;

    uint32_t get_recurrent_capacity() const override;
    uint32_t get_recurrent_used()     const override;
    uint32_t get_recurrent_seq_used(llama_seq_id seq_id) const override;

    // RERoT grouped layout queries (§§B.3.2, B.6, B.13 Phase 3)
    uint32_t get_brain_capacity() const override;
    uint32_t get_hand_capacity()  const override;
    uint32_t get_brain_used()     const override;
    uint32_t get_hand_used()      const override;

    void set_grouped_layout(uint32_t n_brains, uint32_t n_hands);
    bool is_grouped_layout() const { return n_brain_rows > 0 && n_hand_rows > 0; }

    // RBB frontier mean commit (§14.1, §B.6):
    // Deterministically reduces valid PUBLIC candidate hand rows to update S_global for this person.
    // N=1: exact identity with single candidate.
    // PRIVATE/PENDING: excluded from global S commit.
    // Isolation: S_global[person_a] and S_global[person_b] are in separate rows.
    bool commit_rbb_frontier_mean(
        uint32_t person_id,
        const std::vector<int32_t> & candidate_hand_rows,
        const std::vector<bool> & is_public_write);

    // Parallel Delta order-free block DeltaNet update (§14.1.2):
    // Combines concurrent candidate writes to shared S without dilution.
    // N=1: exact identity with single candidate.
    // N>1: regularized Gram solve over orthogonal write components.
    bool commit_rbb_frontier_parallel_delta(
        uint32_t person_id,
        const std::vector<int32_t> & candidate_hand_rows,
        const std::vector<bool> & is_public_write,
        const std::vector<float> & candidate_log_decays = {});

    // Shared fork hand seed (§B.6.4):
    // Materializes/captures immutable fork seed for queued children without duplicating full recurrent rows.
    struct hand_seed {
        uint64_t fork_id = 0;
        int32_t source_hand_row = -1;
        std::vector<std::vector<uint8_t>> conv_tail_bytes; // layer -> conv state bytes
        std::vector<std::vector<uint8_t>> private_s_bytes; // layer -> R0-R2 private S bytes
    };

    std::shared_ptr<hand_seed> capture_hand_seed(uint64_t fork_id, llama_seq_id source_seq);
    bool apply_hand_seed(llama_seq_id dest_seq, const std::shared_ptr<hand_seed> & seed);

    bool rerot_capture_hand_seed(llama_seq_id source_seq, std::vector<uint8_t> & seed_out) override;
    bool rerot_apply_hand_seed(llama_seq_id dest_seq, const std::vector<uint8_t> & seed_in) override;
    bool rerot_commit_rbb_frontier(
            uint32_t person_id,
            const llama_seq_id * candidate_seqs,
            const uint8_t * is_public_write,
            size_t n_candidates) override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // RERoT grouped layout fields (§§B.6, B.13 Phase 3)
    uint32_t n_brain_rows = 0;
    uint32_t n_hand_rows  = 0;
    uint32_t brain_capacity = 0;
    uint32_t hand_capacity  = 0;

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;
    std::vector<int32_t> tails; // logical seq_id -> physical cell

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;

    int32_t s_copy(int i) const;

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
