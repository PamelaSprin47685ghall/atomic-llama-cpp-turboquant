#include "llama-memory-hybrid.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

//
// llama_memory_hybrid
//

llama_memory_hybrid::llama_memory_hybrid(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                 uint32_t   n_brain_max,
                 uint32_t   n_hand_max,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const llama_cparams   * cparams) :
    hparams(model.hparams),
    mem_attn(new llama_kv_cache(
        model,
        model.hparams,
        type_k,
        type_v,
        v_trans,
        offload,
        unified,
        kv_size,
        n_seq_max,
        n_pad,
        n_swa,
        swa_type,
        nullptr,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn,
        nullptr,
        nullptr,
        cparams
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload,
        rs_size,
        unified ? LLAMA_MAX_SEQ : n_seq_max,
        n_rs_seq,
        n_brain_max,
        n_hand_max,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr
    )) {}

llama_memory_context_ptr llama_memory_hybrid::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (mem_attn->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = mem_recr->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!mem_recr->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = mem_attn->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // Target attention uses real hot reservations when bounded; the
        // recurrent cache takes none. Failure fails prepare with zero residue.
        std::vector<llama_xkv::xkv_hot_reservation> hot_res_attn;
        {
            std::string res_err;
            if (!mem_attn->reserve_hot_slots(heads_attn, ubatches, hot_res_attn, &res_err)) {
                LLAMA_LOG_ERROR("%s: %s\n", __func__, res_err.c_str());
                return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
            }
        }

        return std::make_unique<llama_memory_hybrid_context>(
                this, std::move(heads_attn), std::move(ubatches), std::move(hot_res_attn));
    } while(false);

    return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid::init_full() {
    return std::make_unique<llama_memory_hybrid_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_context>(this, lctx, optimize);
}

bool llama_memory_hybrid::get_can_shift() const {
    // Shifting is trivially supported for recurrent
    return mem_attn->get_can_shift();
}

uint32_t llama_memory_hybrid::get_kv_capacity() const {
    return mem_attn->get_kv_capacity();
}

uint32_t llama_memory_hybrid::get_kv_used() const {
    return mem_attn->get_kv_used();
}

uint32_t llama_memory_hybrid::get_kv_seq_used(llama_seq_id seq_id) const {
    return mem_attn->get_kv_seq_used(seq_id);
}

uint32_t llama_memory_hybrid::get_kv_hot_capacity() const {
    return mem_attn ? mem_attn->get_kv_hot_capacity() : get_kv_capacity();
}

bool llama_memory_hybrid::can_use_legacy_attention() const {
    return mem_attn ? mem_attn->can_use_legacy_attention() : true;
}

bool llama_memory_hybrid::is_xkv_bounded_hot() const {
    return mem_attn ? mem_attn->is_xkv_bounded_hot() : false;
}

bool llama_memory_hybrid::get_admission_snapshot(struct llama_memory_admission_snapshot * out) const {
    if (out == nullptr) {
        return false;
    }
    llama_memory_admission_snapshot attn = {};
    if (mem_attn != nullptr && !mem_attn->get_admission_snapshot(&attn)) {
        attn = {};
    }
    const uint32_t recr_cap  = mem_recr ? mem_recr->get_recurrent_capacity() : 0;
    const uint32_t recr_used = mem_recr ? mem_recr->get_recurrent_used()     : 0;
    *out = llama_memory_admission_combine(attn, recr_cap, recr_used);
    return out->logical_capacity != 0 || out->recurrent_capacity != 0;
}

llama_memory_maintenance_status llama_memory_hybrid::maintain_safe_boundary() {
    return mem_attn ? mem_attn->maintain_safe_boundary()
                      : LLAMA_MEMORY_MAINTENANCE_NO_ACTION;
}

bool llama_memory_hybrid::get_xkv_runtime_snapshot(
        struct llama_memory_xkv_runtime_snapshot * out) const {
    if (out == nullptr) {
        return false;
    }
    // Runtime observation lives in the attention cache; recurrent-only
    // hybrids report unbound so the server shows not_evaluated.
    return mem_attn ? mem_attn->get_xkv_runtime_snapshot(out) : false;
}

uint32_t llama_memory_hybrid::get_recurrent_capacity() const {
    return mem_recr->get_recurrent_capacity();
}

uint32_t llama_memory_hybrid::get_recurrent_used() const {
    return mem_recr->get_recurrent_used();
}

uint32_t llama_memory_hybrid::get_recurrent_seq_used(llama_seq_id seq_id) const {
    return mem_recr->get_recurrent_seq_used(seq_id);
}

void llama_memory_hybrid::clear(bool data) {
    mem_attn->clear(data);
    mem_recr->clear(data);
}

bool llama_memory_hybrid::try_clear(bool data, std::string * err) {
    if (!mem_attn->try_clear(data, err)) {
        return false;
    }
    return mem_recr->try_clear(data, err);
}

bool llama_memory_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // Attention removal can fail if bounded removal release/preflight fails.
    // Remove attention first; if attention removal fails, recurrent cache is
    // left untouched.
    if (!mem_attn->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return mem_recr->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

bool llama_memory_hybrid::seq_rm_attention(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    return mem_attn->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid::seq_cp_attention(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos p0,
        llama_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

bool llama_memory_hybrid::seq_rm_recurrent(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    return mem_recr->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid::seq_cp_recurrent(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos p0,
        llama_pos p1) {
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

bool llama_memory_hybrid::rerot_set_write_tag(
        llama_seq_id seq_id,
        const llama_kv_rerot_meta & tag) {
    if (!mem_recr->rerot_set_write_tag(seq_id, tag)) {
        return false;
    }
    if (!mem_attn->rerot_set_write_tag(seq_id, tag)) {
        mem_recr->rerot_clear_write_tag(seq_id);
        return false;
    }
    return true;
}

void llama_memory_hybrid::rerot_clear_write_tag(llama_seq_id seq_id) {
    mem_attn->rerot_clear_write_tag(seq_id);
    mem_recr->rerot_clear_write_tag(seq_id);
}

void llama_memory_hybrid::rerot_release_episode(uint64_t episode_id) {
    mem_recr->rerot_release_episode(episode_id);
}

bool llama_memory_hybrid::rerot_can_publish_run(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        size_t * count) const {
    return mem_attn->rerot_can_publish_run(episode_id, run_id, count);
}

bool llama_memory_hybrid::rerot_can_reclassify_run(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        llama_rerot_visibility expected,
        llama_rerot_visibility replacement,
        uint64_t publish_epoch,
        size_t * count) const {
    return mem_attn->rerot_can_reclassify_run(
        episode_id, run_id, expected, replacement, publish_epoch, count);
}

size_t llama_memory_hybrid::rerot_publish_run(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        uint64_t publish_epoch) {
    return mem_attn->rerot_publish_run(episode_id, run_id, publish_epoch);
}

size_t llama_memory_hybrid::rerot_reclassify_run(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        llama_rerot_visibility expected,
        llama_rerot_visibility replacement,
        uint64_t publish_epoch) {
    return mem_attn->rerot_reclassify_run(
        episode_id, run_id, expected, replacement, publish_epoch);
}

bool llama_memory_hybrid::rerot_can_add_run_ref(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        llama_seq_id seq_id,
        size_t * count) const {
    return mem_attn->rerot_can_add_run_ref(episode_id, run_id, seq_id, count);
}

size_t llama_memory_hybrid::rerot_add_run_ref(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        llama_seq_id seq_id) {
    return mem_attn->rerot_add_run_ref(episode_id, run_id, seq_id);
}

bool llama_memory_hybrid::rerot_set_reader_view(
        llama_seq_id seq_id,
        const llama_rerot_reader_state & view) {
    return mem_attn->rerot_set_reader_view(seq_id, view);
}

void llama_memory_hybrid::rerot_clear_reader_view(llama_seq_id seq_id) {
    mem_attn->rerot_clear_reader_view(seq_id);
}

size_t llama_memory_hybrid::rerot_hand_seed_size(llama_seq_id source_seq) const {
    return mem_recr ? mem_recr->rerot_hand_seed_size(source_seq) : 0;
}

bool llama_memory_hybrid::rerot_capture_hand_seed(llama_seq_id source_seq, std::vector<uint8_t> & seed_out) {
    return mem_recr ? mem_recr->rerot_capture_hand_seed(source_seq, seed_out) : false;
}

bool llama_memory_hybrid::rerot_apply_hand_seed(llama_seq_id dest_seq, const std::vector<uint8_t> & seed_in) {
    return mem_recr ? mem_recr->rerot_apply_hand_seed(dest_seq, seed_in) : false;
}

bool llama_memory_hybrid::rerot_commit_rbb_frontier(
        uint32_t person_id,
        const llama_seq_id * candidate_seqs,
        const uint8_t * is_public_write,
        size_t n_candidates) {
    return mem_recr ? mem_recr->rerot_commit_rbb_frontier(person_id, candidate_seqs, is_public_write, n_candidates) : false;
}

void llama_memory_hybrid::seq_keep(llama_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
}

void llama_memory_hybrid::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_hybrid::seq_pos_min(llama_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid::seq_pos_max(llama_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

// Both halves, because a probe has to see the real footprint of both: the attention cache and the
// recurrent state are backed lazily by the driver just like everything else.
void llama_memory_hybrid::materialize() const {
    mem_attn->materialize();
    mem_recr->materialize();
}

void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_write(io, seq_id, flags);
    }
    mem_recr->state_write(io, seq_id, flags);
}

void llama_memory_hybrid::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_read(io, seq_id, flags);
    }
    mem_recr->state_read(io, seq_id, flags);
}

llama_memory_kv_reclaim_result llama_memory_hybrid::reclaim_kv(const llama_memory_kv_reclaim_request & request) {
    return mem_attn->reclaim_kv(request);
}

bool llama_memory_hybrid::positions_are_sparse() const {
    return mem_attn->positions_are_sparse();
}

llama_kv_cache * llama_memory_hybrid::get_mem_attn() const {
    return mem_attn.get();
}

llama_memory_recurrent * llama_memory_hybrid::get_mem_recr() const {
    return mem_recr.get();
}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_status status) : status(status) {}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_hybrid * mem) :
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize) :
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llama_ubatch>   ubatches,
        std::vector<llama_xkv::xkv_hot_reservation> hot_res_attn) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    // Target attention receives the real hot reservations.
    ctx_attn(new llama_kv_cache_context(mem->get_mem_attn(), std::move(sinfos_attn), this->ubatches, std::move(hot_res_attn))),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

bool llama_memory_hybrid_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    postcompute_finalized = false;
    postcompute_ok = false;

    ctx_attn->next();
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    postcompute_finalized = false;
    postcompute_ok = false;

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();

    return res;
}

bool llama_memory_hybrid_context::postcompute_success() {
    // Exactly-once forward to attention and recurrent (null-guarded for
    // failure-status contexts). A failed success leaves the forward open.
    if (postcompute_finalized) {
        return postcompute_ok;
    }
    const bool ok_attn = ctx_attn ? ctx_attn->postcompute_success() : true;
    const bool ok_recr = ctx_recr ? ctx_recr->postcompute_success() : true;
    postcompute_ok = ok_attn && ok_recr;
    if (postcompute_ok) {
    postcompute_finalized = true;
    }
    return postcompute_ok;
    }

bool llama_memory_hybrid_context::postcompute_failure() {
    if (postcompute_finalized) {
        return postcompute_ok;
    }
    const bool ok_attn = ctx_attn ? ctx_attn->postcompute_failure() : true;
    const bool ok_recr = ctx_recr ? ctx_recr->postcompute_failure() : true;
    postcompute_ok = ok_attn && ok_recr;
    if (postcompute_ok) {
    postcompute_finalized = true;
    }
    return postcompute_ok;
    }

ggml_tensor * llama_memory_hybrid_context::get_xkv_hot_k(ggml_context * ctx, int32_t il) const {
    return ctx_attn ? ctx_attn->get_xkv_hot_k(ctx, il) : nullptr;
}

ggml_tensor * llama_memory_hybrid_context::get_xkv_hot_v(ggml_context * ctx, int32_t il) const {
    return ctx_attn ? ctx_attn->get_xkv_hot_v(ctx, il) : nullptr;
}

llama_memory_status llama_memory_hybrid_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_context * llama_memory_hybrid_context::get_attn() const {
    return static_cast<const llama_kv_cache_context *>(ctx_attn.get());
}

ggml_tensor * llama_memory_hybrid_context::get_turbo_rot_forward() const {
    return ctx_attn ? ctx_attn->get_turbo_rot_forward() : nullptr;
}

ggml_tensor * llama_memory_hybrid_context::get_turbo_rot_inverse() const {
    return ctx_attn ? ctx_attn->get_turbo_rot_inverse() : nullptr;
}

ggml_tensor * llama_memory_hybrid_context::get_turbo_innerq_scale_inv() const {
    return ctx_attn ? ctx_attn->get_turbo_innerq_scale_inv() : nullptr;
}

const llama_memory_recurrent_context * llama_memory_hybrid_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}
