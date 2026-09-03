#include "llama-kv-cache-iswa.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>

//
// llama_kv_cache_iswa
//

llama_kv_cache_iswa::llama_kv_cache_iswa(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   swa_full,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_ubatch,
                 uint32_t   n_pad,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share) :
    llama_kv_cache_iswa(model, model.hparams, type_k, type_v, v_trans, offload, swa_full, unified,
            kv_size, n_seq_max, n_ubatch, n_pad, mem_other, filter, reuse, share) {
}

llama_kv_cache_iswa::llama_kv_cache_iswa(
        const llama_model & model,
        const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   swa_full,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_ubatch,
                 uint32_t   n_pad,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share) : unified(unified) {

    // chain filters
    const layer_filter_cb filter_base = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return !model.hparams.is_swa(il);
    };

    const layer_filter_cb filter_swa  = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return  model.hparams.is_swa(il);
    };

    const uint32_t size_base = kv_size;

    // note: the SWA cache is always padded to 256 for performance
    //       https://github.com/ggml-org/llama.cpp/issues/17037
    uint32_t size_swa = GGML_PAD(std::min(size_base, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);

    // when using full-size SWA cache, we set the SWA cache size to be equal to the base cache size
    if (swa_full) {
        LLAMA_LOG_WARN("%s: using full-size SWA cache (ref: %s)\n",
                __func__, "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");

        size_swa = size_base;
    }

    LLAMA_LOG_INFO("%s: creating non-SWA KV cache, size = %u cells\n", __func__, size_base);

    llama_memory_t mem_other_base = nullptr;
    if (mem_other) {
        mem_other_base = static_cast<llama_kv_cache_iswa *>(mem_other)->get_base();
    }

    llama_memory_t mem_other_swa = nullptr;
    if (mem_other) {
        mem_other_swa = static_cast<llama_kv_cache_iswa *>(mem_other)->get_swa();
    }

    kv_base = std::make_unique<llama_kv_cache>(
            model, hparams, type_k, type_v,
            v_trans, offload, unified, size_base, n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, mem_other_base, filter_base, reuse, share);

    LLAMA_LOG_INFO("%s: creating     SWA KV cache, size = %u cells\n", __func__, size_swa);

    kv_swa = std::make_unique<llama_kv_cache>(
            model, hparams, type_k, type_v,
            v_trans, offload, unified, size_swa, n_seq_max, n_pad,
            hparams.n_swa, hparams.swa_type, mem_other_swa, filter_swa, reuse, share);
}

void llama_kv_cache_iswa::clear(bool data) {
    kv_base->clear(data);
    kv_swa ->clear(data);
}

bool llama_kv_cache_iswa::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    bool res = true;

    res = res & kv_base->seq_rm(seq_id, p0, p1);
    res = res & kv_swa ->seq_rm(seq_id, p0, p1);

    return res;
}

void llama_kv_cache_iswa::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    kv_base->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_swa ->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_iswa::seq_keep(llama_seq_id seq_id) {
    kv_base->seq_keep(seq_id);
    kv_swa ->seq_keep(seq_id);
}

void llama_kv_cache_iswa::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kv_base->seq_add(seq_id, p0, p1, shift);
    kv_swa ->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_iswa::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_base->seq_div(seq_id, p0, p1, d);
    kv_swa ->seq_div(seq_id, p0, p1, d);
}

bool llama_kv_cache_iswa::rerot_set_write_tag(
        llama_seq_id seq_id,
        const llama_kv_rerot_meta & tag) {
    // Validate both sides before making either write tag visible.
    if (!kv_base->rerot_set_write_tag(seq_id, tag)) {
        return false;
    }
    if (!kv_swa->rerot_set_write_tag(seq_id, tag)) {
        kv_base->rerot_clear_write_tag(seq_id);
        return false;
    }
    return true;
}

void llama_kv_cache_iswa::rerot_clear_write_tag(llama_seq_id seq_id) {
    kv_base->rerot_clear_write_tag(seq_id);
    kv_swa->rerot_clear_write_tag(seq_id);
}

bool llama_kv_cache_iswa::rerot_can_publish_run(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        size_t * count) const {
    size_t count_base = 0;
    size_t count_swa = 0;
    const bool base_ok = kv_base->rerot_can_publish_run(episode_id, run_id, &count_base);
    const bool swa_ok = kv_swa->rerot_can_publish_run(episode_id, run_id, &count_swa);

    if (count) {
        *count = base_ok && swa_ok ? count_base + count_swa : 0;
    }
    return base_ok && swa_ok;
}

size_t llama_kv_cache_iswa::rerot_publish_run(
        uint64_t episode_id,
        llama_rerot_run_id run_id,
        uint64_t publish_epoch) {
    size_t count = 0;
    if (publish_epoch == 0 || !rerot_can_publish_run(episode_id, run_id, &count)) {
        return 0;
    }

    const size_t base_count = kv_base->rerot_publish_run(episode_id, run_id, publish_epoch);
    const size_t swa_count = kv_swa->rerot_publish_run(episode_id, run_id, publish_epoch);
    GGML_ASSERT(base_count + swa_count == count);
    return count;
}

bool llama_kv_cache_iswa::rerot_set_reader_view(
        llama_seq_id seq_id,
        const llama_rerot_reader_state & view) {
    if (!kv_base->rerot_set_reader_view(seq_id, view)) {
        return false;
    }
    if (!kv_swa->rerot_set_reader_view(seq_id, view)) {
        kv_base->rerot_clear_reader_view(seq_id);
        return false;
    }
    return true;
}

void llama_kv_cache_iswa::rerot_clear_reader_view(llama_seq_id seq_id) {
    kv_base->rerot_clear_reader_view(seq_id);
    kv_swa->rerot_clear_reader_view(seq_id);
}

llama_pos llama_kv_cache_iswa::seq_pos_min(llama_seq_id seq_id) const {
    // the base cache is a superset of the SWA cache, so we can just check the SWA cache
    return kv_swa->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_iswa::seq_pos_max(llama_seq_id seq_id) const {
    return kv_swa->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_iswa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = kv_base->memory_breakdown();
    for (const auto & buft_size : kv_swa->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

llama_memory_context_ptr llama_kv_cache_iswa::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    GGML_UNUSED(embd_all);

    // first try simple split
    do {
        if (!unified) {
            // requires equal splits, so we skip the simple split
            break;
        }

        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_base = kv_base->prepare(ubatches);
        if (sinfos_base.empty()) {
            break;
        }

        auto sinfos_swa = kv_swa->prepare(ubatches);
        if (sinfos_swa.empty()) {
            break;
        }

        assert(sinfos_base.size() == sinfos_swa.size());

        return std::make_unique<llama_kv_cache_iswa_context>(
                this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
    } while (false);

    // if it fails, try equal split
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_equal(n_ubatch, !unified, 0);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_base = kv_base->prepare(ubatches);
        if (sinfos_base.empty()) {
            break;
        }

        auto sinfos_swa = kv_swa->prepare(ubatches);
        if (sinfos_swa.empty()) {
            break;
        }

        assert(sinfos_base.size() == sinfos_swa.size());

        return std::make_unique<llama_kv_cache_iswa_context>(
                this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
    } while (false);

    // TODO: if we fail again, we should attempt different splitting strategies
    //       but to do that properly, we first have to refactor the batches to be more flexible

    return std::make_unique<llama_kv_cache_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_iswa::init_full() {
    return std::make_unique<llama_kv_cache_iswa_context>(this);
}

llama_memory_context_ptr llama_kv_cache_iswa::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_kv_cache_iswa_context>(this, lctx, optimize);
}

llama_memory_context_ptr llama_kv_cache_iswa::init_mtp(llama_seq_id seq_id, llama_ubatch ubatch) {
    llama_kv_cache::slot_info_vec_t sinfos_base;
    llama_kv_cache::slot_info_vec_t sinfos_swa;

    sinfos_base.push_back(kv_base->mtp_slot_info(seq_id));
    sinfos_swa.push_back(kv_swa->mtp_slot_info(seq_id));

    std::vector<llama_ubatch> ubatches;
    ubatches.push_back(std::move(ubatch));

    return std::make_unique<llama_kv_cache_iswa_context>(
            this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
}

bool llama_kv_cache_iswa::get_can_shift() const {
    return kv_base->get_can_shift() &&
           kv_swa->get_can_shift();
}

uint32_t llama_kv_cache_iswa::get_kv_capacity() const {
    return kv_base->get_kv_capacity();
}

uint32_t llama_kv_cache_iswa::get_kv_used() const {
    return kv_base->get_kv_used();
}

uint32_t llama_kv_cache_iswa::get_kv_seq_used(llama_seq_id seq_id) const {
    return kv_base->get_kv_seq_used(seq_id);
}

void llama_kv_cache_iswa::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        kv_base->state_write(io, seq_id, flags);
    }

    kv_swa->state_write(io, seq_id, flags);
}

void llama_kv_cache_iswa::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        kv_base->state_read(io, seq_id, flags);
    }

    kv_swa->state_read(io, seq_id, flags);
}

llama_memory_kv_reclaim_result llama_kv_cache_iswa::reclaim_kv(const llama_memory_kv_reclaim_request & request) {
    return kv_base->reclaim_kv(request);
}

bool llama_kv_cache_iswa::positions_are_sparse() const {
    return kv_base->positions_are_sparse();
}

llama_kv_cache * llama_kv_cache_iswa::get_base() const {
    return kv_base.get();
}

llama_kv_cache * llama_kv_cache_iswa::get_swa() const {
    return kv_swa.get();
}

//
// llama_kv_cache_iswa_context
//

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(llama_memory_status status) : status(status) {}

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(
        llama_kv_cache_iswa * kv) :
    ctx_base(kv->get_base()->init_full()),
    ctx_swa (kv->get_swa ()->init_full()),
    status(llama_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(
        llama_kv_cache_iswa * kv,
        llama_context * lctx,
        bool optimize) :
    ctx_base(kv->get_base()->init_update(lctx, optimize)),
    ctx_swa (kv->get_swa ()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(
        llama_kv_cache_iswa * kv,
        slot_info_vec_t sinfos_base,
        slot_info_vec_t sinfos_swa,
        std::vector<llama_ubatch> ubatches) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_base(new llama_kv_cache_context(kv->get_base(), std::move(sinfos_base), this->ubatches)),
    ctx_swa (new llama_kv_cache_context(kv->get_swa (), std::move(sinfos_swa),  this->ubatches)),
    status(llama_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

llama_kv_cache_iswa_context:: ~llama_kv_cache_iswa_context() = default;

bool llama_kv_cache_iswa_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_base->next();
    ctx_swa ->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_iswa_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_base->apply();
    res = res & ctx_swa ->apply();

    return res;
}

llama_memory_status llama_kv_cache_iswa_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_iswa_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

const llama_kv_cache_context * llama_kv_cache_iswa_context::get_base() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return static_cast<const llama_kv_cache_context *>(ctx_base.get());
}

const llama_kv_cache_context * llama_kv_cache_iswa_context::get_swa()  const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return static_cast<const llama_kv_cache_context *>(ctx_swa.get());
}
