#include "../include/llama-predefined.h"

#include "llama-context.h"
#include "llama-model.h"
#include "llama-predefined-session.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>

const ggml_predefined_capacity * llama_context::predefined_capacity() const {
    return predefined_session ? &predefined_session->capacity : nullptr;
}

bool llama_context::prepare_predefined_mtp(llama_context & draft, uint32_t max_draft_tokens) {
    // This first resource integration is intentionally scoped to the requested
    // TP5 Qwen path, not a silent policy change for all speculative models.
    if (model.arch != LLM_ARCH_QWEN4EXP || !model.get_split_state_ud.has_tp5_plan) {
        return true;
    }
    if (&draft == this || draft.model.arch != model.arch || draft.cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP ||
        cparams.n_seq_max != 1 || draft.cparams.n_seq_max != 1 || cparams.rerot_enabled ||
        draft.cparams.rerot_enabled || model.hparams.no_alloc || draft.model.hparams.no_alloc ||
        compute_pool.get() != draft.compute_pool.get() || n_queued_tokens || draft.n_queued_tokens) {
        LLAMA_LOG_ERROR("%s: maximum-capacity TP5 MTP requires idle single-sequence target/draft with one shared pool\n",
                        __func__);
        return false;
    }
    const uint64_t verify = uint64_t(max_draft_tokens) + 1;
    if (verify > cparams.n_ubatch || verify > cparams.n_batch || verify > cparams.n_outputs_max ||
        verify > draft.cparams.n_ubatch || verify > draft.cparams.n_batch ||
        model.hparams.n_embd_out() != draft.model.hparams.n_embd_out()) {
        LLAMA_LOG_ERROR("%s: maximum MTP verification/catch-up does not fit configured batch/output capacity\n", __func__);
        return false;
    }

    const char * wire = std::getenv("GGML_TP5_WIRE");
    if (wire && std::strcmp(wire, "f16") != 0 && std::strcmp(wire, "f32") != 0) {
        LLAMA_LOG_ERROR("%s: unsupported TP5 wire width\n", __func__);
        return false;
    }
    ggml_predefined_limits limits{};
    limits.version        = GGML_PREDEFINED_ABI_VERSION;
    limits.struct_size    = sizeof(limits);
    limits.sequences      = 1;
    limits.draft_tokens   = max_draft_tokens;
    limits.ubatch_tokens  = std::max(cparams.n_ubatch, draft.cparams.n_ubatch);
    limits.context_tokens = std::min(cparams.n_ctx_seq, draft.cparams.n_ctx_seq);
    limits.output_rows    = std::min(cparams.n_outputs_max, limits.ubatch_tokens);
    limits.model_width    = model.hparams.n_embd;
    limits.hidden_width   = model.hparams.n_embd_out();
    limits.vocabulary     = model.vocab.n_tokens();
    limits.wire_bytes     = wire && std::strcmp(wire, "f32") == 0 ? 4u : 2u;
    limits.ranks          = (uint32_t) model.get_split_state_ud.n_devices;

    ggml_predefined_capacity cap{};
    char error[192];
    if (!ggml_predefined_make_capacity(&limits, &cap, error, sizeof(error))) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, error);
        return false;
    }
    if (predefined_session || draft.predefined_session) {
        // Repeating startup setup is a no-op only for the same immutable
        // session. A new configuration must explicitly destroy the old owner.
        const auto * old = predefined_capacity();
        if (predefined_session != draft.predefined_session || !old ||
            std::memcmp(&old->limits, &limits, sizeof(limits)) != 0) {
            LLAMA_LOG_ERROR("%s: cannot redefine a live maximum-capacity session\n", __func__);
            return false;
        }
        return true;
    }
    const bool old_retain = ggml_backend_sched_compute_pool_get_retain_capacity(compute_pool.get());
    const bool old_sealed = ggml_backend_sched_compute_pool_get_capacity_sealed(compute_pool.get());
    if (old_sealed) {
        LLAMA_LOG_ERROR("%s: compute pool already belongs to a sealed definition\n", __func__);
        return false;
    }
    try {
        auto definition = std::make_shared<llama_predefined_session>(cap);
        ggml_backend_sched_compute_pool_set_retain_capacity(compute_pool.get(), true);
        // Embedding/sampler policy is already finalized by the MTP constructor.
        // Measure/reserve both mathematical entries before sealing. Existing
        // pooling shares max(target,draft) only for IDENTICAL buffer types.
        // Separate model-owned meta buffer types must not be aliased merely
        // because their physical device lists happen to match: split hooks
        // and model userdata belong to their respective wrappers.
        sched_need_reserve = true;
        draft.sched_need_reserve = true;
        sched_reserve();
        draft.sched_reserve();
        if (output_reserve(limits.output_rows) < limits.output_rows ||
            draft.output_reserve(draft.cparams.n_outputs_max) < draft.cparams.n_outputs_max) {
            ggml_backend_sched_compute_pool_set_retain_capacity(compute_pool.get(), old_retain);
            LLAMA_LOG_ERROR("%s: maximum MTP output storage allocation failed\n", __func__);
            return false;
        }
        ggml_backend_sched_compute_pool_set_capacity_sealed(compute_pool.get(), true);
        predefined_session = definition;
        draft.predefined_session = std::move(definition);
        LLAMA_LOG_INFO("%s: rows=%u verify=%u outputs=%u hidden=%u wire=%u; retained/sealed per-buffer-type maximum pool (resource layer)\n",
                       __func__, cap.tokens, cap.verify_tokens, limits.output_rows, limits.hidden_width, limits.wire_bytes);
        return true;
    } catch (const std::exception & e) {
        ggml_backend_sched_compute_pool_set_retain_capacity(compute_pool.get(), old_retain);
        LLAMA_LOG_ERROR("%s: maximum MTP resource definition failed: %s\n", __func__, e.what());
        return false;
    }
}

bool llama_predefined_mtp_reserve(llama_context * target, llama_context * draft, uint32_t max_draft_tokens) {
    return target && draft && target->prepare_predefined_mtp(*draft, max_draft_tokens);
}

bool llama_predefined_get_capacity(const llama_context * ctx, ggml_predefined_capacity * out) {
    if (!ctx || !out || !ctx->predefined_capacity()) {
        return false;
    }
    *out = *ctx->predefined_capacity();
    return true;
}
