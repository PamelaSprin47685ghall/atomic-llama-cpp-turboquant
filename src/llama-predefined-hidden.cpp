#include "llama-predefined-hidden.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-batch.h"
#include "ggml-alloc.h"

#include <cinttypes>
#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>

namespace {
std::unique_ptr<llama_predefined_hidden_store> make_store(ggml_backend_t executor, uint32_t rows,
                                                         uint32_t result_rows, uint32_t width) {
    if (!executor || !rows || !result_rows || !width || result_rows > rows ||
        uint64_t(rows) * width > SIZE_MAX / sizeof(float)) {
        throw std::invalid_argument("invalid device hidden capacity");
    }
    auto out = std::make_unique<llama_predefined_hidden_store>();
    out->executor = executor;
    out->capacity = rows;
    out->result_capacity = result_rows;
    out->width = width;
    ggml_init_params params{};
    params.mem_size = out->tensors.size() * ggml_tensor_overhead();
    params.no_alloc = true;
    out->context.reset(ggml_init(params));
    if (!out->context) {
        throw std::bad_alloc();
    }
    const char * names[] = {"mtp_persistent_result", "mtp_persistent_carry", "mtp_persistent_seed"};
    for (size_t i = 0; i < out->tensors.size(); ++i) {
        out->tensors[i] = ggml_new_tensor_2d(out->context.get(), GGML_TYPE_F32, width,
            i == LLAMA_PREDEFINED_H_RESULT ? result_rows : 1);
        ggml_set_name(out->tensors[i], names[i]);
        ggml_set_input(out->tensors[i]);
    }
    out->buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(out->context.get(),
        ggml_backend_get_default_buffer_type(executor)));
    if (!out->buffer) {
        throw std::bad_alloc();
    }
    return out;
}

uint32_t readable_rows(const llama_predefined_hidden_store & h, unsigned slot) {
    switch (slot) {
        case LLAMA_PREDEFINED_H_RESULT: return h.valid_rows;
        case LLAMA_PREDEFINED_H_CARRY:
        case LLAMA_PREDEFINED_H_SEED:   return 1;
        default:                       return 0;
    }
}

template<typename T, typename F>
T hidden_api(const char * name, T failure, F && operation) {
    try {
        return operation();
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", name, e.what());
        return failure;
    }
}
} // namespace

bool llama_context::enable_predefined_hidden(llama_context & draft) {
    if (&draft == this || model.arch != LLM_ARCH_QWEN4EXP || draft.model.arch != model.arch ||
        !model.get_split_state_ud.has_tp5_plan || cparams.n_seq_max != 1 || draft.cparams.n_seq_max != 1 ||
        draft.model.hparams.n_layer_nextn != 1 ||
        draft.cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP || cparams.rerot_enabled || draft.cparams.rerot_enabled ||
        cparams.pooling_type != LLAMA_POOLING_TYPE_NONE || draft.cparams.pooling_type != LLAMA_POOLING_TYPE_NONE ||
        model.hparams.no_alloc || draft.model.hparams.no_alloc ||
        !cparams.embeddings_nextn || !draft.cparams.embeddings_nextn ||
        cparams.embeddings_nextn_masked || !draft.cparams.embeddings_nextn_masked ||
        model.hparams.n_embd_out() != draft.model.hparams.n_embd_out() ||
        n_queued_tokens || draft.n_queued_tokens || predefined_hidden || draft.predefined_hidden) {
        LLAMA_LOG_ERROR("%s: device hidden requires idle single-sequence TP5 Qwen4EXP target/MTP\n", __func__);
        return false;
    }
    auto executor_for_output = [](llama_context & ctx) -> ggml_backend_t {
        for (auto * backend : ctx.backend_ptrs) {
            if (ggml_backend_get_device(backend) == ctx.model.dev_output()) {
                return backend;
            }
        }
        return nullptr;
    };
    try {
        auto target_store = make_store(executor_for_output(*this), cparams.n_batch,
            cparams.n_batch, model.hparams.n_embd_out());
        auto draft_store = make_store(executor_for_output(draft),
            std::max(cparams.n_batch, draft.cparams.n_batch), 1, draft.model.hparams.n_embd_out());
        const size_t bytes = size_t(target_store->width) * sizeof(float);
        const ggml_device_copy_range test{target_store->tensors[LLAMA_PREDEFINED_H_RESULT],
            draft_store->tensors[LLAMA_PREDEFINED_H_CARRY], 0, 0, bytes};
        const ggml_device_copy_range self{draft_store->tensors[LLAMA_PREDEFINED_H_RESULT],
            draft_store->tensors[LLAMA_PREDEFINED_H_CARRY], 0, 0, bytes};
        if (!ggml_backend_device_copy_ranges(draft_store->executor, &test, 1, true) ||
            !ggml_backend_device_copy_ranges(draft_store->executor, &self, 1, true)) {
            LLAMA_LOG_ERROR("%s: native per-rank local copy unsupported; no host fallback\n", __func__);
            return false;
        }
        // Startup initialization only; future resets touch carry/seed, not the
        // full result/input capacity. No source graph allocation is retained.
        ggml_backend_buffer_clear(target_store->buffer.get(), 0);
        ggml_backend_buffer_clear(draft_store->buffer.get(), 0);
        predefined_hidden = std::move(target_store);
        draft.predefined_hidden = std::move(draft_store);
        LLAMA_LOG_INFO("%s: device hidden width=%u target_result_rows=%u draft_result_rows=%u input_rows=%u\n",
            __func__, predefined_hidden->width, predefined_hidden->result_capacity,
            draft.predefined_hidden->result_capacity, draft.predefined_hidden->capacity);
        return true;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return false;
    }
}

bool llama_context::predefined_hidden_capture(ggml_backend_t producer, const ggml_tensor * tensor,
                                             uint32_t offset, uint32_t rows) {
    if (!predefined_hidden || !tensor || !rows) {
        return false;
    }
    auto & h = *predefined_hidden;
    if (offset != h.captured_rows || offset > h.result_capacity || rows > h.result_capacity - offset) {
        return false;
    }
    const size_t row_bytes = size_t(h.width) * sizeof(float);
    const ggml_device_copy_range copy{tensor, h.tensors[LLAMA_PREDEFINED_H_RESULT], 0,
        size_t(offset) * row_bytes, size_t(rows) * row_bytes};
    // Use the producing backend: the copy follows its actual compute, not an
    // assumed target/draft queue ordering. Executor equality also ensures our
    // pending-copy flag is retired by the same scheduler.
    if (producer != h.executor || !ggml_backend_device_copy_ranges(producer, &copy, 1, true)) {
        return false;
    }
    h.pending = true; // retain/retire even if a later rank fails while recording
    if (!ggml_backend_device_copy_ranges(producer, &copy, 1, false)) {
        return false;
    }
    h.captured_rows += rows;
    h.host_current = false;
    h.pending = true;
    h.synchronized_generation = 0;
    return true;
}

uint32_t llama_context::predefined_hidden_rows() {
    if (!predefined_hidden) {
        return 0;
    }
    // Replaces the old hidden getter's existing completion point. This first
    // integration removes the host byte bridge, not the phase dependency.
    synchronize();
    predefined_hidden->synchronized_generation = predefined_hidden->generation;
    return predefined_hidden->valid_rows;
}

uint64_t llama_context::predefined_hidden_generation() const {
    return predefined_hidden && predefined_hidden->valid_rows ? predefined_hidden->generation : 0;
}

bool llama_context::predefined_hidden_copy(const llama_predefined_hidden_range * ranges, size_t n) {
    if (!predefined_hidden || n > GGML_DEVICE_COPY_MAX_RANGES || (n && !ranges)) {
        return false;
    }
    auto & dst = *predefined_hidden;
    ggml_device_copy_range copies[GGML_DEVICE_COPY_MAX_RANGES]{};
    for (size_t i = 0; i < n; ++i) {
        const auto & r = ranges[i];
        if (!r.source || !r.source->predefined_hidden || unsigned(r.src_slot) >= 3 ||
            unsigned(r.dst_slot) >= 3 || r.dst_slot == LLAMA_PREDEFINED_H_RESULT || !r.rows) {
            return false;
        }
        const auto & src = *r.source->predefined_hidden;
        if (!llama_predefined_hidden_generation_matches(src, r.src_slot, r.generation)) {
            if (std::getenv("GGML_TP5_PROFILE") != nullptr || std::getenv("GGML_TP5_MTP_PROFILE") != nullptr) {
                LLAMA_LOG_WARN("[tp5-mtp-hidden] copy gen mismatch: src_slot=%u exp_gen=%" PRIu64 " actual_gen=%" PRIu64 " valid_rows=%u\n",
                               r.src_slot, r.generation, src.generation, src.valid_rows);
            }
            return false;
        }
        const uint32_t src_rows = readable_rows(src, r.src_slot);
        const uint32_t dst_rows = 1;
        if (src.width != dst.width || r.src_row > src_rows || r.rows > src_rows - r.src_row ||
            r.dst_row > dst_rows || r.rows > dst_rows - r.dst_row) {
            return false;
        }
        const size_t row_bytes = size_t(dst.width) * sizeof(float);
        copies[i] = {src.tensors[r.src_slot], dst.tensors[r.dst_slot],
            size_t(r.src_row) * row_bytes, size_t(r.dst_row) * row_bytes, size_t(r.rows) * row_bytes};
    }
    if (!ggml_backend_device_copy_ranges(dst.executor, copies, n, true)) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        if (ranges[i].source != this) {
            bool seen = false;
            for (size_t j = 0; j < i; ++j) {
                seen |= ranges[j].source == ranges[i].source;
            }
            if (!seen) {
                auto * src_store = ranges[i].source->predefined_hidden.get();
                if (src_store && src_store->synchronized_generation == ranges[i].generation) {
                    if (std::getenv("GGML_TP5_PROFILE") != nullptr || std::getenv("GGML_TP5_MTP_PROFILE") != nullptr) {
                        LLAMA_LOG_INFO("[tp5-mtp-hidden] redundant sync avoided gen=%" PRIu64 "\n", ranges[i].generation);
                    }
                } else {
                    if (std::getenv("GGML_TP5_PROFILE") != nullptr || std::getenv("GGML_TP5_MTP_PROFILE") != nullptr) {
                        LLAMA_LOG_INFO("[tp5-mtp-hidden] redundant CPU sync executed gen=%" PRIu64 " src=%p\n",
                                       ranges[i].generation, (const void *) ranges[i].source);
                    }
                    ranges[i].source->synchronize();
                    if (src_store) {
                        src_store->synchronized_generation = src_store->generation;
                    }
                }
            }
        }
    }
    dst.pending |= n != 0; // includes an exception/partial recording failure
    if (!ggml_backend_device_copy_ranges(dst.executor, copies, n, false)) {
        return false;
    }
    return true;
}

int llama_context::decode_predefined_hidden(llama_batch batch, const llama_predefined_hidden_range * ranges, size_t n) {
    if (!predefined_hidden || cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP ||
        predefined_hidden->bound_input || !batch.token || batch.n_tokens <= 0 ||
        uint32_t(batch.n_tokens) > predefined_hidden->capacity || !ranges || n == 0 || n > GGML_DEVICE_COPY_MAX_RANGES) {
        return -1;
    }
    uint64_t covered = 0;
    std::array<llama_predefined_hidden_store::bound_range, GGML_DEVICE_COPY_MAX_RANGES> bound{};
    for (size_t i = 0; i < n; ++i) {
        const auto & r = ranges[i];
        if (!r.source || !r.source->predefined_hidden || unsigned(r.src_slot) >= 3 ||
            r.dst_slot != LLAMA_PREDEFINED_H_INPUT || r.dst_row != covered || !r.rows) {
            return -1;
        }
        const auto & source = *r.source->predefined_hidden;
        if (!llama_predefined_hidden_generation_matches(source, r.src_slot, r.generation)) {
            if (std::getenv("GGML_TP5_PROFILE") != nullptr || std::getenv("GGML_TP5_MTP_PROFILE") != nullptr) {
                LLAMA_LOG_WARN("[tp5-mtp-hidden] decode gen mismatch: src_slot=%u exp_gen=%" PRIu64 " actual_gen=%" PRIu64 " valid_rows=%u\n",
                               r.src_slot, r.generation, source.generation, source.valid_rows);
            }
            return -1;
        }
        const uint32_t available = readable_rows(source, r.src_slot);
        if (source.width != predefined_hidden->width || r.src_row > available || r.rows > available - r.src_row) {
            return -1;
        }
        bound[i] = {source.tensors[r.src_slot], size_t(r.src_row) * source.width * sizeof(float), r.dst_row, r.rows};
        covered += r.rows;
    }
    if (covered != uint32_t(batch.n_tokens)) {
        return -1;
    }
    try {
        // Cross-owner dependencies still use the existing context completion
        // point. No hidden bytes are downloaded; same-owner feedback stays in
        // queue order and is copied before RESULT is overwritten by this step.
        for (size_t i = 0; i < n; ++i) {
            if (ranges[i].source != this) {
                bool seen = false;
                for (size_t j = 0; j < i; ++j) seen |= ranges[j].source == ranges[i].source;
                if (!seen) {
                    auto * src_store = ranges[i].source->predefined_hidden.get();
                    if (src_store && src_store->synchronized_generation == ranges[i].generation) {
                        if (std::getenv("GGML_TP5_PROFILE") != nullptr || std::getenv("GGML_TP5_MTP_PROFILE") != nullptr) {
                            LLAMA_LOG_INFO("[tp5-mtp-hidden] redundant sync avoided gen=%" PRIu64 "\n", ranges[i].generation);
                        }
                    } else {
                        if (std::getenv("GGML_TP5_PROFILE") != nullptr || std::getenv("GGML_TP5_MTP_PROFILE") != nullptr) {
                            LLAMA_LOG_INFO("[tp5-mtp-hidden] redundant CPU sync executed gen=%" PRIu64 " src=%p\n",
                                           ranges[i].generation, (const void *) ranges[i].source);
                        }
                        ranges[i].source->synchronize();
                        if (src_store) {
                            src_store->synchronized_generation = src_store->generation;
                        }
                    }
                }
            }
        }
        struct input_guard {
            llama_predefined_hidden_store & h;
            ~input_guard() { h.bound_input = false; h.input_rows = 0; h.input_count = 0; h.input = {}; }
        } guard{*predefined_hidden};
        predefined_hidden->input = bound;
        predefined_hidden->input_count = n;
        predefined_hidden->bound_input = true;
        predefined_hidden->input_rows = uint32_t(batch.n_tokens);
        predefined_hidden->pending = true; // includes failures after an input copy was queued
        batch.embd = nullptr; // no hidden bytes copied by the batch allocator
        return decode(batch);
    } catch (const std::exception & e) {
        predefined_hidden->valid_rows = 0;
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return -3;
    }
}

bool llama_context::predefined_hidden_bind_ubatch(llama_ubatch & ubatch, llama_device_hidden_input & input) {
    if (!predefined_hidden || !predefined_hidden->bound_input) {
        return true;
    }
    auto & h = *predefined_hidden;
    if (!llama_predefined_hidden_slice(h.input.data(), h.input_count, h.input_rows, h.width,
                                       ubatch.source_row, ubatch.n_tokens, h.executor, input)) return false;
    ubatch.device_hidden = &input;
    return true;
}

bool llama_context::predefined_hidden_readback() {
    if (!predefined_hidden) {
        return true;
    }
    synchronize();
    auto & h = *predefined_hidden;
    if (!h.valid_rows || !embd_nextn.data || size_t(h.valid_rows) * h.width > embd_nextn.size) {
        return false;
    }
    if (!h.host_current) {
        ggml_backend_tensor_get(h.tensors[LLAMA_PREDEFINED_H_RESULT], embd_nextn.data, 0,
            size_t(h.valid_rows) * h.width * sizeof(float));
        h.host_current = true;
    }
    return true;
}

bool llama_context::predefined_hidden_carry_io(float * data, size_t bytes, bool write) {
    if (!predefined_hidden || !data || bytes != size_t(predefined_hidden->width) * sizeof(float)) {
        return false;
    }
    synchronize();
    auto * tensor = predefined_hidden->tensors[LLAMA_PREDEFINED_H_CARRY];
    if (write) {
        ggml_backend_tensor_set(tensor, data, 0, bytes);
    } else {
        ggml_backend_tensor_get(tensor, data, 0, bytes);
    }
    return true;
}

bool llama_context::predefined_hidden_reset() {
    if (!predefined_hidden || predefined_hidden->bound_input) {
        return false;
    }
    synchronize();
    auto & h = *predefined_hidden;
    ggml_backend_tensor_memset(h.tensors[LLAMA_PREDEFINED_H_CARRY], 0, 0, size_t(h.width) * sizeof(float));
    ggml_backend_tensor_memset(h.tensors[LLAMA_PREDEFINED_H_SEED], 0, 0, size_t(h.width) * sizeof(float));
    h.valid_rows = h.captured_rows = h.input_rows = 0;
    h.synchronized_generation = 0;
    h.input_count = 0;
    h.input = {};
    h.host_current = false;
    return true;
}

bool llama_predefined_hidden_enable(llama_context * target, llama_context * draft) {
    return hidden_api(__func__, false, [&] { return target && draft && target->enable_predefined_hidden(*draft); });
}
uint32_t llama_predefined_hidden_rows(llama_context * ctx) {
    return hidden_api(__func__, uint32_t(0), [&] { return ctx ? ctx->predefined_hidden_rows() : uint32_t(0); });
}
uint64_t llama_predefined_hidden_generation(const llama_context * ctx) {
    return ctx ? ctx->predefined_hidden_generation() : 0;
}
bool llama_predefined_hidden_copy(llama_context * ctx, const llama_predefined_hidden_range * r, size_t n) {
    return hidden_api(__func__, false, [&] { return ctx && ctx->predefined_hidden_copy(r, n); });
}
int llama_predefined_decode_hidden(llama_context * ctx, llama_batch batch, const llama_predefined_hidden_range * r, size_t n) {
    return hidden_api(__func__, -3, [&] { return ctx ? ctx->decode_predefined_hidden(batch, r, n) : -1; });
}
bool llama_predefined_hidden_carry_get(llama_context * ctx, float * data, size_t bytes) {
    return hidden_api(__func__, false, [&] { return ctx && ctx->predefined_hidden_carry_io(data, bytes, false); });
}
bool llama_predefined_hidden_carry_set(llama_context * ctx, const float * data, size_t bytes) {
    return hidden_api(__func__, false, [&] {
        return ctx && ctx->predefined_hidden_carry_io(const_cast<float *>(data), bytes, true);
    });
}
bool llama_predefined_hidden_reset(llama_context * ctx) {
    return hidden_api(__func__, false, [&] { return ctx && ctx->predefined_hidden_reset(); });
}
