#include "native.h"
#include "text.h"

#include <ggml-backend.h>
#include <tq/relay/submissions.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tq::rerot_compose {
namespace {

class batch_owner {
public:
    explicit batch_owner(std::uint32_t capacity) : value(llama_batch_init(std::int32_t(capacity), 0, 1)) {}
    ~batch_owner() { llama_batch_free(value); }
    batch_owner(const batch_owner &) = delete;
    batch_owner & operator=(const batch_owner &) = delete;
    llama_batch value;
};

llama_pos position(std::int64_t value) {
    if (value < 0 || value > INT32_MAX) { throw std::invalid_argument("reader position exceeds A's position domain"); }
    return llama_pos(value);
}

} // namespace

struct native_execution::impl {
    explicit impl(const options & selected, std::uint32_t stages, std::uint32_t pens,
                  std::function<bool()> stop_query)
        : config(selected), stopped(std::move(stop_query)) {
        if (stages > llama_max_parallel_sequences() || pens == 0 || pens >= stages) {
            throw std::invalid_argument("task graph exceeds A's logical sequence capacity");
        }
        active.assign(stages, false);
        check_cancelled("before model loading");
        if (!config.devices.empty()) {
            std::size_t begin = 0;
            std::unordered_set<ggml_backend_dev_t> unique;
            while (true) {
                const auto end = config.devices.find(',', begin);
                const auto name = config.devices.substr(begin, end == std::string::npos ? end : end - begin);
                const auto device = ggml_backend_dev_by_name(name.c_str());
                if (name.empty() || !device || !unique.insert(device).second ||
                    ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    throw std::invalid_argument("--devices contains an unknown, duplicate, or CPU offload device");
                }
                devices.push_back(device);
                if (end == std::string::npos) { break; }
                begin = end + 1;
            }
            if (devices.size() > llama_max_devices()) { throw std::invalid_argument("too many devices"); }
            devices.push_back(nullptr);
        }
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = std::int32_t(config.gpu_layers);
        model_params.split_mode = config.split == "none" ? LLAMA_SPLIT_MODE_NONE :
                                  config.split == "row" ? LLAMA_SPLIT_MODE_ROW : LLAMA_SPLIT_MODE_LAYER;
        if (!devices.empty()) { model_params.devices = devices.data(); }
        model_params.progress_callback = [](float, void * data) noexcept {
            return !static_cast<impl *>(data)->cancelled();
        };
        model_params.progress_callback_user_data = this;
        model.reset(llama_model_load_from_file(config.model.c_str(), model_params));
        check_cancelled("during model loading");
        if (!model) { throw std::runtime_error("A could not load the model"); }
        check_cancelled("after model loading");
        vocabulary = llama_model_get_vocab(model.get());
        if (!vocabulary || llama_vocab_n_tokens(vocabulary) <= 0) { throw std::runtime_error("model has no vocabulary"); }

        auto params = llama_context_default_params();
        params.n_ctx = config.context;
        params.n_ctx_kv = config.total_kv ? config.total_kv : config.context;
        params.n_seq_max = stages;
        params.n_rs_slots = pens + 1; // One immutable C_base, plus the bound workers.
        params.n_batch = config.batch;
        params.n_ubatch = config.ubatch;
        params.n_outputs_max = std::min(config.batch, stages);
        params.n_threads = std::int32_t(config.threads);
        params.n_threads_batch = std::int32_t(config.threads);
        params.abort_callback = [](void * data) noexcept {
            return static_cast<impl *>(data)->cancelled();
        };
        params.abort_callback_data = this;
        params.kv_unified = true;
        params.batch_invariant_matmul = true;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        // The installed reader interface qualifies F32 KV. F16 reordering
        // can change native rounding enough to cross quantized-model boundaries.
        params.type_k = GGML_TYPE_F32;
        params.type_v = GGML_TYPE_F32;
        context.reset(llama_init_from_model(model.get(), params));
        if (!context) { throw std::runtime_error("A rejected the context configuration"); }
        if (!llama_get_batch_invariant_matmul(context.get())) {
            throw std::runtime_error("A did not retain the requested matrix policy");
        }
        check_cancelled("after context initialization");
        if (const char * reason = llama_attention_views_unsupported_reason(context.get())) {
            throw std::invalid_argument(std::string("reader-view attention is unavailable: ") + reason);
        }
        if (llama_n_seq_max(context.get()) < stages) { throw std::runtime_error("A reduced the required logical sequence capacity"); }
        memory = llama_get_memory(context.get());
        if (!memory) { throw std::invalid_argument("RERoT requires a decoder model with attention memory"); }
        batch = std::make_unique<batch_owner>(llama_n_batch(context.get()));
    }

    ~impl() {
        if (context) { llama_synchronize(context.get()); }
    }

    bool cancelled() noexcept {
        if (stop_seen) { return true; }
        try { stop_seen = stopped && stopped(); }
        catch (...) {
            stop_error = std::current_exception();
            stop_seen = true;
        }
        return stop_seen;
    }

    void check_cancelled(const char * boundary) {
        if (cancelled()) {
            failed = true;
            if (stop_error) { std::rethrow_exception(stop_error); }
            throw std::runtime_error(std::string("cancelled ") + boundary);
        }
    }

    void submit(std::uint64_t frontier) {
        if (failed || !is_drained) { throw std::logic_error("native execution is not ready for a new call"); }
        check_cancelled("before native submission");
        if (next_call == UINT64_MAX) { throw std::overflow_error("native submission identity exhausted"); }
        const auto call = next_call++;
        // Each call is fully drained before the next one. Retaining completed
        // attempts for the entire episode would grow memory and make each
        // drained(call) scan all prior tokens without preserving useful work.
        // The local ledger outlives real completion on both exit paths; call
        // identities and this executor's failure state remain persistent.
        tq::relay::run_submissions submissions(call, 1);
        const auto attempt = submissions.begin_attempt(call, 0);
        if (!attempt) { throw std::logic_error("TQRelay rejected a native submission"); }
        is_drained = false;
        try {
            const auto status = llama_decode(context.get(), batch->value);
            const bool recorded = status == 0 ? submissions.resolve_accepted(*attempt) :
                                               submissions.resolve_unknown(*attempt);
            if (!recorded) { throw std::logic_error("native submission receipt identity mismatch"); }
            // This is the actual native completion proof. No synthetic rank
            // completions are inferred from the logical RERoT cohort size.
            llama_synchronize(context.get());
            is_drained = true;
            if (!submissions.completed(*attempt) || !submissions.drained(call)) {
                throw std::logic_error("TQRelay did not retire completed native work");
            }
            llama_attention_clear_views(context.get());
            if (status == 2) { check_cancelled("during native execution"); }
            if (status != 0) {
                throw std::runtime_error("A decode failed at frontier " + std::to_string(frontier) +
                                         " with status " + std::to_string(status) + "; episode cannot continue");
            }
            // In particular, a long FRAME or base prefill must not submit its
            // remaining waves after cancellation. The current call is really
            // drained and its owned reader metadata is already released.
            check_cancelled("after native completion");
        } catch (...) {
            failed = true;
            // A can leave successful earlier microbatches in its memory on
            // error. Drain the actual call, but do not retry or call it a
            // rollback. This object will never submit again.
            if (!is_drained) {
                static_cast<void>(submissions.resolve_unknown(*attempt));
                llama_synchronize(context.get());
                is_drained = true;
                static_cast<void>(submissions.completed(*attempt));
            }
            llama_attention_clear_views(context.get());
            submissions.fail("native A execution failed");
            throw;
        }
    }

    options config;
    std::function<bool()> stopped;
    std::exception_ptr stop_error;
    bool stop_seen = false;
    std::vector<ggml_backend_dev_t> devices;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    const llama_vocab * vocabulary = nullptr;
    std::unique_ptr<llama_context, decltype(&llama_free)> context{nullptr, llama_free};
    llama_memory_t memory = nullptr;
    std::unique_ptr<batch_owner> batch;
    std::vector<bool> active;
    std::uint64_t next_call = 1;
    bool is_drained = true;
    bool failed = false;
    bool base_ready = false;
};

native_execution::native_execution(const options & config, std::uint32_t stages, std::uint32_t pens,
                                   std::function<bool()> stopped)
    : state_(std::make_unique<impl>(config, stages, pens, std::move(stopped))) {}
native_execution::~native_execution() = default;
const llama_model * native_execution::model() const noexcept { return state_->model.get(); }
const llama_vocab * native_execution::vocab() const noexcept { return state_->vocabulary; }
std::uint32_t native_execution::context_size() const noexcept { return llama_n_ctx_seq(state_->context.get()); }
std::uint32_t native_execution::kv_capacity() const noexcept { return llama_n_ctx_kv(state_->context.get()); }
std::uint32_t native_execution::sequence_capacity() const noexcept { return llama_n_seq_max(state_->context.get()); }
std::uint64_t native_execution::compute_bytes() const noexcept { return llama_memory_compute_bytes(state_->context.get()); }
bool native_execution::drained() const noexcept { return state_->is_drained; }
bool native_execution::pipeline_parallel_enabled() const noexcept {
    return llama_get_pipeline_parallel(state_->context.get());
}

std::string native_execution::device_summary() const {
    std::string output = "[";
    bool first = true;
    for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        const auto device = ggml_backend_dev_get(index);
        if (!first) { output += ','; } first = false;
        output += "{\"name\":" + json_string(ggml_backend_dev_name(device)) +
                  ",\"description\":" + json_string(ggml_backend_dev_description(device)) +
                  ",\"scope\":\"available\"}";
    }
    output += ']';
    return output;
}

std::string native_execution::context_summary() const {
    const auto * context = state_->context.get();
    const auto type_name = [](ggml_type type) {
        return type == GGML_TYPE_COUNT ? "unavailable" : ggml_type_name(type);
    };
    std::ostringstream output;
    output << "{\"pipeline_parallel_enabled\":" << (pipeline_parallel_enabled() ? "true" : "false")
           << ",\"flash_attention_configured\":" << (llama_get_flash_attention(context) ? "true" : "false")
           << ",\"batch_invariant_matmul_requested\":" << (llama_get_batch_invariant_matmul(context) ? "true" : "false")
           << ",\"batch_size\":" << llama_n_batch(context)
           << ",\"ubatch_size\":" << llama_n_ubatch(context)
           << ",\"kv_type_k\":" << json_string(type_name(llama_get_kv_cache_type_k(context)))
           << ",\"kv_type_v\":" << json_string(type_name(llama_get_kv_cache_type_v(context)))
           << ",\"reader_kv_policy\":\"resident-f32\""
           << ",\"reader_kv_policy_reason\":" << json_string(
               "This reader implementation accepts only resident F32 K and V; other types need a verified reader precision path.")
           << ",\"model_devices\":[";
    const auto count = llama_model_n_devices(state_->model.get());
    for (int index = 0; index < count; ++index) {
        const auto device = llama_model_get_device(state_->model.get(), index);
        if (index != 0) { output << ','; }
        output << "{\"name\":" << json_string(ggml_backend_dev_name(device))
               << ",\"description\":" << json_string(ggml_backend_dev_description(device))
               << ",\"scope\":\"model-selected\"}";
    }
    output << "]}";
    return output.str();
}

void native_execution::prefill(const std::vector<llama_token> & tokens) {
    auto & state = *state_;
    state.check_cancelled("before base prefill");
    if (state.base_ready || tokens.empty() || tokens.size() > context_size()) {
        throw std::invalid_argument("invalid or duplicate C_base prefill");
    }
    bool advanced = false;
    try {
        for (std::size_t begin = 0; begin < tokens.size();) {
            const auto count = std::min(tokens.size() - begin, std::size_t(llama_n_batch(state.context.get())));
            auto & batch = state.batch->value;
            batch.n_tokens = std::int32_t(count);
            for (std::size_t index = 0; index < count; ++index) {
                batch.token[index] = tokens[begin + index];
                batch.pos[index] = llama_pos(begin + index);
                batch.n_seq_id[index] = 1;
                batch.seq_id[index][0] = 0;
                batch.logits[index] = begin + index + 1 == tokens.size();
            }
            state.submit(0);
            advanced = true;
            begin += count;
        }
        state.check_cancelled("after base prefill");
        state.active[0] = true;
        state.base_ready = true;
    } catch (...) {
        // submit owns the actual drain. A completed earlier batch still
        // prevents retry if preparing a later batch throws before submit.
        if (advanced) { state.failed = true; }
        throw;
    }
}

void native_execution::start(tq::rerot::StageId stage) {
    auto & state = *state_;
    state.check_cancelled("before stage start");
    if (!state.base_ready || state.failed || !state.is_drained || stage == 0 ||
        stage >= state.active.size() || state.active[stage]) {
        throw std::logic_error("invalid native stage start");
    }
    // A shares the immutable prefix KV and gives native recurrent state its
    // own copy-on-write lineage. The base sequence is never advanced again.
    llama_memory_seq_cp(state.memory, 0, llama_seq_id(stage), -1, -1);
    state.active[stage] = true;
}

void native_execution::seal(tq::rerot::StageId stage) {
    auto & state = *state_;
    if (!state.is_drained || stage == 0 || stage >= state.active.size() || !state.active[stage]) {
        throw std::logic_error("invalid native stage seal");
    }
    if (!llama_memory_seq_rm_recurrent(state.memory, llama_seq_id(stage))) {
        state.failed = true;
        throw std::runtime_error("A could not retire a sealed stage's recurrent state");
    }
    // Keep its shared attention keys. Only the native recurrent/conv slot is
    // retired, so later readers still see a completed non-ancestor's work.
    state.active[stage] = false;
}

void native_execution::release_compute() {
    auto & state = *state_;
    if (!state.is_drained || state.failed || !llama_memory_release_compute(state.context.get())) {
        throw std::runtime_error("A could not release transient compute memory");
    }
}

std::vector<output_row> native_execution::forward(std::uint64_t frontier, const std::vector<input_row> & rows) {
    auto & state = *state_;
    state.check_cancelled("before frontier preparation");
    if (!state.base_ready || state.failed || rows.empty()) { throw std::logic_error("invalid native frontier"); }
    // A accepts one current query per sequence in each native call. FRAMEs
    // may have many rows, but they remain one unpublished B transaction.
    // Advance their own causal prefixes in token waves while all peer
    // views keep the revision that B froze before the first wave.
    std::vector<std::vector<std::size_t>> by_stage(state.active.size());
    std::vector<tq::rerot::StageId> stage_order;
    std::size_t depth = 0;
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const auto & row = rows[index];
        if (row.stage == 0 || row.stage >= state.active.size() || !state.active[row.stage] ||
            row.view.stamp().stage != row.stage || row.view.stamp().frontier != frontier) {
            throw std::logic_error("reader view does not belong to the active stage and frontier");
        }
        auto & indices = by_stage[row.stage];
        if (indices.empty()) { stage_order.push_back(row.stage); }
        indices.push_back(index);
        depth = std::max(depth, indices.size());
    }
    std::vector<output_row> outputs;
    bool advanced = false;
    try {
        for (std::size_t token = 0; token < depth; ++token) {
            std::vector<std::size_t> wave;
            for (const auto stage : stage_order) {
                if (token < by_stage[stage].size()) { wave.push_back(by_stage[stage][token]); }
            }
            for (std::size_t begin = 0; begin < wave.size();) {
                state.check_cancelled("before native wave preparation");
                const auto count = std::min(wave.size() - begin, std::size_t(llama_n_batch(state.context.get())));
                std::vector<std::vector<llama_attention_entry>> entries(count);
                std::vector<llama_attention_view> views;
                views.reserve(count);
                auto & batch = state.batch->value;
                batch.n_tokens = std::int32_t(count);
                for (std::size_t index = 0; index < count; ++index) {
                    const auto & row = rows[wave[begin + index]];
                    const auto expanded = tq::rerot::expand(row.view, kv_capacity());
                    entries[index].reserve(expanded.size());
                    for (const auto & entry : expanded) {
                        if (entry.stage >= state.active.size()) { throw std::logic_error("reader names an unknown key owner"); }
                        entries[index].push_back({llama_seq_id(entry.stage), position(entry.storage_pos), position(entry.virtual_pos)});
                    }
                    views.push_back({llama_seq_id(row.stage), row.position, position(row.view.query_virtual_pos()),
                                     entries[index].size(), entries[index].data()});
                    batch.token[index] = row.token;
                    batch.pos[index] = row.position;
                    batch.n_seq_id[index] = 1;
                    batch.seq_id[index][0] = llama_seq_id(row.stage);
                    batch.logits[index] = row.logits;
                }
                if (!llama_attention_set_views(state.context.get(), views.data(), views.size())) {
                    state.failed = true;
                    throw std::runtime_error("A rejected the prepared reader views before native execution");
                }
                state.submit(frontier);
                advanced = true;
                for (std::size_t index = 0; index < count; ++index) {
                    const auto & row = rows[wave[begin + index]];
                    if (!row.logits) { continue; }
                    const auto data = llama_get_logits_ith(state.context.get(), std::int32_t(index));
                    if (!data) { state.failed = true; throw std::runtime_error("A returned no logits for a requested stage row"); }
                    const auto size = std::size_t(llama_vocab_n_tokens(state.vocabulary));
                    outputs.push_back({row.stage, std::vector<float>(data, data + size)});
                }
                state.check_cancelled("after native wave");
                begin += count;
            }
        }
        return outputs;
    } catch (...) {
        // The completed waves cannot be rolled back. This also covers output
        // allocation and later view preparation, outside submit's own catch.
        if (advanced) { state.failed = true; }
        throw;
    }
}

} // namespace tq::rerot_compose
