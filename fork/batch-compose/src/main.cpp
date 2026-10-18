#include <llama.h>
#include <ggml-backend.h>
#include <tq/scheduling/prefill.h>
#include <tq/scheduling/admission.h>
#include <tq/text/stream.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using tq::scheduling::stage;

struct options {
    std::string model;
    std::string prompts = "-";
    std::uint32_t predict = 32;
    std::uint32_t context = 2048;
    std::uint32_t total_kv = 0;
    std::uint32_t batch = 512;
    std::uint32_t ubatch = 128;
    std::uint32_t threads = 4;
    std::uint32_t gpu_layers = 0;
    std::uint32_t parallel = 0;
    std::optional<std::uint32_t> inject_after;
    std::vector<std::string> stops;
    bool phase_memory = true;
    bool ignore_eog = false;
    bool json = false;
    bool stream = false;
    bool help = false;
};

std::uint32_t number(std::string_view text, const std::string & name, bool allow_zero = false) {
    std::uint32_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
        (!allow_zero && value == 0) || value > std::uint32_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(name + " requires a " + (allow_zero ? "non-negative" : "positive") +
                                    " decimal integer <= INT32_MAX");
    }
    return value;
}

void usage() {
    std::cout <<
        "Usage: tq-batch-compose --model MODEL.gguf [options]\n"
        "  --prompts FILE        One raw prompt per line; default: standard input (-).\n"
        "  --predict N           Generated tokens per request; default: 32.\n"
        "  --ctx-size N          Logical context per sequence; default: 2048.\n"
        "  --total-kv N          Explicit physical unified KV capacity in tokens.\n"
        "                        Default: enough for the largest concurrent requests.\n"
        "  --parallel N          Maximum resident requests; default: min(input count,\n"
        "                        batch-size). Queue excess requests in input order.\n"
        "  --batch-size N        Maximum rows per decode call; default: 512.\n"
        "  --ubatch-size N       Maximum physical microbatch; default: 128.\n"
        "  --threads N           CPU threads; default: 4.\n"
        "  --gpu-layers N        Requested offloaded layers; default: 0.\n"
        "  --inject-after N      Start two requests, then admit the remaining prompts\n"
        "                        after N successful generation batches.\n"
        "  --phase-memory        Release compute memory at prompt-to-generation\n"
        "                        transitions (default); errors are fatal.\n"
        "  --no-phase-memory     Keep compute memory for a real comparison.\n"
        "  --ignore-eog          Ignore model EOG tokens; stop strings still apply.\n"
        "  --stop TEXT           Stop at the first completed string; repeatable.\n"
        "  --stream              Flush UTF-8 text deltas; requires --json.\n"
        "  --json                Write JSON Lines for real batches and results.\n";
}

options parse(int argc, char ** argv, bool & json_errors) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--json") { result.json = true; json_errors = true; continue; }
        if (arg == "--help") { result.help = true; continue; }
        if (arg == "--stream") { result.stream = true; continue; }
        if (arg == "--phase-memory") { result.phase_memory = true; continue; }
        if (arg == "--no-phase-memory") { result.phase_memory = false; continue; }
        if (arg == "--ignore-eog") { result.ignore_eog = true; continue; }
        if (arg != "--model" && arg != "--prompts" && arg != "--predict" && arg != "--ctx-size" &&
            arg != "--total-kv" && arg != "--batch-size" && arg != "--ubatch-size" &&
            arg != "--threads" && arg != "--gpu-layers" && arg != "--inject-after" && arg != "--parallel" && arg != "--stop") {
            throw std::invalid_argument("unknown option: " + arg);
        }
        if (++i == argc) { throw std::invalid_argument("missing value for " + arg); }
        const std::string value = argv[i];
        if (arg == "--model") { result.model = value; }
        else if (arg == "--stop") { result.stops.push_back(value); }
        else if (arg == "--prompts") { result.prompts = value; }
        else if (arg == "--predict") { result.predict = number(value, arg); }
        else if (arg == "--ctx-size") { result.context = number(value, arg); }
        else if (arg == "--total-kv") { result.total_kv = number(value, arg); }
        else if (arg == "--batch-size") { result.batch = number(value, arg); }
        else if (arg == "--ubatch-size") { result.ubatch = number(value, arg); }
        else if (arg == "--threads") { result.threads = number(value, arg); }
        else if (arg == "--gpu-layers") { result.gpu_layers = number(value, arg, true); }
        else if (arg == "--parallel") { result.parallel = number(value, arg); }
        else { result.inject_after = number(value, arg); }
    }
    if (result.help) { return result; }
    if (result.model.empty()) { throw std::invalid_argument("--model is required"); }
    if (result.stream && !result.json) { throw std::invalid_argument("--stream requires --json"); }
    const tq::text::generation_stream validated_stops(result.stops);
    if (result.ubatch > result.batch) { throw std::invalid_argument("--ubatch-size exceeds --batch-size"); }
    if (result.parallel > result.batch) { throw std::invalid_argument("--parallel exceeds --batch-size"); }
    if (result.inject_after && result.parallel == 1) { throw std::invalid_argument("--inject-after requires at least two resident requests"); }
    if (result.inject_after && std::uint64_t(*result.inject_after) + 2 > result.predict) {
        throw std::invalid_argument("injection while two requests are generating requires --predict >= --inject-after + 2");
    }
    return result;
}

// Keep valid UTF-8; replace incomplete or invalid bytes in display text. Token
// ids in results and traces remain exact, including a truncated UTF-8 tail.
std::string quote(std::string_view text) {
    std::string result = "\"";
    const char hex[] = "0123456789abcdef";
    for (std::size_t i = 0; i < text.size();) {
        const auto c = static_cast<unsigned char>(text[i]);
        if (c < 0x80) {
            if (c == '"' || c == '\\') { result += '\\'; result += char(c); }
            else if (c < 0x20) { result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15]; }
            else { result += char(c); }
            ++i;
            continue;
        }
        std::size_t count = c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
        bool valid = count != 0 && i + count <= text.size();
        std::uint32_t scalar = count == 2 ? c & 31u : count == 3 ? c & 15u : c & 7u;
        for (std::size_t j = 1; valid && j < count; ++j) {
            const auto tail = static_cast<unsigned char>(text[i + j]);
            valid = (tail & 0xc0) == 0x80;
            scalar = (scalar << 6) | (tail & 63u);
        }
        valid = valid && scalar >= (count == 2 ? 0x80u : count == 3 ? 0x800u : 0x10000u) &&
                scalar <= 0x10ffff && !(scalar >= 0xd800 && scalar <= 0xdfff);
        if (valid) { result.append(text.substr(i, count)); i += count; }
        else { result += "\\ufffd"; ++i; }
    }
    result += '"';
    return result;
}

const char * phase_name(stage value) {
    return value == stage::prompt ? "prompt" : "generation";
}

class backend_owner {
public:
    backend_owner() { ggml_backend_load_all(); llama_backend_init(); }
    ~backend_owner() { llama_backend_free(); }
    backend_owner(const backend_owner &) = delete;
    backend_owner & operator=(const backend_owner &) = delete;
};

using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
using sampler_ptr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;

class batch_owner {
public:
    explicit batch_owner(std::uint32_t capacity) : value(llama_batch_init(std::int32_t(capacity), 0, 1)) {}
    ~batch_owner() { llama_batch_free(value); }
    batch_owner(const batch_owner &) = delete;
    batch_owner & operator=(const batch_owner &) = delete;
    llama_batch value;
};

struct request {
    std::vector<llama_token> prompt;
    std::vector<llama_token> generated;
    std::string text;
    std::optional<tq::text::generation_stream> text_stream;
    std::optional<std::size_t> stop_index;
    sampler_ptr sampler{nullptr, llama_sampler_free};
    std::optional<tq::scheduling::reservation> reservation;
    std::optional<llama_token> pending;
    std::size_t prompt_done = 0;
    llama_pos position = 0;
    stage state = stage::idle;
    bool finished = false;
    bool eligible = false;
    std::string finish_reason;
};

struct row {
    std::uint32_t request_id;
    llama_pos position;
    llama_token token;
    bool logits;
};

struct sampled {
    std::uint32_t request_id;
    llama_token token;
    bool eog;
    bool finished;
    float logit_best;
    float logit_next;
    llama_token candidate_next;
    tq::text::stream_delta output;
};

class composition {
public:
    explicit composition(options config) : config_(std::move(config)) {
        auto params = llama_model_default_params();
        params.n_gpu_layers = std::int32_t(config_.gpu_layers);
        model_.reset(llama_model_load_from_file(config_.model.c_str(), params));
        if (!model_) { throw std::runtime_error("A could not load the model"); }
        vocab_ = llama_model_get_vocab(model_.get());
        if (!vocab_ || llama_vocab_n_tokens(vocab_) <= 0) { throw std::runtime_error("A returned no vocabulary"); }
        load_prompts();

        active_limit_ = std::uint32_t(std::min(requests_.size(), std::size_t(config_.parallel ? config_.parallel : config_.batch)));
        std::vector<std::uint64_t> needs;
        for (const auto & item : requests_) {
            const auto cells = token_demand(item);
            if (cells > config_.context) {
                throw std::invalid_argument("a prompt plus requested generation exceeds --ctx-size");
            }
            needs.push_back(cells);
        }
        std::sort(needs.rbegin(), needs.rend());
        std::uint64_t needed = 0;
        for (std::size_t i = 0; i < active_limit_; ++i) {
            if (needs[i] > std::numeric_limits<std::uint64_t>::max() - needed) {
                throw std::invalid_argument("concurrent token demand overflow");
            }
            needed += needs[i];
        }
        const auto physical = config_.total_kv ? std::uint64_t(config_.total_kv) :
            std::max(std::uint64_t(config_.context), needed);
        if (physical > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("required physical KV exceeds INT32_MAX");
        }
        auto context_params = llama_context_default_params();
        context_params.n_ctx = config_.context;
        context_params.n_ctx_kv = std::uint32_t(physical);
        context_params.kv_unified = true;
        context_params.n_seq_max = active_limit_;
        context_params.n_batch = config_.batch;
        context_params.n_ubatch = config_.ubatch;
        context_params.n_outputs_max = active_limit_;
        context_params.n_threads = std::int32_t(config_.threads);
        context_params.n_threads_batch = std::int32_t(config_.threads);
        context_.reset(llama_init_from_model(model_.get(), context_params));
        if (!context_) { throw std::runtime_error("A rejected the context configuration"); }
        if (!llama_get_memory(context_.get())) { throw std::runtime_error("this runner requires a decoder model with sequence memory"); }
        if (llama_n_batch(context_.get()) < active_limit_ || llama_n_seq_max(context_.get()) < active_limit_) {
            throw std::runtime_error("A cannot hold the configured resident request count");
        }
        admission_ = std::make_unique<tq::scheduling::admission_budget>(llama_n_ctx_kv(context_.get()), active_limit_);
        if (config_.inject_after && (active_limit_ < 2 ||
            token_demand(requests_[0]) + token_demand(requests_[1]) > admission_->capacity())) {
            throw std::invalid_argument("--inject-after requires capacity for both initial requests");
        }
        batch_.reset(new batch_owner(llama_n_batch(context_.get())));
        candidates_.resize(std::size_t(llama_vocab_n_tokens(vocab_)));
        if (config_.json) {
            std::cout << "{\"event\":\"config\",\"requests\":" << requests_.size()
                      << ",\"logical_context_requested\":" << config_.context
                      << ",\"logical_context_actual\":" << llama_n_ctx_seq(context_.get())
                      << ",\"total_kv_requested\":" << physical
                      << ",\"total_kv_actual\":" << llama_n_ctx_kv(context_.get())
                      << ",\"batch_actual\":" << llama_n_batch(context_.get())
                      << ",\"parallel_actual\":" << active_limit_
                      << ",\"gpu_layers_requested\":" << config_.gpu_layers
                      << ",\"phase_memory\":" << (config_.phase_memory ? "true" : "false")
                      << ",\"stream\":" << (config_.stream ? "true" : "false") << ",\"stops\":[";
            for (std::size_t i = 0; i < config_.stops.size(); ++i) {
                if (i) { std::cout << ','; }
                std::cout << quote(config_.stops[i]);
            }
            std::cout << "]}\n";
            if (config_.stream) { flush_output(); }
        }
    }

    void run() {
        while (true) {
            inject_if_ready();
            admit_ready();
            std::vector<tq::scheduling::slot_state> snapshot;
            for (std::size_t i = 0; i < requests_.size(); ++i) {
                if (requests_[i].reservation) {
                    snapshot.push_back({std::uint32_t(i), requests_[i].state});
                }
            }
            const auto selected = selector_.choose(snapshot);
            if (!selected.prompt_slot && selected.generation_slots.empty()) {
                if (config_.inject_after && !injected_) {
                    throw std::runtime_error("initial requests stopped before injection; reduce --inject-after or use --ignore-eog");
                }
                break;
            }
            change_phase(selected.next);
            std::vector<row> rows;
            if (selected.next == stage::prompt) {
                auto & item = requests_.at(*selected.prompt_slot);
                const auto count = std::min(std::size_t(llama_n_batch(context_.get())), item.prompt.size() - item.prompt_done);
                for (std::size_t i = 0; i < count; ++i) {
                    rows.push_back({*selected.prompt_slot, item.position + llama_pos(i), item.prompt[item.prompt_done + i],
                                    item.prompt_done + i + 1 == item.prompt.size()});
                }
            } else {
                for (const auto id : selected.generation_slots) {
                    auto & item = requests_.at(id);
                    if (!item.pending) { throw std::logic_error("generation has no sampled input token"); }
                    rows.push_back({id, item.position, *item.pending, true});
                }
            }
            consume(selected.next, rows);
        }
        llama_synchronize(context_.get());
        if (admission_->active_count() != 0 || admission_->reserved_tokens() != 0) {
            throw std::logic_error("completed requests retained an admission reservation");
        }
        for (std::size_t id = 0; id < requests_.size(); ++id) {
            const auto & item = requests_[id];
            if (!item.finished) { throw std::logic_error("request did not finish"); }
            if (config_.json) {
                std::cout << "{\"event\":\"result\",\"request_id\":" << id
                          << ",\"prompt_tokens\":" << item.prompt.size()
                          << ",\"decoded_rows\":" << item.position
                          << ",\"finish_reason\":" << quote(item.finish_reason) << ",\"tokens\":[";
                for (std::size_t i = 0; i < item.generated.size(); ++i) {
                    if (i) { std::cout << ','; }
                    std::cout << item.generated[i];
                }
                std::cout << "],\"text\":" << quote(item.text) << ",\"stop_index\":";
                if (item.stop_index) { std::cout << *item.stop_index; }
                else { std::cout << "null"; }
                std::cout << "}\n";
            } else {
                std::cout << "[request " << id << "]\n" << item.text << "\n\n";
            }
        }
        if (!std::cout) { throw std::runtime_error("could not write output"); }
    }

private:
    std::uint64_t token_demand(const request & item) const {
        return std::uint64_t(item.prompt.size()) + config_.predict - 1;
    }

    void admit_ready() {
        for (std::size_t id = 0; id < requests_.size(); ++id) {
            auto & item = requests_[id];
            if (!item.eligible || item.finished || item.reservation) { continue; }
            auto held = admission_->try_acquire(id, token_demand(item));
            if (!held) {
                if (config_.json) {
                    std::cout << "{\"event\":\"wait\",\"request_id\":" << id
                              << ",\"tokens_required\":" << token_demand(item)
                              << ",\"tokens_available\":" << admission_->available_tokens()
                              << ",\"active_requests\":" << admission_->active_count() << "}\n";
                }
                break;
            }
            try {
                item.text_stream.emplace(config_.stops);
                item.sampler.reset(llama_sampler_init_greedy());
                if (!item.sampler) { throw std::runtime_error("A could not create a greedy sampler"); }
            } catch (...) {
                item.text_stream.reset();
                item.sampler.reset();
                admission_->release(*held);
                throw;
            }
            item.reservation = std::move(held);
            item.state = stage::prompt;
            if (config_.json) {
                std::cout << "{\"event\":\"admit\",\"request_id\":" << id
                          << ",\"sequence_id\":" << item.reservation->slot()
                          << ",\"generation\":" << item.reservation->generation()
                          << ",\"tokens_reserved\":" << item.reservation->tokens()
                          << ",\"total_reserved\":" << admission_->reserved_tokens() << "}\n";
            }
        }
    }

    void load_prompts() {
        std::ifstream file;
        std::istream * input = &std::cin;
        if (config_.prompts != "-") {
            file.open(config_.prompts);
            if (!file) { throw std::runtime_error("cannot open prompt file: " + config_.prompts); }
            input = &file;
        }
        std::string line;
        while (std::getline(*input, line)) {
            if (!line.empty() && line.back() == '\r') { line.pop_back(); }
            if (line.size() > std::size_t(std::numeric_limits<std::int32_t>::max())) {
                throw std::invalid_argument("prompt is too large to tokenize");
            }
            request item;
            const auto required = llama_tokenize(vocab_, line.data(), std::int32_t(line.size()), nullptr, 0, true, false);
            if (required == std::numeric_limits<std::int32_t>::min()) { throw std::runtime_error("token count overflow"); }
            item.prompt.resize(std::size_t(required < 0 ? -required : required));
            const auto count = llama_tokenize(vocab_, line.data(), std::int32_t(line.size()), item.prompt.data(),
                                              std::int32_t(item.prompt.size()), true, false);
            if (count <= 0) { throw std::runtime_error("A could not tokenize a nonempty sequence"); }
            item.prompt.resize(std::size_t(count));
            item.eligible = !config_.inject_after || requests_.size() < 2;
            if (requests_.size() >= std::size_t(std::numeric_limits<std::int32_t>::max())) {
                throw std::invalid_argument("request count exceeds INT32_MAX");
            }
            requests_.push_back(std::move(item));
        }
        if (input->bad()) { throw std::runtime_error("could not read prompts"); }
        if (requests_.empty()) { throw std::invalid_argument("the prompt input has no lines"); }
        if (config_.inject_after && requests_.size() < 3) { throw std::invalid_argument("--inject-after needs at least three prompts"); }
    }

    void inject_if_ready() {
        if (!config_.inject_after || injected_ || generation_batches_ < *config_.inject_after) { return; }
        if (requests_[0].state != stage::generation || requests_[1].state != stage::generation) {
            throw std::runtime_error("two initial requests must still be generating at injection; use --ignore-eog for fixed-length runs");
        }
        for (std::size_t id = 2; id < requests_.size(); ++id) { requests_[id].eligible = true; }
        injected_ = true;
        if (config_.json) {
            std::cout << "{\"event\":\"inject\",\"after_generation_batches\":" << generation_batches_ << ",\"request_ids\":[";
            for (std::size_t id = 2; id < requests_.size(); ++id) {
                if (id > 2) { std::cout << ','; }
                std::cout << id;
            }
            std::cout << "]}\n";
        }
    }

    void change_phase(stage next) {
        if (phase_ && *phase_ == next) { return; }
        llama_synchronize(context_.get());
        const auto bytes_before = llama_memory_compute_bytes(context_.get());
        llama_set_decode_phase(context_.get(), next == stage::prompt ?
            LLAMA_DECODE_PHASE_PROMPT : LLAMA_DECODE_PHASE_GENERATION);
        const bool release = phase_ && *phase_ == stage::prompt && next == stage::generation && config_.phase_memory;
        if (release && !llama_memory_release_compute(context_.get())) {
            throw std::runtime_error("A refused to release compute memory at a prompt-to-generation transition");
        }
        if (config_.json) {
            std::cout << "{\"event\":\"phase\",\"from\":" << (phase_ ? quote(phase_name(*phase_)) : "null")
                      << ",\"to\":" << quote(phase_name(next)) << ",\"synchronized\":true"
                      << ",\"release_compute_requested\":" << (release ? "true" : "false")
                      << ",\"release_compute_success\":" << (release ? "true" : "null")
                      << ",\"compute_bytes_before\":" << bytes_before
                      << ",\"compute_bytes_after\":" << llama_memory_compute_bytes(context_.get()) << "}\n";
        }
        phase_ = next;
    }

    std::string piece(llama_token token) const {
        char buffer[128];
        const auto count = llama_token_to_piece(vocab_, token, buffer, sizeof(buffer), 0, false);
        if (count >= 0) { return std::string(buffer, std::size_t(count)); }
        if (count == std::numeric_limits<std::int32_t>::min()) { throw std::runtime_error("token piece size overflow"); }
        std::string result(std::size_t(-count), '\0');
        const auto actual = llama_token_to_piece(vocab_, token, result.data(), std::int32_t(result.size()), 0, false);
        if (actual < 0 || std::size_t(actual) > result.size()) { throw std::runtime_error("A could not detokenize a sample"); }
        result.resize(std::size_t(actual));
        return result;
    }

    sampled sample(std::uint32_t id, std::int32_t batch_row) {
        auto & item = requests_.at(id);
        if (!item.text_stream) { throw std::logic_error("resident request has no text stream"); }
        const auto * logits = llama_get_logits_ith(context_.get(), batch_row);
        if (!logits) { throw std::runtime_error("A returned no logits for a requested output row"); }
        float best = -std::numeric_limits<float>::infinity();
        float next = best;
        llama_token best_id = LLAMA_TOKEN_NULL;
        llama_token next_id = LLAMA_TOKEN_NULL;
        for (std::size_t i = 0; i < candidates_.size(); ++i) {
            if (std::isnan(logits[i]) || logits[i] == std::numeric_limits<float>::infinity()) {
                throw std::runtime_error("A returned invalid logits");
            }
            if (logits[i] > best) {
                next = best;
                next_id = best_id;
                best = logits[i];
                best_id = llama_token(i);
            } else if (logits[i] > next) {
                next = logits[i];
                next_id = llama_token(i);
            }
            candidates_[i] = {llama_token(i), logits[i], 0.0f};
        }
        if (best_id == LLAMA_TOKEN_NULL) { throw std::runtime_error("A returned no finite logit candidate"); }
        llama_token_data_array view{candidates_.data(), candidates_.size(), -1, false};
        llama_sampler_apply(item.sampler.get(), &view);
        if (view.selected < 0 || std::size_t(view.selected) >= view.size) {
            throw std::runtime_error("A greedy sampler returned no selected token");
        }
        const auto token = view.data[view.selected].id;
        llama_sampler_accept(item.sampler.get(), token);
        const bool eog = llama_vocab_is_eog(vocab_, token);
        tq::text::stream_delta output;
        if (eog && !config_.ignore_eog) {
            output = item.text_stream->finish();
            item.finish_reason = output.stop_index ? "stop" : "eog";
            item.finished = true;
        } else {
            item.generated.push_back(token);
            output = item.text_stream->push(piece(token));
            if (!output.finished && item.generated.size() >= config_.predict) {
                auto tail = item.text_stream->finish();
                output.text += tail.text;
                output.finished = tail.finished;
                output.stop_index = tail.stop_index;
            }
            if (output.finished) {
                item.finish_reason = output.stop_index ? "stop" : "length";
                item.finished = true;
            } else {
                item.pending = token;
            }
        }
        item.text += output.text;
        item.stop_index = output.stop_index;
        item.state = item.finished ? stage::idle : stage::generation;
        return {id, token, eog, item.finished, best, next, next_id, std::move(output)};
    }

    static void flush_output() {
        std::cout.flush();
        if (!std::cout) { throw std::runtime_error("could not write output"); }
    }

    void publish_text(const sampled & sample) const {
        if (!config_.stream || (sample.output.text.empty() && !sample.output.finished)) { return; }
        const auto & item = requests_.at(sample.request_id);
        std::cout << "{\"event\":\"text\",\"request_id\":" << sample.request_id
                  << ",\"offset_bytes\":" << sample.output.offset
                  << ",\"text\":" << quote(sample.output.text)
                  << ",\"generated_tokens\":" << item.generated.size()
                  << ",\"finished\":" << (sample.output.finished ? "true" : "false")
                  << ",\"finish_reason\":" << (sample.output.finished ? quote(item.finish_reason) : "null")
                  << ",\"stop_index\":";
        if (sample.output.stop_index) { std::cout << *sample.output.stop_index; }
        else { std::cout << "null"; }
        std::cout << "}\n";
        flush_output();
    }

    void trace_batch(stage phase, const std::vector<row> & rows, int returned, const std::vector<sampled> & samples) const {
        if (!config_.json) { return; }
        std::cout << "{\"event\":\"batch\",\"step\":" << step_ << ",\"phase\":" << quote(phase_name(phase))
                  << ",\"decode_return\":" << returned
                  << ",\"compute_bytes\":" << llama_memory_compute_bytes(context_.get()) << ",\"rows\":[";
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (i) { std::cout << ','; }
            const auto & row = rows[i];
            std::cout << "{\"request_id\":" << row.request_id << ",\"position\":" << row.position
                      << ",\"sequence_id\":" << requests_[row.request_id].reservation->slot()
                      << ",\"token\":" << row.token << ",\"logits\":" << (row.logits ? "true" : "false") << '}';
        }
        std::cout << "],\"samples\":[";
        for (std::size_t i = 0; i < samples.size(); ++i) {
            if (i) { std::cout << ','; }
            const auto & sample = samples[i];
            std::cout << "{\"request_id\":" << sample.request_id << ",\"token\":" << sample.token
                      << ",\"eog\":" << (sample.eog ? "true" : "false")
                      << ",\"finished\":" << (sample.finished ? "true" : "false")
                      << ",\"logit_best\":" << std::setprecision(std::numeric_limits<float>::max_digits10) << sample.logit_best
                      << ",\"logit_next\":";
            if (std::isfinite(sample.logit_next)) { std::cout << sample.logit_next; }
            else { std::cout << "null"; }
            std::cout << ",\"candidate_next\":" << sample.candidate_next << '}';
        }
        std::cout << "]}\n";
    }

    void consume(stage phase, const std::vector<row> & rows) {
        if (rows.empty() || rows.size() > llama_n_batch(context_.get())) { throw std::logic_error("invalid planned batch size"); }
        auto & batch = batch_->value;
        batch.n_tokens = std::int32_t(rows.size());
        for (std::size_t i = 0; i < rows.size(); ++i) {
            batch.token[i] = rows[i].token;
            batch.pos[i] = rows[i].position;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = llama_seq_id(requests_.at(rows[i].request_id).reservation->slot());
            batch.logits[i] = rows[i].logits ? 1 : 0;
        }
        const auto returned = llama_decode(context_.get(), batch);
        if (returned != 0) {
            trace_batch(phase, rows, returned, {});
            throw std::runtime_error("A llama_decode failed at batch " + std::to_string(step_) + " with code " + std::to_string(returned));
        }
        std::vector<sampled> samples;
        try {
            // Finish A before changing our committed positions or freeing a sequence.
            llama_synchronize(context_.get());
            for (std::size_t i = 0; i < rows.size(); ++i) {
                auto & item = requests_.at(rows[i].request_id);
                ++item.position;
                if (phase == stage::prompt) { ++item.prompt_done; }
                else { item.pending.reset(); }
                if (rows[i].logits) { samples.push_back(sample(rows[i].request_id, std::int32_t(i))); }
            }
        } catch (...) {
            // A already returned success. Keep that fact even when C cannot
            // consume all of its outputs; the next event is a fatal error.
            trace_batch(phase, rows, returned, samples);
            throw;
        }
        trace_batch(phase, rows, returned, samples);
        for (const auto & sample : samples) { publish_text(sample); }
        ++step_;
        // All output rows have been consumed before any memory mutation.
        for (const auto & sample : samples) {
            if (!sample.finished) { continue; }
            auto & item = requests_.at(sample.request_id);
            const auto held = *item.reservation;
            if (!llama_memory_seq_rm(llama_get_memory(context_.get()), llama_seq_id(held.slot()), 0, -1)) {
                throw std::runtime_error("A could not release a completed sequence");
            }
            if (!admission_->release(held)) { throw std::logic_error("completed reservation was not active"); }
            item.reservation.reset();
            item.sampler.reset();
            item.text_stream.reset();
            if (config_.json) {
                std::cout << "{\"event\":\"release\",\"request_id\":" << sample.request_id
                          << ",\"sequence_id\":" << held.slot() << ",\"generation\":" << held.generation()
                          << ",\"total_reserved\":" << admission_->reserved_tokens() << "}\n";
            }
        }
        if (phase == stage::generation) { ++generation_batches_; }
    }

    options config_;
    backend_owner backend_;
    model_ptr model_{nullptr, llama_model_free};
    context_ptr context_{nullptr, llama_free};
    const llama_vocab * vocab_ = nullptr;
    std::unique_ptr<batch_owner> batch_;
    std::vector<request> requests_;
    std::vector<llama_token_data> candidates_;
    tq::scheduling::prefill_selector selector_;
    std::unique_ptr<tq::scheduling::admission_budget> admission_;
    std::uint32_t active_limit_ = 0;
    std::optional<stage> phase_;
    std::uint64_t step_ = 0;
    std::uint64_t generation_batches_ = 0;
    bool injected_ = false;
};

} // namespace

int main(int argc, char ** argv) {
    bool json = false;
    try {
        auto config = parse(argc, argv, json);
        if (config.help) { usage(); return std::cout ? 0 : 1; }
        composition app(std::move(config));
        app.run();
        return 0;
    } catch (const std::exception & error) {
        if (json) { std::cout << "{\"event\":\"error\",\"message\":" << quote(error.what()) << "}\n"; }
        std::cerr << "tq-batch-compose: " << error.what() << '\n';
        return 1;
    }
}
