#include <llama.h>
#include <ggml.h>
#include <ggml-backend.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using values = std::vector<float>;
constexpr llama_seq_id reader_id = 5;
constexpr llama_seq_id peer_id = 6;

void require(bool condition, const std::string & message) {
    if (!condition) { throw std::runtime_error(message); }
}

bool starts(const std::string & value, const char * prefix) {
    return value.rfind(prefix, 0) == 0;
}

std::string json_string(const std::string & value) {
    std::string result = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (const auto raw : value) {
        const auto c = static_cast<unsigned char>(raw);
        if (c == '\\' || c == '"') { result += '\\'; result += char(c); }
        else if (c < 0x20) { result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15]; }
        else { result += char(c); }
    }
    result += '"';
    return result;
}

struct placement_log {
    placement_log() {
        llama_log_get(&previous, &previous_user);
        llama_log_set(record, this);
    }
    ~placement_log() { llama_log_set(previous, previous_user); }
    placement_log(const placement_log &) = delete;
    placement_log & operator=(const placement_log &) = delete;

    static void record(ggml_log_level level, const char * text, void * user) noexcept {
        auto & self = *static_cast<placement_log *>(user);
        try {
            if (text) {
                if (const auto * start = std::strstr(text, "offloaded ")) {
                    int actual = 0, total = 0;
                    if (std::sscanf(start, "offloaded %d/%d layers to GPU", &actual, &total) == 2) {
                        self.gpu_layers = std::max(self.gpu_layers, actual);
                    }
                }
                if (std::strstr(text, "offloaded ") || std::strstr(text, "model buffer size") ||
                    std::strstr(text, "KV buffer size")) {
                    std::string line(text);
                    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) { line.pop_back(); }
                    self.buffers.push_back(std::move(line));
                }
            }
        } catch (...) { self.failed = true; }
        if (self.previous) { self.previous(level, text, self.previous_user); }
        else if (text) { std::fputs(text, stderr); }
    }

    ggml_log_callback previous = nullptr;
    void * previous_user = nullptr;
    int gpu_layers = 0;
    bool failed = false;
    std::vector<std::string> buffers;
};

struct sample {
    std::array<int64_t, 4> shape;
    std::array<size_t, 4> strides;
    std::string operation;
    std::string source_type;
    std::string source1_type;
    std::string buffer_type;
    std::string device;
    int32_t precision = -1;
    values data;
};

class trace {
public:
    static bool callback(ggml_tensor * tensor, bool ask, void * user) {
        auto & self = *static_cast<trace *>(user);
        const int feature_dims = select(tensor);
        if (ask) { return !self.rows_.empty() && feature_dims > 0; }
        if (self.rows_.empty() || feature_dims == 0) { return true; }
        try { self.copy(tensor, feature_dims); }
        catch (const std::exception & error) {
            self.error_ = error.what();
            return false;
        }
        return true;
    }

    void begin(std::vector<llama_seq_id> rows, bool prefill = false) {
        rows_ = std::move(rows);
        prefill_ = prefill;
        next_prefill_row_ = 0;
        active_rows_.clear();
    }
    void end() {
        const bool complete = !prefill_ || next_prefill_row_ == rows_.size();
        rows_.clear();
        require(error_.empty(), error_);
        require(complete, "prefill trace did not observe every embedding row");
    }

    const std::vector<std::string> & order() const { return order_; }
    const std::map<std::string, std::map<llama_seq_id, sample>> & samples() const { return samples_; }

private:
    static bool embedding(const std::string & name) {
        return name == "model.embed_tokens" || name == "inp_embd";
    }

    static int select(const ggml_tensor * tensor) {
        if (tensor->type != GGML_TYPE_F32) { return 0; }
        if (tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_RESHAPE ||
            tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_PERMUTE) { return 0; }
        const std::string name = ggml_get_name(tensor);
        if (tensor->op == GGML_OP_MUL_MAT &&
            (starts(name, "Qcur-") || starts(name, "Kcur-") || starts(name, "Vcur-") ||
             starts(name, "wqkv-"))) { return 1; }
        if (tensor->op == GGML_OP_MUL_MAT_ID &&
            (starts(name, "ffn_moe_up-") || starts(name, "ffn_moe_gate-") ||
             starts(name, "ffn_moe_down-"))) { return 2; }
        if (name.find("self_attn.q_layernorm") != std::string::npos ||
            name.find("self_attn.k_layernorm") != std::string::npos) { return 2; }
        if (embedding(name) || name == "result_norm" || name == "result_output" ||
            starts(name, "l_out-") || starts(name, "kqv_out-") ||
            starts(name, "ffn_up-") || starts(name, "ffn_gate-") || starts(name, "ffn_down-") ||
            starts(name, "ffn_swiglu-") || starts(name, "ffn_moe_logits-") ||
            starts(name, "ffn_moe_probs-") || starts(name, "ffn_moe_out-") ||
            name.find(".operator_norm-") != std::string::npos ||
            name.find(".conv.in_proj-") != std::string::npos ||
            name.find(".conv.conv-") != std::string::npos ||
            name.find(".conv.out_proj-") != std::string::npos ||
            name.find(".self_attn.out_proj-") != std::string::npos ||
            name.find(".ffn_norm-") != std::string::npos ||
            name.find(".ffn_out-") != std::string::npos) { return 1; }
        return 0;
    }

    void copy(ggml_tensor * tensor, int feature_dims) {
        const std::string name = ggml_get_name(tensor);
        size_t width = 1;
        size_t row_count = 1;
        for (int i = 0; i < feature_dims; ++i) { width *= size_t(tensor->ne[i]); }
        for (int i = feature_dims; i < 4; ++i) { row_count *= size_t(tensor->ne[i]); }
        if (prefill_ && embedding(name)) {
            require(next_prefill_row_ + row_count <= rows_.size(), "too many prefill embedding rows");
            active_rows_.assign(rows_.begin() + next_prefill_row_, rows_.begin() + next_prefill_row_ + row_count);
            next_prefill_row_ += row_count;
        }
        const auto & logical_rows = prefill_ ? active_rows_ : rows_;
        require(row_count == logical_rows.size(), "unexpected token layout for " + name);
        require(ggml_nbytes(tensor) < size_t(128) * 1024 * 1024, "trace tensor is too large");
        std::vector<uint8_t> raw(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
        if (samples_.count(name) == 0) { order_.push_back(name); }
        for (size_t row = 0; row < row_count; ++row) {
            sample item;
            std::copy(tensor->ne, tensor->ne + 4, item.shape.begin());
            std::copy(tensor->nb, tensor->nb + 4, item.strides.begin());
            item.operation = ggml_op_name(tensor->op);
            item.source_type = tensor->src[0] ? ggml_type_name(tensor->src[0]->type) : "none";
            item.source1_type = tensor->src[1] ? ggml_type_name(tensor->src[1]->type) : "none";
            item.precision = tensor->op == GGML_OP_MUL_MAT || tensor->op == GGML_OP_MUL_MAT_ID ?
                    tensor->op_params[0] : -1;
            const auto buffer = tensor->buffer ? tensor->buffer :
                tensor->view_src ? tensor->view_src->buffer : nullptr;
            const auto buffer_type = buffer ? ggml_backend_buffer_get_type(buffer) : nullptr;
            const auto device = buffer_type ? ggml_backend_buft_get_device(buffer_type) : nullptr;
            item.buffer_type = buffer_type ? ggml_backend_buft_name(buffer_type) : "none";
            item.device = device ? ggml_backend_dev_name(device) : "none";
            item.data.resize(width);
            for (size_t column = 0; column < width; ++column) {
                size_t flat = row * width + column;
                size_t offset = 0;
                for (int dim = 0; dim < 4; ++dim) {
                    offset += (flat % size_t(tensor->ne[dim])) * tensor->nb[dim];
                    flat /= size_t(tensor->ne[dim]);
                }
                require(offset + sizeof(float) <= raw.size(), "invalid trace stride for " + name);
                std::memcpy(&item.data[column], raw.data() + offset, sizeof(float));
                require(std::isfinite(item.data[column]), "non-finite trace value in " + name);
            }
            require(samples_[name].emplace(logical_rows[row], std::move(item)).second,
                    "duplicate trace row for " + name);
        }
    }

    std::vector<llama_seq_id> rows_;
    std::vector<llama_seq_id> active_rows_;
    bool prefill_ = false;
    size_t next_prefill_row_ = 0;
    std::string error_;
    std::vector<std::string> order_;
    std::map<std::string, std::map<llama_seq_id, sample>> samples_;
};

struct batch_owner {
    explicit batch_owner(int count) : value(llama_batch_init(count, 0, 1)) {}
    ~batch_owner() { llama_batch_free(value); }
    llama_batch value;
};

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool special) {
    const auto count = llama_tokenize(vocab, text.data(), int32_t(text.size()), nullptr, 0, special, false);
    require(count < 0 && count != INT32_MIN, "could not size tokens");
    std::vector<llama_token> result(static_cast<size_t>(-count));
    const auto actual = llama_tokenize(vocab, text.data(), int32_t(text.size()), result.data(), -count, special, false);
    require(actual > 0 && actual <= -count, "could not tokenize input");
    result.resize(size_t(actual));
    return result;
}

std::vector<uint8_t> snapshot(llama_context * context, llama_seq_id seq) {
    const auto size = llama_state_seq_get_size(context, seq);
    require(size > 0 && size < size_t(512) * 1024 * 1024, "invalid snapshot size");
    std::vector<uint8_t> result(size);
    require(llama_state_seq_get_data(context, result.data(), size, seq) == size, "could not save sequence");
    return result;
}

struct run_result {
    trace observed;
    std::map<llama_seq_id, values> logits;
    std::vector<uint8_t> prefix;
    bool base_unchanged = false;
    uint32_t ubatch = 0;
};

run_result run(llama_model * model, bool serial, bool observe, int threads, bool batch_invariant) {
    run_result result;
    auto params = llama_context_default_params();
    params.n_ctx = 512;
    params.n_ctx_kv = 512;
    params.n_seq_max = 8;
    params.n_rs_slots = 3;
    params.n_batch = 64;
    params.n_ubatch = 64;
    params.n_outputs_max = 64;
    params.batch_invariant_matmul = batch_invariant;
    params.n_threads = threads;
    params.n_threads_batch = threads;
    params.n_rs_seq = 0;
    params.kv_unified = true;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.type_k = GGML_TYPE_F32;
    params.type_v = GGML_TYPE_F32;
    params.no_perf = true;
    if (observe) {
        params.cb_eval = trace::callback;
        params.cb_eval_user_data = &result.observed;
    }
    std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model, params), llama_free);
    require(context != nullptr, "could not create context");
    require(llama_get_batch_invariant_matmul(context.get()) == batch_invariant,
            "context changed the requested matrix policy");
    result.ubatch = llama_n_ubatch(context.get());
    const auto * vocab = llama_model_get_vocab(model);
    const auto base = tokenize(vocab, "The shared work begins.", true);
    require(base.size() <= 64, "prefix exceeds fixture capacity");
    batch_owner prefix(int(base.size()));
    prefix.value.n_tokens = int32_t(base.size());
    for (size_t i = 0; i < base.size(); ++i) {
        prefix.value.token[i] = base[i];
        prefix.value.pos[i] = llama_pos(i);
        prefix.value.n_seq_id[i] = 1;
        prefix.value.seq_id[i][0] = 0;
        prefix.value.logits[i] = true;
    }
    require(llama_decode(context.get(), prefix.value) == 0, "prefix decode failed");
    llama_synchronize(context.get());
    llama_memory_seq_cp(llama_get_memory(context.get()), 0, reader_id, -1, -1);
    llama_memory_seq_cp(llama_get_memory(context.get()), 0, peer_id, -1, -1);
    result.prefix = snapshot(context.get(), 0);
    const std::array<llama_seq_id, 2> ids{peer_id, reader_id};
    const std::array<llama_token, 2> tokens{
        tokenize(vocab, " red", false).back(), tokenize(vocab, " alpha", false).back()};
    const int step = serial ? 1 : 2;
    for (int begin = 0; begin < 2; begin += step) {
        batch_owner current(step);
        current.value.n_tokens = step;
        std::vector<llama_seq_id> rows;
        for (int i = 0; i < step; ++i) {
            current.value.token[i] = tokens[size_t(begin + i)];
            current.value.pos[i] = llama_pos(base.size());
            current.value.n_seq_id[i] = 1;
            current.value.seq_id[i][0] = ids[size_t(begin + i)];
            current.value.logits[i] = true;
            rows.push_back(ids[size_t(begin + i)]);
        }
        if (observe) { result.observed.begin(rows); }
        const auto status = llama_decode(context.get(), current.value);
        llama_synchronize(context.get());
        if (observe) { result.observed.end(); }
        require(status == 0, "fork decode failed");
        for (int i = 0; i < step; ++i) {
            const float * data = llama_get_logits_ith(context.get(), i);
            require(data != nullptr, "missing final logits");
            result.logits.emplace(ids[size_t(begin + i)], values(data, data + llama_vocab_n_tokens(vocab)));
        }
    }
    result.base_unchanged = result.prefix == snapshot(context.get(), 0);
    return result;
}

run_result run_prefill(llama_model * model, const std::vector<llama_token> & tokens,
                       uint32_t ubatch, bool observe, int threads, bool batch_invariant) {
    require(!tokens.empty() && tokens.size() <= 128, "prefill must contain 1 to 128 tokens");
    run_result result;
    auto params = llama_context_default_params();
    params.n_ctx = 512;
    params.n_ctx_kv = 512;
    params.n_seq_max = 1;
    params.n_rs_slots = 1;
    params.n_batch = 128;
    params.n_ubatch = ubatch;
    params.n_outputs_max = 128;
    params.batch_invariant_matmul = batch_invariant;
    params.n_threads = threads;
    params.n_threads_batch = threads;
    params.n_rs_seq = 0;
    params.kv_unified = true;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.type_k = GGML_TYPE_F32;
    params.type_v = GGML_TYPE_F32;
    params.no_perf = true;
    if (observe) {
        params.cb_eval = trace::callback;
        params.cb_eval_user_data = &result.observed;
    }
    std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model, params), llama_free);
    require(context != nullptr, "could not create prefill context");
    require(llama_get_batch_invariant_matmul(context.get()) == batch_invariant,
            "prefill context changed the requested matrix policy");
    result.ubatch = llama_n_ubatch(context.get());
    require(result.ubatch == ubatch, "prefill microbatch size was changed by the context");
    batch_owner input(int(tokens.size()));
    input.value.n_tokens = int32_t(tokens.size());
    std::vector<llama_seq_id> rows;
    for (size_t i = 0; i < tokens.size(); ++i) {
        input.value.token[i] = tokens[i];
        input.value.pos[i] = llama_pos(i);
        input.value.n_seq_id[i] = 1;
        input.value.seq_id[i][0] = 0;
        input.value.logits[i] = true;
        rows.push_back(llama_seq_id(i));
    }
    // Submit exactly the same logical native batch in both runs. The embedding
    // callback marks each native microbatch's rows by their token positions.
    if (observe) { result.observed.begin(rows, true); }
    const auto status = llama_decode(context.get(), input.value);
    llama_synchronize(context.get());
    if (observe) { result.observed.end(); }
    require(status == 0, "prefill decode failed");
    const auto vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto * data = llama_get_logits_ith(context.get(), int32_t(i));
        require(data != nullptr, "missing prefill logits");
        result.logits.emplace(llama_seq_id(i), values(data, data + vocabulary));
    }
    result.prefix = snapshot(context.get(), 0);
    return result;
}

struct difference {
    double maximum = 0;
    double rms = 0;
    size_t at = 0;
    size_t unequal = 0;
};

difference compare(const values & left, const values & right) {
    require(left.size() == right.size() && !left.empty(), "trace comparison shape mismatch");
    difference result;
    for (size_t i = 0; i < left.size(); ++i) {
        require(std::isfinite(left[i]) && std::isfinite(right[i]), "non-finite comparison value");
        const auto error = std::abs(double(left[i]) - double(right[i]));
        if (error > result.maximum) { result.maximum = error; result.at = i; }
        result.rms += error * error;
        result.unequal += left[i] != right[i];
    }
    result.rms = std::sqrt(result.rms / double(left.size()));
    return result;
}

template <class T> void print_shape(const std::array<T, 4> & shape) {
    std::cout << '[' << shape[0] << ',' << shape[1] << ',' << shape[2] << ',' << shape[3] << ']';
}

void print_values(const std::string & name, llama_seq_id row, const values & left, const values & right,
                  const char * row_kind) {
    const auto difference = compare(left, right);
    std::cout << "{\"kind\":\"logits\",\"comparison\":\"" << name << "\",\"" << row_kind << "\":" << row
              << ",\"max_abs\":" << difference.maximum << ",\"rms\":" << difference.rms
              << ",\"unequal\":" << difference.unequal
              << ",\"left_greedy\":" << std::distance(left.begin(), std::max_element(left.begin(), left.end()))
              << ",\"right_greedy\":" << std::distance(right.begin(), std::max_element(right.begin(), right.end()))
              << "}\n";
}
} // namespace

int main(int argc, char ** argv) {
    try {
        const char * usage = "usage: test-rerot-native-trace --model MODEL [--gpu-layers N] [--prefill-tokens N] [--batch-invariant-matmul 0|1]";
        require(argc >= 3 && argc % 2 == 1, usage);
        std::string model_path;
        int gpu_layers = 0;
        int prefill_count = 0;
        bool layers_seen = false;
        bool batch_invariant = false;
        bool policy_seen = false;
        for (int i = 1; i < argc; i += 2) {
            const std::string option = argv[i];
            const std::string value = argv[i + 1];
            if (option == "--model") {
                require(model_path.empty() && !value.empty(), usage);
                model_path = value;
            } else if (option == "--gpu-layers") {
                require(!layers_seen, usage);
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(), gpu_layers);
                require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() && gpu_layers >= 0,
                        "GPU layer count must be a nonnegative integer");
                layers_seen = true;
            } else if (option == "--prefill-tokens") {
                require(prefill_count == 0, usage);
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(), prefill_count);
                require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() &&
                        prefill_count >= 1 && prefill_count <= 128, "prefill token count must be between 1 and 128");
            } else if (option == "--batch-invariant-matmul") {
                require(!policy_seen && (value == "0" || value == "1"), usage);
                batch_invariant = value == "1";
                policy_seen = true;
            } else { throw std::invalid_argument(usage); }
        }
        require(!model_path.empty(), usage);
        ggml_backend_load_all();
        llama_backend_init();
        struct backend_owner { ~backend_owner() { llama_backend_free(); } } backend;
        auto params = llama_model_default_params();
        params.n_gpu_layers = gpu_layers;
        params.split_mode = LLAMA_SPLIT_MODE_NONE;
        placement_log placement;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(model_path.c_str(), params), llama_model_free);
        require(model != nullptr, "could not load model");
        require(gpu_layers == 0 || placement.gpu_layers > 0,
                "GPU layers requested but no actual model layer offload was reported");
        const bool prefill = prefill_count > 0;
        std::vector<llama_token> prefill_tokens;
        std::vector<llama_seq_id> comparison_rows;
        if (prefill) {
            std::string text;
            while (prefill_tokens.size() < size_t(prefill_count)) {
                text += "The shared work begins. Compare the available evidence, record each assumption, "
                        "and calculate the result carefully. The first reader checks the observations "
                        "while the second reader verifies every conclusion. ";
                prefill_tokens = tokenize(llama_model_get_vocab(model.get()), text, true);
            }
            prefill_tokens.resize(size_t(prefill_count));
            for (int i = 0; i < prefill_count; ++i) { comparison_rows.push_back(llama_seq_id(i)); }
        } else {
            comparison_rows = {peer_id, reader_id};
        }
        const auto execute = [&](bool serial, bool observe) {
            return prefill ? run_prefill(model.get(), prefill_tokens, serial ? 1u : 128u, observe, 2, batch_invariant) :
                             run(model.get(), serial, observe, 2, batch_invariant);
        };
        const auto batch = execute(false, true);
        const auto serial = execute(true, true);
        const auto batch_control = execute(false, false);
        const auto serial_control = execute(true, false);
        const char * row_kind = prefill ? "token_pos" : "seq";
        require(!placement.failed, "could not retain placement evidence");
        std::cout << std::setprecision(17);
        std::cout << "{\"kind\":\"placement\",\"gpu_layers_requested\":" << gpu_layers
                  << ",\"batch_invariant_matmul\":" << (batch_invariant ? "true" : "false")
                  << ",\"gpu_layers_offloaded_observed\":" << placement.gpu_layers << ",\"buffers\":[";
        for (size_t i = 0; i < placement.buffers.size(); ++i) {
            if (i) { std::cout << ','; }
            std::cout << json_string(placement.buffers[i]);
        }
        std::cout << "]}\n";
        if (prefill) {
            std::cout << "{\"kind\":\"prefill\",\"batch_ubatch\":" << batch.ubatch
                      << ",\"serial_ubatch\":" << serial.ubatch
                      << ",\"logical_native_decode_calls_per_run\":1,\"tokens\":[";
            for (size_t i = 0; i < prefill_tokens.size(); ++i) {
                if (i) { std::cout << ','; }
                std::cout << prefill_tokens[i];
            }
            std::cout << "]}\n";
            std::cout << "{\"kind\":\"prefill_state\",\"batch_serial_equal\":"
                      << (batch.prefix == serial.prefix ? "true" : "false")
                      << ",\"batch_trace_equal\":" << (batch.prefix == batch_control.prefix ? "true" : "false")
                      << ",\"serial_trace_equal\":" << (serial.prefix == serial_control.prefix ? "true" : "false")
                      << "}\n";
        } else {
            std::cout << "{\"kind\":\"state\",\"batch_base_unchanged\":" << (batch.base_unchanged ? "true" : "false")
                      << ",\"serial_base_unchanged\":" << (serial.base_unchanged ? "true" : "false")
                      << ",\"batch_serial_prefix_equal\":" << (batch.prefix == serial.prefix ? "true" : "false") << "}\n";
            require(batch.base_unchanged && serial.base_unchanged && batch.prefix == serial.prefix,
                    "base state changed across fork operations or trace shapes");
        }
        require(!batch.observed.order().empty(), "no trace tensors were captured");
        for (const auto & name : batch.observed.order()) {
            const auto other = serial.observed.samples().find(name);
            if (other == serial.observed.samples().end()) {
                std::cout << "{\"kind\":\"missing_serial_node\",\"name\":\"" << name << "\"}\n";
                continue;
            }
            for (const llama_seq_id row : comparison_rows) {
                const auto & left_rows = batch.observed.samples().at(name);
                const auto left_sample = left_rows.find(row);
                const auto right_sample = other->second.find(row);
                if (left_sample == left_rows.end() || right_sample == other->second.end()) {
                    std::cout << "{\"kind\":\"missing_trace_row\",\"name\":" << json_string(name)
                              << ",\"" << row_kind << "\":" << row << ",\"batch_present\":"
                              << (left_sample != left_rows.end() ? "true" : "false")
                              << ",\"serial_present\":" << (right_sample != other->second.end() ? "true" : "false")
                              << "}\n";
                    continue;
                }
                const auto & left = left_sample->second;
                const auto & right = right_sample->second;
                const auto difference = compare(left.data, right.data);
                std::cout << "{\"kind\":\"tensor\",\"name\":\"" << name << "\",\"" << row_kind << "\":" << row
                          << ",\"op\":\"" << left.operation << "\",\"source_type\":\"" << left.source_type
                          << "\",\"source1_type\":\"" << left.source1_type
                          << "\",\"batch_precision\":" << left.precision
                          << ",\"serial_precision\":" << right.precision
                          << ",\"batch_device\":" << json_string(left.device)
                          << ",\"serial_device\":" << json_string(right.device)
                          << ",\"batch_buffer_type\":" << json_string(left.buffer_type)
                          << ",\"serial_buffer_type\":" << json_string(right.buffer_type)
                          << ",\"batch_shape\":";
                print_shape(left.shape);
                std::cout << ",\"serial_shape\":"; print_shape(right.shape);
                std::cout << ",\"batch_strides\":"; print_shape(left.strides);
                std::cout << ",\"serial_strides\":"; print_shape(right.strides);
                std::cout << ",\"max_abs\":" << difference.maximum << ",\"rms\":" << difference.rms
                          << ",\"unequal\":" << difference.unequal << ",\"max_at\":" << difference.at
                          << ",\"batch_value\":" << left.data[difference.at]
                          << ",\"serial_value\":" << right.data[difference.at] << "}\n";
            }
        }
        for (const llama_seq_id row : comparison_rows) {
            print_values("traced_batch_vs_serial", row, batch.logits.at(row), serial.logits.at(row), row_kind);
            print_values("untraced_batch_vs_serial", row, batch_control.logits.at(row), serial_control.logits.at(row), row_kind);
            print_values("batch_trace_effect", row, batch.logits.at(row), batch_control.logits.at(row), row_kind);
            print_values("serial_trace_effect", row, serial.logits.at(row), serial_control.logits.at(row), row_kind);
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "native trace failed: " << error.what() << '\n';
        return 1;
    }
}
