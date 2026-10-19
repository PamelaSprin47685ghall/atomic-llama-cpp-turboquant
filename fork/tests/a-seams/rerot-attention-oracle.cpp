#include <llama.h>
#include <ggml-backend.h>
#include <gguf.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr int width = 32;
constexpr int q_heads = 4;
constexpr int kv_heads = 2;
constexpr int head_width = width / q_heads;
constexpr int vocabulary = 32;
constexpr double epsilon = double(1e-5f);
constexpr double frequency_base = 100.0;
using vector = std::vector<double>;
using identity = std::pair<llama_seq_id, llama_pos>;

void require(bool condition, const std::string & message) {
    if (!condition) { throw std::runtime_error(message); }
}

struct weights {
    int columns;
    int rows;
    std::vector<float> data;
};

struct close_file {
    void operator()(FILE * file) const noexcept { if (file) { std::fclose(file); } }
};

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

// The public model API does not expose its tensors. Retain the native loader
// and KV allocation records as placement evidence, forwarding every original
// message. Merely enumerating a GPU or requesting layers is not evidence.
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
                    self.lines.push_back(std::move(line));
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
    std::vector<std::string> lines;
};

class fixture {
public:
    fixture(std::string arch, int rotary, bool neox, float amplitude, int gpu_layers)
        : arch_(std::move(arch)), rotary_(rotary), neox_(neox), amplitude_(amplitude),
          gpu_layers_requested_(gpu_layers),
          metadata_(gguf_init_empty(), gguf_free) {
        require(metadata_ != nullptr, "could not allocate GGUF metadata");
        gguf_set_val_str(metadata_.get(), "general.architecture", arch_.c_str());
        gguf_set_val_str(metadata_.get(), "general.name", "RERoT independent numerical fixture");
        gguf_set_val_str(metadata_.get(), "tokenizer.ggml.model", "no_vocab");
        const auto u32 = [&](const char * key, std::uint32_t value) {
            gguf_set_val_u32(metadata_.get(), (arch_ + "." + key).c_str(), value);
        };
        const auto f32 = [&](const char * key, float value) {
            gguf_set_val_f32(metadata_.get(), (arch_ + "." + key).c_str(), value);
        };
        u32("vocab_size", vocabulary);
        u32("context_length", 128);
        u32("embedding_length", width);
        u32("block_count", 1);
        u32("feed_forward_length", 64);
        u32("attention.head_count", q_heads);
        u32("attention.head_count_kv", kv_heads);
        u32("rope.dimension_count", std::uint32_t(rotary_));
        f32("attention.layer_norm_rms_epsilon", float(epsilon));
        f32("rope.freq_base", float(frequency_base));
        add("token_embd.weight", width, vocabulary, 1);
        add("output_norm.weight", width, 1, 0, true);
        add("output.weight", width, vocabulary, 2);
        add("blk.0.attn_norm.weight", width, 1, 0, true);
        add("blk.0.attn_q.weight", width, width, 3);
        add("blk.0.attn_k.weight", width, head_width * kv_heads, 4);
        add("blk.0.attn_v.weight", width, head_width * kv_heads, 5);
        add("blk.0.attn_output.weight", width, width, 6);
        add("blk.0.ffn_norm.weight", width, 1, 0, true);
        add("blk.0.ffn_gate.weight", width, 64, 0);
        add("blk.0.ffn_up.weight", width, 64, 0);
        add("blk.0.ffn_down.weight", 64, width, 0);

        auto params = llama_model_default_params();
        params.n_gpu_layers = gpu_layers_requested_;
        params.split_mode = LLAMA_SPLIT_MODE_NONE;
        params.load_mode = LLAMA_LOAD_MODE_NONE;
        // The user-initializer route creates optional tensors even when they
        // are absent from metadata. A complete GGUF declares the exact model
        // being tested: only these weights, no implicit bias or scale.
        for (const auto & item : tensors_) {
            gguf_set_tensor_data(metadata_.get(), item.first.c_str(), item.second.data.data());
        }
        std::unique_ptr<FILE, close_file> file(std::tmpfile());
        require(file != nullptr, "could not create a temporary fixture GGUF");
        require(gguf_write_to_file_ptr(metadata_.get(), file.get(), false), "could not write fixture GGUF");
        require(std::fseek(file.get(), 0, SEEK_SET) == 0, "could not rewind fixture GGUF");
        placement_log placement;
        model_.reset(llama_model_load_from_file_ptr(file.get(), params));
        require(model_ != nullptr, "could not create the actual A fixture model: " + arch_);
        require(llama_model_n_head(model_.get()) == q_heads &&
                llama_model_n_head_kv(model_.get()) == kv_heads, "GQA geometry changed");
        auto context = llama_context_default_params();
        context.n_ctx = 128;
        context.n_ctx_kv = 128;
        context.n_seq_max = 8;
        context.n_batch = 16;
        context.n_ubatch = 16;
        context.n_outputs_max = 16;
        context.n_threads = 1;
        context.n_threads_batch = 1;
        context.kv_unified = true;
        context.type_k = GGML_TYPE_F32;
        context.type_v = GGML_TYPE_F32;
        context.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        context.offload_kqv = gpu_layers_requested_ > 0;
        context.op_offload = gpu_layers_requested_ > 0;
        context.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
        context.rope_freq_base = float(frequency_base);
        context.yarn_ext_factor = 0.0f;
        context.yarn_attn_factor = amplitude_;
        context.no_perf = true;
        context_params_ = context;
        context_.reset(llama_init_from_model(model_.get(), context));
        require(context_ != nullptr && llama_attention_views_supported(context_.get()),
                "actual A does not support this fixture's reader views");
        require(llama_attention_views_unsupported_reason(context_.get()) == nullptr,
                "supported reader profile has an unsupported reason");
        require(!placement.failed, "could not retain actual placement evidence");
        gpu_layers_observed_ = placement.gpu_layers;
        placement_evidence_ = std::move(placement.lines);
        require(gpu_layers_requested_ == 0 || gpu_layers_observed_ > 0,
                "GPU layers were requested but the native loader reported no layer offload");
    }

    ~fixture() { if (context_) { llama_synchronize(context_.get()); } }

    std::size_t verify_unsupported_profiles() {
        const llama_attention_entry entry{0, 0, 0};
        const llama_attention_view view{0, 0, 0, 1, &entry};
        require(llama_attention_set_views(context_.get(), &view, 1),
                "supported profile rejected the capability test's valid view");
        llama_attention_clear_views(context_.get());

        std::size_t rejected = 0;
        const auto check = [&](llama_context * context, const char * detail) {
            const char * reason = llama_attention_views_unsupported_reason(context);
            require(reason != nullptr && std::strstr(reason, detail) != nullptr,
                    std::string("wrong unsupported-profile reason for ") + detail + ": " +
                    (reason ? reason : "<supported>"));
            require(!llama_attention_views_supported(context),
                    "boolean capability query disagrees with the public reason");
            require(!llama_attention_set_views(context, &view, 1),
                    "unsupported profile accepted a reader view");
            ++rejected;
        };
        const auto restored = [&]() {
            require(llama_attention_views_supported(context_.get()) &&
                    llama_attention_views_unsupported_reason(context_.get()) == nullptr,
                    "restoring a supported mode did not restore reader capability");
        };
        check(nullptr, "context is null");
        llama_set_embeddings(context_.get(), true);
        check(context_.get(), "embedding output");
        llama_set_embeddings(context_.get(), false);
        restored();
        llama_set_causal_attn(context_.get(), false);
        check(context_.get(), "causal attention");
        llama_set_causal_attn(context_.get(), true);
        restored();

        const auto profile = [&](llama_context_params params, const char * detail) {
            std::unique_ptr<llama_context, decltype(&llama_free)> context(
                llama_init_from_model(model_.get(), params), llama_free);
            require(context != nullptr, std::string("could not create actual profile: ") + detail);
            check(context.get(), detail);
        };
        auto params = context_params_;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        profile(params, "Flash Attention");
        params = context_params_;
        params.type_k = GGML_TYPE_F16;
        profile(params, "K cache");
        params = context_params_;
        params.type_v = GGML_TYPE_F16;
        profile(params, "V cache");
        params = context_params_;
        params.kv_unified = false;
        params.n_ctx_kv = 0;
        profile(params, "unified KV cache");
        return rejected;
    }

    struct row {
        llama_seq_id seq;
        llama_pos pos;
        llama_token token;
        std::vector<llama_attention_entry> visible;
    };

    void forward(std::vector<row> rows, bool reader_views) {
        std::vector<llama_attention_view> views;
        for (const auto & row : rows) {
            require(!row.visible.empty(), "empty oracle fixture view");
            require(tokens_.emplace(identity{row.seq, row.pos}, row.token).second,
                    "fixture attempted to overwrite a physical logical key");
            views.push_back({row.seq, row.pos, row.visible.back().virtual_pos,
                             row.visible.size(), row.visible.data()});
        }
        if (reader_views) {
            require(llama_attention_set_views(context_.get(), views.data(), views.size()),
                    "actual A rejected a valid numerical fixture view");
        } else { llama_attention_clear_views(context_.get()); }
        const auto batch = llama_batch_init(int32_t(rows.size()), 0, 1);
        struct cleanup { llama_batch value; ~cleanup() { llama_batch_free(value); } } owner{batch};
        owner.value.n_tokens = int32_t(rows.size());
        for (std::size_t i = 0; i < rows.size(); ++i) {
            owner.value.token[i] = rows[i].token;
            owner.value.pos[i] = rows[i].pos;
            owner.value.n_seq_id[i] = 1;
            owner.value.seq_id[i][0] = rows[i].seq;
            owner.value.logits[i] = true;
        }
        const auto status = llama_decode(context_.get(), owner.value);
        llama_synchronize(context_.get());
        require(status == 0, "actual A fixture decode failed: " + std::to_string(status));
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto expected = oracle(rows[i].token, rows[i].visible);
            const auto actual = llama_get_logits_ith(context_.get(), int32_t(i));
            require(actual != nullptr, "fixture query has no actual output");
            double maximum = 0;
            for (int token = 0; token < vocabulary; ++token) {
                require(std::isfinite(actual[token]) && std::isfinite(expected[std::size_t(token)]),
                        "non-finite numerical fixture output");
                const auto difference = std::abs(double(actual[token]) - expected[std::size_t(token)]);
                maximum = std::max(maximum, difference);
                const auto limit = 3e-5 + 1e-5 * std::abs(expected[std::size_t(token)]);
                require(difference <= limit,
                        arch_ + (reader_views ? " reader" : " ordinary") +
                        " differs from independent F64 logits for seq " + std::to_string(rows[i].seq) +
                        " pos " + std::to_string(rows[i].pos) + " input " + std::to_string(rows[i].token) +
                        " amplitude " + std::to_string(amplitude_) + " at token " + std::to_string(token) +
                        ": actual=" + std::to_string(actual[token]) +
                        " expected=" + std::to_string(expected[std::size_t(token)]));
            }
            if (reader_views) { reader_max_ = std::max(reader_max_, maximum); ++reader_queries_; }
            else { ordinary_max_ = std::max(ordinary_max_, maximum); ++ordinary_queries_; }
        }
        llama_attention_clear_views(context_.get());
    }

    void copy_base(llama_seq_id destination) {
        llama_memory_seq_cp(llama_get_memory(context_.get()), 0, destination, -1, -1);
        tokens_.emplace(identity{destination, 0}, tokens_.at({0, 0}));
        tokens_.emplace(identity{destination, 1}, tokens_.at({0, 1}));
    }

    void run(bool reverse, bool serial) {
        reversed_ = reverse;
        serial_cohort_ = serial;
        // Calibrate the independent complete one-layer calculation against
        // ordinary A inference before using it to judge the new seam.
        forward({{0, 0, 1, {{0, 0, 0}}}, {0, 1, 2, {{0, 0, 0}, {0, 1, 1}}}}, false);
        copy_base(5);
        copy_base(6);
        forward({{5, 2, 3, {{0, 0, 0}, {0, 1, 1}, {5, 2, 2}}},
                 {6, 2, 4, {{0, 0, 0}, {0, 1, 1}, {6, 2, 2}}}}, false);
        // Nonuniform virtual offsets force several rotary spans. Key counts
        // and distributions differ; averaging span outputs cannot pass.
        std::vector<row> queries = {
            {5, 3, 5, {{0, 0, 0}, {6, 2, 1}, {0, 1, 3}, {5, 2, 4}, {5, 3, 7}}},
            {6, 3, 6, {{6, 2, 0}, {0, 1, 2}, {5, 2, 4}, {0, 0, 7}, {6, 3, 9}}},
        };
        if (reverse) { std::reverse(queries.begin(), queries.end()); }
        if (serial) {
            // The same complete owned cohort was frozen above. Submitting
            // its queries separately must preserve the F64 reader result,
            // even after an earlier call allocates a hidden peer current key.
            for (const auto & query : queries) { forward({query}, true); }
        } else { forward(queries, true); }
        forward({{5, 4, 7, {{0, 0, 0}, {6, 2, 1}, {6, 3, 2},
                           {0, 1, 4}, {5, 2, 5}, {5, 3, 7}, {5, 4, 10}}}}, true);
        require(ordinary_queries_ == 4 && reader_queries_ == 3, "empty numerical fixture");
    }

    void print(std::ostream & out) const {
        out << "{\"architecture\":\"" << arch_ << "\",\"q_heads\":" << q_heads
                  << ",\"kv_heads\":" << kv_heads << ",\"head_width\":" << head_width
                  << ",\"rotary_width\":" << rotary_ << ",\"rotary_amplitude\":" << amplitude_
                  << ",\"serial_cohort\":" << (serial_cohort_ ? "true" : "false")
                  << ",\"reversed_query_order\":" << (reversed_ ? "true" : "false")
                  << ",\"gpu_layers_requested\":" << gpu_layers_requested_
                  << ",\"gpu_layers_offloaded_observed\":" << gpu_layers_observed_
                  << ",\"ordinary_queries\":" << ordinary_queries_
                  << ",\"reader_queries\":" << reader_queries_
                  << ",\"ordinary_f64_max_abs\":" << ordinary_max_
                  << ",\"reader_f64_max_abs\":" << reader_max_
                  << ",\"placement_evidence\":[";
        for (std::size_t i = 0; i < placement_evidence_.size(); ++i) {
            if (i != 0) { out << ','; }
            out << json_string(placement_evidence_[i]);
        }
        out << "]}";
    }

private:
    void add(const std::string & name, int columns, int rows, int seed, bool norm = false) {
        weights value{columns, rows, std::vector<float>(std::size_t(columns * rows))};
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < columns; ++col) {
                value.data[std::size_t(row * columns + col)] = norm ? 1.0f : seed == 0 ? 0.0f :
                    float(0.17 * std::sin(double((row + 1) * (col + seed + 1))) +
                          0.09 * std::cos(double((row + seed + 2) * (col + 1))));
            }
        }
        // Use real public GGML tensor descriptors only to set the file type.
        // The independent oracle below does not call a GGML numerical op.
        ggml_init_params params{std::size_t(4096), nullptr, true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> owner(ggml_init(params), ggml_free);
        require(owner != nullptr, "could not allocate a tensor descriptor");
        auto tensor = rows == 1 ? ggml_new_tensor_1d(owner.get(), GGML_TYPE_F32, columns) :
                                 ggml_new_tensor_2d(owner.get(), GGML_TYPE_F32, columns, rows);
        ggml_set_name(tensor, name.c_str());
        gguf_add_tensor(metadata_.get(), tensor);
        tensors_.emplace(name, std::move(value));
    }

    vector embedding(llama_token token) const {
        const auto & values = tensors_.at("token_embd.weight").data;
        vector out(width);
        for (int col = 0; col < width; ++col) { out[std::size_t(col)] = values[std::size_t(token * width + col)]; }
        return out;
    }

    static vector normalize(vector input) {
        double sum = 0;
        for (const double value : input) { sum += value * value; }
        const auto factor = 1.0 / std::sqrt(sum / double(input.size()) + epsilon);
        for (double & value : input) { value *= factor; }
        return input;
    }

    vector project(const std::string & name, const vector & input) const {
        const auto & matrix = tensors_.at(name);
        require(input.size() == std::size_t(matrix.columns), "oracle matrix geometry mismatch");
        vector output(std::size_t(matrix.rows), 0.0);
        for (int row = 0; row < matrix.rows; ++row) {
            for (int column = 0; column < matrix.columns; ++column) {
                output[std::size_t(row)] += double(matrix.data[std::size_t(row * matrix.columns + column)]) *
                                          input[std::size_t(column)];
            }
        }
        return output;
    }

    vector rotate(vector input, int heads, llama_pos position) const {
        for (int head = 0; head < heads; ++head) {
            for (int pair = 0; pair < rotary_ / 2; ++pair) {
                const auto angle = double(position) * std::pow(frequency_base, -2.0 * pair / rotary_);
                const auto c = std::cos(angle) * double(amplitude_);
                const auto s = std::sin(angle) * double(amplitude_);
                const int first = head * head_width + (neox_ ? pair : 2 * pair);
                const int second = first + (neox_ ? rotary_ / 2 : 1);
                const auto x = input[std::size_t(first)], y = input[std::size_t(second)];
                input[std::size_t(first)] = x * c - y * s;
                input[std::size_t(second)] = x * s + y * c;
            }
        }
        return input;
    }

    vector oracle(llama_token token, const std::vector<llama_attention_entry> & visible) const {
        const auto residual = embedding(token);
        const auto normalized = normalize(residual);
        const auto q = rotate(project("blk.0.attn_q.weight", normalized), q_heads,
                              visible.back().virtual_pos);
        std::vector<vector> keys, values;
        for (const auto & entry : visible) {
            const auto state = normalize(embedding(tokens_.at({entry.key_seq, entry.key_pos})));
            keys.push_back(rotate(project("blk.0.attn_k.weight", state), kv_heads, entry.virtual_pos));
            values.push_back(project("blk.0.attn_v.weight", state));
        }
        vector attention(width, 0.0);
        for (int head = 0; head < q_heads; ++head) {
            const int kv = head / (q_heads / kv_heads);
            vector scores(visible.size(), 0.0);
            for (std::size_t key = 0; key < visible.size(); ++key) {
                for (int dim = 0; dim < head_width; ++dim) {
                    scores[key] += q[std::size_t(head * head_width + dim)] *
                                   keys[key][std::size_t(kv * head_width + dim)];
                }
                scores[key] /= std::sqrt(double(head_width));
            }
            const auto maximum = *std::max_element(scores.begin(), scores.end());
            double denominator = 0;
            for (double & score : scores) { score = std::exp(score - maximum); denominator += score; }
            for (std::size_t key = 0; key < scores.size(); ++key) {
                for (int dim = 0; dim < head_width; ++dim) {
                    attention[std::size_t(head * head_width + dim)] +=
                        (scores[key] / denominator) * values[key][std::size_t(kv * head_width + dim)];
                }
            }
        }
        auto hidden = project("blk.0.attn_output.weight", attention);
        for (int dim = 0; dim < width; ++dim) { hidden[std::size_t(dim)] += residual[std::size_t(dim)]; }
        // All FFN weights are exactly zero. Its output is zero, so the final
        // residual and output RMS norm retain the full attention effect.
        return project("output.weight", normalize(std::move(hidden)));
    }

    std::string arch_;
    int rotary_;
    bool neox_;
    float amplitude_;
    int gpu_layers_requested_ = 0;
    int gpu_layers_observed_ = 0;
    llama_context_params context_params_{};
    std::vector<std::string> placement_evidence_;
    bool serial_cohort_ = false;
    bool reversed_ = false;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata_;
    std::map<std::string, weights> tensors_;
    std::map<identity, llama_token> tokens_;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model_{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> context_{nullptr, llama_free};
    int ordinary_queries_ = 0;
    int reader_queries_ = 0;
    double ordinary_max_ = 0;
    double reader_max_ = 0;
};
} // namespace

int main(int argc, char ** argv) {
    try {
        int gpu_layers = 0;
        if (argc != 1) {
            require(argc == 3 && std::string(argv[1]) == "--gpu-layers",
                    "usage: test-rerot-attention-oracle [--gpu-layers N]");
            const std::string argument = argv[2];
            const auto parsed = std::from_chars(argument.data(), argument.data() + argument.size(), gpu_layers);
            require(parsed.ec == std::errc{} && parsed.ptr == argument.data() + argument.size() && gpu_layers >= 0,
                    "GPU layer count must be a nonnegative integer");
        }
        ggml_backend_load_all();
        llama_backend_init();
        struct owner { ~owner() { llama_backend_free(); } } backend;
        std::ostringstream receipt;
        receipt << std::setprecision(17) << "{\"status\":\"PASS\",\"scope\":\"actual A graph vs independent F64 logits\",\"cases\":[";
        bool first = true;
        std::size_t unsupported_profiles = 0;
        for (const auto & arch : {std::string("llama"), std::string("qwen2")}) {
            for (const float amplitude : {1.0f, 1.375f}) {
                for (const bool reverse : {false, true}) {
                    for (const bool serial : {false, true}) {
                        fixture test(arch, head_width, arch == "qwen2", amplitude, gpu_layers);
                        if (first && gpu_layers == 0) {
                            unsupported_profiles = test.verify_unsupported_profiles();
                        }
                        test.run(reverse, serial);
                        if (!first) { receipt << ','; } first = false;
                        test.print(receipt);
                    }
                }
            }
        }
        receipt << "],\"unsupported_profile_checks\":" << unsupported_profiles << "}\n";
        std::cout << receipt.str();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "RERoT F64 oracle failed: " << error.what() << '\n';
        return 1;
    }
}
