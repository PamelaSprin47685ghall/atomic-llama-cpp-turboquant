#include "tq/compose/sampler.h"
#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct options {
    std::string model;
    std::string prompt = "Hello";
    int32_t predict = 64;
    int32_t context = 2048;
    int32_t gpu_layers = 0;
    int32_t threads = 4;
    bool json = false;
    tq::compose::sampling_options sampling;
};

void usage() {
    std::cout <<
        "Usage: llama-compose -m model.gguf -p prompt [options]\n"
        "  -n, --n-predict N     Maximum new tokens (default 64)\n"
        "  -c, --ctx-size N      Context size (default 2048)\n"
        "  -ngl, --gpu-layers N  Layers on the GPU (default 0)\n"
        "  -t, --threads N       CPU threads (default 4)\n"
        "  --top-k N            First sampling filter (default 40)\n"
        "  --top-p P            Nucleus threshold (default 0.95)\n"
        "  --min-p P            Minimum probability ratio (default 0.05)\n"
        "  --temp T             Temperature (default 0, greedy)\n"
        "  --seed N             Random seed (default 42)\n"
        "  --no-compact         Use the full upstream sampling row\n"
        "  --json               Print generated text, tokens, and route counts\n"
        "The prompt is raw model input. Use the model's chat format when needed.\n";
}

template<class T>
T integer(const std::string & text, const std::string & name) {
    T value{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument(name + " requires an integer");
    }
    return value;
}

float number(const std::string & text, const std::string & name) {
    char * end = nullptr;
    errno = 0;
    const float value = std::strtof(text.c_str(), &end);
    if (errno || end == text.c_str() || *end || !std::isfinite(value)) {
        throw std::invalid_argument(name + " requires a finite number");
    }
    return value;
}

options parse(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--no-compact") {
            result.sampling.compact = false;
            continue;
        }
        if (flag == "--json") {
            result.json = true;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + flag);
        }
        const std::string value = argv[++i];
        if (flag == "-m" || flag == "--model") result.model = value;
        else if (flag == "-p" || flag == "--prompt") result.prompt = value;
        else if (flag == "-n" || flag == "--n-predict") result.predict = integer<int32_t>(value, flag);
        else if (flag == "-c" || flag == "--ctx-size") result.context = integer<int32_t>(value, flag);
        else if (flag == "-ngl" || flag == "--gpu-layers") result.gpu_layers = integer<int32_t>(value, flag);
        else if (flag == "-t" || flag == "--threads") result.threads = integer<int32_t>(value, flag);
        else if (flag == "--top-k") result.sampling.top_k = integer<int32_t>(value, flag);
        else if (flag == "--top-p") result.sampling.top_p = number(value, flag);
        else if (flag == "--min-p") result.sampling.min_p = number(value, flag);
        else if (flag == "--temp") result.sampling.temperature = number(value, flag);
        else if (flag == "--seed") result.sampling.seed = integer<uint32_t>(value, flag);
        else throw std::invalid_argument("unknown option: " + flag);
    }
    if (result.model.empty()) throw std::invalid_argument("a model path is required");
    if (result.predict < 0 || result.context <= 0 || result.threads <= 0 || result.gpu_layers < 0) {
        throw std::invalid_argument("token and GPU counts must be nonnegative; context and threads must be positive");
    }
    return result;
}

struct backend_runtime {
    backend_runtime() {
        llama_backend_init();
        ggml_backend_load_all();
    }
    ~backend_runtime() { llama_backend_free(); }
};

using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & prompt) {
    if (prompt.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("prompt is too large");
    }
    const int32_t size = static_cast<int32_t>(prompt.size());
    const int32_t needed = llama_tokenize(vocab, prompt.data(), size, nullptr, 0, true, true);
    if (needed >= 0 || needed == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("prompt has no tokens or tokenization failed");
    }
    std::vector<llama_token> tokens(static_cast<size_t>(-needed));
    const int32_t count = llama_tokenize(vocab, prompt.data(), size, tokens.data(), -needed, true, true);
    if (count <= 0 || count > -needed) throw std::runtime_error("tokenization failed");
    tokens.resize(static_cast<size_t>(count));
    return tokens;
}

std::string piece(const llama_vocab * vocab, llama_token token) {
    std::string result(128, '\0');
    int32_t size = llama_token_to_piece(vocab, token, result.data(), 128, 0, false);
    if (size < 0) {
        if (size == std::numeric_limits<int32_t>::min()) throw std::runtime_error("token piece is too large");
        result.resize(static_cast<size_t>(-size));
        size = llama_token_to_piece(vocab, token, result.data(), -size, 0, false);
    }
    if (size < 0) throw std::runtime_error("token conversion failed");
    result.resize(static_cast<size_t>(size));
    return result;
}

std::string quote(const std::string & text) {
    static const char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (size_t i = 0; i < text.size();) {
        const auto c = static_cast<unsigned char>(text[i]);
        if (c < 128) {
            if (c == '"' || c == '\\') {
                result += '\\';
                result += static_cast<char>(c);
            } else if (c < 32) {
                result += "\\u00";
                result += hex[c >> 4];
                result += hex[c & 15];
            } else {
                result += static_cast<char>(c);
            }
            ++i;
            continue;
        }
        const size_t width = c >= 0xc2 && c <= 0xdf ? 2 :
                             c >= 0xe0 && c <= 0xef ? 3 :
                             c >= 0xf0 && c <= 0xf4 ? 4 : 0;
        bool valid = width && width <= text.size() - i;
        for (size_t j = 1; valid && j < width; ++j) {
            const auto next = static_cast<unsigned char>(text[i + j]);
            valid = next >= 0x80 && next <= 0xbf;
            if (j == 1) {
                valid = valid && !(c == 0xe0 && next < 0xa0) &&
                        !(c == 0xed && next > 0x9f) &&
                        !(c == 0xf0 && next < 0x90) &&
                        !(c == 0xf4 && next > 0x8f);
            }
        }
        if (valid) {
            result.append(text, i, width);
            i += width;
        } else {
            // Token limits can end in the middle of a UTF-8 character.
            // Preserve exact token IDs; represent incomplete text as U+FFFD.
            result += "\\ufffd";
            ++i;
        }
    }
    return result + '"';
}

int generate(const options & config) {
    tq::compose::sampler sample(config.sampling);
    backend_runtime runtime;
    auto model_options = llama_model_default_params();
    model_options.n_gpu_layers = config.gpu_layers;
    model_ptr model(llama_model_load_from_file(config.model.c_str(), model_options), llama_model_free);
    if (!model) throw std::runtime_error("cannot load model: " + config.model);
    if (llama_model_has_encoder(model.get()) || !llama_model_has_decoder(model.get())) {
        throw std::invalid_argument("this command requires a decoder-only model");
    }
    const auto * vocab = llama_model_get_vocab(model.get());
    auto prompt = tokenize(vocab, config.prompt);
    if (prompt.size() + static_cast<size_t>(config.predict) > static_cast<size_t>(config.context)) {
        throw std::invalid_argument("prompt plus requested tokens exceeds the context size");
    }
    auto context_options = llama_context_default_params();
    context_options.n_ctx = static_cast<uint32_t>(config.context);
    context_options.n_batch = static_cast<uint32_t>(std::min(config.context, 512));
    context_options.n_ubatch = context_options.n_batch;
    context_options.n_threads = config.threads;
    context_options.n_threads_batch = config.threads;
    context_ptr context(llama_init_from_model(model.get(), context_options), llama_free);
    if (!context) throw std::runtime_error("cannot create model context");

    const auto start = std::chrono::steady_clock::now();
    std::vector<llama_token> tokens;
    std::string output;
    bool eog = false;
    if (config.predict > 0) {
        for (size_t offset = 0; offset < prompt.size();) {
            const int32_t count = static_cast<int32_t>(std::min<size_t>(context_options.n_batch, prompt.size() - offset));
            auto batch = llama_batch_get_one(prompt.data() + offset, count);
            const int status = llama_decode(context.get(), batch);
            if (status != 0) throw std::runtime_error("prompt decode failed with status " + std::to_string(status));
            offset += static_cast<size_t>(count);
        }
        for (int32_t i = 0; i < config.predict; ++i) {
            const auto * logits = llama_get_logits_ith(context.get(), -1);
            llama_token token = sample.sample(logits, llama_vocab_n_tokens(vocab));
            tokens.push_back(token);
            if (llama_vocab_is_eog(vocab, token)) {
                eog = true;
                break;
            }
            const auto text = piece(vocab, token);
            output += text;
            if (!config.json) std::cout << text << std::flush;
            if (i + 1 < config.predict) {
                auto batch = llama_batch_get_one(&token, 1);
                const int status = llama_decode(context.get(), batch);
                if (status != 0) throw std::runtime_error("decode failed with status " + std::to_string(status));
            }
        }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto stats = sample.stats();
    if (config.json) {
        std::cout << "{\"text\":" << quote(output) << ",\"tokens\":[";
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << tokens[i];
        }
        std::cout << "],\"stop\":" << quote(eog ? "eog" : "limit")
                  << ",\"compact_rows\":" << stats.compact_rows
                  << ",\"full_rows\":" << stats.full_rows
                  << ",\"prompt_tokens\":" << prompt.size()
                  << ",\"prefill_and_generation_seconds\":" << seconds << "}\n";
    } else {
        std::cout << '\n';
    }
    std::cerr << "sampled=" << tokens.size() << " compact=" << stats.compact_rows
              << " full=" << stats.full_rows << " prefill_and_generation_seconds=" << seconds << '\n';
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 1 || (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))) {
        usage();
        return argc == 1 ? 1 : 0;
    }
    try {
        return generate(parse(argc, argv));
    } catch (const std::exception & error) {
        std::cerr << "llama-compose: " << error.what() << '\n';
        return 1;
    }
}
