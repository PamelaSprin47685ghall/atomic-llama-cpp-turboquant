#include <llama.h>
#include <ggml.h>
#include <ggml-backend.h>

#include "tape-writer.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct options {
    std::string model;
    std::string prompt;
    std::string output;
    std::uint32_t context = 2048;
    std::uint32_t batch = 512;
    std::uint32_t prefill = 1;
    std::uint32_t threads = 4;
    std::uint32_t gpu_layers = 0;
    std::uint64_t max_output_bytes = 1024ULL * 1024 * 1024;
    bool add_special = true;
};

void usage() {
    std::cout <<
        "Usage: tq-logits-compose --model MODEL.gguf --prompt-file FILE --output FILE [options]\n"
        "Collect full, unsampled logits for one teacher-forced input sequence.\n"
        "  --ctx-size N          Maximum input tokens; default: 2048, maximum: 2000000.\n"
        "  --prefill N           Initial prefix tokens; default: 1. Save its final row,\n"
        "                        then one row after each remaining input token.\n"
        "  --batch-size N        Maximum prefill chunk; default: 512.\n"
        "  --threads N           CPU threads; default: 4, bounded by GGML_MAX_N_THREADS.\n"
        "  --gpu-layers N        Requested GPU layers; default: 0.\n"
        "  --max-output-bytes N  Tape size limit; default: 1073741824 (1 GiB).\n"
        "  --no-add-special      Disable tokenizer BOS/EOS insertion.\n"
        "Input is one raw UTF-8 file, at most 16 MiB. Special-token spellings stay text.\n"
        "Output is a new TP5LOG2 file; existing files and symlinks are refused.\n";
}

std::uint64_t number(std::string_view text, const std::string & name, std::uint64_t maximum, bool zero = false) {
    std::uint64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || (!zero && value == 0) || value > maximum) {
        throw std::invalid_argument("invalid decimal integer for " + name);
    }
    return value;
}

options parse(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--no-add-special") { result.add_special = false; continue; }
        if (arg != "--model" && arg != "--prompt-file" && arg != "--output" && arg != "--ctx-size" &&
            arg != "--prefill" && arg != "--batch-size" && arg != "--threads" && arg != "--gpu-layers" && arg != "--max-output-bytes") {
            throw std::invalid_argument("unknown option: " + arg);
        }
        if (++i == argc) { throw std::invalid_argument("missing value for " + arg); }
        const std::string value = argv[i];
        if (arg == "--model") { result.model = value; }
        else if (arg == "--prompt-file") { result.prompt = value; }
        else if (arg == "--output") { result.output = value; }
        else if (arg == "--max-output-bytes") { result.max_output_bytes = number(value, arg, INT64_MAX); }
        else {
            const auto limit = arg == "--threads" ? std::uint64_t(GGML_MAX_N_THREADS) :
                               arg == "--gpu-layers" ? std::uint64_t(INT32_MAX) : 2000000ULL;
            const auto count = static_cast<std::uint32_t>(number(value, arg, limit, arg == "--gpu-layers"));
            if (arg == "--ctx-size") { result.context = count; }
            else if (arg == "--prefill") { result.prefill = count; }
            else if (arg == "--batch-size") { result.batch = count; }
            else if (arg == "--threads") { result.threads = count; }
            else { result.gpu_layers = count; }
        }
    }
    if (result.model.empty() || result.prompt.empty() || result.output.empty()) {
        throw std::invalid_argument("--model, --prompt-file and --output are required");
    }
    if (result.prefill > result.context) { throw std::invalid_argument("--prefill exceeds --ctx-size"); }
    return result;
}

std::string read_prompt(const std::string & path) {
    constexpr std::size_t limit = 16 * 1024 * 1024;
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot open prompt file"); }
    std::string text;
    std::array<char, 16384> buffer;
    while (input.read(buffer.data(), buffer.size()) || input.gcount()) {
        if (text.size() + std::size_t(input.gcount()) > limit) { throw std::invalid_argument("prompt exceeds 16 MiB"); }
        text.append(buffer.data(), std::size_t(input.gcount()));
    }
    if (!input.eof()) { throw std::runtime_error("cannot read prompt file"); }
    if (text.empty()) { throw std::invalid_argument("prompt file is empty"); }
    for (std::size_t i = 0; i < text.size();) {
        const auto c = static_cast<unsigned char>(text[i]);
        if (c < 128) { ++i; continue; }
        const unsigned count = c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
        std::uint32_t point = c & (count == 2 ? 31 : count == 3 ? 15 : 7);
        bool valid = count && i + count <= text.size();
        for (unsigned j = 1; valid && j < count; ++j) {
            const auto tail = static_cast<unsigned char>(text[i + j]);
            valid = (tail & 0xc0) == 0x80;
            point = (point << 6) | (tail & 63);
        }
        if (!valid || (count == 3 && point < 0x800) || (count == 4 && point < 0x10000) ||
            point > 0x10ffff || (point >= 0xd800 && point <= 0xdfff)) {
            throw std::invalid_argument("prompt is not valid UTF-8");
        }
        i += count;
    }
    return text;
}

struct backend_scope {
    backend_scope() { ggml_backend_load_all(); llama_backend_init(); }
    ~backend_scope() { llama_backend_free(); }
};

struct batch_storage {
    std::vector<llama_token> tokens;
    std::vector<llama_pos> positions;
    std::vector<std::int32_t> counts;
    std::vector<llama_seq_id *> sequences;
    std::vector<std::int8_t> outputs;
    llama_seq_id sequence = 0;

    explicit batch_storage(std::uint32_t capacity)
        : tokens(capacity), positions(capacity), counts(capacity, 1), sequences(capacity, &sequence), outputs(capacity) {}

    batch_storage(const batch_storage &) = delete;
    batch_storage & operator=(const batch_storage &) = delete;
    batch_storage(batch_storage &&) = delete;
    batch_storage & operator=(batch_storage &&) = delete;

    llama_batch fill(const std::vector<std::int32_t> & input, std::uint32_t first, std::uint32_t count, bool save) {
        for (std::uint32_t i = 0; i < count; ++i) {
            tokens[i] = input[first + i];
            positions[i] = static_cast<llama_pos>(first + i);
            outputs[i] = save && i + 1 == count;
        }
        return {static_cast<std::int32_t>(count), tokens.data(), nullptr, positions.data(), counts.data(), sequences.data(), outputs.data()};
    }
};

int collect(const options & args) {
    tq_logits::new_file output(args.output);
    const auto prompt = read_prompt(args.prompt);
    backend_scope backend;
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = static_cast<std::int32_t>(args.gpu_layers);
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(args.model.c_str(), model_params), llama_model_free);
    if (!model) { throw std::runtime_error("cannot load model"); }
    if (llama_model_has_encoder(model.get()) || !llama_model_has_decoder(model.get()) ||
        llama_model_is_diffusion(model.get())) {
        throw std::invalid_argument("collector requires an autoregressive decoder-only text model");
    }
    const auto * vocab = llama_model_get_vocab(model.get());
    const auto vocabulary = llama_vocab_n_tokens(vocab);
    if (vocabulary <= 0 || vocabulary > 2000000) { throw std::invalid_argument("vocabulary exceeds TP5LOG2 limits"); }
    const auto needed = llama_tokenize(vocab, prompt.data(), static_cast<std::int32_t>(prompt.size()), nullptr, 0, args.add_special, false);
    if (needed == INT32_MIN || needed >= 0 || -std::int64_t(needed) > args.context) {
        throw std::invalid_argument("input token count is empty, invalid, or exceeds --ctx-size");
    }
    std::vector<std::int32_t> tokens(static_cast<std::size_t>(-std::int64_t(needed)));
    const auto actual = llama_tokenize(vocab, prompt.data(), static_cast<std::int32_t>(prompt.size()), tokens.data(),
                                      static_cast<std::int32_t>(tokens.size()), args.add_special, false);
    if (actual != static_cast<std::int32_t>(tokens.size())) { throw std::runtime_error("tokenizer count changed"); }
    tq_logits::tape_writer tape(output, static_cast<std::uint32_t>(vocabulary), args.prefill, tokens, args.max_output_bytes);

    auto context_params = llama_context_default_params();
    const auto capacity = std::min(args.batch, args.prefill);
    context_params.n_ctx = args.context;
    context_params.n_batch = capacity;
    context_params.n_ubatch = capacity;
    context_params.n_seq_max = 1;
    context_params.n_outputs_max = 1;
    context_params.n_threads = static_cast<std::int32_t>(args.threads);
    context_params.n_threads_batch = static_cast<std::int32_t>(args.threads);
    context_params.no_perf = true;
    if (!args.gpu_layers) { context_params.offload_kqv = false; context_params.op_offload = false; }
    std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), context_params), llama_free);
    if (!context) { throw std::runtime_error("cannot create context"); }
    if (llama_n_ctx(context.get()) < tokens.size()) { throw std::runtime_error("actual context cannot contain the input"); }

    batch_storage batch(capacity);
    std::uint32_t calls = 0;
    for (std::uint32_t first = 0; first < tokens.size();) {
        const auto count = first < args.prefill ? std::min(capacity, args.prefill - first) : 1u;
        const bool save = first + count >= args.prefill;
        const auto result = llama_decode(context.get(), batch.fill(tokens, first, count, save));
        llama_synchronize(context.get());
        if (result != 0) { throw std::runtime_error("llama_decode failed at input position " + std::to_string(first) + " with code " + std::to_string(result)); }
        ++calls;
        if (save) { tape.append(llama_get_logits_ith(context.get(), static_cast<std::int32_t>(count - 1))); }
        first += count;
    }
    tape.finish();
    std::cout << "{\"format\":\"TP5LOG2\",\"vocabulary\":" << vocabulary
              << ",\"input_tokens\":" << tokens.size() << ",\"prefill\":" << args.prefill
              << ",\"rows\":" << tokens.size() - args.prefill + 1 << ",\"bytes\":" << tape.bytes()
              << ",\"decode_calls\":" << calls << ",\"context\":" << llama_n_ctx(context.get())
              << ",\"batch_size\":" << capacity << ",\"threads\":" << args.threads
              << ",\"gpu_layers_requested\":" << args.gpu_layers << ",\"add_special\":" << (args.add_special ? "true" : "false")
              << ",\"parse_special\":false,\"sampled\":false}\n";
    std::cout.flush();
    if (!std::cout) { throw std::runtime_error("complete tape published, but stdout summary failed"); }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") { usage(); return 0; }
    try {
        return collect(parse(argc, argv));
    } catch (const std::invalid_argument & error) {
        std::cerr << "tq-logits-compose: " << error.what() << '\n';
        return 2;
    } catch (const std::exception & error) {
        std::cerr << "tq-logits-compose: " << error.what() << '\n';
        return 1;
    }
}
