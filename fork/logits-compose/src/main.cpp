#include <llama.h>
#include <ggml.h>
#include <ggml-backend.h>

#include "tape-writer.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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
    bool mtp_ctx = false;      // create the context as LLAMA_CONTEXT_TYPE_MTP (draft head)
    bool dual = false;         // two MTP contexts over one model, interleaved decode
    std::string prefab_stats;  // dump llama_prefab_stats JSON per context at exit
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
        "  --mtp-ctx             Create the context as LLAMA_CONTEXT_TYPE_MTP. The batch\n"
        "                        then carries the wide hidden-state plane (width\n"
        "                        llama_model_n_embd_out) next to the token ids, exactly\n"
        "                        like the production draft decode; ubatches are forced to\n"
        "                        one token per step. Requires a Qwen4Exp GGUF with a\n"
        "                        nextn/MTP head.\n"
        "  --dual                Run a second --mtp-ctx context over the same model and\n"
        "                        interleave decode steps A,B,A,B. Writes a second tape at\n"
        "                        OUTPUT.dual and reports both contexts in --prefab-stats.\n"
        "                        Requires --mtp-ctx.\n"
        "  --prefab-stats FILE   Write per-context llama_prefab_stats (public C API) as\n"
        "                        JSON after the run; refuses an existing FILE.\n"
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
            arg != "--prefill" && arg != "--batch-size" && arg != "--threads" && arg != "--gpu-layers" && arg != "--max-output-bytes" &&
            arg != "--mtp-ctx" && arg != "--dual" && arg != "--prefab-stats") {
            throw std::invalid_argument("unknown option: " + arg);
        }
        if (arg == "--mtp-ctx") { result.mtp_ctx = true; continue; }
        if (arg == "--dual")    { result.dual = true; continue; }
        if (arg == "--prefab-stats") {
            if (++i == argc) { throw std::invalid_argument("missing value for " + arg); }
            result.prefab_stats = argv[i];
            continue;
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
    if (result.dual && !result.mtp_ctx) { throw std::invalid_argument("--dual requires --mtp-ctx"); }
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

// deterministic wide hidden-state plane for --mtp-ctx: pure integer math (splitmix64
// finalizer over a fixed seed, the position and the channel), identical across runs,
// arms and hosts. These values are fixed inputs to the production decode path, not a
// model of anything: the arm comparison needs the same input stream everywhere, and a
// real trunk export can replace it by driving the same llama_batch::embd plane.
constexpr std::uint64_t H_PLANE_SEED = 20261031ull;

inline float h_plane_value(std::uint64_t seed, llama_pos pos, std::int32_t channel) {
    std::uint64_t x = seed ^ (std::uint64_t(pos) * 0x9E3779B97F4A7C15ull) ^ (std::uint64_t(channel) * 0xC2B2AE3D27D4EB4Full);
    x ^= x >> 33;
    x *= 0xFF51AFD7ED558CCDull;
    x ^= x >> 33;
    return static_cast<float>(static_cast<std::int64_t>(x % 2000001ull) - 1000000ll) / 1000000.0f;
}

struct batch_storage {
    std::vector<llama_token> tokens;
    std::vector<llama_pos> positions;
    std::vector<std::int32_t> counts;
    std::vector<llama_seq_id *> sequences;
    std::vector<std::int8_t> outputs;
    std::vector<float> embd;        // wide hidden-state plane [capacity * embd_width]
    std::uint32_t embd_width = 0;   // 0: token-only batch (llama_batch::embd stays null)
    llama_seq_id sequence = 0;

    explicit batch_storage(std::uint32_t capacity, std::uint32_t embd_width_ = 0)
        : tokens(capacity), positions(capacity), counts(capacity, 1), sequences(capacity, &sequence), outputs(capacity),
          embd(std::size_t(capacity) * embd_width_), embd_width(embd_width_) {}

    batch_storage(const batch_storage &) = delete;
    batch_storage & operator=(const batch_storage &) = delete;
    batch_storage(batch_storage &&) = delete;
    batch_storage & operator=(batch_storage &&) = delete;

    llama_batch fill(const std::vector<std::int32_t> & input, std::uint32_t first, std::uint32_t count, bool save) {
        for (std::uint32_t i = 0; i < count; ++i) {
            tokens[i] = input[first + i];
            positions[i] = static_cast<llama_pos>(first + i);
            outputs[i] = save && i + 1 == count;
            if (embd_width) {
                float * row = embd.data() + std::size_t(i) * embd_width;
                for (std::uint32_t c = 0; c < embd_width; ++c) {
                    row[c] = h_plane_value(H_PLANE_SEED, positions[i], static_cast<std::int32_t>(c));
                }
            }
        }
        return {static_cast<std::int32_t>(count), tokens.data(), embd_width ? embd.data() : nullptr,
                positions.data(), counts.data(), sequences.data(), outputs.data()};
    }
};

// per-context llama_prefab_stats JSON (public C API); field order mirrors
// struct llama_prefab_stats in include/llama.h
std::string prefab_stats_json(const std::vector<std::pair<const char *, const llama_context *>> & contexts) {
    std::string out = "{\"contexts\":[";
    bool first_ctx = true;
    for (const auto & entry : contexts) {
        if (!first_ctx) { out += ","; }
        first_ctx = false;
        llama_prefab_stats ps{};
        llama_prefab_get_stats(entry.second, &ps);
        out += "{\"name\":\"";
        out += entry.first;
        out += "\",\"is_enabled\":";
        out += llama_prefab_is_enabled(entry.second) ? "true" : "false";
        out += ",\"n_cached_factors\":";
        out += std::to_string(llama_prefab_n_cached_factors(entry.second));
        out += ",\"stats\":{";
        const std::pair<const char *, std::uint64_t> fields[] = {
            {"n_ubatch_prefab",       ps.n_ubatch_prefab},
            {"n_ubatch_stock",        ps.n_ubatch_stock},
            {"n_stock_builds",        ps.n_stock_builds},
            {"n_stock_reuses",        ps.n_stock_reuses},
            {"n_envelope_builds",     ps.n_envelope_builds},
            {"n_envelope_reuses",     ps.n_envelope_reuses},
            {"n_envelope_allocs",     ps.n_envelope_allocs},
            {"n_sub_envelope",        ps.n_sub_envelope},
            {"n_pad_rows",            ps.n_pad_rows},
            {"n_factor_compiles",     ps.n_factor_compiles},
            {"n_factor_cache_hits",   ps.n_factor_cache_hits},
            {"n_factor_splices",      ps.n_factor_splices},
            {"n_factor_clones",       ps.n_factor_clones},
            {"n_cache_resets",        ps.n_cache_resets},
            {"n_fallback_no_scratch", ps.n_fallback_no_scratch},
            {"n_fallback_runtime",    ps.n_fallback_runtime},
            {"last_n_env",            ps.last_n_env},
            {"last_n_tokens",         ps.last_n_tokens},
            {"last_graph_nodes",      ps.last_graph_nodes},
            {"last_inp_tokens_ne0",   ps.last_inp_tokens_ne0},
            {"n_decouple_declines",   ps.n_decouple_declines},
            {"n_decouple_handled",    ps.n_decouple_handled},
            {"n_mtp_compose_hits",    ps.n_mtp_compose_hits},
            {"n_mtp_a_compiles",      ps.n_mtp_a_compiles},
            {"n_mtp_a_reuses",        ps.n_mtp_a_reuses},
            {"n_mtp_b_builds",        ps.n_mtp_b_builds},
            {"n_mtp_a_clones",        ps.n_mtp_a_clones},
            {"n_mtp_declines",        ps.n_mtp_declines},
            {"build_ns",              ps.build_ns},
            {"n_nodes_built",         ps.n_nodes_built},
            {"template_hits",         ps.template_hits},
        };
        for (std::size_t k = 0; k < sizeof(fields) / sizeof(fields[0]); ++k) {
            if (k) { out += ","; }
            out += "\"";
            out += fields[k].first;
            out += "\":";
            out += std::to_string(fields[k].second);
        }
        out += "}}";
    }
    out += "]}";
    return out;
}

void write_prefab_stats(const options & args, const std::vector<std::pair<const char *, const llama_context *>> & contexts) {
    if (args.prefab_stats.empty()) { return; }
    std::error_code ec;
    if (std::filesystem::exists(args.prefab_stats, ec)) {
        throw std::invalid_argument("--prefab-stats file already exists: " + args.prefab_stats);
    }
    std::ofstream out(args.prefab_stats, std::ios::binary | std::ios::trunc);
    if (!out) { throw std::runtime_error("cannot write --prefab-stats file"); }
    out << prefab_stats_json(contexts) << "\n";
    out.flush();
    if (!out) { throw std::runtime_error("failed to publish --prefab-stats file"); }
}

int collect(const options & args) {
    tq_logits::new_file output(args.output);
    const auto prompt = read_prompt(args.prompt);
    backend_scope backend;
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = static_cast<std::int32_t>(args.gpu_layers);
    // --mtp-ctx contract: the Qwen4Exp nextn/MTP block is a draft head, and the
    // loader skips its tensors (TENSOR_SKIP, src/models/qwen4exp.cpp:286)
    // unless load_mtp is requested at model-load time. The flag is part of the
    // MODEL, not the context: a context created later as LLAMA_CONTEXT_TYPE_MTP
    // needs nextn.eh_proj/enorm/hnorm to exist, otherwise the MTP graph builder
    // aborts on its eh_proj assert. A parameter change after this call cannot
    // bring the tensors back, so it is derived here from the already-parsed
    // --mtp-ctx flag (parse() runs before collect()). Without --mtp-ctx the
    // default (false) is kept and the loaded model is exactly the baseline one.
    model_params.load_mtp = args.mtp_ctx;
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

    // --dual: a second independent context over the same model (two private request
    // contexts in the production sense), its own tape published at OUTPUT.dual.
    std::unique_ptr<tq_logits::new_file> output_b;
    std::unique_ptr<tq_logits::tape_writer> tape_b;
    if (args.dual) {
        output_b = std::make_unique<tq_logits::new_file>(args.output + ".dual");
        tape_b = std::make_unique<tq_logits::tape_writer>(*output_b, static_cast<std::uint32_t>(vocabulary), args.prefill, tokens, args.max_output_bytes);
    }

    auto context_params = llama_context_default_params();
    // --mtp-ctx forces one token per ubatch: the production draft decode shape.
    const auto capacity = args.mtp_ctx ? 1u : std::min(args.batch, args.prefill);
    context_params.n_ctx = args.context;
    context_params.n_batch = capacity;
    context_params.n_ubatch = capacity;
    context_params.n_seq_max = 1;
    context_params.n_outputs_max = 1;
    context_params.n_threads = static_cast<std::int32_t>(args.threads);
    context_params.n_threads_batch = static_cast<std::int32_t>(args.threads);
    context_params.no_perf = true;
    context_params.ctx_type = args.mtp_ctx ? LLAMA_CONTEXT_TYPE_MTP : LLAMA_CONTEXT_TYPE_DEFAULT;
    if (!args.gpu_layers) { context_params.offload_kqv = false; context_params.op_offload = false; }
    std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), context_params), llama_free);
    if (!context) { throw std::runtime_error("cannot create context"); }
    if (llama_n_ctx(context.get()) < tokens.size()) { throw std::runtime_error("actual context cannot contain the input"); }

    // the wide hidden-state plane width of the production MTP decode
    const auto embd_width = args.mtp_ctx ? static_cast<std::uint32_t>(llama_model_n_embd_out(model.get())) : 0u;
    if (args.mtp_ctx && embd_width == 0) { throw std::invalid_argument("--mtp-ctx: model reports n_embd_out == 0"); }

    // the deleter is a function pointer type, which is not default-constructible in
    // C++17: initialise it explicitly (with a null handle) so reset() can adopt the
    // second context and the default-constructed state stays well-defined
    std::unique_ptr<llama_context, decltype(&llama_free)> context_b(nullptr, llama_free);
    if (args.dual) {
        context_b.reset(llama_init_from_model(model.get(), context_params));
        if (!context_b) { throw std::runtime_error("cannot create the --dual context"); }
    }

    batch_storage batch(capacity, embd_width);
    batch_storage batch_b(capacity, embd_width);
    std::uint32_t calls = 0;
    for (std::uint32_t first = 0; first < tokens.size();) {
        const auto count = first < args.prefill ? std::min(capacity, args.prefill - first) : 1u;
        const bool save = first + count >= args.prefill;
        const auto result = llama_decode(context.get(), batch.fill(tokens, first, count, save));
        llama_synchronize(context.get());
        if (result != 0) { throw std::runtime_error("llama_decode failed at input position " + std::to_string(first) + " with code " + std::to_string(result)); }
        ++calls;
        if (save) { tape.append(llama_get_logits_ith(context.get(), static_cast<std::int32_t>(count - 1))); }
        if (context_b) {
            const auto result_b = llama_decode(context_b.get(), batch_b.fill(tokens, first, count, save));
            llama_synchronize(context_b.get());
            if (result_b != 0) { throw std::runtime_error("dual context decode failed at input position " + std::to_string(first) + " with code " + std::to_string(result_b)); }
            if (save) { tape_b->append(llama_get_logits_ith(context_b.get(), static_cast<std::int32_t>(count - 1))); }
        }
        first += count;
    }
    tape.finish();
    if (tape_b) { tape_b->finish(); }

    {
        std::vector<std::pair<const char *, const llama_context *>> stats_ctxs{{"A", context.get()}};
        if (context_b) { stats_ctxs.push_back({"B", context_b.get()}); }
        write_prefab_stats(args, stats_ctxs);
    }
    std::cout << "{\"format\":\"TP5LOG2\",\"vocabulary\":" << vocabulary
              << ",\"input_tokens\":" << tokens.size() << ",\"prefill\":" << args.prefill
              << ",\"rows\":" << tokens.size() - args.prefill + 1 << ",\"bytes\":" << tape.bytes()
              << ",\"decode_calls\":" << calls << ",\"context\":" << llama_n_ctx(context.get())
              << ",\"batch_size\":" << capacity << ",\"threads\":" << args.threads
              << ",\"gpu_layers_requested\":" << args.gpu_layers << ",\"add_special\":" << (args.add_special ? "true" : "false")
              << ",\"mtp_ctx\":" << (args.mtp_ctx ? "true" : "false")
              << ",\"dual\":" << (args.dual ? "true" : "false")
              << ",\"embd_width\":" << embd_width
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
