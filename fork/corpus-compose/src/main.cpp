#include "tq/corpus/trie.h"
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct options {
    std::string command;
    std::string model;
    std::string corpus;
    std::string input;
    int32_t context = 2048;
    int32_t gpu_layers = 0;
    int32_t threads = 4;
    bool reference = false;
};

void usage() {
    std::cout
        << "Build a recorded token trie or score it with a real llama model.\n"
        << "  llama-corpus build -m MODEL --input PROMPTS.txt --corpus NEW_DIR\n"
        << "  llama-corpus score -m MODEL --corpus DIR [--reference]\n"
        << "  -c, --ctx-size N      maximum request tokens (default 2048)\n"
        << "  -ngl, --gpu-layers N  offloaded layers (default 0)\n"
        << "  -t, --threads N       CPU threads (default 4)\n"
        << "Input has one raw prompt per line; build uses the model tokenizer.\n"
        << "Score replays recorded token IDs. Use the same tokenizer as capture.\n"
        << "--reference decodes every request independently, without prefix reuse.\n";
}

int32_t integer(const std::string & value) {
    int32_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::invalid_argument("invalid integer: " + value);
    }
    return result;
}

options parse(int argc, char ** argv) {
    if (argc < 2) {
        throw std::invalid_argument("expected build or score");
    }
    options result;
    result.command = argv[1];
    if (result.command != "build" && result.command != "score") {
        throw std::invalid_argument("expected build or score");
    }
    for (int i = 2; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--reference") {
            result.reference = true;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + key);
        }
        const std::string value = argv[++i];
        if (key == "-m" || key == "--model") result.model = value;
        else if (key == "--corpus") result.corpus = value;
        else if (key == "--input") result.input = value;
        else if (key == "-c" || key == "--ctx-size") result.context = integer(value);
        else if (key == "-ngl" || key == "--gpu-layers") result.gpu_layers = integer(value);
        else if (key == "-t" || key == "--threads") result.threads = integer(value);
        else throw std::invalid_argument("unknown option: " + key);
    }
    if (result.model.empty() || result.corpus.empty()) {
        throw std::invalid_argument("--model and --corpus are required");
    }
    if (result.context <= 0 || result.threads <= 0 || result.gpu_layers < 0) {
        throw std::invalid_argument("context and threads must be positive; gpu layers must be nonnegative");
    }
    if (result.command == "build" && (result.input.empty() || result.reference)) {
        throw std::invalid_argument("build requires --input and does not accept --reference");
    }
    if (result.command == "score" && !result.input.empty()) {
        throw std::invalid_argument("--input is only valid for build");
    }
    return result;
}

struct backend {
    backend() { ggml_backend_load_all(); llama_backend_init(); }
    ~backend() { llama_backend_free(); }
};
struct model_deleter {
    void operator()(llama_model * model) const { llama_model_free(model); }
};
struct context_deleter {
    void operator()(llama_context * context) const { llama_free(context); }
};

std::vector<int32_t> tokenize(const llama_vocab * vocab, const std::string & text) {
    if (text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("prompt exceeds tokenizer length range");
    }
    const auto count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                     nullptr, 0, true, true);
    if (count == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("tokenizer length overflow");
    }
    std::vector<int32_t> tokens(static_cast<size_t>(count < 0 ? -count : count));
    const auto written = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                       tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    if (written <= 0 || static_cast<size_t>(written) > tokens.size()) {
        throw std::runtime_error("tokenizer produced no usable token sequence");
    }
    tokens.resize(static_cast<size_t>(written));
    return tokens;
}

class sequence {
public:
    struct snapshot {
        std::vector<uint8_t> bytes;
        llama_pos last_position = -1;
    };

    sequence(llama_model * model, int32_t vocab_size, const options & args) : vocab_size_(vocab_size) {
        auto params = llama_context_default_params();
        params.n_ctx = static_cast<uint32_t>(args.context);
        params.n_batch = 1;
        params.n_ubatch = 1;
        params.n_threads = args.threads;
        params.n_threads_batch = args.threads;
        context_.reset(llama_init_from_model(model, params));
        if (!context_) throw std::runtime_error("cannot create model context");
    }

    void clear() { llama_memory_clear(llama_get_memory(context_.get()), true); }

    void decode(int32_t token) {
        auto batch = llama_batch_get_one(&token, 1);
        if (llama_decode(context_.get(), batch) != 0) {
            throw std::runtime_error("model decode failed");
        }
        ++decoded_;
    }

    std::vector<double> losses(const std::vector<int32_t> & tokens) const {
        const float * logits = llama_get_logits_ith(context_.get(), -1);
        if (!logits) throw std::runtime_error("model returned no logits");
        double maximum = -std::numeric_limits<double>::infinity();
        for (int32_t i = 0; i < vocab_size_; ++i) {
            if (std::isnan(logits[i]) || logits[i] == std::numeric_limits<float>::infinity()) {
                throw std::runtime_error("model returned invalid logits");
            }
            maximum = std::max(maximum, static_cast<double>(logits[i]));
        }
        if (!std::isfinite(maximum)) throw std::runtime_error("model returned no finite logit");
        double denominator = 0;
        for (int32_t i = 0; i < vocab_size_; ++i) {
            denominator += std::exp(static_cast<double>(logits[i]) - maximum);
        }
        const double normalizer = maximum + std::log(denominator);
        std::vector<double> result;
        result.reserve(tokens.size());
        for (const auto token : tokens) {
            const double loss = normalizer - logits[token];
            if (!std::isfinite(loss)) throw std::runtime_error("request token has non-finite log loss");
            result.push_back(loss);
        }
        return result;
    }

    snapshot save() const {
        const auto size = llama_state_seq_get_size(context_.get(), 0);
        if (!size) throw std::runtime_error("model does not provide sequence snapshots");
        std::vector<uint8_t> result(size);
        if (llama_state_seq_get_data(context_.get(), result.data(), result.size(), 0) != size) {
            throw std::runtime_error("sequence snapshot failed");
        }
        return {std::move(result), llama_memory_seq_pos_max(llama_get_memory(context_.get()), 0)};
    }

    void restore(const snapshot & state) {
        if (llama_state_seq_set_data(context_.get(), state.bytes.data(), state.bytes.size(), 0) != state.bytes.size() ||
            llama_memory_seq_pos_max(llama_get_memory(context_.get()), 0) != state.last_position) {
            throw std::runtime_error("sequence restore failed; use --reference for full replay");
        }
    }

    uint64_t decoded() const { return decoded_; }

private:
    std::unique_ptr<llama_context, context_deleter> context_;
    int32_t vocab_size_;
    uint64_t decoded_ = 0;
};

struct request_score {
    double total = 0;
    uint64_t positions = 0;
};

struct score {
    double total = 0;
    uint64_t positions = 0;
    std::vector<request_score> requests;
};

score full_replay(const tq::corpus::token_trie & trie, sequence & model) {
    score result;
    for (const auto leaf : trie.request_leaves()) {
        const auto path = trie.path_to(leaf);
        model.clear();
        request_score request;
        for (size_t i = 0; i < path.size(); ++i) {
            if (i) {
                request.total += model.losses({path[i]}).front();
                ++request.positions;
            }
            model.decode(path[i]);
        }
        result.total += request.total;
        result.positions += request.positions;
        result.requests.push_back(request);
    }
    return result;
}

score prefix_replay(const tq::corpus::token_trie & trie, sequence & model) {
    struct frame {
        std::vector<uint64_t> children;
        std::vector<double> losses;
        sequence::snapshot state;
        size_t next = 0;
    };
    std::vector<frame> frames;
    std::vector<double> loss_by_node(static_cast<size_t>(trie.node_count()) + 1,
                                     std::numeric_limits<double>::quiet_NaN());
    auto push = [&](uint64_t id) {
        auto children = trie.children(id);
        if (children.empty()) return;
        std::vector<int32_t> tokens;
        for (const auto child : children) tokens.push_back(trie.node(child).token);
        // Sequence snapshots do not promise to restore logits. Capture all edge
        // scores while the parent logits are current, before visiting children.
        auto losses = model.losses(tokens);
        auto state = children.size() > 1 ? model.save() : sequence::snapshot{};
        frames.push_back({std::move(children), std::move(losses), std::move(state), 0});
    };

    for (const auto root : trie.children(0)) {
        model.clear();
        model.decode(trie.node(root).token);
        push(root);
        while (!frames.empty()) {
            auto & current = frames.back();
            if (current.next == current.children.size()) {
                frames.pop_back();
                continue;
            }
            if (current.next && !current.state.bytes.empty()) model.restore(current.state);
            const auto index = current.next++;
            const auto child = current.children[index];
            loss_by_node[static_cast<size_t>(child)] = current.losses[index];
            model.decode(trie.node(child).token);
            push(child);
        }
    }
    score result;
    for (auto leaf : trie.request_leaves()) {
        request_score request;
        while (leaf) {
            const auto node = trie.node(leaf);
            if (node.parent) {
                const double loss = loss_by_node[static_cast<size_t>(leaf)];
                if (!std::isfinite(loss)) throw std::runtime_error("request edge was not scored");
                request.total += loss;
                ++request.positions;
            }
            leaf = node.parent;
        }
        result.total += request.total;
        result.positions += request.positions;
        result.requests.push_back(request);
    }
    return result;
}

int run(const options & args) {
    backend lifetime;
    auto params = llama_model_default_params();
    params.n_gpu_layers = args.gpu_layers;
    params.vocab_only = args.command == "build";
    std::unique_ptr<llama_model, model_deleter> model(llama_model_load_from_file(args.model.c_str(), params));
    if (!model) throw std::runtime_error("cannot load model");
    const auto * vocab = llama_model_get_vocab(model.get());
    const auto n_vocab = llama_vocab_n_tokens(vocab);
    if (args.command == "build") {
        std::ifstream input(args.input);
        if (!input) throw std::runtime_error("cannot open prompt input");
        tq::corpus::token_trie trie(n_vocab);
        std::string line;
        while (std::getline(input, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            trie.append(tokenize(vocab, line));
        }
        if (!input.eof()) throw std::runtime_error("cannot read prompt input");
        if (!trie.request_count()) throw std::runtime_error("input contains no requests");
        trie.save(args.corpus);
        std::cout << "{\"requests\":" << trie.request_count()
                  << ",\"nodes\":" << trie.node_count()
                  << ",\"request_tokens\":" << trie.leaf_positions() << "}\n";
        return 0;
    }
    if (llama_model_has_encoder(model.get()) || !llama_model_has_decoder(model.get())) {
        throw std::invalid_argument("score requires a decoder-only model");
    }
    const auto trie = tq::corpus::token_trie::load(args.corpus, n_vocab);
    if (!trie.request_count()) throw std::invalid_argument("corpus contains no requests");
    for (const auto leaf : trie.request_leaves()) {
        if (trie.path_to(leaf).size() > static_cast<size_t>(args.context)) {
            throw std::invalid_argument("request exceeds --ctx-size; no request was truncated");
        }
    }
    sequence state(model.get(), n_vocab, args);
    const auto result = args.reference ? full_replay(trie, state) : prefix_replay(trie, state);
    if (!result.positions) throw std::invalid_argument("corpus has no token position to score");
    std::cout << std::setprecision(17)
              << "{\"requests\":" << trie.request_count()
              << ",\"nodes\":" << trie.node_count()
              << ",\"request_tokens\":" << trie.leaf_positions()
              << ",\"scored_positions\":" << result.positions
              << ",\"decoded_tokens\":" << state.decoded()
              << ",\"nll_sum\":" << result.total
              << ",\"mean_nll\":" << result.total / static_cast<double>(result.positions)
              << ",\"reference\":" << (args.reference ? "true" : "false")
              << ",\"request_scores\":[";
    for (size_t i = 0; i < result.requests.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << "{\"nll_sum\":" << result.requests[i].total
                  << ",\"scored_positions\":" << result.requests[i].positions << '}';
    }
    std::cout << "]}\n";
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
            usage();
            return 0;
        }
        return run(parse(argc, argv));
    } catch (const std::exception & error) {
        std::cerr << "llama-corpus: " << error.what() << '\n';
        return 1;
    }
}
