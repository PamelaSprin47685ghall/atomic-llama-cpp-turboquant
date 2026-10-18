#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

using bytes = std::vector<uint8_t>;
using tokens = std::vector<llama_token>;

struct batch_owner {
    llama_batch value;

    explicit batch_owner(int32_t count) : value(llama_batch_init(count, 0, 1)) {}
    ~batch_owner() { llama_batch_free(value); }
    batch_owner(const batch_owner &) = delete;
    batch_owner & operator=(const batch_owner &) = delete;
};

class session {
public:
    explicit session(llama_model * model) : vocab(llama_model_get_vocab(model)) {
        auto params = llama_context_default_params();
        params.n_ctx = 768;
        params.n_batch = 64;
        params.n_ubatch = 64;
        params.n_seq_max = 3;
        params.n_rs_seq = 0;
        params.kv_unified = true;
        params.n_threads = 2;
        params.n_threads_batch = 2;
        params.no_perf = true;
        ctx.reset(llama_init_from_model(model, params));
        require(ctx != nullptr, "could not create the test context");
        require(llama_n_seq_max(ctx.get()) == 3, "the test needs three sequence slots");
        require(llama_n_rs_seq(ctx.get()) == 0, "the test needs rollback disabled");
        require(llama_get_memory(ctx.get()) != nullptr, "the model has no memory");
    }

    ~session() { llama_synchronize(ctx.get()); }

    tokens tokenize(const std::string & text) const {
        const int32_t size = llama_tokenize(vocab, text.data(), int32_t(text.size()), nullptr, 0, true, false);
        require(size < 0 && size != std::numeric_limits<int32_t>::min(), "could not size the prompt tokens");
        tokens result(size_t(-size));
        const int32_t count = llama_tokenize(vocab, text.data(), int32_t(text.size()), result.data(), -size, true, false);
        require(count > 3 && count <= 64, "the test prompt must contain 4 to 64 tokens");
        result.resize(size_t(count));
        return result;
    }

    void submit(llama_seq_id id, llama_pos first, const tokens & input) {
        require(!input.empty() && input.size() <= 64, "invalid test batch size");
        llama_synchronize(ctx.get());
        pending_batch = std::make_unique<batch_owner>(int32_t(input.size()));
        auto & batch = pending_batch->value;
        batch.n_tokens = int32_t(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            batch.token[i] = input[i];
            batch.pos[i] = first + llama_pos(i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = id;
            batch.logits[i] = i + 1 == input.size();
        }
        const int32_t status = llama_decode(ctx.get(), batch);
        if (status != 0) {
            llama_synchronize(ctx.get());
            throw std::runtime_error("llama_decode failed with status " + std::to_string(status));
        }
        ++decode_calls;
        input_rows += input.size();
    }

    std::vector<float> logits() const {
        llama_synchronize(ctx.get());
        const float * data = llama_get_logits_ith(ctx.get(), -1);
        const int32_t count = llama_vocab_n_tokens(vocab);
        require(data != nullptr && count > 0, "missing output logits");
        std::vector<float> result(data, data + count);
        for (const float value : result) {
            require(std::isfinite(value), "output contains a non-finite logit");
        }
        return result;
    }

    bytes save(llama_seq_id id) const {
        llama_synchronize(ctx.get());
        const size_t size = llama_state_seq_get_size(ctx.get(), id);
        require(size > 0 && size <= size_t(512) * 1024 * 1024, "invalid or excessive test snapshot size");
        bytes result(size);
        require(llama_state_seq_get_data(ctx.get(), result.data(), result.size(), id) == size,
                "could not save the full sequence state");
        return result;
    }

    void restore(llama_seq_id id, const bytes & state) {
        llama_synchronize(ctx.get());
        require(llama_state_seq_set_data(ctx.get(), state.data(), state.size(), id) == state.size(),
                "could not restore the full sequence state");
    }

    void remove(llama_seq_id id) {
        llama_synchronize(ctx.get());
        require(llama_memory_seq_rm(memory(), id, -1, -1), "could not remove the full sequence");
        require(position(id) == -1, "the removed sequence still has a position");
    }

    void copy(llama_seq_id source, llama_seq_id destination) {
        llama_synchronize(ctx.get());
        llama_memory_seq_cp(memory(), source, destination, -1, -1);
        require(position(destination) == position(source), "sequence copy lost its position");
    }

    void clear(bool data) {
        llama_synchronize(ctx.get());
        llama_memory_clear(memory(), data);
        llama_synchronize(ctx.get());
        for (llama_seq_id id = 0; id < 3; ++id) {
            require(position(id) == -1, "clear left a live sequence");
        }
    }

    void reject_partial_remove(llama_seq_id id) {
        llama_synchronize(ctx.get());
        const auto before = save(id);
        const auto last = position(id);
        require(last > 0, "the partial-remove test needs a live prefix");
        require(!llama_memory_seq_rm(memory(), id, last, last + 1), "partial rollback unexpectedly succeeded");
        require(position(id) == last && save(id) == before, "failed partial removal changed the sequence");
    }

    llama_pos position(llama_seq_id id) const {
        return llama_memory_seq_pos_max(memory(), id);
    }

    size_t decode_calls = 0;
    size_t input_rows = 0;

private:
    llama_memory_t memory() const { return llama_get_memory(ctx.get()); }
    const llama_vocab * vocab;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx{nullptr, llama_free};
    // The next operation owns the fence. Keep this storage until then.
    std::unique_ptr<batch_owner> pending_batch;
};

double compare_logits(const std::vector<float> & expected, const std::vector<float> & actual) {
    require(expected.size() == actual.size() && !expected.empty(), "logit dimensions changed");
    double maximum = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        require(std::isfinite(expected[i]) && std::isfinite(actual[i]), "non-finite comparison input");
        maximum = std::max(maximum, std::abs(double(expected[i]) - double(actual[i])));
    }
    require(maximum <= 1e-5, "same-shape continuation logits changed: max_abs=" + std::to_string(maximum));
    return maximum;
}

size_t check_neighbors(session & run, const std::array<tokens, 3> & prompts) {
    run.clear(false);
    for (llama_seq_id id = 0; id < 3; ++id) {
        run.submit(id, 0, prompts[size_t(id)]);
    }
    size_t comparisons = 0;
    for (size_t round = 0; round < 2; ++round) {
        for (llama_seq_id victim = 0; victim < 3; ++victim) {
            std::array<bytes, 3> before;
            for (llama_seq_id id = 0; id < 3; ++id) {
                if (id != victim) {
                    before[size_t(id)] = run.save(id);
                }
            }
            run.remove(victim);
            for (llama_seq_id id = 0; id < 3; ++id) {
                if (id != victim) {
                    require(run.save(id) == before[size_t(id)], "removal changed a neighbor snapshot");
                    ++comparisons;
                }
            }
            run.submit(victim, 0, prompts[(size_t(victim) + round + 1) % prompts.size()]);
            for (llama_seq_id id = 0; id < 3; ++id) {
                if (id != victim) {
                    require(run.save(id) == before[size_t(id)], "slot reuse changed a neighbor snapshot");
                    ++comparisons;
                }
            }
        }
    }
    run.reject_partial_remove(1);
    return comparisons;
}

size_t check_shared_tail(session & run, const std::array<tokens, 3> & prompts) {
    run.clear(false);
    run.submit(0, 0, prompts[0]);
    run.copy(0, 1);
    run.copy(0, 2);
    const auto first = run.save(1);
    const auto second = run.save(2);
    run.remove(0);
    require(run.save(1) == first && run.save(2) == second, "removal changed a shared survivor");
    run.submit(0, 0, prompts[1]);
    require(run.save(1) == first && run.save(2) == second, "reuse changed a shared survivor");
    const auto separate = run.save(0);
    run.submit(1, llama_pos(prompts[0].size()), tokens{prompts[2].back()});
    require(run.save(0) == separate && run.save(2) == second, "copy-on-write changed an untouched sequence");
    const auto advanced = run.save(1);
    run.remove(2);
    require(run.save(0) == separate && run.save(1) == advanced, "last shared release changed another sequence");
    return 8;
}

double check_restore_and_clear(session & run, const std::array<tokens, 3> & prompts) {
    run.clear(false);
    run.submit(1, 0, prompts[0]);
    const auto checkpoint = run.save(1);
    const auto initial_logits = run.logits();
    const tokens continuation(prompts[2].end() - 3, prompts[2].end());
    std::vector<std::vector<float>> expected;
    for (size_t i = 0; i < continuation.size(); ++i) {
        run.submit(1, llama_pos(prompts[0].size() + i), tokens{continuation[i]});
        expected.push_back(run.logits());
    }
    run.remove(1);
    run.submit(1, 0, prompts[1]);
    run.remove(1);
    run.restore(1, checkpoint);
    require(run.save(1) == checkpoint, "snapshot restore changed the saved bytes");
    double maximum = 0.0;
    for (size_t i = 0; i < continuation.size(); ++i) {
        run.submit(1, llama_pos(prompts[0].size() + i), tokens{continuation[i]});
        maximum = std::max(maximum, compare_logits(expected[i], run.logits()));
    }
    for (const bool data : {false, true}) {
        run.submit(1, run.position(1) + 1, tokens{continuation.front()});
        run.clear(data);
        run.submit(1, 0, prompts[0]);
        require(run.save(1) == checkpoint, "clear and refill changed the sequence state");
        maximum = std::max(maximum, compare_logits(initial_logits, run.logits()));
    }
    return maximum;
}

int parse_layers(const std::string & text) {
    int value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    require(result.ec == std::errc{} && result.ptr == text.data() + text.size() && value >= 0 && value <= 999,
            "gpu layers must be an integer from 0 to 999");
    return value;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        std::string model_path;
        int gpu_layers = 0;
        bool layers_set = false;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            require(i + 1 < argc, "usage: test-recurrent-lifecycle --model MODEL [--gpu-layers N]");
            if (option == "--model" && model_path.empty()) {
                model_path = argv[++i];
            } else if (option == "--gpu-layers" && !layers_set) {
                gpu_layers = parse_layers(argv[++i]);
                layers_set = true;
            } else {
                throw std::runtime_error("unknown or repeated option: " + option);
            }
        }
        require(!model_path.empty(), "usage: test-recurrent-lifecycle --model MODEL [--gpu-layers N]");
        ggml_backend_load_all();
        llama_backend_init();
        struct backend_owner { ~backend_owner() { llama_backend_free(); } } backend;
        auto params = llama_model_default_params();
        params.n_gpu_layers = gpu_layers;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
                llama_model_load_from_file(model_path.c_str(), params), llama_model_free);
        require(model != nullptr, "could not load the model");
        require(llama_model_is_recurrent(model.get()) || llama_model_is_hybrid(model.get()),
                "this test requires a recurrent or hybrid model");
        session run(model.get());
        const std::array<tokens, 3> prompts = {
            run.tokenize("A small boat crossed the quiet lake before sunrise."),
            run.tokenize("The red box contains seven wooden blocks and a key."),
            run.tokenize("Write the next word in this sentence about a garden.")
        };
        const auto neighbors = check_neighbors(run, prompts);
        const auto shared = check_shared_tail(run, prompts);
        const auto maximum = check_restore_and_clear(run, prompts);
        std::cout << std::setprecision(17) << "{\"status\":\"pass\",\"neighbor_snapshot_checks\":" << neighbors
                  << ",\"shared_snapshot_checks\":" << shared
                  << ",\"continuation_max_abs\":" << maximum
                  << ",\"decode_calls\":" << run.decode_calls
                  << ",\"input_rows\":" << run.input_rows << "}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "recurrent lifecycle test failed: " << error.what() << '\n';
        return 1;
    }
}
