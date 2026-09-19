// Fixed-token numerical acceptance probe, deliberately separate from timing.
// Uses normal model/context initialization, batched prefill and one-token
// decode. Every vocabulary logit is saved, not just argmax or top-k samples.
// Run serially on idle GPUs; it never starts a server or resets a device.
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct backend_guard {
    backend_guard() { llama_backend_init(); }
    ~backend_guard() { llama_backend_free(); }
};

struct batch_guard {
    llama_batch value;
    explicit batch_guard(int capacity) : value(llama_batch_init(capacity, 0, 1)) {}
    ~batch_guard() { llama_batch_free(value); }
};

uint32_t positive(const std::string & text) {
    size_t used = 0;
    const auto value = std::stoll(text, &used);
    if (used != text.size() || value <= 0 || value > 1000000) {
        throw std::invalid_argument("audit lengths must be in 1..1000000");
    }
    return uint32_t(value);
}

void write_u32(std::ostream & out, uint32_t value) {
    // Explicit little-endian tape header, independent of struct padding.
    const char bytes[] = {char(value), char(value >> 8), char(value >> 16), char(value >> 24)};
    out.write(bytes, sizeof(bytes));
}
}

int main(int argc, char ** argv) {
    try {
        std::setlocale(LC_NUMERIC, "C");
        std::string output;
        uint32_t prefill = 16;
        uint32_t steps = 280;
        std::vector<char *> forwarded{argv[0]};
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--audit-output" || arg == "--audit-prefill" || arg == "--audit-steps") {
                if (++i == argc) {
                    throw std::invalid_argument("missing value for " + arg);
                }
                if (arg == "--audit-output") output = argv[i];
                else if (arg == "--audit-prefill") prefill = positive(argv[i]);
                else steps = positive(argv[i]);
            } else {
                forwarded.push_back(argv[i]);
            }
        }
        if (output.empty() || std::filesystem::exists(output)) {
            throw std::invalid_argument("provide --audit-output with a new path; --audit-prefill and --audit-steps set tape lengths");
        }
        const uint16_t endian = 1;
        if (*reinterpret_cast<const uint8_t *>(&endian) != 1 || sizeof(float) != 4) {
            throw std::runtime_error("the F32 tape requires a little-endian, 32-bit float host");
        }

        common_params params;
        params.prompt = "The river flows past the quiet town. Count carefully: 1,2,3,4,5,6,7,8,9,10. ";
        params.n_ctx = 4096;
        params.n_batch = 512;
        params.n_ubatch = 512;
        params.n_parallel = 1;
        params.warmup = false;
        common_init();
        if (!common_params_parse(int(forwarded.size()), forwarded.data(), params, LLAMA_EXAMPLE_COMMON)) {
            return 1;
        }
        // A sampler must not replace full-vocabulary logits with GPU top-k.
        params.sampling.backend_sampling = false;
        params.warmup = false;
        if (params.n_parallel != 1 || params.n_batch <= 0 || params.n_ctx <= 0 ||
            uint32_t(params.n_ctx) < prefill + steps) {
            throw std::invalid_argument("use one sequence and a context covering prefill + steps");
        }
        backend_guard backend;
        auto init = common_init_from_params(params);
        if (!init || !init->model() || !init->context()) {
            throw std::runtime_error("model/context initialization failed");
        }
        auto * ctx = init->context();
        const auto * vocab = llama_model_get_vocab(init->model());
        const uint32_t n_vocab = llama_vocab_n_tokens(vocab);
        const auto seed = common_tokenize(vocab, params.prompt, true, true);
        if (seed.empty()) throw std::runtime_error("empty token seed");
        std::vector<llama_token> tokens(prefill + steps);
        for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = seed[i % seed.size()];

        std::ofstream out(output, std::ios::binary | std::ios::out);
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out.write("TP5LOG2\0", 8);
        write_u32(out, n_vocab);
        write_u32(out, prefill);
        write_u32(out, steps);
        write_u32(out, uint32_t(tokens.size()));
        for (const auto token : tokens) write_u32(out, uint32_t(token));

        batch_guard batch(params.n_batch);
        auto decode = [&](uint32_t first, uint32_t count) {
            common_batch_clear(batch.value);
            for (uint32_t j = 0; j < count; ++j) {
                common_batch_add(batch.value, tokens[first + j], first + j, {0}, j + 1 == count);
            }
            const int status = llama_decode(ctx, batch.value);
            if (status != 0) throw std::runtime_error("llama_decode failed: " + std::to_string(status));
        };
        auto save = [&](uint32_t row) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            if (!logits) throw std::runtime_error("missing logits");
            for (uint32_t i = 0; i < n_vocab; ++i) {
                if (!std::isfinite(logits[i])) {
                    throw std::runtime_error("non-finite logit at row " + std::to_string(row) + ", token " + std::to_string(i));
                }
            }
            out.write(reinterpret_cast<const char *>(logits), size_t(n_vocab) * sizeof(float));
            if (row % 32 == 0 || row == steps) {
                fprintf(stderr, "tp5-logits: saved row %u/%u, vocabulary=%u\n", row, steps, n_vocab);
            }
        };
        for (uint32_t first = 0; first < prefill;) {
            const auto count = std::min<uint32_t>(params.n_batch, prefill - first);
            decode(first, count);
            first += count;
        }
        save(0);
        for (uint32_t step = 0; step < steps; ++step) {
            decode(prefill + step, 1);
            save(step + 1);
        }
        out.close();
        fprintf(stderr, "tp5-logits: PASS, %u x %u finite logits; fixed input tape, no sampled tokens\n", steps + 1, n_vocab);
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "tp5-logits: %s\n", error.what());
        return 1;
    }
}
