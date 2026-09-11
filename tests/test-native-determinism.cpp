#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static void add_token(llama_batch & batch, llama_token token, llama_pos pos, bool logits) {
    const int i = batch.n_tokens++;
    batch.token[i] = token;
    batch.pos[i] = pos;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = logits;
}

static llama_context * make_context(llama_model * model, bool f16, uint32_t rollback, uint32_t ubatch) {
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_ctx_kv = 4096;
    cp.kv_unified = true;
    cp.n_batch = 256;
    cp.n_ubatch = ubatch;
    cp.n_seq_max = 1;
    cp.n_seq_recurrent = 1;
    cp.n_rs_seq = rollback;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.type_k = f16 ? GGML_TYPE_F16 : GGML_TYPE_TURBO4_0;
    cp.type_v = f16 ? GGML_TYPE_F16 : GGML_TYPE_TURBO2_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.rerot = false;
    return llama_init_from_model(model, cp);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL [steps=32] [turbo|f16] [--prompt TEXT] [--rollback N] [--tape-in FILE] [--tape-out FILE]\n", argv[0]);
        return 2;
    }
    int status = 0;
    llama_model * model = nullptr;
    llama_context * ctx1 = nullptr;
    llama_context * ctx2 = nullptr;
    llama_batch b1{};
    llama_batch b2{};

    try {
        const int steps = argc > 2 ? std::stoi(argv[2]) : 32;
        const bool f16 = argc > 3 && std::strcmp(argv[3], "f16") == 0;
        std::string prompt = "9.11 or 9.9, which is larger?";
        int rollback = 0;
        std::string tape_in, tape_out;

        for (int i = 4; i < argc; ++i) {
            const std::string arg = argv[i];
            require(i + 1 < argc, "missing arg value");
            const std::string val = argv[++i];
            if (arg == "--prompt") prompt = val;
            else if (arg == "--rollback") rollback = std::stoi(val);
            else if (arg == "--tape-in") tape_in = val;
            else if (arg == "--tape-out") tape_out = val;
            else throw std::runtime_error("unknown arg: " + arg);
        }

        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        model = llama_model_load_from_file(argv[1], mp);
        require(model != nullptr, "model load failed");

        ctx1 = make_context(model, f16, uint32_t(rollback), 256);
        ctx2 = make_context(model, f16, uint32_t(rollback), 256);
        require(ctx1 && ctx2, "context init failed");

        auto * vocab = llama_model_get_vocab(model);
        const int n_prompt = -llama_tokenize(vocab, prompt.data(), int(prompt.size()), nullptr, 0, true, true);
        require(n_prompt > 0 && n_prompt < 256, "bad prompt tokens");
        std::vector<llama_token> tokens(n_prompt);
        require(llama_tokenize(vocab, prompt.data(), int(prompt.size()), tokens.data(), n_prompt, true, true) == n_prompt, "tokenize failed");

        b1 = llama_batch_init(256, 0, 1);
        b2 = llama_batch_init(256, 0, 1);

        for (int i = 0; i < n_prompt; ++i) {
            add_token(b1, tokens[i], i, i == n_prompt - 1);
            add_token(b2, tokens[i], i, i == n_prompt - 1);
        }

        require(llama_decode(ctx1, b1) == 0, "ctx1 prefill failed");
        require(llama_decode(ctx2, b2) == 0, "ctx2 prefill failed");

        const int nv = llama_vocab_n_tokens(vocab);
        const float * l1 = llama_get_logits_ith(ctx1, -1);
        const float * l2 = llama_get_logits_ith(ctx2, -1);

        double max_diff = 0.0;
        for (int i = 0; i < nv; ++i) {
            max_diff = std::max(max_diff, double(std::abs(l1[i] - l2[i])));
        }
        std::printf("PREFILL native vs native max_diff=%g\n", max_diff);
        require(max_diff == 0.0, "native prefill determinism failed (max_diff > 0)");

        std::vector<llama_token> tape;
        if (!tape_in.empty()) {
            std::ifstream file(tape_in);
            std::string magic;
            size_t prompt_count = 0, count = 0;
            file >> magic >> prompt_count >> count;
            require(bool(file) && magic == "REROT_TOKEN_TAPE_V1" && prompt_count == tokens.size() && count >= size_t(steps), "invalid tape header");
            for (llama_token exp : tokens) {
                llama_token v = -1; file >> v;
                require(v == exp, "tape prompt mismatch");
            }
            tape.resize(count);
            for (auto & v : tape) file >> v;
        }

        llama_token next = llama_token(std::max_element(l1, l1 + nv) - l1);
        std::vector<llama_token> recorded_tape;
        int mismatches = 0;

        for (int step = 0; step < steps; ++step) {
            if (!tape.empty()) next = tape[step];
            recorded_tape.push_back(next);
            const llama_pos pos = n_prompt + step;

            b1.n_tokens = 0;
            b2.n_tokens = 0;
            add_token(b1, next, pos, true);
            add_token(b2, next, pos, true);

            require(llama_decode(ctx1, b1) == 0, "ctx1 decode failed");
            require(llama_decode(ctx2, b2) == 0, "ctx2 decode failed");

            const float * log1 = llama_get_logits_ith(ctx1, 0);
            const float * log2 = llama_get_logits_ith(ctx2, 0);

            double step_diff = 0.0;
            for (int i = 0; i < nv; ++i) {
                step_diff = std::max(step_diff, double(std::abs(log1[i] - log2[i])));
            }
            const int top1 = int(std::max_element(log1, log1 + nv) - log1);
            const int top2 = int(std::max_element(log2, log2 + nv) - log2);
            if (top1 != top2 || step_diff > 0.0) {
                std::printf("MISMATCH step=%d top1=%d top2=%d max_diff=%g\n", step, top1, top2, step_diff);
                ++mismatches;
            }
            next = top1;
        }

        if (!tape_out.empty()) {
            std::ofstream file(tape_out);
            file << "REROT_TOKEN_TAPE_V1\n" << tokens.size() << ' ' << recorded_tape.size() << '\n';
            for (auto t : tokens) file << t << ' ';
            file << '\n';
            for (auto t : recorded_tape) file << t << '\n';
            file.flush();
        }

        std::printf("SUMMARY native-vs-native cache=%s steps=%d mismatches=%d\n", f16 ? "f16" : "turbo", steps, mismatches);
        if (mismatches != 0) status = 1;

    } catch (const std::exception & ex) {
        std::fprintf(stderr, "FAIL: %s\n", ex.what());
        status = 1;
    }

    if (b1.token) llama_batch_free(b1);
    if (b2.token) llama_batch_free(b2);
    if (ctx1) llama_free(ctx1);
    if (ctx2) llama_free(ctx2);
    if (model) llama_model_free(model);
    llama_backend_free();
    return status;
}
