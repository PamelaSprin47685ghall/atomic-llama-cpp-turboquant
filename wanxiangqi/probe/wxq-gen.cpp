// Greedy generation from a prompt file, for the behavioural collapse check.
//
// llama-cli segfaults before it logs anything in this build, while trie-calib,
// wxq-kl and wxq-manifold all load and run this model fine, so this is the
// short path to an actual generation rather than a debugging detour.
//
// The point is not perplexity. Every number measured so far -- KL, top-1
// agreement, weight rel-err -- is a proxy for "does the model still work", and
// that has never once been checked. The prompts are real agentic-coding contexts
// cut at <|im_start|>assistant, so the model has to produce a turn.

#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    common_params params;
    params.n_batch  = 2048;
    params.n_ubatch = 512;
    params.n_ctx    = 8192;
    params.n_predict = 200;

    std::string prompt_file, out_file;

    std::vector<char *> rest{argv[0]};
    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--gen-prompt") && i + 1 < argc) prompt_file = argv[++i];
        else if (!std::strcmp(argv[i], "--gen-out")    && i + 1 < argc) out_file    = argv[++i];
        else rest.push_back(argv[i]);
    }

    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) return 1;
    if (prompt_file.empty()) { LOG_ERR("usage: %s -m MODEL --gen-prompt FILE [--gen-out FILE] [-n N]\n", argv[0]); return 1; }

    std::string text;
    {
        std::ifstream f(prompt_file);
        if (!f) { LOG_ERR("cannot open %s\n", prompt_file.c_str()); return 1; }
        text.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    common_init_result_ptr ir = common_init_from_params(params);
    if (!ir || !ir->model() || !ir->context()) { LOG_ERR("failed to load model\n"); return 1; }
    llama_context * ctx = ir->context();
    const llama_vocab * vocab = llama_model_get_vocab(ir->model());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> inp = common_tokenize(ctx, text, true, true);
    LOG_INF("%s: prompt %zu tokens, generating %d\n", __func__, inp.size(), params.n_predict);

    llama_batch batch = llama_batch_init((int32_t) params.n_batch, 0, 1);
    llama_pos pos = 0;
    for (size_t off = 0; off < inp.size(); off += (size_t) params.n_batch) {
        const size_t n = std::min((size_t) params.n_batch, inp.size() - off);
        common_batch_clear(batch);
        for (size_t i = 0; i < n; ++i)
            common_batch_add(batch, inp[off + i], pos + (llama_pos) i, { 0 }, off + i + 1 == inp.size());
        if (llama_decode(ctx, batch) != 0) { LOG_ERR("prompt decode failed\n"); return 1; }
        pos += (llama_pos) n;
    }

    std::string gen;
    std::vector<llama_token> hist;
    for (int t = 0; t < params.n_predict; ++t) {
        const float * lg = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        if (!lg) { LOG_ERR("no logits\n"); break; }
        llama_token best = 0;
        float bv = lg[0];
        for (int32_t i = 1; i < n_vocab; ++i) if (lg[i] > bv) { bv = lg[i]; best = i; }
        if (llama_vocab_is_eog(vocab, best)) { gen += "\n[EOG]"; break; }
        gen += common_token_to_piece(ctx, best);
        hist.push_back(best);

        common_batch_clear(batch);
        common_batch_add(batch, best, pos++, { 0 }, true);
        if (llama_decode(ctx, batch) != 0) { LOG_ERR("decode failed at step %d\n", t); break; }
    }

    // cheap collapse signals: 8-gram repetition and distinct-token ratio
    double rep = 0.0;
    if (hist.size() >= 16) {
        std::map<std::string, int> seen;
        int total = 0, dup = 0;
        for (size_t i = 0; i + 8 <= hist.size(); ++i) {
            std::string key;
            for (size_t j = 0; j < 8; ++j) key += std::to_string(hist[i + j]) + ",";
            if (++seen[key] > 1) ++dup;
            ++total;
        }
        rep = total ? (double) dup / total : 0.0;
    }
    std::map<llama_token, int> uniq;
    for (llama_token x : hist) uniq[x]++;
    const double distinct = hist.empty() ? 0.0 : (double) uniq.size() / (double) hist.size();

    LOG_INF("\n%s: 8-gram repeat %.3f   distinct-token ratio %.3f   generated %zu\n",
            __func__, rep, distinct, hist.size());
    printf("=== GENERATION ===\n%s\n=== END ===\n", gen.c_str());

    if (!out_file.empty()) {
        std::ofstream o(out_file, std::ios::trunc);
        o << gen;
    }

    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
