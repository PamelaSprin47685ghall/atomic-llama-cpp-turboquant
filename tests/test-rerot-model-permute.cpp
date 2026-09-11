// Opt-in real-model permutation invariant. Two logical PUBLIC lanes execute
// the same fixed frontier in opposite physical batch-row orders. DDVR views
// are keyed by logical node/run and RBB is order-free, so mapped outputs must
// agree. A violation is below task semantics.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct perm_trace {
    int step = -1;
    std::array<int, 2> logical_by_row{0, 1};
    std::map<std::pair<int, std::string>, std::vector<float>> values;

    static bool eval(ggml_tensor * tensor, bool ask, void * user) {
        auto & trace = *static_cast<perm_trace *>(user);
        if (trace.step < 0) return false;
        const std::string name = ggml_get_name(tensor);
        const bool selected =
            (name.find("attn_norm-") == 0 || name.find("linear_attn_out-") == 0 ||
             name.find("attn_pregate-") == 0 || name.find("l_out-") == 0) &&
            tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
            tensor->ne[1] == 2 && tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
            ggml_nbytes(tensor) <= 1024 * 1024;
        if (ask) return selected;
        if (!selected) return true;
        const size_t width = size_t(tensor->ne[0]);
        std::vector<float> raw(size_t(ggml_nelements(tensor)));
        ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size() * sizeof(float));
        auto & dst = trace.values[{trace.step, name}];
        dst.resize(raw.size());
        for (int row = 0; row < 2; ++row) {
            const int logical = trace.logical_by_row[row];
            std::copy_n(raw.data() + size_t(row) * width, width,
                        dst.data() + size_t(logical) * width);
        }
        return true;
    }
};

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static llama_context * make_context(llama_model * model, bool f16, perm_trace * trace) {
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_ctx_kv = 4096;
    cp.kv_unified = true;
    cp.n_batch = 256;
    cp.n_ubatch = 256;
    cp.n_seq_max = 2;
    // The shared prefix row remains a COW source while both children split
    // on their first write, so the diagnostic needs source + 2 destinations.
    cp.n_seq_recurrent = 3;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.type_k = f16 ? GGML_TYPE_F16 : GGML_TYPE_TURBO4_0;
    cp.type_v = f16 ? GGML_TYPE_F16 : GGML_TYPE_TURBO2_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.rerot = true;
    cp.rerot_frontier = LLAMA_REROT_FRONTIER_STRONG;
    cp.n_person_max = 1;
    cp.n_pen_max = 3;
    cp.cb_eval = perm_trace::eval;
    cp.cb_eval_user_data = trace;
    return llama_init_from_model(model, cp);
}

static void add(llama_batch & batch, llama_token token, llama_pos pos, llama_seq_id seq) {
    const int i = batch.n_tokens++;
    batch.token[i] = token;
    batch.pos[i] = pos;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = seq;
    batch.logits[i] = true;
}

static void begin_episode(llama_context * ctx) {
    llama_rerot_episode_params ep{};
    ep.frontier_mode = LLAMA_REROT_FRONTIER_STRONG;
    ep.topology_epoch = ep.publish_epoch = ep.layout_epoch = 1;
    require(llama_rerot_episode_begin(ctx, 1, &ep), "episode begin failed");
}

static void install_frontier(llama_context * ctx, uint64_t frontier) {
    const uint32_t order0[2] = {2, 1};
    const uint32_t order1[2] = {1, 2};
    for (int logical = 0; logical < 2; ++logical) {
        llama_rerot_write_tag tag{};
        tag.episode_id = 1;
        tag.node_id = uint32_t(logical + 1);
        tag.run_id = uint32_t(logical + 1);
        tag.publish_epoch = frontier;
        tag.frontier = frontier;
        tag.visibility = LLAMA_REROT_KV_PUBLIC_LIVE;
        require(llama_rerot_set_write_tag(ctx, logical, &tag), "write tag failed");
    }
    llama_rerot_frontier_reader_view views[2]{};
    for (int logical = 0; logical < 2; ++logical) {
        views[logical].seq_id = logical;
        views[logical].episode_id = 1;
        views[logical].reader_node_id = uint32_t(logical + 1);
        views[logical].query_run_id = uint32_t(logical + 1);
        views[logical].frontier = frontier;
        views[logical].frontier_mode = LLAMA_REROT_FRONTIER_STRONG;
        views[logical].stamp = {1, frontier, 1};
        views[logical].ordered_run_ids = logical == 0 ? order0 : order1;
        views[logical].n_ordered_runs = 2;
    }
    require(llama_rerot_set_frontier_views(ctx, views, 2), "frontier views failed");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL [steps=8] [turbo|f16]\n", argv[0]);
        return 2;
    }
    int status = 0;
    llama_model * model = nullptr;
    llama_context * forward = nullptr;
    llama_context * reverse = nullptr;
    llama_batch bf{}, br{};
    try {
        const int steps = argc > 2 ? std::stoi(argv[2]) : 8;
        const bool f16 = argc > 3 && std::strcmp(argv[3], "f16") == 0;
        require(steps > 0 && steps <= 64, "invalid steps");
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        model = llama_model_load_from_file(argv[1], mp);
        require(model != nullptr, "model load failed");
        perm_trace tf, tr;
        forward = make_context(model, f16, &tf);
        reverse = make_context(model, f16, &tr);
        require(forward && reverse, "context load failed");
        auto * vocab = llama_model_get_vocab(model);
        const std::string prompt = "世界上每个大洲有哪些国家";
        const int n_prompt = -llama_tokenize(vocab, prompt.data(), int(prompt.size()), nullptr, 0, true, true);
        require(n_prompt > 4 && n_prompt < 256, "bad prompt size");
        std::vector<llama_token> tokens(n_prompt);
        require(llama_tokenize(vocab, prompt.data(), int(prompt.size()), tokens.data(), n_prompt, true, true) == n_prompt,
                "tokenization failed");
        bf = llama_batch_init(256, 0, 2);
        br = llama_batch_init(256, 0, 2);
        for (int i = 0; i < n_prompt; ++i) {
            add(bf, tokens[i], i, 0);
            add(br, tokens[i], i, 0);
        }
        require(llama_decode(forward, bf) == 0, "forward prefill failed");
        require(llama_decode(reverse, br) == 0, "reverse prefill failed");
        llama_memory_seq_cp(llama_get_memory(forward), 0, 1, -1, -1);
        llama_memory_seq_cp(llama_get_memory(reverse), 0, 1, -1, -1);
        begin_episode(forward);
        begin_episode(reverse);

        const int n_vocab = llama_vocab_n_tokens(vocab);
        double global_rel = 0.0, global_max = 0.0;
        for (int step = 0; step < steps; ++step) {
            const llama_token ta = tokens[(2 * step + 1) % n_prompt];
            const llama_token tb = tokens[(2 * step + 4) % n_prompt];
            const llama_pos pos = n_prompt + step;
            install_frontier(forward, uint64_t(step + 1));
            install_frontier(reverse, uint64_t(step + 1));
            tf.step = tr.step = step;
            tf.logical_by_row = {0, 1};
            tr.logical_by_row = {1, 0};
            bf.n_tokens = 0;
            br.n_tokens = 0;
            add(bf, ta, pos, 0);
            add(bf, tb, pos, 1);
            add(br, tb, pos, 1);
            add(br, ta, pos, 0);
            require(llama_decode(forward, bf) == 0, "forward frontier failed");
            require(llama_decode(reverse, br) == 0, "reverse frontier failed");

            for (int logical = 0; logical < 2; ++logical) {
                const int row_f = logical;
                const int row_r = logical == 0 ? 1 : 0;
                const float * a = llama_get_logits_ith(forward, row_f);
                const float * b = llama_get_logits_ith(reverse, row_r);
                double ref2 = 0.0, err2 = 0.0, max_abs = 0.0;
                int nonfinite = 0;
                for (int i = 0; i < n_vocab; ++i) {
                    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) ++nonfinite;
                    const double delta = double(b[i]) - a[i];
                    err2 += delta * delta;
                    ref2 += double(a[i]) * a[i];
                    max_abs = std::max(max_abs, std::abs(delta));
                }
                const double rel = std::sqrt(err2 / std::max(1e-30, ref2));
                global_rel = std::max(global_rel, rel);
                global_max = std::max(global_max, max_abs);
                const int top_a = int(std::max_element(a, a + n_vocab) - a);
                const int top_b = int(std::max_element(b, b + n_vocab) - b);
                std::printf("LOGITS step=%d logical=%d max=%g rel=%g top=%d perm_top=%d nonfinite=%d\n",
                    step, logical, max_abs, rel, top_a, top_b, nonfinite);
            }
            double first_rel = 0.0;
            std::string first_name;
            for (const auto & [key, lhs] : tf.values) {
                if (key.first != step) continue;
                const auto rhs = tr.values.find(key);
                if (rhs == tr.values.end() || rhs->second.size() != lhs.size()) continue;
                double ref2 = 0.0, err2 = 0.0, max_abs = 0.0;
                for (size_t i = 0; i < lhs.size(); ++i) {
                    const double delta = double(rhs->second[i]) - lhs[i];
                    err2 += delta * delta;
                    ref2 += double(lhs[i]) * lhs[i];
                    max_abs = std::max(max_abs, std::abs(delta));
                }
                const double rel = std::sqrt(err2 / std::max(1e-30, ref2));
                if (rel > 1e-6) std::printf("LAYER step=%d name=%s max=%g rel=%g\n", step, key.second.c_str(), max_abs, rel);
                if (first_name.empty() && rel > 1e-5) {
                    first_name = key.second;
                    first_rel = rel;
                }
            }
            if (!first_name.empty()) std::printf("FIRST_DIVERGENCE step=%d name=%s rel=%g\n", step, first_name.c_str(), first_rel);
        }
        std::printf("SUMMARY cache=%s steps=%d logits_max=%g logits_rel_max=%g\n",
            f16 ? "f16" : "turbo", steps, global_max, global_rel);
        llama_rerot_episode_end(forward, 1);
        llama_rerot_episode_end(reverse, 1);
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "FAIL: %s\n", ex.what());
        status = 1;
    }
    if (bf.token) llama_batch_free(bf);
    if (br.token) llama_batch_free(br);
    if (forward) llama_free(forward);
    if (reverse) llama_free(reverse);
    if (model) llama_model_free(model);
    llama_backend_free();
    return status;
}
