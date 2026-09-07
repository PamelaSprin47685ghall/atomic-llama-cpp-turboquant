// Opt-in real-model equivalence diagnostic; no model is downloaded and this
// is not a hermetic CTest. A one-Lane RERoT episode has no peer sharing, so
// with identical teacher-forced tokens it must reduce to ordinary inference.
// Any early divergence is implementation/numerical, not "task semantics".
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct layer_trace {
    int step = -1;
    std::map<std::pair<int, std::string>, std::vector<float>> values;

    static bool eval(ggml_tensor * tensor, bool ask, void * user) {
        auto & trace = *static_cast<layer_trace *>(user);
        if (trace.step < 0) return false;
        const std::string name = ggml_get_name(tensor);
        const bool selected =
            (name.find("attn_norm-") == 0 || name.find("linear_attn_out-") == 0 ||
             name.find("attn_pregate-") == 0 || name.find("l_out-") == 0 ||
             name.find("Qcur_normed-") == 0 || name.find("Qcur-") == 0 ||
             name.find("Kcur_normed-") == 0 || name.find("Vcur-") == 0 ||
             name.find("rerot_q_grouped_raw-") == 0 || name.find("rerot_q_grouped-") == 0 ||
             name.find("rerot_indexed_attn-") == 0 || name.find("kqv_out-") == 0) &&
            tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
            tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
            ggml_nbytes(tensor) <= 1024 * 1024;
        if (ask) return selected;
        if (!selected) return true;
        auto & dst = trace.values[{trace.step, name}];
        dst.resize(size_t(ggml_nelements(tensor)));
        ggml_backend_tensor_get(tensor, dst.data(), 0, dst.size() * sizeof(float));
        return true;
    }
};

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

static llama_context * make_context(llama_model * model, bool rerot, bool f16, layer_trace * trace) {
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_ctx_kv = 4096;
    cp.kv_unified = true;
    cp.n_batch = 256;
    cp.n_ubatch = 256;
    cp.n_seq_max = 1;
    cp.n_seq_recurrent = 1;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.type_k = f16 ? GGML_TYPE_F16 : GGML_TYPE_TURBO4_0;
    cp.type_v = f16 ? GGML_TYPE_F16 : GGML_TYPE_TURBO2_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.rerot = rerot;
    cp.rerot_frontier = LLAMA_REROT_FRONTIER_STRONG;
    cp.n_person_max = rerot ? 1 : 0;
    cp.n_pen_max = rerot ? 1 : 0;
    cp.cb_eval = layer_trace::eval;
    cp.cb_eval_user_data = trace;
    return llama_init_from_model(model, cp);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL [steps=16] [turbo|f16]\n", argv[0]);
        return 2;
    }
    int status = 0;
    llama_model * model = nullptr;
    llama_context * native = nullptr;
    llama_context * rerot = nullptr;
    llama_batch native_batch{};
    llama_batch rerot_batch{};
    try {
        const int steps = argc > 2 ? std::stoi(argv[2]) : 16;
        const bool f16 = argc > 3 && std::strcmp(argv[3], "f16") == 0;
        require(steps > 0 && steps <= 128, "invalid step count");
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        model = llama_model_load_from_file(argv[1], mp);
        require(model != nullptr, "model load failed");

        layer_trace native_trace, rerot_trace;
        native = make_context(model, false, f16, &native_trace);
        rerot = make_context(model, true, f16, &rerot_trace);
        require(native && rerot, "context load failed");
        auto * vocab = llama_model_get_vocab(model);
        const std::string prompt = "世界上每个大洲有哪些国家";
        const int n_prompt = -llama_tokenize(vocab, prompt.data(), int(prompt.size()), nullptr, 0, true, true);
        require(n_prompt > 0 && n_prompt < 256, "invalid prompt size");
        std::vector<llama_token> tokens(n_prompt);
        require(llama_tokenize(vocab, prompt.data(), int(prompt.size()), tokens.data(), n_prompt, true, true) == n_prompt,
                "tokenization failed");
        native_batch = llama_batch_init(256, 0, 1);
        rerot_batch = llama_batch_init(256, 0, 1);
        for (int i = 0; i < n_prompt; ++i) {
            add_token(native_batch, tokens[i], i, i == n_prompt - 1);
            add_token(rerot_batch, tokens[i], i, i == n_prompt - 1);
        }
        require(llama_decode(native, native_batch) == 0, "native prefill failed");
        require(llama_decode(rerot, rerot_batch) == 0, "rerot-cold prefill failed");

        const int n_vocab = llama_vocab_n_tokens(vocab);
        const float * native_prefill = llama_get_logits_ith(native, -1);
        const float * rerot_prefill = llama_get_logits_ith(rerot, -1);
        double prefill_max = 0.0, prefill_ref2 = 0.0, prefill_err2 = 0.0;
        for (int i = 0; i < n_vocab; ++i) {
            const double delta = double(rerot_prefill[i]) - native_prefill[i];
            prefill_max = std::max(prefill_max, std::abs(delta));
            prefill_err2 += delta * delta;
            prefill_ref2 += double(native_prefill[i]) * native_prefill[i];
        }
        std::printf("PREFILL max=%g rel=%g native_top=%td rerot_top=%td\n", prefill_max,
            std::sqrt(prefill_err2 / std::max(1e-30, prefill_ref2)),
            std::max_element(native_prefill, native_prefill + n_vocab) - native_prefill,
            std::max_element(rerot_prefill, rerot_prefill + n_vocab) - rerot_prefill);

        llama_rerot_episode_params ep{};
        ep.frontier_mode = LLAMA_REROT_FRONTIER_STRONG;
        ep.topology_epoch = ep.publish_epoch = ep.layout_epoch = 1;
        require(llama_rerot_episode_begin(rerot, 1, &ep), "episode begin failed");
        llama_token next = llama_token(std::max_element(native_prefill, native_prefill + n_vocab) - native_prefill);
        const uint32_t run_id = 1;

        double global_logits_max = 0.0, global_logits_rel = 0.0;
        for (int step = 0; step < steps; ++step) {
            const llama_pos pos = n_prompt + step;
            native_trace.step = step;
            rerot_trace.step = step;

            native_batch.n_tokens = 0;
            add_token(native_batch, next, pos, true);
            require(llama_decode(native, native_batch) == 0, "native decode failed");

            llama_rerot_write_tag tag{};
            tag.episode_id = 1;
            tag.node_id = 1;
            tag.run_id = run_id;
            tag.publish_epoch = uint64_t(step + 1);
            tag.frontier = uint64_t(step + 1);
            tag.visibility = LLAMA_REROT_KV_PUBLIC_LIVE;
            require(llama_rerot_set_write_tag(rerot, 0, &tag), "write tag failed");
            llama_rerot_frontier_reader_view view{};
            view.seq_id = 0;
            view.episode_id = 1;
            view.reader_node_id = 1;
            view.query_run_id = run_id;
            view.frontier = tag.frontier;
            view.frontier_mode = LLAMA_REROT_FRONTIER_STRONG;
            view.stamp = {1, tag.publish_epoch, 1};
            view.ordered_run_ids = &run_id;
            view.n_ordered_runs = 1;
            require(llama_rerot_set_frontier_views(rerot, &view, 1), "reader view failed");
            rerot_batch.n_tokens = 0;
            add_token(rerot_batch, next, pos, true);
            require(llama_decode(rerot, rerot_batch) == 0, "rerot decode failed");

            const float * expected = llama_get_logits_ith(native, 0);
            const float * actual = llama_get_logits_ith(rerot, 0);
            double max_abs = 0.0, ref2 = 0.0, err2 = 0.0;
            int nonfinite = 0;
            for (int i = 0; i < n_vocab; ++i) {
                if (!std::isfinite(actual[i])) ++nonfinite;
                const double delta = double(actual[i]) - expected[i];
                max_abs = std::max(max_abs, std::abs(delta));
                err2 += delta * delta;
                ref2 += double(expected[i]) * expected[i];
            }
            const double rel = std::sqrt(err2 / std::max(1e-30, ref2));
            global_logits_max = std::max(global_logits_max, max_abs);
            global_logits_rel = std::max(global_logits_rel, rel);
            const int ref_top = int(std::max_element(expected, expected + n_vocab) - expected);
            const int top = int(std::max_element(actual, actual + n_vocab) - actual);
            std::printf("LOGITS step=%d max=%g rel=%g top=%d ref_top=%d nonfinite=%d\n",
                step, max_abs, rel, top, ref_top, nonfinite);

            double first_layer_rel = 0.0;
            std::string first_layer;
            for (const auto & [key, ref] : native_trace.values) {
                if (key.first != step) continue;
                const auto found = rerot_trace.values.find(key);
                if (found == rerot_trace.values.end() || found->second.size() != ref.size()) continue;
                double lref2 = 0.0, lerr2 = 0.0, lmax = 0.0;
                for (size_t i = 0; i < ref.size(); ++i) {
                    const double delta = double(found->second[i]) - ref[i];
                    lerr2 += delta * delta;
                    lref2 += double(ref[i]) * ref[i];
                    lmax = std::max(lmax, std::abs(delta));
                }
                const double lrel = std::sqrt(lerr2 / std::max(1e-30, lref2));
                std::printf("LAYER step=%d name=%s max=%g rel=%g\n", step, key.second.c_str(), lmax, lrel);
                if (first_layer.empty() && lrel > 1e-5) {
                    first_layer = key.second;
                    first_layer_rel = lrel;
                }
            }
            if (!first_layer.empty()) {
                std::printf("FIRST_DIVERGENCE step=%d name=%s rel=%g\n", step, first_layer.c_str(), first_layer_rel);
            }
            if (step == 0) {
                const auto report_cross = [&](const char * lhs_name, const char * rhs_name) {
                    const auto lhs = native_trace.values.find({step, lhs_name});
                    const auto rhs = rerot_trace.values.find({step, rhs_name});
                    if (lhs == native_trace.values.end() || rhs == rerot_trace.values.end() ||
                        lhs->second.size() != rhs->second.size()) return;
                    double cmax = 0.0, cref2 = 0.0, cerr2 = 0.0;
                    for (size_t i = 0; i < lhs->second.size(); ++i) {
                        const double delta = double(rhs->second[i]) - lhs->second[i];
                        cmax = std::max(cmax, std::abs(delta));
                        cerr2 += delta * delta;
                        cref2 += double(lhs->second[i]) * lhs->second[i];
                    }
                    std::printf("CROSS step=0 native=%s rerot=%s max=%g rel=%g\n",
                        lhs_name, rhs_name, cmax, std::sqrt(cerr2 / std::max(1e-30, cref2)));
                };
                report_cross("Qcur_normed-3", "rerot_q_grouped_raw-3");
                report_cross("Qcur-3", "rerot_q_grouped-3");
                report_cross("attn_pregate-3", "attn_pregate-3");
            }
            next = llama_token(ref_top); // teacher-force the native trajectory
        }
        std::printf("SUMMARY cache=%s steps=%d logits_max=%g logits_rel_max=%g\n",
            f16 ? "f16" : "turbo", steps, global_logits_max, global_logits_rel);
        llama_rerot_episode_end(rerot, 1);
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "FAIL: %s\n", ex.what());
        status = 1;
    }
    if (native_batch.token) llama_batch_free(native_batch);
    if (rerot_batch.token) llama_batch_free(rerot_batch);
    if (native) llama_free(native);
    if (rerot) llama_free(rerot);
    if (model) llama_model_free(model);
    llama_backend_free();
    return status;
}
