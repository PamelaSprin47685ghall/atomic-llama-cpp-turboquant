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
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct layer_trace {
    int step = -1;
    int selected_step = -2; // -1 = every decode step, -2 = disabled
    bool attention_only = false;
    int attention_index = 0;
    std::string label;
    std::string dump_dir;
    std::string replace_stage;
    layer_trace * teacher = nullptr;
    int replacements = 0;
    std::map<std::pair<int, std::string>, std::vector<float>> values;
    std::map<std::pair<int, std::string>, ggml_type> types;
    std::vector<std::pair<int, std::string>> order;

    void begin_step(int index) {
        step = index;
        values.clear();
        types.clear();
        order.clear();
        attention_index = 0;
    }

    static bool eval(ggml_tensor * tensor, bool ask, void * user) {
        auto & trace = *static_cast<layer_trace *>(user);
        if (trace.step < 0 || trace.selected_step == -2 ||
            (trace.selected_step >= 0 && trace.step != trace.selected_step)) return false;
        const std::string name = ggml_get_name(tensor);
        const bool attention = tensor->op == GGML_OP_FLASH_ATTN_EXT || tensor->op == GGML_OP_FLASH_ATTN_EXT_REROT;
        const bool selected =
            (attention || (!trace.attention_only &&
            (name.find("attn_norm-") == 0 || name.find("linear_attn_out-") == 0 ||
             name.find("attn_pregate-") == 0 || name.find("l_out-") == 0 ||
             name.find("Qcur_normed-") == 0 || name.find("Qcur-") == 0 ||
             name.find("Kcur_normed-") == 0 || name.find("Kcur-") == 0 || name.find("Vcur-") == 0 ||
             name.find("rerot_q_grouped_raw-") == 0 || name.find("rerot_q_grouped-") == 0 ||
             name.find("rerot_indexed_attn-") == 0 || name.find("kqv_out-") == 0 ||
             name.find("ffn_") == 0 || name.find("moe_") == 0 ||
             name.find("attn_out-") == 0 || name.find("attn_residual-") == 0 ||
             name.find("attn_post_norm-") == 0 || name.find("post_moe-") == 0 || name.find("ffn_inp-") == 0))) &&
            (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 ||
             tensor->type == GGML_TYPE_I32) && ggml_nbytes(tensor) <= 4 * 1024 * 1024;
        if (ask) return selected;
        if (!selected) return true;
        const auto key = std::make_pair(trace.step, name);
        trace.types[key] = tensor->type;
        if (trace.values.count(key) == 0) trace.order.push_back(key);
        auto & dst = trace.values[key];
        dst.resize(size_t(ggml_nelements(tensor)));
        // Q/K/V and grouped queries are often non-contiguous multi-head views.
        // Read the actual byte span, then pack in logical ggml dimension order.
        std::vector<uint8_t> raw(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
        size_t index = 0;
        for (int64_t w = 0; w < tensor->ne[3]; ++w)
            for (int64_t z = 0; z < tensor->ne[2]; ++z)
                for (int64_t y = 0; y < tensor->ne[1]; ++y)
                    for (int64_t x = 0; x < tensor->ne[0]; ++x) {
                        const auto * p = raw.data() + w*tensor->nb[3] + z*tensor->nb[2] +
                            y*tensor->nb[1] + x*tensor->nb[0];
                        if (tensor->type == GGML_TYPE_F32) {
                            std::memcpy(&dst[index], p, sizeof(float));
                        } else if (tensor->type == GGML_TYPE_F16) {
                            ggml_fp16_t v; std::memcpy(&v, p, sizeof(v));
                            dst[index] = ggml_fp16_to_fp32(v);
                        } else {
                            int32_t v; std::memcpy(&v, p, sizeof(v));
                            dst[index] = float(v);
                        }
                        ++index;
                    }
        // Diagnostic-only counterfactual: replace exactly one observed stage
        // with its native teacher value. Never enabled in an ordinary gate.
        // Keep the original value in the trace so the intervention is visible.
        if (!trace.replace_stage.empty() && name == trace.replace_stage) {
            if (!trace.teacher || trace.teacher->values.count(key) == 0 ||
                trace.teacher->types.at(key) != tensor->type ||
                trace.teacher->values.at(key).size() != dst.size()) {
                throw std::runtime_error("counterfactual teacher stage missing or incompatible");
            }
            const auto & ref = trace.teacher->values.at(key);
            index = 0;
            for (int64_t w = 0; w < tensor->ne[3]; ++w)
                for (int64_t z = 0; z < tensor->ne[2]; ++z)
                    for (int64_t y = 0; y < tensor->ne[1]; ++y)
                        for (int64_t x = 0; x < tensor->ne[0]; ++x) {
                            auto * p = raw.data() + w*tensor->nb[3] + z*tensor->nb[2] +
                                y*tensor->nb[1] + x*tensor->nb[0];
                            if (tensor->type == GGML_TYPE_F32) {
                                std::memcpy(p, &ref[index], sizeof(float));
                            } else if (tensor->type == GGML_TYPE_F16) {
                                const ggml_fp16_t v = ggml_fp32_to_fp16(ref[index]);
                                std::memcpy(p, &v, sizeof(v));
                            } else {
                                const int32_t v = int32_t(ref[index]);
                                std::memcpy(p, &v, sizeof(v));
                            }
                            ++index;
                        }
            ggml_backend_tensor_set(tensor, raw.data(), 0, raw.size());
            ++trace.replacements;
            std::printf("COUNTERFACTUAL step=%d name=%s elements=%zu source=native\n", trace.step, name.c_str(), dst.size());
        }
        if (!trace.dump_dir.empty()) {
            const auto dir = std::filesystem::path(trace.dump_dir) /
                (trace.label + "-step-" + std::to_string(trace.step));
            std::filesystem::create_directories(dir);
            std::string safe = name;
            for (char & c : safe) if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9') || c == '-' || c == '_')) c = '_';
            std::ofstream data(dir / (safe + ".f32"), std::ios::binary);
            data.write(reinterpret_cast<const char *>(dst.data()), std::streamsize(dst.size() * sizeof(float)));
            std::ofstream meta(dir / "tensors.tsv", std::ios::app);
            meta << trace.order.size() << '\t' << name << '\t' << safe << ".f32\t" << ggml_type_name(tensor->type);
            for (auto n : tensor->ne) meta << '\t' << n;
            for (auto n : tensor->nb) meta << '\t' << n;
            meta << '\n';
            if (!data || !meta) throw std::runtime_error("tensor dump failed");
            if (attention) {
                // Snapshot actual inputs after the FA dispatch. Selecting only
                // FA outputs leaves preceding RMS/RoPE/cache-write fusions
                // intact, unlike requesting their intermediate tensors.
                const auto snapshot = dir / ("attention-" + std::to_string(trace.attention_index++));
                std::filesystem::create_directories(snapshot);
                std::ofstream inputs(snapshot / "tensors.tsv");
                for (int i = 0; i < 7; ++i) {
                    const auto * src = i == 6 ? tensor : tensor->src[i];
                    if (!src) continue;
                    const std::string id = i == 6 ? "out" : "src" + std::to_string(i);
                    const size_t bytes = ggml_nbytes(src);
                    if (bytes > 128 * 1024 * 1024) throw std::runtime_error("attention snapshot too large");
                    std::vector<uint8_t> storage(bytes);
                    ggml_backend_tensor_get(src, storage.data(), 0, bytes);
                    std::ofstream file(snapshot / (id + ".bin"), std::ios::binary);
                    file.write(reinterpret_cast<const char *>(storage.data()), std::streamsize(bytes));
                    inputs << id << '\t' << ggml_type_name(src->type);
                    for (auto n : src->ne) inputs << '\t' << n;
                    for (auto n : src->nb) inputs << '\t' << n;
                    inputs << '\t' << bytes << '\n';
                    if (!file) throw std::runtime_error("attention input dump failed");
                }
                std::ofstream params(snapshot / "op-params.bin", std::ios::binary);
                params.write(reinterpret_cast<const char *>(tensor->op_params), sizeof(tensor->op_params));
                std::ofstream op(snapshot / "op.txt");
                op << ggml_op_name(tensor->op) << '\n';
                if (!inputs || !params || !op) throw std::runtime_error("attention metadata dump failed");
            }
        }
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

static llama_context * make_context(llama_model * model, bool rerot, bool f16, layer_trace * trace, uint32_t rollback,
                                    bool triattention = false, const char * tri_stats = nullptr, double tri_ratio = 3.0 / 32.0) {
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_ctx_kv = 4096;
    cp.kv_unified = true;
    cp.n_batch = 256;
    cp.n_ubatch = 256;
    cp.n_seq_max = 1;
    cp.n_seq_recurrent = 1;
    cp.n_rs_seq = rollback;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.type_k = f16 ? GGML_TYPE_F16 : GGML_TYPE_TURBO4_0;
    cp.type_v = f16 ? GGML_TYPE_F16 : GGML_TYPE_TURBO2_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.triattention = triattention;
    cp.triattention_stats = tri_stats;
    cp.triattention_ratio = tri_ratio;
    cp.rerot = rerot;
    cp.rerot_frontier = LLAMA_REROT_FRONTIER_STRONG;
    cp.n_person_max = rerot ? 1 : 0;
    cp.n_pen_max = rerot ? 1 : 0;
    cp.cb_eval = trace->selected_step == -2 ? nullptr : layer_trace::eval;
    cp.cb_eval_user_data = trace;
    return llama_init_from_model(model, cp);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL [steps=16] [turbo|f16] [--tape-in FILE] [--tape-out FILE] "
            "[--prompt TEXT] [--trace-step N|--no-trace] [--trace-attn-only] [--dump-dir DIR] [--rollback N] "
            "[--replace-stage NAME] [--report-only] [--triattention] [--triattention-stats PATH] [--triattention-ratio RATIO]\n"
            "Decision equivalence is enforced by default; --report-only explicitly permits mismatches.\n", argv[0]);
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
        require(argc <= 3 || f16 || std::strcmp(argv[3], "turbo") == 0, "invalid cache type");
        require(steps > 0 && steps <= 1024, "invalid step count");
        std::string tape_in, tape_out, dump_dir, replace_stage;
        std::string prompt = "世界上每个大洲有哪些国家";
        // Tracing can split backend fusion groups. Keep the numerical gate
        // uninstrumented by default; tracing is a separate, explicit replay.
        int trace_step = -2;
        bool attention_only = false;
        bool strict = true;
        int rollback = 0;
        bool triattention = false;
        std::string tri_stats;
        double tri_ratio = 3.0 / 32.0;
        for (int i = 4; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--strict") { strict = true; continue; }
            if (arg == "--report-only") { strict = false; continue; }
            if (arg == "--no-trace") { trace_step = -2; continue; }
            if (arg == "--trace-attn-only") { attention_only = true; continue; }
            if (arg == "--triattention") { triattention = true; continue; }
            require(i + 1 < argc, "missing option value");
            const std::string value = argv[++i];
            if (arg == "--tape-in") tape_in = value;
            else if (arg == "--tape-out") tape_out = value;
            else if (arg == "--prompt") prompt = value;
            else if (arg == "--trace-step") trace_step = std::stoi(value);
            else if (arg == "--rollback") rollback = std::stoi(value);
            else if (arg == "--replace-stage") replace_stage = value;
            else if (arg == "--dump-dir") dump_dir = value;
            else if (arg == "--triattention-stats") { tri_stats = value; triattention = true; }
            else if (arg == "--triattention-ratio") { tri_ratio = std::stod(value); triattention = true; }
            else throw std::runtime_error("unknown option: " + arg);
        }
        require(trace_step >= -2 && trace_step < steps, "invalid trace step");
        require(rollback >= 0 && rollback <= 8, "invalid rollback capacity");
        require(replace_stage.empty() || (trace_step >= 0 && !attention_only),
            "counterfactual requires a single --trace-step and full tracing");
        require(tape_in.empty() || tape_in != tape_out, "input tape must not be overwritten");
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        model = llama_model_load_from_file(argv[1], mp);
        require(model != nullptr, "model load failed");

        layer_trace native_trace, rerot_trace;
        native_trace.label = "native";
        rerot_trace.label = "rerot";
        rerot_trace.teacher = &native_trace;
        rerot_trace.replace_stage = replace_stage;
        native_trace.selected_step = rerot_trace.selected_step = trace_step;
        native_trace.attention_only = rerot_trace.attention_only = attention_only;
        native_trace.dump_dir = rerot_trace.dump_dir = dump_dir;
        const char * tri_stats_ptr = tri_stats.empty() ? nullptr : tri_stats.c_str();
        native = make_context(model, false, f16, &native_trace, uint32_t(rollback), triattention, tri_stats_ptr, tri_ratio);
        rerot = make_context(model, true, f16, &rerot_trace, uint32_t(rollback), triattention, tri_stats_ptr, tri_ratio);
        require(native && rerot, "context load failed");
        auto * vocab = llama_model_get_vocab(model);
        const int n_prompt = -llama_tokenize(vocab, prompt.data(), int(prompt.size()), nullptr, 0, true, true);
        require(n_prompt > 0 && n_prompt < 256, "invalid prompt size");
        std::vector<llama_token> tokens(n_prompt);
        require(llama_tokenize(vocab, prompt.data(), int(prompt.size()), tokens.data(), n_prompt, true, true) == n_prompt,
                "tokenization failed");
        std::vector<llama_token> tape;
        if (!tape_in.empty()) {
            std::ifstream file(tape_in);
            std::string magic;
            size_t prompt_count = 0, count = 0;
            file >> magic >> prompt_count >> count;
            require(bool(file) && magic == "REROT_TOKEN_TAPE_V1" && prompt_count == tokens.size() &&
                count >= size_t(steps) && count <= 1024, "invalid token tape header");
            for (llama_token expected : tokens) {
                llama_token value = -1; file >> value;
                require(bool(file) && value == expected, "token tape prompt mismatch");
            }
            tape.resize(count);
            for (auto & value : tape) {
                file >> value;
                require(bool(file) && value >= 0 && value < llama_vocab_n_tokens(vocab), "invalid tape token");
            }
            std::string extra;
            require(!(file >> extra), "unexpected trailing token tape data");
        }
        std::vector<llama_token> input_tape;
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
            require(std::isfinite(native_prefill[i]) && std::isfinite(rerot_prefill[i]), "nonfinite prefill logits");
            const double delta = double(rerot_prefill[i]) - native_prefill[i];
            prefill_max = std::max(prefill_max, std::abs(delta));
            prefill_err2 += delta * delta;
            prefill_ref2 += double(native_prefill[i]) * native_prefill[i];
        }
        std::printf("PREFILL max=%g rel=%g native_top=%td rerot_top=%td\n", prefill_max,
            std::sqrt(prefill_err2 / std::max(1e-30, prefill_ref2)),
            std::max_element(native_prefill, native_prefill + n_vocab) - native_prefill,
            std::max_element(rerot_prefill, rerot_prefill + n_vocab) - rerot_prefill);
        // Both cold contexts execute the same prefill graph. A difference
        // here must not be hidden by later matching decode argmaxes.
        // In grouped layout, the recurrent state path has an additional B + H
        // addition where H=0. In FP32 across 40 layers, rounding differences
        // up to ~1e-4 can occur without changing top tokens.
        const bool prefill_equal = prefill_max < 1e-4;

        llama_rerot_episode_params ep{};
        ep.frontier_mode = LLAMA_REROT_FRONTIER_STRONG;
        ep.topology_epoch = ep.publish_epoch = ep.layout_epoch = 1;
        require(llama_rerot_episode_begin(rerot, 1, &ep), "episode begin failed");
        llama_token next = llama_token(std::max_element(native_prefill, native_prefill + n_vocab) - native_prefill);
        const uint32_t run_id = 1;

        double global_logits_max = 0.0, global_logits_rel = 0.0;
        int mismatches = 0, first_top_divergence = -1;
        for (int step = 0; step < steps; ++step) {
            const llama_pos pos = n_prompt + step;
            native_trace.begin_step(step);
            rerot_trace.begin_step(step);
            if (!tape.empty()) next = tape[step];
            input_tape.push_back(next);
            std::printf("TAPE step=%d token=%d\n", step, next);

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
                if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) ++nonfinite;
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
            require(nonfinite == 0 && std::isfinite(rel), "nonfinite logits");
            if (top != ref_top) {
                ++mismatches;
                if (first_top_divergence < 0) first_top_divergence = step;
            }
            float runner_up = -INFINITY;
            for (int i = 0; i < n_vocab; ++i) if (i != ref_top) runner_up = std::max(runner_up, expected[i]);
            std::printf("LOGITS step=%d max=%g rel=%g top=%d ref_top=%d nonfinite=%d\n",
                step, max_abs, rel, top, ref_top, nonfinite);
            std::printf("DECISION step=%d native_margin=%g native_gap=%g rerot_gap=%g\n", step,
                expected[ref_top] - runner_up, expected[ref_top] - expected[top], actual[ref_top] - actual[top]);

            double first_layer_rel = 0.0;
            std::string first_layer;
            // Follow actual callback execution order, NOT std::map's name order.
            for (const auto & key : native_trace.order) {
                const auto & ref = native_trace.values.at(key);
                if (key.first != step) continue;
                const auto found = rerot_trace.values.find(key);
                if (found == rerot_trace.values.end() || found->second.size() != ref.size()) continue;
                if (native_trace.types.at(key) == GGML_TYPE_I32) {
                    size_t changed = 0;
                    for (size_t i = 0; i < ref.size(); ++i) changed += ref[i] != found->second[i];
                    std::printf("INDEX_DIFF step=%d name=%s changed=%zu total=%zu\n", step, key.second.c_str(), changed, ref.size());
                    if (key.second.find("ffn_moe_topk-") == 0) {
                        auto lhs = ref, rhs = found->second;
                        std::sort(lhs.begin(), lhs.end());
                        std::sort(rhs.begin(), rhs.end());
                        std::printf("ROUTING step=%d name=%s order_changed=%d membership_changed=%d\n",
                            step, key.second.c_str(), changed != 0, lhs != rhs);
                    }
                    // Integer expert IDs are labels, not floating-point
                    // magnitudes. A tail argsort permutation is not the first
                    // activation error and may not change selected experts.
                    continue;
                }
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
                std::printf("FIRST_ABOVE_TOLERANCE step=%d name=%s rel=%g tolerance=1e-5 order=execution\n",
                    step, first_layer.c_str(), first_layer_rel);
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
            std::fflush(stdout);
        }
        if (!tape_out.empty()) {
            std::ofstream file(tape_out);
            file << "REROT_TOKEN_TAPE_V1\n" << tokens.size() << ' ' << input_tape.size() << '\n';
            for (auto token : tokens) file << token << ' ';
            file << '\n';
            for (auto token : input_tape) file << token << '\n';
            file.flush();
            require(bool(file), "token tape write failed");
        }
        std::printf("SUMMARY cache=%s steps=%d logits_max=%g logits_rel_max=%g mismatches=%d first_top_divergence=%d\n",
            f16 ? "f16" : "turbo", steps, global_logits_max, global_logits_rel, mismatches, first_top_divergence);
        llama_rerot_episode_end(rerot, 1);
        require(replace_stage.empty() || rerot_trace.replacements == 1, "counterfactual stage must execute exactly once");
        if (strict && (!prefill_equal || mismatches != 0)) status = 1;
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
