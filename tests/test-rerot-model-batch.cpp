// Opt-in model-backed counterfactual, not a hermetic CTest. Replay identical
// tokens from an identical prefix through one and several independent lanes.
// This tests ordinary inference (RERoT OFF), without autoregressive divergence
// or a different planner/task prompt confounding a batching comparison.
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

struct observer {
    int step = -1;
    int lanes = 1;
    bool reference = true;
    int reference_lane = 0;
    bool distinct = false;
    std::map<std::pair<int, std::string>, std::vector<float>> baseline;
    static bool eval(ggml_tensor * t, bool ask, void * user) {
        auto & o = *static_cast<observer *>(user);
        std::string name = ggml_get_name(t);
        const bool selected = o.step >= 0 && (name.find("l_out-") == 0 || name.find("attn_norm-") == 0 ||
            name.find("linear_attn_out-") == 0 || name.find("attn_pregate-") == 0) &&
            t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[1] == o.lanes && t->ne[2] == 1 && t->ne[3] == 1;
        if (ask) return selected;
        if (!selected) return true;
        std::vector<float> data(size_t(ggml_nelements(t)));
        ggml_backend_tensor_get(t, data.data(), 0, data.size()*sizeof(float));
        auto key = std::make_pair(o.step * 32 + o.reference_lane, name);
        if (o.reference) { o.baseline[key] = std::move(data); return true; }
        auto it = o.baseline.find(std::make_pair(o.step * 32, name));
        if (it == o.baseline.end()) return true;
        const auto & ref = it->second;
        double err = 0, ref2 = 0, err2 = 0, spread = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            const auto & lane_ref = o.distinct
                ? o.baseline.at(std::make_pair(o.step * 32 + int(i / ref.size()), name)) : ref;
            const double d = double(data[i]) - lane_ref[i % ref.size()];
            err = std::max(err, std::abs(d)); err2 += d*d;
            ref2 += double(lane_ref[i % ref.size()])*lane_ref[i % ref.size()];
            spread = std::max(spread, std::abs(double(data[i])-data[i % ref.size()]));
        }
        std::printf("LAYER lanes=%d step=%d name=%s max=%g rel=%g spread=%g\n",
            o.lanes, o.step, name.c_str(), err, std::sqrt(err2/std::max(1e-30,ref2)), spread);
        return true;
    }
};

static void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s MODEL [max_lanes=4] [steps=12] [cache=turbo|f16] [same|distinct]\n", argv[0]); return 2; }
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_batch batch{};
    int status = 0;
    try {
        std::puts("Ordinary model batch diagnostic: exit 0 checks execution/finite values, not task quality or exact batch equivalence.");
        const int max_lanes = argc > 2 ? std::stoi(argv[2]) : 4;
        const int steps = argc > 3 ? std::stoi(argv[3]) : 12;
        require(max_lanes >= 2 && max_lanes <= 16 && steps > 0 && steps <= 128, "invalid shape");
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        model = llama_model_load_from_file(argv[1], mp);
        require(model != nullptr, "model load failed");
        observer obs;
        obs.distinct = argc > 5 && std::strcmp(argv[5], "distinct") == 0;
        auto cp = llama_context_default_params();
        cp.n_ctx = 4096; cp.n_ctx_kv = 4096; cp.kv_unified = true;
        cp.n_batch = 256; cp.n_ubatch = 256; cp.n_seq_max = max_lanes+1; cp.n_seq_recurrent = max_lanes+1;
        cp.n_threads = 4; cp.n_threads_batch = 4;
        cp.type_k = GGML_TYPE_TURBO4_0; cp.type_v = GGML_TYPE_TURBO2_0;
        if (argc > 4 && std::strcmp(argv[4], "f16") == 0) cp.type_k = cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.cb_eval = observer::eval; cp.cb_eval_user_data = &obs;
        ctx = llama_init_from_model(model, cp);
        require(ctx != nullptr, "context load failed");
        auto * vocab = llama_model_get_vocab(model);
        const std::string prompt = "世界上每个大洲有哪些国家";
        const int count = -llama_tokenize(vocab, prompt.data(), int(prompt.size()), nullptr, 0, true, true);
        require(count > 0 && count < 256, "bad prompt size");
        std::vector<llama_token> prefix(count);
        require(llama_tokenize(vocab, prompt.data(), int(prompt.size()), prefix.data(), count, true, true) == count, "tokenize failed");
        batch = llama_batch_init(256, 0, 1);
        auto add = [&](llama_token token, int pos, int seq, bool logits) {
            const int i = batch.n_tokens++;
            batch.token[i] = token; batch.pos[i] = pos;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = seq; batch.logits[i] = logits;
        };
        for (int i = 0; i < count; ++i) add(prefix[i], i, 0, i == count-1);
        require(llama_decode(ctx, batch) == 0, "prefill failed");
        const int nv = llama_vocab_n_tokens(vocab);
        const auto * initial = llama_get_logits_ith(ctx, -1);
        llama_token next = llama_token(std::max_element(initial, initial+nv)-initial);
        std::vector<uint8_t> checkpoint(llama_state_seq_get_size(ctx, 0));
        require(llama_state_seq_get_data(ctx, checkpoint.data(), checkpoint.size(), 0) == checkpoint.size(), "checkpoint failed");
        std::vector<llama_token> tape;
        std::vector<std::vector<float>> expected;
        for (int step = 0; step < steps; ++step) {
            obs.step = step;
            tape.push_back(next);
            batch.n_tokens = 0; add(next, count+step, 0, true);
            require(llama_decode(ctx, batch) == 0, "single decode failed");
            const auto * logits = llama_get_logits_ith(ctx, 0);
            expected.emplace_back(logits, logits+nv);
            next = llama_token(std::max_element(logits, logits+nv)-logits);
        }
        std::vector<std::vector<std::vector<float>>> lane_expected(max_lanes);
        std::vector<std::vector<llama_token>> lane_tape(max_lanes, tape);
        lane_expected[0] = expected;
        for (int seq = 1; seq < max_lanes; ++seq) {
            if (!obs.distinct) { lane_expected[seq] = expected; continue; }
            obs.step = -1; obs.reference_lane = seq;
            llama_memory_clear(llama_get_memory(ctx), true);
            require(llama_state_seq_set_data(ctx, checkpoint.data(), checkpoint.size(), 0) == checkpoint.size(), "reference restore failed");
            for (int step = 0; step < steps; ++step) {
                // Different teacher-forced rows reveal permutations that an
                // identical-lane test cannot detect. No new sampling policy.
                const llama_token token = prefix[(seq + step) % prefix.size()];
                lane_tape[seq][step] = token;
                obs.step = step; batch.n_tokens = 0; add(token, count+step, 0, true);
                require(llama_decode(ctx, batch) == 0, "distinct reference decode failed");
                const float * logits = llama_get_logits_ith(ctx, 0);
                lane_expected[seq].emplace_back(logits, logits+nv);
            }
        }
        obs.reference = false;
        obs.reference_lane = 0;
        for (const int lanes : {1, 2, max_lanes}) {
            obs.step = -1; obs.lanes = lanes;
            auto memory = llama_get_memory(ctx);
            llama_memory_clear(memory, true);
            require(llama_state_seq_set_data(ctx, checkpoint.data(), checkpoint.size(), 0) == checkpoint.size(), "restore failed");
            for (int seq = 1; seq < lanes; ++seq) llama_memory_seq_cp(memory, 0, seq, -1, -1);
            for (int step = 0; step < steps; ++step) {
                obs.step = step; batch.n_tokens = 0;
                for (int seq = 0; seq < lanes; ++seq) add(lane_tape[seq][step], count+step, seq, true);
                require(llama_decode(ctx, batch) == 0, "batch decode failed");
                for (int seq = 0; seq < lanes; ++seq) {
                    const auto * logits = llama_get_logits_ith(ctx, seq);
                    const auto & reference = lane_expected[seq][step];
                    double err = 0, ref2 = 0, err2 = 0;
                    for (int i = 0; i < nv; ++i) {
                        require(std::isfinite(logits[i]), "nonfinite logits");
                        const double delta = double(logits[i])-reference[i];
                        err = std::max(err, std::abs(delta)); err2 += delta*delta;
                        ref2 += double(reference[i])*reference[i];
                    }
                    const int top = int(std::max_element(logits, logits+nv)-logits);
                    const int ref_top = int(std::max_element(reference.begin(), reference.end())-reference.begin());
                    std::printf("LOGITS lanes=%d step=%d seq=%d max=%g rel=%g top=%d ref_top=%d\n", lanes, step, seq, err, std::sqrt(err2/std::max(1e-30,ref2)),top,ref_top);
                }
            }
        }
    } catch (const std::exception & ex) { std::fprintf(stderr, "FAIL: %s\n", ex.what()); status = 1; }
    if (batch.token) llama_batch_free(batch);
    if (ctx) llama_free(ctx);
    if (model) llama_model_free(model);
    llama_backend_free();
    return status;
}
