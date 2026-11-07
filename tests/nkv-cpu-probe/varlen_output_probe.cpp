// Paired CPU decode probe. Contexts share weights, not KV state.
// The fixed gate checks actual output-tail rows without an evaluation callback.
// Variable token width remains a separate diagnostic.
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr int steps = 200;
constexpr int width_max = 4;

struct log_capture {
    ggml_log_callback previous = nullptr;
    void * previous_data = nullptr;
    bool recording = false;
    std::string text;
};

void capture_log(ggml_log_level level, const char * text, void * data) {
    auto & log = *static_cast<log_capture *>(data);
    if (log.recording) log.text += text;
    if (log.previous) log.previous(level, text, log.previous_data);
}

bool stat_value(const std::string & line, const char * name, size_t & value) {
    const std::string key = std::string(" ") + name + "=";
    const size_t at = line.find(key);
    if (at == std::string::npos || line.find(key, at + key.size()) != std::string::npos) return false;
    const char * begin = line.c_str() + at + key.size();
    char * end = nullptr;
    const unsigned long long n = std::strtoull(begin, &end, 10);
    if (end == begin || (*end != ' ' && *end != '\n' && *end != '\0')) return false;
    value = size_t(n);
    return static_cast<unsigned long long>(value) == n;
}

bool check_binder_stats(const std::string & text, int transitions, int growths) {
    const std::string marker = "NKV_BINDER_STATS ";
    const size_t at = text.find(marker);
    if (at == std::string::npos || text.find(marker, at + marker.size()) != std::string::npos) return false;
    const size_t end = text.find('\n', at);
    const std::string line = text.substr(at, end == std::string::npos ? end : end - at);
    size_t builds, env_builds, analyses, shape_fail, patches, cap, disabled, exact, grow, grow_rebuilds;
    if (!stat_value(line, "graph_builds", builds) || !stat_value(line, "env_builds", env_builds) ||
        !stat_value(line, "shape_analyses", analyses) || !stat_value(line, "shape_fail", shape_fail) ||
        !stat_value(line, "patch_binds", patches) || !stat_value(line, "cap_nout", cap) ||
        !stat_value(line, "env_disabled", disabled) || !stat_value(line, "exact", exact) ||
        !stat_value(line, "nout_grow_binds", grow) || !stat_value(line, "nout_grow_rebuilds", grow_rebuilds)) return false;
    return builds == 1 && env_builds == 1 && analyses == 1 && shape_fail == 0 &&
           patches >= size_t(transitions) && cap == width_max && disabled == 0 && exact == 1 &&
           grow >= size_t(growths) && grow_rebuilds == 0;
}

struct kqv_capture {
    bool seen = false;
    bool valid = false;
    int64_t shape[4] = {};
    std::vector<float> values;
};

bool capture_kqv(ggml_tensor * t, bool ask, void * user_data) {
    const bool wanted = std::strcmp(ggml_get_name(t), "kqv-0") == 0;
    if (ask) return wanted;
    if (!wanted) return true;
    auto & out = *static_cast<kqv_capture *>(user_data);
    out.seen = true;
    out.valid = t->type == GGML_TYPE_F32 && t->src[0] && t->src[0]->type == GGML_TYPE_F16 &&
                t->nb[0] == sizeof(float) && t->ne[0] > 0;
    if (!out.valid) return true;
    size_t count = 1;
    for (int i = 0; i < 4; ++i) {
        out.shape[i] = t->ne[i];
        count *= size_t(t->ne[i]);
    }
    out.values.resize(count);
    size_t offset = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                ggml_backend_tensor_get(t, out.values.data() + offset,
                        i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3], t->ne[0]*sizeof(float));
                offset += size_t(t->ne[0]);
            }
        }
    }
    return true;
}

bool compare_kqv(const kqv_capture & a, const kqv_capture & b, int step) {
    if (!a.seen || !b.seen || !a.valid || !b.valid ||
            std::memcmp(a.shape, b.shape, sizeof(a.shape)) != 0 || a.values.size() != b.values.size()) {
        std::fprintf(stderr, "KQV_CAPTURE_FAIL step=%d seen=%d/%d valid=%d/%d\n",
                     step, int(a.seen), int(b.seen), int(a.valid), int(b.valid));
        return false;
    }
    const size_t width = size_t(a.shape[0]);
    for (size_t pos = 0; pos < a.values.size(); pos += width) {
        if (std::memcmp(a.values.data() + pos, b.values.data() + pos, width*sizeof(float)) != 0) {
            std::fprintf(stderr, "KQV_ROW_MISMATCH step=%d row=%zu\n", step, pos / width);
            return false;
        }
    }
    return true;
}

struct tail_tensor {
    int visits = 0;
    int64_t width = 0;
    int64_t rows = 0;
    bool valid = false;
    std::vector<float> values;
};

struct tail_capture {
    tail_tensor norm;
    tail_tensor logits;
};

bool capture_tail(ggml_tensor * t, bool ask, void * user_data) {
    const bool norm = std::strcmp(ggml_get_name(t), "result_norm") == 0;
    const bool logits = std::strcmp(ggml_get_name(t), "result_output") == 0;
    if (ask) return norm || logits;
    if (!norm && !logits) return true;
    auto & out = *static_cast<tail_capture *>(user_data);
    tail_tensor & snap = norm ? out.norm : out.logits;
    ++snap.visits;
    snap.valid = t->type == GGML_TYPE_F32 && t->ne[0] > 0 && t->ne[1] > 0 &&
                 t->ne[2] == 1 && t->ne[3] == 1 && t->nb[0] == sizeof(float);
    if (!snap.valid) return true;
    snap.width = t->ne[0];
    snap.rows = t->ne[1];
    snap.values.resize(size_t(snap.width * snap.rows));
    for (int64_t row = 0; row < snap.rows; ++row) {
        ggml_backend_tensor_get(t, snap.values.data() + row*snap.width,
                                row*t->nb[1], size_t(snap.width)*sizeof(float));
    }
    return true;
}

bool compare_tail_tensor(const tail_tensor & a, const tail_tensor & b, int rows) {
    return a.visits == 1 && b.visits == 1 && a.valid && b.valid &&
           a.width == b.width && a.rows == rows && b.rows == rows &&
           a.values.size() == b.values.size() && !a.values.empty() &&
           std::memcmp(a.values.data(), b.values.data(), a.values.size()*sizeof(float)) == 0;
}

bool compare_tail_capture(const tail_capture & a, const tail_capture & b,
                          int rows, int vocab, const float * a_logits, const float * b_logits) {
    if (!compare_tail_tensor(a.norm, b.norm, rows) || !compare_tail_tensor(a.logits, b.logits, rows) ||
        a.logits.width != vocab || b.logits.width != vocab) return false;
    const size_t offset = size_t(rows - 1)*vocab;
    return std::memcmp(a.logits.values.data() + offset, a_logits, size_t(vocab)*sizeof(float)) == 0 &&
           std::memcmp(b.logits.values.data() + offset, b_logits, size_t(vocab)*sizeof(float)) == 0;
}
bool write_rows(FILE * file, int step, int width, int pos, int vocab,
                const std::vector<int> & indices, const std::vector<float> & rows) {
    const uint32_t header[] = {uint32_t(step), uint32_t(width), uint32_t(pos),
                               uint32_t(vocab), uint32_t(indices.size())};
    if (std::fwrite(header, sizeof(header), 1, file) != 1) return false;
    for (int index : indices) {
        const uint32_t i = uint32_t(index);
        if (std::fwrite(&i, sizeof(i), 1, file) != 1) return false;
    }
    return std::fwrite(rows.data(), sizeof(float), rows.size(), file) == rows.size();
}

bool capture(llama_context * ctx, const std::vector<int> & indices, int vocab,
             std::vector<float> & rows) {
    rows.clear();
    for (int index : indices) {
        const float * logits = llama_get_logits_ith(ctx, index);
        if (!logits) return false;
        rows.insert(rows.end(), logits, logits + vocab);
    }
    return true;
}

bool run(llama_model * model, const char * mode, const char * on_path, const char * off_path, bool instrumented) {
    const bool variable = std::strcmp(mode, "variable") == 0;
    if (!variable && std::strcmp(mode, "fixed") != 0) return false;
    const int widths[] = {1, 3, 2, width_max};
    int total_tokens = 0;
    for (int step = 0; step < steps; ++step) total_tokens += variable ? widths[step % 4] : width_max;

    auto cp = llama_context_default_params();
    cp.n_ctx = total_tokens;
    cp.n_ctx_kv = total_tokens;
    cp.n_batch = width_max;
    cp.n_ubatch = width_max;
    cp.n_seq_max = 1;
    cp.n_outputs_max = width_max; // explicit caller contract, not an implicit default
    cp.n_threads = cp.n_threads_batch = 1;
    cp.kv_unified = true;
    cp.offload_kqv = cp.op_offload = false;
    cp.no_perf = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    const char * capture_env = std::getenv("NKV_CAPTURE_KQV");
    const bool capture_node = !instrumented && capture_env && std::strcmp(capture_env, "1") == 0;
    kqv_capture on_node, off_node;
    tail_capture on_tail, off_tail;
    if (instrumented) {
        cp.cb_eval = capture_tail;
        cp.cb_eval_user_data = &on_tail;
    } else if (capture_node) {
        cp.cb_eval = capture_kqv;
        cp.cb_eval_user_data = &on_node;
    }

    setenv("GGML_TP5_UNION_BINDER", "1", 1);
    unsetenv("GGML_TP5_RSZ_RT");
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    llama_context * on = llama_init_from_model(model, cp);
    if (instrumented) cp.cb_eval_user_data = &off_tail;
    else if (capture_node) cp.cb_eval_user_data = &off_node;
    setenv("GGML_TP5_UNION_BINDER", "0", 1);
    llama_context * off = llama_init_from_model(model, cp);
    if (!on || !off) {
        if (on) llama_free(on);
        if (off) llama_free(off);
        std::fprintf(stderr, "CONTEXT_FAIL mode=%s\n", mode);
        return false;
    }
    FILE * out_on = on_path ? std::fopen(on_path, "wb") : std::tmpfile();
    FILE * out_off = off_path ? std::fopen(off_path, "wb") : std::tmpfile();
    if (!out_on || !out_off) {
        if (out_on) std::fclose(out_on);
        if (out_off) std::fclose(out_off);
        llama_free(on);
        llama_free(off);
        return false;
    }

    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    int pos = 0, pass_steps = 0, pass_rows = 0, pass_tails = 0, on_builds = 0, off_builds = 0;
    int previous_count = 0, growths = 0, shrinks = 0, transitions = 0;
    bool tail_changed = false;
    std::vector<float> first_tail;
    bool ok = vocab > 3;
    std::fprintf(stderr, "DECODE_BEGIN mode=%s steps=%d total_tokens=%d declared_outputs_max=%d "
                 "on_nctx=%u off_nctx=%u on_kv=%u off_kv=%u\n",
                 mode, steps, total_tokens, width_max, llama_n_ctx(on), llama_n_ctx(off),
                 llama_n_ctx_kv(on), llama_n_ctx_kv(off));
    for (int step = 0; ok && step < steps; ++step) {
        const int width = variable ? widths[step % 4] : width_max;
        const int count = variable ? 1 + (step % width) : widths[(step + 3) % 4];
        if (step != 0 && count != previous_count) {
            ++transitions;
            if (count > previous_count) ++growths; else ++shrinks;
        }
        previous_count = count;
        llama_batch batch = llama_batch_init(width, 0, 1);
        std::vector<int> indices;
        for (int j = 0; j < width; ++j) {
            batch.token[j] = 3 + ((pos + j) * 7 + 1) % (vocab - 3);
            batch.pos[j] = pos + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = j >= width - count;
            if (batch.logits[j]) indices.push_back(j);
        }
        batch.n_tokens = width;
        // Build counting uses the actual model.build_graph counter (base line),
        // which also counts internal fallback rebuilds; the strict line's
        // n_reused proxy cannot see them, so the stricter counter is kept.
        if (instrumented) {
            on_tail = tail_capture{};
            off_tail = tail_capture{};
        }
        if (capture_node) {
            on_node.seen = off_node.seen = false;
            on_node.valid = off_node.valid = false;
        }
        const uint64_t on_before = llama_graph_build_count(on);
        const uint64_t off_before = llama_graph_build_count(off);
        const int on_rc = llama_decode(on, batch);
        const int off_rc = llama_decode(off, batch);
        const int on_build = int(llama_graph_build_count(on) - on_before);
        const int off_build = int(llama_graph_build_count(off) - off_before);
        llama_batch_free(batch);
        if (on_rc || off_rc || int(indices.size()) != count) {
            std::fprintf(stderr, "DECODE_FAIL mode=%s step=%d on=%d off=%d builds=%d/%d rows=%zu\n",
                         mode, step, on_rc, off_rc, on_build, off_build, indices.size());
            ok = false;
            break;
        }
        if (capture_node && !compare_kqv(on_node, off_node, step)) ok = false;
        on_builds += on_build;
        off_builds += off_build;
        std::vector<float> a, b;
        if (!capture(on, indices, vocab, a) || !capture(off, indices, vocab, b) || a.size() != b.size() ||
            !write_rows(out_on, step, width, pos, vocab, indices, a) ||
            !write_rows(out_off, step, width, pos, vocab, indices, b)) {
            std::fprintf(stderr, "CAPTURE_FAIL mode=%s step=%d\n", mode, step);
            ok = false;
            break;
        }
        const float * tail_on = llama_get_logits_ith(on, -1);
        const float * tail_off = llama_get_logits_ith(off, -1);
        if (!tail_on || !tail_off || indices.empty() || indices.back() != width - 1 ||
            std::memcmp(tail_on, a.data() + (indices.size() - 1)*vocab, size_t(vocab)*sizeof(float)) ||
            std::memcmp(tail_off, b.data() + (indices.size() - 1)*vocab, size_t(vocab)*sizeof(float))) {
            std::fprintf(stderr, "TAIL_CAPTURE_FAIL mode=%s step=%d\n", mode, step);
            ok = false;
            break;
        }
        if (instrumented && !compare_tail_capture(on_tail, off_tail, count, vocab, tail_on, tail_off)) {
            std::fprintf(stderr, "OUTPUT_TAIL_NODE_FAIL mode=%s step=%d norm=%d/%d output=%d/%d\n",
                         mode, step, on_tail.norm.visits, off_tail.norm.visits,
                         on_tail.logits.visits, off_tail.logits.visits);
            ok = false;
            break;
        }
        bool finite = true;
        for (float x : a) finite = finite && std::isfinite(x);
        for (float x : b) finite = finite && std::isfinite(x);
        if (!finite) { std::fprintf(stderr, "NONFINITE mode=%s step=%d\n", mode, step); ok = false; break; }
        const bool tail_equal = std::memcmp(tail_on, tail_off, size_t(vocab)*sizeof(float)) == 0;
        pass_tails += tail_equal;
        if (first_tail.empty()) first_tail.assign(tail_on, tail_on + vocab);
        else if (std::memcmp(first_tail.data(), tail_on, size_t(vocab)*sizeof(float)) != 0) tail_changed = true;
        int equal_rows = 0;
        for (size_t r = 0; r < indices.size(); ++r) {
            if (std::memcmp(a.data() + r*vocab, b.data() + r*vocab, size_t(vocab)*sizeof(float)) == 0) ++equal_rows;
        }
        pass_rows += equal_rows;
        const bool equal = equal_rows == count;
        pass_steps += equal;
        std::fprintf(stderr, "STEP mode=%s step=%d ntok=%d active_outputs=%d tail_token=%d "
                     "declared_limit=%d max_possible_outputs=%d on_build=%d off_build=%d "
                     "bitwise_rows=%d/%d tail_equal=%d\n", mode, step, width, count, indices.back(), width_max,
                     std::min(width, width_max), on_build, off_build, equal_rows, count, int(tail_equal));
        pos += width;
    }
    const bool builds_ok = on_builds == 1 && off_builds == steps;
    std::fprintf(stderr, "DECODE_END mode=%s steps=%d bitwise_steps=%d bitwise_rows=%d tails=%d "
                 "growths=%d shrinks=%d changed_tail=%d on_decode_builds=%d off_decode_builds=%d\n",
                 mode, steps, pass_steps, pass_rows, pass_tails, growths, shrinks, int(tail_changed),
                 on_builds, off_builds);
    log_capture log;
    llama_log_get(&log.previous, &log.previous_data);
    llama_log_set(capture_log, &log);
    log.recording = true;
    llama_free(on);
    log.recording = false;
    llama_log_set(log.previous, log.previous_data);
    const bool binder_ok = check_binder_stats(log.text, transitions, growths);
    std::fprintf(stderr, "BINDER_EVIDENCE mode=%s valid=%d\n", mode, int(binder_ok));
    llama_free(off);
    if (std::fclose(out_on) != 0 || std::fclose(out_off) != 0) ok = false;
    return ok && pos == total_tokens && pass_steps == steps && pass_tails == steps &&
           transitions > 0 && growths > 0 && shrinks > 0 && tail_changed && builds_ok && binder_ok;
}
} // namespace

int main(int argc, char ** argv) {
    const bool plain_gate = argc == 2 && std::strcmp(argv[1], "--gate") == 0;
    const bool tail_gate = argc == 2 && std::strcmp(argv[1], "--tail-gate") == 0;
    const bool gate = plain_gate || tail_gate;
    if (!gate && argc != 5) {
        std::fprintf(stderr, "usage: %s --gate|--tail-gate | model.gguf fixed|variable on.bin off.bin\n", argv[0]);
        return 2;
    }
    const char * model_path = gate ? std::getenv("NOUT_TAIL_MODEL") : argv[1];
    if (!model_path || !*model_path) {
        std::fprintf(stderr, "NOUT_TAIL_MODEL is required for the fixed gate\n");
        return 2;
    }
    if (gate) unsetenv("NKV_CAPTURE_KQV");
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    ggml_backend_dev_t devices[] = {nullptr};
    mp.devices = devices;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    const bool ok = model && run(model, gate ? "fixed" : argv[2], gate ? nullptr : argv[3], gate ? nullptr : argv[4], tail_gate);
    if (model) llama_model_free(model);
    llama_backend_free();
    std::fprintf(stderr, "PROBE_RC=%d\n", ok ? 0 : 1);
    return ok ? 0 : 1;
}
