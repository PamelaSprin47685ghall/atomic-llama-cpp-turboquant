#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// The model must be a CPU-supported dense LLAMA GGUF.
int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s model.gguf\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = { nullptr };
    mp.devices = devices;
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { llama_backend_free(); return 1; }

    auto cp = llama_context_default_params();
    cp.n_ctx = cp.n_ctx_kv = 2048;
    cp.n_batch = cp.n_ubatch = cp.n_outputs_max = 4;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 1;
    cp.kv_unified = true;
    cp.offload_kqv = cp.op_offload = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.no_perf = false;

    setenv("GGML_TP5_UNION_BINDER", "1", 1);
    setenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK", "1", 1);
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    llama_context * on = llama_init_from_model(model, cp);
    uint64_t previous_input_bytes = 0;
    setenv("GGML_TP5_UNION_BINDER", "0", 1);
    llama_context * off = llama_init_from_model(model, cp);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    bool ok = on && off && vocab > 3;
    const int widths[] = { 1, 3, 2, 4 };
    const int counts[] = { 1, 0, 1, 4 };
    int pos = 0;
    for (int step = 0; ok && step < 800; ++step) {
        const int width = widths[step % 4];
        const int count = counts[step % 4];
        llama_batch batch = llama_batch_init(width, 0, 1);
        batch.n_tokens = width;
        for (int i = 0; i < width; ++i) {
            batch.token[i] = 3 + ((pos + i) * 7 + 1) % (vocab - 3);
            batch.pos[i] = pos + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i >= width - count;
        }
        const int a = llama_decode(on, batch);
        const int b = llama_decode(off, batch);
        llama_batch_free(batch);
        if (a || b) {
            std::fprintf(stderr, "decode step=%d on=%d off=%d\n", step, a, b);
            ok = false;
            break;
        }
        llama_ntok_plan_stats stats = {};
        if (!llama_get_ntok_plan_stats(on, &stats) || stats.graph_builds != 1 ||
                stats.active_tokens != uint32_t(width) || stats.active_kv != uint32_t(pos + width) ||
                stats.active_outputs != uint32_t(count) || stats.checked_tensors == 0 ||
                stats.mismatched_tensors != 0 || stats.planned_bytes != stats.active_bytes ||
                stats.token_input_bytes != size_t(width) * sizeof(llama_token) ||
                stats.kv_mask_bytes != size_t(pos + width) * size_t(width) * sizeof(float) ||
                stats.output_ids_bytes != size_t(count) * sizeof(int32_t) ||
                (step > 0 && stats.token_input_bytes == previous_input_bytes)) {
            std::fprintf(stderr, "allocation step=%d width=%d builds=%llu active=%u "
                    "planned=%llu requested=%llu arena=%llu input=%llu checked=%llu mismatch=%llu\n",
                    step, width, (unsigned long long) stats.graph_builds, stats.active_tokens,
                    (unsigned long long) stats.planned_bytes, (unsigned long long) stats.active_bytes,
                    (unsigned long long) stats.arena_bytes, (unsigned long long) stats.token_input_bytes,
                    (unsigned long long) stats.checked_tensors, (unsigned long long) stats.mismatched_tensors);
            ok = false;
            break;
        }
        // Machine-readable plan-size lines for the no-headroom gate. Each
        // planned byte count is an allocator request for a root that depends
        // on the active KV, token or output width; the requested value is the
        // exact width this step asked for. A plan above the request is
        // reserved headroom and fails here as well as in the gate script.
        const size_t kv_width    = size_t(pos + width);
        const size_t token_width = size_t(width);
        const size_t output_width = size_t(count);
        const size_t kv_mask_bytes     = kv_width * token_width * sizeof(float);
        const size_t token_input_bytes = token_width * sizeof(llama_token);
        const size_t output_ids_bytes  = output_width * sizeof(int32_t);
        std::fprintf(stderr, "PLAN_SIZE step=%d kind=kv active_kv=%zu planned_bytes=%llu requested_bytes=%zu\n",
                step, kv_width, (unsigned long long) stats.kv_mask_bytes, kv_mask_bytes);
        std::fprintf(stderr, "PLAN_SIZE step=%d kind=token active_tokens=%zu planned_bytes=%llu requested_bytes=%zu\n",
                step, token_width, (unsigned long long) stats.token_input_bytes, token_input_bytes);
        std::fprintf(stderr, "PLAN_SIZE step=%d kind=output active_outputs=%zu planned_bytes=%llu requested_bytes=%zu\n",
                step, output_width, (unsigned long long) stats.output_ids_bytes, output_ids_bytes);
        if (stats.kv_mask_bytes > kv_mask_bytes || stats.token_input_bytes > token_input_bytes ||
                stats.output_ids_bytes > output_ids_bytes) {
            std::fprintf(stderr, "headroom step=%d kv=%llu/%zu token=%llu/%zu output=%llu/%zu\n",
                    step, (unsigned long long) stats.kv_mask_bytes, kv_mask_bytes,
                    (unsigned long long) stats.token_input_bytes, token_input_bytes,
                    (unsigned long long) stats.output_ids_bytes, output_ids_bytes);
            ok = false;
            break;
        }
        previous_input_bytes = stats.token_input_bytes;
        for (int i = 0; i < width; ++i) {
            if (i < width - count) continue;
            const float * lhs = llama_get_logits_ith(on, i);
            const float * rhs = llama_get_logits_ith(off, i);
            if (!lhs || !rhs || std::memcmp(lhs, rhs, size_t(vocab) * sizeof(float)) != 0) {
                std::fprintf(stderr, "logits differ step=%d row=%d\n", step, i);
                ok = false;
                break;
            }
        }
        pos += width;
    }
    ok = ok && pos == 2000;
    if (on) llama_free(on);
    if (off) llama_free(off);
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
