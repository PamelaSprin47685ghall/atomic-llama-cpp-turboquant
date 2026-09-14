#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../src/llama-memory-hybrid-idx.h"
#include "../src/llama-model.h"
#include "ggml.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

// Stub model to initialize production llama_memory_hybrid_idx
struct qsa_test_model : public llama_model {
    qsa_test_model() : llama_model(llama_model_default_params()) {
        arch = LLM_ARCH_QWEN4EXP;
        hparams.n_ctx_train = 4096;
        hparams.n_layer_all = 2;
        hparams.is_recr_impl[1] = true;
        hparams.n_head_arr.fill(4);
        hparams.n_head_kv_arr.fill(4);
        hparams.n_embd             = 128;
        hparams.n_embd_head_k_full = 128;
        hparams.n_embd_head_v_full = 128;
        // SSM / Gated DeltaNet recurrent dimensions (matching test-llama-archs conventions):
        hparams.ssm_d_conv     = 4;
        hparams.ssm_d_inner    = 128;
        hparams.ssm_d_state    = 32;
        hparams.ssm_dt_rank    = 2;
        hparams.ssm_n_group    = 2;
        hparams.dsv4_hc_mult   = 2;
        hparams.hc_low_rank    = 32;
        // QSA Indexer dimensions:
        hparams.indexer_head_size  = 32;
        hparams.indexer_n_head     = 2;
        hparams.indexer_top_k      = 8;
        hparams.dsv4_compress_ratios.fill(4);
        hparams.no_alloc = false;
    }

    void load_stats(llama_model_loader &) override {}
    void load_hparams(llama_model_loader &) override {}
    void load_vocab(llama_model_loader &) override {}
    bool load_tensors(llama_model_loader &) override { return true; }
    void load_arch_hparams(llama_model_loader &) override {}
    void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override {
        return nullptr;
    }
};

int main() {
    qsa_test_model model;

    // Filters: layer 0 is dense attention (with QSA), layer 1 is recurrent
    auto filter_attn = [](uint32_t il) { return il == 0; };
    auto filter_recr = [](uint32_t il) { return il == 1; };
    auto filter_idx  = [](uint32_t il) { return il == 0; };

    const uint32_t kv_size = 128;
    const uint32_t n_seq_max = 2;

    llama_memory_hybrid_idx mem(
        model,
        GGML_TYPE_F32, GGML_TYPE_F32, false,
        kv_size, 1, 0, LLAMA_SWA_TYPE_NONE,
        GGML_TYPE_F32, GGML_TYPE_F32, n_seq_max,
        n_seq_max, 1, false, true,
        filter_attn, filter_recr, filter_idx
    );

    mem.materialize();

    const llama_seq_id seq0 = 0;
    const llama_seq_id seq1 = 1;

    // Verify production pooled-cache geometry
    assert(mem.get_mem_idx() != nullptr);
    assert(mem.get_pooled_k(0) != nullptr);
    assert(mem.get_pooled_k(1) == nullptr); // Recurrent layer must have no pooled cache
    assert(mem.get_pooled_rows() == kv_size / 4 + 2);

    // Initial watermark is 0
    assert(mem.pooled_valid(seq0) == 0);

    // Set watermark directly as graph execution does
    mem.pooled_valid(seq0) = 8;
    assert(mem.pooled_valid(seq0) == 8);

    // 1. Test seq_rm clamping (dropping tokens starting at pos 20: 20 / 4 = 5 blocks)
    mem.seq_rm(seq0, 20, -1);
    assert(mem.pooled_valid(seq0) == 5);

    // 2. Test seq_rm p0 <= 0 drops all valid blocks
    mem.seq_rm(seq0, 0, -1);
    assert(mem.pooled_valid(seq0) == 0);

    // Reset and test seq_cp
    mem.pooled_valid(seq0) = 6;
    mem.seq_cp(seq0, seq1, 0, -1);
    // Destination watermark is reset to 0 to force fresh pooling
    assert(mem.pooled_valid(seq1) == 0);

    // 3. Test seq_add (position shift remaps blocks -> watermark must reset)
    mem.pooled_valid(seq0) = 6;
    mem.seq_add(seq0, 0, -1, 4);
    assert(mem.pooled_valid(seq0) == 0);

    // 4. Test seq_div
    mem.pooled_valid(seq0) = 6;
    mem.seq_div(seq0, 0, -1, 2);
    assert(mem.pooled_valid(seq0) == 0);

    // 5. Test seq_keep
    mem.pooled_valid(seq0) = 7;
    mem.pooled_valid(seq1) = 3;
    mem.seq_keep(seq0);
    assert(mem.pooled_valid(seq0) == 7);
    assert(mem.pooled_valid(seq1) == 0);

    // 6. Test clear(true)
    mem.clear(true);
    assert(mem.pooled_valid(seq0) == 0);

    printf("test-qsa-pooled-cache: all real llama_memory_hybrid_idx lifecycle invariants verified successfully.\n");
    return 0;
}
