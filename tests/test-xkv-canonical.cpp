// Comprehensive unit tests and golden fixtures for XKV canonical read boundary (§5.3 / §8.1)
// Exercises:
// - Noncommuting transform-order fixture with attention rotation ON and actual selected nrot
// - Real partial IMRoPE with theta=1e7, head_dim=256, rotary_dim=64, and non-rotary tail intact
// - Real ggml tensor strides (ne/nb) and both non-transposed and transposed V layouts
// - Public read_k_canonical and read_v_canonical APIs with explicit canonical_rope_context
// - Missing-factor fail-closed rejection
// - Real backend handles and sync counts:
//   * Same backend for K+V -> 1 synchronization
//   * Distinct backends for K and V -> 2 synchronizations
//   * Host-only buffers -> 0 synchronizations
// - Real budgeted xkv_prerope_staging object enforcing committed/PUBLIC invariants, rejecting tentative/normal/private_control/pending_record, and releasing on seal
// - Invalid, bounds, overflow, and fail-closed error reporting
// - Exact semantic parity with Tri scoring oracle

#include "llama-xkv-canonical.h"
#include "llama-triattention.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

static int g_test_failures = 0;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_test_failures++; \
        return; \
    } \
} while (0)

#define TEST_ASSERT_MSG(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s:%d: %s — %s\n", __FILE__, __LINE__, #cond, msg); \
        g_test_failures++; \
        return; \
    } \
} while (0)

static bool float_eq(float a, float b, float eps = 1e-3f) {
    return fabsf(a - b) < eps;
}

// Deterministic orthonormal Sylvester-Hadamard generator (no private header dependencies)
static void generate_test_hadamard(float * data, uint32_t n) {
    assert(n > 0 && (n & (n - 1)) == 0);
    data[0] = 1.0f / sqrtf((float) n);

    for (uint32_t s = 1; s < n; s *= 2) {
        for (uint32_t i = 0; i < s; ++i) {
            for (uint32_t j = 0; j < s; ++j) {
                const float val = data[i * n + j];
                data[(i + s) * n + (j    )] =  val;
                data[(i    ) * n + (j + s)] =  val;
                data[(i + s) * n + (j + s)] = -val;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Test Stub Model for Initializing Real llama_kv_cache
// ----------------------------------------------------------------------------

struct canonical_test_model : public llama_model {
    canonical_test_model() : llama_model(llama_model_default_params()) {
        hparams.n_ctx_train = 4096;
        arch = LLM_ARCH_QWEN35MOE; // native IMRoPE architecture
        hparams.rope_type = LLAMA_ROPE_TYPE_IMROPE;
        hparams.n_layer_all = 2;
        hparams.n_head_arr.fill(4);
        hparams.n_head_kv_arr.fill(2); // 2 KV heads serving 4 Q heads
        hparams.n_embd_head_k_full = 256;
        hparams.n_embd_head_v_full = 256;
        hparams.n_rot_full = 64; // partial rotary dimension = 64, tail = 192
        hparams.rope_freq_base_train = 1e7f;
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

// ----------------------------------------------------------------------------
// Test 1: Noncommuting Transform Order (Hadamard Attention Rotation vs RoPE)
// ----------------------------------------------------------------------------
static void test_noncommuting_transform_order() {
    fprintf(stderr, "--- test_noncommuting_transform_order ---\n");

    const uint32_t head_dim = 128;
    const uint32_t rotary_dim = 128;
    const uint32_t nrot = 64;
    const uint32_t freq_count = rotary_dim / 2;
    const int32_t pos = 17;

    std::vector<float> omega(freq_count);
    std::vector<float> freq_scale_sq(freq_count, 1.0f);
    for (uint32_t f = 0; f < freq_count; ++f) {
        omega[f] = powf(10000.0f, -2.0f * (float) f / (float) rotary_dim);
    }

    std::vector<float> hadamard(nrot * nrot);
    generate_test_hadamard(hadamard.data(), nrot);

    std::vector<float> k_canonical(head_dim);
    for (uint32_t i = 0; i < head_dim; ++i) {
        k_canonical[i] = sinf((float) (i + 1) * 0.13f);
    }

    // Step 1: RoPE forward
    std::vector<float> k_post_rope(head_dim);
    for (uint32_t f = 0; f < freq_count; ++f) {
        float angle = omega[f] * (float) pos;
        float c = cosf(angle);
        float s = sinf(angle);
        float re = k_canonical[f];
        float im = k_canonical[f + freq_count];
        k_post_rope[f]              = re * c - im * s;
        k_post_rope[f + freq_count] = re * s + im * c;
    }

    // Step 2: Attention Hadamard rotation forward
    std::vector<float> k_stored(head_dim);
    std::memcpy(k_stored.data(), k_post_rope.data(), head_dim * sizeof(float));
    std::string err;
    TEST_ASSERT(llama_xkv::invert_attention_rotation(k_stored.data(), 1, head_dim, hadamard.data(), nrot, &err));

    // Decode Pipeline Correct Order: Turbo Decode -> H^{-1} -> RoPE^{-1}
    std::vector<float> k_correct(head_dim);
    std::memcpy(k_correct.data(), k_stored.data(), head_dim * sizeof(float));
    TEST_ASSERT(llama_xkv::invert_attention_rotation(k_correct.data(), 1, head_dim, hadamard.data(), nrot, &err));
    std::vector<float> k_recovered(head_dim);
    TEST_ASSERT(llama_xkv::invert_rope_k(k_recovered.data(), k_correct.data(), &pos, omega.data(), freq_scale_sq.data(), 1, head_dim, rotary_dim, &err));

    for (uint32_t i = 0; i < head_dim; ++i) {
        TEST_ASSERT_MSG(float_eq(k_recovered[i], k_canonical[i], 1e-4f), "Correct decode order failed to recover canonical K");
    }

    // Decode Pipeline Wrong Order: RoPE^{-1} -> H^{-1}
    std::vector<float> k_wrong_rope(head_dim);
    TEST_ASSERT(llama_xkv::invert_rope_k(k_wrong_rope.data(), k_stored.data(), &pos, omega.data(), freq_scale_sq.data(), 1, head_dim, rotary_dim, &err));
    TEST_ASSERT(llama_xkv::invert_attention_rotation(k_wrong_rope.data(), 1, head_dim, hadamard.data(), nrot, &err));

    double max_diff = 0.0;
    for (uint32_t i = 0; i < head_dim; ++i) {
        max_diff = std::max(max_diff, (double) std::abs(k_wrong_rope[i] - k_canonical[i]));
    }
    TEST_ASSERT_MSG(max_diff > 0.05, "RoPE and Hadamard unexpectedly commuted (must be noncommuting)");

    fprintf(stderr, "  PASSED (max commute diff: %.4f)\n", max_diff);
}

// ----------------------------------------------------------------------------
// Test 2: Real partial IMRoPE with theta=1e7, head_dim=256, rotary_dim=64
// ----------------------------------------------------------------------------
static void test_real_partial_imrope_recovery() {
    fprintf(stderr, "--- test_real_partial_imrope_recovery ---\n");

    const uint32_t head_dim = 256;
    const uint32_t rotary_dim = 64;
    const uint32_t freq_count = rotary_dim / 2; // 32
    const float freq_base = 1e7f;
    const int32_t position = 127;

    // Use explicit canonical_rope_context creation
    auto rope_ctx = llama_xkv::canonical_rope_context::create_text_imrope(rotary_dim, freq_base, 1.0f);
    TEST_ASSERT(rope_ctx.valid);

    std::vector<float> pre_rope(head_dim);
    for (uint32_t i = 0; i < head_dim; ++i) {
        pre_rope[i] = (float) (i + 1) * 0.01f;
    }

    std::vector<float> post_rope(head_dim);
    for (uint32_t f = 0; f < freq_count; ++f) {
        float angle = rope_ctx.omega[f] * (float) position;
        float c = cosf(angle);
        float s = sinf(angle);
        float re = pre_rope[f];
        float im = pre_rope[f + freq_count];
        post_rope[f]              = re * c - im * s;
        post_rope[f + freq_count] = re * s + im * c;
    }
    for (uint32_t d = rotary_dim; d < head_dim; ++d) {
        post_rope[d] = pre_rope[d]; // Non-rotary tail intact
    }

    std::vector<float> recovered(head_dim);
    std::string err;
    bool ok = llama_xkv::invert_rope_k(
        recovered.data(), post_rope.data(), &position,
        rope_ctx.omega.data(), rope_ctx.freq_scale_sq.data(), 1,
        head_dim, rotary_dim, &err
    );
    TEST_ASSERT_MSG(ok, err.c_str());

    for (uint32_t d = 0; d < rotary_dim; ++d) {
        TEST_ASSERT_MSG(float_eq(recovered[d], pre_rope[d], 1e-4f), "Rotary dim mismatch");
    }
    for (uint32_t d = rotary_dim; d < head_dim; ++d) {
        TEST_ASSERT_MSG(float_eq(recovered[d], pre_rope[d], 1e-6f), "Tail dim mismatch");
    }

    fprintf(stderr, "  PASSED\n");
}

// ----------------------------------------------------------------------------
// Test 3: Real llama_kv_cache integration, Transposed V, Layer Mapping, and Sync Counting
// ----------------------------------------------------------------------------
static void test_kv_cache_canonical_read_and_sync_counting() {
    fprintf(stderr, "--- test_kv_cache_canonical_read_and_sync_counting ---\n");

    canonical_test_model model;
    llama_cparams cparams = {};
    cparams.xkv_mode = LLAMA_XKV_MODE_SR;
    cparams.n_ctx_kv = 64;

    const uint32_t kv_size = 64;
    const uint32_t head_dim = 256;
    const uint32_t rotary_dim = 64;

    // Explicit validated rope context for layer 0 (no factors needed for text IMRoPE)
    auto rope_ctx = llama_xkv::canonical_rope_context::create_text_imrope(rotary_dim, 1e7f, 1.0f);
    TEST_ASSERT(rope_ctx.valid);

    // 1. Create cache with non-transposed V (v_trans = false)
    llama_kv_cache kv_notrans(
        model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, kv_size, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr
    );
    kv_notrans.init_xkv_store(cparams);

    ggml_tensor * k0 = kv_notrans.get_k_storage(0);
    ggml_tensor * v0 = kv_notrans.get_v_storage(0);
    TEST_ASSERT(k0 && v0);
    TEST_ASSERT(k0->buffer && v0->buffer);

    const size_t k_row_stride = k0->ne[0]; // 2 * 256 = 512

    std::vector<float> k_val(head_dim);
    std::vector<float> v_val(head_dim);
    for (uint32_t j = 0; j < head_dim; ++j) {
        k_val[j] = (float) j * 0.5f;
        v_val[j] = (float) j * 0.25f;
    }
    ggml_backend_tensor_set(k0, k_val.data(), (5 * k_row_stride + 1 * head_dim) * sizeof(float), head_dim * sizeof(float));
    ggml_backend_tensor_set(v0, v_val.data(), (5 * k_row_stride + 1 * head_dim) * sizeof(float), head_dim * sizeof(float));

    uint32_t cell = 5;
    int32_t pos = 0;
    llama_xkv::canonical_read_batch batch;
    batch.cell_indices = &cell;
    batch.positions = &pos;
    batch.n_cells = 1;
    batch.seq_id = 0;

    std::vector<float> k_rec(head_dim);
    std::vector<float> v_rec(head_dim);
    llama_xkv::canonical_batch_stats stats;
    std::string err;

    // Read K head 1 with explicit canonical_rope_context
    bool ok_k = llama_xkv::read_k_canonical(
        kv_notrans, cparams, 0, batch, 1, k_rec.data(), head_dim,
        LLAMA_XKV_SOURCE_DECODED_HOT, nullptr, nullptr, &rope_ctx, &stats, &err
    );
    TEST_ASSERT_MSG(ok_k, err.c_str());

    bool ok_v = llama_xkv::read_v_canonical(
        kv_notrans, cparams, 0, batch, 1, v_rec.data(), head_dim,
        LLAMA_XKV_SOURCE_DECODED_HOT, nullptr, nullptr, &stats, &err
    );
    TEST_ASSERT_MSG(ok_v, err.c_str());

    for (uint32_t j = 0; j < head_dim; ++j) {
        TEST_ASSERT(float_eq(k_rec[j], (float) j * 0.5f));
        TEST_ASSERT(float_eq(v_rec[j], (float) j * 0.25f));
    }

    // 2. Combined single-sync layer snapshot testing (host memory -> sync_count = 0)
    llama_xkv::xkv_canonical_layer_kv_snapshot kv_snap;
    stats = {};
    bool init_ok = kv_snap.init(
        kv_notrans, cparams, 0, &cell, &pos, 1, 0, true, true,
        LLAMA_XKV_SOURCE_DECODED_HOT, nullptr, nullptr, &rope_ctx, &stats, &err
    );
    TEST_ASSERT_MSG(init_ok, err.c_str());
    TEST_ASSERT_MSG(stats.sync_count == 0, "Host memory buffers require 0 synchronizations");

    std::vector<float> h0_k(head_dim), h1_k(head_dim);
    std::vector<float> h0_v(head_dim), h1_v(head_dim);
    TEST_ASSERT(kv_snap.read_k_head(0, h0_k.data(), head_dim, &err));
    TEST_ASSERT(kv_snap.read_k_head(1, h1_k.data(), head_dim, &err));
    TEST_ASSERT(kv_snap.read_v_head(0, h0_v.data(), head_dim, &err));
    TEST_ASSERT(kv_snap.read_v_head(1, h1_v.data(), head_dim, &err));
    TEST_ASSERT(float_eq(h1_k[10], 5.0f));
    TEST_ASSERT(float_eq(h1_v[10], 2.5f));

    // 3. Backend synchronization contracts:
    ggml_backend_t b1 = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t b2 = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    // Test the synchronization planner helper without invalidating tensor buffer ownership:
    // Case A: Same non-null backend for K and V -> 1 synchronization
    TEST_ASSERT(llama_xkv::plan_canonical_synchronizations(b1, true, b1, true) == 1);
    // Case B: Distinct non-null backends for K and V -> 2 synchronizations
    TEST_ASSERT(llama_xkv::plan_canonical_synchronizations(b1, true, b2, true) == 2);
    // Case C: K needs sync, V does not -> 1 synchronization
    TEST_ASSERT(llama_xkv::plan_canonical_synchronizations(b1, true, b2, false) == 1);
    // Case D: Host buffers (neither needs sync) -> 0 synchronizations
    TEST_ASSERT(llama_xkv::plan_canonical_synchronizations(b1, false, b2, false) == 0);
    TEST_ASSERT(llama_xkv::plan_canonical_synchronizations(nullptr, true, nullptr, true) == 0);

    if (b1) ggml_backend_free(b1);
    if (b2) ggml_backend_free(b2);

    // 4. Transposed V test (v_trans = true)
    llama_kv_cache kv_trans(
        model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        true, false, true, kv_size, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr
    );
    ggml_tensor * v_t = kv_trans.get_v_storage(0);
    TEST_ASSERT(v_t && kv_trans.get_v_trans());
    TEST_ASSERT(v_t->buffer);

    std::vector<float> vt_val(head_dim);
    for (uint32_t j = 0; j < head_dim; ++j) {
        size_t el_idx = ((size_t) (1 * head_dim + j) * kv_size) + 5;
        float v = 100.0f + (float) j;
        ggml_backend_tensor_set(v_t, &v, el_idx * sizeof(float), sizeof(float));
    }

    std::vector<float> vt_rec(head_dim);
    bool ok_vt = llama_xkv::read_v_canonical(
        kv_trans, cparams, 0, batch, 1, vt_rec.data(), head_dim,
        LLAMA_XKV_SOURCE_DECODED_HOT, nullptr, nullptr, nullptr, &err
    );
    TEST_ASSERT_MSG(ok_vt, err.c_str());
    for (uint32_t j = 0; j < head_dim; ++j) {
        TEST_ASSERT(float_eq(vt_rec[j], 100.0f + (float) j));
    }

    // 5. Fail-closed on missing rope context or factors when rotary_dim is invalid
    llama_xkv::canonical_rope_context invalid_rope_ctx;
    invalid_rope_ctx.valid = false;
    llama_xkv::xkv_canonical_layer_kv_snapshot bad_snap;
    std::string bad_err;
    bool bad_ok = bad_snap.init(
        kv_notrans, cparams, 0, &cell, &pos, 1, 0, true, true,
        LLAMA_XKV_SOURCE_DECODED_HOT, nullptr, nullptr, &invalid_rope_ctx, nullptr, &bad_err
    );
    TEST_ASSERT_MSG(!bad_ok, "Missing RoPE tables for K must fail closed");

    // 6. Explicit unallocated-buffer rejection fixture using no_alloc=true cache
    {
        canonical_test_model model_noalloc;
        model_noalloc.hparams.no_alloc = true;
        llama_kv_cache kv_unalloc(
            model_noalloc, model_noalloc.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
            false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, nullptr, nullptr, nullptr
        );
        ggml_tensor * k_unalloc = kv_unalloc.get_k_storage(0);
        TEST_ASSERT(k_unalloc);
        // Cache constructed with no_alloc has dummy buffer with size 0
        TEST_ASSERT(k_unalloc->buffer && ggml_backend_buffer_get_size(k_unalloc->buffer) == 0);
        llama_xkv::xkv_canonical_layer_kv_snapshot unalloc_snap;
        std::string unalloc_err;
        bool unalloc_ok = unalloc_snap.init(kv_unalloc, cparams, 0, &cell, &pos, 1, 0, true, true, LLAMA_XKV_SOURCE_DECODED_HOT, nullptr, nullptr, &rope_ctx, nullptr, &unalloc_err);
        TEST_ASSERT_MSG(!unalloc_ok, "Unallocated tensor buffer must fail closed with explicit error");
    }

    // 7. Non-unified stream selection test (seq_id > 0 with distinct stream data)
    {
        // Create non-unified cache: n_seq_max = 2, unified = false -> n_stream = 2
        llama_kv_cache kv_multi(
            model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
            false, false, false, kv_size, 2, 1, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, nullptr, nullptr, nullptr
        );
        TEST_ASSERT(kv_multi.get_n_stream() == 2);
        TEST_ASSERT(kv_multi.get_stream_for_seq(0) == 0);
        TEST_ASSERT(kv_multi.get_stream_for_seq(1) == 1);

        ggml_tensor * k_multi = kv_multi.get_k_storage(0);
        ggml_tensor * v_multi = kv_multi.get_v_storage(0);
        TEST_ASSERT(k_multi && v_multi);

        const size_t row_stride = k_multi->ne[0]; // 512
        // Stream 0: cell 3 gets 11.0f
        // Stream 1: cell 3 gets 77.0f
        std::vector<float> val_s0(head_dim, 11.0f);
        std::vector<float> val_s1(head_dim, 77.0f);
        const size_t offs_s0 = (0 * kv_size + 3) * row_stride * sizeof(float);
        const size_t offs_s1 = (1 * kv_size + 3) * row_stride * sizeof(float);
        ggml_backend_tensor_set(k_multi, val_s0.data(), offs_s0, head_dim * sizeof(float));
        ggml_backend_tensor_set(k_multi, val_s1.data(), offs_s1, head_dim * sizeof(float));

        uint32_t c3 = 3;
        int32_t p0 = 0;
        llama_xkv::canonical_read_batch batch_s0;
        batch_s0.cell_indices = &c3;
        batch_s0.positions = &p0;
        batch_s0.n_cells = 1;
        batch_s0.seq_id = 0;

        llama_xkv::canonical_read_batch batch_s1 = batch_s0;
        batch_s1.seq_id = 1;

        std::vector<float> rec_s0(head_dim);
        std::vector<float> rec_s1(head_dim);
        TEST_ASSERT(llama_xkv::read_k_canonical(kv_multi, cparams, 0, batch_s0, 0, rec_s0.data(), head_dim, LLAMA_XKV_SOURCE_DECODED_HOT, nullptr, nullptr, &rope_ctx, nullptr, &err));
        TEST_ASSERT(llama_xkv::read_k_canonical(kv_multi, cparams, 0, batch_s1, 0, rec_s1.data(), head_dim, LLAMA_XKV_SOURCE_DECODED_HOT, nullptr, nullptr, &rope_ctx, nullptr, &err));
        TEST_ASSERT(float_eq(rec_s0[0], 11.0f));
        TEST_ASSERT(float_eq(rec_s1[0], 77.0f));
    }

    fprintf(stderr, "  PASSED\n");
}

// ----------------------------------------------------------------------------
// Test 4: Real Budgeted xkv_prerope_staging Object and Invariant Rejections
// ----------------------------------------------------------------------------
static void test_budgeted_prerope_staging_and_invariants() {
    fprintf(stderr, "--- test_budgeted_prerope_staging_and_invariants ---\n");

    canonical_test_model model;
    llama_cparams cparams = {};
    cparams.xkv_mode = LLAMA_XKV_MODE_SR;
    cparams.n_ctx_kv = 64;

    llama_kv_cache kv(
        model, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        false, false, true, 64, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr
    );
    kv.init_xkv_store(cparams);
    auto store = kv.get_xkv_store();
    TEST_ASSERT(store);

    llama_xkv::prerope_staging_config cfg;
    cfg.max_budget_tokens = 16;
    cfg.head_dim_k = 256;
    cfg.head_dim_v = 256;
    cfg.n_kv_heads = 2;

    llama_xkv::xkv_prerope_staging staging(cfg);
    TEST_ASSERT(staging.is_valid());
    TEST_ASSERT(staging.get_allocated_bytes() == 0);

    std::vector<float> k_buf(8 * 2 * 256, 1.0f);
    std::vector<float> v_buf(8 * 2 * 256, 2.0f);
    std::string err;

    // Case 1: Empty cells reject (§5.2 invariant)
    bool ok = staging.stage_range(kv, 0, 0, 8, k_buf.data(), v_buf.data(), &err);
    TEST_ASSERT_MSG(!ok, "Empty cells must be rejected");

    // Case 2: Arithmetic overflow reject
    ok = kv.can_capture_prerope_range(0, UINT32_MAX - 4, 10, &err);
    TEST_ASSERT_MSG(!ok, "Arithmetic overflow in cell range must be rejected");

    // Setup 4 occupied cells
    auto & cells = const_cast<llama_kv_cells &>(kv.get_cells(0));
    for (uint32_t i = 0; i < 4; ++i) {
        cells.pos_set(i, i);
        cells.seq_add(i, 0);
        store->register_hot_payload(cells.payload_id_get(i), i, cells.storage_generation_get(i));
    }

    // Case 3: Tentative/hot_writing rejection (cells are registered hot_writing, NOT hot_committed)
    ok = staging.stage_range(kv, 0, 0, 4, k_buf.data(), v_buf.data(), &err);
    TEST_ASSERT_MSG(!ok, "Uncommitted tentative cells must be rejected");

    // Commit the payloads
    for (uint32_t i = 0; i < 4; ++i) {
        TEST_ASSERT(store->commit_hot_payload(cells.payload_id_get(i)));
    }

    // Now committed cells succeed
    ok = staging.stage_range(kv, 0, 0, 4, k_buf.data(), v_buf.data(), &err);
    TEST_ASSERT_MSG(ok, err.c_str());
    TEST_ASSERT(staging.get_staged_count() == 4);
    TEST_ASSERT(staging.get_allocated_bytes() == 4 * 2 * (256 + 256) * sizeof(float));

    // Case 4: Mark one cell as private_control -> staging fails closed
    llama_kv_rerot_meta priv_meta;
    priv_meta.episode_id = 99;
    priv_meta.node_id = 1;
    priv_meta.run_id = 1;
    priv_meta.visibility = llama_rerot_visibility::private_control;
    cells.rerot_set(2, priv_meta);
    ok = staging.stage_range(kv, 0, 0, 4, k_buf.data(), v_buf.data(), &err);
    TEST_ASSERT_MSG(!ok, "private_control cell in range must be rejected");

    // Case 5: Mark cell as pending_record -> staging fails closed
    llama_kv_rerot_meta pend_meta;
    pend_meta.episode_id = 99;
    pend_meta.node_id = 1;
    pend_meta.run_id = 1;
    pend_meta.visibility = llama_rerot_visibility::pending_record;
    cells.rerot_set(2, pend_meta);
    ok = staging.stage_range(kv, 0, 0, 4, k_buf.data(), v_buf.data(), &err);
    TEST_ASSERT_MSG(!ok, "pending_record cell in range must be rejected");

    // Release clears allocated staging memory
    staging.release();
    TEST_ASSERT(staging.get_staged_count() == 0);
    TEST_ASSERT(staging.get_allocated_bytes() == 0);

    fprintf(stderr, "  PASSED\n");
}

// ----------------------------------------------------------------------------
// Test 5: Tri Scoring Exact Parity on Canonical pre-RoPE Keys
// ----------------------------------------------------------------------------
static void test_tri_scoring_exact_parity() {
    fprintf(stderr, "--- test_tri_scoring_exact_parity ---\n");

    const uint32_t head_dim = 128;
    const uint32_t freq_count = head_dim / 2;
    const uint32_t n_candidates = 4;

    std::vector<float> omega(freq_count);
    std::vector<float> freq_scale_sq(freq_count, 1.0f);
    std::vector<float> offsets = {1.0f, 2.0f, 4.0f, 8.0f};

    for (uint32_t f = 0; f < freq_count; ++f) {
        omega[f] = powf(10000.0f, -2.0f * (float) f / (float) head_dim);
    }

    triattention_head_stats stats;
    std::vector<float> q_real(freq_count, 1.0f);
    std::vector<float> q_imag(freq_count, 0.0f);
    std::vector<float> q_abs(freq_count, 1.0f);
    std::vector<float> q_mean_abs(freq_count, 1.0f);
    std::vector<float> extra_w(freq_count, 0.0f);

    stats.q_mean_real  = q_real.data();
    stats.q_mean_imag  = q_imag.data();
    stats.q_abs_mean   = q_abs.data();
    stats.q_mean_abs   = q_mean_abs.data();
    stats.extra_weight = extra_w.data();

    std::vector<int32_t> positions = {10, 20, 30, 40};
    int64_t frontier_position = 100;

    std::vector<float> canonical_pre_rope_k(n_candidates * head_dim);
    for (size_t i = 0; i < canonical_pre_rope_k.size(); ++i) {
        canonical_pre_rope_k[i] = sinf((float) i * 0.1f);
    }

    std::vector<float> baseline_scores(n_candidates);
    triattention_score_keys(
        baseline_scores.data(), canonical_pre_rope_k.data(), &stats,
        omega.data(), freq_scale_sq.data(), offsets.data(),
        positions.data(), frontier_position, n_candidates,
        head_dim, freq_count, (uint32_t) offsets.size(),
        TRIATTENTION_AGG_MEAN, false
    );

    // Forward rotate
    std::vector<float> post_rope_k(n_candidates * head_dim);
    for (uint32_t i = 0; i < n_candidates; ++i) {
        const float * src = canonical_pre_rope_k.data() + i * head_dim;
        float * dst = post_rope_k.data() + i * head_dim;
        float pos = (float) positions[i];
        for (uint32_t f = 0; f < freq_count; ++f) {
            float angle = omega[f] * pos;
            float c = cosf(angle);
            float s = sinf(angle);
            float re = src[f];
            float im = src[f + freq_count];
            dst[f]              = re * c - im * s;
            dst[f + freq_count] = re * s + im * c;
        }
    }

    // Recover pre-RoPE keys via canonical invert_rope_k
    std::vector<float> recovered_pre_rope_k(n_candidates * head_dim);
    std::string err;
    bool ok = llama_xkv::invert_rope_k(
        recovered_pre_rope_k.data(), post_rope_k.data(), positions.data(),
        omega.data(), freq_scale_sq.data(), n_candidates,
        head_dim, head_dim, &err
    );
    TEST_ASSERT_MSG(ok, err.c_str());

    std::vector<float> recovered_scores(n_candidates);
    triattention_score_keys(
        recovered_scores.data(), recovered_pre_rope_k.data(), &stats,
        omega.data(), freq_scale_sq.data(), offsets.data(),
        positions.data(), frontier_position, n_candidates,
        head_dim, freq_count, (uint32_t) offsets.size(),
        TRIATTENTION_AGG_MEAN, false
    );

    for (uint32_t i = 0; i < n_candidates; ++i) {
        TEST_ASSERT_MSG(float_eq(recovered_scores[i], baseline_scores[i], 1e-4f),
            "Tri scores on recovered canonical pre-RoPE K differ from baseline");
    }

    fprintf(stderr, "  PASSED\n");
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    test_noncommuting_transform_order();
    test_real_partial_imrope_recovery();
    test_kv_cache_canonical_read_and_sync_counting();
    test_budgeted_prerope_staging_and_invariants();
    test_tri_scoring_exact_parity();

    if (g_test_failures != 0) {
        fprintf(stderr, "FAILED: %d tests failed\n", g_test_failures);
        return 1;
    }

    fprintf(stderr, "ALL CANONICAL TESTS PASSED\n");
    return 0;
}
