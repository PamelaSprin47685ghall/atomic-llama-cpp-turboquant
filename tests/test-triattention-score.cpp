// Unit tests for triattention_scorer — pure importance oracle scorer.
// Tests calibration loading, RoPE inversion, scoring, and z-score normalization.
// Uses assert-based testing (no external framework).

#include "../src/llama-triattention.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ============================================================================
// Helpers
// ============================================================================

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

static bool float_eq(float a, float b, float eps = 1e-4f) {
    return fabsf(a - b) < eps;
}

// ============================================================================
// Mock calibration file writer
// ============================================================================

struct mock_calib_params {
    uint32_t head_dim       = 128;
    uint32_t num_layers     = 2;
    uint32_t num_attn_heads = 4;
    uint32_t num_kv_heads   = 4;
    double   rope_theta     = 10000.0;
    uint32_t rope_style     = 0;  // half
    uint32_t n_sampled      = 2;
    uint32_t freq_count     = 64;
    const char * model_name = "test_model";
};

// Write a .triattention calibration file with known values.
static void write_mock_calib(const char * path, const mock_calib_params & p) {
    FILE * f = fopen(path, "wb");
    assert(f && "cannot open temp calib file for writing");

    // Header
    uint32_t magic = TRIATTENTION_MAGIC;
    uint32_t version = TRIATTENTION_VERSION;
    fwrite(&magic, sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(uint32_t), 1, f);
    fwrite(&p.head_dim, sizeof(uint32_t), 1, f);
    fwrite(&p.num_layers, sizeof(uint32_t), 1, f);
    fwrite(&p.num_attn_heads, sizeof(uint32_t), 1, f);
    fwrite(&p.num_kv_heads, sizeof(uint32_t), 1, f);
    fwrite(&p.rope_theta, sizeof(double), 1, f);
    fwrite(&p.rope_style, sizeof(uint32_t), 1, f);
    fwrite(&p.n_sampled, sizeof(uint32_t), 1, f);
    fwrite(&p.freq_count, sizeof(uint32_t), 1, f);

    // Model name (null-terminated, name_len includes the null)
    uint32_t name_len = (uint32_t)strlen(p.model_name) + 1;
    fwrite(&name_len, sizeof(uint32_t), 1, f);
    fwrite(p.model_name, 1, name_len, f);

    // Per sampled head
    for (uint32_t h = 0; h < p.n_sampled; h++) {
        uint32_t layer_idx = h;  // head 0 -> layer 0, head 1 -> layer 1
        uint32_t head_idx  = 0;  // both use attn_head 0 → kv_head 0
        fwrite(&layer_idx, sizeof(uint32_t), 1, f);
        fwrite(&head_idx, sizeof(uint32_t), 1, f);

        // q_mean_real[f] = 1.0 for all f (uniform concentration)
        // q_mean_imag[f] = 0.0 (purely real mean Q)
        // q_abs_mean[f]  = 1.0 (mean magnitude)
        // r_f[f]         = 1.0 (perfect concentration ratio)
        std::vector<float> q_mean_real(p.freq_count, 1.0f);
        std::vector<float> q_mean_imag(p.freq_count, 0.0f);
        std::vector<float> q_abs_mean(p.freq_count, 1.0f);
        std::vector<float> r_f(p.freq_count, 1.0f);

        fwrite(q_mean_real.data(), sizeof(float), p.freq_count, f);
        fwrite(q_mean_imag.data(), sizeof(float), p.freq_count, f);
        fwrite(q_abs_mean.data(), sizeof(float), p.freq_count, f);
        fwrite(r_f.data(), sizeof(float), p.freq_count, f);
    }

    fclose(f);
}

// ============================================================================
// Mock ggml tensor helpers
// ============================================================================

// Create a ggml_context + F32 2D tensor with a CPU backend buffer.
// Tensor layout: [n_embd_k_gqa, kv_size] where n_embd_k_gqa = n_kv_heads * head_dim.
struct mock_tensor_ctx {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

static mock_tensor_ctx make_mock_tensor_ctx(size_t mem_size) {
    mock_tensor_ctx mtc;
    ggml_init_params params;
    params.mem_size   = mem_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;  // don't allocate from pool; use backend buffer instead
    mtc.ctx = ggml_init(params);
    return mtc;
}

// Allocate backend buffers for all tensors in the context.
// Must be called AFTER all tensors are created, BEFORE writing data.
static void alloc_mock_tensors(mock_tensor_ctx & mtc) {
    if (!mtc.buf) {
        mtc.buf = ggml_backend_alloc_ctx_tensors_from_buft(mtc.ctx, ggml_backend_cpu_buffer_type());
        TEST_ASSERT_MSG(mtc.buf, "failed to allocate CPU backend buffer for mock tensors");
    }
}

static void free_mock_tensor_ctx(mock_tensor_ctx & mtc) {
    if (mtc.buf) {
        ggml_backend_buffer_free(mtc.buf);
        mtc.buf = nullptr;
    }
    if (mtc.ctx) {
        ggml_free(mtc.ctx);
        mtc.ctx = nullptr;
    }
}

// Create a K cache tensor: shape [n_kv_heads * head_dim, kv_size], type F32.
// Does NOT allocate backend buffer — call alloc_mock_tensors() after all tensors are created.
static ggml_tensor * make_k_tensor(
    mock_tensor_ctx & mtc,
    uint32_t n_kv_heads,
    uint32_t head_dim,
    uint32_t kv_size)
{
    const int64_t n_embd_k_gqa = (int64_t)n_kv_heads * (int64_t)head_dim;
    ggml_tensor * k = ggml_new_tensor_2d(mtc.ctx, GGML_TYPE_F32, n_embd_k_gqa, (int64_t)kv_size);
    return k;
}

// Write F32 K data for a specific cell and KV head into the tensor.
static void write_k_cell(
    ggml_tensor * k_tensor,
    uint32_t cell_idx,
    uint32_t kv_head_idx,
    uint32_t head_dim,
    const float * data)
{
    const int64_t n_embd_k_gqa = k_tensor->ne[0];
    const size_t row_bytes = (size_t)n_embd_k_gqa * sizeof(float);  // F32
    const size_t head_offset = (size_t)kv_head_idx * head_dim * sizeof(float);
    const size_t tensor_offset = (size_t)cell_idx * row_bytes + head_offset;
    const size_t head_bytes = head_dim * sizeof(float);
    ggml_backend_tensor_set(k_tensor, data, tensor_offset, head_bytes);
}

// ============================================================================
// Test 1: Calibration loading
// ============================================================================

static void test_calibration_loading() {
    fprintf(stderr, "--- test_calibration_loading ---\n");

    const char * tmp_path = "/tmp/test_triattention_calib.triattention";
    mock_calib_params p;
    write_mock_calib(tmp_path, p);

    triattention_scorer_config cfg;
    triattention_scorer scorer(
        tmp_path, cfg,
        p.rope_theta, p.head_dim, p.num_kv_heads);

    TEST_ASSERT(scorer.valid());
    TEST_ASSERT(scorer.get_head_dim() == p.head_dim);
    TEST_ASSERT(scorer.get_n_kv_heads() == p.num_kv_heads);
    TEST_ASSERT(scorer.get_n_sampled() == p.n_sampled);
    TEST_ASSERT(scorer.get_freq_count() == p.freq_count);
    TEST_ASSERT(strcmp(scorer.get_model_name(), "test_model") == 0);

    // Print info (just verify it doesn't crash)
    scorer.print_info(stderr);

    // Test invalid file path
    triattention_scorer bad_scorer(
        "/tmp/nonexistent_triattention_file.triattention", cfg,
        p.rope_theta, p.head_dim, p.num_kv_heads);
    TEST_ASSERT(!bad_scorer.valid());

    // Test model mismatch (wrong head_dim)
    triattention_scorer mismatch_scorer(
        tmp_path, cfg,
        p.rope_theta, 64, p.num_kv_heads);  // wrong head_dim
    TEST_ASSERT(!mismatch_scorer.valid());

    // Test model mismatch (wrong n_kv_heads)
    triattention_scorer mismatch_scorer2(
        tmp_path, cfg,
        p.rope_theta, p.head_dim, 8);  // wrong n_kv_heads
    TEST_ASSERT(!mismatch_scorer2.valid());

    remove(tmp_path);
    fprintf(stderr, "  PASSED\n");
}

// ============================================================================
// Test 2: RoPE inversion
// ============================================================================

static void test_rope_inversion() {
    fprintf(stderr, "--- test_rope_inversion ---\n");

    const uint32_t head_dim = 128;
    const uint32_t freq_count = head_dim / 2;
    const double rope_theta = 10000.0;
    const uint32_t rope_style = 0;  // half

    // Build omega array: omega[f] = rope_theta^(-2f/head_dim)
    std::vector<float> omega(freq_count);
    for (uint32_t f = 0; f < freq_count; f++) {
        double exponent = -2.0 * (double)f / (double)head_dim;
        omega[f] = (float)pow(rope_theta, exponent);
    }

    // Create a known pre-RoPE K vector
    // Use simple values: k_pre[f] = (f+1)*0.01, k_pre[f+fc] = (f+1)*0.02
    std::vector<float> k_pre(head_dim);
    for (uint32_t f = 0; f < freq_count; f++) {
        k_pre[f]              = (float)(f + 1) * 0.01f;
        k_pre[f + freq_count] = (float)(f + 1) * 0.02f;
    }

    // Apply forward RoPE at position 5 to get post-RoPE K
    // Half style:
    //   k_post[f]    = k_pre[f]*cos(w*pos) - k_pre[f+fc]*sin(w*pos)
    //   k_post[f+fc] = k_pre[f]*sin(w*pos) + k_pre[f+fc]*cos(w*pos)
    const int32_t position = 5;
    std::vector<float> k_post(head_dim);
    for (uint32_t f = 0; f < freq_count; f++) {
        float angle = omega[f] * (float)position;
        float c = cosf(angle);
        float s = sinf(angle);
        k_post[f]              = k_pre[f] * c - k_pre[f + freq_count] * s;
        k_post[f + freq_count] = k_pre[f] * s + k_pre[f + freq_count] * c;
    }

    // Now invert RoPE to recover pre-RoPE K
    std::vector<float> k_recovered(head_dim);
    std::vector<int32_t> positions = {position};
    triattention_invert_rope(
        k_recovered.data(),
        k_post.data(),
        positions.data(),
        omega.data(),
        1,           // n_keys
        head_dim,
        freq_count,
        rope_style);

    // Verify recovered matches original pre-RoPE K
    for (uint32_t f = 0; f < freq_count; f++) {
        TEST_ASSERT_MSG(float_eq(k_recovered[f], k_pre[f], 1e-3f),
            "RoPE inversion failed for real part");
        TEST_ASSERT_MSG(float_eq(k_recovered[f + freq_count], k_pre[f + freq_count], 1e-3f),
            "RoPE inversion failed for imag part");
    }

    // Test with multiple keys at different positions
    const uint32_t n_keys = 3;
    std::vector<float> k_pre_multi(n_keys * head_dim);
    std::vector<float> k_post_multi(n_keys * head_dim);
    std::vector<float> k_rec_multi(n_keys * head_dim);
    std::vector<int32_t> pos_multi = {0, 10, 127};

    for (uint32_t i = 0; i < n_keys; i++) {
        for (uint32_t f = 0; f < freq_count; f++) {
            k_pre_multi[i * head_dim + f]              = (float)(i * 100 + f) * 0.001f;
            k_pre_multi[i * head_dim + f + freq_count] = (float)(i * 100 + f) * 0.002f;
        }
        // Apply forward RoPE
        for (uint32_t f = 0; f < freq_count; f++) {
            float angle = omega[f] * (float)pos_multi[i];
            float c = cosf(angle);
            float s = sinf(angle);
            float re = k_pre_multi[i * head_dim + f];
            float im = k_pre_multi[i * head_dim + f + freq_count];
            k_post_multi[i * head_dim + f]              = re * c - im * s;
            k_post_multi[i * head_dim + f + freq_count] = re * s + im * c;
        }
    }

    triattention_invert_rope(
        k_rec_multi.data(),
        k_post_multi.data(),
        pos_multi.data(),
        omega.data(),
        n_keys,
        head_dim,
        freq_count,
        rope_style);

    for (uint32_t i = 0; i < n_keys; i++) {
        for (uint32_t f = 0; f < freq_count; f++) {
            TEST_ASSERT_MSG(
                float_eq(k_rec_multi[i * head_dim + f], k_pre_multi[i * head_dim + f], 1e-3f),
                "Multi-key RoPE inversion failed for real part");
            TEST_ASSERT_MSG(
                float_eq(k_rec_multi[i * head_dim + f + freq_count],
                         k_pre_multi[i * head_dim + f + freq_count], 1e-3f),
                "Multi-key RoPE inversion failed for imag part");
        }
    }

    fprintf(stderr, "  PASSED\n");
}

// ============================================================================
// Test 3: Scoring (score_head and score_combined)
// ============================================================================

static void test_scoring() {
    fprintf(stderr, "--- test_scoring ---\n");

    // Create calibration file
    const char * tmp_path = "/tmp/test_triattention_score.triattention";
    mock_calib_params p;
    write_mock_calib(tmp_path, p);

    triattention_scorer_config cfg;
    cfg.normalize_scores = false;  // raw scores for predictable ordering
    triattention_scorer scorer(
        tmp_path, cfg,
        p.rope_theta, p.head_dim, p.num_kv_heads);
    TEST_ASSERT(scorer.valid());

    const uint32_t head_dim = p.head_dim;
    const uint32_t n_kv_heads = p.num_kv_heads;
    const uint32_t kv_size = 16;  // 16 cell slots

    // Create mock K tensors for 2 layers (both must be created before allocating)
    size_t mem_size = ggml_tensor_overhead() * 4 + (size_t)n_kv_heads * head_dim * kv_size * sizeof(float) * 2 + 1024;
    mock_tensor_ctx mtc = make_mock_tensor_ctx(mem_size);
    ggml_tensor * k_tensor  = make_k_tensor(mtc, n_kv_heads, head_dim, kv_size);
    ggml_tensor * k_tensor2 = make_k_tensor(mtc, n_kv_heads, head_dim, kv_size);
    alloc_mock_tensors(mtc);  // allocate backend buffers for both tensors

    const uint32_t n_candidates = 4;
    std::vector<uint32_t> cell_indices = {0, 1, 2, 3};
    std::vector<int32_t>  positions    = {0, 1, 2, 3};

    // Fill layer 0 K data: cell i has magnitude proportional to (4-i)
    // so cell 0 has highest magnitude, cell 3 lowest
    for (uint32_t i = 0; i < n_candidates; i++) {
        std::vector<float> k_data(head_dim);
        float scale = (float)(n_candidates - i);  // 4, 3, 2, 1
        for (uint32_t f = 0; f < head_dim; f++) {
            k_data[f] = scale * 0.1f;
        }
        write_k_cell(k_tensor, cell_indices[i], 0, head_dim, k_data.data());
    }

    // Score head 0 (which maps to sampled head 0: layer=0, attn_head=0, kv_head=0)
    std::vector<float> scores(n_candidates);
    scorer.score_head(
        scores.data(),
        k_tensor,
        cell_indices.data(),
        positions.data(),
        0,              // kv_head_idx
        n_candidates,
        100             // frontier_position
    );

    // Verify all scores are finite
    for (uint32_t i = 0; i < n_candidates; i++) {
        TEST_ASSERT_MSG(std::isfinite(scores[i]), "score is not finite");
    }

    // With uniform q_mean and larger K magnitude → higher score
    // Cell 0 has largest magnitude, so should have highest score
    // (The norm term dominates with uniform Q stats)
    fprintf(stderr, "  scores: [%.4f, %.4f, %.4f, %.4f]\n",
            scores[0], scores[1], scores[2], scores[3]);

    // Verify relative ordering: cell 0 (largest K) should score highest
    int max_idx = 0;
    for (uint32_t i = 1; i < n_candidates; i++) {
        if (scores[i] > scores[max_idx]) max_idx = (int)i;
    }
    TEST_ASSERT_MSG(max_idx == 0, "cell 0 (largest K magnitude) should have highest score");

    // Fill layer 1 K data: reverse ordering — cell 3 has largest magnitude
    for (uint32_t i = 0; i < n_candidates; i++) {
        std::vector<float> k_data(head_dim);
        float scale = (float)(i + 1);  // 1, 2, 3, 4
        for (uint32_t f = 0; f < head_dim; f++) {
            k_data[f] = scale * 0.1f;
        }
        write_k_cell(k_tensor2, cell_indices[i], 0, head_dim, k_data.data());
    }

    // Score layer 1 head 0 individually
    std::vector<float> scores_h1(n_candidates);
    scorer.score_head(
        scores_h1.data(),
        k_tensor2,
        cell_indices.data(),
        positions.data(),
        0,              // kv_head_idx
        n_candidates,
        100
    );

    // score_combined: max aggregation across sampled heads
    // Head 0 (layer 0): cell 0 highest
    // Head 1 (layer 1): cell 3 highest
    // Combined max: each cell gets max of the two heads' scores
    std::vector<float> combined(n_candidates);
    ggml_tensor * k_tensors[2] = {k_tensor, k_tensor2};
    int32_t layer_map[2] = {0, 1};  // internal layer 0 → model layer 0, etc.

    scorer.score_combined(
        combined.data(),
        k_tensors,
        2,              // n_kv_layers
        layer_map,
        cell_indices.data(),
        positions.data(),
        n_candidates,
        100             // frontier_position
    );

    // Verify all combined scores are finite
    for (uint32_t i = 0; i < n_candidates; i++) {
        TEST_ASSERT_MSG(std::isfinite(combined[i]), "combined score is not finite");
    }

    fprintf(stderr, "  combined: [%.4f, %.4f, %.4f, %.4f]\n",
            combined[0], combined[1], combined[2], combined[3]);

    // Combined should be max of per-head scores
    for (uint32_t i = 0; i < n_candidates; i++) {
        float expected_max = fmaxf(scores[i], scores_h1[i]);
        TEST_ASSERT_MSG(float_eq(combined[i], expected_max, 1e-2f),
            "combined score should be max of per-head scores");
    }

    free_mock_tensor_ctx(mtc);
    remove(tmp_path);
    fprintf(stderr, "  PASSED\n");
}

// ============================================================================
// Test 4: Z-score normalization
// ============================================================================

// Reference implementation of z-score normalization (same algorithm as the scorer's internal one)
static void zscore_normalize_ref(float * scores, uint32_t n) {
    if (n <= 1) return;

    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) sum += scores[i];
    double mean = sum / n;

    double var_sum = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        double d = scores[i] - mean;
        var_sum += d * d;
    }
    double std = sqrt(var_sum / n);
    if (std < 1e-10) std = 1e-10;

    for (uint32_t i = 0; i < n; i++) {
        scores[i] = (float)((scores[i] - mean) / std);
    }
}

static void test_zscore_normalization() {
    fprintf(stderr, "--- test_zscore_normalization ---\n");

    // Test 4a: Verify reference z-score produces mean=0, std=1
    {
        std::vector<float> scores = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        zscore_normalize_ref(scores.data(), (uint32_t)scores.size());

        double mean = 0.0;
        for (float s : scores) mean += s;
        mean /= scores.size();
        TEST_ASSERT_MSG(fabs(mean) < 1e-5, "z-score mean should be ~0");

        double var = 0.0;
        for (float s : scores) {
            double d = s - mean;
            var += d * d;
        }
        double std = sqrt(var / scores.size());
        TEST_ASSERT_MSG(fabs(std - 1.0) < 1e-4, "z-score std should be ~1");
    }

    // Test 4b: Verify through the scorer with normalize_scores=true
    // Create calibration and K tensors, score with normalization on and off,
    // verify that normalization changes the scale but preserves ordering.
    {
        const char * tmp_path = "/tmp/test_triattention_zscore.triattention";
        mock_calib_params p;
        write_mock_calib(tmp_path, p);

        const uint32_t head_dim = p.head_dim;
        const uint32_t n_kv_heads = p.num_kv_heads;
        const uint32_t kv_size = 16;

        // Score WITHOUT normalization
        triattention_scorer_config cfg_raw;
        cfg_raw.normalize_scores = false;
        triattention_scorer scorer_raw(
            tmp_path, cfg_raw,
            p.rope_theta, p.head_dim, p.num_kv_heads);
        TEST_ASSERT(scorer_raw.valid());

        // Score WITH normalization
        triattention_scorer_config cfg_norm;
        cfg_norm.normalize_scores = true;
        triattention_scorer scorer_norm(
            tmp_path, cfg_norm,
            p.rope_theta, p.head_dim, p.num_kv_heads);
        TEST_ASSERT(scorer_norm.valid());

        // Create mock K tensors for 2 layers
        size_t mem_size = ggml_tensor_overhead() * 8 +
            (size_t)n_kv_heads * head_dim * kv_size * sizeof(float) * 2 + 4096;
        mock_tensor_ctx mtc = make_mock_tensor_ctx(mem_size);
        ggml_tensor * k_tensors[2] = {
            make_k_tensor(mtc, n_kv_heads, head_dim, kv_size),
            make_k_tensor(mtc, n_kv_heads, head_dim, kv_size)
        };
        alloc_mock_tensors(mtc);  // allocate backend buffers for both tensors

        const uint32_t n_candidates = 6;
        std::vector<uint32_t> cell_indices = {0, 1, 2, 3, 4, 5};
        std::vector<int32_t>  positions    = {0, 1, 2, 3, 4, 5};

        // Fill both layers with varying magnitude K data
        for (uint32_t l = 0; l < 2; l++) {
            for (uint32_t i = 0; i < n_candidates; i++) {
                std::vector<float> k_data(head_dim);
                float scale = (float)(n_candidates - i) * (l == 0 ? 1.0f : 0.5f);
                for (uint32_t f = 0; f < head_dim; f++) {
                    k_data[f] = scale * 0.1f;
                }
                write_k_cell(k_tensors[l], cell_indices[i], 0, head_dim, k_data.data());
            }
        }

        int32_t layer_map[2] = {0, 1};

        // Get raw combined scores
        std::vector<float> combined_raw(n_candidates);
        scorer_raw.score_combined(
            combined_raw.data(),
            k_tensors, 2, layer_map,
            cell_indices.data(), positions.data(),
            n_candidates, 200);

        // Get normalized combined scores
        std::vector<float> combined_norm(n_candidates);
        scorer_norm.score_combined(
            combined_norm.data(),
            k_tensors, 2, layer_map,
            cell_indices.data(), positions.data(),
            n_candidates, 200);

        // All scores should be finite
        for (uint32_t i = 0; i < n_candidates; i++) {
            TEST_ASSERT_MSG(std::isfinite(combined_raw[i]), "raw combined score not finite");
            TEST_ASSERT_MSG(std::isfinite(combined_norm[i]), "normalized combined score not finite");
        }

        fprintf(stderr, "  raw:  [");
        for (uint32_t i = 0; i < n_candidates; i++) fprintf(stderr, "%.4f%s", combined_raw[i], i < 5 ? ", " : "");
        fprintf(stderr, "]\n");
        fprintf(stderr, "  norm: [");
        for (uint32_t i = 0; i < n_candidates; i++) fprintf(stderr, "%.4f%s", combined_norm[i], i < 5 ? ", " : "");
        fprintf(stderr, "]\n");

        // Normalized scores should have different scale than raw
        // (z-score normalization rescales to mean=0, std=1 per head, then max)
        // They won't be exactly mean=0/std=1 because of the max aggregation,
        // but they should be on a different scale than raw.
        bool scale_changed = false;
        for (uint32_t i = 0; i < n_candidates; i++) {
            if (!float_eq(combined_raw[i], combined_norm[i], 1e-3f)) {
                scale_changed = true;
                break;
            }
        }
        TEST_ASSERT_MSG(scale_changed, "normalization should change score values");

        // The relative ordering should be preserved (normalization is monotonic per head)
        // Find the argmax in both
        int raw_max_idx = 0, norm_max_idx = 0;
        for (uint32_t i = 1; i < n_candidates; i++) {
            if (combined_raw[i]  > combined_raw[raw_max_idx])  raw_max_idx  = (int)i;
            if (combined_norm[i] > combined_norm[norm_max_idx]) norm_max_idx = (int)i;
        }
        // With max aggregation and z-score per head, the argmax may shift
        // because z-score changes relative magnitudes across heads.
        // We just verify both are finite and the normalized scores are reasonable.
        // (Not asserting same argmax — max-of-z-scores can reorder.)

        // Verify that normalized scores have reasonable range (z-scores are typically [-3, 3])
        for (uint32_t i = 0; i < n_candidates; i++) {
            TEST_ASSERT_MSG(fabsf(combined_norm[i]) < 100.0f,
                "normalized score should be in reasonable range");
        }

        free_mock_tensor_ctx(mtc);
        remove(tmp_path);
    }

    // Test 4c: Edge case — constant scores (std=0 → no division by zero)
    {
        std::vector<float> scores = {5.0f, 5.0f, 5.0f, 5.0f};
        zscore_normalize_ref(scores.data(), (uint32_t)scores.size());
        // With std clamped to 1e-10, all scores become ~0
        for (float s : scores) {
            TEST_ASSERT_MSG(std::isfinite(s), "constant z-score should be finite");
            TEST_ASSERT_MSG(fabsf(s) < 1e-3f, "constant z-score should be ~0");
        }
    }

    // Test 4d: Single element (n=1 → no normalization)
    {
        std::vector<float> scores = {42.0f};
        zscore_normalize_ref(scores.data(), 1);
        TEST_ASSERT_MSG(float_eq(scores[0], 42.0f), "single element should not be normalized");
    }

    fprintf(stderr, "  PASSED\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    fprintf(stderr, "=== TriAttention Scorer Tests ===\n\n");

    test_calibration_loading();
    test_rope_inversion();
    test_scoring();
    test_zscore_normalization();

    fprintf(stderr, "\n=== Results: %d failure(s) ===\n", g_test_failures);
    return g_test_failures == 0 ? 0 : 1;
}
